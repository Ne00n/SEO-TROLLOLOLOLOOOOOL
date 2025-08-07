#!/usr/bin/env python3
"""
Email Auto Responder with IMAP fetch + SMTP replies + Ollama integration.

Features:
- Fetch latest 50 emails via IMAP (STARTTLS).
- Track seen/replied message-ids in JSON state file.
- Subject keyword matching to trigger auto-replies.
- Business hours logic (Europe/Berlin by default; configurable).
- Outside business hours: send standard auto-reply.
- Inside business hours: call Ollama to generate reply, sanitize & send. Also queue any prior matching, unreplied emails.
- Strip & ignore any <think>...</think> blocks from content before using in prompt.
- STARTTLS for both IMAP (port 143 default) and SMTP (port 587 default).
- Config via .env environment variables.

Author: Roo
"""

import imaplib
import smtplib
import ssl
import re
import json
import os
import time
import logging
from email import message_from_bytes
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import parsedate_to_datetime, parseaddr
from datetime import datetime, time as dt_time, timedelta
from typing import List, Dict, Any, Tuple, Optional
import requests
from zoneinfo import ZoneInfo
import sys

# Fallback for environments where email.policy isn't available on the email module attribute
try:
    from email import policy as email_policy
    DEFAULT_EMAIL_POLICY = email_policy.default
except Exception:
    # Use compat by parsing without explicit policy
    email_policy = None
    DEFAULT_EMAIL_POLICY = None

STATE_FILE = os.environ.get("STATE_FILE", "state.json")
SEEN_LIMIT = int(os.environ.get("FETCH_LIMIT", "50"))

# Logging config
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format=LOG_FORMAT)
logger = logging.getLogger("auto_responder")

# Env config with defaults
IMAP_HOST = os.environ.get("IMAP_HOST", "")
IMAP_PORT = int(os.environ.get("IMAP_PORT", "143"))  # STARTTLS
IMAP_USER = os.environ.get("IMAP_USER", "")
IMAP_PASS = os.environ.get("IMAP_PASS", "")

SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))  # STARTTLS
SMTP_USER = os.environ.get("SMTP_USER", IMAP_USER)
SMTP_PASS = os.environ.get("SMTP_PASS", IMAP_PASS)

MAIL_FROM = os.environ.get("MAIL_FROM", SMTP_USER or IMAP_USER)

# Keywords comma-separated; case-insensitive
KEYWORDS = [k.strip() for k in os.environ.get("KEYWORDS", "urgent,asap,help,support,issue,error,problem").split(",") if k.strip()]

# Ollama
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Default to a widely available small model; override via .env
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "60"))

# Business hours
TZ_NAME = os.environ.get("BUSINESS_HOURS_TZ", "Europe/Berlin")
BUSINESS_DAYS = os.environ.get("BUSINESS_DAYS", "Mon-Fri")
BUSINESS_START = os.environ.get("BUSINESS_START", "09:00")
BUSINESS_END = os.environ.get("BUSINESS_END", "17:00")

# Outside-hours standard reply
OUT_OF_HOURS_MESSAGE = os.environ.get(
    "OUT_OF_HOURS_MESSAGE",
    "Thank you for your email. Our team is currently outside business hours and will get back to you during our next working period."
)

# Safety: strip <think> blocks pattern (case-insensitive, multiline, dotall)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>|<\s*think\s*>.*?<\s*/\s*think\s*>|<THINK>.*?</THINK>|<Think>.*?</Think>", re.DOTALL | re.IGNORECASE)
RAW_THINK_BLOCK_RE = re.compile(r"<.*?think.*?>.*?<.*?/.*?think.*?>", re.DOTALL | re.IGNORECASE)

# Real HTML variants if emails contain actual tags
HTML_THINK_BLOCK_RE = re.compile(r"<?\s*think\s*>.*?<?/\s*think\s*>|<think>.*?</think>|<THINK>.*?</THINK>", re.DOTALL | re.IGNORECASE)
REAL_TAG_THINK_BLOCK_RE = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_FILE):
        return {"seen_ids": [], "replied_ids": []}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"seen_ids": [], "replied_ids": []}


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    os.replace(tmp, STATE_FILE)
    logger.debug(f"State saved to {STATE_FILE}: seen={len(state.get('seen_ids', []))}, replied={len(state.get('replied_ids', []))}")


def decode_mime_header(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def get_message_id(msg: "EmailMessage") -> str:
    mid = msg.get("Message-ID") or msg.get("Message-Id") or ""
    return (mid or "").strip()


def is_business_day(dt: datetime) -> bool:
    # BUSINESS_DAYS can be "Mon-Fri" or "Mon,Wed,Fri"
    weekdays_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    spec = BUSINESS_DAYS.replace(" ", "")
    if "-" in spec:
        start, end = spec.split("-", 1)
        try:
            i_start = weekdays_map.index(start)
            i_end = weekdays_map.index(end)
        except ValueError:
            return dt.weekday() < 5  # default Mon-Fri
        if i_start <= i_end:
            valid = set(range(i_start, i_end + 1))
        else:
            # wrap-around like Fri-Mon
            valid = set(list(range(i_start, 7)) + list(range(0, i_end + 1)))
        return dt.weekday() in valid
    else:
        parts = [p for p in spec.split(",") if p]
        valid = set()
        for p in parts:
            if p in weekdays_map:
                valid.add(weekdays_map.index(p))
        if not valid:
            return dt.weekday() < 5
        return dt.weekday() in valid


def parse_hhmm(s: str) -> Tuple[int, int]:
    s = s.strip()
    if ":" in s:
        h, m = s.split(":", 1)
        return int(h), int(m)
    return int(s), 0


def in_business_hours(now: Optional[datetime] = None) -> bool:
    tz = ZoneInfo(TZ_NAME)
    now = now or datetime.now(tz)
    if not is_business_day(now):
        logger.debug(f"Outside business day ({BUSINESS_DAYS}), now={now.isoformat()}")
        return False
    sh, sm = parse_hhmm(BUSINESS_START)
    eh, em = parse_hhmm(BUSINESS_END)
    start = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    end = now.replace(hour=eh, minute=em, second=0, microsecond=0)
    inside = start <= now <= end
    logger.debug(f"Business hours check: {start.time()} - {end.time()}, now={now.time()}, inside={inside}")
    return inside


def sanitize_text(text: str) -> str:
    # Remove any think blocks in various encodings
    # 1) Real HTML tags
    text = REAL_TAG_THINK_BLOCK_RE.sub("", text)
    # 2) HTML-escaped variants
    text = HTML_THINK_BLOCK_RE.sub("", text)
    text = THINK_BLOCK_RE.sub("", text)
    text = RAW_THINK_BLOCK_RE.sub("", text)
    # Also collapse excessive whitespace
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def message_to_plaintext(msg: "EmailMessage") -> str:
    # Prefer text/plain, fallback to text/html stripped
    parts: List[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = (part.get("Content-Disposition") or "").lower()
            if ctype == "text/plain" and "attachment" not in disp:
                try:
                    parts.append(part.get_content().strip())
                except Exception:
                    try:
                        payload = part.get_payload(decode=True) or b""
                        charset = part.get_content_charset() or "utf-8"
                        parts.append(payload.decode(charset, errors="replace").strip())
                    except Exception:
                        pass
            elif ctype == "text/html" and "attachment" not in disp and not parts:
                # Use HTML only if no text/plain collected
                try:
                    html = part.get_content().strip()
                except Exception:
                    try:
                        payload = part.get_payload(decode=True) or b""
                        charset = part.get_content_charset() or "utf-8"
                        html = payload.decode(charset, errors="replace").strip()
                    except Exception:
                        html = ""
                if html:
                    txt = re.sub("<[^>]+>", " ", html)  # if HTML is already escaped
                    txt = re.sub("&nbsp;", " ", txt)
                    txt = re.sub(r"&[a-z]+;", " ", txt)
                    txt = re.sub(r"</?[^>]+>", " ", txt)
                    txt = re.sub(r"<!--.*?-->", " ", txt, flags=re.DOTALL)
                    parts.append(re.sub(r"\s+", " ", txt).strip())
    else:
        ctype = msg.get_content_type()
        try:
            content = msg.get_content().strip()
        except Exception:
            payload = msg.get_payload(decode=True) or b""
            charset = msg.get_content_charset() or "utf-8"
            content = payload.decode(charset, errors="replace").strip()
        if ctype == "text/plain":
            parts.append(content)
        elif ctype == "text/html":
            txt = re.sub("<[^>]+>", " ", content)
            txt = re.sub("&nbsp;", " ", txt)
            txt = re.sub(r"&[a-z]+;", " ", txt)
            txt = re.sub(r"</?[^>]+>", " ", txt)
            txt = re.sub(r"<!--.*?-->", " ", txt, flags=re.DOTALL)
            parts.append(re.sub(r"\s+", " ", txt).strip())

    text = "\n\n".join([p for p in parts if p]).strip()
    return sanitize_text(text)


def subject_matches(subject: str) -> bool:
    if not subject:
        return False
    subj = subject.lower()
    return any(k.lower() in subj for k in KEYWORDS)


def ensure_starttls_imap(host: str, port: int, user: str, password: str) -> imaplib.IMAP4:
    logger.debug(f"Connecting IMAP STARTTLS to {host}:{port} as {user}")
    imap = imaplib.IMAP4(host=host, port=port)
    typ, resp = imap.starttls(ssl_context=ssl.create_default_context())
    logger.debug(f"IMAP STARTTLS response: {typ} {resp}")
    typ, resp = imap.login(user, password)
    logger.debug(f"IMAP login response: {typ}")
    return imap


def ensure_starttls_smtp(host: str, port: int, user: str, password: str) -> smtplib.SMTP:
    logger.debug(f"Connecting SMTP STARTTLS to {host}:{port} as {user}")
    smtp = smtplib.SMTP(host=host, port=port, timeout=60)
    code, msg = smtp.ehlo()
    logger.debug(f"SMTP EHLO pre-STARTTLS: {code} {msg}")
    code, msg = smtp.starttls(context=ssl.create_default_context())
    logger.debug(f"SMTP STARTTLS: {code} {msg}")
    code, msg = smtp.ehlo()
    logger.debug(f"SMTP EHLO post-STARTTLS: {code} {msg}")
    if user:
        smtp.login(user, password)
        logger.debug("SMTP login successful")
    return smtp


def send_email(to_addr: str, subject: str, body: str, in_reply_to: Optional[str] = None, references: Optional[str] = None) -> None:
    msg = EmailMessage()
    msg["From"] = MAIL_FROM
    msg["To"] = to_addr
    msg["Subject"] = subject
    if in_reply_to:
        msg["In-Reply-To"] = in_reply_to
    if references:
        msg["References"] = references
    msg.set_content(body)

    logger.info(f"Sending email to={to_addr} subject={subject!r} in_reply_to={bool(in_reply_to)}")
    with ensure_starttls_smtp(SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS) as smtp:
        smtp.send_message(msg)
    logger.debug("Email sent successfully")


def fetch_latest_messages() -> List[Tuple[bytes, EmailMessage]]:
    with ensure_starttls_imap(IMAP_HOST, IMAP_PORT, IMAP_USER, IMAP_PASS) as imap:
        typ, mailbox_info = imap.select("INBOX")
        logger.debug(f"IMAP select INBOX: {typ} {mailbox_info}")
        # Search all; could refine e.g., UNSEEN
        typ, data = imap.search(None, "ALL")
        logger.debug(f"IMAP search ALL: {typ} count={len(data[0].split()) if typ == 'OK' and data and data[0] else 0}")
        if typ != "OK" or not data or not data[0]:
            return []
        ids = data[0].split()
        latest_ids = ids[-SEEN_LIMIT:]
        logger.info(f"Fetching {len(latest_ids)} latest emails (limit={SEEN_LIMIT})")
        messages: List[Tuple[bytes, EmailMessage]] = []
        for uid in reversed(latest_ids):
            typ, msg_data = imap.fetch(uid, "(RFC822)")
            if typ == "OK" and msg_data and isinstance(msg_data[0], tuple):
                raw = msg_data[0][1]
                if DEFAULT_EMAIL_POLICY is not None:
                    msg = message_from_bytes(raw, policy=DEFAULT_EMAIL_POLICY)
                else:
                    msg = message_from_bytes(raw)
                # Ensure type is EmailMessage
                if not isinstance(msg, EmailMessage):
                    # Convert to EmailMessage if possible by rebuilding
                    em = EmailMessage()
                    for k, v in msg.items():
                        em[k] = v
                    try:
                        em.set_content(msg.get_content())
                    except Exception:
                        payload = msg.get_payload(decode=True) or b""
                        charset = getattr(msg, "get_content_charset", lambda: "utf-8")() or "utf-8"
                        try:
                            em.set_content(payload.decode(charset, errors="replace"))
                        except Exception:
                            em.set_content("")
                    msg = em
                subj = decode_mime_header(msg.get("Subject"))
                frm = decode_mime_header(msg.get("From"))
                logger.debug(f"Fetched UID={uid.decode()} MID={get_message_id(msg)} Subject={subj!r} From={frm!r}")
                messages.append((uid, msg))
            else:
                logger.warning(f"Failed to fetch UID={uid.decode()} typ={typ}")
        return messages


def ollama_generate_reply(prompt: str) -> str:
    """
    Generate a reply using Ollama /api/generate (non-streaming).
    Adds extra logging to help debug payload/endpoint issues.
    """
    def _sanitize_out(text: str) -> str:
        text = sanitize_text(text)
        # Extra guard: remove bracketed placeholders like [NAME], {placeholder}, <placeholder>
        text = re.sub(r"\[[^\]]+\]", "", text)
        text = re.sub(r"\{[^}]+\}", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        return text.strip()

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # Common options supported by Ollama
        "options": {
            "temperature": 0.7
        }
    }
    logger.debug(f"Ollama generate URL={url}")
    logger.debug(f"Ollama model={OLLAMA_MODEL}, prompt_len={len(prompt)}")
    # Explicitly prevent any GET due to proxies/misconfig by forcing method and verifying
    r = requests.request("POST", url, json=payload, timeout=OLLAMA_TIMEOUT)
    logger.debug(f"Ollama status={r.status_code}, content-type={r.headers.get('content-type')}")
    r.raise_for_status()
    data = r.json()
    # Standard response schema from /api/generate contains 'response'
    text = data.get("response", "") or data.get("text", "")
    logger.debug(f"Ollama response text length={len(text)}")
    return _sanitize_out(text)


def build_ollama_prompt(original_subject: str, original_from: str, original_body: str) -> str:
    # Clear, strict instruction: no placeholders, provide a concise professional reply.
    # Include context but sanitized.
    guidelines = (
        "You are an assistant drafting a professional, helpful, and concise email reply. "
        "Write a direct response to the customer's message without using any placeholders or variables. "
        "Do not include any bracketed items like [name], {company}, or <anything>. "
        "Do not add boilerplate signatures; just the main body of the email.\n\n"
        "Constraints:\n"
        "- No placeholders or TODO markers.\n"
        "- Keep it 3-8 sentences.\n"
        "- Be specific to the user's issue if possible.\n"
        "- If clarification is required, ask 1-2 targeted questions.\n"
        "- Maintain a polite, professional tone.\n"
    )
    prompt = (
        f"{guidelines}\n"
        f"Original Subject: {original_subject}\n"
        f"From: {original_from}\n"
        "Customer Message (sanitized, any think blocks removed):\n"
        "-----\n"
        f"{original_body}\n"
        "-----\n\n"
        "Now draft the reply message body only (no greeting placeholders like 'Dear [Name]', no signatures)."
    )
    return prompt


def collect_unreplied_queue(messages: List[Tuple[bytes, EmailMessage]], state: Dict[str, Any]) -> List[EmailMessage]:
    queue: List[EmailMessage] = []
    for _, msg in messages:
        mid = get_message_id(msg)
        subject = decode_mime_header(msg.get("Subject"))
        if not mid:
            continue
        if mid in state.get("replied_ids", []):
            continue
        if not subject_matches(subject):
            continue
        queue.append(msg)
    return queue


def reply_to_message(msg: "EmailMessage", use_ollama: bool, state: Dict[str, Any]) -> None:
    mid = get_message_id(msg)
    if not mid:
        logger.warning("Skipping reply: missing Message-ID")
        return

    subject = decode_mime_header(msg.get("Subject"))
    from_header = decode_mime_header(msg.get("From"))
    # Extract email address for reply
    addr = parseaddr(from_header)[1]
    if not addr:
        logger.warning(f"Skipping reply for MID={mid}: could not parse a reply address from From={from_header!r}")
        return

    logger.info(f"Preparing reply for MID={mid} to={addr} subject={subject!r} use_ollama={use_ollama}")
    original_body = message_to_plaintext(msg)
    logger.debug(f"Original body (sanitized, length={len(original_body)} chars)")

    if use_ollama:
        prompt = build_ollama_prompt(subject, from_header, original_body)
        logger.debug(f"Ollama prompt length={len(prompt)}")
        try:
            body = ollama_generate_reply(prompt)
            logger.debug(f"Ollama reply length={len(body)}")
        except Exception as e:
            logger.exception(f"Ollama generation failed for MID={mid}: {e}")
            body = ("Thank you for your email. We encountered an internal issue generating an automated reply. "
                    "We will follow up shortly with a detailed response.")
    else:
        body = OUT_OF_HOURS_MESSAGE

    reply_subject = f"Re: {subject}" if subject and not subject.lower().startswith("re:") else (subject or "Re: your email")
    in_reply_to = msg.get("Message-ID")
    references = msg.get("References") or in_reply_to

    send_email(addr, reply_subject, body, in_reply_to=in_reply_to, references=references)

    # Mark replied
    rep = state.setdefault("replied_ids", [])
    if mid not in rep:
        rep.append(mid)
        logger.debug(f"Marked replied MID={mid}")


def main() -> None:
    # Basic config validation
    missing = []
    for key in ["IMAP_HOST", "IMAP_USER", "IMAP_PASS", "SMTP_HOST", "MAIL_FROM"]:
        if not globals().get(key):
            missing.append(key)
    if missing:
        logger.error(f"Configuration missing environment variables: {', '.join(missing)}")
        logger.error("Create a .env file based on .env.example and re-run.")
        return

    logger.info("Starting email auto responder")
    logger.info(f"Config: IMAP_HOST={IMAP_HOST}, IMAP_PORT={IMAP_PORT}, SMTP_HOST={SMTP_HOST}, SMTP_PORT={SMTP_PORT}, FETCH_LIMIT={SEEN_LIMIT}")
    logger.info(f"Ollama: base={OLLAMA_BASE_URL}, model={OLLAMA_MODEL}, timeout={OLLAMA_TIMEOUT}s")
    logger.info(f"Business hours: {BUSINESS_DAYS} {BUSINESS_START}-{BUSINESS_END} TZ={TZ_NAME}")
    logger.info(f"Keywords: {KEYWORDS}")

    state = load_state()
    logger.debug(f"Loaded state: seen={len(state.get('seen_ids', []))}, replied={len(state.get('replied_ids', []))}")

    # Fetch latest messages
    try:
        messages = fetch_latest_messages()
    except Exception as e:
        logger.exception(f"IMAP fetch failed: {e}")
        return

    logger.info(f"Fetched {len(messages)} messages")

    # Update seen_ids
    for _, msg in messages:
        mid = get_message_id(msg)
        if mid and mid not in state.get("seen_ids", []):
            state["seen_ids"].append(mid)

    # Process messages with keyword subjects
    matching_msgs = []
    for _, msg in messages:
        subject = decode_mime_header(msg.get("Subject"))
        if subject_matches(subject):
            matching_msgs.append(msg)
    logger.info(f"Subject-matching messages: {len(matching_msgs)}")

    # Determine business hours
    inside = in_business_hours()
    logger.info(f"In business hours: {inside}")

    if not inside:
        # Outside hours: reply standard to any matching not already replied
        for msg in matching_msgs:
            mid = get_message_id(msg)
            if not mid or mid in state.get("replied_ids", []):
                continue
            try:
                reply_to_message(msg, use_ollama=False, state=state)
                time.sleep(0.5)
            except Exception as e:
                logger.exception(f"Failed to send out-of-hours reply for {mid}: {e}")
    else:
        # Inside hours: queue includes prior unreplied matching emails too
        queue = collect_unreplied_queue(messages, state)
        logger.info(f"Reply queue (inside hours): {len(queue)}")
        # Ensure chronological by email Date header (oldest first)
        def parse_date(m: "EmailMessage") -> float:
            try:
                dt = parsedate_to_datetime(m.get("Date"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=ZoneInfo(TZ_NAME))
                return dt.timestamp()
            except Exception:
                return 0.0
        queue.sort(key=parse_date)

        for msg in queue:
            mid = get_message_id(msg)
            if not mid:
                continue
            try:
                reply_to_message(msg, use_ollama=True, state=state)
                time.sleep(0.8)
            except Exception as e:
                logger.exception(f"Failed to send business-hours reply for {mid}: {e}")

    save_state(state)
    logger.info("Done.")


if __name__ == "__main__":
    main()