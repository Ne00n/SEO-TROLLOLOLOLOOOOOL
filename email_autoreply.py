#!/usr/bin/env python3
"""
Email Auto-Reply Script

Workflow:
- Load IMAP/SMTP + Ollama config from config/creds.json
- Log into IMAP, fetch last 50 messages (by UID)
- Track seen Message-IDs in data/seen_emails.json
- For unseen messages, check subject/title for trigger words
- If triggered:
    - During business hours: call Ollama via curl to generate a reply
    - Outside business hours: send a standard "we will reply soon" message
- Send a reply via SMTP, quoting original message-id and in-reply-to headers
- Persist newly seen Message-IDs

Usage:
  python3 scripts/email_autoreply.py           # normal run
  python3 scripts/email_autoreply.py --dry-run # no external send/curl

Requirements:
- Python 3.9+
- curl installed and reachable in PATH
"""

import argparse
import email
import imaplib
import json
import os
import re
import smtplib
import ssl
import subprocess
import sys
import time
from base64 import b64decode
from datetime import datetime, timezone
from email.header import decode_header, make_header
from email.message import EmailMessage
from email.utils import make_msgid, parseaddr, formataddr
try:
    # Python 3.9+ recommended
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:
    ZoneInfo = None  # Fallback will use naive local time if zoneinfo not available

# -----------------------
# Constants and Defaults
# -----------------------

DEFAULT_CONFIG_PATH = "config/creds.json"
DEFAULT_SEEN_PATH = "data/seen_emails.json"
DEFAULT_THREADS_PATH = "data/threads.json"
MAX_FETCH = 50
REPLY_RATE_LIMIT_SEC = 2.0  # brief pause between replies to avoid throttling

# -----------------------
# Utilities
# -----------------------

def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def decode_mime_header(value: str) -> str:
    if not value:
        return ""
    try:
        decoded = make_header(decode_header(value))
        return str(decoded)
    except Exception:
        return value

def get_text_from_message(msg: email.message.Message) -> str:
    # Prefer text/plain; if not found, fallback to text/html stripped
    if msg.is_multipart():
        parts = []
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition", "")).lower()
            if ctype == "text/plain" and "attachment" not in disp:
                try:
                    charset = part.get_content_charset() or "utf-8"
                    parts.append(part.get_payload(decode=True).decode(charset, errors="replace"))
                except Exception:
                    continue
        if parts:
            return "\n".join(parts)
        # fallback to html
        for part in msg.walk():
            ctype = part.get_content_type()
            if ctype == "text/html":
                try:
                    charset = part.get_content_charset() or "utf-8"
                    html = part.get_payload(decode=True).decode(charset, errors="replace")
                    return strip_html(html)
                except Exception:
                    continue
        return ""
    else:
        ctype = msg.get_content_type()
        try:
            charset = msg.get_content_charset() or "utf-8"
            payload = msg.get_payload(decode=True)
            if payload is None:
                return msg.get_payload()
            text = payload.decode(charset, errors="replace")
            if ctype == "text/html":
                return strip_html(text)
            return text
        except Exception:
            return msg.get_payload()

def strip_html(html: str) -> str:
    # very naive strip; good enough for fallback
    text = re.sub(r"<script[\s\S]*?</script>", "", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def word_trigger_match(subject: str, triggers) -> bool:
    if not subject:
        return False
    subj = subject.lower()
    for t in triggers:
        t_norm = t.lower().strip()
        if not t_norm:
            continue
        # word-boundary; allow hyphenated or punctuation around
        pattern = r"\b" + re.escape(t_norm) + r"\b"
        if re.search(pattern, subj):
            return True
    return False

def run_curl_ollama(base_url: str, model: str, system_prompt: str, user_prompt: str, timeout: int = 60):
    """
    Call Ollama HTTP API using curl to generate response text.
    POST /api/generate
    Body: {"model": model, "prompt": "...", "system": "...", "stream": false}
    Returns the 'response' field.
    """
    payload = {
        "model": model,
        "prompt": user_prompt,
        "system": system_prompt,
        "stream": False
    }
    cmd = [
        "curl", "-sS", "-X", "POST",
        f"{base_url.rstrip('/')}/api/generate",
        "-H", "Content-Type: application/json",
        "--max-time", str(timeout),
        "--data", json.dumps(payload),
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        data = json.loads(res.stdout)
        return data.get("response", "").strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"curl error: {e.stderr.strip()}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError("Failed to parse Ollama response as JSON") from e

def smtp_connect_send(config: dict, msg: EmailMessage, dry_run: bool = False):
    smtp_conf = config.get("smtp", {})
    host = smtp_conf.get("host")
    port = int(smtp_conf.get("port", 587))
    use_ssl = bool(smtp_conf.get("use_ssl", False))
    use_starttls = bool(smtp_conf.get("use_starttls", True))
    username = smtp_conf.get("username")
    password = smtp_conf.get("password")

    if dry_run:
        print("[DRY-RUN] Would send email via SMTP:")
        print(f"  From: {msg['From']}")
        print(f"  To:   {msg['To']}")
        print(f"  Subj: {msg['Subject']}")
        return

    if not host or not username or not password:
        raise ValueError("SMTP config incomplete (host/username/password)")

    if use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=context) as server:
            server.login(username, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port) as server:
            server.ehlo()
            if use_starttls:
                context = ssl.create_default_context()
                server.starttls(context=context)
                server.ehlo()
            server.login(username, password)
            server.send_message(msg)

def imap_connect(config: dict) -> imaplib.IMAP4:
    imap_conf = config.get("imap", {})
    host = imap_conf.get("host")
    port = int(imap_conf.get("port", 993))
    use_ssl = bool(imap_conf.get("use_ssl", True))
    username = imap_conf.get("username")
    password = imap_conf.get("password")

    if not host or not username or not password:
        raise ValueError("IMAP config incomplete (host/username/password)")

    if use_ssl:
        M = imaplib.IMAP4_SSL(host, port)
    else:
        M = imaplib.IMAP4(host, port)
        if imap_conf.get("use_starttls", False):
            M.starttls()
    typ, resp = M.login(username, password)
    if typ != "OK":
        raise RuntimeError(f"IMAP login failed: {resp}")
    return M

def fetch_last_n_messages(M: imaplib.IMAP4, mailbox: str, n: int) -> list[tuple[bytes, bytes]]:
    # Select mailbox
    typ, _ = M.select(mailbox, readonly=True)
    if typ != "OK":
        raise RuntimeError(f"Could not select mailbox {mailbox}")

    # Search all UIDs
    typ, data = M.uid("SEARCH", None, "ALL")
    if typ != "OK":
        raise RuntimeError("IMAP UID SEARCH failed")

    uids = [uid for uid in data[0].split() if uid]
    if not uids:
        return []

    last_uids = uids[-n:]

    # Fetch RFC822 for each UID
    results = []
    for uid in last_uids:
        typ, msg_data = M.uid("FETCH", uid, "(RFC822)")
        if typ != "OK" or not msg_data:
            continue
        # msg_data is a list of tuples: [(b'UID ... RFC822 {bytes}', raw), b')']
        for part in msg_data:
            if isinstance(part, tuple) and len(part) == 2:
                results.append((uid, part[1]))
    return results

def build_reply(original_msg: email.message.Message, reply_text: str, from_addr: str) -> EmailMessage:
    orig_from = parseaddr(original_msg.get("From", ""))
    orig_to = parseaddr(original_msg.get("To", ""))
    orig_subject = decode_mime_header(original_msg.get("Subject", ""))
    orig_msgid = original_msg.get("Message-ID") or make_msgid()

    # Choose reply recipient: reply-to if present, else From
    reply_to_hdr = original_msg.get("Reply-To")
    if reply_to_hdr:
        reply_to = parseaddr(reply_to_hdr)[1]
    else:
        reply_to = orig_from[1] or orig_to[1]

    msg = EmailMessage()
    msg["Subject"] = f"Re: {orig_subject}" if not orig_subject.lower().startswith("re:") else orig_subject
    msg["From"] = from_addr
    msg["To"] = reply_to
    msg["In-Reply-To"] = orig_msgid
    # Extend References with prior References if present
    prior_refs = original_msg.get_all("References", [])
    refs_chain = " ".join(prior_refs + [orig_msgid]) if prior_refs else orig_msgid
    msg["References"] = refs_chain
    msg.set_content(f"{reply_text}\n\n---\nAutomated response generated by Ollama.")
    return msg

def generate_prompt(original_subject: str, original_body: str) -> tuple[str, str]:
    system_prompt = (
        "You are a helpful email assistant. Provide concise, polite, and professional "
        "email replies. Use European/Berlin business formal style. Do not hallucinate. "
        "Ask for missing details if necessary. Keep under 180 words."
    )
    user_prompt = (
        f"Subject: {original_subject}\n\n"
        f"Email:\n{original_body}\n\n"
        "Write an appropriate reply."
    )
    return system_prompt, user_prompt

def summarize_for_thread(history_bodies: list[str], max_chars: int = 2000) -> str:
    """
    Simple truncation-based summarizer for context. Keeps last messages within limit.
    """
    if not history_bodies:
        return ""
    joined = "\n\n---\n\n".join(history_bodies)[-max_chars:]
    return joined

# -----------------------
# Main flow
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Email Auto-Reply")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to creds/config JSON")
    parser.add_argument("--seen", default=DEFAULT_SEEN_PATH, help="Path to seen emails JSON")
    parser.add_argument("--threads", default=DEFAULT_THREADS_PATH, help="Path to per-thread state JSON")
    parser.add_argument("--mailbox", default="INBOX", help="IMAP mailbox to scan")
    parser.add_argument("--dry-run", action="store_true", help="Do not call curl or send emails")
    parser.add_argument("--max", type=int, default=MAX_FETCH, help="How many latest emails to inspect")
    args = parser.parse_args()

    config = load_json(args.config, default={})
    if not config:
        print(f"Config not found or empty at {args.config}. Creating example config.")
        example = example_config()
        save_json(args.config, example)
        print("Fill in your credentials, then re-run.")
        return

    seen = load_json(args.seen, default={"message_ids": []})
    seen_ids: set[str] = set(seen.get("message_ids", []))

    # Thread state:
    # threads = {
    #   "root_msgid_or_thread_id": {
    #       "status": "open"|"resolved",
    #       "last_updated": "iso8601",
    #       "messages": [{"id": "...", "from": "...", "date": "...", "subject": "..."}],
    #       "auto_reply_count": 1
    #   }
    # }
    threads = load_json(args.threads, default={})

    triggers = config.get("triggers", ["urgent", "support", "refund", "invoice"])
    ollama_conf = config.get("ollama", {"base_url": "http://localhost:11434", "model": "llama3.1:8b"})
    from_identity = config.get("from", {"name": "", "email": ""})
    from_email = formataddr((from_identity.get("name", ""), from_identity.get("email", ""))).strip()
    if not from_email or "@" not in from_email:
        # fallback to SMTP username as from
        from_email = config.get("smtp", {}).get("username", "")

    # Business hours configuration
    bh_conf = config.get("business_hours", {})
    tz_name = bh_conf.get("timezone", "Europe/Berlin")
    open_hour = int(bh_conf.get("open_hour", 8))   # 0-23
    close_hour = int(bh_conf.get("close_hour", 20)) # 0-23
    after_hours_text = bh_conf.get(
        "after_hours_message",
        "Thank you for your email. Our team will review your message during business hours and get back to you as soon as possible."
    )

    def is_business_open(now_dt=None):
        # Determine local time for configured timezone
        if ZoneInfo:
            try:
                tz = ZoneInfo(tz_name)
                now_local = (now_dt or datetime.now(timezone.utc)).astimezone(tz)
            except Exception:
                now_local = now_dt or datetime.now()
        else:
            now_local = now_dt or datetime.now()
        hour = now_local.hour
        # Open if open_hour <= hour < close_hour
        return open_hour <= hour < close_hour

    # Connect IMAP
    try:
        M = imap_connect(config)
    except Exception as e:
        print(f"[ERROR] IMAP connection failed: {e}")
        sys.exit(1)

    new_seen = False
    try:
        messages = fetch_last_n_messages(M, args.mailbox, max(1, args.max))
        if not messages:
            print("No messages found.")
        to_reply = []

        for uid, raw in messages:
            try:
                msg = email.message_from_bytes(raw)
                msgid = msg.get("Message-ID")
                subject = decode_mime_header(msg.get("Subject", ""))
                date_hdr = msg.get("Date", "")
                frm_hdr = decode_mime_header(msg.get("From", ""))

                if not msgid:
                    # Create a synthetic ID from UID if Message-ID is missing
                    msgid = f"<uid-{uid.decode()}@local>"

                # Thread identification: prefer topmost References root, else In-Reply-To, else own Message-ID
                in_reply_to = msg.get("In-Reply-To")
                references = msg.get_all("References", [])
                if references:
                    # The first reference is typically the thread root
                    ref_tokens = " ".join(references).split()
                    thread_root = ref_tokens[0] if ref_tokens else (in_reply_to or msgid)
                else:
                    thread_root = in_reply_to or msgid

                if msgid in seen_ids:
                    # Already processed
                    # But update threads metadata if missing
                    th = threads.get(thread_root)
                    if th:
                        if not any(m.get("id") == msgid for m in th.get("messages", [])):
                            th.setdefault("messages", []).append({
                                "id": msgid,
                                "from": frm_hdr,
                                "date": date_hdr,
                                "subject": subject
                            })
                            th["last_updated"] = datetime.now(timezone.utc).isoformat()
                    continue

                # Mark as seen no matter what, so we don't reprocess forever
                seen_ids.add(msgid)
                new_seen = True

                # Update or create thread record
                th = threads.setdefault(thread_root, {
                    "status": "open",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "messages": [],
                    "auto_reply_count": 0
                })
                th["messages"].append({
                    "id": msgid,
                    "from": frm_hdr,
                    "date": date_hdr,
                    "subject": subject
                })
                th["last_updated"] = datetime.now(timezone.utc).isoformat()

                # Decide reply policy:
                # - If thread is resolved -> no further auto replies.
                # - Otherwise, check triggers OR if this email is a reply in an existing open thread we already engaged in.
                body = get_text_from_message(msg)
                engaged = th.get("auto_reply_count", 0) > 0
                should_reply = word_trigger_match(subject, triggers) or engaged

                if should_reply:
                    if is_business_open():
                        # Build context summary from thread history (last few bodies if available)
                        history_bodies = []
                        # We only have the current body here; optionally we could fetch thread again for bodies.
                        history_bodies.append(body)
                        context = summarize_for_thread(history_bodies)
                        sys_p, usr_p = generate_prompt(subject, context)
                        to_reply.append(("ollama", msg, sys_p, usr_p, thread_root))
                    else:
                        to_reply.append(("after_hours", msg, None, None, thread_root))
                else:
                    # Not triggered and not engaged; skip
                    continue
            except Exception as e:
                print(f"[WARN] Failed processing a message: {e}")
                continue

        # Generate and send replies
        for idx, item in enumerate(to_reply, 1):
            print(f"Preparing reply {idx}/{len(to_reply)}...")
            try:
                # item shape: (mode, orig_msg, sys_prompt, usr_prompt, thread_root)
                if len(item) == 5:
                    mode, orig_msg, sys_prompt, usr_prompt, thread_root = item
                else:
                    # backward compat
                    mode, orig_msg, sys_prompt, usr_prompt = item
                    # derive thread_root
                    in_reply_to = orig_msg.get("In-Reply-To")
                    refs = orig_msg.get_all("References", [])
                    if refs:
                        ref_tokens = " ".join(refs).split()
                        thread_root = ref_tokens[0] if ref_tokens else (in_reply_to or orig_msg.get("Message-ID"))
                    else:
                        thread_root = in_reply_to or orig_msg.get("Message-ID")

                if mode == "after_hours":
                    reply_text = after_hours_text if not args.dry_run else "[DRY-RUN] After-hours auto-reply."
                else:
                    if args.dry_run:
                        reply_text = "[DRY-RUN] This would be a generated reply."
                    else:
                        reply_text = run_curl_ollama(
                            base_url=ollama_conf.get("base_url", "http://localhost:11434"),
                            model=ollama_conf.get("model", "llama3.1:8b"),
                            system_prompt=sys_prompt,
                            user_prompt=usr_prompt,
                            timeout=int(ollama_conf.get("timeout", 60)),
                        )

                reply_msg = build_reply(
                    original_msg=orig_msg,
                    reply_text=reply_text,
                    from_addr=from_email,
                )
                smtp_connect_send(config, reply_msg, dry_run=args.dry_run)
                print("Reply sent.")

                # Update thread stats
                th = threads.setdefault(thread_root, {
                    "status": "open",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "messages": [],
                    "auto_reply_count": 0
                })
                th["auto_reply_count"] = th.get("auto_reply_count", 0) + 1
                th["last_updated"] = datetime.now(timezone.utc).isoformat()

                time.sleep(REPLY_RATE_LIMIT_SEC)
            except Exception as e:
                print(f"[ERROR] Reply failed: {e}")

    finally:
        try:
            M.logout()
        except Exception:
            pass

    if new_seen:
        save_json(args.seen, {"message_ids": sorted(seen_ids)})
        print(f"Updated seen list at {args.seen}")

def example_config():
    return {
        "imap": {
            "host": "imap.example.com",
            "port": 993,
            "use_ssl": True,
            "use_starttls": False,
            "username": "user@example.com",
            "password": "your-IMAP-password-or-app-password"
        },
        "smtp": {
            "host": "smtp.example.com",
            "port": 587,
            "use_ssl": False,
            "use_starttls": True,
            "username": "user@example.com",
            "password": "your-SMTP-password-or-app-password"
        },
        "from": {
            "name": "Support Bot",
            "email": "user@example.com"
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "model": "llama3.1:8b",
            "timeout": 60
        },
        "triggers": ["urgent", "support", "refund", "invoice"],
        "business_hours": {
            "timezone": "Europe/Berlin",
            "open_hour": 8,
            "close_hour": 20,
            "after_hours_message": "Thank you for your email. Our team will review your message during business hours and get back to you as soon as possible."
        }
    }

if __name__ == "__main__":
    main()