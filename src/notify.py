"""Lightweight notification module for bet alerts.

Supports Slack webhooks and SMTP email. All failures are silent — caught and
written to logs/notify_errors.log so they never crash the main prediction
pipeline.

Configuration via environment variables:
    NOTIFY_SLACK_WEBHOOK  — Slack incoming webhook URL
    NOTIFY_EMAIL_TO       — recipient email address
    NOTIFY_EMAIL_FROM     — sender email address
    NOTIFY_SMTP_HOST      — SMTP server (default: smtp.gmail.com)
    NOTIFY_SMTP_PORT      — SMTP port (default: 587)
    NOTIFY_SMTP_PASSWORD  — SMTP password / app password

CLI test:
    python -m notify --test
"""

from __future__ import annotations

import logging
import os
import smtplib
import threading
from email.mime.text import MIMEText
from pathlib import Path

# ── Error logger (writes to logs/notify_errors.log, never to stderr) ─────────

_LOG_PATH = Path("logs/notify_errors.log")


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("notify")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(_LOG_PATH)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    except Exception:
        # Last resort: NullHandler so nothing propagates to root logger
        logger.addHandler(logging.NullHandler())
    return logger


_logger = _get_logger()


# ── Slack ─────────────────────────────────────────────────────────────────────

def send_slack(message: str, webhook_url: str) -> bool:
    """POST a plain-text message to a Slack incoming webhook.

    Returns True on success, False on any error.
    Prefers the `requests` library; falls back to `urllib` if unavailable.
    """
    payload = {"text": message}

    # Try requests first
    try:
        import requests as _req  # type: ignore
        r = _req.post(webhook_url, json=payload, timeout=10)
        r.raise_for_status()
        return True
    except ImportError:
        pass  # fall through to urllib
    except Exception as exc:
        _logger.error("send_slack (requests) failed: %s", exc)
        return False

    # urllib fallback
    try:
        import json
        import urllib.request

        data = json.dumps(payload).encode("utf-8")
        req  = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        _logger.error("send_slack (urllib) failed: %s", exc)
        return False


# ── Email ─────────────────────────────────────────────────────────────────────

def send_email(subject: str, body: str) -> bool:
    """Send an email using SMTP env vars.

    Returns True on success, False on any error.
    """
    to_addr   = os.getenv("NOTIFY_EMAIL_TO", "").strip()
    from_addr = os.getenv("NOTIFY_EMAIL_FROM", "").strip()
    host      = os.getenv("NOTIFY_SMTP_HOST", "smtp.gmail.com").strip()
    password  = os.getenv("NOTIFY_SMTP_PASSWORD", "").strip()

    try:
        port = int(os.getenv("NOTIFY_SMTP_PORT", "587"))
    except ValueError:
        port = 587

    if not (to_addr and from_addr and password):
        _logger.debug("send_email: missing env vars (TO=%s FROM=%s PWD=%s)",
                      bool(to_addr), bool(from_addr), bool(password))
        return False

    try:
        msg           = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"]    = from_addr
        msg["To"]      = to_addr

        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(from_addr, password)
            smtp.sendmail(from_addr, [to_addr], msg.as_string())
        return True
    except Exception as exc:
        _logger.error("send_email failed: %s", exc)
        return False


# ── Configuration check ───────────────────────────────────────────────────────

def notify_configured() -> bool:
    """Return True if at least one notification channel is configured."""
    slack_ok = bool(os.getenv("NOTIFY_SLACK_WEBHOOK", "").strip())
    email_ok = bool(
        os.getenv("NOTIFY_EMAIL_TO", "").strip()
        and os.getenv("NOTIFY_EMAIL_FROM", "").strip()
        and os.getenv("NOTIFY_SMTP_PASSWORD", "").strip()
    )
    return slack_ok or email_ok


# ── Bet alert ─────────────────────────────────────────────────────────────────

def _format_bet_alert(
    bet_info: dict,
    p1: str,
    p2: str,
    surface: str,
    tournament: str,
    paper_trade: bool = False,
) -> str:
    """Build a human-readable bet-alert message."""
    edge      = bet_info.get("edge", 0.0)
    bet_on    = bet_info.get("bet_on")          # "p1" or "p2"
    best_odds = (
        bet_info.get("best_odds_p1") if bet_on == "p1"
        else bet_info.get("best_odds_p2")
    )
    best_book = (
        bet_info.get("best_book_p1") if bet_on == "p1"
        else bet_info.get("best_book_p2")
    )
    kelly_q   = (
        bet_info.get("kelly_p1", {}).get("quarter", 0.0) if bet_on == "p1"
        else bet_info.get("kelly_p2", {}).get("quarter", 0.0)
    )

    bet_player = p1 if bet_on == "p1" else p2
    odds_str   = f"{best_odds:.2f}" if best_odds else "N/A"
    book_str   = f" ({best_book})" if best_book else ""

    lines = [
        f"TENNIS BET ALERT: {p1} vs {p2}",
        f"Surface: {surface}  |  Tournament: {tournament}",
        f"Bet on: {bet_player} @ {odds_str}{book_str}",
        f"Edge: +{edge:.1%}  |  QKelly: {kelly_q:.2%}",
    ]
    if paper_trade:
        lines.append("[PAPER TRADE]")
    return "\n".join(lines)


def send_bet_alert(
    bet_info: dict,
    p1: str,
    p2: str,
    surface: str,
    tournament: str,
    paper_trade: bool = False,
) -> None:
    """Send a bet alert to all configured channels in a background thread.

    Returns immediately; notifications are fire-and-forget.
    Fails silently if no channels are configured.
    """
    if not notify_configured():
        _logger.debug("send_bet_alert: no notification channels configured; skipping.")
        return

    message = _format_bet_alert(bet_info, p1, p2, surface, tournament, paper_trade)
    subject = f"Bet Alert: {p1} vs {p2} — {surface}"

    def _send() -> None:
        webhook = os.getenv("NOTIFY_SLACK_WEBHOOK", "").strip()
        if webhook:
            ok = send_slack(message, webhook)
            if not ok:
                _logger.warning("Slack alert failed for %s vs %s", p1, p2)

        if os.getenv("NOTIFY_EMAIL_TO", "").strip():
            ok = send_email(subject, message)
            if not ok:
                _logger.warning("Email alert failed for %s vs %s", p1, p2)

    t = threading.Thread(target=_send, daemon=True)
    t.start()


# ── Test helper ───────────────────────────────────────────────────────────────

def test_notify() -> None:
    """Send a test message to all configured channels."""
    if not notify_configured():
        print("No notification channels configured.")
        print("Set NOTIFY_SLACK_WEBHOOK and/or NOTIFY_EMAIL_TO + NOTIFY_EMAIL_FROM "
              "+ NOTIFY_SMTP_PASSWORD.")
        return

    message = (
        "Tennis-odds notify test\n"
        "If you see this, your notification channel is working correctly."
    )

    webhook = os.getenv("NOTIFY_SLACK_WEBHOOK", "").strip()
    if webhook:
        ok = send_slack(message, webhook)
        print(f"Slack: {'OK' if ok else 'FAILED (see logs/notify_errors.log)'}")

    if os.getenv("NOTIFY_EMAIL_TO", "").strip():
        ok = send_email("Tennis-odds notify test", message)
        print(f"Email: {'OK' if ok else 'FAILED (see logs/notify_errors.log)'}")


# ── CLI entrypoint ────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the notify module's configured channels.",
        prog="python -m notify",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Send a test message to all configured channels.",
    )
    args = parser.parse_args()

    if args.test:
        test_notify()
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
