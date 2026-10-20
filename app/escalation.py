"""Human handoff: send a conversation summary to an n8n webhook (ticket, Slack, email)."""

from __future__ import annotations

import logging
import time

import httpx

log = logging.getLogger("bot.escalation")


def notify(url: str, secret: str, payload: dict) -> bool:
    """POST with retries. Returns True on success. Never raises (runs as a background task)."""
    if not url:
        log.warning("escalation requested but BOT_ESCALATION_WEBHOOK_URL is not set", extra={"reason": payload.get("reason")})
        return False
    headers = {"X-Webhook-Secret": secret} if secret else {}
    for attempt in range(3):
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=8.0)
            if r.status_code < 300:
                return True
            log.warning("escalation webhook returned %s", r.status_code)
        except httpx.HTTPError as exc:
            log.warning("escalation webhook error: %s", type(exc).__name__)
        time.sleep(1.5 * (attempt + 1))
    log.error("escalation failed after retries")
    return False
