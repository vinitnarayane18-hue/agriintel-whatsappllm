"""
whatsapp.py
───────────
WhatsApp Cloud API wrapper for the AgriIntel farmer agent.
Called by main.py — handles ALL WhatsApp in/out. No AI, no business logic here
(same philosophy as delivery.py: pure I/O, deterministic).

Responsibilities:
    1. Webhook handshake verification (GET  /webhook)
    2. Webhook payload signature verification (POST /webhook, X-Hub-Signature-256)
    3. Parsing raw Meta payloads into a normalized message dict
    4. Deduplication — Meta retries webhook delivery; without this a farmer
       could get charged/replied to twice for one message (critical given
       x402 + Razorpay payments downstream).
    5. Sending messages: text, interactive buttons, CTA-URL (payment links),
       location-request (native "share location" button instead of asking
       the farmer to type an address).

CRITICAL RULES (mirrors x402_client.py conventions — do not break):
    - All Graph API calls are async httpx, with a shared timeout.
    - Never trust an incoming webhook without verify_signature() passing.
    - WHATSAPP_APP_SECRET, WHATSAPP_TOKEN, WHATSAPP_VERIFY_TOKEN come from
      env vars only — never hardcode.
    - dedup uses the SAME Redis instance as session_store.py (get_redis_client)
      so we don't spin up a second connection pool.
"""

import hashlib
import hmac
import logging
import os
from typing import Optional

import httpx

from session_store import get_redis_client

logger = logging.getLogger(__name__)

# ─── Config (env-only, same pattern as x402_client.py / consent_log.py) ──────

GRAPH_API_VERSION      = os.getenv("GRAPH_API_VERSION", "v23.0")
WHATSAPP_TOKEN          = os.getenv("WHATSAPP_TOKEN", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_VERIFY_TOKEN   = os.getenv("WHATSAPP_VERIFY_TOKEN", "")
WHATSAPP_APP_SECRET     = os.getenv("WHATSAPP_APP_SECRET", "")

GRAPH_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"

_DEDUP_TTL_SECONDS = 3600  # 1hr — Meta's retry window is well under this


# ─── Shared error response builder (same shape as x402_client._err) ─────────

def _err(error_type: str, reason: str, status_code: int = 0) -> dict:
    payload = {"error": True, "error_type": error_type, "error_reason": reason}
    if status_code:
        payload["status_code"] = status_code
    return payload


# ─── 1. Webhook handshake (GET /webhook) ─────────────────────────────────────

def verify_webhook_subscription(
    mode: Optional[str], token: Optional[str], challenge: Optional[str]
) -> Optional[str]:
    """
    Meta calls GET /webhook once, at setup time, with:
        hub.mode=subscribe, hub.verify_token=<yours>, hub.challenge=<random>
    Return the challenge string as-is (main.py returns it as plain text,
    status 200) if the token matches. Return None to reject (main.py
    should respond 403).
    """
    if mode == "subscribe" and token and WHATSAPP_VERIFY_TOKEN and hmac.compare_digest(
        token, WHATSAPP_VERIFY_TOKEN
    ):
        logger.info("[WhatsApp] Webhook verification succeeded")
        return challenge
    logger.warning(f"[WhatsApp] Webhook verification failed (mode={mode})")
    return None


# ─── 2. Payload signature verification (POST /webhook) ──────────────────────

def verify_signature(raw_body: bytes, signature_header: Optional[str]) -> bool:
    """
    Meta signs every webhook POST with X-Hub-Signature-256: sha256=<hex>,
    computed as HMAC-SHA256(app_secret, raw_body).

    MUST be called with the RAW request body (bytes), before any JSON
    parsing — main.py has to read the body as bytes first, not request.json().

    Without this check, anyone who finds your webhook URL could POST fake
    "payment succeeded" or fake farmer messages.
    """
    if not signature_header or not signature_header.startswith("sha256="):
        logger.warning("[WhatsApp] Missing/malformed signature header")
        return False

    if not WHATSAPP_APP_SECRET:
        logger.error("[WhatsApp] WHATSAPP_APP_SECRET not set — cannot verify signature")
        return False

    expected = hmac.new(
        WHATSAPP_APP_SECRET.encode("utf-8"), raw_body, hashlib.sha256
    ).hexdigest()
    received = signature_header.split("sha256=", 1)[1]

    valid = hmac.compare_digest(expected, received)
    if not valid:
        logger.warning("[WhatsApp] Signature mismatch — rejecting payload")
    return valid


# ─── 3. Deduplication (Meta retries webhook delivery) ────────────────────────

def is_duplicate_message(message_id: str) -> bool:
    """
    Meta may deliver the same webhook event more than once (retries on
    slow/failed 200 responses). Returns True if we've already seen this
    message_id in the last hour — main.py should skip processing.

    Uses SET NX (atomic) so two concurrent requests can't both pass.
    """
    if not message_id:
        return False
    try:
        client = get_redis_client()
        key = f"wa_seen:{message_id}"
        was_set = client.set(key, "1", nx=True, ex=_DEDUP_TTL_SECONDS)
        return not was_set  # if set failed (key existed), it's a duplicate
    except Exception as e:
        logger.warning(f"[WhatsApp] Dedup check failed, processing anyway: {e}")
        return False  # fail-open — better to risk a dupe than drop a real message


# ─── 4. Parsing incoming payloads ────────────────────────────────────────────

def parse_incoming_webhook(payload: dict) -> Optional[dict]:
    """
    Normalizes Meta's nested webhook JSON into a flat dict main.py/orchestrator
    can consume directly:

        {
            "message_id": str,
            "phone": str,                 # sender's WhatsApp number
            "timestamp": str,
            "type": "text"|"location"|"button_reply"|"list_reply"|"unsupported",
            "text": str or None,
            "lat": float or None,
            "lon": float or None,
            "reply_id": str or None,      # button/list id if applicable
            "reply_title": str or None,
        }

    Returns None for non-message events (status updates like "delivered",
    "read" receipts — Meta sends those on the same webhook URL, and they
    are noise for us, not farmer messages).
    """
    try:
        entry = payload.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})

        messages = value.get("messages")
        if not messages:
            # This was a status callback (sent/delivered/read), not a message
            return None

        msg = messages[0]
        msg_type = msg.get("type")
        phone = msg.get("from")
        message_id = msg.get("id")
        timestamp = msg.get("timestamp")

        parsed = {
            "message_id": message_id,
            "phone": phone,
            "timestamp": timestamp,
            "type": msg_type,
            "text": None,
            "lat": None,
            "lon": None,
            "reply_id": None,
            "reply_title": None,
        }

        if msg_type == "text":
            parsed["text"] = msg.get("text", {}).get("body")

        elif msg_type == "location":
            loc = msg.get("location", {})
            parsed["lat"] = loc.get("latitude")
            parsed["lon"] = loc.get("longitude")

        elif msg_type == "interactive":
            interactive = msg.get("interactive", {})
            itype = interactive.get("type")
            if itype == "button_reply":
                parsed["type"] = "button_reply"
                parsed["reply_id"] = interactive["button_reply"]["id"]
                parsed["reply_title"] = interactive["button_reply"]["title"]
            elif itype == "list_reply":
                parsed["type"] = "list_reply"
                parsed["reply_id"] = interactive["list_reply"]["id"]
                parsed["reply_title"] = interactive["list_reply"]["title"]

        else:
            # image/audio/video/document/sticker/contacts/unsupported etc.
            parsed["type"] = "unsupported"

        return parsed

    except (IndexError, KeyError, TypeError) as e:
        logger.warning(f"[WhatsApp] Could not parse webhook payload: {e}")
        return None


# ─── Internal: shared POST helper ────────────────────────────────────────────

async def _post(payload: dict) -> dict:
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        return _err("CONFIG_ERROR", "WHATSAPP_TOKEN / WHATSAPP_PHONE_NUMBER_ID not set")

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            response = await http.post(GRAPH_URL, headers=headers, json=payload)

        if response.status_code == 200:
            return response.json()

        logger.error(f"[WhatsApp] Send failed {response.status_code}: {response.text}")
        return _err("SEND_FAILED", response.text, response.status_code)

    except Exception as e:
        logger.error(f"[WhatsApp] Send pipeline error: {e}")
        return _err("PIPELINE_ERROR", str(e))


# ─── 5a. Read receipts (good UX — farmer sees blue tick, knows bot got it) ──

async def mark_as_read(message_id: str) -> dict:
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id,
    }
    return await _post(payload)


# ─── 5b. Plain text ───────────────────────────────────────────────────────────

async def send_text(to: str, body: str, preview_url: bool = False) -> dict:
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"body": body, "preview_url": preview_url},
    }
    return await _post(payload)


# ─── 5c. Reply buttons (max 3) — for service/language clarification ─────────

async def send_buttons(
    to: str,
    body_text: str,
    buttons: list[tuple[str, str]],
    header_text: Optional[str] = None,
    footer_text: Optional[str] = None,
) -> dict:
    """
    buttons: list of (id, title) tuples, max 3, title max 20 chars (WhatsApp limit).
    Example: send_buttons(to, "Which service?",
                           [("mandi", "मंडी भाव"), ("weather", "हवामान")])
    """
    if len(buttons) > 3:
        logger.warning("[WhatsApp] >3 buttons given, truncating to 3 (WhatsApp limit)")
        buttons = buttons[:3]

    interactive: dict = {
        "type": "button",
        "body": {"text": body_text},
        "action": {
            "buttons": [
                {"type": "reply", "reply": {"id": bid, "title": title[:20]}}
                for bid, title in buttons
            ]
        },
    }
    if header_text:
        interactive["header"] = {"type": "text", "text": header_text}
    if footer_text:
        interactive["footer"] = {"text": footer_text}

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "interactive",
        "interactive": interactive,
    }
    return await _post(payload)


# ─── 5d. CTA-URL button — for Razorpay payment links ────────────────────────

async def send_payment_link(
    to: str, body_text: str, button_label: str, url: str, header_text: Optional[str] = None
) -> dict:
    """
    Sends the payment link as a tappable button instead of a raw URL in text
    — farmers are far more likely to tap a clean button than a long rzp.io
    link pasted in a paragraph. razorpay_handler.py should call this (not
    send_text) once it generates the payment link.
    """
    interactive: dict = {
        "type": "cta_url",
        "body": {"text": body_text},
        "action": {
            "name": "cta_url",
            "parameters": {"display_text": button_label[:20], "url": url},
        },
    }
    if header_text:
        interactive["header"] = {"type": "text", "text": header_text}

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "interactive",
        "interactive": interactive,
    }
    return await _post(payload)


# ─── 5e. Location request — native "share location" button ─────────────────

async def send_location_request(to: str, body_text: str) -> dict:
    """
    Sends a one-tap "Send Location" button instead of asking the farmer to
    type/paste coordinates. orchestrator.py's ack messages currently say
    "📍 share your location" as plain text — main.py should call THIS
    function right after sending that ack, so the farmer gets a real button,
    not just an instruction.
    """
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "interactive",
        "interactive": {
            "type": "location_request_message",
            "body": {"text": body_text},
            "action": {"name": "send_location"},
        },
    }
    return await _post(payload)
