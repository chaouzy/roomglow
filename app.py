import base64
import hashlib
import hmac
import json
import os
import re
import time
from typing import Any, Dict

import requests as http
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 35 * 1024 * 1024

client = OpenAI()

AFFILIATE_TAG         = "technova050-20"
SUPABASE_URL          = os.environ.get("SUPABASE_URL", "")
SUPABASE_SECRET_KEY   = os.environ.get("SUPABASE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")


# ── Supabase helpers ───────────────────────────────────────────────────────

def _supa_headers():
    return {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
        "Content-Type": "application/json",
    }

def save_pro_user(email: str, stripe_customer_id: str = ""):
    url = f"{SUPABASE_URL}/rest/v1/pro_users"
    headers = {**_supa_headers(), "Prefer": "resolution=merge-duplicates"}
    payload = {"email": email.lower().strip(), "active": True}
    if stripe_customer_id:
        payload["stripe_customer_id"] = stripe_customer_id
    try:
        r = http.post(url, json=payload, headers=headers, timeout=10)
        return r.status_code in (200, 201)
    except Exception:
        return False

def is_pro_user(email: str) -> bool:
    email = email.lower().strip()
    url = f"{SUPABASE_URL}/rest/v1/pro_users?email=eq.{email}&active=eq.true&select=email"
    try:
        r = http.get(url, headers=_supa_headers(), timeout=10)
        data = r.json()
        return isinstance(data, list) and len(data) > 0
    except Exception:
        return False


# ── Stripe webhook verification ────────────────────────────────────────────

def verify_stripe_signature(payload: bytes, sig_header: str, secret: str) -> bool:
    try:
        parts = {}
        for part in sig_header.split(","):
            k, v = part.split("=", 1)
            parts[k.strip()] = v.strip()
        timestamp = int(parts.get("t", 0))
        v1_sig    = parts.get("v1", "")
        if abs(time.time() - timestamp) > 300:
            return False
        signed   = f"{timestamp}.".encode() + payload
        expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, v1_sig)
    except Exception:
        return False


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "RoomGlow API"})


@app.post("/webhook")
def stripe_webhook():
    payload    = request.get_data()
    sig_header = request.headers.get("Stripe-Signature", "")

    if not verify_stripe_signature(payload, sig_header, STRIPE_WEBHOOK_SECRET):
        return jsonify({"error": "Invalid signature"}), 400

    try:
        event = json.loads(payload)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    if event.get("type") == "checkout.session.completed":
        session         = event.get("data", {}).get("object", {})
        email           = (
            (session.get("customer_details") or {}).get("email")
            or session.get("customer_email")
            or ""
        )
        stripe_customer = session.get("customer", "")
        if email:
            save_pro_user(email, stripe_customer)

    return jsonify({"status": "ok"})


@app.get("/check-pro")
def check_pro():
    email = request.args.get("email", "").strip()
    if not email or "@" not in email:
        return jsonify({"pro": False, "error": "Invalid email"}), 400
    return jsonify({"pro": is_pro_user(email)})


# ── Prompt builder ─────────────────────────────────────────────────────────

def build_prompt(room_type: str) -> str:
    rt = room_type.strip() if isinstance(room_type, str) and room_type.strip() else "room"
    return (
        f"You are an expert interior designer and aesthetics consultant. Analyze this photo of a {rt} "
        "with deep attention to detail. Be specific about what you actually see — reference real objects, "
        "colors, and features visible in the photo. "
        "Return a single JSON object with EXACTLY these keys:\n\n"
        '"score": integer 1-100 (overall aesthetic score),\n\n'
        '"score_breakdown": object with exactly these 5 integer keys, each 0-20:\n'
        '  "lighting", "color_harmony", "furniture_arrangement", "decor_personality", "cleanliness_organization",\n\n'
        '"style": short string — the aesthetic style name,\n\n'
        '"style_description": string — exactly 2 sentences describing this style,\n\n'
        '"whats_working": string — 2-3 sentences of specific praise,\n\n'
        '"whats_not_working": string — 2-3 sentences about biggest opportunities,\n\n'
        '"style_report": string — 3-4 sentences of deeper personality analysis,\n\n'
        '"tips": array of exactly 5 objects, each with: title, tip, icon, category, difficulty, '
        'estimated_score_boost, product_name, product_description, product_price_range, product_search_url.\n\n'
        "Be hyper-specific. Output ONLY valid JSON. No markdown fences, no extra text."
    )


# ── JSON helpers ───────────────────────────────────────────────────────────

def _extract_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("Not text")
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = text.find("{"); end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    parsed = json.loads(text[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Not a dict")
    return parsed


def _amazon_url(query: str) -> str:
    q = re.sub(r"\s+", "+", query.strip())
    return f"https://www.amazon.com/s?k={q}&tag={AFFILIATE_TAG}"


def _ensure_affiliate_tag(url: str) -> str:
    if not isinstance(url, str) or not url.strip().startswith("http"):
        return url
    url = url.strip()
    if f"tag={AFFILIATE_TAG}" in url:
        return url
    if "tag=" in url:
        url = re.sub(r"tag=[^&]*", f"tag={AFFILIATE_TAG}", url)
    else:
        url += ("&" if "?" in url else "?") + f"tag={AFFILIATE_TAG}"
    return url


def _coerce_tip(obj: Any, idx: int) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        obj = {}
    def _str(key, fallback):
        v = obj.get(key)
        return v.strip() if isinstance(v, str) and v.strip() else fallback
    difficulty = obj.get("difficulty", "Medium")
    if difficulty not in ("Easy", "Medium", "Hard"):
        difficulty = "Medium"
    try:
        boost = max(3, min(12, int(obj.get("estimated_score_boost", 5))))
    except (TypeError, ValueError):
        boost = 5
    product_name = _str("product_name", "room decor accent")
    raw_url = obj.get("product_search_url", "")
    url = _ensure_affiliate_tag(raw_url.strip()) if isinstance(raw_url, str) and raw_url.strip().startswith("http") else _amazon_url(product_name)
    return {
        "title": _str("title", f"Upgrade Your Space — Tip {idx}"),
        "tip": _str("tip", f"Look for one small change that could make a big impact (tip {idx})."),
        "icon": _str("icon", "✨"),
        "category": _str("category", "Decor"),
        "difficulty": difficulty,
        "estimated_score_boost": boost,
        "product_name": product_name,
        "product_description": _str("product_description", "A great addition that will elevate your room."),
        "product_price_range": _str("product_price_range", "$20-50"),
        "product_search_url": url,
    }


def _coerce_score_breakdown(raw: Any, overall: int) -> Dict[str, int]:
    keys = ["lighting", "color_harmony", "furniture_arrangement", "decor_personality", "cleanliness_organization"]
    if not isinstance(raw, dict):
        base = overall // 5; rem = overall % 5
        result = {k: base for k in keys}
        for k in keys[:rem]: result[k] += 1
        return result
    result = {}
    for k in keys:
        try: result[k] = max(0, min(20, int(raw.get(k, overall // 5))))
        except (TypeError, ValueError): result[k] = overall // 5
    return result


def _coerce_response(data: Dict[str, Any]) -> Dict[str, Any]:
    try: score = max(1, min(100, int(data.get("score", 72))))
    except (TypeError, ValueError): score = 72
    style = data.get("style", "")
    style = style.strip() if isinstance(style, str) and style.strip() else "Modern Cozy"
    def _s(key, fallback):
        v = data.get(key, "")
        return v.strip() if isinstance(v, str) and v.strip() else fallback
    tips_raw = data.get("tips", [])
    tips = [_coerce_tip(item, i) for i, item in enumerate(tips_raw[:5], 1)] if isinstance(tips_raw, list) else []
    while len(tips) < 5:
        idx = len(tips) + 1
        tips.append(_coerce_tip({"title": f"Add a Cohesive Accent — Tip {idx}", "tip": "One element could tie your room together.", "icon": "✨", "category": "Decor", "difficulty": "Easy", "estimated_score_boost": 5, "product_name": "aesthetic room decor", "product_description": "Adds warmth and personality.", "product_price_range": "$15-40", "product_search_url": _amazon_url("aesthetic room decor")}, idx))
    return {
        "score": score,
        "score_breakdown": _coerce_score_breakdown(data.get("score_breakdown"), score),
        "style": style,
        "style_description": _s("style_description", f"Your room has a {style} aesthetic. With targeted upgrades it could feel even more cohesive."),
        "whats_working": _s("whats_working", "Your space has a solid foundation with thoughtful arrangement."),
        "whats_not_working": _s("whats_not_working", "There are a few key areas where small changes could make a big difference."),
        "style_report": _s("style_report", f"This {style} space reflects a personality that values comfort and intentionality."),
        "tips": tips[:5],
    }


@app.post("/analyze")
def analyze():
    if not request.is_json:
        return jsonify({"error": "Expected JSON payload."}), 400
    payload   = request.get_json(silent=True) or {}
    image_b64 = payload.get("image")
    mime      = payload.get("mime", "image/jpeg")
    room_type = payload.get("room_type", "Bedroom")
    if not isinstance(image_b64, str) or not image_b64.strip():
        return jsonify({"error": "Missing or invalid 'image' field."}), 400
    if not isinstance(mime, str) or not mime.startswith("image/"):
        mime = "image/jpeg"
    if not isinstance(room_type, str):
        room_type = "Bedroom"
    try: base64.b64decode(image_b64, validate=True)
    except Exception: return jsonify({"error": "Invalid base64."}), 400
    try:
        completion = client.chat.completions.create(
            model="gpt-4o", temperature=0.4, max_tokens=1800,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": build_prompt(room_type)},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
            ]}],
        )
    except Exception as e:
        return jsonify({"error": f"OpenAI request failed: {e}"}), 502
    try: content = completion.choices[0].message.content
    except Exception: content = None
    if not isinstance(content, str) or not content.strip():
        return jsonify({"error": "No response from model."}), 502
    try: result = _coerce_response(_extract_json_object(content))
    except Exception: result = _coerce_response({})
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
