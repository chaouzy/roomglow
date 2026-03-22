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
    url = f"{SUPABASE_URL}/rest/v1/pro_users?on_conflict=email"
    headers = {**_supa_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"}
    payload = {"email": email.lower().strip(), "active": True}
    if stripe_customer_id:
        payload["stripe_customer_id"] = stripe_customer_id
    try:
        r = http.post(url, json=payload, headers=headers, timeout=10)
        return r.status_code in (200, 201)
    except Exception:
        return False

def deactivate_pro_user(stripe_customer_id: str):
    """Mark user inactive when subscription is cancelled."""
    if not stripe_customer_id:
        return False
    url = f"{SUPABASE_URL}/rest/v1/pro_users?stripe_customer_id=eq.{stripe_customer_id}"
    try:
        r = http.patch(url, json={"active": False}, headers=_supa_headers(), timeout=10)
        return r.status_code in (200, 204)
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

    elif event.get("type") in ("customer.subscription.deleted", "customer.subscription.updated"):
        sub = event.get("data", {}).get("object", {})
        status = sub.get("status", "")
        if status in ("canceled", "unpaid", "past_due"):
            stripe_customer = sub.get("customer", "")
            if stripe_customer:
                deactivate_pro_user(stripe_customer)

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
        f"You are a brutally honest but genuinely helpful interior designer with a sharp eye and a great sense of humor. "
        f"You're analyzing a photo of someone's {rt}. Your feedback is specific, vivid, and personal — not generic. "
        f"You reference actual objects, colors, textures and layout details you can literally see in the photo. "
        f"Your tone is like a brutally honest friend who happens to be a professional designer — warm but direct. "
        f"People should want to screenshot your analysis and share it. Make it feel personal and real.\n\n"

        "Return a single JSON object with EXACTLY these keys:\n\n"

        '"score": integer 1-100. Be honest — most rooms score 45-75. A truly exceptional room scores 85+. '
        'A genuinely bad room can score below 40. Do not be generous.\n\n'

        '"score_breakdown": object with exactly these 5 integer keys, each 0-20:\n'
        '  "lighting" — quality, warmth and placement of light sources you can see,\n'
        '  "color_harmony" — how well the colors work together,\n'
        '  "furniture_arrangement" — layout, proportion and flow,\n'
        '  "decor_personality" — character, intentionality, personal style,\n'
        '  "cleanliness_organization" — visual tidiness and order.\n'
        'The 5 values should roughly sum to the overall score.\n\n'

        '"style": the aesthetic style name — be specific and creative. Not just "Modern" but "Moody Japandi" '
        'or "Chaotic Maximalist" or "Accidental Minimalist" or "IKEA Transitional" or "Dark Academia Adjacent". '
        'Make it feel like a personality type result people want to share.\n\n'

        '"style_description": exactly 2 sentences. First sentence: what this style actually is. '
        'Second sentence: how this specific room embodies it — reference something you can literally see.\n\n'

        '"verdict": one punchy sentence — the single most important thing about this room. '
        'Like "This room is 90% of the way there but one ceiling light is sabotaging everything." '
        'Or "Strong bones, weak follow-through." Make it quotable and shareable.\n\n'

        '"whats_working": 2-3 sentences of genuine, specific praise. Reference actual things you see. '
        'Not "good color choices" but "the warm terracotta throw against the grey sofa creates exactly the contrast this space needed." '
        'Make the person feel good about something real.\n\n'

        '"whats_not_working": 2-3 sentences. Be direct but constructive. Reference specific things you can see. '
        'Not "the lighting could be better" but "that overhead light is flattening every shadow in the room and '
        'making it feel like a waiting room instead of a bedroom." This is the part people screenshot.\n\n'

        '"style_report": 3-4 sentences of deeper personality analysis. What does this room say about the person '
        'who lives here? What are they going for, and how close are they to nailing it? '
        'Be insightful and a little playful. This is the Pro locked section — make it feel worth unlocking.\n\n'

        '"tips": array of exactly 5 objects. Each tip must:\n'
        '- Reference something specific you can actually see in the photo\n'
        '- Give concrete, actionable advice (not "add more lighting" but "replace the overhead with a '
        'dimmable arc floor lamp positioned behind the seating area")\n'
        '- Mention a specific product type with a real price range\n'
        '- Feel achievable, not overwhelming\n\n'
        'Each tip object must have:\n'
        '  "title": punchy 5-7 word title,\n'
        '  "tip": 3-4 sentences of specific advice referencing what you see,\n'
        '  "icon": one relevant emoji,\n'
        '  "category": one of: Lighting | Color | Furniture | Decor | Organization | Texture,\n'
        '  "difficulty": one of: Easy | Medium | Hard,\n'
        '  "estimated_score_boost": integer 3-12 (be realistic),\n'
        '  "product_name": specific product type (e.g. "Dimmable Arc Floor Lamp" not just "lamp"),\n'
        '  "product_description": one sentence why THIS specific product fits THIS specific room,\n'
        '  "product_price_range": realistic price range (e.g. "$35-65"),\n'
        '  "product_search_url": https://www.amazon.com/s?k=specific+search+terms+here&tag=technova050-20\n\n'

        "CRITICAL RULES:\n"
        "- Be hyper-specific — if you see a grey sofa, say 'grey sofa'. If you see string lights, say 'string lights'.\n"
        "- The verdict and whats_not_working fields should be memorable and quotable.\n"
        "- The style name should feel like a personality type result.\n"
        "- Amazon search URLs must use specific terms that would actually find the right product.\n"
        "- Output ONLY valid JSON. No markdown fences, no extra text whatsoever."
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
        "verdict": _s("verdict", f"This {style} space has real potential — a few targeted changes would make a significant difference."),
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
    # Pro users get GPT-4o, free users get GPT-4o-mini
    # This creates a real quality difference worth upgrading for
    is_pro = payload.get("is_pro", False)
    model  = "gpt-4o" if is_pro else "gpt-4o-mini"

    try:
        completion = client.chat.completions.create(
            model=model, temperature=0.4, max_tokens=1800,
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


REDESIGN_STYLES = {
    "scandinavian": "Scandinavian minimal style — clean white walls, light oak wood floors, simple furniture, lots of natural light, neutral tones with small green plants",
    "modern":       "Modern luxury style — sleek dark surfaces, statement lighting, high-end furniture, rich textures, moody atmosphere",
    "cozy":         "Cozy cottage style — warm earthy tones, soft textiles, candles, wooden furniture, layered rugs, warm amber lighting",
    "dark_academia": "Dark academia style — deep greens and browns, bookshelves, vintage furniture, atmospheric moody lighting, rich wood tones",
    "industrial":   "Modern industrial style — exposed brick, concrete, metal accents, Edison bulb lighting, raw materials with warm touches",
    "japandi":      "Japandi style — minimalist Japanese-Scandinavian fusion, neutral palette, natural materials, zen atmosphere, no clutter",
}

@app.post("/redesign")
def redesign():
    """Generate an AI room redesign for Pro users only."""
    if not request.is_json:
        return jsonify({"error": "Expected JSON payload."}), 400

    payload   = request.get_json(silent=True) or {}
    image_b64 = payload.get("image")
    mime      = payload.get("mime", "image/jpeg")
    room_type = payload.get("room_type", "room")
    style_key = payload.get("style", "scandinavian")
    is_pro    = payload.get("is_pro", False)

    if not is_pro:
        return jsonify({"error": "Pro required"}), 403

    if not isinstance(image_b64, str) or not image_b64.strip():
        return jsonify({"error": "Missing image."}), 400

    if not isinstance(mime, str) or not mime.startswith("image/"):
        mime = "image/jpeg"

    style_desc = REDESIGN_STYLES.get(style_key, REDESIGN_STYLES["scandinavian"])

    # Use GPT-4o-mini vision to describe the room first, then generate redesign
    # This gives DALL-E much better context about the actual room layout
    try:
        vision = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Describe the layout of this {room_type} in 2-3 sentences: room dimensions, window placement, main furniture positions. Be specific and brief."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                ]
            }]
        )
        room_desc = vision.choices[0].message.content.strip()
    except Exception:
        room_desc = f"a {room_type}"

    # Generate the redesigned room with DALL-E 3
    dalle_prompt = (
        f"Interior design photo of {room_desc}. "
        f"Redesigned in {style_desc}. "
        f"Professional interior photography, beautiful lighting, magazine quality, "
        f"photorealistic, high resolution. No people."
    )

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="b64_json",
        )
        image_data = response.data[0].b64_json
        return jsonify({"image": image_data, "style": style_key})
    except Exception as e:
        return jsonify({"error": f"Image generation failed: {e}"}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

