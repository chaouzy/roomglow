import base64
import json
import re
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 35 * 1024 * 1024

client = OpenAI()

AFFILIATE_TAG = "technova050-20"


def build_prompt(room_type: str) -> str:
    rt = room_type.strip() if isinstance(room_type, str) and room_type.strip() else "room"
    return (
        f"You are an expert interior designer and aesthetics consultant. Analyze this photo of a {rt} "
        f"with deep attention to detail. Be specific about what you actually see — reference real objects, "
        f"colors, and features visible in the photo. "
        "Return a single JSON object with EXACTLY these keys:\n\n"

        '"score": integer 1-100 (overall aesthetic score),\n\n'

        '"score_breakdown": object with exactly these 5 integer keys, each 0-20:\n'
        '  "lighting" (quality and warmth of light sources visible),\n'
        '  "color_harmony" (how well colors work together),\n'
        '  "furniture_arrangement" (layout, spacing, flow),\n'
        '  "decor_personality" (unique character, styling, personal touches),\n'
        '  "cleanliness_organization" (tidiness and visual order),\n\n'

        '"style": short string — the aesthetic style name (e.g. "Warm Minimalist", "Dark Academia", "Boho Chic"),\n\n'

        '"style_description": string — exactly 2 sentences describing this style and how this room embodies it,\n\n'

        '"whats_working": string — 2-3 sentences of genuine specific praise about what already looks good. '
        'Reference actual things you see in the image.\n\n'

        '"tips": array of exactly 5 objects. Each must have:\n'
        '  "title": string, punchy tip title 5-7 words,\n'
        '  "tip": string, 3-4 sentences of specific actionable advice referencing things visible in the image,\n'
        '  "icon": string, one relevant emoji,\n'
        '  "category": one of: Lighting | Color | Furniture | Decor | Organization | Texture,\n'
        '  "difficulty": one of: Easy | Medium | Hard,\n'
        '  "estimated_score_boost": integer 3-12,\n'
        '  "product_name": string, specific product type (e.g. "Warm LED Floor Lamp"),\n'
        '  "product_description": string, one sentence why this product fits this specific room,\n'
        '  "product_price_range": string, realistic price range (e.g. "$25-45"),\n'
        '  "product_search_url": Amazon URL https://www.amazon.com/s?k=search+terms+here&tag=technova050-20\n\n'

        "Be hyper-specific — reference actual colors, objects, and layout you see. "
        "The 5 score_breakdown values should roughly sum to the overall score. "
        "Output ONLY valid JSON. No markdown fences, no extra text whatsoever."
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("Model output was not text.")
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate a JSON object in the model output.")
    parsed = json.loads(text[start: end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON was not an object.")
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

    title = _str("title", f"Upgrade Your Space — Tip {idx}")
    tip = _str("tip", f"Look for one small change that could make a big impact in your room (tip {idx}).")
    icon = _str("icon", "✨")
    category = _str("category", "Decor")
    difficulty = obj.get("difficulty", "Medium")
    if difficulty not in ("Easy", "Medium", "Hard"):
        difficulty = "Medium"

    try:
        boost = int(obj.get("estimated_score_boost", 5))
        boost = max(3, min(12, boost))
    except (TypeError, ValueError):
        boost = 5

    product_name = _str("product_name", "room decor accent")
    product_desc = _str("product_description", "A great addition that will elevate the feel of your room.")
    price_range = _str("product_price_range", "$20-50")

    raw_url = obj.get("product_search_url", "")
    if isinstance(raw_url, str) and raw_url.strip().startswith("http"):
        url = _ensure_affiliate_tag(raw_url.strip())
    else:
        url = _amazon_url(product_name)

    return {
        "title": title,
        "tip": tip,
        "icon": icon,
        "category": category,
        "difficulty": difficulty,
        "estimated_score_boost": boost,
        "product_name": product_name,
        "product_description": product_desc,
        "product_price_range": price_range,
        "product_search_url": url,
    }


def _coerce_score_breakdown(raw: Any, overall: int) -> Dict[str, int]:
    keys = ["lighting", "color_harmony", "furniture_arrangement", "decor_personality", "cleanliness_organization"]
    if not isinstance(raw, dict):
        base = overall // 5
        rem = overall % 5
        result = {k: base for k in keys}
        for k in keys[:rem]:
            result[k] += 1
        return result
    result = {}
    for k in keys:
        try:
            v = max(0, min(20, int(raw.get(k, overall // 5))))
        except (TypeError, ValueError):
            v = overall // 5
        result[k] = v
    return result


def _coerce_response(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        score = max(1, min(100, int(data.get("score", 72))))
    except (TypeError, ValueError):
        score = 72

    style = data.get("style", "")
    style = style.strip() if isinstance(style, str) and style.strip() else "Modern Cozy"

    style_desc = data.get("style_description", "")
    style_desc = (
        style_desc.strip()
        if isinstance(style_desc, str) and style_desc.strip()
        else f"Your room has a {style} aesthetic with its own unique character. With a few targeted upgrades, it could feel even more cohesive and intentional."
    )

    whats_working = data.get("whats_working", "")
    whats_working = (
        whats_working.strip()
        if isinstance(whats_working, str) and whats_working.strip()
        else "Your space already has a solid foundation to build from. The overall layout shows thoughtful arrangement and there is real potential to elevate it further."
    )

    score_breakdown = _coerce_score_breakdown(data.get("score_breakdown"), score)

    tips_raw = data.get("tips", [])
    tips = []
    if isinstance(tips_raw, list):
        for i, item in enumerate(tips_raw[:5], start=1):
            tips.append(_coerce_tip(item, i))

    while len(tips) < 5:
        idx = len(tips) + 1
        tips.append(_coerce_tip({
            "title": f"Add a Cohesive Accent — Tip {idx}",
            "tip": "Look for one element — lighting, texture, or color — that could tie your room together more intentionally.",
            "icon": "✨",
            "category": "Decor",
            "difficulty": "Easy",
            "estimated_score_boost": 5,
            "product_name": "aesthetic room decor accent",
            "product_description": "A versatile piece that adds warmth and personality to your space.",
            "product_price_range": "$15-40",
            "product_search_url": _amazon_url("aesthetic room decor"),
        }, idx))

    return {
        "score": score,
        "score_breakdown": score_breakdown,
        "style": style,
        "style_description": style_desc,
        "whats_working": whats_working,
        "tips": tips[:5],
    }


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "RoomGlow API"})


@app.post("/analyze")
def analyze():
    if not request.is_json:
        return jsonify({"error": "Expected JSON payload."}), 400

    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image")
    mime = payload.get("mime", "image/jpeg")
    room_type = payload.get("room_type", "Bedroom")

    if not isinstance(image_b64, str) or not image_b64.strip():
        return jsonify({"error": "Missing or invalid 'image' field."}), 400

    if not isinstance(mime, str) or not mime.startswith("image/"):
        mime = "image/jpeg"

    if not isinstance(room_type, str):
        room_type = "Bedroom"

    try:
        base64.b64decode(image_b64, validate=True)
    except Exception:
        return jsonify({"error": "The provided image was not valid base64."}), 400

    data_url = f"data:{mime};base64,{image_b64}"
    prompt = build_prompt(room_type)

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.4,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
    except Exception as e:
        return jsonify({"error": f"OpenAI request failed: {e}"}), 502

    content = None
    try:
        content = completion.choices[0].message.content
    except Exception:
        content = None

    if not isinstance(content, str) or not content.strip():
        return jsonify({"error": "No usable response from the model."}), 502

    try:
        obj = _extract_json_object(content)
        result = _coerce_response(obj)
    except Exception:
        result = _coerce_response({})

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
