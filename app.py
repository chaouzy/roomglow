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


def build_prompt(room_type: str) -> str:
    rt = room_type.strip() if isinstance(room_type, str) and room_type.strip() else "room"
    return (
        f"You are an interior design and aesthetics expert. Analyze this photo of a {rt} "
        f"and give 5 specific aesthetic improvement tips tailored to what you see. "
        "Return a single JSON object with exactly these keys: "
        '"score": an integer from 1 to 100 (overall aesthetic score), '
        '"style": a short string describing the aesthetic (e.g. "Warm Minimalist", "Dark Academia"), '
        '"tips": an array of exactly 5 objects. Each object must have: '
        '"tip" (string, actionable friendly advice), '
        '"icon" (string, one relevant emoji), '
        '"category" (string, e.g. Lighting, Color, Texture), '
        '"product_name" (string, a specific product type to shop for), '
        '"product_search_url" (string, a real Amazon search URL using https://www.amazon.com/s?k= '
        "with query terms joined by + for spaces, e.g. https://www.amazon.com/s?k=warm+floor+lamp). "
        "Be specific to what is visible in the image. "
        "Output ONLY valid JSON, no markdown fences or extra text."
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("Model output was not text.")
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
    candidate = text[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON was not an object.")
    return parsed


def _amazon_search_url(query: str) -> str:
    q = re.sub(r"\s+", "+", query.strip())
    return f"https://www.amazon.com/s?k={q}"


def _coerce_tip(obj: Any, idx: int) -> Dict[str, str]:
    if not isinstance(obj, dict):
        obj = {}
    tip = obj.get("tip")
    icon = obj.get("icon")
    category = obj.get("category")
    product_name = obj.get("product_name")
    url = obj.get("product_search_url")

    tip_s = tip.strip() if isinstance(tip, str) else f"Improve one detail in your space (tip {idx})."
    icon_s = icon.strip() if isinstance(icon, str) and icon.strip() else "✨"
    cat_s = category.strip() if isinstance(category, str) and category.strip() else "Aesthetic"
    prod_s = (
        product_name.strip()
        if isinstance(product_name, str) and product_name.strip()
        else "room decor accent"
    )
    if isinstance(url, str) and url.strip().startswith("http"):
        url_s = url.strip()
    else:
        url_s = _amazon_search_url(prod_s)

    return {
        "tip": tip_s,
        "icon": icon_s,
        "category": cat_s,
        "product_name": prod_s,
        "product_search_url": url_s,
    }


def _coerce_response(data: Dict[str, Any]) -> Dict[str, Any]:
    score_raw = data.get("score")
    try:
        score = int(score_raw)
    except (TypeError, ValueError):
        score = 72
    score = max(1, min(100, score))

    style = data.get("style")
    style_s = style.strip() if isinstance(style, str) and style.strip() else "Modern Cozy"

    tips_raw = data.get("tips")
    tips: List[Dict[str, str]] = []
    if isinstance(tips_raw, list):
        for i, item in enumerate(tips_raw[:5], start=1):
            tips.append(_coerce_tip(item, i))

    while len(tips) < 5:
        idx = len(tips) + 1
        tips.append(
            _coerce_tip(
                {
                    "tip": f"Add one cohesive upgrade that matches your space (lighting, texture, or color). Tip {idx}.",
                    "icon": "✨",
                    "category": "Style",
                    "product_name": "decorative accent",
                    "product_search_url": _amazon_search_url("aesthetic room decor"),
                },
                idx,
            )
        )

    return {"score": score, "style": style_s, "tips": tips[:5]}


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
            temperature=0.2,
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
