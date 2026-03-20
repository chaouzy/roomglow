import base64
import json
import os
from typing import Any, List

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
CORS(app)

# JSON + base64 overhead makes payloads larger than the raw image file.
app.config["MAX_CONTENT_LENGTH"] = 35 * 1024 * 1024  # ~35MB request body cap

client = OpenAI()

PROMPT = (
    "You are an interior design and aesthetics expert. Analyze this photo of a living space and provide "
    "exactly 5 specific, actionable, and friendly tips to make it more aesthetic. Format your response as a "
    "JSON array of 5 objects, each with a 'tip' field and an 'icon' field (use a relevant emoji as the icon). "
    "Be specific to what you actually see in the image."
)


def _extract_json_array(text: str) -> List[Any]:
    """Parse a JSON array from model output (tolerates leading/trailing text)."""
    if not isinstance(text, str):
        raise ValueError("Model output was not text.")

    # First try: raw JSON.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Fallback: find the first complete [...] block.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not locate a JSON array in the model output.")

    candidate = text[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, list):
        raise ValueError("Parsed JSON was not an array.")
    return parsed


def _coerce_tips(items: Any) -> List[dict]:
    tips: List[dict] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            tip = item.get("tip")
            icon = item.get("icon")
            if isinstance(tip, str) and isinstance(icon, str) and tip.strip() and icon.strip():
                tips.append({"tip": tip.strip(), "icon": icon.strip()})
            if len(tips) >= 5:
                break

    # If the model didn't follow instructions perfectly, pad deterministically.
    while len(tips) < 5:
        idx = len(tips) + 1
        tips.append(
            {
                "tip": f"Refine your space by focusing on one cohesive change at a time (consider lighting, color harmony, or texture). Tip {idx}.",
                "icon": "✨",
            }
        )
    return tips[:5]


@app.post("/analyze")
def analyze():
    if not request.is_json:
        return jsonify({"error": "Expected JSON payload."}), 400

    payload = request.get_json(silent=True) or {}
    image_b64 = payload.get("image")
    mime = payload.get("mime", "image/jpeg")

    if not isinstance(image_b64, str) or not image_b64.strip():
        return jsonify({"error": "Missing or invalid 'image' field."}), 400

    if not isinstance(mime, str) or not mime.startswith("image/"):
        mime = "image/jpeg"

    # Basic sanity check that base64 decodes.
    try:
        base64.b64decode(image_b64, validate=True)
    except Exception:
        return jsonify({"error": "The provided image was not valid base64."}), 400

    data_url = f"data:{mime};base64,{image_b64}"

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
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
        items = _extract_json_array(content)
        tips = _coerce_tips(items)
    except Exception:
        # Keep the API stable: return 5 tips even if parsing fails.
        tips = _coerce_tips([])

    return jsonify(tips)


if __name__ == "__main__":
    # Local dev server. Frontend calls http://localhost:5000/analyze.
    app.run(host="0.0.0.0", port=5000, debug=True)

