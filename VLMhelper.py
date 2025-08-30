# VLMhelper.py
import os, base64, re, time
from typing import Dict, Any
import openai
import json

# Single-line, pipe-delimited spec
PROMPT_TEXT = (
    "You are an image analysis agent for chest X-rays (CXR).\n"
    "Return exactly ONE line, fields separated by '|', wrapped by markers:\n"
    "===LINE===\n"
    "<LABEL>|<CONF>|<EXPLANATION>\n"
    "===END===\n"
    "Definitions:\n"
    "- CONF is a directional confidence in [0.0, 1.0]: 0.0 means definitely NO pulmonary edema; 1.0 means definitely YES pulmonary edema.\n"
    "- LABEL must equal 1 if CONF >= 0.5, otherwise 0.\n"
    "- EXPLANATION is a short phrase WITHOUT the '|' character.\n"
    "Examples:\n"
    "===LINE===\n"
    "1|0.87|bilateral perihilar haze and interstitial markings\n"
    "===END===\n"
    "===LINE===\n"
    "0|0.12|clear lungs without interstitial congestion\n"
    "===END==="
)

def _encode_image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def _messages_for_prompt(data_url: str):
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT_TEXT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }]

_LINE_RE = re.compile(r"^\s*([01])\s*\|\s*([0-9]*\.?[0-9]+)\s*\|\s*(.+?)\s*$")

def _extract_line(text: str) -> str | None:
    try:
        start = text.index("===LINE===") + len("===LINE===")
        end   = text.index("===END===")
        return text[start:end].strip()
    except ValueError:
        return None

def _parse_line(line: str) -> Dict[str, Any] | None:
    m = _LINE_RE.match(line)
    if not m:
        return None
    label = int(m.group(1))
    conf  = float(m.group(2))
    expl  = m.group(3)
    if not (0.0 <= conf <= 1.0):
        return None
    if "|" in expl:
        return None
    return {"label": label, "confidence": conf, "explanation": expl}

def vlm_diag_line(
    image_path: str,
    model: str = "gpt-4.1-mini",
    max_retries: int = 3,
    retry_delay: float = 5.0
) -> Dict[str, Any]:
    """
    Run a VLM and return:
      {
        'ok': bool,
        'model': str,
        'raw': str|None,          # full raw response
        'label': int|None,        # 0/1
        'confidence': float|None, # 0..1
        'explanation': str|None
      }
    """
    # Expect OPENAI_API_KEY in env; do NOT hardcode secrets
    with open("APIconfig.json") as f:
        config = json.load(f)

    api_key = config["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key

    data_url = _encode_image_to_data_url(image_path)
    messages = _messages_for_prompt(data_url)
    last_text = None

    for _ in range(max_retries):
        resp = openai.chat.completions.create(model=model, messages=messages)
        text = resp.choices[0].message.content.strip()
        last_text = text

        line = _extract_line(text)
        parsed = _parse_line(line) if line else None
        if parsed is not None:
            return {
                "ok": True,
                "model": model,
                "raw": text,
                "label": parsed["label"],
                "confidence": parsed["confidence"],
                "explanation": parsed["explanation"],
            }

        # Give explicit corrective feedback and retry
        messages.append({"role": "assistant", "content": text})
        messages.append({
            "role": "user",
            "content": (
                "FORMAT ERROR. Return exactly:\n===LINE===\n"
                "<0 or 1>|<0..1>|<short explanation without '|'>\n===END===\n"
                "No extra text."
            ),
        })
        time.sleep(retry_delay)

    return {"ok": False, "model": model, "raw": last_text, "label": None, "confidence": None, "explanation": None}
