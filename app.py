from __future__ import annotations

import os
from flask import Flask, render_template, request

from model import SpamDetector


BASE_DIR = os.path.dirname(__file__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024  


detector = SpamDetector()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get("email_text", "")
    email_text = email_text.strip()

    if not email_text:
        return render_template(
            "index.html",
            error="Paste an email before checking.",
            result=None,
            email_text="",
        )

    result = detector.predict(email_text)

    try:
        import json
        import datetime as _dt

        os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
        log_path = os.path.join(BASE_DIR, "output", "predictions.jsonl")
        payload = {
            "ts": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "label": result.label,
            "probability": result.probability,
            "top_signals": result.top_signals,
            "email_preview": email_text[:500],
            "email_length": len(email_text),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
  
        pass

    return render_template(
        "index.html",
        error=None,
        result=result,
        email_text=email_text,
    )



if __name__ == "__main__":

    app.run(host="127.0.0.1", port=5000, debug=True)

