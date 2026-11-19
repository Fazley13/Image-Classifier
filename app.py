"""Local web app for MobileNetV2 image classification.

Run:  python app.py      then open  http://127.0.0.1:8000
"""
import os

# Keep the console quiet (must be set before TensorFlow is imported).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from flask import Flask, jsonify, render_template, request

import classifier

MAX_UPLOAD_MB = 16

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html", max_upload_mb=MAX_UPLOAD_MB)


@app.get("/api/health")
def health():
    return jsonify(classifier.status())


@app.post("/api/predict")
def predict():
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify(error="No image was uploaded."), 400

    try:
        predictions = classifier.classify(file.read())
    except classifier.ClassifierError as exc:
        return jsonify(error=str(exc)), exc.status_code
    except Exception:  # noqa: BLE001
        app.logger.exception("Unexpected error while classifying %r", file.filename)
        return jsonify(error="Something went wrong while analyzing this image."), 500

    return jsonify(filename=file.filename, predictions=predictions)


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error=f"This file is larger than {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    classifier.start_loading()  # load the model while the page opens
    print(f"\n  Image Classifier running at http://{host}:{port}\n")
    app.run(host=host, port=port, debug=False, threaded=True)
