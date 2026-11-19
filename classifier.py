"""MobileNetV2 (ImageNet) classification helpers.

The model is loaded lazily and only once. Images are decoded with Pillow,
converted to RGB, center-cropped to 224x224 and passed through the same
preprocessing MobileNetV2 was trained with.
"""
from __future__ import annotations

import io
import threading

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_SIZE = (224, 224)
TOP_K = 5

_model = None
_tf = None
_state = "idle"  # idle | loading | ready | error
_error: str | None = None
_load_lock = threading.Lock()
_predict_lock = threading.Lock()


class ClassifierError(Exception):
    """An error whose message is safe to show to the user."""

    status_code = 400


class ModelUnavailable(ClassifierError):
    status_code = 503


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def get_model():
    """Return the MobileNetV2 model, loading it on first use (thread-safe)."""
    global _model, _tf, _state, _error

    if _model is not None:
        return _model

    with _load_lock:
        if _model is not None:  # another thread finished while we waited
            return _model

        _state, _error = "loading", None
        try:
            import tensorflow as tf  # imported here so the server starts fast
        except ImportError as exc:
            _state = "error"
            _error = "TensorFlow is not installed. Run: pip install -r requirements.txt"
            raise ModelUnavailable(_error) from exc

        try:
            model = tf.keras.applications.MobileNetV2(weights="imagenet")
            # Warm-up: builds the graph and downloads the ImageNet class index
            # now, so the first real request is fast and failures show up here.
            warm = model.predict(np.zeros((1, *IMAGE_SIZE, 3), dtype="float32"), verbose=0)
            tf.keras.applications.mobilenet_v2.decode_predictions(warm, top=1)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            _state = "error"
            _error = (
                "The model could not be loaded. The first run needs an internet "
                f"connection to download the ImageNet weights. ({exc.__class__.__name__}: {exc})"
            )
            raise ModelUnavailable(_error) from exc

        _tf, _model, _state = tf, model, "ready"
        return _model


def start_loading() -> None:
    """Begin loading the model in a background thread."""

    def _run():
        try:
            get_model()
        except ModelUnavailable:
            pass  # state/error are recorded; the UI reads them from /api/health

    threading.Thread(target=_run, name="model-loader", daemon=True).start()


def status() -> dict:
    return {"state": _state, "error": _error}


# --------------------------------------------------------------------------
# Image handling
# --------------------------------------------------------------------------
def _to_rgb(img: Image.Image) -> Image.Image:
    """Fix orientation and flatten transparency onto white (not black)."""
    img = ImageOps.exif_transpose(img)
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


def load_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes into a 224x224 RGB image."""
    if not data:
        raise ClassifierError("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            rgb = _to_rgb(img)
    except UnidentifiedImageError as exc:
        raise ClassifierError(
            "This file isn't a supported image. Use a JPG, PNG, WebP, GIF, BMP or TIFF file."
        ) from exc
    except Image.DecompressionBombError as exc:
        raise ClassifierError("This image has too many pixels to process safely.") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        raise ClassifierError("This image file looks damaged and could not be read.") from exc

    # Center-crop to a square, then resize (the dashed square in the UI).
    return ImageOps.fit(rgb, IMAGE_SIZE, method=Image.Resampling.BICUBIC)


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def _pretty(label: str) -> str:
    label = label.replace("_", " ").strip()
    return label[:1].upper() + label[1:]


def classify(data: bytes, top: int = TOP_K) -> list[dict]:
    """Classify image bytes and return the top predictions."""
    image = load_image(data)  # validate the image before waiting on the model
    model = get_model()

    batch = np.expand_dims(np.asarray(image, dtype="float32"), axis=0)
    batch = _tf.keras.applications.mobilenet_v2.preprocess_input(batch)

    with _predict_lock:
        probabilities = model.predict(batch, verbose=0)
    decoded = _tf.keras.applications.mobilenet_v2.decode_predictions(probabilities, top=top)[0]

    return [
        {
            "rank": rank,
            "class_id": class_id,
            "label": _pretty(name),
            "confidence": float(score),
        }
        for rank, (class_id, name, score) in enumerate(decoded, start=1)
    ]
