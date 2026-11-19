# Image Classifier

A local web app that uses **MobileNetV2** (pre-trained on ImageNet) to name the objects in your photos.
Drop in one or more images and get the top 5 guesses with a confidence score for each.

Everything runs on your own computer. Images are analyzed in memory and never saved.

## Features

- Drag and drop, file picker, or paste an image with Ctrl+V
- Several images at once, with a thumbnail strip to switch between them
- Top 5 predictions with confidence bars and a plain-language verdict
- A dashed square on the image shows the exact area the model analyzes
- Light and dark mode (follows your system), responsive on phones
- Clear error messages for bad files, large files, and a missing model

## Run it locally

You need **Python 3.9 – 3.12** (TensorFlow does not support every newer Python release yet).

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv

# 2. activate it
#    Windows (PowerShell):  .venv\Scripts\Activate.ps1
#    Windows (cmd):         .venv\Scripts\activate.bat
#    macOS / Linux:         source .venv/bin/activate

# 3. install the dependencies
pip install -r requirements.txt

# 4. start the app
python app.py
```

Then open **http://127.0.0.1:8000** in your browser.

> **First run:** TensorFlow downloads the ImageNet weights (about 14 MB) and the class
> list, so you need an internet connection once. After that they are cached
> in `~/.keras` and the app works offline.

### Options

| Environment variable | Default     | Purpose                              |
| -------------------- | ----------- | ------------------------------------ |
| `PORT`               | `8000`      | Port to listen on                    |
| `HOST`               | `127.0.0.1` | Use `0.0.0.0` to open it to your LAN |

Example (PowerShell): `$env:PORT=5000; python app.py`

## Project structure

```
Image-Classifier/
├── app.py            Flask server and API routes
├── classifier.py     Model loading, image preprocessing, prediction
├── requirements.txt
├── templates/
│   └── index.html    Page markup
└── static/
    ├── css/style.css Design (light table theme)
    └── js/app.js     Uploads, queue, and results UI
```

## API

| Method | Route          | Description                                                        |
| ------ | -------------- | ------------------------------------------------------------------ |
| GET    | `/api/health`  | `{"state": "idle \| loading \| ready \| error", "error": null}`      |
| POST   | `/api/predict` | Multipart form with an `image` file. Returns the top 5 predictions. |

```json
{
  "filename": "dog.jpg",
  "predictions": [
    { "rank": 1, "class_id": "n02099601", "label": "Golden retriever", "confidence": 0.9241 }
  ]
}
```

## What changed from the Colab version

The original script was a Google Colab notebook export. It could not run locally, so these were fixed:

- **Colab-only code removed.** `from google.colab import files` and `files.upload()` do not exist
  outside Colab. Uploads now go through the browser.
- **Matplotlib removed.** Results are shown in the web UI instead of a pop-up plot.
- **Deprecated Keras helpers replaced.** `load_img` and `img_to_array` from
  `tensorflow.keras.preprocessing.image` are gone in newer Keras versions. Images are now read with Pillow.
- **Better image handling.** Photo orientation (EXIF) is respected, transparent PNGs are flattened onto
  white instead of black, and the image is center-cropped and resized with bicubic
  interpolation instead of nearest-neighbor.
- **Top 5 instead of top 1**, with confidence for each.
- **Model loads once**, in the background at startup, instead of on every run.
- **Validation and errors.** Non-image files, corrupt files, oversized files (16 MB limit), and
  a missing TensorFlow install all return a clear message instead of a stack trace.
- **Quiet inference.** `model.predict(..., verbose=0)` hides the progress bar spam.

## Troubleshooting

- **"TensorFlow is not installed"**: run `pip install -r requirements.txt` inside your active
  virtual environment.
- **`pip` can't find a TensorFlow version**: your Python is probably too new. Use Python 3.9 – 3.12.
- **"The model could not be loaded"**: the first run needs internet to download the weights. Check
  your connection and click *Try again*.
- **Port already in use**: start on another port, e.g. `PORT=8080 python app.py`.
- **The fonts look different offline**: the page loads the Bricolage Grotesque font from Google Fonts and
  falls back to your system font when you are offline.

## Limitations

MobileNetV2 only knows the 1,000 ImageNet classes. It cannot recognize people by name, text, or anything
outside those classes, and it will still give its best guess with a low confidence score.
