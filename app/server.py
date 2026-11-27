"""Web app: FastAPI backend (models, Grad-CAM, stats) + the static frontend in app/web/.

    python app/server.py                  # http://localhost:8000
    python app/server.py --host 0.0.0.0   # reachable from other devices on the LAN
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference as I  # noqa: E402
from cropdoc.data import IMAGE_SIZE, RESIZE, geometry_transform  # noqa: E402
from cropdoc.explain import CAM_METHODS, compute_cam, mask_from_hsv  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
WEB = APP_DIR / "web"
STATS = ROOT / "reports" / "site_stats.json"
DISPLAY = 2 * IMAGE_SIZE  # 448 px specimen image

app = FastAPI(title="Crop Disease Detection", docs_url="/api/docs", redoc_url=None)


@app.middleware("http")
async def no_stale_frontend(request, call_next):
    """Make browsers revalidate the page and its JS/CSS, so an edit never mixes old and new files."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@lru_cache(maxsize=4)
def get_model(name: str) -> I.LoadedModel:
    if name not in I.available_models():
        raise HTTPException(404, f"Model '{name}' is not installed. Run `python scripts/fetch_models.py`.")
    return I.load_model(name)


@lru_cache(maxsize=1)
def disease_info() -> dict:
    return I.load_disease_info()


def _png_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _jpeg_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _heatmap_rgba(cam: np.ndarray) -> np.ndarray:
    """CAM in [0,1] -> RGBA heatmap (turbo colormap); alpha rises with importance."""
    rgb = cv2.cvtColor(cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    alpha = np.uint8(np.clip(0.25 + 0.75 * cam, 0, 1) * 255)
    return np.dstack([rgb, alpha])


@app.get("/api/health")
def health():
    return {"ok": True, "models": I.available_models()}


@app.get("/api/models")
def models():
    out = []
    for name in I.available_models():
        lm = get_model(name)
        out.append({"name": name, "display": lm.display, "unsure_threshold": lm.unsure_threshold,
                    "runtime": "ONNX Runtime" if lm.session is not None else "PyTorch", "classes": len(lm.classes)})
    return out


@app.get("/api/stats")
def stats():
    if not STATS.exists():
        raise HTTPException(404, "reports/site_stats.json is missing. Run `python scripts/build_site_stats.py`.")
    return JSONResponse(json.loads(STATS.read_text()))


@app.get("/api/samples")
def samples():
    readme = APP_DIR / "samples" / "README.md"
    out = []
    for line in readme.read_text().splitlines() if readme.exists() else []:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 2 and cells[0].endswith(".jpg"):
            crop, disease = I.pretty(cells[1])
            out.append({"file": cells[0], "url": f"/samples/{cells[0]}", "class": cells[1], "crop": crop, "disease": disease})
    return out


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), model: str = Form("efficientnet_b0")):
    data = await file.read()
    try:
        img = I.read_upload(data)
    except I.InvalidImage as e:
        raise HTTPException(422, str(e)) from e
    lm = get_model(model)

    t0 = time.perf_counter()
    probs, x = I.predict(lm, img)
    predict_ms = (time.perf_counter() - t0) * 1000
    top = I.top_k(probs, lm.classes, 3)
    reason = I.unsure_reason(probs, lm.unsure_threshold)
    top_idx = lm.classes.index(top[0][0])

    t1 = time.perf_counter()
    cams = {m: compute_cam(lm.model, x, [top_idx], m)[0] for m in CAM_METHODS}
    cam_ms = (time.perf_counter() - t1) * 1000

    # Display copy: the same centre crop the model saw, at 2x resolution (Resize 512 -> crop 448
    # covers exactly the region of Resize 256 -> crop 224), with CAMs upscaled to match.
    crop_rgb = np.asarray(geometry_transform(DISPLAY, 2 * RESIZE)(img))
    cams_big = {m: np.clip(cv2.resize(c, (DISPLAY, DISPLAY), interpolation=cv2.INTER_CUBIC), 0, 1) for m, c in cams.items()}
    leaf_mask = mask_from_hsv(img)
    leaf_share = float(leaf_mask.mean())
    on_leaf = {m: float((c * leaf_mask).sum() / max(c.sum(), 1e-8)) for m, c in cams.items()}

    info = disease_info().get(top[0][0], {})
    return {
        "model": {"name": lm.name, "display": lm.display, "unsure_threshold": lm.unsure_threshold,
                  "runtime": "ONNX Runtime" if lm.session is not None else "PyTorch"},
        "top": [{"class": c, "crop": I.pretty(c)[0], "disease": I.pretty(c)[1], "prob": p} for c, p in top],
        "unsure": reason,
        "no_leaf": leaf_share < 0.05,
        "image": _jpeg_b64(crop_rgb),
        "heatmaps": {m: _png_b64(_heatmap_rgba(c)) for m, c in cams_big.items()},
        "attention_on_leaf": on_leaf,
        "leaf_share": leaf_share,
        "timing_ms": {"predict": round(predict_ms, 1), "gradcam": round(cam_ms, 1)},
        "info": info,
    }


app.mount("/samples", StaticFiles(directory=APP_DIR / "samples"), name="samples")
app.mount("/figures", StaticFiles(directory=ROOT / "reports" / "figures"), name="figures")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
