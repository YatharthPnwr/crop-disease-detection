"""Web API: health, stats, samples, predict and input validation (needs trained models)."""

from pathlib import Path

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HAVE_MODELS = (ROOT / "app/models/efficientnet_b0.pt").exists()


@pytest.fixture(scope="module")
def client():
    import server

    return TestClient(server.app)


def test_static_and_stats(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    stats = client.get("/api/stats").json()
    assert stats["dataset"]["classes"] == 38
    assert set(stats["models"]) == {"custom_cnn", "efficientnet_b0"}


def test_bad_upload_is_422(client):
    r = client.post("/api/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code in (404, 422)  # 404 when no models are installed


@pytest.mark.skipif(not HAVE_MODELS, reason="trained models not installed (python scripts/fetch_models.py)")
def test_predict_sample(client):
    sample = ROOT / "app/samples/tomato_early_blight.jpg"
    r = client.post("/api/predict", files={"file": (sample.name, sample.read_bytes(), "image/jpeg")},
                    data={"model": "efficientnet_b0"})
    assert r.status_code == 200
    d = r.json()
    assert d["top"][0]["class"] == "Tomato___Early_blight"
    assert set(d["heatmaps"]) == {"gradcam", "gradcam++"}
    assert d["image"].startswith("data:image/jpeg;base64,")
