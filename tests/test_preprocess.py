"""Train-eval and app preprocessing must be the same function with the same output."""

import io

import numpy as np
import torch
from PIL import Image

from cropdoc.data import eval_transform, load_image, preprocess


def _photo(w=320, h=240, mode="RGB"):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr).convert(mode)


def test_shape_and_dtype():
    x = preprocess(_photo())
    assert x.shape == (3, 224, 224) and x.dtype == torch.float32


def test_preprocess_matches_eval_transform():
    img = _photo()
    assert torch.equal(preprocess(img), eval_transform()(load_image(img)))


def test_rgba_and_greyscale_become_rgb():
    for mode in ("RGBA", "L"):
        assert preprocess(_photo(mode=mode)).shape == (3, 224, 224)


def test_exif_orientation_is_applied():
    img = _photo(w=300, h=200)
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90 degrees on display
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    buf.seek(0)
    assert load_image(buf).size == (200, 300)


def test_bytes_path_and_pil_agree(tmp_path):
    img = _photo()
    p = tmp_path / "leaf.png"
    img.save(p)
    assert torch.equal(preprocess(p), preprocess(Image.open(p)))
