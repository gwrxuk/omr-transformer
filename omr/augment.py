"""Scan / phone-photo style degradations applied to clean renders."""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter


def degrade(img: np.ndarray, rng: random.Random, strength: float = 1.0) -> np.ndarray:
    """Return a degraded copy. strength=0 is a no-op, 1 is the training mix."""
    if strength <= 0:
        return img
    im = Image.fromarray(img)
    h = im.height
    # slight rotation and scale, as from a hand-held phone
    if rng.random() < 0.8 * strength:
        im = im.rotate(rng.uniform(-1.8, 1.8) * strength, resample=Image.BILINEAR,
                       expand=False, fillcolor=255)
    if rng.random() < 0.5 * strength:
        s = rng.uniform(0.85, 1.1)
        im = im.resize((max(8, int(im.width * s)), max(8, int(h * s))), Image.BILINEAR)
        canvas = Image.new("L", (im.width, h), 255)
        canvas.paste(im, (0, (h - im.height) // 2))
        im = canvas
    # ink spread / thin strokes
    if rng.random() < 0.3 * strength:
        im = im.filter(ImageFilter.MinFilter(3) if rng.random() < 0.5 else ImageFilter.MaxFilter(3))
    if rng.random() < 0.6 * strength:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2) * strength))
    a = np.asarray(im).astype(np.float32)
    # uneven lighting, paper tint, contrast
    if rng.random() < 0.6 * strength:
        grad = np.linspace(rng.uniform(-40, 0), rng.uniform(-40, 0), a.shape[1])[None, :]
        a = a + grad * strength
    if rng.random() < 0.6 * strength:
        lo, hi = rng.uniform(0, 70) * strength, 255 - rng.uniform(0, 50) * strength
        a = lo + a * (hi - lo) / 255.0
    if rng.random() < 0.7 * strength:
        a = a + np.random.default_rng(rng.getrandbits(32)).normal(0, 12 * strength, a.shape)
    a = np.clip(a, 0, 255).astype(np.uint8)
    if rng.random() < 0.5 * strength:
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=rng.randint(25, 70))
        a = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("L"))
    return a
