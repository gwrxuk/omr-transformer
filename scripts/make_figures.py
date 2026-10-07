"""Plot training curves and render a clean/scan example grid into docs/."""
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omr.augment import degrade  # noqa: E402
from omr.render import render  # noqa: E402
from omr.sequence import random_staff  # noqa: E402

run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/base")
docs = Path("docs")
docs.mkdir(exist_ok=True)

# example grid: clean render above its degraded "scan"
rows = []
for s in range(3):
    r = random.Random(100 + s)
    seq = random_staff(r)
    img = render(seq, r)
    rows += [img, degrade(img, random.Random(s), 1.0)]
W = max(a.shape[1] for a in rows)
grid = np.full((128 * len(rows), W), 255, np.uint8)
for i, a in enumerate(rows):
    grid[i * 128:(i + 1) * 128, : a.shape[1]] = a
Image.fromarray(grid).save(docs / "examples.png")

# SER curves from log.csv (simple PIL plot to avoid a matplotlib dependency)
pts = {}
with open(run / "log.csv") as f:
    for row in csv.DictReader(f):
        if row["split"].startswith("val") and row["SER"]:
            pts.setdefault(row["split"], []).append((int(row["step"]), float(row["SER"])))
if pts:
    Wd, Hd, m = 640, 360, 50
    im = Image.new("RGB", (Wd, Hd), "white")
    d = ImageDraw.Draw(im)
    smax = max(s for v in pts.values() for _, s in v)
    xmax = max(x for v in pts.values() for x, _ in v)
    d.line([(m, m), (m, Hd - m), (Wd - m, Hd - m)], fill="black")
    colors = {"val_clean": (31, 119, 180), "val_scan": (214, 39, 40), "val": (44, 160, 44)}
    for k, v in pts.items():
        xy = [(m + (x / xmax) * (Wd - 2 * m), Hd - m - (s / smax) * (Hd - 2 * m)) for x, s in v]
        d.line(xy, fill=colors.get(k, "gray"), width=3)
        d.text((xy[-1][0] - 60, xy[-1][1] - 16), f"{k} {v[-1][1]:.3f}", fill=colors.get(k, "gray"))
    d.text((m, 15), f"Validation symbol error rate (max {smax:.2f}) vs step (max {xmax})", fill="black")
    im.save(docs / "ser_curve.png")
print("wrote docs/examples.png", "and docs/ser_curve.png" if pts else "")
