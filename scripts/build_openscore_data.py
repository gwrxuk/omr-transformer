"""Build real-engraving training data from the OpenScore Lieder corpus (CC0).

Songs are split by song, never by excerpt:
  test  = the 60 songs sampled by openscore_eval.py (seed 0): excluded entirely
  val   = 50 further songs (seed 1)
  train = everything else
For each train/val song, up to --per-song excerpts of the vocal line are cut
with the same rules as the evaluation, engraved with MuseScore in one batch
job, cropped to the model's staff geometry, and written with their tokens.

    python scripts/build_openscore_data.py --per-song 8
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openscore_eval import MSCORE, iter_excerpts, staff_crop  # noqa: E402
from omr.vocab import Vocab  # noqa: E402


def splits(all_paths: list[str]) -> dict[str, list[str]]:
    test = random.Random(0).sample(all_paths, 60)       # identical to openscore_eval.py
    rest = [p for p in all_paths if p not in set(test)]
    val = random.Random(1).sample(rest, 50)
    train = [p for p in rest if p not in set(val)]
    return {"test": test, "val": val, "train": train}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/openscore")
    ap.add_argument("--out", default="data/openscore_train")
    ap.add_argument("--per-song", type=int, default=8)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    (out / "xml").mkdir(parents=True, exist_ok=True)
    (out / "png").mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(parents=True, exist_ok=True)
    vocab = Vocab()
    all_paths = sorted(p for p in (src / "all_paths.txt").read_text().split("\n") if p)
    sp = splits(all_paths)
    (out / "splits.json").write_text(json.dumps(sp, indent=1, ensure_ascii=False))
    print({k: len(v) for k, v in sp.items()}, flush=True)

    items, jobs = [], []
    for split in ("train", "val"):
        for n, path in enumerate(sp[split]):
            stem = Path(path).stem
            xml = src / f"{stem}.musicxml"
            if not xml.exists():
                continue
            try:
                for k, (sc, toks, where) in enumerate(iter_excerpts(xml, vocab, step=2)):
                    if k >= args.per_song:
                        break
                    eid = f"{stem}_{where['start_measure']:03d}"
                    exml = out / "xml" / f"{eid}.musicxml"
                    sc.write("musicxml", fp=str(exml))
                    jobs.append({"in": str(exml.resolve()), "out": str((out / "png" / f"{eid}.png").resolve())})
                    items.append({"id": eid, "song": path, "split": split, "tokens": toks, **where})
            except Exception as e:  # unparseable song: skip it
                print("skip", stem, type(e).__name__, flush=True)
            if n % 100 == 0:
                print(split, n, len(items), flush=True)
    (out / "render_job.json").write_text(json.dumps(jobs))
    print("excerpts", len(items), "rendering...", flush=True)
    subprocess.run([MSCORE, "-r", "300", "-T", "20", "-j", str(out / "render_job.json")],
                   capture_output=True, timeout=7200)

    kept = []
    for it in items:
        pngs = sorted((out / "png").glob(f"{it['id']}*.png"))
        img = staff_crop(pngs[0]) if pngs else None
        if img is None:
            continue
        Image.fromarray(img).save(out / "img" / f"{it['id']}.png")
        kept.append(it)
    with open(out / "labels.jsonl", "w") as f:
        for it in kept:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print("kept", len(kept), dict(Counter(it["split"] for it in kept)), flush=True)


if __name__ == "__main__":
    main()
