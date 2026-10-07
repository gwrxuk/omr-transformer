"""Evaluate the model on real engravings from the OpenScore Lieder corpus (CC0).

For a fixed random sample of songs: download the MuseScore file, take the vocal
line, keep the first 2-3 measure window whose notation the model's vocabulary
can express (no ties, tuplets, chords, grace notes, pickups or unsupported
clefs/keys/metres), engrave it with MuseScore 4, crop the staff to the model's
input geometry, transcribe, and score against the exact source tokens.

    python scripts/openscore_eval.py --songs 60 --ckpt runs/base/best.pt
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omr.metrics import Scores  # noqa: E402
from omr.vocab import DURATIONS, KEYS, Vocab  # noqa: E402

MSCORE = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"
RAW = "https://raw.githubusercontent.com/OpenScore/Lieder/main/"
KEY_BY_SHARPS = {v: k for k, v in KEYS.items()}
TYPE_NAME = {"whole": "whole", "half": "half", "quarter": "quarter", "eighth": "eighth", "16th": "sixteenth"}


def list_scores() -> list[str]:
    out = subprocess.run(["gh", "api", "repos/OpenScore/Lieder/git/trees/main?recursive=1",
                          "--jq", '.tree[] | select(.path|endswith(".mscx")) | .path'],
                         capture_output=True, text=True, check=True).stdout
    return sorted(out.split())


def note_token(n, vocab: Vocab) -> str | None:
    from music21 import note
    d = n.duration
    if d.isGrace or d.tuplets or (getattr(n, "tie", None) is not None):
        return None
    base = TYPE_NAME.get(d.type)
    if base is None or d.dots > 1:
        return None
    name = base + ("." if d.dots == 1 else "")
    if name not in DURATIONS or abs(DURATIONS[name] - float(d.quarterLength)) > 1e-6:
        return None
    if isinstance(n, note.Rest):
        tok = f"rest-{name}"
    elif isinstance(n, note.Note):
        alter = int(n.pitch.alter)
        acc = "#" * alter if alter > 0 else "b" * (-alter)
        tok = f"note-{n.pitch.step}{acc}{n.pitch.octave}_{name}"
    else:
        return None  # chords and anything else
    return tok if tok in vocab.index else None


def excerpt(path: Path, vocab: Vocab, rng: random.Random):
    """Return (music21 score of the excerpt, ground-truth tokens) or None."""
    from music21 import clef, converter, key, meter, metadata, stream
    sc = converter.parse(str(path))
    part = sc.parts[0]
    measures = list(part.getElementsByClass("Measure"))
    for start in range(0, max(0, len(measures) - 2)):
        for length in (3, 2):
            win = measures[start:start + length]
            if len(win) < length:
                continue
            c = win[0].getContextByClass(clef.Clef)
            k = win[0].getContextByClass(key.KeySignature)
            t = win[0].getContextByClass(meter.TimeSignature)
            if c is None or t is None:
                continue
            ctok = {"G2": "clef-G2", "F4": "clef-F4"}.get(f"{c.sign}{c.line}")
            ktok = KEY_BY_SHARPS.get(k.sharps if k is not None else 0)
            ttok = f"timeSignature-{t.ratioString}"
            if ctok is None or ktok is None or ttok not in vocab.index:
                continue
            toks = [ctok, f"keySignature-{ktok}", ttok]
            ok = True
            for m in win:
                if m.voices or m.getElementsByClass(clef.Clef) and m is not win[0] \
                        or m.getElementsByClass(meter.TimeSignature) and m is not win[0] \
                        or m.getElementsByClass(key.KeySignature) and m is not win[0]:
                    ok = False
                    break
                if abs(float(m.duration.quarterLength) - t.barDuration.quarterLength) > 1e-6:
                    ok = False
                    break
                for n in m.notesAndRests:
                    tok = note_token(n, vocab)
                    if tok is None:
                        ok = False
                        break
                    toks.append(tok)
                if not ok:
                    break
                toks.append("barline")
            n_notes = sum(t.startswith("note-") for t in toks)
            n_rests = sum(t.startswith("rest-") for t in toks)
            # skip windows that are mostly rests (piano introductions) and whole-bar rests,
            # which engravers draw as a centred whole rest whatever the metre
            full_bar = any(len(m.notesAndRests) == 1 and m.notesAndRests[0].isRest for m in win)
            if not ok or n_notes < 4 or n_rests > n_notes / 3 or full_bar:
                continue
            # build a clean excerpt: voice notes only, no lyrics, dynamics or text
            p = stream.Part()
            p.partName = ""
            for i, m in enumerate(win):
                nm = stream.Measure(number=i + 1)
                if i == 0:
                    nm.append([copy.deepcopy(c), key.KeySignature(k.sharps if k is not None else 0),
                               meter.TimeSignature(t.ratioString)])
                for n in m.notesAndRests:
                    nn = copy.deepcopy(n)
                    nn.lyrics = []
                    nn.expressions = []
                    nn.articulations = []
                    nm.append(nn)
                p.append(nm)
            out = stream.Score([p])
            out.metadata = metadata.Metadata()
            out.metadata.title = ""
            out.metadata.composer = ""
            return out, toks, {"start_measure": start + 1, "measures": length}
    return None


def staff_crop(png: Path) -> np.ndarray | None:
    """Find the staff, scale so line spacing = 10 px, crop to the 128 px training band."""
    a = np.asarray(Image.open(png).convert("L"))
    dark = (a < 128).mean(1)
    rows = np.where(dark > 0.5)[0]
    if len(rows) < 5:
        return None
    lines, cur = [], [rows[0]]
    for r in rows[1:]:
        if r == cur[-1] + 1:
            cur.append(r)
        else:
            lines.append(np.mean(cur))
            cur = [r]
    lines.append(np.mean(cur))
    if len(lines) < 5:
        return None
    lines = lines[:5]
    s = float(np.mean(np.diff(lines)))
    scale = 10.0 / s
    im = Image.fromarray(a).resize((max(8, round(a.shape[1] * scale)), max(8, round(a.shape[0] * scale))),
                                   Image.LANCZOS)
    b = np.asarray(im)
    bottom = lines[4] * scale
    y0 = int(round(bottom - 84))           # training: bottom staff line at y = 84
    band = np.full((128, b.shape[1]), 255, np.uint8)
    src0, src1 = max(0, y0), min(b.shape[0], y0 + 128)
    band[src0 - y0: src1 - y0] = b[src0:src1]
    cols = np.where((band < 128).any(0))[0]
    return band[:, max(0, cols[0] - 8): cols[-1] + 12] if len(cols) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--out", default="docs/openscore_test")
    ap.add_argument("--work", default="data/openscore")
    args = ap.parse_args()

    import torch
    from omr.evaluate import load
    from omr.predict import transcribe
    from omr.train import pick_device

    work, out = Path(args.work), Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    vocab = Vocab()
    rng = random.Random(args.seed)
    paths = rng.sample(list_scores(), args.songs)

    dev = pick_device("auto")
    model, vocab, _ = load(args.ckpt, dev)
    scores, rows = Scores(), []
    for i, path in enumerate(paths):
        name = Path(path).stem
        try:
            mscx = work / f"{name}.mscx"
            if not mscx.exists():
                urllib.request.urlretrieve(RAW + urllib.parse.quote(path), mscx)
            xml = work / f"{name}.musicxml"
            if not xml.exists():
                subprocess.run([MSCORE, "-o", str(xml), str(mscx)], capture_output=True, timeout=120)
            ex = excerpt(xml, vocab, rng)
            if ex is None:
                rows.append({"song": path, "status": "no in-vocabulary excerpt"})
                continue
            sc, truth, where = ex
            exml = work / f"{name}_ex.musicxml"
            sc.write("musicxml", fp=str(exml))
            png = work / f"{name}_ex.png"
            subprocess.run([MSCORE, "-o", str(png), "-r", "300", "-T", "20", str(exml)],
                           capture_output=True, timeout=120)
            rendered = sorted(work.glob(f"{name}_ex*.png"))
            img = staff_crop(rendered[0]) if rendered else None
            if img is None:
                rows.append({"song": path, "status": "render or staff detection failed"})
                continue
            Image.fromarray(img).save(out / "images" / f"{name}.png")
            pred = transcribe(model, vocab, img, dev, beam=4)
            scores.add(pred, truth)
            one = Scores()
            one.add(pred, truth)
            rows.append({"song": path, "status": "ok", **where, "truth": " ".join(truth),
                         "pred": " ".join(pred), "SER": round(one.summary()["SER"], 4)})
            print(f"[{i + 1}/{len(paths)}] {name} SER {one.summary()['SER']:.3f}", flush=True)
        except Exception as e:  # keep going; record why
            rows.append({"song": path, "status": f"error: {type(e).__name__}: {e}"[:200]})
    summary = scores.summary()
    (out / "results.json").write_text(json.dumps({"summary": summary, "items": rows}, indent=1,
                                                 ensure_ascii=False))
    print(json.dumps(summary))
    print("kept", summary["samples"], "of", len(paths), "songs")


if __name__ == "__main__":
    main()
