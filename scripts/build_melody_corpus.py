"""Extract the vocal melody of every OpenScore Lieder song as token phrases.

A phrase is a run of consecutive measures that the vocabulary can express
(same rules as the OMR evaluation: no ties, tuplets, chords or grace notes),
with one clef, key and metre. Runs of two or more whole-bar rests (piano
interludes) split phrases, and leading/trailing rest bars are trimmed.
Songs keep the train/val/test split of build_openscore_data.py.

    python scripts/build_melody_corpus.py          # -> data/melody/phrases.jsonl
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_openscore_data import splits  # noqa: E402
from openscore_eval import KEY_BY_SHARPS, note_token  # noqa: E402
from omr.vocab import Vocab  # noqa: E402

MIN_BARS = 4


def song_phrases(xml: str) -> list[dict]:
    import warnings
    warnings.filterwarnings("ignore")
    from music21 import clef, converter, key, meter
    vocab = Vocab()
    part = converter.parse(xml).parts[0]
    phrases, cur, header = [], [], None

    def close():
        nonlocal cur
        while cur and cur[0]["rest"]:
            cur.pop(0)
        while cur and cur[-1]["rest"]:
            cur.pop()
        if len(cur) >= MIN_BARS:
            phrases.append({"header": list(header), "bars": [b["toks"] for b in cur]})
        cur = []

    for m in part.getElementsByClass("Measure"):
        c = m.getContextByClass(clef.Clef)
        k = m.getContextByClass(key.KeySignature)
        t = m.getContextByClass(meter.TimeSignature)
        if c is None or t is None:
            close()
            continue
        ctok = {"G2": "clef-G2", "F4": "clef-F4"}.get(f"{c.sign}{c.line}")
        ktok = KEY_BY_SHARPS.get(k.sharps if k is not None else 0)
        ttok = f"timeSignature-{t.ratioString}"
        h = (ctok, f"keySignature-{ktok}", ttok)
        if ctok is None or ktok is None or ttok not in vocab.index:
            close()
            continue
        if h != header:
            close()
            header = h
        toks = [note_token(n, vocab) for n in m.notesAndRests]
        if m.voices or not toks or None in toks or \
                abs(float(m.duration.quarterLength) - t.barDuration.quarterLength) > 1e-6:
            close()
            continue
        rest = all(x.startswith("rest-") for x in toks)
        if rest and cur and cur[-1]["rest"]:  # second rest bar in a row: interlude
            close()
            continue
        cur.append({"toks": toks, "rest": rest})
    close()
    return phrases


def work(args):
    path, split, src = args
    xml = Path(src) / f"{Path(path).stem}.musicxml"
    if not xml.exists():
        return []
    try:
        return [{"song": path, "split": split, **p} for p in song_phrases(str(xml))]
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/openscore")
    ap.add_argument("--out", default="data/melody")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_paths = sorted(p for p in (Path(args.src) / "all_paths.txt").read_text().split("\n") if p)
    jobs = [(p, s, args.src) for s, ps in splits(all_paths).items() for p in ps]
    with mp.Pool(8) as pool, open(out / "phrases.jsonl", "w") as f:
        n = 0
        for i, res in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            for ph in res:
                f.write(json.dumps(ph, ensure_ascii=False) + "\n")
                n += 1
            if i % 200 == 0:
                print(i, "songs", n, "phrases", flush=True)
    print("phrases", n)


if __name__ == "__main__":
    main()
