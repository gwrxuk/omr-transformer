"""Compose new melodies with the melody LM, check they are new, engrave them,
and read the engravings back with the OMR model.

Two kinds of piece:
  free      - 16 bars from a header only (clef, key, metre)
  continue  - the first 2 bars of a held-out test song (never seen in
              training), continued to 16 bars

For each piece, --candidates samples are drawn with bar-filling constraints.
The kept sample must end on the tonic, stay within a 15-semitone range, have
at most one empty bar, and share no run longer than --max-copy notes with any
training phrase (compared as interval + duration, so transposition does not
hide a copy). Among those, the highest mean log-probability wins.

    python scripts/compose.py --lm runs/melody/best.pt --omr runs/real_ft/best.pt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from openscore_eval import MSCORE, staff_crop  # noqa: E402
from omr.export import midi_number, to_midi, to_musicxml  # noqa: E402
from omr.melody import TONIC, MelodyLM, sample  # noqa: E402
from omr.vocab import Vocab  # noqa: E402

FREE = [("clef-G2", "keySignature-FM", "timeSignature-3/4"),
        ("clef-G2", "keySignature-GM", "timeSignature-6/8"),
        ("clef-G2", "keySignature-EbM", "timeSignature-4/4")]
BPM = {"3/4": 96, "6/8": 120, "4/4": 84, "2/4": 90, "3/8": 108}
N_BARS = 16


def symbols(tokens: list[str]) -> list[tuple]:
    """Transposition-invariant melody: (interval from previous note, duration) per note or rest."""
    out, prev = [], None
    for t in tokens:
        if t.startswith("note-"):
            p, d = t[5:].rsplit("_", 1)
            m = midi_number(p)
            out.append((None if prev is None else m - prev, d))
            prev = m
        elif t.startswith("rest-"):
            out.append(("r", t[5:]))
    return out


class CopyIndex:
    def __init__(self, phrases: list[dict], n_min: int = 4, n_max: int = 24):
        self.n_min, self.n_max, self.sets = n_min, n_max, {}
        self.seqs = []
        for p in phrases:
            s = symbols([t for bar in p["bars"] for t in bar])
            self.seqs.append((p["song"], s))
            for n in range(n_min, n_max + 1):
                st = self.sets.setdefault(n, set())
                for i in range(len(s) - n + 1):
                    st.add(self.key(s, i, n))

    @staticmethod
    def key(s, i, n):
        return tuple(s[i + 1:i + n]) + (s[i][1],)

    def longest(self, tokens: list[str]) -> int:
        """Longest run of notes/rests (first note's interval ignored) found in the training phrases."""
        s = symbols(tokens)
        best = 0
        for i in range(len(s)):
            n = max(best + 1, self.n_min)
            while i + n <= len(s) and n <= self.n_max and self.key(s, i, n) in self.sets[n]:
                best, n = n, n + 1
        return best

    def source_of(self, tokens: list[str], n: int) -> str | None:
        s = symbols(tokens)
        for i in range(len(s) - n + 1):
            k = self.key(s, i, n)
            for song, t in self.seqs:
                if any(self.key(t, j, n) == k for j in range(len(t) - n + 1)):
                    return song
        return None


def checks(tokens: list[str], header) -> dict:
    notes = [t for t in tokens if t.startswith("note-")]
    pitches = [midi_number(t[5:].rsplit("_", 1)[0]) for t in notes]
    last = notes[-1][5:].rsplit("_", 1)
    tonic = TONIC[header[1].split("-", 1)[1]].replace("-", "b")
    bars, cur = [], []
    for t in tokens[3:]:
        if t == "barline":
            bars.append(cur)
            cur = []
        else:
            cur.append(t)
    empty = sum(all(t.startswith("rest-") for t in b) for b in bars)
    return {"ends_on_tonic": last[0].rstrip("0123456789") == tonic,
            "final_long": last[1] in ("quarter", "quarter.", "half", "half.", "whole"),
            "range": max(pitches) - min(pitches), "empty_bars": empty, "notes": len(notes)}


def hide_part_name(xml: str) -> str:
    return xml.replace("<part-name>Music</part-name>", '<part-name print-object="no"></part-name>')


def bars_of(tokens):
    bars, cur = [], []
    for t in tokens[3:]:
        if t == "barline":
            bars.append(cur)
            cur = []
        else:
            cur.append(t)
    return bars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", default="runs/melody/best.pt")
    ap.add_argument("--omr", default="runs/real_ft/best.pt")
    ap.add_argument("--data", default="data/melody/phrases.jsonl")
    ap.add_argument("--out", default="docs/new_music")
    ap.add_argument("--candidates", type=int, default=128)
    ap.add_argument("--max-copy", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "render").mkdir(parents=True, exist_ok=True)
    vocab = Vocab()
    phrases = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    index = CopyIndex([p for p in phrases if p["split"] == "train"])

    ck = torch.load(args.lm, map_location="cpu")
    a = ck["args"]
    lm = MelodyLM(len(vocab.tokens), a["d_model"], 4, a["layers"], 0.0, 256).eval()
    lm.load_state_dict(ck["model"])
    g = torch.Generator().manual_seed(args.seed)

    # prompts from held-out test songs: first phrase per song whose 2 opening bars hold >= 4 notes
    tests, seen = [], set()
    for p in sorted((p for p in phrases if p["split"] == "test"), key=lambda p: p["song"]):
        opening = [t for bar in p["bars"][:2] for t in bar]
        if p["song"] in seen or sum(t.startswith("note-") for t in opening) < 4 or len(p["bars"]) < 6:
            continue
        if p["header"][0] != "clef-G2":
            continue
        seen.add(p["song"])
        tests.append(p)
    jobs = [{"kind": "free", "header": list(h), "prompt_bars": []} for h in FREE]
    for p in tests[:2]:
        jobs.append({"kind": "continue", "header": p["header"], "prompt_bars": p["bars"][:2],
                     "source": p["song"], "source_next_bars": p["bars"][2:6]})

    pieces = []
    for n, job in enumerate(jobs, 1):
        prompt = ["<bos>", *job["header"]]
        for bar in job["prompt_bars"]:
            prompt += bar + ["barline"]
        cands = []
        for _ in range(args.candidates):
            toks, lp = sample(lm, vocab, prompt, N_BARS, generator=g)
            new = toks[3 + sum(len(b) + 1 for b in job["prompt_bars"]):]
            c = checks(toks, job["header"])
            c["copy_run"] = index.longest(new)
            c["ok"] = (c["ends_on_tonic"] and c["final_long"] and c["range"] <= 15
                       and c["empty_bars"] <= 1 and c["copy_run"] <= args.max_copy)
            cands.append((c["ok"], lp, toks, c))
        ok = [x for x in cands if x[0]]
        pool = ok or cands
        _, lp, toks, c = max(pool, key=lambda x: x[1])
        title = f"Lied-style melody {n}" + (" (continuation)" if job["kind"] == "continue" else "")
        meta = {"n": n, "title": title, **job, "tokens": toks, "mean_logp": round(lp, 4), **c,
                "candidates": args.candidates, "passed_filters": len(ok),
                "copy_source": index.source_of(toks[3:], c["copy_run"]) if c["copy_run"] >= 4 else None}
        stem = f"melody_{n}"
        metre = job["header"][2].split("-")[1]
        (out / f"{stem}.musicxml").write_text(hide_part_name(to_musicxml(toks, title)))
        (out / f"{stem}.mid").write_bytes(to_midi(toks, bpm=BPM.get(metre, 90)))
        pieces.append(meta)
        print(n, job["kind"], job["header"], f"passed {len(ok)}/{args.candidates}",
              {k: c[k] for k in ("copy_run", "range", "ends_on_tonic")}, flush=True)

    # engrave: full pieces (PNG + PDF) and 2-bar windows for the OMR round trip
    rjobs, windows = [], []
    for p in pieces:
        stem = f"melody_{p['n']}"
        src = str((out / f"{stem}.musicxml").resolve())
        rjobs += [{"in": src, "out": str((out / f"{stem}.png").resolve())},
                  {"in": src, "out": str((out / f"{stem}.pdf").resolve())}]
        bars = bars_of(p["tokens"])
        for w in range(0, N_BARS, 2):
            toks = list(p["header"])
            for bar in bars[w:w + 2]:
                toks += bar + ["barline"]
            wid = f"{stem}_w{w // 2 + 1}"
            x = out / "render" / f"{wid}.musicxml"
            x.write_text(hide_part_name(to_musicxml(toks, "")))
            rjobs.append({"in": str(x.resolve()), "out": str((out / "render" / f"{wid}.png").resolve())})
            windows.append((p["n"], wid, toks))
    job = out / "render" / "job.json"
    job.write_text(json.dumps(rjobs))
    subprocess.run([MSCORE, "-j", str(job)], capture_output=True, timeout=3600)
    # window PNGs need the OMR resolution; rerender those at 300 dpi
    wjobs = [j for j in rjobs if "/render/" in j["out"]]
    job.write_text(json.dumps(wjobs))
    subprocess.run([MSCORE, "-r", "300", "-T", "20", "-j", str(job)], capture_output=True, timeout=3600)

    for p in pieces:  # audio one file at a time: a failed MP3 export must not stop the batch
        stem = out / f"melody_{p['n']}"
        try:
            subprocess.run([MSCORE, "-o", str(stem.with_suffix(".mp3")), str(stem.with_suffix(".musicxml"))],
                           capture_output=True, timeout=300)
        except subprocess.TimeoutExpired:
            print("mp3 export timed out for", stem.name)

    from omr.evaluate import load
    from omr.metrics import Scores
    from omr.predict import transcribe
    from omr.train import pick_device
    dev = pick_device("auto")
    model, ovocab, _ = load(args.omr, dev)
    total = Scores()
    per = {p["n"]: Scores() for p in pieces}
    rt = []
    for n, wid, truth in windows:
        pngs = sorted((out / "render").glob(f"{wid}*.png"))
        img = staff_crop(pngs[0]) if pngs else None
        if img is None:
            rt.append({"window": wid, "status": "staff not found"})
            continue
        Image.fromarray(img).save(out / "render" / f"{wid}_crop.png")
        pred = transcribe(model, ovocab, img, dev, beam=4)
        total.add(pred, truth)
        per[n].add(pred, truth)
        one = Scores()
        one.add(pred, truth)
        rt.append({"window": wid, "SER": round(one.summary()["SER"], 4),
                   "truth": " ".join(truth), "pred": " ".join(pred)})
    for p in pieces:
        p["omr_roundtrip"] = {k: round(v, 4) for k, v in per[p["n"]].summary().items()}
    report = {"settings": vars(args), "pieces": pieces,
              "omr_roundtrip_total": {k: round(v, 4) for k, v in total.summary().items()},
              "omr_windows": rt}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("OMR round trip", report["omr_roundtrip_total"])


if __name__ == "__main__":
    main()
