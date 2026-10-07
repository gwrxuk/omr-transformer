"""Evaluate a checkpoint on the held-out synthetic test sets.

    python -m omr.evaluate --ckpt runs/base/best.pt --size 1000
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import SyntheticStaves, collate
from .metrics import Scores
from .model import OMRTransformer
from .train import pick_device
from .vocab import Vocab


def load(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location=device)
    vocab = Vocab.load(Path(ckpt_path).parent / "vocab.json")
    a = ck["args"]
    model = OMRTransformer(len(vocab), a["d_model"], enc_layers=a["layers"], dec_layers=a["layers"],
                           ff=4 * a["d_model"], dropout=0.0, pad_id=vocab.pad).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), vocab, ck


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--beam", type=int, default=0, help="beam width (0 = greedy)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    model, vocab, ck = load(args.ckpt, device)

    report = {"checkpoint": args.ckpt, "step": ck["step"], "decoding": f"beam{args.beam}" if args.beam else "greedy"}
    confusions: Counter = Counter()
    for name, strength in (("test_clean", 0.0), ("test_scan", 1.0)):
        ds = SyntheticStaves(vocab, "test", args.size, strength)
        dl = DataLoader(ds, batch_size=1 if args.beam else 64, num_workers=2,
                        collate_fn=partial(collate, pad_id=vocab.pad))
        scores = Scores()
        with torch.no_grad():
            for x, w, y in dl:
                x = x.to(device)
                if args.beam:
                    preds = [model.beam(x, w, vocab.bos, vocab.eos, k=args.beam)]
                else:
                    preds = model.greedy(x, w, vocab.bos, vocab.eos, max_len=y.shape[1] + 8)
                for p, r in zip(preds, y.tolist()):
                    pt, rt = vocab.decode(p), vocab.decode(r)
                    scores.add(pt, rt)
                    if len(pt) == len(rt):
                        confusions.update((a, b) for a, b in zip(rt, pt) if a != b)
        report[name] = scores.summary()
        print(name, json.dumps(report[name]))
    report["top_confusions"] = [f"{a} -> {b} ({n})" for (a, b), n in confusions.most_common(10)]
    suffix = (f"_beam{args.beam}" if args.beam else "") + (f"_n{args.size}" if args.size != 1000 else "")
    out = Path(args.ckpt).parent / f"test_results{suffix}.json"
    out.write_text(json.dumps(report, indent=2))
    print("top confusions:", *report["top_confusions"], sep="\n  ")
    print("wrote", out)


if __name__ == "__main__":
    main()
