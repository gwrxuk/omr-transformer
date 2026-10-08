"""Train the melody language model on OpenScore Lieder vocal lines.

Training phrases are transposed into every key the vocabulary has (7 major
keys) when all pitches stay in range; validation phrases are left as written.

    python scripts/train_melody_lm.py --steps 6000 --out runs/melody
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omr.melody import MelodyLM, encode, transpose_phrase  # noqa: E402
from omr.vocab import KEYS, Vocab  # noqa: E402

MAX_LEN = 256


def chunks(header, bars, vocab):
    """Split a phrase into sequences of at most MAX_LEN tokens, each starting with the header."""
    out, cur = [], []
    for bar in bars:
        if cur and len(encode(header, cur + [bar], vocab)) > MAX_LEN:
            out.append(encode(header, cur, vocab, eos=False))
            cur = []
        cur.append(bar)
    if cur:
        out.append(encode(header, cur, vocab))
    return [s for s in out if len(s) <= MAX_LEN]


def load(path: Path, vocab: Vocab):
    data = {"train": [], "val": [], "test": []}
    for line in path.read_text().splitlines():
        p = json.loads(line)
        if p["split"] == "train":
            for k in KEYS:
                t = transpose_phrase(p["header"], p["bars"], k, vocab)
                if t:
                    data["train"] += chunks(*t, vocab)
        else:
            data[p["split"]] += chunks(p["header"], p["bars"], vocab)
    return data


def batchify(seqs, pad):
    x = torch.full((len(seqs), MAX_LEN), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/melody/phrases.jsonl")
    ap.add_argument("--out", default="runs/melody")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rng = random.Random(0)
    vocab = Vocab()
    pad = vocab.index["<pad>"]
    data = load(Path(args.data), vocab)
    print({k: len(v) for k, v in data.items()},
          "train tokens", sum(map(len, data["train"])), flush=True)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = MelodyLM(len(vocab.tokens), args.d_model, 4, args.layers, args.dropout, MAX_LEN).to(dev)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))

    def loss_on(x):
        x = x.to(dev)
        logits = model(x[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), ignore_index=pad)

    @torch.no_grad()
    def evaluate(split):
        model.eval()
        tot, n = 0.0, 0
        seqs = data[split]
        for i in range(0, len(seqs), args.batch):
            b = seqs[i:i + args.batch]
            b = b + [b[0]] * (args.batch - len(b))  # keep one shape on MPS
            x = batchify(b, pad).to(dev)
            logits = model(x[:, :-1])
            y = x[:, 1:]
            l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                ignore_index=pad, reduction="none").view(len(b), -1)
            k = min(len(seqs) - i, args.batch)
            tot += l[:k].sum().item()
            n += (y[:k] != pad).sum().item()
        model.train()
        return tot / n

    best, log, t0 = float("inf"), [], time.time()
    for step in range(1, args.steps + 1):
        x = batchify(rng.sample(data["train"], args.batch), pad)
        loss = loss_on(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == args.steps:
            v = evaluate("val")
            rec = {"step": step, "train_loss": round(loss.item(), 4), "val_loss": round(v, 4),
                   "val_ppl": round(math.exp(v), 2), "min": round((time.time() - t0) / 60, 1)}
            log.append(rec)
            print(rec, flush=True)
            if v < best:
                best = v
                torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev)["model"])
    res = {"val_loss": evaluate("val"), "test_loss": evaluate("test")}
    res = {**{k: round(v, 4) for k, v in res.items()},
           **{k.replace("loss", "ppl"): round(math.exp(v), 2) for k, v in res.items()},
           "best_step": torch.load(out / "best.pt", map_location="cpu")["step"]}
    print(res)
    (out / "log.json").write_text(json.dumps(log, indent=1))
    (out / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
