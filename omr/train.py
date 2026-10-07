"""Train the OMR transformer.

    python -m omr.train --steps 6000 --out runs/base
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from functools import partial
from pathlib import Path

# Cap the MPS allocator so a leak raises an error instead of swapping the machine.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import torch  # noqa: E402
from torch.utils.data import DataLoader

from .data import PrimusDataset, SyntheticStaves, collate
from .metrics import Scores
from .model import OMRTransformer
from .vocab import Vocab


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(model, loader, vocab, device, max_batches: int | None = None) -> dict:
    model.eval()
    scores, loss_sum, tok = Scores(), 0.0, 0
    for b, (x, w, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x, w, y[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1),
            ignore_index=vocab.pad, reduction="sum")
        loss_sum += loss.item()
        tok += (y[:, 1:] != vocab.pad).sum().item()
        preds = model.greedy(x, w, vocab.bos, vocab.eos, max_len=y.shape[1] + 8)
        for p, r in zip(preds, y.tolist()):
            scores.add(vocab.decode(p), vocab.decode(r))
    model.train()
    return {"loss": loss_sum / max(1, tok), **scores.summary()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--degrade", type=float, nargs=2, default=[0.0, 1.0],
                    help="per-image degradation strength range for training")
    ap.add_argument("--val-size", type=int, default=512)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--primus-root", default=None, help="fine-tune on PrIMuS instead of synthetic")
    ap.add_argument("--init", default=None, help="checkpoint to start from")
    ap.add_argument("--fixed-width", type=int, default=1024, help="pad every batch to this width (0 = buckets)")
    ap.add_argument("--fixed-len", type=int, default=40, help="pad every target to this length (0 = buckets)")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    vocab = Vocab()
    vocab.save(out / "vocab.json")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    if args.primus_root:
        full = PrimusDataset(vocab, args.primus_root, distorted=True)
        n_val = min(args.val_size, len(full) // 10)
        train_ds = torch.utils.data.Subset(full, range(n_val, len(full)))
        val_sets = {"val": torch.utils.data.Subset(full, range(n_val))}
    else:
        train_ds = SyntheticStaves(vocab, "train", args.steps * args.batch, tuple(args.degrade))
        val_sets = {"val_clean": SyntheticStaves(vocab, "val", args.val_size, 0.0),
                    "val_scan": SyntheticStaves(vocab, "val", args.val_size, 1.0)}

    coll = partial(collate, pad_id=vocab.pad, fixed_w=args.fixed_width or None,
                   fixed_l=args.fixed_len or None)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=bool(args.primus_root),
                          num_workers=args.workers, collate_fn=coll,
                          persistent_workers=args.workers > 0, drop_last=True)
    val_dls = {k: DataLoader(v, batch_size=args.batch, num_workers=0, collate_fn=coll)
               for k, v in val_sets.items()}

    model = OMRTransformer(len(vocab), args.d_model, enc_layers=args.layers, dec_layers=args.layers,
                           ff=4 * args.d_model, dropout=args.dropout, pad_id=vocab.pad).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params/1e6:.2f}M vocab={len(vocab)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    crit = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad, label_smoothing=args.label_smoothing)
    log = open(out / "log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "split", "loss", "SER", "pitch_ER", "duration_ER", "sequence_acc", "lr", "elapsed_s"])

    best, step, t0, run_loss = float("inf"), 0, time.time(), 0.0
    model.train()
    while step < args.steps:
        for x, w, y in train_dl:
            if step >= args.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.lr, args.warmup, args.steps)
            x, y = x.to(device), y.to(device)
            logits = model(x, w, y[:, :-1])
            loss = crit(logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss = 0.98 * run_loss + 0.02 * loss.item() if step else loss.item()
            step += 1
            if step % 50 == 0:
                mem = f"  gpu {torch.mps.driver_allocated_memory() / 2**30:.1f}GB" if device.type == "mps" else ""
                print(f"step {step:5d}  loss {run_loss:.3f}  lr {opt.param_groups[0]['lr']:.2e}  "
                      f"{time.time() - t0:.0f}s{mem}", flush=True)
                writer.writerow([step, "train", f"{run_loss:.4f}", "", "", "", "",
                                 f"{opt.param_groups[0]['lr']:.2e}", f"{time.time() - t0:.0f}"])
            if step % args.eval_every == 0 or step == args.steps:
                results = {k: evaluate(model, dl, vocab, device) for k, dl in val_dls.items()}
                if device.type == "mps":
                    torch.mps.empty_cache()
                for k, r in results.items():
                    print(f"  [{k}] " + "  ".join(f"{m}={v:.4f}" for m, v in r.items() if m != "samples"),
                          flush=True)
                    writer.writerow([step, k, f"{r['loss']:.4f}", f"{r['SER']:.4f}", f"{r['pitch_ER']:.4f}",
                                     f"{r['duration_ER']:.4f}", f"{r['sequence_acc']:.4f}", "",
                                     f"{time.time() - t0:.0f}"])
                log.flush()
                key = "val_scan" if "val_scan" in results else "val"
                ckpt = {"model": model.state_dict(), "step": step, "args": vars(args),
                        "val": results}
                torch.save(ckpt, out / "last.pt")
                if results[key]["SER"] < best:
                    best = results[key]["SER"]
                    torch.save(ckpt, out / "best.pt")
                    print(f"  saved best ({key} SER {best:.4f})", flush=True)
    log.close()
    print(f"done in {time.time() - t0:.0f}s, best SER {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
