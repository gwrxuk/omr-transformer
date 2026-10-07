"""Transcribe a staff image to tokens, MusicXML and MIDI.

    python -m omr.predict --ckpt runs/base/best.pt --image staff.png --out out/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import to_tensor
from .evaluate import load
from .export import to_midi, to_musicxml
from .train import pick_device


def transcribe(model, vocab, img: np.ndarray, device, beam: int = 4) -> list[str]:
    x = to_tensor(img)
    W = x.shape[-1] + (-x.shape[-1]) % 8
    xb = torch.zeros(1, 1, x.shape[1], W)
    xb[0, :, :, : x.shape[-1]] = x
    w = torch.tensor([x.shape[-1]])
    xb = xb.to(device)
    ids = model.beam(xb, w, vocab.bos, vocab.eos, k=beam) if beam else \
        model.greedy(xb, w, vocab.bos, vocab.eos)[0]
    return vocab.decode(ids)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    device = pick_device(args.device)
    model, vocab, _ = load(args.ckpt, device)
    img = np.asarray(Image.open(args.image).convert("L"))
    tokens = transcribe(model, vocab, img, device, args.beam)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    (out / f"{stem}.tokens.txt").write_text(" ".join(tokens) + "\n")
    (out / f"{stem}.musicxml").write_text(to_musicxml(tokens, stem))
    (out / f"{stem}.mid").write_bytes(to_midi(tokens))
    print(" ".join(tokens))
    print(f"wrote {out}/{stem}.tokens.txt, .musicxml, .mid")


if __name__ == "__main__":
    main()
