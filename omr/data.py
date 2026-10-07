"""Datasets and batching.

SyntheticStaves generates sample `i` from a seed derived from (split, i), so
the validation and test sets are fixed and every run sees the same data.
PrimusDataset reads the public PrIMuS / Camera-PrIMuS layout for fine-tuning.
"""
from __future__ import annotations

import random
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augment import degrade
from .render import H, render
from .sequence import random_staff
from .vocab import Vocab

SPLIT_SALT = {"train": 1, "val": 2, "test": 3}


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HxW (white=255) -> float 1xHxW with ink=1, paper=0."""
    if img.shape[0] != H:
        w = max(8, round(img.shape[1] * H / img.shape[0]))
        img = np.asarray(Image.fromarray(img).resize((w, H), Image.BILINEAR))
    return torch.from_numpy(1.0 - img.astype(np.float32) / 255.0)[None]


class SyntheticStaves(Dataset):
    def __init__(self, vocab: Vocab, split: str, size: int,
                 degrade_strength: float | tuple[float, float] = 0.0, offset: int = 0):
        """degrade_strength: fixed value, or (lo, hi) to sample per image."""
        self.vocab, self.split, self.size = vocab, split, size
        self.strength, self.offset = degrade_strength, offset

    def __len__(self) -> int:
        return self.size

    def sample(self, i: int) -> tuple[np.ndarray, list[str]]:
        rng = random.Random(zlib.crc32(f"{self.split}:{i + self.offset}".encode()))
        seq = random_staff(rng)
        img = render(seq, rng, jitter=True)
        s = self.strength
        s = rng.uniform(*s) if isinstance(s, tuple) else s
        return degrade(img, rng, s), seq

    def __getitem__(self, i: int):
        img, seq = self.sample(i)
        return to_tensor(img), torch.tensor(self.vocab.encode(seq))


class PrimusDataset(Dataset):
    """PrIMuS layout: <root>/<id>/<id>.png and <id>.semantic (tab-separated tokens).

    Camera-PrIMuS adds <id>_distorted.jpg; pass distorted=True to use it.
    Tokens outside our vocabulary map to <unk>; the vocab can be rebuilt from
    the corpus with Vocab(tokens=...) before fine-tuning.
    """

    def __init__(self, vocab: Vocab, root: str | Path, ids: list[str] | None = None,
                 distorted: bool = False):
        self.vocab, self.root, self.distorted = vocab, Path(root), distorted
        self.ids = ids or sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        d = self.root / self.ids[i]
        name = f"{self.ids[i]}_distorted.jpg" if self.distorted else f"{self.ids[i]}.png"
        img = np.asarray(Image.open(d / name).convert("L"))
        seq = (d / f"{self.ids[i]}.semantic").read_text().split()
        return to_tensor(img), torch.tensor(self.vocab.encode(seq))


def collate(batch, pad_id: int, w_bucket: int = 128, l_bucket: int = 8,
            fixed_w: int | None = None, fixed_l: int | None = None):
    """Pad images on the right (paper=0) and targets with <pad>.

    On Apple MPS every distinct tensor shape gets its own cached buffers, and
    with varying widths the cache grew to 24 GB and pushed the machine into
    swap. Training therefore pads every batch to one fixed shape (fixed_w,
    fixed_l); otherwise sizes are rounded up to buckets.
    """
    imgs, tgts = zip(*batch)
    widths = torch.tensor([im.shape[-1] for im in imgs])
    W = int(widths.max())
    W = max(W, fixed_w) if fixed_w else W + (-W) % w_bucket  # multiple of 8 = encoder stride
    W += (-W) % 8
    x = torch.zeros(len(imgs), 1, H, W)
    for k, im in enumerate(imgs):
        x[k, :, :, : im.shape[-1]] = im
    L = max(len(t) for t in tgts)
    L = max(L, fixed_l) if fixed_l else L + (-L) % l_bucket
    y = torch.full((len(tgts), L), pad_id, dtype=torch.long)
    for k, t in enumerate(tgts):
        y[k, : len(t)] = t
    return x, widths, y
