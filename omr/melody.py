"""A small melody language model over the OMR token vocabulary.

The OMR model reads staves into tokens; this model writes new token
sequences in the same vocabulary, so its output can be exported with
omr.export and engraved with MuseScore (and read back by the OMR model).

Sequence: <bos> clef key metre  bar-tokens barline  bar-tokens barline ... <eos>
Sampling is constrained so every bar adds up to the metre exactly.
"""
from __future__ import annotations

import math
from fractions import Fraction

import torch
from torch import nn

from .model import PositionalEncoding
from .vocab import DURATIONS, KEYS, Vocab

TONIC = {"CM": "C", "GM": "G", "DM": "D", "AM": "A", "FM": "F", "BbM": "B-", "EbM": "E-"}


def bar_length(metre: str) -> Fraction:
    beats, unit = metre.split("/")
    return Fraction(int(beats) * 4, int(unit))


def token_length(tok: str) -> Fraction | None:
    if tok.startswith("note-"):
        return Fraction(DURATIONS[tok.rsplit("_", 1)[1]]).limit_denominator(16)
    if tok.startswith("rest-"):
        return Fraction(DURATIONS[tok[5:]]).limit_denominator(16)
    return None


def transpose_phrase(header: list[str], bars: list[list[str]], target_key: str,
                     vocab: Vocab) -> tuple[list[str], list[list[str]]] | None:
    """Transpose to target_key by the smaller interval between tonics (keeps spelling diatonic).

    Returns None if any pitch would fall outside the vocabulary.
    """
    from music21 import interval, pitch
    src_key = header[1].split("-", 1)[1]
    if src_key == target_key:
        return header, bars
    a, b = pitch.Pitch(TONIC[src_key] + "4"), pitch.Pitch(TONIC[target_key] + "4")
    iv = interval.Interval(a, b)
    if iv.semitones > 6:
        iv = interval.Interval(a, pitch.Pitch(TONIC[target_key] + "3"))
    elif iv.semitones < -6:
        iv = interval.Interval(a, pitch.Pitch(TONIC[target_key] + "5"))
    out = []
    for bar in bars:
        nb = []
        for t in bar:
            if t.startswith("note-"):
                p, d = t[5:].rsplit("_", 1)
                q = pitch.Pitch(p.replace("b", "-") if len(p) > 2 else p).transpose(iv)
                if abs(q.alter) > 1:
                    return None
                acc = "#" if q.alter == 1 else ("b" if q.alter == -1 else "")
                t = f"note-{q.step}{acc}{q.octave}_{d}"
                if t not in vocab.index:
                    return None
            nb.append(t)
        out.append(nb)
    return [header[0], f"keySignature-{target_key}", header[2]], out


def encode(header: list[str], bars: list[list[str]], vocab: Vocab, eos: bool = True) -> list[int]:
    toks = ["<bos>", *header]
    for bar in bars:
        toks += bar + ["barline"]
    if eos:
        toks.append("<eos>")
    return [vocab.index[t] for t in toks]


class MelodyLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, layers: int = 4,
                 dropout: float = 0.2, max_len: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model, nhead, 4 * d_model, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.body = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.d = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B x T -> B x T x V
        T = x.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
        h = self.body(self.pos(self.embed(x) * math.sqrt(self.d)), mask=mask, is_causal=True)
        return self.head(self.norm(h))


class BarConstraint:
    """Which tokens are legal next, given the metre and the time left in the bar."""

    def __init__(self, vocab: Vocab, metre: str, n_bars: int):
        self.v, self.bar, self.n_bars = vocab, bar_length(metre), n_bars
        self.lengths = {i: token_length(t) for i, t in enumerate(vocab.tokens)}
        self.left, self.done_bars, self.bar_has_note = self.bar, 0, False

    def mask(self, vocab_size: int, device) -> torch.Tensor:
        m = torch.full((vocab_size,), float("-inf"), device=device)
        if self.done_bars == self.n_bars:
            m[self.v.index["<eos>"]] = 0
            return m
        if self.left == 0:
            m[self.v.index["barline"]] = 0
            return m
        for i, L in self.lengths.items():
            if L is not None and L <= self.left:
                # every bar gets at least one note: no rest may finish a bar that has none
                if not self.bar_has_note and L == self.left and self.v.tokens[i].startswith("rest-"):
                    continue
                m[i] = 0
        return m

    def step(self, tok_id: int) -> None:
        t = self.v.tokens[tok_id]
        if t == "barline":
            self.done_bars += 1
            self.left, self.bar_has_note = self.bar, False
        elif self.lengths[tok_id] is not None:
            self.left -= self.lengths[tok_id]
            self.bar_has_note |= t.startswith("note-")


@torch.no_grad()
def sample(model: MelodyLM, vocab: Vocab, prompt: list[str], n_bars: int, temperature: float = 0.9,
           top_p: float = 0.92, generator: torch.Generator | None = None, device="cpu"):
    """Continue `prompt` (starting <bos> clef key metre, optionally with whole bars) to n_bars.

    Returns (tokens without <bos>/<eos>, mean log-probability of the sampled tokens).
    """
    metre = prompt[3].split("-", 1)[1]
    c = BarConstraint(vocab, metre, n_bars)
    ids = [vocab.index[t] for t in prompt]
    for i in ids[4:]:
        c.step(i)
    logps = []
    V = len(vocab.tokens)
    while True:
        x = torch.tensor([ids[-511:]], device=device)
        logits = model(x)[0, -1].float().cpu()
        logits = logits + c.mask(V, "cpu")
        logp = torch.log_softmax(logits, -1)
        probs = torch.softmax(logits / temperature, -1)
        sp, si = probs.sort(descending=True)
        keep = sp.cumsum(0) - sp < top_p
        sp = sp * keep
        nxt = si[torch.multinomial(sp / sp.sum(), 1, generator=generator)].item()
        logps.append(logp[nxt].item())
        ids.append(nxt)
        c.step(nxt)
        if vocab.tokens[nxt] == "<eos>":
            break
    toks = [vocab.tokens[i] for i in ids[1:-1]]
    return toks, sum(logps) / len(logps)
