"""Draw a token sequence as a single-staff image (grayscale, white background).

This is a deliberately simple engraver: no beams, one voice, accidentals
redrawn on every altered note. It exists so training data can be generated on
the fly with exact ground truth; real scans come later via the PrIMuS loader.
"""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .sequence import FLAT_ORDER, SHARP_ORDER, key_alteration
from .vocab import ALTERS, KEYS, parse_pitch, step_number

H = 128            # image height
SP = 10            # staff space in px
BOTTOM = 84        # y of bottom staff line
TOP = BOTTOM - 4 * SP
INK = 20
BOTTOM_LINE = {"G2": step_number("E", 4), "F4": step_number("G", 2)}
# treble-clef staff positions of key-signature accidentals
SHARP_POS = [8, 5, 9, 6, 3, 7, 4]
FLAT_POS = [4, 7, 3, 6, 2, 5, 1]


def y_of(pos: int) -> float:
    """Staff position (half-spaces above bottom line) -> y pixel."""
    return BOTTOM - pos * SP / 2


@lru_cache(maxsize=4)
def _font(size: int) -> ImageFont.ImageFont:
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


class Engraver:
    def __init__(self, rng: random.Random, jitter: bool = True):
        self.rng = rng
        self.jitter = jitter
        self.img = Image.new("L", (4000, H), 255)
        self.d = ImageDraw.Draw(self.img)
        self.x = 12.0
        self.lw = rng.choice([1, 1, 2]) if jitter else 1
        self.clef = "G2"
        self.key = "CM"

    # ---------- helpers ----------
    def gap(self, base: float) -> None:
        self.x += base + (self.rng.uniform(-2, 3) if self.jitter else 0)

    def line(self, *xy, w=None):
        self.d.line(xy, fill=INK, width=w or self.lw)

    # ---------- glyphs ----------
    def clef_glyph(self, clef: str) -> None:
        x = self.x
        if clef == "G2":
            self.line(x + 8, TOP - 14, x + 8, BOTTOM + 12, w=2)
            self.d.ellipse([x, y_of(2) - 9, x + 16, y_of(2) + 9], outline=INK, width=2)
            self.d.arc([x + 4, TOP - 18, x + 16, TOP - 4], 180, 360, fill=INK, width=2)
            self.d.ellipse([x + 2, BOTTOM + 8, x + 8, BOTTOM + 14], fill=INK)
        else:  # F4
            self.d.ellipse([x, y_of(6) - 4, x + 8, y_of(6) + 4], fill=INK)
            self.d.arc([x, y_of(6) - 7, x + 20, y_of(2) + 4], 270, 90, fill=INK, width=3)
            for dy in (-4, 4):
                self.d.ellipse([x + 22, y_of(6) + dy - 2, x + 26, y_of(6) + dy + 2], fill=INK)
        self.gap(30)

    def accidental(self, alter: int, x: float, y: float) -> None:
        if alter == 1:      # sharp
            self.line(x + 2, y - 9, x + 2, y + 9)
            self.line(x + 6, y - 10, x + 6, y + 8)
            self.line(x - 1, y - 2, x + 9, y - 5, w=2)
            self.line(x - 1, y + 4, x + 9, y + 1, w=2)
        elif alter == -1:   # flat
            self.line(x + 2, y - 14, x + 2, y + 4)
            self.d.arc([x + 2, y - 4, x + 9, y + 5], 270, 120, fill=INK, width=2)
        else:               # natural
            self.line(x + 1, y - 10, x + 1, y + 5)
            self.line(x + 7, y - 5, x + 7, y + 10)
            self.line(x + 1, y - 2, x + 7, y - 4, w=2)
            self.line(x + 1, y + 4, x + 7, y + 2, w=2)

    def key_glyph(self, key: str) -> None:
        n = KEYS[key]
        shift = 0 if self.clef == "G2" else -2
        positions = SHARP_POS[:n] if n > 0 else FLAT_POS[:-n]
        for p in positions:
            self.accidental(1 if n > 0 else -1, self.x, y_of(p + shift))
            self.x += 10
        self.gap(8)

    def time_glyph(self, ts: str) -> None:
        num, den = ts.split("/")
        f = _font(19)
        for txt, p in ((num, 6), (den, 2)):
            self.d.text((self.x, y_of(p)), txt, fill=INK, font=f, anchor="lm")
        self.gap(26)

    def ledger(self, x: float, pos: int) -> None:
        for p in range(-2, pos - 1, -2):
            self.line(x - 9, y_of(p), x + 9, y_of(p))
        for p in range(10, pos + 1, 2):
            self.line(x - 9, y_of(p), x + 9, y_of(p))

    def note_glyph(self, pitch: str, dur: str) -> None:
        letter, acc, octave = parse_pitch(pitch)
        pos = step_number(letter, octave) - BOTTOM_LINE[self.clef]
        alter = ALTERS[acc]
        y = y_of(pos)
        if alter != key_alteration(self.key, letter):
            self.accidental(alter, self.x, y)
            self.x += 13
        x = self.x + 6
        base = dur.rstrip(".")
        hollow = base in ("whole", "half")
        w = 7 if base == "whole" else 6
        self.ledger(x, pos)
        self.d.ellipse([x - w, y - 4.5, x + w, y + 4.5], outline=INK,
                       fill=None if hollow else INK, width=2)
        if base != "whole":
            up = pos < 4
            sx = x + w - 1 if up else x - w + 1
            ey = y - 35 if up else y + 35
            self.line(sx, y, sx, ey, w=2)
            for k in range({"eighth": 1, "sixteenth": 2}.get(base, 0)):
                fy = ey + (8 * k if up else -8 * k)
                self.line(sx, fy, sx + 9, fy + (12 if up else -12), w=2)
        if dur.endswith("."):
            dy = -SP / 2 if pos % 2 == 0 else 0
            self.d.ellipse([x + w + 3, y + dy - 2, x + w + 7, y + dy + 2], fill=INK)
        self.gap(18 + 10 * min(4.0, {"whole": 4, "half": 2.5, "quarter": 1.4,
                                    "eighth": 0.9, "sixteenth": 0.6}[base]))

    def rest_glyph(self, dur: str) -> None:
        x = self.x
        base = dur.rstrip(".")
        if base == "whole":
            self.d.rectangle([x, y_of(6), x + 12, y_of(6) + 5], fill=INK)
        elif base == "half":
            self.d.rectangle([x, y_of(4) - 5, x + 12, y_of(4)], fill=INK)
        elif base == "quarter":
            pts = [(x + 3, y_of(7)), (x + 9, y_of(5)), (x + 3, y_of(3)),
                   (x + 9, y_of(1.5)), (x + 4, y_of(0.5))]
            self.d.line(pts, fill=INK, width=3)
        else:
            n = 1 if base == "eighth" else 2
            for k in range(n):
                cy = y_of(5) + 8 * k
                self.d.ellipse([x + 1, cy - 3, x + 6, cy + 2], fill=INK)
                self.line(x + 5, cy, x + 10, cy - 4, w=2)
            self.line(x + 10, y_of(5) - 4, x + 4, y_of(1), w=2)
        if dur.endswith("."):
            self.d.ellipse([x + 15, y_of(5) - 2, x + 19, y_of(5) + 2], fill=INK)
        self.gap(26)

    def barline(self) -> None:
        self.line(self.x, TOP, self.x, BOTTOM, w=2)
        self.gap(14)

    # ---------- driver ----------
    def draw(self, seq: list[str]) -> np.ndarray:
        for tok in seq:
            kind, _, val = tok.partition("-")
            if kind == "clef":
                self.clef = val
                self.clef_glyph(val)
            elif kind == "keySignature":
                self.key = val
                self.key_glyph(val)
            elif kind == "timeSignature":
                self.time_glyph(val)
            elif kind == "note":
                pitch, dur = val.split("_")
                self.note_glyph(pitch, dur)
            elif kind == "rest":
                self.rest_glyph(val)
            elif tok == "barline":
                self.barline()
        width = int(self.x) + 10
        for k in range(5):
            y = BOTTOM - k * SP
            self.d.line([(4, y), (width - 4, y)], fill=INK, width=self.lw)
        return np.asarray(self.img.crop((0, 0, width, H)))


def render(seq: list[str], rng: random.Random | None = None, jitter: bool = True) -> np.ndarray:
    return Engraver(rng or random.Random(0), jitter).draw(seq)
