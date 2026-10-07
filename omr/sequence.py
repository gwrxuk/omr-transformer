"""Random but musically valid monophonic staves.

Every measure is filled exactly to the time signature, pitches move mostly by
step with occasional leaps, notes follow the key signature, and a small share
carry an explicit accidental (so the model must read the sign, not just the key).
"""
from __future__ import annotations

import random

from .vocab import (ALTERS, DURATIONS, KEYS, LETTERS, RANGE, TIME_SIGS,
                    parse_pitch, step_number)

SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"


def key_alteration(key: str, letter: str) -> int:
    n = KEYS[key]
    if n > 0 and letter in SHARP_ORDER[:n]:
        return 1
    if n < 0 and letter in FLAT_ORDER[:-n]:
        return -1
    return 0


def measure_length(ts: str) -> float:
    num, den = (int(x) for x in ts.split("/"))
    return num * 4.0 / den


def fill_measure(rng: random.Random, length: float) -> list[str]:
    """Pick durations that sum exactly to `length` quarter notes."""
    out, left = [], length
    while left > 1e-9:
        options = [d for d, q in DURATIONS.items() if q <= left + 1e-9]
        # favour common values
        weights = [{"quarter": 6, "eighth": 5, "half": 3, "sixteenth": 2,
                    "quarter.": 2, "eighth.": 1, "half.": 1, "whole": 1}[d]
                   for d in options]
        d = rng.choices(options, weights)[0]
        out.append(d)
        left -= DURATIONS[d]
    return out


def pitch_name(step: int, alter: int) -> str:
    acc = {v: k for k, v in ALTERS.items()}[alter]
    return f"{LETTERS[step % 7]}{acc}{step // 7}"


def random_staff(rng: random.Random, n_measures: tuple[int, int] = (2, 4),
                 accidental_rate: float = 0.08, rest_rate: float = 0.12) -> list[str]:
    clef = rng.choice(list(RANGE))
    key = rng.choice(list(KEYS))
    ts = rng.choice(TIME_SIGS)
    lo, hi = (parse_pitch(x) for x in RANGE[clef])
    lo_s, hi_s = step_number(lo[0], lo[2]), step_number(hi[0], hi[2])
    step = rng.randint(lo_s + 2, hi_s - 2)

    seq = [f"clef-{clef}", f"keySignature-{key}", f"timeSignature-{ts}"]
    for m in range(rng.randint(*n_measures)):
        for d in fill_measure(rng, measure_length(ts)):
            if rng.random() < rest_rate:
                seq.append(f"rest-{d}")
                continue
            move = rng.choices([-2, -1, 0, 1, 2, rng.randint(-5, 5)],
                               [2, 6, 2, 6, 2, 1])[0]
            step = min(max(step + move, lo_s), hi_s)
            alter = key_alteration(key, LETTERS[step % 7])
            if rng.random() < accidental_rate:
                alter = rng.choice([a for a in (-1, 0, 1) if a != alter])
            seq.append(f"note-{pitch_name(step, alter)}_{d}")
        seq.append("barline")
    return seq
