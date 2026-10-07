"""Music-token vocabulary.

Tokens follow the PrIMuS "semantic" encoding so a model trained here can be
fine-tuned on the real PrIMuS / Camera-PrIMuS data without changing the head:

    clef-G2  keySignature-DM  timeSignature-3/4
    note-F#4_quarter  note-A4_eighth.  rest-half  barline
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]

CLEFS = ["G2", "F4"]
# key name -> number of sharps (+) or flats (-)
KEYS = {"CM": 0, "GM": 1, "DM": 2, "AM": 3, "FM": -1, "BbM": -2, "EbM": -3}
TIME_SIGS = ["2/4", "3/4", "4/4", "3/8", "6/8"]
# duration name -> length in quarter notes
DURATIONS = {
    "whole": 4.0, "half.": 3.0, "half": 2.0, "quarter.": 1.5,
    "quarter": 1.0, "eighth.": 0.75, "eighth": 0.5, "sixteenth": 0.25,
}
LETTERS = "CDEFGAB"
ALTERS = {"bb": -2, "b": -1, "": 0, "#": 1, "##": 2}
# written pitch range per clef, as diatonic step numbers (octave*7 + letter index)
RANGE = {"G2": ("A3", "C6"), "F4": ("C2", "E4")}


def step_number(letter: str, octave: int) -> int:
    return octave * 7 + LETTERS.index(letter)


def parse_pitch(p: str) -> tuple[str, str, int]:
    """'F#4' -> ('F', '#', 4)."""
    letter, rest = p[0], p[1:]
    i = 0
    while i < len(rest) and rest[i] in "#b":
        i += 1
    return letter, rest[:i], int(rest[i:])


def all_pitches(clef: str) -> list[str]:
    lo, hi = (parse_pitch(x) for x in RANGE[clef])
    out = []
    for n in range(step_number(lo[0], lo[2]), step_number(hi[0], hi[2]) + 1):
        letter, octave = LETTERS[n % 7], n // 7
        for acc in ("b", "", "#"):
            out.append(f"{letter}{acc}{octave}")
    return out


def build_tokens() -> list[str]:
    toks = list(SPECIALS)
    toks += [f"clef-{c}" for c in CLEFS]
    toks += [f"keySignature-{k}" for k in KEYS]
    toks += [f"timeSignature-{t}" for t in TIME_SIGS]
    toks += ["barline"]
    toks += [f"rest-{d}" for d in DURATIONS]
    pitches = sorted({p for c in CLEFS for p in all_pitches(c)},
                     key=lambda p: (parse_pitch(p)[2], LETTERS.index(p[0]), p))
    toks += [f"note-{p}_{d}" for p in pitches for d in DURATIONS]
    return toks


@dataclass
class Vocab:
    tokens: list[str] = field(default_factory=build_tokens)

    def __post_init__(self) -> None:
        self.index = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad(self) -> int:
        return self.index[PAD]

    @property
    def bos(self) -> int:
        return self.index[BOS]

    @property
    def eos(self) -> int:
        return self.index[EOS]

    def encode(self, seq: list[str], add_special: bool = True) -> list[int]:
        ids = [self.index.get(t, self.index[UNK]) for t in seq]
        return [self.bos, *ids, self.eos] if add_special else ids

    def decode(self, ids: list[int]) -> list[str]:
        out = []
        for i in ids:
            t = self.tokens[i]
            if t == EOS:
                break
            if t not in (PAD, BOS):
                out.append(t)
        return out

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.tokens, indent=0))

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls(tokens=json.loads(Path(path).read_text()))
