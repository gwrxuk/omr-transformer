"""Symbol error rate and finer-grained error breakdowns."""
from __future__ import annotations


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def split_note(tok: str) -> tuple[str, str]:
    """note-F#4_quarter -> ('F#4', 'quarter'); other tokens -> (tok, tok)."""
    if tok.startswith("note-"):
        p, d = tok[5:].split("_")
        return p, d
    if tok.startswith("rest-"):
        return "rest", tok[5:]
    return tok, tok


class Scores:
    """Accumulates corpus-level metrics over (prediction, reference) pairs."""

    def __init__(self) -> None:
        self.n = self.exact = 0
        self.sym_err = self.sym_ref = 0
        self.pitch_err = self.dur_err = self.part_ref = 0

    def add(self, pred: list[str], ref: list[str]) -> None:
        self.n += 1
        self.exact += pred == ref
        self.sym_err += edit_distance(pred, ref)
        self.sym_ref += len(ref)
        pp, pd = zip(*map(split_note, pred)) if pred else ((), ())
        rp, rd = zip(*map(split_note, ref))
        self.pitch_err += edit_distance(list(pp), list(rp))
        self.dur_err += edit_distance(list(pd), list(rd))
        self.part_ref += len(ref)

    def summary(self) -> dict[str, float]:
        return {
            "samples": self.n,
            "SER": self.sym_err / max(1, self.sym_ref),
            "pitch_ER": self.pitch_err / max(1, self.part_ref),
            "duration_ER": self.dur_err / max(1, self.part_ref),
            "sequence_acc": self.exact / max(1, self.n),
        }
