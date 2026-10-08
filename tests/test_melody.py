import torch

from omr.melody import BarConstraint, MelodyLM, bar_length, sample, transpose_phrase
from omr.vocab import Vocab


def test_bar_length():
    assert bar_length("3/4") == 3
    assert bar_length("6/8") == 3
    assert bar_length("2/4") == 2


def test_transpose_keeps_diatonic_spelling():
    v = Vocab()
    header = ["clef-G2", "keySignature-CM", "timeSignature-2/4"]
    bars = [["note-C4_quarter", "note-F4_quarter"], ["note-B4_half"]]
    h, b = transpose_phrase(header, bars, "FM", v)
    assert h[1] == "keySignature-FM"
    assert b == [["note-F4_quarter", "note-Bb4_quarter"], ["note-E5_half"]]
    h, b = transpose_phrase(header, bars, "DM", v)
    assert b == [["note-D4_quarter", "note-G4_quarter"], ["note-C#5_half"]]


def test_transpose_out_of_range_returns_none():
    v = Vocab()
    header = ["clef-G2", "keySignature-CM", "timeSignature-2/4"]
    assert transpose_phrase(header, [["note-B5_half"]], "FM", v) is None  # up a 4th -> E6


def test_constraint_fills_bars_exactly():
    v = Vocab()
    c = BarConstraint(v, "3/4", n_bars=1)
    m = c.mask(len(v.tokens), "cpu")
    assert m[v.index["note-C5_half."]] == 0 and m[v.index["note-C5_whole"]] == float("-inf")
    assert m[v.index["barline"]] == float("-inf")
    c.step(v.index["note-C5_half"])
    m = c.mask(len(v.tokens), "cpu")
    assert m[v.index["note-C5_half"]] == float("-inf") and m[v.index["note-C5_quarter"]] == 0
    c.step(v.index["note-C5_quarter"])
    m = c.mask(len(v.tokens), "cpu")
    assert (m == 0).nonzero().flatten().tolist() == [v.index["barline"]]
    c.step(v.index["barline"])
    assert (c.mask(len(v.tokens), "cpu") == 0).nonzero().flatten().tolist() == [v.index["<eos>"]]


def test_sample_from_untrained_model_is_well_formed():
    v = Vocab()
    torch.manual_seed(0)
    lm = MelodyLM(len(v.tokens), d_model=32, nhead=4, layers=1, dropout=0.0).eval()
    prompt = ["<bos>", "clef-G2", "keySignature-GM", "timeSignature-6/8"]
    toks, lp = sample(lm, v, prompt, n_bars=4, generator=torch.Generator().manual_seed(1))
    assert toks[:3] == prompt[1:]
    assert toks.count("barline") == 4 and toks[-1] == "barline"
    bars, cur = [], 0.0
    from omr.melody import token_length
    for t in toks[3:]:
        if t == "barline":
            bars.append(cur)
            cur = 0
        else:
            cur += token_length(t)
    assert bars == [3, 3, 3, 3]
    assert lp < 0


def test_constraint_forbids_rest_only_bar():
    v = Vocab()
    c = BarConstraint(v, "2/4", n_bars=2)
    m = c.mask(len(v.tokens), "cpu")
    assert m[v.index["rest-half"]] == float("-inf") and m[v.index["rest-quarter"]] == 0
    c.step(v.index["rest-quarter"])
    m = c.mask(len(v.tokens), "cpu")
    assert m[v.index["rest-quarter"]] == float("-inf") and m[v.index["note-G4_quarter"]] == 0
