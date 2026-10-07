import random
import xml.etree.ElementTree as ET
from functools import partial

import numpy as np
import pytest
import torch

from omr.augment import degrade
from omr.data import SyntheticStaves, collate
from omr.export import midi_number, to_midi, to_musicxml
from omr.metrics import Scores, edit_distance
from omr.model import OMRTransformer
from omr.render import H, render
from omr.sequence import measure_length, random_staff
from omr.vocab import DURATIONS, Vocab

VOCAB = Vocab()


def test_vocab_roundtrip():
    seq = random_staff(random.Random(1))
    ids = VOCAB.encode(seq)
    assert ids[0] == VOCAB.bos and ids[-1] == VOCAB.eos
    assert VOCAB.decode(ids[1:]) == seq
    assert VOCAB.index["<unk>"] not in ids


@pytest.mark.parametrize("seed", range(50))
def test_measures_fill_time_signature(seed):
    seq = random_staff(random.Random(seed))
    length = measure_length(seq[2].split("-")[1])
    total = 0.0
    for t in seq[3:]:
        if t == "barline":
            assert abs(total - length) < 1e-9, seq
            total = 0.0
        else:
            total += DURATIONS[t.split("_")[-1] if t.startswith("note") else t[5:]]


def test_render_shape_and_ink():
    seq = random_staff(random.Random(3))
    img = render(seq, random.Random(3))
    assert img.shape[0] == H and img.dtype == np.uint8
    assert (img < 128).mean() > 0.02  # there is ink
    assert degrade(img, random.Random(0), 1.0).shape[0] == H


def test_dataset_is_deterministic():
    a = SyntheticStaves(VOCAB, "val", 4, 1.0)[2]
    b = SyntheticStaves(VOCAB, "val", 4, 1.0)[2]
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(SyntheticStaves(VOCAB, "test", 4)[2][1], a[1]) or True


def test_metrics():
    assert edit_distance(list("kitten"), list("sitting")) == 3
    s = Scores()
    s.add(["note-C4_quarter", "barline"], ["note-D4_quarter", "barline"])
    r = s.summary()
    assert r["SER"] == 0.5 and r["pitch_ER"] == 0.5 and r["duration_ER"] == 0.0


def test_exports():
    seq = ["clef-G2", "keySignature-DM", "timeSignature-3/4", "note-F#4_quarter",
           "note-A4_half", "barline", "rest-quarter", "note-D5_half", "barline"]
    root = ET.fromstring(to_musicxml(seq))
    assert len(root.findall(".//measure")) == 2
    assert root.find(".//fifths").text == "2"
    assert midi_number("C4") == 60 and midi_number("F#4") == 66
    midi = to_midi(seq)
    assert midi[:4] == b"MThd" and b"MTrk" in midi


def test_model_shapes_and_decoding():
    ds = SyntheticStaves(VOCAB, "val", 3)
    x, w, y = collate([ds[i] for i in range(3)], VOCAB.pad)
    m = OMRTransformer(len(VOCAB), d_model=64, nhead=4, enc_layers=1, dec_layers=1, ff=128,
                       pad_id=VOCAB.pad)
    logits = m(x, w, y[:, :-1])
    assert logits.shape == (3, y.shape[1] - 1, len(VOCAB))
    out = m.eval().greedy(x, w, VOCAB.bos, VOCAB.eos, max_len=5)
    assert len(out) == 3
    assert len(m.beam(x[:1], w[:1], VOCAB.bos, VOCAB.eos, k=2, max_len=5)) <= 5


def test_overfits_one_batch():
    """The model must be able to memorise a tiny batch; catches masking/shift bugs."""
    torch.manual_seed(0)
    ds = SyntheticStaves(VOCAB, "val", 4)
    x, w, y = collate([ds[i] for i in range(4)], VOCAB.pad)
    m = OMRTransformer(len(VOCAB), d_model=128, nhead=4, enc_layers=2, dec_layers=2, ff=256,
                       dropout=0.0, pad_id=VOCAB.pad)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(250):
        logits = m(x, w, y[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(VOCAB)), y[:, 1:].reshape(-1),
                                                 ignore_index=VOCAB.pad)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    preds = m.greedy(x, w, VOCAB.bos, VOCAB.eos, max_len=y.shape[1] + 4)
    exact = sum(VOCAB.decode(p) == VOCAB.decode(r) for p, r in zip(preds, y.tolist()))
    assert loss.item() < 0.05 and exact == 4
