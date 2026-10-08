# omr-transformer

End-to-end optical music recognition in PyTorch: a single-staff image goes in;
a sequence of music tokens, MusicXML and MIDI come out. The model is a CNN plus
Transformer encoder-decoder, pre-trained on generated staves and fine-tuned on
real engravings from the CC0 OpenScore Lieder corpus.

![real engravings](docs/openscore_examples.png)
*Real engravings from songs held out of training (OpenScore Lieder, CC0,
engraved with MuseScore 4). The fine-tuned model transcribes the first three
exactly; the last is the worst of 37 held-out excerpts, where a flat written
once holds for the rest of the measure and the model reads the unmarked
repetitions as naturals.*

## Results at a glance

| Test set | Synthetic-only model | Fine-tuned on real engravings |
|---|---|---|
| **Real engravings**, 37 excerpts from held-out OpenScore songs | 73.6% SER, 0% exact | **1.9% SER, 75.7% exact** |
| Synthetic clean, 1,000 staves | 2.8% | 1.8% |
| Synthetic scan-degraded, 1,000 staves | 7.3% | 5.2% |

Details: synthetic pre-training below; real engravings and fine-tuning in
[Real engravings (OpenScore Lieder)](#real-engravings-openscore-lieder) and
[docs/openscore_test/](docs/openscore_test/README.md). A low-resolution scanned
orchestral page still fails ([docs/real_score_test/](docs/real_score_test/README.md)).

## Synthetic pre-training results

Trained 4,000 steps (batch 32, about 128k generated staves) on an Apple M4 GPU
in 78 minutes. Test sets are 1,000 staves each from a seed range never used in
training or validation.

| Test set | SER | Pitch ER | Duration ER | Exact staves |
|---|---|---|---|---|
| Clean renders | **2.8%** | 2.5% | 1.0% | 67.4% |
| Scan-style degraded | **7.3%** | 6.3% | 2.9% | 53.1% |

SER is the symbol error rate: token-level edit distance divided by reference
length. Pitch ER and duration ER apply the same measure after splitting each
note token into its pitch and duration parts, so they show which half of the
symbol is wrong. Beam search (k=4) lowers scan SER from 6.9% to 6.3% on a
200-staff subset (`test_results_beam4_n200.json`).

![validation SER](docs/ser_curve.png)

**Where it fails.** Almost all of the ten most frequent confusions are
accidentals: a sharp, flat or natural read as the wrong sign or missed
(`C#4 -> C4`, `Gb2 -> G#2`). Only about 8% of notes carry an explicit
accidental, and the glyphs are small, so this is the class to target next:
oversample altered notes, or give the encoder more vertical resolution near the
notehead. Dots are the second weakness, as in the demo below.

**Demo** (`docs/demo_input.png`, degraded): the prediction matches the ground
truth on 14 of 15 symbols; the miss is the final note, a dotted half read as a plain half.

## How it works

```
image 1x128xW
  -> CNN: strided stem + 3 conv blocks (height /16, width /8)   -> 128 x 8 x W/8
  -> flatten height into channels, linear to d=256, sinusoidal positions
  -> Transformer encoder (4 layers, 8 heads, pre-norm)
  -> Transformer decoder (4 layers, causal self-attention, cross-attention)
  -> logits over 723 music tokens
```

8.6M parameters. Training uses teacher forcing, label smoothing 0.1, AdamW,
linear warmup and cosine decay, and gradient clipping at 1.0. The best
checkpoint is chosen on validation SER over the degraded set.

**Tokens** follow the PrIMuS "semantic" encoding (`clef-G2`, `keySignature-DM`,
`timeSignature-6/8`, `note-F#4_quarter.`, `rest-eighth`, `barline`), so a model
trained here can be fine-tuned on real PrIMuS / Camera-PrIMuS data without
changing the output head (`--primus-root`).

**Data.** `omr/sequence.py` generates musically valid staves: measures fill the
time signature exactly, melodies move mostly by step, notes follow the key
signature, and about 8% carry an explicit accidental. `omr/render.py` engraves
them (two clefs, seven keys, five time signatures, eight durations including
dotted values, ledger lines, flags and rests), and `omr/augment.py` adds
rotation, scaling, ink spread, blur, uneven lighting, noise and JPEG
compression. Examples of generated staves, clean and degraded:
[docs/examples.png](docs/examples.png). Each sample is seeded by its split and index, so validation and
test sets are fixed and runs are reproducible.

## Engineering notes

- **Apple MPS memory.** The first full run grew to 24 GB of GPU memory and
  stalled the machine in swap. With a fixed batch the footprint was flat at
  6.9 GB, which traced the growth to variable tensor shapes: MPS keeps cached
  buffers per shape and does not reuse them across shapes. The fix is to pad
  every batch to one shape (1024 px x 40 tokens; the widest generated staff is
  968 px), use a strided first convolution to shrink activations, and cap the
  allocator with `PYTORCH_MPS_HIGH_WATERMARK_RATIO` so a leak fails fast
  instead of swapping. Memory then stayed at 5.7 GB for the whole run.
- **Masking test.** `test_overfits_one_batch` trains a small model until it
  memorises four staves exactly. That catches off-by-one errors in the
  target shift and broken causal or padding masks, which otherwise only show up
  as a model that never converges.

## Usage

```bash
pip install -r requirements.txt
python -m omr.train --out runs/base --steps 4000          # train
python -m omr.evaluate --ckpt runs/base/best.pt --size 1000   # test sets
python -m omr.evaluate --ckpt runs/base/best.pt --size 200 --beam 4
python -m omr.predict --ckpt runs/base/best.pt --image staff.png --out out/
python scripts/make_figures.py runs/base
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q      # 57 tests
```

`predict` writes `<name>.tokens.txt`, `<name>.musicxml` (opens in MuseScore)
and `<name>.mid`.

## Real engravings (OpenScore Lieder)

Tested on 37 vocal-line excerpts from held-out songs of the CC0
[OpenScore Lieder Corpus](https://github.com/OpenScore/Lieder), engraved with
MuseScore 4:

| Model | OpenScore SER | Exact excerpts | Synthetic clean / degraded SER |
|---|---|---|---|
| Synthetic-only (`runs/base`) | 73.6% | 0% | 2.8% / 7.3% |
| + fine-tuned on 5,327 real engravings (`runs/real_ft`) | **1.9%** | **75.7%** | 1.8% / 5.2% |

The synthetic-only model read clefs and keys but output 3/8 for every time
signature, and its decoder forced barlines to fit that metre. Fine-tuning on
MuseScore renders of 1,242 other songs (split by song, so test songs are never
seen) fixed that without hurting the synthetic tests. Build and evaluation:
`python scripts/build_openscore_data.py`, then
`python -m omr.train --init runs/base/best.pt --real-root data/openscore_train --steps 3000 --lr 2e-4 --warmup 200 --out runs/real_ft`,
then `python scripts/openscore_eval.py --ckpt runs/real_ft/best.pt`.
Details: [docs/openscore_test/](docs/openscore_test/README.md).

## Out-of-domain check on a real score

Run on six staves cropped from the first page of Holst's *Mars* (public-domain
music; low-resolution page image from the Dover full-score listing, source in
the linked note), the model **does not transfer**: it misreads alto and bass
clefs as treble, invents key signatures, cannot express 5/4, and turns beamed
triplets into unrelated durations. Details, crops, raw predictions and source:
[docs/real_score_test/](docs/real_score_test/README.md).

## Writing new music from the corpus

The same token vocabulary also works for generation. `omr/melody.py` is a
small decoder-only Transformer (3.5M parameters, 4 layers, d=256) trained on
the vocal lines of the OpenScore Lieder songs, so its output can be exported,
engraved and read back by the OMR model.

- **Data** (`scripts/build_melody_corpus.py`): 2,502 phrases from 827 songs
  that the vocabulary can express (no ties, tuplets or chords). The split is
  by song, the same as the OMR data: 2,294 training phrases (764 songs), 103
  validation and 105 test. Training phrases are transposed into each of the
  7 vocabulary keys when every pitch stays in range (786k tokens).
- **Model fit** (held-out test songs, perplexity per token, lower is better):

  | Model | Test perplexity |
  |---|---|
  | Unigram (add-one) | 57.4 |
  | Bigram (add-0.01) | 22.4 |
  | Melody Transformer (best step 5,500 of 6,000) | **11.0** |

- **Sampling** (`scripts/compose.py`): nucleus sampling (T=0.9, p=0.92) with
  a mask that only allows durations fitting the time left in the bar, forces
  the barline when the bar is full, and requires at least one note per bar.
  For each piece, 128 samples are drawn. A sample is kept only if it ends
  on the tonic with a note of a quarter or longer, stays within 15 semitones,
  and shares no run longer than 8 notes with any training phrase (compared
  as interval + duration, so a transposed copy still counts). The kept
  sample with the highest mean log-probability is used.

| # | Start | Passed filters | Longest run shared with training | OMR read-back SER |
|---|---|---|---|---|
| 1 | F major, 3/4, from scratch | 7 / 128 | 6 notes | 0.0% |
| 2 | G major, 6/8, from scratch | 6 / 128 | 6 notes | 0.0% |
| 3 | E♭ major, 4/4, from scratch | 5 / 128 | 5 notes | 1.9% |
| 4 | First 2 bars of Harriet Abrams, *Crazy Jane* (test song) | 2 / 128 | 6 notes | 2.0% |
| 5 | First 2 bars of Brahms, *Der Schmied*, Op. 19 No. 4 (test song) | 10 / 128 | 7 notes | 0.0% |

Melody 4, continuing the opening of a song the model never saw in training
(bars 1-2 are Abrams; bars 3-16 are generated):

![Generated melody 4](docs/new_music/melody_4_preview.png)

Each piece is in [docs/new_music/](docs/new_music/) as MusicXML, MIDI, PDF,
PNG and MP3 (MuseScore 4 rendering), with every check in `report.json`.
Reading the engravings back with the fine-tuned OMR model (2-bar windows,
40 in total) gives SER 0.85% and 37/40 windows exact. Two errors drop the
second of two repeated quarter notes. The third reads two C♯ eighths as C,
because the sharp is written once and carries through the bar: the same
accidental-carry error seen on the OpenScore test set.

These are monophonic melodies without words or accompaniment. The filters
and the copy check catch obvious copying, but they do not judge musical
quality. A short shared run (5-7 notes) is common in tonal melody and is
reported per piece, with the training song it matches.

## Limits and next steps

- The training images are synthetic and engraved by a simple renderer: one
  voice, no beams, no chords, no slurs or dynamics. The degraded test set
  measures robustness to scan damage, not accuracy on real printed scores.
- Next: fine-tune on Camera-PrIMuS (87k real incipits with photo distortion)
  and report SER on its official split; add beams and chords to the renderer;
  try a CTC head as a faster baseline; target the accidental errors above.
- The checkpoint (`runs/base/best.pt`, 34 MB) is not committed.

## Layout

```
omr/vocab.py      token vocabulary (PrIMuS-compatible)
omr/sequence.py   random valid staves
omr/render.py     engraver
omr/augment.py    scan degradations
omr/data.py       synthetic and PrIMuS datasets, fixed-shape batching
omr/model.py      CNN + Transformer encoder-decoder, greedy and beam search
omr/train.py      training loop
omr/evaluate.py   test-set metrics and confusion analysis
omr/predict.py    image -> tokens / MusicXML / MIDI
omr/export.py     MusicXML and MIDI writers
omr/metrics.py    SER, pitch and duration error rates
omr/melody.py     melody language model and bar-constrained sampling
scripts/fetch_openscore.py      download + convert the OpenScore Lieder corpus
scripts/build_melody_corpus.py  vocal-line phrases for the melody model
scripts/train_melody_lm.py      train the melody model
scripts/compose.py              compose, copy-check, engrave, OMR read-back
tests/            63 tests
```
