# Real engravings: OpenScore Lieder corpus

**Result: on 37 real engraved vocal-line excerpts the model's symbol error rate
is 73.6%, against 2.8% on its synthetic test set.** It reads clefs (37/37) and
key signatures (34/37) correctly, but it outputs 3/8 as the time signature for
every excerpt (correct 1/37), then places barlines to fit that wrong metre.
Even scored on notes and rests alone, ignoring barlines and the time
signature, 56% of pitches and 65% of durations are wrong.

## Source

Scores from the [OpenScore Lieder Corpus](https://github.com/OpenScore/Lieder)
(about 1,350 nineteenth-century songs transcribed in MuseScore), released under
the CC0 public-domain dedication. Project page:
https://fourscoreandmore.org/openscore/. Reference: Gotham, M. and Jonas, P.
(2021), *The OpenScore Lieder Corpus*, and Gotham, M. (2026), *Journal of Open
Humanities Data* 12: 43. The exact song file behind every excerpt is listed in
`results.json`.

## Method (`scripts/openscore_eval.py`)

1. Sample 60 songs from the corpus with a fixed seed (0).
2. Convert each MuseScore file to MusicXML with MuseScore 4.6.5 and take the
   vocal line (first part).
3. Take the first 2-3 measure window the model's vocabulary can express: treble
   or bass clef, 0-3 sharps or flats, a supported metre (2/4, 3/4, 4/4, 3/8,
   6/8), no ties, tuplets, chords, grace notes, pickups or mid-window changes,
   at least 4 notes, rests no more than a third of the notes, and no whole-bar
   rests (engravers draw those as a centred whole rest whatever the metre, which
   the label would not match). 37 of 60 songs had such a window.
4. Rebuild the excerpt without lyrics, dynamics or text, engrave it with
   MuseScore at 300 dpi, find the staff, scale it so the line spacing is 10 px
   as in training, and crop the 128 px band.
5. Transcribe with beam search (k = 4) and score against the exact source tokens.

The images are clean digital engravings, not scans, so this isolates the effect
of real engraving (fonts, beams, spacing) from scan damage.

## Results

| Measure | Real engravings (37) | Synthetic clean test (1,000) |
|---|---|---|
| Symbol error rate | **73.6%** | 2.8% |
| Pitch error rate (all tokens) | 60.6% | 2.5% |
| Duration error rate (all tokens) | 64.5% | 1.0% |
| Exact excerpts | 0% | 67.4% |
| Notes and rests only: pitch / duration error | 55.9% / 64.9% | n/a |
| Clef / key / time signature correct | 37 / 34 / 1 of 37 | n/a |

Median per-excerpt SER is 71%; the best is 56%
(`example_best.png`, Holmès, *Le ruban rose*):

```
truth  clef-G2 keySignature-BbM timeSignature-4/4 G4 D5 D5 C5 | D5(half) Eb5 F5 | D5 C5(8th) Bb4(8th) C5 D5 |
pred   clef-G2 keySignature-BbM timeSignature-3/8 G4 | D5 D5 | C5 | D5(half) | D5(half) E5 | D5 Bb4 |
```

The opening pitches are read correctly, but the predicted 3/8 metre makes the
decoder insert barlines every few notes and drop or merge notes to fit them.

## Why it fails

- **Glyph style.** Training images use a simple hand-written engraver and Arial
  numerals; MuseScore uses the Leland music font. Clefs and accidentals in a key
  signature are distinctive enough to survive the change; time-signature
  numerals and note shapes are not.
- **Beams.** Real engraving beams eighths and sixteenths; training data only has
  flags. Duration errors are the largest category.
- **Spacing.** MuseScore justifies measures across the page width, so notes are
  much further apart than in training.
- **The decoder as a language model.** Having learned that bars must fill the
  time signature, the decoder enforces a wrongly read metre on everything after
  it. One wrong token cascades into many.

The next step is the same as for the scanned-page check (`../real_score_test/`):
train on real engraving. Rendering corpus excerpts like these with MuseScore
would give unlimited real-engraving training data with exact labels; held-out
songs would then serve as the test set.
