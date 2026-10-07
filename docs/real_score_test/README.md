# Out-of-domain check: a real orchestral score page

**Result: the model does not transfer to this page.** It was trained only on
synthetic single-voice staves (Section "Limits" of the main README), and this
sample breaks almost every assumption it was trained under. It is kept here as
an honest negative result and a baseline for future fine-tuning.

## Source

- Music: Gustav Holst, *The Planets*, Op. 32, I. "Mars, the Bringer of War",
  first page of the full score. The composition (1914–16) is in the public domain.
- Page image (`score.png`): from the product listing of the Dover full-score
  edition, *The Planets in Full Score* (Dover Music Scores, ISBN 0-486-29277-0):
  https://www.amazon.com/dp/0486292770
  (original link as provided: https://www.amazon.com/Planets-Score-Dover-Music-Scores/dp/0486292770).
  Used as a single low-resolution sample for research evaluation only.

## Method

The page is 1050 x 1256 px, compressed and thresholded, with staff lines about
3-4 px apart and mostly broken, so automatic staff detection failed. Six staves
were cropped by position (1st and 2nd violins, violas, cellos, double basses,
timpani I), each as a 46 px band scaled to the model's 128 px input height, and
transcribed with beam search (k = 4). Crops: `violin1.png` ... `timpani1.png`;
raw output: `predictions.json`.

## What the page contains vs. what the model knows

| Property of the page | In training data? |
|---|---|
| Alto and bass clefs on violas, cellos, basses | Only treble (G2) and bass (F4) clefs; no alto clef |
| 5/4 time | No: vocabulary has 2/4, 3/4, 4/4, 3/8, 6/8 |
| No key signature | Yes (C major) |
| Beamed eighths and triplets | No: the engraver draws flags only, no beams, no tuplets |
| Staff spacing about 3.5 px, broken lines, heavy compression | No: 10 px spacing, unbroken lines; degradation is milder |
| Dynamics and text (col legno, p) around the staff | No |

## Results

The opening is an ostinato: each string staff and the timpani repeat a single
pitch (G) in a 5/4 rhythm built from triplets, eighths and quarters.

| Staff | Clef read | Key read (page: none) | Time read (page: 5/4) | Pitches |
|---|---|---|---|---|
| Violins I | treble (correct) | G major (wrong) | none | repeated A3: repetition caught, pitch wrong |
| Violins II | treble (correct) | D major (wrong) | 3/8 (wrong) | mostly repeated A3, same pattern |
| Violas | treble (wrong, alto) | E-flat major (wrong) | none | varied F4/E-flat4 (wrong) |
| Cellos | treble (wrong, bass) | D major (wrong) | 3/4 (wrong) | varied E4-F#5 (wrong) |
| Double basses | treble (wrong, bass) | B-flat major (wrong) | 3/8 (wrong) | varied, up to C6 (wrong) |
| Timpani I | bass (correct) | E-flat major (wrong) | 3/8 (wrong) | varied E-flat2-C4 (wrong) |

No staff is transcribed usefully. The only signals that carry over are the
treble clef on the violins, the bass clef on the timpani, and the repeated-note
pattern on the violins. Rhythms come out as unrelated quarter and eighth values,
because beamed groups and triplets never appeared in training.

## What it would take

Fine-tuning on Camera-PrIMuS (real engraving with photo distortion), adding the
alto clef, more time signatures, beams and tuplets to the vocabulary and
engraver, and a staff-detection step that copes with broken lines. A page like
this one also needs scanning at a higher resolution (staff spacing of 10 px or
more) before any single-staff model can read it.
