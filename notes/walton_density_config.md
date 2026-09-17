# Walton density run — why this configuration

Written before the run. Every choice below is argued from what the knob does
and what the scene is, not from what any earlier run produced. If a setting
here only makes sense because it once gave a tidy answer, it is the wrong
setting.

Camera: `walton-lighthouse-santa-cruz-ca`. Tool: `check_camera_geometry.py`
(patch-based, with the whole-frame coarse pass). Command at the end.

## Sampling: `--every 3`

The two routes are not equally easy. Registering January against a reference
in June asks the correlator to match two frames that differ in sun angle,
season, haze and tide as well as in camera position. Registering a frame
against the one before it asks far less. The gap between those two routes is
what sets the noise floor, so the floor is dominated by the harder of them —
and the sequential route gets easier the closer together the samples are.

At 7 days the two frames of a pair share almost nothing but the scene's
permanent structure. At 3 days they share weather and sun angle as well. The
cost is three times the frames; the benefit is a tighter gap, which is the
quantity the entire verdict rests on. That is the trade worth making.

Nothing about `--every 3` biases the answer toward "one epoch" or "several".
Denser sampling can only reveal steps that weekly sampling stepped over.

## Reference frame: derived, not forced

`check_camera_geometry.py` has no flag to force a reference, and that is
correct here. The reference is chosen by `best_reference()`: every frame is
correlated against a handful of probes spread through the record, and the one
with the best median peak-to-sidelobe ratio wins.

The two obvious alternatives are both wrong. The first frame is wrong because
a record can open on its worst day and every measurement is relative to the
reference, so a soft anchor degrades the whole run. The sharpest frame is
worse, because sharpness scored as gradient magnitude is maximised by noise —
a rainy or high-ISO frame beats any real scene, and then nothing registers
against the anchor at all.

"The frame the rest of the record matches best" is the property actually
wanted, and it is measured rather than assumed. Forcing a date would mean
choosing the anchor, which is the one choice most able to shape the result
without showing that it did.

## Search window: `--max-shift 150` (the default)

This is a PRIOR, and it is stated as one. A correlation surface spans the
whole frame, so a spurious peak 600 px away competes on equal terms with the
true peak 5 px away. Some bound is required; the only question is whether the
bound is honest.

150 px on a 2560×1920 frame is 5.9% of the width and 7.8% of the height. A
camera bolted to a building does not move that far in three days. The window
is wide enough to contain real mount settling, thermal drift and a modest
knock, and narrow enough to exclude the far side of the frame.

Crucially the run reports how many frames had their tallest peak *outside*
the window and were re-measured inside it. That count is the evidence for the
prior. If it is a handful, the prior held. If it is most of the record, the
prior is wrong and the run says so rather than hiding it — at which point the
answer is to raise the window and re-run, not to believe the numbers.

## Clarity floor: `--min-clarity 0.45` (the default)

A frame registers on the structure it contains, and fog removes structure
without removing the frame. A correlator handed an empty frame still returns
a number, and that number clears the confidence floor often enough to poison
the record. Confidence cannot catch this, because the failure is that there
is nothing to be confident about.

Clarity is the RMS gradient in grey levels per pixel — how much edge there is
to align on. The threshold is a fraction of *the record's own median* rather
than an absolute, so it travels between cameras with different optics and
exposure instead of being tuned per site. 0.45 keeps anything at least
roughly half as structured as a typical frame of this record, which excludes
whiteouts and keeps ordinary overcast.

Frames below it are not called wrong. They are called unmeasured, which is
what they are.

## Step threshold: `--step-px 0.5`

The tool does not take this number at face value. It computes
`threshold = max(--step-px, resolution)`, where `resolution` is three times
the error implied by the gap between the two routes. So `--step-px` is a
floor, not the threshold, and the only thing a value below the derived
resolution does is get out of the way.

That is exactly what is wanted. The default 3.0 is a guess made in advance of
any measurement; whenever the record measures itself better than 3 px, that
guess is the binding constraint and the record's own precision is discarded.
Passing 0.5 means the derived resolution decides in every case where it is
larger, and the printed line says which number was used.

This is not "lowering the bar to find more steps". The derived resolution is
three times the measured error, so a step that clears it is three sigma on
this record's own terms. If the record registers poorly, the derivation
raises the threshold on its own and the run reports that it did.

### CORRECTION, written after the run

The paragraph above was true of the **coarse** route and of nothing else, and
I did not check before asserting it. `--step-px` also set the patch-candidate
agreement tolerance, the patch-route step threshold and the per-date
trustworthiness test, none of which were floored by the derived resolution.
So 0.5 did get out of the way in the route I was reasoning about, and in the
same breath demanded that twelve patches on a 2560x1920 frame agree with each
other to half a pixel over 336 dates.

The run duly printed

    agreement between candidates (median px apart over the record, threshold 0)
    No 3 of 8 patches agree with each other to within 0 px.

That is not a finding about Walton. It is the flag, and the `threshold 0` is
0.5 rounded for display, which hid it further. **The patch cross-check from
that run says nothing and should not be read.**

The tool now has `--agree-px` for the tolerance (default 3.0, which is what
`check_geometry_strict.py` already used), every threshold runs through one
`not_finer_than_the_record` helper so the measured resolution can raise any of
them, and sub-pixel values no longer print as `0`. Re-run with the command at
the foot of this note; `--step-px 0.5` is still the right choice for the
reasons given above, and it now only does the thing those reasons are about.

## Frame margins: `--top-margin 100`

Walton's frames carry a composited banner across the top — roughly 70 px. A
composited overlay is identical in every frame, so it is perfectly stationary
high-contrast structure, which is the same hazard as a burned-in watermark:
it votes for "no motion" in exactly the frames where the answer matters, and
it raises confidence while destroying the measurement.

The tool has an automatic test for this (rows whose variation between frames
is far below the scene's), and it catches the banner in most windows. It is
not reliable across every window, and the cost of the margin when the
automatic test would have caught it anyway is 100 rows of sky. Stating it is
cheaper than depending on the detector.

No bottom margin: the bottom of this frame is the harbour and the built-up
shore, which is the most rigid content in the scene.

## Patch set: `--candidates 12`, `--roi-size 128` (both defaults)

Candidates are proposed by a picker and then filtered by an agreement test.
The picker looks at how a patch *appears*; the agreement test measures how it
*behaves* over the record. Only the second is evidence. More candidates
therefore costs almost nothing — the frames are already decoded, and a 128×128
FFT is trivial beside a 2560×1920 JPEG decode — and it is the only defence
against a frame whose sharpest-looking content is weather.

128 px is the patch edge. Bigger patches hold more structure and register a
soft scene better, but a patch cannot see a move larger than half its own
width. That limitation is handled separately by the whole-frame coarse pass,
which is not capped by patch size; the patches then measure the residual. So
128 does not have to be large enough to catch a big move, and keeping it small
means more independent cells on the same land.

Do not pass `--no-coarse`. It exists only to reproduce the failure the coarse
pass was written to fix.

## The March 2024 1280×720 span

Five frames in the record are 1280×720 against a 2560×1920 archive. That is
16:9 against 4:3 — not a scaled version of the same view, a different field of
view or a different crop of the sensor.

**These frames are not registered against the rest and never pooled with it.**
Until today both routes cropped a mismatched frame to the top-left corner it
shares with the reference and registered it anyway, which compares different
ground and lets a meaningless number vote in the median, the two-route gap and
the noise floor derived from that gap. Both routes now drop such frames,
count them, and say so. The span is reported on its own.

Five frames is not enough to say anything about geometry within the span. That
is the report, not a gap in it.

## The command

    python3 check_camera_geometry.py \
      --camera walton-lighthouse-santa-cruz-ca \
      --sample --cached --every 3 --top-margin 100 --step-px 0.5 \
      --max-shift 450 --contact-sheet --open

`--min-clarity` is left at its default deliberately, per the argument above.
`--agree-px` is left at its default because 3.0 is a tolerance on measurement
error and nothing here has measured that error to be different.

`--max-shift` is **no longer** at its default, and the reason is the test the
section above said to apply. The 150 px run reported 185 of 336 frames with
their tallest peak outside the window — 55%, not a handful — and its own
warning said the prior is likely wrong and the epochs provisional. That is the
prior failing its stated check, so the window is raised and the run repeated.
450 px is 17.6% of the width and 23.4% of the height: wide enough that a real
move of the size 55% of the frames are hinting at fits inside it, and still
short of the far side of the frame.

`--cached` is included because the frames are already on disk from the 150 px
run; nothing is re-downloaded and the comparison is against the same imagery.
