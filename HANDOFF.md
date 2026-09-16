# Handoff brief — for the Claude that plans this project

Self-contained. Written for a model that writes prompts for Claude Code and
cannot see the terminal. Leads with what is settled and what is dead, because
re-proposing a ruled-out test costs a day.

Last regenerated: 2026-09-16.

> **Three results dominate, and two of them changed today.**
>
> **1. Detector confidence does not rank rips — but 2.7% is NOT the detector's
> precision.** 360 hand labels: high-confidence band 3/116, low band 3/116,
> Fisher p = 1.00. That comparison stands. The absolute figure does not: a
> blind 60-frame calibration against the RipAID annotators shows the labeller
> catches only **19% of rips a trained annotator draws** (54% counting his
> doubts) while agreeing on 96% of negatives. Correcting for that puts the
> detector's true precision near **14%, plausibly 5-20%** — still poor, five
> times the headline, and not measurable more precisely without better labels.
>
> **2. Walton's "year effect" was a score-threshold change on 2024-12-13**,
> floor 0.5075 -> 0.7033, +30 sd. Not weather, not a retrain. Cut Walton at
> that date or read it at one floor.
>
> **3. The midday effect is REAL and it is NOT the camera.** Walton's detection
> rate carries a sun term peaking at bearing 185 — 5 deg from solar noon, 21
> from the camera. Human annotators at Cala Millor, a camera facing 90 deg
> AWAY from solar noon, draw bigger (dR2 +0.0108, p=0.015) and more numerous
> (+0.0160, p=0.0019) rips near solar noon too. A person sees more rip near
> midday. The detector is tracking something real, not lens flare.

---

## What the project is

WebCOOS publishes rip-detector output for 8 beach cameras. We join it hourly to
observed and modelled conditions and ask what drives it. The standing hazard:
a driver of the DETECTOR (light, contrast, texture) is indistinguishable from a
driver of the RIP, because the label is model confidence, not a verified rip.

| | Walton Lighthouse, Santa Cruz CA | Hampton Inn Oceanfront S, Virginia Beach |
|---|---|---|
| analysable hours, class-filtered | 8,405 | 1,938 |
| ...split at 2024-12-13 | 1,400 pre / 7,005 post | post era only |
| rip-class record | **2024-06-24** to 2026-09-15 | 2026-04-25 to 2026-08-29 |
| frames, rip class | 28,721 of 35,394 (81%) | 2,352 of 2,352 (**100%**) |
| usable days for a time series | **643** | 81 |
| busiest 64px detection cell | 0.8% | 7.7% |
| nearshore waves | CDIP MOP SC130, 98% | none |
| buoy | 46236 — 22%, all post-split (2025-08-31 to 2026-08-31) | none |
| shore normal | 206.0 deg, PUBLISHED by CDIP | 90.0 deg, **ASSUMED** |
| cloud cover | sidecar, 100% | sidecar, 100% |

Three models publish into one feed, separable exactly by `score_classes`:

| model | class | cameras |
|---|---|---|
| `ripdetect_walton / yolov8x_1.1` | `rip_current` | Walton, Virginia Beach, both Panama City |
| `rip_current_detector / 1` | `rip` | Corolla Hampton Inn, Corolla Sailfish, Carova |
| `yolo / v8n` | COCO objects | Walton + both Corollas, 2024-05-31 to 2024-06-19 ONLY |

| camera | frames | rip class | usable days |
|---|---|---|---|
| Walton | 35,394 | 28,721 | 643 |
| Virginia Beach | 2,352 | 2,352 | 81 |
| Panama City west | 980 | 980 | 28 |
| Panama City east | 709 | 709 | 27 |
| Corolla Hampton Inn | 10,766 | 1,064 | 34 |
| Corolla Sailfish | 10,239 | 472 | 14 |
| Carova | 34 | 34 | 1 |

**RipAID** (Zenodo, v1.0.0 = 10.5281/zenodo.15082427, v2.0.0 = 10.5281/zenodo.
18196300) supplies human-drawn rip annotations. v2.0.0 is 6,789 images = 2,815
SIRENA fixed-camera frames + 2,944 RipScout aerial/drone + 1,030 UFSC CoastSnap
smartphone. **Only the 2,815 SIRENA frames** have a bearing, a lat/lon and a
clock; `load_ripaid.frames_from()` keeps those and reports the rest as dropped.

---

## SETTLED — cite these, do not re-derive

**A single observer on a single still misses most rips.** 60 RipAID frames
labelled blind (30 annotated rips, 30 real negatives, doubt and sediment frames
excluded), compared to the annotators:

| doubt handled as | you / their rips | you / their nones | Cohen's kappa |
|---|---|---|---|
| excluded (n=39) | **3/16 = 19%** | 22/23 = 96% | +0.163 [−0.050, +0.431] |
| counted as yes (n=56) | **15/28 = 54%** | 22/28 = 79% | +0.321 [+0.080, +0.543] |
| counted as no (n=56) | 3/28 = 11% | 27/28 = 96% | +0.071 [−0.053, +0.221] |

25 misses, 1 false alarm. The profile is a conservative reader: nearly never
calls a rip that is not there, and does not see most of the ones that are. The
strict interval includes zero.

**Therefore 2.7% is a joint measurement of detector AND labeller.** Measured
precision = true precision x labeller sensitivity (the additive false-alarm
term is zero: the Walton never-fired cells were 0/70). Strict: 2.59% / 0.19 =
**14%**. Doubt-as-yes: the 7-11% per-cell bracket / 0.54 = **13-21%**. Quote
the detector's precision as **roughly 5-20%, best estimate near 14%**, and say
it rests on the labeller's RipAID sensitivity transferring to Walton's camera,
which is the weak link.

**What that does NOT touch: "confidence does not rank rips."** High 3/116
against low 3/116 is a comparison BETWEEN bands, and attenuation hits both
equally. Fisher p = 1.00 stands unaltered. Only the absolute figure moves.

**Sun position changes what a PERSON sees, not just what the detector fires
on.** RipAID v2.0.0, within frames that already contain an annotated rip — the
one question the dataset's selection cannot reach — at Cala Millor, whose
camera faces 90 deg and whose solar noon is 180, a 90 deg separation:

| target | peak bearing | dR2 | p | at camera (90) | at noon (180) |
|---|---|---|---|---|---|
| drawn size, z within camera | 185 | +0.0108 | 0.015 | +0.0029 | **+0.0092** |
| number of rips drawn | 215 | +0.0160 | 0.0019 | +0.0020 | **+0.0090** |
| annotator doubt | 5 | +0.0044 | 0.19 | — | — |

Both significant targets peak near solar noon at a camera pointing 90 deg away
from it. **For this to be a camera effect, Cala Millor's seaward bearing would
have to be within ~10 deg of due south; it is on Mallorca's EAST coast.** The
map-read bearing being imprecise does not matter, only being 90 deg wrong would.
Annotator doubt shows nothing (p=0.19), so "people hedge more at some sun
angles" is not the mechanism.

Son Bou behaved as predicted and settles nothing: its shore normal IS 180, a
0 deg separation, so every verdict reads "cannot separate". Its effects are
larger and highly significant (drawn size +0.0275 p=1.5e-05, count +0.0428
p=1.9e-08), which corroborates that a sun-position effect exists without
saying which pole owns it.

**Walton's own rotation agrees.** dR2 against every bearing, 5 deg grid,
variant A: detection_rate peaks at 185 (+0.0486), detections at 185 (+0.0573),
bbox_area_max at 175. The camera's 206 collects 67% of the maximum; solar noon
is 180. Trough +0.0003 at 120, a 160-fold range, so the sweep has real
contrast. Under `pooled` both count targets peak at exactly 180.

**The sun terms drive the detection RATE.** `sun_in_view` + `sun_glare` over a
base of wave height/period, tide, onshore wind, wind speed, temperature,
precipitation, 48h rain, solar elevation, cloud:

| variant | detection_rate | detections |
|---|---|---|
| pooled (class-filtered) | +0.0293, F=131.5 | +0.0348, F=159.2 |
| **A, one 0.70 floor** | **+0.0325, F=148.6** | **+0.0381, F=179.3** |
| B, pre era (n=1,400) | +0.0224, F=21.9 | +0.0559, F=58.7 |
| B, post era (n=6,862) | +0.0293, F=109.3 | +0.0299, F=114.1 |

All p below 1e-9. Scale: `mop_wave_height` rho_hrmo +0.2205 (detections, A).
dR2 and rho-squared are different quantities — never divide one by the other.

**The 2024-12-13 threshold change.** `score_floor` 0.5075 -> 0.7033, +30 sd
raw, **+35.7 sd residualized** on wave height and cloud. `score_median` +0.098
(+6.4 sd), `score_p90` +0.048 (+2.4 sd), same date. Class-filtered histograms
read 0.500 through 2024-12 and 0.700 from 2025-01. Model version did NOT
change. A CENSORING change: everything 0.50-0.70 stopped being published.

**Re-censoring the pre era to 0.70 removes 69.1% of its detections:** 7,400
detected frames -> 2,283, 9,236 detections -> 3,273, 1,008 detection-hours ->
645. Anything pooled across 2024-12-13 uncorrected compares a record where two
thirds of one side does not exist on the other.

**Reading Walton at one floor STRENGTHENS the ocean signal.** `mop_wave_height`
rho_hrmo, pooled -> A: detections +0.1826 -> **+0.2205**, rate +0.1451 ->
**+0.1816**. Both eras agree. The old floor's weak detections were diluting the
physics, which is evidence the correction corrects rather than merely differs.

**Cloud cover raises largest-box size at both sites.** VB rho +0.247 (+0.211
hour/month removed), single-era and 100% rip class so never needed correcting.
Walton rip frames: pooled +0.137 (+0.143), A +0.162 (+0.159), B post +0.163
(+0.167), B pre +0.041 (+0.076). The one cross-site replication, clean at both.

**The nearshore model beats the distant buoy — POST-ERA only.** On the 1,884
hours where MOP and buoy 46236 both report, demeaned within them: detection
rate 3.15x (+0.180 vs +0.057), detections 2.93x (+0.199 vs +0.068), largest box
1.54x (+0.235 vs +0.153). A ratio of two modest correlations — quote the ratio,
not the strength.

**The object stream was a 20-day fleet-wide event.** `yolo/v8n` ran
**2024-05-31 to 2024-06-19** and stopped dead — at Walton AND both Corollas, on
the same two dates, across two rip-model families and opposite coasts. A
WebCOOS platform action, now dated. So at Walton the class filter is **exactly
a truncation of the first 24 days**: `rip_current` starts 2024-06-24, the two
never overlap, every month from 2024-07 is 0% object, and 28,721 + 6,673 =
35,394 exactly. Measured two ways — unfiltered hourly table 5,918 hours from
2024-05-31 21:00 against filtered 5,719, and pre-era detection-hours 1,207
against 1,008 — both give 199 hours. Virginia Beach, both Panama City views and
Carova have **zero object frames ever** (deployed after). Walton has **no
blank-class frames at all**. **October 2024 is missing entirely** from Walton's
table, inside the thin pre-era half.

**Panama City west is half one fixed object.** 464 of 980 detections in one
64px cell at (960,192), centroids varying 10.2 x 5.5 px — 41% of the 18.5px
spread an even scatter would show.

**Walton is NOT dominated by a fixed feature.** Busiest cell 0.8% of 28,721
detections across 746 occupied cells. Hand labelling does show boxes on a rock
jetty and moored boats, real but minor.

**Only Walton can carry a time series.** 643 usable days against 81, 34, 28,
27, 14 and 1.

**Box coordinates are source pixels.** Walton stills are 2560x1920; payload
coordinates reach 2490 x 1919.

**Virginia Beach's air-temperature result is not an artifact of light.** All
four targets read SUPPRESSED at 1.46x-1.73x — adding light terms makes the
temperature coefficient GROW.

---

## RULED OUT — do not propose these again

- **Camera-specific glare / lens flare as the midday mechanism.** Two
  independent lines kill it. Walton's rotation peaks at 185, which is 21 deg
  from its camera and 5 from solar noon. And human annotators at Cala Millor,
  a camera 90 deg from solar noon, peak at solar noon anyway. A lens story
  predicts the peak AT the camera at both sites and gets it at neither.
- **2.7% as a measurement of the detector.** It is detector x labeller. See
  SETTLED. Any future precision figure from single-still labelling carries the
  same attenuation and must be reported as a lower bound.
- **The front/behind sun split as a way to separate lens from sea breeze.** At
  Walton sun-behind is 06:00-09:00 local and nothing else, so the split is
  morning-versus-midday and every story predicts a front-heavy result. It ran:
  +0.0372 against +0.0005, enormous, and decides nothing.
- **Afternoon sea breeze as what the sun terms carry.** An afternoon effect is
  azimuth ~220-260 at Walton and the rotation would peak there. It peaks at 185.
- **The published bearing-term F-test as evidence that 206 deg is special.**
  cos(az-b) = cos(az)cos(b) + sin(az)sin(b), so `sun_in_view` IS the azimuth
  harmonics rotated: fitted R2 on cos(az), sin(az) is 1.000000 with
  coefficients -0.8989, -0.4381 = cos(206), sin(206). The comparator holds no
  azimuth, so every bearing buys something. Only the rotation distinguishes.
- **Son Bou as a test site for camera-versus-midday.** Its shore normal is 180
  and so is solar noon — a 0 deg separation. Structurally incapable whatever
  its n. Cala Millor (90 deg apart) is the discriminator.
- **RipAID rip PRESENCE as a target.** Frames were pulled around lifeguard
  sightings +/-2h and "most of the images that did not show rip currents were
  removed". The run confirms the damage: the positive rate swings 0.40-0.67
  across azimuth bins at Cala Millor and 0.19-0.76 at Son Bou, which is the
  curator's retention, not the ocean.
- **The four rain correlations reversing between years.** Gone on rip frames
  under every variant: all four pairs POSITIVE on both sides (pre +0.128 to
  +0.172, post +0.050 to +0.076, every p below 1e-5). It was the 20-day object
  block in the 2024 half — 6,582 detections in June alone against a typical
  Walton month of 700-2,600 — not "rain empties a beach".
- **The MOP-vs-buoy ratio as a cross-era result.** Buoy 46236 runs 2025-08-31
  to 2026-08-31, entirely post-split. Zero matched hours in the low-floor era.
- **VB temperature as a GLARE proxy** (bearing terms dR2 <= 0.0007, p
  0.36-0.97), **as DAYLIGHT** (solar elevation moves the coefficient the wrong
  way), **as HAZE** (cloud moves it 2-4%, sign inconsistent, and cloud's own
  correlation is POSITIVE), **as people-on-the-beach** (VB is 2,352 of 2,352
  rip class; no person detections there, ever).
- **A letterbox / model-space transform for box coordinates.** Coordinates
  reach 97-100% of the still's dimensions. Six transforms built; none needed.
- **A fixed scene feature explaining Walton's driver results.** Busiest cell 0.8%.
- **Walton's year effect as weather.** It is the threshold change, surviving
  residualizing at +35.7 sd.
- **"Three cameras have never detected a rip."** Corolla Hampton Inn, Corolla
  Sailfish and Carova emit class `rip` from a different model. Carova is 100%
  rip. Reached twice by different routes — any code bucketing classes must
  match every name in `RIP_CLASSES`, not just `rip_current`.
- **The 2026-08-01..08 Walton / Panama City coincidence, as firmly
  unexplained.** Still unexplained, but weaker than it was: score FLOORS are
  set per model family (Corolla near 0.70 from Feb 2024 while Walton sat at
  0.50), yet the platform demonstrably acts fleet-wide across families on a
  single day. A same-week coincidence is not evidence against a platform cause.

---

## OPEN — ranked

0. **What is the midday mechanism?** The largest live question now that the
   effect is established and the camera is excluded. Two readings remain and
   nothing separates them: sun angle changes VISIBILITY (glint geometry, how
   far into the water column a viewer sees, contrast against the surface), or
   rips genuinely differ near midday (tidal phase correlating with time of day,
   afternoon-morning wind asymmetry). The RipAID result cannot tell them apart
   — a person drawing bigger rips at noon fits both.
   Candidate test: the two mechanisms predict different TIDE couplings. A
   visibility effect should be independent of tide; a real-occurrence effect
   should move with it. Walton has tide at 100% coverage and 8,405 hours.

1. **Re-estimate Walton's precision properly, or stop quoting one.** The
   correction gives ~14% with an interval no one should defend. Two ways out:
   label a second RipAID calibration set to tighten the sensitivity estimate
   (cheap, another hour), or relabel the Walton sample knowing what a rip looks
   like after seeing the annotators' 25 misses (better, slower). The second
   also repairs OPEN #2, which is currently unanswerable.

2. **What are the false positives made of?** Unanswerable as things stand: the
   notes column was left blank on all 60 calibration frames and on the Walton
   misses, so there is no record of WHY anything was called. Any relabelling
   must fill it.

3. **What IS Virginia Beach's temperature tracking?** Four explanations dead;
   the -0.36 is still there and still suppressed by light. Proposed: a
   sea-breeze / storminess index. Walton may echo it — in the pre era
   `temperature_2m` is the top predictor of both count targets and flips sign
   under controls (-0.0595 -> **+0.1867** rate, -0.0530 -> **+0.1895**
   detections with hour and month removed). n=1,400, one era: a lead.

4. **VB's shore normal is a guess (90.0 deg).** Every onshore wind and wave
   component there is computed from it. `check_camera_geometry.py` exists.

5. **Deck slide 6** still carries unmatched-hours ratios and two verdict flags
   the matched-hours run changed, plus everything above.

---

## CONSTRAINTS the planner must respect

**Any precision from single-still labelling is attenuated ~5x.** The labeller
catches 19% of annotator-drawn rips (54% counting doubt). Report such figures
as lower bounds, name the sensitivity used, and never compare an absolute
precision from this project against a published one without it. Comparisons
BETWEEN bands of the same labelling are unaffected.

**Filter by `score_classes` first.** Two rip names (`rip_current`, `rip`) and
one object stream share the column. Blank class lists are observed zeros and
must be KEPT where a camera has any — Walton has none. At Walton the filter is
a truncation of the first 24 days; at VB, both Panama City views and Carova a
no-op.

**Cut Walton at 2024-12-13, or read it at one 0.70 floor.**

    python analyze_walton_eras.py --keep-workspace data/era_workspace
    python analyze_walton_eras.py --check-recensoring data/era_workspace/A/data/rip_detection
    python analyze_walton_eras.py --check-recensoring data/rip_detection

The second must report the pre era APPLIED and the third NOT APPLIED. If they
agree, the correction silently did nothing.

**Variant A has a residual bias, quotable wherever A is.** The frame table
carries one score per FRAME, not per detection, so re-censoring demotes whole
frames: a pre-era frame holding a 0.9 and a 0.55 keeps both. 506 Walton frames
prove it (`score_mean < 0.70 <= score_max`), and that is a LOWER bound.

**RipAID inputs.** `load_ripaid.frames_from(path)` takes a v1.0.0 COCO export
or a v2.0.0 YOLO-OBB root. The v2.0.0 download ships no data.yaml; the class
mapping is recovered from counts (4,103 / 1,437 / 4,591 for 0 / 1 / 2, matching
the README exactly, so it is forced) and re-verified on every load. `sediment`,
new in v2.0.0, leaves both calibration strata for the same reason `doubt` does.
YOLO coordinates are NORMALISED, so area there is a fraction of the frame where
the COCO and WebCOOS paths give pixels — never pool them. Harmless within this
project because every area analysis z-scores within a camera. The files carry
six decimal places, so a recovered box is exact to ~2e-7 in area and ~1.2e-5
degrees; tolerances below that fail on correct arithmetic.

**Any bearing-derived term carries the whole azimuth unless azimuth is in the
comparator.** To ask whether a specific bearing matters, rotate it and race the
peak against the SOLAR-NOON azimuth, not against a tolerance —
`analyze_glare_split.classify_peak`. Pass lat/lon to `solar_noon_bearing`:
without them it takes the hourly argmax of elevation, which rounds solar noon
to the hour and lands 4-6 deg off, half the deciding margin.

**Open-Meteo bills by variable-hours.** Six ERA5 variables over Walton's
20,832-hour window to get one costs 6x quota and returned HTTP 429.
`pull_cloud_cover.py` fetches one variable into a sidecar. One site at a time.

**Never re-pull observations on the default one-year window.** It narrows a
site's conditions file below its rip record and silently shrinks every n.

**Concurrency.** `pull_rip_detection.py` hits only app.webcoos.org, safe
alongside anything. `pull_site_observations.py` and `pull_cloud_cover.py` share
the Open-Meteo quota — one at a time. `wq/` writes only `data/wq/**` plus
`wq_manifest.json` and contends with nothing.

**Read rho_hrmo, except for solar elevation** — hour-of-day IS solar elevation,
so demeaning by hour-by-month removes it by construction.

**`bbox_area_max` is pixels** on the WebCOOS and COCO paths. Comparable within
a camera, meaningless across, in either unit.

**Match a still to its detection's timestamp, not its hour.** The labelling
sample was built twice on the wrong images; the first draw averaged 606s off.

**Read a detection cluster's spread against `cell / sqrt(12)`** — 18.5px at
64px. A spread near it is NO clustering.

**`analyze_drivers.py` is shared infrastructure; its defaults are frozen.** New
questions get new scripts that import it. A variant needing a different
`rip_<slug>_hourly.csv` gets a mirror of `data/` — real directories, symlinked
files — with `ad.DATA_DIR` repointed. glob's `**` DOES follow symlinked
directories, so a scratch directory must be excluded from its own mirror.

---

## Prompt-writing notes

Claude Code has no access to `data/` — gitignored, absent from the sandbox. It
writes and unit-tests analysis code; Riley runs it and pastes output back.
**Ask for code plus the exact command, never for numbers.**

25 offline suites, all passing, named `test_*_offline.py`. Every new analysis
script ships one, and every fixture must have an answer known by construction.
The era fixture is the model: it pairs a floor change with a driver moving the
OTHER way, so pooling inverts the SIGN (truth +0.78, pooled -0.13, corrected
+0.78) — a method that merely blunted the bias would pass a tolerance test and
fail that one.

**Calibrate the instrument before believing the measurement.** The 2.7%
precision figure survived six weeks and a deck draft before anyone asked
whether the labeller could see a rip. One blind hour against an annotated
dataset moved it five-fold. Any project measuring a model against human labels
should budget that hour first.

**Ask for the diagnostic before the analysis.** The box-overlay investigation
went: overlay looks wrong -> audit coordinates -> audit still timing -> audit
class names. The third invalidated the premise of the first two.

**Race two hypotheses; never threshold one.** The first rotation run printed
"3 of 3 targets within 45 deg of the camera's axis" and concluded a lens
effect. Peak 185, camera 206, solar noon 180 — it was reporting the threshold.
Where two explanations predict nearby values, compare them against EACH OTHER
with a margin and let "cannot separate" be an allowed answer.

**Ask whether a variant could even move the number.** An era run printed a full
MOP-vs-buoy verdict under variant A that was a byte-identical copy of the
uncorrected one, because every matched hour was post-era.

**Assume the first fix misses.** Fixes applied to the wrong layer and re-done
after a real run disproved them: a class default changed in one module while
another kept its own; a guard on regression slopes when the overflow was in the
product; a sign asserted backwards in a fixture; a mirror built on a false
belief about glob and symlinks; a class bucketer matching one of two rip names;
a filename regex fed a stem when it required an extension. Ask for the fix AND
the command that would show it failed.
