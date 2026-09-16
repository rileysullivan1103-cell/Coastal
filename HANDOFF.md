# Handoff brief — for the Claude that plans this project

Regenerated on request ("summarize for claude"). It is written for a model that
writes prompts for Claude Code and cannot see this terminal, so it leads with
what is settled and what is ruled out. Re-asking a ruled-out question costs a
day: several of the entries below took multi-hour data pulls to answer.

Last regenerated: 2026-09-16.

> **Read this first.** Three results now dominate everything else.
>
> **The detector's confidence carries no information about rips.** 360 hand
> labels: high band 3/116, low band 3/116 — identical, Fisher p = 1.00.
> Population-weighted precision 2.7%. Every result in this project whose
> dependent variable is `score_max` is a result about a variable that does not
> discriminate rips.
>
> **Walton's "year effect" was a confidence threshold change on 2024-12-13**,
> 0.50 -> 0.70, +30 sd. Not weather, not a retrain. Walton's record must be cut
> into pre- and post-2024-12-13 eras, or read at one floor.
>
> **Sun geometry moves whether the detector fires at all.** Two bearing terms
> add dR2 +0.0325 to detection rate and +0.0381 to detections over a model that
> already holds wave height, period, tide, wind, temperature, rain and cloud.
> F = 148.6 and 179.3 on n = 8,262. This survives the class filter, survives
> re-censoring to one floor, and survives in each era separately. It used to be
> a `score_max` curiosity; it is now a statement about the detection rate.

---

## What the project is

WebCOOS publishes detector output for 8 beach cameras under a product named
"rip-detection-results". We join that hourly output to observed and modelled
conditions and ask what drives it. The recurring hazard is that a driver of the
DETECTOR (light, glare, contrast, image texture) is indistinguishable from a
driver of the RIP, because the label is model confidence and not a verified rip.

| | Walton Lighthouse, Santa Cruz CA | Hampton Inn Oceanfront S, Virginia Beach |
|---|---|---|
| analysable hours, class-filtered | 8,405 | 1,938 |
| ...split at 2024-12-13 | 1,400 pre / 7,005 post | post era only |
| span | 2024-05-31 to 2026-09-15 | 2026-04-25 to 2026-08-29 |
| frames, all classes | 35,394 | 2,352 |
| frames, rip class only | 28,721 (81%) | 2,352 (**100%**) |
| usable days for a time series | **643** | 81 |
| busiest 64px detection cell | 0.8% | 7.7% |
| nearshore waves | CDIP MOP SC130, 98% | none |
| buoy | 46236 — 22%, and **all of it post-split** (2025-08-31 to 2026-08-31) | none |
| shore normal | 206.0 deg, PUBLISHED by CDIP | 90.0 deg, **ASSUMED** |
| cloud cover | sidecar, 100% | sidecar, 100% |

**Three models publish into this one feed.** They are separable exactly by
`score_classes`, which never mixes a rip class with an object class:

| model | class it emits | cameras |
|---|---|---|
| `ripdetect_walton / yolov8x_1.1` | `rip_current` | Walton, Virginia Beach, both Panama City views |
| `rip_current_detector / 1` | `rip` | Corolla Hampton Inn, Corolla Sailfish, Carova |
| `yolo / v8n` | COCO objects | Walton and both Corolla cameras, 2024-05 to 2024-07 only |

| camera | frames | rip class | usable days |
|---|---|---|---|
| Walton | 35,394 | 28,721 (81%) | 643 |
| Virginia Beach | 2,352 | 2,352 (100%) | 81 |
| Panama City west | 980 | 980 (100%) | 28 |
| Panama City east | 709 | 709 (100%) | 27 |
| Corolla Hampton Inn | 10,766 | 1,064 (10%) | 34 |
| Corolla Sailfish | 10,239 | 472 (5%) | 14 |
| Carova | 34 | 34 (100%) | 1 |

---

## SETTLED — cite these, do not re-derive

**Detector confidence does not discriminate rips.** 360 Walton stills hand
labelled (319 judgeable, 41 unusable), drawn stratified by confidence band x
MOP wave tercile, every row `rip_current`:

| | rips found | rate | 95% CI |
|---|---|---|---|
| high band | 3 / 116 | 2.59% | [0.9%, 7.3%] |
| low band | 3 / 116 | 2.59% | [0.9%, 7.3%] |
| all fired | 6 / 232 | 2.59% | [1.2%, 5.5%] |
| never fired | 0 / 70 | 0.00% | [0.0%, 5.2%] |

High vs low: Fisher p = 1.00. Fired vs not fired: p = 0.34. Population-weighted
precision 2.7%; doubt-as-yes raises the per-cell figures to 7-11%. The false
omission rate is 0/28, 0/23, 0/19 across the three `none` cells.

**Walton's year effect is a confidence threshold change on 2024-12-13.**
`score_floor` (minimum nonzero score) 0.5075 -> 0.7033, **+30 sd raw, +35.7 sd
residualized on wave height and cloud**. `score_median` +0.098 (+6.4 sd) and
`score_p90` +0.048 (+2.4 sd) on the same date. The monthly histograms, filtered
to `rip_current`, show floor 0.500 every month through 2024-12 and 0.700 every
month from 2025-01. The model version string did NOT change
(`ripdetect_walton / yolov8x_1.1` throughout), so this is a config change, not
a retrain. It is a CENSORING change: everything scored 0.50-0.70 stopped being
published, which is why detection rate fell too.

**How much of Walton's pre era that censoring was.** Re-censoring the low-floor
era to 0.70 removes **69.1% of its detections**: 7,400 detected frames -> 2,283,
9,236 detections -> 3,273, 1,008 detection-hours -> 645. Anything computed
across 2024-12-13 without correction is comparing a record where two thirds of
one side does not exist on the other.

**Glare geometry drives the detection RATE, not just confidence.** `sun_in_view`
+ `sun_glare` over a base holding wave height and period, tide, onshore wind,
wind speed, temperature, precipitation, 48h rain, solar elevation and cloud:

| variant | detection_rate | detections | bbox_area_max |
|---|---|---|---|
| pooled (class-filtered, one span) | +0.0293, F=131.5 | +0.0348, F=159.2 | +0.0009, p=0.059 |
| **A, re-censored to one 0.70 floor** | **+0.0325, F=148.6** | **+0.0381, F=179.3** | +0.0019, p=0.0034 |
| B, pre era alone (n=1,400) | +0.0224, F=21.9 | +0.0559, F=58.7 | +0.0091, p=0.0059 |
| B, post era alone (n=6,862) | +0.0293, F=109.3 | +0.0299, F=114.1 | +0.0034, p=1.1e-4 |

Every p on the count targets is below 1e-9. The original number (`score_max`,
dR2 +0.0248, F(2,5824) = 78.43) is reproduced and enlarged — pooled +0.0424,
A +0.0295 — but `score_max` is the variable the labelling killed, so quote the
count rows. For scale against the ocean: `mop_wave_height` is the strongest
physical predictor in the same data at rho_hrmo +0.2205 (detections, variant A).
dR2 and rho-squared are not the same quantity; do not divide one by the other
in a deck.

**Cloud cover raises largest-box size at both sites, and it survives everything.**
VB rho +0.247 (+0.211 with hour and month removed) — VB is 100% `rip_current`
and single-era, so that half never needed correcting. Walton, rip frames only:

| variant | rho | rho_hrmo | p (hrmo) |
|---|---|---|---|
| pooled | +0.137 | +0.143 | 1.5e-27 |
| A, one floor | +0.162 | +0.159 | 7.6e-32 |
| B pre (n=1,008) | +0.041 | +0.076 | 0.016 |
| B post (n=4,708) | +0.163 | +0.167 | 6.4e-31 |

Originally +0.122 (+0.128) on the unfiltered pooled record. This is the one
cross-site replication in the project and it is now clean at both sites.

**The nearshore model beats the distant buoy — as a POST-ERA statement.** On the
1,884 hours where MOP and buoy 46236 both report, demeaned within them:
detection rate 3.15x (MOP rho +0.180 vs buoy +0.057), detections 2.93x (+0.199
vs +0.068), largest box 1.54x (+0.235 vs +0.153). Against the original 3.2x /
2.9x / 1.5x on the same n — the class filter barely moved it. But see RULED OUT:
the buoy has no pre-era data at all, so this is not and never was a cross-era
result. The ratio is a ratio of two modest correlations; quote the ratio, not
the strength. The fourth ratio (confidence, deck 3.6x) has a non-significant
buoy denominator (p=0.19) and is arithmetic, not a finding — stop quoting it.

**Reading Walton at one floor STRENGTHENS the ocean signal.** `mop_wave_height`
rho_hrmo, pooled -> variant A: detections +0.1826 -> **+0.2205**, detection rate
+0.1451 -> **+0.1816**. Both eras alone agree (B_pre +0.1667 / +0.1779, B_post
+0.2220 / +0.1760). The old floor's weak detections were diluting the physics,
which is evidence that the correction is correcting rather than merely differing.

**The three rip classes and three models above.** `rip_current` never co-occurs
with an object class, so `set(classes) <= {"rip_current", "rip"}` is an exact
filter. Frames with a blank class list are the observed zeros and must be KEPT.
At Walton the object stream is 6,673 of 35,394 frames — 18.9%.

**Panama City west view is half one fixed object.** 464 of 980 detections in a
single 64px cell at (960,192), centroids varying 10.2 x 5.5 px — 41% of the
18.5px spread an even scatter in that cell would show. Read every spread
against that baseline: Virginia Beach's busiest cell at 18.1 x 14.2 is NO
clustering, not a tight one.

**Walton is NOT dominated by a fixed feature.** Busiest cell 0.8% of 28,721
detections across 746 occupied cells. Hand labelling does show boxes on a rock
jetty and on moored boats, so that false-positive mode is real, but it is a
minor share.

**Only Walton can carry a time series.** 643 usable days against 81, 34, 28, 27,
14 and 1 elsewhere. The other six are rip records, not time series.

**Box coordinates are source pixels.** Walton stills are 2560x1920 and payload
coordinates reach 2490 x 1919.

**Virginia Beach's air-temperature result is not an artifact of light.** All
four targets read SUPPRESSED at 1.46x-1.73x: adding light terms makes the
temperature coefficient GROW. VB is 100% `rip_current`, so this is not
contaminated by object detections.

---

## RULED OUT — do not propose these again

- **The four rain correlations reversing between years.** THE REVERSAL WAS THE
  OBJECT DETECTIONS. On rip frames it is gone under every variant, including
  the one closest to the original run: all four predictor/target pairs are
  POSITIVE on both sides of the split (pre +0.128 to +0.172, post +0.050 to
  +0.076, every p below 1e-5). `pooled` differs from the original only by the
  class filter — the MOP/buoy n of 1,884 is identical across both runs, so the
  underlying hours did not change. Rain empties a beach, the `person` class
  falls with it, and that is what made the pre era read negative. What is left
  is a consistent modest POSITIVE rain term, which belongs with cloud and glare
  as image degradation, not ocean physics.
- **The MOP-vs-buoy ratio as a cross-era result.** Buoy 46236's record runs
  2025-08-31 to 2026-08-31 — entirely after the split. Zero matched hours fall
  in the low-floor era, so re-censoring cannot touch it and a pre-era split
  cannot compute it. Any future prompt asking to "check the buoy ratio per era"
  is asking for a number that does not exist.
- **VB temperature as a GLARE proxy.** Bearing terms dR2 <= 0.0007, every p
  between 0.36 and 0.97.
- **VB temperature as a DAYLIGHT proxy.** Solar elevation moves the coefficient
  the wrong way (larger, not smaller).
- **VB temperature as a HAZE proxy.** Cloud moves it 2-4%, sign inconsistent,
  and cloud's own correlation is POSITIVE — backwards for image degradation.
- **VB temperature as a people-on-the-beach effect.** VB is 100%
  `rip_current`, 2,352 of 2,352. There are no person detections there.
- **A letterbox / model-space transform for the box coordinates.** Coordinates
  reach 97-100% of the still's dimensions. Six transforms were built; none is
  needed.
- **A fixed scene feature explaining Walton's driver results.** Busiest cell
  0.8%.
- **Walton's year effect as weather.** It is the 2024-12-13 threshold change,
  and it survives residualizing at +35.7 sd.
- **"Three cameras have never detected a rip."** Corolla Hampton Inn, Corolla
  Sailfish and Carova emit class `rip` from a different model. Carova's record
  is 100% rip.
- **The 2026-08-01..08 Walton / Panama City west coincidence as a platform
  deployment.** Two cameras, one metric, no version change at either, and
  Walton's date is its only non-transient one. Corolla's floor sat near 0.70
  from February 2024 while Walton was still at 0.50 — confirmed again by the
  per-era floor sweep, which reads Corolla HI at 0.7149 and Sailfish at 0.7120
  in the pre era. The two model families are configured independently. File as
  unexplained.

Solar position needs no data source; it is computed from camera lat/lon and
timestamp in solar.py. Cloud cover is on disk at both sites.

---

## OPEN — ranked

0. **Is 2.7% the detector's failure, or the medium's?** The single most
   important open question, because it decides whether this project reports a
   broken detector or an untestable one. Only 6 rips were seen in 302 judgeable
   frames — a 2.0% base rate overall. Two readings fit equally well: the
   detector is near-useless, or rips are rarely identifiable in a SINGLE STILL
   by ONE observer (practitioners use video or time-averaged imagery). Nothing
   in the labelling separates them, and more of the same labelling cannot: at a
   2.6% rate, detecting even a doubling between bands needs ~868 labels per
   band at 80% power.
   **The test that decides it is already in the repo.** `load_ripaid.py` reads
   RipAID (Zenodo 15082427), 2,815 frames of human-drawn rip annotations with
   948 real negatives. Label ~60 RipAID frames blind through the existing page
   and compare to the annotators. Agreement means the 2.0% base rate is real;
   disagreement means the labels measure visibility in a still, and the
   detector has not been fairly tested. An hour's work.

1. **Is the glare result the detector or the ocean?** Now the second-biggest
   question, because it is the largest surviving effect in the project and it
   is exactly the confound this project was set up to worry about. Sun geometry
   could be degrading the image, or afternoon sea breeze could genuinely change
   the surf, or offshore-facing light could make real rips visible. Nothing run
   so far separates them. Two candidate tests, neither started: (a) the RipAID
   frames carry human annotations and timestamps, so the same bearing terms can
   be run against HUMAN labels — if they vanish there, it is the detector;
   (b) split the Walton effect by whether the sun is in front of or behind the
   camera, since a physical sea-breeze story does not care and a lens-flare
   story does.

2. **What are the false positives made of?** The notes column on the 360 labels
   is the only record. Jetty, boat wake and whitewater-on-rocks recurred. If
   they concentrate, the finding is actionable (mask those regions) rather than
   a score.

3. **What IS Virginia Beach's temperature tracking?** Four explanations dead;
   the -0.36 is still there and still suppressed by light. Proposed next test:
   a sea-breeze / storminess index. Not started. NEW: Walton may echo it.
   In the pre era `temperature_2m` is the TOP predictor of both count targets,
   and it flips sign under controls — detection rate -0.0595 raw to **+0.1867**
   with hour and month removed, detections -0.0530 to **+0.1895**. That is
   suppression, the same shape as VB. On `bbox_area_max` it is negative instead
   (A -0.1227, B_post -0.1323). n = 1,400, one era, so this is a lead and not a
   finding — but a sea-breeze index would now be testable at two sites.

4. **VB's shore normal is a guess (90.0 deg).** Every onshore wind and wave
   component at VB is computed from it. check_camera_geometry.py exists.

5. **Deck slide 6** still carries the unmatched-hours ratios and two verdict
   flags the matched-hours run changed. Now also pending the class filter, the
   era correction, the precision result and the glare upgrade.

---

## CONSTRAINTS the planner must respect

**Filter by `score_classes` before computing anything from a rip record.** Two
rip class names (`rip_current`, `rip`) and one object stream share the column.
Blank class lists are the observed zeros and must be KEPT. At Walton the filter
moves 18.9% of frames and it has already overturned one published finding.

**Cut Walton at 2024-12-13, or read it at one 0.70 floor.**
`analyze_walton_eras.py` does both and prints them side by side:

    python analyze_walton_eras.py --keep-workspace data/era_workspace
    python analyze_walton_eras.py --check-recensoring data/era_workspace/A/data/rip_detection
    python analyze_walton_eras.py --check-recensoring data/rip_detection

The second command must report the pre era APPLIED and the third NOT APPLIED.
If they agree, the correction silently did nothing.

**Variant A has a residual bias and it must be stated wherever A is quoted.**
The frame table carries one score per FRAME, not per detection, so re-censoring
can only demote whole frames: a pre-era frame holding a 0.9 and a 0.55 keeps
both. 506 Walton frames prove it (`score_mean < 0.70 <= score_max`), and that
is a LOWER bound, because a strong detection can pull the mean back over the
floor and hide one. A is therefore slightly LESS censored than the post era.

**Buoy 46236 can only speak to the post era.** 2025-08-31 to 2026-08-31. Its
"22% of hours" is one contiguous year at the end of the record, not a scatter.

**Open-Meteo bills by variable-hours, not by calls.** Six ERA5 variables over
Walton's 20,832-hour window to obtain one costs 6x the quota and returned HTTP
429. pull_cloud_cover.py fetches one variable into a sidecar; that is the
pattern. One site at a time.

**Never re-pull observations on the default one-year window.** It narrows a
site's conditions file below its rip record and silently shrinks every n
downstream. Always pass explicit --start/--end at least as wide as the rip
record.

**Concurrency.** pull_rip_detection.py hits only app.webcoos.org and is safe
alongside anything. pull_site_observations.py and pull_cloud_cover.py share the
Open-Meteo hourly quota — run them one at a time. The wq/ pipeline writes only
data/wq/** plus wq_manifest.json and contends with nothing; analyze_walton_eras
makes no Open-Meteo calls at all and runs safely beside it.

**Read rho_hrmo, except for solar elevation.** Hour-of-day IS solar elevation,
so demeaning by hour-by-month removes it by construction. For the light columns
read rho and rho_mo; section (b) of analyze_glare.py is where light competes
fairly.

**bbox_area_max is in pixels.** Comparable within a camera, meaningless pooled
across cameras.

**A still must be matched to its detection's timestamp, not its hour.** The
labelling sample was built twice on the wrong images; the first draw averaged
606s from the detection it carried boxes for. The rebuilt sample runs at a
median 18s, worst 39s, for every box-carrying row.

**Read a detection cluster's spread against `cell / sqrt(12)`.** At 64px that
is 18.5. A spread near it is NO clustering. Read alone, every camera looks like
it has a fixed object.

**`rip_<slug>_hourly.csv` and `rip_<slug>_index.csv` are not cameras.** Any
glob over data/rip_detection/ must exclude them.

**`analyze_drivers.py` is shared infrastructure and its defaults are frozen.**
New questions get new scripts that import it. Where a variant needs a different
`rip_<slug>_hourly.csv`, the pattern is a mirror of data/ — real directories,
symlinked files — with `ad.DATA_DIR` repointed. Note that glob's `**` DOES
follow a symlinked directory, so any scratch directory must be excluded from
its own mirror or the variant tables collide under one filename.

---

## Prompt-writing notes

Claude Code has no access to data/ — it is gitignored and absent from the
sandbox. It can write and test analysis code but cannot run it on real data;
Riley runs it locally and pastes output back. Prompts should ask for code plus
the exact command, not for numbers.

Offline test suites are the contract: 23 of them, all passing, named
test_*_offline.py. Any new analysis script should arrive with one, and any
fixture must be built so the expected answer is known by construction rather
than copied from a previous run of the same code. The era fixture is the model
to copy: it pairs a floor change with a driver moving the OTHER way, so pooling
inverts the SIGN (truth +0.78, pooled -0.13, corrected +0.78) rather than
merely weakening it — a method that only blunted the bias would still pass a
tolerance test and fail this one.

**Ask for the diagnostic before the analysis.** The box-overlay investigation
went: overlay looks wrong -> audit the coordinates -> audit the still timing ->
audit the class names. Each was one command, and the third invalidated the
premise of the first two. Going straight to "fix the box transform" would have
produced a confident wrong answer; six transforms were built and none was
needed.

**Assume the first fix misses.** Several fixes have been applied to the wrong
layer and re-done after a real run disproved them: a class default changed in
one module while another kept its own; a guard put on regression slopes when
the overflow was in the product; a sign asserted backwards in a test fixture;
a mirror built on a false belief about glob and symlinks. Prompts should ask
for the fix AND the command that would show it failed.

**Ask whether a variant could even move the number.** The first era run printed
a full MOP-vs-buoy verdict under variant A that was a byte-identical copy of
the uncorrected one, because every matched hour was post-era and A had nothing
to re-censor there. It read as independent corroboration. Any side-by-side
comparison should be asked to report how many rows the treatment actually
touched, not just the result.
