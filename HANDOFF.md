# Handoff brief — for the Claude that plans this project

Regenerated on request ("summarize for claude"). It is written for a model that
writes prompts for Claude Code and cannot see this terminal, so it leads with
what is settled and what is ruled out. Re-asking a ruled-out question costs a
day: several of the entries below took multi-hour data pulls to answer.

Last regenerated: 2026-09-16.

> **Read this first.** Two results now dominate everything else.
>
> **The detector's confidence carries no information about rips.** 360 hand
> labels: high band 3/116, low band 3/116 — identical, Fisher p = 1.00.
> Population-weighted precision 2.7%. Every result in this project whose
> dependent variable is `score_max` is a result about a variable that does not
> discriminate rips.
>
> **Walton's "year effect" was a confidence threshold change on 2024-12-13**,
> 0.50 -> 0.70, +30 sd. Not weather, not a retrain. Walton's record must be cut
> into pre- and post-2024-12-13 eras.

---

## What the project is

WebCOOS publishes detector output for 8 beach cameras under a product named
"rip-detection-results". We join that hourly output to observed and modelled
conditions and ask what drives it. The recurring hazard is that a driver of the
DETECTOR (light, glare, contrast, image texture) is indistinguishable from a
driver of the RIP, because the label is model confidence and not a verified rip.

| | Walton Lighthouse, Santa Cruz CA | Hampton Inn Oceanfront S, Virginia Beach |
|---|---|---|
| analysable hours (pre-filter) | 8,712 | 1,938 |
| span | 2024-05-31 to 2026-09-15 | 2026-04-25 to 2026-08-29 |
| frames, all classes | 35,394 | 2,352 |
| frames, rip class only | 28,721 (81%) | 2,352 (**100%**) |
| usable days for a time series | **643** | 81 |
| busiest 64px detection cell | 0.8% | 7.7% |
| nearshore waves | CDIP MOP SC130, 98% | none |
| buoy | 46236, 22% of hours only | none |
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

**The three rip classes and three models above.** `rip_current` never co-occurs
with an object class, so `set(classes) <= {"rip_current", "rip"}` is an exact
filter. Frames with a blank class list are the observed zeros and must be KEPT.

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

**Cloud cover raises largest-box size at both sites.** VB rho +0.247 (+0.211
with hour and month removed), Walton +0.122 (+0.128). The VB half is clean; the
Walton half is provisional.

### Provisional — computed over Walton's unfiltered, un-split record

Each pooled ~19% object detections into the dependent variable, AND straddles
the 2024-12-13 threshold change. None is disproven; all need re-running on rip
frames, cut by era.

- **The nearshore model beats the distant buoy.** On 1,884 matched hours,
  demeaned within them: detection rate 3.2x, detections/hour 2.9x, largest box
  1.5x. The fourth ratio (confidence, deck 3.6x) has a non-significant buoy
  denominator (p=0.19) and is arithmetic, not a finding — stop quoting it.
- **Walton has a year effect larger than its weather.** R2 0.020->0.074
  (detection rate), 0.013->0.079, 0.040->0.095, 0.053->0.104. Now EXPLAINED by
  the threshold change; re-run with an era split rather than year dummies.
- **Four rain correlations reverse between years.** rain_24h_mm and rain_48h_mm
  on both count targets: significantly negative in 2024, positive in 2026. Both
  sides of 2024-12-13. Never quote the pooled rain number.
- **Glare geometry earns its place in exactly one place.** Walton score_max,
  dR2 = +0.0248, F(2,5824) = 78.43, p = 2.4e-34. A real, well-tested result
  about a variable the labelling has now shown does not discriminate rips.

---

## RULED OUT — do not propose these again

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
- **"Three cameras have never detected a rip."** THIS WAS WRONG in the previous
  brief. Corolla Hampton Inn, Corolla Sailfish and Carova emit class `rip` from
  a different model. Carova's record is 100% rip.
- **The 2026-08-01..08 Walton / Panama City west coincidence as a platform
  deployment.** Two cameras, one metric, no version change at either, and
  Walton's date is its only non-transient one. Corolla's floor sat near 0.70
  from February 2024 while Walton was still at 0.50, so the two model families
  are configured independently. File as unexplained.

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

1. **Re-run every Walton driver result on rip frames, split at 2024-12-13.**
   Four SETTLED-adjacent findings depend on it. `build_label_sample.keep_class()`
   is the filter; it needs applying upstream in the frame table, and the hourly
   aggregates (`rip_<slug>_hourly.csv`) predate class filtering and must be
   regenerated. Deliverable: which findings survive, before/after side by side,
   per era.

2. **What are the false positives made of?** The notes column on the 360 labels
   is the only record. Jetty, boat wake and whitewater-on-rocks recurred. If
   they concentrate, the finding is actionable (mask those regions) rather than
   a score.

3. **What IS Virginia Beach's temperature tracking?** Four explanations dead;
   the -0.36 is still there and still suppressed by light. Proposed next test:
   a sea-breeze / storminess index. Not started.

4. **Cloud -> largest box, as its own question.** The one cross-site
   replication. VB half clean, Walton half needs the filter and the era split.

5. **VB's shore normal is a guess (90.0 deg).** Every onshore wind and wave
   component at VB is computed from it. check_camera_geometry.py exists.

6. **Deck slide 6** still carries the unmatched-hours ratios and two verdict
   flags the matched-hours run changed. Now also pending the class filter, the
   era split, and the precision result.

---

## CONSTRAINTS the planner must respect

**Filter by `score_classes` before computing anything from a rip record.** Two
rip class names (`rip_current`, `rip`) and one object stream share the column.
Blank class lists are the observed zeros and must be KEPT.

**Cut Walton at 2024-12-13.** Anything pooled across it mixes two censoring
regimes.

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
data/wq/** and contends with nothing.

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

---

## Prompt-writing notes

Claude Code has no access to data/ — it is gitignored and absent from the
sandbox. It can write and test analysis code but cannot run it on real data;
Riley runs it locally and pastes output back. Prompts should ask for code plus
the exact command, not for numbers.

Offline test suites are the contract: 22 of them, all passing, named
test_*_offline.py. Any new analysis script should arrive with one, and any
fixture must be built so the expected answer is known by construction rather
than copied from a previous run of the same code.

**Ask for the diagnostic before the analysis.** The box-overlay investigation
went: overlay looks wrong -> audit the coordinates -> audit the still timing ->
audit the class names. Each was one command, and the third invalidated the
premise of the first two. Going straight to "fix the box transform" would have
produced a confident wrong answer; six transforms were built and none was
needed.

**Assume the first fix misses.** Several fixes this week were applied to the
wrong layer and had to be re-done after a real run disproved them: a class
default changed in one module while another kept its own; a guard put on
regression slopes when the overflow was in the product; a sign asserted
backwards in a test fixture. Prompts should ask for the fix AND the command
that would show it failed.
