# Handoff brief — for the Claude that plans this project

Regenerated on request ("summarize for claude"). It is written for a model that
writes prompts for Claude Code and cannot see this terminal, so it leads with
what is settled and what is ruled out. Re-asking a ruled-out question costs a
day: several of the entries below took multi-hour data pulls to answer.

Last regenerated: 2026-09-16.

> **Read this first.** On 2026-09-16 the rip feed was found to carry TWO models'
> output, and three of the cameras have never produced a rip detection at all.
> Every Walton number in SETTLED was computed over a record that is 19% object
> detections and is therefore PROVISIONAL until re-run. Virginia Beach is
> unaffected. Do not commission work that builds on a provisional number without
> first commissioning the re-run.

---

## What the project is

WebCOOS publishes an object-detection API's output for 8 beach cameras under a
product named "rip-detection-results". We join that hourly output to observed
and modelled conditions and ask what drives it. The recurring hazard is that a
driver of the DETECTOR (light, glare, contrast, image texture) is
indistinguishable from a driver of the RIP, because the label is model
confidence and not a verified rip.

| | Walton Lighthouse, Santa Cruz CA | Hampton Inn Oceanfront S, Virginia Beach |
|---|---|---|
| analysable hours (pre-filter) | 8,712 | 1,938 |
| span | 2024-05-31 to 2026-09-15 | 2026-04-25 to 2026-08-29 |
| frames, all classes | 35,394 | 2,352 |
| frames, rip_current only | 28,721 (81%) | 2,352 (**100%**) |
| busiest 64px cell | 0.8% of detections | 7.7% |
| nearshore waves | CDIP MOP SC130, 98% | none |
| buoy | 46236, 22% of hours only | none |
| shore normal | 206.0 deg, PUBLISHED by CDIP | 90.0 deg, **ASSUMED** |
| cloud cover | sidecar, 100% | sidecar, 100% |

Class mix across every record on disk, `score_classes` in the frame table:

| camera | frames | rip_current |
|---|---|---|
| Virginia Beach | 2,352 | 100% |
| Panama City east | 709 | 100% |
| Panama City west | 980 | 100% |
| Walton | 35,394 | 81% |
| Corolla Hampton Inn | 10,766 | **0%** |
| Corolla Sailfish St | 10,239 | **0%** |
| Carova | 34 | **0%** |

---

## SETTLED — cite these, do not re-derive

**The "rip" feed carries two models.** At Walton, 28,721 frames are
`rip_current` and 6,673 are COCO object classes — person 3,395, boat 894,
boat+person 798, then car, bird, truck, kite, bicycle, dog, motorcycle.
`rip_current` NEVER co-occurs with any object class, so the two separate
exactly rather than by threshold. The annotated frames are served from
`stage-webcoos-object-detector-api.../outputs/yolo/v8n/`, which is YOLOv8-nano;
those URLs now 404, so WebCOOS's own renderings are unavailable as ground truth.

**Three cameras have never detected a rip.** Corolla Hampton Inn (10,766
frames), Corolla Sailfish (10,239) and Carova (34) are 0% `rip_current`. Their
entire 21,039-frame "rip record" is people, boats and cars. These are not thin
rip records; they are object-detection records. Any cross-site statement that
included them compared a rip record against a people record.

**Panama City west view is half one fixed object.** 464 of 980 detections sit in
a single 64px cell at (960,192), with centroids varying 10.2 x 5.5 px — 41% of
the 18.5px spread an even scatter in that cell would show. Its `detection_rate`
is largely "was that feature visible". Panama City east: busiest cell 11.6%.

**Walton is NOT dominated by a fixed feature.** Busiest cell 0.8% of 28,721
detections, spread over 746 occupied cells. Hand labelling does show boxes on a
rock jetty and on moored boats, so that false-positive mode is real, but it is a
minor share and does not explain any Walton driver result.

**Box coordinates are source pixels.** Walton stills are 2560x1920 and payload
coordinates reach 2490 x 1919 — 97% and 100% of the frame. No letterbox or
model-space transform applies.

**Virginia Beach's air-temperature result is not an artifact of light.** All four
targets read SUPPRESSED at 1.46x-1.73x: adding light terms makes the temperature
coefficient GROW, which is evidence against the proxy reading, not a weak
version of it. VB is 100% `rip_current`, so this is NOT contaminated by object
detections and stands as written.

**Cloud cover raises largest-box size at both sites.** VB rho +0.247 (+0.211
with hour and month removed), Walton +0.122 (+0.128). The only cross-site
replication the project has. The VB half is clean; the Walton half is
provisional (see the banner).

### Provisional — computed over Walton's unfiltered record

Each of these pooled ~19% object detections into the dependent variable.
`score_max` for an object frame is a person's or a boat's confidence;
`bbox_area_max` is a person's bounding box, a very different size distribution
from a rip's. None is disproven. All need re-running on `rip_current` frames
before being quoted.

- **The nearshore model beats the distant buoy.** On the 1,884 hours where both
  CDIP MOP and buoy 46236 exist, demeaned within that shared set: detection rate
  3.2x, detections/hour 2.9x, largest box 1.5x. The fourth ratio (confidence,
  deck 3.6x) has a non-significant buoy denominator (p=0.19) and is arithmetic,
  not a finding — stop quoting it as a ratio regardless of the re-run.
- **Walton has a year effect larger than its weather.** Two year dummies raise
  R2 from 0.020 to 0.074 (detection rate), 0.013 to 0.079 (detections), 0.040 to
  0.095 (score_max), 0.053 to 0.104 (bbox area). Year betas -0.31 to +0.31.
  score_max moves OPPOSITE the other three.
- **Four rain correlations reverse between years.** rain_24h_mm and rain_48h_mm
  on both count targets: significantly negative in 2024, significantly positive
  in 2026. Never quote the pooled rain number for Walton.
- **Glare geometry earns its place in exactly one place.** Walton score_max,
  dR2 = +0.0248, F(2,5824) = 78.43, p = 2.4e-34. Every other site/target pair:
  dR2 <= 0.0035.

---

## RULED OUT — do not propose these again

- **VB temperature as a GLARE proxy.** Bearing terms contribute dR2 <= 0.0007 on
  all four targets, every p between 0.36 and 0.97.
- **VB temperature as a DAYLIGHT proxy.** Solar elevation moves the coefficient
  the wrong way (larger, not smaller).
- **VB temperature as a HAZE proxy.** Cloud moves it 2-4%, sign inconsistent
  across targets, and cloud's own correlation is POSITIVE — backwards for an
  image-degradation story.
- **VB temperature as a people-on-the-beach effect.** Proposed and killed the
  same day: VB is 100% `rip_current`, 2,352 of 2,352 frames. There are no
  person detections in that record to drive it.
- **A letterbox / model-space transform for the box coordinates.** Coordinates
  reach 97-100% of the still's dimensions; model-space coordinates cannot exceed
  the canvas. Six candidate transforms were built and tested and none is needed.
- **A fixed scene feature explaining Walton's driver results.** Busiest cell
  0.8%. This was a live hypothesis for the glare -> score_max result and it is
  now dead.

Solar position needs no data source; it is computed from camera lat/lon and
timestamp in solar.py. Cloud cover is on disk at both sites. Neither is a reason
to decline a follow-up.

---

## OPEN — ranked

0. **Re-run every Walton driver result on `rip_current` frames only.** This is
   now the highest-value prompt in the project, because four SETTLED entries
   depend on it. `build_label_sample.keep_class()` is the filter; it needs
   applying upstream in the frame table, and the hourly aggregates
   (`rip_<slug>_hourly.csv`) were built before class filtering existed and must
   be regenerated. Deliverable: which of the four provisional findings survive,
   with the before/after numbers side by side.

1. **Hand-labelled ground truth — LABELLING COMPLETE, analysis not yet run.**
   All 360 Walton frames labelled on 2026-09-16 (yes / no / doubt / unusable,
   plus per-box marks and free-text notes). `analyze_precision.py` reports
   precision by stratum with Wilson intervals, false-omission rates for the
   three `none` cells, a doubt-as-no / doubt-as-yes bracket, and a
   population-weighted headline from `strata.csv`. The number that matters most:
   whether the `high` confidence band's precision is meaningfully above `low`.
   If it is not, `score_max` is not a usable ranking signal and the glare result
   is a finding about a meaningless variable. NO RESULTS YET — do not write
   prompts that assume a value.

2. **What are the false positives made of?** The notes column is the only record
   of this. Jetty, boat wake and whitewater-on-rocks were seen repeatedly during
   labelling. If they concentrate, the finding is actionable (mask those
   regions) rather than just a score.

3. **Drop or re-describe Corolla, Sailfish and Carova.** They are currently
   listed as thin rip cameras. They are not rip cameras.

4. **Walton's year effect — what changed and when.**
   `analyze_detector_changepoints.py` is built and tested but NOT YET RUN. It
   finds abrupt shifts in daily detection rate, median and p90 score_max, score
   floor (minimum nonzero score, which moves when a confidence threshold
   changes) and median bbox area, with binary segmentation and BIC, optionally
   on weather-residualized series, and flags dates where two or more cameras
   shift within 7 days. It also reports each camera's recorded
   model_name/model_version changes, which may answer the question outright.
   Run `--spans` first for the per-camera windows.

5. **What IS Virginia Beach's temperature tracking?** Four explanations are now
   dead; the -0.36 is still there and still suppressed by light. Proposed next
   test: a sea-breeze / storminess index. Not started.

6. **Cloud -> largest box, as its own question.** The one cross-site
   replication. The VB half is clean; the Walton half needs the class filter.

7. **VB's shore normal is a guess (90.0 deg).** Every onshore wind and wave
   component at VB is computed from it. check_camera_geometry.py exists.

8. **Deck slide 6** still carries the unmatched-hours ratios and two verdict
   flags that the matched-hours run changed. Left alone deliberately, and now
   also pending the class re-run.

---

## CONSTRAINTS the planner must respect

**Filter by `score_classes` before computing anything from a rip record.** The
column is in the frame table. `rip_current` never co-occurs with an object
class, so `set(classes) == {"rip_current"}` is exact. Frames with a blank class
list are the observed zeros and must be KEPT.

**Open-Meteo bills by variable-hours, not by calls.** Asking for 6 ERA5
variables over Walton's 20,832-hour window to obtain 1 costs 6x the quota and
kept returning HTTP 429. pull_cloud_cover.py fetches one variable into a
sidecar; that is the pattern for any new ERA5 column. Budget one site at a time.

**Never re-pull observations on the default one-year window.** The default
narrows a site's conditions file to less than its rip record, which silently
shrinks every n in every table downstream. That has already cost this project
two analyses. Always pass explicit --start/--end at least as wide as the rip
record.

**Concurrency.** pull_rip_detection.py hits only app.webcoos.org and is safe
alongside anything. pull_site_observations.py and pull_cloud_cover.py share the
Open-Meteo hourly quota with each other — run them one at a time. The wq/
pipeline writes only data/wq/** and contends with nothing.

**Read rho_hrmo, except for solar elevation.** Hour-of-day IS solar elevation,
so demeaning by hour-by-month removes it by construction. For the light columns
read rho and rho_mo; section (b) of analyze_glare.py is where light competes
fairly, because that regression applies no hour control.

**bbox_area_max is in pixels.** Comparable within a camera, meaningless pooled
across cameras.

**A still must be matched to its detection's timestamp, not its hour.** The
labelling sample was built twice on the wrong images: the first draw took the
still nearest the middle of each hour, a median 606s from the detection it
carried boxes for. A one-minute stills feed plus a moving sea means the water
under the box is not the water the detector saw. The rebuilt sample runs at a
median 18s, worst 39s, for every row carrying boxes.

**`rip_<slug>_hourly.csv` and `rip_<slug>_index.csv` are not cameras.** Both sit
beside the frame table and match `rip_*.csv`. Any glob over that directory must
exclude them.

---

## Prompt-writing notes

Claude Code has no access to data/ — it is gitignored and absent from the
sandbox. It can write and test analysis code but cannot run it on real data;
Riley runs it locally and pastes output back. Prompts should therefore ask for
code plus the exact command to run, not for numbers.

Offline test suites are the contract: 22 of them, all passing, named
test_*_offline.py. Any new analysis script should arrive with one, and any
fixture must be built so the expected answer is known by construction rather
than copied from a previous run of the same code. Two bugs this week were found
only because a fixture disagreed with the code: a sign error that had the
diagnosis chasing the wrong failure mode, and a shift-size measurement that
overstated a residual by sevenfold.

**Ask for the diagnostic before the analysis.** Today's sequence was: overlay
looks wrong -> audit the coordinates -> audit the still timing -> audit the
class names. Each step was one command, and the third one invalidated the
premise of the first two. A prompt that had gone straight to "fix the box
transform" would have produced a confident wrong answer; six transforms were
built and none was needed.
