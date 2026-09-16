# Handoff brief — for the Claude that plans this project

Self-contained. Written for a model that writes prompts for Claude Code and
cannot see the terminal. Leads with what is settled and what is dead, because
re-proposing a ruled-out test costs a day.

Last regenerated: 2026-09-16.

> **Three results dominate.**
>
> **1. Detector confidence carries no information about rips.** 360 hand
> labels: high-confidence band 3/116, low band 3/116 — identical, Fisher
> p = 1.00. Population-weighted precision **2.7%**. Any result whose dependent
> variable is `score_max` is about a variable that does not discriminate rips.
>
> **2. Walton's "year effect" was a score-threshold change on 2024-12-13**,
> floor 0.5075 → 0.7033, +30 sd. Not weather, not a retrain. Cut Walton at that
> date or read it at one floor.
>
> **3. The detector fires more near solar noon, and it is NOT glare.** Two
> sun-bearing terms add dR2 +0.0325 (detection rate) and +0.0381 (detections)
> over a model already holding waves, tide, wind, temperature, rain, solar
> elevation and cloud. But rotating the assumed bearing through 360 deg puts
> the peak at 185 deg — 5 deg from solar noon, 21 from the camera's 206. Do not
> call it glare.

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
| buoy | 46236 — 22%, all of it post-split (2025-08-31 to 2026-08-31) | none |
| shore normal | 206.0 deg, PUBLISHED by CDIP | 90.0 deg, **ASSUMED** |
| cloud cover | sidecar, 100% | sidecar, 100% |

Three models publish into one feed, separable exactly by `score_classes`,
which never mixes a rip class with an object class:

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

---

## SETTLED — cite these, do not re-derive

**Precision is 2.7%, and confidence does not rank rips.** 360 Walton stills
hand-labelled (319 judgeable, 41 unusable), stratified by confidence band x MOP
wave tercile, every row `rip_current`: high 3/116, low 3/116, all-fired 6/232,
never-fired 0/70. Fisher p = 1.00 high vs low, 0.34 fired vs not. Doubt-as-yes
raises per-cell figures to 7-11%.

**The 2024-12-13 threshold change.** `score_floor` 0.5075 → 0.7033, +30 sd raw,
**+35.7 sd residualized** on wave height and cloud. `score_median` +0.098
(+6.4 sd), `score_p90` +0.048 (+2.4 sd), same date. Class-filtered monthly
histograms read 0.500 through 2024-12 and 0.700 from 2025-01. Model version did
NOT change. It is a CENSORING change — everything scoring 0.50-0.70 stopped
being published — which is why the detection rate fell too.

**Re-censoring the pre era to 0.70 removes 69.1% of its detections:** 7,400
detected frames → 2,283, 9,236 detections → 3,273, 1,008 detection-hours → 645.
Anything pooled across 2024-12-13 uncorrected compares a record where two
thirds of one side does not exist on the other.

**The sun terms drive the detection RATE.** `sun_in_view` + `sun_glare` over a
base of wave height/period, tide, onshore wind, wind speed, temperature,
precipitation, 48h rain, solar elevation, cloud:

| variant | detection_rate | detections |
|---|---|---|
| pooled (class-filtered) | +0.0293, F=131.5 | +0.0348, F=159.2 |
| **A, one 0.70 floor** | **+0.0325, F=148.6** | **+0.0381, F=179.3** |
| B, pre era (n=1,400) | +0.0224, F=21.9 | +0.0559, F=58.7 |
| B, post era (n=6,862) | +0.0293, F=109.3 | +0.0299, F=114.1 |

All p below 1e-9. Scale against the ocean: `mop_wave_height` rho_hrmo +0.2205
(detections, A). dR2 and rho-squared are different quantities — do not divide
one by the other in a deck.

**But the rotation points at solar noon, not the camera.** dR2 recomputed
against every bearing, 5 deg grid, variant A: detection_rate peaks at 185
(+0.0486), detections at 185 (+0.0573), bbox_area_max at 175. The camera's own
206 collects 67% of the maximum. Solar noon is 180. Trough +0.0003 at 120, so
the sweep has real contrast (160-fold), not a flat curve over-read. Under
`pooled` both count targets peak at exactly 180.

**The effect is entirely on the sunlit side, which proves less than it looks.**
sun-in-front (n=6,121) detection_rate +0.0249 / detections +0.0372; sun-behind
(n=2,141) +0.0002 / +0.0005. The behind null is REAL — an effect the front
group's size would land at F=37.9, p=7e-17 there. It survives the signed
parameterisation and dropping night hours. But sun-behind is **06:00-09:00
local and almost nothing else** (1,426 of 1,974 lit hours), so the split is
morning-versus-midday and every candidate story predicts it.

**Cloud cover raises largest-box size at both sites.** VB rho +0.247 (+0.211
hour/month removed) — VB is single-era and 100% rip class, so that half never
needed correcting. Walton, rip frames: pooled +0.137 (+0.143), A +0.162
(+0.159), B post +0.163 (+0.167), B pre +0.041 (+0.076, p=0.016). The one
cross-site replication in the project, clean at both sites.

**The nearshore model beats the distant buoy — POST-ERA only.** On the 1,884
hours where MOP and buoy 46236 both report, demeaned within them: detection
rate 3.15x (+0.180 vs +0.057), detections 2.93x (+0.199 vs +0.068), largest box
1.54x (+0.235 vs +0.153). A ratio of two modest correlations — quote the ratio,
not the strength.

**Reading Walton at one floor STRENGTHENS the ocean signal.** `mop_wave_height`
rho_hrmo, pooled → A: detections +0.1826 → **+0.2205**, rate +0.1451 →
**+0.1816**. Both eras alone agree. The old floor's weak detections were
diluting the physics, which is evidence the correction corrects rather than
merely differs.

**The object stream was a 20-day fleet-wide event, not contamination
throughout.** `yolo/v8n` ran **2024-05-31 to 2024-06-19** and stopped dead — at
Walton AND both Corolla cameras, on the same two dates, across two different
rip-model families and opposite coasts. That is a WebCOOS platform action, now
dated. Consequences:
- At Walton the class filter is **exactly a truncation of the first 24 days**.
  `rip_current` starts 2024-06-24, five days after the object model stopped;
  the two never overlap in time; every month from 2024-07 on is 0% object.
  28,721 + 6,673 = 35,394 exactly. Measured two ways: the unfiltered hourly
  table runs 5,918 hours from 2024-05-31 21:00 against the filtered 5,719, and
  pre-era detection-hours are 1,207 unfiltered against 1,008 filtered — both
  give 199 hours, the object window at ~10 daylight hours a day.
- The huge June frame counts (Corolla Sailfish: ~50/month typical, 9,653 in
  June 2024, of which **one** is a rip) are the object model's FIRING RATE, not
  sampling. The feed publishes on a detection, and `yolo/v8n` finds a person or
  bird on nearly every daytime beach frame.
- **Virginia Beach, both Panama City views and Carova have zero object frames
  ever** — all deployed after the object era. Nothing to filter at those four.
- **Walton has no blank-class frames at all**, so its observed zeros come
  entirely from the coverage file.
- **October 2024 is missing entirely** from Walton's frame table (2024-09: 339
  frames, 2024-11: 2,647, nothing between). That gap sits in the pre-era half
  of the era split, already the thin side at 1,400 hours.

**Panama City west is half one fixed object.** 464 of 980 detections in one
64px cell at (960,192), centroids varying 10.2 x 5.5 px — 41% of the 18.5px
spread an even scatter would show.

**Walton is NOT dominated by a fixed feature.** Busiest cell 0.8% of 28,721
detections across 746 occupied cells. Hand labelling does show boxes on a rock
jetty and moored boats, so that false-positive mode is real but minor.

**Only Walton can carry a time series.** 643 usable days against 81, 34, 28,
27, 14 and 1. The other six are rip records, not time series.

**Box coordinates are source pixels.** Walton stills are 2560x1920; payload
coordinates reach 2490 x 1919.

**Virginia Beach's air-temperature result is not an artifact of light.** All
four targets read SUPPRESSED at 1.46x-1.73x — adding light terms makes the
temperature coefficient GROW.

---

## RULED OUT — do not propose these again

- **The front/behind sun split as a way to separate lens glare from sea
  breeze.** It cannot, at this camera: sun-behind is 06:00-09:00 local and
  nothing else, so the split is morning-versus-midday and every story predicts
  a front-heavy result. The test ran, the contrast is enormous (+0.0372 vs
  +0.0005), and it decides nothing.
- **Afternoon sea breeze as what the sun terms carry.** An afternoon effect is
  azimuth ~220-260 here and the rotation would peak there. It peaks at 185.
- **The published bearing-term F-test as evidence that 206 deg is special.**
  cos(az-b) = cos(az)cos(b) + sin(az)sin(b), so `sun_in_view` IS the azimuth
  harmonics rotated: fitted R2 on cos(az), sin(az) is 1.000000 with
  coefficients -0.8989, -0.4381 = cos(206), sin(206). The comparator rung holds
  no azimuth, so every bearing buys something. Only the rotation distinguishes.
- **The four rain correlations reversing between years.** Gone on rip frames
  under every variant: all four pairs POSITIVE on both sides (pre +0.128 to
  +0.172, post +0.050 to +0.076, every p below 1e-5). It was the 20-day object
  block sitting in the 2024 half — 6,582 detections in June alone against a
  typical Walton month of 700-2,600 — not "rain empties a beach", since there
  are no person detections across 2024 outside those 20 days.
- **The MOP-vs-buoy ratio as a cross-era result.** Buoy 46236 runs 2025-08-31
  to 2026-08-31, entirely post-split. Zero matched hours in the low-floor era.
  A prompt asking for that number per era is asking for one that cannot exist.
- **VB temperature as a GLARE proxy** (bearing terms dR2 <= 0.0007, p 0.36-0.97),
  **as DAYLIGHT** (solar elevation moves the coefficient the wrong way),
  **as HAZE** (cloud moves it 2-4%, sign inconsistent, and cloud's own
  correlation is POSITIVE — backwards for image degradation), **as
  people-on-the-beach** (VB is 2,352 of 2,352 rip class; there are no person
  detections there, ever).
- **A letterbox / model-space transform for box coordinates.** Coordinates
  reach 97-100% of the still's dimensions. Six transforms built; none needed.
- **A fixed scene feature explaining Walton's driver results.** Busiest cell 0.8%.
- **Walton's year effect as weather.** It is the threshold change, surviving
  residualizing at +35.7 sd.
- **"Three cameras have never detected a rip."** Corolla Hampton Inn, Corolla
  Sailfish and Carova emit class `rip` from a different model. Carova is 100%
  rip. This error has now been reached twice by different routes — any code
  that buckets classes must match every name in `RIP_CLASSES`, not just
  `rip_current`.
- **The 2026-08-01..08 Walton / Panama City coincidence, as firmly
  unexplained.** Still unexplained, but the argument is weaker than it was.
  Score FLOORS are set per model family (Corolla near 0.70 from Feb 2024 while
  Walton sat at 0.50), but the platform demonstrably acts fleet-wide across
  families on a single day — the object block proves it. A same-week
  coincidence at two cameras is therefore not evidence against a platform
  cause. Do not re-run the floor comparison; it has been done.

---

## OPEN — ranked

0. **Is 2.7% the detector's failure, or the medium's?** Decides whether this
   project reports a broken detector or an untestable one. 6 rips in 302
   judgeable frames. Either the detector is near-useless, or rips are rarely
   identifiable in a SINGLE STILL by ONE observer (practitioners use video or
   time-averaged imagery). More of the same labelling cannot separate them: at
   a 2.6% rate, detecting a doubling between bands needs ~868 labels per band.
   **The test is built but BLOCKED on the data.** `build_ripaid_calibration.py`
   draws 60 blinded RipAID frames (30 human-annotated rips, 30 real negatives,
   doubt frames excluded); `analyze_ripaid_calibration.py` reports Cohen's
   kappa against the annotators. An hour's labelling once the input exists.
   **BLOCKED: no annotations on disk.** `RipAID_v1.0.0/` holds
   `images/default/` only — 480 PNGs, cameras clm_s_01..05 and snb_s_01..03,
   2011-10 to 2024-09 — a partial extract with no annotation file beside it.
   **RipAID v2.0.0's `yolo-obb` export is the input, and the reader is
   written.** `load_ripaid.frames_from(path)` takes either a v1.0.0 COCO
   export or a v2.0.0 YOLO-OBB dataset root, and returns the same frame table,
   so both calibration scripts accept either unchanged. Take the yolo-obb
   download, not the CVAT backup: the backup exists to be restored into CVAT
   and carries CVAT-format annotations, while YOLO-OBB is per-image TXT that
   parses directly.
   **The class mapping is recovered from the data, not assumed.** The download
   ships no data.yaml, so nothing in it states which index is which. Counting
   instances per index over the 6,789 label files returns 4,103 / 1,437 /
   4,591 for 0 / 1 / 2, matching the README's published per-class totals
   exactly; because the three differ, the assignment is forced. `0 =
   rip_current, 1 = doubt, 2 = sediment`, and `verify_classes()` re-runs that
   check on every load and says so loudly if it stops matching.
   **Verified against the download:** 6,789 label files, 1,082 empty (the
   no-annotation images), 2,815 named `clm_*`/`snb_*` (the SIRENA subset, the
   only frames with a bearing, a lat/lon and a clock). All three match the
   README.

1. **Is the midday effect the detector or the water?** Largest surviving effect
   in the project. Sea breeze is out and camera-specific glare is unsupported
   (above). What remains: something in the WATER that peaks near solar noon
   (sun angle changing what is visible through the surface), or something in
   the DETECTOR that does (contrast, exposure, overhead light).
   **The test is built**, as part 2 of `analyze_ripaid_calibration.py`: the
   same rotation against HUMAN annotations. Read section (b) — drawn rip size
   and annotator doubt within annotated rips. Section (a), rip presence, cannot
   contribute (see CONSTRAINTS).
   **Cala Millor is the best discriminator in the project**: shore normal 90
   deg against a solar noon of 180, a 90 deg separation, versus Walton's 26.
   **Son Bou is structurally incapable** — its shore normal IS 180, so no peak
   can be closer to one than the other, whatever its n.

2. **What are the false positives made of?** The notes column on the 360 labels
   is the only record. Jetty, boat wake and whitewater-on-rocks recurred. If
   they concentrate spatially the finding is actionable (mask those regions).

3. **What IS Virginia Beach's temperature tracking?** Four explanations dead;
   the -0.36 is still there and still suppressed by light. Proposed: a
   sea-breeze / storminess index. NEW: Walton may echo it — in the pre era
   `temperature_2m` is the top predictor of both count targets and flips sign
   under controls (-0.0595 → **+0.1867** rate, -0.0530 → **+0.1895** detections
   with hour and month removed). n=1,400, one era: a lead, not a finding.

4. **VB's shore normal is a guess (90.0 deg).** Every onshore wind and wave
   component there is computed from it. `check_camera_geometry.py` exists.

5. **Deck slide 6** still carries unmatched-hours ratios and two verdict flags
   the matched-hours run changed, plus everything above.

---

## CONSTRAINTS the planner must respect

**Filter by `score_classes` first.** Two rip names (`rip_current`, `rip`) and
one object stream share the column. Blank class lists are observed zeros and
must be KEPT where a camera has any — Walton has none. At Walton the filter is
a truncation of the first 24 days; at VB, both Panama City views and Carova it
is a no-op.

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

**RipAID rip PRESENCE is not a valid target.** Frames were pulled around
lifeguard sightings +/-2h and "most of the images that did not show rip
currents were removed", so the negatives are the residue of a hand deletion.
What survives the selection is asked WITHIN annotated rips: drawn size
(z-scored within camera — pixel areas are not comparable across them) and
annotator doubt.

**RipAID v2.0.0 is three datasets in one, and only one third is usable here.**
Confirmed from its README: 6,789 images = 2,815 from v1.0.0 (SIRENA fixed
cameras, two Mediterranean beaches) + 2,944 from RipScout (aerial, drone and
Google Earth) + 1,030 from UFSC CoastSnap (crowd-sourced smartphone shots,
Brazil). **Only the 2,815 SIRENA frames have a fixed viewing bearing, a known
lat/lon and a timestamp in the filename**, so only they can carry the bearing
rotation — a drone image has no camera bearing to rotate against.
`build_frames` drops the rest as unparsed, which is the right behaviour, but
it means v2.0.0 buys no extra rotation power over v1.0.0.

**v2.0.0 adds a third class, `sediment`** — "a sediment plume that might
relate to a rip current", 4,591 instances across 1,588 images (23.4%). It is
neither a rip nor a clean negative, so like `doubt` it is excluded from both
strata of the calibration draw (`clean_strata` does this; a v1.0.0 table with
no such column is unaffected). Label
counts: rip_current 4,103 instances / 3,577 images (52.7%), doubt 1,437 /
1,259 (18.5%), sediment 4,591 / 1,588 (23.4%), no annotation 1,082 images
(15.9%). Those are WHOLE-dataset figures; the SIRENA-only split is not given
and must be measured.

**The DOI in the code is v1.0.0's.** v1.0.0 = 10.5281/zenodo.15082427,
v2.0.0 = 10.5281/zenodo.18196300. `load_ripaid.py` and this brief cite the
first.

**Any bearing-derived term carries the whole azimuth unless azimuth is in the
comparator.** To ask whether a specific bearing matters, rotate it and race the
peak against the SOLAR-NOON azimuth, not against a tolerance — at mid-latitude
the two candidate bearings are tens of degrees apart and any tolerance wide
enough to accept one accepts the other. `analyze_glare_split.classify_peak`.
Pass lat/lon to `solar_noon_bearing`: without them it takes the hourly argmax
of elevation, which rounds solar noon to the hour and lands 4-6 deg off.

**Open-Meteo bills by variable-hours.** Six ERA5 variables over Walton's
20,832-hour window to obtain one costs 6x quota and returned HTTP 429.
`pull_cloud_cover.py` fetches one variable into a sidecar. One site at a time.

**Never re-pull observations on the default one-year window.** It narrows a
site's conditions file below its rip record and silently shrinks every n
downstream. Always pass explicit --start/--end.

**Concurrency.** `pull_rip_detection.py` hits only app.webcoos.org, safe
alongside anything. `pull_site_observations.py` and `pull_cloud_cover.py` share
the Open-Meteo quota — one at a time. `wq/` writes only `data/wq/**` plus
`wq_manifest.json` and contends with nothing.

**Read rho_hrmo, except for solar elevation** — hour-of-day IS solar elevation,
so demeaning by hour-by-month removes it by construction. For light columns
read rho and rho_mo.

**`bbox_area_max` is pixels — EXCEPT on the YOLO-OBB path.** WebCOOS and the
RipAID COCO export give pixel areas; YOLO coordinates are normalised to 0-1,
so `load_ripaid.build_frames_yolo` returns a fraction of the frame. Never pool
the two. It is harmless inside this project because every area analysis
z-scores within a camera and a camera's resolution is fixed, making the
normalised area a constant multiple of the pixel area and the z-score
identical. It would not be harmless in anything comparing raw areas.
Comparable within a camera, meaningless across, in either unit.

**YOLO-OBB coordinates carry six decimal places.** A box recovered from them
is exact to about 2e-7 in area and 1.2e-5 degrees in orientation — fine for
everything here, but the reason the offline tolerances are what they are.
Tightening them below the file's own precision fails on correct arithmetic.

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
script ships one, and every fixture must have an answer known by construction,
not copied from a previous run of the same code. The era fixture is the model:
it pairs a floor change with a driver moving the OTHER way, so pooling inverts
the SIGN (truth +0.78, pooled -0.13, corrected +0.78) — a method that merely
blunted the bias would pass a tolerance test and fail that one.

**Ask for the diagnostic before the analysis.** The box-overlay investigation
went: overlay looks wrong → audit coordinates → audit still timing → audit
class names. The third invalidated the premise of the first two; six transforms
were built and none was needed.

**Race two hypotheses; never threshold one.** The first rotation run printed
"3 of 3 targets within 45 deg of the camera's axis" and concluded a lens
effect. Peak 185, camera 206, solar noon 180 — it was reporting the threshold.
Where two explanations predict nearby values, compare them against EACH OTHER
with a margin and let "cannot separate" be an allowed answer.

**Ask whether a variant could even move the number.** An era run printed a full
MOP-vs-buoy verdict under variant A that was a byte-identical copy of the
uncorrected one, because every matched hour was post-era. Any side-by-side
should report how many rows the treatment actually touched.

**Assume the first fix misses.** Fixes applied to the wrong layer and re-done
after a real run disproved them: a class default changed in one module while
another kept its own; a guard on regression slopes when the overflow was in the
product; a sign asserted backwards in a fixture; a mirror built on a false
belief about glob and symlinks; a class bucketer matching one of two rip names.
Ask for the fix AND the command that would show it failed.
