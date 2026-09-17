# California scope amendment — DRAFT for confirmation

Every rule below was fixed **before any Californian coefficient existed**.
Nothing here was chosen in response to a result.

## Choices to confirm

| # | Choice | Reversible after fitting? |
|---|---|---|
| 1 | Add California; exclude the Great Lakes | yes, but re-fitting |
| 2 | `site_cluster` at 150 m as the unit of independence; nothing deduplicated | yes |
| 3 | Flat series flagged and reported, never excluded | yes |
| 4 | EPA federal fallback = the **36 per 1,000** magnitude set (marine ENT 110 → **130**) | yes, re-fit |
| 5 | AB 411 two-limb rule applied wherever a same-sample fecal result exists | yes, re-fit |
| 6 | California marine E. coli has **no applicable standard** (exceedance = NA) | yes |
| 7 | `outfall_type` retired as a Phase 2 grouping | yes |
| 8 | `region` tested but reported as **descriptive, confounded** | yes |
| 9 | D6 headline reported net of flat series and the permutation null; 0.30 primary | yes |
| 9b | Covariate build must clear per-source coverage: ERA5 99% of all stations, marine 95% of wet cells, tide 95% of gauged stations, water temp ungated | yes |
| 10 | Holdout = 20% of clusters + last 12 months, seed `20260916` | **NO — redrawing invalidates every comparison** |
| 11 | Santa Cruz Wharf and Carpinteria excluded from replication statistics | **NO — they are already excluded from the holdout too** |
| 12 | Exceedance comparisons across `region` are confounded by criterion and unit (§4b) | yes |
| 13 | **DECIDED** — NE fecal coliform scored against NSSP via a `programme` block (§4c i) | yes, re-fit |
| 14 | **DECIDED** — the 22 Great Lakes stations excluded (§4c ii) | yes, re-fit |
| 15 | **DECIDED** — evaluation drops training stations within **0.5 km** of a test cluster; 1.0 km as sensitivity (§6) | yes |

Items 10 and 11 are the ones that stop being free once Phase 2 runs.

## 1. Why California is added

The registered pass described itself as national and covered five Northeast
states. Two independent causes, both fixed before this amendment:

* **A timeout.** `scan_cameras.TIMEOUT` is 180 s; California's WQP station
  request answers in ~198 s. Every retry timed out, `pull_stations` logged one
  line and carried on, and the guard for "everything is failing" never fired
  because CA is fourth alphabetically after three states that succeeded.
  Fixed in `1c5d014`; failures made fatal in `e910937`.
* **A pilot scope.** Results were pulled for `CT,MA,NJ,NY,RI` only — 50 of 360
  chunks, a deliberate `--states` restriction. The pipeline reported it; the
  "national" framing came from outside the pipeline.

`1c5d014` also fixed a third defect found while fixing the first:
`--stations --states CA` rebuilt `stations.csv` from the requested states
alone, silently shrinking it from 32,513 rows to 2,313.

**Great Lakes excluded**, measured rather than assumed: the results are on
disk and cost no API quota, but 356 stations clear the sample floor and they
yield **351 fittable pairs, every one of them E. coli** — freshwater beaches
are regulated on E. coli and the data follows the regulation. It would add
164 Open-Meteo cells (~3–4 days staggered) to move one analyte. Two further
reasons to keep it separate: 59% of Great Lakes records merge as field
replicates against a fraction of a percent in California, which is a different
sampling design; and MN is missing 8 of 10 year-chunks with IN missing 2024.

## 2. Rules adopted before fitting

* **`site_cluster`, single linkage at 150 m.** Stations are not beaches —
  Cowell Beach in Santa Cruz is 22 identifiers inside 472 m. 3,575 stations
  are **3,173 beaches**. *Nothing is dropped.* Deduplication was tried and the
  data refused it: co-located same-agency pairs agree on a median 20% of their
  `(date, analyte, value)` readings against 1.7% for distant pairs of the same
  agency — far above chance, nowhere near copied rows. Dropping them would
  have deleted 50,430 real samples. `SITE_CLUSTER_RADIUS_KM` is outside
  `SPEC_KEYS`: D2's shuffle unit is still the station, as registered.
* **Flat-series flags**, reported not excluded. `constant_series` is
  parameter-free; `floor_pinned` is ≥90% of samples on the series minimum.
  On the five-state run, **154 of D6's 348 "no predictor clears 0.20" pairs
  (44%) are flat, not failed.**
* **EPA 36 per 1,000 magnitude set**, consistently. The block previously mixed
  sets — marine ENT from 32/1,000 (110), marine ECOLI from 36/1,000 (410).
  36/1,000 chosen because fresh ECOLI is already 235, its Beach Action Value.
  marine ENT moves **110 → 130**.
* **AB 411 two-limb rule** (17 CCR 7958). Total coliform is 10,000/100 mL and
  1,000/100 mL when fecal/total on the *same sample* exceeds 0.1. "Same
  sample" is `(station_id, sampled_at)` where a real time was reported and
  `(station_id, date)` where it was not, with midnight counting as no time —
  the rule `covariates.join_samples` already uses. Both limbs are data in
  `thresholds.json`; neither number appears in `fit.py`.
* **California marine E. coli: no applicable standard.** 7958 lists three
  analytes and E. coli is not among them. `threshold_for` refuses to fall
  through to EPA's *freshwater* 410. Exceedance columns go NA with a reason;
  coefficients are untouched.

## 3. Hypothesis-generating sites

| beach | cluster | stations |
|---|---|---|
| Santa Cruz Wharf | `CABEACH_WQX-Cowell` | 22 (11 CABEACH_WQX + 11 CEDEN) |
| Carpinteria State Beach | `CABEACH_WQX-WP0000180` | 1 |

Reported individually, **excluded from every rain replication statistic** and
from the holdout. They cannot replicate a hypothesis they generated. The
exclusion is driven by `wq/hypothesis_sites.csv` — a committed list, applied
in `task1_rain.py` through `common.drop_hypothesis_sites`, overridable only
with `--include-hypothesis-sites`.

## 4. `outfall_type` retired; `region` is descriptive

Phase 1 killed `outfall_type` as a grouping: baseline `iqr_ratio` 0.863 against
chance 0.982 (p=0.000), but **0.995 against 0.999 at p=0.165** once only the
levels holding 98% of stations remain. The effect was 47 stations, of which
`major/NON-POTW` was 81% one organization, 81% one state and **71% one NPDES
permit**. It may enter Phase 2 as a site-level covariate, never as a grouping.

`region` is tested with the same machinery
(`task2_stratum --stratum region`) and reported as **descriptive only**: in
this study region is confounded with the state programme that sampled it, the
lab method it used, and the analyte mix — California is essentially the entire
Pacific level, carries all 687 total-coliform pairs, and reports in MPN where
the Northeast reports largely in CFU. A region effect here is not a coastal
effect.

## 4b. The criteria are not symmetric between regions

`region` is the stratum this amendment exists to make testable, and the thing
that makes it hardest to read is that the two levels are not judged by the
same rule.

| | California | Northeast |
|---|---|---|
| criterion | **state regulation** — 17 CCR 7958 (AB 411) | **federal fallback** — EPA 2012 / 1976 |
| ENT | 104 MPN/100 mL | 130 cfu/100 mL (2012 RWQC STV, 36/1,000) |
| FECAL | 400 MPN/100 mL | 400 cfu/100 mL (EPA 1976, the 10%-exceedance value) |
| TOTAL | 10,000, **and 1,000** when fecal/total > 0.1 | no fitted pairs |
| ECOLI | **no applicable standard** | 410 (freshwater STV, used as a marine fallback and flagged) |
| unit reported | mostly MPN | mostly CFU |

So a Pacific exceedance is a California posting decision and an Atlantic one
is an EPA recommendation that no state in this study has been checked against.
The enterococcus limits differ by 25% (104 against 130) in opposite directions
from what the regional difference in bacteria would suggest, and CFU and MPN
are different estimators that this pipeline deliberately never interconverts.

**Therefore: any region difference in the exceedance columns (`auc_exceedance`,
and anything Phase 2 scores on exceedance classification) is confounded with
the criterion, the unit and the programme, and is not evidence about the
coast.** The rank correlations are unaffected — `rho_ctrl` never touches a
threshold — so D1, D2, D3 and D6 remain comparable across regions and only the
C4 exceedance column carries this asymmetry. Report it that way.

This is on top of the confounding already noted in section 4: region is also
the state programme, the lab method and the analyte mix.

## 4c. Decisions taken from the pre-report checks

Both findings below are now **acted on**, decided before any Californian
coefficient existed. Commits `afecf12` (both) and `d6f49e7` (the buffer);
manifest scope amendment 2 records the exclusion.

**(i) The Northeast fecal coliform is not beach monitoring.** Of 1,692 fitted
FECAL pairs outside California, **zero** come from a BEACH Act bathing-beach
station, and **1,473 (87.1%)** match a shellfish growing-area pattern exactly:
fecal-coliform-only, estuarine, numerically coded, run by the New Jersey
Bureau of Marine Water Monitoring. NSSP classifies shellfish waters on fecal
coliform; the BEACH Act programme posts bathing beaches on enterococcus, and
in this data every `BEACH Program Site-*` station measures ENT or E. coli and
none measures fecal coliform. California's FECAL, by contrast, is 488 of 498
pairs from `BEACH Program Site-Ocean`.

Left as it stood, the FECAL distribution would pool a shellfish harvest
classification with a beach posting decision and call the result one analyte.
It also means the Phase 1 `outfall_type` finding — carried by NJDEP stations —
was a finding about shellfish waters.

**DECIDED: a `programme` stratum with NSSP thresholds.** `thresholds.json`
gains a `programmes` block, looked up *before* the state block, carrying the
NSSP approved-growing-area fecal limit of **43 MPN/100 mL** (verified verbatim
against Virginia's adoption of the Model Ordinance). Two caveats travel with
it: 43 is a **90th percentile** of a station's distribution, not a
single-sample maximum, so roughly a tenth of samples at a compliant station
are expected to exceed it; and 43 is the MPN figure where the
membrane-filtration equivalent is 31 CFU, which this pipeline never
interconverts. The companion geometric-mean limit of 14 is not applied at all,
because the pipeline scores samples and not stations.

`programme` is derived by `strata.programme_of` from the **WQP location type
and nothing else** — deriving it from the analyte mix would be easier and is
precisely what `strata.py` may not do, since strata must be assignable before
results exist. So the label claims only what it can: published under the BEACH
Act, or not. It does **not** assert that a `non_beach` station is a growing
area, and one that is not would be scored against a standard it is not managed
under. Split: **1,777 `beach_act` / 1,798 `non_beach`**, of which New Jersey
is 1,744.

**(ii) The Great Lakes level is 22 stations.** All `21NYBCH` bathing beaches,
all `water_class: fresh`, contributing 22 fitted pairs, every one E. coli.
They are the only fresh-water stations in the study, so excluding them would
make the whole study marine and retire the `_default` fresh block entirely.
Without them `region` is a clean two-level stratum, Atlantic 2,830 against
Pacific 723.

**DECIDED: excluded.** Listed in `wq/excluded_stations.csv` with a reason, and
reported in the attrition census under their own outcome rather than quietly
absent. Manifest scope amendment 2.

## 5. D6 headline

Reported net of both corrections: flat series subtracted, and the observed
share compared against the within-month permutation null (200 permutations,
seed 0). **`USABLE_RHO_FLOOR` 0.30 is primary and 0.20 secondary** — at 0.20,
noise alone cleared the floor in 67% of ENT and 81% of FECAL pairs.

## 6. Phase 2 holdout — fixed now

Drawn by `python -m wq.holdout --write`, seed **`20260916`**, fraction 0.20.
Unit is the **site cluster**, stratified by `(state, analyte availability)`.

| | |
|---|---|
| eligible clusters | 3,092 |
| **held-out clusters** | **617 (19.4%)** |
| held-out stations | 696 |
| held-out samples | 183,623 |
| time holdout | all samples after **2025-06-30** (65,049 rows), at every site |
| hypothesis clusters excluded | 2 |
| `wq/holdout_sites.csv` SHA-256 | `0324ce1613abe707359d5e3686dfed5e9f6bb814a47436c1d677ad359d763ad5` |

Per state — 19.2% to 20.5% everywhere, and 19.4% to 21.7% per analyte:

| state | clusters | stations | samples |
|---|---|---|---|
| CA | 104 | 142 | 146,990 |
| NJ | 363 | 370 | 22,945 |
| MA | 85 | 93 | 5,722 |
| CT | 21 | 31 | 2,966 |
| NY | 29 | 34 | 2,159 |
| RI | 15 | 26 | 2,841 |

**Adjacency and the training buffer.** Holding out a cluster is not the same as
holding out a beach: the median held-out cluster has a training station **561 m**
away, 44.6% have one inside 500 m and **76.3% inside 1 km**, and the closest
possible is 150 m because that is the cluster radius. The evaluation fit
therefore drops training stations within **0.5 km** of any test cluster
(`holdout.buffered_training_stations`), costing 16.1% of training stations;
**1.0 km is the recorded sensitivity**, costing 40.3% overall and 51.4% of New
Jersey's — and since New Jersey is the whole shellfish population, at 1 km the
two arms differ by coast as well as by beach. `holdout_sites.csv` is unchanged:
the test set is fixed and hashed, and moving it to suit the geometry would be
choosing a test set after looking at it.

**Guard:** every diagnostic drops held-out clusters *and* the held-out months
by default, via `common.drop_held_out`. Keeping them requires
`--evaluate-holdout`, which prints that the result is a held-out score. Phase 2
training code must call the same function.

## 7. Phase 2 success criteria

The partial-pooling model must beat **both** baselines on held-out clusters:
(i) independent per-site fits, (ii) one fully pooled national model. Failing
either is failing.

Two measures:

* **log-count error** — per-sample, on held-out clusters.
* **exceedance classification**, scored **pooled across held-out clusters per
  analyte**, not per site. Exceedances are too rare for a per-site AUC: only
  **14% of coefficient rows** have both classes present at ≥5 days.

Report **Brier score and AUC with bootstrap confidence intervals resampling
CLUSTERS**, not samples — resampling samples would treat 22 Cowell Beach
identifiers as 22 independent draws, which is the error this whole document is
built to avoid.

## Still unresolved

* `beach_type` is 0 of 3,575 reviewed. It is the one stratum with no automatic
  fallback and plausibly the one that does what `outfall_type` failed to.
* `_default` fresh ENT (61, EPA 1986), marine FECAL (400, EPA 1976) and marine
  TOTAL (10,000, no federal criterion) remain unverified.
