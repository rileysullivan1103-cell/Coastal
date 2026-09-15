# wq/ — how much does the bacteria coefficient vary between beaches?

The question is not *does rain predict bacteria*. It is **how much does the
coefficient vary between sites, and is that variation explained by site type**.
The deliverable is a distribution of site-level coefficients, not a headline
number, so no mean of a coefficient is printed anywhere in this module and
nothing is ever pooled across sites.

The pullers come from the existing pipeline — `scan_cameras.get_with_retry`,
`pull_wqp_results`' verified WQP column names, `pull_observations.pull_buoy` /
`pull_coops_series` / `add_tide_state`, `pull_site_observations.open_meteo` /
`fetch_marine`, `analyze_drivers.spearman` / `demean_by`. This module adds the
study design, the hygiene, and the caching that makes a national run finish.

## Running it

```bash
./wq/run_wq.sh --all
```

`caffeinate` is first on that line, and `run_wq.py` also re-execs itself under
`caffeinate -i -m` (or `systemd-inhibit` on Linux), so a multi-hour national
pull survives the lid closing however it was started. `--no-caffeinate` opts
out. The inhibitor is probed before the re-exec, because `systemd-inhibit`
exists inside containers that have no bus to inhibit and exits non-zero there —
an unprobed `execvpe` would end the run instead of protecting it.

Stage by stage:

```bash
python -m wq.run_wq --stations      # coastal recreational stations, per state
python -m wq.run_wq --results       # 10 years of samples, per (state, year)
python -m wq.run_wq --ckan          # California's own resource
python -m wq.run_wq --clean         # hygiene, with a count for every decision
python -m wq.run_wq --spatial       # site covariates, for the analysable sites
python -m wq.run_wq --review        # the beach_type worklist, for a HUMAN
python -m wq.run_wq --strata        # assign strata from METADATA ONLY
python -m wq.run_wq --manifest      # freeze the specification, with a timestamp
python -m wq.run_wq --covariates    # conditions per site
python -m wq.run_wq --fit           # per site, per analyte, per predictor
python -m wq.run_wq --report        # the distribution
```

### Why the samples come before the spatial layers

The WQP pull returns **32,513 coastal stations** nationally. The first version
of this ran the spatial layers over all of them, at roughly three requests
each: ~100,000 requests, thirty hours, and — against a free community service
like Overpass — straightforwardly abusive. It was also mostly wasted, because
a station with four bacteria samples in ten years can never clear the
pre-registered floor and will never be fitted whatever its coastline looks
like.

So the samples are pulled first, and the spatial layers run only for stations
that could actually enter the study. **The pre-registration is not weakened by
this**: A1 requires the specification frozen before any *model* runs, and the
manifest is still written before `--covariates` and `--fit`. The covariate
values still come from the coordinate alone. Only the question of which
stations are worth computing them for is informed by the sample counts — which
is the same attrition filter C2 already reports, and those stations are named
in it as excluded on sample count.

Two further reductions, for the same reason:

**Coastline and permitted-discharge data are fetched per TILE, not per
station.** Both are shared between neighbouring stations, so a quarter-degree
tile with forty monitoring stations on it costs one query rather than forty.
The tile box is widened by the full search radius, so a station beside a tile
boundary still sees the coastline on the other side of it — without that
margin it would read as open water. In a realistic coastal cluster, 200
stations collapse to 4 tiles.

**429 is answered in minutes, not seconds.** The shared `get_with_retry`
ladder is 2/4/8 seconds, which on a rate-limited service means three retries
inside fourteen seconds and then a failed site — the national run produced
pages of those and no data. Requests to each host are now throttled to a
minimum interval, `Retry-After` is honoured where the service states one, and
the backoff runs 15s / 60s / 180s.

`--spatial` prints the station count, the tile count and the expected request
count before it starts, so a run this size cannot begin by surprise.

Everything is resumable. Stations cache per state, results per `(state, year)`,
ERA5 and waves per 0.1-degree grid cell, tide and water temperature per CO-OPS
gauge. A failed chunk is recorded and skipped; re-running the same command
retries exactly the chunks that are missing. If the first three chunks all fail
the run stops and says so, because that is a connectivity problem rather than a
data one and there is no point spending four hours discovering it.

Offline checks, no network:

```bash
python wq/test_wq_offline.py           # hygiene, strata, the manifest guard
python wq/test_wq_fit_offline.py       # the statistics, against planted answers
python wq/test_wq_geo_offline.py       # coastline geometry, against known shapes
python wq/test_wq_pipeline_offline.py  # the stages composed end to end
python test_lint_offline.py            # covers wq/ too
```

## The pre-registration has teeth

`wq_manifest.json` is written before any model runs and records the analyte
list, the predictor list, the control, the strata, the sample floor and a hash
of every pre-registered constant in `wq/config.py`.

`wq.fit` calls `manifest.require_manifest()` and **stops** if that hash no
longer matches. Lowering `MIN_SAMPLES_PER_SITE` until an interesting site
qualifies, or adding the predictor that turned out to work, fails the run:

```
wq/config.py has changed since the manifest was written.
    manifest 7594aaf0624ffdcc
    config   c1d0e93a77b41f82
This is the pre-registration doing its job.
```

Re-registering is allowed and deliberate — move the old manifest aside and
write a new one, so the two are diffable and the change is visible.

The A2 coverage rule runs at the same moment: a stratum populated for fewer
sites than its `required_coverage` is **dropped before fitting**, and the
manifest records the observed coverage that dropped it. `watershed_area_km2`
and `impervious_frac` have no offline source wired in and are expected to be
dropped this way. They are emitted as empty columns on purpose rather than
quietly left out, so the drop is on the record instead of in someone's memory.

## Site covariates, and where each one comes from

Everything below is derived automatically from the station's lat/lon. The
source and the **vintage** of every layer is written into `wq_manifest.json`
under `layers`, beside the count of stations it populated — because NLCD 2011
and NLCD 2021 are not the same covariate, and a coefficient stratified on one
is not a result about the other.

| covariate | layer | notes |
|---|---|---|
| `dist_to_stream_m`, `stream_order`, `upstream_area_km2`, `n_streams_within_2km` | NHDPlus V2.1 via USGS NLDI, flowline attributes via EPA WATERS | a stream **mouth**, not a stream: only flowline endpoints landing within 300 m of the coastline count, so a creek passing 300 m inland on its way elsewhere is not counted as an input |
| `dist_to_outfall_m`, `outfall_type`, `n_outfalls_within_2km` | EPA ECHO (CWA/NPDES) | major/minor and POTW/non-POTW where ECHO carries them |
| `impervious_frac`, `developed_frac` | NLCD, accumulated to the upstream catchment by NLDI | the NLCD **year is read from the service catalogue**, not hardcoded, and written into the manifest |
| `shore_normal_deg`, `curvature_1_per_km`, `embayment_ratio`, `land_fraction_5km`, `fetch_km_*` | OSM coastline via Overpass | see below |
| `tidal_range_m`, `datum_gauge_dist_km` | NOAA CO-OPS datums | MHHW − MLLW at the nearest gauge |

Two caveats that travel with the layers in the manifest rather than living
only here:

**ECHO gives the facility, not the pipe.** A treatment plant sited a
kilometre inland of its own diffuser makes `dist_to_outfall_m` an
overestimate, and the error is not random — big coastal plants discharge
further offshore than small ones.

**NLCD is accumulated over the upstream catchment of the flowline nearest the
beach.** That is the right denominator for a creek-mouth beach and the wrong
one for a beach whose nearest flowline drains somewhere else entirely.

### The coastline covariates

All five come from one Overpass request per site and one pass over the local
linework, in a local tangent plane in metres. The whole thing rests on one
convention: **OSM draws `natural=coastline` with land on the left of the way's
direction**. That single fact makes land/water computable from linework alone,
with no polygon fill and no raster — nearest segment, sign of the cross
product.

It is also the assumption most likely to be wrong in a specific place, because
OSM ways are edited piecemeal and one reversed way inverts land and sea
exactly where it is wrong. `coastline_sanity()` therefore checks that ways
meeting at a shared endpoint chain head-to-**tail**; two ends or two starts
meeting means a reversal, and **every coastline covariate at that site is
withheld** rather than returned upside down.

> An earlier version of that check probed each segment against itself and
> agreed 100% of the time on every input, including a deliberately reversed
> one — a segment cannot disagree with itself. It is in the test suite now
> precisely because a check that cannot fail is worse than no check: it gets
> read as evidence.

- `shore_normal_deg` — the outward normal of a least-squares tangent fitted
  over ±250 m, then **verified** by stepping 100 m along it and confirming
  that point is water. Same convention as the rip pipeline's `sites.yaml`.
- `curvature_1_per_km` — signed, from a circle through the station and points
  ±1 km along the shore. **Negative is concave/embayed.** The sign comes from
  which side of the coast the fitted centre falls on: in a bay the centre sits
  in the water, on a headland it sits in the land.
- `embayment_ratio` — straight-line over along-shore distance, ±2 km. 1.0 is
  a straight coast. It **cannot tell a bay from a headland** — both score the
  same — which is why curvature carries a sign and there is a test asserting
  exactly that.
- `land_fraction_5km` — sampled on rings whose radii go as √, so every sample
  stands for the same area and the answer is an area fraction rather than a
  count biased toward the middle.
- `fetch_km_by_octant` — marches outward until it hits land. A value at the
  25 km cap is **censored, not measured**, and carries a `fetch_capped_*`
  companion saying so. The eight octants are recorded per site;
  `fetch_km_min/mean/max` are what the pre-registration stratifies on, because
  eight more groupings over the same sites is eight more chances to find a
  flattering split.

`natural=coastline` covers ocean and Gulf shorelines only — the Great Lakes
are mapped as water polygons — so every coastline covariate is absent at a
Great Lakes site, and the coverage rule sees that rather than the code
pretending otherwise.

## beach_type is assigned by hand. Deliberately.

Enclosure is continuous. Any threshold on `land_fraction_5km` or
`embayment_ratio` gets the obvious sites right and misclassifies exactly the
ambiguous ones — a half-open embayment, a beach inside a breakwater, a lagoon
mouth — and those are the sites that decide whether stratifying on beach type
explains anything. An automatic label would put the hardest cases on whichever
side of a cutoff nobody chose deliberately, and D2 would then be reporting the
cutoff.

So the continuous covariates **sort the work**; a person assigns the label.

```bash
python -m wq.review --worklist                    # hardest cases first
python -m wq.review --ingest <csv> --by "name"    # provenance is required
python -m wq.review --status
```

The worklist carries an imagery link and the ambiguity score that ordered it.
Every ingested label records **who** assigned it and **when**; an ingest
without `--by` is refused, and an invented label is refused. `strata.py` reads
`beach_type` only from `beach_type_reviewed.csv` — putting one in
`strata_overrides.csv` is an error, not a shortcut.

A previous version classified `beach_type` from station-name keywords
("Harbor" meant enclosed, "Ocean" meant open). That is a threshold dressed up
as a rule, and it was confidently wrong on the sites that mattered. If nobody
has done the review, `beach_type` is empty, the coverage rule drops it, and D2
says nothing about beach type. **That is the correct outcome, not a bug.**

## Freezing the covariate list, and amending it honestly

The covariate list is frozen in `wq_manifest.json` before fitting and hashed
with everything else, so adding one afterwards fails the run:

```
wq/config.py has changed since the manifest was written.
To ADD a covariate without disturbing the registered pass:
    python -m wq.manifest --amend <covariate> --why "..." --by "your name"
```

Amending is allowed and **labelled**, not forbidden. `--amend` appends an
entry with its own timestamp, marks it `exploratory: true`, records who added
it and why, and leaves the registered entry untouched. An amendment with no
reason and no author is refused — that is indistinguishable from the thing the
file exists to prevent. Anything fitted with an amended covariate is an
exploratory pass and the report says so; the pre-registered result is whatever
entry 0 says it is.

### Coverage: the ~70% rule

Every covariate's populated share is reported and written into the manifest,
and anything under **70% of stations** is dropped from the pre-registered list
rather than fitted on a biased subset. One threshold for all of them, so it
cannot be tuned per covariate after the fact.

`wq/spatial.py --all` prints the coverage table; `--manifest` applies the rule
and records the observed share that dropped each one.

### Recorded but never stratified on

`STRATIFY_ON` in `config.py` is a **subset** of the covariates. Three are
recorded and deliberately excluded as groupings: `shore_normal_deg` (a
direction — 359° and 1° are adjacent, so a median split is meaningless; it
feeds the wind predictors instead), `datum_gauge_dist_km` (a data-quality
figure — grouping on it would be grouping on measurement quality), and
`fetch_km_min`/`max` (redundant with the mean). Each additional grouping is
another chance for a split to look explanatory by luck, and D5 already has
enough of those to account for.

## Hygiene: the counts are the point

`data/wq/out/hygiene_log.csv` carries a row for every decision — duplicates,
rejected statuses, blanks, field replicates, unit conversions, non-detects,
over-range results. "n = 41,000 samples" is not a finding until the records
that did not make it are accounted for.

Three that matter more than they look:

**Non-detects are substituted at DL/2, never dropped and never zeroed.**
`<10` parses to `(10.0, "below")`, not to `NaN`. A pipeline that lets
`pd.to_numeric` turn `<10` into `NaN` and then drops NaN has deleted every
clean sample at a site and kept every dirty one — every beach then looks worse
than it is and the rain coefficient inflates. The non-detect share is carried
per site and per analyte, and a site over 30% is reported **separately** from
the headline distribution, because its coefficient rests mostly on the
substitution rather than on measured variation.

**Over-range results are kept at the limit.** `>24196` is a Quanti-Tray
saturating, and it saturates on exactly the wet days the study is about.

**CFU and MPN are recorded, not converted.** They are different estimators and
no factor relates them. Per-mL and per-L *are* converted (×100, ÷10) and the
conversions are counted. A site mixing both estimators is flagged.

**Field replicates are collapsed into their sample.** A replicate counted as a
second sample inflates `n`, and `n` is the number every coefficient here is
reported beside.

## The n assertion

Carried forward from the `n=730` bug. Every fit recomputes the number of rows
with **both** sides non-null and compares it against the `n` the correlation
reports; they are written side by side in `coefficients.csv` as `n` /
`n_paired_check` and `n_ctrl` / `n_paired_check_ctrl`, and a mismatch raises
`SampleCountMismatch` and ends the run. `--lenient` downgrades it to a warning
and should not be used to make a run finish.

`wq/test_wq_fit_offline.py` reproduces the bug deliberately — it monkeypatches
`spearman` to report 730 on 60 rows — and checks the run stops.

## What the fit does

Per site, per analyte, per predictor: Spearman after removing each month's own
mean from **both** sides, the same control `analyze_drivers` applies for
hour-of-day. Both `rho` and `rho_ctrl` are reported so the collapse is visible.
In the fixture a purely seasonal predictor goes 0.81 raw → 0.05 controlled.

Alongside it, the decision-relevant target: the AUC of each predictor against
exceedance of the applicable single-sample criterion, looked up per state and
analyte from `wq/thresholds.json`. **No threshold is written in the code.**
Every fitted row records which entry it used in `threshold_source`, so a result
computed against the federal fallback is never mistaken for one computed
against a state's own rule. Great Lakes sites read the *fresh* block, because
the enterococci criterion genuinely differs.

> Every entry in `thresholds.json` is marked `_verified: false`. The values are
> transcribed from the cited documents but were not checked against the live
> regulation, because outbound network access was blocked when this was
> written. Re-read each cited criterion and flip the flag before publishing.

## Reading the report

**D1** is the distribution: median, IQR, 10th/90th, min, max, n sites, and a
histogram, per analyte/predictor. The spread is the result.

**D2** is the headline test, and it does **not** simply compare within-stratum
IQR against overall IQR. Splitting any group into subgroups narrows an IQR
mechanically, so that comparison finds structure in noise: in the test fixture,
strata assigned at *random* still produce an IQR ratio of 0.84, which a report
reading that number alone would call a finding. D2 therefore permutes the
stratum labels and asks whether the observed narrowing beats an arbitrary split
of the same group sizes. Planted structure scores p=0.000 with a ratio of 0.12
against 0.94 for the shuffle; random labels score p≈0.10.

**D3** is sign agreement, reported and not led with. A predictor can agree in
sign at 95% of sites and be useless at all of them.

**D4** is `per_site.csv`: station, region, strata, n, non-detect fraction,
coefficient and AUC per predictor, plus `shore_normal_source` and
`join_resolution`.

**D5** is the expected false-positive count at alpha beside the observed count,
with a Benjamini-Hochberg count as the honest version. The rain family is
nested, so the tests are not independent and the expectation is an
approximation, not a bound.

**D6** is how many sites have no usable predictor, which sizes where this
approach would not work at all. Read the **chance criterion** first, not the
fixed floor: the "strongest predictor" is a maximum over eleven predictors, and
the best of eleven pure-noise correlations at n=120 clears 0.20 most of the
time — so the fixed floor flatters small sites. `best_of_k_threshold` gives the
|rho| that the best of k predictors clears by chance at that site's own n
(0.256 at n=120 with 11 predictors, 0.358 at n=60).

## What this pass deliberately does not do

No pooled model. No hierarchical model. No partial pooling. This pass
establishes whether the coefficients vary and whether the pre-registered strata
explain the variation; partial pooling is the next step and should be informed
by D1 and D2 rather than run blind — in particular by how much spread survives
stratification, which is what decides how much a hierarchical prior can borrow.

## Known limits

- **The shore normal now comes from the coastline**, fitted over ±250 m, which
  is a local measurement rather than a guess. The fallback chain is still
  recorded per site in `shore_normal_source`: manual override, then CDIP MOP
  (California only), then the coastline tangent, then the bearing to the ocean
  cell the wave model answered on, then a regional default. The rip pipeline
  computed Santa Cruz onshore components off a 26-degree error until MOP
  published the real normal, which is why the source travels.
- **Every spatial endpoint is unverified.** Outbound access to NLDI, EPA
  WATERS, ECHO and Overpass was blocked when this was written, so every
  endpoint and field name in `wq/layers.py` is transcribed from documentation
  and flagged `verified: False` in the manifest. Each fetcher has a `--probe`
  mode that prints what actually comes back; run them once, correct anything
  that moved, and flip the flags.
- **Sampling is not random.** Agencies sample in swim season, on a schedule, and
  sometimes after known spills. The per-month control handles the calendar, not
  the schedule.
- **The month control cannot separate "season caused it" from "the cause only
  varies with season"** — a predictor with little within-month variation
  collapses either way. The same limit `analyze_drivers` documents.
- **Sample times are mixed.** WQP carries a time and the California CKAN feed
  does not, so the national distribution mixes hourly and daily joins.
  `join_resolution` says which per sample; "the tide at 09:30" and "the mean
  tide that day" are not the same predictor.
