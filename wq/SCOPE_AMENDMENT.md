# Scope amendment — California added after registration

Machine-readable record: entry `kind: "scope amendment"` in `wq_manifest.json`,
written by `python -m wq.manifest --amend-scope CA --why "..." --by "..."`.
This file is the prose that entry points at.

## What the registered pass was

Entry 0 of `wq_manifest.json`, `spec_hash e4da2f400be5c6f0`, written
2026-09-16T03:31:14Z against 2,854 stations. It froze the analytes, the
predictors and their families, the month control, the strata and the coverage
rule, `MIN_SAMPLES_PER_SITE`, `MIN_PAIRED_N`, `NONDETECT_FLAG_FRACTION`,
`ALPHA`, both `USABLE_RHO_FLOOR`s and `STRATUM_PERMUTATIONS`.

**None of that changes here.** The spec hash after this amendment is still
`e4da2f400be5c6f0`; `wq/fit.py` would refuse to run otherwise, which is the
guard doing its job. What changes is the set of stations those frozen rules
are applied to.

## What the registered pass actually covered, and why

It described itself as national. It was not. Fitted stations were
NJ 1,879 / MA 455 / NY 168 / CT 157 / RI 120 — five Northeast states, all
Atlantic except 22 New York Great Lakes beaches, with one organization
(`NJDEP_BMWM`) supplying 59% of them.

Two independent defects, both now fixed and both visible in the git history:

1. **California was lost to a timeout.** `scan_cameras.TIMEOUT` is 180
   seconds; California's WQP station request answers in about 198. All four
   retries timed out, `pull_stations` recorded CA as failed, printed one line,
   and carried on. CA is the only state large enough to exceed the timeout,
   and the "first N states all failed" guard never fired because CA is fourth
   alphabetically and AK, AL and AS had already succeeded. Fixed by
   `pull.WQP_TIMEOUT = 600`, applied only to WQP calls.
2. **Results were only ever pulled for five states.** 50 of 360 chunks. This
   was a deliberate `--states` pilot restriction, not a failure. The pipeline
   said so — `load_raw_results` printed the never-pulled count and
   `report_scope()` printed the five states. The "national" framing came from
   outside the pipeline.

A third defect surfaced while fixing the first: `pull_stations` rebuilt
`stations.csv` from only the states the invocation asked for, so
`--stations --states CA` replaced 32,513 national stations with 2,313
Californian ones. Now assembled from every cached chunk on disk.

Since this amendment, a state whose request **fails** is fatal
(`--allow-failed-states` to override, which puts the decision on the record).
A state that was never **requested** stays a warning: that is a scope choice,
not a failure, and conflating the two would make a deliberate single-state
pass impossible to run.

## What this amendment adds

California. WQP `statecode US:06`, the same `REQUEST_SITE_TYPES` and the same
ten-year window as every other state — nothing about the query is special.

- 2,313 CA coastal stations added to `stations.csv` (34,826 total, all 36
  coastal states now have a chunk on disk).
- 720,704 CA sample rows after hygiene, against 201,203 from the five
  Northeast states.
- Stations clearing `MIN_SAMPLES_PER_SITE`: 3,575, of which 723 are Californian.

### What it unlocks

- **`region` becomes testable.** It was a pre-registered stratum that could
  not vary: 2,757 Atlantic against 22 Great Lakes. With California it has a
  populated Pacific level, so D2 can ask the question it was registered to ask.
- **Total coliform becomes fittable.** 687 site-analyte pairs clear the floor,
  against **zero** before — the five-state pull held 216 TOTAL samples in
  total, across five stations all under the floor. One of the analytes in the
  frozen `ANALYTES` list had no result at all and this was not visible in D1.
- **E. coli stops being a rounding error.** 315 pairs against 31, and the 31
  were 22 Great Lakes freshwater beaches plus 9 inland river stations.
- **Two of the three prior small-sample findings become checkable.** Santa Cruz
  Wharf is `CABEACH_WQX-Wharf-East` (268 ENT / 268 TOTAL / 267 FECAL) with
  `CEDEN-Wharf` and `CEDEN-Wharf-East` alongside it; Carpinteria State Beach is
  `CABEACH_WQX-WP0000180` (1,412 rows 2017–2025, including 471 TOTAL and 179
  FECAL), which is the analyte the prior reversal was reported on. Virginia
  Beach remains outside the study.

### Two identity traps in the California data

- **`21CABCH` is a duplicate identifier set with no results.** 989 CA stations
  carry it, 407 of them at a coordinate also held by a `CABEACH_WQX` station,
  and **none** has a single row in the Result service. They pass through
  attrition as "no samples after hygiene". Carpinteria State Beach is listed
  under `21CABCH-823` with nothing attached; its data is under
  `CABEACH_WQX-WP0000180`. A name lookup that stops at the first match finds
  the empty one.
- **`CEDEN` and `CABEACH_WQX` can describe the same physical site.** The Santa
  Cruz wharf appears as both. `wq/clean.py` deduplicates WQP against the
  California CKAN feed, not WQP against itself, so co-located pairs from two
  WQP organizations enter as two sites. This is the same non-independence that
  makes the NJDEP stations dangerous, and D2's once-per-site shuffle cannot
  see it.

## What is NOT re-evaluated, and why that matters

The coverage rule in `manifest.write()` ran **once**, at registration, against
the 2,854 stations that existed then. Its decisions are frozen in entry 0:
`beach_type` dropped at 0.00 coverage, the four NHD covariates and
`impervious_frac`/`developed_frac` dropped at 0.00, `region`, `outfall_type`,
`tidal_range_m` and the coastline geometry kept.

Those decisions are **not** re-run for the added stations, and that is what
pre-registration means. It is also the thing most likely to mislead here: a
covariate kept because the Northeast had it may be thin or absent in
California. `tidal_range_m` was kept at 0.8935 coverage on Northeast gauges;
whether CO-OPS covers the Pacific stations as well is an empirical question
this amendment does not answer.

**So read the report's own coverage table before reading its D2.** The
`predictor_coverage` block and `strata.coverage` print the observed, current
numbers; entry 0 prints the numbers that made the decision. Where they
disagree, the coverage table is the fact and the manifest is the history.

## How to read results from before and after

Pre-registered results produced before this entry describe 2,854 Northeast
stations. Results produced after it describe a different population. They are
not interchangeable and no number should be quoted without saying which one it
came from. The five-state outputs are archived under
`data/wq/archive_5state_<date>/` rather than overwritten, so the comparison is
a diff and not a memory.

In particular, the Phase 1 diagnostics in `wq/diagnostics/` were run against
the five-state population. Their headline conclusions — outfall_type carried by
47 stations, the D6 addressable share falling from 87.7% to ~32–42% against a
permutation null, rain positive at 80–90% of sites at a median rho near 0.2 —
are statements about the Northeast. Re-running them is the point of the
amendment, not a formality.

## Still outstanding

Great Lakes results (IL, IN, MI, MN, OH, PA, WI) are being pulled and cost no
API quota, only WQP time. They are **not** part of this amendment: their
covariates were not fetched, because Great Lakes stations spread over a much
longer shoreline than California's and cost roughly 410 further Open-Meteo
cells against California's 106. Adding them is a second amendment and a
separate decision.
