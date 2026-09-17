# Phase 1 diagnostics

Post-hoc checks on the completed `python -m wq.run_wq --fit --report` run.

Nothing here refits, re-pulls or re-runs any pipeline stage. Every script
reads the existing files under `data/wq/` and writes **only** into
`data/wq/out/diagnostics/`. The pre-registered outputs in `data/wq/out/` are
read-only evidence and are never opened for writing.

```sh
source ~/Coastal/.venv/bin/activate     # Python 3.9, from the main checkout
cd ~/Coastal-wq
```

## Commands

```sh
python -m wq.diagnostics.task0_reconcile
python -m wq.diagnostics.task1_rain
python -m wq.diagnostics.task2_stratum                     # default: outfall_type
python -m wq.diagnostics.task2_stratum --stratum region
python -m wq.diagnostics.task3_null                       # 200 permutations, seed 0
python -m wq.diagnostics.task3_null --permutations 500 --seed 7
python -m wq.diagnostics.task3_null --permutations 20 --limit 300   # timing probe
```

Runtimes on the machine this was written on: task0 ~5s, task1 ~10s,
task2 ~40s per stratum (400 permutations per variant), task3 ~10s at 200
permutations.

## What each one does

| script | task | writes into `data/wq/out/diagnostics/` |
| --- | --- | --- |
| `task0_reconcile.py` | Closes the gap between `coefficients.csv` (32,857 rows) and D5's test count (24,596) | `task0_reconciliation.csv`, `task0_untested_by_predictor.csv`, `task0_by_analyte.csv` |
| `task1_rain.py` | Rain vs bacteria per analyte and window, with sign, raw significance, BH and magnitude side by side; broken out by `outfall_type`; looks for the three prior small-sample beaches | `task1_rain_by_analyte.csv`, `task1_rain_by_outfall_type.csv`, `task1_negative_by_outfall_type.csv`, `task1_prior_sites_lookup.csv` |
| `task2_stratum.py` | **EXPLORATORY / post hoc.** Stress-tests any pre-registered grouping (`--stratum`, default `outfall_type`): small levels excluded, small levels merged, leave-one-level-out, large-levels-only, pooled and per analyte | `task2_<stratum>_variants.csv`, `task2_<stratum>_small_level_stations.csv`, `task2_<stratum>_small_level_shared.csv` |
| `task3_null.py` | A permutation null for D6's addressable-site count: the outcome is shuffled within calendar-month blocks and the max-of-k `\|rho_ctrl\|` is recomputed | `task3_null_summary.csv`, `task3_null_per_pair.csv`, `task3_ecoli_detail.csv` |

`common.py` holds the loaders and the definitions the scripts share
(`headline`, `untested_reason`, `bh_significant`). It reads those definitions
off `wq/fit.py` and `wq/report.py` rather than restating them.

## Things worth knowing before reading the numbers

* **`headline` is not `coefficients.csv`.** `wq.report.run` drops every row
  belonging to a site-analyte pair over `NONDETECT_FLAG_FRACTION` before D1-D6
  see it. Any diagnostic reproducing a headline number has to drop them too.
* **BH is global.** `report_multiple_testing` runs one Benjamini-Hochberg
  step-up over all 24,596 headline tests. `common.bh_significant` reproduces
  that set, not a friendlier per-analyte one.
* **Every task takes `--evaluate-holdout` and `--include-hypothesis-sites`.**
  By default each one drops the 617 held-out site clusters, the samples after
  2025-06-30, and the two hypothesis beaches (Santa Cruz Wharf, Carpinteria
  State Beach). Those defaults are the point: a Phase 1 number computed over
  the held-out clusters is not a held-out number any more, and a replication
  statistic that includes the beaches which generated the hypothesis is
  answering a question with itself. See `wq/holdout.py`.
* **Task 2 takes `--stratum`.** It stress-tests whichever registered grouping
  you name; the outputs are named after it, so runs on different strata do not
  overwrite each other. It warns if the stratum is not in the manifest's active
  strata.
* **Task 2 reuses the pipeline's test.** `report._stratum_significance` is
  imported, not reimplemented, so a variant differs from D2 only in the labels
  it is handed. That function shuffles the stratum label once per *site* and
  reuses the shuffle across every analyte/predictor cell, which is what makes
  its p-values mean anything.
* **Task 2 is not pre-registered.** It re-asks a question D2 already answered
  under the frozen specification. Whatever it shows is a hypothesis for Phase
  2, not a result of this pass.
* **Task 3 checks itself against the fit first.** It recomputes every observed
  `rho_ctrl` from `samples_with_covariates.csv` and asserts agreement with
  `coefficients.csv` to 1e-6 before permuting. All 24,596 agree to 3.3e-16.
  The permutation preserves each predictor's paired-row set exactly, so every
  `n_ctrl` in the null equals the `n_ctrl` in the fit.
* **The month control is by calendar month.** `wq.fit.fit_site_analyte` uses
  `.dt.month`, which pools January across all years. Task 3's permutation
  blocks are those same calendar months, so the control survives the shuffle
  untouched.
