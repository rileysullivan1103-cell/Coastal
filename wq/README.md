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

Stage by stage, which is also the order the pre-registration requires:

```bash
python -m wq.run_wq --stations      # coastal recreational stations, per state
python -m wq.run_wq --neighbours    # streams and facilities, for the A2 proxies
python -m wq.run_wq --strata        # assign strata from METADATA ONLY
python -m wq.run_wq --manifest      # freeze the specification, with a timestamp
python -m wq.run_wq --results       # 10 years of samples, per (state, year)
python -m wq.run_wq --ckan          # California's own resource
python -m wq.run_wq --clean         # hygiene, with a count for every decision
python -m wq.run_wq --covariates    # conditions per site
python -m wq.run_wq --fit           # per site, per analyte, per predictor
python -m wq.run_wq --report        # the distribution
```

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

## What the strata actually are

| variable | source | how good |
|---|---|---|
| `region` | state code and coordinate boxes | solid |
| `beach_type` | WQP site type, then station-name keywords | mixed — `beach_type_source` says which, per site |
| `freshwater_input` | a WQP Stream/Spring station within 500 m | a proxy: detects creeks somebody monitors, misses creeks nobody does, so its **yes is stronger than its no** |
| `outfall_present` | a WQP Facility station within 1 km, plus name keywords | same asymmetry |
| `tidal_range_m` | CO-OPS MHHW − MLLW at the nearest gauge | solid where a gauge is within 50 km |
| `watershed_area_km2` | nothing | dropped by the coverage rule |
| `impervious_frac` | nothing | dropped by the coverage rule |

Nothing in `wq/strata.py` may read a sample value. It runs before the results
are even pulled, which is the structural version of "assign these from station
metadata, not from the results".

Where a variable was never *checked* it stays `NaN` rather than becoming `no`.
"Nothing is there" and "nothing was looked for" are different answers and only
one of them is evidence.

`wq/strata_overrides.csv` beats every rule above, per station.

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

- **The shore normal is not known nationally.** Wind onshore/alongshore needs
  the direction each beach faces. It is taken from an override file, then CDIP
  MOP (California only), then the bearing to the ocean cell the wave model
  actually answered on, then a regional default — and `shore_normal_source` is
  carried per site so wind results computed off a guess can be separated. The
  rip pipeline computed Santa Cruz onshore components off a 26-degree error
  until MOP published the real normal.
- **Sampling is not random.** Agencies sample in swim season, on a schedule, and
  sometimes after known spills. The per-month control handles the calendar, not
  the schedule.
- **The month control cannot separate "season caused it" from "the cause only
  varies with season"** — a predictor with little within-month variation
  collapses either way. The same limit `analyze_drivers` documents.
- **`freshwater_input` and `outfall_present` are proximity proxies**, not
  surveys.
- **Sample times are mixed.** WQP carries a time and the California CKAN feed
  does not, so the national distribution mixes hourly and daily joins.
  `join_resolution` says which per sample; "the tide at 09:30" and "the mean
  tide that day" are not the same predictor.
