# Coastal WQ — working rules

## Hard rule: the offline suites gate every change

Run all four before you start and again before you move on:

```sh
source ~/Coastal/.venv/bin/activate
python wq/test_wq_offline.py
python wq/test_wq_pipeline_offline.py
python wq/test_wq_fit_offline.py
python wq/test_wq_geo_offline.py
```

**This is not advisory and "it was a small edit" is not an exception.** On
2026-09-17 a one-constant edit to `wq/config.py` silently deleted
`STRATUM_MIN_NARROWING` and `STRATUM_MIN_SMALLEST_LEVEL_SHARE`, because the
replaced span reached further than intended. The suites would have caught it
in seconds. They were not run, and it surfaced only when `--report` died
mid-run with an `AttributeError` — after the fit had already been redone.

`wq/config.py` is the worst file in the repo to edit carelessly: it holds the
pre-registered specification, its hash is what `wq/fit.py` refuses to run
without, and constants that look decorative are load-bearing. A `PostToolUse`
hook in `.claude/settings.json` now runs the suites automatically after any
edit to it. The hook is a backstop, not a substitute for running them.

## The specification is frozen

`config.SPEC_KEYS` is hashed into `wq_manifest.json` and `fit.py` stops if the
hash moves. Anything that changes a coefficient, a floor, or which pairs enter
the headline belongs inside it and needs a deliberate re-registration.
Anything that changes only what the report SAYS — wording gates, diagnostic
flags, coverage bars — belongs outside it. When adding a constant, say which
it is and why, in the comment beside it.

## The study population changes by amendment, not by directory contents

`wq/study_states.csv` is the declared population. Result chunks arrive for
reasons unrelated to the study — Virginia was pulled to audit one deck figure,
the Great Lakes states were pulled and then excluded — so an unrestricted
`--clean` is gated and will refuse to read undeclared states. Adopt one with
`python -m wq.manifest --amend-scope`, which writes a dated, authored entry.

## Do not delete data to fix a number

Non-detects are substituted and flagged, never dropped. Co-located stations
are labelled, never deduplicated. Heavy-tie pairs are flagged, never excluded.
Every one of these was considered, measured, and rejected because dropping the
clean low end makes every beach look worse than it is — see
`notes/carpinteria_sign_flip.md` and the docstring of
`wq/test_wq_offline.py`.

## Never report a coefficient without its n — or a rank coefficient without its ties

`n=220` on a series holding 13 distinct values is not the study `n=220`
implies. `modal_share` and `n_distinct_values` travel with every coefficient
row for that reason.
