# Carpinteria's −0.32: a small-sample fluke, not a defect

**Verdict: the current code reproduces the old number on the old subset.
There is no bug. The prior figure was a one-year window read as if it were a
finding.**

## What the old figure actually used

| | |
|---|---|
| station | `WP0000180` — "WP0000180-Carpinteria State, Santa Barbara" |
| source | `data/water_quality.csv`, the California CKAN export, via `analyze_drivers._load_ckan_wq` |
| analyte | `Coliform, Total` |
| rows at that station | **46** |
| **window** | **2025-09-02 → 2026-08-24 — eleven months** |
| rain series | a GHCND gauge (`data/precip_GHCND_*.csv`), not ERA5 |
| month control | `demean_by(series, month)` — the same function the pipeline uses |

The new figure uses `CABEACH_WQX-WP0000180`, **n=471, 2017-01-03 → 2025-12-15**,
with ERA5 rain. These are not the same study. They are an eleven-month window
and a nine-year one, and they barely overlap.

## The reproduction

Current code, current data, restricted to the CKAN window:

| subset | n | 24h | 48h | 72h |
|---|---|---|---|---|
| full record | 471 | **+0.195** | **+0.198** | **+0.217** |
| CKAN window | 15 | **−0.324** | **−0.324** | −0.024 |

−0.324 against a prior −0.32. It reproduces. Note the overlap is n=15 rather
than 42, because the pipeline's samples stop at 2025-12-15 while the CKAN file
runs to 2026-08 — so the prior figure rested on even fewer points than the
reproduction did, and still landed in the same place.

It reproduces **across a change of rain series**, GHCND gauge to ERA5, which
rules out the rain source as the mechanism.

## How unremarkable is −0.32?

Subsamples drawn without replacement from the same 471 paired samples,
seed 20260917, 20,000 draws:

| subsample size | mean ρ | sd | 95% interval | **P(ρ ≤ −0.32)** |
|---|---|---|---|---|
| n = 42 (as reported) | +0.182 | 0.172 | −0.169 … +0.499 | **0.29%** |
| n = 15 (the actual overlap) | +0.136 | 0.409 | −0.687 … +0.819 | **15.6%** |

At the n that was reported, −0.32 is a 1-in-350 draw — rare, and a p=0.036 was
duly attached to it. At the n the window actually supports, it happens **one
time in six**. The published p-value was computed on the sample it had, not on
the sample it needed, and nothing in that calculation knew the window was
eleven months of a nine-year record.

## The defect hypotheses, and why each is out

Each was checked and none is the mechanism, because the current code — which
has none of these — reproduces the number:

* **rain window alignment / backward-looking sums** — same ERA5 rolling sums
  the whole study uses, and the flip survives changing the rain source entirely.
* **sign convention** — the same `spearman` reports +0.195 on the full record.
* **month-demeaning applied to one side only** — `fit_site_analyte` demeans
  both sides; the reproduction above demeans both and still flips.
* **misaligned join** — the sign is stable across 24h and 48h and vanishes at
  72h, which is a noise signature, not an offset.
* **the 21CABCH empty-duplicate trap** — real, and documented elsewhere, but
  not this: the CKAN file keys on `StationCode`, never touches `21CABCH`, and
  the station it resolves is the one holding the data.

## One real defect found on the way, not responsible here

`_load_ckan_wq` sets `frame["nondetect"] = False` unconditionally and never
reads the `ResultQualCode` column, which is where the CKAN export puts its
censoring flag. A `<10` therefore enters as a measured 10. At Carpinteria only
2 of 46 TOTAL rows are flagged `<`, so it cannot explain a sign flip — but the
same loader feeds every CKAN figure in the deck, and at a cleaner beach the
censored share is larger. The wq pipeline does not have this problem: it reads
the qualifier, substitutes at DL/2, and flags the pair.

## What else the same path touches

Every CKAN station in the file is the same shape — **one year, n between 19 and 47**:

| station | analytes | n | window |
|---|---|---|---|
| Main Beach (Santa Cruz) | TOTAL / FECAL / ENT | 45 / 45 / 44 | 2025-09-04 → 2026-06-15 |
| Carpinteria State | TOTAL / ECOLI / ENT | 46 each | 2025-09-02 → 2026-08-24 |
| Capitola City Beach | three | 47 each | 2025-09-02 → 2026-08-24 |
| Twin Lakes State Beach | three | 23 each | 2025-09-08 → 2026-08-17 |
| Stinson, Schoonmaker, Cardiff | three | 19–26 | 2025-09 → 2026-08 |

**Santa Cruz (deck: +0.47 / +0.45 / +0.43 across three analytes).** Same
direction, roughly double the magnitude:

| analyte | full record (n≈355) | deck window (n=18) |
|---|---|---|
| TOTAL | +0.176 / +0.256 / +0.253 | +0.366 / +0.366 / +0.403 |
| FECAL | +0.124 / +0.149 / +0.148 | +0.296 / +0.296 / +0.369 |
| ENT | +0.265 / +0.332 / +0.313 | +0.282 / +0.282 / +0.394 |

Not a sign flip, but the "replicated across three analytes at ~0.45" claim is
an eleven-month window doubling a real effect of about 0.15–0.33. FECAL is the
worst affected: 0.30 in the window, 0.15 over nine years.

**Virginia Beach (deck: +0.32).** Different path — `wqp_Hampton_Inn_...csv`,
not CKAN, so the non-detect defect above does not apply. 96 enterococcus rows,
2023-04-03 → 2025-08-27. Larger and longer than the CKAN figures but still a
single small site, and **it cannot be checked here at all**: Virginia's results
were never pulled, so the state is absent from `samples_clean.csv`.

## What this changes

Nothing in the pipeline. The code is right and was right.

What it changes is how the deck's per-site figures should be read: they are
eleven-month windows at single stations with n in the twenties and forties,
and the sampling distribution at that n is wide enough to produce a confident
wrong sign about once in 350 draws, or once in six if the effective n is
smaller than the reported one. The national fit exists precisely because that
is not a basis for a claim.

---

# Addendum: Virginia Beach audited, and it fails too

Virginia's results were pulled on 2026-09-17 (10/10 chunks, no failures — the
timeout fix made this cheap). The deck's only cross-coast claim can now be
checked, and it does not survive.

`21VABCH-VA824084`, enterococcus, ERA5 rain at its own cell:

| | deck | full record |
|---|---|---|
| n | 96 | **181** |
| window | 2023-04-03 → 2025-08-27 | 2017-05-16 → 2025-09-30 |
| 24h | **+0.32** | **+0.087** (p=0.25) |
| 48h | — | +0.054 (p=0.47) |
| 72h | — | +0.046 (p=0.54) |

Not significant at any window. The same shape as the other two: a short window
roughly doubling an effect that is small or absent over the full record.

It is also tie-limited, and undeclared: **106 of 181 values (58.6%) sit on
1.0**, there are 12 distinct values in the whole series, and **zero** declared
non-detects. A "1" that appears in 59% of samples is a detection limit, not a
measurement, and WQP's Virginia feed does not say so — the same gap found in
California's `CABEACH_WQX` feed.

## All three deck beach findings, audited

| beach | deck | full record | verdict |
|---|---|---|---|
| Carpinteria | TOTAL −0.32 | **+0.198** (n=471) | wrong sign |
| Santa Cruz | +0.45 × 3 analytes | **+0.332** ENT only; FECAL +0.149 | one analyte, half the size |
| Virginia Beach | ENT +0.32 | **+0.087** n.s. (n=181) | does not replicate |

Three for three. The common cause is not a code defect in any of them: it is
that each was a one-to-two-year window at a single station, and at n in the
tens the sampling distribution of rho is wide enough to produce confident
numbers that nine years of the same beach do not support.

**The cross-coast claim should be struck.** Virginia Beach was the only
non-Californian beach figure in the deck, and on its own full record it is
indistinguishable from zero.

## One operational note

`data/wq/raw/results_VA_*.csv` now exist. `pull.load_raw_results` with no
`--states` reads every chunk on disk, so the next unrestricted `--clean` will
pull Virginia into the study population without anyone asking for it. Either
pass `--states` explicitly or amend the scope deliberately.
