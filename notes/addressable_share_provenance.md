# Where the addressable share moved from ~31–33% to 46–61%

Provenance, not new analysis. The figure is
`share_beating_own_null_p05` — the share of site-analyte pairs whose strongest
|rho_ctrl| beats that pair's own permutation null at p ≤ 0.05.

## The three stages, measured

| analyte | A. 5-state (original) | B. +California, pooled | C. beach-only | A→B | B→C |
|---|---|---|---|---|---|
| ENT | 0.330 | 0.432 | 0.460 | **+0.102** | +0.028 |
| FECAL | 0.315 | 0.371 | 0.552 | +0.055 | **+0.182** |
| ECOLI | 0.742 *(n=31)* | 0.515 | 0.520 | **−0.227** | +0.005 |
| TOTAL | — *(0 pairs)* | 0.604 | 0.607 | new | +0.003 |

**The "31–33%" was ENT 0.330 and FECAL 0.315.** The "46–61%" is the beach-only
column, which spans ENT 0.460 to TOTAL 0.607.

## What each stage contributed

**California entering did most of ENT** (+0.102) and a little of FECAL
(+0.055). More beaches, sampled far more intensively: the Pacific kept-rate is
77.3% against the Atlantic's 26.9%.

**Restricting to beaches did most of FECAL** (+0.182), and it is a power
effect rather than a quality one. Pooled FECAL has a median n_ctrl of 57;
beach-only FECAL has **297**. The shellfish growing-area stations that
`--scope beach` removes are short records, and a pair with 57 samples rarely
beats its own null whatever it measures.

**E. coli went DOWN, by a lot** (−0.227), and that is the most honest number
in the table. The original 0.742 rested on **31 pairs** — 22 New York Great
Lakes beaches and 9 inland river gauges, several with n over 150. Adding two
hundred ordinary Californian beaches regressed it to the mean. The early figure
was not wrong, it was a small unrepresentative sample of unusually long records.

**Total coliform did not exist before.** It contributes 0.607 and is the
best-powered analyte in the study (median n 330, cleanest null). Part of the
widening from "31–33%" to "46–61%" is therefore a new analyte at the top of
the range rather than movement in any existing one.

## What contributed nothing

**The tie flags did not move this metric at all.** Lowering
`FLAT_SERIES_SHARE` to 0.875 and dropping the `modal == min` requirement
changed which pairs are *flagged* and what D6 *reports*; it does not enter the
null computation, which is a per-pair permutation of that pair's own outcome.
Beach-only before and after those changes: ECOLI 0.524 → 0.520, ENT 0.460 →
0.460, FECAL 0.558 → 0.552, TOTAL 0.611 → 0.607. Noise.

**The shore-normal fixes likewise.** They improved two predictors that were
below chance to begin with, so the max-of-k rarely selected them either way.

**The D6 "comparable count" is a different quantity and drove none of this.**
D6 counts pairs with nothing clearing a fixed floor and now subtracts the ones
whose series could not move — "of 777 pairs with nothing clearing 0.20, 161
sit on a series that could not move, so the comparable count is 616". That is
a subtraction inside D6's own reporting. It shares no arithmetic with the
null-beating share.

## The short version

**The move is population, not methodology.** California entering and then
restricting to beaches account for essentially all of it; the methodological
changes made during the same period moved it by under a point. The one caveat
worth carrying is that the two ends of the range are not the same measurement:
"31–33%" was two analytes on 2,791 pairs, "46–61%" is four analytes on 2,366,
and the top of that range is an analyte that did not previously exist.
