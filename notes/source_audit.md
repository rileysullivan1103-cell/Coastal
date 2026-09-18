# Source and variable audit — 2026-09-17

Brief: check that no source or variable is unaccounted for, particularly in
California, now that the beach headline is ~100% Californian for total and
fecal coliform.

Structure: for each of the eleven pre-registered predictors, what feeds it,
how good that is, and what better exists. Then what is missing entirely.

---

## 1. The one real defect: 61% of California has a guessed shore normal

**`shore_normal_deg` feeds `wind_onshore_ms` and `wind_alongshore_ms` — two of
the eleven predictors — and for 442 of California's 723 stations it is not
measured at all.** Those stations fall back to `REGION_SHORE_NORMAL["Pacific"]
= 270.0`, one constant, due west, for every one of them.

California does not face west. Of the 281 CA stations where the normal WAS
measured from the coastline, the compass distribution is:

| octant | stations |
|---|---|
| S–SW | **117** |
| SW–W | 90 |
| SE–S | 36 |
| N–NE | 11 |
| W–NW | 10 |
| everything else | 17 |

Using 270° on those 281 instead of their measured value would give:

| | |
|---|---|
| median angular error | **55.2°** |
| >30° wrong | 82.6% |
| >60° wrong | 42.3% |
| **>90° wrong** | **20.6%** |

A 90° error does not degrade the two wind predictors, it **swaps** them: what
is recorded as onshore wind is alongshore wind and vice versa. On the measured
sample that happens to one station in five.

**Cause, and it is not a data limitation.** `coastline_note` on the affected
stations reads:

> `POST overpass-api.de: ConnectionError | GET overpass-api.de: skipped — 3
> consecutive failures earlier in this run … | POST overpass.kumi.systems:
> ConnectionError | … | POST overpass.private.coffee: ConnectionError`

All three Overpass mirrors threw `ConnectionError` during the spatial run and
all three circuit breakers tripped. 383 of the 442 never had their coastline
requested at all. Re-tested 2026-09-17: `overpass-api.de` answers the
Carpinteria bounding box in **1 second**. The other two return 504, but the
mirror loop tries the primary first, so that does not matter.

**Action taken: `--spatial` re-run.** This is free — no new source, no
Open-Meteo quota, the pipeline's own primary layer.

---

## 2. Checked and adequate — no action

**ERA5 rainfall resolution.** Initially flagged and then withdrawn: forcing
`models=era5` returns a 0.25° cell 12.4 km inland of Carpinteria, in the Santa
Ynez Mountains, which would be a serious orographic bias. **The pipeline does
not pass `models=`.** Its default request returns a cell **3.4 km** away, and
across eight randomly sampled CA stations the offset is a median 3.6 km, max
6.4 km. The rain series is fine; the alarm was an artifact of the test.

**Tide.** CO-OPS water level, 100% of eligible stations after the polarity
fix. No better source for the US coast.

**Waves.** Open-Meteo marine, 96.1% of wet cells. Adequate but see §4.

---

## 3. Missing entirely, and free — derivable from data already on disk

These need no new source, no request and no quota. All three have a stronger
literature basis than most of what is already in the predictor list.

**Solar radiation is the strongest omission.** Sunlight-driven die-off is the
dominant loss term for enterococci and E. coli in marine water — it is why
bacteria peak in the early morning and after overcast days. `shortwave_radiation`
and `direct_radiation` are available on the **same archive endpoint, the same
cells** the pipeline already fetches, with full history (verified: 72/72
non-null for a 3-day probe at Carpinteria). This is not free — adding a
variable means re-fetching the 333 ERA5 cells — but it is one variable on
cells that are already mapped, roughly a day of the daily quota.

**Antecedent dry days.** First-flush: the bacterial spike when a long dry
spell breaks is larger than the same rainfall mid-season. 73.1% of beach rows
have zero 24h rain, so the dry-spell length is well defined and entirely
derivable from the rain series already on disk. **Zero cost.**

**Time of day.** 70.5% of beach samples carry a real clock time, heavily
concentrated at 06:00–07:00, and `join_resolution` already distinguishes
hourly from daily joins. Diurnal decay is the same physics as the radiation
term. **Zero cost.**

Day of week was checked and **rejected**: the distribution is Monday 316k,
Sunday 6.7k. That is the sampling schedule, not swimmer density, and modelling
it would fit the agency's calendar.

---

## 4. Real, costly, and genuinely uncertain

**CDIP MOP — California only, already built here (`pull_cdip_mop.py`).**
Nearshore waves propagated to the 10 m depth contour, roughly 1.5 km from the
beach, against Open-Meteo's global reanalysis cell. It also publishes
`metaShoreNormal`, the measured shore normal — which would have fixed §1 from
a different direction. If the Overpass re-run succeeds, the shore-normal case
for MOP mostly evaporates and the wave case remains: the difference between a
modelled surf-zone wave height and a 0.1° reanalysis cell is real but its
relevance to bacteria is not established. **Recommend: not now.**

**USGS discharge.** 4,623 California sites publish daily discharge, and 81% of
CA beach stations have one within 10 km (median 5.8 km). Freshwater discharge
is the dominant transport mechanism for beach bacteria, so this is the most
physically motivated missing predictor. The problem is linkage: a gauge 6 km
away on a different creek is not the creek that discharges at that beach, and
establishing which is which needs the NHD flowline network — which this
pipeline already tried and got **0% coverage** on, so it was dropped at
registration. **Recommend: only with a working catchment linkage.**

**GloFAS river discharge** (Open-Meteo flood API, verified working, free,
different host so probably a separate quota). Global 5 km model. Returns
0.02–0.03 m³/s at Carpinteria, which is plausible for a small coastal creek —
but a 5 km model on a creek that size is a guess with a decimal point.
**Recommend: no.**

---

## 5. Already investigated and closed in earlier sessions

| source | verdict |
|---|---|
| Open-Meteo marine `sea_surface_temperature` | **forecast-only** — 0 of 168 values for a 2021 window |
| NOAA ERDDAP MUR SST | works, ~100% coverage with a ≤1.45 km nudge, **declined**: a water pixel 1.2 km from a shellfish station is bay water and estuaries stratify |
| CO-OPS water temperature | 37% coverage — the gauge network, not a failure. `water_temp_c` is effectively absent and the study runs on ten predictors |

---

## 6. The data-quality problem that outranks every new source

Undeclared censoring. **2,700 of 5,115 fitted pairs (53%) have at least half
their samples on a single value**; 1,369 (27%) have 70% or more. The modal
values are 3.0 and 10.0 — method detection limits. Only 337 pairs (6.6%) carry
any declared non-detect, because **WQP publishes no censoring flag at all** for
`CABEACH_WQX` (confirmed against California's own CKAN export, which says 61%
of Carpinteria's enterococcus is `<`) or for Virginia.

No new sensor improves a series that is 73% one value. **The highest-value
request in this whole audit is not a new source: it is asking WQP for
`ResultDetectionConditionText` and `DetectionQuantitationLimitMeasure` on the
California feed specifically.** The pipeline already reads those columns. If
they exist and are simply not being returned by the current query, the problem
disappears rather than being estimated.

---

## Priority

1. **Shore normal** — in flight, free, fixes 61% of California. *(done this session)*
2. **Antecedent dry days, time of day** — free, derivable, well motivated.
3. **Ask WQP for the California censoring fields** — costs one query, could
   retire §6 entirely.
4. **Solar radiation** — one ERA5 variable, ~a day of quota, best-founded
   omission.
5. Everything else — no.

---

# Addendum, 2026-09-18 — decisions taken and two expectations overturned

## The permutation null was conservative, not inflated

The concern was that the within-month shuffle destroys the outcome's serial
correlation, making the null too easy and every "excess over null" an upper
bound. `--null circular_shift` rolls the series along the time axis instead,
so every autocorrelation survives and only the covariate alignment breaks.

On the beach population it goes the other way:

| analyte | null@0.30 within → circular | beats own null within → circular |
|---|---|---|
| ECOLI | 0.079 → **0.047** | 52.4% → **53.8%** |
| ENT | 0.182 → **0.163** | 46.0% → **47.1%** |
| FECAL | 0.114 → **0.071** | 55.8% → **58.4%** |
| TOTAL | 0.068 → **0.050** | 61.1% → **61.9%** |

Within-month shuffling keeps a sample inside its own calendar month and so
near similar covariate conditions, which makes it the **tighter** null. The
headline share of pairs beating their own null was understated, and the two
nulls differ by only one to three points — which is reassuring about both.

## E. coli river stations: already excluded

Of the ten non-beach E. coli stations, six are USGS gauges on the Connecticut,
Quinnipiac and Naugatuck rivers, one is a Delaware river, one the Passaic at
Clay St, one a Harborwatch estuary point and one a Yurok tribal estuary site.
**`study_scope == beach` already excludes all ten**, leaving 283 beach E. coli
stations. No further action needed.

Two of them — Smyrna River and Passaic River — landed in `shellfish` rather
than `unclassified`, because they report fecal coliform and the shellfish rule
is an inference from that. That is the documented error mode of the rule,
affecting two stations.

## Great Lakes: IN by the stated rule, pending a quota decision

All chunks now complete (IN 2024 and MN 2019–2026 were transient WQP failures
and fetched on retry). **368 stations clear the 30-sample floor** — MI 194,
OH 81, IL 29, PA 27, IN 25, MN 12 — against a rule of "in if roughly 100
clear". 363 fittable pairs, **every one E. coli**, which is the regulatory
split doing its work: freshwater beaches are posted on E. coli.

Including them would roughly double the study's E. coli and give it a genuine
second group. The cost is **164 new Open-Meteo cells**, about 1.7 days of ERA5
against the daily ceiling plus marine.

**DECIDED 2026-09-18: not worth it. The Great Lakes stay out.** The count
cleared the stated rule, so this is a decision against the rule rather than an
application of it, and the reason is what the rest of the audit found: the
Great Lakes would add 363 pairs of a single analyte to a study in which no
grouping explains the spread and only rain predicts anything. A second E. coli
group buys a comparison between two populations whose coefficients are already
known not to be explained by which population they came from. That is not
worth 1.7 days of quota.

Nothing changes in the code: scope amendment 2 and `wq/excluded_stations.csv`
already exclude the 22 New York stations, and the seven Great Lakes states
were never adopted into `wq/study_states.csv`. The results stay on disk and
cost nothing; the gate in `pull.load_raw_results` stops them entering a clean
by accident.

---

# Addendum 2 — the three candidate predictors, tested

All three were proposed in §3 above on the strength of their physics. All
three were then measured on the beach population, month-demeaned within site,
against `rain_48h_mm` as the incumbent benchmark. **The null rate for a single
predictor at the median n of 65 is about 11.6%**, which is the number
"share clearing 0.20" has to beat.

| candidate | pairs | median rho | median \|rho\| | share \|rho\| ≥ 0.20 | verdict |
|---|---|---|---|---|---|
| **rain_48h** (incumbent) | 3,223 | **+0.191** | **0.197** | **49.0%** | the benchmark |
| antecedent dry days | 3,099 | −0.007 | 0.074 | **12.0%** | **indistinguishable from noise** |
| time of day | 2,392 | −0.032 | 0.088 | 18.0% | marginal |
| solar radiation | 798 | **−0.111** | 0.112 | 19.8% | real, and weak |

**Antecedent dry days is exactly the null rate.** 12.0% against 11.6%
expected. The first-flush effect is real in the literature and is not visible
here at all — plausibly because 73% of beach rows have zero 24-hour rain, so
the dry-spell variable is mostly measuring "it is summer in California".

**Time of day is marginal** and confounded: sampling is concentrated at
06:00–07:00, so the variation is between a handful of agencies' schedules
rather than across the diurnal cycle.

**Solar radiation is the real one, and it is still weak.** The sign is
negative, as sunlight-driven die-off requires, and it is consistent across all
four analytes — ENT −0.134, TOTAL −0.116, FECAL −0.080, ECOLI −0.072 — with
enterococcus the strongest, which is what UV sensitivity predicts. So this is
a genuine signal rather than noise. But at a median |rho| of 0.112 it performs
like `wave_height` (0.09), which is already in the predictor list and
contributes nothing usable.

**Recommendation: do not add any of the three.** Solar would cost a full
333-cell ERA5 re-fetch to buy a predictor in the same band as the ones already
doing no work. One caveat worth recording: this tested same-day mean
shortwave, and the physics is about cumulative dose between contamination and
sampling. A lagged solar term might do better and was not tested.

The broader point is the one that keeps recurring. Rain is the only predictor
in this study that does anything, no site attribute explains why it varies
between beaches, and adding more weather does not change either fact.
