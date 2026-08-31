# Coastal

Site-discovery pipeline for coastal monitoring: find WebCOOS camera locations
that also have a nearby NDBC buoy, a high-coverage NOAA precipitation station,
and a water-quality station.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then put both tokens in `.env`. Nothing needs exporting — every script does
`import env`, which reads `.env` straight into `os.environ`.

## Coming back to this after closing the terminal

Two things do not survive a closed window: the activated virtualenv, and
anything exported into the shell. The `.env` loader takes care of the second,
but the venv still has to be re-activated each time:

```bash
cd /path/to/Coastal && source .venv/bin/activate
```

If a script fails and it is not obvious why, run the doctor first:

```bash
python doctor.py
```

It reports which python is running, whether it is this project's `.venv`,
which of the five packages import, whether both tokens are readable, and which
intermediate files exist — then prints the exact commands to fix whatever is
missing. It imports nothing outside the standard library, so it still runs on
a bare system python where `pandas` is absent, which is one of the things it
is there to detect.

`pywebcoos` is **not on PyPI** — `pip install pywebcoos` fails. It installs from
GitHub only, which `requirements.txt` handles:
`git+https://github.com/WebCOOS/py-webcoos-client.git`

## Tokens

| Token | Where to get it | How it is sent |
|---|---|---|
| `WEBCOOS_TOKEN` | Register at <https://app.webcoos.org/u/accounts/login/>, then copy the token from your profile page | `Authorization: Token <value>` |
| `NOAA_CDO_TOKEN` | Request at <https://www.ncdc.noaa.gov/cdo-web/token> (emailed immediately) | `token: <value>` header |

Both are read from the environment. Never hardcode them — `.env` is gitignored.

NOAA CDO limits: 5 requests/second, 10,000 requests/day.

## Run order

Run these in order. Each one is cheap and catches a class of failure before the
expensive bulk download.

```bash
python doctor.py                 # 0. is this checkout runnable at all?
python check_tokens.py           # 1. are both tokens actually accepted?
python verify_webcoos_fields.py  # 2. where do camera coordinates really live?
python verify_ca_ckan.py         # 3. find the CA water quality resource id
python test_matching_offline.py  # 4. matching logic sanity check (no network)
python find_candidate_sites.py   # 5. the real run
```

Step 2 may tell you to update `_GEOM_PATHS` in `find_candidate_sites.py` before
step 4 will work. That is the point of running it.

`verify_wqp_fields.py` is only needed if you widen `REGION` beyond California —
see Region scoping below.

## Scripts

- **`check_tokens.py`** — one cheap authenticated request per service. Reports
  "rejected" separately from "unreachable" so you can tell a bad token from a
  network problem.
- **`verify_webcoos_fields.py`** — probes `GET /webcoos/api/v1/assets/` and prints
  the full field map plus every coordinate-like path, and dumps the raw JSON.
  This is what confirms where camera coordinates actually live.
- **`verify_wqp_fields.py`** — pulls a small bbox from the Water Quality Portal
  and prints the real column names against the ones the pipeline expects.
- **`verify_ca_ckan.py`** — searches data.ca.gov for the California water
  quality dataset and dumps a resource's column names.
- **`find_candidate_sites.py`** — the pipeline. Bulk-downloads each source once,
  then matches locally.
- **`test_matching_offline.py`** — exercises the matching/ranking logic against
  synthetic fixtures, no network needed.
- **`env.py`** — loads `.env` into `os.environ` on import. An already-set
  variable always wins, and `.env` is looked up beside the script rather than
  in the current directory, so running from anywhere works.
- **`doctor.py`** — standard library only. Diagnoses a checkout that will not
  run and prints the commands to fix it.
- **`test_env_offline.py`** — edge cases for the loader: quoting, `export`
  prefixes, and not clobbering a variable the shell already set.
- **`pull_rip_detection.py`** — downloads WebCOOS's own rip-detection product
  for Walton Lighthouse. Resolves feed and product by slug, so it is not
  subject to the pywebcoos label bug described below.
- **`test_rip_offline.py`** — synthetic-fixture checks for the rip pull: slug
  matching, camera resolution, and payload-to-table conversion.

## Verified field names

Confirmed by reading the installed library sources:

- `pywebcoos.API.get_cameras()` returns a DataFrame with **one column,
  `'Camera Name'`** — no coordinates. (`pywebcoos/API.py::_get_camera_list`
  builds it from `asset['data']['common']['label']` only.) Distance matching
  therefore reads the `/assets/` endpoint directly rather than using this.
- WebCOOS assets endpoint is `https://app.webcoos.org/webcoos/api/v1/assets/`,
  auth header `Authorization: Token <token>`, DRF-paginated
  (`count`/`next`/`results`).
- `NdbcApi().stations()` columns are **capitalized**: `Station`, `Lat`, `Lon`,
  `Elevation`, `Name`, `Owner`, `Program`, `Type`, `Includes Meteorology`,
  `Includes Currents`, `Includes Water Quality`, `DART Program`.
  (`ndbc_api/api/parsers/http/active_stations.py`)

## Still unverified

Neither of these could be checked against a live response yet, so both have a
probe script and both fail loudly rather than silently producing zero results.

- The exact JSON path to camera coordinates in the WebCOOS asset record.
  `_GEOM_PATHS` in `find_candidate_sites.py` tries the plausible shapes and
  raises if none match — run `verify_webcoos_fields.py` and pin the real path.
- Water Quality Portal field names (`LatitudeMeasure`, `LongitudeMeasure`,
  `MonitoringLocationIdentifier`) and the `/data/Station/search` parameters —
  run `verify_wqp_fields.py`.

## Region scoping

`REGION` in `find_candidate_sites.py` selects the search area. It currently
defaults to `"california"`; `"us_coastal"` is the original nationwide box.

The region drives three things: cameras outside it are dropped, the NOAA CDO
`extent` is scoped to it, and the WQP `bBox` is scoped to it. Note the two
formats use opposite coordinate order — `region_extent()` emits
`min_lat,min_lon,max_lat,max_lon` for CDO, `region_bbox()` emits
`min_lon,min_lat,max_lon,max_lat` for WQP.

**The California region skips the national Water Quality Portal**, using the
dedicated `data.ca.gov` source instead. That also means `verify_wqp_fields.py`
is not on the critical path for a CA-only run.

## California water quality

Configured to the **Beach Water Quality Monitoring Stations** resource on
data.ca.gov (`98e628ff-d012-4982-ad32-b9f9ad8ab524`), 1041 rows, one per
monitoring station.

| config | value | why |
|---|---|---|
| `CA_CKAN_LAT_COL` / `CA_CKAN_LON_COL` | `Station_UpperLat` / `Station_UpperLon` | stations carry Upper and Lower pairs, identical for point stations |
| `CA_CKAN_ID_COL` | `Station_id` | `Station_Name` and `AgencyStationIdentifier` are both literally `"0"` on many rows |
| `CA_CKAN_LABEL_COL` | `Beach_Name` | so output names a beach, not a bare number |
| `CA_CKAN_ACTIVE_ONLY` | `True` | a decommissioned station would imply coverage that no longer exists |

Rows with `0.0` coordinates are dropped — that is this dataset's stand-in for a
missing location, not a point off West Africa. The resource is denormalised
(station joined to beach and agency), so rows are deduplicated on `Station_id`.

The sibling **Fecal Indicator Bacteria Monitoring Results** resource
(`15a63495-8d9f-4a49-b43a-3092ef3106b9`) has coordinates too, but it is ~627k
sample rows — the wrong shape for site discovery. It is the right source for
the *measurements* once sites are chosen, and it carries precomputed
`30DayGeoMean` / `6WeekGeoMean` columns.

If `CA_CKAN_RESOURCE_ID` is set to `None`, California sites fall back to being
**assumed** to have coverage rather than checked. Those rows are reported
honestly — `wq_station_id` is `CA_CKAN_ASSUMED`, `wq_distance_km` is `NaN`,
`wq_source_confirmed` is `False` — and the run prints a warning.

To re-explore the catalog:

```
python verify_ca_ckan.py
python verify_ca_ckan.py RESOURCE_ID [RESOURCE_ID ...]
```

## Distance thresholds

## Current results (California, 2026-08-31)

86 WebCOOS cameras nationally, 10 in California, of which **7 qualify** on
buoy + precipitation + water quality.

| camera | buoy | km | beach | km |
|---|---|---|---|---|
| San Elijo State Beach | 46274 | 5.8 | Cardiff State Beach | 0.1 |
| Stinson Beach | 46237 | 12.0 | Stinson Beach | 0.1 |
| Sausalito - Galilee Harbor | 46237 | 15.1 | Schoonmaker Beach | 0.2 |
| Walton Lighthouse, Santa Cruz | 46236 | 22.9 | Twin Lakes State Beach | 0.4 |
| Santa Cruz Wharf | 46236 | 23.2 | Main Beach | 0.3 |
| Capitola Wharf | 46236 | 23.6 | Capitola City Beach | 0.3 |
| Carpinteria State Beach | 46053 | 33.7 | Carpinteria State | 0.3 |

The three that do not qualify — Crescent City, Humboldt Bay/Arcata, and Point
Reyes — all have a buoy and a precipitation station. They fail only on water
quality: no active monitoring station within `MAX_WQ_DISTANCE_KM`.

Beach names come from CKAN and camera coordinates from WebCOOS, so the two
sources agreeing at 0.1-0.4 km is an independent check on the matching.

## The stations-to-results join

`verify_ckan_join.py` confirmed that `stations.Station_Name` <-> `results.StationCode`
is the correct key: 1041 x 2705 rows join to 636, median coordinate gap 0.03 km,
97% of pairs within 1 km.

The remaining 3% matters. Use `ckan_join.join_stations_to_results()` rather than
a raw merge, because three separate problems live in that tail:

| symptom | cause |
|---|---|
| `1100` -> Rincon Beach vs Crescent City, 963 km, names disagree | short agency-local codes are not unique statewide |
| `EH-130`, `BC-010`, `BC-020` ~13,400 km, names agree | one side carries the `(0,0)` placeholder |
| `BNB25` 29.7 km, Laguna Beach 13.8 km, names agree | a coordinate is simply wrong in one table |

Only the first is a true mis-join, and it is the dangerous one: it would
attribute Crescent City bacteria readings to a Rincon Beach camera with nothing
in the output looking wrong. The helper validates every pair against the one
thing the key cannot fake -- both tables independently recording the station's
location -- and drops disagreements past `MAX_JOIN_DISAGREEMENT_KM`.

Also worth knowing about the overlap: 633 keys are in both tables, 404 stations
have no results in the 2020-present resource (check the 2010-2020 one), and
2063 results stations are not beaches at all -- the results resource is
statewide surface water, so codes like `514SAC011` are Sacramento River sites.

## Pulling observations

`pull_observations.py` fetches a year for each qualifying site. The four
sources have four different native resolutions and the script does not paper
over that:

| source | resolution | what you get |
|---|---|---|
| NDBC `stdmet` | hourly | wind, waves, water temp (`WTMP`), mean wave direction (`MWD`) |
| CO-OPS `water_level` | 6-minute | level, rate of change, rising/falling/slack |
| CO-OPS `wind` | 6-minute | speed, direction, gust |
| CO-OPS `water_temperature` | 6-minute | water temp at the coast, not offshore |
| NOAA CDO GHCND | **daily** | precipitation, plus 24/48/72h rolling totals |

A precipitation station must also have reported within
`MAX_PRECIP_STALENESS_DAYS`. `datacoverage` is a lifetime figure, so a station
that stopped reporting years ago still scores 0.98 and gets matched — three of
the seven sites initially drew stations that returned no data at all for the
past year.
| data.ca.gov CKAN | irregular | bacteria samples, a few a week in swim season |

**Hourly precipitation is not available.** GHCND publishes one PRCP total per
day, so `rain_24h_mm` / `rain_48h_mm` / `rain_72h_mm` are 1/2/3-day rolling
sums. `RAIN_INCLUDE_SAME_DAY` decides whether the window ends on the sample day
(so rain falling after a morning sample still counts) or the day before (strictly
antecedent, but blind to same-day rain). Daily data cannot separate the two.

A day the station did not report stays `NaN` rather than becoming 0, so any
window spanning a gap is `NaN` instead of a silent under-count.

Tide direction is derived from the real elapsed time between readings, so a gap
in the record does not manufacture a huge apparent rate. Movement slower than
`TIDE_SLACK_M_PER_HR` is reported as `slack` rather than a direction, which
keeps noise around high and low water from reading as a trend.

CO-OPS needs no token, and caps its 6-minute products at 31 days per request,
so a year is stitched from twelve chunks. It answers HTTP 200 with an `error`
body when a station lacks a product, so the status code alone proves nothing —
the script checks the body.

`test_pull_offline.py` exercises the rolling sums, tide direction and wind
parsing against fixtures, with no network.

### Water temperature and swell direction

Both already arrive in the NDBC `stdmet` feed — `WTMP` and `MWD` — so neither
needs a separate source. `MWD` is only published by buoys carrying a directional
wave sensor, so the run reports per buoy which of `WTMP`/`MWD`/`WVHT`/`DPD` are
actually populated rather than letting an all-empty column pass unnoticed.

Coastal water temperature is pulled separately from CO-OPS. A tide gauge at the
beach and a buoy 20-30 km offshore are measuring different water; for surf-zone
bacteria the near one is likelier to be the relevant predictor, so both are kept.

## Gridded hourly weather (Open-Meteo)

`pull_gridded_weather.py` fetches hourly precipitation and wind per site from
Open-Meteo's ERA5 archive. No API key, free for non-commercial use.

It exists because two station-data gaps cannot be closed by picking a better
station:

- **GHCND is daily**, so "rainfall in the 24 hours before this sample" is not
  computable from it. From an hourly source it is, and `rain_24h_mm` becomes 24
  clock hours rather than a calendar day.
- **No buoy in the matched set publishes wind**, and CO-OPS wind is absent at
  some gauges. A gridded product has no station gaps at all.

The trade-off is real: this is ERA5 reanalysis, a model reconstruction on a
roughly 9-25 km grid, not a rain gauge reading. For antecedent rainfall it is
usually a better input than a gauge 20 km away, but it is not an observation.
Both pulls write to `data/`, so compare them before choosing.

Requested variables are checked against the response — any that do not come
back are reported rather than silently becoming a column of NaN — and the run
prints the grid cell actually used, which is not the coordinate requested.

## Is the wind any good

`compare_wind_sources.py` answers this per site. Two things make wind harder
than rainfall:

**Three of the seven sites have no observed wind at all.** No buoy in the
matched set publishes `WSPD`, and Monterey serves no CO-OPS wind, so Walton
Lighthouse, Santa Cruz Wharf and Capitola have only the gridded source and
nothing to validate it against. The script names them separately rather than
quietly omitting them.

**Direction is circular.** An ordinary mean or correlation on degrees treats
359 and 1 as nearly opposite and averages them to 180 — due south for two
readings that are both due north. Everything here goes through u/v components:
the 6-minute CO-OPS readings are vector-averaged to hourly, and the comparison
reports mean absolute circular difference rather than a correlation.

`within45` — the share of blowing hours where the two sources agree to within
45 degrees, about one compass octant — is the column to judge on for a surf
zone, since onshore versus offshore is what matters. Calm hours are excluded,
because a calm has no direction to compare.

Both scalar and vector mean speed are kept. They differ whenever direction
swings within the hour, and the gap is itself informative.

### What the comparison found

                            site station   km  hours  obs_ms  grid_ms  spd_corr  dir_err  dir_med  within45
       San Elijo State Beach 9410230 16.9   8623    2.19     2.00     0.683     26.6     18.7        83
    Stinson Beach (nw view)  9414290 18.3   8638    3.69     3.10     0.457     46.4     38.5        58
      Sausalito - Galilee    9414290  6.5   8638    3.69     2.58     0.459     29.4     20.4        82
      Carpinteria State Bch  9411340 16.1   8640    2.42     1.45     0.532     33.7     23.5        75

**Direction is usable, speed is not.** `within45` is 83/82/75 at three of the
four sites — good enough to call onshore versus offshore. Speed correlations of
0.46–0.68 are too weak to trust an individual hour's value.

**The grid reads low at every site**, worst at Carpinteria (1.45 against 2.42,
a 0.60x ratio). That is a consistent bias, not noise — ERA5's footprint smooths
away the coastal sea breeze. Correctable with a per-site scale factor if you
ever need absolute speed.

**Exposure beats distance.** Stinson and Sausalito share station 9414290 —
identical `obs_ms` of 3.69 over the same 8638 hours — yet score 58 and 82.
Same observation, two grid cells. Sausalito sits inside the Bay beside the
station; Stinson is over the headlands on the open coast, where the station
simply does not see the same wind. Distance alone explains nothing: San Elijo
is 16.9 km out and scores 83.

`dir_med` is below `dir_err` at every site, so typical agreement is better than
the means suggest — a minority of badly wrong hours drags the mean up, which is
what light-wind hours do when direction is barely defined.

The consequence that matters: the three sites with no observed wind include
**Walton Lighthouse, the only rip-detection camera**. Its nearest analogues,
Santa Cruz Wharf and Capitola, are equally blind. Treat Walton's gridded
direction as plausible but unproven, and its speed as indicative only.

## Choosing between the gauge and the grid

`compare_precip_sources.py` aggregates the hourly grid to daily, aligns it
against the GHCND gauge, and reports totals, ratio, daily correlation and
wet-day agreement per site, alongside the elevation of each source.

The first run produced ratios on **both sides of 1** — Stinson 1.27, Walton
1.36, but Sausalito 0.84 and Carpinteria 0.76 — so this is not a simple model
wet bias. The elevation columns are there because the likelier explanation is
the gauge: Carpinteria's is **Juncal Dam**, inland and up in the Santa Ynez
range, and Sausalito's is **Muir Woods**, in a coastal redwood canyon. Both
collect orographic rain that a beach a few kilometres away never sees. Where a
gauge sits well above its grid cell, the gauge is the one measuring the wrong
place.

Daily correlation of 0.58–0.81 between a point gauge and a 9–25 km cell is
normal, not a failure.

**Raw wet/dry agreement is not reported.** The first version printed an
`agree_pct` that came out at 92% for all six sites; on this data an always-dry
predictor scores 83–90%, so the metric was nearly uninformative. It is replaced
by wet-day recall and precision, which describe the days that actually matter.

Three of the seven sites share one gauge (Soquel), and two share a grid cell
(Walton and Santa Cruz Wharf are ~1 km apart), so those rows are not
independent evidence.

## Why a precipitation station returns nothing

An empty PRCP response never means "it did not rain": GHCND records `PRCP = 0`
on dry days, so empty means the station published nothing.

Running `diagnose_precip.py` across the matched stations settled it — **they
were offline**, and `maxdate` correlates perfectly with whether data came back:

| station | last report | data returned |
|---|---|---|
| `USC00042150` Crescent City | 27,364 days ago (1951) | — |
| `US1CASZ0001` Santa Cruz | 6,240 days ago (2009) | none |
| `US1CASD0092` Solana Beach | 4,353 days ago (2014) | none |
| `USC00047916` Santa Cruz | 1,601 days ago (2022) | none |
| `USC00046027` Muir Woods | 31 days ago | 366 days |
| `USC00044422` Juncal Dam | 15 days ago | 366 days |
| `US1CAMR0030` Bolinas | 3 days ago | 366 days |
| `US1CASZ0028` Soquel | 2 days ago | 366 days |

Every station reporting within a month returned a full year; every station
silent for years returned nothing. `MAX_PRECIP_STALENESS_DAYS` in
`find_candidate_sites.py` is what stops them being matched.

Note that `datacoverage` was 1.0000 for three of the four dead stations. It is a
lifetime figure and carries no information about whether a station still runs.

**The script tests PRCP by requesting it**, not by reading metadata. An earlier
version inferred availability from CDO's `/datatypes?stationid=` listing and
reported "does not report PRCP" for all ten stations, including four that had
just returned a year of rainfall each. `/datatypes/PRCP?stationid=` is worse: it
ignores the station filter and returns the global 1781-to-present range for
every station. Neither is usable.

### Known coverage gaps

Established by running the pull, not assumed:

- **No buoy supplies wind.** `46236`, `46237` and `46274` all return `WDIR` and
  `WSPD` as entirely empty columns. Wind has to come from CO-OPS.
- **`46053` (Carpinteria) has sparse waves** — `MWD`/`WVHT` on ~9,560 of 26,459
  rows, so swell direction is missing about two thirds of the time there.
- **CO-OPS products vary by station even within a listed type.** Monterey
  (`9413450`) is in the `met` list but serves no wind; La Jolla, San Francisco
  and Santa Barbara serve wind but no water temperature. The pull tries the
  three nearest stations for each product rather than only the closest, and
  pulls each station/product combination once even when several sites share it.

## Reading the WebCOOS product catalogue

`explore_webcoos_products.py` lists every feed, product and service per camera
from the saved `webcoos_assets_raw.json`, so it costs nothing to re-run.

`pywebcoos` hardcodes `feed_name = 'raw-video-data'` in `get_products()`,
`get_inventory()` and `download()`, so a product under any other feed would be
invisible to it. Running the explorer settled that: **every product on all 86
cameras is under the `raw-video-data` feed slug**, so nothing is hidden behind
a second feed. Whether the library can actually reach them is a separate
question — see the label bug below.

Derived products that exist: `rip-detection-results` (8 cameras),
`object-detection-results` (14), `seal-detection-results` (1), plus
`annotated-image` (22).

**`rip-detection-results` in California exists on exactly one camera: Walton
Lighthouse, Santa Cruz** — 35,158 elements — and that camera is one of the seven
qualifying sites.

## Pulling the rip-detection product

    python pull_rip_detection.py --list
    python pull_rip_detection.py --probe
    python pull_rip_detection.py --pull --start 2025-06-01 --end 2025-09-01 --interval 30

`--probe` downloads six hours and reports what actually came down — file
extensions, sizes, the first 2 KB, and the JSON key structure if it parses.
Run it before `--pull`. The output format of this product has not been
observed yet, so nothing downstream assumes one: `build_table()` flattens JSON
or CSV payloads into a single table, and if the payloads turn out to be
imagery it writes only the element index (filename, timestamp, url) rather
than inventing columns. The index is what you need to join frames to the
observation tables either way.

### Why this does not go through pywebcoos

`API._get_camera_products` matches on the feed **label**:

    if feed['data']['common']['label'] == 'raw-video-data':

That is a label compared against a slug. WebCOOS's slug is `raw-video-data`;
if the label is anything else (`Raw Video Data`, say), the loop never matches,
the local `products` is never assigned, and the library raises
`UnboundLocalError` from inside `download()` — not a clear "not found". The
same label matching applies to product names, so `get_products()` returns
labels where the catalogue shows slugs, and `download()` wants the label.

`pull_rip_detection.py` therefore resolves feed, product and service by slug
and calls `/elements/` directly. `--via-pywebcoos` runs the library path
instead, so you can see for yourself which one works against the live API.
Both write to the same directory.

## Buoy station types

`NdbcApi().stations()` returns 1351 stations nationally: 709 land-based
`fixed`, 439 moored `buoy`, plus `dart`/`oilrig`/`tao`/`other`/`usv`. These are
different instruments — a moored buoy measures offshore waves and sea-surface
temperature, while a `fixed` station measures whatever is bolted to a pier.

`BUOY_TYPES` is set to `("buoy",)` so only moored buoys match. Without it, four
of the seven qualifying California sites matched a `fixed` station: three NOAA
tide gauges, and — for Capitola Wharf, an open-coast site — Azevedo Pond in the
Elkhorn Slough Reserve, 22.5 km inland, with no meteorology feed at all.

Filtering re-matches rather than drops: a site keeps whatever buoy is nearest
within `MAX_BUOY_DISTANCE_KM`.

`BUOY_REQUIRE_METEOROLOGY` additionally drops stations with no standard met
feed (NDBC's `met` flag). It is off by default, because some wave buoys report
waves without a met feed — `46237` San Francisco Bar is one.

`buoy_name` and `buoy_type` are in the output so a questionable match is
visible rather than hidden behind a station code.


| source | radius | rationale |
|---|---|---|
| buoy | 50 km | offshore conditions generalise over distance |
| precipitation | 30 km | rainfall is regional |
| water quality | 2 km | a bacteria reading only speaks for the water it came from |

Water quality previously shared the precipitation radius. It has its own
constant, `MAX_WQ_DISTANCE_KM`, because 30 km was letting a station anywhere in
the same town qualify a site.

## Scoring

`combined_score` is `has_all_four` minus a distance term, so qualifying sites
sort above non-qualifying ones and closer sites sort first within each group.

The distance term uses the **mean** of the measured distances rather than the
sum. A site with assumed water quality coverage has no `wq_distance_km`;
summing would treat that gap as either 0 km (flattering it) or a 999 km penalty
(sinking it below sites that qualify no better than it does). The mean scores
each site on what is actually known about it.

## Known scaling notes

- The CDO pull paginates 1000 stations at a time. CDO allows 5 req/sec and
  10,000/day; the loop sleeps between pages and backs off on HTTP 429. A
  California extent is a small fraction of the nationwide request count.
- The nationwide WQP pull is large and can take several minutes. Scoping
  `REGION` avoids it entirely for California.
