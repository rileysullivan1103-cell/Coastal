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
- **`analyze_drivers.py`** — ranks what tracks rip detections and bacteria
  counts across the CSVs in `data/`.
- **`test_analyze_offline.py`** — plants a known driver in synthetic data and
  checks the analysis recovers it, and that a pure clock variable is demoted.
- **`test_lint_offline.py`** — every bare name read in a module must be
  defined, imported, or a builtin there. Catches a constant deleted during a
  refactor, or a module used without being imported, on a branch the tests do
  not reach. It was ALL_CAPS-only until a missing `import glob` walked past it.
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
    python pull_rip_detection.py --inventory
    python pull_rip_detection.py --probe
    python pull_rip_detection.py --pull --start 2025-06-01 --end 2025-09-01
    python pull_rip_detection.py --camera Corolla --pull --match-observations

`--inventory` asks the service when it actually has data and downloads
nothing. Run it first. The catalogue's element count (35,158 for Walton) says
how much data exists but never *when* — the first probe against "the last six
hours" came back with zero elements, which looks like a broken product and is
really just an empty window. The script now consults the inventory before
choosing a range, and defaults both `--probe` and `--pull` to the end of the
covered period rather than to now.

`--probe` downloads six hours and reports what actually came down — file
extensions, sizes, the first 2 KB, and the JSON key structure if it parses.
Run it before `--pull`. The output format of this product has not been
observed yet, so nothing downstream assumes one: `build_table()` flattens JSON
or CSV payloads into a single table, and if the payloads turn out to be
imagery it writes only the element index (filename, timestamp, url) rather
than inventing columns. The index is what you need to join frames to the
observation tables either way.

### The other seven cameras

`rip-detection-results` exists on **eight cameras nationally**, and for a long
while this project used one. `--camera` always took a name, so nothing stopped
the others being pulled except knowing they were there:

    python pull_rip_detection.py --list                 the roster
    python pull_rip_detection.py --list --inventory     ...and when each has data
    python pull_rip_detection.py --all-rip --inventory  one request per camera
    python pull_rip_detection.py --all-rip --pull --match-observations

**The catalogue was being read one page deep.** `load_assets()` GET the
`/assets/` endpoint once and kept whatever came back, but WebCOOS paginates it,
so any camera past the first page was invisible — and asking for one produced
`No camera matching ...`, which reads as *that camera has no rip detection* and
is a different statement. It now uses `scan_cameras.fetch_assets()`, which
follows the pagination and caches the complete list, rather than keeping a
second and shorter copy of the catalogue.

**Judge a camera on its populated bins, not its element count.** Corolla
catalogues 10,239 elements against Walton's 35,158 and looks like a third the
size. Its inventory says those elements sit in **46 populated bins of 818**,
last data 50 days old — a few dense weeks, not a third of a record. The element
count says how much data exists and never how much of the record it covers, and
the joinable hours follow the coverage. `--list --inventory` prints the
populated share beside the count for exactly this reason.

**`--all-rip` isolates each camera.** A missing stills product, a service the
token cannot reach or a dropped connection ends that camera and nothing else,
and the run closes with a summary line per camera. `SystemExit` is caught
alongside ordinary exceptions, because the single-camera helpers exit rather
than raise, and an uncaught one would abandon the sweep with no record of what
had already succeeded.

### --match-observations is scoped to one camera

`observation_window()` intersects the spans of the observation CSVs, because a
rip hour is only usable where the conditions that would explain it also exist.
It used to intersect **every** file in `data/`, which was fine while the
project held one region and silently fatal the moment it held two: Cala
Millor's weather ends 2024-09-28 and Corolla's begins 2025-09-08, so with both
on disk the intersection was empty and `--match-observations` skipped all eight
cameras — Walton included, whose own observations were perfectly good.

It now matches only the files written for the camera being pulled, found by
the stem its writer chose. Note that is **not** `slugify()`:
`pull_site_observations.py` and `pull_gridded_weather.py` build filenames with
underscores cut at 48 characters (`gridded_Beachfront_from_Hampton_Inn__Corolla__NC.csv`)
while this module uses dashes for service and directory names. Two slug
conventions in one pipeline is a trap, so `obs_slug()` exists to name the
difference rather than leave it to be rediscovered.

Tide and buoy files are named after a **station**, not a camera, and are shared
between sites — `tide_8651370.csv` serves both Corolla cameras. Which one
belongs to a camera is a distance question `analyze_drivers.py` answers at join
time, so they are listed but do not constrain the window.

### The denominator is camera uptime, not detector uptime

`--coverage` answers "was the camera looking". It does not answer "was the
detector running", and the rip feed cannot: it publishes an element on a
detection, so silence means *no rip* or *no detector* and nothing separates
them. Two different mistakes follow, and the national sweep surfaced both.

**Outside the rip record, the hours are not zeros.** `--coverage` defaults to
the last year of the **stills** inventory, which is longer than the rip record
at every camera checked — Virginia Beach publishes stills from 2026-02-03 and
rips from 2026-04-25. Counting that gap would manufacture about 1,200 quiet
hours from eleven weeks when nothing was watching for rips, all of them in one
season, so the invented zeros land on the month control too. `apply_coverage()`
now clips to the rip record and says how many hours it dropped.

The clip is by **day**, not by hour. The feed fires on a detection, so the last
detection is not the end of the detector's shift; an hour-level clip would
discard the genuinely quiet hours after the final firing of the last day, which
are exactly the observed zeros the coverage file exists to supply.

**Inside the record, a silent day is ambiguous and stays ambiguous.** At
Corolla the rip feed has data on **89 days of 906** while the camera was up for
4,627 hours, so nearly the whole denominator is days that published nothing.
Resolving that silently either way is wrong: keeping them makes
`detection_rate` mostly a measure of detector uptime, and dropping them biases
the rate up by deleting calm days. The run reports the share, warns when it is
over half, and `--detector-days-only` drops them if you decide that is the
question you want:

    python analyze_drivers.py --target rip --site <slug> --detector-days-only

Walton is the one camera where this barely matters — 713 populated rip bins
against 1,093 stills bins. It is the reason the problem went unnoticed.

### Making a new camera analysable

The rip feed on its own correlates against nothing. For a camera outside the
California seven, the conditions come from `pull_site_observations.py`, which
works from a coordinate rather than a site list and splits its sources by what
exists there:

    python scan_cameras.py --rip-only --weather grid        # what each one has
    python pull_site_observations.py --camera "Corolla"     # ERA5, marine, buoy, tide
    python pull_rip_detection.py --camera <slug> --inventory
    python pull_rip_detection.py --camera <slug> --coverage
    python pull_rip_detection.py --camera <slug> --pull --match-observations
    python analyze_drivers.py --target rip --site <slug>

`--coverage` and `--pull` need no dates: with neither `--start` nor `--end` the
run takes the last year the inventory says the product actually covers, which
is what you want and is not what "now" would give you.

**Name the camera by slug.** `--camera` takes a label, a slug, or a unique
substring, and a substring is often not unique — Corolla has *two* rip cameras,
`beachfront-from-hampton-inn-corolla-nc` and
`beachfront-from-sailfish-street-beach-access-corolla-nc`, so `--camera Corolla`
cannot mean anything on its own. An ambiguous name now prints the slugs rather
than the labels, so the fix is a paste. The slug is also what
`analyze_drivers.py --site` takes and what the output directories are named
after: one identifier across the pipeline.

`analyze_drivers.py` globs `data/**/rip_*_hourly.csv`, so a newly pulled camera
appears in the run without any list to edit. Do not skip `--coverage`: without
the stills denominator every hour with no detection is *unknown* rather than an
observed zero, and `detection_rate` collapses to the constant 1.0.

Note what does **not** travel east. CDIP MOP — the 1.45 km nearshore model that
produced the strongest wave result in this project — is California only. A
camera in North Carolina or Florida gets Open-Meteo Marine and whatever NDBC
buoy is within 50 km, which is the distant-wave problem MOP was adopted to fix.
Any cross-camera wave comparison has to carry that asymmetry explicitly.

### What the product actually contains

Probed against Walton on 2026-08-31. The elements are **`.jsonl`**, about 800
bytes each, one JSON object per line:

```json
{"time":"2026-08-31T14:05:10Z",
 "annotated_image_url":"http://stage-webcoos-rip-detector-api.srv.axds.co/outputs/...jpg",
 "classification_result":{
   "classification_model_name":"ripdetect_walton",
   "classification_model_version":"yolov8x_1.1",
   "detected":true,"detection_count":1,
   "classification_scores":[{"rip_current":0.7011650204658508}],
   "classification_bboxes":[[{"x":1853,"y":974},{"x":2255,"y":1123}]]},
 "original_image_reference":"walton_lighthouse-2026-08-31-140451Z.jpg"}
```

So it is a YOLOv8 detector's output, not a hand-labelled record: a confidence
score and a pixel bounding box per detection. `classification_scores` is a list
of single-key dicts, so the class name (`rip_current`) is data rather than
schema — the parser reads it generically and a second class would need no code
change.

Two CSVs come out of a pull:

- `rip_<camera>.csv` — one row per frame: `detected`, `detection_count`,
  `score_max`, `score_mean`, `score_classes`, `bbox_count`, `bbox_area_max`,
  `bbox_x`, `bbox_y`, model name and version, image references.
- `rip_<camera>_hourly.csv` — collapsed to hourly, which is the resolution
  everything else in the pipeline uses: `frames`, `frames_with_detection`,
  `detections`, `detection_rate`, `score_max`, `score_mean`, `bbox_area_max`.

An hour with frames but no detection is kept as an observed zero. Dropping it
would turn "the camera looked and saw nothing" into "the camera was not
looking", and those mean opposite things when you regress rips on rainfall.

Both `time` (the model's stamp) and `element_time` are kept — they differ by
seconds, and the image filename carries a third, slightly earlier stamp
(`140451Z` against `14:05:10`), which is the frame capture.

### Coverage

    data runs 2024-05-31 21:28 to 2026-08-31 14:41 UTC
    35,158 elements across 700 populated bins of 823

The product is live — 27 months, still publishing. 123 of 823 daily bins are
empty, so roughly 15% of days have nothing at all: absence of a detection is
not the same as absence of a rip, and the frame count per hour is what
distinguishes them.

At ~50 elements a day the frames are roughly 20 minutes apart across daylight
hours, not continuous. `--interval 30` would discard a real fraction of them,
so the full pull takes everything and thins later.

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

## Widening the funnel: which requirement is actually shut

    python scan_cameras.py                    # rainfall must come from a gauge
    python scan_cameras.py --weather grid     # rainfall comes from Open-Meteo
    python scan_cameras.py --rip-only --weather grid

A camera qualifies on four sources: a buoy within 50 km, a tide gauge within
50 km, a bacteria station within **2 km**, and precipitation within 30 km. The
run used to print only how many cleared all four, which says nothing about
*which* requirement is doing the excluding — so relaxing one was guesswork.

Every run now prints both totals and a per-requirement cost:

```
N/86 qualify with a GHCND rain gauge
M/86 qualify with gridded precipitation (+K)

what each requirement is costing ON ITS OWN
(cameras failing that one and nothing else — the rest fail several)
  water quality   ...
  precipitation   ...
```

`gate_cost()` counts only cameras failing **exactly one** requirement. A camera
with no buoy *and* no gauge is not evidence against either: relaxing one would
not qualify it, and charging it to both would overstate what the change buys.

### Why the grid may be dropped in as a gate, and why it is not a lower bar

`--weather grid` removes precipitation from the gate entirely, because
Open-Meteo's ERA5 archive answers on a latitude and longitude — there is no
station to be absent, stale, or 30 km inland. `pull_gridded_weather.py` already
supplies it for every analysed site; the gate was requiring a *second*,
redundant rainfall source and failing cameras that have perfectly good rainfall.

It is also not obviously the worse number. `compare_precip_sources.py` found the
gauge measuring the wrong place at two of the seven California sites —
Carpinteria's is **Juncal Dam**, inland and up in the Santa Ynez range, and
Sausalito's is **Muir Woods**, in a coastal redwood canyon — both collecting
orographic rain that a beach a few kilometres away never sees. Where a gauge
sits well above its grid cell, the gauge is the one in the wrong place.

The nearest gauge is still recorded in every row either way, so
`compare_precip_sources.py` keeps working. Only the gate changes.

### Why not AccuWeather or the Weather Channel

Both are commercial APIs behind a key, with short free retention and terms that
restrict storing or redistributing the series. Neither publishes a multi-decade
hourly archive on an arbitrary coordinate. Open-Meteo does, needs no key, and is
already wired in — for hourly rainfall at a point, a commercial forecast API
would be a downgrade paid for with a signup.

### What the grid does not rescue

Water quality. A bacteria station has to be **at** the beach the camera
watches, so no gridded product substitutes for it, and nationally it is the
requirement that excludes the most cameras — 68 of 86 have one within 2 km,
and the four-source count is 58. The rain gauge is worth perhaps ten cameras;
the 2 km bacteria radius is worth eighteen. In California specifically the
gauge is worth **nothing**: all three non-qualifying cameras (Crescent City,
Humboldt Bay/Arcata, Point Reyes) fail on water quality and have a rain gauge
already.

### Feeding the wider list forward

`pull_gridded_weather.py` reads `candidate_sites_ranked.csv` (the California
run) by default and `camera_candidates.csv` (the national scan) with `--sites`:

    python pull_gridded_weather.py --sites camera_candidates.csv

`load_sites()` accepts either column convention — `camera_name`/`has_all_four`
or `camera`/`qualifies` — and a list carrying neither qualification column is
used whole, on the grounds that it is a hand-picked set rather than a scan.
Running `--weather grid` and then not pulling the grid would qualify cameras on
a promise nothing kept.

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

## Ranking the drivers

    python analyze_drivers.py
    python analyze_drivers.py --target rip
    python analyze_drivers.py --target wq

Spearman rank correlation throughout. None of these variables is close to
normal — rainfall is zero-inflated, bacteria counts are log-distributed with
censored non-detects, detection rate is a bounded proportion — and rank
correlation is unbothered by all three. The p-value uses a Fisher z
approximation, so scipy is not required. Nothing under `MIN_N = 30` paired
observations is reported at all.

Two traps the output is built to avoid:

**Diurnal confounding.** The rip detector only sees daylight, and tide, sea
breeze and air temperature all cycle daily, so a raw correlation can be pure
clock. Every rip correlation is reported twice: `rho` raw, and `rho_ctrl`
after subtracting each hour-of-day's own mean from both sides. In the test
fixture a deliberately planted clock variable scores **rho 0.44 and rho_ctrl
0.0001** — it looks like a strong driver until controlled. Judge on
`rho_ctrl`.

**Nested predictors.** `rain_24h`, `rain_48h` and `rain_72h` are sums of one
another and will always rank together. They are one family, not three
findings. The standardized regression prints a condition number and warns
above 30, where individual coefficients stop being interpretable.

### What it cannot tell you

`detection_rate` is a YOLOv8 model's confidence, not a verified rip. Anything
that drives the *detector* — glare, swell texture, contrast, water colour — is
indistinguishable here from something that drives the *rip*. Two WebCOOS
cameras carry the product and RipAID's eight are a different instrument
entirely, so nothing here generalizes to a beach that was not analysed, and
Walton has no observed wind, so every wind column there is unvalidated ERA5.

Bacteria samples are not a random sample of conditions: agencies sample in
swim season, on a schedule, and sometimes after known spills. Non-detects are
dropped rather than imputed, which thins the low end. `SampleDate` carries no
time of day, so tide and wind are daily means — not the value at the moment of
sampling.

The shore normal used to turn wind and swell direction into onshore
components is an **assumption** (`SHORE_NORMAL_DEG`) everywhere CDIP MOP does
not reach. Where a MOP file exists the published `metaShoreNormal` wins, and at
Walton that is **206.0 degrees against the 180 the constant assumed for all
three Santa Cruz sites** — so every onshore figure printed at Santa Cruz before
the MOP pull was computed off a 26-degree error. Virginia Beach, Cala Millor
and Son Bou are still on bearings read off a map, and every onshore/offshore
result at those sites is conditional on them.

## What the rip feed does and does not contain

The June-August 2025 pull returned 2,795 elements and **every one of them has
`detected: true`**. The feed publishes an element when the detector fires, not
one per frame examined, so it contains no negatives at all.

The consequence is not a tuning problem, it is a limit on the question. Absence
of a file means either "no rip" or "no image", and nothing in the feed
separates them. Presence/absence cannot be modelled from this source alone.
`detection_rate` is therefore constant at 1.0 and is reported as constant
rather than correlated; what remains analysable is how OFTEN the detector
fires (`detections` per hour) and how confident it is (`score_max`,
`bbox_area_max`).

### Getting the zeros honestly

    python pull_rip_detection.py --coverage --start 2025-06-01 --end 2025-09-01

`--coverage` enumerates the stills product and writes an hourly count of
images captured. It **downloads nothing**: element listing already returns a
timestamp per element, which is all a denominator needs, so hundreds of
thousands of JPEGs never move. Pagination is the only cost — and Walton's
stills service holds 652,697 elements, so that cost is real.

It is **resumable**. Work is committed one day at a time to a
`*_progress.csv` sidecar, so an interrupted run continues where it stopped
instead of restarting. A first attempt died on a read timeout at page 443 of
roughly 1,300 and lost everything; requests now retry with exponential backoff
(5 attempts, on timeouts, connection errors and 429/5xx), and a day is the
most any single failure can cost. Re-run the same command to continue.

`analyze_drivers.py` picks the coverage file up automatically. With it, an
hour holding images and no detection becomes an **observed zero**, an hour
with no images stays absent, and `detection_rate` becomes detections per image
examined rather than the constant 1.0.

Do NOT skip this and treat every missing hour as a zero. 123 of 823 daily bins
are empty, and those are camera or pipeline outages, not days with no rips.
Scoring an outage as "no rip" biases the target toward zero exactly when the
camera was down — and cameras go down in bad weather, which is correlated with
the very drivers being tested. That turns a data gap into a fake negative
result. Without a coverage file the analysis says so and leaves those hours
UNKNOWN.

## Seasonal confounding in the water quality analysis

California rainfall is concentrated in winter and beach sampling is
concentrated in the dry swim season. Rain and bacteria can therefore correlate
in either direction purely through the calendar, with no mechanism between
them. Every water quality correlation is reported twice, `rho` raw and
`rho_ctrl` after removing per-month means from both sides -- the same guard the
rip analysis applies for hour-of-day. Judge on `rho_ctrl`.

## Why the regression withholds coefficients

`precip_mm` and `rain_24h_mm` are the same series whenever
`RAIN_INCLUDE_SAME_DAY` is on: a one-day rolling sum IS the daily total. Left
in the design matrix they make it exactly singular, and `lstsq` responds with a
minimum-norm solution that splits one variable's effect arbitrarily across the
copies -- which is how a first run reported condition numbers around 1e17 and
sign-flipped rainfall coefficients that looked like findings.

Exact and near-duplicate columns are now dropped before fitting, with a line
saying which. Above `MAX_REPORTABLE_CONDITION` the coefficients are withheld
entirely rather than printed, because a number nobody can trust is worse than
no number.

## Per-site, not pooled

Water quality is reported three ways per analyte: pooled, pooled with each
site's own mean removed, and one table per site.

The middle one is the guard that matters. Sites differ in both how dirty they
are and which tide gauge serves them, so a between-site difference arrives
looking exactly like a within-site relationship. The test fixture plants
precisely this: two beaches, one dirtier and sitting at a higher-water gauge,
with no relationship inside either. Pooled it scores **rho 0.73**; within site
it collapses to **-0.10**, and neither beach shows anything alone. A predictor
that survives raw but not within-site was telling you which beach the sample
came from.

### The tide check

`TIDE LEVEL vs BACTERIA, PER SITE` reports `level_m` against each analyte for
each site separately, with the site's setting alongside — enclosed bay, open
embayment or open coast, classified by hand in `SITE_SETTING`.

That column is a prior, not decoration. Inside an enclosed bay, water level
plausibly tracks flushing and the arrival of bay water at the shoreline. On
open coast it is mostly the astronomical tide, so a strong effect there earns
suspicion rather than confirmation. If the pooled tide effect turns out to sit
only on open-coast sites, the mechanism does not fit and something else is
doing the work.

Sites below `MIN_N` are listed with their sample count and marked
`underpowered`, with the correlation withheld rather than the row hidden — for
a single pre-specified predictor, knowing a site could not be judged beats a
silent omission.

## Matching the rip range to the observations

    python pull_rip_detection.py --pull --match-observations
    python pull_rip_detection.py --coverage --match-observations

`--match-observations` reads the time span of every `gridded_*`, `buoy_*` and
`tide_*` CSV already in `data/`, takes their **intersection**, clips it to what
the rip product's inventory actually holds, and pulls that. A rip hour is only
usable where the conditions meant to explain it also exist, so the union would
be the wrong answer.

This exists because the first run got it wrong: the rip pull covered June-August
2025 while `pull_observations.py` had fetched August 2025 to August 2026. The
join landed on 39 hours of gridded weather and **one** hour of buoy data, and
the correlation tables printed anyway, looking like results.

`analyze_drivers.py` now refuses to be quiet about it. Each source prints the
percentage of rip hours it covers, and below 20% the run prints
`*** THE WINDOWS DO NOT LINE UP ***` with each source's actual span and the
command to fix it.

## Season is the second confound

The rip window now spans a full year, which makes the calendar a confound of
the same shape that time of day already was. Water temperature, air
temperature and the whole wave climate cycle annually, so a raw correlation
can be the season and nothing else.

Each rip correlation therefore prints three ways:

    rho        raw
    rho_hr     hour-of-day means removed from both sides
    rho_hrmo   hour-of-day AND month means removed

Tables rank by `rho_hrmo`. In the test fixture a variable that is purely
seasonal scores **0.94 raw and 0.38 after the month control**, while a genuine
driver goes the other way, **0.34 raw to 0.90 controlled** — the calendar was
hiding it. Judge on `rho_hrmo`.

Two limits worth knowing.

Demeaning by month removes the between-month signal but not the trend inside
each month, so a perfectly seasonal driver drops a long way without reaching
zero.

More importantly, **this control cannot distinguish "season caused it" from
"the cause only varies with season"**. A predictor that barely moves within a
month has almost nothing left after demeaning, so it collapses whether or not
it is causal. The fixture shows a genuinely causal driver going from **0.83
raw to 0.09 controlled**, while a fast-varying driver holds at 0.96 under the
identical control. So a collapse means *indistinguishable from season here* —
not *not a cause*. `rho_mo` is printed beside `rho_hrmo` to show which control
did the damage.

## How much of the target is just the calendar

`HOW MUCH OF EACH TARGET IS JUST THE CALENDAR` reports, per target, the share
of its variance that hour-of-day, month, and the two together account for on
their own — eta squared on the control key, no predictors involved.

This is the number to read before interpreting any collapse. If month explains
little of the target, then a predictor collapsing under the month control is
telling you about the predictor (it has no within-month variation to test),
not that season drives the outcome. If month explains a lot, season is a real
competitor to every driver in the table.

Without it the two cases look identical in the correlation columns.

## Is the camera one geometric record, or several

    python check_camera_geometry.py --camera <slug> --detections   # read-only
    python check_camera_geometry.py --camera <slug> --sample        # downloads

A pan, zoom, remount or housing shift breaks the pixel-to-ground mapping. A
shoreline trend computed across a camera move looks exactly like erosion, and
`bbox_area_max` pooled across one is comparing two different scales. This is a
gate on any coastal-change work, and it is worth running before the pixel
metric is trusted rather than after.

**The stills are not on disk.** `--coverage` enumerates element timestamps and
downloads nothing — that is what lets it build a denominator over 650,000
frames without moving a byte — so there is no local imagery to register. Any
imagery pass has to fetch frames.

`--detections` is the part that is genuinely read-only, and it answers what the
rip table can answer:

- **detector version.** `model_name` / `model_version` runs, with dates and row
  counts. A retrain is a discontinuity in `score_max` and `bbox_area_max` even
  when the camera never moved, and the pixel metrics are not comparable across
  one.
- **frame extent.** Boxes are in native image pixels, so the largest coordinate
  ever seen is a floor on the frame size. A resolution change shows up here and
  nowhere else in what has already been downloaded.
- **where detections sit.** Reported, but the weakest of the three — detections
  are in the surf zone and the surf moves on its own.

It cannot see a pan or a remount that left the detector and the resolution
alone, and says so rather than reading as a clean bill.

`--sample` takes one frame per `--every` days (7 by default) at a fixed UTC
hour, caches them under `data/geometry/<slug>/`, and registers each against the
first readable frame.

### Phase correlation, not ORB

Descriptor matching answers "where did this corner go" per frame and then needs
outlier rejection to survive fog, glare and a gull on the railing. Phase
correlation on a fixed patch answers "how far did this patch move" directly,
carries its own confidence in the height of the correlation peak, and needs no
feature library — the whole thing is an FFT and a parabolic peak refinement.
Frames whose peak is flat (fog, night, heavy rain) are reported and dropped
rather than averaged in.

Sign convention is the easy mistake: the correlation peak gives the shift that
maps the frame back onto the reference, which is the **negative** of the
displacement. It is negated before returning, so a feature that slid 5 px right
reads `dx = +5`.

### The banner at the top of the frame

Walton's stills carry a composited strip across the top — "Walton Lighthouse
Cam by UCSC" on the left, a running timestamp on the right. It is drawn after
capture, so it **cannot move when the camera does**, and it is the single most
attractive thing in the frame to a picker that rewards sharpness and stillness:
letter edges are the hardest edges anywhere in the image. All four patches
landed on it and reported 0.03 px of agreement, which is a measurement of
nothing.

An absolute floor on temporal variation does not catch it. The timestamp digits
genuinely change every frame, and JPEG ringing around the static letters moves
by several grey levels, so the strip measured 4.6–9.1 and read as live scene.
The test that works is **relative and anchored to the frame edges**: a banner is
a contiguous strip touching the top or bottom that varies far less than the
scene rows around it. Flooding inward from an edge lets the ratio be generous
without risk — the flood stops at the first row that behaves like scene — and
`MAX_EDGE_FLOOD` caps how far it can eat.

If the automatic test does not fire — and at Walton it did not, through two
rounds of increasingly careful detection — use the margin instead:

    python check_camera_geometry.py --camera <slug> --sample --top-margin 100

A camera's overlay is a fixed, known property of that camera. Stating its
height directly beats another inference that might also miss it. Every run
prints the row-variation profile at the top edge against the frame's typical
row, so you can see whether the banner is quiet enough for the automatic test
to have had a chance.

### Agreement between features is the actual test

One patch moving is a sign that blew over. Every patch moving by the same
vector on the same date is the camera. The run reports both — the median offset
across features, and the spread between them — and splits epochs on the median.
A large offset with a large spread is a feature problem; a large offset with a
near-zero spread is a camera move.

Features are auto-proposed by preferring patches that are sharp in space
(something to correlate against; flat sky aligns equally well everywhere) and
still in time (structure rather than weather), restricted to the top
`--land-fraction` of the frame, because the beach and the water move for real
reasons. **Eyeball them before trusting a result** — an auto-picked patch on a
moored boat tracks the boat. `--roi name:x,y,w,h` overrides.

### Dating a step

A step is a level change that persists, not a spike, so the test compares the
median of the `PERSIST` samples before a date against the median of the
`PERSIST` after. That test is deliberately blunt and fires on every index whose
window straddles the change, so one event produces a run of hits; the event is
then dated by the largest single-sample jump inside the run. Without that
second step the reported date is the last frame *before* the move, because a
window centred there already sees the new level in its tail.

Epochs are reported in **stills**, not in sampled frames, by joining the
coverage file. A weekly sample says nothing about how much imagery an epoch
holds, and a short epoch can be the dense one — that is the number that decides
whether an epoch is worth salvaging.

Nothing is corrected or re-registered. This pass only answers whether the
archive is one record or several.

## Rip detection at Walton, once the waves came from the beach

This section used to say "no conclusions yet", because the observation window
barely overlapped the rip window — `n=39` on the gridded join and `n=1` on the
buoy. That is no longer the constraint. A year of rip output, a stills-feed
denominator and a nearshore wave model give **4,843 hours analysed** (2,550
with detections, 2,295 observed zeros), of which 2,548 carry the gridded and
MOP columns and 1,094 carry the buoy.

Ranked on `rho_hrmo` — hour-of-day and month both removed:

| target | leader | rho_hrmo | `mop_wave_height` | buoy `WVHT` | calendar share |
|---|---|---|---|---|---|
| `bbox_area_max` | `mop_wave_height` | **0.3037** | **0.3037** (1st) | 0.1508 | 0.070 |
| `score_max` | `mop_wave_height` | **0.1183** | **0.1183** (1st) | 0.0333 | 0.078 |
| `detections` | `level_m` | 0.1852 | 0.1451 (5th) | 0.0627 | 0.242 |
| `detection_rate` | `precipitation` | 0.1486 | 0.0774 (7th) | 0.0387 | 0.279 |

All at `n=2548` for the MOP and gridded columns, `n=1094` for the buoy;
everything above `p=0.001` except `WVHT` on `detection_rate` (`p=0.20`).

**The nearshore model beats the buoy on all four targets**, by 2x to 4x. Same
physical quantity, same camera, same hours where they overlap: what changed is
1.45 km instead of 22.9 km, 15 m of water instead of 133 m, and 100% coverage
instead of 43%. This is the single clearest thing the project has shown about
where a wave number should be measured — and it is the reason the earlier
Walton wave null was reported as a suspected instrument failure rather than as
a finding.

**But it splits by target, and the split is not flattering.** The two targets
`mop_wave_height` leads are the ones that describe *how big* a detection is,
and both are nearly calendar-free (7-8% of variance in the control key). The
two it does not lead describe *how often* the detector fires, are the most
calendar-loaded targets here (24% and 28%), and are led by rain and tide. So
the honest sentence is: bigger waves at the beach go with bigger, more
confident detection boxes; they do not much change how often a box appears.

### What that does not settle

`bbox_area_max` is the area in pixels of a YOLOv8 box. A bigger wave breaking
closer in produces a wider, brighter foam signature, which produces a bigger
box, whether or not the water inside it is a rip. The cleanest correlation in
the rip half of this project is therefore also the one most exposed to the
detector-versus-rip ambiguity — a stronger result on a weaker target.

The rain columns did not go away either: `rain_24h_mm` is 0.2234 on
`bbox_area_max`, second behind the waves, and `precipitation` still leads
`detection_rate`. Rain and nearshore wave height are not the same weather but
they are not independent in a Californian winter, and nothing here separates a
detector responding to swell texture from one responding to runoff plumes.

One number in the output is smaller than it looks: the standardized regression
reports `mop_wave_height` at beta **0.2345** on `bbox_area_max`, the largest
coefficient in that fit, but the fit is `n=1094` — dropping any row without
buoy data pins the joint model to 43% of the hours the correlation used. The
correlation column is the one with 2,548 hours behind it.

### Virginia Beach still disagrees

At Virginia Beach, modelled wave height is the top driver of `detection_rate`
at 0.40. At Walton, with a wave source that is now closer and more complete
than Virginia Beach's, the same target gives `mop_wave_height` 0.0774. Getting
Walton a better wave record did not make the two cameras agree on that target.
The instrument explanation for the old disagreement was half right: it was
hiding a real wave signal at Walton, but that signal is in box size, not in
detection frequency, so the two cameras genuinely behave differently and one
of them is not simply mismeasured.

## Rip-current casualties, and why two beaches on one coast disagree

Every rip target elsewhere in this project is our own instrument: `detection_rate`
is a YOLOv8 model's output, RipAID's boxes are a curator's pen. NOAA's Storm
Events database records something else — a day on which a rip current killed or
injured somebody — and it was written by people who had never seen our cameras.
`pull_storm_events.py` writes it as a daily series, and `analyze_storm.py` joins
it to conditions.

Three properties of the record decide how it can be read.

**It has real zeros.** RipAID's no-rip frames were deleted by a curator, so its
zeros are not zeros. Here a day with no logged event is a genuine observed zero
for "a rip hurt somebody today", and the daily table writes those days out
explicitly rather than leaving gaps.

**It samples attendance, not the ocean.** An event is logged when a person is in
the water. In New Hanover County, Saturday holds 31 of 72 events against
Thursday's 2 — no ocean process distinguishes Saturday from Sunday, let alone
from Thursday. Every table is therefore reported with each predictor ranked
within its own month and weekday, so 50 (or rho 0) is the null whatever the
season and the weekend are doing.

**It is not comparable between zones.** New Hanover logs 33 casualties across 72
events; Florida's Coastal Bay logs 70 across 63. That is the local forecast
office's filing practice, not the water. Nothing here compares one zone's event
*count* to another's — only the sign and size of each zone's own drivers.

### The zones are named twice

NWS re-cut its coastal zones during this period, so one stretch of coast appears
under two names in two eras: `NEW HANOVER` 2000-2010 filed against the county,
`COASTAL NEW HANOVER` 2012-2026 filed against the forecast zone. Escambia has
three names in three eras. Analysed apart, each name carries years of zeros that
are a filing change rather than a quiet ocean. `--zone` matches on a substring
and pools them, printing each name's span and saying whether the spans are
disjoint (a rename being repaired) or overlapping (two places being conflated).
`--exclude` drops the ones that are not beaches — `INLAND NEW HANOVER` is caught
by the same substring that correctly pools the other two.

### Two zones on the same coastline, opposite signs

Both are Atlantic beaches, Wrightsville Beach at 34.19N and Jupiter Inlet at
26.94N — about 800 km apart on one shoreline, not two oceans. Both zones have
enough positives to model (72 and 79 event days over 9,736) and both have
complete conditions from `era5_ocean`. Season-only, ranked within month and
weekday:

| predictor | New Hanover, NC | Palm Beach, FL | do they agree? |
|---|---|---|---|
| `wind_speed_10m` | **-0.042** (p=0.002) | **+0.069** (p<0.0001) | opposite, z=6.5, p=8e-11 |
| `wave_height` | -0.008 (ns) | **+0.074** (p<0.0001) | opposite, z=4.8, p=2e-06 |
| `wave_height_max` | -0.024 (ns) | **+0.068** (p<0.0001) | opposite, z=5.5, p=5e-08 |
| `wave_period` | **+0.092** (p<0.0001) | +0.005 (ns) | NC only, z=5.1, p=3e-07 |

In percentile terms — where the typical casualty day sat among ordinary days of
the same month and weekday — the two signatures are different phenomena:

- **New Hanover:** wave period at the 76th percentile, wind at the 32nd. Long
  swell arriving under light wind. A clean, calm-looking day.
- **Palm Beach:** wind at the 77th, wave height at the 75th, gusts at the 74th,
  wave period at the 51st. A windy day with a short, steep sea.

The seasons differ to match. New Hanover's events peak in July and August and
there are none at all from November to March. Palm Beach's peak in **April and
May** (17 and 22 of 79) and it logs events in every month of the year — a
spring cold-front regime, not a summer swell regime.

### What this does and does not establish

It does not validate the cameras. Storm Events carries no coordinates, its zone
is a stretch of county coast, and the imagery starts in 2023 while the casualty
record starts in 2000. These are two independent regressions on the same coast,
compared — never merged.

**Neither zone has a strong driver.** The largest controlled rho in either is
0.09. The percentile shifts are real (the null sd on 72-79 event days is about
6 points, and the shifts run 20-27), but the ocean explains very little of when
somebody gets hurt, which is unsurprising for an outcome that requires a person
to be present.

**The replication failed, and that is the result.** New Hanover's signature does
not appear at Palm Beach; the wind term is significantly *reversed*. So there is
no general rule here of the form "long-period swell is the dangerous case" — and
equally, the earlier suspicion that our detector might be tracking whitewater
rather than hazard is not supported, because at Palm Beach casualty days really
are the bigger-wave, windier days the detector responds to.

What survives is the same shape as the water-quality result: rain against
enterococcus is +0.47 at Santa Cruz Wharf and -0.32 at Carpinteria; wind against
a rip casualty is -0.04 in New Hanover and +0.07 in Palm Beach. Two entirely
independent outcome types, two independent sampling programmes, and in both the
sign flips between beaches.

The rip pair sharpens it rather than weakening it. Santa Cruz and Carpinteria
are both Californian; New Hanover and Palm Beach are both Atlantic. Nothing here
is a Pacific-versus-Atlantic contrast, which one would expect to differ. The
sign reverses **within a single coastline**, between beaches you would model the
same way. **A driver fitted at one beach should not be deployed at another.**

### Zones with a camera in them

`camera_candidates.csv` contains cameras inside four zones with enough events to
model: New Hanover (74 events, Wrightsville Beach and Carolina Beach), Palm Beach
(87, Jupiter Inlet), Bay (63, Panama City Beach) and Horry (22, Cherry Grove
Pier — below the floor). None of our original cameras qualify: Virginia Beach has
**6** casualty days in seventeen years, Corolla 5, Santa Cruz 4, Carpinteria 1.
That is why this analysis moved to other beaches rather than checking the ones
the project started with.

### Two data traps found here

**The default wave model is a forecast model.** The first Wrightsville pull wrote
233,664 hours and carried a wave height in 18% of them — not patchy data, a
series that began in 2021-10, fifteen years after the casualty record. The
request succeeded and the file was written. `--marine-probe` asks each model what
span it actually has at that coordinate; only `era5_ocean` reaches 2000. Use it
for anything historical, and note that it serves the combined height, direction
and period but **not** the swell/wind-wave partition, which is exactly the split
that would test the New Hanover story directly.

**A catalogued station is not a publishing station.** Buoy 41110 sits 10.2 km off
Wrightsville Beach and publishes no standard meteorological feed at all
(`Includes Meteorology: False`). CO-OPS 8658163 sits 3.6 km away and serves no
water level. Both appear in `camera_candidates.csv` with their distances, and
those distances fed the site ranking. `buoy_km` and `tide_km` measure how far
away a station is, not whether it has ever published anything — the same class of
error as counting a wave column that is silently all NaN.

## Waves at the beach, from CDIP MOP

Every wave number in this project so far has come from far away: Walton
Lighthouse joined to buoy **46236, 22.9 km out in 133 m of water**, Wrightsville
Beach to an Open-Meteo reanalysis cell. Surfline's Santa Cruz wave data does not
come from Surfline — the upstream is **CDIP** (Coastal Data Information Program,
Scripps), which pushes its buoy measurements to NDBC every 30 minutes and also
runs **MOP** (MOnitoring and Prediction), a model that propagates those waves
inshore and publishes hourly series at points along the 10–15 m isobath.

`pull_cdip_mop.py` reads them over OPeNDAP. For Walton the nearest point is
**SC130, 1.45 km from the camera**, hourly from 1999-12-31, with the hindcast and
nowcast files abutting at one shared timestamp. It is a model, so its columns
stay `lower_snake_case` alongside ERA5 — but a model initialised by a real buoy
and propagated to the surf zone carries three things nothing else here has:
`metaShoreNormal` (published, against four bearings this project read off a map),
`waveModelInputSource` (which buoy constrained each timestep), and `waveSxy`,
the alongshore radiation stress that actually drives longshore current.

California only. MOP does not exist for Virginia Beach, Wrightsville or Jupiter.

### Three data traps found here

**A region search that stops at the first candidate.** The first run wrote 229,867
hours from **B1788, 251 km away**, under the Walton camera's name. Two faults: the
loop broke at the first region whose *first* point was within 400 km (B0001 is
367 km from Santa Cruz), and nothing afterwards refused the result. The search now
scores every region and `MAX_KM = 25.0` makes the run exit rather than write. A
wrong file that looks complete is worse than no file.

**Guessing an identifier shape.** Dataset ids come in two families — `B0001` (one
letter, four digits) and `SC001` (two letters, three) — and three separate
attempts to pattern-match them cost a turn each, once producing a confident
"only 4 regions" census that was also doubled because each dataset appears twice
in the catalogue HTML. Santa Cruz exists only in the second family.

**`_FillValue` is a number.** CDIP declares `_FillValue = -999.99` on its Float32
wave variables and the `.ascii` service hands it back as plain text. `float()`
accepts it, `notna()` is True, and the puller's own coverage report called a
column of nothing but fill **100.0% populated**. At SC130 that hit `waveDm`,
`waveSxy` and `waveSxx` — near-zero denormals (`1.2397983E-33`, uninitialised
memory) at the start of the record and explicit `-999.99` mid-record — while
`waveHs` reads a healthy 0.47 m at the very same timestamps. Unfixed, it would
have poisoned `wave_direction` with a −999.99 bearing. `clean_fill()` masks fill
values and denormals before any percentage is computed, and drops a column that
is under 50% usable at the chosen point rather than writing it under a name other
sites fill honestly. It never drops rows: the height and period at those
timestamps are good, and throwing them away to protect a direction column that
has nothing in it anywhere would lose real data. `--keep-degenerate` overrides.

**A value that looks like a variable name.** The full SC130 pull died claiming
`waveSxx` returned zero values. It was not a truncated response — the block was
there, reading `NaN, NaN, NaN, ...`, and CDIP writes a scalar as
`metaWaterDepth, 10.0`: a name, a comma, a value. A data line beginning `NaN`
has exactly that shape, so `parse_ascii` filed the whole block under a variable
called `NaN` and left `waveSxx` empty. A line now names a variable only if the
response's own DDS header declares that name, and never if it parses as a
number — `float()` accepts `NaN` and `Inf`.

The first reading of that failure was "the server truncates large multi-variable
requests", which was wrong: the same bytes would have parsed correctly. The
recovery built on that reading stayed, because verifying a response against the
count it asked for is worth having either way. `fetch_span()` retries once, then
asks for short variables individually, then halves the row span, and raises with
the byte count and the response tail — which is what identified this bug.

**A number can be garbage without being a sentinel.** `radiation_stress_sxx`
survived the fill and denormal checks at 64.3% "usable" and was written. Its
values run from 1e-11 to **2.3e+29 with a median of 1.6e+19** — a radiation
stress is O(0.01), and B1788 reads 0.042. Forty orders of magnitude is
uninitialised memory read as Float32, and the `TINY` floor only caught the small
end. The fix is not another hand-picked ceiling: CDIP declares `valid_min` and
`valid_max` per variable in the `.das` (`waveHs` is 0.0–20.0 m), so the puller
reads the server's own stated range and masks anything outside it. Where a
variable declares no range, `HUGE = 1e6` is the backstop — nothing here, in
metres, seconds, degrees or stress, is legitimately larger. Fill values are not
double-counted: −999.99 is below every declared minimum, and counting it as both
fill and out-of-range would subtract it from the usable share twice.

Its coverage was also scattered rather than an early gap — full in 2003–2007 and
2015–2024, zero in 2000–2001 and 2009–2013 — so even physical values would have
needed an era cut rather than the whole column.

### Using it

`analyze_drivers.py` loads `data/mop_<site>.csv` when it exists, prefixing every
column `mop_`. Open-Meteo Marine and CDIP MOP publish the same names —
`wave_height`, `wave_period`, `wave_direction` — and one is a global reanalysis
cell while the other is a buoy-driven model 1.45 km off the beach. Sharing a
column name would let a result about one be read as a result about the other:
the same mistake the ALL-CAPS/lower_snake convention exists to prevent between
measured and modelled.

The MOP columns are named in `RIP_PREDICTORS` and `WQ_PREDICTORS` explicitly.
The first Walton run joined the file at 100% overlap and then showed it in no
table at all: a predictor is analysed only if a list names it, so loading a
column and reading it are two separate things, and the join report saying
"2550 overlapping (100%)" said nothing about whether anything used it.

**The shore normal is now read, not guessed.** CDIP publishes one per point, and
the file carries it. Where a MOP file exists its value wins, and the run prints
the gap: **206.0° at SC130 against the 180.0° `SHORE_NORMAL_DEG` assumes for all
three Santa Cruz sites** — with 185.6° at SC125 and 166.9° at SC150, two and
three km along the same beach. A single bearing for the bay was never going to
be right, and every `wind_onshore` and `rip_axis_offset_deg` figure at those
sites was computed from the guess.

This is the same failure this project keeps catching in other people's data — a
sentinel counted as a measurement — found this time in the checking code itself.

SC125 and SC150, 2.2 and 2.8 km along the same coast, return byte-identical
opening rows and the same 67.8% for `waveSxx`. These columns are not broken at
one point — they are the same uninitialised buffer across the MOP hindcast, so
no neighbouring point rescues them.

Zeros are masked only in a column already caught holding fill values or
denormals. An exact 0.0 is ambiguous alone — 0° is due north — but at SC130 the
zeros sit inside the same run as the denormals. Counting a zero against
usability in the report and then writing it to the CSV anyway would leave the
two disagreeing, which is how a bad number gets into an analysis.

`--probe` now reads **every 553rd hour across the whole record** in one strided
OPeNDAP request (`waveDm[0:553:221327]`) rather than five rows off the front, and
prints each column's usable share and the month its first bad sample appears.
The head sample saw only the denormals; the fill values start after 2013. A probe
that reads the beginning of a record answers a question nobody asked.

### What it bought, and what it cost

Bought: at Walton, `mop_wave_height` beats buoy `WVHT` on every rip target, by
two to four times, and tops `bbox_area_max` at rho_hrmo **0.3037** against the
buoy's 0.1508 — full numbers under [Rip detection at Walton](#rip-detection-at-walton-once-the-waves-came-from-the-beach).
It also replaced a guessed shore normal with a published one, 26 degrees away.

Cost: **`waveSxy` was the reason for coming here and it does not exist.** The
alongshore radiation stress is the term that actually drives longshore current,
and at SC130 it is `-999.99` in 233,581 of 233,601 hours — as is `waveDm`, and
`waveSxx` fails the range check at 28.4% usable. SC125 and SC150, two and three
km along the same beach, return byte-identical rows, so no neighbouring point
rescues them. What MOP delivered is height, mean period, peak period and peak
direction, well measured and close in. The rip-forcing term is not available
from this source at this stretch of coast, and nothing here should be read as
having tested it.

## The stricter geometry check (`check_geometry_strict.py`)

`check_camera_geometry.py` answers "is this one geometric record or several?"
with phase correlation over the whole frame. Running it end to end at Walton
Lighthouse exposed four ways that method can be confidently wrong, and this
script is the same question asked so that none of them is available.

    python check_geometry_strict.py --camera <slug> --inventory
    python check_geometry_strict.py --camera <slug> --cached --mask-preview
    python check_geometry_strict.py --camera <slug> --cached --survey

**The water is masked out before anything is registered, not after.** At Walton
six of twelve surviving patches sat on buildings and the six that failed sat on
surf and open water — the frame is majority moving ocean, so a whole-frame
correlation is partly a measurement of the waves. The land mask is declared by
hand in the `MASKS` table, with a note saying which frame it was drawn from, and
**registration exits rather than run without one**. A synthetic scene where the
water moves 30 px and the land moves 5 px is in the test suite: unmasked, the
run returns the water's 30 px at a confidence of 374.

**The primary route fits a homography, not a shift.** SIFT (or ORB) with RANSAC,
matched only on the masked land. RANSAC rejects outliers instead of averaging
them in, and a homography carries rotation, scale and perspective. This matters
more than it sounds: *a rotation about the frame centre moves the centre pixel
by exactly zero*, so a camera that turned 2° — moving its corners by 13 px —
reads as perfectly stable to any shift-only method. Every run reports rotation,
scale, anisotropy and bend alongside translation. **A zoom change leaves the
frame size untouched and so never appears in metadata**; the scale column is the
only place it can show up.

**Phase correlation is kept as a second, independent route.** Not for
redundancy: the gap between two routes is the only thing in the file that
measures *correctness*. Confidence does not. Walton's per-quarter table read
88–100% registered in quarters that demonstrably did not register, because
peak-to-sidelobe ratio says how distinctive a peak is and never whether it is in
the right place. The pass rate is reported here and is never the answer.

**Only boundaries both routes find are called confirmed.** Everything else is
labelled as one route's candidate, and distinguished from the case where the
other route had no frames there to look at.

### The noise algebra is not the same as the Walton one

Within one route, a neighbour step is compared against the difference of two
measurements against the reference, so the gap carries `e·√3` and `noise =
gap/√3` — as before. **Between the two routes it is `e·√2`, not `e·√3`**, because
those are two direct measurements of the same quantity rather than a step
against a difference. Using √3 there would understate the error by 22% and
promise a resolution the data does not support. The step threshold is three
times the *worst* of the available estimates, never the kindest.

### Two things measured here that are easy to get wrong

**Do not difference the survivors.** Dropping weak frames and then calling
`.diff()` pairs each neighbour step with whatever row happened to survive before
it. Across a fortnight of fog that is a two-week gap, so the camera's real
motion over that fortnight gets booked as measurement noise and the floor
inflates until nothing can be resolved.

**A masked band caps its range, it does not lose its axis** — and the earlier
version of this note had that backwards, which is worth recording because the
wrong version would have had us throw away true vertical agreement.

The band the mask leaves at a beach camera is a strip along the bottom of the
frame, so this is the normal case, not a corner one. The claim here used to be
that phase correlation across such a strip returns ~0 *at ordinary confidence*
— a silent lie, the worst failure a stability check can have. That reading came
from cropping to the **feathered** mask's bounding box, which hands the
correlator a ramp instead of an edge. Cropping to the **binary** box instead
(and keeping the Hann window, since the band runs into the frame edge) removed
it, and it does not reproduce at any band height down to 16 rows.

What is actually left is narrower and self-announcing. Shifting a band across
its short axis destroys overlap, so confidence falls as the displacement grows.
Swept on synthetic texture, `dx` held at +5:

| true `dy` | 72-row band | 300-row band |
| --------- | ----------- | ------------ |
| 2 px      | +1.99, conf 297 | +2.00, conf 959 |
| 10 px     | +9.98, conf 152 | +10.00, conf 819 |
| 20 px     | +19.95, conf 64 | +20.00, conf 599 |
| 30 px     | +29.90, conf 21 | +30.00, conf 459 |
| 34 px     | +33.86, conf 11 | +34.00, conf 415 |
| 40 px     | **−33.79, conf 6** | +40.00, conf 364 |

The thin band reads every displacement it can see correctly, to a hundredth of
a pixel, out to about half its height — the search is clamped there. Past that
the answer is wrong, but confidence has already fallen through `MIN_CONFIDENCE`
(8) by the time it is, so the frame is rejected rather than believed. The cost
is **range, not truth**: a gap in route 2 along a thin axis may mean a step too
big for that axis to see, not a camera that held still. `THIN_BAND_PX` now
marks that ceiling rather than a blind spot, and the run prints the ceiling in
pixels for the mask actually declared.

### A mask is a claim about every frame, so `--mask-audit` tests it there

The mask is drawn by hand on one frame. That frame was taken at one tide, in
one season, with the beach at one width — and the mask asserts something about
all 1.1 million stills. The failure mode is invisible to a preview by
construction: a boundary traced along the waterline at low tide sits over dry
sand in the frame it was drawn on and over swash on every spring high in the
record. It looks right exactly once.

`--mask-audit` walks the cached frames and counts, per pixel, how often that
pixel looked like water. The discriminator is colour rather than motion,
because daily sampling leaves no short timescale in which water moves and sand
does not: dry sand is warm (`R - B` strongly positive), water, foam and wet
sand are not. It reads a fog whiteout as water too, which is why the per-frame
shares are reported rather than collapsed into a verdict — a bad mask wets a
fraction of the frames, fog wets all of them at once, and the median tells them
apart.

On a synthetic record whose waterline swings between rows 180 and 260, a mask
drawn at the calmest moment (boundary at row 0.55) is caught with **15.6% of
its area reading as water in a quarter of the frames**; the same mask pushed to
row 0.70 takes none. Both pass a single-frame look.

**The first version of this audit was confounded, and said so loudly without
anyone noticing.** Run against the deliberately over-cautious Sailfish mask —
one sitting entirely on dry sand in the frame it was drawn from — it reported
**76.6% of that mask "wet in a quarter of frames"**, with the intrusion
reaching row **0.997**: the dune fence at the bottom of the frame. Water does
not reach the dune line.

The cause is not fog alone. A four-year beach record is roughly a third dim —
dusk, overcast, rain, a camera dropping to monochrome — and on those frames dry
sand's `R - B` falls under any fixed threshold, so sand reads as water. Once
the dim frames have eaten most of the "quarter of frames" budget, a pixel needs
only a few percent of the remainder to clear the bar, and essentially every
pixel does. Reproduced on a synthetic record with that lighting mix: a mask on
dry sand in *every* frame scores **100% often-wet, 90th percentile 100%, reach
0.998** — against Sailfish's 76.6% / 99.5% / 0.997.

A clean fixture does **not** reproduce this, which is why the first attempt at
a regression test passed while the real run was nonsense. Two fixes, both
measurements rather than thresholds tuned to taste:

1. **A frame with no colour in it cannot answer the question.** If `R - B`
   spans under 20 levels across the whole frame, nothing in it is
   distinguishable from anything else, and it is skipped and counted as
   skipped — not counted as all-water.
2. **The threshold is anchored to water the frame itself shows.** A band
   seaward of the mask is ocean by construction; its median `R - B` is what
   water looks like in *this* frame's light, and the split sits above it. An
   overcast afternoon is no longer read as a flooded beach.

On the same synthetic mix the fixed test scores the dry-sand mask **0.0%** and
still catches a mask drawn on the waterline at **9.1%**.

The audit also fits the edge rather than leaving it to judgement. For each
column it takes the most landward row that looked like water in at least a
quarter of the frames, fits a line through those, and prints a paste-ready
polygon. On a synthetic record with a shoreline of `y = 0.60 - 0.20x` and a
tide swinging ±0.04 of the frame, it recovers `y = 0.625 - 0.200x` — the slope
exact, the intercept inside the swing. It **proposes**; the mask stays
hand-declared and logged, because an edge fitted automatically to a record
containing a fog winter or a beach renourishment would be fitted to that
instead, and nothing downstream would say so.

**Vegetation was the audit's second confound.** With the light handled, the
Sailfish run still put the intrusion at row 0.997 — the dune fence at the very
bottom of the frame, which the ocean cannot reach. Sea oats and the dark growth
on the dune back read `R - B` around +15 to +30, under any threshold set to
catch an ocean that runs about −40. Green separates them: vegetation is the only
thing in the scene whose green channel leads both others, while water, foam, wet
sand and dry sand all have green between red and blue.

**But vegetation was not what set that 0.997, and the guard did not move it.**
The intrusion line was a *maximum*: the deepest row holding any often-wet pixel,
taken over 1,871 scattered specks in 97,399 mask pixels, on a mask whose typical
frame reads 2.5% wet. One pixel of shadow at the foot of the frame sets it, and
no improvement to the discriminator can move a maximum. The reach is now the
deepest row that is **at least 10% habitually wet across its own masked width**,
with the deepest single pixel still printed beside it and labelled an extreme.
On a well-placed mask there is no such row, and the audit now says so instead of
naming the dune fence. This is the same discipline as the confidence rule: *a
statistic that a single pixel can move is not a measurement.*

**A fit is not a measurement until it is tight.** The first Sailfish audit fitted
`y = 0.665 - 0.163x` at scatter 0.006 over 526 of 672 columns — a real shoreline,
worth pasting. The re-run against the mask built *from that line* fitted
`y = 0.690 + 0.008x` at scatter **0.172** over **672 of 672** columns: 29× the
scatter, the slope collapsed to nothing, and not one column rejected, because two
standard deviations of that much scatter covers the whole set. It printed with
exactly the same confidence as the good fit, and pasting it would have flattened
the edge and thrown away the right-hand half of the beach. That is what success
looks like from inside the fit — the habitual water is already outside the mask,
what is left are specks, and a line through specks is a line through nothing. The
audit now **refuses to offer a polygon** when the scatter exceeds 0.02 of the
frame height, and says to keep the declared edge.

### Masks are declared by hand, and the burned-in overlay is cut back out

Every `MASKS` entry records who drew it and from which frame. The Sailfish
shoreline was fitted, not eyeballed: the sand/water colour boundary across 178
columns (84 kept, the rest rejected as canopies and shadow) gives
`y = 0.615 - 0.195x` with a scatter of 0.003 of the frame height.

**The margin on top of that line was guessed twice and measured once, and the
guesses missed in both directions.** The first draft used 0.10 "for tide" —
about 150 rows — which visibly threw away the upper beach where the umbrellas
sit. Tightening it to 0.04 fixed that frame and broke the record: the audit over
1,602 usable frames fits habitual water at **`y = 0.665 - 0.163x`** (scatter
0.006 of the frame height, 526 of 672 columns), which is 0.05 further landward
than the reference frame's own shoreline at the left and **0.08 further at the
right** — so 0.04 put the seaward edge *inside* habitual water at the right-hand
end. The slope is shallower than any single frame shows, too.

The declared edge is now that measured line plus 0.03, which lands at 38.1% of
the frame — within 0.1 points of the over-cautious first draft, but for a
measured reason and with the right slope. **A single frame cannot show a tide,**
and neither guess was going to.

The re-run against that edge closed the loop: habitually-wet pixels inside the
mask fell from **7.69% to 1.92%**, and the median frame's wet share from 4.0% to
**2.5%**. The edge is final at 38.1% of the frame, 73 survey cells of 128 px.

The `drop` polygon is not cosmetic. **The "Sailfish" watermark is burned into
the sensor, not the scene** — measured at x 0.033–0.104, y 0.927–0.956. It is
bright, sharp and perfectly stationary, and it does not move when the camera
moves. On a beach the sand is texture-poor and this is the highest-contrast
thing in the land region, so SIFT weights it heavily; it votes for "no motion"
in exactly the frames where the answer matters. Left in, it anchors both routes
and reports a camera that has turned as a camera that has not: the single most
dangerous kind of false stability this pipeline can produce, because it raises
confidence while destroying the measurement.

The `drop` box is sized to the text and no larger. The first draft blacked out
the whole bottom-left corner, which was the same mistake as the 0.10 margin in
miniature — more land given up for nothing.

### The survey is the decisive test, and it is run early

Tile the land into non-overlapping cells and register each independently. A
rigid camera translation displaces every cell by the same vector, so a common
vector either exists or it does not — no view about which part of the scene is
interesting is required. At Walton this was the strongest single piece of
evidence: 280 cells, no three agreeing within 3 px.

**The grid is anchored to the mask, not to the frame origin.** Land at rows
298–479 of a 480-row frame is 182 rows — room for a 128 px cell with 54 to
spare — but the only frame-anchored row starts at 384 and runs off the bottom,
so the survey found zero cells and the decisive test silently did not run. Cells
stay non-overlapping so their agreement is independent; if the band cannot hold
three, the cell size drops rather than letting cells share pixels.

### What it will not do

It will not correct or re-register anything. It will not pool two cameras on the
same beach. It will not register across a frame-size change — that is a hard
epoch boundary, and the inventory that finds it runs before anything else,
because weekly sampling at Walton missed a five-frame resolution change entirely
and it took the full daily pull to surface.
