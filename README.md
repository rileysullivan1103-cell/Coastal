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
indistinguishable here from something that drives the *rip*. Rip data exists
at one camera only, so nothing generalizes to another beach, and Walton has no
observed wind, so every wind column there is unvalidated ERA5.

Bacteria samples are not a random sample of conditions: agencies sample in
swim season, on a schedule, and sometimes after known spills. Non-detects are
dropped rather than imputed, which thins the low end. `SampleDate` carries no
time of day, so tide and wind are daily means — not the value at the moment of
sampling.

The shore normal used to turn wind and swell direction into onshore
components is an **assumption** (`SHORE_NORMAL_DEG`, 180 degrees for the Santa
Cruz sites, which face south across Monterey Bay). Any onshore/offshore result
is conditional on it.

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

## Rip detection: no conclusions yet

Deliberately not drawing any. One camera, and the observation window barely
overlaps the rip window (`n=39` on the gridded join, `n=1` on the buoy). Until
more cameras carry the product, or enough time accumulates for a real overlap,
anything the rip tables show is noise with a p-value.

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
