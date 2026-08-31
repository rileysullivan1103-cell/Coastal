# Coastal

Site-discovery pipeline for coastal monitoring: find WebCOOS camera locations
that also have a nearby NDBC buoy, a high-coverage NOAA precipitation station,
and a water-quality station.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # fill in both tokens
export $(grep -v '^#' .env | xargs)
```

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

## Public discharge data (SFPUC)

`probe_sfpuc.py` finds the data endpoint behind
`webapps.sfpuc.org/sapps/beachesandbay.html`. That URL is a rendered page, not
an API, so the client has to be written against whatever backend it actually
calls — the probe extracts endpoint-like URLs from the page, scans its linked
scripts, and can fetch each candidate to show the real response shape.

**Note the geography.** SFPUC's combined sewer discharge monitoring covers the
San Francisco shoreline, and none of the seven qualifying sites are in San
Francisco — they are in San Diego, Marin, Santa Cruz and Santa Barbara
counties. Sausalito is the nearest, across the Golden Gate in Marin. So this
source has no site to attach to yet unless San Francisco cameras are added, or
unless Bay discharges are taken to influence Sausalito, which is a modelling
assumption rather than a given.

## Reading the WebCOOS product catalogue

`explore_webcoos_products.py` lists every feed, product and service per camera
from the saved `webcoos_assets_raw.json`, so it costs nothing to re-run.

It exists because **`pywebcoos` hardcodes `feed_name = 'raw-video-data'`** in
`get_products()`, `get_inventory()` and `download()`. Any product published under
a different feed — which is where a derived rip-detection output would most
likely live — is invisible to that library and needs a direct API call. The
script flags each feed accordingly.

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
