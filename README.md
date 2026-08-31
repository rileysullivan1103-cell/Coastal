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
python verify_wqp_fields.py      # 3. are the WQP column names right?
python test_matching_offline.py  # 4. matching logic sanity check (no network)
python find_candidate_sites.py   # 5. the real run
```

Steps 2 and 3 may tell you to update field names in `find_candidate_sites.py`
before step 5 will work. That is the point of running them.

## Scripts

- **`check_tokens.py`** — one cheap authenticated request per service. Reports
  "rejected" separately from "unreachable" so you can tell a bad token from a
  network problem.
- **`verify_webcoos_fields.py`** — probes `GET /webcoos/api/v1/assets/` and prints
  the full field map plus every coordinate-like path, and dumps the raw JSON.
  This is what confirms where camera coordinates actually live.
- **`verify_wqp_fields.py`** — pulls a small bbox from the Water Quality Portal
  and prints the real column names against the ones the pipeline expects.
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

## Known scaling notes

- The CDO pull paginates 1000 stations at a time over the whole US extent, which
  is 100+ requests. CDO allows 5 req/sec and 10,000/day; the loop sleeps between
  pages and backs off on HTTP 429.
- The national WQP pull is large and can take several minutes. Narrow
  `US_COASTAL_EXTENT` to your actual region of interest to cut both this and the
  CDO pull down substantially.
