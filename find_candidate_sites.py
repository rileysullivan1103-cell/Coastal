"""
Site discovery v2 — bulk-download each data source ONCE, then match locally.
Faster, fewer API calls, avoids NOAA CDO rate limits entirely by using their
pre-computed 'datacoverage' metadata field instead of pulling and counting
actual data records.

Setup:
    pip install -r requirements.txt
    cp .env.example .env   # then fill in both tokens
    export $(grep -v '^#' .env | xargs)
    python find_candidate_sites.py

Requires outbound access to app.webcoos.org, www.ncei.noaa.gov,
www.ndbc.noaa.gov and www.waterqualitydata.us.
"""

import os
import sys
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests
from ndbc_api import NdbcApi

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

# Tokens come from the environment. Never hardcode them — this file is in git.
WEBCOOS_TOKEN = os.environ.get("WEBCOOS_TOKEN")
NOAA_CDO_TOKEN = os.environ.get("NOAA_CDO_TOKEN")

WEBCOOS_API_BASE = "https://app.webcoos.org/webcoos/api/v1"

# CDO permits 5 req/sec; 0.25s between pages keeps a safe margin.
CDO_PAGE_SIZE = 1000  # CDO's maximum
CDO_REQUEST_INTERVAL_S = 0.25

MAX_BUOY_DISTANCE_KM = 75
MAX_PRECIP_DISTANCE_KM = 30
MIN_ACCEPTABLE_DATACOVERAGE = 0.90  # NOAA's own precomputed metric, 0-1
TOP_N_SITES = 20

# Rough California coastal bounding box — good enough to flag "use the CA
# CKAN source" without needing an extra geocoding call. Not precise at the
# Oregon/Mexico borders, but fine for this purpose.
CA_LAT_RANGE = (32.5, 42.0)
CA_LON_RANGE = (-124.5, -117.0)


def is_california(lat, lon):
    return CA_LAT_RANGE[0] <= lat <= CA_LAT_RANGE[1] and CA_LON_RANGE[0] <= lon <= CA_LON_RANGE[1]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance. lat2/lon2 may be numpy arrays (vectorized)."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# STEP 1 — Bulk-download all four station/camera lists, ONCE each
# ---------------------------------------------------------------------------

# Candidate paths for an asset's coordinates, most likely first. WebCOOS assets
# are GeoJSON-ish, so the coordinate pair may be nested a few different ways.
# Run verify_webcoos_fields.py to see the real shape; if none of these match,
# get_all_cameras() raises rather than silently returning zero usable rows.
_GEOM_PATHS = [
    ("data", "properties", "location"),
    ("data", "common", "geometry"),
    ("geometry",),
    ("data", "geometry"),
    ("location",),
]


def _dig(node, path):
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _extract_lat_lon(asset):
    """Return (lat, lon) for a WebCOOS asset, or (None, None)."""
    for path in _GEOM_PATHS:
        geom = _dig(asset, path)
        if not isinstance(geom, dict):
            continue
        # GeoJSON Point: coordinates are [lon, lat] — note the order.
        coords = geom.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            try:
                return float(coords[1]), float(coords[0])
            except (TypeError, ValueError):
                pass
        # Plain lat/lon keys under any of the usual spellings.
        for lat_key, lon_key in (("latitude", "longitude"), ("lat", "lon"), ("lat", "lng")):
            if lat_key in geom and lon_key in geom:
                try:
                    return float(geom[lat_key]), float(geom[lon_key])
                except (TypeError, ValueError):
                    pass
    return None, None


def get_all_cameras() -> pd.DataFrame:
    """Camera name + coordinates, straight from the WebCOOS assets endpoint.

    NOT pywebcoos.API.get_cameras() — that returns a single 'Camera Name'
    column with no coordinates at all (see pywebcoos/API.py::_get_camera_list),
    which is useless for distance matching.
    """
    if not WEBCOOS_TOKEN:
        sys.exit("WEBCOOS_TOKEN is not set. See .env.example.")

    headers = {"Authorization": f"Token {WEBCOOS_TOKEN}", "Accept": "application/json"}
    url = f"{WEBCOOS_API_BASE}/assets/"
    rows, missing = [], []

    while url:
        resp = requests.get(url, headers=headers, timeout=60)
        if resp.status_code in (401, 403):
            sys.exit(f"WebCOOS rejected the token ({resp.status_code}): {resp.text[:300]}")
        resp.raise_for_status()
        payload = resp.json()

        for asset in payload.get("results", []):
            label = _dig(asset, ("data", "common", "label")) or asset.get("slug")
            lat, lon = _extract_lat_lon(asset)
            if lat is None:
                missing.append(label)
                continue
            rows.append({"camera_name": label, "latitude": lat, "longitude": lon})

        url = payload.get("next")

    if not rows:
        sys.exit(
            "No camera coordinates found in the WebCOOS response. The geometry "
            "field path has changed — run verify_webcoos_fields.py and update "
            "_GEOM_PATHS to match."
        )
    if missing:
        print(f"  warning: {len(missing)} assets had no usable coordinates, skipped")
    return pd.DataFrame(rows)


def get_all_buoys() -> pd.DataFrame:
    """Every NDBC station. Columns are 'Station', 'Lat', 'Lon', 'Name', ...
    (capitalized — see ndbc_api/api/parsers/http/active_stations.py)."""
    return NdbcApi().stations()


def get_all_precip_stations(bounding_extent: str, token: str) -> pd.DataFrame:
    """bounding_extent format: 'min_lat,min_lon,max_lat,max_lon' —
    covering your whole region of interest, not per-camera."""
    if not token:
        sys.exit("NOAA_CDO_TOKEN is not set. See .env.example.")

    url = "https://www.ncei.noaa.gov/cdo-web/api/v2/stations"
    all_stations = []
    offset = 1
    while True:
        resp = requests.get(url, headers={"token": token},
                            params={"extent": bounding_extent, "datasetid": "GHCND",
                                    "limit": CDO_PAGE_SIZE, "offset": offset}, timeout=120)

        # CDO allows 5 requests/second and 10,000/day. A full US extent is well
        # over a hundred pages, so back off rather than tripping the limiter.
        if resp.status_code == 429:
            print("  rate limited by CDO, backing off 10s...")
            time.sleep(10)
            continue
        if resp.status_code in (400, 401, 403):
            sys.exit(f"NOAA CDO rejected the request ({resp.status_code}): {resp.text[:300]}")
        resp.raise_for_status()

        payload = resp.json()
        # CDO returns an empty body (not an error) for an invalid token.
        if not payload:
            sys.exit("NOAA CDO returned an empty response — usually an invalid token. "
                     "Run check_tokens.py.")

        batch = payload.get("results", [])
        if not batch:
            break
        all_stations.extend(batch)

        total = payload.get("metadata", {}).get("resultset", {}).get("count")
        print(f"  fetched {len(all_stations)}" + (f"/{total}" if total else "") + " stations")

        offset += CDO_PAGE_SIZE
        time.sleep(CDO_REQUEST_INTERVAL_S)

    return pd.DataFrame(all_stations)  # includes 'datacoverage' field directly


def get_all_water_quality_stations(bounding_extent: str) -> pd.DataFrame:
    """Water Quality Portal (waterqualitydata.us) — national aggregator that
    BEACON itself draws from. No token needed. bBox format:
    'min_lon,min_lat,max_lon,max_lat' (NOTE: lon/lat order, opposite of the
    NOAA CDO extent format above — easy to mix up).
    UNVERIFIED: exact endpoint/param names below are based on WQP's documented
    conventions, not tested live — check the real response shape before
    trusting field names in the matching step."""
    url = "https://www.waterqualitydata.us/data/Station/search"
    params = {
        "bBox": bounding_extent,
        "characteristicName": "Escherichia coli;Enterococcus;Fecal Coliform",
        "mimeType": "csv",
        "zip": "no",  # without this WQP can hand back a zip, which read_csv cannot parse
    }
    resp = requests.get(url, params=params, timeout=600)
    resp.raise_for_status()
    if resp.content[:2] == b"PK":
        sys.exit("WQP returned a zip archive despite zip=no — unpack it or adjust params.")
    return pd.read_csv(StringIO(resp.text))


# ---------------------------------------------------------------------------
# STEP 2 — Match locally (no further API calls needed)
# ---------------------------------------------------------------------------

def nearest_with_min_distance(lat, lon, candidates_df, lat_col, lon_col, max_km):
    """Nearest row within max_km, or (None, None). Vectorized — the WQP pull
    can be six figures of rows, and iterrows() per camera would take hours."""
    if candidates_df is None or candidates_df.empty:
        return None, None
    for col in (lat_col, lon_col):
        if col not in candidates_df.columns:
            raise KeyError(
                f"Column {col!r} not in dataframe. Available: {list(candidates_df.columns)}"
            )

    lats = pd.to_numeric(candidates_df[lat_col], errors="coerce").to_numpy(dtype=float)
    lons = pd.to_numeric(candidates_df[lon_col], errors="coerce").to_numpy(dtype=float)
    valid = ~(np.isnan(lats) | np.isnan(lons))
    if not valid.any():
        return None, None

    dists = np.full(lats.shape, np.inf)
    dists[valid] = haversine_km(lat, lon, lats[valid], lons[valid])

    idx = int(np.argmin(dists))
    best_dist = float(dists[idx])
    if best_dist > max_km:
        return None, None
    return candidates_df.iloc[idx], best_dist


def _round_km(value):
    # `if value` would turn an exact 0.0 km match into None.
    return round(value, 1) if value is not None else None


def rank_candidate_sites(cameras_df, buoys_df, precip_df, wq_df):
    results = []
    good_precip = precip_df[precip_df["datacoverage"] >= MIN_ACCEPTABLE_DATACOVERAGE]
    print(f"  {len(good_precip)}/{len(precip_df)} precip stations meet the "
          f"{MIN_ACCEPTABLE_DATACOVERAGE} datacoverage floor")

    for _, cam in cameras_df.iterrows():
        lat, lon = cam.get("latitude"), cam.get("longitude")
        if lat is None or lon is None:
            continue

        # NDBC columns are capitalized: 'Lat'/'Lon'/'Station', not 'lat'/'lon'.
        buoy_row, buoy_dist = nearest_with_min_distance(
            lat, lon, buoys_df, "Lat", "Lon", MAX_BUOY_DISTANCE_KM)

        precip_row, precip_dist = nearest_with_min_distance(
            lat, lon, good_precip, "latitude", "longitude", MAX_PRECIP_DISTANCE_KM)

        # NOTE: WQP field names (LatitudeMeasure/LongitudeMeasure) are the
        # documented convention but unverified live — check real response first
        wq_row, wq_dist = nearest_with_min_distance(
            lat, lon, wq_df, "LatitudeMeasure", "LongitudeMeasure", MAX_PRECIP_DISTANCE_KM)

        # Override: California sites always count as having water quality
        # coverage, regardless of what the national WQP search finds — we
        # already have a confirmed, working, openly-licensed source for CA
        # specifically (data.ca.gov CKAN API), so don't let an unverified
        # generic search disqualify or deprioritize a CA site.
        ca_site = is_california(lat, lon)
        wq_satisfied = ca_site or (wq_row is not None)
        wq_source = "CA_CKAN" if ca_site else (
            wq_row["MonitoringLocationIdentifier"] if wq_row is not None else None)

        results.append({
            "camera_name": cam.get("camera_name", "unknown"),
            "lat": lat, "lon": lon,
            "buoy_id": buoy_row["Station"] if buoy_row is not None else None,
            "buoy_distance_km": _round_km(buoy_dist),
            "precip_station_id": precip_row["id"] if precip_row is not None else None,
            "precip_datacoverage": precip_row["datacoverage"] if precip_row is not None else None,
            "precip_distance_km": _round_km(precip_dist),
            "wq_station_id": wq_source,
            "wq_distance_km": 0 if ca_site else _round_km(wq_dist),
            "wq_source_confirmed": ca_site,  # True = data.ca.gov, tested; False = WQP, unverified
            "has_all_four": all([buoy_row is not None, precip_row is not None, wq_satisfied]),
        })

    if not results:
        sys.exit("No cameras had usable coordinates — nothing to rank.")

    df = pd.DataFrame(results)
    dist_cols = ["buoy_distance_km", "precip_distance_km", "wq_distance_km"]
    # These columns hold None alongside floats, so they arrive as object dtype;
    # coerce to numeric before fillna or pandas downcasts with a FutureWarning.
    dists = df[dist_cols].apply(pd.to_numeric, errors="coerce").fillna(999)
    df["combined_score"] = df["has_all_four"].astype(int) - dists.sum(axis=1) / 1000
    return df.sort_values("combined_score", ascending=False)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    US_COASTAL_EXTENT = "24.0,-125.0,49.0,-66.0"

    print("Pulling camera list...")
    cameras_df = get_all_cameras()
    print(f"  {len(cameras_df)} cameras found")

    print("Pulling buoy list...")
    buoys_df = get_all_buoys()
    print(f"  {len(buoys_df)} buoys found")

    print("Pulling precipitation station list (one bulk call, paginated)...")
    precip_df = get_all_precip_stations(US_COASTAL_EXTENT, NOAA_CDO_TOKEN)
    print(f"  {len(precip_df)} precip stations found")

    print("Pulling water quality station list (Water Quality Portal, national)...")
    # WQP wants lon,lat order — converting from the lat,lon extent above
    min_lat, min_lon, max_lat, max_lon = [float(x) for x in US_COASTAL_EXTENT.split(",")]
    wq_bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    wq_df = get_all_water_quality_stations(wq_bbox)
    print(f"  {len(wq_df)} water quality stations found")

    print("Matching locally...")
    ranked = rank_candidate_sites(cameras_df, buoys_df, precip_df, wq_df)

    qualified = ranked[ranked["has_all_four"]]
    print(f"Cameras with buoy + precip + water quality all qualifying: {len(qualified)}")

    top_sites = (qualified if len(qualified) >= TOP_N_SITES else ranked).head(TOP_N_SITES)
    top_sites.to_csv("candidate_sites_ranked.csv", index=False)
    print(f"Saved top {len(top_sites)} to candidate_sites_ranked.csv")
    print(top_sites[["camera_name", "buoy_id", "buoy_distance_km",
                     "precip_station_id", "precip_datacoverage",
                     "wq_station_id", "wq_distance_km"]])
