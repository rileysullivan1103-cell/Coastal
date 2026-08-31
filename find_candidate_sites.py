"""
Site discovery v2 — bulk-download each data source ONCE, then match locally.
Faster, fewer API calls, avoids NOAA CDO rate limits entirely by using their
pre-computed 'datacoverage' metadata field instead of pulling and counting
actual data records.

Setup:
    pip install -r requirements.txt
    cp .env.example .env   # then fill in both tokens
    python find_candidate_sites.py

Requires outbound access to app.webcoos.org, www.ncei.noaa.gov,
www.ndbc.noaa.gov and www.waterqualitydata.us.
"""

import env  # noqa: F401  -- loads .env into os.environ

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

# NdbcApi().stations() returns 709 land-based 'fixed' stations alongside 439
# moored 'buoy's (plus dart/oilrig/tao/other). They are different instruments:
# a moored buoy reports offshore conditions, a fixed station reports whatever
# is bolted to a pier. Restrict to the types whose data suits the pipeline.
# None keeps every type. Restricted to moored buoys: the pipeline wants genuine
# offshore conditions (waves, sea-surface temperature), which a tide gauge or a
# pier-mounted sensor does not measure.
BUOY_TYPES = ("buoy",)
# NDBC's 'met' flag. Drops stations with no standard meteorological feed.
BUOY_REQUIRE_METEOROLOGY = False

MAX_BUOY_DISTANCE_KM = 50
MAX_PRECIP_DISTANCE_KM = 30
# Water quality gets its own, much tighter radius: a bacteria reading is only
# representative of the stretch of water it was taken from, so a station has to
# be effectively at the beach the camera watches, not merely in the same town.
MAX_WQ_DISTANCE_KM = 2
MIN_ACCEPTABLE_DATACOVERAGE = 0.90  # NOAA's own precomputed metric, 0-1
# datacoverage is a lifetime figure: a station that stopped reporting in 2019
# can still score 0.98 and be useless for a recent window. CDO's 'maxdate' is
# the last day the station reported, so require it to be recent. Set to None to
# disable the check.
MAX_PRECIP_STALENESS_DAYS = 90
TOP_N_SITES = 20

# Rough California coastal bounding box — good enough to flag "use the CA
# CKAN source" without needing an extra geocoding call. Not precise at the
# Oregon/Mexico borders, but fine for this purpose.
CA_LAT_RANGE = (32.5, 42.0)
CA_LON_RANGE = (-124.5, -117.0)

# Region to search. Cameras outside it are dropped, and the NOAA/WQP pulls are
# scoped to it — which is most of the runtime, so keep it as tight as you can.
REGIONS = {
    "california": (CA_LAT_RANGE[0], CA_LON_RANGE[0], CA_LAT_RANGE[1], CA_LON_RANGE[1]),
    "us_coastal": (24.0, -125.0, 49.0, -66.0),
}
REGION = "california"

# California water quality via the data.ca.gov CKAN datastore.
# Set CA_CKAN_RESOURCE_ID to a real resource UUID to actually pull stations.
# While it is None the pipeline only ASSUMES California sites have water
# quality coverage — it does not verify it. Run verify_ca_ckan.py to find the
# resource id and its column names, then fill these in.
CA_CKAN_BASE = "https://data.ca.gov/api/3/action"

# "Beach Water Quality Monitoring Stations" from the Beach Advisories dataset:
# 1041 rows, one per monitoring station. The sibling Fecal Indicator Bacteria
# results table has coordinates too, but it is ~627k sample rows — the wrong
# shape and far too slow for site discovery.
CA_CKAN_RESOURCE_ID = "98e628ff-d012-4982-ad32-b9f9ad8ab524"
# Stations carry Upper and Lower coordinate pairs, identical for point
# stations. Upper is the reference point.
CA_CKAN_LAT_COL = "Station_UpperLat"
CA_CKAN_LON_COL = "Station_UpperLon"
# Station_id is the only reliable key: Station_Name and
# AgencyStationIdentifier are both literally "0" on many rows.
CA_CKAN_ID_COL = "Station_id"
# Carried into the output so a row names a beach rather than a bare number.
CA_CKAN_LABEL_COL = "Beach_Name"
# Decommissioned stations would otherwise match cameras and imply coverage
# that no longer exists.
CA_CKAN_STATUS_COL = "Status"
CA_CKAN_ACTIVE_ONLY = True
CA_CKAN_PAGE_SIZE = 1000

# If no CKAN resource is configured, still treat California sites as having
# water quality coverage (the original behaviour). Such rows are reported with
# wq_source_confirmed=False, because nothing was actually checked.
ASSUME_CA_WATER_QUALITY = True


def region_extent(region=REGION):
    """CDO 'extent' string: 'min_lat,min_lon,max_lat,max_lon'."""
    return ",".join(str(v) for v in REGIONS[region])


def region_bbox(region=REGION):
    """WQP 'bBox' string: 'min_lon,min_lat,max_lon,max_lat' — lon first."""
    min_lat, min_lon, max_lat, max_lon = REGIONS[region]
    return f"{min_lon},{min_lat},{max_lon},{max_lat}"


def in_region(lat, lon, region=REGION):
    min_lat, min_lon, max_lat, max_lon = REGIONS[region]
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


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

        # Paging lives under 'pagination', not at the top level:
        # {'next', 'previous', 'count', 'page', 'total_pages', ...}
        pagination = payload.get("pagination") or {}
        url = pagination.get("next")
        page, total_pages = pagination.get("page"), pagination.get("total_pages")
        if url is None and page is not None and total_pages is not None \
                and page < total_pages:
            print(f"  warning: on page {page} of {total_pages} but the response "
                  "carries no next link; results are incomplete")

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
    """NDBC stations. Columns are 'Station', 'Lat', 'Lon', 'Name', 'Type', ...
    (capitalized — see ndbc_api/api/parsers/http/active_stations.py)."""
    df = NdbcApi().stations()

    if BUOY_TYPES is not None and "Type" in df.columns:
        before = len(df)
        df = df[df["Type"].isin(BUOY_TYPES)]
        print(f"  kept {len(df)}/{before} stations of type {tuple(BUOY_TYPES)}")

    if BUOY_REQUIRE_METEOROLOGY and "Includes Meteorology" in df.columns:
        before = len(df)
        df = df[df["Includes Meteorology"]]
        print(f"  kept {len(df)}/{before} stations with a meteorology feed")

    return df.reset_index(drop=True)


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
        # CDO answers {} both for an invalid token and for an offset past the
        # end of the resultset. Only the first page can distinguish them.
        if not payload:
            if not all_stations:
                sys.exit("NOAA CDO returned an empty response on the first page — "
                         "usually an invalid token. Run check_tokens.py.")
            break

        batch = payload.get("results", [])
        if not batch:
            break
        all_stations.extend(batch)

        total = payload.get("metadata", {}).get("resultset", {}).get("count")
        print(f"  fetched {len(all_stations)}" + (f"/{total}" if total else "") + " stations")

        # Stop on the reported total rather than probing one page past the end.
        if total is not None and len(all_stations) >= total:
            break

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


def get_ca_wq_stations() -> pd.DataFrame:
    """Unique monitoring stations from the data.ca.gov CKAN datastore.

    Uses CKAN's standard datastore_search action, which is a fixed spec:
    result.records / result.fields / result.total. The resource id and the
    column names are dataset-specific — run verify_ca_ckan.py to find them.

    Returns an empty frame when nothing is configured.
    """
    if not CA_CKAN_RESOURCE_ID:
        return pd.DataFrame()

    missing = [n for n, v in (("CA_CKAN_LAT_COL", CA_CKAN_LAT_COL),
                              ("CA_CKAN_LON_COL", CA_CKAN_LON_COL),
                              ("CA_CKAN_ID_COL", CA_CKAN_ID_COL)) if not v]
    if missing:
        sys.exit(f"CA_CKAN_RESOURCE_ID is set but {', '.join(missing)} are not. "
                 "Run verify_ca_ckan.py to see the resource's column names.")

    records, offset = [], 0
    while True:
        resp = requests.get(f"{CA_CKAN_BASE}/datastore_search",
                            params={"resource_id": CA_CKAN_RESOURCE_ID,
                                    "limit": CA_CKAN_PAGE_SIZE, "offset": offset},
                            timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("success"):
            sys.exit(f"CKAN request failed: {str(payload.get('error'))[:300]}")

        batch = payload.get("result", {}).get("records", [])
        if not batch:
            break
        records.extend(batch)

        total = payload.get("result", {}).get("total")
        print(f"  fetched {len(records)}" + (f"/{total}" if total else "") + " CKAN records")
        if total is not None and len(records) >= total:
            break
        offset += CA_CKAN_PAGE_SIZE

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    for col in (CA_CKAN_LAT_COL, CA_CKAN_LON_COL, CA_CKAN_ID_COL):
        if col not in df.columns:
            sys.exit(f"Column {col!r} not in the CKAN resource. "
                     f"Available: {list(df.columns)}")

    if CA_CKAN_ACTIVE_ONLY and CA_CKAN_STATUS_COL in df.columns:
        before = len(df)
        df = df[df[CA_CKAN_STATUS_COL].astype(str).str.strip().str.lower() == "active"]
        if before != len(df):
            print(f"  dropped {before - len(df)} non-active stations")

    keep = [CA_CKAN_ID_COL, CA_CKAN_LAT_COL, CA_CKAN_LON_COL]
    if CA_CKAN_LABEL_COL and CA_CKAN_LABEL_COL in df.columns:
        keep.append(CA_CKAN_LABEL_COL)
    df = df[keep].copy()

    df[CA_CKAN_LAT_COL] = pd.to_numeric(df[CA_CKAN_LAT_COL], errors="coerce")
    df[CA_CKAN_LON_COL] = pd.to_numeric(df[CA_CKAN_LON_COL], errors="coerce")
    df = df.dropna(subset=[CA_CKAN_LAT_COL, CA_CKAN_LON_COL])

    # 0.0 is this dataset's stand-in for a missing coordinate, not a location
    # off West Africa.
    before = len(df)
    df = df[(df[CA_CKAN_LAT_COL] != 0) & (df[CA_CKAN_LON_COL] != 0)]
    if before != len(df):
        print(f"  dropped {before - len(df)} stations with zero coordinates")

    # The resource is denormalised (station joined to beach and agency), so the
    # same station can appear more than once.
    df = df.drop_duplicates(subset=[CA_CKAN_ID_COL]).reset_index(drop=True)
    return df


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


def rank_candidate_sites(cameras_df, buoys_df, precip_df, wq_df, ca_wq_df=None):
    results = []
    have_ckan = ca_wq_df is not None and not ca_wq_df.empty
    if not have_ckan and ASSUME_CA_WATER_QUALITY:
        print("  WARNING: no CKAN resource configured — California sites are being "
              "ASSUMED to have water quality coverage, not verified. Those rows "
              "carry wq_source_confirmed=False and wq_distance_km=NaN.")
    good_precip = precip_df[precip_df["datacoverage"] >= MIN_ACCEPTABLE_DATACOVERAGE]
    print(f"  {len(good_precip)}/{len(precip_df)} precip stations meet the "
          f"{MIN_ACCEPTABLE_DATACOVERAGE} datacoverage floor")

    if MAX_PRECIP_STALENESS_DAYS is not None and "maxdate" in good_precip.columns:
        cutoff = (pd.Timestamp.now().normalize()
                  - pd.Timedelta(days=MAX_PRECIP_STALENESS_DAYS))
        last_report = pd.to_datetime(good_precip["maxdate"], errors="coerce")
        before = len(good_precip)
        good_precip = good_precip[last_report >= cutoff]
        print(f"  {len(good_precip)}/{before} of those reported within the last "
              f"{MAX_PRECIP_STALENESS_DAYS} days")

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
            lat, lon, wq_df, "LatitudeMeasure", "LongitudeMeasure", MAX_WQ_DISTANCE_KM)

        # California sites use the dedicated data.ca.gov CKAN source in
        # preference to the generic national WQP search. If that source is
        # actually configured we match against it and get a real station id and
        # distance; if it is not, we fall back to merely ASSUMING coverage,
        # which is recorded honestly via wq_source_confirmed below.
        ca_site = is_california(lat, lon)
        wq_name = None
        if ca_site and have_ckan:
            ca_row, ca_dist = nearest_with_min_distance(
                lat, lon, ca_wq_df, CA_CKAN_LAT_COL, CA_CKAN_LON_COL,
                MAX_WQ_DISTANCE_KM)
            wq_row, wq_dist = ca_row, ca_dist
            wq_source = ca_row[CA_CKAN_ID_COL] if ca_row is not None else None
            if ca_row is not None and CA_CKAN_LABEL_COL in ca_wq_df.columns:
                wq_name = ca_row[CA_CKAN_LABEL_COL]
            wq_measured = ca_row is not None
            wq_satisfied = ca_row is not None
        elif ca_site:
            wq_source = "CA_CKAN_ASSUMED"
            wq_dist = None
            wq_measured = False
            wq_satisfied = ASSUME_CA_WATER_QUALITY
        else:
            wq_source = (wq_row["MonitoringLocationIdentifier"]
                         if wq_row is not None else None)
            wq_measured = wq_row is not None
            wq_satisfied = wq_row is not None

        results.append({
            "camera_name": cam.get("camera_name", "unknown"),
            "lat": lat, "lon": lon,
            "buoy_id": buoy_row["Station"] if buoy_row is not None else None,
            "buoy_name": buoy_row["Name"] if buoy_row is not None else None,
            "buoy_type": buoy_row["Type"] if buoy_row is not None else None,
            "buoy_distance_km": _round_km(buoy_dist),
            "precip_station_id": precip_row["id"] if precip_row is not None else None,
            "precip_datacoverage": precip_row["datacoverage"] if precip_row is not None else None,
            "precip_distance_km": _round_km(precip_dist),
            "wq_station_id": wq_source,
            "wq_station_name": wq_name,
            "wq_distance_km": _round_km(wq_dist),
            # True only when an actual station was matched. An assumed CA site
            # is False — nothing was checked.
            "wq_source_confirmed": wq_measured,
            "has_all_four": all([buoy_row is not None, precip_row is not None, wq_satisfied]),
        })

    if not results:
        sys.exit("No cameras had usable coordinates — nothing to rank.")

    df = pd.DataFrame(results)
    dist_cols = ["buoy_distance_km", "precip_distance_km", "wq_distance_km"]
    # These columns hold None alongside floats, so they arrive as object dtype;
    # coerce to numeric explicitly rather than letting fillna downcast.
    dists = df[dist_cols].apply(pd.to_numeric, errors="coerce")

    # Score on the MEAN of the measured distances, not the sum. A site whose
    # water quality coverage is assumed rather than measured has no wq distance;
    # summing would treat that gap as either 0 km (flattering it) or a 999 km
    # penalty (sinking it below sites that qualify no better). The mean scores
    # each site on what is actually known about it. Rows with nothing measured
    # fall back to the full penalty.
    mean_dist = dists.mean(axis=1).fillna(999)
    df["combined_score"] = df["has_all_four"].astype(int) - mean_dist * len(dist_cols) / 1000
    return df.sort_values("combined_score", ascending=False)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Region: {REGION} ({region_extent()})\n")

    print("Pulling camera list...")
    cameras_df = get_all_cameras()
    print(f"  {len(cameras_df)} cameras found nationally")
    cameras_df = cameras_df[
        cameras_df.apply(lambda c: in_region(c["latitude"], c["longitude"]), axis=1)
    ].reset_index(drop=True)
    print(f"  {len(cameras_df)} inside the {REGION} region")
    if cameras_df.empty:
        sys.exit(f"No cameras inside the {REGION} region — widen REGION and retry.")

    print("Pulling buoy list...")
    buoys_df = get_all_buoys()
    print(f"  {len(buoys_df)} buoys found")

    print("Pulling precipitation station list (paginated)...")
    precip_df = get_all_precip_stations(region_extent(), NOAA_CDO_TOKEN)
    print(f"  {len(precip_df)} precip stations found")

    # The California override already satisfies water quality for any CA site,
    # so if every camera in the region is in California the WQP pull cannot
    # change a single result. Skip it rather than spend minutes on the slowest,
    # least-verified source for nothing.
    all_ca = cameras_df.apply(lambda c: is_california(c["latitude"], c["longitude"]),
                              axis=1).all()

    print("Pulling California water quality stations (data.ca.gov CKAN)...")
    ca_wq_df = get_ca_wq_stations()
    if ca_wq_df.empty:
        print("  none — CA_CKAN_RESOURCE_ID is not configured "
              "(run verify_ca_ckan.py to find it)")
    else:
        print(f"  {len(ca_wq_df)} unique CA monitoring stations found")

    if all_ca:
        print("All cameras in region are in California; skipping the national "
              "Water Quality Portal pull.")
        wq_df = pd.DataFrame(columns=["MonitoringLocationIdentifier",
                                      "LatitudeMeasure", "LongitudeMeasure"])
    else:
        print("Pulling water quality station list (Water Quality Portal)...")
        wq_df = get_all_water_quality_stations(region_bbox())
        print(f"  {len(wq_df)} water quality stations found")

    print("Matching locally...")
    ranked = rank_candidate_sites(cameras_df, buoys_df, precip_df, wq_df, ca_wq_df)

    # Default pandas width truncates this table to "[10 rows x 8 columns]".
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)

    qualified = ranked[ranked["has_all_four"]]
    print(f"Cameras with buoy + precip + water quality all qualifying: {len(qualified)}")

    top_sites = (qualified if len(qualified) >= TOP_N_SITES else ranked).head(TOP_N_SITES)
    top_sites.to_csv("candidate_sites_ranked.csv", index=False)
    print(f"Saved top {len(top_sites)} to candidate_sites_ranked.csv")
    print(top_sites[["camera_name", "buoy_id", "buoy_type", "buoy_distance_km",
                     "precip_datacoverage", "wq_station_name",
                     "wq_distance_km", "has_all_four"]].to_string(index=False))
