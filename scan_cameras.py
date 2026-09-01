"""Score every WebCOOS camera in the country for instrumentability.

find_candidate_sites.py works region-first: pick California, bulk-download every
buoy, rain gauge and monitoring station in it, then match. That is the wrong
shape for going national. There are only ~86 cameras in the whole country, and
the Water Quality Portal holds hundreds of thousands of stations — pulling the
national table to match 86 points would be absurd, and WQP would likely time
out before it finished.

This script inverts it. The cameras are the small set, so it queries outward
from each one:

  buoys         one bulk NDBC pull (~1,350 rows) matched locally
  tide          one bulk CO-OPS pull matched locally
  precipitation one NOAA CDO request per camera, over a small extent
  bacteria      one Water Quality Portal request per camera, over a small bbox

That is roughly 175 requests total instead of one national download, and every
per-camera answer is cached, so a re-run after a timeout costs nothing for the
cameras already done.

    python scan_cameras.py --limit 5          # try it on five cameras first
    python scan_cameras.py                    # the whole country
    python scan_cameras.py --rip-only         # only cameras carrying rip detection
    python scan_cameras.py --refresh          # ignore the cache

Writes camera_candidates.csv, ranked.
"""

import env  # noqa: F401  -- loads .env into os.environ

import argparse
import json
import math
import os
import sys
import time
from io import StringIO

import pandas as pd
import requests

import find_candidate_sites as f

API_BASE = "https://app.webcoos.org/webcoos/api/v1"
# verify_webcoos_fields.py and explore_webcoos_products.py each save ONE page of
# /assets/. That is fine for eyeballing the schema and wrong for a national
# census, so this script pages through the endpoint itself and keeps its own
# complete copy. The single-page dump is only a fallback for running offline.
ASSETS_ALL = "data/webcoos_assets_all.json"
ASSETS_ONE_PAGE = "webcoos_assets_raw.json"
CACHE = "data/camera_scan_cache.json"
OUT_CSV = "camera_candidates.csv"

# Same thresholds the California run settled on, so results are comparable.
MAX_BUOY_KM = 50
MAX_PRECIP_KM = 30
MAX_WQ_KM = 2
MAX_TIDE_KM = 50
# Bacteria sampling is the binding constraint, so the query goes out further
# than the threshold: knowing the nearest station is 3.4 km away is a useful
# answer, and "nothing found" would not distinguish that from 300 km away.
WQ_QUERY_KM = 15

WQP_URL = "https://www.waterqualitydata.us/data/Station/search"
WQP_CHARACTERISTICS = "Escherichia coli;Enterococcus;Fecal Coliform"
# Stations that stopped reporting years ago are no use for a live pipeline.
WQP_SINCE = "01-01-2023"  # WQP wants MM-DD-YYYY
CDO_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/stations"
COOPS_MDAPI = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"

REQUEST_PAUSE = 0.25
TIMEOUT = 180
MAX_RETRIES = 4

# WQP's documented column names. Never confirmed against a live response in
# this project — the California path used the state CKAN source instead. The
# scan checks the first real response and stops with the actual names rather
# than silently matching on columns that do not exist.
WQP_LAT = "LatitudeMeasure"
WQP_LON = "LongitudeMeasure"
WQP_ID = "MonitoringLocationIdentifier"
WQP_NAME = "MonitoringLocationName"

# Set to False the first time WQP rejects startDateLo, so the rest of the scan
# stops sending a parameter this endpoint does not accept.
_WQP_DATE_FILTER = [True]


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def get_with_retry(url, **kwargs):
    delay = 2
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"      {type(exc).__name__}; retry {attempt} in {delay}s")
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
            print(f"      HTTP {resp.status_code}; retry {attempt} in {delay}s")
            time.sleep(delay)
            delay *= 2
            continue
        return resp
    raise RuntimeError("unreachable")


def dig(node, *path):
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE) as fh:
            return json.load(fh)
    return {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as fh:
        json.dump(cache, fh, indent=1)


def bbox_around(lat, lon, km):
    """(min_lon, min_lat, max_lon, max_lat) -- WQP's lon-first order."""
    dlat = km / 111.0
    dlon = km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def extent_around(lat, lon, km):
    """'min_lat,min_lon,max_lat,max_lon' -- CDO's lat-first order.

    The two APIs take opposite orders for the same rectangle, which is exactly
    the kind of thing that silently returns the wrong stations rather than an
    error, so they are built by separate functions that each name their order.
    """
    dlat = km / 111.0
    dlon = km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
    return f"{lat - dlat:.4f},{lon - dlon:.4f},{lat + dlat:.4f},{lon + dlon:.4f}"


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

def fetch_assets(refresh=False):
    """Every WebCOOS asset, following pagination. Cached to ASSETS_ALL."""
    if not refresh and os.path.exists(ASSETS_ALL):
        with open(ASSETS_ALL) as fh:
            return json.load(fh)

    token = os.environ.get("WEBCOOS_TOKEN")
    if not token:
        if os.path.exists(ASSETS_ONE_PAGE):
            print(f"WEBCOOS_TOKEN not set — falling back to {ASSETS_ONE_PAGE}, "
                  "which holds only the first page.")
            with open(ASSETS_ONE_PAGE) as fh:
                return json.load(fh).get("results", [])
        sys.exit("WEBCOOS_TOKEN is not set and no local asset dump exists.")

    headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
    url = f"{API_BASE}/assets/"
    assets, expected = [], None
    while url:
        resp = get_with_retry(url, headers=headers)
        if resp.status_code in (401, 403):
            sys.exit(f"WebCOOS rejected the token ({resp.status_code}): {resp.text[:300]}")
        resp.raise_for_status()
        payload = resp.json()
        assets.extend(payload.get("results", []))
        pagination = payload.get("pagination") or {}
        if expected is None:
            expected = payload.get("count") or pagination.get("count")
        url = pagination.get("next")
        print(f"  {len(assets)} assets so far...")

    if expected and len(assets) != expected:
        print(f"  warning: endpoint reports {expected} assets but {len(assets)} "
              "were collected — pagination may have stopped early")

    os.makedirs(os.path.dirname(ASSETS_ALL), exist_ok=True)
    with open(ASSETS_ALL, "w") as fh:
        json.dump(assets, fh)
    print(f"  saved {ASSETS_ALL}")
    return assets


def load_cameras(refresh=False):
    assets = fetch_assets(refresh=refresh)
    rows, skipped = [], 0
    for asset in assets:
        label = dig(asset, "data", "common", "label") or asset.get("slug")
        lat, lon = f._extract_lat_lon(asset)
        if lat is None:
            skipped += 1
            continue

        products = set()
        for feed in asset.get("feeds") or []:
            for product in feed.get("products") or []:
                slug = dig(product, "data", "common", "slug")
                if slug:
                    products.add(slug)

        rows.append({
            "camera": label,
            "state": dig(asset, "data", "properties", "state_or_territory"),
            "lat": lat, "lon": lon,
            "has_rip": any("rip" in p for p in products),
            "products": len(products),
        })
    if skipped:
        print(f"  {skipped} assets had no usable coordinates, skipped")
    if not rows:
        sys.exit("No cameras with coordinates — run verify_webcoos_fields.py and "
                 "check _GEOM_PATHS in find_candidate_sites.py.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bulk sources -- one request each, matched locally
# ---------------------------------------------------------------------------

def load_buoys():
    print("NDBC stations (one bulk pull)...")
    df = f.get_all_buoys()
    print(f"  {len(df)} stations")
    return df


def load_coops(product="waterlevels"):
    print("CO-OPS stations (one bulk pull)...")
    resp = get_with_retry(COOPS_MDAPI, params={"type": product})
    resp.raise_for_status()
    stations = resp.json().get("stations", [])
    if not stations:
        # The mdapi type vocabulary is not versioned anywhere reliable; if the
        # filter matches nothing, take every station rather than reporting that
        # no camera in the country is near a tide gauge.
        print(f"  type={product!r} returned nothing — retrying unfiltered")
        resp = get_with_retry(COOPS_MDAPI)
        resp.raise_for_status()
        stations = resp.json().get("stations", [])
    rows = [{"station_id": s.get("id"), "name": s.get("name"),
             "lat": s.get("lat"), "lon": s.get("lng")} for s in stations]
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("CO-OPS returned no stations at all — check the mdapi endpoint.")
    df = df.dropna(subset=["lat", "lon"])
    print(f"  {len(df)} tide stations")
    return df


# ---------------------------------------------------------------------------
# Per-camera sources
# ---------------------------------------------------------------------------

def wqp_near(lat, lon, km):
    """Bacteria monitoring stations within km of a point, via WQP.

    Returns (dataframe, note). An empty dataframe means nothing is nearby,
    which is a real answer; None means the request failed and should be retried.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_around(lat, lon, km)
    params = {
        "bBox": f"{min_lon:.4f},{min_lat:.4f},{max_lon:.4f},{max_lat:.4f}",
        "characteristicName": WQP_CHARACTERISTICS,
        "mimeType": "csv",
        "zip": "no",
    }
    if _WQP_DATE_FILTER[0]:
        params["startDateLo"] = WQP_SINCE

    resp = get_with_retry(WQP_URL, params=params)
    if resp.status_code == 400 and _WQP_DATE_FILTER[0]:
        print("    water quality: WQP rejected startDateLo — dropping the date "
              "filter for the rest of the scan (stations may be inactive)")
        _WQP_DATE_FILTER[0] = False
        return wqp_near(lat, lon, km)
    if resp.status_code == 404:
        return pd.DataFrame(), "no stations"
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}: {resp.text[:120]}"
    if resp.content[:2] == b"PK":
        return None, "zip returned despite zip=no"

    text = resp.text.strip()
    if not text:
        return pd.DataFrame(), "no stations"
    frame = pd.read_csv(StringIO(text))
    if frame.empty:
        return frame, "no stations"

    missing = [c for c in (WQP_LAT, WQP_LON, WQP_ID) if c not in frame.columns]
    if missing:
        sys.exit(
            "WQP column names differ from what this script expects.\n"
            f"  missing: {missing}\n"
            f"  actual:  {list(frame.columns)[:25]}\n"
            "Update WQP_LAT / WQP_LON / WQP_ID at the top of scan_cameras.py."
        )
    return frame, "ok"


def cdo_near(lat, lon, km, token):
    """GHCND precipitation stations within km of a point."""
    params = {"extent": extent_around(lat, lon, km), "datasetid": "GHCND",
              "datatypeid": "PRCP", "limit": 1000}
    resp = get_with_retry(CDO_URL, headers={"token": token}, params=params)
    if resp.status_code in (400, 401, 403):
        sys.exit(f"NOAA CDO rejected the request ({resp.status_code}): {resp.text[:200]}")
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"
    # CDO answers "nothing here" with an empty BODY, not an empty JSON object,
    # so .json() has to be guarded rather than called straight.
    if not resp.text.strip():
        return pd.DataFrame(), "no stations"
    frame = pd.DataFrame(resp.json().get("results", []))
    return frame, "ok" if not frame.empty else "no stations"


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def nearest(lat, lon, frame, lat_col, lon_col, max_km):
    """(row, km) of the closest row within max_km, else (None, None)."""
    if frame is None or frame.empty:
        return None, None
    if lat_col not in frame.columns or lon_col not in frame.columns:
        return None, None
    lats = pd.to_numeric(frame[lat_col], errors="coerce")
    lons = pd.to_numeric(frame[lon_col], errors="coerce")
    keep = (lats.notna() & lons.notna()).to_numpy()
    if not keep.any():
        return None, None
    subset = frame[keep]
    dist = f.haversine_km(lat, lon, lats.to_numpy()[keep], lons.to_numpy()[keep])
    best = int(dist.argmin())
    if dist[best] > max_km:
        return None, None
    return subset.iloc[best], float(dist[best])


def _as_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def scan(cameras, buoys, tide_stations, token, cache, refresh):
    results = []
    frame = cameras.reset_index(drop=True)
    for index, cam in frame.iterrows():
        key = f"{cam['camera']}|{cam['lat']:.4f},{cam['lon']:.4f}"
        print(f"[{index + 1}/{len(frame)}] {cam['camera']}  [{cam['state']}]")

        buoy_row, buoy_km = nearest(cam["lat"], cam["lon"], buoys, "Lat", "Lon",
                                    MAX_BUOY_KM)
        tide_row, tide_km = nearest(cam["lat"], cam["lon"], tide_stations, "lat", "lon",
                                    MAX_TIDE_KM)

        entry = {} if refresh else dict(cache.get(key, {}))

        if "wq" not in entry:
            wq_frame, note = wqp_near(cam["lat"], cam["lon"], WQ_QUERY_KM)
            if wq_frame is None:
                print(f"    water quality: {note} — leaving uncached to retry")
            else:
                row, km = nearest(cam["lat"], cam["lon"], wq_frame,
                                  WQP_LAT, WQP_LON, WQ_QUERY_KM)
                entry["wq"] = None if row is None else {
                    "id": str(row[WQP_ID]),
                    "name": str(row.get(WQP_NAME, "")),
                    "km": round(km, 2),
                    "candidates": int(len(wq_frame)),
                }
            time.sleep(REQUEST_PAUSE)

        if "precip" not in entry and token:
            precip_frame, note = cdo_near(cam["lat"], cam["lon"], MAX_PRECIP_KM, token)
            if precip_frame is None:
                print(f"    precipitation: {note} — leaving uncached to retry")
            else:
                row, km = nearest(cam["lat"], cam["lon"], precip_frame,
                                  "latitude", "longitude", MAX_PRECIP_KM)
                entry["precip"] = None if row is None else {
                    "id": str(row["id"]),
                    "name": str(row.get("name", "")),
                    "km": round(km, 2),
                    "coverage": _as_float(row.get("datacoverage")),
                    "maxdate": str(row.get("maxdate", "")),
                }
            time.sleep(REQUEST_PAUSE)

        cache[key] = entry
        save_cache(cache)

        wq = entry.get("wq")
        precip = entry.get("precip")
        # A station 9 km away is recorded but does not count as coverage.
        wq_close = bool(wq) and wq["km"] <= MAX_WQ_KM
        results.append({
            "camera": cam["camera"],
            "state": cam["state"],
            "lat": cam["lat"], "lon": cam["lon"],
            "has_rip": cam["has_rip"],
            "buoy_id": None if buoy_row is None else buoy_row["Station"],
            "buoy_km": None if buoy_km is None else round(buoy_km, 1),
            "tide_id": None if tide_row is None else tide_row["station_id"],
            "tide_km": None if tide_km is None else round(tide_km, 1),
            "wq_id": None if not wq else wq["id"],
            "wq_name": None if not wq else wq["name"],
            "wq_km": None if not wq else wq["km"],
            "wq_within_2km": wq_close,
            "wq_nearby": 0 if not wq else wq["candidates"],
            "precip_id": None if not precip else precip["id"],
            "precip_km": None if not precip else precip["km"],
            "precip_coverage": None if not precip else precip["coverage"],
            "precip_maxdate": None if not precip else precip["maxdate"],
        })
        got = [n for n, v in (("buoy", buoy_row is not None),
                              ("tide", tide_row is not None),
                              (f"water quality @{wq['km']}km" if wq else "water quality",
                               wq_close),
                              ("precip", bool(precip))) if v]
        note = ", ".join(got) if got else "nothing within range"
        if wq and not wq_close:
            note += f"  (nearest bacteria station {wq['km']} km — too far)"
        print(f"    {note}")

    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser(description="Scan WebCOOS cameras for nearby data sources.")
    ap.add_argument("--limit", type=int, help="scan only the first N cameras")
    ap.add_argument("--rip-only", action="store_true",
                    help="only cameras carrying a rip-detection product")
    ap.add_argument("--state", help="restrict to one state or territory")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the cache and re-fetch everything")
    args = ap.parse_args()

    token = os.environ.get("NOAA_CDO_TOKEN")
    if not token:
        print("NOAA_CDO_TOKEN is not set — precipitation will be skipped.\n")

    print("WebCOOS assets...")
    cameras = load_cameras(refresh=args.refresh)
    print(f"{len(cameras)} cameras with coordinates, "
          f"{int(cameras['has_rip'].sum())} carrying rip detection")
    if args.state:
        cameras = cameras[cameras["state"] == args.state]
    if args.rip_only:
        cameras = cameras[cameras["has_rip"]]
    if args.limit:
        cameras = cameras.head(args.limit)
    if cameras.empty:
        sys.exit("No cameras selected.")
    print(f"scanning {len(cameras)}\n")

    buoys = load_buoys()
    tide_stations = load_coops()
    print()

    cache = load_cache()
    table = scan(cameras, buoys, tide_stations, token, cache, args.refresh)

    table["sources"] = (table["buoy_id"].notna().astype(int)
                        + table["tide_id"].notna().astype(int)
                        + table["wq_within_2km"].astype(int)
                        + table["precip_id"].notna().astype(int))
    table["instrumentable"] = table["sources"] == 4
    table = table.sort_values(["has_rip", "sources", "wq_km"],
                              ascending=[False, False, True])
    table.to_csv(OUT_CSV, index=False)

    print(f"\n{'=' * 74}\nRESULTS\n{'=' * 74}")
    print(f"{int(table['wq_within_2km'].sum())}/{len(table)} cameras have a bacteria "
          f"station within {MAX_WQ_KM} km")
    print(f"{int(table['wq_id'].notna().sum())}/{len(table)} have one within "
          f"{WQ_QUERY_KM} km")
    print(f"{int(table['instrumentable'].sum())}/{len(table)} have all four sources")

    cols = ["camera", "state", "sources", "wq_km", "buoy_km", "precip_km", "tide_km"]
    rip = table[table["has_rip"]]
    if not rip.empty:
        print(f"\n--- cameras carrying rip detection ({len(rip)}) ---")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(rip[cols].to_string(index=False))

    full = table[table["instrumentable"]]
    if not full.empty:
        print(f"\n--- every camera with all four sources ({len(full)}) ---")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(full[["camera", "state", "has_rip", "wq_km", "buoy_km",
                        "precip_km", "tide_km"]].to_string(index=False))

    print(f"\nwrote {OUT_CSV}")
    print(f"Water quality is the binding constraint — it is required within "
          f"{MAX_WQ_KM} km, the others at {MAX_PRECIP_KM}-{MAX_BUOY_KM} km.")


if __name__ == "__main__":
    main()
