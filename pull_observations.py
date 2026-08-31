"""Pull a year of observations for the qualifying candidate sites.

Three sources at three different native resolutions -- the script does not
pretend otherwise:

  buoy           hourly    NDBC stdmet (wind, waves, air/water temperature)
  precipitation  DAILY     NOAA CDO GHCND. There is no hourly option here:
                           GHCND publishes one PRCP total per day, so the
                           24/48/72h antecedent sums are 1/2/3-day rolling sums.
  water quality  irregular data.ca.gov bacteria samples, a few per week in
                           swim season

    python pull_observations.py

Writes one CSV per site per source into data/.
"""

import env  # noqa: F401  -- loads .env into os.environ

import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from ndbc_api import NdbcApi

import ckan_join
import find_candidate_sites as f

SITES_CSV = "candidate_sites_ranked.csv"
OUT_DIR = "data"
DAYS_BACK = 365  # CDO caps a /data query at a one-year range

# Rolling rainfall windows, in days. With daily data 24h == 1 day.
RAIN_WINDOWS_DAYS = (1, 2, 3)
# True: the window ends on the sample date, so rain falling AFTER a morning
# sample still counts toward its "preceding 24h". False: windows end the day
# before, which is strictly antecedent but ignores same-day rain entirely.
# Daily data cannot separate the two; this is the cost of GHCND.
RAIN_INCLUDE_SAME_DAY = True

CDO_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"

# NOAA CO-OPS (Tides & Currents). No token required.
COOPS_MDAPI = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json"
COOPS_DATA = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
# The 6-minute products cap a single request at 31 days, so a year is chunked.
COOPS_CHUNK_DAYS = 31
COOPS_DATUM = "MLLW"
MAX_TIDE_DISTANCE_KM = 50
# Below this rate of change the tide is treated as slack rather than
# rising/falling, so noise around high and low water is not read as direction.
TIDE_SLACK_M_PER_HR = 0.05


# ---------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------

def load_sites():
    if not os.path.exists(SITES_CSV):
        sys.exit(f"{SITES_CSV} not found — run find_candidate_sites.py first.")
    df = pd.read_csv(SITES_CSV)
    qualifying = df[df["has_all_four"]] if "has_all_four" in df.columns else df
    if qualifying.empty:
        sys.exit("No qualifying sites in the CSV.")
    print(f"{len(qualifying)} qualifying sites")
    return qualifying.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Buoy — hourly
# ---------------------------------------------------------------------------

# NDBC stdmet columns, named as in NDBC's own file header. Water temperature
# and mean wave direction are already part of this feed -- no extra source
# needed. MWD is only published by buoys with a directional wave sensor, so its
# absence is reported rather than passing silently.
STDMET_WANTED = {
    "WTMP": "water temperature",
    "MWD": "mean wave direction (swell direction)",
    "WVHT": "significant wave height",
    "DPD": "dominant wave period",
    "APD": "average wave period",
    "WDIR": "wind direction",
    "WSPD": "wind speed",
    "ATMP": "air temperature",
}


def report_stdmet_coverage(df, station_id):
    """Say which of the wanted measurements this buoy actually publishes."""
    for col, label in STDMET_WANTED.items():
        if col not in df.columns:
            print(f"      {col} ({label}): column absent")
            continue
        present = df[col].notna().sum()
        if present == 0:
            print(f"      {col} ({label}): column present but entirely empty")
        elif present < len(df) * 0.5:
            print(f"      {col} ({label}): only {present}/{len(df)} rows")


def pull_buoy(station_id, start, end):
    """Hourly standard meteorological data.

    Includes WTMP (water temperature) and MWD (mean wave direction) alongside
    wind and wave height, so swell direction and water temp need no separate
    source. All columns are kept; nothing is filtered out here.
    """
    try:
        df = NdbcApi().get_data(station_id=str(station_id), mode="stdmet",
                                start_time=start, end_time=end, as_df=True)
    except Exception as exc:
        print(f"    stdmet failed for {station_id}: {exc}")
        return None
    if df is None or len(df) == 0:
        print(f"    {station_id}: no stdmet rows. Stations with "
              "'Includes Meteorology' False publish no standard met feed.")
        return None
    return df


# ---------------------------------------------------------------------------
# Precipitation — daily
# ---------------------------------------------------------------------------

def pull_precip(station_id, start, end, token):
    """Daily PRCP totals in mm from GHCND."""
    rows, offset = [], 1
    while True:
        resp = requests.get(f"{CDO_BASE}/data", headers={"token": token}, timeout=120,
                            params={"datasetid": "GHCND", "stationid": station_id,
                                    "datatypeid": "PRCP", "units": "metric",
                                    "startdate": start.strftime("%Y-%m-%d"),
                                    "enddate": end.strftime("%Y-%m-%d"),
                                    "limit": 1000, "offset": offset})
        if resp.status_code == 429:
            print("      rate limited, backing off 10s")
            time.sleep(10)
            continue
        if resp.status_code in (400, 401, 403):
            print(f"      CDO refused ({resp.status_code}): {resp.text[:160]}")
            return None
        resp.raise_for_status()
        payload = resp.json()
        if not payload:
            break
        batch = payload.get("results", [])
        if not batch:
            break
        rows.extend(batch)
        total = payload.get("metadata", {}).get("resultset", {}).get("count")
        if total is not None and len(rows) >= total:
            break
        offset += 1000
        time.sleep(f.CDO_REQUEST_INTERVAL_S)

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[["date", "value"]].rename(columns={"value": "precip_mm"})


def add_rain_windows(df, start, end):
    """Rolling antecedent rainfall totals.

    Reindexes onto a complete daily calendar first. A day the station did not
    report is left as NaN rather than 0, so any window containing a gap comes
    out NaN instead of silently under-reporting the rainfall.
    """
    calendar = pd.date_range(pd.Timestamp(start).normalize(),
                             pd.Timestamp(end).normalize(), freq="D")
    daily = df.groupby("date", as_index=True)["precip_mm"].sum().reindex(calendar)
    out = pd.DataFrame({"date": daily.index, "precip_mm": daily.to_numpy()})

    source = daily if RAIN_INCLUDE_SAME_DAY else daily.shift(1)
    for days in RAIN_WINDOWS_DAYS:
        # min_periods=days => a window with any missing day yields NaN.
        out[f"rain_{days * 24}h_mm"] = (
            source.rolling(days, min_periods=days).sum().to_numpy())

    missing = int(out["precip_mm"].isna().sum())
    if missing:
        print(f"      {missing}/{len(out)} days have no observation; "
              "windows spanning them are NaN")
    return out


# ---------------------------------------------------------------------------
# Tides and coastal wind — NOAA CO-OPS, 6-minute
# ---------------------------------------------------------------------------

def nearest_serving(lat, lon, stations, max_km, tries=3):
    """The nearest stations by distance, closest first.

    Being listed as offering a product type does not mean a station serves it
    for the requested window -- Monterey is in the 'met' list but returns no
    wind -- so the caller needs more than one candidate to try.
    """
    if stations is None or stations.empty:
        return []
    d = f.haversine_km(lat, lon,
                       stations["lat"].to_numpy(dtype=float),
                       stations["lon"].to_numpy(dtype=float))
    ordered = stations.assign(_km=d).nsmallest(tries, "_km")
    return [(row, row["_km"]) for _, row in ordered.iterrows()
            if row["_km"] <= max_km]


def coops_stations(station_type):
    """Every CO-OPS station offering a product type ('waterlevels', 'met')."""
    resp = requests.get(COOPS_MDAPI, params={"type": station_type}, timeout=120)
    resp.raise_for_status()
    stations = resp.json().get("stations", [])
    df = pd.DataFrame([{"station_id": s.get("id"), "name": s.get("name"),
                        "lat": s.get("lat"), "lon": s.get("lng")}
                       for s in stations])
    print(f"    {len(df)} CO-OPS stations offering {station_type!r}")
    return df.dropna(subset=["lat", "lon"])


def pull_coops_series(station_id, product, start, end):
    """A CO-OPS product over the window, stitched from 31-day chunks."""
    frames, chunk_start = [], start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=COOPS_CHUNK_DAYS), end)
        params = {"product": product, "station": station_id,
                  "begin_date": chunk_start.strftime("%Y%m%d"),
                  "end_date": chunk_end.strftime("%Y%m%d"),
                  "time_zone": "gmt", "units": "metric", "format": "json",
                  "application": "coastal-pipeline"}
        if product == "water_level":
            params["datum"] = COOPS_DATUM

        resp = requests.get(COOPS_DATA, params=params, timeout=120)
        resp.raise_for_status()
        payload = resp.json()
        # CO-OPS answers 200 with an {"error": ...} body for a station that does
        # not carry the product, so status code alone proves nothing.
        if "error" in payload:
            print(f"      {product}: {payload['error'].get('message', '')[:120]}")
            return None
        rows = payload.get("data") or payload.get("predictions") or []
        if rows:
            frames.append(pd.DataFrame(rows))
        chunk_start = chunk_end
        time.sleep(0.2)

    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["t"])
    df["time"] = pd.to_datetime(df["t"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def add_tide_state(df):
    """Water level plus its rate of change and direction.

    Rate is computed against the real elapsed time between readings, not an
    assumed 6-minute step, so a gap in the record does not manufacture a huge
    apparent rate.
    """
    out = df[["time"]].copy()
    out["level_m"] = pd.to_numeric(df["v"], errors="coerce")

    hours = out["time"].diff().dt.total_seconds() / 3600.0
    out["rate_m_per_hr"] = out["level_m"].diff() / hours
    # A jump across a long gap is not a measured rate.
    out.loc[hours > 1.0, "rate_m_per_hr"] = pd.NA
    rate = pd.to_numeric(out["rate_m_per_hr"], errors="coerce")

    out["tide_state"] = pd.NA
    out.loc[rate > TIDE_SLACK_M_PER_HR, "tide_state"] = "rising"
    out.loc[rate < -TIDE_SLACK_M_PER_HR, "tide_state"] = "falling"
    out.loc[rate.abs() <= TIDE_SLACK_M_PER_HR, "tide_state"] = "slack"
    return out


def add_wind(df):
    out = df[["time"]].copy()
    out["wind_speed_m_s"] = pd.to_numeric(df.get("s"), errors="coerce")
    out["wind_dir_deg"] = pd.to_numeric(df.get("d"), errors="coerce")
    out["wind_gust_m_s"] = pd.to_numeric(df.get("g"), errors="coerce")
    if "dr" in df.columns:
        out["wind_dir_text"] = df["dr"]
    return out


# ---------------------------------------------------------------------------
# Water quality — irregular samples
# ---------------------------------------------------------------------------

def _sql(query):
    resp = requests.get(f"{f.CA_CKAN_BASE}/datastore_search_sql",
                        params={"sql": query}, timeout=300)
    if resp.status_code != 200 or not resp.json().get("success"):
        return None
    return pd.DataFrame(resp.json()["result"]["records"])


def canonical_station_id(value):
    """CKAN numeric ids arrive as int, but a CSV round-trip through a column
    that holds NaN for non-qualifying rows makes the column float64, turning
    101 into 101.0. Compare both sides on one canonical form."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value).strip()
    return str(int(number)) if number.is_integer() else str(number)


def station_code_map(site_station_ids):
    """Station_id -> StationCode, via the coordinate-validated join."""
    stations = pd.DataFrame(_fetch_all(f.CA_CKAN_RESOURCE_ID))
    distinct = _sql(f'SELECT DISTINCT "StationCode", "TargetLatitude", '
                    f'"TargetLongitude" FROM "{ckan_join.RESULTS_ID}"')
    if distinct is None:
        sys.exit("datastore_search_sql unavailable; cannot map stations to codes.")

    joined, stats = ckan_join.join_stations_to_results(stations, distinct)
    print(f"    join kept {stats['kept']}/{stats['joined']} pairs")

    wanted = {canonical_station_id(x) for x in site_station_ids}
    keys = joined[f.CA_CKAN_ID_COL].map(canonical_station_id)
    subset = joined[keys.isin(wanted)]
    if subset.empty and wanted:
        print(f"    no match. wanted {sorted(wanted)[:5]}, "
              f"join has {sorted(set(keys))[:5]}")
    return dict(zip(keys[keys.isin(wanted)], subset[ckan_join.RESULT_KEY].astype(str)))


def _fetch_all(resource_id):
    rows, offset = [], 0
    while True:
        r = requests.get(f"{f.CA_CKAN_BASE}/datastore_search", timeout=120,
                         params={"resource_id": resource_id, "limit": 1000,
                                 "offset": offset})
        r.raise_for_status()
        result = r.json().get("result", {})
        batch = result.get("records", [])
        if not batch:
            break
        rows.extend(batch)
        if len(rows) >= (result.get("total") or 0):
            break
        offset += 1000
    return rows


def pull_water_quality(codes, start):
    if not codes:
        return None
    quoted = ",".join("'" + c.replace("'", "''") + "'" for c in codes)
    query = (f'SELECT "StationCode","StationName","SampleDate","Analyte","Result",'
             f'"Unit","ResultQualCode","30DayGeoMean","6WeekGeoMean" '
             f'FROM "{ckan_join.RESULTS_ID}" WHERE "StationCode" IN ({quoted}) '
             f"""AND "SampleDate" >= '{start.strftime('%Y-%m-%d')}'""")
    return _sql(query)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    token = os.environ.get("NOAA_CDO_TOKEN")
    if not token:
        sys.exit("NOAA_CDO_TOKEN is not set. See .env.example.")
    os.makedirs(OUT_DIR, exist_ok=True)

    end = datetime.now()
    start = end - timedelta(days=DAYS_BACK)
    print(f"Window: {start:%Y-%m-%d} to {end:%Y-%m-%d}\n")

    sites = load_sites()

    print("\nBuoy (hourly stdmet)...")
    for buoy_id in sorted(set(sites["buoy_id"].dropna())):
        print(f"  {buoy_id}")
        df = pull_buoy(buoy_id, start, end)
        if df is not None:
            path = f"{OUT_DIR}/buoy_{buoy_id}.csv"
            df.to_csv(path)
            print(f"    {len(df)} rows -> {path}")
            report_stdmet_coverage(df, buoy_id)

    print("\nPrecipitation (daily GHCND + rolling windows)...")
    for station_id in sorted(set(sites["precip_station_id"].dropna())):
        print(f"  {station_id}")
        df = pull_precip(station_id, start, end, token)
        if df is None:
            print("    no data")
            continue
        windowed = add_rain_windows(df, start, end)
        path = f"{OUT_DIR}/precip_{station_id.replace(':', '_')}.csv"
        windowed.to_csv(path, index=False)
        print(f"    {len(windowed)} days -> {path}")

    print("\nTides and coastal wind (CO-OPS, 6-minute)...")
    tide_stations = coops_stations("waterlevels")
    met_stations = coops_stations("met")
    # Several sites share a gauge; pull each station/product once.
    done = set()

    def pull_once(station, product, transform, prefix):
        key = (station["station_id"], product)
        if key in done:
            print(f"      {product} already pulled for {station['station_id']}")
            return True
        raw = pull_coops_series(station["station_id"], product, start, end)
        if raw is None or raw.empty:
            return False
        done.add(key)
        path = f"{OUT_DIR}/{prefix}_{station['station_id']}.csv"
        out = transform(raw)
        out.to_csv(path, index=False)
        extra = ""
        if prefix == "tide":
            extra = f"  {out['tide_state'].value_counts().to_dict()}"
        print(f"      {len(out)} readings -> {path}{extra}")
        return True

    def water_temp(raw):
        out = raw[["time"]].copy()
        out["water_temp_c"] = pd.to_numeric(raw["v"], errors="coerce")
        return out

    for _, site in sites.iterrows():
        lat, lon = site["lat"], site["lon"]
        print(f"  {site['camera_name']}")

        for product, source, transform, prefix in (
                ("water_level", tide_stations, add_tide_state, "tide"),
                ("water_temperature", tide_stations, water_temp, "watertemp"),
                ("wind", met_stations, add_wind, "wind")):
            served = False
            for station, dist in nearest_serving(lat, lon, source,
                                                 MAX_TIDE_DISTANCE_KM):
                print(f"    {product}: trying {station['station_id']} "
                      f"{station['name']} ({dist:.1f} km)")
                if pull_once(station, product, transform, prefix):
                    served = True
                    break
            if not served:
                print(f"    no {product} within {MAX_TIDE_DISTANCE_KM} km")

    print("\nWater quality (bacteria samples)...")
    codes = station_code_map(sites["wq_station_id"].dropna())
    print(f"    {len(codes)} of {len(sites)} sites mapped to a results StationCode")
    for station_id, code in codes.items():
        print(f"      Station_id {station_id} -> StationCode {code}")
    wq = pull_water_quality(list(codes.values()), start)
    if wq is None or wq.empty:
        print("    no samples in the window")
    else:
        path = f"{OUT_DIR}/water_quality.csv"
        wq.to_csv(path, index=False)
        print(f"    {len(wq)} samples -> {path}")
        print(f"    analytes: {sorted(wq['Analyte'].unique())}")

    print(f"\nDone. CSVs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
