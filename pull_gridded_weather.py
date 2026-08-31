"""Hourly rainfall and wind for each site from Open-Meteo's ERA5 archive.

This exists because the station data has two gaps that are not fixable by
picking a better station:

  * GHCND is daily, so a true "rainfall in the 24h before this sample" is not
    computable from it. Here it is, because the source is hourly.
  * No buoy in the matched set publishes wind, and CO-OPS wind is missing at
    some gauges (Monterey serves none). A gridded product has no station gaps.

The trade-off is real and worth stating: this is ERA5 reanalysis, a model
reconstruction on a grid of roughly 9-25 km, not a rain gauge reading. For
antecedent rainfall it is usually the better input than a gauge 20 km away, but
it is not an observation. Keep the GHCND pull alongside it and compare.

No API key. Free for non-commercial use.

    python pull_gridded_weather.py
    python pull_gridded_weather.py --probe    one site, print the raw schema
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import requests

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
SITES_CSV = "candidate_sites_ranked.csv"
OUT_DIR = "data"
DAYS_BACK = 365

# Requested hourly variables. Verified against the response rather than assumed:
# any that do not come back are reported instead of silently becoming NaN.
HOURLY_VARS = [
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "temperature_2m",
]

# True hourly windows now, not calendar days.
RAIN_WINDOWS_HOURS = (24, 48, 72)


def fetch(lat, lon, start, end, probe=False):
    params = {"latitude": lat, "longitude": lon,
              "start_date": start.strftime("%Y-%m-%d"),
              "end_date": end.strftime("%Y-%m-%d"),
              "hourly": ",".join(HOURLY_VARS),
              "timezone": "UTC",
              # Open-Meteo defaults to km/h. NDBC stdmet WSPD and CO-OPS wind
              # are both m/s, so ask for m/s rather than leaving two wind
              # sources in different units for something downstream to mix.
              "wind_speed_unit": "ms"}
    resp = requests.get(ARCHIVE, params=params, timeout=300)
    if resp.status_code != 200:
        print(f"    HTTP {resp.status_code}: {resp.text[:300]}")
        return None
    payload = resp.json()
    if "error" in payload or payload.get("error"):
        print(f"    API error: {str(payload.get('reason') or payload)[:300]}")
        return None

    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        print(f"    unexpected response shape, top-level keys: "
              f"{list(payload.keys())}")
        return None

    units = payload.get("hourly_units") or {}
    for var in ("wind_speed_10m", "wind_gusts_10m"):
        if units.get(var) not in (None, "m/s"):
            print(f"    WARNING: {var} came back in {units[var]}, not m/s. "
                  "The wind_speed_unit request was not honoured; do not mix "
                  "this with the CO-OPS or NDBC wind without converting.")

    if probe:
        print(f"    top-level keys : {list(payload.keys())}")
        print(f"    hourly keys    : {list(hourly.keys())}")
        print(f"    units          : {payload.get('hourly_units')}")
        print(f"    grid cell at   : {payload.get('latitude')}, "
              f"{payload.get('longitude')} "
              f"(requested {lat}, {lon})")

    missing = [v for v in HOURLY_VARS if v not in hourly]
    if missing:
        print(f"    requested but not returned: {missing}")

    df = pd.DataFrame({k: v for k, v in hourly.items()})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def add_rain_windows(df):
    """Rolling rainfall over true clock hours.

    Reindexed onto a complete hourly range first, so a gap in the series yields
    NaN for any window spanning it rather than a quiet under-count -- the same
    rule the daily GHCND path uses.
    """
    if "precipitation" not in df.columns:
        print("    no precipitation column; skipping rain windows")
        return df

    series = df.set_index("time")["precipitation"]
    full = pd.date_range(series.index.min(), series.index.max(), freq="h")
    series = series.reindex(full)

    out = df.set_index("time").reindex(full)
    out.index.name = "time"
    for hours in RAIN_WINDOWS_HOURS:
        out[f"rain_{hours}h_mm"] = series.rolling(hours, min_periods=hours).sum()

    gaps = int(series.isna().sum())
    if gaps:
        print(f"    {gaps}/{len(full)} hours missing; windows spanning them are NaN")
    return out.reset_index()


def main():
    probe = "--probe" in sys.argv
    if not os.path.exists(SITES_CSV):
        sys.exit(f"{SITES_CSV} not found — run find_candidate_sites.py first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    sites = pd.read_csv(SITES_CSV)
    if "has_all_four" in sites.columns:
        sites = sites[sites["has_all_four"]]
    if probe:
        sites = sites.head(1)

    end = datetime.now() - timedelta(days=6)  # the archive lags a few days
    start = end - timedelta(days=DAYS_BACK)
    print(f"Window: {start:%Y-%m-%d} to {end:%Y-%m-%d}\n")

    for _, site in sites.iterrows():
        name = site["camera_name"]
        print(f"  {name}")
        df = fetch(site["lat"], site["lon"], start, end, probe=probe)
        if df is None:
            continue
        df = add_rain_windows(df)
        slug = "".join(c if c.isalnum() else "_" for c in str(name))[:48]
        path = f"{OUT_DIR}/gridded_{slug}.csv"
        df.to_csv(path, index=False)
        print(f"    {len(df)} hours -> {path}")
        if "precipitation" in df.columns:
            total = df["precipitation"].sum()
            wet = int((df["precipitation"] > 0).sum())
            print(f"    {total:.0f} mm total, {wet} wet hours of {len(df)}")

    print(f"\nDone. CSVs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
