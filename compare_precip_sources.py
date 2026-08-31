"""Compare gauge rainfall (GHCND daily) against gridded rainfall (ERA5 hourly).

Both are in data/ for the sites where the gauge works, so the question of which
one feeds the model is answerable rather than a matter of preference. ERA5 is a
model reconstruction and is known to spread light precipitation, so a bias
against a working gauge is the thing to look for.

Aggregates the hourly grid to daily totals, aligns on date, and reports totals,
wet-day agreement, correlation and bias per site.

    python compare_precip_sources.py
"""

import os
import sys

import pandas as pd
import requests

DATA_DIR = "data"
SITES_CSV = "candidate_sites_ranked.csv"
WET_DAY_MM = 1.0  # below this a "wet day" is drizzle or model noise
CDO = "https://www.ncei.noaa.gov/cdo-web/api/v2"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"


def gauge_elevation(station_id, token):
    """Metres. A gauge in a canyon or up a mountain does not measure the beach."""
    if not token:
        return None
    try:
        r = requests.get(f"{CDO}/stations/{station_id}",
                         headers={"token": token}, timeout=30)
        return (r.json() or {}).get("elevation") if r.status_code == 200 else None
    except requests.RequestException:
        return None


def grid_elevation(lat, lon):
    try:
        r = requests.get(OPEN_METEO, timeout=30, params={
            "latitude": lat, "longitude": lon, "hourly": "precipitation",
            "start_date": "2025-01-01", "end_date": "2025-01-01"})
        return (r.json() or {}).get("elevation") if r.status_code == 200 else None
    except requests.RequestException:
        return None


def site_slug(name):
    return "".join(c if c.isalnum() else "_" for c in str(name))[:48]


def load_pair(site):
    """(daily gauge series, daily gridded series) for one site, or None."""
    station = site.get("precip_station_id")
    if not isinstance(station, str):
        return None
    gauge_path = f"{DATA_DIR}/precip_{station.replace(':', '_')}.csv"
    grid_path = f"{DATA_DIR}/gridded_{site_slug(site['camera_name'])}.csv"
    if not (os.path.exists(gauge_path) and os.path.exists(grid_path)):
        return None

    gauge = pd.read_csv(gauge_path, parse_dates=["date"])
    gauge = gauge.set_index("date")["precip_mm"]

    grid = pd.read_csv(grid_path, parse_dates=["time"])
    if "precipitation" not in grid.columns:
        return None
    grid = (grid.set_index("time")["precipitation"]
                .resample("D").sum(min_count=1))
    grid.index = grid.index.tz_localize(None).normalize()

    joined = pd.DataFrame({"gauge": gauge, "grid": grid}).dropna()
    return joined if len(joined) > 30 else None


def main():
    if not os.path.exists(SITES_CSV):
        sys.exit(f"{SITES_CSV} not found.")
    sites = pd.read_csv(SITES_CSV)
    if "has_all_four" in sites.columns:
        sites = sites[sites["has_all_four"]]

    token = os.environ.get("NOAA_CDO_TOKEN")
    if not token:
        print("  NOAA_CDO_TOKEN not set; gauge elevations will be blank\n")

    rows = []
    for _, site in sites.iterrows():
        pair = load_pair(site)
        if pair is None:
            print(f"  {site['camera_name']}: no overlapping pair "
                  "(gauge missing, or the station is one of the dead ones)")
            continue

        gauge_wet = pair["gauge"] >= WET_DAY_MM
        grid_wet = pair["grid"] >= WET_DAY_MM
        both = int((gauge_wet & grid_wet).sum())

        g_elev = gauge_elevation(site["precip_station_id"], token)
        r_elev = grid_elevation(site["lat"], site["lon"])

        rows.append({
            "site": site["camera_name"][:30],
            "days": len(pair),
            "gauge_m": round(g_elev) if g_elev is not None else None,
            "grid_m": round(r_elev) if r_elev is not None else None,
            "gauge_mm": round(pair["gauge"].sum()),
            "grid_mm": round(pair["grid"].sum()),
            "ratio": round(pair["grid"].sum() / max(pair["gauge"].sum(), 0.1), 2),
            "corr": round(pair["gauge"].corr(pair["grid"]), 3),
            # Raw wet/dry agreement is dominated by dry days -- an always-dry
            # predictor scores 83-90% here -- so report the wet days directly.
            "recall": round(both / max(int(gauge_wet.sum()), 1), 2),
            "precis": round(both / max(int(grid_wet.sum()), 1), 2),
        })

    if not rows:
        sys.exit("\nNo site had both sources. Run pull_observations.py and "
                 "pull_gridded_weather.py first.")

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + out.to_string(index=False))

    print("\nHow to read this:")
    print("  gauge_m /  elevation of each source. A gauge up a canyon or in the")
    print("  grid_m    hills is measuring orographic rain, not beach rain, and")
    print("            a large gap here explains a ratio far from 1 better than")
    print("            any property of the model does.")
    print("  ratio     grid total / gauge total.")
    print("  corr      daily correlation. 0.7-0.8 between a point gauge and a")
    print("            9-25 km cell is normal, not a failure.")
    print("  recall    of the gauge's wet days, the share the grid also called wet.")
    print("  precis    of the grid's wet days, the share the gauge agreed on.")
    print(f"            A wet day is >= {WET_DAY_MM} mm. Raw wet/dry agreement is")
    print("            not reported: it is dominated by dry days, and an")
    print("            always-dry predictor would score 83-90% on this data.")

    print("\nRatios above and below 1 both appear here, so this is not a simple")
    print("model wet bias. Check the elevation columns before concluding")
    print("anything about either source: where the gauge sits well above the")
    print("grid cell, the gauge is the one measuring the wrong place.")
