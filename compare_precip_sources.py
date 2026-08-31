"""Compare gauge rainfall (GHCND daily) against gridded rainfall (ERA5 hourly).

Both are in data/ for the sites where the gauge works, so the question of which
one feeds the model is answerable rather than a matter of preference. ERA5 is a
model reconstruction and is known to spread light precipitation, so a bias
against a working gauge is the thing to look for.

Aggregates the hourly grid to daily totals, aligns on date, and reports totals,
wet-day agreement, correlation and bias per site.

    python compare_precip_sources.py
"""

import glob
import os
import sys

import pandas as pd

DATA_DIR = "data"
SITES_CSV = "candidate_sites_ranked.csv"
WET_DAY_MM = 1.0  # below this a "wet day" is drizzle or model noise


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

    rows = []
    for _, site in sites.iterrows():
        pair = load_pair(site)
        if pair is None:
            print(f"  {site['camera_name']}: no overlapping pair "
                  "(gauge missing, or the station is one of the dead ones)")
            continue

        gauge_wet = pair["gauge"] >= WET_DAY_MM
        grid_wet = pair["grid"] >= WET_DAY_MM
        rows.append({
            "site": site["camera_name"][:34],
            "days": len(pair),
            "gauge_mm": round(pair["gauge"].sum()),
            "grid_mm": round(pair["grid"].sum()),
            "ratio": round(pair["grid"].sum() / max(pair["gauge"].sum(), 0.1), 2),
            "gauge_wet_d": int(gauge_wet.sum()),
            "grid_wet_d": int(grid_wet.sum()),
            "corr": round(pair["gauge"].corr(pair["grid"]), 3),
            # Do they agree on which days were wet at all?
            "agree_pct": round(100 * (gauge_wet == grid_wet).mean()),
        })

    if not rows:
        sys.exit("\nNo site had both sources. Run pull_observations.py and "
                 "pull_gridded_weather.py first.")

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + out.to_string(index=False))

    print("\nHow to read this:")
    print("  ratio      grid total / gauge total. Near 1.0 means they agree on")
    print("             how much rain fell. Much above 1 means ERA5 is wetter,")
    print("             which is its known failure mode.")
    print("  corr       daily correlation. High corr with ratio far from 1")
    print("             means the timing is right and only the amount is off,")
    print("             which a scale factor fixes.")
    print("  agree_pct  share of days both call wet or both call dry.")
    print(f"             A wet day here is >= {WET_DAY_MM} mm.")

    median_ratio = out["ratio"].median()
    median_corr = out["corr"].median()
    print(f"\nMedian ratio {median_ratio}, median daily correlation {median_corr}.")
    if median_corr >= 0.8 and 0.8 <= median_ratio <= 1.25:
        print("The two agree closely. Use the gridded source: same signal, "
              "hourly resolution, and no station-liveness problem.")
    elif median_corr >= 0.8:
        print("Timing agrees but totals do not. The gridded source is usable "
              "for antecedent-rainfall features, where relative magnitude "
              "matters more than absolute mm, but do not treat its mm as gauge mm.")
    else:
        print("They disagree on timing, not just amount. Prefer the gauge where "
              "one is live, and inspect a few storms by hand before trusting "
              "either at these sites.")


if __name__ == "__main__":
    main()
