"""Is the wind data any good?

Two sources, and neither is uncomplicated:

  CO-OPS   6-minute, real observations, but the station can be 16-18 km away
           in different terrain, and Monterey serves no wind at all -- so the
           three Santa Cruz sites have no observed wind whatsoever.
  ERA5     hourly, at every site, but model output on a 9-25 km cell.

Direction is a circular quantity. An ordinary mean or correlation on degrees
treats 359 and 1 as almost opposite, so this uses vector components throughout:
6-minute readings are averaged as u/v, and the comparison reports mean absolute
circular difference rather than a correlation coefficient.

    python compare_wind_sources.py
"""

import env  # noqa: F401  -- loads .env into os.environ

import os
import sys

import numpy as np
import pandas as pd

import pull_observations as po

DATA_DIR = "data"
SITES_CSV = "candidate_sites_ranked.csv"
# Below this, direction is meaningless -- a calm has no direction to compare.
CALM_MS = 1.0


def to_uv(speed, direction_deg):
    """Meteorological direction (where wind comes FROM) to u/v components."""
    rad = np.radians(direction_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def from_uv(u, v):
    speed = np.sqrt(u ** 2 + v ** 2)
    direction = (np.degrees(np.arctan2(-u, -v))) % 360
    return speed, direction


def circular_diff(a, b):
    """Smallest signed angle from b to a, in degrees, within [-180, 180)."""
    return (a - b + 180) % 360 - 180


def hourly_from_coops(df):
    """6-minute CO-OPS wind to hourly, averaging direction as vectors."""
    df = df.dropna(subset=["wind_speed_m_s", "wind_dir_deg"]).copy()
    if df.empty:
        return None
    u, v = to_uv(df["wind_speed_m_s"].to_numpy(), df["wind_dir_deg"].to_numpy())
    df["u"], df["v"] = u, v
    hourly = df.set_index("time")[["u", "v", "wind_speed_m_s"]].resample("h").mean()
    speed_vec, direction = from_uv(hourly["u"], hourly["v"])
    return pd.DataFrame({
        # Scalar mean is the fair comparison against a model hourly mean;
        # vector speed is lower whenever the direction swings within the hour.
        "speed": hourly["wind_speed_m_s"],
        "speed_vec": speed_vec,
        "dir": direction,
    })


def coops_wind_for(site, met_stations):
    """The CO-OPS wind file pull_observations would have used, if any."""
    for station, dist in po.nearest_serving(site["lat"], site["lon"],
                                            met_stations, po.MAX_TIDE_DISTANCE_KM):
        path = f"{DATA_DIR}/wind_{station['station_id']}.csv"
        if os.path.exists(path):
            return path, station, dist
    return None, None, None


def main():
    if not os.path.exists(SITES_CSV):
        sys.exit(f"{SITES_CSV} not found.")
    sites = pd.read_csv(SITES_CSV)
    if "has_all_four" in sites.columns:
        sites = sites[sites["has_all_four"]]

    met_stations = po.coops_stations("met")
    rows, unobserved = [], []

    for _, site in sites.iterrows():
        name = site["camera_name"][:30]
        grid_path = (f"{DATA_DIR}/gridded_"
                     f"{''.join(c if c.isalnum() else '_' for c in str(site['camera_name']))[:48]}.csv")
        if not os.path.exists(grid_path):
            print(f"  {name}: no gridded file")
            continue

        coops_path, station, dist = coops_wind_for(site, met_stations)
        if coops_path is None:
            unobserved.append(name)
            continue

        coops = pd.read_csv(coops_path, parse_dates=["time"])
        coops_h = hourly_from_coops(coops)
        if coops_h is None:
            unobserved.append(name)
            continue

        grid = pd.read_csv(grid_path, parse_dates=["time"])
        if "wind_speed_10m" not in grid.columns:
            print(f"  {name}: gridded file has no wind column")
            continue
        grid = grid.set_index("time")[["wind_speed_10m", "wind_direction_10m"]]

        pair = coops_h.join(grid, how="inner").dropna()
        if len(pair) < 100:
            print(f"  {name}: only {len(pair)} overlapping hours")
            continue

        # Direction only means something when it is actually blowing.
        blowing = pair[(pair["speed"] >= CALM_MS)
                       & (pair["wind_speed_10m"] >= CALM_MS)]
        dir_err = circular_diff(blowing["dir"].to_numpy(),
                                blowing["wind_direction_10m"].to_numpy())

        rows.append({
            "site": name,
            "station": station["station_id"],
            "km": round(dist, 1),
            "hours": len(pair),
            "obs_ms": round(pair["speed"].mean(), 2),
            "grid_ms": round(pair["wind_speed_10m"].mean(), 2),
            "spd_corr": round(pair["speed"].corr(pair["wind_speed_10m"]), 3),
            "dir_err": round(float(np.mean(np.abs(dir_err))), 1),
            "dir_med": round(float(np.median(np.abs(dir_err))), 1),
            "within45": round(100 * float(np.mean(np.abs(dir_err) <= 45))),
        })

    if unobserved:
        print("\nNO OBSERVED WIND AT ALL — gridded is the only source, and there")
        print("is nothing to validate it against at these sites:")
        for name in unobserved:
            print(f"  {name}")

    if not rows:
        sys.exit("\nNo site had both sources.")

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\n" + out.to_string(index=False))

    print("\nHow to read this:")
    print("  km        distance to the observing station. Wind decorrelates over")
    print("            distance far faster than rainfall does.")
    print("  spd_corr  hourly wind speed correlation.")
    print("  dir_err   mean absolute circular difference in direction, degrees,")
    print(f"            over hours where both read >= {CALM_MS} m/s. Computed on")
    print("            vectors, so 359 vs 1 is 2 degrees apart, not 358.")
    print("  within45  share of those hours agreeing to within 45 degrees, i.e.")
    print("            roughly the same compass octant.")

    print(f"\nMedian speed correlation {out['spd_corr'].median()}, "
          f"median direction error {out['dir_err'].median()} degrees.")
    print("For onshore/offshore -- which is what matters for a surf zone -- the")
    print("within45 column is the one to judge on, not spd_corr.")


if __name__ == "__main__":
    main()
