"""How many Open-Meteo cells this run still has to buy.

ERA5 and the wave model are cached per 0.1-degree cell, and Open-Meteo's
free tier is metered by variables x days, not by requests -- one cell over
this project's window is worth roughly a hundred of its call units. So the
number that decides whether a re-run fits in one night is not the station
count, it is the count of cells not already cached. Run this before
spending a day's quota.
"""
import os, sys
import pandas as pd
sys.path.insert(0, ".")
from wq import config
from wq.covariates import cell_key, CACHE_DIR, _EMPTY_MARKER

stations = pd.read_csv(os.path.join(config.OUT_DIR, "stations.csv"))
meta_path = os.path.join(config.OUT_DIR, "covariate_sources.csv")
if os.path.exists(meta_path):
    keep = set(pd.read_csv(meta_path)["station_id"].astype(str))
    stations = stations[stations["station_id"].astype(str).isin(keep)]
    label = "fitted stations"
else:
    label = "all stations"

good = stations.dropna(subset=["lat", "lon"])
cells = {cell_key(r.lat, r.lon) for r in good.itertuples()}
print(f"{len(good)} {label} -> {len(cells)} distinct 0.1-deg cells")

def survey(prefix):
    real = empty = 0
    for key in cells:
        path = os.path.join(CACHE_DIR, f"{prefix}_{key}.csv")
        if not os.path.exists(path):
            continue
        try:
            head = open(path).readline()
        except OSError:
            continue
        if _EMPTY_MARKER in head:
            empty += 1
        else:
            real += 1
    print(f"  {prefix:7s} cached with data: {real:4d}   cached as absent: {empty:4d}"
          f"   still to fetch: {len(cells) - real - empty:4d}")

survey("era5")
survey("marine")
