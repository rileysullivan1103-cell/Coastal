"""How many Open-Meteo cells a re-run still has to buy.

ERA5 and the wave model are cached per 0.1-degree cell, and Open-Meteo's
free tier is metered by variables x days rather than by requests -- one cell
over this project's window is worth roughly a hundred of its call units. So
the number that decides whether a re-run fits in one night is not the
station count, it is the count of cells not already cached.

This looks for its inputs rather than assuming a layout, because a working
copy's data directory does not always sit where config says it should, and a
planning tool that dies on a path is worth nothing. It prints what it used.
"""
import os
import sys
import glob

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wq import config  # noqa: E402
from wq.covariates import cell_key, _EMPTY_MARKER  # noqa: E402


def find(name, extra_roots=()):
    """The newest file called `name` under any plausible root, or None."""
    roots = [config.OUT_DIR, config.DATA_DIR, os.getcwd()]
    roots.extend(extra_roots)
    seen = []
    for root in roots:
        if root and os.path.isdir(root):
            seen.extend(glob.glob(os.path.join(root, "**", name),
                                  recursive=True))
    if not seen:
        return None
    return max(set(seen), key=os.path.getmtime)


def find_cache():
    """The directory holding era5_*.csv, wherever it ended up."""
    for root in (config.RAW_DIR, config.DATA_DIR, os.getcwd()):
        if not root or not os.path.isdir(root):
            continue
        hits = glob.glob(os.path.join(root, "**", "era5_*.csv"), recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return None


def main():
    stations_path = find("stations.csv")
    if stations_path is None:
        sys.exit("no stations.csv found under data/ -- run --stations first, "
                 "or run this from the directory holding data/")
    print(f"stations : {stations_path}")
    stations = pd.read_csv(stations_path)

    meta_path = find("covariate_sources.csv")
    if meta_path is not None and "station_id" in stations.columns:
        print(f"fitted   : {meta_path}")
        keep = set(pd.read_csv(meta_path)["station_id"].astype(str))
        subset = stations[stations["station_id"].astype(str).isin(keep)]
        if not subset.empty:
            stations, label = subset, "fitted stations"
        else:
            label = "all stations (no station_id overlap with the meta table)"
    else:
        label = "all stations (no covariate_sources.csv yet)"

    cache = find_cache()
    print(f"cache    : {cache or '(none found -- nothing is cached yet)'}")
    print()

    lat = next((c for c in ("lat", "latitude") if c in stations.columns), None)
    lon = next((c for c in ("lon", "longitude") if c in stations.columns), None)
    if not lat or not lon:
        sys.exit(f"stations.csv has no lat/lon columns; it has {list(stations.columns)}")

    good = stations.dropna(subset=[lat, lon])
    cells = {cell_key(r, n) for r, n in zip(good[lat], good[lon])}
    print(f"{len(good)} {label} -> {len(cells)} distinct 0.1-deg cells")

    if cache is None:
        print(f"  every cell is still to fetch: {len(cells)}")
        return

    for prefix in ("era5", "marine"):
        data = absent = 0
        for key in cells:
            path = os.path.join(cache, f"{prefix}_{key}.csv")
            if not os.path.exists(path):
                continue
            try:
                with open(path) as handle:
                    head = handle.readline()
            except OSError:
                continue
            if _EMPTY_MARKER in head:
                absent += 1
            else:
                data += 1
        todo = len(cells) - data - absent
        print(f"  {prefix:6s} cached with data {data:5d}   "
              f"cached as absent {absent:5d}   STILL TO FETCH {todo:5d}")


if __name__ == "__main__":
    main()
