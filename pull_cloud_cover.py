"""Fetch ONLY cloud cover, into a sidecar file, for the glare/haze test.

analyze_glare.py needs total cloud cover and the gridded_*.csv files predate it
being requested. The obvious fix -- re-run pull_site_observations.py -- has cost
two failed attempts to HTTP 429, and it is the wrong shape of request anyway:

  * Open-Meteo charges by variable-hours, not by call. Walton's window is 20,832
    hours; asking for six variables to obtain one is six times the quota for the
    same answer, which is most of why the re-pull kept being refused.
  * A re-pull rewrites gridded_*.csv, marine_*.csv, tide_*.csv and
    watertemp_*.csv. Every one of those is currently correct. Putting four good
    files at risk to add a column to a fifth is a bad trade, and narrowing one
    of them is the exact bug that has already cost this project two analyses.

So this writes data/cloud_<slug>.csv and touches nothing else. The window is
read from the site's existing gridded file, so the two line up by construction
and there is no date for anyone to get wrong.

    python pull_cloud_cover.py --camera "Walton Lighthouse, Santa Cruz, CA"
    python pull_cloud_cover.py --all-rip

A refusal writes nothing, so a rate-limited run is safe to repeat.
"""

import argparse
import sys

import pandas as pd

import analyze_drivers as ad
import pull_site_observations as pso

VARIABLE = "cloud_cover"


def existing_window(camera_name):
    """(start, end) of the site's ERA5 file, so the sidecar matches it exactly."""
    frame = ad.read_csv(f"{ad.DATA_DIR}/gridded_{ad.grid_slug(camera_name)}.csv")
    if frame is None or frame.empty:
        return None, None
    column = "time" if "time" in frame.columns else frame.columns[0]
    stamps = pd.to_datetime(frame[column], utc=True, errors="coerce").dropna()
    if stamps.empty:
        return None, None
    return stamps.min(), stamps.max()


def pull(name, lat, lon, start, end):
    path = f"{ad.DATA_DIR}/cloud_{ad.grid_slug(name)}.csv"
    print(f"\n{name}  ({lat:.4f}, {lon:.4f})")
    print(f"  window {start:%Y-%m-%d} to {end:%Y-%m-%d}, matching the ERA5 file")
    frame, note = pso.open_meteo(pso.ERA5, lat, lon, start, end, [VARIABLE])
    if frame is None:
        print(f"  FAILED: {note}")
        print("  Nothing written; the run is safe to repeat once the quota resets.")
        return False
    if VARIABLE not in frame.columns:
        print(f"  the service answered without a {VARIABLE} column; nothing written")
        return False
    keep = frame[["time", VARIABLE]]
    share = float(pd.to_numeric(keep[VARIABLE], errors="coerce").notna().mean())
    keep.to_csv(path, index=False)
    print(f"  {len(keep)} hours -> {path}  ({share:.0%} populated)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", action="append",
                        help="substring of a camera name; repeatable")
    parser.add_argument("--all-rip", action="store_true",
                        help="every site that has a gridded file on disk")
    parser.add_argument("--start", help="override the window start (YYYY-MM-DD)")
    parser.add_argument("--end", help="override the window end (YYYY-MM-DD)")
    args = parser.parse_args()

    sites = ad.load_sites()
    if args.all_rip:
        wanted = list(sites["camera_name"])
    elif args.camera:
        wanted = []
        for needle in args.camera:
            hits = [s for s in sites["camera_name"]
                    if needle.lower() in str(s).lower()]
            if not hits:
                print(f"no camera matching {needle!r}")
            wanted += hits
    else:
        parser.error("give --camera or --all-rip")

    done = 0
    for name in dict.fromkeys(wanted):
        row = next((s for _, s in sites.iterrows()
                    if s["camera_name"] == name), None)
        if row is None or pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
            continue
        start, end = existing_window(name)
        if args.start:
            start = pd.Timestamp(args.start, tz="UTC")
        if args.end:
            end = pd.Timestamp(args.end, tz="UTC")
        if start is None or end is None:
            print(f"\n{name}: no gridded file to take a window from; "
                  "pass --start and --end, or run pull_site_observations.py")
            continue
        done += pull(name, float(row["lat"]), float(row["lon"]), start, end)
    print(f"\n{done} site(s) written.")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
