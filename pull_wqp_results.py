"""Bacteria sample RESULTS from the Water Quality Portal, for any US site.

Everything bacteria-related in this project so far has come from California's
CKAN portal, which has no equivalent in Virginia, Michigan or North Carolina.
The Water Quality Portal is the national source. scan_cameras.py has used it to
locate stations, but no result has ever been pulled from it here, so the column
names below follow WQP's documented convention and are UNVERIFIED. The script
checks them against the first real response and stops with the actual names
rather than matching on columns that do not exist.

    python pull_wqp_results.py --camera "Virginia Beach" --probe   # columns only
    python pull_wqp_results.py --camera "Virginia Beach"
    python pull_wqp_results.py --lat 36.83 --lon -75.97 --name "VB" --km 3

One thing this gains over the California path: WQP carries a sample TIME, not
just a date. The CKAN results did not, which forced every condition to be a
daily mean — 'the tide on the day of the sample' rather than at the moment of
sampling. Where the time is populated here, that limitation goes away.

Writes data/wqp_<slug>.csv.
"""

import env  # noqa: F401  -- loads .env into os.environ

import argparse
import os
import sys
from io import StringIO

import pandas as pd
import requests

import scan_cameras as scan

RESULT_URL = "https://www.waterqualitydata.us/data/Result/search"
CANDIDATES_CSV = "camera_candidates.csv"
OUT_DIR = "data"

CHARACTERISTICS = "Escherichia coli;Enterococcus;Fecal Coliform;Total Coliform"
DEFAULT_KM = 2
DEFAULT_SINCE = "01-01-2023"  # MM-DD-YYYY, as WQP wants

# WQP's documented result columns. Unverified — see the module docstring.
COL_STATION = "MonitoringLocationIdentifier"
COL_DATE = "ActivityStartDate"
COL_TIME = "ActivityStartTime/Time"
COL_TZ = "ActivityStartTime/TimeZoneCode"
COL_ANALYTE = "CharacteristicName"
COL_VALUE = "ResultMeasureValue"
COL_UNIT = "ResultMeasure/MeasureUnitCode"
COL_NONDETECT = "ResultDetectionConditionText"

REQUIRED = (COL_STATION, COL_DATE, COL_ANALYTE, COL_VALUE)


def slugify(text):
    """Identical to pull_site_observations.slugify, so a site's condition
    files and its bacteria file carry the same name."""
    return "".join(c if c.isalnum() else "_" for c in str(text))[:48]


def fetch(lat, lon, km, since):
    min_lon, min_lat, max_lon, max_lat = scan.bbox_around(lat, lon, km)
    params = {
        "bBox": f"{min_lon:.4f},{min_lat:.4f},{max_lon:.4f},{max_lat:.4f}",
        "characteristicName": CHARACTERISTICS,
        "startDateLo": since,
        "mimeType": "csv",
        "zip": "no",
    }
    print(f"  bBox {params['bBox']}  (lon-first, {km} km around the site)")
    resp = scan.get_with_retry(RESULT_URL, params=params)
    print(f"  HTTP {resp.status_code}, {len(resp.content) / 1000:.0f} kB")
    if resp.status_code == 400:
        sys.exit(f"WQP rejected the request:\n{resp.text[:400]}")
    resp.raise_for_status()
    if resp.content[:2] == b"PK":
        sys.exit("Got a zip back despite zip=no.")
    text = resp.text.strip()
    if not text:
        sys.exit("Empty response — no results in that box since " + since)
    return pd.read_csv(StringIO(text), low_memory=False)


def check_columns(frame, probe):
    print(f"\n  {len(frame)} rows, {len(frame.columns)} columns")
    if probe:
        print("\n=== ACTUAL COLUMNS ===")
        for col in frame.columns:
            sample = frame[col].dropna()
            example = "" if sample.empty else str(sample.iloc[0])[:44]
            print(f"  {col:<46} e.g. {example}")

    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        print("\n=== COLUMNS THIS SCRIPT NEEDS ===")
        for col in REQUIRED:
            print(f"  {col:<46} {'PRESENT' if col in frame.columns else 'MISSING'}")
        print("\nSimilar columns that ARE present:")
        for col in frame.columns:
            if any(w in col.lower() for w in
                   ("identifier", "date", "time", "characteristic", "result", "unit")):
                print(f"  {col}")
        sys.exit(f"\n{len(missing)} required column(s) missing — update the "
                 "COL_* constants at the top of pull_wqp_results.py.")
    print("  all required columns present")


def build(frame, name):
    out = pd.DataFrame({
        "station": frame[COL_STATION].astype(str),
        "analyte": frame[COL_ANALYTE].astype(str),
        "value_raw": frame[COL_VALUE],
        "unit": frame[COL_UNIT].astype(str) if COL_UNIT in frame else "",
    })
    out["value"] = pd.to_numeric(frame[COL_VALUE], errors="coerce")

    # WQP records a non-detect as a blank value plus a condition text. Dropping
    # those silently biases the low end, so they are counted and flagged rather
    # than quietly discarded.
    if COL_NONDETECT in frame.columns:
        out["nondetect"] = frame[COL_NONDETECT].notna()
    else:
        out["nondetect"] = False

    stamp = frame[COL_DATE].astype(str)
    has_time = COL_TIME in frame.columns and frame[COL_TIME].notna().any()
    if has_time:
        times = frame[COL_TIME].fillna("").astype(str)
        stamp = stamp.str.cat(times, sep=" ").str.strip()
    out["sampled_at"] = pd.to_datetime(stamp, errors="coerce", utc=True)
    out["has_sample_time"] = has_time and frame[COL_TIME].notna()
    out["site"] = name
    return out.sort_values("sampled_at").reset_index(drop=True)


def summarize(out):
    print(f"\n{'=' * 74}\nRESULTS\n{'=' * 74}")
    dated = out["sampled_at"].notna()
    print(f"{len(out)} results from {out['station'].nunique()} stations")
    if dated.any():
        print(f"{out.loc[dated, 'sampled_at'].min():%Y-%m-%d} to "
              f"{out.loc[dated, 'sampled_at'].max():%Y-%m-%d}")
    print(f"{int((~dated).sum())} rows have no parseable date")

    timed = int(out["has_sample_time"].sum())
    print(f"{timed}/{len(out)} carry a sample TIME "
          f"({'hourly joins possible' if timed else 'daily means only'})")

    nd = int(out["nondetect"].sum())
    numeric = int(out["value"].notna().sum())
    print(f"{numeric} numeric values, {nd} non-detects, "
          f"{len(out) - numeric - nd} neither")

    print("\nper analyte:")
    for analyte, group in out.groupby("analyte"):
        vals = group["value"].dropna()
        stations = group["station"].nunique()
        span = ""
        if not vals.empty:
            span = (f"median {vals.median():g}, "
                    f"{vals.min():g}-{vals.max():g}")
        print(f"  {analyte:<28} {len(group):>5} results, "
              f"{stations} stations   {span}")

    units = out.loc[out["unit"].astype(str) != "", "unit"].value_counts()
    if len(units) > 1:
        print(f"\nWARNING: {len(units)} different units in one file — "
              "do not pool these without converting:")
        for unit, count in units.items():
            print(f"  {unit:<20} {count}")
    elif len(units) == 1:
        print(f"\nunits: {units.index[0]} (consistent)")

    per_station = out.groupby("station")["sampled_at"].agg(["count", "min", "max"])
    print("\nsampling density (this decides whether a site is analysable):")
    for station, row in per_station.sort_values("count", ascending=False).head(10).iterrows():
        if pd.isna(row["min"]):
            print(f"  {station:<34} {row['count']:>5} results, no dates")
            continue
        days = max((row["max"] - row["min"]).days, 1)
        print(f"  {station:<34} {row['count']:>5} results over {days} days "
              f"(~{row['count'] / (days / 30.4):.1f}/month)")


def main():
    ap = argparse.ArgumentParser(description="Bacteria results from the Water Quality Portal.")
    ap.add_argument("--camera", help="substring of a name in camera_candidates.csv")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--name")
    ap.add_argument("--km", type=float, default=DEFAULT_KM)
    ap.add_argument("--since", default=DEFAULT_SINCE, help="MM-DD-YYYY")
    ap.add_argument("--probe", action="store_true", help="print every column and stop")
    args = ap.parse_args()

    if args.lat is not None and args.lon is not None:
        name, lat, lon = args.name or f"{args.lat},{args.lon}", args.lat, args.lon
    elif args.camera:
        if not os.path.exists(CANDIDATES_CSV):
            sys.exit(f"{CANDIDATES_CSV} not found — run scan_cameras.py first.")
        frame = pd.read_csv(CANDIDATES_CSV)
        hits = frame[frame["camera"].str.contains(args.camera, case=False, na=False)]
        if hits.empty:
            sys.exit(f"No camera matching {args.camera!r}.")
        row = hits.iloc[0]
        name, lat, lon = row["camera"], float(row["lat"]), float(row["lon"])
        if len(hits) > 1:
            print(f"{len(hits)} match; using {name}")
    else:
        sys.exit("Give --camera, or --lat and --lon.")

    print(f"{name}  ({lat:.4f}, {lon:.4f})")
    frame = fetch(lat, lon, args.km, args.since)
    check_columns(frame, args.probe)
    if args.probe:
        print("\nColumns look right — re-run without --probe to write the CSV.")
        return

    out = build(frame, name)
    summarize(out)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/wqp_{slugify(name)}.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
