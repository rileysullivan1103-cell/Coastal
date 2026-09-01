"""Describe a downloaded rip-current dataset so the pipeline can be pointed at it.

Written for the Zenodo record at https://zenodo.org/records/15082427, but it
makes no assumptions about that record specifically — it reports what is
actually in the files rather than what the landing page says should be.

Nothing in this project can reach Zenodo, so the structure of the download is
unknown here. Run this against the copy on disk and paste the output; the
columns it finds are what the weather pull gets keyed on.

    python probe_rip_dataset.py ~/Downloads/some-folder
    python probe_rip_dataset.py ~/Downloads/rips.csv
    python probe_rip_dataset.py ~/Downloads/rips.csv --full   # every column
"""

import argparse
import json
import os
import sys
import zipfile

import pandas as pd

TABULAR = (".csv", ".tsv", ".txt", ".xlsx", ".xls")
MAX_PREVIEW_FILES = 12

# Column names that plausibly carry the three things the weather pull needs.
# Matched as case-insensitive substrings, most specific first.
TIME_HINTS = ("datetime", "timestamp", "date_time", "date", "time", "utc")
LAT_HINTS = ("latitude", "lat_", "lat", "y_coord", "northing")
LON_HINTS = ("longitude", "long", "lon", "lng", "x_coord", "easting")
SITE_HINTS = ("site", "beach", "location", "station", "name", "region", "country")
RIP_HINTS = ("rip", "hazard", "flash", "current", "label", "class", "type",
             "present", "detected", "count")


def find_files(target):
    if os.path.isfile(target):
        return [target]
    found = []
    for root, _, names in os.walk(target):
        for name in sorted(names):
            if name.startswith("."):
                continue
            found.append(os.path.join(root, name))
    return found


def guess(columns, hints):
    """Columns matching any hint, in hint order — most specific first."""
    hits = []
    for hint in hints:
        for col in columns:
            if hint in str(col).lower() and col not in hits:
                hits.append(col)
    return hits


def describe_table(df, path, full=False):
    print(f"  {len(df)} rows x {len(df.columns)} columns")
    print("\n  columns:")
    for col in (df.columns if full else df.columns[:40]):
        series = df[col]
        non_null = int(series.notna().sum())
        sample = series.dropna()
        example = "" if sample.empty else str(sample.iloc[0])[:48]
        print(f"    {str(col)[:38]:<40} {str(series.dtype):<10} "
              f"{non_null}/{len(df)} non-null   e.g. {example}")
    if not full and len(df.columns) > 40:
        print(f"    ... {len(df.columns) - 40} more (use --full)")

    print("\n  what the weather pull needs:")
    for label, hints in (("time", TIME_HINTS), ("latitude", LAT_HINTS),
                         ("longitude", LON_HINTS), ("site", SITE_HINTS),
                         ("rip signal", RIP_HINTS)):
        hits = guess(df.columns, hints)
        print(f"    {label:<12} {', '.join(str(h) for h in hits[:6]) if hits else 'NOT FOUND'}")

    # A rip record without a usable timestamp cannot be joined to conditions at
    # all, so report the actual parsed range rather than trusting the dtype.
    for col in guess(df.columns, TIME_HINTS)[:2]:
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        ok = int(parsed.notna().sum())
        if ok:
            print(f"\n  {col}: {ok}/{len(df)} parse as dates, "
                  f"{parsed.min()} to {parsed.max()}")
        else:
            print(f"\n  {col}: nothing parses as a date — check the format")

    lats = guess(df.columns, LAT_HINTS)[:1]
    lons = guess(df.columns, LON_HINTS)[:1]
    if lats and lons:
        lat = pd.to_numeric(df[lats[0]], errors="coerce")
        lon = pd.to_numeric(df[lons[0]], errors="coerce")
        pairs = (lat.notna() & lon.notna()).sum()
        if pairs:
            print(f"  {lats[0]}/{lons[0]}: {pairs} usable pairs, "
                  f"lat {lat.min():.3f}..{lat.max():.3f}  "
                  f"lon {lon.min():.3f}..{lon.max():.3f}")
            distinct = df[[lats[0], lons[0]]].dropna().drop_duplicates()
            print(f"  {len(distinct)} distinct coordinate pairs "
                  "(= sites the weather pull would fetch)")

    print("\n  first rows:")
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print("    " + df.head(3).to_string().replace("\n", "\n    "))


def read_table(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for sep in (None, ",", ";", "\t"):
        try:
            df = pd.read_csv(path, sep=sep, engine="python", nrows=200000)
        except Exception:
            continue
        if len(df.columns) > 1:
            return df
    return pd.read_csv(path, engine="python", nrows=200000)


def main():
    ap = argparse.ArgumentParser(description="Describe a rip-current dataset.")
    ap.add_argument("target", help="file or directory holding the download")
    ap.add_argument("--full", action="store_true", help="list every column")
    args = ap.parse_args()

    if not os.path.exists(args.target):
        sys.exit(f"{args.target} does not exist.")

    files = find_files(args.target)
    if not files:
        sys.exit(f"No files under {args.target}.")

    print(f"{len(files)} file(s) under {args.target}\n")
    for path in files:
        size = os.path.getsize(path)
        print(f"  {size / 1e6:9.2f} MB  {os.path.relpath(path, args.target)}")
    print()

    shown = 0
    for path in files:
        lower = path.lower()
        if lower.endswith(".zip"):
            print(f"{'=' * 74}\n{os.path.basename(path)}  (zip)\n{'=' * 74}")
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist()[:40]:
                    print(f"  {info.file_size / 1e6:9.2f} MB  {info.filename}")
            print("\n  Unzip it and re-run this against the folder.\n")
            continue

        if lower.endswith(".json") or lower.endswith(".geojson"):
            print(f"{'=' * 74}\n{os.path.basename(path)}  (json)\n{'=' * 74}")
            with open(path) as fh:
                try:
                    payload = json.load(fh)
                except json.JSONDecodeError as exc:
                    print(f"  will not parse as JSON: {exc}\n")
                    continue
            if isinstance(payload, dict):
                print(f"  top-level keys: {list(payload.keys())[:20]}")
                if payload.get("type") == "FeatureCollection":
                    feats = payload.get("features") or []
                    print(f"  GeoJSON FeatureCollection, {len(feats)} features")
                    if feats:
                        print(f"  first feature properties: "
                              f"{list((feats[0].get('properties') or {}).keys())}")
                        print(f"  first geometry: {feats[0].get('geometry')}")
            elif isinstance(payload, list):
                print(f"  list of {len(payload)}")
                if payload:
                    print(f"  first item keys: "
                          f"{list(payload[0].keys()) if isinstance(payload[0], dict) else type(payload[0])}")
            print()
            continue

        if not lower.endswith(TABULAR):
            # .nc, .tif, .mp4, .jpg and friends — name them, do not guess.
            continue

        if shown >= MAX_PREVIEW_FILES:
            print(f"(stopping after {MAX_PREVIEW_FILES} tables)")
            break
        print(f"{'=' * 74}\n{os.path.relpath(path, args.target)}\n{'=' * 74}")
        try:
            df = read_table(path)
        except Exception as exc:
            print(f"  could not read: {type(exc).__name__}: {exc}\n")
            continue
        describe_table(df, path, full=args.full)
        print()
        shown += 1

    other = [p for p in files if not p.lower().endswith(TABULAR + (".zip", ".json", ".geojson"))]
    if other:
        print(f"{len(other)} non-tabular file(s) not inspected:")
        for path in other[:15]:
            print(f"  {os.path.relpath(path, args.target)}")
        print("\nIf the rip observations live in one of these (NetCDF, imagery, "
              "video), say which and I will add a reader.")


if __name__ == "__main__":
    main()
