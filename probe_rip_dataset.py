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
import re
import sys
import zipfile
# Reading only, never round-tripping, and the file is one the user downloaded
# themselves -- ElementTree is adequate here and has no extra dependency.
import xml.etree.ElementTree as ET

import pandas as pd

TABULAR = (".csv", ".tsv", ".xlsx", ".xls")
MAX_PREVIEW_FILES = 12
IMAGE = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
VIDEO = (".mp4", ".avi", ".mov", ".mkv")

# The whole question for an annotation dataset is whether a frame can be placed
# in TIME. Without that there is no weather to join to. These are the filename
# patterns coastal camera archives actually use; each captures a timestamp.
FILENAME_TIME = [
    # Contiguous-digit forms first: 201105211100 also matches the date-only
    # pattern below, and matching that one would silently discard the clock
    # time and downgrade every join to a daily mean.
    (r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})(?!\d)", "YYYYMMDDHHMMSS"),
    (r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(?!\d)", "YYYYMMDDHHMM"),
    (r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", "YYYY-MM-DD-HH-MM"),
    (r"(\d{4})-(\d{2})-(\d{2})[-T_](\d{2})(\d{2})(\d{2})", "YYYY-MM-DD-HHMMSS"),
    (r"(\d{4})-(\d{2})-(\d{2})[-T_](\d{2}):(\d{2}):(\d{2})", "YYYY-MM-DDTHH:MM:SS"),
    (r"(\d{4})(\d{2})(\d{2})[-T_](\d{2})(\d{2})(\d{2})", "YYYYMMDD_HHMMSS"),
    (r"(\d{4})(\d{2})(\d{2})[-T_](\d{2})(\d{2})", "YYYYMMDD_HHMM"),
    (r"(\d{4})-(\d{2})-(\d{2})", "YYYY-MM-DD (date only)"),
    (r"(\d{4})(\d{2})(\d{2})", "YYYYMMDD (date only)"),
]
UNIX_TIME = re.compile(r"(?<!\d)(1[0-9]{9})(?!\d)")   # 2001-2033 as seconds

# Column names that plausibly carry the three things the weather pull needs.
# Matched as case-insensitive substrings, most specific first.
TIME_HINTS = ("datetime", "timestamp", "date_time", "date", "time", "utc")
LAT_HINTS = ("latitude", "lat_", "lat", "y_coord", "northing")
LON_HINTS = ("longitude", "long", "lon", "lng", "x_coord", "easting")
SITE_HINTS = ("site", "beach", "location", "station", "name", "region", "country")
RIP_HINTS = ("rip", "hazard", "flash", "current", "label", "class", "type",
             "present", "detected", "count")


def timestamp_style(names):
    """Which filename time pattern, if any, the sample of names follows."""
    for pattern, label in FILENAME_TIME:
        rx = re.compile(pattern)
        hits = [n for n in names if rx.search(os.path.basename(n))]
        if len(hits) >= max(1, len(names) // 2):
            example = rx.search(os.path.basename(hits[0]))
            return label, example.group(0), len(hits)
    hits = [n for n in names if UNIX_TIME.search(os.path.basename(n))]
    if len(hits) >= max(1, len(names) // 2):
        return "unix seconds", UNIX_TIME.search(os.path.basename(hits[0])).group(0), len(hits)
    return None, None, 0


def report_frame_names(names, source):
    """The make-or-break check: can these frames be placed in time?"""
    print(f"\n  {len(names)} frame names from {source}")
    for name in names[:5]:
        print(f"    {name}")
    if len(names) > 5:
        print(f"    ... and {len(names) - 5} more")

    label, example, hits = timestamp_style(names)
    print("\n  CAN THESE BE PLACED IN TIME?")
    if label:
        print(f"    YES — {hits}/{len(names)} names carry a timestamp")
        print(f"    pattern: {label}   e.g. {example}")
        if "date only" in label:
            print("    Date but no clock time, so conditions can only be joined")
            print("    as daily means — the same limitation the California")
            print("    bacteria work had.")
    else:
        print("    NO — no timestamp in the filenames.")
        print("    Without a time there is no weather to join to. The rips would")
        print("    still be usable for training or for a detector-vs-human")
        print("    comparison, but not for a driver analysis. Check whether the")
        print("    dataset ships a separate index mapping frame -> date/site.")


def describe_cvat(path):
    """CVAT XML: <annotations><meta>..</meta><image name=..><box label=../></image>"""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        print(f"  will not parse as XML: {exc}")
        return
    if root.tag != "annotations":
        print(f"  root element is <{root.tag}>, not <annotations> — not CVAT")
        return

    task = root.find(".//task")
    if task is not None:
        for field in ("name", "size", "mode", "created", "start_frame", "stop_frame"):
            node = task.find(field)
            if node is not None and node.text:
                print(f"  task {field:<12} {node.text}")
        for source in task.findall(".//segment/url") + task.findall(".//source"):
            if source.text:
                print(f"  source        {source.text}")

    images = root.findall("image")
    tracks = root.findall("track")
    print(f"  {len(images)} <image> elements, {len(tracks)} <track> elements")

    shapes = {}
    for parent in (images or tracks):
        for shape in parent:
            label = shape.get("label", "?")
            shapes.setdefault(label, {"count": 0, "kinds": set()})
            shapes[label]["count"] += 1
            shapes[label]["kinds"].add(shape.tag)

    if shapes:
        print("\n  annotations by label:")
        total = sum(v["count"] for v in shapes.values())
        for label, info in sorted(shapes.items(), key=lambda kv: -kv[1]["count"]):
            share = 100 * info["count"] / total
            print(f"    {label:<20} {info['count']:>7}  ({share:4.1f}%)  "
                  f"as {', '.join(sorted(info['kinds']))}")
        # 'doubt' is the interesting one: it is the annotators' own uncertainty,
        # which is exactly the quantity the detector score cannot supply.
        if any("doubt" in k.lower() for k in shapes):
            doubtful = sum(v["count"] for k, v in shapes.items() if "doubt" in k.lower())
            print(f"\n    {doubtful} of {total} shapes ({100 * doubtful / total:.1f}%) "
                  "are 'doubt' — human uncertainty, recorded explicitly.")

    names = [im.get("name", "") for im in images if im.get("name")]
    if names:
        report_frame_names(names, "the XML")
    elif tracks:
        print("\n  Track-based (video), so frames are numbered, not named.")
        print("  A frame number becomes a time only via the video start time "
              "and frame rate — look for those in the task metadata above.")


def describe_coco(payload, path):
    images = payload.get("images") or []
    annotations = payload.get("annotations") or []
    categories = payload.get("categories") or []
    print(f"  COCO-style: {len(images)} images, {len(annotations)} annotations, "
          f"{len(categories)} categories")
    by_id = {c.get("id"): c.get("name", "?") for c in categories}
    counts = {}
    for ann in annotations:
        name = by_id.get(ann.get("category_id"), str(ann.get("category_id")))
        counts[name] = counts.get(name, 0) + 1
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {name:<20} {count:>7}")
    if images:
        keys = list(images[0].keys())
        print(f"  image record keys: {keys}")
        names = [im.get("file_name", "") for im in images if im.get("file_name")]
        if names:
            report_frame_names(names, "the JSON")


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

        if lower.endswith(".xml"):
            print(f"{'=' * 74}\n{os.path.basename(path)}  (xml)\n{'=' * 74}")
            describe_cvat(path)
            print()
            continue

        if lower.endswith((".json", ".geojson", ".txt")):
            with open(path, errors="replace") as fh:
                head = fh.read(2048).lstrip()
            if lower.endswith(".txt") and not head.startswith(("{", "[")):
                continue  # a YOLO label file or free text, not JSON
            print(f"{'=' * 74}\n{os.path.basename(path)}  (json)\n{'=' * 74}")
            with open(path) as fh:
                try:
                    payload = json.load(fh)
                except json.JSONDecodeError as exc:
                    print(f"  will not parse as JSON: {exc}\n")
                    continue
            if isinstance(payload, dict) and "annotations" in payload and "images" in payload:
                describe_coco(payload, path)
                print()
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

    frames = [p for p in files if p.lower().endswith(IMAGE)]
    if frames:
        print(f"{'=' * 74}\nIMAGERY\n{'=' * 74}")
        print(f"  {len(frames)} image files")
        report_frame_names([os.path.basename(p) for p in frames], "the image filenames")
        print()

    clips = [p for p in files if p.lower().endswith(VIDEO)]
    if clips:
        print(f"  {len(clips)} video file(s):")
        for path in clips[:10]:
            print(f"    {os.path.relpath(path, args.target)}")
        print()

    known = TABULAR + (".zip", ".json", ".geojson", ".xml", ".txt") + IMAGE + VIDEO
    other = [p for p in files if not p.lower().endswith(known)]
    if other:
        print(f"{len(other)} non-tabular file(s) not inspected:")
        for path in other[:15]:
            print(f"  {os.path.relpath(path, args.target)}")
        print("\nIf the rip observations live in one of these (NetCDF, imagery, "
              "video), say which and I will add a reader.")


if __name__ == "__main__":
    main()
