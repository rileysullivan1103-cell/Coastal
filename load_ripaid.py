"""Turn the RipAID annotation export into hourly rip tables the pipeline can use.

RipAID (Zenodo 15082427) is a CVAT/COCO export of HUMAN-drawn rip annotations
on fixed coastal camera imagery. It differs from the WebCOOS rip feed in three
ways that matter more than its size:

  * It has real negatives. 948 of its 2,815 frames carry no annotation at all,
    which means a person looked and saw no rip. The WebCOOS feed publishes an
    element only when the detector fires, so 'no file' there means 'no rip OR
    no image' and the denominator had to be rebuilt from a separate stills
    product. Here the denominator is the frame list itself.
  * The target is a person, not a model. A driver of the WebCOOS score cannot
    be told apart from a driver of the detector. Here it can.
  * It carries a 'doubt' class -- the annotators' own uncertainty, recorded
    explicitly rather than hidden inside a confidence number.

Frames are named <site>_<camera>_<YYYY-MM-DD-HH-MM>, so each one places in time
to the hour and belongs to a known camera. Output matches the column contract
of pull_rip_detection.py's hourly summary, so analyze_drivers.py reads it
unchanged.

    python load_ripaid.py instances_default.json
    python load_ripaid.py instances_default.json --by-camera
    python load_ripaid.py instances_default.json --sites ripaid_sites.csv

Writes data/ripaid/rip_<site>_hourly.csv, one per site.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter

import pandas as pd

OUT_DIR = "data/ripaid"
SITES_TEMPLATE = "ripaid_sites.csv"

# clm_s_01_2011-05-21-11-00.png -> site 'clm', camera 'clm_s_01', that hour.
FILENAME = re.compile(
    r"^(?P<camera>(?P<site>[a-z]+)_[a-z]+_\d+)_"
    r"(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})-(?P<h>\d{2})-(?P<mi>\d{2})\.",
    re.IGNORECASE)

RIP_LABEL = "rip_current"
DOUBT_LABEL = "doubt"


def axial_mean_deg(angles):
    """Mean of orientations that are lines, not arrows.

    A rip at 10 deg and one at 190 deg lie along the same axis. An arithmetic
    mean calls that 100 deg, which is perpendicular to both and simply wrong.
    Doubling the angles before averaging and halving the result folds the two
    directions onto one axis, which is what an orientation actually is.
    """
    values = [a for a in angles if a is not None and not math.isnan(a)]
    if not values:
        return float("nan")
    x = sum(math.cos(math.radians(2 * a)) for a in values)
    y = sum(math.sin(math.radians(2 * a)) for a in values)
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return float("nan")  # perfectly opposed; no meaningful axis
    return (math.degrees(math.atan2(y, x)) / 2) % 180


def load(path):
    with open(path) as fh:
        payload = json.load(fh)
    for key in ("images", "annotations", "categories"):
        if key not in payload:
            sys.exit(f"{path} has no '{key}' — is this a COCO export? "
                     f"Top-level keys: {list(payload.keys())}")
    return payload


def build_frames(payload):
    labels = {c["id"]: c["name"] for c in payload["categories"]}
    print(f"categories: {labels}")

    per_image = {}
    for ann in payload["annotations"]:
        entry = per_image.setdefault(ann["image_id"], {"rip": [], "doubt": [], "other": []})
        name = labels.get(ann.get("category_id"), "?")
        bucket = ("rip" if name == RIP_LABEL
                  else "doubt" if name == DOUBT_LABEL else "other")
        entry[bucket].append(ann)

    other = sum(len(v["other"]) for v in per_image.values())
    if other:
        print(f"WARNING: {other} annotations are neither {RIP_LABEL!r} nor "
              f"{DOUBT_LABEL!r} and are being ignored — labels seen: "
              f"{sorted(set(labels.values()))}")

    rows, unparsed = [], []
    for image in payload["images"]:
        match = FILENAME.match(image["file_name"])
        if not match:
            unparsed.append(image["file_name"])
            continue
        entry = per_image.get(image["id"], {"rip": [], "doubt": [], "other": []})
        rips, doubts = entry["rip"], entry["doubt"]
        rows.append({
            "timestamp": pd.Timestamp(
                int(match["y"]), int(match["mo"]), int(match["d"]),
                int(match["h"]), int(match["mi"]), tz="UTC"),
            "site": match["site"].lower(),
            "camera": match["camera"].lower(),
            "file_name": image["file_name"],
            "n_rip": len(rips),
            "n_doubt": len(doubts),
            # A frame a person annotated with nothing is an OBSERVED ZERO.
            "detected": len(rips) > 0,
            "area_max": max((a.get("area") or 0) for a in rips) if rips else 0.0,
            "area_sum": sum((a.get("area") or 0) for a in rips),
            "rotation_axial": axial_mean_deg(
                [(a.get("attributes") or {}).get("rotation") for a in rips]),
        })

    if unparsed:
        print(f"\nWARNING: {len(unparsed)} filenames did not match the expected "
              f"<site>_<cam>_<YYYY-MM-DD-HH-MM> pattern and were dropped:")
        for name in unparsed[:5]:
            print(f"  {name}")
        print("  If these are a different naming scheme, say so — dropping them "
              "silently would bias the frame count.")

    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        sys.exit("No frames parsed. Check the file_name format.")
    return frame


def summarize(frames):
    print(f"\n{'=' * 74}\nWHAT IS IN THE EXPORT\n{'=' * 74}")
    print(f"{len(frames)} frames, {frames['timestamp'].min():%Y-%m-%d} to "
          f"{frames['timestamp'].max():%Y-%m-%d}")
    zeros = int((~frames["detected"]).sum())
    print(f"{int(frames['detected'].sum())} frames with a rip, {zeros} without "
          f"({100 * zeros / len(frames):.1f}% observed negatives)")
    print(f"{int(frames['n_rip'].sum())} rip annotations, "
          f"{int(frames['n_doubt'].sum())} 'doubt' annotations")

    doubt_only = int(((frames["n_rip"] == 0) & (frames["n_doubt"] > 0)).sum())
    print(f"{doubt_only} frames carry ONLY doubt — these are neither a clean "
          "positive nor a clean negative")

    print("\nper site and camera:")
    for site, group in frames.groupby("site"):
        span_years = group["timestamp"].dt.year.nunique()
        print(f"  {site}: {len(group)} frames, {group['camera'].nunique()} cameras, "
              f"{span_years} years, {100 * group['detected'].mean():.1f}% with a rip")
        for camera, sub in group.groupby("camera"):
            print(f"    {camera:<12} {len(sub):>5} frames  "
                  f"{sub['timestamp'].min():%Y-%m}..{sub['timestamp'].max():%Y-%m}  "
                  f"{100 * sub['detected'].mean():.0f}% rip")

    print("\nframes per year (a season control needs several years):")
    counts = frames["timestamp"].dt.year.value_counts().sort_index()
    print("  " + "  ".join(f"{y}:{n}" for y, n in counts.items()))
    print("\nframes per hour UTC (daylight only, as at Walton):")
    counts = frames["timestamp"].dt.hour.value_counts().sort_index()
    print("  " + "  ".join(f"{h:02d}:{n}" for h, n in counts.items()))


def to_hourly(frames):
    """Same columns pull_rip_detection.py's hourly summary writes."""
    work = frames.copy()
    work["hour"] = work["timestamp"].dt.floor("h")
    hourly = work.groupby("hour").agg(
        frames=("detected", "size"),
        frames_with_detection=("detected", "sum"),
        detections=("n_rip", "sum"),
        doubts=("n_doubt", "sum"),
        bbox_area_max=("area_max", "max"),
        bbox_area_sum=("area_sum", "sum"),
    )
    hourly["detection_rate"] = (hourly["frames_with_detection"]
                                / hourly["frames"]).round(4)
    hourly["doubt_rate"] = (hourly["doubts"] / hourly["frames"]).round(4)
    # Orientation is averaged axially across the hour's frames, for the same
    # reason it is averaged axially within a frame.
    orientation = work.groupby("hour")["rotation_axial"].apply(
        lambda s: axial_mean_deg(list(s)))
    hourly["rip_axis_deg"] = orientation
    # An hour whose frames held no rip has a real zero area, not a missing one.
    hourly.loc[hourly["frames_with_detection"] == 0, "bbox_area_max"] = 0.0
    return hourly.reset_index()


def main():
    ap = argparse.ArgumentParser(description="Load a RipAID COCO export.")
    ap.add_argument("coco_json", help="instances_default.json from the download")
    ap.add_argument("--by-camera", action="store_true",
                    help="one file per camera instead of per site")
    ap.add_argument("--sites", help="CSV mapping site code to latitude/longitude")
    args = ap.parse_args()

    if not os.path.exists(args.coco_json):
        sys.exit(f"{args.coco_json} not found.")

    frames = build_frames(load(args.coco_json))
    summarize(frames)

    os.makedirs(OUT_DIR, exist_ok=True)
    key = "camera" if args.by_camera else "site"
    print(f"\n{'=' * 74}\nHOURLY TABLES\n{'=' * 74}")
    for name, group in frames.groupby(key):
        hourly = to_hourly(group)
        path = f"{OUT_DIR}/rip_{name}_hourly.csv"
        hourly.to_csv(path, index=False)
        zeros = int((hourly["frames_with_detection"] == 0).sum())
        print(f"  {path}  ({len(hourly)} hours, {zeros} observed-zero hours)")

    codes = sorted(frames["site"].unique())
    if args.sites and os.path.exists(args.sites):
        mapping = pd.read_csv(args.sites)
        print(f"\nsite coordinates from {args.sites}:")
        print(mapping.to_string(index=False))
    else:
        # Coordinates are not in the COCO export, and without them there is no
        # weather to fetch. Write a stub rather than inventing a location.
        stub = pd.DataFrame({"site": codes, "name": ["" for _ in codes],
                             "latitude": ["" for _ in codes],
                             "longitude": ["" for _ in codes]})
        if not os.path.exists(SITES_TEMPLATE):
            stub.to_csv(SITES_TEMPLATE, index=False)
            print(f"\nThe export carries no coordinates, so I cannot fetch "
                  f"weather yet.\nWrote {SITES_TEMPLATE} with the site codes "
                  f"found: {', '.join(codes)}")
            print("Fill in latitude/longitude (the Zenodo record or its README "
                  "should name the beaches), then:")
            print(f"  python pull_site_observations.py --sites-csv {SITES_TEMPLATE} "
                  f"--start {frames['timestamp'].min():%Y-%m-%d} "
                  f"--end {frames['timestamp'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
