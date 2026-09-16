"""Turn the RipAID annotation export into hourly rip tables the pipeline can use.

RipAID (Zenodo 15082427) is a CVAT/COCO export of HUMAN-drawn rip annotations
on fixed coastal camera imagery. It differs from the WebCOOS rip feed in three
ways that matter more than its size:

  * It has real negatives. The 2,815-frame count is confirmed by the v2.0.0
    README; the "948 of them carry no annotation" figure is NOT -- v2.0.0
    reports 1,082 unannotated images across all 6,789, and gives no v1.0.0
    split. summarize() prints the real counts and those are the ones to quote
    --
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
import glob
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

# The README names the two SIRENA stations: Cala Millor (Mallorca) and Son Bou
# (Menorca), Balearic Islands, Spain. The coordinates below are beach-centroid
# estimates read off a map, not survey positions or camera positions. They are
# good enough to pick an ERA5 / Marine grid cell, which is all the weather pull
# needs; they are NOT good enough to trust a shore-normal result on without
# checking the orientation against Figure 2 of the README.
KNOWN_SITES = {
    "clm": {"name": "Cala Millor", "latitude": 39.5965, "longitude": 3.3835},
    "snb": {"name": "Son Bou", "latitude": 39.9085, "longitude": 4.0755},
}


def slug(text):
    """The same slug pull_rip_detection.py writes, so every rip table in the
    project follows one naming rule. Without this a camera named clm_s_01
    would be written as rip_clm_s_01_hourly.csv while analyze_drivers looked
    for clm-s-01, and the by-camera tables would silently never be found."""
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


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
        # Matched on the BASENAME. A CVAT COCO export names frames
        # "default/clm_s_01_2011-05-21-11-00.png" -- the subset folder is part
        # of the recorded name -- and the pattern is anchored, so matching the
        # full string drops every frame in the export and leaves an empty
        # table. The full name is still what gets stored, because it is what
        # locates the file on disk.
        match = FILENAME.match(os.path.basename(image["file_name"]))
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

    # Checked BEFORE the sort: an empty frame has no timestamp column, so
    # sorting it raises KeyError('timestamp') and buries the actual problem,
    # which is that nothing matched the naming pattern.
    frame = pd.DataFrame(rows)
    if frame.empty:
        sys.exit(f"No frames parsed from {len(payload['images'])} image "
                 "entries. Every file_name failed the\n  "
                 "<site>_<cam>_<YYYY-MM-DD-HH-MM> pattern — check the first "
                 "few listed above.")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


# ---------------------------------------------------------------------------
# YOLO-OBB input (RipAID v2.0.0)
# ---------------------------------------------------------------------------

# Index -> class, RECOVERED FROM THE DATA, not assumed. The v2.0.0 download
# ships no data.yaml, so nothing in it states the mapping. What it does ship is
# an instance count per index, and the README publishes a distinct total per
# class: rip_current 4103, doubt 1437, sediment 4591. Counting indices over the
# 6,789 label files returns exactly 4103/1437/4591 for 0/1/2, and because the
# three totals differ the assignment is forced. verify_classes() re-runs that
# check on every load rather than trusting this comment.
YOLO_CLASSES = {0: RIP_LABEL, 1: DOUBT_LABEL, 2: "sediment"}
PUBLISHED_INSTANCES = {RIP_LABEL: 4103, DOUBT_LABEL: 1437, "sediment": 4591}

SEDIMENT_LABEL = "sediment"


def obb_area(points):
    """Area of the oriented box, by the shoelace formula.

    Not width x height of the enclosing rectangle: these boxes are ROTATED, and
    the axis-aligned bound of a box at 45 degrees is twice its area. The
    coordinates are normalised to 0-1, so this is a fraction of the frame, NOT
    pixels -- see build_frames_yolo for why that is harmless here and where it
    would not be.
    """
    total = 0.0
    for index in range(len(points)):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def obb_angle_deg(points):
    """Orientation of the box's LONG axis, folded onto 0-180 degrees.

    The COCO path reads a `rotation` attribute; YOLO-OBB has no such field and
    the orientation has to come out of the corner order. Measured along the
    longer of the two edges meeting at the first corner, because the short edge
    would report a direction 90 degrees away from the rip's axis. Folded to
    0-180 for the same reason axial_mean_deg exists: a box at 10 degrees and
    one at 190 lie along one line.
    """
    if len(points) < 3:
        return None
    (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
    first = ((x1 - x0), (y1 - y0))
    second = ((x2 - x1), (y2 - y1))
    long_edge = first if (first[0] ** 2 + first[1] ** 2) >= \
        (second[0] ** 2 + second[1] ** 2) else second
    if long_edge == (0.0, 0.0):
        return None
    return math.degrees(math.atan2(long_edge[1], long_edge[0])) % 180.0


def read_obb(path):
    """One label file as [(class_name, [(x, y) x4]), ...]. Empty file -> []."""
    out = []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                index = int(float(parts[0]))
                values = [float(v) for v in parts[1:9]]
            except ValueError:
                continue
            name = YOLO_CLASSES.get(index, f"class_{index}")
            out.append((name, list(zip(values[0::2], values[1::2]))))
    return out


def verify_classes(counts):
    """Say whether the index->class mapping is the one the README implies.

    Printed rather than enforced: a subset of the labels legitimately will not
    match the published totals, and refusing to load one would be worse than
    saying the mapping is unconfirmed for it. What must never happen is a
    silent mismatch, because every rip count downstream rides on index 0.
    """
    print("\n  class index -> label, checked against the README's totals:")
    exact = True
    for index in sorted(YOLO_CLASSES):
        name = YOLO_CLASSES[index]
        seen = counts.get(name, 0)
        published = PUBLISHED_INSTANCES.get(name)
        mark = ""
        if published is not None and seen != published:
            exact = False
            mark = f"   <- README says {published}"
        print(f"    {index} = {name:<12} {seen:>6} instances{mark}")
    unknown = {k: v for k, v in counts.items() if k.startswith("class_")}
    if unknown:
        exact = False
        print(f"    UNKNOWN INDICES: {unknown} — a class this loader does not "
              "know about.")
    if exact:
        print("    exact match on all three, so the mapping is forced by the "
              "counts, not assumed.")
    else:
        print("    NOT an exact match. That is expected for a subset of the "
              "labels and\n    alarming for the full set — if this is the "
              "whole download, the index\n    mapping above is wrong and every "
              "rip count below with it.")
    return exact


def build_frames_yolo(target):
    """Frame table from a YOLO-OBB export, same columns as build_frames().

    `target` may be the dataset root, or its labels/ directory.

    ONE DIFFERENCE FROM THE COCO PATH, and it matters only if you forget it:
    YOLO coordinates are normalised to 0-1, so area_max here is a FRACTION OF
    THE FRAME and the COCO path's is PIXELS. The two must never be pooled. It
    is harmless within this project because every area analysis z-scores within
    a camera, and a camera's resolution is fixed -- so the normalised area is a
    constant multiple of the pixel area and the z-score is identical. It would
    not be harmless in anything comparing raw areas.
    """
    labels_dir = target
    if os.path.isdir(os.path.join(target, "labels")):
        labels_dir = os.path.join(target, "labels")
    if not os.path.isdir(labels_dir):
        sys.exit(f"{target} has no labels/ directory and is not one.")
    images_dir = os.path.join(os.path.dirname(labels_dir.rstrip("/")), "images")
    by_stem = {}
    if os.path.isdir(images_dir):
        for name in os.listdir(images_dir):
            by_stem.setdefault(os.path.splitext(name)[0], name)

    paths = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    if not paths:
        sys.exit(f"No .txt label files under {labels_dir}.")
    print(f"{len(paths)} label files under {labels_dir}")

    counts = Counter()
    rows, unparsed = [], []
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        # Matched on the basename WITH its extension: FILENAME ends in `\.`,
        # so it requires the dot and a bare stem never matches it. The COCO
        # path feeds it a file_name that has one; this path has to keep it too.
        match = FILENAME.match(os.path.basename(path))
        boxes = read_obb(path)
        for name, _ in boxes:
            counts[name] += 1
        if not match:
            unparsed.append(stem)
            continue
        rips = [pts for name, pts in boxes if name == RIP_LABEL]
        doubts = [pts for name, pts in boxes if name == DOUBT_LABEL]
        sediment = [pts for name, pts in boxes if name == SEDIMENT_LABEL]
        areas = [obb_area(pts) for pts in rips]
        rows.append({
            "timestamp": pd.Timestamp(
                int(match["y"]), int(match["mo"]), int(match["d"]),
                int(match["h"]), int(match["mi"]), tz="UTC"),
            "site": match["site"].lower(),
            "camera": match["camera"].lower(),
            "file_name": by_stem.get(stem, stem),
            "n_rip": len(rips),
            "n_doubt": len(doubts),
            "n_sediment": len(sediment),
            # A frame a person annotated with no RIP is an observed zero.
            # Sediment and doubt are neither rips nor clean negatives.
            "detected": len(rips) > 0,
            "area_max": max(areas) if areas else 0.0,
            "area_sum": sum(areas),
            "rotation_axial": axial_mean_deg(
                [obb_angle_deg(pts) for pts in rips]),
        })

    verify_classes(counts)

    if unparsed:
        print(f"\n  {len(unparsed)} of {len(paths)} label files are NOT from "
              "the fixed-camera\n  subset and are dropped — they carry no "
              "site, camera or timestamp:")
        for stem in unparsed[:3]:
            print(f"    {stem}")
        print("  RipAID v2.0.0 merges drone and smartphone imagery into the "
              "original\n  fixed-camera set. Those have no viewing bearing and "
              "no clock, so they\n  cannot join any analysis in this project "
              "that needs either.")

    frame = pd.DataFrame(rows)
    if frame.empty:
        sys.exit("No fixed-camera frames found. Every label file failed the\n"
                 "  <site>_<cam>_<YYYY-MM-DD-HH-MM> pattern.")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    print(f"\n  {len(frame)} fixed-camera frames kept")
    return frame


def frames_from(path):
    """Frame table from either input format, dispatched on what `path` is.

    RipAID v1.0.0 shipped a COCO export; v2.0.0 ships YOLO-OBB and a CVAT
    backup. Callers should not have to care which is on disk.
    """
    if os.path.isdir(path):
        return build_frames_yolo(path)
    return build_frames(load(path))


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

    print(f"\n{'!' * 74}")
    print("HOW THESE FRAMES WERE SELECTED — READ BEFORE CORRELATING ANYTHING")
    print(f"{'!' * 74}")
    print("""The RipAID README, section 2.1, describes the sampling:

  Balearic Islands lifeguards logged rip sightings (date, time, beach).
  Images were pulled from all cameras at that beach for that time +/- 2h.
  'After visual inspection, most of the images that did not show rip
  currents were removed.'

Two separate selections, and together they are fatal to a driver analysis
of rip PRESENCE:

  1. The hours in this dataset are hours a lifeguard was on duty AND
     reported a rip. That is a sample of lifeguard staffing and of rip
     occurrence at once, and the two cannot be separated here.
  2. Frames without a rip were then deliberately deleted. The frames that
     remain without one are the residue of that deletion, not a control
     group. They are not the hours when no rip happened.

So 'what conditions predict a rip being present' cannot be asked of this
data. A correlation would measure the curator's selection at least as much
as the ocean. This is not a flaw in the dataset — it was built to train
detectors, and for that the enrichment is a feature.

What the selection does NOT destroy is a question asked WITHIN the frames
that contain an annotated rip: given that a rip is there and a person drew
a box round it, does its size or orientation track the conditions? The
sample of rips is not random, but the measurement of each one is a property
of that rip rather than of the choice to include it.""")

    print("\nOne more trap in that analysis: bbox area is in PIXELS, and the")
    print("cameras have very different focal lengths — the README puts clm's")
    print("cross-shore pixel resolution between 0.2 and 15 m depending on the")
    print("camera. A pixel area from clm_c01 and one from clm_c05 are not the")
    print("same physical area. Compare areas WITHIN a camera, never across.")

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
        path = f"{OUT_DIR}/rip_{slug(name)}_hourly.csv"
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
        # 'name' is pre-filled with the site code so the weather pull produces
        # sane filenames even if only latitude/longitude get filled in.
        stub = pd.DataFrame([{
            "site": code,
            "name": KNOWN_SITES.get(code, {}).get("name", code),
            "latitude": KNOWN_SITES.get(code, {}).get("latitude", ""),
            "longitude": KNOWN_SITES.get(code, {}).get("longitude", ""),
        } for code in codes])
        if not os.path.exists(SITES_TEMPLATE):
            stub.to_csv(SITES_TEMPLATE, index=False)
            unknown = [c for c in codes if c not in KNOWN_SITES]
            print(f"\nWrote {SITES_TEMPLATE} for: {', '.join(codes)}")
            if unknown:
                print(f"  {', '.join(unknown)} have no coordinates — fill them in.")
            print("  Coordinates for known sites are map estimates, not survey "
                  "positions. Fine for a weather grid cell; check them before "
                  "trusting a shore-normal result.")
            print("Then:")
            print(f"  python pull_site_observations.py --sites-csv {SITES_TEMPLATE} "
                  f"--start {frames['timestamp'].min():%Y-%m-%d} "
                  f"--end {frames['timestamp'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
