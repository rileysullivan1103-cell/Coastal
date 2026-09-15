"""Draw a stratified sample of Walton stills for hand labelling.

Why stratify rather than sample at random: a random 300 from 8,712 hours would
be almost entirely low-confidence, low-surf frames, because that is what the
record mostly contains. Precision on the frames the detector is confident about
-- the ones a warning system would act on -- would rest on a handful of rows.
Stratifying spends the labelling budget where the answer is uncertain.

Strata are detector confidence (none / low / high) crossed with MOP wave height
tercile, nine cells, plus two boosters drawn from the hours most likely to fool
a vision model: low sun elevation and heavy cloud. The boosters are drawn AFTER
the grid and marked, so they can be reported separately or pooled, and either
way the grid cells stay balanced.

    python build_label_sample.py --dry-run     # stratum counts, no downloads
    python build_label_sample.py               # writes images + the CSV

A re-run is safe: images already on disk are kept, and an existing label CSV is
never overwritten (pass --force to rebuild one, losing any labels in it).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import analyze_drivers as ad
import pull_rip_detection as prd
import solar

CAMERA = "Walton Lighthouse, Santa Cruz, CA"
OUT_DIR = f"{ad.DATA_DIR}/label_sample"
IMAGE_DIR = f"{OUT_DIR}/images"
LABEL_CSV = f"{OUT_DIR}/labels.csv"
STRATA_CSV = f"{OUT_DIR}/strata.csv"

TARGET = 300
BOOSTER_EACH = 30
LOW_SUN_DEG = 15.0          # below this the sun is in the water, not over it
HIGH_CLOUD_PCT = 80.0

LABEL_COLUMNS = ["rip_present", "notes", "labeled_at"]

# Hours with no MOP wave height. They are a real part of the record and are
# reported, but they are not drawn from: a frame with no wave height cannot
# sit in a wave tercile, so labelling it answers nothing the cross-tab asks.
# Left in the draw pool they are three more "cells", and draw_grid splits the
# budget twelve ways instead of nine -- 75 of 300 images spent on 0.7% of the
# record, while every real cell drops from 33 rows to 25.
MISSING_TERCILE = "H? missing"


def load_frames(slug):
    """Frame-level detector output, one row per still the detector scored."""
    frame = ad.read_csv(f"{ad.DATA_DIR}/rip_detection/rip_{slug}.csv")
    if frame is None or frame.empty:
        sys.exit(f"No data/rip_detection/rip_{slug}.csv — run pull_rip_detection.py --pull")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame["hour"] = ad.to_hour(frame["timestamp"])
    return frame


def load_coverage(slug):
    """Hours the camera published imagery, detector output or not.

    The 'none' stratum lives here and nowhere else: an hour with imagery and no
    detection has no row in the frame table, so sampling only the frame table
    would silently make the sample conditional on the detector having fired --
    which is the very thing being measured.
    """
    path = f"{ad.DATA_DIR}/rip_detection/coverage_{slug}_hourly.csv"
    frame = ad.read_csv(path)
    if frame is None or frame.empty:
        sys.exit(f"No {path} — run pull_rip_detection.py --coverage")
    frame = frame.copy()
    column = "hour" if "hour" in frame.columns else frame.columns[0]
    frame["hour"] = ad.to_hour(frame[column])
    return frame.dropna(subset=["hour"]).drop_duplicates("hour")


def hourly_conditions(camera, lat, lon):
    """MOP wave height, cloud cover and solar elevation, one row per hour.

    Read from the same files the driver analysis uses, so a stratum boundary
    here means the same thing a tercile means there.
    """
    slug = ad.grid_slug(camera)
    mop = ad.read_csv(f"{ad.DATA_DIR}/mop_{slug}.csv")
    if mop is None or mop.empty:
        mop = ad.read_csv(f"{ad.DATA_DIR}/mop_{ad.rip_slug(camera)}.csv")
    frames = []
    if mop is not None and not mop.empty:
        column = "time" if "time" in mop.columns else mop.columns[0]
        mop = mop.copy()
        mop["hour"] = ad.to_hour(mop[column])
        height = next((c for c in mop.columns if "height" in c.lower()), None)
        if height:
            frames.append(mop.groupby("hour")[[height]].mean()
                          .rename(columns={height: "mop_wave_height"}))

    cloud = ad.read_csv(f"{ad.DATA_DIR}/cloud_{slug}.csv")
    if cloud is not None and not cloud.empty and "cloud_cover" in cloud.columns:
        column = "time" if "time" in cloud.columns else cloud.columns[0]
        cloud = cloud.copy()
        cloud["hour"] = ad.to_hour(cloud[column])
        frames.append(cloud.groupby("hour")[["cloud_cover"]].mean())

    if not frames:
        sys.exit("Neither a MOP file nor a cloud file on disk; nothing to stratify on.")
    out = frames[0]
    for extra in frames[1:]:
        out = out.join(extra, how="outer")
    out = out.reset_index()
    elevation, _ = solar.position(pd.DatetimeIndex(out["hour"]), lat, lon)
    out["solar_elevation"] = elevation
    return out


def camera_latlon(name):
    """The camera's position, from the same sites table the analysis uses.

    Solar elevation is one of the two booster strata, so a wrong position here
    would quietly draw the wrong hours. Exit rather than fall back to a guess.
    """
    sites = ad.load_sites()
    if sites is None or sites.empty:
        sys.exit("No qualifying-sites table on disk; cannot place the camera.")
    hit = sites[sites["camera_name"] == name]
    if hit.empty:
        hit = sites[sites["camera_name"].str.contains(name, case=False, na=False,
                                                      regex=False)]
    if len(hit) != 1:
        sys.exit(f"{name!r} matched {len(hit)} rows in the sites table; "
                 "pass the exact camera_name.")
    return float(hit.iloc[0]["lat"]), float(hit.iloc[0]["lon"])


def stills_service(name):
    """The stills service slug, or None if the catalogue will not give one.

    find_camera and find_stills_service both exit on failure, which is right
    for a pull script and wrong here: rows whose detector payload names its own
    source image can still be labelled without the catalogue, so a failure
    downgrades the run rather than ending it.
    """
    try:
        asset = prd.find_camera(prd.load_assets(), name)
        slug, _ = prd.find_stills_service(asset)
        return slug
    except SystemExit as exc:
        print(f"  stills service unavailable ({exc}).")
        print("  Rows that name their own source image will still be fetched;")
        print("  'none'-stratum rows need the service and will be reported as skipped.")
        return None


def confidence_band(score, cut):
    if pd.isna(score):
        return "none"
    return "high" if score >= cut else "low"


def assign_strata(pool, cut, edges):
    """Label every candidate row with its confidence band and wave tercile."""
    pool = pool.copy()
    pool["confidence"] = [confidence_band(s, cut) for s in pool["score_max"]]
    pool["wave_tercile"] = pd.cut(
        pool["mop_wave_height"], bins=edges,
        labels=["H1 low", "H2 mid", "H3 high"], include_lowest=True)
    pool["wave_tercile"] = pool["wave_tercile"].astype(object)
    pool.loc[pool["mop_wave_height"].isna(), "wave_tercile"] = MISSING_TERCILE
    pool["stratum"] = pool["confidence"] + " / " + pool["wave_tercile"].astype(str)
    return pool


def drawable_pool(pool):
    """The candidates the grid may draw from: everything with a wave tercile.

    Kept separate from assign_strata so the counts printed to the terminal show
    the whole record, including the hours that are about to be excluded, rather
    than quietly hiding them.
    """
    return pool[pool["wave_tercile"] != MISSING_TERCILE]


def draw_grid(pool, target, rng):
    """Equal draws per populated cell, with the remainder spread by cell size.

    An empty or thin cell is not backfilled from elsewhere: a cell with 4 hours
    in it should report n=4 and a useless interval, not borrow rows from a cell
    that answers a different question.
    """
    cells = [c for c, group in pool.groupby("stratum") if len(group)]
    if not cells:
        return pool.iloc[0:0]
    per = max(1, target // len(cells))
    picks = []
    for name, group in pool.groupby("stratum"):
        take = min(per, len(group))
        picks.append(group.sample(n=take, random_state=rng.integers(1 << 31)))
    drawn = pd.concat(picks)
    short = target - len(drawn)
    if short > 0:
        rest = pool.drop(index=drawn.index)
        if len(rest):
            drawn = pd.concat([drawn, rest.sample(n=min(short, len(rest)),
                                                  random_state=rng.integers(1 << 31))])
    return drawn


def draw_boosters(pool, already, rng):
    """Extra rows from the two conditions most likely to fool a vision model."""
    out = []
    rest = pool.drop(index=already.index, errors="ignore")
    # The test is evaluated against the CURRENT rest on each pass, not captured
    # once up front: the first booster removes its rows, so a mask built before
    # the loop would be indexed against a frame it no longer matches. pandas
    # reindexes that rather than raising, which is how it would have shipped.
    tests = (("low sun", lambda f: f["solar_elevation"] < LOW_SUN_DEG),
             ("high cloud", lambda f: f["cloud_cover"] >= HIGH_CLOUD_PCT))
    for name, predicate in tests:
        if rest.empty:
            print(f"  booster {name!r}: nothing left to draw from; skipped")
            continue
        subset = rest[predicate(rest).fillna(False)]
        if subset.empty:
            print(f"  booster '{name}': no rows match; skipped")
            continue
        take = min(BOOSTER_EACH, len(subset))
        picked = subset.sample(n=take, random_state=rng.integers(1 << 31))
        picked = picked.copy()
        picked["booster"] = name
        out.append(picked)
        rest = rest.drop(index=picked.index)
        print(f"  booster '{name}': {take} rows")
    return pd.concat(out) if out else pool.iloc[0:0]


def source_index(slug):
    """filename -> on-disk path for the raw detector payloads.

    The frame CSV records source_file as a bare basename ("...json"), not a
    path, so the payload cannot be reopened from that column alone -- an
    os.path.exists on it is false from any directory, and boxes_for quietly
    returned no boxes for every detection in the sample. The index written
    beside the frame CSV carries both the filename and the real path.
    """
    index = ad.read_csv(f"{ad.DATA_DIR}/rip_detection/rip_{slug}_index.csv")
    if index is None or index.empty:
        return {}
    if "filename" not in index.columns or "path" not in index.columns:
        return {}
    return {os.path.basename(str(name)): str(path)
            for name, path in zip(index["filename"], index["path"])
            if isinstance(path, str) or path == path}


def boxes_for(row, index):
    """Every detector box for ONE frame, as [{x,y,w,h}], in source pixels.

    The flattened frame CSV keeps only the largest box's area and centroid,
    which cannot be drawn. The raw payload is still on disk, so re-read it --
    labelling against a box that is not the box the detector drew would make
    the whole exercise measure the wrong thing.

    A payload file can hold many records. Pooling every box in the file would
    overlay other frames' boxes on this one, which is the same failure wearing
    a different hat, so the record is matched on its original image reference
    and a file that cannot be narrowed to one record contributes nothing.
    """
    path = index.get(os.path.basename(str(row.get("source_file") or "")))
    if not path or not os.path.exists(path):
        return []
    try:
        records = prd.read_records(path) or []
    except Exception:
        return []

    wanted = row.get("original_image")
    if isinstance(wanted, str) and wanted:
        records = [r for r in records
                   if r.get("original_image_reference") == wanted]
    elif len(records) != 1:
        records = []

    out = []
    for record in records:
        result = record.get("classification_result") or {}
        for box in result.get("classification_bboxes") or []:
            points = [(p.get("x"), p.get("y")) for p in box or []
                      if isinstance(p, dict)]
            points = [(x, y) for x, y in points if x is not None and y is not None]
            if len(points) < 2:
                continue
            xs, ys = [p[0] for p in points], [p[1] for p in points]
            out.append({"x": min(xs), "y": min(ys),
                        "w": abs(max(xs) - min(xs)), "h": abs(max(ys) - min(ys))})
    return out


def still_url(row, service, cache):
    """The URL of the still this row should be labelled against.

    A detected frame names its own source image, which is exact. An hour with
    no detection names nothing, so the stills service is asked for that hour and
    the frame nearest the middle is taken -- one call per hour, cached, because
    several sampled rows can land in the same hour.
    """
    direct = row.get("original_image")
    if isinstance(direct, str) and direct.startswith("http"):
        return direct, "detector's own source reference"

    hour = pd.Timestamp(row["hour"])
    if hour in cache:
        return cache[hour], "stills service"
    elements = prd.fetch_elements(service, hour.to_pydatetime(),
                                  (hour + pd.Timedelta(hours=1)).to_pydatetime(),
                                  quiet=True)
    time.sleep(0.2)
    if not elements:
        cache[hour] = None
        return None, "stills service"
    middle = hour + pd.Timedelta(minutes=30)
    best = min(elements, key=lambda e: abs(pd.Timestamp(e["timestamp"]) - middle))
    cache[hour] = best["url"]
    return best["url"], "stills service"


def report_boxes(table):
    """Say how many rows carry an overlay, and complain if detections do not.

    A detection with no box is the failure mode worth shouting about: the page
    still shows the image, so labelling proceeds against a blank overlay and
    the result looks like a clean run.
    """
    drawn = int((table["boxes"] != "[]").sum())
    print(f"  {drawn} rows carry detector boxes to overlay; "
          f"{len(table) - drawn} have none")
    detections = table[table["confidence"] != "none"]
    missing = int((detections["boxes"] == "[]").sum())
    if missing:
        print(f"  WARNING: {missing} of {len(detections)} detection rows have no "
              f"box to draw.\n           Labelling those is labelling blind -- "
              f"check data/rip_detection/ before starting.")
    return drawn


def count_labelled(table):
    """How many rows carry a verdict.

    fillna before astype(str): a blank cell reads back from the CSV as NaN, and
    str(NaN) is the non-empty string "nan", so the obvious spelling counts every
    unlabelled row as labelled. It reported "360 existing labels preserved" on a
    sample with no labels in it -- harmless there, but this line is the only
    evidence a repair run mid-labelling did not discard the work.
    """
    if "rip_present" not in table.columns:
        return 0
    verdicts = table["rip_present"].fillna("").astype(str).str.strip()
    return int(verdicts.ne("").sum())


def repair_boxes(slug, frames):
    """Recompute only the boxes column of an existing labels.csv.

    Rebuilding the sample would re-resolve 360 stills urls against the service
    and discard any labels already entered. The boxes are derived from payloads
    already on disk, so they can be recomputed in place instead.
    """
    table = ad.read_csv(LABEL_CSV)
    if table is None or table.empty:
        sys.exit(f"No {LABEL_CSV} to repair — run without --repair-boxes first.")

    lookup = {}
    duplicates = 0
    for _, frame in frames.iterrows():
        key = pd.Timestamp(frame["timestamp"]).strftime("%Y%m%dT%H%M%SZ")
        if key in lookup:
            duplicates += 1
            continue
        lookup[key] = frame
    if duplicates:
        print(f"  {duplicates} frames share a frame_id to the second; kept the first")

    index = source_index(slug)
    print(f"  {len(index)} payload files in the detection index")

    boxes = []
    for frame_id in table["frame_id"]:
        frame = lookup.get(str(frame_id))
        boxes.append(json.dumps(boxes_for(frame, index)) if frame is not None
                     else "[]")
    table["boxes"] = boxes

    labelled = count_labelled(table)
    temporary = LABEL_CSV + ".tmp"
    table.to_csv(temporary, index=False)
    os.replace(temporary, LABEL_CSV)
    print(f"\n  rewrote the boxes column of {LABEL_CSV}")
    print(f"  {labelled} existing labels preserved")
    report_boxes(table)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default=CAMERA)
    parser.add_argument("--target", type=int, default=TARGET)
    parser.add_argument("--seed", type=int, default=20260915,
                        help="fixed so the same sample can be rebuilt exactly")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the stratum table and stop, downloading nothing")
    parser.add_argument("--repair-boxes", action="store_true",
                        help="recompute the boxes column of an existing "
                             "labels.csv in place, keeping images and labels")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing labels.csv, losing its labels")
    args = parser.parse_args()

    lat, lon = camera_latlon(args.camera)

    slug = ad.rip_slug(args.camera)
    frames = load_frames(slug)
    # Before coverage and conditions: a repair needs neither, and loading
    # conditions can reach for Open-Meteo, which bills by variable-hours.
    if args.repair_boxes:
        return repair_boxes(slug, frames)

    coverage = load_coverage(slug)
    conditions = hourly_conditions(args.camera, lat, lon)

    detected = frames[frames["detected"].astype(bool)].copy()
    blank_hours = coverage[~coverage["hour"].isin(frames["hour"])].copy()
    for column in ("score_max", "detection_count", "bbox_count", "bbox_area_max",
                   "source_file", "original_image", "timestamp"):
        if column not in blank_hours.columns:
            blank_hours[column] = np.nan
    blank_hours["timestamp"] = blank_hours["hour"]

    keep = ["timestamp", "hour", "score_max", "detection_count", "bbox_count",
            "bbox_area_max", "source_file", "original_image"]
    pool = pd.concat([detected[keep], blank_hours[keep]], ignore_index=True)
    pool = pool.merge(conditions, on="hour", how="left")

    scores = pool["score_max"].dropna()
    if scores.empty:
        sys.exit("No scored frames; nothing to band by confidence.")
    cut = float(scores.median())
    heights = pool["mop_wave_height"].dropna()
    edges = ([-np.inf, np.inf] if heights.empty
             else [-np.inf, float(heights.quantile(1 / 3)),
                   float(heights.quantile(2 / 3)), np.inf])
    pool = assign_strata(pool, cut, edges)

    print(f"\n{args.camera}")
    print(f"  {len(detected)} scored frames, {len(blank_hours)} imagery hours with no detection")
    print(f"  confidence split at score_max {cut:.3f} (median of scored frames)")
    if len(edges) == 4:
        # The cut points alone hide how much physical contrast the wave axis
        # has. Equal-count terciles on a tightly clustered record can put the
        # middle third inside a 20 cm band, which makes "H2 mid" a label for a
        # slice too narrow to expect a precision difference across.
        print(f"  wave terciles at {edges[1]:.2f} m and {edges[2]:.2f} m")
        print(f"    H1 low   {heights.min():.2f} - {edges[1]:.2f} m")
        print(f"    H2 mid   {edges[1]:.2f} - {edges[2]:.2f} m "
              f"({edges[2] - edges[1]:.2f} m wide)")
        print(f"    H3 high  {edges[2]:.2f} - {heights.max():.2f} m")
    print(f"\n  candidates per stratum:")
    counts = pool.groupby("stratum").size().sort_index()
    for name, n in counts.items():
        print(f"    {name:<22} {n:>6}")

    drawable = drawable_pool(pool)
    dropped = len(pool) - len(drawable)
    if dropped:
        print(f"\n  excluding {dropped} candidates with no MOP wave height "
              f"({dropped / len(pool):.1%} of the record)")

    rng = np.random.default_rng(args.seed)
    grid = draw_grid(drawable, args.target, rng)
    grid = grid.copy()
    grid["booster"] = ""
    print(f"\n  grid draw: {len(grid)} rows across {grid['stratum'].nunique()} cells")
    boosters = draw_boosters(drawable, grid, rng)
    sample = pd.concat([grid, boosters]) if len(boosters) else grid
    sample = sample.sort_values("timestamp").reset_index(drop=True)
    print(f"  total sample: {len(sample)} rows")

    if args.dry_run:
        print("\n  --dry-run: nothing downloaded, nothing written.")
        return 0

    if os.path.exists(LABEL_CSV) and not args.force:
        sys.exit(f"\n{LABEL_CSV} already exists. Labelling it again from scratch "
                 "would discard the labels in it.\nPass --force if that is what you want.")

    os.makedirs(IMAGE_DIR, exist_ok=True)
    service = stills_service(args.camera)

    print(f"\n  resolving image urls for {len(sample)} rows")
    cache, rows, sources = {}, [], set()
    for _, row in sample.iterrows():
        url, how = (still_url(row, service, cache) if service or
                    isinstance(row.get("original_image"), str) else (None, "unavailable"))
        if not url:
            continue
        sources.add(how)
        rows.append({"timestamp": row["timestamp"], "url": url,
                     "filename": os.path.basename(url.split("?")[0]),
                     "_row": row})
    print(f"  {len(rows)} of {len(sample)} rows have an image url ({', '.join(sorted(sources)) or 'none'})")
    if not rows:
        sys.exit("  No image urls resolved; nothing to download.")

    box_index = source_index(slug)
    got = prd.download([{k: v for k, v in r.items() if k != "_row"} for r in rows],
                       IMAGE_DIR)
    by_name = {os.path.basename(g["path"]): g["path"] for g in got}

    out = []
    for entry in rows:
        row = entry["_row"]
        local = by_name.get(entry["filename"].replace(":", ""))
        if not local:
            continue
        out.append({
            "frame_id": f"{pd.Timestamp(row['timestamp']).strftime('%Y%m%dT%H%M%SZ')}",
            "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
            "stratum": row["stratum"],
            "confidence": row["confidence"],
            "wave_tercile": row["wave_tercile"],
            "booster": row.get("booster", ""),
            "score_max": row["score_max"],
            "detection_count": row["detection_count"],
            "bbox_count": row["bbox_count"],
            "bbox_area_max": row["bbox_area_max"],
            "mop_wave_height": row.get("mop_wave_height"),
            "cloud_cover": row.get("cloud_cover"),
            "solar_elevation": row.get("solar_elevation"),
            "image": os.path.basename(local),
            "boxes": json.dumps(boxes_for(row, box_index)),
            "rip_present": "",
            "notes": "",
            "labeled_at": "",
        })

    table = pd.DataFrame(out)
    table.to_csv(LABEL_CSV, index=False)

    # How common each stratum is in the whole record, not in the sample. The
    # sample deliberately over-represents high confidence and the boosters, so
    # without these weights the precision analysis would report the sample's
    # precision and call it the detector's.
    populations = (drawable.groupby("stratum").size().rename("population")
                   .reset_index())
    populations["sampled"] = populations["stratum"].map(
        table["stratum"].value_counts()).fillna(0).astype(int)
    populations.to_csv(STRATA_CSV, index=False)
    print(f"  wrote {STRATA_CSV}  (stratum populations, for reweighting)")
    print(f"\n  wrote {LABEL_CSV}  ({len(table)} rows)")
    print(f"  images in {IMAGE_DIR}/")
    report_boxes(table)
    print(f"\nNext:  python label_server.py")
    print(f"Then:  python analyze_precision.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
