"""Find detections that never move -- fixed scene features read as rip currents.

Hand labelling turned up boxes sitting on a rock jetty and on moored boats,
in frames whose class is rip_current. A jetty is a dark linear feature running
seaward with white water breaking along both sides, which from a fixed camera
is close to what a rip channel looks like, so the confusion is expected. What
matters is how MUCH of the record it accounts for.

The test is that a real rip moves and a jetty does not. Rip channels migrate
with tide and swell; a rock groyne is in the same pixels every frame for two
years. So the detections are gridded by centroid and the cells that fire far
more often than any moving feature could are reported, along with the share of
the whole record they carry.

If one cell holds a large share, then detection_rate at that camera is mostly
"was the jetty visible today", and every driver result computed from it --
cloud, glare, wave height -- is a result about visibility of a fixed object
rather than about rip currents. That is worth knowing before those numbers go
anywhere.

Two further checks separate a static object from a genuinely rip-prone spot:

  stationarity   how tightly the centroids cluster INSIDE the hot cell. A rip
                 recurring in a favoured channel still wanders a few tens of
                 pixels; a jetty does not move at all.
  condition-free a real rip's detections rise and fall with the surf. A fixed
                 object's do not, except through visibility.

    python analyze_static_detections.py
    python analyze_static_detections.py --cell 48 --top 12

Reads only what is on disk. No network.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

import analyze_drivers as ad

RIP_CLASS = "rip_current"
CELL = 64
TOP = 8
OUT_DIR = f"{ad.DATA_DIR}/static_detections"


REQUIRED = ("timestamp", "bbox_x", "bbox_y")


def rip_frames(path, wanted=RIP_CLASS):
    """Detected frames of one class, with a usable box centroid.

    Every column is checked before it is used. Not defensive habit: one file
    under data/rip_detection/ matching rip_*.csv has no timestamp column at
    all, and assuming the schema killed the whole run on the first camera
    reached rather than reporting that one file and carrying on to the other
    six. What that file is matters too, so its columns are printed rather than
    swallowed.
    """
    frame = ad.read_csv(path)
    if frame is None or frame.empty:
        return None
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        print(f"  {os.path.basename(path)}: no {', '.join(missing)} column"
              f"{'s' if len(missing) > 1 else ''} — skipped")
        print(f"    it has: {', '.join(map(str, frame.columns[:12]))}"
              f"{' ...' if len(frame.columns) > 12 else ''}")
        return None

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True,
                                        errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    if "score_classes" in frame.columns:
        classes = frame["score_classes"].fillna("")
        frame = frame[classes.map(
            lambda v: {p.strip() for p in str(v).split(",") if p.strip()}
            == {wanted})]
    else:
        print(f"  {os.path.basename(path)}: no score_classes column; "
              f"cannot separate {wanted} from object detections")
    for column in ("bbox_x", "bbox_y"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["bbox_x", "bbox_y"])
    return frame if not frame.empty else None


def grid_counts(frame, cell):
    """Detections per grid cell, and the cell each detection fell in."""
    column = (frame["bbox_x"] // cell).astype(int)
    row = (frame["bbox_y"] // cell).astype(int)
    key = list(zip(column, row))
    tally = pd.Series(key, index=frame.index).value_counts()
    return tally, pd.Series(key, index=frame.index)


def spread_within(frame, cell_key, cells, cell):
    """Std deviation of centroids inside one cell, in pixels.

    The diagnostic that separates a fixed object from a favoured channel. A
    rip recurring in the same rough place still wanders; a groyne is in the
    same pixels every time. Reported against the cell size so a reader can see
    whether the cluster fills its cell or sits in a corner of it.
    """
    inside = frame[cell_key == cells]
    if len(inside) < 3:
        return float("nan"), float("nan")
    return float(inside["bbox_x"].std()), float(inside["bbox_y"].std())


def condition_coupling(frame, cell_key, cells, conditions):
    """Spearman rho between a cell's DAILY detection count and wave height.

    A real rip fires more when the surf is up. A jetty does not, except
    through whatever visibility the weather allows. Returned alongside the same
    number for every OTHER cell, because the comparison is what carries the
    meaning: an absolute rho near zero could just be a weak driver.
    """
    if conditions is None or conditions.empty:
        return None
    frame = frame.copy()
    frame["date"] = frame["timestamp"].dt.floor("D")
    frame["in_cell"] = (cell_key == cells).to_numpy()

    daily = frame.groupby("date")["in_cell"].agg(["sum", "size"])
    daily.columns = ["hot", "all"]
    daily["rest"] = daily["all"] - daily["hot"]
    joined = daily.join(conditions, how="inner").dropna(
        subset=["wave_height"])
    if len(joined) < ad.MIN_N:
        return None
    hot = ad.spearman(joined["hot"], joined["wave_height"])
    rest = ad.spearman(joined["rest"], joined["wave_height"])
    return {"n_days": len(joined), "hot": hot, "rest": rest}


def daily_waves(camera):
    """Daily mean MOP wave height from the file already on disk."""
    mop = ad.read_csv(f"{ad.DATA_DIR}/mop_{ad.grid_slug(camera)}.csv")
    if mop is None or mop.empty:
        return None
    column = "time" if "time" in mop.columns else mop.columns[0]
    mop = mop.copy()
    mop["date"] = pd.to_datetime(mop[column], utc=True,
                                 errors="coerce").dt.floor("D")
    height = next((c for c in mop.columns if "height" in c.lower()), None)
    if not height:
        return None
    return (mop.groupby("date")[[height]].mean()
            .rename(columns={height: "wave_height"}))


def cameras_on_disk():
    out = []
    try:
        sites = ad.load_sites()
        by_slug = {ad.rip_slug(n): n for n in sites["camera_name"].dropna()}
    except SystemExit:
        by_slug = {}
    for path in sorted(glob.glob(f"{ad.DATA_DIR}/rip_detection/rip_*.csv")):
        stem = os.path.basename(path)[len("rip_"):-len(".csv")]
        if stem.endswith("_index"):
            continue
        out.append((by_slug.get(stem, stem), stem, path))
    return out


def report(camera, frame, cell, top, conditions):
    tally, cell_key = grid_counts(frame, cell)
    total = len(frame)
    print(f"\n  {total} {RIP_CLASS} detections with a centroid, "
          f"gridded at {cell}px")
    print(f"  {len(tally)} occupied cells; the busiest carries "
          f"{tally.iloc[0] / total:.1%} of them")

    print(f"\n  {'cell (x,y) px':<20} {'n':>7} {'share':>7} {'score':>6} "
          f"{'area':>9} {'spread x,y px':>16}")
    rows = []
    for cells, count in tally.head(top).items():
        inside = frame[cell_key == cells]
        sx, sy = spread_within(frame, cell_key, cells, cell)
        score = pd.to_numeric(inside.get("score_max"), errors="coerce").median()
        area = pd.to_numeric(inside.get("bbox_area_max"),
                             errors="coerce").median()
        label = f"{cells[0] * cell},{cells[1] * cell}"
        print(f"  {label:<20} {count:>7} {count / total:>6.1%} "
              f"{score:>6.2f} {area:>9.0f} {sx:>7.1f},{sy:>7.1f}")
        rows.append({"camera": camera, "cell_x": cells[0] * cell,
                     "cell_y": cells[1] * cell, "n": int(count),
                     "share": count / total, "median_score": score,
                     "median_area": area, "spread_x": sx, "spread_y": sy})

    hottest = tally.index[0]
    share = tally.iloc[0] / total
    sx, sy = spread_within(frame, cell_key, hottest, cell)

    print()
    if share >= 0.25:
        print(f"    ONE CELL CARRIES {share:.0%} OF THIS CAMERA'S DETECTIONS.")
        print("    detection_rate here is largely 'was that feature visible', "
              "not 'was there a rip'.\n    Every driver result computed from "
              "it is about the visibility of a fixed object.")
    elif share >= 0.10:
        print(f"    the busiest cell carries {share:.0%} — enough to shift a "
              "pooled result, not\n    enough to be the whole of it. Worth "
              "excluding as a sensitivity check.")
    else:
        print(f"    no cell dominates (busiest {share:.1%}); detections are "
              "spread across the scene.")

    if sx == sx and sx < cell / 6 and sy < cell / 6:
        print(f"    centroids inside it vary by only {sx:.1f} x {sy:.1f} px. "
              "A rip recurring in a\n    favoured channel still wanders; this "
              "does not move at all.")

    # Only where a cell actually dominates. On a scene with detections spread
    # evenly the "busiest" cell is whichever one noise favoured, and a coupling
    # verdict about it reads as a finding while being about nothing.
    coupling = (condition_coupling(frame, cell_key, hottest, conditions)
                if share >= 0.10 else None)
    if coupling:
        hot, rest = coupling["hot"], coupling["rest"]
        print(f"\n    daily count vs wave height over {coupling['n_days']} days")
        # ad.spearman returns (rho, n, p) -- n before p, which is easy to get
        # backwards and would print a p-value of 300 without complaining.
        print(f"      busiest cell   rho {hot[0]:+.3f}  n {hot[1]}  "
              f"p {hot[2]:.3g}")
        print(f"      every other    rho {rest[0]:+.3f}  n {rest[1]}  "
              f"p {rest[2]:.3g}")
        if (hot[0] == hot[0] and rest[0] == rest[0]
                and abs(hot[0]) + 0.1 < abs(rest[0])):
            print("      the busiest cell is markedly less coupled to the surf "
                  "than the rest of\n      the scene, which is what a fixed "
                  "object looks like.")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", nargs="*", default=None)
    parser.add_argument("--cell", type=int, default=CELL,
                        help="grid size in source pixels")
    parser.add_argument("--top", type=int, default=TOP)
    parser.add_argument("--detection-class", default=RIP_CLASS)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    found = cameras_on_disk()
    if args.cameras:
        wanted = [c.lower() for c in args.cameras]
        found = [f for f in found
                 if any(w in f[0].lower() or w in f[1].lower() for w in wanted)]
    if not found:
        sys.exit("No rip records on disk.")

    everything = []
    for camera, slug, path in found:
        print(f"\n{'=' * 72}\n{camera}\n{'=' * 72}")
        frame = rip_frames(path, args.detection_class)
        if frame is None:
            print(f"  no {args.detection_class} detections with a box centroid")
            continue
        everything += report(camera, frame, args.cell, args.top,
                             daily_waves(camera))

    if everything:
        os.makedirs(args.out_dir, exist_ok=True)
        target = f"{args.out_dir}/hot_cells.csv"
        pd.DataFrame(everything).to_csv(target, index=False)
        print(f"\n  wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
