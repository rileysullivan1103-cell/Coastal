"""Walton's driver results, re-run on rip frames under one score floor.

Every Walton driver number on file was computed over a record that does two
things at once. About 19% of its frames are a different model's COCO object
detections -- people, boats, kites -- counted as rip detections in the rate, in
score_max and in bbox_area_max. And on 2024-12-13 the detector's score floor
moved from 0.5075 to 0.7033, which analyze_detector_changepoints.py measured at
+30 sd raw and +35.7 sd residualized on waves, tide and cloud, with no model
version change. That is a CENSORING change: everything scoring 0.50-0.70
stopped being published. A pooled correlation across it is reading two
different instruments as one.

So each finding is re-run two ways, and both are printed beside the pooled
filtered number and beside what was originally reported:

  A. RE-CENSORED. Filter to rip classes, then demote every pre-era frame
     scoring under the floor to an observed zero, so the whole span is read at
     one 0.70 floor. Hours whose detections all go are observed zeros, not
     gaps: the frame stays in the denominator with detected=False.

  B. ERA SPLIT. Same class filter, no re-censoring, each era analysed on its
     own. B throws away the cross-era comparison and keeps every detection; A
     keeps the span and throws away the weak detections. They fail differently,
     which is why both are here: a finding that survives both is not an
     artifact of the threshold change under any reading.

THE APPROXIMATION IN A, STATED UP FRONT. The frame table carries score_max and
score_mean per FRAME, not a score per detection. So A can only demote whole
frames, and a pre-era frame holding a 0.9 detection and a 0.55 detection keeps
both. Those frames are counted and printed -- score_mean < floor <= score_max
proves at least one sub-threshold detection survived -- and that count is the
residual bias in A. It is reported, not hidden, because it is the one thing
that could make A look like B by accident.

score_max is reported as a DESCRIPTIVE ONLY, flagged everywhere it appears:
hand labelling of 360 stills found high-confidence frames no likelier to hold a
rip than low-confidence ones (3/116 against 3/116, Fisher p=1.00), so a driver
of score_max is a driver of something that does not discriminate rips.

analyze_drivers.py is not modified. Each variant is assembled by pointing
ad.DATA_DIR at a mirror of data/ -- real directories, symlinked files -- with
only rip_<slug>_hourly.csv replaced. The real data/ is never written to.

    python analyze_walton_eras.py
    python analyze_walton_eras.py --keep-workspace data/era_workspace
    python analyze_walton_eras.py --check-recensoring data/era_workspace/A
"""

import argparse
import contextlib
import glob
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

import analyze_drivers as ad
import analyze_glare as ag
import build_label_sample as bls
import pull_rip_detection as prd

DEFAULT_CAMERA = "walton"
DEFAULT_SPLIT = "2024-12-13"
DEFAULT_FLOOR = 0.70

# The three that answer "was a rip there, and how big did the detector call
# it". score_max is handled apart, under its caveat.
TARGETS = ["detection_rate", "detections", "bbox_area_max"]
DESCRIPTIVE = "score_max"
SCORE_CAVEAT = "does not discriminate rips per hand labels"

# Columns that describe a detection and must go when a frame is demoted. Left
# behind, an hour whose detections were all censored would still report the
# box area and confidence of the detection that is no longer there.
DETECTION_COLUMNS = ["score_max", "score_mean", "bbox_area_max", "bbox_count",
                     "bbox_x", "bbox_y", "rip_axis_deg"]

# What was reported over the unfiltered, un-split record, from HANDOFF.md.
# Quoted here so the new numbers are read against them rather than in place of
# them. These are prior results, not this run's output.
ORIGINAL = {
    "mop_buoy": {"detection_rate": 3.2, "detections": 2.9, "bbox_area_max": 1.5,
                 "n": 1884},
    "glare": {"target": "score_max", "dR2": 0.0248, "F": 78.43, "df2": 5824,
              "p": 2.4e-34},
    "cloud_box": {"rho": 0.122, "rho_hrmo": 0.128},
}
# Set once in main(). mop_vs_buoy needs the split to say whether A had any
# hours to re-censor, and threading it through every caller for one boolean
# would be worse than a module-level note that it is set exactly once.
SPLIT_HOURS = []

RAIN_COLUMNS = ["rain_24h_mm", "rain_48h_mm"]
RAIN_TARGETS = ["detection_rate", "detections"]


# ---------------------------------------------------------------------------
# frame-level surgery
# ---------------------------------------------------------------------------

def as_bool(series):
    """`detected` read back from CSV can be bool or the strings True/False.

    astype(bool) on the string column makes every row True, including "False",
    which would turn the entire observed-zero half of the record into
    detections without raising anything.
    """
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0) != 0
    return series.astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "t"})


def era_of(frames, split):
    """Boolean mask: True for frames before the split (the low-floor era)."""
    return frames["timestamp"] < split


def censor(frames, split, floor):
    """Demote pre-split frames that do not clear `floor` to observed zeros.

    DEMOTED, NOT DROPPED. A frame the detector looked at and (under the new
    floor) would not have published is a frame with no rip in it, which is an
    observed zero and belongs in the denominator. Dropping the row instead
    would remove it from `frames` in the hourly summary and quietly raise the
    detection rate of the very era being censored -- the opposite of the
    correction intended.
    """
    out = frames.copy()
    out["detected"] = as_bool(out["detected"])
    out["detection_count"] = pd.to_numeric(out.get("detection_count"),
                                           errors="coerce").fillna(0)
    score = pd.to_numeric(out.get("score_max"), errors="coerce")
    mean = pd.to_numeric(out.get("score_mean"), errors="coerce")

    pre = era_of(out, split)
    demote = pre & out["detected"] & (score < floor)

    stats = {
        "pre_frames": int(pre.sum()),
        "pre_detected": int((pre & out["detected"]).sum()),
        "pre_detections": float(out.loc[pre & out["detected"],
                                        "detection_count"].sum()),
        "demoted": int(demote.sum()),
        # score_mean < floor <= score_max PROVES a surviving detection under
        # the floor: the mean cannot sit below it unless something does.
        "residual_frames": int((pre & out["detected"] & (score >= floor)
                                & (mean < floor)).sum()),
        "post_frames": int((~pre).sum()),
        "post_detected": int(((~pre) & out["detected"]).sum()),
    }

    out.loc[demote, "detected"] = False
    out.loc[demote, "detection_count"] = 0
    for column in DETECTION_COLUMNS:
        if column in out.columns:
            out.loc[demote, column] = np.nan
    if "score_classes" in out.columns:
        out.loc[demote, "score_classes"] = np.nan

    hours = out["timestamp"].dt.floor("h")
    stats["pre_hours"] = int(hours[pre].nunique())
    stats["pre_hours_with_detection"] = int(
        hours[pre & as_bool(frames["detected"])].nunique())
    stats["pre_hours_with_detection_after"] = int(
        hours[pre & out["detected"]].nunique())
    stats["pre_detections_after"] = float(
        out.loc[pre & out["detected"], "detection_count"].sum())
    return out, stats


def post_era_violations(frames, split, floor):
    """(all-below, some-below) counts that prove the post era is not at `floor`.

    Two separate proofs, because score_max alone only catches the blatant case:

      * score_max < floor -- nothing in the frame reaches the floor at all.
      * score_mean < floor -- the mean of the frame's detections is under it,
        so at least one detection is, whatever score_max says.

    Either one means the 0.70 floor this whole exercise assumes is not there,
    and every number below would be re-censoring one era to match a floor the
    other never had.
    """
    detected = as_bool(frames["detected"])
    post = (~era_of(frames, split)) & detected
    score = pd.to_numeric(frames.get("score_max"), errors="coerce")
    mean = pd.to_numeric(frames.get("score_mean"), errors="coerce")
    all_below = frames[post & (score < floor)]
    some_below = frames[post & (score >= floor) & (mean < floor)]
    return all_below, some_below


# ---------------------------------------------------------------------------
# a data/ that analyze_drivers can read without data/ being touched
# ---------------------------------------------------------------------------

def mirror_data(source, destination, replacements, exclude=()):
    """A stand-in data/ directory: real dirs, symlinked files, some replaced.

    Every directory on the path to a replaced file is made REAL, and only its
    contents are linked. That is not cosmetic: writing the replacement into a
    symlinked directory would write it straight into the real data/, which is
    the one thing this whole mechanism exists to avoid. Directories with
    nothing replaced inside them stay a single symlink, so the per-camera
    payload folders cost one link rather than tens of thousands.

    `exclude` is a list of absolute paths to keep out of the mirror. It matters
    because glob's `**` DOES follow a symlinked directory -- so a workspace
    living under data/ would be linked back into every variant's mirror, and
    assemble_rip, globbing for rip_*_hourly.csv, would then find all four
    variants' tables under the same filename and analyse whichever sorted
    first. The variant labels would still print correctly over the wrong
    numbers, which is the worst way for this to fail.
    """
    replacements = {os.path.normpath(k): v for k, v in replacements.items()}
    exclude = {os.path.abspath(p) for p in exclude}
    real_dirs = set()
    for key in replacements:
        parts = key.split(os.sep)[:-1]
        for depth in range(1, len(parts) + 1):
            real_dirs.add(os.sep.join(parts[:depth]))
    _mirror(source, destination, "", real_dirs, exclude)

    for relative, frame in replacements.items():
        target = os.path.join(destination, relative)
        os.makedirs(os.path.dirname(target) or destination, exist_ok=True)
        if os.path.lexists(target):
            os.unlink(target)
        frame.to_csv(target, index=False)
    return destination


# Dropped into every workspace this module builds, so a LATER run mirroring
# data/ can recognise an earlier run's leftovers and step around them. Without
# it, excluding only the current workspace is not enough: HANDOFF tells people
# to keep data/era_workspace, and the next script to mirror data/ then finds
# four stale variant tables under the one filename assemble_rip globs for.
WORKSPACE_MARKER = ".coastal_variant_workspace"


def is_variant_workspace(path):
    return os.path.isfile(os.path.join(path, WORKSPACE_MARKER))


def _mirror(source, destination, prefix, real_dirs, exclude):
    os.makedirs(destination, exist_ok=True)
    if not os.path.isdir(source):
        return
    for entry in sorted(os.listdir(source)):
        src = os.path.join(source, entry)
        dst = os.path.join(destination, entry)
        relative = os.path.join(prefix, entry) if prefix else entry
        absolute = os.path.abspath(src)
        if absolute in exclude:
            continue
        if os.path.isdir(src) and is_variant_workspace(src):
            continue
        # A directory that CONTAINS an excluded path is still mirrored, as a
        # real directory, so the rest of its contents survive.
        contains = any(e.startswith(absolute + os.sep) for e in exclude)
        if os.path.isdir(src) and (relative in real_dirs or contains):
            _mirror(src, dst, relative, real_dirs, exclude)
        elif not os.path.lexists(dst):
            os.symlink(absolute, dst)


@contextlib.contextmanager
def data_dir(path):
    """Point analyze_drivers (and, through it, analyze_glare) at a mirror."""
    previous = ad.DATA_DIR
    ad.DATA_DIR = path
    try:
        yield path
    finally:
        ad.DATA_DIR = previous


def build_variant(workspace, label, table, slug):
    """Write one variant's hourly table into its own mirror of data/."""
    folder = os.path.join(workspace, label)
    os.makedirs(folder, exist_ok=True)
    marker = os.path.join(workspace, WORKSPACE_MARKER)
    if not os.path.exists(marker):
        with open(marker, "w") as handle:
            handle.write("Written by analyze_walton_eras.build_variant.\n"
                         "Its presence tells a later mirror of data/ to skip "
                         "this directory:\nthe rip_*_hourly.csv files under it "
                         "are variant tables, not cameras.\n")
    scratch = os.path.join(folder, f"rip_{slug}_hourly.csv")
    hourly = prd.hourly_summary(table, scratch)
    if hourly is None or hourly.empty:
        print(f"  {label}: no hours survive; variant skipped")
        return None
    mirror = os.path.join(folder, "data")
    relative = os.path.join("rip_detection", f"rip_{slug}_hourly.csv")
    mirror_data(ad.DATA_DIR, mirror, {relative: hourly},
                exclude=[workspace])

    # The guard, not a formality. assemble_rip picks hits[0] out of a glob, so
    # a second table under this stem anywhere in the mirror means the variant
    # label above the numbers stops describing the numbers.
    found = glob.glob(os.path.join(mirror, "**", f"rip_{slug}_hourly.csv"),
                      recursive=True)
    expected = os.path.join(mirror, relative)
    if found != [expected]:
        sys.exit(f"  {label}: the mirror holds {len(found)} table(s) named "
                 f"rip_{slug}_hourly.csv:\n    "
                 + "\n    ".join(found)
                 + f"\n  Exactly one was expected, at {expected}. "
                 "assemble_rip globs for that\n  name and takes the first "
                 "match, so this run would have labelled one\n  variant's "
                 "numbers with another variant's name.\n"
                 f"\n  A workspace this tool built carries a {WORKSPACE_MARKER} "
                 "file and is skipped\n  automatically. A stray table under "
                 "data/ that has no marker has to go by\n  hand — delete it, "
                 "or move --keep-workspace outside the mirrored tree.")
    return mirror


# ---------------------------------------------------------------------------
# assembling one variant into an hourly frame with conditions on it
# ---------------------------------------------------------------------------

def assemble(mirror, sites, want, camera):
    """The joined hourly frame for one variant, with cloud and solar attached."""
    with data_dir(mirror):
        frame, name, has_coverage = ad.assemble_rip(sites, want=want)
        if frame is None:
            return None
        observed = (frame.copy() if has_coverage
                    else frame[frame.get("frames", 0) > 0].copy())
        if observed.empty:
            return None
        observed["month"] = observed["hour"].dt.month
        observed["hr_mo"] = (observed["hour_of_day"].astype(str) + "-"
                             + observed["month"].astype(str))
        observed, cloud_source = ag.load_cloud_sidecar(observed, name)
    lat, lon = float(camera["lat"]), float(camera["lon"])
    weather = camera.get("weather_name")
    weather = name if not isinstance(weather, str) or not weather else weather
    bearing, _ = ag.seaward_bearing(name, weather)
    ag.add_solar(observed, lat, lon, bearing)
    return {"observed": observed, "name": name, "bearing": bearing,
            "cloud": (ag.CLOUD_COLUMN in observed.columns
                      and observed[ag.CLOUD_COLUMN].notna().any()),
            "cloud_source": cloud_source, "has_coverage": has_coverage}


# ---------------------------------------------------------------------------
# the four findings
# ---------------------------------------------------------------------------

def demeaned_rho(frame, predictor, target, key=None):
    """(rho, n, p), optionally with each group's own mean removed from both."""
    if predictor not in frame.columns or target not in frame.columns:
        return np.nan, 0, np.nan
    left, right = frame[predictor], frame[target]
    if key is not None:
        left, right = ad.demean_by(left, frame[key]), ad.demean_by(right, frame[key])
    return ad.spearman(left, right)


def mop_vs_buoy(observed, targets):
    """Nearshore model against the distant buoy, on the hours both cover.

    Restricted to matched hours and demeaned WITHIN them, which is what the
    original ratio did. Demeaning first and restricting afterwards would remove
    each variable's mean over a different set of hours and make the two sides
    incomparable.
    """
    if not {"mop_wave_height", "WVHT"} <= set(observed.columns):
        return None
    both = observed.dropna(subset=["mop_wave_height", "WVHT"]).copy()
    if both.empty:
        return None
    rows = []
    # How many matched hours A actually re-censors. At Walton the answer is
    # zero: buoy 46236's record starts well after the split, so every hour
    # where MOP and the buoy both report is post-era, and A's series on those
    # hours is pooled's series unchanged. Printing a verdict for A there would
    # be reporting the pooled number under A's name.
    pre_hours = int((both["hour"] < SPLIT_HOURS[0]).sum()) if SPLIT_HOURS \
        else -1
    for target in targets:
        if target not in both.columns:
            continue
        sub = both.dropna(subset=[target])
        if sub.empty:
            continue
        mop_rho, n, mop_p = demeaned_rho(sub, "mop_wave_height", target, "hr_mo")
        buoy_rho, _, buoy_p = demeaned_rho(sub, "WVHT", target, "hr_mo")
        ratio = (abs(mop_rho) / abs(buoy_rho)
                 if pd.notna(mop_rho) and pd.notna(buoy_rho) and buoy_rho
                 else np.nan)
        rows.append({"target": target, "n": n, "mop_rho": mop_rho,
                     "mop_p": mop_p, "buoy_rho": buoy_rho, "buoy_p": buoy_p,
                     "ratio": ratio, "pre_hours": pre_hours})
    return pd.DataFrame(rows) if rows else None


def rain_table(observed, targets=RAIN_TARGETS, columns=RAIN_COLUMNS):
    rows = []
    for column in columns:
        for target in targets:
            rho, n, p = demeaned_rho(observed, column, target)
            rho_c, _, p_c = demeaned_rho(observed, column, target, "hr_mo")
            rows.append({"predictor": column, "target": target, "n": n,
                         "rho": rho, "p": p, "rho_hrmo": rho_c, "p_hrmo": p_c})
    if not rows:
        return pd.DataFrame(columns=["predictor", "target", "n", "rho", "p",
                                     "rho_hrmo", "p_hrmo"])
    return pd.DataFrame(rows)


def rain_reversals(pre, post, alpha=0.05):
    """Pairs whose sign flips between eras with both sides significant.

    Both sides significant, because a sign flip between a p=0.9 and a p=0.8 is
    not a reversal, it is two absences of evidence pointed different ways.
    """
    merged = pre.merge(post, on=["predictor", "target"], suffixes=("_pre", "_post"))
    flipped = merged[
        (np.sign(merged["rho_pre"]) != np.sign(merged["rho_post"]))
        & merged["rho_pre"].notna() & merged["rho_post"].notna()
        & (merged["p_pre"] < alpha) & (merged["p_post"] < alpha)]
    return merged, flipped


def glare_test(observed, target, base, rungs, bearing_base):
    """(dR2, F, p) for sun_in_view + sun_glare over the comparator rung."""
    terms = [c for c in ("sun_in_view", "sun_glare") if c in observed.columns]
    if len(terms) != 2 or target not in observed.columns:
        return None
    rows = ag.ladder(observed, target, base, rungs)
    by_model = {r["model"]: r["fit"] for r in rows}
    small, big = by_model.get(bearing_base), by_model.get("+ glare geometry")
    gain, f_stat, p_value = ag.added_term_test(small, big, len(terms))
    n = big["n"] if big else 0
    df2 = (n - len(big["names"]) - 1) if big and big.get("beta") is not None \
        else np.nan
    return {"dR2": gain, "F": f_stat, "p": p_value, "n": n, "df2": df2,
            "comparator": bearing_base}


def cloud_box(observed):
    if ag.CLOUD_COLUMN not in observed.columns:
        return None
    rho, n, p = demeaned_rho(observed, ag.CLOUD_COLUMN, "bbox_area_max")
    rho_c, _, p_c = demeaned_rho(observed, ag.CLOUD_COLUMN, "bbox_area_max",
                                 "hr_mo")
    return {"rho": rho, "n": n, "p": p, "rho_hrmo": rho_c, "p_hrmo": p_c}


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "     —"
    return f"{value:+.{digits}f}" if isinstance(value, float) else str(value)


def fmt_p(value):
    if value is None or pd.isna(value):
        return "    —"
    # spearman returns exactly 0 where |rho| >= 1, and a printed "0" reads as
    # a p-value of zero rather than as "the rank order is identical".
    if value == 0:
        return "<1e-300"
    return f"{value:.2g}"


def verdict_line(name, survives, detail):
    mark = {True: "SURVIVES", False: "DOES NOT SURVIVE",
            None: "CANNOT TELL"}[survives]
    print(f"    {name:<24} {mark:<18} {detail}")


# ---------------------------------------------------------------------------
# the check that would show A did nothing
# ---------------------------------------------------------------------------

def check_recensoring(folder, split, floor):
    """Read an hourly table back off disk and say whether the floor is one floor.

    Point this at the A mirror and it should say APPLIED. Point it at the real
    data/rip_detection and it must say NOT APPLIED -- that is the negative
    control, and a run where both say the same thing is a run where the
    re-censoring silently did not happen.
    """
    paths = sorted(glob.glob(os.path.join(folder, "**", "rip_*_hourly.csv"),
                             recursive=True))
    if not paths:
        print(f"  no rip_*_hourly.csv under {folder}")
        return 1
    split = pd.Timestamp(split)
    if split.tzinfo is None:
        split = split.tz_localize("UTC")
    applied = []
    for path in paths:
        frame = pd.read_csv(path)
        if "score_max" not in frame.columns or "hour" not in frame.columns:
            print(f"  {path}: no score_max/hour column; nothing to check")
            continue
        frame["hour"] = ad.to_hour(frame["hour"])
        score = pd.to_numeric(frame["score_max"], errors="coerce")
        pre = frame["hour"] < split
        for label, mask in (("pre", pre), ("post", ~pre)):
            scores = score[mask].dropna()
            if scores.empty:
                print(f"  {os.path.basename(path)}  {label:<4} era: no hours")
                continue
            below = int((scores < floor).sum())
            state = "APPLIED" if below == 0 else "NOT APPLIED"
            if label == "pre":
                applied.append(below == 0)
            print(f"  {os.path.basename(path)}  {label:<4} era: "
                  f"{len(scores)} hours with a detection, min score_max "
                  f"{scores.min():.4f}, {below} below {floor:.2f}  -> {state}")
    if applied:
        print(f"\n  {sum(applied)} of {len(applied)} file(s) read as APPLIED "
              "for the pre era.")
    print("\n  A pre-era minimum under the floor means the re-censoring did not"
          "\n  reach this file. Against the untouched data/rip_detection that is"
          "\n  the expected answer, and is what 'not applied' looks like: run "
          "both\n  and compare, because a variant that silently fell back to "
          "the original\n  file reads exactly like a variant where the "
          "censoring changed nothing.")
    return 0


# ---------------------------------------------------------------------------

def resolve_camera(sites, want):
    slug_want = ad.rip_slug(want)
    hits = [s for _, s in sites.iterrows()
            if slug_want in ad.rip_slug(s["camera_name"])]
    if not hits:
        print(f"No site whose slug contains {slug_want!r}. Known:")
        for _, s in sites.iterrows():
            print(f"  {ad.rip_slug(s['camera_name'])}")
        sys.exit(1)
    camera = hits[0]
    return camera, ad.rip_slug(camera["camera_name"])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default=DEFAULT_CAMERA,
                        help="substring of the camera slug (default walton)")
    parser.add_argument("--split", default=DEFAULT_SPLIT,
                        help=f"era boundary, UTC (default {DEFAULT_SPLIT})")
    parser.add_argument("--floor", type=float, default=DEFAULT_FLOOR,
                        help=f"single score floor for variant A (default {DEFAULT_FLOOR})")
    parser.add_argument("--classes", default=",".join(bls.RIP_CLASSES),
                        help="detection classes to keep (default rip_current,rip)")
    parser.add_argument("--keep-workspace", default=None,
                        help="write the variant mirrors here instead of a temp dir")
    parser.add_argument("--check-recensoring", default=None, metavar="DIR",
                        help="read rip_*_hourly.csv under DIR and report the "
                             "per-era score floor, then exit")
    parser.add_argument("--top", type=int, default=8,
                        help="predictors shown per correlation table (default 8)")
    args = parser.parse_args()

    split = pd.Timestamp(args.split, tz="UTC")
    SPLIT_HOURS.clear()
    SPLIT_HOURS.append(split)
    if args.check_recensoring:
        print(f"\nPER-ERA SCORE FLOOR IN {args.check_recensoring}")
        print(f"  split {split:%Y-%m-%d}, floor {args.floor:.2f}\n")
        return check_recensoring(args.check_recensoring, split, args.floor)

    sites = ad.load_sites()
    camera, slug = resolve_camera(sites, args.camera)
    name = camera["camera_name"]
    wanted = tuple(c.strip() for c in args.classes.split(",") if c.strip())

    print(f"\n{'=' * 78}")
    print(f"DRIVER RE-RUN UNDER ONE SCORE FLOOR — {name}")
    print("=" * 78)
    print(f"  era boundary {split:%Y-%m-%d} UTC, floor {args.floor:.2f}, "
          f"classes kept: {', '.join(wanted)}")
    print("  A = re-censored to one floor over the whole span")
    print("  B = each era analysed on its own, nothing censored")
    print(f"  {DESCRIPTIVE} is DESCRIPTIVE ONLY throughout: {SCORE_CAVEAT}")

    frames = bls.load_frames(slug)
    print(f"\n--- frame table ---")
    print(f"  {len(frames)} frames, {frames['timestamp'].min():%Y-%m-%d} to "
          f"{frames['timestamp'].max():%Y-%m-%d}")
    frames, dropped = bls.keep_class(frames, wanted=wanted, label="")
    print(f"  {len(frames)} frames after the class filter "
          f"({dropped} dropped, blank-class observed zeros kept)")
    if frames.empty:
        sys.exit("  nothing left after the class filter")
    frames["detected"] = as_bool(frames["detected"])

    # -- the assertion, before anything is computed on the result --
    all_below, some_below = post_era_violations(frames, split, args.floor)
    print(f"\n--- assertion: the post era is at {args.floor:.2f} ---")
    print(f"  post-era detected frames: "
          f"{int(((~era_of(frames, split)) & frames['detected']).sum())}")
    print(f"  with score_max  < {args.floor:.2f} (nothing in the frame clears it): "
          f"{len(all_below)}")
    print(f"  with score_mean < {args.floor:.2f} (something in it does not): "
          f"{len(some_below)}")
    if len(all_below) or len(some_below):
        print("\n  *** THE POST ERA IS NOT AT THIS FLOOR ***")
        offenders = pd.concat([all_below, some_below]).sort_values("timestamp")
        columns = [c for c in ("timestamp", "score_max", "score_mean",
                               "detection_count", "model_version")
                   if c in offenders.columns]
        print(offenders[columns].head(12).to_string(index=False))
        sys.exit(f"\n  {len(offenders)} post-era detections sit under "
                 f"{args.floor:.2f}. Re-censoring the pre era to match a floor\n"
                 "  the post era does not have would manufacture the very "
                 "comparability it\n  claims to restore. Fix --floor or "
                 "--split, or re-check the changepoint.")
    print("  clear: every post-era detection clears the floor.")

    # -- variant A --
    censored, stats = censor(frames, split, args.floor)
    print(f"\n--- pre-flight: what re-censoring the pre era costs ---")
    print(f"  pre-era frames:                      {stats['pre_frames']}")
    print(f"  pre-era frames with a detection:     {stats['pre_detected']}"
          f"  ->  {stats['pre_detected'] - stats['demoted']}")
    print(f"  pre-era detections (sum of counts):  "
          f"{stats['pre_detections']:.0f}  ->  {stats['pre_detections_after']:.0f}")
    print(f"  pre-era hours with a detection:      "
          f"{stats['pre_hours_with_detection']}  ->  "
          f"{stats['pre_hours_with_detection_after']}")
    print(f"  pre-era hours with imagery:          {stats['pre_hours']}"
          "   (unchanged — demoted frames stay in the denominator)")
    print(f"  frames demoted to observed zeros:    {stats['demoted']}"
          f"  ({stats['demoted'] / max(stats['pre_detected'], 1):.1%} of pre-era"
          " detections)")
    print(f"\n  RESIDUAL BIAS IN A: {stats['residual_frames']} pre-era frames "
          f"have score_mean < {args.floor:.2f} <= score_max,")
    print("  so they hold at least one detection under the floor that A cannot")
    print("  remove — the table has no per-detection score. A is therefore a")
    print("  floor on whole frames, not on detections, and is slightly LESS")
    print("  censored than the post era. Read every A number with that slack.")

    workspace = args.keep_workspace or tempfile.mkdtemp(prefix="walton_eras_")
    os.makedirs(workspace, exist_ok=True)
    pre_mask = era_of(frames, split)

    print(f"\n--- building the variant hourly tables in {workspace} ---")
    specs = [("pooled", frames), ("A", censored),
             ("B_pre", frames[pre_mask]), ("B_post", frames[~pre_mask])]
    variants = {}
    try:
        mirrors = {}
        for label, table in specs:
            if table is None or table.empty:
                print(f"  {label}: no frames")
                continue
            mirror = build_variant(workspace, label, table, slug)
            if mirror:
                mirrors[label] = mirror

        for label, mirror in mirrors.items():
            print(f"\n{'-' * 78}")
            print(f"ASSEMBLING {label}")
            print("-" * 78)
            built = assemble(mirror, sites, slug, camera)
            if built is None:
                print(f"  {label}: nothing assembled")
                continue
            variants[label] = built
            observed = built["observed"]
            print(f"  {len(observed)} hours, {observed['hour'].min():%Y-%m-%d}"
                  f" to {observed['hour'].max():%Y-%m-%d}; cloud "
                  f"{'present' if built['cloud'] else 'ABSENT'}")

        if not variants:
            sys.exit("no variant assembled; nothing to compare")
        report(variants, args)
    finally:
        if not args.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)
        else:
            print(f"\n  workspace kept at {workspace}")
            print("  the command that would show the re-censoring did NOT apply:")
            print(f"    python analyze_walton_eras.py --check-recensoring "
                  f"{os.path.join(workspace, 'A', 'data', 'rip_detection')}")
            print("    python analyze_walton_eras.py --check-recensoring "
                  "data/rip_detection")
            print("    the first must say APPLIED for the pre era and the "
                  "second NOT APPLIED;\n    if they agree, A did nothing and "
                  "every A number above is a copy of pooled.")
    return 0


ORDER = ["pooled", "A", "B_pre", "B_post"]


def present(variants):
    return [label for label in ORDER if label in variants]


def report(variants, args):
    print(f"\n{'=' * 78}")
    print("RANKED DRIVERS, SIDE BY SIDE")
    print("=" * 78)
    print("  'pooled' is the class-filtered whole record with no floor "
          "correction:\n  the closest thing to the original run, and the "
          "comparator for A and B.")
    for target in TARGETS + [DESCRIPTIVE]:
        header = target + (f"   <- DESCRIPTIVE ONLY: {SCORE_CAVEAT}"
                           if target == DESCRIPTIVE else "")
        print(f"\n{'-' * 78}\n{header}\n{'-' * 78}")
        for label in present(variants):
            observed = variants[label]["observed"]
            if target not in observed.columns:
                print(f"\n  {label}: no {target} column")
                continue
            ad.report_correlations(
                observed, target, ad.RIP_PREDICTORS,
                controls=[("hr", observed["hour_of_day"]),
                          ("mo", observed["month"]),
                          ("hrmo", observed["hr_mo"])],
                title=f"  [{label}] {target}", top=args.top)

    print(f"\n{'=' * 78}")
    print("THE FOUR PROVISIONAL FINDINGS")
    print("=" * 78)
    finding_mop_buoy(variants)
    finding_rain(variants, args)
    finding_glare(variants)
    finding_cloud_box(variants)

    print(f"\n{'=' * 78}")
    print("HOW TO READ THE VERDICTS")
    print("=" * 78)
    print("  SURVIVES means the finding is there under that variant on its own")
    print("  terms, stated with the rule used. It does NOT mean the finding is")
    print("  about rips: precision on this camera is 2.7% population-weighted,")
    print(f"  and {DESCRIPTIVE} {SCORE_CAVEAT}. A driver of a 2.7%-precision")
    print("  detector is a driver of what the detector fires on, which is")
    print("  mostly not a rip. That caveat is not removed by either variant.")


def finding_mop_buoy(variants):
    print("\n--- 1. the nearshore model beats the distant buoy ---")
    original = ORIGINAL["mop_buoy"]
    print(f"  originally: detection_rate {original['detection_rate']}x, "
          f"detections {original['detections']}x, bbox_area_max "
          f"{original['bbox_area_max']}x, on {original['n']} matched hours")
    print("  rule: survives where |rho_mop| / |rho_buoy| > 1 with the MOP side "
          "significant.")
    print("  'pre hrs' is how many of the matched hours fall before the "
          "split. Where it\n  is 0 the buoy never reported in the low-floor "
          "era, so A has nothing to\n  re-censor here and B_pre has nothing "
          "to compute.")
    print(f"\n  {'variant':<8}{'target':<16}{'n':>6}{'mop rho':>10}{'p':>10}"
          f"{'buoy rho':>10}{'p':>10}{'ratio':>8}{'pre hrs':>10}")
    verdicts = {}
    for label in present(variants):
        table = mop_vs_buoy(variants[label]["observed"], TARGETS)
        if table is None or table.empty:
            print(f"  {label:<8}no hours where both MOP and the buoy report")
            verdicts[label] = None
            continue
        for _, row in table.iterrows():
            ratio = (f"{row['ratio']:.2f}" if pd.notna(row["ratio"]) else "—")
            print(f"  {label:<8}{row['target']:<16}{row['n']:>6}"
                  f"{fmt(row['mop_rho']):>10}{fmt_p(row['mop_p']):>10}"
                  f"{fmt(row['buoy_rho']):>10}{fmt_p(row['buoy_p']):>10}"
                  f"{ratio:>8}{int(row['pre_hours']):>10}")
        verdicts[label] = table
    print()
    # One verdict per target, not one per variant. The original finding is
    # three separate ratios, and a single line reading DOES NOT SURVIVE
    # because the weakest of the three fell over would bury the two that
    # held -- which is the opposite of what a side-by-side is for.
    for label in present(variants):
        table = verdicts.get(label)
        if table is None:
            verdict_line(label, None, "MOP and buoy never overlap here")
            continue
        vacuous = label == "A" and int(table["pre_hours"].iloc[0]) == 0
        if vacuous:
            verdict_line("A", None,
                         "A cannot test this: no matched hour lies before the "
                         "split,")
            print(f"    {'':<24} {'':<18} so A's series here IS pooled's and "
                  "the row above is a copy.")
            continue
        for _, row in table.iterrows():
            survives = bool(pd.notna(row["ratio"]) and row["ratio"] > 1
                            and pd.notna(row["mop_p"]) and row["mop_p"] < 0.05)
            detail = (f"ratio {row['ratio']:.2f}" if pd.notna(row["ratio"])
                      else "no ratio: the buoy correlation is zero or absent")
            if pd.notna(row["mop_p"]) and row["mop_p"] >= 0.05:
                detail += f", but the MOP side is not significant (p={row['mop_p']:.2g})"
            verdict_line(f"{label} / {row['target']}", survives, detail)


def finding_rain(variants, args):
    print("\n--- 2. the rain correlations reverse between eras ---")
    print("  originally: rain_24h_mm and rain_48h_mm significantly NEGATIVE in")
    print("  2024 and POSITIVE in 2026, on detection_rate and detections, "
          "either\n  side of the split — so the pooled rain number was never "
          "quotable.")
    print("  rule: the reversal survives where a predictor/target pair flips "
          "sign\n  across the eras with BOTH sides significant at p<0.05.")

    split = pd.Timestamp(args.split, tz="UTC")
    pairs = {}
    if "A" in variants:
        observed = variants["A"]["observed"]
        pairs["A"] = (rain_table(observed[observed["hour"] < split]),
                      rain_table(observed[observed["hour"] >= split]))
    if "B_pre" in variants and "B_post" in variants:
        pairs["B"] = (rain_table(variants["B_pre"]["observed"]),
                      rain_table(variants["B_post"]["observed"]))
    if "pooled" in variants:
        # The original test, as it was run: one record, split by date only for
        # this table. It should track B closely -- B differs only in that each
        # era's coverage is clipped to its own span -- and a wide gap between
        # the two would mean that clipping is doing more than bookkeeping.
        observed = variants["pooled"]["observed"]
        pairs["pooled"] = (rain_table(observed[observed["hour"] < split]),
                           rain_table(observed[observed["hour"] >= split]))

    for label in ("pooled", "A", "B"):
        if label not in pairs:
            continue
        pre, post = pairs[label]
        merged, flipped = rain_reversals(pre, post)
        print(f"\n  [{label}]  {'predictor':<14}{'target':<16}"
              f"{'pre n':>7}{'pre rho':>10}{'p':>10}"
              f"{'post n':>8}{'post rho':>10}{'p':>10}")
        for _, row in merged.iterrows():
            flag = "   <- reverses" if (
                pd.notna(row["rho_pre"]) and pd.notna(row["rho_post"])
                and np.sign(row["rho_pre"]) != np.sign(row["rho_post"])
                and row["p_pre"] < 0.05 and row["p_post"] < 0.05) else ""
            print(f"          {row['predictor']:<14}{row['target']:<16}"
                  f"{int(row['n_pre']):>7}{fmt(row['rho_pre']):>10}"
                  f"{fmt_p(row['p_pre']):>10}"
                  f"{int(row['n_post']):>8}{fmt(row['rho_post']):>10}"
                  f"{fmt_p(row['p_post']):>10}{flag}")
        verdict_line(label, len(flipped) > 0,
                     f"{len(flipped)} of {len(merged)} pairs reverse with both "
                     "sides significant")
    if "A" in pairs:
        print("\n  A splits the SAME re-censored series at the same date, so a")
        print("  reversal that survives A is not the floor change: both sides "
              "of\n  A's split are read at one floor.")


def finding_glare(variants):
    print("\n--- 3. glare geometry earns its place ---")
    original = ORIGINAL["glare"]
    print(f"  originally: {original['target']} dR2 {original['dR2']:+.4f}, "
          f"F(2,{original['df2']}) = {original['F']}, p = {original['p']:.1g}")
    print(f"  NOTE: that was on {DESCRIPTIVE}, which {SCORE_CAVEAT}. It is "
          "repeated\n  here for continuity and tested on the three real "
          "targets as well.")
    print("  rule: survives where dR2 > 0 with p < 0.01 for the bearing pair.")
    print(f"\n  {'variant':<8}{'target':<16}{'n':>7}{'dR2':>10}{'F':>10}"
          f"{'p':>12}")
    for label in present(variants):
        built = variants[label]
        observed = built["observed"]
        height, period = ag.wave_columns(observed)
        if height not in observed.columns:
            print(f"  {label:<8}no wave height column; the ladder has no base")
            continue
        base = [c for c in [height, period] + ag.BASE_OTHER
                if c in observed.columns]
        rungs, bearing_base = ag.build_rungs(built["cloud"])
        survived = []
        for target in TARGETS + [DESCRIPTIVE]:
            result = glare_test(observed, target, base, rungs, bearing_base)
            if result is None:
                continue
            mark = "   <- DESCRIPTIVE ONLY" if target == DESCRIPTIVE else ""
            f_stat = "—" if pd.isna(result["F"]) else f"{result['F']:.2f}"
            print(f"  {label:<8}{target:<16}{result['n']:>7}"
                  f"{fmt(result['dR2'], 4):>10}{f_stat:>10}"
                  f"{fmt_p(result['p']):>12}{mark}")
            if target != DESCRIPTIVE and pd.notna(result["dR2"]) \
                    and result["dR2"] > 0 and pd.notna(result["p"]) \
                    and result["p"] < 0.01:
                survived.append(target)
        verdict_line(label, len(survived) > 0,
                     f"bearing terms do significant work on: "
                     f"{', '.join(survived) if survived else 'no real target'}"
                     f"  (comparator {bearing_base})")


def finding_cloud_box(variants):
    print("\n--- 4. cloud cover raises the largest box ---")
    original = ORIGINAL["cloud_box"]
    print(f"  originally at Walton: rho {original['rho']:+.3f} "
          f"({original['rho_hrmo']:+.3f} with hour and month removed).")
    print("  Virginia Beach's half (+0.247 / +0.211) is clean and is not "
          "re-run here.")
    print("  rule: survives where rho_hrmo > 0 at p < 0.05.")
    print(f"\n  {'variant':<8}{'n':>7}{'rho':>10}{'p':>10}{'rho_hrmo':>11}"
          f"{'p':>10}")
    for label in present(variants):
        result = cloud_box(variants[label]["observed"])
        if result is None:
            print(f"  {label:<8}no cloud column on disk for this variant")
            verdict_line(label, None, "cloud_cover absent")
            continue
        print(f"  {label:<8}{result['n']:>7}{fmt(result['rho']):>10}"
              f"{fmt_p(result['p']):>10}{fmt(result['rho_hrmo']):>11}"
              f"{fmt_p(result['p_hrmo']):>10}")
        survives = (pd.notna(result["rho_hrmo"]) and result["rho_hrmo"] > 0
                    and pd.notna(result["p_hrmo"]) and result["p_hrmo"] < 0.05)
        verdict_line(label, bool(survives),
                     f"rho_hrmo {fmt(result['rho_hrmo'])} at p={fmt_p(result['p_hrmo'])}")


if __name__ == "__main__":
    raise SystemExit(main())
