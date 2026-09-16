"""Find abrupt shifts in WebCOOS detector behaviour, independent of conditions.

The question this answers is not "what drives rips" but "did the thing doing
the measuring change under us". A retrained model, a raised confidence
threshold or a redeployed pipeline moves every downstream number, and it does
so on a DATE rather than with the weather -- which is what makes it findable
without knowing anything about the site.

Five daily metrics per camera:

    detection_rate     detections per hour the camera was up
    score_median       median score_max of that day's detections
    score_p90          90th percentile of the same
    score_floor        smallest NONZERO score_max seen that day
    bbox_area_median   median bbox_area_max

score_floor earns its place: a confidence threshold is a floor, so raising one
from 0.25 to 0.40 moves this metric immediately and cleanly while barely
touching the median. It is the sharpest single indicator of a calibration
change in the list, and the only one whose movement is hard to explain by
weather.

For cameras named with --residualize (Walton by default), each metric is also
regressed on daily MOP wave height, tide and cloud cover, and the changepoint
search is run again on the residuals. A month of big surf raises detection
rate on its own; the residual series asks whether anything moved that the
weather does not account for. Both are reported, because a shift that
survives residualization is a much stronger claim than one that does not.

The small cameras are too thin to fit drivers on and are NOT residualized --
see --min-driver-days. Their raw series are still perfectly good for dates,
which is all this script is for, and the report says which cameras got which
treatment rather than leaving it to be inferred.

Changepoints come from binary segmentation with the number of splits chosen by
BIC. ruptures is used when it is installed, as a cross-check against the
built-in implementation; neither is required, and the report says which ran.

    python analyze_detector_changepoints.py --spans
    python analyze_detector_changepoints.py

Reads only what is already on disk under data/. Makes no network calls.
"""

import argparse
import glob
import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import analyze_drivers as ad
import build_label_sample as bls
import diagnose_class_timeline as ct

OUT_DIR = f"{ad.DATA_DIR}/changepoints"
RESIDUALIZE_DEFAULT = ["Walton Lighthouse, Santa Cruz, CA"]

METRICS = ["detection_rate", "score_median", "score_p90", "score_floor",
           "bbox_area_median"]

MIN_FRAMES_PER_DAY = 5
MIN_SEGMENT_DAYS = 14
MAX_CHANGEPOINTS = 6
MIN_DRIVER_DAYS = 120
COINCIDE_DAYS = 7

HIST_BINS = np.round(np.arange(0.0, 1.0001, 0.05), 4)


# ---------------------------------------------------------------- changepoints

def _best_split(values, lo, hi, min_size):
    """(index, RSS gain) of the best single split of values[lo:hi], or (None, 0).

    Cumulative sums rather than a loop of means: the search is over every
    admissible cut in the segment, and doing that with a Python loop over a
    900-day series inside a greedy outer loop is slow enough to discourage
    running it, which is its own kind of bug.
    """
    n = hi - lo
    if n < 2 * min_size:
        return None, 0.0
    seg = np.asarray(values[lo:hi], dtype=float)
    csum = np.concatenate([[0.0], np.cumsum(seg)])
    csq = np.concatenate([[0.0], np.cumsum(seg ** 2)])
    total = csq[n] - csum[n] ** 2 / n

    cuts = np.arange(min_size, n - min_size + 1)
    if cuts.size == 0:
        return None, 0.0
    left = csq[cuts] - csum[cuts] ** 2 / cuts
    right_n = n - cuts
    right = (csq[n] - csq[cuts]) - (csum[n] - csum[cuts]) ** 2 / right_n
    gains = total - left - right
    best = int(np.argmax(gains))
    if not np.isfinite(gains[best]) or gains[best] <= 0:
        return None, 0.0
    return lo + int(cuts[best]), float(gains[best])


def binary_segmentation(values, max_k=MAX_CHANGEPOINTS, min_size=MIN_SEGMENT_DAYS):
    """Split indices in the order the greedy search found them.

    Returned in DISCOVERY order, not sorted, because the BIC step below has to
    be able to take the first k of them: the sequence of models it compares is
    nested only if k=2 means "the two best splits", not "the two earliest".
    """
    n = len(values)
    splits, bounds = [], [0, n]
    for _ in range(max_k):
        best_gain, best_at = 0.0, None
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            at, gain = _best_split(values, lo, hi, min_size)
            if at is not None and gain > best_gain:
                best_gain, best_at = gain, at
        if best_at is None:
            break
        splits.append(best_at)
        bounds = sorted(bounds + [best_at])
    return splits


def piecewise_rss(values, splits):
    bounds = [0] + sorted(splits) + [len(values)]
    total = 0.0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        seg = np.asarray(values[lo:hi], dtype=float)
        if seg.size:
            total += float(((seg - seg.mean()) ** 2).sum())
    return total


def choose_k_bic(values, splits):
    """How many of the greedy splits BIC keeps, and the table it chose from.

    BIC for a piecewise-constant mean with common variance:

        n log(RSS/n) + 2k log(n)

    k locations and k+1 means, plus a variance, is 2k+2 parameters; the +2 is
    the same for every k and drops out of the comparison. Without a penalty the
    RSS falls monotonically with k and the answer is always max_k, which is how
    a changepoint search returns six confident dates from pure noise.
    """
    n = len(values)
    table, best_k, best_bic = [], 0, None
    for k in range(len(splits) + 1):
        rss = piecewise_rss(values, splits[:k])
        rss = max(rss, 1e-12)
        bic = n * math.log(rss / n) + 2 * k * math.log(n) if n > 1 else 0.0
        table.append({"k": k, "rss": rss, "bic": bic})
        if best_bic is None or bic < best_bic - 1e-9:
            best_k, best_bic = k, bic
    return best_k, table


def detect(values, max_k=MAX_CHANGEPOINTS, min_size=MIN_SEGMENT_DAYS):
    """Sorted split indices BIC keeps. Index i means "values[i] starts a new run"."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 * min_size or not np.isfinite(values).all():
        return []
    if float(np.std(values)) == 0.0:
        return []
    splits = binary_segmentation(values, max_k, min_size)
    keep, _ = choose_k_bic(values, splits)
    return sorted(splits[:keep])


def ruptures_cross_check(values, max_k, min_size, penalty):
    """The same question asked of ruptures, if it happens to be installed.

    Not required and not authoritative -- it is here so a disagreement between
    two implementations shows up as a disagreement rather than as confidence.
    """
    try:
        import ruptures
    except ImportError:
        return None
    try:
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        algo = ruptures.Pelt(model="l2", min_size=min_size).fit(values)
        found = algo.predict(pen=penalty)
        return sorted(i for i in found if 0 < i < len(values))
    except Exception as exc:  # noqa: BLE001 -- a cross-check must never be fatal
        print(f"    (ruptures cross-check failed: {exc})")
        return None


def segment_means(values, splits):
    bounds = [0] + sorted(splits) + [len(values)]
    return [float(np.mean(values[lo:hi])) for lo, hi in zip(bounds[:-1], bounds[1:])]


def reverted_flags(values, splits, tolerance=0.34):
    """Which changepoints are undone by a later one, returning to the old level.

    A four-day outage, or any transient, does not produce one changepoint. It
    produces a PAIR: out and back. Both are real mean shifts and BIC is right
    to keep them, but reporting them as two regime changes invites a hunt for
    two release notes that do not exist. A shift whose level is restored later
    is marked rather than hidden, because the excursion itself is worth seeing
    -- an outage is a fact about the record even when it is not a deployment.

    tolerance is a fraction of the shift being undone: the level has to come
    back within a third of the way to count as restored.
    """
    order = sorted(splits)
    means = segment_means(values, order)
    flags = []
    for position in range(len(order)):
        before, after = means[position], means[position + 1]
        size = abs(after - before)
        if size <= 0:
            flags.append(False)
            continue
        # Forward: a later segment comes back to where this one started.
        # Backward: this shift IS the coming back. Both legs of an excursion
        # are transient, and marking only the outbound one leaves the return
        # leg looking like a lone regime change on its own date.
        restored = any(abs(later - before) <= tolerance * size
                       for later in means[position + 2:])
        returning = any(abs(earlier - after) <= tolerance * size
                        for earlier in means[:position])
        flags.append(bool(restored or returning))
    return flags


def shift_sizes(series, index, lo=None, hi=None):
    """Mean either side of a split, the difference, and it in SD units.

    Bounded by the NEIGHBOURING splits, not by the ends of the series. With one
    changepoint the two are the same; with two they are not, and comparing
    whole prefix against whole suffix mixes the earlier regime into the later
    shift's "before". On the offline fixture that reported a residual weather
    shift of 0.144 where the actual step between adjacent segments was 0.020 --
    a sevenfold overstatement, in the direction that makes a nothing look like
    a finding.

    The SD is the WITHIN-segment spread, not the spread across the step: the
    latter includes the step itself, so scaling by it shrinks every large shift
    toward the same unremarkable number.
    """
    values = np.asarray(series, dtype=float)
    lo = 0 if lo is None else lo
    hi = len(values) if hi is None else hi
    before, after = values[lo:index], values[index:hi]
    if not before.size or not after.size:
        return None
    pooled = np.concatenate([before - before.mean(), after - after.mean()])
    sd = float(np.std(pooled, ddof=1)) if pooled.size > 1 else 0.0
    delta = float(after.mean() - before.mean())
    return {"before": float(before.mean()), "after": float(after.mean()),
            "delta": delta, "sd": sd,
            "delta_sd": (delta / sd) if sd > 0 else float("nan")}


def neighbours(splits, index, length):
    """(previous split or 0, next split or length) around this one."""
    order = sorted(splits)
    position = order.index(index)
    lo = order[position - 1] if position else 0
    hi = order[position + 1] if position + 1 < len(order) else length
    return lo, hi


# ------------------------------------------------------------------- the data

def cameras_on_disk():
    """Every camera with a rip record, by the name the rest of the repo uses."""
    paths = sorted(glob.glob(f"{ad.DATA_DIR}/rip_detection/rip_*.csv"))
    sites = None
    try:
        sites = ad.load_sites()
    except SystemExit:
        sites = None
    by_slug = {}
    if sites is not None and "camera_name" in sites.columns:
        by_slug = {ad.rip_slug(n): n for n in sites["camera_name"].dropna()}

    found = []
    for path in paths:
        stem = os.path.basename(path)[len("rip_"):-len(".csv")]
        # rip_<slug>_index.csv and rip_<slug>_hourly.csv sit beside the frame
        # table and match this glob; neither is a camera.
        if stem.endswith("_index") or stem.endswith("_hourly"):
            continue
        found.append((by_slug.get(stem, stem), stem, path))
    return found


def class_names(args):
    """The class list the run is filtering on, or None for 'any'."""
    wanted = getattr(args, "detection_class", None)
    if not wanted or wanted == "any":
        return None
    return [c.strip() for c in str(wanted).split(",") if c.strip()]


def class_mix(frame):
    """Counts per score_classes value, for the header line."""
    if "score_classes" not in frame.columns:
        return {}
    from collections import Counter
    tally = Counter()
    for value in frame["score_classes"].fillna(""):
        name = str(value).strip()
        if name:
            tally[name] += 1
    return tally


def daily_metrics(path, min_frames, wanted=bls.RIP_CLASS):
    """One row per day: the five metrics, plus the counts they rest on.

    A day is kept only if it has at least min_frames scored frames. A median
    over two detections is not a median, and an unfiltered series puts its
    noisiest points at exactly the thin edges of the record where a deployment
    change is most likely to sit -- so the noise and the signal arrive in the
    same place and the method cannot tell them apart.
    """
    frame = ad.read_csv(path)
    if frame is None or frame.empty or "timestamp" not in frame.columns:
        return None, "no rip record"
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True,
                                        errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    if frame.empty:
        return None, "no parseable timestamps"
    frame["date"] = frame["timestamp"].dt.floor("D")
    frame["hour"] = ad.to_hour(frame["timestamp"])

    # Written before the feed was found to carry two models' output. Without
    # this filter, "detection rate" at Walton is 19% people and boats, and at
    # Corolla it is 100% of them -- so a changepoint would be a changepoint in
    # how often the beach was busy. bls.keep_class keeps rows with no class at
    # all, which are the observed zeros.
    if wanted and wanted != "any":
        names = ([wanted] if isinstance(wanted, str) and "," not in wanted
                 else [c.strip() for c in str(wanted).split(",") if c.strip()])
        frame, dropped = bls.keep_class(frame, names, label="class: ")
        if frame.empty or not frame.get("detected", pd.Series(dtype=bool)).any():
            return None, (f"no {' or '.join(names)} detections at all "
                          f"({dropped} frames of other classes) — "
                          "pass --detection-class any to analyse them")

    if "detected" in frame.columns:
        detected = frame[frame["detected"].astype(bool)]
    else:
        detected = frame[frame.get("score_max", pd.Series(dtype=float)).notna()]

    rows = []
    for date, group in frame.groupby("date"):
        hits = detected[detected["date"] == date]
        scores = pd.to_numeric(hits.get("score_max"), errors="coerce").dropna()
        nonzero = scores[scores > 0]
        areas = pd.to_numeric(hits.get("bbox_area_max"), errors="coerce").dropna()
        rows.append({
            "date": date,
            "n_frames": len(group),
            "n_detections": len(hits),
            "detected_hours": int(hits["hour"].nunique()),
            "score_median": float(scores.median()) if len(scores) else np.nan,
            "score_p90": float(scores.quantile(0.9)) if len(scores) else np.nan,
            "score_floor": float(nonzero.min()) if len(nonzero) else np.nan,
            "bbox_area_median": float(areas.median()) if len(areas) else np.nan,
        })
    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    thin = int((daily["n_frames"] < min_frames).sum())
    daily = daily[daily["n_frames"] >= min_frames].reset_index(drop=True)
    return daily, (f"{thin} days dropped under {min_frames} frames" if thin else "")


def attach_detection_rate(daily, slug):
    """detections per hour the camera was actually up, where coverage exists.

    Falling back to detections-per-frame when there is no coverage file would
    be quietly wrong in the one direction that matters here: the rip feed
    publishes only on a detection, so its own frame count is not a denominator
    at all, and the ratio would be pinned near 1 by construction.
    """
    path = f"{ad.DATA_DIR}/rip_detection/coverage_{slug}_hourly.csv"
    coverage = ad.read_csv(path)
    if coverage is None or coverage.empty or "hour" not in coverage.columns:
        daily["covered_hours"] = np.nan
        daily["detection_rate"] = np.nan
        return daily, False
    coverage = coverage.copy()
    coverage["hour"] = ad.to_hour(coverage["hour"])
    coverage = coverage.dropna(subset=["hour"]).drop_duplicates("hour")
    coverage["date"] = coverage["hour"].dt.floor("D")
    per_day = coverage.groupby("date").size().rename("covered_hours")
    daily = daily.merge(per_day, left_on="date", right_index=True, how="left")
    daily["detection_rate"] = np.where(
        daily["covered_hours"] > 0,
        daily["detected_hours"] / daily["covered_hours"], np.nan)
    return daily, True


def model_versions(path, wanted=None):
    """When each recorded model version is present, as a SPAN per version.

    Not as row-to-row transitions. Two models publish into one feed and their
    rows interleave in time, so a walk down the sorted rows calls every row a
    version change: Corolla Hampton Inn printed fifty-four lines alternating
    "rip_current_detector / 1 -> yolo / v8n -> rip_current_detector / 1" across
    five dates. That is the two models taking turns, not fifty-four
    deployments, and it buried the one real transition at 2024-07-28.

    A version's first and last appearance is what a deployment looks like, and
    it is immune to interleaving.
    """
    frame = ad.read_csv(path)
    if frame is None or frame.empty:
        return []
    columns = [c for c in ("model_name", "model_version") if c in frame.columns]
    if not columns:
        return []
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True,
                                        errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    if wanted and "score_classes" in frame.columns:
        keep = frame["score_classes"].fillna("").map(
            lambda v: not bls.class_set(v) or bls.class_set(v) <= set(wanted))
        if keep.any():
            frame = frame[keep]
    if frame.empty:
        return []
    frame["tag"] = frame[columns].astype(str).agg(" / ".join, axis=1)

    span_first = frame["timestamp"].min()
    span_last = frame["timestamp"].max()
    out = []
    for tag, group in frame.groupby("tag"):
        first, last = group["timestamp"].min(), group["timestamp"].max()
        out.append({"tag": tag, "n": len(group), "first": first, "last": last,
                    "starts_late": (first - span_first).days > 14,
                    "ends_early": (span_last - last).days > 14})
    return sorted(out, key=lambda v: v["first"])



def daily_controls(camera, lat, lon):
    """Daily MOP wave height, tide and cloud cover, from files already on disk.

    Deliberately NOT ad.load_coops: that resolves a station by asking CO-OPS
    for its station list, and this script promises to touch no network. Tide
    files are matched by name here and skipped when the match is not certain,
    because a tide series from the wrong ocean would residualize a real shift
    away.
    """
    grid = ad.grid_slug(camera)
    pieces, found = [], []

    mop = ad.read_csv(f"{ad.DATA_DIR}/mop_{grid}.csv")
    if mop is not None and not mop.empty:
        column = "time" if "time" in mop.columns else mop.columns[0]
        mop = mop.copy()
        mop["date"] = pd.to_datetime(mop[column], utc=True,
                                     errors="coerce").dt.floor("D")
        height = next((c for c in mop.columns if "height" in c.lower()), None)
        if height:
            pieces.append(mop.groupby("date")[[height]].mean()
                          .rename(columns={height: "wave_height"}))
            found.append("MOP wave height")

    cloud = ad.read_csv(f"{ad.DATA_DIR}/cloud_{grid}.csv")
    if cloud is not None and not cloud.empty and "cloud_cover" in cloud.columns:
        column = "time" if "time" in cloud.columns else cloud.columns[0]
        cloud = cloud.copy()
        cloud["date"] = pd.to_datetime(cloud[column], utc=True,
                                       errors="coerce").dt.floor("D")
        pieces.append(cloud.groupby("date")[["cloud_cover"]].mean())
        found.append("cloud cover")

    tide_paths = sorted(glob.glob(f"{ad.DATA_DIR}/tide_*.csv"))
    if len(tide_paths) == 1:
        tide = ad.read_csv(tide_paths[0])
        if tide is not None and "time" in tide.columns:
            tide = tide.copy()
            tide["date"] = pd.to_datetime(tide["time"], utc=True,
                                          errors="coerce").dt.floor("D")
            level = next((c for c in tide.columns
                          if c not in ("time", "date")
                          and pd.api.types.is_numeric_dtype(tide[c])), None)
            if level:
                daily = tide.groupby("date")[level].agg(["mean", "std"])
                daily.columns = ["tide_mean", "tide_range"]
                pieces.append(daily)
                found.append(f"tide ({os.path.basename(tide_paths[0])})")
    elif len(tide_paths) > 1:
        found.append(f"tide SKIPPED ({len(tide_paths)} files, none identifiable "
                     "offline)")

    if not pieces:
        return None, found
    out = pieces[0]
    for extra in pieces[1:]:
        out = out.join(extra, how="outer")
    return out, found


def control_slopes(target, design):
    """Control coefficients fitted on FIRST DIFFERENCES, not on levels.

    Fitting on levels is what the obvious version does, and on this data it is
    badly wrong: a real step that happens to correlate with the weather biases
    the weather coefficient upward, so subtracting it OVER-corrects and leaves
    a phantom shift -- sign flipped -- on the date the weather moved. Measured
    on the offline fixture, a true slope of 0.200 came back as 0.269.

    The obvious repair, level dummies at the shifts the raw pass already found,
    is circular and fragile: when the raw pass puts splits either side of a
    weather step (which it does, because the weather step IS a step), those
    dummies absorb the variance the coefficient needs.

    Differencing needs no changepoint input at all. A level shift contributes
    to exactly one difference out of several hundred, so its influence is
    negligible, while day-to-day co-variation -- which is what identifies the
    relationship -- is untouched. On the same fixture this returns 0.188, and
    the weather step collapses from 0.20 to 0.02 while the real step is
    preserved at full size.

    The cost, stated because it is real: differencing leans on day-to-day
    variation, so a control that only moves slowly (a seasonal cycle with no
    daily wiggle) is weakly identified and will be under-corrected. Wave
    height, tide and cloud cover all vary strongly day to day, which is why
    this is the right trade here.
    """
    d_target = np.diff(target)
    d_design = np.diff(design, axis=0)
    if d_target.size <= d_design.shape[1] + 2:
        return np.zeros(d_design.shape[1])

    # A control that never moves between consecutive kept days -- which happens
    # when the series has long gaps and the differences straddle them, or when
    # a column is constant -- leaves lstsq with a rank-deficient system. It
    # returns a solution, but on the real Walton run that solution contained
    # infinities and every subtraction below raised "divide by zero encountered
    # in matmul". The residuals printed anyway, which is the dangerous part.
    usable = np.isfinite(d_design).all(axis=1) & np.isfinite(d_target)
    if usable.sum() <= d_design.shape[1] + 2:
        return np.zeros(d_design.shape[1])
    moving = np.std(d_design[usable], axis=0) > 0
    slopes = np.zeros(d_design.shape[1])
    if not moving.any():
        return slopes
    fitted, *_ = np.linalg.lstsq(d_design[usable][:, moving],
                                 d_target[usable], rcond=None)
    slopes[moving] = fitted
    if not np.isfinite(slopes).all():
        return np.zeros(d_design.shape[1])
    return slopes


def residualize(daily, controls, metrics):
    """Each metric with the controls' linear contribution removed -- only that.

    Rows where any control is missing become NaN rather than keeping an
    unadjusted value. Mixing residualized and raw points in one series puts a
    step at every boundary of the controls' coverage, manufacturing exactly the
    shape being hunted for.

    Level shifts are deliberately LEFT IN: the weather goes, the steps stay,
    and the changepoint search runs again on what is left. See control_slopes
    for why the coefficients are fitted on differences.
    """
    merged = daily.merge(controls, left_on="date", right_index=True, how="left")
    names = list(controls.columns)
    design = merged[names].to_numpy(dtype=float)
    out = pd.DataFrame({"date": merged["date"]})
    usable = np.isfinite(design).all(axis=1)

    for metric in metrics:
        if metric not in merged.columns:
            continue
        target = pd.to_numeric(merged[metric], errors="coerce").to_numpy(dtype=float)
        rows = usable & np.isfinite(target)
        column = np.full(len(merged), np.nan)
        if rows.sum() > len(names) + 3:
            # Centred and scaled before fitting. The guard added earlier was on
            # the SLOPES, and they were finite; the overflow was in the product,
            # because an ill-conditioned fit on raw columns (bbox areas run to
            # 1e5, cloud cover to 1e2) produced slopes large enough that
            # design @ slopes overflowed. On standardized columns the slopes are
            # O(1) and the product cannot blow up. Centring also leaves the
            # residual's mean equal to the target's, so the before/after levels
            # printed for a residualized series stay on the metric's own scale.
            block = design[rows]
            centre = block.mean(axis=0)
            spread = block.std(axis=0)
            spread[spread == 0] = 1.0
            scaled = (block - centre) / spread
            slopes = control_slopes(target[rows], scaled)
            with np.errstate(all="ignore"):
                adjusted = target[rows] - scaled @ slopes
            # Never let a bad fit through as data: a non-finite residual would
            # be dropped silently by the changepoint search's isfinite mask,
            # shortening the series without saying so.
            column[rows] = np.where(np.isfinite(adjusted), adjusted, np.nan)
        out[metric] = column
    return out, names


def monthly_histograms(path, camera, wanted=None):
    """score_max distribution per month, as counts in fixed 0.05 bins.

    Fixed bins, never per-month quantiles: the whole point is that two months
    are comparable by eye, and a bin edge that moves with the data hides the
    move it was drawn to show.
    """
    frame = ad.read_csv(path)
    if frame is None or frame.empty or "score_max" not in frame.columns:
        return None
    frame = frame.copy()
    # Filtered to the analysed class. Unfiltered, the floor a month shows is
    # whichever model scored lowest that month, so Corolla read 0.501 in June
    # 2024 from the object model while the rip model sat at 0.70 throughout --
    # a calibration shift that never happened to the series being analysed.
    if wanted and "score_classes" in frame.columns:
        keep = frame["score_classes"].fillna("").map(
            lambda v: not bls.class_set(v) or bls.class_set(v) <= set(wanted))
        if keep.any():
            frame = frame[keep]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True,
                                        errors="coerce")
    frame["score_max"] = pd.to_numeric(frame["score_max"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "score_max"])
    if frame.empty:
        return None
    frame["month"] = frame["timestamp"].dt.strftime("%Y-%m")
    rows = []
    for month, group in frame.groupby("month"):
        counts, _ = np.histogram(group["score_max"], bins=HIST_BINS)
        row = {"camera": camera, "month": month, "n": len(group),
               "min_nonzero": float(group.loc[group["score_max"] > 0,
                                              "score_max"].min())
               if (group["score_max"] > 0).any() else np.nan,
               "median": float(group["score_max"].median())}
        for edge, count in zip(HIST_BINS[:-1], counts):
            row[f"b{edge:.2f}"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def print_histograms(hist, width=48):
    """The histograms as text, because a calibration shift is a shape change."""
    if hist is None or hist.empty:
        print("    no score_max on this camera; no histogram")
        return
    bins = [c for c in hist.columns if c.startswith("b")]
    peak = max(1, int(hist[bins].to_numpy().max()))
    print(f"    {'month':<9} {'n':>6} {'floor':>6} {'med':>5}  "
          f"score_max 0.0 -> 1.0")
    for _, row in hist.iterrows():
        bar = ""
        for name in bins:
            share = int(row[name]) / peak
            bar += " .:-=+*#@"[min(8, int(share * 8 + 0.5))]
        floor = row["min_nonzero"]
        print(f"    {row['month']:<9} {int(row['n']):>6} "
              f"{floor:>6.3f} {row['median']:>5.2f}  |{bar}|"
              if floor == floor else
              f"    {row['month']:<9} {int(row['n']):>6} "
              f"{'-':>6} {row['median']:>5.2f}  |{bar}|")


# ----------------------------------------------------------------- the report

def analyze_series(label, dates, values, args, findings, camera, kind):
    """Run the detector on one metric's series, print it, return the dates.

    The dates come back because the residualized pass needs them: they become
    the level dummies that keep a real step from biasing the weather
    coefficients it is being separated from.
    """
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(values)
    if keep.sum() < 2 * args.min_segment:
        print(f"    {label:<18} too short after dropping gaps "
              f"({int(keep.sum())} days)")
        return []
    dates = pd.DatetimeIndex(pd.Series(dates)[keep].to_numpy())
    values = values[keep]

    splits = detect(values, args.max_k, args.min_segment)
    penalty = 2 * math.log(len(values)) * float(np.var(values))
    other = ruptures_cross_check(values, args.max_k, args.min_segment, penalty)

    if not splits:
        note = ""
        if other:
            note = f"   (ruptures suggested {len(other)}; BIC kept none)"
        print(f"    {label:<18} no changepoint{note}")
        return []

    reverted = reverted_flags(values, splits)
    for index, undone in zip(splits, reverted):
        lo, hi = neighbours(splits, index, len(values))
        shift = shift_sizes(values, index, lo, hi)
        date = dates[index]
        gap = (date - dates[index - 1]).days if index else 0
        flags = ("  GAP" if gap > args.min_segment else "")
        flags += "  TRANSIENT" if undone else ""
        print(f"    {label:<18} {date:%Y-%m-%d}  "
              f"{shift['before']:>9.4f} -> {shift['after']:>9.4f}  "
              f"delta {shift['delta']:>+9.4f}  "
              f"{shift['delta_sd']:>+5.1f} sd{flags}")
        findings.append({"camera": camera, "kind": kind, "metric": label,
                         "date": date.strftime("%Y-%m-%d"),
                         "before": shift["before"], "after": shift["after"],
                         "delta": shift["delta"], "delta_sd": shift["delta_sd"],
                         "transient": bool(undone),
                         "days_since_previous_observation": int(gap),
                         "n_days": int(len(values))})
    if other is not None:
        agree = sum(1 for i in splits
                    if any(abs(i - j) <= args.coincide for j in other))
        print(f"      ruptures found {len(other)}; "
              f"{agree}/{len(splits)} of ours within {args.coincide} days")
    return [dates[i] for i in splits]


def dates_for(daily, metrics, args):
    """{metric: [date, ...]} for one daily table, printing nothing."""
    out = {}
    for metric in metrics:
        if metric not in daily.columns:
            continue
        values = pd.to_numeric(daily[metric], errors="coerce").to_numpy(float)
        keep = np.isfinite(values)
        if keep.sum() < 2 * args.min_segment:
            out[metric] = []
            continue
        kept = pd.DatetimeIndex(daily["date"][keep].to_numpy())
        splits = detect(values[keep], args.max_k, args.min_segment)
        flags = reverted_flags(values[keep], splits)
        out[metric] = [(kept[i], undone)
                       for i, undone in zip(splits, flags)]
    return out


def near_class_event(moment, events, window):
    """The object-class start/stop within `window` days of a date, if any.

    This is the whole point of running both passes. A class switching on
    mid-record steps the UNFILTERED series on that date, and a changepoint
    search will name it and invite a hunt for a release note that does not
    exist. Marking the coincidence is what tells the two apart.
    """
    best, best_gap = None, None
    for event in events:
        gap = abs((pd.Timestamp(event["date"]).tz_localize(None)
                   - pd.Timestamp(moment).tz_localize(None)).days)
        if gap <= window and (best_gap is None or gap < best_gap):
            best, best_gap = event, gap
    return best


def compare_classes(camera, path, slug, args, class_events):
    """Filtered and unfiltered changepoint dates, side by side."""
    print("\n  filtered (rip_current) vs unfiltered (every class)")
    passes = {}
    for label, wanted in (("filtered", args.detection_class), ("unfiltered", "any")):
        with _quiet():
            daily, _ = daily_metrics(path, args.min_frames, wanted)
            if daily is not None and not daily.empty:
                daily, _ = attach_detection_rate(daily, slug)
        if daily is None or daily.empty:
            passes[label] = {}
            continue
        metrics = [m for m in METRICS
                   if m in daily.columns and daily[m].notna().any()]
        passes[label] = dates_for(daily, metrics, args)

    every = sorted(set(passes["filtered"]) | set(passes["unfiltered"]))
    if not every:
        print("    neither pass produced a usable series")
        return []

    rows = []
    print(f"    {'metric':<18} {'filtered':<34} unfiltered")
    for metric in every:
        left = _stamp(passes["filtered"].get(metric, []))
        right = _stamp(passes["unfiltered"].get(metric, []))
        print(f"    {metric:<18} {left:<34} {right}")
        for moment, undone in passes["unfiltered"].get(metric, []):
            hit = near_class_event(moment, class_events, args.coincide)
            rows.append({"camera": camera, "metric": metric, "pass": "unfiltered",
                         "date": moment.strftime("%Y-%m-%d"),
                         "transient": bool(undone),
                         "class_event": ("" if hit is None else
                                         f"{hit['group']} {hit['kind']} "
                                         f"{pd.Timestamp(hit['date']):%Y-%m-%d}")})
        for moment, undone in passes["filtered"].get(metric, []):
            rows.append({"camera": camera, "metric": metric, "pass": "filtered",
                         "date": moment.strftime("%Y-%m-%d"),
                         "transient": bool(undone), "class_event": ""})

    marked = [r for r in rows if r["class_event"]]
    if marked:
        print(f"\n    {len(marked)} unfiltered date(s) coincide with an object "
              f"class switching\n    within {args.coincide} days — these are "
              "the class, not the detector:")
        for row in marked:
            print(f"      {row['date']}  {row['metric']:<18} {row['class_event']}")
    elif class_events:
        print(f"\n    no unfiltered date falls within {args.coincide} days of "
              "an object class switching")

    only_unfiltered = {r["date"] for r in rows if r["pass"] == "unfiltered"} \
        - {r["date"] for r in rows if r["pass"] == "filtered"}
    if only_unfiltered:
        print(f"    {len(only_unfiltered)} date(s) appear ONLY unfiltered: "
              + ", ".join(sorted(only_unfiltered)))
    return rows


class _quiet:
    """Silence a helper that prints, without silencing the report around it."""

    def __enter__(self):
        self._saved = sys.stdout
        sys.stdout = io.StringIO()

    def __exit__(self, *exc):
        sys.stdout = self._saved
        return False


def _stamp(pairs):
    if not pairs:
        return "-"
    return ", ".join(f"{d:%Y-%m-%d}" + ("*" if undone else "")
                     for d, undone in pairs)


def coincidences(findings, window):
    """Dates where two or more CAMERAS shift within `window` days.

    Two metrics on one camera moving together is one event seen twice, not two
    events, so the clustering is over distinct cameras -- counting rows would
    make every single-camera shift look platform-wide as soon as it touched
    two metrics.
    """
    if not findings:
        return []
    frame = pd.DataFrame(findings)
    if "transient" in frame.columns:
        # An excursion that reverses itself is not a deployment. Leaving these
        # in would let two cameras having a bad week together read as a release.
        frame = frame[~frame["transient"].fillna(False).astype(bool)]
        if frame.empty:
            return []
    frame["day"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("day")
    clusters, current = [], []
    for _, row in frame.iterrows():
        if current and (row["day"] - current[0]["day"]).days > window:
            clusters.append(current)
            current = []
        current.append(row)
    if current:
        clusters.append(current)

    out = []
    for cluster in clusters:
        cameras = sorted({row["camera"] for row in cluster})
        if len(cameras) < 2:
            continue
        days = [row["day"] for row in cluster]
        out.append({"first": min(days).strftime("%Y-%m-%d"),
                    "last": max(days).strftime("%Y-%m-%d"),
                    "cameras": cameras,
                    "metrics": sorted({row["metric"] for row in cluster})})
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="substring filters; default is every camera on disk")
    parser.add_argument("--start", default=None,
                        help="global window start (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="global window end")
    parser.add_argument("--window", action="append", default=[],
                        metavar="CAMERA=START:END",
                        help="per-camera window, repeatable; overrides --start/--end")
    parser.add_argument("--spans", action="store_true",
                        help="print each camera's rip record span and stop")
    parser.add_argument("--residualize", nargs="*", default=None,
                        help="cameras to also run on weather-residualized series; "
                             "default Walton, '' for none")
    parser.add_argument("--detection-class", "--class-filter",
                        dest="detection_class",
                        default=",".join(bls.RIP_CLASSES),
                        help="analyse only this class; 'any' pools every class, "
                             "which is what the earlier version did")
    parser.add_argument("--compare-classes", action="store_true",
                        help="run filtered and unfiltered side by side and mark "
                             "dates that coincide with an object class starting "
                             "or stopping")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES_PER_DAY)
    parser.add_argument("--min-segment", type=int, default=MIN_SEGMENT_DAYS,
                        help="shortest run of days that can be called a regime")
    parser.add_argument("--max-k", type=int, default=MAX_CHANGEPOINTS)
    parser.add_argument("--min-driver-days", type=int, default=MIN_DRIVER_DAYS,
                        help="below this a camera is too thin to fit drivers on")
    parser.add_argument("--coincide", type=int, default=COINCIDE_DAYS)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    found = cameras_on_disk()
    if not found:
        sys.exit(f"No {ad.DATA_DIR}/rip_detection/rip_*.csv — run "
                 "pull_rip_detection.py first.")
    if args.cameras:
        wanted = [c.lower() for c in args.cameras]
        found = [f for f in found
                 if any(w in f[0].lower() or w in f[1].lower() for w in wanted)]
        if not found:
            sys.exit(f"No camera on disk matches {args.cameras}")

    windows = {}
    for entry in args.window:
        if "=" not in entry or ":" not in entry.split("=", 1)[1]:
            sys.exit(f"--window wants CAMERA=START:END, got {entry!r}")
        name, span = entry.split("=", 1)
        lo, hi = span.split(":", 1)
        windows[name.lower()] = (lo, hi)

    if args.spans:
        print("\ncamera rip records on disk\n")
        print(f"  {'camera':<44} {'first':<12} {'last':<12} {'days':>6} {'rows':>8}")
        for camera, slug, path in found:
            frame = ad.read_csv(path)
            if frame is None or "timestamp" not in frame.columns:
                print(f"  {camera[:43]:<44} {'-':<12} {'-':<12}")
                continue
            stamps = pd.to_datetime(frame["timestamp"], utc=True,
                                    errors="coerce").dropna()
            print(f"  {camera[:43]:<44} {stamps.min():%Y-%m-%d}   "
                  f"{stamps.max():%Y-%m-%d}   "
                  f"{stamps.dt.floor('D').nunique():>6} {len(frame):>8}")
        print("\n  Paste these into --window CAMERA=START:END to pin the "
              "windows explicitly;\n  with no --window the script uses each "
              "camera's own span, which is these dates.\n")
        return 0

    residual_targets = (RESIDUALIZE_DEFAULT if args.residualize is None
                        else [c for c in args.residualize if c])

    os.makedirs(args.out_dir, exist_ok=True)
    findings, histograms, daily_tables = [], [], []
    treatment, comparisons = [], []

    for camera, slug, path in found:
        print(f"\n{'=' * 72}\n{camera}\n{'=' * 72}")

        whole = ad.read_csv(path)
        mix = class_mix(whole) if whole is not None else {}
        if mix:
            top = ", ".join(f"{name} {count}" for name, count
                            in sorted(mix.items(), key=lambda kv: -kv[1])[:6])
            rip = sum(count for name, count in mix.items()
                      if set(bls.class_set(name)) <= set(bls.RIP_CLASSES))
            print(f"  classes: {top}")
            print(f"  rip classes: {rip} of {sum(mix.values())} "
                  f"({rip / max(sum(mix.values()), 1):.0%})")

        class_events = []
        if args.compare_classes:
            timeline, why = ct.load(path)
            if timeline is not None:
                class_events = ct.transitions(
                    ct.monthly_table(timeline), ct.class_events(timeline),
                    timeline["timestamp"].min(), timeline["timestamp"].max())

        daily, note = daily_metrics(path, args.min_frames, args.detection_class)
        if daily is None or daily.empty:
            print(f"  skipped: {note or 'no usable days'}")
            treatment.append({"camera": camera, "treatment": "skipped",
                              "reason": note or "no usable days"})
            continue
        daily, have_coverage = attach_detection_rate(daily, slug)

        window = windows.get(camera.lower()) or windows.get(slug.lower())
        lo, hi = (window if window else (args.start, args.end))
        if lo:
            daily = daily[daily["date"] >= pd.Timestamp(lo, tz="UTC")]
        if hi:
            daily = daily[daily["date"] <= pd.Timestamp(hi, tz="UTC")]
        daily = daily.reset_index(drop=True)
        if daily.empty:
            print("  skipped: the requested window holds no days")
            continue

        print(f"  {len(daily)} days, {daily['date'].min():%Y-%m-%d} to "
              f"{daily['date'].max():%Y-%m-%d}"
              + (f"   ({note})" if note else ""))
        if not have_coverage:
            print("  no coverage file: detection_rate is UNAVAILABLE, not zero "
                  "(run pull_rip_detection.py --coverage)")

        versions = model_versions(path, wanted=class_names(args))
        if not versions:
            print("\n  payload records no model name or version")
        elif len(versions) == 1:
            print(f"\n  one model version throughout: {versions[0]['tag']}")
        else:
            print(f"\n  {len(versions)} model versions in the payload:")
            for entry in versions:
                mark = ""
                if entry["starts_late"]:
                    mark += "  STARTS mid-record"
                if entry["ends_early"]:
                    mark += "  STOPS mid-record"
                print(f"    {entry['first']:%Y-%m-%d} to {entry['last']:%Y-%m-%d}"
                      f"  {entry['n']:>7} frames  {entry['tag']}{mark}")

        print("\n  raw series:")
        metrics = [m for m in METRICS if m in daily.columns
                   and daily[m].notna().any()]
        raw_breaks = {}
        for metric in metrics:
            raw_breaks[metric] = analyze_series(
                metric, daily["date"], daily[metric], args, findings, camera,
                "raw")

        thin = len(daily) < args.min_driver_days
        if camera in residual_targets and not thin:
            controls, available = daily_controls(camera, None, None)
            if controls is None:
                print("\n  residualized series: NO control files on disk; skipped")
                treatment.append({"camera": camera, "treatment": "raw only",
                                  "reason": "no control files"})
            else:
                residuals, names = residualize(daily, controls, metrics)
                print(f"\n  residualized on {', '.join(available)} "
                      f"({len(names)} columns):")
                for metric in metrics:
                    if metric in residuals.columns:
                        analyze_series(metric, residuals["date"],
                                       residuals[metric], args, findings,
                                       camera, "residualized")
                treatment.append({"camera": camera,
                                  "treatment": "raw + residualized",
                                  "reason": ", ".join(available)})
        else:
            why = (f"only {len(daily)} usable days, under --min-driver-days "
                   f"{args.min_driver_days}" if thin
                   else "not in --residualize")
            print(f"\n  DRIVERS NOT FITTED on this camera: {why}.")
            print("  Dates below come from the raw series alone, which is what "
                  "this script\n  is for; a weather-driven swing here is NOT "
                  "ruled out.")
            treatment.append({"camera": camera, "treatment": "raw only",
                              "reason": why})

        if args.compare_classes:
            comparisons += compare_classes(camera, path, slug, args,
                                           class_events)

        print("\n  monthly score_max histograms:")
        hist = monthly_histograms(path, camera, wanted=class_names(args))
        print_histograms(hist)
        if hist is not None:
            histograms.append(hist)

        daily = daily.copy()
        daily.insert(0, "camera", camera)
        daily_tables.append(daily)

    print(f"\n{'=' * 72}\ncross-camera coincidences\n{'=' * 72}")
    clusters = coincidences(findings, args.coincide)
    if not clusters:
        print(f"  no date where two or more cameras shift within "
              f"{args.coincide} days.")
        print("  Every shift found is site-specific on this evidence.")
    else:
        print(f"  {len(clusters)} window(s) where two or more cameras shift "
              f"within {args.coincide} days.")
        print("  That is the signature of a platform-wide deployment rather "
              "than a site change:\n")
        for cluster in clusters:
            print(f"    {cluster['first']} .. {cluster['last']}   "
                  f"{len(cluster['cameras'])} cameras")
            for name in cluster["cameras"]:
                print(f"      {name}")
            print(f"      metrics: {', '.join(cluster['metrics'])}\n")

    if findings:
        table = pd.DataFrame(findings)
        table.to_csv(f"{args.out_dir}/changepoints.csv", index=False)
        print(f"  wrote {args.out_dir}/changepoints.csv ({len(table)} rows)")
    if histograms:
        pd.concat(histograms, ignore_index=True).to_csv(
            f"{args.out_dir}/monthly_score_histograms.csv", index=False)
        print(f"  wrote {args.out_dir}/monthly_score_histograms.csv")
    if daily_tables:
        pd.concat(daily_tables, ignore_index=True).to_csv(
            f"{args.out_dir}/daily_metrics.csv", index=False)
        print(f"  wrote {args.out_dir}/daily_metrics.csv")
    if treatment:
        pd.DataFrame(treatment).to_csv(f"{args.out_dir}/treatment.csv", index=False)
        print(f"  wrote {args.out_dir}/treatment.csv  "
              "(which cameras were residualized and which were not)")
    if comparisons:
        pd.DataFrame(comparisons).to_csv(
            f"{args.out_dir}/class_comparison.csv", index=False)
        print(f"  wrote {args.out_dir}/class_comparison.csv  "
              "(filtered vs unfiltered dates, with class events marked)")
    if clusters:
        with open(f"{args.out_dir}/coincidences.json", "w") as handle:
            json.dump(clusters, handle, indent=2)
        print(f"  wrote {args.out_dir}/coincidences.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
