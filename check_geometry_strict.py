#!/usr/bin/env python3
"""Is a camera's stills record ONE geometric record, or several?

The same question check_camera_geometry.py asks, asked with a stricter method.
Four things went wrong at Walton Lighthouse and each one is designed out here
rather than warned about:

  1. THE WATER WAS IN THE MEASUREMENT.  Six of twelve candidate patches
     survived the agreement test there and all six sat on buildings; the six
     that failed sat on surf and open water. The frame is majority moving
     ocean, so whole-frame phase correlation was partly being driven by waves.
     Here a land mask is applied BEFORE anything is registered, the mask is
     declared by hand per camera, and registration refuses to run without one.

  2. TRANSLATION WAS THE ONLY THING THAT COULD BE SEEN.  Phase correlation
     measures a shift and nothing else, so a camera that rotated a degree or
     crept 2% in zoom would have registered as "stable" with slightly worse
     confidence. Here the primary route fits a full homography by
     ORB/SIFT + RANSAC, which represents rotation, scale and perspective, and
     RANSAC REJECTS the water outliers instead of averaging them in. Phase
     correlation is kept as an independent second route, because the gap
     between two routes is the only thing in this file that measures
     correctness rather than confidence.

  3. THE FRAME SIZE CHANGED AND NOBODY LOOKED.  Five frames at 1280x720 sat in
     Walton's 2560x1920 record and weekly sampling never saw them; it took the
     full daily pull. A size change is a hard epoch boundary that costs nothing
     to detect, so the inventory runs FIRST, before any registration, and
     registration is confined to one size group.

  4. A PASS RATE WAS READ AS A RESULT.  Walton's per-quarter table read 88-100%
     registered in quarters that demonstrably did not register. Peak-to-sidelobe
     ratio says how DISTINCTIVE a correlation peak is, never whether it is in
     the right place. The pass rate is reported here and is never the answer.

Nothing is corrected and nothing is re-registered. This measures.

    python check_geometry_strict.py --camera <name> --inventory
    python check_geometry_strict.py --camera <name> --mask-preview --cached
    python check_geometry_strict.py --camera <name> --cached --survey
    python check_geometry_strict.py --camera <name> --cached
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

import check_camera_geometry as geo


OUT_DIR = geo.OUT_DIR

# Sampling. Daily, because weekly missed a real boundary at Walton. 17:00 UTC
# is roughly midday on the US east coast, which is the steadiest light the
# record offers for a North Carolina camera.
DEFAULT_EVERY_DAYS = 1
DEFAULT_HOUR_UTC = 17

CELL = 128              # survey cell, px
MIN_CLUSTER = 3         # cells that must agree before a common vector exists
AGREE_PX = 3.0          # how close "agree" is, px
THIN_BAND_PX = 128      # below this a band caps how far it can measure

# Feature route.
MIN_MATCHES = 20        # raw correspondences before a fit is even attempted
MIN_INLIERS = 15        # RANSAC inliers before the fit is believed
MIN_INLIER_SHARE = 0.25
RANSAC_PX = 3.0
MAX_FEATURES = 4000

# Phase route. Reused wholesale from the Walton module.
MIN_CONFIDENCE = geo.MIN_CONFIDENCE
MAX_SHIFT_PX = 150

PERSIST = geo.PERSIST


# ---------------------------------------------------------------------------
# Land masks
#
# DECLARED BY HAND, ONE PER CAMERA, AND LOGGED WITH EVERY RUN.
#
# Polygons are in FRACTIONAL frame coordinates (0-1, x then y) so one
# declaration survives a resolution change and so the numbers can be read off
# any preview regardless of what it was rendered at. Each entry must say who
# drew it and from which frame, because a mask is a judgement about the scene
# and an undocumented judgement is indistinguishable from a guess.
#
# `keep` polygons are land. `drop` polygons are cut back out of them, for a
# sign, a timestamp banner or a pier that sits inside the land region.
# ---------------------------------------------------------------------------

MASKS = {
    "beachfront-from-sailfish-street-beach-access-corolla-nc": {
        # The shoreline in the reference frame is a straight line. Fitting the
        # sand/water colour boundary across 178 columns (84 kept; the rest
        # rejected as canopies, tents and shadow) gives y = 0.615 - 0.195x
        # with a scatter of 0.003 of the frame height about it.
        #
        # THE MARGIN IS 0.04, NOT THE 0.10 IT WAS. The first draft pushed the
        # edge a tenth of the frame landward of that line "for tide", which
        # threw away about 150 rows of plainly dry sand -- the upper beach
        # where the umbrellas sit, which is land in every frame and is the
        # part with the most texture to register on. A margin is for what one
        # frame cannot show, and 0.10 was not a margin, it was a guess with no
        # measurement behind it. 0.04 is a working value pending the audit:
        # --mask-audit measures where water actually reaches across the whole
        # record and prints the empirical edge, and that is what this should
        # be set from.
        "keep": [[(0.0, 0.655), (1.0, 0.460), (1.0, 1.0), (0.0, 1.0)]],
        # THE "Sailfish" WATERMARK IS BURNED INTO THE SENSOR, NOT THE SCENE.
        # Measured at x 0.033-0.104, y 0.927-0.956. It does not move when the
        # camera moves, so leaving it in hands both routes a bright, sharp,
        # perfectly stationary feature -- and on a beach, where the sand is
        # texture-poor and this is the highest-contrast thing in the land
        # region, SIFT will weight it heavily. It votes for "no motion" in
        # exactly the frames where the answer matters. Dropped with a little
        # padding, but only a little: the first draft blocked out the whole
        # bottom-left corner, which was more land given up for nothing.
        "drop": [[(0.02, 0.915), (0.12, 0.915), (0.12, 0.97), (0.02, 0.97)]],
        "note": "drawn by Claude from the fractional grid preview of "
                "currituck_sailfish-2024-06-02-165953Z.jpg; shoreline fitted "
                "(y = 0.615 - 0.195x) rather than eyeballed, plus a 0.04 "
                "margin that is PROVISIONAL until --mask-audit measures the "
                "real water excursion across the record",
    },
    # "beachfront-from-hampton-inn-corolla-nc": {
    #     "keep": [[(0.0, 0.62), (1.0, 0.55), (1.0, 1.0), (0.0, 1.0)]],
    #     "drop": [],
    #     "note": "drawn by hand from <frame>; dune line and buildings only",
    # },
}


def mask_spec(slug, override=None):
    """The declared land mask for a camera, or an explanation of its absence.

    Registration is refused without one. That is the point: an unmasked run
    over a frame that is most ocean produces a confident number driven by
    waves, and the number looks exactly like a measurement of the camera.
    """
    if override:
        return override
    for key, spec in MASKS.items():
        if key == slug or key in slug or slug in key:
            return spec
    return None


def polygon_mask(shape, polygon):
    """Boolean array, True inside a fractional-coordinate polygon.

    Even-odd ray casting, written out rather than imported, so the mask does
    not depend on whether OpenCV or matplotlib happens to be installed.
    """
    height, width = shape
    points = [(x * width, y * height) for x, y in polygon]
    ys, xs = np.mgrid[0:height, 0:width]
    xs = xs + 0.5
    ys = ys + 0.5
    inside = np.zeros(shape, dtype=bool)
    count = len(points)
    for index in range(count):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % count]
        if y0 == y1:
            continue
        straddles = ((ys >= min(y0, y1)) & (ys < max(y0, y1)))
        crossing = x0 + (ys - y0) * (x1 - x0) / (y1 - y0)
        inside ^= straddles & (xs < crossing)
    return inside


def build_mask(shape, spec):
    """Land mask for one frame shape from a declared spec."""
    keep = np.zeros(shape, dtype=bool)
    for polygon in spec.get("keep") or []:
        keep |= polygon_mask(shape, polygon)
    for polygon in spec.get("drop") or []:
        keep &= ~polygon_mask(shape, polygon)
    return keep


def describe_mask(mask, spec, shape):
    share = float(mask.mean())
    print(f"\nLAND MASK  ({spec.get('note', 'no provenance recorded')})")
    print(f"  {share * 100:.1f}% of the {shape[1]}x{shape[0]} frame is land "
          f"({int(mask.sum()):,} px)")
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) and len(cols):
        print(f"  spans rows {rows[0]}-{rows[-1]}, cols {cols[0]}-{cols[-1]}")
    if share < 0.05:
        print("  WARNING: under 5% of the frame. Too little to register on.")
    if share > 0.85:
        print("  WARNING: over 85% of the frame. This is probably not a land "
              "mask;\n  check that the water was actually excluded.")

    # A THIN BAND CAPS HOW FAR IT CAN MEASURE ACROSS ITSELF, which is a
    # smaller problem than the one this warning used to claim and is worth
    # stating precisely, because the earlier claim would have had us discard
    # true vertical agreement as untrustworthy.
    #
    # Shifting a band by dy across its short axis throws away dy/height of the
    # overlap, so confidence falls as the displacement grows. Swept on
    # synthetic texture, a 72-row band against a 300-row one, dx held at +5:
    #
    #     true dy      72-row             300-row
    #        2 px   +1.99 conf 297     +2.00 conf 959
    #       10 px   +9.98 conf 152    +10.00 conf 819
    #       20 px  +19.95 conf  64    +20.00 conf 599
    #       30 px  +29.90 conf  21    +30.00 conf 459
    #       34 px  +33.86 conf  11    +34.00 conf 415
    #       40 px  -33.79 conf   6    +40.00 conf 364
    #
    # The thin band reads every displacement it can see CORRECTLY, to a
    # hundredth of a pixel, right up to about half its height -- the limit of
    # the search, since `reach` is clamped there. Past that the answer is
    # wrong, and confidence has already collapsed through MIN_CONFIDENCE (8)
    # by the time it is: the wrong answer at 40 px carries confidence 6 and is
    # rejected. So the failure announces itself rather than posing as zero.
    #
    # What a thin band therefore costs is RANGE, not truth: it cannot see a
    # jump bigger than half its short axis, and near that ceiling it runs out
    # of confidence and drops out of the record. A camera that steps 60 px
    # down would leave a gap in route 2 rather than a row of stable-looking
    # zeros. Read a thin axis as measuring small motion well and large motion
    # not at all.
    box = mask_box(mask.astype(float))
    if box:
        top, bottom, left, right = box
        for name, extent in (("rows", bottom - top), ("columns", right - left)):
            if extent < THIN_BAND_PX:
                print(f"  NOTE: the mask spans only {extent} {name}, so route "
                      f"2 can measure\n  displacement across that axis only "
                      f"out to about {extent // 2} px. Inside that range it "
                      "is\n  accurate; beyond it confidence collapses and "
                      "the frame drops out rather\n  than reading as stable, "
                      "so a gap in route 2 along this axis may mean a\n  step "
                      "too big to see, not a camera that held still. The "
                      "homography route\n  has no such ceiling.")
    return share


def feathered(mask, radius=16):
    """Mask with a soft edge, for the phase route.

    A hard-edged mask is itself a huge step function, and phase correlation
    sees the edge of the mask as the strongest feature in the frame -- every
    frame then "registers" perfectly to the mask rather than to the scene.
    Smoothing the edge over a few pixels removes that.

    Box-blurred twice via a summed-area table, which approximates a Gaussian
    closely enough and needs nothing but numpy.
    """
    soft = mask.astype(np.float64)
    for _ in range(2):
        padded = np.pad(soft, radius, mode="edge")
        summed = padded.cumsum(axis=0).cumsum(axis=1)
        summed = np.pad(summed, ((1, 0), (1, 0)), mode="constant")
        size = 2 * radius + 1
        rows, cols = soft.shape
        block = (summed[size:size + rows, size:size + cols]
                 - summed[0:rows, size:size + cols]
                 - summed[size:size + rows, 0:cols]
                 + summed[0:rows, 0:cols])
        soft = block / (size * size)
    return soft


# ---------------------------------------------------------------------------
# A. Frame geometry inventory
# ---------------------------------------------------------------------------

def availability(service):
    """Per-day frame counts straight from the service inventory.

    Cheap, needs no imagery, and says where the record is thin before an
    absent epoch gets mistaken for a stable one. Corolla's feed is known to
    swing between roughly 122 and 718 frames a day.
    """
    import pull_rip_detection as rip
    frame = rip.fetch_inventory(service)
    if frame is None or "Count" not in frame.columns:
        print("  inventory gave no per-bin counts")
        return None
    starts = pd.to_datetime(frame.get("Bin Start"), errors="coerce", utc=True)
    counts = pd.to_numeric(frame["Count"], errors="coerce").fillna(0)
    series = pd.Series(counts.values, index=starts).dropna().sort_index()
    if series.empty:
        return None

    live = series[series > 0]
    print(f"\nPER-DAY FRAME AVAILABILITY  ({len(series)} bins)")
    print(f"  populated bins : {len(live)} of {len(series)}")
    if len(live):
        print(f"  frames per populated day: min {int(live.min())}, "
              f"median {int(live.median())}, max {int(live.max())}")
    empty = series[series == 0]
    if len(empty):
        runs, run = [], [empty.index[0]]
        for when in empty.index[1:]:
            if (when - run[-1]).days <= 1:
                run.append(when)
            else:
                runs.append(run)
                run = [when]
        runs.append(run)
        runs.sort(key=len, reverse=True)
        print(f"  {len(empty)} empty days in {len(runs)} gaps; longest:")
        for run in runs[:6]:
            print(f"    {run[0]:%Y-%m-%d} to {run[-1]:%Y-%m-%d}  "
                  f"{len(run)} days")
    return series


def sampled_gaps(dates, every_days):
    """Runs of intended sample days that produced no frame."""
    if len(dates) < 2:
        return []
    span = pd.date_range(min(dates).normalize(), max(dates).normalize(),
                         freq=f"{every_days}D")
    have = {d.normalize() for d in dates}
    missing = [d for d in span if d not in have]
    if not missing:
        return []
    runs, run = [], [missing[0]]
    for when in missing[1:]:
        if (when - run[-1]).days <= every_days:
            run.append(when)
        else:
            runs.append(run)
            run = [when]
    runs.append(run)
    return runs


# ---------------------------------------------------------------------------
# Route 1: features + RANSAC homography
# ---------------------------------------------------------------------------

def require_cv2():
    try:
        import cv2
    except ImportError:
        sys.exit("the homography route needs OpenCV:\n"
                 "    pip install opencv-python-headless\n"
                 "Run with --no-homography to use the phase route alone, but "
                 "note that\nleaves nothing to cross-check it against.")
    return cv2


def make_detector(kind, cv2):
    if kind == "sift":
        if not hasattr(cv2, "SIFT_create"):
            sys.exit("this OpenCV has no SIFT; use --detector orb")
        return cv2.SIFT_create(nfeatures=MAX_FEATURES), "l2"
    return cv2.ORB_create(nfeatures=MAX_FEATURES), "hamming"


def frame_features(image, mask, detector, cv2):
    """Keypoints and descriptors, computed only where the mask allows.

    OpenCV takes the mask at detection time, so water keypoints are never
    generated rather than generated and then discarded.
    """
    grey = np.clip(image, 0, 255).astype(np.uint8)
    where = (mask * 255).astype(np.uint8)
    keypoints, descriptors = detector.detectAndCompute(grey, where)
    return keypoints, descriptors


def fit_homography(src_kp, src_desc, dst_kp, dst_desc, norm, cv2):
    """Homography mapping REFERENCE pixel coords onto IMAGE pixel coords.

    Direction matters and is easy to invert. A scene feature sitting at x in
    the reference and at x+5 in the image must come out as +5, matching the
    sign convention phase_shift already uses, so the two routes can be
    subtracted from one another without a silent flip.

    Returns a dict, or None with a reason, so a failed frame is a recorded
    fact rather than a hole.
    """
    if src_desc is None or dst_desc is None:
        return None, "no descriptors"
    if len(src_desc) < MIN_MATCHES or len(dst_desc) < MIN_MATCHES:
        return None, "too few features"

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING if norm == "hamming"
                            else cv2.NORM_L2)
    try:
        pairs = matcher.knnMatch(src_desc, dst_desc, k=2)
    except Exception:
        return None, "matcher failed"
    # Lowe's ratio test: a correspondence is only trusted when the best match
    # is clearly better than the second best. On repetitive texture -- dune
    # grass, shingles, wave crests -- the two are equally good and the match is
    # meaningless, which is exactly what this throws away.
    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < MIN_MATCHES:
        return None, f"only {len(good)} ratio-test matches"

    src = np.float32([src_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([dst_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX,
                                         maxIters=5000, confidence=0.999)
    if matrix is None or inliers is None:
        return None, "no consensus"
    inliers = inliers.ravel().astype(bool)
    kept = int(inliers.sum())
    share = kept / len(good)
    if kept < MIN_INLIERS or share < MIN_INLIER_SHARE:
        return None, f"{kept}/{len(good)} inliers"

    projected = cv2.perspectiveTransform(src[inliers], matrix).reshape(-1, 2)
    residual = float(np.median(np.hypot(*(projected - dst[inliers].reshape(-1, 2)).T)))
    return {"H": matrix, "matches": len(good), "inliers": kept,
            "share": share, "residual": residual}, None


def decompose(matrix, shape):
    """Translation, rotation, scale and bend of a homography at frame centre.

    A homography is not one number, and reporting only the centre translation
    would reproduce exactly the blindness this route exists to remove. So:

      dx, dy      where the centre pixel went
      rotation    the closest rotation in the local Jacobian, degrees
      scale       sqrt of its determinant -- a zoom shows up here and NOWHERE
                  in the frame metadata, which is why a resolution check alone
                  cannot clear a camera
      aniso       ratio of the Jacobian's singular values; 1.0 is a similarity
      bend        how far the corners depart from the affine approximation
                  taken at the centre, in px -- the genuinely projective part
    """
    height, width = shape
    cx, cy = width / 2.0, height / 2.0
    denominator = matrix[2, 0] * cx + matrix[2, 1] * cy + matrix[2, 2]
    if abs(denominator) < 1e-12:
        return None
    ux = (matrix[0, 0] * cx + matrix[0, 1] * cy + matrix[0, 2]) / denominator
    uy = (matrix[1, 0] * cx + matrix[1, 1] * cy + matrix[1, 2]) / denominator

    jac = np.array([
        [(matrix[0, 0] - ux * matrix[2, 0]) / denominator,
         (matrix[0, 1] - ux * matrix[2, 1]) / denominator],
        [(matrix[1, 0] - uy * matrix[2, 0]) / denominator,
         (matrix[1, 1] - uy * matrix[2, 1]) / denominator],
    ])
    rotation = math.degrees(math.atan2(jac[1, 0] - jac[0, 1],
                                       jac[0, 0] + jac[1, 1]))
    determinant = float(np.linalg.det(jac))
    scale = math.sqrt(abs(determinant)) if determinant else 0.0
    singular = np.linalg.svd(jac, compute_uv=False)
    aniso = float(singular[0] / singular[1]) if singular[1] > 1e-12 else float("inf")

    corners = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    bend = 0.0
    for corner in corners:
        weight = matrix[2, 0] * corner[0] + matrix[2, 1] * corner[1] + matrix[2, 2]
        if abs(weight) < 1e-12:
            return None
        exact = np.array([
            (matrix[0, 0] * corner[0] + matrix[0, 1] * corner[1] + matrix[0, 2]) / weight,
            (matrix[1, 0] * corner[0] + matrix[1, 1] * corner[1] + matrix[1, 2]) / weight,
        ])
        affine = np.array([ux, uy]) + jac @ (corner - np.array([cx, cy]))
        bend = max(bend, float(np.hypot(*(exact - affine))))

    return {"dx": ux - cx, "dy": uy - cy, "rotation": rotation,
            "scale": scale, "aniso": aniso, "bend": bend}


# ---------------------------------------------------------------------------
# Route 2: phase correlation on the masked land
# ---------------------------------------------------------------------------

def mask_box(soft, floor=0.5):
    """Bounding box of the region the mask actually keeps.

    The default floor is 0.5, which is the INSIDE of the mask rather than the
    outer end of its feathered ramp. Cropping to the ramp instead costs the
    short axis: the ramp attenuates the first rows of real land, the Hann
    window applied inside phase correlation attenuates the rest, and between
    them a 144-row band has too few rows left at full weight to carry a
    vertical displacement. Cropping at 0.5 leaves the ramp outside the crop,
    where it belongs, and a known 2 px rise comes back as 2.00 instead of 0.2.
    """
    rows = np.where(soft.max(axis=1) > floor)[0]
    cols = np.where(soft.max(axis=0) > floor)[0]
    if not len(rows) or not len(cols):
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def masked_phase(reference, image, soft, max_shift=MAX_SHIFT_PX):
    """phase_shift, with everything outside the land mask removed first.

    Two things have to happen and neither is optional.

    THE MEAN IS SUBTRACTED OVER THE MASKED REGION, not the whole frame.
    Subtracting the frame mean would leave the land sitting on a pedestal
    whose height depends on how bright the sea was that day and whose SHAPE is
    the mask, so the correlation would go back to finding the mask.

    THE FRAME IS CROPPED TO THE MASK BEFORE CORRELATING. phase_shift applies a
    Hann window across whatever array it is handed, and a Hann window is zero
    at the edges. At a beach camera the land is a band along the BOTTOM of the
    frame -- exactly where that window has driven the signal to nothing. Left
    uncropped, this recovered horizontal shifts correctly and vertical shifts
    not at all: a 2 px rise read as 0.1 px, which is indistinguishable from a
    stable camera. Cropping to the mask's bounding box puts the land in the
    middle of its own array, where the window leaves it alone.

    Translation is unaffected by an identical crop of both frames, so nothing
    is biased by this; what it does cost is range, since the correlation
    surface wraps at half the CROPPED size.
    """
    weight = float(soft.sum())
    if weight <= 0:
        return None
    box = mask_box(soft)
    if box is None:
        return None
    top, bottom, left, right = box
    window = soft[top:bottom, left:right]
    first = reference[top:bottom, left:right]
    second = image[top:bottom, left:right]
    if min(window.shape) < 16:
        return None
    level = float((first * window).sum() / max(window.sum(), 1e-9))
    other = float((second * window).sum() / max(window.sum(), 1e-9))
    reach = min(window.shape[0] // 2, window.shape[1] // 2)
    if max_shift:
        reach = min(int(max_shift), reach)
    # The Hann window inside phase_shift is KEPT. The band runs into the frame
    # edge, and under vertical motion the rows at that edge have no counterpart
    # in the other frame; a flat-topped window keeps them at full weight and
    # they pull the answer back towards zero shift. Hann suppresses them.
    return geo.phase_shift((first - level) * window,
                           (second - other) * window, max_shift=reach)


# ---------------------------------------------------------------------------
# D. Frame-wide survey over the land
# ---------------------------------------------------------------------------

def land_cells(mask, size=CELL, coverage=0.9):
    """Grid cells lying (almost) wholly inside the land mask.

    A rigid camera translation moves EVERY cell by the SAME vector. That is
    the decisive test in this file: it needs no view about which part of the
    scene is interesting, and when it fails there is no rigid transform to
    find, whatever the confidence numbers say.

    THE GRID IS ANCHORED TO THE MASK, NOT TO THE FRAME ORIGIN. Anchored at
    (0, 0) a 182-row band of land at rows 298-479 yields ZERO cells, because
    the only grid row that would fit starts at 384 and runs off the bottom --
    the decisive test silently does not run, on a mask that plainly has room
    for it. Cells stay non-overlapping so their agreement is independent
    evidence; if the band is too narrow for three of them the cell size drops
    rather than the cells being allowed to share pixels.
    """
    box = mask_box(mask.astype(float))
    if box is None:
        return [], size
    top, bottom, left, right = box
    for trial in (size, size // 2, size // 4):
        if trial < 32:
            break
        cells = []
        for y in range(top, bottom - trial + 1, trial):
            for x in range(left, right - trial + 1, trial):
                if mask[y:y + trial, x:x + trial].mean() >= coverage:
                    cells.append({"name": f"c{len(cells) + 1:03d}", "x": x,
                                  "y": y, "w": trial, "h": trial})
        if len(cells) >= MIN_CLUSTER:
            return cells, trial
    return cells, trial


# ---------------------------------------------------------------------------
# Noise algebra
# ---------------------------------------------------------------------------

def within_route_noise(direct, dates):
    """Per-measurement error of ONE route, from its own two internal paths.

    Same construction as the Walton run and the same algebra. Each consecutive
    pair is measured twice: directly against its neighbour, and as the
    difference of the two measurements against the reference. If a single
    measurement carries error e, the neighbour step carries e, the difference
    of two reference measurements carries e*sqrt(2), and the gap between the
    two carries e*sqrt(3). So noise = gap / sqrt(3).

    `direct` must carry dx/dy indexed by date AND a `step_dx`/`step_dy` column
    holding the neighbour measurement, or None is returned.
    """
    needed = {"dx", "dy", "step_dx", "step_dy"}
    if direct is None or direct.empty or not needed <= set(direct.columns):
        return None
    # DIFFERENCE ADJACENT ROWS OF THE FULL RECORD, NOT OF THE SURVIVORS.
    # step_dx at row i was measured against row i-1 of the record. Dropping the
    # weak frames first and then calling .diff() silently pairs each step with
    # whatever row happened to survive before it, which on a foggy stretch can
    # be a week away -- so the "gap" would be measuring the camera's real
    # motion over that week and calling it noise.
    ordered = direct.sort_index()
    predicted_x = ordered["dx"].diff()
    predicted_y = ordered["dy"].diff()
    gap = np.hypot(ordered["step_dx"] - predicted_x,
                   ordered["step_dy"] - predicted_y).dropna()
    if len(gap) < 3:
        return None
    median = float(np.median(gap))
    return {"pairs": int(len(gap)), "gap": median,
            "noise": median / math.sqrt(3.0),
            "resolution": 3.0 * median / math.sqrt(3.0)}


def between_route_gap(first, second):
    """How far the two routes disagree about the SAME frame.

    Two independent measurements of one displacement, each with error e1, e2,
    differ by sqrt(e1^2 + e2^2) -- sqrt(2) times a common e, NOT the sqrt(3)
    of the within-route check, which compares a step against a difference of
    two measurements. Using the wrong constant here would understate the error
    by about 22% and hand back a resolution finer than the data supports.

    The predicted gap is computed from each route's own stated noise, so the
    comparison answers a question neither route can ask alone: do these two
    methods disagree by more than they each claim to be uncertain by?
    """
    if first is None or second is None or first.empty or second.empty:
        return None
    shared = first.index.intersection(second.index)
    if len(shared) < 3:
        return None
    gap = np.hypot(first.loc[shared, "dx"] - second.loc[shared, "dx"],
                   first.loc[shared, "dy"] - second.loc[shared, "dy"])
    gap = gap.dropna()
    if gap.empty:
        return None
    return {"frames": int(len(gap)), "gap": float(np.median(gap)),
            "worst": float(gap.max())}


def resolution_from(parts, cross):
    """The step size this record can actually resolve, from every estimate.

    Takes the WORST of the per-route noises and the between-route implication,
    because a resolution claim is a promise and the promise has to hold for
    whichever route a step is read from.
    """
    errors = [p["noise"] for p in parts if p]
    if cross:
        errors.append(cross["gap"] / math.sqrt(2.0))
    if not errors:
        return None
    noise = max(errors)
    return {"noise": noise, "resolution": 3.0 * noise}


# ---------------------------------------------------------------------------
# The two routes, measured over the record
# ---------------------------------------------------------------------------

def phase_series(paths, dates, soft, pick, max_shift):
    """Route 2. Masked phase correlation against the reference, plus the
    neighbour step, which is what gives this route its own error bar."""
    reference = geo.load_gray(paths[pick])
    if reference is None:
        sys.exit("the reference frame is not a readable image")
    rows, previous, previous_date = [], None, None
    weak = outside = unreadable = 0
    for path, date in zip(paths, dates):
        image = geo.load_gray(path)
        if image is None or image.shape != reference.shape:
            unreadable += 1
            previous, previous_date = None, None
            continue
        got = masked_phase(reference, image, soft, max_shift)
        row = {"date": date, "frame": os.path.basename(path)}
        if got is None or got[2] < MIN_CONFIDENCE:
            weak += 1
        else:
            dy, dx, confidence, beyond = got
            outside += int(beyond)
            row.update({"dx": dx, "dy": dy, "confidence": confidence})
        if previous is not None:
            step = masked_phase(previous, image, soft, max_shift)
            if step is not None and step[2] >= MIN_CONFIDENCE:
                row["step_dx"], row["step_dy"] = step[1], step[0]
        rows.append(row)
        previous, previous_date = image, date
    if unreadable:
        print(f"  {unreadable} frames unreadable or a different size, skipped")
    if weak:
        print(f"  {weak}/{len(paths)} frames below confidence "
              f"{MIN_CONFIDENCE} (fog, night, rain) — dropped")
    if outside:
        share = outside / max(1, len(paths))
        print(f"  {outside}/{len(paths)} frames had their tallest peak beyond "
              f"{max_shift} px and were\n  re-measured inside it")
        if share > geo.OVERRULE_SHARE:
            print(f"  WARNING: that is {share * 100:.0f}% of the record. Re-run "
                  f"with --max-shift {max_shift * 3} before\n  believing any "
                  "offset below; those frames were given a position rather "
                  "than measured at one.")
    frame = pd.DataFrame(rows)
    return frame.set_index("date") if not frame.empty else frame


def homography_series(paths, dates, mask, pick, kind):
    """Route 1. ORB/SIFT + RANSAC homography against the reference, plus the
    neighbour fit. Features are detected on the land mask only."""
    cv2 = require_cv2()
    detector, norm = make_detector(kind, cv2)

    reference = geo.load_gray(paths[pick])
    if reference is None:
        sys.exit("the reference frame is not a readable image")
    shape = reference.shape
    ref_kp, ref_desc = frame_features(reference, mask, detector, cv2)
    print(f"  reference carries {0 if ref_desc is None else len(ref_desc)} "
          f"{kind.upper()} features on the land")
    if ref_desc is None or len(ref_desc) < MIN_MATCHES:
        sys.exit("the reference frame has too few land features to register "
                 "against.\nPick another reference with --reference, or widen "
                 "the mask.")

    rows, previous, reasons = [], None, {}
    for path, date in zip(paths, dates):
        image = geo.load_gray(path)
        if image is None or image.shape != shape:
            reasons["unreadable or resized"] = reasons.get("unreadable or resized", 0) + 1
            previous = None
            continue
        keypoints, descriptors = frame_features(image, mask, detector, cv2)
        row = {"date": date, "frame": os.path.basename(path),
               "features": 0 if descriptors is None else len(descriptors)}

        fit, why = fit_homography(ref_kp, ref_desc, keypoints, descriptors,
                                  norm, cv2)
        if fit is None:
            reasons[why] = reasons.get(why, 0) + 1
        else:
            parts = decompose(fit["H"], shape)
            if parts:
                row.update(parts)
                row.update({"inliers": fit["inliers"], "matches": fit["matches"],
                            "share": fit["share"], "residual": fit["residual"]})

        if previous is not None:
            step, _ = fit_homography(previous[0], previous[1], keypoints,
                                     descriptors, norm, cv2)
            if step is not None:
                moved = decompose(step["H"], shape)
                if moved:
                    row["step_dx"], row["step_dy"] = moved["dx"], moved["dy"]
                    row["step_rotation"] = moved["rotation"]
        rows.append(row)
        previous = (keypoints, descriptors)

    for why, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  {count} frames did not fit: {why}")
    frame = pd.DataFrame(rows)
    return frame.set_index("date") if not frame.empty else frame


def run_survey(paths, dates, mask, coarse):
    """D. Tile the land, register every cell, ask whether they agree.

    A rigid camera translation displaces every cell by the same vector, so a
    common vector either exists or it does not. This is the strongest single
    piece of evidence available and it appeals to nothing about what the scene
    looks like.
    """
    print("\n" + "=" * 74)
    print("D. FRAME-WIDE SURVEY OF THE LAND")
    print("=" * 74)
    cells, used = land_cells(mask)
    print(f"{len(cells)} non-overlapping cells of {used} px lie wholly inside "
          "the land mask")
    if used != CELL:
        print(f"  (dropped from {CELL} px: the land band is too narrow to fit "
              f"{MIN_CLUSTER} of them)")
    if len(cells) < MIN_CLUSTER:
        print("  NOT TESTED. The mask cannot hold three independent cells even "
              "at 32 px, so\n  this record gets no survey — which is a gap in "
              "the evidence, not a pass.")
        return "untested", cells
    tracked, _ = geo.track(paths, dates, cells, pick=0, coarse=coarse)
    if tracked.empty:
        print("  NOT TESTED. No cell registered on any frame.")
        return "untested", cells
    kept, rejected, distance = geo.agreeing_features(tracked, tolerance=AGREE_PX,
                                                     minimum=MIN_CLUSTER)
    print("\n" + "=" * 74)
    print("SURVEY RESULT — DO THE LAND CELLS AGREE ON ONE VECTOR?")
    print("=" * 74)
    if len(kept) >= MIN_CLUSTER:
        inner = [distance[name] for name in kept if name in distance]
        print(f"  YES. {len(kept)} of {len(cells)} cells agree to within "
              f"{AGREE_PX} px"
              + (f" (typical {np.median(inner):.2f} px)" if inner else ""))
        xs = [c["x"] for c in cells if c["name"] in kept]
        ys = [c["y"] for c in cells if c["name"] in kept]
        print(f"  they span x {min(xs)}-{max(xs) + used}, "
              f"y {min(ys)}-{max(ys) + used}")
        if max(ys) - min(ys) < 2 * used:
            print("  WARNING: they sit in one band of rows, so their agreement "
                  "is not fully\n  independent evidence.")
        return kept, cells
    closest = min(distance.values()) if distance else float("nan")
    print(f"  NO. No {MIN_CLUSTER} of {len(cells)} cells agree to within "
          f"{AGREE_PX} px.")
    print(f"  The closest any cell comes to the rest is {closest:.1f} px.")
    print("  There is no consistent rigid transform to find. Every offset "
          "reported below\n  is a number the method produced, not a "
          "displacement of the camera.")
    return None, cells


def route_steps(series, resolution, label):
    """Steps in one route's displacement vector, at the derived resolution."""
    if series is None or series.empty:
        return []
    usable = series.dropna(subset=["dx", "dy"])[["dx", "dy"]]
    if len(usable) < 2 * PERSIST + 1:
        print(f"  {label}: only {len(usable)} usable frames; no step test")
        return []
    return geo.find_steps(usable, threshold=resolution, persist=PERSIST)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def epoch_table(steps, dates, resolution, title):
    print(f"\n{title}")
    if not steps:
        print(f"  no step above {resolution:.2g} px persists — ONE epoch "
              "AT THIS RESOLUTION")
    epochs = geo.describe_epochs(steps, dates)
    for index, epoch in enumerate(epochs, 1):
        days = (epoch["end"] - epoch["start"]).days
        worth = "usable" if epoch["frames"] >= 20 and days >= 30 else "too short to use"
        print(f"  {index:>2}. {epoch['start']:%Y-%m-%d} to {epoch['end']:%Y-%m-%d}"
              f"  {days:>4}d  {epoch['frames']:>4} frames   {worth}")
    return epochs


def verdict(epochs, dates, resolution, slug):
    """One plain-language answer per camera."""
    print("\n" + "=" * 74)
    print(f"VERDICT  {slug}")
    print("=" * 74)
    if not epochs:
        print("  No registration succeeded. There is no verdict to give.")
        return
    best = max(epochs, key=lambda e: ((e["end"] - e["start"]).days, e["frames"]))
    days = (best["end"] - best["start"]).days
    total = (dates[-1] - dates[0]).days or 1
    if len(epochs) == 1:
        print(f"  The record reads as ONE geometric epoch at this resolution:")
        print(f"  {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}, {total} days, "
              f"{len(dates)} frames.")
        print(f"  A move smaller than {resolution:.2g} px, or one inside a gap "
              "in the sampling,\n  would not appear. That is evidence of "
              "stability, not proof of it.")
    else:
        print(f"  The record breaks into {len(epochs)} epochs.")
        print(f"  Longest continuous stable stretch: {best['start']:%Y-%m-%d} "
              f"to {best['end']:%Y-%m-%d}")
        print(f"  — {days} days, {best['frames']} frames, "
              f"{days / total * 100:.0f}% of the record.")
    print(f"  Smallest move this record can resolve: {resolution:.2g} px.")
    print("  Nothing has been corrected or re-registered.")


# ---------------------------------------------------------------------------

def parse_polygon(text):
    """--mask-poly 'x,y x,y x,y' in fractional coordinates."""
    points = []
    for pair in text.replace(",", " ").split():
        points.append(float(pair))
    if len(points) < 6 or len(points) % 2:
        sys.exit("--mask-poly needs at least three x,y pairs")
    return [(points[i], points[i + 1]) for i in range(0, len(points), 2)]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", required=True,
                    help="camera slug, label or unique substring")
    ap.add_argument("--inventory", action="store_true",
                    help="frame geometry and availability only; stop there")
    ap.add_argument("--cached", action="store_true",
                    help="reuse frames already downloaded; no network")
    ap.add_argument("--every", type=int, default=DEFAULT_EVERY_DAYS,
                    help=f"sample spacing in days (default {DEFAULT_EVERY_DAYS})")
    ap.add_argument("--hour", type=int, default=DEFAULT_HOUR_UTC,
                    help=f"target hour UTC (default {DEFAULT_HOUR_UTC})")
    ap.add_argument("--limit", type=int, help="first N sample targets only")
    ap.add_argument("--since", help="ignore frames before this date")
    ap.add_argument("--until", help="ignore frames after this date")
    ap.add_argument("--mask-poly", action="append",
                    help="fractional polygon 'x,y x,y x,y' (repeatable)")
    ap.add_argument("--mask-drop", action="append",
                    help="fractional polygon to cut back out (repeatable)")
    ap.add_argument("--mask-audit", action="store_true",
                    help="check the declared mask against every cached frame "
                         "for water, and stop")
    ap.add_argument("--warm", type=int, default=25,
                    help="R-B level below which a pixel is called water in "
                         "the mask audit")
    ap.add_argument("--mask-preview", action="store_true",
                    help="draw the mask over a frame and stop")
    ap.add_argument("--survey", action="store_true",
                    help="tile the land and test for a common vector")
    ap.add_argument("--detector", choices=("sift", "orb"), default="sift")
    ap.add_argument("--no-homography", action="store_true",
                    help="phase route only (leaves nothing to cross-check)")
    ap.add_argument("--reference", help="force the reference frame date, "
                                        "YYYY-MM-DD, for the swap test")
    ap.add_argument("--max-shift", type=int, default=MAX_SHIFT_PX)
    ap.add_argument("--step-px", type=float, default=None,
                    help="override the derived step threshold (rarely right)")
    args = ap.parse_args()

    geo.require_imaging()

    # THE CHEAP HALF OF THE INVENTORY COMES BEFORE THE DOWNLOAD. Per-day
    # availability is one request against the service inventory and needs no
    # imagery at all, so asking for it should not first cost a daily pull of
    # the whole record. Reported first, it also says whether a daily sample is
    # even meaningful before the frames are fetched.
    if args.inventory and not args.cached:
        import pull_rip_detection as rip
        asset = rip.find_camera(rip.load_assets(), args.camera)
        service, _ = rip.find_stills_service(asset)
        availability(service)

    # ---- frames -----------------------------------------------------------
    if args.cached:
        slug, paths, dates = geo.cached_frames(args.camera)
    else:
        slug, paths, dates = geo.sample_frames(args.camera, args.every,
                                               args.hour, args.limit)
    order = np.argsort(dates)
    paths = [paths[i] for i in order]
    dates = [dates[i] for i in order]

    # ---- A. inventory, before anything is registered ----------------------
    print("\n" + "=" * 74)
    print(f"A. FRAME GEOMETRY INVENTORY  {slug}")
    print("=" * 74)
    print(f"{len(paths)} frames on disk, {dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}")
    groups, unreadable = geo.frame_sizes(paths, dates)
    if unreadable:
        # A file that will not open is a dud download, not an epoch. Drop it
        # here and renumber the size groups onto the shortened record, rather
        # than re-scanning (which would print the inventory a second time).
        drop = set(unreadable)
        keep = [i for i in range(len(paths)) if i not in drop]
        renumber = {old: new for new, old in enumerate(keep)}
        paths = [paths[i] for i in keep]
        dates = [dates[i] for i in keep]
        groups = {size: [renumber[i] for i in indices]
                  for size, indices in groups.items()}
    if not groups:
        sys.exit("no readable frames in the sample")

    gaps = sampled_gaps(dates, args.every)
    if gaps:
        print(f"\n  {sum(len(g) for g in gaps)} intended sample days produced "
              f"no frame, in {len(gaps)} gaps; longest:")
        for run in sorted(gaps, key=len, reverse=True)[:6]:
            print(f"    {run[0]:%Y-%m-%d} to {run[-1]:%Y-%m-%d}  {len(run)} days")

    if args.inventory:
        print("\nInventory only. Nothing registered.")
        print("Frame sizes above cover the SAMPLED frames. A size change "
              "shorter than the\nsampling interval can still hide; at Walton "
              "the change lasted five days and\nweekly sampling missed it, "
              "which is why the default here is daily.")
        return

    # Registration happens INSIDE one frame size. Across a size change there is
    # no pixel correspondence to measure, so pooling them would be measuring
    # the change itself over and over.
    if len(groups) > 1:
        biggest = max(groups.items(), key=lambda kv: len(kv[1]))
        keep = set(biggest[1])
        print(f"\n  registering only the {biggest[0][0]}x{biggest[0][1]} group "
              f"({len(keep)} frames).")
        print("  The other sizes are their own epochs by definition and are "
              "not compared to it.")
        paths = [p for i, p in enumerate(paths) if i in keep]
        dates = [d for i, d in enumerate(dates) if i in keep]

    if args.since:
        keep = [i for i, d in enumerate(dates) if d >= pd.Timestamp(args.since, tz="UTC")]
        paths, dates = [paths[i] for i in keep], [dates[i] for i in keep]
    if args.until:
        keep = [i for i, d in enumerate(dates) if d <= pd.Timestamp(args.until, tz="UTC")]
        paths, dates = [paths[i] for i in keep], [dates[i] for i in keep]
    if len(paths) < 2 * PERSIST + 1:
        sys.exit(f"only {len(paths)} frames after filtering; too few to test")

    # ---- the land mask ----------------------------------------------------
    override = None
    if args.mask_poly:
        override = {"keep": [parse_polygon(p) for p in args.mask_poly],
                    "drop": [parse_polygon(p) for p in (args.mask_drop or [])],
                    "note": "passed on the command line for this run only"}
    spec = mask_spec(slug, override)

    # THE PREVIEW IS THE TOOL FOR NOT HAVING A MASK YET, so it has to work
    # before one exists. Exiting here with "declare a mask first" would send
    # you to find a frame, open it, and guess at fractions by eye -- which is
    # the step this is supposed to replace.
    if args.mask_preview and spec is None:
        sample = geo.load_gray(paths[len(paths) // 2])
        if sample is None:
            sys.exit("the middle frame is not a readable image")
        out = os.path.join(OUT_DIR, f"grid_{slug}.jpg")
        draw_grid_preview(paths[len(paths) // 2], out)
        print(f"\nNo mask is declared for {slug!r} yet.\nwrote {out}")
        print("\nThe grid is in FRACTIONS of the frame, which is what MASKS "
              "takes. Read the\nland region off it and add an entry:\n")
        print(f'    "{slug}": {{')
        print('        "keep": [[(0.0, 0.60), (1.0, 0.56), (1.0, 1.0), '
              '(0.0, 1.0)]],')
        print('        "drop": [],')
        print(f'        "note": "drawn by hand from '
              f'{os.path.basename(paths[len(paths) // 2])}",')
        print("    },")
        print("\nOr try one without committing to it:")
        print('    --mask-poly "0,0.60 1,0.56 1,1 0,1" --mask-preview')
        return

    if spec is None:
        sys.exit(
            f"\nNo land mask is declared for {slug!r}, and registration will "
            "not run without one.\n"
            "The frame is mostly moving water; an unmasked run returns a "
            "confident number\ndriven by waves that looks exactly like a "
            "measurement of the camera.\n\n"
            "  1. python check_geometry_strict.py --camera "
            f"{args.camera} --cached --mask-preview\n"
            "  2. read the land region off the preview as fractions of the "
            "frame\n"
            "  3. add it to MASKS in this file, with a note saying which "
            "frame it came from")

    sample = geo.load_gray(paths[len(paths) // 2])
    if sample is None:
        sys.exit("the middle frame is not a readable image")
    shape = sample.shape
    mask = build_mask(shape, spec)
    describe_mask(mask, spec, shape)

    if args.mask_preview:
        out = os.path.join(OUT_DIR, f"mask_{slug}.jpg")
        draw_mask_preview(paths[len(paths) // 2], mask, out)
        print(f"\nwrote {out}\nCheck that every bright area is land that does "
              "not move, then\nrecord the polygon in MASKS and re-run.")

    # THE AUDIT IS THE ONLY THING THAT TESTS THE MASK AGAINST THE RECORD. The
    # preview above shows one frame at one tide; the mask is a claim about
    # every frame, and only this checks it. It runs before registration so a
    # mask that catches swash is caught before its results are believed.
    if args.mask_audit or args.mask_preview:
        audit_mask(paths, dates, mask, spec, warm=args.warm, slug=slug)
    if args.mask_preview or args.mask_audit:
        return

    print("  mask as run: " + json.dumps(
        {"keep": [[list(p) for p in poly] for poly in spec.get("keep") or []],
         "drop": [[list(p) for p in poly] for poly in spec.get("drop") or []]}))
    print("\nEverything below is measured on the masked land only.")

    soft = feathered(mask)

    # ---- reference --------------------------------------------------------
    if args.reference:
        wanted = pd.Timestamp(args.reference, tz="UTC")
        pick = int(np.argmin([abs((d - wanted).total_seconds()) for d in dates]))
        print(f"\nreference FORCED to {dates[pick]:%Y-%m-%d} "
              "(reference-swap test)")
    else:
        pick = pick_reference(paths, spec)
        print(f"\nreference frame: {dates[pick]:%Y-%m-%d} "
              f"({os.path.basename(paths[pick])}) — of {len(paths)} frames, "
              "the one\nthe rest of the record matches best on the land")

    # ---- route 2 first: it is cheap and it gives the survey its coarse -----
    print("\n" + "=" * 74)
    print("ROUTE 2  masked phase correlation  (cross-check)")
    print("=" * 74)
    phase = phase_series(paths, dates, soft, pick, args.max_shift)
    coarse = phase.dropna(subset=["dx", "dy"])[["dx", "dy"]] \
        if not phase.empty else None

    # ---- D. the survey, early and decisive --------------------------------
    survey_kept = None
    if args.survey:
        survey_kept, _ = run_survey(paths, dates, mask, coarse)

    # ---- route 1: the primary measurement ---------------------------------
    homography = pd.DataFrame()
    if not args.no_homography:
        print("\n" + "=" * 74)
        print(f"ROUTE 1  {args.detector.upper()} + RANSAC homography  (primary)")
        print("=" * 74)
        homography = homography_series(paths, dates, mask, pick, args.detector)

    # ---- the error bars ---------------------------------------------------
    print("\n" + "=" * 74)
    print("HOW WELL THIS RECORD MEASURES ITSELF")
    print("=" * 74)
    phase_noise = within_route_noise(phase, dates)
    homog_noise = within_route_noise(homography, dates)
    for name, part in (("homography", homog_noise), ("phase", phase_noise)):
        if part:
            print(f"  {name:<12} {part['pairs']:>5} pairs measured both ways, "
                  f"gap {part['gap']:.2f} px → ±{part['noise']:.2g} px")
        else:
            print(f"  {name:<12} not enough paired measurements for an "
                  "error bar")
    cross = between_route_gap(
        homography.dropna(subset=["dx", "dy"]) if not homography.empty else None,
        phase.dropna(subset=["dx", "dy"]) if not phase.empty else None)
    if cross:
        predicted = math.hypot(*(p["noise"] if p else 0.0
                                 for p in (homog_noise, phase_noise)))
        print(f"\n  the two routes differ by a median of {cross['gap']:.2f} px "
              f"on the same frame\n  ({cross['frames']} frames measured by "
              f"both; worst {cross['worst']:.1f} px)")
        if predicted > 0:
            print(f"  their own error bars predict a gap of "
                  f"{predicted:.2f} px")
            if cross["gap"] > 3 * max(predicted, 1e-6):
                print("  !! THE ROUTES DISAGREE BY MORE THAN THEY CLAIM TO BE "
                      "UNCERTAIN BY.\n  At least one is locking onto something "
                      "other than the scene. Neither\n  route's offsets are "
                      "evidence until that is resolved.")
    limit = resolution_from([homog_noise, phase_noise], cross)
    if limit is None:
        print("\n  No error bar could be derived, so no step threshold can be "
              "justified.\n  Nothing below is reported as a finding.")
        return
    resolution = args.step_px or limit["resolution"]
    print(f"\n  per-frame error: ±{limit['noise']:.2g} px")
    print(f"  SMALLEST MOVE THIS RECORD CAN RESOLVE: {resolution:.2g} px "
          "(3x that error)")
    if args.step_px:
        print(f"  (overridden from {limit['resolution']:.2g} px on the "
              "command line)")

    # ---- E. rotation and scale, not just translation ----------------------
    if not homography.empty and "rotation" in homography.columns:
        turned = homography.dropna(subset=["rotation"])
        if not turned.empty:
            print("\n" + "=" * 74)
            print("ROTATION, SCALE AND BEND  (invisible to a shift-only method)")
            print("=" * 74)
            for column, unit, note in (
                    ("rotation", "deg", "0 is level with the reference"),
                    ("scale", "x", "1.0 is the same zoom"),
                    ("aniso", "x", "1.0 is a similarity; higher is stretched"),
                    ("bend", "px", "0 is affine; higher is genuinely projective")):
                values = turned[column].replace([np.inf, -np.inf], np.nan).dropna()
                if values.empty:
                    continue
                print(f"  {column:<9} median {values.median():>9.4f} {unit:<4} "
                      f"range {values.min():>9.4f} to {values.max():>9.4f}"
                      f"   ({note})")
            span = float(turned["rotation"].max() - turned["rotation"].min())
            if span > 0.5:
                print(f"\n  The frame turns through {span:.2f} deg across the "
                      "record. A shift-only\n  method cannot see this at all, "
                      "and it is a geometric break.")
            zoom = turned["scale"].dropna()
            if not zoom.empty and float(zoom.max() / max(zoom.min(), 1e-9)) > 1.02:
                print(f"  Scale moves by {(zoom.max() / zoom.min() - 1) * 100:.1f}%"
                      " across the record — a zoom change,\n  which leaves the "
                      "frame size untouched and so never shows in metadata.")

    # ---- steps, per route, then against each other ------------------------
    homog_steps = route_steps(homography, resolution, "homography")
    phase_steps = route_steps(phase, resolution, "phase")

    print("\n" + "=" * 74)
    print("CANDIDATE EPOCH BOUNDARIES")
    print("=" * 74)
    measured = lambda frame: (list(frame.dropna(subset=["dx"]).index)
                              if not frame.empty and "dx" in frame.columns
                              else [])
    rows = compare_routes(homog_steps, measured(homography),
                          phase_steps, measured(phase))
    if not rows:
        print("  Neither route found a step above the resolution.")
    for verdict_text, when, first, second in rows:
        sizes = f"{first:.1f} px" if second is None else f"{first:.1f} / {second:.1f} px"
        print(f"  {when:%Y-%m-%d}  {sizes:<20} {verdict_text}")

    single = homography.empty or "dx" not in homography.columns
    both = [r for r in rows if r[0].startswith("both")]
    if single:
        # ONE ROUTE CANNOT CORROBORATE ITSELF. With the homography route off
        # there is no second opinion, so nothing here is confirmed and the
        # epoch list has to say so rather than quietly print a verdict.
        print("\n  Only ONE route ran, so NOTHING here is corroborated. Every "
              "date above is a\n  candidate that no independent method has "
              "seen. This is the state the Walton\n  run was in when its "
              "per-quarter pass rate looked reassuring.")
        confirmed = []
    else:
        print(f"\n  {len(both)} of {len(rows)} candidate moves are seen by "
              "BOTH routes.")
        print("  Only those are treated as confirmed. A step one route found "
              "alone is a\n  question, not a finding.")
        confirmed = [s for s in homog_steps
                     if any(abs((s["date"] - r[1]).days) <= 1 for r in both)]

    epochs = epoch_table(confirmed, list(dates), resolution,
                         "EPOCHS FROM THE CONFIRMED BOUNDARIES ONLY"
                         if not single else
                         "EPOCHS — NONE CONFIRMED, SHOWN AS ONE BLOCK")

    # ---- outputs ----------------------------------------------------------
    write_series(slug, homography, phase)
    if not homography.empty and "dx" in homography.columns:
        offsets = np.hypot(homography["dx"], homography["dy"]).dropna()
        for line in geo.spark(offsets, marks=[s["date"] for s in confirmed],
                              label="offset from reference, px"):
            print(line)

    if args.survey and survey_kept == "untested":
        print("\n  NOTE: the survey could not run, so the strongest available "
              "check on these\n  epochs was not made. That is missing "
              "evidence, not evidence of stability.")
    elif args.survey and survey_kept is None:
        print("\n  NOTE: the survey found NO common vector, so the epoch list "
              "above rests on a\n  rigid transform the land itself does not "
              "agree exists. Treat every number\n  above as a number the "
              "method produced rather than a displacement.")
    verdict(epochs, list(dates), resolution, slug)

    if not args.reference:
        print("\n  F. REFERENCE-SWAP TEST — one run, decisive. Re-run with")
        print(f"       --reference {dates[len(dates) // 4]:%Y-%m-%d}")
        print("     If the offsets are real they all shift by one constant and "
              "the steps keep\n     their dates and sizes. If they are "
              "spurious the pattern reorganises.")


def pick_reference(paths, spec, probes=8, downsample=8):
    """The frame the rest of the record matches best ON THE LAND.

    geo.best_reference scores whole frames, which on a majority-water view is
    partly scoring how well the waves happened to line up. Same idea, same
    median-peak criterion, but measured through the mask.
    """
    images = [geo.load_gray(path, downsample=downsample) for path in paths]
    usable = [i for i, image in enumerate(images)
              if image is not None and float(image.std()) >= geo.BLANK_STD]
    if len(usable) < 3:
        return 0
    shape = min((images[i].shape for i in usable), key=lambda s: (s[0], s[1]))
    small = feathered(build_mask(shape, spec), radius=max(2, 16 // downsample))
    step = max(1, len(usable) // probes)
    probe_set = usable[::step][:probes]
    scores = []
    for index in usable:
        base = images[index][:shape[0], :shape[1]]
        peaks = []
        for other in probe_set:
            if other == index:
                continue
            got = masked_phase(base, images[other][:shape[0], :shape[1]], small,
                               max_shift=None)
            if got is not None:
                peaks.append(got[2])
        scores.append((float(np.median(peaks)) if peaks else 0.0, index))
    return max(scores)[1]


def compare_routes(first, first_dates, second, second_dates,
                   persist=PERSIST, window_days=None):
    """Which candidate boundaries both routes found, and which only one did.

    A step the other route missed is not automatically wrong; what matters is
    whether that route had frames on both sides of the date at all. So an
    unmatched step is reported as contradicted or as unmeasured, never just
    as absent.
    """
    if window_days is None:
        window_days = max(2.0 * geo.sampling_days(first_dates or second_dates),
                          1.0)

    def near(date, dates):
        return [d for d in dates if abs((d - date).days) <= window_days]

    def covered(date, dates):
        before = [d for d in dates if d < date]
        after = [d for d in dates if d >= date]
        return len(before) >= persist and len(after) >= persist

    rows = []
    for step in first:
        match = near(step["date"], [s["date"] for s in second])
        if match:
            twin = min(second, key=lambda s: abs((s["date"] - step["date"]).days))
            big, small = sorted((step["jump"], twin["jump"]), reverse=True)
            label = "both routes" if big <= 1.5 * max(small, 1e-6) \
                else "both routes, but the sizes disagree"
            rows.append((label, step["date"], step["jump"], twin["jump"]))
        elif covered(step["date"], second_dates):
            rows.append(("homography only — phase saw the date and found "
                         "nothing", step["date"], step["jump"], None))
        else:
            rows.append(("homography only — phase had no frames there",
                         step["date"], step["jump"], None))
    for step in second:
        if not near(step["date"], [s["date"] for s in first]):
            if covered(step["date"], first_dates):
                rows.append(("phase only — homography saw the date and found "
                             "nothing", step["date"], step["jump"], None))
            else:
                rows.append(("phase only — homography had no frames there",
                             step["date"], step["jump"], None))
    return sorted(rows, key=lambda row: row[1])


def write_series(slug, homography, phase):
    """Per-frame offset, rotation and scale, as CSV, for both routes."""
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for name, frame in (("homography", homography), ("phase", phase)):
        if frame is None or frame.empty:
            continue
        path = os.path.join(OUT_DIR, f"strict_{name}_{slug}.csv")
        frame.to_csv(path)
        written.append(path)
    if written:
        print("\nwrote:")
        for path in written:
            print(f"  {path}")


def draw_grid_preview(path, out_path, step=0.05, label_every=2):
    """The frame with a labelled grid in FRACTIONAL coordinates.

    Fractions rather than pixels because that is what MASKS stores, and
    because a mask read off a preview in pixels silently becomes wrong the
    moment the camera's resolution changes.
    """
    from PIL import Image, ImageDraw
    with Image.open(path) as img:
        rgb = img.convert("RGB").copy()
    draw = ImageDraw.Draw(rgb)
    width, height = rgb.size
    count = int(round(1.0 / step))
    for index in range(count + 1):
        fraction = index * step
        x, y = int(fraction * (width - 1)), int(fraction * (height - 1))
        heavy = index % label_every == 0
        colour = (255, 200, 0) if heavy else (120, 120, 120)
        draw.line([(x, 0), (x, height)], fill=colour, width=2 if heavy else 1)
        draw.line([(0, y), (width, y)], fill=colour, width=2 if heavy else 1)
        if heavy:
            for spot, text in (((x + 3, 3), f"{fraction:.2f}"),
                               ((3, y + 3), f"{fraction:.2f}")):
                draw.text(spot, text, fill=(0, 0, 0))
                draw.text((spot[0] - 1, spot[1] - 1), text, fill=(255, 255, 0))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rgb.save(out_path, quality=90)
    return out_path


def water_frequency(paths, shape, warm=25, downsample=4):
    """How often each pixel looked like water, across every frame given.

    A mask drawn on ONE frame is a hypothesis about every other frame. The
    tide moves the waterline, storms move it further, and a mask that holds at
    the hour it was drawn can still swallow swash on a spring high. Nothing
    in a single-frame preview can show that; this can.

    The discriminator is colour, not motion, because the sampling here is
    daily and consecutive samples are a day apart -- there is no short
    timescale left in which water moves and sand does not. Dry sand is warm
    (R - B strongly positive); water, foam and wet sand are not. `warm` is
    that threshold in levels.

    It reads whitewater and wet sand as water, which is the conservative
    direction, and it also reads a fog whiteout as water, which is not a mask
    fault -- so the per-frame shares are reported rather than averaged into a
    verdict. Returns (frequency, per_frame, size) where `frequency` is the
    share of frames in which each pixel looked like water.
    """
    from PIL import Image
    total, count, per_frame = None, 0, []
    for path in paths:
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                if downsample > 1:
                    rgb = rgb.reduce(downsample)
                array = np.asarray(rgb).astype(np.int16)
        except Exception:
            continue
        wet = (array[..., 0] - array[..., 2]) < warm
        if total is None:
            total = np.zeros(wet.shape, dtype=np.float64)
        elif wet.shape != total.shape:
            continue
        total += wet
        count += 1
        per_frame.append(wet)
    if not count:
        return None, [], None
    return total / count, per_frame, total.shape


def measured_edge(frequency, often=0.25, margin=0.03, slug="<camera>"):
    """Where water actually reaches, fitted from the record rather than drawn.

    The mask's seaward edge is the one number in the whole file that is pure
    judgement, and the first Sailfish draft got it wrong by a tenth of the
    frame in the cautious direction -- which is not safe, it is 150 rows of
    the most textured land thrown away. This replaces the judgement with a
    measurement: for each column, the most landward row that looked like water
    in at least `often` of the frames, fitted to a line and offset by
    `margin`.

    It PROPOSES; it does not install. The mask stays hand-declared and
    logged, because a mask fitted automatically to a record that contains a
    fog winter or a beach renourishment would be fitted to that instead, and
    nothing downstream would say so.
    """
    height, width = frequency.shape
    columns, rows = [], []
    for x in range(width):
        wet = np.where(frequency[:, x] >= often)[0]
        if len(wet):
            columns.append(x / max(width - 1, 1))
            rows.append(wet[-1] / max(height - 1, 1))
    if len(columns) < width // 4:
        print("\n  too few columns carry habitual water to fit an edge; "
              "either the mask\n  already clears it everywhere, or the record "
              "is too short to say.")
        return None
    columns, rows = np.array(columns), np.array(rows)
    # Robust fit: drop the columns that sit far off the line -- a pier, a
    # groyne, a parked truck. A PERFECT fit is the case that breaks this
    # naively: the residuals are then floating-point dust, two standard
    # deviations of dust excludes everything, and the next polyfit is handed
    # an empty vector. So a round that would empty the set ends the loop.
    keep = np.ones(len(columns), bool)
    slope, intercept = np.polyfit(columns, rows, 1)
    for _ in range(5):
        resid = rows - (slope * columns + intercept)
        spread = float(np.std(resid[keep]))
        if spread < 1e-6:
            break
        nearer = np.abs(resid) < 2.0 * spread
        if nearer.sum() < max(4, len(columns) // 10):
            break
        keep = nearer
        slope, intercept = np.polyfit(columns[keep], rows[keep], 1)
    scatter = float(np.std(rows[keep] - (slope * columns[keep] + intercept)))
    print(f"\n  WHERE WATER ACTUALLY REACHES, over the whole record:")
    print(f"    y = {intercept:.3f} {slope:+.3f} x   "
          f"(scatter {scatter:.3f} of the frame, {keep.sum()} of "
          f"{len(columns)} columns)")
    print(f"    left edge {intercept:.3f}, right edge {intercept + slope:.3f}")
    left, right = intercept + margin, intercept + slope + margin
    print(f"\n  the same line with a {margin:.2f} margin, ready to paste:")
    print(f'    "{slug}": {{')
    print(f'        "keep": [[(0.0, {left:.3f}), (1.0, {right:.3f}), '
          f'(1.0, 1.0), (0.0, 1.0)]],')
    print("    }")
    print("  Compare it with what is declared. A declared edge well landward "
          "of this one\n  is land being thrown away; one seaward of it is "
          "water being registered.")
    return {"slope": float(slope), "intercept": float(intercept),
            "scatter": scatter}


def audit_mask(paths, dates, mask, spec, warm=25, downsample=4,
               often=0.25, slug="<camera>"):
    """Does the declared mask ever contain water, anywhere in the record?

    Measurement, not correction: this reports what the mask caught and where.
    It changes nothing and proposes no new mask.
    """
    print("\n" + "=" * 74)
    print("MASK AUDIT  -- the mask was drawn on one frame; this is every frame")
    print("=" * 74)
    from PIL import Image
    frequency, frames, shape = water_frequency(paths, mask.shape, warm,
                                               downsample)
    if frequency is None:
        print("  no readable frames to audit against")
        return None
    small = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                       .resize((shape[1], shape[0]))) > 127
    inside = small.sum()
    if not inside:
        print("  the mask is empty at audit resolution")
        return None

    print(f"  {len(frames)} frames, {downsample}x downsampled, "
          f"'water' = R-B below {warm} levels")
    print(f"  (foam, wet sand and a fog whiteout all read as water here; "
          f"that is\n   deliberate -- it is the direction that fails safe)")

    often_wet = (frequency >= often) & small
    share = often_wet.sum() / inside
    print(f"\n  pixels inside the mask that looked like water in >= "
          f"{often:.0%} of frames:")
    print(f"    {often_wet.sum():,} of {inside:,}  ({share:.2%} of the mask)")

    if often_wet.any():
        reach = np.where(often_wet.any(axis=1))[0][-1] / shape[0]
        edge = np.where(small.any(axis=1))[0][0] / shape[0]
        print(f"    the mask's seaward edge is row {edge:.3f}; water reaches "
              f"row {reach:.3f},\n    so the intrusion runs "
              f"{reach - edge:.3f} of the frame height INTO the land.")

    # The per-frame share is what names the dates worth looking at.
    shares = np.array([(wet & small).sum() / inside for wet in frames])
    print(f"\n  per-frame share of the mask that looked like water:")
    print(f"    median {np.median(shares):.1%}   "
          f"90th pct {np.percentile(shares, 90):.1%}   "
          f"worst {shares.max():.1%}")
    if shares.max() < 0.01:
        print("    no frame in the record put water inside this mask.")
    else:
        order = np.argsort(shares)[::-1]
        print("    the six worst frames (check these by eye before believing "
              "the mask):")
        for index in order[:6]:
            print(f"      {dates[index]:%Y-%m-%d}  {shares[index]:>6.1%}")
    settled = np.median(shares)
    if settled > 0.35:
        print("\n  WARNING: the TYPICAL frame has over a third of the mask "
              "reading as water.\n  That is a mask fault, not a tide -- "
              "redraw it landward.")
    elif shares.max() > 0.6:
        print("\n  NOTE: some frames read mostly wet while the median is "
              "low. That is the\n  signature of fog and low sun rather than "
              "of water in the mask; the dates\n  above say which.")
    edge = measured_edge(frequency, often=often, slug=spec.get("slug", slug))
    return {"often_wet_share": float(share),
            "median_frame_share": float(settled),
            "worst_frame_share": float(shares.max()),
            "measured_edge": edge}


def draw_mask_preview(path, mask, out_path):
    """The frame with the water dimmed, so the mask can be checked by eye."""
    from PIL import Image
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        if rgb.size != (mask.shape[1], mask.shape[0]):
            rgb = rgb.resize((mask.shape[1], mask.shape[0]))
        array = np.asarray(rgb).astype(np.float64)
    dimmed = array * 0.25
    dimmed[..., 0] += 40 * (~mask)
    blended = np.where(mask[..., None], array, dimmed)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)).save(
        out_path, quality=88)
    return out_path


if __name__ == "__main__":
    main()
