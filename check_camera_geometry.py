"""Has the camera stayed geometrically fixed across its whole record?

A pan, zoom, remount or housing shift breaks the pixel-to-ground mapping. A
shoreline trend computed across a camera move looks exactly like erosion, and
`bbox_area_max` pooled across one is comparing two different scales. This
answers whether the archive is ONE geometric record or several, and does not
attempt to correct anything.

TWO PASSES, because they need different things.

    python check_camera_geometry.py --camera <slug> --detections

`--detections` is read-only and needs no network. It runs on the rip table
already on disk and answers the questions that table can answer: did the
DETECTOR change (model name or version), and does the distribution of detection
pixels shift or exceed a frame size that changed. It cannot see a camera move
that left the detector alone, so a clean result here is not a clean bill.

    python check_camera_geometry.py --camera <slug> --sample

`--sample` is the real test and it needs imagery. The stills are NOT on disk:
pull_rip_detection.py --coverage enumerates element timestamps and downloads
nothing, by design, so 650,000 JPEGs never move. This mode downloads one frame
per week at a consistent hour -- about 165 frames over Walton's record, a few
hundred MB -- caches them under data/geometry/<slug>/, and registers each
against a reference by phase correlation on static patches.

    python check_camera_geometry.py --camera <slug> --sample --every 3
    python check_camera_geometry.py --camera <slug> --roi roofline:120,80,96,96

Method is phase correlation (FFT cross-power spectrum, parabolic sub-pixel
peak refinement) on fixed regions of interest, not ORB or SIFT. Descriptor
matching answers "where did this corner go" per frame and needs outlier
rejection to survive fog, glare and a gull on the railing; phase correlation on
a patch answers "how far did this patch move" directly, carries its own
confidence in the peak sharpness, and needs no feature library. Frames whose
peak is not sharp are reported and dropped rather than averaged in.

THE KEY TEST IS AGREEMENT BETWEEN FEATURES. One patch moving is a sign that
blew over. Every patch moving by the same vector on the same date is the
camera. Both are reported, and the epoch split uses the agreeing signal.

That test is also how the patches are CHOSEN. Picking four patches by how they
look in one frame and then trusting them failed at Walton three times running:
sky, then a banner, then fog. Sharpness in a single frame cannot tell a
roofline from a bright bank of cloud. So the run now picks a dozen candidates,
tracks all of them, and keeps the largest group whose offset series agree with
each other over the whole record. Rigid patches agree because they are bolted
to the same building; fog, surf and glare do not agree with anything, including
each other. The patches are selected by the evidence rather than by my reading
of one JPEG, and the ones that were thrown out are drawn on the preview in
grey so the choice is inspectable.

Outputs, per camera, under data/geometry/:
  geometry_<slug>.csv   one row per frame per feature: dx, dy, confidence
  geometry_<slug>.png   per-feature offset against date, plus the step trace
and to stdout: candidate discontinuity dates and the epochs they imply, with
how many stills fall in each.
"""

import env  # noqa: F401  -- loads .env into os.environ

import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

OUT_DIR = "data/geometry"
RIP_DIR = "data/rip_detection"

# One frame a week is enough to find a remount; --every changes it.
DEFAULT_EVERY_DAYS = 7
# Local solar noon is the steadiest light and the least shadow movement. UTC
# here because every timestamp in this project is UTC; Walton's noon is 20:00.
DEFAULT_HOUR_UTC = 20
# A frame within this many hours of the target is close enough to stand in.
HOUR_TOLERANCE = 3

# Patch size for tracking.
ROI_SIZE = 128
# A patch of N pixels can only resolve a displacement of N/2. Past that the two
# patches hold non-overlapping ground and the correlation peak is spurious --
# and spurious in a DIFFERENT direction in every patch, because each one is
# matching different accidental content. That is not a subtle failure: it is
# exactly "no two patches agree with each other", which is also the signature
# of patches on water, so the two are indistinguishable from the disagreement
# alone. The whole frame is registered first, at this downsample, and the
# patches then measure only what is left over. At downsample 2 on a 2560 px
# frame the coarse pass resolves +/- 640 px, which is any camera move short of
# a repoint.
COARSE_DOWNSAMPLE = 2
# Grey-level standard deviation below which a frame carries no scene at all:
# a black frame, a whiteout, a dropped feed. See phase_shift.
BLANK_STD = 1.0
# Confidence reported for two identical arrays, which happens at least once per
# run when the reference is compared with itself. Finite, so it cannot poison a
# median or a comparison downstream.
PERFECT_MATCH = 1e6
# How far the camera is allowed to have moved, in original pixels. This is a
# PRIOR, not a measurement, and it is here because the alternative is worse: a
# correlation surface spans the whole frame, so a spurious peak 600 px away
# competes on equal terms with the true one 5 px away, and at Walton three
# "independent" frame-to-frame steps landed within 1.5 px of each other at
# ~592 px, which is not what independent errors do. A camera bolted to a
# building does not move a quarter of its frame between two weekly stills.
# Raise it with --max-shift if a genuine repoint is being looked for; the run
# reports how often the best peak anywhere was outside this window, so the
# prior can be checked rather than trusted.
MAX_SHIFT_PX = 150
# THE RESOLUTION MUST NOT BE A FUNCTION OF THE FLAG. --max-shift decides how
# much of the correlation surface may win. If the peak is really on the scene,
# changing the window moves nothing but the count of frames overruled. If there
# is NO dominant peak, the best position inside the box is found near the box,
# and every number downstream -- the two-route gap, the per-frame error, the
# resolution, the step threshold and the epochs built on it -- scales with the
# window instead of with the imagery.
#
# Walton is the case that forced this. At --max-shift 150 the derived
# resolution came out 145 px, 0.97x the window, and the largest offset sat on
# the wall at 149.88. Re-run at 450 to test the prior -- the overruled count
# duly fell from 55% to 6%, which looks like the prior being fixed -- and the
# resolution came out 376 px, 0.84x the window, with the largest offset at
# 629.56 px: 98.9% of the way to the corner of a 450 px box. Two windows, both
# saturated, the answer tracking the flag. It produced 43 epochs and every one
# of them was the search box.
#
# So: when the smallest move a record claims to resolve is this share of the
# window it was given, the window is bounding the answer and there is no
# measurement here to report.
WANDER_SHARE = 0.5
# A frame registers on the structure it contains, and fog removes structure
# without removing the frame. Walton's contact sheet settles this: every frame
# that failed to register against its neighbour is a whiteout, and the clear
# ones are not. Those frames are not WRONG, they are EMPTY -- and a correlator
# handed an empty frame still returns a number, with a peak that clears the
# confidence floor often enough to poison the record.
#
# Clarity is the RMS gradient of the frame, in grey levels per pixel: how much
# edge there is to align on. The threshold is a fraction of the RECORD'S OWN
# median rather than an absolute, because it has to travel to cameras with
# different optics, exposure and scenes.
MIN_CLARITY = 0.45
# How many candidates to propose and track before the agreement test picks the
# keepers. More candidates is cheap -- the frames are already loaded, and a
# 128x128 FFT is nothing next to decoding a 2560x1920 JPEG -- and it is the
# only defence against a frame where the sharpest-looking thing is weather.
CANDIDATES = 12
# Vertical bands the candidates are spread across. Patches from one band are
# four samples of one thing and agree with each other whatever the camera did.
BANDS = 4
# Fewer than this many mutually-agreeing features is not a rigid scene, it is a
# coincidence. Two patches agreeing could both be on the same drifting fogbank.
MIN_CLUSTER = 3
# How far the two columns of the quarter table may part before the record is
# said to have drifted away from its anchor: the share of frames registering
# against the PREVIOUS frame, minus the share registering against the
# REFERENCE. Measurement error hits both columns equally, so a persistent gap
# is not error.
DRIFT_GAP = 0.15
# Below this share, the patch pass is not adding an independent measurement:
# it is the coarse pass plus a small residual, and its agreement with the
# coarse pass is arithmetic rather than evidence.
INDEPENDENT_SHARE = 0.25
# Features are taken from the top of the frame by default: land, roofline and
# structure live there, and the beach and water -- which move for real reasons
# -- live below. --land-fraction moves the line.
LAND_FRACTION = 0.55

# A shift is only a discontinuity if it is bigger than this and it persists.
STEP_PX = 3.0
# How close two patches must track each other before they count as being on the
# same rigid body. This used to BE --step-px, and that was wrong: the two
# numbers answer different questions. "How big must a shift be before I call it
# a move" is a threshold on the signal, and it is right to push it below the
# record's own resolution and let the derived floor take over. "How close must
# two measurements of the same camera be before I believe they are measuring
# one thing" is a tolerance on the ERROR, and pushing it down does not make the
# test stricter -- it makes it unpassable, because nothing on a 2.5 MP frame
# agrees to half a pixel across three hundred dates. Sharing one knob meant a
# --step-px of 0.5, chosen so the coarse route's measured resolution would
# bind, silently demanded half-pixel agreement from twelve patches and then
# reported "no patches agree" as if the record had said it.
AGREE_PX = 3.0
PERSIST = 3          # samples on each side that must agree
# Peak-to-sidelobe ratio: how far the correlation peak stands above the rest
# of its own surface, in standard deviations. NOT the raw peak height, which
# was the first version and does not survive being asked about two array sizes:
# a delta in an N-pixel surface has height 1 whatever N is, but a real partial
# match has height proportional to the correlated SHARE, so the same 0.05 that
# dropped fog in a 128px patch dropped 29 of 30 perfectly good whole frames.
# PSR has no such problem, because the quantity it is compared against is the
# surface's own noise. Pure noise peaks at about sqrt(2 ln N) sidelobes -- 4.4
# for a 128px patch, 5.0 for a 2.5 MP frame -- so a single threshold above that
# means "better than chance" at any size.
MIN_CONFIDENCE = 8.0
OVERRULE_SHARE = 0.2   # frames forced inside the search window before it is suspect
# Temporal spread, in grey levels, below which a patch is not scene at all.
# Real imagery weeks apart never repeats exactly: sun angle, haze and JPEG
# noise alone put the median absolute deviation well above one level. A region
# that is byte-identical across months is composited after capture -- a
# timestamp bar, a logo, a letterbox -- and it does NOT move when the camera
# does, so registering against it reports a rock-steady camera no matter what
# the camera did. It is the most dangerous thing in the frame for this test
# and the most attractive to a picker that rewards stillness.
OVERLAY_SPREAD = 0.5
# An absolute floor is not enough on its own. Walton's banner is a solid strip
# with white text, and JPEG ringing around those letter edges varies by a few
# grey levels between frames -- comfortably above any fixed floor, while the
# strip itself is still welded to the sensor. So the real test is RELATIVE: a
# row of composited pixels varies far less than the scene rows around it,
# whatever the absolute numbers are. A row under this share of the frame's
# typical row-to-row variation is not scene.
# The test is also ANCHORED TO THE EDGES and flooded inward. A banner is a
# contiguous strip touching the top or the bottom of the frame, never a stripe
# through the middle of the scene, so flooding from the edge lets the ratio be
# generous without any risk of eating real content: the flood stops at the
# first row that behaves like scene.
COMPOSITED_ROW_RATIO = 0.6
# ...and never more than this share of the frame from either edge, so a
# pathological record cannot exclude everything.
MAX_EDGE_FLOOD = 0.25
# And how much of a patch may be composited before the patch is unusable.
# Essentially none: an overlay's edges are the sharpest content in the frame,
# so even a sliver of banner inside a patch dominates its correlation peak and
# pins it to zero. There is always clean scene elsewhere to use instead.
MAX_DEAD_FRACTION = 0.02


# ---------------------------------------------------------------------------
# Read-only pass: what the detection table already knows
# ---------------------------------------------------------------------------

def detection_report(slug):
    """Epochs visible in the rip table, with no network and no imagery.

    Three things in that table are geometry-adjacent:

      * model_name / model_version -- a detector change is a discontinuity in
        every pixel metric even when the camera never moved. Confidence and
        box size are not comparable across a retrain.
      * the extent of bbox_x / bbox_y -- boxes are in native image pixels, so
        the largest coordinate ever seen is a floor on the frame width and
        height. A resolution change shows up here and nowhere else in the data
        already downloaded.
      * the centre of the bbox distribution -- a remount moves where in the
        frame rips get found. This is the weakest of the three: surf moves too.
    """
    path = os.path.join(RIP_DIR, f"rip_{slug}.csv")
    if not os.path.exists(path):
        sys.exit(f"{path} not found — pull the camera first:\n"
                 f"  python pull_rip_detection.py --camera {slug} --pull "
                 f"--match-observations")
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True,
                                        errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    print(f"{len(frame):,} detection rows, "
          f"{frame['timestamp'].min():%Y-%m-%d} to "
          f"{frame['timestamp'].max():%Y-%m-%d}\n")

    print("DETECTOR VERSION")
    version_cols = [c for c in ("model_name", "model_version")
                    if c in frame.columns]
    if not version_cols:
        print("  the table carries no model columns")
    else:
        frame["_model"] = frame[version_cols].astype(str).agg(" / ".join, axis=1)
        runs = (frame["_model"] != frame["_model"].shift()).cumsum()
        spans = frame.groupby(runs).agg(
            model=("_model", "first"), rows=("_model", "size"),
            first=("timestamp", "min"), last=("timestamp", "max"))
        for _, row in spans.iterrows():
            print(f"  {row['first']:%Y-%m-%d} to {row['last']:%Y-%m-%d}  "
                  f"{row['rows']:>7,} rows  {row['model']}")
        if len(spans) > 1:
            print(f"  {len(spans)} runs — a version change is a break in "
                  "score_max and bbox_area_max even if the camera never moved")
        else:
            print("  one version throughout")

    print("\nFRAME EXTENT  (largest detection pixel seen per month; a floor on")
    print("the frame size, not the frame size — a quiet month reads small)")
    if {"bbox_x", "bbox_y"}.issubset(frame.columns):
        monthly = frame.set_index("timestamp").resample("MS").agg(
            max_x=("bbox_x", "max"), max_y=("bbox_y", "max"),
            med_x=("bbox_x", "median"), med_y=("bbox_y", "median"),
            n=("bbox_x", "size"))
        monthly = monthly[monthly["n"] > 0]
        for stamp, row in monthly.iterrows():
            print(f"  {stamp:%Y-%m}  n={int(row['n']):>5}  "
                  f"max ({row['max_x']:.0f}, {row['max_y']:.0f})  "
                  f"median ({row['med_x']:.0f}, {row['med_y']:.0f})")
        ceiling_x, ceiling_y = monthly["max_x"].max(), monthly["max_y"].max()
        print(f"\n  the record never exceeds ({ceiling_x:.0f}, {ceiling_y:.0f})")
        # A resolution change is the one thing this pass can call outright: a
        # month whose maximum exceeds an earlier ceiling by a wide margin means
        # the frame got bigger, which nothing but a camera change does.
        running = monthly["max_x"].cummax()
        jumps = monthly.index[(monthly["max_x"] > running.shift() * 1.2)
                              & running.shift().notna()]
        if len(jumps):
            print("  frame WIDENED in: " +
                  ", ".join(f"{d:%Y-%m}" for d in jumps))
        else:
            print("  no month exceeds the running maximum by >20%, so no "
                  "resolution change is visible here")
    else:
        print("  the table carries no bbox columns")

    print("\nWHAT THIS PASS CANNOT SEE")
    print("  A pan, a small zoom or a remount that left the detector and the")
    print("  resolution alone. Detections are in the surf zone and the surf")
    print("  moves on its own, so their distribution is not a fixed reference.")
    print("  Only --sample, on the imagery, answers the question you asked.")
    return frame


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def require_imaging():
    """Fail before the downloads, not after them.

    The first run of --sample fetched eight frames and then died inside
    propose_rois on a missing Pillow. Checking at the top costs nothing and
    means a missing dependency is a one-line message rather than a traceback
    that arrives after the network work is already done.
    """
    try:
        import PIL  # noqa: F401
    except ImportError:
        sys.exit("--sample needs Pillow to read the frames.\n"
                 "  pip install -r requirements.txt\n"
                 "(paste that line ALONE — a trailing shell comment is passed "
                 "to pip as an argument\n on a zsh without "
                 "interactive_comments, and pip rejects it.)")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("  matplotlib is not installed; the run will write the CSV and "
              "the text report\n  but no plot.  pip install matplotlib")


def frame_clarity(paths, dates, downsample=4):
    """RMS gradient per frame: how much structure there is to register on.

    Not brightness and not variance. A whiteout can be bright and a foggy frame
    can have a perfectly ordinary spread of grey levels while carrying no EDGE
    at all, and edges are the only thing phase correlation can use.
    """
    rows = []
    for path, date in zip(paths, dates):
        image = load_gray(path, downsample=downsample)
        if image is None:
            continue
        gy, gx = np.gradient(image)
        rows.append({"date": date,
                     "clarity": float(np.sqrt(np.mean(gy ** 2 + gx ** 2)))})
    if not rows:
        return pd.Series(dtype=float)
    return pd.DataFrame(rows).set_index("date")["clarity"].sort_index()


def frame_sizes(paths, dates):
    """Distinct pixel dimensions in the sampled record, with dates.

    This should have been the first thing checked and was not. If the camera
    was replaced or reconfigured, the stills change size, and then NOTHING in
    this module means what it says: a patch at (1160, 276) is looking at
    different ground in the two halves of the record, every feature disagrees
    with every other, and the disagreement is the answer rather than an
    obstacle to it. The header carries the size, so this costs no decoding.

    A FILE THAT WILL NOT OPEN IS NOT A FRAME SIZE. Counting the unreadable
    ones as a group of their own made a single corrupt download at Corolla
    print as "2 DIFFERENT FRAME SIZES" under the epoch-boundary banner, which
    is the loudest thing this module says and was saying something false. They
    are counted and named on their own line instead, and returned separately
    so the caller can drop them rather than treat them as an epoch.

    Returns (groups, unreadable): a dict of (width, height) -> list of
    indices, and the list of indices whose file did not open.
    """
    from PIL import Image
    groups, unreadable = {}, []
    for index, path in enumerate(paths):
        try:
            with Image.open(path) as img:
                size = img.size
        except Exception:
            unreadable.append(index)
            continue
        groups.setdefault(size, []).append(index)
    if len(groups) > 1:
        print("\n" + "!" * 74)
        print(f"THE RECORD HOLDS {len(groups)} DIFFERENT FRAME SIZES.")
        print("That is a hard epoch boundary on its own: the same pixel is")
        print("different ground on either side of it, so no patch can track")
        print("across it and no pixel metric pools across it.")
    for size, indices in sorted(groups.items(),
                                key=lambda kv: -len(kv[1])):
        name = f"{size[0]}x{size[1]}"
        span = (f"{dates[indices[0]]:%Y-%m-%d} to "
                f"{dates[indices[-1]]:%Y-%m-%d}")
        print(f"  {name:>12}  {len(indices):>4} frames  {span}")
    if len(groups) > 1:
        print("!" * 74)
    if unreadable:
        span = (f"{dates[unreadable[0]]:%Y-%m-%d} to "
                f"{dates[unreadable[-1]]:%Y-%m-%d}")
        print(f"  {len(unreadable)} files did not open ({span}) -- corrupt or "
              f"truncated downloads,")
        print("  not a frame size and not an epoch boundary; they are dropped.")
    return groups, unreadable


def load_gray(path, downsample=1):
    """Greyscale float array, or None if the file is not a readable image."""
    from PIL import Image
    try:
        with Image.open(path) as img:
            img = img.convert("L")
            if downsample > 1:
                img = img.reduce(downsample)
            return np.asarray(img, dtype=np.float64)
    except Exception:
        return None


def _parabolic(values, index):
    """Sub-pixel offset of a peak from three samples, wrapping at the ends."""
    n = len(values)
    before, here, after = (values[(index - 1) % n], values[index],
                           values[(index + 1) % n])
    denominator = before - 2 * here + after
    if denominator == 0:
        return 0.0
    return 0.5 * (before - after) / denominator


def phase_shift(reference, image, max_shift=None):
    """(dy, dx, confidence, outside): how far `image` has MOVED from `reference`.

    `max_shift` limits the search to displacements within that many pixels of
    zero. `outside` is True when the tallest peak ANYWHERE was outside that
    window -- the count of those is how the prior gets audited against the
    record rather than assumed.

    Sign convention matters here and is easy to get backwards. The correlation
    peak gives the shift that maps `image` back onto `reference`, which is the
    negative of the displacement; it is negated before returning, so a feature
    that has slid 5 px right reads dx = +5 on the plot rather than -5.

    Confidence is the PEAK-TO-SIDELOBE RATIO: how far the peak stands above
    the rest of its own correlation surface, in standard deviations of that
    surface, with a small neighbourhood of the peak excluded so a broad peak
    does not inflate its own noise estimate. A sharp peak means one
    unambiguous alignment; fog, night and heavy rain flatten it, and those
    frames are dropped rather than averaged in.

    It is a ratio rather than the raw peak height so that one threshold works
    for a 128 px patch and for a 2.5 MP frame. See MIN_CONFIDENCE.
    """
    if reference.shape != image.shape:
        return None
    # A BLANK FRAME HAS NO ALIGNMENT TO FIND, and must not be allowed to claim
    # one. Walton's record contains several fully black frames -- outages, or
    # night where the sampled hour drifted -- and a constant array flattens to
    # zero once its mean is removed, so the cross-power surface is identically
    # zero: argmax picks index 0, the shift reads (0, 0), and the sidelobe
    # spread is zero, which the perfect-match branch below scores as INFINITE
    # confidence. A black frame therefore beat every real frame in the
    # reference selection, and the whole archive then registered against it at
    # exactly 0.00 px with perfect confidence -- a flawless-looking result that
    # was measuring nothing at all.
    if float(reference.std()) < BLANK_STD or float(image.std()) < BLANK_STD:
        return None
    window = np.outer(np.hanning(reference.shape[0]),
                      np.hanning(reference.shape[1]))
    first = np.fft.fft2((reference - reference.mean()) * window)
    second = np.fft.fft2((image - image.mean()) * window)
    cross = first * np.conj(second)
    magnitude = np.abs(cross)
    magnitude[magnitude < 1e-12] = 1e-12
    surface = np.fft.ifft2(cross / magnitude).real

    rows, cols = surface.shape
    free = np.unravel_index(int(np.argmax(surface)), surface.shape)
    peak, outside = free, False
    if max_shift:
        # The surface is circular, so the allowed band wraps: rows 0..R and
        # rows n-R..n-1 are both "within R of no movement".
        reach_y, reach_x = min(int(max_shift), rows // 2), min(int(max_shift),
                                                              cols // 2)
        window = np.zeros(surface.shape, dtype=bool)
        window[np.ix_(np.r_[0:reach_y + 1, rows - reach_y:rows],
                      np.r_[0:reach_x + 1, cols - reach_x:cols])] = True
        if not window[free]:
            outside = True
            near = np.where(window, surface, -np.inf)
            peak = np.unravel_index(int(np.argmax(near)), surface.shape)
    # Mask a few pixels around the peak before measuring the background, so a
    # peak two or three pixels wide is not counted as part of its own noise.
    background = np.ones(surface.shape, dtype=bool)
    guard = 3
    rows_, cols_ = surface.shape
    background[max(0, peak[0] - guard): peak[0] + guard + 1,
               max(0, peak[1] - guard): peak[1] + guard + 1] = False
    sidelobes = surface[background]
    spread = float(sidelobes.std())
    if spread <= 1e-12:
        # A surface with no sidelobe variation at all is a perfect delta: the
        # two arrays are identical, which happens at least once per run when
        # the reference is measured against itself. The first version scored it
        # ZERO, by dividing by the spread it had just found to be nothing,
        # which dropped every reference frame as if it were fog.
        #
        # Reported as a large FINITE number rather than infinity: an infinity
        # in the confidence column propagates through every median, mean and
        # comparison downstream, and "perfect" does not need to be unbounded to
        # beat every real measurement.
        confidence = PERFECT_MATCH
    else:
        confidence = float(surface[peak] - sidelobes.mean()) / spread

    dy = peak[0] + _parabolic(surface[:, peak[1]], peak[0])
    dx = peak[1] + _parabolic(surface[peak[0], :], peak[1])
    # The correlation surface is circular: a shift of -2 appears at n-2.
    if dy > rows / 2:
        dy -= rows
    if dx > cols / 2:
        dx -= cols
    return float(-dy), float(-dx), confidence, outside


def coarse_shifts(paths, dates, pick, downsample=COARSE_DOWNSAMPLE,
                  min_confidence=MIN_CONFIDENCE, max_shift=MAX_SHIFT_PX):
    """Whole-frame displacement per date, against the reference frame.

    Two reasons this runs before the patches rather than instead of them.

    RANGE. A 128 px patch cannot see a 200 px move; it reports something, and
    what it reports is noise. The full frame can, so the coarse pass puts every
    patch back on its own ground before the patch is asked anything.

    ROBUSTNESS. Phase correlation is not confused by a frame that is mostly
    water. Uncorrelated content -- surf, cloud, glare -- contributes a flat
    pedestal to the cross-power surface rather than a competing peak, so the
    only thing that can produce a peak is content common to both frames, which
    is the rigid scene however small a share of the frame it occupies. A
    scene-wide correlation is therefore the RIGHT primary measurement here and
    the patches are the cross-check, not the other way round.

    Returns a frame indexed by date with dy, dx, offset and confidence, in
    ORIGINAL pixels.
    """
    reference = load_gray(paths[pick], downsample=downsample)
    if reference is None:
        return pd.DataFrame()
    rows, mismatched = [], 0
    for path, date in zip(paths, dates):
        image = load_gray(path, downsample=downsample)
        if image is None:
            continue
        if image.shape != reference.shape:
            # A DIFFERENT FRAME SIZE IS DIFFERENT GROUND, and cropping to the
            # common area does not make it comparable -- it makes it look
            # comparable. Walton's five 1280x720 frames are 16:9 against a 4:3
            # record, so the top-left 1280x720 of the big frame is not the same
            # scene as the small frame at all; registering them returns a
            # number with no meaning, and that number then votes in the median,
            # the two-route gap and the noise floor derived from it.
            #
            # A frame size change is a hard epoch boundary. Frames on the other
            # side of one are not measured against this reference at all; they
            # are counted and reported, and they belong to their own run.
            mismatched += 1
            continue
        base, moved = reference, image
        got = phase_shift(base, moved,
                          max_shift=max_shift / downsample if max_shift
                          else None)
        if got is None:
            continue
        dy, dx, confidence, outside = got
        rows.append({"date": date, "dy": dy * downsample, "dx": dx * downsample,
                     "confidence": confidence, "outside": outside})
    if not rows:
        return pd.DataFrame()
    if mismatched:
        print(f"  {mismatched} frames are a DIFFERENT SIZE from the reference "
              f"and were not\n  registered against it: a frame size change is "
              f"its own epoch, and cropping\n  to a common area would compare "
              f"different ground. Run them separately.")
    frame = pd.DataFrame(rows).set_index("date").sort_index()
    strayed = int(frame["outside"].sum())
    if strayed:
        print(f"  {strayed}/{len(frame)} frames had their tallest peak beyond "
              f"{max_shift:.0f} px and were\n  re-measured inside it — that "
              "count IS the evidence for the limit; if it is\n  most of the "
              "record the limit is wrong, not the record")
        # AND WHEN IT IS A QUARTER OF THE RECORD, SAY SO LOUDLY. A frame whose
        # true peak lies outside the window is not measured; it is assigned the
        # best position inside a box it does not belong in, and that number
        # then votes in every median and every step. Printing the count and
        # doing nothing with it let Walton's winter stretch report epochs from
        # a record where one frame in four had been overruled.
        share = strayed / len(frame)
        if share > OVERRULE_SHARE:
            print(f"\n  WARNING: that is {share:.0%} of the frames. Each one "
                  "was given the best")
            print("  position inside the window rather than its own, and those "
                  "numbers vote in")
            print("  every median and step below. The test is to re-run with "
                  f"--max-shift {max_shift * 3:.0f}:")
            print("  if the count collapses and the offsets change, the limit "
                  "was wrong; if the")
            print("  count holds, those frames do not register at all and are "
                  "unmeasured either")
            print("  way. Until then treat the epochs below as provisional.")
    weak = int((frame["confidence"] < min_confidence).sum())
    if weak:
        print(f"  {weak}/{len(frame)} frames below confidence "
              f"{min_confidence} (fog, night, rain) — dropped")
        frame = frame[frame["confidence"] >= min_confidence]
    frame["offset"] = np.hypot(frame["dx"], frame["dy"])
    return frame


def contact_sheet(paths, dates, out_path, columns=12, thumb_width=200,
                  flagged=()):
    """Every sampled frame as one labelled grid.

    Numbers said the early record does not register and the late record does.
    Numbers cannot say WHY, and the reasons are all things a person sees at a
    glance and a correlator cannot: a different field of view, a lens change, a
    camera that returns to a different preset, a dirty dome, night frames from
    a season when local noon is not the sampled hour.

    Dates whose registration was not confident are marked, so the question
    "what do the failing frames have in common" can be answered by looking at
    them rather than by inference.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    label_height = 16
    flagged = {pd.Timestamp(d) for d in flagged}
    cells = []
    for path, date in zip(paths, dates):
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                height = max(1, int(image.height * thumb_width / image.width))
                cells.append((image.resize((thumb_width, height)), date))
        except Exception:
            continue
    if not cells:
        return None
    thumb_height = max(cell.height for cell, _ in cells)
    rows = (len(cells) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width,
                              rows * (thumb_height + label_height)),
                      (18, 33, 43))
    draw = ImageDraw.Draw(sheet)
    for index, (cell, date) in enumerate(cells):
        column, row = index % columns, index // columns
        x = column * thumb_width
        y = row * (thumb_height + label_height)
        sheet.paste(cell, (x, y))
        bad = pd.Timestamp(date) in flagged
        draw.text((x + 3, y + thumb_height + 2),
                  f"{date:%Y-%m-%d}" + ("  X" if bad else ""),
                  fill=(228, 87, 46) if bad else (220, 226, 230))
        if bad:
            draw.rectangle([x, y, x + thumb_width - 1, y + cell.height - 1],
                           outline=(228, 87, 46), width=3)
    sheet.save(out_path, quality=85)
    return out_path


def registration_quality(direct, sequential, dates, min_confidence=MIN_CONFIDENCE):
    """Share of dates that registered confidently, by half-year.

    A record that fails everywhere is a broken method. A record that fails for
    two years and then works is telling you something happened to the camera,
    and this says WHEN to look.
    """
    index = pd.DatetimeIndex(dates)
    table = pd.DataFrame(index=index)
    table["direct"] = direct["confidence"].reindex(index) \
        if direct is not None and not direct.empty else np.nan
    table["sequential"] = sequential["confidence"].reindex(index) \
        if sequential is not None and not sequential.empty else np.nan
    # tz_localize(None) first: to_period drops the timezone and warns, and a
    # warning in the middle of a report reads like a problem with the data.
    table["period"] = index.tz_localize(None).to_period("Q")
    rows = []
    for period, group in table.groupby("period"):
        rows.append({
            "period": str(period),
            "frames": len(group),
            "direct_ok": float((group["direct"] >= min_confidence).mean()),
            "seq_ok": float((group["sequential"] >= min_confidence).mean())})
    return pd.DataFrame(rows)


def sequential_shifts(paths, dates, downsample=COARSE_DOWNSAMPLE,
                      min_confidence=MIN_CONFIDENCE, max_shift=MAX_SHIFT_PX):
    """Each frame against the one BEFORE it, rather than against a reference.

    This is the check the direct-to-reference pass cannot perform on itself.
    Registering January against a reference in June asks the correlator to
    match two frames that differ in sun angle, season, haze and tide as well as
    in camera position, and it can lock onto the wrong thing while still
    producing a confident peak. Consecutive weeks are the easiest possible
    pair, so a sequential pass is the most reliable measurement available -- and
    its CUMULATIVE SUM should reproduce the direct measurement exactly, because
    both are describing the same camera.

    Where the two agree, a single translation describes the record and the
    steps are real. Where they diverge, the direct pass is registering
    something other than the scene, and no number it produced can be trusted.
    That is a test with a right answer, which nothing before it in this module
    has been.
    """
    rows, crossings = [], 0
    previous, previous_date = None, None
    for path, date in zip(paths, dates):
        image = load_gray(path, downsample=downsample)
        if image is None:
            continue
        if previous is None:
            rows.append({"date": date, "previous": pd.NaT, "dy": 0.0,
                         "dx": 0.0, "confidence": PERFECT_MATCH})
            previous, previous_date = image, date
            continue
        if image.shape != previous.shape:
            # Same rule as the direct route: a step ACROSS a frame size change
            # is not a small number, it is an unmeasurable one. The chain
            # continues from the new size, and the pair that spanned the change
            # is left out rather than carried as anything.
            crossings += 1
            previous, previous_date = image, date
            continue
        base, moved = previous, image
        got = phase_shift(base, moved,
                          max_shift=max_shift / downsample if max_shift
                          else None)
        # The pair this step spans is recorded, so the cross-check can compare
        # it against the SAME pair measured the other way.
        spanned = previous_date
        previous, previous_date = image, date
        if got is None:
            continue
        dy, dx, confidence, _ = got
        rows.append({"date": date, "previous": spanned,
                     "dy": dy * downsample, "dx": dx * downsample,
                     "confidence": confidence})
    if crossings:
        print(f"  {crossings} consecutive pairs span a frame size change and "
              f"were NOT measured;\n  a step across one is unmeasurable, not "
              f"small.")
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("date").sort_index()
    # A weak link breaks the chain rather than one row of it: everything after
    # an unmeasured step is offset by whatever that step was. Carrying zero is
    # the least-wrong choice and the count is reported so it can be judged.
    weak = int((frame["confidence"] < min_confidence).sum())
    if weak:
        print(f"  {weak}/{len(frame)} consecutive pairs below confidence "
              f"{min_confidence}; their step is carried as zero")
        frame.loc[frame["confidence"] < min_confidence, ["dy", "dx"]] = 0.0
    frame["step"] = np.hypot(frame["dx"], frame["dy"])
    frame["cum_dy"] = frame["dy"].cumsum()
    frame["cum_dx"] = frame["dx"].cumsum()
    return frame


def still_threshold(values, bins=64):
    """Otsu's split of `values` into a still group and a moving one.

    Returns the cut that maximises the between-group variance. On a single
    cluster that is near the middle; on two clusters it lands in the gap.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 4:
        return float(values.max()) if values.size else 0.0
    counts, edges = np.histogram(values, bins=bins)
    weight = np.cumsum(counts)
    total = weight[-1]
    if total == 0 or weight[-1] == counts[0]:
        return float(np.median(values))
    centres = (edges[:-1] + edges[1:]) / 2.0
    running = np.cumsum(counts * centres)
    below, above = weight, total - weight
    usable = (below > 0) & (above > 0)
    if not usable.any():
        return float(np.median(values))
    mean_below = np.where(below > 0, running / np.maximum(below, 1), 0.0)
    mean_above = np.where(above > 0,
                          (running[-1] - running) / np.maximum(above, 1), 0.0)
    between = below * above * (mean_below - mean_above) ** 2
    between = np.where(usable, between, -np.inf)
    return float(edges[int(np.argmax(between)) + 1])


def propose_rois(paths, size=ROI_SIZE, count=CANDIDATES, bands=BANDS,
                 land_fraction=LAND_FRACTION, sample=16,
                 top_margin=0, bottom_margin=0):
    """Pick patches that are sharp in space and still in time.

    Sharp in space so there is something to correlate: a patch of flat sky
    aligns equally well everywhere and returns noise. Still in time so the
    patch is structure rather than weather.

    Two things the first version got wrong, both visible in its first real
    run at Walton, which put three of four patches at y=0 -- the sky.

      * It scored per PIXEL. The argmax of a pixel-wise gradient is a single
        bright speck, and a patch centred on one has nothing else in it to
        align. Both terms are now averaged over the patch footprint, so the
        question is "does this 128 px box contain structure", not "is there
        one sharp pixel here".
      * It scored structure DIVIDED BY spread. A ratio rewards being still as
        much as being sharp, and nothing is stiller than sky: a featureless
        patch with a spread near zero beat a roofline whose shadows move.
        Stability is now a constraint -- reject the patches that move most --
        and sharpness alone is the objective among those that pass.
    """
    chosen_paths = paths[:: max(1, len(paths) // sample)][:sample]
    stack = [g for g in (load_gray(p, downsample=2) for p in chosen_paths)
             if g is not None]
    if len(stack) < 3:
        return []
    shape = min((g.shape for g in stack), key=lambda s: (s[0], s[1]))
    stack = np.stack([g[:shape[0], :shape[1]] for g in stack])

    # LEVEL THE ILLUMINATION BEFORE ASKING WHAT MOVED. Fog, sun and exposure
    # change the whole frame at once -- a gain and an offset -- and that global
    # swing is far larger than the difference between a roofline that never
    # moves and surf that never stops. Measured raw at Walton, land and water
    # came out 15 and 27: a ratio of 1.76, which no threshold can split, and
    # every candidate duly reported variation ~20 whether it sat on a house or
    # on the open sea. Four of twelve landed on water.
    #
    # Dividing each frame by its OWN WHOLE-FRAME median and spread removes the
    # gain and the offset while leaving the water's own churn intact. On the
    # same fixture that separation goes from 1.76x to 17x. The scale has to come
    # from the whole frame: taken per patch it would divide out exactly the
    # variance being looked for.
    middle = np.median(stack, axis=(1, 2), keepdims=True)
    swing = np.median(np.abs(stack - middle), axis=(1, 2), keepdims=True)
    levelled = (stack - middle) / np.maximum(swing, 1e-6)

    median = np.median(levelled, axis=0)
    spread = np.median(np.abs(levelled - median), axis=0)
    # The banner and blank-region tests below are about ABSOLUTE grey levels
    # ("this never changes by even half a level"), so they keep the raw scale.
    raw_median = np.median(stack, axis=0)
    raw_spread = np.median(np.abs(stack - raw_median), axis=0)
    gy, gx = np.gradient(median)
    structure = np.hypot(gy, gx)

    # Average both over the patch footprint. `size` is in full-resolution
    # pixels and the stack is at downsample 2, so the window is size // 2.
    from scipy.ndimage import uniform_filter
    window = max(3, size // 2)
    patch_structure = uniform_filter(structure, window)
    patch_spread = uniform_filter(spread, window)
    # Reported in grey levels beside the levelled figure, because "this patch
    # varies by 0.08 levels" is how an overlay announces itself and the
    # levelled number cannot say it.
    patch_raw_spread = uniform_filter(raw_spread, window)

    row_spread = np.median(raw_spread, axis=1)
    typical = float(np.median(row_spread))

    half = size // 4  # working at downsample=2
    usable = np.zeros(shape, dtype=bool)
    usable[half: shape[0] - half, half: shape[1] - half] = True
    # Land only. Everything below the line is beach and water, which move.
    usable[int(shape[0] * land_fraction):, :] = False
    # Explicit margins, in full-resolution pixels, for when the automatic
    # banner test does not fire. It has failed on Walton repeatedly, and a
    # camera's overlay is a fixed, known property of that camera: being able
    # to say "ignore the top 100 px" beats another round of inference.
    # The margin has to exclude the whole PATCH, not its centre. --top-margin
    # 100 put a centre at row 100 and therefore a patch spanning rows 36-164,
    # which is still half on the banner. Add the patch half-height.
    if top_margin:
        usable[: (top_margin + size // 2) // 2, :] = False
    if bottom_margin:
        usable[-((bottom_margin + size // 2) // 2):, :] = False
    if not usable.any():
        print("  the margins leave nothing searchable")
        return []

    # Why the banner test did or did not fire, in numbers. Printed near the
    # edges because that is where composited strips live.
    edge = min(60, len(row_spread) // 4)
    head = ", ".join(f"{v:.1f}" for v in row_spread[:edge:max(1, edge // 6)])
    print(f"  row variation, top edge inward: {head}  "
          f"(frame typical {typical:.1f})")

    # Stability as a constraint, then maximise sharpness among what passes.
    # Sky passes the stability test and then loses on sharpness, which is the
    # behaviour that was missing.
    #
    # THE THRESHOLD CANNOT BE "THE CALMER HALF". A median split asserts that
    # half the searchable frame is still, and at Walton it is not: the water
    # reaches most of the way up the usable rows, so the calmer half still
    # contained surf, and surf has the sharpest edges in the frame -- every
    # whitecap is an edge -- so it won the sharpness contest and four patches
    # landed on the sea. The levelling made land and water separable; it was
    # this fixed 50% that let the water back in.
    #
    # Let the frame say where the split is. Otsu's threshold maximises the
    # variance BETWEEN the two groups, so when the patches really do fall into
    # a still class and a moving class it lands in the gap between them,
    # wherever that gap sits -- and on a frame with no water, where the values
    # are one cluster, it cuts near the middle, which is what the old rule did.
    # So it is never worse than the median split and is much better when it
    # matters.
    calm = patch_spread <= still_threshold(patch_spread[usable])

    # ...but not TOO still. See OVERLAY_SPREAD. The test has to be the SHARE OF
    # DEAD PIXELS inside the patch, not the patch's mean variation: a patch
    # straddling the bottom edge of a banner averages the banner's zero against
    # live scene below and passes comfortably, while half its body still cannot
    # move. That is exactly the patch the picker reaches for, because the
    # banner's edge is the sharpest thing in the frame.
    # Rows that barely change compared with the rest of the frame. A banner
    # spans the full width, so this is a per-ROW question: a scene row somewhere
    # in the frame always has weather, shadow or surf moving through it.
    quiet = row_spread < COMPOSITED_ROW_RATIO * typical
    limit = int(len(row_spread) * MAX_EDGE_FLOOD)
    composited = np.zeros(len(row_spread), dtype=bool)
    for row in range(min(limit, len(quiet))):          # flood down from the top
        if not quiet[row]:
            break
        composited[row] = True
    for row in range(min(limit, len(quiet))):          # and up from the bottom
        if not quiet[-1 - row]:
            break
        composited[-1 - row] = True
    dead = (raw_spread <= OVERLAY_SPREAD) | composited[:, None]
    dead_fraction = uniform_filter(dead.astype(float), window)
    if composited.any():
        hit = np.flatnonzero(composited)
        print(f"  rows {hit[0] * 2}-{hit[-1] * 2 + 1} at the frame edge vary "
              f"{row_spread[composited].max() / max(typical, 1e-9):.0%} as much "
              "as the scene — composited banner, excluded")
    alive = dead_fraction < MAX_DEAD_FRACTION
    dead_share = float((usable & ~alive).sum()) / max(int(usable.sum()), 1)
    if dead_share > 0.005:
        print(f"  {dead_share:.0%} of the searchable area is composited rather "
              "than scene — never varies between frames, excluded")
    score = np.where(usable & calm & alive, patch_structure, 0.0)
    if not score.any():
        print("  nothing is both calm and alive; dropping the calm test")
        score = np.where(usable & alive, patch_structure, 0.0)

    # Spread the candidates across horizontal bands rather than taking the best
    # N overall. Greedy selection with local suppression walks along one row: at
    # Walton it put all four patches at y=0, spanning the full width but
    # sampling a single band, and four samples of one band agree with each
    # other whatever the camera did. Bands force the patches apart vertically,
    # which is what makes their agreement mean something.
    rows = np.flatnonzero(usable.any(axis=1))
    bands = max(1, min(bands, count))
    edges = np.linspace(rows[0], rows[-1] + 1, bands + 1).astype(int)
    per_band = int(np.ceil(count / bands))

    rois = []
    working = score.copy()
    for index in range(bands):
        band_top, band_bottom = edges[index], edges[index + 1]
        for _ in range(per_band):
            if len(rois) >= count:
                break
            banded = np.zeros_like(working)
            banded[band_top:band_bottom, :] = working[band_top:band_bottom, :]
            # A band with nothing usable left in it -- all water, all overlay,
            # or already suppressed -- falls back to the best patch anywhere,
            # so a frame whose structure really is all in one place still gets
            # its candidates. Bands express a preference for vertical spread,
            # not a mandate to accept rubbish: a band of nothing but fog has a
            # best patch, and it is still fog. Below a quarter of the best
            # score anywhere, take the best remaining patch instead.
            floor = 0.25 * float(working.max()) if working.any() else 0.0
            source = banded if banded.max(initial=0.0) >= floor and banded.any() \
                else working
            if not source.any():
                break
            cy, cx = np.unravel_index(int(np.argmax(source)), source.shape)
            rois.append({"name": f"f{len(rois) + 1}",
                         "x": int(cx * 2 - size // 2),
                         "y": int(cy * 2 - size // 2),
                         "w": size, "h": size,
                         # Carried so the run can print why each patch was
                         # chosen. A patch reported with near-zero variation is
                         # an overlay whatever else the output says.
                         "structure": float(patch_structure[cy, cx]),
                         "spread": float(patch_raw_spread[cy, cx]),
                         "motion": float(patch_spread[cy, cx])})
            # Suppress a generous neighbourhood so the patches are not all one
            # corner of one roof.
            y0, y1 = max(0, cy - half * 3), cy + half * 3
            x0, x1 = max(0, cx - half * 3), cx + half * 3
            working[y0:y1, x0:x1] = 0
    return rois


def survey_rois(paths, size=ROI_SIZE, top_margin=0, bottom_margin=0,
                limit=400):
    """Tile the WHOLE frame, and let the agreement test say where land is.

    propose_rois answers "where does this frame look sharp and still", which at
    Walton has now picked sky, a translucent banner and fog in turn. The survey
    asks nothing about appearance at all: it lays a grid over every row the
    margins allow, tracks every cell, and reports which cells agree with each
    other. Whatever is bolted down shows up as the region whose cells move
    together, wherever it turns out to be -- including below the land fraction,
    which is a guess about framing that Walton may simply not obey.

    It is the expensive mode and the honest one. The frames are decoded once
    for all cells, so the cost over --candidates is one small FFT per cell per
    frame, not one more pass over the JPEGs.
    """
    first = load_gray(paths[0])
    if first is None:
        return []
    height, width = first.shape
    top = top_margin
    bottom = height - bottom_margin
    rois = []
    for y in range(top, bottom - size + 1, size):
        for x in range(0, width - size + 1, size):
            rois.append({"name": f"c{len(rois) + 1}", "x": x, "y": y,
                         "w": size, "h": size})
    if len(rois) > limit:
        # Thin evenly rather than truncating, so the survey still spans the
        # whole frame instead of stopping part way down it.
        step = int(np.ceil(len(rois) / limit))
        rois = rois[::step]
        for index, roi in enumerate(rois, 1):
            roi["name"] = f"c{index}"
    print(f"  {len(rois)} grid cells of {size}px over rows "
          f"{top}-{bottom} of {height}")
    return rois


def draw_rois(path, rois, out_path, rejected=()):
    """Write the reference frame with the chosen patches boxed and labelled.

    "Eyeball these before trusting the result" is not actionable when the
    result is four x/y triples and the frame is 2560 px wide. A picture is.

    `rejected` patches are drawn thin and grey. Seeing which candidates the
    agreement test threw out is how you tell a working run from a lucky one:
    grey boxes on sky and surf with orange boxes on buildings is the picture
    this method is supposed to produce.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except Exception:
        return None
    draw = ImageDraw.Draw(image)
    width = max(2, image.width // 500)
    for roi in rejected:
        box = [roi["x"], roi["y"], roi["x"] + roi["w"], roi["y"] + roi["h"]]
        draw.rectangle(box, outline=(124, 143, 153), width=max(1, width // 2))
    for roi in rois:
        box = [roi["x"], roi["y"], roi["x"] + roi["w"], roi["y"] + roi["h"]]
        draw.rectangle(box, outline=(228, 87, 46), width=width)
        draw.text((roi["x"] + 4, max(0, roi["y"] - 14 * width)), roi["name"],
                  fill=(228, 87, 46))
    image.save(out_path, quality=88)
    return out_path


def crop(image, roi):
    patch = image[roi["y"]: roi["y"] + roi["h"], roi["x"]: roi["x"] + roi["w"]]
    return patch if patch.shape == (roi["h"], roi["w"]) else None


def best_reference(paths, probes=8, downsample=8):
    """Index of the frame the REST OF THE RECORD can register against.

    Two wrong answers came before this one. The first frame is wrong: Walton's
    record opens on its foggiest day, and everything is measured relative to
    the reference, so a soft reference degrades every measurement in the run.
    The sharpest frame is also wrong, and worse, because sharpness was scored
    as mean gradient magnitude and NOTHING maximises that like noise -- a rainy
    frame, a high-ISO dusk frame, or a corrupted one beats any real scene, and
    then nothing in the record registers against the anchor.

    The property actually wanted is not sharpness at all. The reference is
    whichever frame the most other frames can be matched to, so that is what is
    measured: correlate every frame against a handful of probes spread through
    the record and keep the one with the best median peak-to-sidelobe ratio.
    Noise scores near chance against everything and loses; a clear frame from a
    period the camera held still wins, which is exactly the anchor wanted.

    Done at downsample 8 so the whole record fits in memory at once and the
    comparison is a few thousand small FFTs rather than a third pass over the
    JPEGs.
    """
    images = [load_gray(path, downsample=downsample) for path in paths]
    usable = [index for index, image in enumerate(images)
              if image is not None and float(image.std()) >= BLANK_STD]
    blank = sum(1 for image in images
                if image is not None and float(image.std()) < BLANK_STD)
    if blank:
        print(f"  {blank} frames carry no scene (black, whiteout or dropped "
              "feed) and cannot be\n  a reference or be registered")
    if len(usable) < 3:
        return 0
    shape = min((images[i].shape for i in usable), key=lambda s: (s[0], s[1]))
    step = max(1, len(usable) // probes)
    sample = usable[::step][:probes]
    scores = []
    for index in usable:
        base = images[index][:shape[0], :shape[1]]
        peaks = []
        for other in sample:
            if other == index:
                continue
            got = phase_shift(base, images[other][:shape[0], :shape[1]])
            if got is not None:
                peaks.append(got[2])  # (dy, dx, confidence, outside)
        scores.append((float(np.median(peaks)) if peaks else 0.0, index))
    return max(scores)[1]


def track(paths, dates, rois, pick=0, min_confidence=MIN_CONFIDENCE,
          coarse=None):
    """dx/dy per frame per feature, against the reference frame at `pick`.

    `coarse` is the whole-frame displacement per date. Each patch is cut from
    the moved position rather than from the reference position, so what the
    correlation measures is the RESIDUAL -- which is inside a patch's range by
    construction -- and the residual is added back to the coarse shift to give
    the displacement. Without this a move larger than half a patch is
    unmeasurable and reads as patches that disagree.
    """
    reference, reference_date = load_gray(paths[pick]), dates[pick]
    rows, unreadable, offscreen = [], 0, 0
    for path, date in zip(paths, dates):
        image = load_gray(path)
        if image is None:
            unreadable += 1
            continue
        if reference is None:
            reference, reference_date = image, date
        shift_y, shift_x = (0.0, 0.0)
        if coarse is not None and date in coarse.index:
            shift_y = float(coarse.at[date, "dy"])
            shift_x = float(coarse.at[date, "dx"])
        for roi in rois:
            base = crop(reference, roi)
            moved = dict(roi, x=int(round(roi["x"] + shift_x)),
                         y=int(round(roi["y"] + shift_y)))
            patch = crop(image, moved)
            if base is None or patch is None:
                # The patch walked off the edge of the moved frame. That is a
                # fact about this date, not a bad measurement to substitute
                # for, so it is counted and skipped.
                offscreen += 1
                continue
            shifted = phase_shift(base, patch)
            if shifted is None:
                continue
            dy, dx, confidence, _ = shifted
            rows.append({"date": date, "feature": roi["name"],
                         "dx": dx + shift_x, "dy": dy + shift_y,
                         "confidence": confidence,
                         "frame": os.path.basename(path)})
    if offscreen:
        print(f"  {offscreen} patch positions fell outside the moved frame, "
              "skipped")
    if unreadable:
        print(f"  {unreadable} files were not readable images, skipped")
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, reference_date
    weak = int((frame["confidence"] < min_confidence).sum())
    if weak:
        print(f"  {weak}/{len(frame)} measurements below confidence "
              f"{min_confidence} (fog, night, rain) — dropped")
        frame = frame[frame["confidence"] >= min_confidence]
    frame["offset"] = np.hypot(frame["dx"], frame["dy"])
    return frame, reference_date


# ---------------------------------------------------------------------------
# Steps and epochs
# ---------------------------------------------------------------------------

def agreeing_signal(frame):
    """Per-date median dx/dy across features.

    The median is the point: one feature moving is that feature, and taking
    the median of four makes a lone sign that blew over invisible while a
    camera move, which shifts all four together, comes through undiminished.
    """
    if frame.empty:
        return pd.DataFrame()
    # Disagreement is the distance from the consensus VECTOR, not the range of
    # offset magnitudes. Two patches that slid 5 px in opposite directions have
    # identical magnitudes and a range of zero, and the earlier version called
    # that perfect agreement.
    rows = []
    for date, group in frame.groupby("date"):
        dx, dy = float(group["dx"].median()), float(group["dy"].median())
        apart = np.hypot(group["dx"] - dx, group["dy"] - dy)
        rows.append({
            "date": date, "dx": dx, "dy": dy,
            "offset": float(np.hypot(dx, dy)),
            # TYPICAL disagreement, which is what the gate is named after and
            # what it should test. The maximum is reported beside it but does
            # not gate: over a survey's worth of cells the worst cell on one
            # foggy date is always several pixels out, and gating on it throws
            # away a record that eleven cells agree about.
            "spread": float(np.median(apart)),
            "spread_max": float(apart.max()),
            "features": int(group["feature"].nunique())})
    return pd.DataFrame(rows).set_index("date").sort_index()


def _cliques(nodes, adjacency):
    """Every maximal set of mutually adjacent nodes (Bron-Kerbosch)."""
    found = []

    def expand(clique, candidates, excluded):
        if not candidates and not excluded:
            found.append(list(clique))
            return
        for node in list(candidates):
            expand(clique + [node],
                   candidates & adjacency[node],
                   excluded & adjacency[node])
            candidates = candidates - {node}
            excluded = excluded | {node}

    expand([], set(nodes), set())
    return found


def _greedy_group(names, adjacency, seeds=40):
    """Largest mutually-agreeing set, found greedily.

    Bron-Kerbosch is exact and fine for a dozen candidates; on a few hundred
    survey cells its worst case is not. Growing a group from each of the
    best-connected nodes, always adding the node that keeps the most options
    open, finds the rigid region in practice -- and the rigid region is a
    dense block, which is the easy case for greedy.
    """
    order = sorted(names, key=lambda n: -len(adjacency[n]))[:seeds]
    best = []
    for seed in order:
        group, pool = [seed], set(adjacency[seed])
        while pool:
            node = max(pool, key=lambda n: len(adjacency[n] & pool))
            group.append(node)
            pool &= adjacency[node]
        if len(group) > len(best):
            best = group
    return [best]


def agreeing_features(frame, tolerance=STEP_PX, minimum=MIN_CLUSTER):
    """The largest group of features whose offset series agree with each other.

    This is what replaces my judgement about which part of the frame is
    rigid. Two patches on the same building report the same dx/dy on every
    date, because there is only one camera; two patches on fog report
    unrelated numbers, because there is nothing holding them together. So the
    rigid scene is recoverable as the largest mutually-consistent subset, with
    no appeal to what the frame looks like.

    Pairwise distance is the MEDIAN over dates of the distance between the two
    features' offset vectors -- median so that a handful of foggy frames one
    patch happened to survive cannot separate two features that track together
    the rest of the time.

    Returns (kept, rejected, distance), where distance maps each feature to its
    median distance from the kept group, so a run can say how far out the
    rejects were rather than only that they were dropped.
    """
    names = sorted(frame["feature"].unique())
    if len(names) < 2:
        return names, [], {}
    wide = frame.pivot_table(index="date", columns="feature",
                             values=["dx", "dy"])

    def distance_between(a, b):
        dxa, dya = wide[("dx", a)], wide[("dy", a)]
        dxb, dyb = wide[("dx", b)], wide[("dy", b)]
        both = dxa.notna() & dxb.notna()
        # Two features that never survive the confidence cut on the same date
        # have no evidence of agreeing. Absence of disagreement is not
        # agreement, so they are not linked.
        if int(both.sum()) < minimum:
            return float("inf")
        return float(np.median(np.hypot(dxa[both] - dxb[both],
                                        dya[both] - dyb[both])))

    pairs = {}
    for index, a in enumerate(names):
        for b in names[index + 1:]:
            pairs[(a, b)] = pairs[(b, a)] = distance_between(a, b)
    adjacency = {a: {b for b in names
                     if b != a and pairs[(a, b)] <= tolerance} for a in names}

    def tightness(clique):
        inner = [pairs[(a, b)] for i, a in enumerate(clique)
                 for b in clique[i + 1:]]
        return float(np.mean(inner)) if inner else 0.0

    # Biggest clique wins; ties go to the tightest, so a spurious pair of
    # patches that happen to be 2.9 px apart never beats a real group.
    groups = (_cliques(names, adjacency) if len(names) <= 20
              else _greedy_group(names, adjacency))
    best = max(groups, key=lambda c: (len(c), -tightness(c)), default=[])
    best = sorted(best)

    def distance_to(group):
        return {n: float(np.median([pairs[(n, k)] for k in group if k != n]))
                for n in names if [k for k in group if k != n]}

    if len(best) < minimum:
        # Nothing agrees. Report each feature's median distance from ALL the
        # others, so the failure says how far apart the frame is rather than
        # only that no group formed.
        return [], names, distance_to(names)
    return best, [n for n in names if n not in best], distance_to(best)


def drifted_quarters(quality, gap=DRIFT_GAP):
    """Quarters that register against the previous frame but not the reference.

    Measurement error hits both columns of the quality table equally -- a frame
    too soft to match its neighbour is too soft to match the anchor. So a
    persistent gap between them is not error. It is the record having moved
    away from the anchor while consecutive frames stayed close to each other,
    which is drift, and the quarter where the columns part is when it started.

    Walton at --max-shift 112: 2024Q4 onward reads 36-76% against the reference
    while holding 76-96% against the previous frame, and the parting is right
    after the 2024-08-31 reference. A wider window recaptures those frames and
    the gap closes, which hides the finding rather than answering it.
    """
    return quality[(quality["seq_ok"] - quality["direct_ok"]) > gap]


def patch_independence(signal, coarse):
    """How much of the patch answer is the patches' own, not the coarse pass's.

    Each patch is cut from the coarse-corrected position and its residual is
    added back to the coarse shift, so the patch offsets CONTAIN the coarse
    offsets. When the residual is small beside the coarse shift, "both passes
    agree" is arithmetic rather than corroboration, and saying so matters more
    than the count of agreements.

    Returns the median distance between the two passes' offsets as a share of
    the offset the patches report, or NaN when they share no dates.
    """
    shared = signal.index.intersection(coarse.index)
    if not len(shared):
        return float("nan")
    apart = np.hypot(
        signal.loc[shared, "dx"].to_numpy() - coarse.loc[shared, "dx"].to_numpy(),
        signal.loc[shared, "dy"].to_numpy() - coarse.loc[shared, "dy"].to_numpy())
    return float(np.median(apart)) / max(
        float(signal.loc[shared, "offset"].median()), 1e-9)


def not_finer_than_the_record(given, derived):
    """A threshold or tolerance, never finer than what the record can measure.

    Both the step threshold and the agreement tolerance are asked for on the
    command line before anything has been measured, and both become nonsense
    below the record's own resolution -- a step smaller than the error is not a
    step, and patches cannot agree more closely than either can be read. So a
    given value is a FLOOR, and the measured resolution raises it. `derived` is
    NaN when the record could not measure itself, and then the given value is
    all there is.
    """
    return max(given, derived) if np.isfinite(derived) else given


def noise_floor(gap):
    """Turn the two-route disagreement into a per-measurement error.

    The cross-check compares a sequential step against the difference of two
    direct measurements. If each measurement carries an independent error e,
    the sequential step carries e, the difference of two direct measurements
    carries e*sqrt(2), and the gap between those two routes carries

        sqrt(e^2 + 2 e^2) = e * sqrt(3)

    BY CONSTRUCTION, on perfect data. An earlier version treated any gap over
    10 px as proof the passes were lying and withheld every verdict -- which,
    for a record with 14 px of per-measurement error, can never be satisfied.
    The gap is not a verdict on the method. It is the method reporting its own
    precision, and what it buys is a resolution limit rather than a refusal.
    """
    noise = gap / math.sqrt(3.0)
    return noise, 3.0 * noise


def find_steps(series, threshold=STEP_PX, persist=PERSIST):
    """Dates where the level shifts by `threshold` and stays shifted.

    A step is not a spike. Comparing the median of `persist` samples before a
    date against the median of `persist` after ignores a single bad frame and
    only fires on a change that the record keeps.

    THE STEP IS IN THE DISPLACEMENT VECTOR, NOT ITS LENGTH. Pass a frame with
    dx and dy and both are compared; pass a bare series and it is treated as
    one number. The distinction is not academic. Every offset here is measured
    from a reference frame, and the length |p(t) - p_ref| is blind in two
    directions at once: a camera that slides from 20 px left of the reference
    to 20 px right of it never changes its distance, so a real move reads as
    nothing; and a reference that sits away from where the camera usually
    points makes the record's ordinary position read as a large constant
    offset, so which frame was chosen as reference starts to decide where the
    steps appear. The vector has neither problem -- it is the same measurement
    in either case, only with the sign kept.
    """
    if isinstance(series, pd.DataFrame):
        values = series[["dx", "dy"]].to_numpy(dtype=float)
    else:
        values = series.to_numpy(dtype=float).reshape(-1, 1)
    dates = list(series.index)

    def level(block):
        return np.median(block, axis=0)

    def apart(one, other):
        return float(np.hypot(*(one - other))) if values.shape[1] == 2 \
            else float(abs(one[0] - other[0]))

    def size(point):
        return float(np.hypot(*point)) if values.shape[1] == 2 \
            else float(point[0])
    hits = []
    for index in range(persist, len(values) - persist + 1):
        if apart(level(values[index: index + persist]),
                 level(values[index - persist: index])) >= threshold:
            hits.append(index)
    if not hits:
        return []

    # The window test is deliberately blunt: it fires on every index whose
    # neighbourhood straddles the change, so one event produces a run of hits.
    # Group the run, then date the event by the largest single-sample jump
    # inside it, which is the first frame actually at the new level. Without
    # that refinement the reported date is the last frame BEFORE the move,
    # because a window centred there already sees the new level in its tail.
    runs, current = [], [hits[0]]
    for index in hits[1:]:
        if index - current[-1] <= persist:
            current.append(index)
        else:
            runs.append(current)
            current = [index]
    runs.append(current)

    steps = []
    for run in runs:
        candidates = [i for i in run if i > 0]
        if not candidates:
            continue
        index = max(candidates, key=lambda i: apart(values[i], values[i - 1]))
        before = level(values[max(0, index - persist): index])
        after = level(values[index: index + persist])
        # HOW MUCH RECORD STANDS BEHIND THIS STEP. The window test needs only
        # `persist` samples on each side, and with exactly that many the two
        # medians are each one sample from being a single reading. Walton's
        # 2026-01-18 step read 31 px measured across 44 frames and 11 px across
        # 8 -- same date, same camera, a third of the size, because narrowing
        # the window starved the medians rather than sharpening them. A step
        # standing on the minimum is a candidate, not a measurement, and has
        # to say so.
        steps.append({"date": dates[index], "jump": apart(after, before),
                      "before": size(before), "after": size(after),
                      "support": min(index, len(values) - index),
                      "thin": min(index, len(values) - index) < 2 * persist})
    return steps


def describe_epochs(steps, dates, counts=None):
    """Split the record at each step and say how much is in each piece."""
    boundaries = [s["date"] for s in steps]
    edges = [dates[0]] + boundaries + [dates[-1]]
    epochs = []
    for index in range(len(edges) - 1):
        start, end = edges[index], edges[index + 1]
        # Half-open except at the tail, so a frame belongs to exactly one epoch
        # and the frame on the step date belongs to the epoch it starts.
        last = index == len(edges) - 2
        inside = [d for d in dates
                  if start <= d <= end] if last else [d for d in dates
                                                      if start <= d < end]
        epochs.append({"start": start, "end": end, "frames": len(inside)})
    if counts is not None:
        for epoch in epochs:
            window = counts[(counts.index >= epoch["start"])
                            & (counts.index <= epoch["end"])]
            epoch["stills"] = int(window.sum())
    return epochs


# Everything the run wrote, so it can be listed -- and opened -- in one place
# at the end instead of scrolling back for paths.
WRITTEN = []
OPEN_IMAGES = False


def wrote(path):
    if path:
        WRITTEN.append(os.path.abspath(path))
    return path


def show(paths):
    """Open the written images with whatever this platform uses.

    `open` on macOS, `xdg-open` on Linux, `start` on Windows. A failure here is
    never worth failing a run over -- the paths are printed either way -- so it
    is reported and swallowed.
    """
    import shutil as _shutil
    import subprocess
    if sys.platform == "darwin":
        opener = ["open"]
    elif os.name == "nt":
        opener = ["cmd", "/c", "start", ""]
    else:
        opener = ["xdg-open"] if _shutil.which("xdg-open") else None
    images = [p for p in paths if p.lower().endswith((".png", ".jpg"))]
    if not opener or not images:
        return
    try:
        subprocess.run(opener + images, check=False)
    except Exception as error:
        print(f"  could not open the images ({error}); the paths are above")


def spark(series, width=86, height=13, marks=(), label=""):
    """The offset series as text, so the finding is readable without a viewer.

    A PNG has to be found, opened, and -- when the person reading it is not at
    the machine -- attached to a message. A chart made of characters is in the
    terminal output already, which means it is in the thing you paste. The PNG
    is still written and is still the better artefact; this is so the shape of
    the record cannot be missed for want of opening it.

    Time is on the x axis by DATE, not by sample, so a gap in the record reads
    as a gap rather than being closed up.
    """
    series = series.dropna()
    if len(series) < 2:
        return []
    start, end = series.index.min(), series.index.max()
    span = (end - start).total_seconds()
    if span <= 0:
        return []
    columns = [[] for _ in range(width)]
    for stamp, value in series.items():
        index = int((stamp - start).total_seconds() / span * (width - 1))
        columns[min(max(index, 0), width - 1)].append(float(value))
    # The MAXIMUM in each bucket, not the mean: this chart exists to make a
    # step or a spike visible, and averaging is exactly what hides one.
    heights = [max(bucket) if bucket else None for bucket in columns]

    # An empty column usually means "no sample was due here", not "the record
    # has a hole": at one frame a week over three years there are more columns
    # than samples, and marking every one of them as missing buries the gaps
    # that matter. A column is only a GAP if the nearest real measurement is
    # further away than the record's own sampling interval.
    gaps = series.index.to_series().diff().dropna()
    spacing = float(gaps.median().total_seconds()) if len(gaps) else span
    seconds_per_column = span / max(width - 1, 1)
    gap_columns = set()
    occupied = [index for index, bucket in enumerate(columns) if bucket]
    for index, value in enumerate(heights):
        if value is not None or not occupied:
            continue
        nearest = min(abs(index - other) for other in occupied)
        if nearest * seconds_per_column > 1.5 * spacing:
            gap_columns.add(index)
    finite = [v for v in heights if v is not None]
    if not finite:
        return []
    low, high = min(min(finite), 0.0), max(finite)
    if high - low < 1e-9:
        high = low + 1.0

    marked = set()
    for stamp in marks:
        index = int((pd.Timestamp(stamp) - start).total_seconds()
                    / span * (width - 1))
        marked.add(min(max(index, 0), width - 1))

    grid = [[" "] * width for _ in range(height)]
    for column, value in enumerate(heights):
        if value is None:
            if column in gap_columns:
                grid[height - 1][column] = "·"   # the record really is empty
            continue
        row = int(round((value - low) / (high - low) * (height - 1)))
        grid[height - 1 - row][column] = "#" if column in marked else "*"
    for column in marked:
        for row in range(height):
            if grid[row][column] == " ":
                grid[row][column] = "|"

    lines = [f"  {label}" if label else "  offset, px"]
    for index, row in enumerate(grid):
        value = high - (high - low) * index / (height - 1)
        lines.append(f"  {value:8.1f} |" + "".join(row))
    lines.append("  " + " " * 9 + "+" + "-" * width)
    lines.append(f"  {'':9} {start:%Y-%m-%d}" +
                 " " * max(1, width - 21) + f"{end:%Y-%m-%d}")
    legend = "           * measurement"
    if gap_columns:
        legend += "   · gap in the record"
    if marked:
        legend += "   | # candidate step"
    lines.append(legend)
    return lines


def sampling_days(dates, fallback=7.0):
    """Median spacing of a record, in days."""
    if len(dates) < 2:
        return fallback
    gaps = pd.Series(sorted(dates)).diff().dropna()
    spacing = gaps.median().total_seconds() / 86400.0 if len(gaps) else fallback
    return spacing if spacing > 0 else fallback


def reconcile(first, first_dates, second, second_dates,
              persist=PERSIST, window_days=None):
    """Set the two passes' step lists against each other, date by date.

    The whole-frame pass and the agreeing patches measure the same camera by
    different means, so a real move shows in both. Printing two lists of
    epochs and leaving the reader to diff them hides the only thing that
    matters: which steps BOTH passes found.

    A step the other pass missed is not automatically wrong. It matters
    whether that pass could see the date at all -- a run of frames it dropped
    for fog leaves a hole that no step detector can fire inside. So each
    unmatched step is reported as contradicted (the other pass had frames on
    both sides and found nothing) or as unmeasured (it did not).
    """
    # TWO STEPS ARE THE SAME EVENT WHEN THEY ARE A SAMPLE OR TWO APART, NOT A
    # FORTNIGHT. The tolerance was a fixed 14 days, which is two samples of a
    # weekly record and fourteen of a daily one. On Walton's daily December
    # window that paired a step on 2026-02-17 with one on 2026-02-05 and called
    # them the same move -- twelve days and a factor of four in size apart --
    # so the summary read "3 of 3 seen by both passes" when one of the three
    # was two different events. The tolerance has to follow the sampling.
    if window_days is None:
        window_days = max(2.0 * sampling_days(first_dates), 1.0)

    def near(date, dates, days):
        return [d for d in dates if abs((d - date).days) <= days]

    def covered(date, dates):
        before = [d for d in dates if d < date]
        after = [d for d in dates if d >= date]
        return len(before) >= persist and len(after) >= persist and \
            near(date, dates, window_days)

    rows = []
    for step in first:
        match = near(step["date"], [s["date"] for s in second], window_days)
        if match:
            twin = min(second, key=lambda s: abs((s["date"] - step["date"]).days))
            # Agreeing that something happened is not agreeing on what. Two
            # passes that report 113 px and 72 px for one move have not
            # measured the same thing, and saying "both" without the caveat
            # reads as corroboration it has not earned. Half again is already
            # a wide allowance for two measurements of one rigid displacement:
            # where the passes really do agree at Walton they land within a
            # few percent of each other (8.5 / 8.8 px, 4.8 / 4.8 px).
            big, small = sorted((step["jump"], twin["jump"]), reverse=True)
            verdict = "both" if big <= 1.5 * max(small, 1e-6) \
                else "both, but the sizes disagree"
            rows.append((verdict, step["date"], step["jump"], twin["jump"]))
        elif covered(step["date"], second_dates):
            rows.append(("whole frame only", step["date"], step["jump"], None))
        else:
            rows.append(("whole frame, patches blind",
                         step["date"], step["jump"], None))
    for step in second:
        if not near(step["date"], [s["date"] for s in first], window_days):
            if covered(step["date"], first_dates):
                rows.append(("patches only", step["date"], step["jump"], None))
            else:
                rows.append(("patches, whole frame blind",
                             step["date"], step["jump"], None))
    return sorted(rows, key=lambda row: row[1])


def report_record(steps, dates, slug, step_px, what, noise=None):
    """Print the epoch split, or its absence, for one registration pass.

    `noise` is the per-measurement error the cross-check implies. When it is
    given, `step_px` is this record's own resolution limit rather than a
    threshold anyone chose, and the null result has to be stated as such: not
    "the camera was stable" but "nothing moved by more than the smallest move
    this record can see".
    """
    if not steps:
        print(f"\nNo step larger than {step_px:.0f} px persists in {what}.")
        if noise is not None:
            print(f"That threshold is not a preference. It is 3x this "
                  f"record's own measurement")
            print(f"error of ~{noise:.0f} px, below which a step cannot be "
                  "told from the noise.")
            print("So the record is ONE epoch AT THIS RESOLUTION. A real move "
                  "smaller than")
            print(f"{step_px:.0f} px is neither found nor ruled out — it is "
                  "invisible to this")
            print("measurement, and denser sampling (--every 3) is what lowers "
                  "the floor.")
            return []
        print("The record reads as ONE geometric epoch. That is evidence of")
        print("stability, not proof: a move smaller than the step threshold,")
        print("or one inside a gap in the sampling, would not appear. Re-run")
        print("with --every 3 and a lower --step-px before committing to a")
        print("shoreline trend.")
        return []
    print(f"\n{len(steps)} candidate discontinuit"
          f"{'y' if len(steps) == 1 else 'ies'} in {what}:")
    for step in steps:
        thin = (f"   <- only {step['support']} frames on one side"
                if step.get("thin") else "")
        print(f"  {step['date']:%Y-%m-%d}  {step['before']:.1f} px -> "
              f"{step['after']:.1f} px  (jump {step['jump']:.1f}){thin}")
    if any(step.get("thin") for step in steps):
        print("\n  A step marked above rests on the fewest frames the test "
              "accepts, so its")
        print("  two medians are each about one reading. Its DATE is worth as "
              "much as any")
        print("  other; its SIZE is not, and narrowing the window further "
              "shrinks the")
        print("  support rather than sharpening it. Sample that stretch more "
              "densely")
        print("  (--every 1) before believing the magnitude.")
    epochs = describe_epochs(steps, list(dates), counts=coverage_counts(slug))
    print(f"\n{len(epochs)} epochs:")
    for index, epoch in enumerate(epochs, 1):
        stills = f", {epoch['stills']:,} stills" if "stills" in epoch else ""
        print(f"  {index}. {epoch['start']:%Y-%m-%d} to {epoch['end']:%Y-%m-%d}"
              f"  {epoch['frames']} sampled frames{stills}")
    print("\nNothing has been corrected or re-registered. Before pooling any")
    print("pixel metric across these dates, decide per epoch.")
    return epochs


def coverage_counts(slug):
    """Hourly image counts, so an epoch can be measured in stills not samples.

    THIS FILE HAS ANOTHER WRITER. pull_rip_detection appends to the same
    coverage CSV as it works through a camera, so a geometry run started while
    a pull is in flight can read it mid-append and hit a torn line. That is a
    decoration -- the stills column beside each epoch -- and it must not take
    down a run that has just spent twenty minutes correlating a thousand
    frames. A missing count prints as an epoch without stills; a crash here
    would lose the dates, the offsets and the cross-check with it.
    """
    paths = glob.glob(os.path.join(RIP_DIR, f"coverage_{slug}_hourly.csv"))
    if not paths:
        return None
    try:
        frame = pd.read_csv(paths[0])
        frame["hour"] = pd.to_datetime(frame["hour"], utc=True, errors="coerce")
        return frame.dropna(subset=["hour"]).set_index("hour")["images"]
    except Exception as problem:                      # torn read, or no rows yet
        print(f"  (still counts unavailable: {type(problem).__name__} reading "
              f"{os.path.basename(paths[0])} — a pull may be writing it)")
        return None


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_registration(direct, sequential, steps, path):
    """The plots the brief actually asked for: offset against date, and the
    frame-to-frame magnitude that catches a step."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(direct.index, direct["dx"], marker=".", lw=0.8, label="dx")
    axes[0].plot(direct.index, direct["dy"], marker=".", lw=0.8, label="dy")
    axes[0].set_ylabel("offset from reference, px")
    axes[0].legend(fontsize=8)
    axes[1].plot(direct.index, direct["offset"], color="#12212B", lw=1.0,
                 label="measured against the reference")
    if sequential is not None and not sequential.empty:
        walk = np.hypot(sequential["cum_dx"], sequential["cum_dy"])
        axes[1].plot(sequential.index, walk, color="#E4572E", lw=1.0, ls="--",
                     label="cumulative sum of frame-to-frame steps")
    axes[1].set_ylabel("displacement, px")
    axes[1].legend(fontsize=8)
    if sequential is not None and not sequential.empty:
        axes[2].plot(sequential.index, sequential["step"], color="#E4572E",
                     lw=0.9)
    axes[2].set_ylabel("frame-to-frame step, px")
    axes[2].set_xlabel("date")
    for step in steps:
        for axis in axes:
            axis.axvline(step["date"], color="#E4572E", ls=":", lw=1)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.axhline(0, color="#7C8F99", lw=0.6)
    axes[0].set_title("Whole-frame registration")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot(frame, signal, steps, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  matplotlib not installed; skipping {path}\n"
              "    pip install matplotlib")
        return None

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for name, group in frame.groupby("feature"):
        axes[0].plot(group["date"], group["dx"], marker=".", lw=0.8, label=name)
        axes[1].plot(group["date"], group["dy"], marker=".", lw=0.8, label=name)
    axes[0].set_ylabel("dx, pixels")
    axes[1].set_ylabel("dy, pixels")
    axes[0].legend(fontsize=8, ncol=4)
    axes[2].plot(signal.index, signal["offset"], color="#12212B", lw=1.0,
                 label="median offset across features")
    axes[2].fill_between(signal.index, 0, signal["spread"], color="#E4572E",
                         alpha=0.25, label="spread between features")
    axes[2].set_ylabel("pixels")
    axes[2].legend(fontsize=8)
    for step in steps:
        for axis in axes:
            axis.axvline(step["date"], color="#E4572E", ls="--", lw=1)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.axhline(0, color="#7C8F99", lw=0.6)
    axes[2].set_xlabel("date")
    axes[0].set_title("Offset from the reference frame, per static feature")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Sampling (the one mode that touches the network)
# ---------------------------------------------------------------------------

def cached_frames(camera):
    """The frames a previous --sample already downloaded. No network at all.

    Choosing patches is iterative -- that is the nature of it -- and every
    iteration was paying for 168 element queries and a fresh look at the
    inventory before it could re-read JPEGs that were already on disk. The
    imagery is the expensive part and it does not change, so once a camera has
    been sampled, every later run over the same frames should be local.

    Dates come from the filenames, which pull_rip_detection writes as
    <label>-YYYY-MM-DD-HHMMSSZ.jpg -- the element timestamp, not the download
    time, so the record is recoverable from the directory alone.
    """
    import re
    directories = sorted(d for d in os.listdir(OUT_DIR)
                         if os.path.isdir(os.path.join(OUT_DIR, d)))
    matches = [d for d in directories if d == camera] or \
              [d for d in directories if camera.lower() in d.lower()]
    if not matches:
        sys.exit(f"no cached frames for {camera!r} under {OUT_DIR}/.\n"
                 "Cached cameras: " + (", ".join(directories) or "none") +
                 "\nRun once without --cached to download them.")
    if len(matches) > 1:
        sys.exit(f"{camera!r} matches {', '.join(matches)} — be specific")
    slug = matches[0]
    save_dir = os.path.join(OUT_DIR, slug)

    stamp = re.compile(r"(\d{4}-\d{2}-\d{2})-(\d{2})(\d{2})(\d{2})Z")
    paths, dates = [], []
    unnamed = 0
    for name in sorted(os.listdir(save_dir)):
        found = stamp.search(name)
        if not found:
            unnamed += 1
            continue
        day, hour, minute, second = found.groups()
        paths.append(os.path.join(save_dir, name))
        dates.append(pd.Timestamp(f"{day}T{hour}:{minute}:{second}Z"))
    if unnamed:
        print(f"  {unnamed} files carry no timestamp in their name, skipped")
    if not paths:
        sys.exit(f"{save_dir}/ holds no timestamped frames")
    print(f"\nreusing {len(paths)} frames already in {save_dir}/ — no network")
    return slug, paths, dates


def sample_frames(camera, every_days, hour_utc, limit=None, workers=6):
    """One still per `every_days`, near `hour_utc`, cached under data/geometry."""
    import pull_rip_detection as rip

    assets = rip.load_assets()
    asset = rip.find_camera(assets, camera)
    label = rip.dig(asset, "data", "common", "label")
    slug = rip.slugify(label)
    service, _ = rip.find_stills_service(asset)
    first, last = rip.report_inventory(service)
    if first is None:
        sys.exit("the stills inventory gave no range; nothing to sample")

    save_dir = os.path.join(OUT_DIR, slug)
    os.makedirs(save_dir, exist_ok=True)

    targets, cursor = [], pd.Timestamp(first).ceil("D") + pd.Timedelta(hours=hour_utc)
    while cursor <= pd.Timestamp(last):
        targets.append(cursor)
        cursor += pd.Timedelta(days=every_days)
    if limit:
        targets = targets[:limit]
    print(f"\n{len(targets)} targets, one every {every_days}d at "
          f"{hour_utc:02d}:00 UTC")

    picked = []
    for index, target in enumerate(targets, 1):
        window_start = (target - pd.Timedelta(hours=HOUR_TOLERANCE)).to_pydatetime()
        window_end = (target + pd.Timedelta(hours=HOUR_TOLERANCE)).to_pydatetime()
        rows = rip.fetch_elements(service, window_start, window_end, quiet=True)
        if not rows:
            print(f"  {target:%Y-%m-%d}: no frame within "
                  f"{HOUR_TOLERANCE}h  ({index}/{len(targets)})")
            continue
        # The one nearest the target hour, so lighting is as steady as the
        # record allows.
        best = min(rows, key=lambda r: abs(r["timestamp"] - target))
        picked.append(best)
        print(f"  {target:%Y-%m-%d}: {best['timestamp']:%H:%M}  "
              f"({index}/{len(targets)})")

    if not picked:
        sys.exit("no frames could be sampled")
    print(f"\ndownloading {len(picked)} frames to {save_dir}/")
    got = rip.download(picked, save_dir, workers)
    paths = [row["path"] for row in got if row.get("path")]
    dates = [pd.Timestamp(row["timestamp"]) for row in got if row.get("path")]
    return slug, paths, dates


# ---------------------------------------------------------------------------

def parse_roi(text):
    name, _, rest = text.partition(":")
    x, y, w, h = (int(v) for v in rest.split(","))
    return {"name": name or "roi", "x": x, "y": y, "w": w, "h": h}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", required=True,
                    help="camera slug, label or unique substring")
    ap.add_argument("--detections", action="store_true",
                    help="read-only pass over the rip table; no network")
    ap.add_argument("--sample", action="store_true",
                    help="download a periodic still and register the record")
    ap.add_argument("--every", type=int, default=DEFAULT_EVERY_DAYS,
                    help=f"days between sampled frames (default {DEFAULT_EVERY_DAYS})")
    ap.add_argument("--hour", type=int, default=DEFAULT_HOUR_UTC,
                    help=f"UTC hour to sample at (default {DEFAULT_HOUR_UTC})")
    ap.add_argument("--limit", type=int, help="stop after N targets, to try it")
    ap.add_argument("--cached", action="store_true",
                    help="re-use the frames a previous --sample downloaded and "
                         "make no network calls. Use this for every re-run "
                         "that only changes --roi, --candidates or margins.")
    ap.add_argument("--since", help="ignore frames before this date "
                                    "(YYYY-MM-DD), to ask the question of one "
                                    "window rather than the whole archive")
    ap.add_argument("--until", help="ignore frames after this date "
                                    "(YYYY-MM-DD)")
    ap.add_argument("--roi", action="append", default=[],
                    help="name:x,y,w,h — repeatable; overrides auto-selection")
    ap.add_argument("--land-fraction", type=float, default=LAND_FRACTION,
                    help="auto-selection uses only the top this much of frame")
    ap.add_argument("--min-clarity", type=float, default=MIN_CLARITY,
                    help=f"drop frames whose RMS gradient is below this "
                         f"fraction of the record's median (default "
                         f"{MIN_CLARITY}). Fog removes the structure a "
                         "correlator needs without removing the frame. 0 keeps "
                         "everything.")
    ap.add_argument("--max-shift", type=float, default=MAX_SHIFT_PX,
                    help=f"how far the camera is allowed to have moved, in "
                         f"pixels (default {MAX_SHIFT_PX}). A PRIOR, not a "
                         "measurement: it stops a spurious peak halfway across "
                         "the frame outcompeting the true one. Raise it to "
                         "look for a genuine repoint; 0 removes it.")
    ap.add_argument("--contact-sheet", action="store_true",
                    help="write every sampled frame as one labelled grid. "
                         "Written automatically when the registration does "
                         "not agree with itself, because that is when the "
                         "answer is in the pictures rather than the numbers.")
    ap.add_argument("--open", dest="open_images", action="store_true",
                    help="open the plots and previews when the run finishes, "
                         "in whatever this machine uses for images")
    ap.add_argument("--no-coarse", action="store_true",
                    help="skip the whole-frame registration and measure with "
                         "patches alone. Only useful for reproducing the "
                         "failure it exists to fix: a move bigger than half a "
                         "patch is invisible to a patch.")
    ap.add_argument("--survey", action="store_true",
                    help="ignore the feature picker: tile the WHOLE frame and "
                         "report which cells agree with each other. Use this "
                         "when the candidates all disagree — it finds the "
                         "rigid region from the record instead of from how "
                         "the frame looks.")
    ap.add_argument("--roi-size", type=int, default=ROI_SIZE,
                    help=f"patch edge in pixels (default {ROI_SIZE}). A bigger "
                         "patch holds more structure and registers a soft "
                         "scene better.")
    ap.add_argument("--candidates", type=int, default=CANDIDATES,
                    help=f"patches to propose and track before the agreement "
                         f"test keeps the rigid ones (default {CANDIDATES})")
    ap.add_argument("--top-margin", type=int, default=0,
                    help="ignore this many pixels at the top of the frame. Use "
                         "it for a composited banner the automatic test misses "
                         "— Walton's is about 70 px, so --top-margin 100.")
    ap.add_argument("--bottom-margin", type=int, default=0,
                    help="ignore this many pixels at the bottom of the frame")
    ap.add_argument("--step-px", type=float, default=STEP_PX,
                    help=f"a shift this large counts as a step (default "
                         f"{STEP_PX}). A value below the resolution the record "
                         f"derives for itself is raised to it, and the run "
                         f"says so.")
    ap.add_argument("--agree-px", type=float, default=AGREE_PX,
                    help=f"how far apart two patches may track and still count "
                         f"as one rigid body (default {AGREE_PX}). This is a "
                         f"tolerance on measurement error, NOT a step size; "
                         f"lowering it does not find more moves, it refuses "
                         f"more records.")
    args = ap.parse_args()
    global OPEN_IMAGES
    OPEN_IMAGES = args.open_images
    import atexit
    atexit.register(finish)

    if not (args.detections or args.sample):
        ap.error("choose --detections (read-only) or --sample (downloads frames)")

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.detections:
        print("=" * 74)
        print(f"READ-ONLY PASS  {args.camera}")
        print("=" * 74)
        detection_report(args.camera)
        if not args.sample:
            return

    print("\n" + "=" * 74)
    print("IMAGERY PASS")
    print("=" * 74)
    require_imaging()
    if args.cached:
        slug, paths, dates = cached_frames(args.camera)
    else:
        slug, paths, dates = sample_frames(args.camera, args.every, args.hour,
                                           args.limit)
    order = np.argsort(dates)
    paths = [paths[i] for i in order]
    dates = [dates[i] for i in order]
    print(f"\n{len(paths)} frames on disk, "
          f"{dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}")

    # A DATE WINDOW IS NOT CHERRY-PICKING WHEN IT IS THE WINDOW THE ANSWER IS
    # FOR. Every measurement here carries the record's own noise, and that
    # noise is set by the WHOLE record: a stretch the correlator cannot follow
    # raises the floor for the part it can, so a question about one year gets
    # answered at three years' precision. Walton's rip detections span
    # 2025-08-31 to 2026-08-25 and nothing pooled across them depends on 2023.
    # Asking about that window alone is a narrower question, honestly answered,
    # rather than the same question answered badly.
    if args.since or args.until:
        window = [i for i, date in enumerate(dates)
                  if (not args.since
                      or date >= pd.Timestamp(args.since, tz="UTC"))
                  and (not args.until
                       or date <= pd.Timestamp(args.until, tz="UTC"))]
        if len(window) < 5:
            sys.exit(f"only {len(window)} frames fall inside "
                     f"{args.since or 'the start'} to {args.until or 'the end'}"
                     "; widen the window or sample more densely")
        paths = [paths[i] for i in window]
        dates = [dates[i] for i in window]
        print(f"  restricted to {len(paths)} frames, "
              f"{dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}")
        print("  every figure below describes THIS window and says nothing "
              "about the rest")
        print("  of the archive, including its noise floor, which the dropped "
              "frames no")
        print("  longer raise")
    frame_sizes(paths, dates)[0]

    # Fog is the obstacle, not geometry. Walton's contact sheet showed every
    # frame that failed to register against its neighbour is a whiteout, and
    # the quarters that fail are the foggy ones. A frame with no edges in it
    # cannot be registered, but it is still handed to the correlator, which
    # still returns a number -- so it has to be excluded before it votes.
    all_paths, all_dates, foggy = paths, dates, set()
    clarity = frame_clarity(paths, dates)
    if args.min_clarity and not clarity.empty:
        floor = args.min_clarity * float(clarity.median())
        foggy = set(clarity.index[clarity < floor])
        print(f"\nCLARITY  (RMS gradient; fog has no edges to align on)")
        print(f"  median {clarity.median():.2f}, "
              f"quartiles {clarity.quantile(0.25):.2f}-"
              f"{clarity.quantile(0.75):.2f}, "
              f"floor {floor:.2f} = {args.min_clarity:.0%} of the median")
        if foggy:
            keep = [i for i, date in enumerate(dates) if date not in foggy]
            print(f"  {len(foggy)}/{len(clarity)} frames are below it and are "
                  "NOT registered.")
            print("  They are not evidence of stability or of a move; they "
                  "are unmeasured.")
            paths = [paths[i] for i in keep]
            dates = [dates[i] for i in keep]
            if len(paths) < 5:
                sys.exit("fewer than 5 frames survive the clarity floor; "
                         "lower --min-clarity or sample a different hour")
        else:
            print("  every frame clears it")

    rois = [parse_roi(text) for text in args.roi]
    if rois:
        pass
    elif args.survey:
        print("\nSURVEY: tiling the whole frame. Nothing is being judged on "
              "how it looks;\nthe agreement test alone decides which cells "
              "are on rigid ground.")
        rois = survey_rois(paths, size=args.roi_size,
                           top_margin=args.top_margin,
                           bottom_margin=args.bottom_margin)
        if not rois:
            sys.exit("could not read the first frame to lay out a grid")
    else:
        print(f"\nproposing {args.candidates} candidate patches (sharp in "
              f"space, still in time,\ntop {args.land_fraction:.0%} of frame); "
              "the agreement test picks the keepers")
        rois = propose_rois(paths, size=args.roi_size, count=args.candidates,
                            land_fraction=args.land_fraction,
                            top_margin=args.top_margin,
                            bottom_margin=args.bottom_margin)
        if not rois:
            sys.exit("could not choose features; pass --roi name:x,y,w,h")
    if len(rois) <= 20:
        for roi in rois:
            extra = ""
            if "spread" in roi:
                # motion is illumination-levelled: water churns, land does not.
                extra = (f"  structure={roi['structure']:.2f} "
                         f"variation={roi['spread']:.2f} "
                         f"motion={roi.get('motion', float('nan')):.2f}")
            print(f"  {roi['name']:<6} x={roi['x']:>5} y={roi['y']:>5} "
                  f"{roi['w']}x{roi['h']}{extra}")

    pick = best_reference(paths)
    print(f"\n  reference frame: {dates[pick]:%Y-%m-%d} "
          f"({os.path.basename(paths[pick])}) — of {len(paths)} frames, the "
          "one\n  the rest of the record matches best")
    preview = wrote(draw_rois(paths[pick], rois,
                              os.path.join(OUT_DIR, f"candidates_{slug}.jpg")))
    if preview:
        print(f"    {preview}")

    coarse = None
    # The resolution the record derives for itself, in the coarse pass. It is a
    # property of the frames, not of the pass, so the patch cross-check below is
    # floored by it too rather than taking a flag at face value.
    derived = float("nan")
    if not args.no_coarse:
        print("\n" + "=" * 74)
        print("WHOLE-FRAME REGISTRATION")
        print("=" * 74)
        print("Every pixel at once, against the reference. This is the primary")
        print("measurement. It is not capped by patch size, and a frame that is")
        print("mostly water does not confuse it: uncorrelated content raises the")
        print("floor of the correlation surface rather than competing for the")
        print("peak, so the peak comes from whatever IS common to both frames.")
        print(f"  peaks are searched within {args.max_shift:.0f} px of no "
              "movement; a real move\n  beyond that reads as the count "
              "below, not as itself")
        coarse = coarse_shifts(paths, dates, pick, max_shift=args.max_shift)
        if coarse.empty:
            print("  no frame could be registered against the reference")
            coarse = None
        else:
            print(f"  registered {len(coarse)} of {len(paths)} frames")
            print(f"  median offset across the record: "
                  f"{coarse['offset'].median():.2f} px")
            print(f"  largest single-date offset:      "
                  f"{coarse['offset'].max():.2f} px on "
                  f"{coarse['offset'].idxmax():%Y-%m-%d}")
            # Does a single translation actually describe this record? The
            # direct pass cannot answer that about itself. Consecutive weeks
            # are the easiest pair to register, so their steps summed must
            # reproduce the direct measurement -- and where they do not, the
            # direct numbers are not measuring the scene.
            print("\nCROSS-CHECK: frame to frame, PER STEP")
            sequential = sequential_shifts(paths, dates,
                                           max_shift=args.max_shift)
            reliable = False
            noise = resolution = float("nan")
            threshold = args.step_px
            if sequential.empty:
                print("  no consecutive pair could be registered")
                sequential = None
            else:
                # COMPARE PER STEP, NOT CUMULATIVELY. The first version of this
                # test summed the frame-to-frame steps and compared the total
                # against the direct measurement -- which compares a RANDOM WALK
                # against something that does not accumulate. With 13 px of
                # per-step noise over 145 frames, a camera that never moved at
                # all drifts ~130 px, and the test called that "the two passes
                # disagree" and withheld every verdict. It was measuring its own
                # error budget.
                #
                # The honest comparison is between two measurements of the SAME
                # pair of frames: the sequential step from A to B, against the
                # difference of the two direct measurements of A and B. Neither
                # accumulates, so agreement means what it says.
                pairs = sequential.dropna(subset=["previous"])
                pairs = pairs[pairs.index.isin(coarse.index)
                              & pairs["previous"].isin(coarse.index)]
                if len(pairs) < 5:
                    print("  too few pairs measured both ways to compare")
                    reliable, gap = False, float("nan")
                else:
                    later = coarse.loc[pairs.index]
                    earlier = coarse.loc[pairs["previous"]]
                    apart = np.hypot(
                        pairs["dx"].to_numpy()
                        - (later["dx"].to_numpy() - earlier["dx"].to_numpy()),
                        pairs["dy"].to_numpy()
                        - (later["dy"].to_numpy() - earlier["dy"].to_numpy()))
                    gap = float(np.median(apart))
                    print(f"  {len(pairs)} frame pairs measured BOTH ways "
                          "(against each other, and as the")
                    print("  difference of their two measurements against the "
                          "reference)")
                    print(f"  the two routes differ by a median of {gap:.1f} px "
                          "on the same pair")
                print(f"  largest single frame-to-frame step: "
                      f"{sequential['step'].max():.1f} px on "
                      f"{sequential['step'].idxmax():%Y-%m-%d}")
                biggest = sequential["step"].nlargest(5)
                for date, value in biggest.items():
                    print(f"    {date:%Y-%m-%d}  {value:8.1f} px")
                # What the gap buys is a precision, not a pass/fail.
                wandering = False
                if np.isfinite(gap):
                    noise, resolution = noise_floor(gap)
                    threshold = not_finer_than_the_record(args.step_px,
                                                           resolution)
                    wandering = resolution >= WANDER_SHARE * args.max_shift
                    # A resolution this coarse must not be allowed to relax
                    # the patch agreement test, which would then pass by
                    # being asked nothing. `derived` is what floors that
                    # test, so it stays unset here.
                    derived = float("nan") if wandering else resolution
                    print(f"  two routes over the same pair disagree by "
                          f"sqrt(3) x the error in one")
                    print(f"  measurement, so this record measures a frame to "
                          f"about +/-{noise:.2g} px")
                    print(f"  SMALLEST MOVE THIS RECORD CAN RESOLVE: "
                          f"{resolution:.2g} px (3x that error)")
                    if threshold > args.step_px:
                        print(f"  the {args.step_px:.2g} px step threshold is "
                              "below that floor, so steps are")
                        print(f"  searched at {threshold:.2g} px instead")
                    if wandering:
                        corner = args.max_shift * 2 ** 0.5
                        print("\n" + "!" * 74)
                        print(f"THAT RESOLUTION IS "
                              f"{resolution / args.max_shift:.0%} OF THE "
                              f"SEARCH WINDOW IT WAS GIVEN.")
                        print("A window is a prior about where the peak is, "
                              "not a measurement. When")
                        print("the smallest move a record can resolve scales "
                              "with the box it was")
                        print("searched in, the box is bounding the answer: "
                              "there is no dominant")
                        print("peak on the scene, so the best position inside "
                              "the window is found")
                        print("near the window, and the gap between the two "
                              "routes measures the")
                        print("width of the box rather than the error of "
                              "either route.")
                        print(f"  the largest offset is "
                              f"{coarse['offset'].max():.0f} px, against a "
                              f"{corner:.0f} px box corner "
                              f"({coarse['offset'].max() / corner:.0%} of the "
                              f"way out)")
                        print("THE TEST: re-run with --max-shift "
                              f"{args.max_shift * 2:.0f} and with "
                              f"--max-shift {args.max_shift // 4:.0f}.")
                        print("If the resolution follows the window both "
                              "ways, nothing here is")
                        print("registering and no epoch table from this "
                              "record means anything. If it")
                        print("stops following, the window that stopped it is "
                              "the one to use.")
                        print("Either way: pool NOTHING across these dates "
                              "until it is settled.")
                        print("!" * 74)
                    steady = float(coarse["offset"].median())
                    if not wandering and steady > 2.0 * noise:
                        print(f"  the median offset of {steady:.0f} px is "
                              f"{steady / noise:.0f}x that error: the "
                              "record's")
                        print("  spread is real motion, not measurement noise")
                else:
                    noise = resolution = float("nan")
                    threshold = args.step_px
                reliable = bool(np.isfinite(gap)) and not wandering
                covered = len(coarse) / max(len(paths), 1)
                if reliable and covered < 0.7:
                    # A precision measured on the frames that registered says
                    # nothing about the ones that did not. An earlier version
                    # announced that one translation described the record
                    # while half of it had been dropped.
                    print(f"  Only {covered:.0%} of the sampled frames "
                          "registered at all.")
                    print("  The verdict below covers THAT subset. The rest of "
                          "the record is not")
                    print("  stable and is not moving; it is unmeasured, and "
                          "the quarters below")
                    print("  say which part. Treat any epoch here as provisional "
                          "until the")
                    print("  unregistered frames are explained.")
                elif not reliable and wandering:
                    pass   # already said, loudly, immediately above
                elif not reliable:
                    print("\n" + "!" * 74)
                    print("TOO FEW PAIRS WERE MEASURED BOTH WAYS TO STATE A "
                          "PRECISION. Without it")
                    print("there is no floor to judge a step against, so "
                          "neither a move nor")
                    print("stability can be claimed. Re-run with --every 3.")
                    print("!" * 74)
            coarse_steps = find_steps(coarse[["dx", "dy"]], threshold=threshold)
            if reliable:
                report_record(coarse_steps, coarse.index, slug, threshold,
                              "the whole frame",
                              noise=noise if threshold > args.step_px else None)
            else:
                print(f"\n{len(coarse_steps)} apparent discontinuit"
                      f"{'y' if len(coarse_steps) == 1 else 'ies'} and any "
                      "epoch split are WITHHELD.")
            registration = coarse.copy()
            if sequential is not None:
                registration = registration.join(
                    sequential[["step", "cum_dx", "cum_dy"]], how="outer")
            reg_csv = os.path.join(OUT_DIR, f"registration_{slug}.csv")
            registration.to_csv(reg_csv)
            wrote(reg_csv)
            wrote(plot_registration(
                coarse, sequential, coarse_steps if reliable else [],
                os.path.join(OUT_DIR, f"registration_{slug}.png")))

            quality = registration_quality(coarse, sequential, dates)
            if len(quality) > 1:
                print("\nWHERE THE RECORD REGISTERS  (share of frames whose "
                      "peak beat chance)")
                print("  quarter   frames   vs reference   vs previous frame")
                for _, row in quality.iterrows():
                    print(f"  {row['period']:<9} {row['frames']:>6}   "
                          f"{row['direct_ok']:>11.0%}   {row['seq_ok']:>16.0%}")
                print("  A record that fails everywhere is a broken method. "
                      "One that fails for a")
                print("  while and then works is telling you when to look at "
                      "the camera.")
                # THE TWO COLUMNS DIVERGING IS ITS OWN FINDING, and reading it
                # off the table by eye is how it gets missed. A frame that
                # registers against the one before it but NOT against the
                # reference is not a bad frame: it is a frame that has moved
                # away from the anchor. Measurement error has no dates in it;
                # drift does, and the quarter where the columns part is the
                # date it started.
                parted = drifted_quarters(quality)
                if len(parted):
                    print(f"\n  IN {len(parted)} QUARTER"
                          f"{'' if len(parted) == 1 else 'S'} THE RECORD "
                          "REGISTERS AGAINST THE PREVIOUS FRAME BUT NOT")
                    print("  AGAINST THE REFERENCE: "
                          + ", ".join(f"{row['period']} "
                                      f"({row['direct_ok']:.0%} vs "
                                      f"{row['seq_ok']:.0%})"
                                      for _, row in parted.iterrows()))
                    print("  Consecutive frames still match, so the imagery is "
                          "fine and the method")
                    print("  works. What those frames no longer match is the "
                          "ANCHOR. That is drift")
                    print(f"  away from {dates[pick]:%Y-%m-%d}, not noise -- "
                          "noise has no dates in it.")
                    print("  A wider --max-shift will recapture them and hide "
                          "it; the honest reads")
                    print("  are to treat the parting quarter as a boundary, "
                          "and to re-run with a")
                    print("  reference inside the late stretch and see whether "
                          "the columns swap.")

            if args.contact_sheet or not reliable:
                weak = []
                if sequential is not None and not sequential.empty:
                    weak = list(sequential.index[
                        sequential["confidence"] < MIN_CONFIDENCE])
                sheet = wrote(contact_sheet(
                    all_paths, all_dates,
                    os.path.join(OUT_DIR, f"frames_{slug}.jpg"),
                    flagged=set(weak) | foggy))
                if sheet:
                    print(f"\n  every sampled frame, labelled: {sheet}")
                    print("  Boxed in orange: too little contrast to register, "
                          "or failed against\n  their neighbour. At Walton "
                          "these are the whiteouts.")

            print("\nDISPLACEMENT FROM THE REFERENCE, by date")
            for line in spark(coarse["offset"],
                              marks=[s["date"] for s in coarse_steps],
                              label="offset from reference, px"):
                print(line)
            if sequential is not None and not sequential.empty:
                print("\nFRAME-TO-FRAME STEP, by date  (a camera move is ONE "
                      "tall bar;")
                print("a record that never settles is tall everywhere)")
                for line in spark(sequential["step"],
                                  marks=[s["date"] for s in coarse_steps],
                                  label="step from the previous frame, px"):
                    print(line)
            if coarse["offset"].max() > args.roi_size / 2:
                print(f"\n  NOTE: the record moves further than half a "
                      f"{args.roi_size} px patch "
                      f"({coarse['offset'].max():.0f} px).")
                print("  Patches could not have measured this on their own, "
                      "and without the")
                print("  coarse pass above they would each have reported a "
                      "different wrong")
                print("  answer. They are now cut from the moved position, so "
                      "what follows")
                print("  measures the residual.")

    print("\nregistering patches" +
          (" against the coarse-corrected position" if coarse is not None
           else " (NO coarse correction — --no-coarse)"))
    frame, reference_date = track(paths, dates, rois, pick, coarse=coarse)
    if frame.empty:
        sys.exit("no usable measurements")

    # Which of those candidates were actually on the rigid scene. This is the
    # step that used to be my guess about the frame contents and is now the
    # record's own answer.
    # The tolerance is floored by the record's own resolution for the same
    # reason the step threshold is: two patches cannot be asked to agree more
    # closely than either of them can be measured. Without the floor, a run
    # that measures itself to +/-48 px would reject every patch in the frame
    # for failing a 3 px test and call that a finding.
    agree = not_finer_than_the_record(args.agree_px, derived)
    if agree > args.agree_px:
        print(f"\n  the {args.agree_px:.2g} px agreement tolerance is below "
              f"the {derived:.2g} px this record\n  can resolve, so patches "
              f"are required to agree to {agree:.2g} px instead")
    kept, rejected, distance = agreeing_features(frame, tolerance=agree)
    if 1 < len(rois) <= 20:
        print(f"\nagreement between candidates (median px apart over the "
              f"record, threshold {agree:.2g})")
        for roi in rois:
            name = roi["name"]
            apart = distance.get(name)
            mark = "keep  " if name in kept else "drop  "
            if apart is None or not np.isfinite(apart):
                reading = "   too few dates in common to compare"
            else:
                reading = f"{apart:6.2f} px from the group"
            print(f"  {mark}{name:<4} {reading}")
    if not kept:
        finite = [v for v in distance.values() if np.isfinite(v)]
        closest = (f"{min(finite):.1f} px" if finite else
                   "— no two patches even share enough readable dates to "
                   "compare")
        message = [
            f"\nNo {MIN_CLUSTER} of {len(rois)} patches agree with each other "
            f"to within {agree:.2g} px.",
            f"The closest any patch comes to the rest is {closest}.",
            "Nothing in the searched part of the frame is behaving like rigid "
            "structure, so",
            "there is no answer to give -- a verdict from these patches would "
            "be noise."]
        if args.survey:
            # The survey already looked everywhere, so the next move is not a
            # different place to look.
            message += [
                "",
                "The survey covered the whole frame, so this is not a patch "
                "placement problem.",
                "Either nothing in this camera's view is rigid at this patch "
                "size -- try",
                f"--roi-size {args.roi_size * 2}, which holds four times the "
                "structure -- or the record",
                "really does not register, which is itself the answer to the "
                "question asked:",
                "the archive would not be one geometric record."]
        else:
            message += [
                "",
                "Let the record find the rigid region instead of the picker:",
                "  --survey        tiles the whole frame and reports which "
                "cells agree",
                "or pass the structure by hand from the preview above:",
                f"  --roi roof:X,Y,{args.roi_size},{args.roi_size} "
                f"--roi pier:X,Y,{args.roi_size},{args.roi_size}"]
        sys.exit("\n".join(message))
    if rejected:
        print(f"\n  {len(rejected)} of {len(rois)} candidates did not track "
              f"with the others and were dropped.")
        print("  That is the expected outcome, not a fault: a patch on fog, "
              "surf or glare\n  agrees with nothing, which is exactly how it "
              "is identified.")
    frame = frame[frame["feature"].isin(kept)]
    kept_rois = [r for r in rois if r["name"] in kept]
    dropped_rois = [r for r in rois if r["name"] not in kept]
    if len(kept_rois) > 1:
        # Where the rigid region turned out to be, in numbers, so the finding
        # survives without the picture.
        xs = [r["x"] for r in kept_rois]
        ys = [r["y"] for r in kept_rois]
        print(f"  the agreeing patches span x {min(xs)}-{max(xs) + args.roi_size}"
              f", y {min(ys)}-{max(ys) + args.roi_size}")
    final = wrote(draw_rois(paths[pick], kept_rois,
                            os.path.join(OUT_DIR, f"rois_{slug}.jpg"),
                            # Hundreds of grey survey cells make the picture
                            # unreadable; the kept ones are the finding.
                            rejected=(dropped_rois if len(dropped_rois) <= 30
                                      else ())))
    if final:
        # Absolute, because data/ is often a symlink into another checkout and
        # a relative path pasted into a browser is a 404 rather than a file.
        print("\n  OPEN THIS BEFORE TRUSTING THE RESULT:")
        print(f"    open {final}")
        print("  Orange = kept, grey = dropped. Every orange box should sit on "
              "something\n  bolted down. Orange on a moored boat or a parked "
              "car tracks the boat, and\n  several boats on one mooring field "
              "agree with each other. Override with\n  --roi name:x,y,w,h.")

    # Patches in one band of rows are several samples of one thing. They will
    # agree with each other beautifully and say nothing about the camera.
    rows = [roi["y"] for roi in kept_rois]
    if len(kept_rois) > 2 and max(rows) - min(rows) < ROI_SIZE:
        print(f"\n  WARNING: every kept patch sits within "
              f"{max(rows) - min(rows)} px of the same row.")
        print("  They are sampling one band of the frame, so their agreement "
              "is not fully")
        print("  independent evidence. Spread them by hand with --roi before "
              "believing a verdict.")

    csv_path = os.path.join(OUT_DIR, f"geometry_{slug}.csv")
    frame.to_csv(csv_path, index=False)
    wrote(csv_path)
    signal = agreeing_signal(frame)
    patch_threshold = not_finer_than_the_record(args.step_px, derived)
    steps = find_steps(signal[["dx", "dy"]], threshold=patch_threshold)
    wrote(plot(frame, signal, steps,
               os.path.join(OUT_DIR, f"geometry_{slug}.png")))

    print("\n" + "=" * 74)
    print("PATCH CROSS-CHECK")
    print("=" * 74)
    print(f"reference frame {reference_date:%Y-%m-%d}; every offset is "
          "measured from it")
    print(f"median offset across the record: "
          f"{signal['offset'].median():.2f} px")
    print(f"largest single-date offset:      {signal['offset'].max():.2f} px "
          f"on {signal['offset'].idxmax():%Y-%m-%d}")
    disagreement = float(signal["spread"].median())
    print(f"typical disagreement between features: {disagreement:.2f} px "
          f"(worst {signal['spread_max'].median():.2f} px)")

    # The whole method rests on the patches tracking one rigid scene. When
    # they disagree by more than this record can measure a frame to, they
    # are not measuring a common thing and NEITHER answer below is worth
    # anything -- not the steps, and not their absence. The first Walton run
    # reported a 0.02 px median offset with 16.6 px of disagreement and then
    # called the record stable, which it had no basis for.
    print(f"features kept: {len(kept)} of {len(rois)} candidates "
          f"({', '.join(kept)})")

    # The kept features passed a MEDIAN agreement test, which a genuine camera
    # move does not break -- every rigid patch moves together -- but a zoom or
    # a lens change does, because rigid patches then move by different amounts.
    # So a per-date disagreement above the agreement tolerance still invalidates
    # the verdict even after the selection.
    trustworthy = disagreement <= agree
    if not trustworthy:
        print("\n" + "!" * 74)
        print(f"KEPT FEATURES STILL DISAGREE ({disagreement:.1f} px apart, "
              f"against a {agree:.2g} px agreement tolerance).")
        print("They passed the median agreement test and fail it date by date,")
        print("which is what a zoom or a lens change looks like: rigid points")
        print("move together under a pan and apart under a zoom. It is also")
        print("what too few surviving frames looks like. The verdict below is")
        print("not evidence either way; open the ROI preview and check the")
        print("orange boxes are on structure, then re-run with --every 3.")
        print("!" * 74)

    if not trustworthy and not steps:
        print("\nNo epoch split can be claimed from the patches — see above.")
    elif not steps:
        report_record([], signal.index, slug, patch_threshold,
                      "the agreeing patches")
    elif not trustworthy:
        # Steps found among features that do not agree are steps in the
        # disagreement, not in the camera. Printing dates and epoch sizes under
        # the warning invites them to be read as findings, which is how the
        # first Walton run produced "2 epochs" out of four patches on fog.
        print(f"\n{len(steps)} apparent step"
              f"{'' if len(steps) == 1 else 's'} found, NOT reported: with the "
              "features this far apart they")
        print("describe the disagreement, not the camera. Fix the patches "
              "first.")
    else:
        report_record(steps, signal.index, slug, patch_threshold,
                      "the agreeing patches")

    # The two passes measure the same camera by different means. The answer is
    # what they AGREE on, and that is worth stating rather than leaving to be
    # diffed out of two epoch lists.
    if coarse is not None and trustworthy:
        rows = reconcile(coarse_steps, list(coarse.index),
                         steps, list(signal.index))
        print("\n" + "=" * 74)
        print("THE TWO PASSES, SIDE BY SIDE")
        print("=" * 74)
        # HOW INDEPENDENT THESE TWO PASSES ARE DEPENDS ON THE NUMBERS. Each
        # patch is cut from the coarse-corrected position and its residual is
        # ADDED BACK to the coarse shift, so the patch offsets carry the coarse
        # offsets inside them. When the residual is small beside the coarse
        # shift, the patch pass is mostly re-reporting the coarse pass and
        # "both passes agree" is closer to arithmetic than to corroboration.
        # Walton at --max-shift 900: coarse median offset 192.61 px, patch
        # median 193.48 px, patch-to-patch scatter about 40 px. Thirty-six of
        # forty-two steps "seen by both" reads as strong agreement and is
        # largely the same measurement twice.
        share = patch_independence(signal, coarse)
        if np.isfinite(share) and share < INDEPENDENT_SHARE:
            print("NOT TWO INDEPENDENT MEASUREMENTS. Each patch was cut from "
                  "the coarse-")
            print("corrected position and its residual added back, so the "
                  "patch offsets")
            print(f"CONTAIN the coarse offsets. The residual the patches "
                  f"contribute is about")
            print(f"{share:.0%} of the offset they report, so agreement below "
                  "is mostly the coarse")
            print("pass being read back. The genuinely independent pair is "
                  "whole-frame-against-")
            print("reference versus frame-to-frame, and those disagree by the "
                  "gap reported")
            print("above. Use --no-coarse for a patch answer that owes the "
                  "coarse pass nothing")
            print("-- at the cost of a pass that cannot see a move bigger than "
                  "half a patch.")
            print("-" * 74)
        if not rows:
            print("Neither pass found a step. Nothing moved by more than each "
                  "pass could see.")
        for verdict, date, jump, twin in rows:
            size = f"{jump:5.1f} px" + (f" / {twin:.1f} px" if twin else "")
            print(f"  {date:%Y-%m-%d}  {size:<20} {verdict}")
        agreed = [row for row in rows if row[0].startswith("both")]
        print(f"\n{len(agreed)} of {len(rows)} candidate move"
              f"{'' if len(rows) == 1 else 's'} are seen BY BOTH passes. "
              "Those are the")
        print("ones to respect when pooling. A step only one pass found is a "
              "question,")
        print("not a finding: check whether the other pass had frames there "
              "at all")
        print("(the label says), and look at those dates in --contact-sheet "
              "before")
        print("either accepting or dismissing it.")

def finish():
    """List everything written, in one block, and open it if asked.

    Registered with atexit rather than called at the end of main, because the
    runs that write the most interesting files are often the ones that stop
    early -- a patch pass with nothing to report has still written the preview
    that shows why, and printing its path only on the happy path is how you end
    up scrolling for it.
    """
    if not WRITTEN:
        return
    print("\n" + "=" * 74)
    print("FILES WRITTEN")
    print("=" * 74)
    for item in WRITTEN:
        print(f"  {item}")
    images = [p for p in WRITTEN if p.lower().endswith((".png", ".jpg"))]
    if OPEN_IMAGES:
        show(WRITTEN)
    elif images:
        print("\n  open them all with:")
        print("    open " + " ".join(images))
        print("  or add --open to the command and the run will do it.")


if __name__ == "__main__":
    main()
