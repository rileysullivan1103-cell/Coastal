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

Outputs, per camera, under data/geometry/:
  geometry_<slug>.csv   one row per frame per feature: dx, dy, confidence
  geometry_<slug>.png   per-feature offset against date, plus the step trace
and to stdout: candidate discontinuity dates and the epochs they imply, with
how many stills fall in each.
"""

import env  # noqa: F401  -- loads .env into os.environ

import argparse
import glob
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

# Patch size for tracking, and how many patches to follow.
ROI_SIZE = 128
N_FEATURES = 4
# Features are taken from the top of the frame by default: land, roofline and
# structure live there, and the beach and water -- which move for real reasons
# -- live below. --land-fraction moves the line.
LAND_FRACTION = 0.55

# A shift is only a discontinuity if it is bigger than this and it persists.
STEP_PX = 3.0
PERSIST = 3          # samples on each side that must agree
MIN_CONFIDENCE = 0.05  # phase-correlation peak sharpness below this is fog
# Temporal spread, in grey levels, below which a patch is not scene at all.
# Real imagery weeks apart never repeats exactly: sun angle, haze and JPEG
# noise alone put the median absolute deviation well above one level. A region
# that is byte-identical across months is composited after capture -- a
# timestamp bar, a logo, a letterbox -- and it does NOT move when the camera
# does, so registering against it reports a rock-steady camera no matter what
# the camera did. It is the most dangerous thing in the frame for this test
# and the most attractive to a picker that rewards stillness.
OVERLAY_SPREAD = 0.5
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


def phase_shift(reference, image):
    """(dy, dx, confidence): how far `image` has MOVED from `reference`.

    Sign convention matters here and is easy to get backwards. The correlation
    peak gives the shift that maps `image` back onto `reference`, which is the
    negative of the displacement; it is negated before returning, so a feature
    that has slid 5 px right reads dx = +5 on the plot rather than -5.

    Confidence is the correlation peak height against the mean of the surface.
    A sharp peak means one unambiguous alignment; fog, night and heavy rain
    flatten it, and those frames are dropped rather than averaged in.
    """
    if reference.shape != image.shape:
        return None
    window = np.outer(np.hanning(reference.shape[0]),
                      np.hanning(reference.shape[1]))
    first = np.fft.fft2((reference - reference.mean()) * window)
    second = np.fft.fft2((image - image.mean()) * window)
    cross = first * np.conj(second)
    magnitude = np.abs(cross)
    magnitude[magnitude < 1e-12] = 1e-12
    surface = np.fft.ifft2(cross / magnitude).real

    peak = np.unravel_index(int(np.argmax(surface)), surface.shape)
    # The cross-power surface is normalised, so the peak height is already a
    # 0-1 measure of how unambiguous the alignment is. A clean structural match
    # gives a spike; fog gives a plateau near 1/N.
    confidence = float(surface[peak])

    dy = peak[0] + _parabolic(surface[:, peak[1]], peak[0])
    dx = peak[1] + _parabolic(surface[peak[0], :], peak[1])
    # The correlation surface is circular: a shift of -2 appears at n-2.
    rows, cols = surface.shape
    if dy > rows / 2:
        dy -= rows
    if dx > cols / 2:
        dx -= cols
    return float(-dy), float(-dx), confidence


def propose_rois(paths, size=ROI_SIZE, count=N_FEATURES,
                 land_fraction=LAND_FRACTION, sample=16):
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

    median = np.median(stack, axis=0)
    spread = np.median(np.abs(stack - median), axis=0)
    gy, gx = np.gradient(median)
    structure = np.hypot(gy, gx)

    # Average both over the patch footprint. `size` is in full-resolution
    # pixels and the stack is at downsample 2, so the window is size // 2.
    from scipy.ndimage import uniform_filter
    window = max(3, size // 2)
    patch_structure = uniform_filter(structure, window)
    patch_spread = uniform_filter(spread, window)

    half = size // 4  # working at downsample=2
    usable = np.zeros(shape, dtype=bool)
    usable[half: shape[0] - half, half: shape[1] - half] = True
    # Land only. Everything below the line is beach and water, which move.
    usable[int(shape[0] * land_fraction):, :] = False
    if not usable.any():
        return []

    # Stability as a constraint: keep the calmer half of the candidates, then
    # maximise sharpness among them. Sky passes the stability test and then
    # loses on sharpness, which is the behaviour that was missing.
    calm = patch_spread <= np.percentile(patch_spread[usable], 50)

    # ...but not TOO still. See OVERLAY_SPREAD. The test has to be the SHARE OF
    # DEAD PIXELS inside the patch, not the patch's mean variation: a patch
    # straddling the bottom edge of a banner averages the banner's zero against
    # live scene below and passes comfortably, while half its body still cannot
    # move. That is exactly the patch the picker reaches for, because the
    # banner's edge is the sharpest thing in the frame.
    dead = spread <= OVERLAY_SPREAD
    dead_fraction = uniform_filter(dead.astype(float), window)
    alive = dead_fraction < MAX_DEAD_FRACTION
    dead_share = float((usable & ~alive).sum()) / max(int(usable.sum()), 1)
    if dead_share > 0.005:
        print(f"  {dead_share:.0%} of the searchable area is composited rather "
              "than scene — never varies between frames, excluded")
    score = np.where(usable & calm & alive, patch_structure, 0.0)
    if not score.any():
        print("  nothing is both calm and alive; dropping the calm test")
        score = np.where(usable & alive, patch_structure, 0.0)

    # Take the best patch from each horizontal band rather than the best four
    # overall. Greedy selection with local suppression walks along one row: at
    # Walton it put all four patches at y=0, spanning the full width but
    # sampling a single band, and four samples of one band agree with each
    # other whatever the camera did. Bands force the patches apart vertically,
    # which is what makes their agreement mean something.
    rows = np.flatnonzero(usable.any(axis=1))
    edges = np.linspace(rows[0], rows[-1] + 1, count + 1).astype(int)
    bands = [(edges[i], edges[i + 1]) for i in range(count)]

    rois = []
    working = score.copy()
    for band_top, band_bottom in bands:
        banded = np.zeros_like(working)
        banded[band_top:band_bottom, :] = working[band_top:band_bottom, :]
        # A band with nothing usable in it -- all water, or all overlay --
        # falls back to the best patch left anywhere, so a frame whose
        # structure really is all in one place still gets its features.
        source = banded if banded.any() else working
        if not source.any():
            break
        cy, cx = np.unravel_index(int(np.argmax(source)), source.shape)
        rois.append({"name": f"feature{len(rois) + 1}",
                     "x": int(cx * 2 - size // 2), "y": int(cy * 2 - size // 2),
                     "w": size, "h": size,
                     # Carried so the run can print why each patch was chosen.
                     # A patch reported with near-zero variation is an overlay
                     # whatever else the output says.
                     "structure": float(patch_structure[cy, cx]),
                     "spread": float(patch_spread[cy, cx])})
        # Suppress a generous neighbourhood so the patches are not all one
        # corner of one roof.
        y0, y1 = max(0, cy - half * 3), cy + half * 3
        x0, x1 = max(0, cx - half * 3), cx + half * 3
        working[y0:y1, x0:x1] = 0
    return rois


def draw_rois(path, rois, out_path):
    """Write the reference frame with the chosen patches boxed and labelled.

    "Eyeball these before trusting the result" is not actionable when the
    result is four x/y triples and the frame is 2560 px wide. A picture is.
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


def track(paths, dates, rois, min_confidence=MIN_CONFIDENCE):
    """dx/dy per frame per feature, against the first readable frame."""
    reference, reference_date = None, None
    rows, unreadable = [], 0
    for path, date in zip(paths, dates):
        image = load_gray(path)
        if image is None:
            unreadable += 1
            continue
        if reference is None:
            reference, reference_date = image, date
            print(f"  reference frame: {date:%Y-%m-%d} ({os.path.basename(path)})")
        for roi in rois:
            base, patch = crop(reference, roi), crop(image, roi)
            if base is None or patch is None:
                continue
            shifted = phase_shift(base, patch)
            if shifted is None:
                continue
            dy, dx, confidence = shifted
            rows.append({"date": date, "feature": roi["name"],
                         "dx": dx, "dy": dy, "confidence": confidence,
                         "frame": os.path.basename(path)})
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
    grouped = frame.groupby("date").agg(
        dx=("dx", "median"), dy=("dy", "median"),
        spread=("offset", lambda s: float(s.max() - s.min())),
        features=("feature", "nunique"))
    grouped["offset"] = np.hypot(grouped["dx"], grouped["dy"])
    return grouped.sort_index()


def find_steps(series, threshold=STEP_PX, persist=PERSIST):
    """Dates where the level shifts by `threshold` and stays shifted.

    A step is not a spike. Comparing the median of `persist` samples before a
    date against the median of `persist` after ignores a single bad frame and
    only fires on a change that the record keeps.
    """
    values = series.to_numpy(dtype=float)
    dates = list(series.index)
    hits = []
    for index in range(persist, len(values) - persist + 1):
        before = np.median(values[index - persist: index])
        after = np.median(values[index: index + persist])
        jump = abs(after - before)
        if jump >= threshold:
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
        index = max(candidates, key=lambda i: abs(values[i] - values[i - 1]))
        before = float(np.median(values[max(0, index - persist): index]))
        after = float(np.median(values[index: index + persist]))
        steps.append({"date": dates[index], "jump": abs(after - before),
                      "before": before, "after": after})
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


def coverage_counts(slug):
    """Hourly image counts, so an epoch can be measured in stills not samples."""
    paths = glob.glob(os.path.join(RIP_DIR, f"coverage_{slug}_hourly.csv"))
    if not paths:
        return None
    frame = pd.read_csv(paths[0])
    frame["hour"] = pd.to_datetime(frame["hour"], utc=True, errors="coerce")
    return frame.dropna(subset=["hour"]).set_index("hour")["images"]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

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
    ap.add_argument("--roi", action="append", default=[],
                    help="name:x,y,w,h — repeatable; overrides auto-selection")
    ap.add_argument("--land-fraction", type=float, default=LAND_FRACTION,
                    help="auto-selection uses only the top this much of frame")
    ap.add_argument("--step-px", type=float, default=STEP_PX,
                    help=f"a shift this large counts as a step (default {STEP_PX})")
    args = ap.parse_args()

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
    slug, paths, dates = sample_frames(args.camera, args.every, args.hour,
                                       args.limit)
    order = np.argsort(dates)
    paths = [paths[i] for i in order]
    dates = [dates[i] for i in order]
    print(f"\n{len(paths)} frames on disk, "
          f"{dates[0]:%Y-%m-%d} to {dates[-1]:%Y-%m-%d}")

    rois = [parse_roi(text) for text in args.roi]
    if not rois:
        print("\nchoosing static features (sharp in space, still in time, "
              f"top {args.land_fraction:.0%} of frame)")
        rois = propose_rois(paths, land_fraction=args.land_fraction)
        if not rois:
            sys.exit("could not choose features; pass --roi name:x,y,w,h")
    for roi in rois:
        extra = ""
        if "spread" in roi:
            extra = (f"  structure={roi['structure']:.2f} "
                     f"variation={roi['spread']:.2f}")
        print(f"  {roi['name']:<10} x={roi['x']} y={roi['y']} "
              f"{roi['w']}x{roi['h']}{extra}")

    # Four patches in one band of rows are four samples of one thing. They will
    # agree with each other beautifully and say nothing about the camera.
    rows = [roi["y"] for roi in rois]
    if len(rois) > 2 and max(rows) - min(rows) < ROI_SIZE:
        print(f"\n  WARNING: every patch sits within {max(rows) - min(rows)} px "
              "of the same row. They are")
        print("  sampling one band of the frame, so their agreement is not "
              "independent evidence.")
        print("  Spread them by hand with --roi before believing a verdict.")
    preview = draw_rois(paths[0], rois,
                        os.path.join(OUT_DIR, f"rois_{slug}.jpg"))
    if preview:
        print(f"\n  OPEN THIS BEFORE TRUSTING THE RESULT: {preview}")
        print("  Each box must sit on something bolted down — the lighthouse,")
        print("  a roofline, a railing. A box on a moored boat or a parked car")
        print("  tracks the boat. Re-run with --roi name:x,y,w,h to override.")

    print("\nregistering")
    frame, reference_date = track(paths, dates, rois)
    if frame.empty:
        sys.exit("no usable measurements")

    csv_path = os.path.join(OUT_DIR, f"geometry_{slug}.csv")
    frame.to_csv(csv_path, index=False)
    signal = agreeing_signal(frame)
    steps = find_steps(signal["offset"], threshold=args.step_px)
    png_path = plot(frame, signal, steps, os.path.join(OUT_DIR,
                                                       f"geometry_{slug}.png"))

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    print(f"reference frame {reference_date:%Y-%m-%d}; every offset is "
          "measured from it")
    print(f"median offset across the record: "
          f"{signal['offset'].median():.2f} px")
    print(f"largest single-date offset:      {signal['offset'].max():.2f} px "
          f"on {signal['offset'].idxmax():%Y-%m-%d}")
    disagreement = float(signal["spread"].median())
    print(f"typical disagreement between features: {disagreement:.2f} px")

    # The whole method rests on the patches tracking one rigid scene. When
    # they disagree by more than the size of the step being looked for, they
    # are not measuring a common thing and NEITHER answer below is worth
    # anything -- not the steps, and not their absence. The first Walton run
    # reported a 0.02 px median offset with 16.6 px of disagreement and then
    # called the record stable, which it had no basis for.
    trustworthy = disagreement <= args.step_px
    if not trustworthy:
        print("\n" + "!" * 74)
        print(f"FEATURES DO NOT AGREE ({disagreement:.1f} px apart, against a "
              f"{args.step_px:.0f} px step threshold).")
        print("They are not tracking one rigid scene, so the verdict below is")
        print("not evidence either way. Open the ROI preview: a patch on sky,")
        print("water or a moored boat produces exactly this. Fix the patches")
        print("with --roi and run again.")
        print("!" * 74)

    if not steps:
        verdict = ("The record reads as ONE geometric epoch."
                   if trustworthy else
                   "No epoch split can be claimed — see the warning above.")
        print(f"\nNo step larger than {args.step_px} px persists. {verdict}")
        print("That is evidence of stability, not proof: a move smaller than")
        print("the step threshold, or one inside a gap in the sampling, would")
        print("not appear. Re-run with --every 3 and a lower --step-px before")
        print("committing to a shoreline trend.")
    else:
        print(f"\n{len(steps)} candidate discontinuit"
              f"{'y' if len(steps) == 1 else 'ies'}:")
        for step in steps:
            print(f"  {step['date']:%Y-%m-%d}  {step['before']:.1f} px -> "
                  f"{step['after']:.1f} px  (jump {step['jump']:.1f})")
        epochs = describe_epochs(steps, list(signal.index),
                                 counts=coverage_counts(slug))
        print(f"\n{len(epochs)} epochs:")
        for index, epoch in enumerate(epochs, 1):
            stills = (f", {epoch['stills']:,} stills" if "stills" in epoch
                      else "")
            print(f"  {index}. {epoch['start']:%Y-%m-%d} to "
                  f"{epoch['end']:%Y-%m-%d}  "
                  f"{epoch['frames']} sampled frames{stills}")
        print("\nNothing has been corrected or re-registered. Before pooling")
        print("any pixel metric across these dates, decide per epoch.")

    print(f"\nwrote {csv_path}")
    if png_path:
        print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
