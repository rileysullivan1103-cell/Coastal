"""Offline checks for check_camera_geometry.py -- no network, no real imagery.

The frames are synthesised here with a KNOWN shift applied on a known date, so
the question under test is whether the registration recovers a shift it was
given and whether the step finder lands on the date it was given. Everything
that matters is arithmetic on arrays, which is testable without a camera.

    python test_geometry_offline.py
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd

import check_camera_geometry as g

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


def scene(width=384, height=256, seed=0):
    """A frame with hard structure up top and moving water below.

    The top half holds fixed high-contrast edges -- a roofline and a tower --
    which is what a real static feature looks like to a correlator. The bottom
    half is re-randomised per frame, standing in for surf: a patch chosen there
    would track nothing.
    """
    rng = np.random.default_rng(seed)
    image = np.full((height, width), 40.0)
    image[:, :] += rng.normal(0, 2, (height, width))
    # roofline: a bright horizontal slab with a stepped edge
    image[60:78, 40:160] = 210
    image[52:60, 90:130] = 235
    # tower: a vertical bar with a cap
    image[30:110, 250:268] = 200
    image[22:30, 240:278] = 240
    return image


def water(image, seed):
    """Re-randomise the lower band, and jitter the whole frame slightly.

    The jitter matters: real land is not byte-identical between frames weeks
    apart -- sun angle, haze and JPEG noise all move it a little. A fixture
    without it makes every static pixel look like a composited overlay to
    OVERLAY_SPREAD, which is a property of the fixture and not of the code.
    """
    # The noise has to be big enough to survive a JPEG round trip and the
    # downsample-by-2 the picker applies, or a flat region compresses to a
    # constant and the fixture claims to be an overlay. Real scenes clear this
    # easily: sun angle alone reshades a wall by far more than this between
    # one week and the next.
    rng = np.random.default_rng(seed)
    out = image + rng.normal(0, 6.0, image.shape) + rng.normal(0, 4.0)
    height = out.shape[0]
    out[int(height * 0.6):, :] = rng.normal(120, 40,
                                            out[int(height * 0.6):, :].shape)
    return out


def shifted(image, dy, dx):
    return np.roll(np.roll(image, dy, axis=0), dx, axis=1)


def test_phase_shift_recovers_a_known_offset():
    print("phase correlation")
    base = scene()
    for dy, dx in ((0, 0), (3, -5), (-7, 2), (11, 9)):
        moved = shifted(base, dy, dx)
        got = g.phase_shift(base, moved)
        check(f"recovers ({dy}, {dx})",
              got is not None and abs(got[0] - dy) < 0.6
              and abs(got[1] - dx) < 0.6, got)

    # Sub-pixel: a half-pixel shift made by averaging two neighbours.
    half = (base.astype(float) + shifted(base, 0, 1)) / 2
    got = g.phase_shift(base, half)
    check("a half-pixel shift reads between 0 and 1",
          got is not None and 0.2 < got[1] < 0.9, got)

    check("mismatched shapes are refused, not broadcast",
          g.phase_shift(base, base[:100, :100]) is None)

    # Fog: no structure to align. The peak should be far weaker than a real
    # match, which is what MIN_CONFIDENCE exists to catch.
    rng = np.random.default_rng(7)
    fog = rng.normal(128, 1.0, base.shape)
    sharp = g.phase_shift(base, shifted(base, 2, 2))[2]
    flat = g.phase_shift(fog, rng.normal(128, 1.0, base.shape))[2]
    check("a structural match scores above a featureless one",
          sharp > flat, (sharp, flat))


def test_features_are_chosen_on_land():
    print("\nfeature proposal")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        base = scene()
        paths = []
        for index in range(12):
            frame = water(base, seed=index)
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
            paths.append(path)

        rois = g.propose_rois(paths, size=64, count=3, land_fraction=0.55)
        check("it proposes features", len(rois) == 3, len(rois))
        below = [r for r in rois if r["y"] > 256 * 0.55]
        check("none of them sit in the water", not below, below)
        check("they do not all sit on one structure",
              len({r["x"] // 64 for r in rois}) > 1 or len(rois) < 2,
              [r["x"] for r in rois])
        for roi in rois:
            check(f"{roi['name']} is inside the frame",
                  0 <= roi["x"] and roi["x"] + roi["w"] <= 384
                  and 0 <= roi["y"] and roi["y"] + roi["h"] <= 256, roi)
    finally:
        shutil.rmtree(tmp)


def beach_scene(width=512, height=384, seed=0):
    """The shape that broke the first auto-picker: sky on top, then land.

    A beach camera looks out, so the top of the frame is sky -- bright, almost
    gradient-free, and the stillest thing in the record. The first version
    scored structure DIVIDED BY temporal spread, which made stillness pay, and
    it put three of four patches at y=0. Nothing up there is correlatable.
    """
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width))
    sky = int(height * 0.28)
    image[:sky, :] = 205 + rng.normal(0, 0.4, (sky, width))   # still, featureless
    image[sky:, :] = 70 + rng.normal(0, 3, (height - sky, width))
    # A lighthouse and a roofline, below the sky and above the waterline.
    image[sky + 10: sky + 90, 120:150] = 225
    image[sky + 2: sky + 12, 112:158] = 245
    image[sky + 50: sky + 70, 300:430] = 215
    image[sky + 40: sky + 50, 330:400] = 180
    return image


def test_the_picker_avoids_sky():
    print("\nfeature proposal on a sky-topped frame")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        base = beach_scene()
        sky = int(384 * 0.28)
        paths = []
        for index in range(12):
            frame = water(base, seed=index)
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
            paths.append(path)

        rois = g.propose_rois(paths, size=64, count=3, land_fraction=0.6)
        check("it proposes features", len(rois) == 3, len(rois))
        # A patch whose whole body is above the horizon is the failure mode.
        in_sky = [r for r in rois if r["y"] + r["h"] <= sky]
        check("no patch lies entirely in the sky", not in_sky,
              [(r["name"], r["y"]) for r in rois])
        check("none is flush against the top edge",
              not [r for r in rois if r["y"] == 0],
              [(r["name"], r["y"]) for r in rois])

        # And the patches it does pick must actually be correlatable: a known
        # shift has to come back out of each one.
        moved = shifted(base, -4, 6)
        for roi in rois:
            got = g.phase_shift(g.crop(base, roi), g.crop(moved, roi))
            check(f"{roi['name']} recovers a planted shift",
                  got is not None and abs(got[0] + 4) < 1.0
                  and abs(got[1] - 6) < 1.0, got)
    finally:
        shutil.rmtree(tmp)


def test_patches_are_spread_vertically():
    """All four patches at y=0 is four samples of one band.

    Greedy selection with local suppression walks along a single row when the
    strongest structure lies in one horizontal strip -- which is what a coastal
    camera looks like, with the far shore near the top and water below. Those
    four patches agree with each other beautifully and say nothing about
    whether the camera moved, because nothing distinguishes a camera shift
    from a shift of that one strip.
    """
    print("\nfeatures are spread down the frame, not along one row")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        # Structure strongest in a strip near the top, weaker but present
        # further down -- the greedy picker takes the strip four times.
        base = np.full((400, 600), 60.0)
        rng = np.random.default_rng(5)
        base += rng.normal(0, 2, base.shape)
        base[20:70, :] = 230                       # the bright strip
        base[30:60, ::40] = 30                     # with hard vertical edges
        for row in (150, 230):                     # weaker structure lower down
            base[row:row + 24, :] = 150
            base[row + 6:row + 18, ::60] = 40

        paths = []
        for index in range(12):
            frame = water(base, seed=index)
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
            paths.append(path)

        rois = g.propose_rois(paths, size=64, count=3, land_fraction=0.7)
        check("it proposes features", len(rois) == 3, len(rois))
        spread = max(r["y"] for r in rois) - min(r["y"] for r in rois)
        check("they do not all share one row", spread > 64, 
              [(r["name"], r["y"]) for r in rois])
        check("and they still sit inside the searchable region",
              all(0 <= r["y"] and r["y"] + r["h"] <= 400 for r in rois), rois)
    finally:
        shutil.rmtree(tmp)


def test_the_picker_rejects_a_burned_in_overlay():
    """The failure that survived the sky fix: a composited banner.

    Walton's second run put all four patches at y=0 and then reported 0.03 px
    of disagreement -- a perfect score from a region that cannot move. An
    overlay is byte-identical between frames (zero temporal spread) and full
    of text edges (high structure), so a picker that rewards stillness and
    sharpness will choose it every time. It is welded to the sensor, not the
    world: register against it and the camera reads as rock steady whatever it
    actually did.
    """
    print("\nfeature proposal against a composited overlay")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        base = beach_scene()
        banner = np.zeros((28, 512))
        rng = np.random.default_rng(99)
        banner[:, :] = 15
        for start in range(10, 500, 26):          # blocky "text"
            banner[6:22, start:start + 14] = 245
        paths = []
        for index in range(12):
            frame = water(base, seed=index)
            # Lighting drifts across the record, as it really does...
            frame = frame + rng.normal(0, 4)
            # ...but the banner is composited afterwards, so it never varies.
            frame[:28, :] = banner
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
            paths.append(path)

        rois = g.propose_rois(paths, size=64, count=3, land_fraction=0.6)
        check("it still proposes features", len(rois) == 3, len(rois))
        on_banner = [r for r in rois if r["y"] < 28]
        check("no patch lands on the overlay", not on_banner,
              [(r["name"], r["y"]) for r in rois])
        check("every chosen patch actually varies over time",
              all(r["spread"] > g.OVERLAY_SPREAD for r in rois),
              [(r["name"], round(r["spread"], 3)) for r in rois])
        check("and each carries the numbers that justify it",
              all("structure" in r and "spread" in r for r in rois))

        # The overlay must be rejected on stillness, not on position: the same
        # banner anywhere in the searchable area has to lose.
        for roi in rois:
            got = g.phase_shift(g.crop(base, roi), g.crop(shifted(base, 5, -3), roi))
            check(f"{roi['name']} tracks the scene, not the banner",
                  got is not None and abs(got[0] - 5) < 1.0
                  and abs(got[1] + 3) < 1.0, got)
    finally:
        shutil.rmtree(tmp)


def test_a_banner_with_a_live_timestamp_is_still_rejected():
    """Walton's real banner, which the absolute threshold let through.

    "Walton Lighthouse Cam by UCSC" on the left, "2023-07-02 12:59:35" on the
    right, on a strip about 70 px tall across a 1920 px frame. Two things beat
    OVERLAY_SPREAD: the timestamp digits genuinely change every frame, and JPEG
    ringing around the static letters moves by a few grey levels. So the strip
    measured 4.6-9.1 variation and passed as scene -- while being welded to the
    sensor, which is the one thing that makes it useless here. All four patches
    landed on it and reported 0.03 px of agreement between them.

    The strip is caught on RELATIVE variation instead: it changes far less than
    the scene rows around it, whatever its absolute numbers.
    """
    print("\nfeature proposal against a banner with a live timestamp")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        base = beach_scene(width=640, height=480)
        rng = np.random.default_rng(11)
        paths = []
        for index in range(12):
            frame = water(base, seed=index)
            strip = np.full((18, 640), 70.0)
            # Static title text on the left...
            for start in range(8, 300, 22):
                strip[4:14, start:start + 12] = 240
            # ...and a timestamp on the right that really does change.
            for start in range(420, 620, 18):
                if rng.random() > 0.4:
                    strip[4:14, start:start + 10] = 240
            # JPEG ringing around the letters, which is what cleared the floor.
            strip += rng.normal(0, 3.0, strip.shape)
            frame[:18, :] = strip
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
            paths.append(path)

        # The trap only exists if the strip looks alive by the absolute test.
        stack = np.stack([g.load_gray(p) for p in paths])
        strip_spread = float(np.median(np.abs(stack[:, :18] -
                                              np.median(stack[:, :18], axis=0))))
        check("the fixture's banner does clear the absolute floor",
              strip_spread > g.OVERLAY_SPREAD, strip_spread)

        rois = g.propose_rois(paths, size=64, count=3, land_fraction=0.6)
        check("it still proposes features", len(rois) == 3, len(rois))
        on_banner = [r for r in rois if r["y"] < 18]
        check("and none of them lands on the banner", not on_banner,
              [(r["name"], r["y"]) for r in rois])
        check("nor straddles it",
              all(r["y"] >= 18 for r in rois), [r["y"] for r in rois])
    finally:
        shutil.rmtree(tmp)


def test_explicit_margins_exclude_the_banner():
    """The escape hatch for when the automatic test does not fire.

    It did not fire on Walton: three of four patches stayed on the banner
    through two rounds of increasingly clever detection. A camera's overlay is
    a fixed, known property of that camera, so being able to say "ignore the
    top 100 px" is worth more than another inference that might also miss.
    """
    print("\nexplicit top and bottom margins")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        base = beach_scene(width=640, height=480)
        rng = np.random.default_rng(23)
        paths = []
        for index in range(12):
            frame = water(base, seed=index)
            # A banner that defeats every automatic test: it varies as much as
            # the scene does, because it is translucent.
            frame[:40, :] = frame[:40, :] * 0.4 + rng.normal(120, 8, (40, 640))
            for start in range(8, 620, 20):
                frame[12:30, start:start + 11] = 245 + rng.normal(0, 4)
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
            paths.append(path)

        # No claim about what the picker does WITHOUT the margin: whether a
        # given banner attracts patches depends on the scene, and the point of
        # the flag is that it does not have to be argued about.
        tight = g.propose_rois(paths, size=64, count=3, land_fraction=0.6,
                               top_margin=80)
        check("with --top-margin none of them start inside it",
              all(r["y"] >= 40 for r in tight), [r["y"] for r in tight])
        check("and the patches are still usable",
              all(r["spread"] > 0 for r in tight), tight)

        # A margin that eats the whole searchable region says so rather than
        # returning patches from nowhere.
        check("an impossible margin returns nothing, loudly",
              g.propose_rois(paths, size=64, count=3, land_fraction=0.6,
                             top_margin=900) == [])
    finally:
        shutil.rmtree(tmp)


def test_roi_preview_is_written():
    print("\nthe ROI preview image")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        path = os.path.join(tmp, "frame.jpg")
        Image.fromarray(scene().clip(0, 255).astype(np.uint8)).save(path)
        rois = [{"name": "roof", "x": 40, "y": 40, "w": 128, "h": 64},
                {"name": "tower", "x": 224, "y": 16, "w": 96, "h": 112}]
        out = os.path.join(tmp, "rois.jpg")
        got = g.draw_rois(path, rois, out)
        check("it writes a file", got == out and os.path.exists(out))
        with Image.open(out) as drawn:
            check("the preview keeps the frame's size",
                  drawn.size == (384, 256), drawn.size)
            check("and is RGB, so the boxes can be coloured",
                  drawn.mode == "RGB", drawn.mode)

        # A frame that is not an image must not take the run down: the preview
        # is a convenience and the registration is the job.
        broken = os.path.join(tmp, "broken.jpg")
        with open(broken, "w") as fh:
            fh.write("not an image")
        check("an unreadable frame returns None rather than raising",
              g.draw_rois(broken, rois, out) is None)
    finally:
        shutil.rmtree(tmp)


def test_the_margin_excludes_the_whole_patch():
    """--top-margin 100 must not leave a patch spanning rows 36-164.

    The first version excluded patch CENTRES above the margin, so a 128 px
    patch centred exactly on the boundary still had half its body on the
    banner. The margin has to account for the patch's half-height.
    """
    print("\nmargins exclude the patch, not its centre")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        base = beach_scene(width=640, height=480)
        paths = []
        for index in range(12):
            frame = water(base, seed=index)
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
            paths.append(path)
        rois = g.propose_rois(paths, size=64, count=3, land_fraction=0.7,
                              top_margin=60)
        check("every patch starts at or below the margin",
              all(r["y"] >= 60 for r in rois), [r["y"] for r in rois])
        check("not merely centred below it",
              all(r["y"] + r["h"] // 2 > 60 for r in rois))
    finally:
        shutil.rmtree(tmp)


def test_the_reference_is_the_sharpest_frame():
    """Not the first. Walton's record opens on its foggiest frame.

    Everything is measured relative to the reference, so a soft reference
    degrades every measurement in the run and not only its own.
    """
    print("\nthe reference frame is chosen, not assumed")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        base = beach_scene(width=512, height=384)
        rois = [{"name": "a", "x": 100, "y": 120, "w": 96, "h": 96},
                {"name": "b", "x": 300, "y": 130, "w": 96, "h": 96}]
        dates = pd.date_range("2024-01-07", periods=6, freq="7D", tz="UTC")
        paths = []
        for index in range(6):
            frame = water(base, seed=index)
            if index != 3:
                # Fog: wash the contrast out of everything but the sharp one.
                frame = frame * 0.25 + 150
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
            paths.append(path)

        check("it picks the one clear frame", g.sharpest(paths, rois) == 3,
              g.sharpest(paths, rois))
        frame, reference = g.track(paths, list(dates), rois,
                                   g.sharpest(paths, rois))
        check("and registers against it", reference == dates[3], reference)
        check("every frame is still measured",
              set(frame["feature"]) == {"a", "b"}, set(frame["feature"]))
    finally:
        shutil.rmtree(tmp)


def test_a_planted_step_is_found_on_the_right_date():
    print("\nstep detection on a planted remount")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        base = scene()
        dates = pd.date_range("2024-01-07", periods=40, freq="7D", tz="UTC")
        move_at = 20                      # the camera is nudged here
        paths = []
        for index, _ in enumerate(dates):
            dy, dx = (0, 0) if index < move_at else (-6, 9)
            frame = water(shifted(base, dy, dx), seed=index)
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
            paths.append(path)

        rois = [{"name": "roof", "x": 40, "y": 40, "w": 128, "h": 64},
                {"name": "tower", "x": 224, "y": 16, "w": 96, "h": 112}]
        frame, reference = g.track(paths, list(dates), rois)
        check("every frame and feature measured",
              len(frame) == len(dates) * len(rois), len(frame))
        check("the reference is the first frame", reference == dates[0])

        signal = g.agreeing_signal(frame)
        check("before the move the offset is ~0",
              signal["offset"].iloc[:move_at].max() < 1.0,
              signal["offset"].iloc[:move_at].max())
        check("after the move it is the planted magnitude",
              abs(signal["offset"].iloc[move_at:].median()
                  - np.hypot(6, 9)) < 1.0,
              signal["offset"].iloc[move_at:].median())
        # All features move together, so the spread between them stays near
        # zero. That is the signature that separates a camera move from one
        # feature falling over.
        check("the features agree throughout", signal["spread"].max() < 1.0,
              signal["spread"].max())

        steps = g.find_steps(signal["offset"], threshold=3.0, persist=3)
        check("exactly one step is reported", len(steps) == 1, steps)
        if steps:
            check("and it lands on the date the move was planted",
                  steps[0]["date"] == dates[move_at],
                  f"{steps[0]['date']} vs {dates[move_at]}")

        epochs = g.describe_epochs(steps, list(signal.index))
        check("the record splits into two epochs", len(epochs) == 2, epochs)
        if len(epochs) == 2:
            check("every sampled frame lands in exactly one epoch",
                  epochs[0]["frames"] + epochs[1]["frames"] == len(dates),
                  [e["frames"] for e in epochs])
            check("the split is where the move was",
                  epochs[0]["frames"] == move_at, epochs[0]["frames"])
    finally:
        shutil.rmtree(tmp)


def test_a_stable_record_reports_no_step():
    print("\nstep detection on a camera that never moved")
    dates = pd.date_range("2024-01-07", periods=30, freq="7D", tz="UTC")
    rng = np.random.default_rng(3)
    # Sub-pixel jitter is what a fixed camera in wind actually looks like.
    offset = pd.Series(np.abs(rng.normal(0, 0.4, len(dates))), index=dates)
    check("jitter under the threshold raises nothing",
          g.find_steps(offset, threshold=3.0, persist=3) == [])

    # A single wild frame is a bad frame, not a remount.
    spike = offset.copy()
    spike.iloc[15] = 40.0
    check("one outlier frame is not a step",
          g.find_steps(spike, threshold=3.0, persist=3) == [],
          g.find_steps(spike, threshold=3.0, persist=3))


def tracked(series_by_feature, start="2024-01-07"):
    """A tracked frame built straight from per-feature (dy, dx) series."""
    rows = []
    length = len(next(iter(series_by_feature.values())))
    dates = pd.date_range(start, periods=length, freq="7D", tz="UTC")
    for name, series in series_by_feature.items():
        for date, (dy, dx) in zip(dates, series):
            rows.append({"date": date, "feature": name, "dx": float(dx),
                         "dy": float(dy), "confidence": 0.5, "frame": "x.jpg"})
    frame = pd.DataFrame(rows)
    frame["offset"] = np.hypot(frame["dx"], frame["dy"])
    return frame


def test_disagreement_is_measured_as_a_vector():
    """Two features that slid opposite ways are not in agreement.

    The first version measured the spread as the range of offset MAGNITUDES.
    A feature at dx=+5 and one at dx=-5 both have magnitude 5, so the range is
    zero and the run called that perfect agreement -- the one arrangement that
    most clearly means the patches are not on a common rigid scene.
    """
    print("\ndisagreement between features")
    frame = tracked({"a": [(0, 5)] * 8, "b": [(0, -5)] * 8})
    signal = g.agreeing_signal(frame)
    check("opposite shifts of equal size are a disagreement",
          signal["spread"].median() > 4.0, signal["spread"].median())

    together = tracked({"a": [(0, 5)] * 8, "b": [(0, 5.2)] * 8})
    check("features that moved together are not",
          g.agreeing_signal(together)["spread"].median() < 1.0,
          g.agreeing_signal(together)["spread"].median())


def test_the_agreeing_group_is_recovered_from_noise():
    """The rigid patches are found without being told which they are.

    This is the whole reason the picker no longer has to be right. Three
    features are given one shared record -- a camera that jumped on sample 20
    -- and three are given unrelated noise, which is what a patch on fog, surf
    or glare produces. The largest mutually-consistent subset must be the three
    real ones, with no appeal to where they sit in the frame.
    """
    print("\nrecovering the rigid features from a field of candidates")
    rng = np.random.default_rng(11)
    truth = [(0.0, 0.0)] * 20 + [(-6.0, 9.0)] * 20
    series = {}
    for name in ("roof", "tower", "wall"):
        series[name] = [(dy + rng.normal(0, 0.3), dx + rng.normal(0, 0.3))
                        for dy, dx in truth]
    for name in ("fog", "surf", "glare"):
        series[name] = [(rng.normal(0, 14), rng.normal(0, 14))
                        for _ in truth]
    frame = tracked(series)

    kept, rejected, distance = g.agreeing_features(frame, tolerance=3.0)
    check("the three rigid features are kept",
          set(kept) == {"roof", "tower", "wall"}, kept)
    check("and the three noise features are dropped",
          set(rejected) == {"fog", "surf", "glare"}, rejected)
    check("each reject is reported with how far out it was",
          all(distance.get(n, 0) > 3.0 for n in rejected),
          {n: round(distance.get(n, -1), 1) for n in rejected})

    # And the step survives the selection: dropping the noise must not drop
    # the signal with it.
    signal = g.agreeing_signal(frame[frame["feature"].isin(kept)])
    steps = g.find_steps(signal["offset"], threshold=3.0, persist=3)
    check("the planted step survives the selection", len(steps) == 1, steps)
    if steps:
        check("on the right date", steps[0]["date"] == signal.index[20],
              steps[0]["date"])


def test_a_frame_with_nothing_rigid_in_it_returns_nothing():
    """Better no answer than an answer from two patches of fog.

    Two features will always agree about as well as any other two, so a run
    that accepted the best pair would report a verdict from whatever noise
    happened to correlate. Below MIN_CLUSTER the honest output is empty.
    """
    print("\na frame with no rigid structure in it")
    rng = np.random.default_rng(5)
    series = {name: [(rng.normal(0, 12), rng.normal(0, 12)) for _ in range(30)]
              for name in ("a", "b", "c", "d")}
    kept, rejected, distance = g.agreeing_features(tracked(series),
                                                   tolerance=3.0)
    check("no group is claimed", kept == [], kept)
    check("everything is reported as rejected", len(rejected) == 4, rejected)
    check("with distances, so the failure says how far apart they are",
          all(distance[n] > 3.0 for n in ("a", "b", "c", "d")), distance)

    # Two of four agreeing is still not enough.
    series["b"] = list(series["a"])
    kept, _, _ = g.agreeing_features(tracked(series), tolerance=3.0)
    check("a lone agreeing pair is not a rigid scene", kept == [], kept)


def test_features_with_no_overlapping_dates_are_not_linked():
    """Absence of disagreement is not agreement.

    A patch that only ever survives the confidence cut in winter and one that
    only survives in summer have no common date to compare, and a distance of
    zero over an empty set would link them into a clique of imaginary friends.
    """
    print("\nfeatures that never overlap in time")
    frame = tracked({"a": [(0, 0)] * 12, "b": [(0, 0)] * 12,
                     "c": [(0, 0)] * 12})
    dates = sorted(frame["date"].unique())
    # 'c' survives only on the last three dates, 'a' and 'b' only on the first.
    keep_early = frame["date"].isin(dates[:6]) & frame["feature"].isin(["a", "b"])
    keep_late = frame["date"].isin(dates[9:]) & (frame["feature"] == "c")
    frame = frame[keep_early | keep_late]
    kept, _, _ = g.agreeing_features(frame, tolerance=3.0)
    check("a non-overlapping feature does not join the group",
          "c" not in kept, kept)


def test_a_move_bigger_than_a_patch():
    """The bug that made every patch at Walton disagree with every other.

    A patch of N pixels can only resolve a displacement of N/2. Past that the
    two patches hold non-overlapping ground, the correlation peak is spurious,
    and -- this is the part that misleads -- it is spurious in a DIFFERENT
    direction in each patch, because each is matching different accidental
    content. The symptom is "no two patches agree", which is also what patches
    on water look like, so the disagreement alone cannot tell you which you
    have.

    The fix is to register the whole frame first, which has no such cap, and
    cut each patch from the moved position so it only ever measures a residual.
    This plants a 150 px move behind a 96 px patch and checks both halves: that
    the naive pass fails, and that the corrected pass recovers the move and
    dates it.
    """
    print("\na camera move larger than the patch")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        # Proportioned like a coastal camera rather than scaled up from the
        # small fixture: a real frame gives phase correlation a decent SHARE
        # of rigid content, and the whole-frame pass only works because of
        # that share. Structure here occupies the upper third; the rest is
        # re-randomised every frame.
        base = np.full((768, 1024), 55.0)
        # IRREGULAR spacing on purpose. Evenly spaced buildings make the scene
        # periodic, and a shift of one period is indistinguishable from no
        # shift: an earlier version of this fixture spaced them 130 px apart
        # and the 138 px move was recovered as 8. Real streets are irregular,
        # but a picket fence or a row of pilings is not, and a camera looking
        # at one has a genuine ambiguity no correlator can resolve.
        edges = [40, 150, 330, 395, 560, 690, 745, 900]
        for index, x in enumerate(edges):
            width = 45 + 11 * (index % 4)
            base[150:260, x:x + width] = 205      # a row of buildings
            base[110:150, x + 8:x + width - 8] = 240   # roofs
        base[90:100, :] = 175                    # a treeline
        base[260:270, :] = 140                   # a sea wall
        dates = pd.date_range("2024-01-07", periods=30, freq="7D", tz="UTC")
        move_at, planted = 15, (-60.0, 138.0)   # 150 px, well past 96/2
        rng = np.random.default_rng(7)
        paths = []
        for index, _ in enumerate(dates):
            dy, dx = (0.0, 0.0) if index < move_at else planted
            frame = shifted(base, int(dy), int(dx)).copy()
            frame[300:, :] = 110 + rng.normal(0, 35, (768 - 300, 1024))
            frame += rng.normal(0, 4, frame.shape)
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
            paths.append(path)

        rois = [{"name": "roof", "x": 180, "y": 120, "w": 96, "h": 96},
                {"name": "tower", "x": 440, "y": 130, "w": 96, "h": 96},
                {"name": "eaves", "x": 700, "y": 125, "w": 96, "h": 96}]

        # 1. the failure, reproduced deliberately
        naive, _ = g.track(paths, list(dates), rois, 0)
        naive_signal = g.agreeing_signal(naive)
        naive_offset = naive_signal["offset"][
            naive_signal.index >= dates[move_at]].median()
        # The patches do not merely lose precision past their own size: what
        # they report bears no relation to the move. Whether they also disagree
        # with each other depends on what accidental content they land on --
        # the frames here are similar enough that they can agree on the same
        # wrong answer, which is the more dangerous of the two failures.
        check("without the coarse pass the patches are simply wrong",
              abs(naive_offset - np.hypot(*planted)) > 20.0, naive_offset)

        # 2. the whole frame, which has no such cap
        coarse = g.coarse_shifts(paths, list(dates), 0, downsample=2)
        check("the whole frame registers every date",
              len(coarse) == len(dates), len(coarse))
        # Slice by DATE, not position: dropped frames shift the positions and
        # a positional slice then compares the wrong halves of the record.
        cut = dates[move_at]
        moved = coarse["offset"][coarse.index >= cut].median()
        before = coarse["offset"][coarse.index < cut].max()
        check("and recovers the planted magnitude",
              abs(moved - np.hypot(*planted)) < 3.0, moved)
        check("with nothing before the move", before < 2.0, before)
        steps = g.find_steps(coarse["offset"], threshold=3.0, persist=3)
        check("the whole-frame pass finds one step", len(steps) == 1, steps)
        if steps:
            check("on the planted date", steps[0]["date"] == dates[move_at],
                  f"{steps[0]['date']} vs {dates[move_at]}")

        # 3. the patches, now measuring a residual
        fixed, _ = g.track(paths, list(dates), rois, 0, coarse=coarse)
        signal = g.agreeing_signal(fixed)
        check("with the coarse pass the patches agree",
              signal["spread"].median() < 1.0, signal["spread"].median())
        kept, rejected, _ = g.agreeing_features(fixed, tolerance=3.0)
        check("and all three survive the agreement test",
              len(kept) == 3, (kept, rejected))
        after = signal["offset"][signal.index >= dates[move_at]].median()
        check("the patch total is the displacement, not the residual",
              abs(after - np.hypot(*planted)) < 3.0, after)
        patch_steps = g.find_steps(signal["offset"], threshold=3.0, persist=3)
        check("the patches date the move too", len(patch_steps) == 1,
              patch_steps)
        if patch_steps:
            check("to the same date",
                  patch_steps[0]["date"] == dates[move_at],
                  patch_steps[0]["date"])
    finally:
        shutil.rmtree(tmp)


def coastal_frames(tmp, dates, motion, seed=7):
    """Frames with a realistic share of rigid structure and a planted motion.

    `motion` maps an index to the (dy, dx) the camera is at by then.
    """
    from PIL import Image
    base = np.full((768, 1024), 55.0)
    edges = [40, 150, 330, 395, 560, 690, 745, 900]
    for index, x in enumerate(edges):
        width = 45 + 11 * (index % 4)
        base[150:260, x:x + width] = 205
        base[110:150, x + 8:x + width - 8] = 240
    base[90:100, :] = 175
    base[260:270, :] = 140
    rng = np.random.default_rng(seed)
    paths = []
    for index, _ in enumerate(dates):
        dy, dx = motion(index)
        frame = shifted(base, int(dy), int(dx)).astype(float)
        frame[300:, :] = 110 + rng.normal(0, 35, (768 - 300, 1024))
        frame += rng.normal(0, 4, frame.shape)
        path = os.path.join(tmp, f"f{index:02d}.jpg")
        Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
        paths.append(path)
    return paths


def test_frame_to_frame_agrees_with_the_reference():
    """The one test in this module with a right answer it can check itself.

    Registering each frame against a distant reference and registering it
    against its neighbour are two routes to the same quantity, because there is
    one camera. Consecutive weeks are the easier pair -- same season, similar
    sun -- so where the two routes agree, a single translation describes the
    record; where they diverge, the direct pass is locking onto something other
    than the scene, and it can do that while still reporting a confident peak.
    """
    print("\nframe-to-frame against direct-to-reference")
    tmp = tempfile.mkdtemp()
    try:
        dates = pd.date_range("2024-01-07", periods=24, freq="7D", tz="UTC")
        # Two moves, so the cumulative sum has something to accumulate.
        def motion(index):
            if index < 8:
                return (0.0, 0.0)
            if index < 16:
                return (-40.0, 90.0)
            return (-40.0, 210.0)
        paths = coastal_frames(tmp, dates, motion)

        direct = g.coarse_shifts(paths, list(dates), 0, downsample=2)
        steps = g.sequential_shifts(paths, list(dates), downsample=2)
        check("every consecutive pair is measured",
              len(steps) == len(dates), len(steps))

        walk = np.hypot(steps["cum_dx"], steps["cum_dy"])
        gap = float(np.nanmedian(np.abs(walk.reindex(direct.index)
                                        - direct["offset"])))
        check("the two routes agree on a clean record", gap < 3.0, gap)

        # Both moves show as a single large frame-to-frame step, on the day.
        big = steps["step"].nlargest(2).sort_index()
        check("exactly the planted moves stand out",
              list(big.index) == [dates[8], dates[16]], list(big.index))
        check("and the rest of the record is quiet",
              steps["step"].drop(list(big.index)).max() < 3.0,
              steps["step"].drop(list(big.index)).max())
        check("the first move's magnitude is right",
              abs(big.iloc[0] - np.hypot(40, 90)) < 3.0, big.iloc[0])
        check("and the second's",
              abs(big.iloc[1] - np.hypot(0, 120)) < 3.0, big.iloc[1])

        # And the failure it exists to catch: a record where the direct pass
        # is fed frames it cannot match. Every frame after the midpoint is
        # replaced by unrelated content, so the direct pass registers noise
        # while the sequential pass sees one huge step and then quiet.
        # The two routes must then disagree.
        rng = np.random.default_rng(1)
        from PIL import Image
        for index in range(12, 24):
            junk = rng.normal(128, 45, (768, 1024))
            Image.fromarray(junk.clip(0, 255).astype(np.uint8)).save(paths[index])
        broken = g.coarse_shifts(paths, list(dates), 0, downsample=2)
        broken_steps = g.sequential_shifts(paths, list(dates), downsample=2)
        if not broken.empty and not broken_steps.empty:
            walk = np.hypot(broken_steps["cum_dx"], broken_steps["cum_dy"])
            gap = float(np.nanmedian(np.abs(walk.reindex(broken.index)
                                            - broken["offset"])))
            check("an unregistrable record makes the two routes diverge",
                  gap > 10.0 or len(broken) < len(dates) * 0.75,
                  f"gap {gap:.1f}, {len(broken)} of {len(dates)} registered")
    finally:
        shutil.rmtree(tmp)


def test_the_text_chart():
    """The chart exists so the finding survives not opening a PNG.

    It is what gets pasted into a message, so the three things it must not get
    wrong are: a step has to look like a step, a real gap in the record has to
    be visible, and ordinary spacing between samples must NOT be drawn as a
    gap -- at one frame a week over three years there are more columns than
    samples, and marking every empty column as missing buries the holes that
    matter.
    """
    print("\nthe terminal chart")
    dates = pd.date_range("2024-01-07", periods=40, freq="7D", tz="UTC")
    flat = pd.Series(np.zeros(len(dates)), index=dates)
    lines = g.spark(flat)
    check("a flat record draws something", len(lines) > 5, len(lines))
    check("and claims no gaps",
          not any("gap in the record" in line for line in lines))
    body = "\n".join(lines)
    check("nothing is drawn above the baseline",
          body.count("*") > 0 and all("*" not in line for line in lines[1:-4]),
          [line for line in lines[1:-4] if "*" in line][:1])

    stepped = pd.Series(np.where(np.arange(len(dates)) < 20, 0.0, 120.0),
                        index=dates)
    top = g.spark(stepped)[1]
    check("a step reaches the top row", "*" in top, repr(top[:40]))
    check("and only in its second half",
          top.index("*") > len(top) * 0.5, top.index("*"))

    # A hole: drop three months out of the middle.
    holed = stepped.drop(dates[12:24])
    drawn = g.spark(holed)
    check("a real gap is marked", any("·" in line for line in drawn))
    check("and the legend says so",
          any("gap in the record" in line for line in drawn))

    marked = g.spark(flat, marks=[dates[20]])
    check("a step date draws a rule", any("|" in line for line in marked[1:6]))
    check("and the legend explains it",
          any("candidate step" in line for line in marked))

    check("too little data draws nothing",
          g.spark(pd.Series([1.0], index=dates[:1])) == [])


def test_the_survey_tiles_the_whole_frame():
    """The survey asks nothing about appearance, so it must cover everything.

    Every failure so far came from a picker deciding where to look. The survey
    exists to stop deciding, so a grid that quietly skipped the bottom third --
    the one part of Walton's frame the land fraction never searched -- would
    reproduce the bug it is meant to end.
    """
    print("\nthe whole-frame survey grid")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        path = os.path.join(tmp, "f00.jpg")
        Image.fromarray(np.zeros((512, 768), dtype=np.uint8)).save(path)
        cells = g.survey_rois([path], size=128)
        check("the grid reaches the right edge",
              max(c["x"] for c in cells) + 128 > 768 - 128,
              max(c["x"] for c in cells))
        check("and the bottom edge",
              max(c["y"] for c in cells) + 128 > 512 - 128,
              max(c["y"] for c in cells))
        check("it starts at the top-left", (cells[0]["x"], cells[0]["y"]) == (0, 0),
              (cells[0]["x"], cells[0]["y"]))
        check("no cell runs off the frame",
              all(c["x"] + c["w"] <= 768 and c["y"] + c["h"] <= 512
                  for c in cells))
        check("names are unique",
              len({c["name"] for c in cells}) == len(cells))

        margined = g.survey_rois([path], size=128, top_margin=200)
        check("a top margin is respected",
              min(c["y"] for c in margined) >= 200,
              min(c["y"] for c in margined))

        # Thinning must still span the frame, not stop part way down it.
        thinned = g.survey_rois([path], size=64, limit=10)
        check("thinning keeps the span, it does not truncate",
              max(c["y"] for c in thinned) > 512 // 2,
              max(c["y"] for c in thinned))
    finally:
        shutil.rmtree(tmp)


def test_a_large_survey_still_finds_the_rigid_block():
    """Greedy grouping, on more patches than Bron-Kerbosch should be given.

    A survey hands the agreement test hundreds of cells. The exact search is
    exponential in the worst case, so past a threshold it switches to growing
    the group greedily -- which has to find the same answer on the shape that
    actually occurs: one dense block of agreeing cells in a sea of noise.
    """
    print("\na survey-sized candidate set")
    rng = np.random.default_rng(23)
    truth = [(0.0, 0.0)] * 15 + [(4.0, -7.0)] * 15
    series = {}
    for index in range(40):                     # the rigid block
        series[f"r{index}"] = [(dy + rng.normal(0, 0.2),
                                dx + rng.normal(0, 0.2)) for dy, dx in truth]
    for index in range(60):                     # water, sky, glare
        series[f"n{index}"] = [(rng.normal(0, 20), rng.normal(0, 20))
                               for _ in truth]
    kept, rejected, _ = g.agreeing_features(tracked(series), tolerance=3.0)
    check("the rigid block is recovered", len(kept) >= 35, len(kept))
    check("and holds no noise cell",
          all(name.startswith("r") for name in kept),
          [n for n in kept if not n.startswith("r")][:5])
    check("the noise cells are rejected", len(rejected) >= 60, len(rejected))

    signal = g.agreeing_signal(tracked(series)[
        tracked(series)["feature"].isin(kept)])
    steps = g.find_steps(signal["offset"], threshold=3.0, persist=3)
    check("the planted step survives a survey-sized selection",
          len(steps) == 1, steps)


def test_a_changed_frame_size_is_reported():
    """A resolution change is an epoch boundary, and the cheapest one to find.

    Nothing downstream means anything across it: the same pixel is different
    ground on either side, so every patch disagrees with every other and the
    disagreement IS the answer rather than an obstacle to it.
    """
    print("\nframe sizes across the record")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        dates = list(pd.date_range("2024-01-07", periods=6, freq="7D", tz="UTC"))
        paths = []
        for index in range(6):
            shape = (256, 384) if index < 3 else (512, 768)
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(path)
            paths.append(path)
        groups = g.frame_sizes(paths, dates)
        check("both sizes are found", len(groups) == 2, sorted(groups))
        check("and each carries its frames",
              sorted(len(v) for v in groups.values()) == [3, 3],
              {k: len(v) for k, v in groups.items()})

        same = g.frame_sizes(paths[:3], dates[:3])
        check("a single-resolution record reports one size", len(same) == 1,
              sorted(same))
    finally:
        shutil.rmtree(tmp)


def test_epochs_count_stills_not_just_samples():
    print("\nepoch sizes in stills")
    dates = list(pd.date_range("2024-01-07", periods=10, freq="7D", tz="UTC"))
    steps = [{"date": dates[4], "jump": 9.0, "before": 0.0, "after": 9.0}]
    hours = pd.date_range(dates[0], dates[-1], freq="h", tz="UTC")
    counts = pd.Series(60, index=hours)
    epochs = g.describe_epochs(steps, dates, counts=counts)
    check("two epochs", len(epochs) == 2, len(epochs))
    check("both carry a still count",
          all("stills" in e for e in epochs), epochs)
    check("the counts are non-trivial",
          all(e["stills"] > 0 for e in epochs), epochs)
    # A weekly sample says nothing about how much imagery an epoch holds; a
    # short epoch can be the dense one. Counting stills is what decides whether
    # an epoch is worth salvaging.
    check("a short epoch is measured in stills, not samples",
          epochs[0]["stills"] != epochs[0]["frames"], epochs[0])


def main():
    for test in (test_phase_shift_recovers_a_known_offset,
                 test_features_are_chosen_on_land,
                 test_the_picker_avoids_sky,
                 test_patches_are_spread_vertically,
                 test_the_picker_rejects_a_burned_in_overlay,
                 test_a_banner_with_a_live_timestamp_is_still_rejected,
                 test_explicit_margins_exclude_the_banner,
                 test_roi_preview_is_written,
                 test_the_margin_excludes_the_whole_patch,
                 test_the_reference_is_the_sharpest_frame,
                 test_a_planted_step_is_found_on_the_right_date,
                 test_a_stable_record_reports_no_step,
                 test_disagreement_is_measured_as_a_vector,
                 test_the_agreeing_group_is_recovered_from_noise,
                 test_a_frame_with_nothing_rigid_in_it_returns_nothing,
                 test_features_with_no_overlapping_dates_are_not_linked,
                 test_a_move_bigger_than_a_patch,
                 test_frame_to_frame_agrees_with_the_reference,
                 test_the_text_chart,
                 test_the_survey_tiles_the_whole_frame,
                 test_a_large_survey_still_finds_the_rigid_block,
                 test_a_changed_frame_size_is_reported,
                 test_epochs_count_stills_not_just_samples):
        test()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}"))
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
