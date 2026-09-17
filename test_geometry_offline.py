"""Offline checks for check_camera_geometry.py -- no network, no real imagery.

The frames are synthesised here with a KNOWN shift applied on a known date, so
the question under test is whether the registration recovers a shift it was
given and whether the step finder lands on the date it was given. Everything
that matters is arithmetic on arrays, which is testable without a camera.

    python test_geometry_offline.py
"""

import os
import shutil
import subprocess
import sys
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

    # Fog: some signal, but no structure to align on. The peak should be far
    # weaker than a real match, which is what MIN_CONFIDENCE exists to catch.
    # The deviation stays above BLANK_STD on purpose -- below it the frame
    # carries no scene at all and is refused outright rather than scored, which
    # is a different rule tested separately.
    rng = np.random.default_rng(7)
    fog = rng.normal(128, 4.0, base.shape)
    sharp = g.phase_shift(base, shifted(base, 2, 2))[2]
    flat = g.phase_shift(fog, rng.normal(128, 4.0, base.shape))[2]
    check("a structural match scores above a featureless one",
          sharp > flat, (sharp, flat))
    check("and the featureless one is below the threshold",
          flat < g.MIN_CONFIDENCE, flat)


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


def test_the_reference_is_the_one_the_record_matches():
    """Not the first frame, and not the sharpest either.

    Everything is measured relative to the reference, so a bad one degrades
    every measurement in the run. The first frame is wrong because Walton's
    record opens on its foggiest day. The SHARPEST frame is wrong too, and
    worse: sharpness scored as gradient magnitude is maximised by noise, so a
    rainy or corrupted frame beats every real scene and then nothing registers
    against the anchor. What is wanted is the frame the rest of the record can
    be matched to, which is what gets measured.
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

        check("it picks the one clear frame", g.best_reference(paths) == 3,
              g.best_reference(paths))
        frame, reference = g.track(paths, list(dates), rois,
                                   g.best_reference(paths))
        check("and registers against it", reference == dates[3], reference)
        check("every frame is still measured",
              set(frame["feature"]) == {"a", "b"}, set(frame["feature"]))
    finally:
        shutil.rmtree(tmp)


def test_water_is_rejected_under_changing_light():
    """The failure visible in Walton's preview: four of twelve patches on the sea.

    The picker rejects water by asking what MOVED between frames. Fog, sun and
    exposure move the whole frame at once -- a gain and an offset -- and that
    global swing dwarfs the difference between a roofline that never moves and
    surf that never stops. Measured raw, land and water came out 15 and 27, a
    ratio no threshold can split, and every Walton candidate reported variation
    ~20 whether it sat on a house or on the open sea.

    Levelling each frame by its own WHOLE-FRAME median and spread removes the
    gain and offset and leaves the churn. The scale must come from the whole
    frame: per patch it would divide out the very variance being looked for.
    """
    print("\nwater, under changing light")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        rng = np.random.default_rng(3)
        paths = []
        for index in range(16):
            gain, lift = rng.uniform(0.55, 1.0), rng.uniform(0, 80)
            frame = np.empty((256, 384))
            # Land across the top: buildings that never move.
            frame[:120] = 60.0
            for x in range(20, 360, 70):
                frame[40:110, x:x + 40] = 200
                frame[20:40, x + 8:x + 32] = 235
            frame[:120] += rng.normal(0, 2, (120, 384))
            # Water below: churns on its own, every frame different.
            frame[120:] = rng.normal(125, 35, (136, 384))
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray((frame * gain + lift).clip(0, 255).astype(np.uint8)
                            ).save(path)
            paths.append(path)

        # land_fraction 1.0: the picker may search the water and must decline.
        rois = g.propose_rois(paths, size=64, count=6, land_fraction=1.0)
        # Fewer than asked for is fine and expected: the land strip is narrow
        # and local suppression keeps the patches apart. Declining to fill the
        # quota from the sea is the behaviour under test.
        check("it proposes patches", len(rois) >= 3, len(rois))
        wet = [r for r in rois if r["y"] > 120]
        check("none of them land on the water", not wet,
              [(r["name"], r["y"]) for r in rois])
        check("and the motion figure separates the two",
              all(r.get("motion", 9) < 2.0 for r in rois),
              [(r["name"], round(r.get("motion", -1), 2)) for r in rois])
    finally:
        shutil.rmtree(tmp)


def test_water_wins_a_median_split_when_it_covers_most_of_the_frame():
    """Walton's actual failure, which the levelling alone did not fix.

    The picker kept "the calmer half" of the searchable patches. That is an
    assertion about the frame -- that half of it is still -- and at Walton it
    is false: water reaches most of the way up the usable rows, so the calmer
    half still contained surf. Surf then WINS the sharpness contest, because
    every whitecap is an edge, and patches land on the sea. Levelling made the
    two separable; the fixed 50% is what let the water back in.

    Otsu's threshold puts the cut in the gap between the two groups wherever
    that gap sits, so the split follows the frame instead of being asserted.
    """
    print("\nwater covering most of the searchable frame")

    # The mechanism, in isolation: 70% water, 30% land, well separated.
    land = np.full(30, 1.0)
    sea = np.full(70, 9.0)
    both = np.concatenate([land, sea])
    median_split = float(np.percentile(both, 50))
    otsu = g.still_threshold(both)
    check("a median split admits the water", median_split >= 9.0, median_split)
    check("Otsu puts the cut in the gap", 1.0 < otsu < 9.0, round(otsu, 2))
    check("so only the land passes",
          int((both <= otsu).sum()) == 30, int((both <= otsu).sum()))

    one_cluster = np.random.default_rng(5).normal(4.0, 0.5, 200)
    cut = g.still_threshold(one_cluster)
    check("with no second group it still cuts near the middle",
          abs(float((one_cluster <= cut).mean()) - 0.5) < 0.25,
          round(float((one_cluster <= cut).mean()), 2))

    # And end to end, on a frame shaped like Walton's: a narrow built-up
    # strip over a lot of moving water, under changing light.
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        rng = np.random.default_rng(11)
        paths = []
        for index in range(16):
            gain, lift = rng.uniform(0.55, 1.0), rng.uniform(0, 80)
            frame = np.empty((256, 384))
            frame[:80] = 60.0                       # buildings, top 31%
            for x in range(15, 370, 63):
                frame[30:74, x:x + 34] = 205
                frame[14:30, x + 6:x + 28] = 238
            frame[:80] += rng.normal(0, 2, (80, 384))
            # Water below, with whitecaps: SHARPER than the rooflines, and
            # different in every frame. This is the trap.
            sea = rng.normal(120, 18, (176, 384))
            caps = rng.random((176, 384)) < 0.04
            sea[caps] = 250
            frame[80:] = sea
            path = os.path.join(tmp, f"w{index:02d}.jpg")
            Image.fromarray((frame * gain + lift).clip(0, 255).astype(np.uint8)
                            ).save(path)
            paths.append(path)

        rois = g.propose_rois(paths, size=64, count=8, land_fraction=1.0)
        check("it proposes patches", len(rois) >= 3, len(rois))
        wet = [(r["name"], r["y"]) for r in rois if r["y"] + 32 > 80]
        check("and none of them sit on the water", not wet, wet)
    finally:
        shutil.rmtree(tmp)


def test_fog_is_measured_and_excluded():
    """Walton's contact sheet: every frame that failed to register is a whiteout.

    Fog removes the structure a correlator needs WITHOUT removing the frame, so
    a foggy still is not a wrong measurement, it is an empty one -- and it is
    still handed to the correlator, which still returns a number. Clarity has
    to separate those frames from clear ones, and it has to do it on EDGE
    content rather than brightness or variance: a whiteout can be bright, and
    a flat grey wall of fog can have a perfectly ordinary spread of grey levels
    while carrying no edge at all.
    """
    print("\nfog, measured")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        base = beach_scene(width=512, height=384)
        dates = list(pd.date_range("2024-01-07", periods=10, freq="7D",
                                   tz="UTC"))
        paths = []
        for index in range(10):
            frame = water(base, seed=index)
            if index in (3, 4, 7):
                # Fog: contrast collapses toward a bright mean. Brightness goes
                # UP, so a brightness test would miss it entirely.
                frame = frame * 0.12 + 205
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
            paths.append(path)

        clarity = g.frame_clarity(paths, dates, downsample=1)
        check("every frame is measured", len(clarity) == 10, len(clarity))
        foggy = [clarity.loc[dates[i]] for i in (3, 4, 7)]
        clear = [clarity.loc[dates[i]] for i in (0, 1, 2, 5, 6, 8, 9)]
        check("fog scores far below clear", max(foggy) < min(clear) / 2,
              (round(max(foggy), 2), round(min(clear), 2)))
        floor = g.MIN_CLARITY * float(clarity.median())
        caught = set(clarity.index[clarity < floor])
        check("the default floor catches exactly the fog",
              caught == {dates[3], dates[4], dates[7]},
              sorted(str(d.date()) for d in caught))

        # The trap this replaces: the foggy frames here are BRIGHTER than the
        # clear ones, so anything keyed on mean level picks the wrong set.
        levels = {}
        for index, path in enumerate(paths):
            levels[index] = float(g.load_gray(path).mean())
        check("fog is brighter, so brightness would have chosen backwards",
              min(levels[i] for i in (3, 4, 7))
              > max(levels[i] for i in (0, 1, 2, 5, 6, 8, 9)),
              {k: round(v) for k, v in levels.items()})
    finally:
        shutil.rmtree(tmp)


def test_the_search_window_is_a_prior_that_reports_itself():
    """A camera bolted to a building does not move a quarter of its frame.

    The correlation surface spans the whole image, so a spurious peak 600 px
    away competes on equal terms with the true one 5 px away. At Walton three
    "independent" frame-to-frame steps landed within 1.5 px of each other at
    ~592 px, which is not what independent errors do.

    Limiting the search is a prior and has to behave like one: it must not
    change a measurement inside the window, it must say so when it overrules a
    peak outside it, and it must be removable -- otherwise a genuine repoint
    would be quietly reported as a small move, which is the failure this whole
    module exists to prevent.
    """
    print("\nthe search window")
    real = np.random.default_rng(0).normal(128, 30, (512, 512))

    small = shifted(real, 4, -9)
    unlimited = g.phase_shift(real, small)
    limited = g.phase_shift(real, small, max_shift=150)
    check("a shift inside the window is unchanged by it",
          unlimited[:2] == limited[:2], (unlimited[:2], limited[:2]))
    check("and is not flagged as overruled", limited[3] is False, limited[3])

    far = shifted(real, 0, 200)
    found = g.phase_shift(real, far, max_shift=150)
    check("a shift beyond the window is overruled", found[3] is True)
    check("and what is returned lies inside it",
          abs(found[1]) <= 151, found[1])
    check("removing the limit finds the real one",
          abs(g.phase_shift(real, far)[1] - 200) < 1,
          g.phase_shift(real, far)[1])
    check("a limit wide enough finds it too",
          abs(g.phase_shift(real, far, max_shift=400)[1] - 200) < 1,
          g.phase_shift(real, far, max_shift=400)[1])

    # The wrap matters: -200 lives at index n-200, so a naive window over
    # rows 0..R would call every negative shift "outside".
    back = shifted(real, 0, -100)
    inside = g.phase_shift(real, back, max_shift=150)
    check("a NEGATIVE shift inside the window is not overruled",
          inside[3] is False and abs(inside[1] + 100) < 1, inside[:2] + (inside[3],))


def test_a_blank_frame_cannot_register_or_anchor():
    """The worst bug in this module's history, and the most convincing one.

    Walton's record holds several fully black frames -- outages, or night where
    the sampled hour drifted. A constant array flattens to zero once its mean
    is removed, so the cross-power surface is identically zero: the shift reads
    (0, 0) and the sidelobe spread is zero, which the perfect-match branch
    scored as INFINITE confidence. A black frame therefore beat every real
    frame in the reference selection, and the whole archive registered against
    it at exactly 0.00 px with perfect confidence.

    That is the shape of a wrong answer worth fearing: not a warning, not a
    crash, but a flawless-looking result that measures nothing.
    """
    print("\na blank frame must not register or anchor")
    real = np.random.default_rng(0).normal(128, 30, (256, 256))
    blank = np.zeros((256, 256))
    white = np.full((256, 256), 255.0)

    got = g.phase_shift(real, shifted(real, 5, -7))
    check("a real pair still registers",
          got is not None and abs(got[0] - 5) < 0.1 and abs(got[1] + 7) < 0.1,
          got)
    check("black against a scene is refused",
          g.phase_shift(blank, real) is None)
    check("a scene against black is refused",
          g.phase_shift(real, blank) is None)
    check("black against black is refused too",
          g.phase_shift(blank, blank) is None)
    check("a whiteout is refused", g.phase_shift(white, real) is None)

    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        base = beach_scene(width=512, height=384)
        paths = []
        for index in range(8):
            frame = (np.zeros((384, 512)) if index in (2, 6)
                     else water(base, seed=index))
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
            paths.append(path)
        pick = g.best_reference(paths, downsample=2)
        check("a black frame is never the reference", pick not in (2, 6), pick)

        dates = list(pd.date_range("2024-01-07", periods=8, freq="7D",
                                   tz="UTC"))
        coarse = g.coarse_shifts(paths, dates, pick, downsample=2)
        check("the black frames are not registered",
              len(coarse) <= 6, len(coarse))
        check("and the real ones are", len(coarse) >= 5, len(coarse))
        check("nothing claims infinite confidence",
              bool(np.isfinite(coarse["confidence"]).all()),
              list(coarse["confidence"].round(1)))
        check("the reference's own perfect match is finite and top",
              coarse["confidence"].max() == g.PERFECT_MATCH,
              coarse["confidence"].max())
    finally:
        shutil.rmtree(tmp)


def test_noise_never_becomes_the_reference():
    """The failure that made the quality table read 0% in a period that worked.

    A frame of pure noise has enormous gradient magnitude, so a picker scoring
    sharpness chooses it over any real scene -- and then every genuine frame
    fails to register against the anchor, and the run reports that the record
    does not register when what does not register is the reference.
    """
    print("\na noisy frame must not become the reference")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        base = beach_scene(width=512, height=384)
        rng = np.random.default_rng(4)
        paths = []
        for index in range(8):
            if index == 5:
                frame = rng.normal(128, 60, (384, 512))   # rain, dusk, junk
            else:
                frame = water(base, seed=index)
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
            paths.append(path)
        pick = g.best_reference(paths, downsample=2)
        check("the noise frame is not chosen", pick != 5, pick)
        check("and a real frame is", pick in range(8) and pick != 5, pick)
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


def test_a_move_the_magnitude_cannot_see():
    """The step is in the displacement vector, not in its length.

    Every offset is measured from a reference frame, so the length
    |p(t) - p_ref| is blind twice over. A camera that slides from 20 px left
    of the reference to 20 px right of it never changes its DISTANCE from the
    reference, so a 40 px move reads as perfectly stable. And a reference that
    sits away from where the camera usually points turns the record's ordinary
    position into a large constant offset, which makes the choice of reference
    frame start to decide where the steps appear.
    """
    print("\na move that leaves the distance unchanged")
    when = pd.date_range("2025-09-07", periods=20, freq="7D", tz="UTC")
    dx = np.where(np.arange(20) < 10, -20.0, 20.0)      # a 40 px slide
    dy = np.zeros(20)
    frame = pd.DataFrame({"dx": dx, "dy": dy}, index=when)
    frame["offset"] = np.hypot(frame["dx"], frame["dy"])

    check("the distance from the reference never changes",
          float(frame["offset"].std()) < 1e-9, float(frame["offset"].std()))
    check("so the magnitude sees nothing",
          g.find_steps(frame["offset"], threshold=5.0) == [], "no step")

    found = g.find_steps(frame[["dx", "dy"]], threshold=5.0)
    check("the vector finds the move", len(found) == 1, len(found))
    if found:
        check("  on the right date", found[0]["date"] == when[10],
              str(found[0]["date"].date()))
        check("  at its true size", abs(found[0]["jump"] - 40.0) < 1.0,
              round(found[0]["jump"], 1))

    # And it must not invent one where the camera really is still.
    rng = np.random.default_rng(9)
    still = pd.DataFrame({"dx": rng.normal(0, 0.4, 40),
                          "dy": rng.normal(0, 0.4, 40)},
                         index=pd.date_range("2025-09-07", periods=40,
                                             freq="7D", tz="UTC"))
    check("and reports nothing on a camera that did not move",
          g.find_steps(still, threshold=5.0) == [], g.find_steps(still, 5.0))


def test_a_step_says_how_much_record_stands_behind_it():
    """Narrowing the window starves the medians rather than sharpening them.

    The window test needs only `persist` samples on each side, and with
    exactly that many each median is about one reading. Walton's 2026-01-18
    step measured 31 px across 44 frames and 11 px across 8 -- same date, same
    camera, a third of the size -- because the narrower run left three frames
    before the date. The date survives that; the magnitude does not, and a
    step standing on the minimum has to say so.
    """
    print("\nhow much record stands behind a step")
    rng = np.random.default_rng(3)

    def record(length, at, jump=30.0):
        when = pd.date_range("2025-12-07", periods=length, freq="7D", tz="UTC")
        dx = np.where(np.arange(length) < at, 0.0, jump)
        return pd.DataFrame({"dx": dx + rng.normal(0, 0.3, length),
                             "dy": rng.normal(0, 0.3, length)}, index=when)

    thin = g.find_steps(record(8, 3), threshold=5.0)
    check("a step with 3 frames on one side is found", len(thin) == 1, thin)
    if thin:
        check("  and is marked as thin", thin[0]["thin"] is True, thin[0])
        check("  with the support named", thin[0]["support"] == 3,
              thin[0]["support"])

    thick = g.find_steps(record(40, 20), threshold=5.0)
    check("a step in the middle of a long record is not marked",
          len(thick) == 1 and thick[0]["thin"] is False,
          [(s["support"], s["thin"]) for s in thick])


def test_the_two_passes_are_set_against_each_other():
    """Two epoch lists are not an answer until someone diffs them.

    The whole-frame pass and the agreeing patches measure the same camera by
    different means, so what they AGREE on is the finding. A step only one of
    them saw is a question -- and the first thing to ask is whether the other
    pass had frames on both sides of that date at all, because a run dropped
    for fog leaves a hole no step detector can fire inside.
    """
    print("\nsetting the two passes against each other")
    when = pd.date_range("2025-09-07", periods=30, freq="7D", tz="UTC")

    def step(date, jump):
        return {"date": date, "jump": jump, "before": 0.0, "after": jump}

    both = g.reconcile([step(when[10], 8.5), step(when[20], 30.0)], list(when),
                       [step(when[11], 8.8)], list(when))
    kinds = [row[0] for row in both]
    check("a step both passes saw, a week apart, is matched",
          kinds.count("both") == 1, kinds)
    check("and it carries both sizes",
          [row[3] for row in both if row[0] == "both"] == [8.8],
          [row[3] for row in both])
    check("the unmatched one is contradicted where the other pass had frames",
          "whole frame only" in kinds, kinds)

    # THE TOLERANCE FOLLOWS THE SAMPLING. Walton's daily December window paired
    # a step on 2026-02-17 with one on 2026-02-05 -- twelve days and a factor
    # of four in size apart -- because the tolerance was a fixed 14 days, which
    # is two samples of a weekly record and fourteen of a daily one.
    daily = pd.date_range("2025-12-01", periods=90, freq="1D", tz="UTC")
    apart = g.reconcile([step(daily[40], 8.0)], list(daily),
                        [step(daily[52], 33.0)], list(daily))
    check("twelve days apart in a DAILY record is two events",
          len(apart) == 2 and not any(row[0].startswith("both") for row in apart),
          [row[0] for row in apart])
    weekly = pd.date_range("2025-12-01", periods=30, freq="7D", tz="UTC")
    together = g.reconcile([step(weekly[10], 8.0)], list(weekly),
                           [step(weekly[11], 8.5)], list(weekly))
    check("but one sample apart in a WEEKLY record is one event",
          len(together) == 1 and together[0][0] == "both",
          [row[0] for row in together])

    # And agreeing that something happened is not agreeing on what.
    loud = g.reconcile([step(weekly[10], 113.0)], list(weekly),
                       [step(weekly[10], 72.0)], list(weekly))
    check("a 113 px and a 72 px reading of one move is flagged, not corroborated",
          loud[0][0] == "both, but the sizes disagree", loud[0][0])

    # Now the same unmatched step, but the patches have no frames near it.
    blind = list(when[:14])
    gapped = g.reconcile([step(when[20], 30.0)], list(when), [], blind)
    check("with no frames there it is unmeasured, not contradicted",
          gapped[0][0] == "whole frame, patches blind", gapped[0][0])

    check("and a pair of empty lists reconciles to nothing",
          g.reconcile([], list(when), [], list(when)) == [], "empty")


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

        def routes_differ(direct, steps):
            """Median disagreement between the two routes ON THE SAME PAIR.

            NOT the cumulative sum against the direct measurement: that
            compares a random walk to something that does not accumulate, so a
            perfectly still camera with per-step noise e over n frames "differs"
            by about e*sqrt(n). At 13 px over 145 frames that is ~130 px of
            pure artefact, which is what the first version of this test
            reported as a broken registration.
            """
            pairs = steps.dropna(subset=["previous"])
            pairs = pairs[pairs.index.isin(direct.index)
                          & pairs["previous"].isin(direct.index)]
            later, earlier = direct.loc[pairs.index], direct.loc[pairs["previous"]]
            return float(np.median(np.hypot(
                pairs["dx"].to_numpy()
                - (later["dx"].to_numpy() - earlier["dx"].to_numpy()),
                pairs["dy"].to_numpy()
                - (later["dy"].to_numpy() - earlier["dy"].to_numpy()))))

        gap = routes_differ(direct, steps)
        check("the two routes agree on a clean record", gap < 3.0, gap)

        # The bias this replaces. A camera that NEVER MOVES, measured with
        # ordinary independent noise on every measurement. The per-step
        # comparison must stay flat as the record lengthens, because nothing
        # accumulates; the cumulative sum must drift further and further,
        # because it is a random walk. The old test compared the walk against
        # the level and therefore punished long records for being long.
        def still_record(length, noise=6.0, seed=4):
            rng = np.random.default_rng(seed)
            when = pd.date_range("2024-01-07", periods=length, freq="7D",
                                 tz="UTC")
            level = rng.normal(0, noise, (length, 2))     # measuring a fixed camera
            step = rng.normal(0, noise, (length, 2))      # measuring zero motion
            return (pd.DataFrame({"dx": level[:, 0], "dy": level[:, 1]},
                                 index=when),
                    pd.DataFrame({"previous": [pd.NaT] + list(when[:-1]),
                                  "dx": step[:, 0], "dy": step[:, 1]},
                                 index=when))

        per_step, cumulative = {}, {}
        for length in (30, 300):
            level, step = still_record(length)
            per_step[length] = routes_differ(level, step)
            walk = np.hypot(np.cumsum(step["dx"]), np.cumsum(step["dy"]))
            cumulative[length] = float(walk.median())
        check("per-step disagreement does not grow with record length",
              abs(per_step[300] - per_step[30]) < 3.0,
              {k: round(v, 1) for k, v in per_step.items()})
        check("the cumulative sum does, on the very same still camera",
              cumulative[300] > 2 * cumulative[30],
              {k: round(v, 1) for k, v in cumulative.items()})
        check("so a long still record passes per-step and failed cumulatively",
              per_step[300] < 20.0 < cumulative[300],
              (round(per_step[300], 1), round(cumulative[300], 1)))

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


def test_the_gap_between_routes_is_a_resolution_not_a_verdict():
    """The cross-check reports the instrument's precision. It is not a pass mark.

    Two routes to the same pair each carry the per-measurement error e: the
    sequential step carries e, the difference of two direct measurements
    carries e*sqrt(2), and the gap between the routes carries e*sqrt(3) BY
    CONSTRUCTION, on data with no defect in it at all. A fixed absolute
    threshold on that gap therefore does not test the registration -- it tests
    whether the record happens to be precise enough to clear an arbitrary
    number, and a record with 14 px of error can never clear 10 px however
    honest it is. What the gap actually buys is the smallest move worth
    believing.
    """
    print("\nthe two-route gap as a precision")
    import io
    import contextlib

    def routes_differ(direct, steps):
        pairs = steps.dropna(subset=["previous"])
        later, earlier = direct.loc[pairs.index], direct.loc[pairs["previous"]]
        return float(np.median(np.hypot(
            pairs["dx"].to_numpy()
            - (later["dx"].to_numpy() - earlier["dx"].to_numpy()),
            pairs["dy"].to_numpy()
            - (later["dy"].to_numpy() - earlier["dy"].to_numpy()))))

    def still_record(error, length=400, seed=11):
        """A camera that never moves, measured with error `error` per frame."""
        rng = np.random.default_rng(seed)
        when = pd.date_range("2024-01-07", periods=length, freq="7D", tz="UTC")
        level = rng.normal(0, error, (length, 2))
        step = rng.normal(0, error, (length, 2))
        return (pd.DataFrame({"dx": level[:, 0], "dy": level[:, 1]}, index=when),
                pd.DataFrame({"previous": [pd.NaT] + list(when[:-1]),
                              "dx": step[:, 0], "dy": step[:, 1]}, index=when))

    for error in (5.0, 10.0, 14.0):
        level, step = still_record(error)
        gap = routes_differ(level, step)
        recovered, resolution = g.noise_floor(gap)
        check(f"a {error:.0f} px error shows as a {error * 1.73:.0f} px gap",
              abs(recovered - error) < 0.25 * error,
              f"gap {gap:.1f} -> error {recovered:.1f}")
        check("  and the resolution limit is 3x the error",
              abs(resolution - 3 * recovered) < 1e-6, round(resolution, 1))

    # The point of the floor: noise must not read as a step, and a move that
    # clears the floor must still be found.
    level, step = still_record(14.0)
    _, resolution = g.noise_floor(routes_differ(level, step))
    offset = np.hypot(level["dx"], level["dy"])
    check("no step survives the floor on a still camera",
          g.find_steps(offset, threshold=resolution) == [],
          f"floor {resolution:.0f} px")
    check("but the old 3 px threshold invents them",
          len(g.find_steps(offset, threshold=3.0)) > 0,
          len(g.find_steps(offset, threshold=3.0)))

    moved = offset.copy()
    moved.iloc[200:] += 2.0 * resolution
    found = g.find_steps(moved, threshold=resolution)
    check("a move well clear of the floor is still found",
          len(found) == 1 and abs((found[0]["date"] - moved.index[200]).days) <= 14,
          [str(s["date"].date()) for s in found])

    # And the words. A null result at a floor of 40 px is not "the camera was
    # stable"; it is "nothing moved by more than the smallest move visible".
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        g.report_record([], moved.index, "nowhere", 40.0, "the whole frame",
                        noise=13.3)
    text = said.getvalue()
    check("the null result names the resolution it holds at",
          "AT THIS RESOLUTION" in text and "40 px" in text, text.strip()[:60])
    check("and does not claim a smaller move was ruled out",
          "neither found nor ruled out" in text, text.strip()[-70:])


def test_the_contact_sheet_and_quality_table():
    """When the numbers cannot say why, show the frames.

    The sheet exists for one job: making it possible to see what the frames
    that failed to register have in common. So it must contain every frame,
    label each with its date, and mark the failures -- a grid that silently
    dropped the unreadable ones would hide exactly the evidence it is for.
    """
    print("\nthe contact sheet and the quality table")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        dates = list(pd.date_range("2024-01-07", periods=14, freq="30D",
                                   tz="UTC"))
        paths = []
        for index, _ in enumerate(dates):
            frame = water(scene(width=320, height=240), seed=index)
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(path)
            paths.append(path)
        out = os.path.join(tmp, "sheet.jpg")
        made = g.contact_sheet(paths, dates, out, columns=5, thumb_width=100,
                               flagged=dates[3:6])
        check("a sheet is written", made is not None and os.path.exists(out))
        with Image.open(out) as sheet:
            width, height = sheet.size
        check("it is five thumbnails wide", width == 500, width)
        check("and three rows tall for fourteen frames",
              height > 3 * 75 and height < 4 * 120, height)

        # An unreadable file must not silently shrink the sheet's job.
        broken = os.path.join(tmp, "broken.jpg")
        with open(broken, "w") as handle:
            handle.write("not an image")
        made = g.contact_sheet(paths + [broken], dates + [dates[-1]],
                               os.path.join(tmp, "sheet2.jpg"), columns=5,
                               thumb_width=100)
        check("an unreadable frame does not stop the sheet", made is not None)

        direct = pd.DataFrame(
            {"confidence": [20.0] * 7 + [2.0] * 7}, index=pd.DatetimeIndex(dates))
        sequential = pd.DataFrame(
            {"confidence": [2.0] * 7 + [20.0] * 7}, index=pd.DatetimeIndex(dates))
        table = g.registration_quality(direct, sequential, dates)
        check("the table is per quarter", len(table) >= 4, len(table))
        check("it separates the good period from the bad",
              table["direct_ok"].iloc[0] > 0.9 and table["direct_ok"].iloc[-1] < 0.1,
              list(table["direct_ok"].round(2)))
        check("and reports both passes independently",
              table["seq_ok"].iloc[0] < 0.1 and table["seq_ok"].iloc[-1] > 0.9,
              list(table["seq_ok"].round(2)))
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
        groups, unreadable = g.frame_sizes(paths, dates)
        check("both sizes are found", len(groups) == 2, sorted(groups))
        check("and each carries its frames",
              sorted(len(v) for v in groups.values()) == [3, 3],
              {k: len(v) for k, v in groups.items()})
        check("nothing is filed as unreadable", unreadable == [], unreadable)

        same, _ = g.frame_sizes(paths[:3], dates[:3])
        check("a single-resolution record reports one size", len(same) == 1,
              sorted(same))

        # A FILE THAT WILL NOT OPEN IS NOT A SECOND FRAME SIZE. One corrupt
        # download at Corolla printed under the "2 DIFFERENT FRAME SIZES"
        # banner -- the loudest claim this module makes, made falsely, about a
        # single dud JPEG.
        dud = os.path.join(tmp, "dud.jpg")
        with open(dud, "w") as fh:
            fh.write("this is not a JPEG")
        one, bad = g.frame_sizes(paths[:3] + [dud], dates[:4])
        check("a corrupt file is not counted as a frame size", len(one) == 1,
              sorted(one))
        check("...it is reported separately, so it is not an epoch boundary",
              bad == [3], bad)
    finally:
        shutil.rmtree(tmp)


def test_a_different_frame_size_is_dropped_not_cropped():
    """Cropping to a common area does not make two geometries comparable.

    Both routes used to crop a mismatched frame to the top-left corner it
    shares with the reference and register it anyway. At Walton that means
    five 1280x720 frames -- 16:9 -- cut against a 2560x1920 record, which is
    4:3. The top-left 1280x720 of the big frame is not the same scene as the
    small frame; it is a different field of view. Registering them returns a
    number with no meaning, and that number then votes in the median offset,
    in the two-route gap, and in the noise floor derived from that gap.

    A frame size change is a hard epoch boundary. Frames across one are
    counted and reported, and they belong to their own run.
    """
    print("\nframes of another size are not registered")
    tmp = tempfile.mkdtemp()
    try:
        from PIL import Image
        def scene(height, width):
            """Structure a correlator can actually lock onto."""
            frame = np.zeros((height, width), dtype=np.uint8)
            frame[:] = np.linspace(20, 90, width, dtype=np.uint8)
            for row, col in ((0.2, 0.25), (0.55, 0.6), (0.7, 0.2)):
                top, left = int(row * height), int(col * width)
                frame[top:top + height // 8, left:left + width // 8] = 235
            return frame

        dates = list(pd.date_range("2024-03-20", periods=7, freq="1D",
                                   tz="UTC"))
        paths = []
        for index in range(7):
            # Four 4:3 frames, then two 16:9 ones, then 4:3 again -- the shape
            # of the Walton span, which reverts rather than staying changed.
            big = index not in (4, 5)
            frame = scene(192, 256) if big else scene(108, 192)
            path = os.path.join(tmp, f"f{index:02d}.jpg")
            Image.fromarray(frame).save(path, quality=95)
            paths.append(path)

        direct = g.coarse_shifts(paths, dates, pick=0, downsample=1)
        check("the odd-sized frames are not measured against the reference",
              len(direct) == 5, f"{len(direct)} rows from 7 frames")
        check("...and the dates kept are the ones of the reference's size",
              list(direct.index) == [dates[i] for i in (0, 1, 2, 3, 6)],
              [str(d.date()) for d in direct.index])

        chain = g.sequential_shifts(paths, dates, downsample=1)
        # Pairs 3->4 and 5->6 span the change; 4->5 is within the small size.
        spans = [(a, b) for a, b in zip(chain.index, chain["previous"])]
        check("no pair that spans the size change is measured",
              all(list(dates).index(a) != 4 and list(dates).index(a) != 6
                  for a, _ in spans),
              [f"{a.date()}<-{b.date()}" for a, b in spans])
        check("...and the chain still runs on either side of it",
              len(chain) >= 3, f"{len(chain)} measured pairs")
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


def test_the_agreement_tolerance_is_not_the_step_threshold():
    """Asking for a small step must not make the agreement test unpassable.

    These are two different questions wearing one flag. "How big must a shift
    be before I call it a move" is a threshold on the signal, and pushing it
    below the record's own resolution is the right thing to do -- the derived
    floor takes over and the run says so. "How close must two patches track
    before I believe they are on one rigid body" is a tolerance on measurement
    ERROR, and pushing THAT down does not make the test stricter. It makes it
    impossible, and the run then reports "nothing in this frame is rigid" as a
    finding about the camera when it is a finding about the flag.

    The Walton density run hit exactly this: --step-px 0.5, chosen so the
    coarse route's measured resolution would bind, also demanded half-pixel
    agreement from twelve patches over 336 dates and reported that none agreed.
    """
    # Four patches on one rigid body, tracking together to within about 2 px
    # of scatter, plus two patches on nothing.
    rng = np.random.default_rng(11)
    dates = pd.date_range("2023-01-01", periods=40, freq="7D")
    truth_dx = np.linspace(0, 6, len(dates))
    rows = []
    for name in ("a", "b", "c", "d"):
        for date, dx in zip(dates, truth_dx):
            rows.append({"date": date, "feature": name,
                         "dx": dx + rng.normal(0, 1.0),
                         "dy": rng.normal(0, 1.0)})
    for name in ("x", "y"):
        for date in dates:
            rows.append({"date": date, "feature": name,
                         "dx": rng.normal(0, 40), "dy": rng.normal(0, 40)})
    frame = pd.DataFrame(rows)

    kept, _, _ = g.agreeing_features(frame, tolerance=0.5)
    check("a half-pixel tolerance finds no rigid body in a rigid frame",
          len(kept) == 0, f"kept {kept}")

    kept, rejected, _ = g.agreeing_features(frame, tolerance=g.AGREE_PX)
    check(f"...and the same frame at the {g.AGREE_PX:.0f} px tolerance "
          "recovers all four",
          sorted(kept) == ["a", "b", "c", "d"], f"kept {sorted(kept)}")
    check("...and the two patches on nothing are still thrown out",
          sorted(rejected) == ["x", "y"], f"rejected {sorted(rejected)}")

    # The two knobs are separately settable, so a small --step-px can no longer
    # reach the agreement test at all. The parser is built inside main(), so
    # the only honest way to ask what flags exist is to ask the CLI.
    helptext = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "check_camera_geometry.py"), "--help"],
        capture_output=True, text=True).stdout
    check("--agree-px exists as its own flag", "--agree-px" in helptext)
    check("...and it says in the help that it is not a step size",
          "NOT a step size" in " ".join(helptext.split()))

    # And the floor: a value finer than the record can measure is raised to it,
    # while a record that could not measure itself leaves the given value alone.
    check("a tolerance finer than the record is raised to the record",
          g.not_finer_than_the_record(0.5, 48.0) == 48.0)
    check("...and one coarser than the record is left alone",
          g.not_finer_than_the_record(10.0, 4.0) == 10.0)
    check("...and an unmeasured record cannot raise anything",
          g.not_finer_than_the_record(3.0, float("nan")) == 3.0)


def test_a_resolution_that_tracks_the_window_is_not_a_resolution():
    """A search box that bounds the answer must not be reported as precision.

    --max-shift is a prior about where the correlation peak is. If the peak is
    really on the scene, changing the window changes only how many frames get
    overruled. If there is no dominant peak, the best position inside the box
    is found near the box, and the two-route gap then measures the WIDTH OF
    THE BOX rather than the error of either route.

    Walton made this concrete and nearly got away with it. At --max-shift 150
    the resolution came out 145 px -- 0.97x the window -- and the largest
    offset sat on the wall at 149.88. Re-running at 450 to test the prior
    dropped the overruled count from 55% to 6%, which reads exactly like a
    prior being fixed, and the resolution came out 376 px: 0.84x the new
    window, largest offset 98.9% of the way to the box corner. Forty-three
    epochs, every one of them the search box.

    The overruled count alone cannot catch this -- it got BETTER while the
    measurement got worse -- so the ratio is tested directly.
    """
    for window, gap in ((150.0, 83.9), (450.0, 217.2)):
        _, resolution = g.noise_floor(gap)
        check(f"a {window:.0f} px window returning a {resolution:.0f} px "
              f"resolution is refused",
              resolution >= g.WANDER_SHARE * window,
              f"{resolution / window:.2f} x the window, "
              f"limit {g.WANDER_SHARE}")

    # A record that genuinely registers is not caught by it: 4 px of gap in a
    # 150 px window is a real measurement and must survive.
    _, resolution = g.noise_floor(4.0)
    check("...and a record that really registers is not",
          resolution < g.WANDER_SHARE * 150.0,
          f"{resolution:.1f} px resolution in a 150 px window")

    # The tolerance that floors the patch agreement test must NOT be raised by
    # a resolution like that -- a test relaxed to 380 px passes by being asked
    # nothing, which is how 6 of 6 patches sitting 39-66 px apart were kept.
    check("a refused resolution cannot relax the agreement test",
          g.not_finer_than_the_record(g.AGREE_PX, float("nan")) == g.AGREE_PX)

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
                 test_the_reference_is_the_one_the_record_matches,
                 test_noise_never_becomes_the_reference,
                 test_a_blank_frame_cannot_register_or_anchor,
                 test_the_search_window_is_a_prior_that_reports_itself,
                 test_fog_is_measured_and_excluded,
                 test_water_is_rejected_under_changing_light,
                 test_water_wins_a_median_split_when_it_covers_most_of_the_frame,
                 test_a_planted_step_is_found_on_the_right_date,
                 test_a_stable_record_reports_no_step,
                 test_the_two_passes_are_set_against_each_other,
                 test_a_step_says_how_much_record_stands_behind_it,
                 test_a_move_the_magnitude_cannot_see,
                 test_disagreement_is_measured_as_a_vector,
                 test_the_agreeing_group_is_recovered_from_noise,
                 test_a_frame_with_nothing_rigid_in_it_returns_nothing,
                 test_features_with_no_overlapping_dates_are_not_linked,
                 test_a_move_bigger_than_a_patch,
                 test_frame_to_frame_agrees_with_the_reference,
                 test_the_gap_between_routes_is_a_resolution_not_a_verdict,
                 test_the_text_chart,
                 test_the_contact_sheet_and_quality_table,
                 test_the_survey_tiles_the_whole_frame,
                 test_a_large_survey_still_finds_the_rigid_block,
                 test_a_changed_frame_size_is_reported,
                 test_a_different_frame_size_is_dropped_not_cropped,
                 test_epochs_count_stills_not_just_samples,
                 test_the_agreement_tolerance_is_not_the_step_threshold,
                 test_a_resolution_that_tracks_the_window_is_not_a_resolution):
        test()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}"))
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
