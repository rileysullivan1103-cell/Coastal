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
    out = image.copy()
    rng = np.random.default_rng(seed)
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
                 test_roi_preview_is_written,
                 test_a_planted_step_is_found_on_the_right_date,
                 test_a_stable_record_reports_no_step,
                 test_epochs_count_stills_not_just_samples):
        test()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}"))
    raise SystemExit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
