#!/usr/bin/env python3
"""Offline checks for check_geometry_strict.py. No network, no real frames.

Every test here exists because a specific way of being wrong was available.
The synthetic scenes are built so that the WRONG method gives a confident
wrong answer -- a water region that moves differently from the land, a record
with a gap in it, a homography carrying rotation and zoom that a shift-only
method reports as perfectly stable. A test that only passes on easy data
proves nothing about a method whose whole purpose is to survive hard data.

    python test_strict_offline.py
"""

import math
import sys
import os

import numpy as np
import pandas as pd

import check_geometry_strict as strict


PASSED = []
FAILED = []


def check(name, condition, detail=""):
    (PASSED if condition else FAILED).append(name)
    mark = "ok  " if condition else "FAIL"
    print(f"  {mark} {name}" + (f"   [{detail}]" if detail else ""))


def texture(shape, seed, blur=3):
    """Deterministic blob texture that SIFT and phase correlation can both use."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, shape)
    try:
        import cv2
        base = cv2.GaussianBlur(base, (0, 0), blur)
    except ImportError:
        for _ in range(4):
            base = (base + np.roll(base, 1, 0) + np.roll(base, -1, 0)
                    + np.roll(base, 1, 1) + np.roll(base, -1, 1)) / 5.0
    base -= base.min()
    return 40 + 180 * base / max(base.max(), 1e-9)


def two_layer_scene(shape, land_top, land_shift, water_shift, seed=7):
    """A frame that is mostly water, where water and land move DIFFERENTLY.

    This is the Walton failure made reproducible: the honest answer is the
    land's shift, and an unmasked method that lets the majority of the pixels
    vote returns the water's.
    """
    height, width = shape
    pad = 80
    big = texture((height + 2 * pad, width + 2 * pad), seed)
    wet = texture((height + 2 * pad, width + 2 * pad), seed + 1)

    def cut(source, shift):
        # MINUS, and the sign is the whole point. Sliding the sampling window
        # right moves the CONTENT left, so `shift` would otherwise mean the
        # opposite of the displacement the code reports and every expectation
        # in these tests would be negated -- which is precisely the confusion
        # phase_shift's docstring warns about. Here `shift` means "the scene
        # has moved by this much", matching what the code returns.
        dy, dx = shift
        return source[pad - dy: pad - dy + height, pad - dx: pad - dx + width]

    frame = np.empty(shape)
    frame[land_top:, :] = cut(big, land_shift)[land_top:, :]
    frame[:land_top, :] = cut(wet, water_shift)[:land_top, :]
    return frame


# ---------------------------------------------------------------------------

def test_polygon_mask_is_the_region_it_names():
    mask = strict.polygon_mask((100, 200),
                               [(0.25, 0.5), (0.75, 0.5), (0.75, 1.0), (0.25, 1.0)])
    check("a rectangle polygon masks that rectangle",
          mask[75, 100] and not mask[25, 100] and not mask[75, 10],
          f"{mask.mean():.3f} of the frame")
    check("its area is the area of the rectangle",
          abs(mask.mean() - 0.25) < 0.02, f"{mask.mean():.3f} vs 0.25")


def test_drop_polygons_are_cut_back_out():
    spec = {"keep": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]],
            "drop": [[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]]}
    mask = strict.build_mask((80, 80), spec)
    check("a drop polygon removes its region from the mask",
          abs(mask.mean() - 0.5) < 0.03 and not mask[40, 10] and mask[40, 70],
          f"{mask.mean():.3f} left")


def test_the_feathered_mask_has_no_cliff():
    mask = np.zeros((200, 200), dtype=bool)
    mask[100:, :] = True
    soft = strict.feathered(mask, radius=10)
    edge = soft[90:110, 100]
    check("feathering leaves the interior alone", soft[180, 100] > 0.98,
          f"{soft[180, 100]:.3f}")
    check("feathering leaves the exterior alone", soft[20, 100] < 0.02,
          f"{soft[20, 100]:.3f}")
    check("feathering removes the cliff at the boundary",
          float(np.max(np.abs(np.diff(edge)))) < 0.2,
          f"biggest single-pixel step {np.max(np.abs(np.diff(edge))):.3f}")


def test_masking_finds_the_land_where_not_masking_finds_the_water():
    """THE reason the mask exists. Both answers are confident; one is wrong."""
    shape = (480, 640)
    land_top = int(0.70 * shape[0])          # 70% water, as at a beach camera
    reference = two_layer_scene(shape, land_top, (0, 0), (0, 0))
    moved = two_layer_scene(shape, land_top, (2, 5), (0, 30))

    unmasked = strict.geo.phase_shift(reference, moved, max_shift=60)
    mask = np.zeros(shape, dtype=bool)
    mask[land_top:, :] = True
    masked = strict.masked_phase(reference, moved, strict.feathered(mask, 8),
                                 max_shift=60)

    check("the land mask recovers the land's shift",
          masked is not None and abs(masked[1] - 5) < 1.5
          and abs(masked[0] - 2) < 1.5,
          "none" if masked is None else f"dx={masked[1]:.2f} dy={masked[0]:.2f}")
    drawn_to_water = (unmasked is not None and abs(unmasked[1] - 30) < 5)
    check("WITHOUT the mask the water wins, confidently",
          drawn_to_water and unmasked[2] > strict.MIN_CONFIDENCE,
          "none" if unmasked is None
          else f"dx={unmasked[1]:.2f} conf={unmasked[2]:.0f}")


def test_the_mask_audit_catches_a_tide_one_frame_could_not_show():
    """A mask is a claim about every frame; a preview shows one.

    The failure this exists to catch is invisible to the preview by
    construction: a mask drawn at low tide sits over dry sand in the frame it
    was drawn on, and over swash on every spring high in the record. Both
    masks here pass a single-frame look; only one survives the archive.
    """
    import tempfile
    import shutil
    import contextlib
    import io
    from PIL import Image

    height, width = 400, 600
    folder = tempfile.mkdtemp()
    try:
        paths, dates = [], []
        for index in range(60):
            waterline = 220 + int(40 * np.sin(index / 3.0))   # tide swing
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:waterline] = (70, 95, 110)      # water: R-B = -40
            frame[waterline:] = (205, 180, 140)    # sand:  R-B = +65
            path = os.path.join(folder, f"f{index:03d}.jpg")
            Image.fromarray(frame).save(path, quality=95)
            paths.append(path)
            dates.append(pd.Timestamp("2024-01-01", tz="UTC")
                         + pd.Timedelta(days=index))

        def audit(edge):
            spec = {"keep": [[(0.0, edge), (1.0, edge), (1.0, 1.0), (0.0, 1.0)]],
                    "drop": [], "note": "test"}
            mask = strict.build_mask((height, width), spec)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                got = strict.audit_mask(paths, dates, mask, spec, downsample=2)
            return got, buffer.getvalue()

        # 0.55 is 220 px -- the waterline in the calmest frame, and under
        # water for a sixth of the record.
        low, low_said = audit(0.55)
        check("a mask drawn at low tide is caught taking water",
              low["often_wet_share"] > 0.05,
              f"{low['often_wet_share']:.1%} of the mask is wet in a quarter "
              f"of frames")
        check("...and the worst frames are named, so they can be looked at",
              "2024-01-" in low_said or "2024-02-" in low_said,
              low_said.strip().splitlines()[-1].strip()[:44])

        # 0.70 is 280 px, clear of the highest water in the record.
        safe, safe_said = audit(0.70)
        check("a mask with margin is not",
              safe["often_wet_share"] == 0 and safe["worst_frame_share"] < 0.01,
              f"{safe['often_wet_share']:.1%} often wet, "
              f"worst frame {safe['worst_frame_share']:.1%}")
        check("...and a clean audit does not print a table of zeroes",
              "no frame in the record put water inside this mask" in safe_said,
              safe_said.strip().splitlines()[-1].strip()[:52])

        # The median is what separates a bad mask from a foggy record: fog
        # whitens every frame at once, a bad mask wets a fraction of them.
        check("the median frame share is reported, not just the worst",
              low["median_frame_share"] < low["worst_frame_share"],
              f"median {low['median_frame_share']:.1%} vs worst "
              f"{low['worst_frame_share']:.1%}")
    finally:
        shutil.rmtree(folder)


def test_the_declared_masks_are_usable_as_declared():
    """A mask that builds to nothing is worse than no mask: it runs."""
    for slug, spec in strict.MASKS.items():
        mask = strict.build_mask((1520, 2688), spec)
        share = mask.mean()
        check(f"{slug[:34]}: leaves a workable land region",
              0.05 < share < 0.85, f"{share:.1%} of the frame")
        cells, size = strict.land_cells(mask)
        check(f"{slug[:34]}: the survey can tile it",
              len(cells) >= strict.MIN_CLUSTER, f"{len(cells)} cells of {size} px")
        check(f"{slug[:34]}: says where it came from",
              len(spec.get("note", "")) > 20, spec.get("note", "")[:40] + "...")


def test_a_thin_land_band_caps_its_range_rather_than_lying():
    """The limit is range, not truth, and it announces itself.

    This test used to assert that a 72-row band recovers ~0 across its short
    axis "with ordinary confidence" -- a silent lie, the worst failure a
    stability check can have. That reading came from cropping to the FEATHERED
    mask's bounding box, which fed the correlator a ramp instead of an edge;
    once masked_phase was fixed to crop to the binary box, it stopped
    reproducing at any band height down to 16 rows.

    What is left is real and much narrower: shifting a band across its short
    axis destroys overlap, so a thin band can only see displacements out to
    about half its height, and past that confidence has already fallen through
    MIN_CONFIDENCE. The frame drops out of the record instead of posing as
    stable. That distinction decides how a gap in route 2 gets read, so it is
    measured here rather than assumed.
    """
    big = texture((600, 480), 7)
    thin_h, thick_h = 72, 300

    def band(height, top, dy):
        first = big[top:top + height, 100:400]
        second = big[top - dy:top - dy + height, 95:395]    # +dy down, +5 right
        return strict.geo.phase_shift(first, second,
                                      max_shift=min(60, height // 2))

    small_thin = band(thin_h, 300, 2)
    small_thick = band(thick_h, 200, 2)
    check("a thin band reads a small shift as exactly as a deep one",
          abs(small_thin[0] - 2) < 0.1 and abs(small_thin[1] - 5) < 0.1,
          f"72 rows: dy={small_thin[0]:.2f} dx={small_thin[1]:.2f} "
          f"(300 rows: dy={small_thick[0]:.2f})")

    # Inside the ceiling -- half the band -- it stays right.
    inside = band(thin_h, 300, 30)
    check("...and stays right out to nearly half its height",
          abs(inside[0] - 30) < 0.5 and inside[2] > strict.MIN_CONFIDENCE,
          f"dy={inside[0]:.2f} (truth 30) conf={inside[2]:.0f}")

    # Past it the answer IS wrong -- and that is the case that matters.
    beyond = band(thin_h, 300, 40)
    check("past the ceiling the answer is wrong",
          abs(beyond[0] - 40) > 5, f"dy={beyond[0]:.2f} (truth 40)")
    check("...but confidence has already collapsed, so it is REJECTED "
          "rather than believed",
          beyond[2] < strict.MIN_CONFIDENCE,
          f"conf={beyond[2]:.0f} < {strict.MIN_CONFIDENCE} threshold")

    deep = band(thick_h, 200, 40)
    check("a deep band has no such ceiling at the same displacement",
          abs(deep[0] - 40) < 0.5 and deep[2] > strict.MIN_CONFIDENCE,
          f"dy={deep[0]:.2f} conf={deep[2]:.0f}")
    check("the threshold in the code is at least twice the shift we care about",
          strict.THIN_BAND_PX >= 2 * strict.AGREE_PX * 10,
          f"{strict.THIN_BAND_PX} px, ceiling {strict.THIN_BAND_PX // 2} px")


def test_a_thin_mask_is_warned_about(capsys=None):
    import io
    import contextlib
    mask = np.zeros((600, 800), dtype=bool)
    mask[540:, :] = True                       # a 60-row strip
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        strict.describe_mask(mask, {"note": "test"}, mask.shape)
    said = buffer.getvalue()
    check("a thin mask draws a warning naming the axis",
          "spans only 60 rows" in said, said.strip().splitlines()[-1][:60])


def test_decompose_recovers_a_transform_it_was_given():
    height, width = 400, 600
    angle = math.radians(1.5)
    scale = 1.03
    shift = np.array([12.0, -7.0])
    centre = np.array([width / 2.0, height / 2.0])
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]]) * scale
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = centre + shift - rotation @ centre

    got = strict.decompose(matrix, (height, width))
    check("decompose recovers translation",
          abs(got["dx"] - 12.0) < 1e-6 and abs(got["dy"] + 7.0) < 1e-6,
          f"dx={got['dx']:.4f} dy={got['dy']:.4f}")
    check("decompose recovers rotation",
          abs(got["rotation"] - 1.5) < 1e-6, f"{got['rotation']:.5f} deg")
    check("decompose recovers scale",
          abs(got["scale"] - 1.03) < 1e-9, f"{got['scale']:.6f}")
    check("a similarity reports no anisotropy and no bend",
          abs(got["aniso"] - 1.0) < 1e-9 and got["bend"] < 1e-6,
          f"aniso={got['aniso']:.6f} bend={got['bend']:.2e}")


def test_a_rotation_a_shift_only_method_calls_stability():
    """Rotation about the frame centre moves the centre pixel by ZERO.

    Phase correlation therefore reports 0 px and full confidence for a camera
    that has visibly turned. This is not a subtle failure mode; it is the
    whole reason the primary route fits a homography.
    """
    height, width = 400, 600
    angle = math.radians(2.0)
    centre = np.array([width / 2.0, height / 2.0])
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]])
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = centre - rotation @ centre

    got = strict.decompose(matrix, (height, width))
    check("a pure rotation shows zero translation at the centre",
          abs(got["dx"]) < 1e-9 and abs(got["dy"]) < 1e-9,
          f"dx={got['dx']:.2e}")
    check("...and the homography route still sees the 2 deg",
          abs(got["rotation"] - 2.0) < 1e-6, f"{got['rotation']:.4f} deg")
    corner_travel = 2 * math.sin(angle / 2) * math.hypot(*centre)
    check("...which had moved the corners by a long way",
          corner_travel > 10, f"{corner_travel:.1f} px at the corner")


def test_the_homography_route_recovers_a_warp_through_the_mask():
    try:
        import cv2
    except ImportError:
        print("  skip cv2 tests (OpenCV not installed)")
        return
    shape = (300, 420)
    land_top = int(0.55 * shape[0])
    mask = np.zeros(shape, dtype=bool)
    mask[land_top:, :] = True

    reference = two_layer_scene(shape, land_top, (0, 0), (0, 0), seed=11)
    angle = math.radians(1.0)
    centre = np.array([shape[1] / 2.0, shape[0] / 2.0])
    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]]) * 1.01
    truth = np.eye(3)
    truth[:2, :2] = rotation
    truth[:2, 2] = centre + np.array([6.0, 3.0]) - rotation @ centre
    moved = cv2.warpPerspective(reference, truth, (shape[1], shape[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)

    detector, norm = strict.make_detector("sift", cv2)
    ref_kp, ref_desc = strict.frame_features(reference, mask, detector, cv2)
    kp, desc = strict.frame_features(moved, mask, detector, cv2)
    fit, why = strict.fit_homography(ref_kp, ref_desc, kp, desc, norm, cv2)
    if fit is None:
        check("the homography route fits a known warp", False, str(why))
        return
    got = strict.decompose(fit["H"], shape)
    check("the homography route fits a known warp", True,
          f"{fit['inliers']}/{fit['matches']} inliers, "
          f"residual {fit['residual']:.2f} px")
    check("...recovering the translation", abs(got["dx"] - 6) < 1.0
          and abs(got["dy"] - 3) < 1.0,
          f"dx={got['dx']:.2f} dy={got['dy']:.2f}")
    check("...recovering the rotation", abs(got["rotation"] - 1.0) < 0.2,
          f"{got['rotation']:.3f} deg")
    check("...recovering the zoom", abs(got["scale"] - 1.01) < 0.01,
          f"{got['scale']:.4f}")
    check("...and detecting no features in the water",
          all(point.pt[1] >= land_top - 2 for point in ref_kp),
          f"{len(ref_kp)} keypoints, all below row {land_top}")


def test_land_cells_stay_inside_the_mask():
    mask = np.zeros((512, 512), dtype=bool)
    mask[256:, 128:384] = True
    cells, used = strict.land_cells(mask, size=128)
    check("survey cells lie inside the land mask",
          cells and all(mask[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]].mean()
                        >= 0.9 for c in cells),
          f"{len(cells)} cells of {used} px")
    check("survey cells do not stray into the water",
          all(c["y"] >= 256 for c in cells))
    check("survey cells do not overlap each other",
          len({(c["x"], c["y"]) for c in cells}) == len(cells))


def test_the_survey_grid_is_anchored_to_the_land_not_the_frame():
    """The band that made the decisive test silently not run.

    Land at rows 298-479 of a 480-row frame is 182 rows -- room for a 128 px
    cell with 54 rows to spare. Anchored at the frame origin the only
    candidate row starts at 384 and runs off the bottom, so the survey
    reported zero cells and the strongest check available was skipped without
    anything being obviously wrong.
    """
    mask = np.zeros((480, 640), dtype=bool)
    mask[298:, :] = True
    origin_anchored = [y for y in range(0, 480 - 128 + 1, 128)
                       if mask[y:y + 128, 0:128].mean() >= 0.9]
    check("the frame-anchored grid would have found nothing",
          not origin_anchored, f"{len(origin_anchored)} rows fit")
    cells, used = strict.land_cells(mask, size=128)
    check("the mask-anchored grid finds cells in the same band",
          len(cells) >= strict.MIN_CLUSTER,
          f"{len(cells)} cells of {used} px")
    check("...and they all sit on land",
          all(mask[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]].all()
              for c in cells))


def test_a_band_too_narrow_for_big_cells_drops_the_cell_size():
    mask = np.zeros((480, 640), dtype=bool)
    mask[400:, :] = True                      # only 80 rows of land
    cells, used = strict.land_cells(mask, size=128)
    check("a narrow band falls back to a smaller cell rather than giving up",
          len(cells) >= strict.MIN_CLUSTER and used < 128,
          f"{len(cells)} cells of {used} px")


def test_the_noise_floor_does_not_measure_the_gap_in_the_record():
    """The bug this guards: dropping weak frames and THEN differencing.

    With a fortnight of fog in the middle, the surviving rows either side are
    two weeks apart. Differencing those and comparing against a one-day step
    measures the camera's real motion across the fog and books it as
    measurement noise -- inflating the floor until nothing can be resolved.
    """
    days = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    drift = np.arange(40) * 4.0            # a steady 4 px/day real motion
    frame = pd.DataFrame({"dx": drift, "dy": np.zeros(40),
                          "step_dx": np.full(40, 4.0),
                          "step_dy": np.zeros(40)}, index=days)
    frame.loc[days[10:24], ["dx", "dy"]] = np.nan       # a fortnight of fog

    got = strict.within_route_noise(frame, list(days))
    check("a gap in the record does not inflate the noise floor",
          got is not None and got["noise"] < 0.5,
          "none" if got is None else f"±{got['noise']:.3f} px from "
                                     f"{got['pairs']} pairs")


def test_the_noise_floor_recovers_an_error_it_was_given():
    rng = np.random.default_rng(3)
    for error in (0.5, 2.0, 8.0):
        days = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
        true = np.cumsum(rng.normal(0, 1.0, 300))
        noise_x = rng.normal(0, error, 300)
        noise_y = rng.normal(0, error, 300)
        frame = pd.DataFrame(
            {"dx": true + noise_x, "dy": noise_y,
             "step_dx": np.diff(true, prepend=true[0]) + rng.normal(0, error, 300),
             "step_dy": rng.normal(0, error, 300)}, index=days)
        got = strict.within_route_noise(frame, list(days))
        # The median of a 2-D magnitude is about 1.18 sigma, so the recovered
        # figure is the right size rather than the exact sigma; a factor-of-two
        # band is what this construction can honestly claim.
        ratio = got["noise"] / error
        check(f"the within-route floor recovers e={error} px",
              0.5 < ratio < 2.0, f"got ±{got['noise']:.2f} px (x{ratio:.2f})")


def test_between_route_gap_uses_sqrt_two_not_sqrt_three():
    """Two direct measurements differ by sqrt(2)*e; a step against a difference
    of two differs by sqrt(3)*e. Using the wrong constant understates the
    error by 22% and promises a resolution the data does not support."""
    days = pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC")
    a = pd.DataFrame({"dx": np.zeros(50), "dy": np.zeros(50)}, index=days)
    b = pd.DataFrame({"dx": np.full(50, 10.0), "dy": np.zeros(50)}, index=days)
    cross = strict.between_route_gap(a, b)
    check("the between-route gap is the distance between the routes",
          abs(cross["gap"] - 10.0) < 1e-9, f"{cross['gap']:.3f} px")
    limit = strict.resolution_from([], cross)
    check("...and converts to error with sqrt(2), not sqrt(3)",
          abs(limit["noise"] - 10.0 / math.sqrt(2)) < 1e-9,
          f"±{limit['noise']:.3f} px (sqrt3 would give "
          f"{10.0 / math.sqrt(3):.3f})")


def test_the_resolution_takes_the_worst_estimate_not_the_kindest():
    parts = [{"noise": 0.3}, {"noise": 4.0}]
    limit = strict.resolution_from(parts, {"gap": 1.0})
    check("the resolution is set by the worst route, not the best",
          abs(limit["noise"] - 4.0) < 1e-9, f"±{limit['noise']:.2f} px")
    check("...and the step threshold is three times it",
          abs(limit["resolution"] - 12.0) < 1e-9,
          f"{limit['resolution']:.1f} px")


def test_a_step_is_confirmed_only_when_both_routes_see_it():
    days = list(pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"))
    shared = days[30]
    homog = [{"date": shared, "jump": 20.0},
             {"date": days[45], "jump": 18.0}]
    phase = [{"date": days[31], "jump": 19.0}]
    rows = strict.compare_routes(homog, days, phase, days)
    labels = {row[1]: row[0] for row in rows}
    check("a step both routes found is labelled as such",
          labels[shared].startswith("both routes"), labels[shared])
    check("a step only one route found is not",
          labels[days[45]].startswith("homography only"), labels[days[45]])


def test_two_routes_that_disagree_on_size_are_not_called_agreement():
    days = list(pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC"))
    rows = strict.compare_routes([{"date": days[20], "jump": 113.0}], days,
                                 [{"date": days[20], "jump": 72.0}], days)
    check("113 px and 72 px is not corroboration",
          rows[0][0] == "both routes, but the sizes disagree", rows[0][0])


def test_an_unmeasured_stretch_is_not_a_contradiction():
    days = list(pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"))
    rows = strict.compare_routes([{"date": days[2], "jump": 30.0}], days,
                                 [], days[40:])
    check("a route with no frames there is blind, not contradicting",
          "had no frames there" in rows[0][0], rows[0][0])


def test_sampled_gaps_finds_the_missing_days():
    days = list(pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")) + \
           list(pd.date_range("2024-01-25", periods=10, freq="D", tz="UTC"))
    runs = strict.sampled_gaps(days, 1)
    check("a fortnight of missing sample days is reported as one gap",
          len(runs) == 1 and len(runs[0]) == 14,
          f"{len(runs)} gaps, longest {max((len(r) for r in runs), default=0)} days")


def test_registration_refuses_to_run_without_a_declared_mask():
    check("an undeclared camera has no mask, so registration cannot start",
          strict.mask_spec("some-camera-nobody-has-looked-at") is None)
    override = {"keep": [[(0, 0), (1, 0), (1, 1), (0, 1)]], "drop": [],
                "note": "test"}
    check("...and an explicit override is honoured",
          strict.mask_spec("anything", override) is override)


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    print(f"check_geometry_strict: {len(tests)} offline checks\n")
    for test in tests:
        print(test.__name__.replace("test_", "").replace("_", " ") + ":")
        test()
        print()
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name in FAILED:
            print(f"  FAILED: {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
