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


def test_a_thin_land_band_loses_its_short_axis_and_says_so():
    """Not a bug to fix, a limit to declare.

    A beach camera's land is a strip along the bottom of the frame. Phase
    correlation over a strip recovers displacement along the strip and loses
    it ACROSS the strip, returning ~0 with ordinary confidence -- which is
    indistinguishable from a camera that did not move. Measured here so the
    threshold in the code is a number from data rather than a guess.
    """
    big = texture((400, 480), 7)
    outcomes = {}
    for height in (72, 200):
        first = big[100:100 + height, 100:400]
        second = big[98:98 + height, 95:395]        # +2 down, +5 right
        got = strict.geo.phase_shift(first, second, max_shift=40)
        outcomes[height] = got
    thin, thick = outcomes[72], outcomes[200]
    check("a 200-row band recovers both axes",
          abs(thick[0] - 2) < 0.2 and abs(thick[1] - 5) < 0.2,
          f"dy={thick[0]:.2f} dx={thick[1]:.2f}")
    check("a 72-row band keeps the long axis",
          abs(thin[1] - 5) < 0.2, f"dx={thin[1]:.2f}")
    check("...and LOSES the short one, at ordinary confidence",
          abs(thin[0] - 2) > 1.0 and thin[2] > strict.MIN_CONFIDENCE,
          f"dy={thin[0]:.2f} (truth 2.0), conf={thin[2]:.0f}")
    check("the threshold in the code sits above the failing width",
          72 < strict.THIN_BAND_PX <= 200, f"{strict.THIN_BAND_PX} px")


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
