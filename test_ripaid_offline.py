"""Offline checks for load_ripaid.py and the filename-timestamp probe.

The two things that would quietly corrupt this dataset are treating a frame a
person annotated as empty as MISSING rather than as an observed zero, and
averaging rip orientations as if they were arrows rather than lines.

    python test_ripaid_offline.py
"""

import contextlib
import io
import math
import json
import os
import sys
import tempfile

import load_ripaid as rip
import probe_rip_dataset as probe

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


def test_axial_mean():
    print("rip orientation is an axis, not an arrow")
    # 10 and 190 degrees are the SAME line. An arithmetic mean says 100, which
    # is perpendicular to both -- the exact error this function exists to avoid.
    got = rip.axial_mean_deg([10.0, 190.0])
    check("10 and 190 average to 10, not 100", abs(got - 10) < 1e-6, f"{got}")
    got = rip.axial_mean_deg([350.0, 10.0])
    check("wraps around 0 correctly", min(abs(got - 0), abs(got - 180)) < 1e-6, f"{got}")
    got = rip.axial_mean_deg([80.0, 100.0])
    check("ordinary case still works", abs(got - 90) < 1e-6, f"{got}")
    check("empty input is NaN", rip.axial_mean_deg([]) != rip.axial_mean_deg([]))
    check("all-None is NaN", rip.axial_mean_deg([None, None]) != rip.axial_mean_deg([None, None]))
    check("result is folded into 0-180", 0 <= rip.axial_mean_deg([200.0]) < 180,
          str(rip.axial_mean_deg([200.0])))


def _coco(images, annotations):
    return {"categories": [{"id": 1, "name": "rip_current"},
                           {"id": 2, "name": "doubt"}],
            "images": images, "annotations": annotations}


def test_observed_zeros():
    print("a frame annotated as empty is a zero, not a gap")
    payload = _coco(
        images=[
            {"id": 1, "file_name": "clm_s_01_2011-05-21-11-00.png"},
            {"id": 2, "file_name": "clm_s_01_2011-05-21-11-30.png"},  # no annotation
            {"id": 3, "file_name": "snb_s_02_2012-06-01-09-00.png"},
        ],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "area": 100.0,
             "attributes": {"rotation": 10.0}},
            {"id": 2, "image_id": 1, "category_id": 2, "area": 50.0},   # doubt
            {"id": 3, "image_id": 3, "category_id": 1, "area": 400.0,
             "attributes": {"rotation": 190.0}},
        ])
    frames = rip.build_frames(payload)
    check("every image becomes a frame", len(frames) == 3, str(len(frames)))

    empty = frames[frames["file_name"].str.contains("11-30")].iloc[0]
    check("the unannotated frame is present", empty is not None)
    check("it is an observed zero, not NaN", empty["detected"] == False
          and empty["n_rip"] == 0 and empty["area_max"] == 0.0)

    first = frames[frames["file_name"].str.contains("11-00")].iloc[0]
    check("doubt is counted separately from rip", first["n_rip"] == 1
          and first["n_doubt"] == 1, f"{first['n_rip']}/{first['n_doubt']}")
    check("doubt does not inflate the rip area", first["area_max"] == 100.0,
          str(first["area_max"]))

    check("site parsed", set(frames["site"]) == {"clm", "snb"}, str(set(frames["site"])))
    check("camera parsed", "clm_s_01" in set(frames["camera"]))
    check("timestamp parsed to the minute",
          str(frames["timestamp"].min()) == "2011-05-21 11:00:00+00:00",
          str(frames["timestamp"].min()))


def test_doubt_only_frames():
    print("frames carrying only doubt")
    payload = _coco(
        images=[{"id": 1, "file_name": "clm_s_01_2011-05-21-11-00.png"}],
        annotations=[{"id": 1, "image_id": 1, "category_id": 2, "area": 50.0}])
    frames = rip.build_frames(payload)
    row = frames.iloc[0]
    check("counted as no rip", not row["detected"] and row["n_rip"] == 0)
    check("but the doubt is retained", row["n_doubt"] == 1)


def test_hourly():
    print("hourly rollup")
    payload = _coco(
        images=[{"id": 1, "file_name": "clm_s_01_2011-05-21-11-00.png"},
                {"id": 2, "file_name": "clm_s_01_2011-05-21-11-40.png"},
                {"id": 3, "file_name": "clm_s_01_2011-05-21-12-00.png"}],
        annotations=[{"id": 1, "image_id": 1, "category_id": 1, "area": 100.0,
                      "attributes": {"rotation": 10.0}}])
    hourly = rip.to_hourly(rip.build_frames(payload))
    check("two hours", len(hourly) == 2, str(len(hourly)))
    first = hourly.iloc[0]
    check("frames counted", first["frames"] == 2, str(first["frames"]))
    check("rate is 1 of 2", abs(first["detection_rate"] - 0.5) < 1e-9,
          str(first["detection_rate"]))
    second = hourly.iloc[1]
    check("an all-empty hour survives as a zero row",
          second["frames"] == 1 and second["frames_with_detection"] == 0)
    check("its area is 0, not NaN", second["bbox_area_max"] == 0.0,
          str(second["bbox_area_max"]))
    for col in ("frames", "frames_with_detection", "detections",
                "detection_rate", "bbox_area_max"):
        check(f"column {col} matches the pipeline contract", col in hourly.columns)


def test_unparsed_names_are_reported():
    print("filenames that do not match")
    payload = _coco(
        images=[{"id": 1, "file_name": "clm_s_01_2011-05-21-11-00.png"},
                {"id": 2, "file_name": "mystery.png"}],
        annotations=[])
    frames = rip.build_frames(payload)
    check("the bad name is dropped, not guessed at", len(frames) == 1)


def test_probe_patterns():
    print("timestamp patterns the probe recognizes")
    cases = [
        (["clm_s_01_201105211100.txt"], "YYYYMMDDHHMM"),
        (["clm_s_01_2011-05-21-11-00.png"], "YYYY-MM-DD-HH-MM"),
        (["walton_lighthouse-2026-08-31-140451Z.jpg"], "YYYY-MM-DD-HHMMSS"),
        (["cam_20210715_143000.jpg"], "YYYYMMDD_HHMMSS"),
        (["cam_20210715.jpg"], "YYYYMMDD (date only)"),
        (["frame_000001.jpg"], None),
    ]
    for names, want in cases:
        got, _, _ = probe.timestamp_style(names)
        check(f"{names[0][:34]} -> {want}", got == want, f"got {got}")


def test_cvat_subset_prefix():
    """A CVAT export records "default/<name>.png", not "<name>.png".

    FILENAME is anchored, so matching the full string drops every frame and
    leaves an empty table -- which then raised KeyError('timestamp') from the
    sort rather than saying what went wrong. Both halves are checked.
    """
    print("a CVAT export with a subset folder in the file name")
    plain = {"images": [{"id": 1, "file_name": "clm_s_01_2011-05-21-11-00.png"}],
             "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                              "area": 10.0}],
             "categories": [{"id": 1, "name": rip.RIP_LABEL}]}
    prefixed = dict(plain, images=[
        {"id": 1, "file_name": "default/clm_s_01_2011-05-21-11-00.png"}])

    bare = rip.build_frames(plain)
    nested = rip.build_frames(prefixed)
    check("the bare name parses", len(bare) == 1)
    check("and so does the prefixed one", len(nested) == 1)
    check("both land on the same camera and timestamp",
          bare["camera"].iloc[0] == nested["camera"].iloc[0]
          and bare["timestamp"].iloc[0] == nested["timestamp"].iloc[0])
    check("the FULL name is kept, since it locates the file on disk",
          nested["file_name"].iloc[0] == "default/clm_s_01_2011-05-21-11-00.png")

    unmatchable = dict(plain, images=[{"id": 1, "file_name": "nope.png"}])
    try:
        rip.build_frames(unmatchable)
        check("a wholly unparseable export exits", False, "it returned")
    except SystemExit as exc:
        check("a wholly unparseable export exits with a readable reason",
              "pattern" in str(exc), str(exc).replace("\n", " ")[:70])


# The label files carry six decimal places -- that is what RipAID ships -- so
# a box recovered from them is exact only to that. Measured: area within 2e-7,
# orientation within 1.2e-5 degrees. The tolerances below are the data's own
# precision, not a fudge; the exact-coordinate case is checked separately at
# machine precision to show the arithmetic itself is not the source of either.
AREA_TOL = 1e-6
ANGLE_TOL = 1e-3


def _obb(cx, cy, w, h, deg):
    """A rotated box as eight normalised numbers, for the fixtures below.

    Rounded to six places exactly as the real files are, so the fixtures share
    their precision limit rather than testing against numbers no file has.
    """
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    out = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2),
                   (-w / 2, h / 2)):
        out += [cx + dx * c - dy * s, cy + dx * s + dy * c]
    return " ".join(f"{v:.6f}" for v in out)


def test_obb_geometry():
    """Area and orientation from four corners, against hand arithmetic."""
    print("oriented-box geometry")
    points = [(0.0, 0.0), (0.4, 0.0), (0.4, 0.1), (0.0, 0.1)]
    check("an axis-aligned box's area is width x height, exactly",
          abs(rip.obb_area(points) - 0.04) < 1e-15,
          f"err {abs(rip.obb_area(points) - 0.04):.2e}")
    exact_turned = [(0.5, 0.5 - 0.2), (0.5 + 0.2, 0.5), (0.5, 0.5 + 0.2),
                    (0.5 - 0.2, 0.5)]
    check("and a diamond's is half its bounding square, exactly",
          abs(rip.obb_area(exact_turned) - 0.08) < 1e-15,
          f"err {abs(rip.obb_area(exact_turned) - 0.08):.2e}")

    # The SAME box turned 45 degrees has the SAME area. The axis-aligned
    # bound of it does not, which is why the shoelace is used.
    turned = [tuple(float(v) for v in pair) for pair in
              zip(*[iter([float(x) for x in _obb(0.5, 0.5, 0.4, 0.1, 45).split()])] * 2)]
    check("and rotating it does not change the area",
          abs(rip.obb_area(turned) - 0.04) < AREA_TOL,
          f"{rip.obb_area(turned):.6f}")

    for planted in (0.0, 30.0, 95.0, 179.0):
        pts = [tuple(float(v) for v in pair) for pair in
               zip(*[iter([float(x) for x in
                           _obb(0.5, 0.5, 0.4, 0.1, planted).split()])] * 2)]
        got = rip.obb_angle_deg(pts)
        check(f"orientation {planted:5.0f} recovered",
              abs(got - planted) < ANGLE_TOL,
              f"{got:.4f}")

    # An orientation is a line, so 190 degrees IS 10 degrees.
    pts = [tuple(float(v) for v in pair) for pair in
           zip(*[iter([float(x) for x in _obb(0.5, 0.5, 0.4, 0.1, 190).split()])] * 2)]
    check("190 degrees folds onto 10",
          abs(rip.obb_angle_deg(pts) - 10.0) < ANGLE_TOL,
          f"{rip.obb_angle_deg(pts):.4f}")

    # The long axis, not the short one: a box wider than tall must report
    # along its width even when the corner order starts on the short edge.
    tall = [tuple(float(v) for v in pair) for pair in
            zip(*[iter([float(x) for x in _obb(0.5, 0.5, 0.1, 0.4, 0).split()])] * 2)]
    check("a tall box reports its long axis, at 90 degrees",
          abs(rip.obb_angle_deg(tall) - 90.0) < ANGLE_TOL,
          f"{rip.obb_angle_deg(tall):.4f}")


def test_yolo_obb_reader():
    """A YOLO-OBB export with a known composition, read back."""
    print("reading a YOLO-OBB export")
    with tempfile.TemporaryDirectory() as folder:
        labels = os.path.join(folder, "labels")
        images = os.path.join(folder, "images")
        os.makedirs(labels)
        os.makedirs(images)

        def write(stem, lines):
            with open(os.path.join(labels, stem + ".txt"), "w") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            open(os.path.join(images, stem + ".png"), "wb").write(b"x")

        write("clm_s_01_2012-03-20-10-00",
              ["0 " + _obb(0.5, 0.5, 0.4, 0.1, 30)])
        write("clm_s_01_2012-03-20-11-00",
              ["0 " + _obb(0.5, 0.5, 0.2, 0.1, 30),
               "0 " + _obb(0.2, 0.2, 0.4, 0.1, 30),
               "1 " + _obb(0.8, 0.8, 0.1, 0.1, 0)])
        write("snb_s_02_2013-07-01-09-00", ["2 " + _obb(0.5, 0.5, 0.3, 0.1, 0)])
        write("snb_s_02_2013-07-01-10-00", [])          # a real negative
        write("DJI_0005_8460_jpg.rf.abc", ["0 " + _obb(0.5, 0.5, 0.2, 0.1, 0)])
        write("PHOTO-2023-03-21-09-18-10_jpg.rf.def",
              ["0 " + _obb(0.5, 0.5, 0.2, 0.1, 0)])

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            frame = rip.build_frames_yolo(folder)
        text = buffer.getvalue()

    check("only the fixed-camera frames survive", len(frame) == 4, str(len(frame)))
    check("and the drone and phone frames are reported, not silently dropped",
          "DJI_0005" in text and "2 of 6" in text, text.split("\n")[-6][:60])

    by_stem = {os.path.splitext(n)[0]: r for n, r in
               zip(frame["file_name"], frame.to_dict("records"))}
    one = by_stem["clm_s_01_2012-03-20-10-00"]
    check("a single rip is counted once and detected",
          one["n_rip"] == 1 and one["detected"] is True)
    check("its area is width x height", abs(one["area_max"] - 0.04) < AREA_TOL,
          f"{one['area_max']:.6f}")
    check("and its orientation is the planted 30 degrees",
          abs(one["rotation_axial"] - 30.0) < ANGLE_TOL,
          f"{one['rotation_axial']:.4f}")

    two = by_stem["clm_s_01_2012-03-20-11-00"]
    check("two rips and a doubt are counted separately",
          (two["n_rip"], two["n_doubt"], two["n_sediment"]) == (2, 1, 0),
          str((two["n_rip"], two["n_doubt"], two["n_sediment"])))
    check("area_max takes the LARGER rip, not the first",
          abs(two["area_max"] - 0.04) < AREA_TOL, f"{two['area_max']:.6f}")
    check("area_sum adds them", abs(two["area_sum"] - 0.06) < AREA_TOL,
          f"{two['area_sum']:.6f}")

    sediment = by_stem["snb_s_02_2013-07-01-09-00"]
    check("a sediment-only frame is NOT a detection",
          sediment["detected"] is False and sediment["n_sediment"] == 1)
    check("and carries no rip area", sediment["area_max"] == 0.0)

    empty = by_stem["snb_s_02_2013-07-01-10-00"]
    check("an empty label file is an observed zero, not a gap",
          empty["detected"] is False
          and (empty["n_rip"], empty["n_doubt"], empty["n_sediment"]) == (0, 0, 0))

    check("the site is taken from the camera name",
          set(frame["site"]) == {"clm", "snb"}, str(sorted(set(frame["site"]))))
    check("file_name resolves to the real image on disk",
          all(str(n).endswith(".png") for n in frame["file_name"]),
          str(list(frame["file_name"])[:2]))


def test_yolo_class_mapping_is_checked():
    """The mapping is recovered from counts, so a mismatch must be announced."""
    print("the class index mapping")
    check("index 0 is the rip class", rip.YOLO_CLASSES[0] == rip.RIP_LABEL)
    check("index 1 is doubt", rip.YOLO_CLASSES[1] == rip.DOUBT_LABEL)
    check("index 2 is sediment", rip.YOLO_CLASSES[2] == "sediment")

    from collections import Counter
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exact = rip.verify_classes(Counter(rip.PUBLISHED_INSTANCES))
    check("counts matching the README report an exact match", exact)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        off = rip.verify_classes(Counter({rip.RIP_LABEL: 7}))
    check("counts that do not are reported as not matching", not off)
    check("and the mismatch says what the README expected",
          "4103" in buffer.getvalue(), buffer.getvalue().strip()[:60])

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        unknown = rip.verify_classes(Counter({"class_9": 4}))
    check("an index this loader does not know about is flagged",
          not unknown and "UNKNOWN" in buffer.getvalue())


def main():
    for test in (test_axial_mean, test_observed_zeros, test_doubt_only_frames,
                 test_hourly, test_unparsed_names_are_reported,
                 test_cvat_subset_prefix, test_obb_geometry,
                 test_yolo_obb_reader, test_yolo_class_mapping_is_checked,
                 test_probe_patterns):
        test()
        print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
