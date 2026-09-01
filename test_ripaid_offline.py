"""Offline checks for load_ripaid.py and the filename-timestamp probe.

The two things that would quietly corrupt this dataset are treating a frame a
person annotated as empty as MISSING rather than as an observed zero, and
averaging rip orientations as if they were arrows rather than lines.

    python test_ripaid_offline.py
"""

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


def main():
    for test in (test_axial_mean, test_observed_zeros, test_doubt_only_frames,
                 test_hourly, test_unparsed_names_are_reported, test_probe_patterns):
        test()
        print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
