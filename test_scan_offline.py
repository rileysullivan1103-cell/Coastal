"""Offline checks for scan_cameras.py -- no network, no credentials.

The two things most likely to be silently wrong in this script are the
coordinate ORDER (WQP wants lon-first, CDO wants lat-first, and both accept the
other order without complaining) and the nearest-station match. Both are pure
functions, so both are testable here.

    python test_scan_offline.py
"""

import json
import os
import sys
import tempfile

import pandas as pd

import scan_cameras as s

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


def test_coordinate_order():
    print("coordinate order")
    lat, lon, km = 36.95, -122.02, 10  # Santa Cruz

    min_lon, min_lat, max_lon, max_lat = s.bbox_around(lat, lon, km)
    check("WQP bbox is lon,lat,lon,lat", min_lon < lon < max_lon and min_lat < lat < max_lat)
    check("WQP bbox brackets the point in longitude", min_lon < -122.02 < max_lon,
          f"{min_lon} .. {max_lon}")
    # A degree of longitude is shorter than a degree of latitude away from the
    # equator, so the same distance must span MORE degrees of longitude.
    check("longitude span widens with latitude",
          (max_lon - min_lon) > (max_lat - min_lat),
          f"lon span {max_lon - min_lon:.4f} vs lat span {max_lat - min_lat:.4f}")

    parts = [float(p) for p in s.extent_around(lat, lon, km).split(",")]
    check("CDO extent has four parts", len(parts) == 4, str(parts))
    check("CDO extent is lat,lon,lat,lon",
          parts[0] < lat < parts[2] and parts[1] < lon < parts[3], str(parts))
    check("the two orders are genuinely different",
          abs(parts[0] - min_lon) > 1, "extent[0] should be a latitude, not a longitude")


def test_nearest():
    print("nearest station match")
    frame = pd.DataFrame([
        {"id": "far", "latitude": 40.0, "longitude": -122.0},
        {"id": "near", "latitude": 36.96, "longitude": -122.03},
        {"id": "bad", "latitude": None, "longitude": -122.0},
    ])
    row, km = s.nearest(36.95, -122.02, frame, "latitude", "longitude", 50)
    check("picks the closest row", row is not None and row["id"] == "near",
          None if row is None else row["id"])
    check("distance is plausible", km is not None and 0 < km < 5, str(km))

    row, km = s.nearest(36.95, -122.02, frame, "latitude", "longitude", 0.1)
    check("respects max_km", row is None and km is None)

    # Non-zero index: iloc on the filtered frame must not fall back to labels.
    shifted = frame.iloc[1:].copy()
    shifted.index = [100, 101]
    row, _ = s.nearest(36.95, -122.02, shifted, "latitude", "longitude", 50)
    check("survives a non-default index", row is not None and row["id"] == "near")

    check("empty frame is not a match", s.nearest(1, 1, pd.DataFrame(), "a", "b", 50)
          == (None, None))
    check("missing columns are not a match",
          s.nearest(1, 1, frame, "nope", "nah", 50) == (None, None))
    all_bad = pd.DataFrame([{"latitude": None, "longitude": None}])
    check("all-unparseable frame is not a match",
          s.nearest(1, 1, all_bad, "latitude", "longitude", 50) == (None, None))


def test_load_cameras():
    print("camera parsing")
    assets = [
        {  # a camera with rip detection
            "slug": "cam-a",
            "data": {
                "common": {"label": "Camera A"},
                "properties": {"state_or_territory": "California",
                               "location": {"coordinates": [-122.02, 36.95]}},
            },
            "feeds": [{"products": [
                {"data": {"common": {"slug": "one-minute-stills"}}},
                {"data": {"common": {"slug": "rip-detection-results"}}},
            ]}],
        },
        {  # imagery only
            "slug": "cam-b",
            "data": {
                "common": {"label": "Camera B"},
                "properties": {"state_or_territory": "Florida",
                               "location": {"coordinates": [-80.1, 25.8]}},
            },
            "feeds": [{"products": [{"data": {"common": {"slug": "raw-video-data"}}}]}],
        },
        {  # no coordinates at all -- must be skipped, not crash
            "slug": "cam-c",
            "data": {"common": {"label": "Camera C"}, "properties": {}},
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "assets.json")
        with open(path, "w") as fh:
            json.dump(assets, fh)
        original = s.ASSETS_ALL
        s.ASSETS_ALL = path
        try:
            cameras = s.load_cameras()
        finally:
            s.ASSETS_ALL = original

    check("drops the asset with no coordinates", len(cameras) == 2, str(len(cameras)))
    check("keeps GeoJSON lon/lat order straight",
          abs(cameras.iloc[0]["lat"] - 36.95) < 1e-9
          and abs(cameras.iloc[0]["lon"] + 122.02) < 1e-9,
          cameras.iloc[0][["lat", "lon"]].to_dict())
    check("flags the rip camera", bool(cameras.iloc[0]["has_rip"]))
    check("does not flag the imagery-only camera", not bool(cameras.iloc[1]["has_rip"]))
    check("reads the state", cameras.iloc[1]["state"] == "Florida")
    check("counts products", int(cameras.iloc[0]["products"]) == 2)


def test_as_float():
    print("numeric coercion")
    check("parses a number", s._as_float("0.95") == 0.95)
    check("None stays None", s._as_float(None) is None)
    check("NaN becomes None", s._as_float(float("nan")) is None)
    check("garbage becomes None", s._as_float("n/a") is None)


def main():
    for test in (test_coordinate_order, test_nearest, test_load_cameras, test_as_float):
        test()
        print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
