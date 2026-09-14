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


def qualification_fixture():
    """Four cameras, each failing a different requirement, plus one that passes.

    Written as literal cells rather than built from a loop so that what each
    row is testing stays readable.
    """
    return pd.DataFrame([
        # everything present
        {"camera": "complete", "buoy_id": "46236", "tide_id": "9413450",
         "wq_within_2km": True, "precip_id": "GHCND:USC00048558"},
        # gauge is the only gap -- this is the row --weather grid rescues
        {"camera": "no gauge", "buoy_id": "46236", "tide_id": "9413450",
         "wq_within_2km": True, "precip_id": None},
        # bacteria station exists but sits outside the 2 km radius
        {"camera": "bacteria too far", "buoy_id": "46236", "tide_id": "9413450",
         "wq_within_2km": False, "precip_id": "GHCND:USC00048558"},
        # nothing offshore
        {"camera": "no buoy", "buoy_id": None, "tide_id": "9413450",
         "wq_within_2km": True, "precip_id": "GHCND:USC00048558"},
        # fails two at once, so it is evidence against neither on its own
        {"camera": "no buoy or gauge", "buoy_id": None, "tide_id": "9413450",
         "wq_within_2km": True, "precip_id": None},
    ])


def test_missing_sources():
    print("which requirement a camera fails")
    table = qualification_fixture()
    rows = {r["camera"]: r for _, r in table.iterrows()}
    check("a complete camera is missing nothing",
          s.missing_sources(rows["complete"]) == [])
    check("a camera with no gauge fails on precipitation",
          s.missing_sources(rows["no gauge"]) == ["precipitation"])
    check("under the grid that same camera fails nothing",
          s.missing_sources(rows["no gauge"], weather="grid") == [])
    check("a bacteria station at 9 km is not coverage",
          s.missing_sources(rows["bacteria too far"]) == ["water quality"])
    check("the grid does not rescue a water quality gap",
          s.missing_sources(rows["bacteria too far"], weather="grid")
          == ["water quality"])
    check("two gaps are both reported",
          s.missing_sources(rows["no buoy or gauge"]) == ["buoy", "precipitation"])
    # NaN is what an id becomes after a round trip through a CSV, and `is None`
    # does not catch it.
    nan_row = {"buoy_id": float("nan"), "tide_id": "9413450",
               "wq_within_2km": True, "precip_id": "GHCND:X"}
    check("a NaN id counts as absent, not present",
          s.missing_sources(nan_row) == ["buoy"])


def test_qualify_by_weather_source():
    print("what swapping the gauge for the grid buys")
    table = qualification_fixture()
    gauge = s.qualify(table, "gauge")
    grid = s.qualify(table, "grid")
    check("one camera qualifies on the gauge", int(gauge["qualifies"].sum()) == 1)
    check("two qualify on the grid", int(grid["qualifies"].sum()) == 2)
    check("the one it rescues is the gauge-only failure",
          list(grid[grid["qualifies"]]["camera"]) == ["complete", "no gauge"])
    check("the missing column names the gap",
          gauge.set_index("camera").loc["no gauge", "missing"] == "precipitation")
    check("and is empty where nothing is missing",
          gauge.set_index("camera").loc["complete", "missing"] == "")
    check("the input frame is not mutated",
          "qualifies" not in table.columns)
    try:
        s.qualify(table, "accuweather")
    except ValueError:
        check("an unknown weather source is refused", True)
    else:
        check("an unknown weather source is refused", False,
              "no ValueError raised")


def test_gate_cost():
    print("what each requirement costs on its own")
    cost = s.gate_cost(qualification_fixture())
    check("precipitation alone blocks one camera", cost["precipitation"] == 1)
    check("water quality alone blocks one", cost["water quality"] == 1)
    check("buoy alone blocks one", cost["buoy"] == 1)
    check("tide blocks none", cost["tide"] == 0)
    # The double-failure row must not be counted against either gate: relaxing
    # one of them would not qualify that camera, so charging it to both would
    # overstate what the change buys.
    check("a camera failing two is charged to neither",
          cost["buoy"] + cost["precipitation"] == 2)
    check("every requirement is reported, even the free ones",
          set(cost) == set(s.REQUIREMENTS))


def main():
    for test in (test_coordinate_order, test_nearest, test_load_cameras,
                 test_as_float, test_missing_sources,
                 test_qualify_by_weather_source, test_gate_cost):
        test()
        print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
