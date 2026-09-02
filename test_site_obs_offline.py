"""Offline checks for pull_site_observations.py and pull_wqp_results.py.

No network, no credentials. Covers the pure logic: which sources a site
qualifies for, how columns are found in an arbitrary CSV, how rain windows
handle gaps, and that the two scripts name a site's files identically.

    python test_site_obs_offline.py
"""

import sys
from datetime import datetime

import pandas as pd

import pull_site_observations as pso
import pull_wqp_results as wqp

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}  {detail}")
        FAILURES.append(name)


def test_us_detection():
    print("which sources a site qualifies for")
    check("Virginia Beach is in the US", pso.in_united_states(36.83, -75.97))
    check("Santa Cruz is in the US", pso.in_united_states(36.95, -122.02))
    check("Holland MI is in the US", pso.in_united_states(42.77, -86.21))
    check("Homer AK is in the US", pso.in_united_states(59.60, -151.42))
    check("Waikiki is in the US", pso.in_united_states(21.27, -157.82))
    # The European sites are the reason this test exists: a false positive here
    # would send the script hunting for an NDBC buoy off Aquitaine.
    check("Biscarrosse (France) is not", not pso.in_united_states(44.47, -1.25))
    check("Perranporth (UK) is not", not pso.in_united_states(50.35, -5.15))
    check("Sydney is not", not pso.in_united_states(-33.89, 151.27))


def test_column_picking():
    print("finding coordinates in an arbitrary CSV")
    exact = pd.DataFrame(columns=["beach_name", "latitude", "longitude", "n"])
    check("exact names", (pso._pick(exact, ("lat", "latitude")) == "latitude"
                          and pso._pick(exact, ("lon", "lng", "longitude")) == "longitude"))
    check("name column", pso._pick(exact, ("name", "camera", "site", "beach")) == "beach_name")

    # 'longitude' contains 'lon' as a substring, so an exact-match pass has to
    # run before the substring pass or a lat lookup can grab the wrong column.
    short = pd.DataFrame(columns=["Site", "Lat", "Lon"])
    check("case-insensitive exact match",
          pso._pick(short, ("lat", "latitude")) == "Lat"
          and pso._pick(short, ("lon", "lng", "longitude")) == "Lon")

    awkward = pd.DataFrame(columns=["id", "y_latitude_dd", "x_longitude_dd"])
    check("falls back to substring",
          pso._pick(awkward, ("lat", "latitude")) == "y_latitude_dd")
    check("returns None when absent", pso._pick(exact, ("elevation",)) is None)


def test_rain_windows():
    print("rolling rainfall across a gap")
    times = pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC")
    frame = pd.DataFrame({"time": times, "precipitation": [1.0, 1, 1, 1, 1, 1]})
    out = pso.add_rain_windows(frame)
    check("adds the window columns",
          all(f"rain_{h}h_mm" in out.columns for h in (24, 48, 72)))
    check("too short a series is NaN, not a partial sum",
          out["rain_24h_mm"].isna().all())

    long_times = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    full = pd.DataFrame({"time": long_times, "precipitation": [1.0] * 48})
    out = pso.add_rain_windows(full)
    check("24h of 1mm sums to 24", out["rain_24h_mm"].iloc[-1] == 24.0,
          str(out["rain_24h_mm"].iloc[-1]))

    gapped = full.drop(index=range(10, 14)).reset_index(drop=True)
    out = pso.add_rain_windows(gapped)
    check("reindexes the missing hours back in", len(out) == 48, str(len(out)))
    check("a window spanning the gap is NaN, not an undercount",
          pd.isna(out["rain_24h_mm"].iloc[30]), str(out["rain_24h_mm"].iloc[30]))

    no_rain = pd.DataFrame({"time": times, "wind_speed_10m": [1.0] * 6})
    check("no precipitation column is not a crash",
          "rain_24h_mm" not in pso.add_rain_windows(no_rain).columns)


def test_nearest():
    print("station matching")
    frame = pd.DataFrame([{"Station": "far", "Lat": 40.0, "Lon": -122.0},
                          {"Station": "near", "Lat": 36.96, "Lon": -122.03}])
    row, km = pso._nearest(36.95, -122.02, frame, "Lat", "Lon", 50)
    check("picks the closest", row is not None and row["Station"] == "near")
    check("rejects beyond max_km",
          pso._nearest(36.95, -122.02, frame, "Lat", "Lon", 0.1) == (None, None))
    check("empty frame is not a match",
          pso._nearest(1, 1, pd.DataFrame(), "Lat", "Lon", 50) == (None, None))


def test_slug_agreement():
    print("file naming")
    for name in ("Hampton Inn Oceanfront South at Virginia Beach",
                 "Walton Lighthouse, Santa Cruz, CA",
                 "Kahului, Hawaiʻi - Harbor (east) view"):
        check(f"both scripts agree on {name[:28]!r}",
              pso.slugify(name) == wqp.slugify(name),
              f"{pso.slugify(name)} vs {wqp.slugify(name)}")
    check("slug is filesystem-safe",
          all(c.isalnum() or c == "_" for c in pso.slugify("a/b c:d")))
    check("slug is bounded", len(pso.slugify("x" * 200)) == 48)


def test_marine_nudges():
    print("marine land-cell fallback")
    check("tries the exact point first", pso.MARINE_NUDGES[0] == 0.0)
    check("nudges outward in order",
          pso.MARINE_NUDGES == sorted(pso.MARINE_NUDGES))
    check("covers all eight directions", len(pso.MARINE_BEARINGS) == 8)
    check("furthest nudge is a plausible coastal offset",
          0.1 <= max(pso.MARINE_NUDGES) <= 0.5,
          f"{max(pso.MARINE_NUDGES) * 111:.0f} km")


def test_marine_model_is_passed_through():
    """The Wrightsville pull returned 233,664 hours and a wave height in 18%
    of them, all at the recent end -- a request that succeeded and produced a
    series too short to join. The model has to be selectable, and the choice
    has to reach the request."""
    print("the marine model reaches the query string")
    sent = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"hourly": {"time": ["2005-01-01T00:00"], "wave_height": [1.0]},
                    "hourly_units": {"wave_height": "m"}}

    def fake_get(url, params=None, timeout=None):
        sent.update(params or {})
        return FakeResponse()

    real_get = pso.requests.get
    try:
        pso.requests.get = fake_get
        frame, note = pso.open_meteo(
            pso.MARINE, 34.19, -77.81,
            datetime(2005, 1, 1), datetime(2005, 1, 2),
            ["wave_height"], models="era5_ocean")
    finally:
        pso.requests.get = real_get

    check("the response still parses", frame is not None, note)
    check("models is in the request", sent.get("models") == "era5_ocean",
          sent.get("models"))

    sent.clear()
    try:
        pso.requests.get = fake_get
        pso.open_meteo(pso.MARINE, 34.19, -77.81,
                       datetime(2005, 1, 1), datetime(2005, 1, 2),
                       ["wave_height"])
    finally:
        pso.requests.get = real_get
    check("and absent when no model is chosen, so the default is unchanged",
          "models" not in sent, sorted(sent))

    check("the probe list starts with the current default",
          pso.MARINE_MODELS_TO_TRY[0] == "best_match",
          pso.MARINE_MODELS_TO_TRY[0])
    check("and offers a reanalysis that could cover 2000",
          "era5_ocean" in pso.MARINE_MODELS_TO_TRY, pso.MARINE_MODELS_TO_TRY)


def main():
    for test in (test_us_detection, test_column_picking, test_rain_windows,
                 test_nearest, test_slug_agreement, test_marine_nudges,
                 test_marine_model_is_passed_through):
        test()
        print()
    if FAILURES:
        sys.exit(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
