"""Offline tests for the derived quantities in pull_observations.py.

The network calls cannot be tested without network, but the arithmetic that
turns raw readings into features can, and it is where the bugs live.

    python test_pull_offline.py
"""

from datetime import datetime

import pandas as pd

import pull_observations as po


def check_rain_windows():
    """Rolling totals must never silently under-report across a missing day."""
    start, end = datetime(2026, 1, 1), datetime(2026, 1, 10)
    obs = pd.DataFrame([
        {"date": pd.Timestamp("2026-01-01"), "precip_mm": 0.0},
        {"date": pd.Timestamp("2026-01-02"), "precip_mm": 10.0},
        # 2026-01-03 absent entirely -- the station did not report.
        {"date": pd.Timestamp("2026-01-04"), "precip_mm": 5.0},
        # Two records on one day must sum, not overwrite.
        {"date": pd.Timestamp("2026-01-05"), "precip_mm": 2.0},
        {"date": pd.Timestamp("2026-01-05"), "precip_mm": 3.0},
        {"date": pd.Timestamp("2026-01-06"), "precip_mm": 0.0},
        {"date": pd.Timestamp("2026-01-07"), "precip_mm": 0.0},
        {"date": pd.Timestamp("2026-01-08"), "precip_mm": 1.0},
        {"date": pd.Timestamp("2026-01-09"), "precip_mm": 0.0},
        {"date": pd.Timestamp("2026-01-10"), "precip_mm": 0.0},
    ])
    r = po.add_rain_windows(obs, start, end).set_index("date")

    assert r.loc["2026-01-05", "precip_mm"] == 5.0, "same-day records must sum"
    assert pd.isna(r.loc["2026-01-03", "precip_mm"]), "a missing day is NaN, not 0"
    assert pd.isna(r.loc["2026-01-04", "rain_48h_mm"]), "window over a gap is NaN"
    assert pd.isna(r.loc["2026-01-05", "rain_72h_mm"])
    assert r.loc["2026-01-02", "rain_48h_mm"] == 10.0
    assert r.loc["2026-01-07", "rain_72h_mm"] == 5.0
    assert r.loc["2026-01-08", "rain_72h_mm"] == 1.0

    original = po.RAIN_INCLUDE_SAME_DAY
    try:
        po.RAIN_INCLUDE_SAME_DAY = False
        shifted = po.add_rain_windows(obs, start, end).set_index("date")
        assert shifted.loc["2026-01-08", "rain_24h_mm"] == 0.0, \
            "excluding the sample day must drop that day's 1mm"
        assert r.loc["2026-01-08", "rain_24h_mm"] == 1.0, \
            "including it must keep that 1mm"
    finally:
        po.RAIN_INCLUDE_SAME_DAY = original
    print("rain windows OK")


def check_tide_state():
    """Direction must follow the real elapsed time, not an assumed 6-min step."""
    raw = pd.DataFrame({
        "t": ["2026-01-01 00:00", "2026-01-01 00:06", "2026-01-01 00:12",
              "2026-01-01 00:18", "2026-01-01 03:30", "2026-01-01 03:36"],
        "v": ["1.00", "1.02", "1.02", "0.98", "0.50", "0.44"],
    })
    raw["time"] = pd.to_datetime(raw["t"], utc=True)
    tide = po.add_tide_state(raw)

    assert pd.isna(tide.loc[0, "rate_m_per_hr"]), "first reading has no prior"
    assert tide.loc[1, "tide_state"] == "rising"
    assert tide.loc[2, "tide_state"] == "slack", "no change is slack, not a direction"
    assert tide.loc[3, "tide_state"] == "falling"
    # The 3-hour gap must not become a measured rate.
    assert pd.isna(tide.loc[4, "rate_m_per_hr"]), "no rate across a long gap"
    assert tide.loc[5, "tide_state"] == "falling"
    print("tide state OK")


def check_wind_parsing():
    raw = pd.DataFrame({"t": ["2026-01-01 00:00"], "s": ["3.4"], "d": ["270"],
                        "g": ["5.1"], "dr": ["W"]})
    raw["time"] = pd.to_datetime(raw["t"], utc=True)
    out = po.add_wind(raw)
    assert out.loc[0, "wind_speed_m_s"] == 3.4
    assert out.loc[0, "wind_dir_deg"] == 270
    assert out.loc[0, "wind_gust_m_s"] == 5.1
    assert out.loc[0, "wind_dir_text"] == "W"
    print("wind parsing OK")


def check_station_id_canonicalisation():
    """A CSV round-trip turns the integer station id 101 into 101.0, because
    the column holds NaN for non-qualifying rows and becomes float64. Both
    spellings must resolve to the same key or nothing matches."""
    assert po.canonical_station_id(101) == "101"
    assert po.canonical_station_id(101.0) == "101"
    assert po.canonical_station_id("101") == "101"
    assert po.canonical_station_id("101.0") == "101"
    assert po.canonical_station_id(" 101 ") == "101"
    # A genuinely non-integer or text id survives unmangled.
    assert po.canonical_station_id("EH-130") == "EH-130"
    assert po.canonical_station_id(101.5) == "101.5"

    ids = pd.Series([101.0, 102.0, None])
    wanted = {po.canonical_station_id(x) for x in ids.dropna()}
    assert wanted == {"101", "102"}, wanted
    print("station id canonicalisation OK")


if __name__ == "__main__":
    check_station_id_canonicalisation()
    check_rain_windows()
    check_tide_state()
    check_wind_parsing()
    print("\nAll offline pull assertions passed.")
