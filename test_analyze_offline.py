"""Checks for analyze_drivers on synthetic data with a KNOWN answer.

The point is not that the code runs. It is that when a relationship is planted
in the data, the analysis finds it and ranks it top; and when a variable is
pure noise or a pure clock, the analysis does not promote it. An analysis
script that cannot recover a planted signal is worse than none, because its
output still looks like a finding.

    python test_analyze_offline.py
"""

import math
import os
import shutil
import tempfile

import numpy as np
import pandas as pd

import analyze_drivers as a

FAILURES = []
RNG = np.random.default_rng(20260831)


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_spearman():
    print("\nspearman")
    x = np.arange(200.0)
    rho, n, p = a.spearman(x, x * 3 + 1)
    check("a monotonic relation is rho 1", abs(rho - 1.0) < 1e-9, rho)
    check("n counts the pairs", n == 200)
    rho, _, _ = a.spearman(x, -x)
    check("a decreasing relation is rho -1", abs(rho + 1.0) < 1e-9, rho)

    # Monotonic but wildly non-linear: this is why rank, not Pearson.
    rho, _, _ = a.spearman(x, np.exp(x / 20))
    check("rank correlation ignores curvature", abs(rho - 1.0) < 1e-9, rho)

    noise_rho, _, noise_p = a.spearman(RNG.normal(size=500), RNG.normal(size=500))
    check("noise gives a small rho", abs(noise_rho) < 0.15, noise_rho)
    check("noise gives a non-significant p", noise_p > 0.05, noise_p)

    strong_rho, _, strong_p = a.spearman(x, x + RNG.normal(scale=20, size=200))
    check("a real signal is significant", strong_p < 1e-6, strong_p)

    rho, n, p = a.spearman([1, 2, 3], [1, 2, 3])
    check("under the sample floor returns NaN, not a confident answer",
          math.isnan(rho) and n == 3, (rho, n))
    rho, _, _ = a.spearman(np.ones(100), np.arange(100.0))
    check("a constant column gives NaN rather than dividing by zero",
          math.isnan(rho), rho)


def test_demean_by():
    print("\ndemean_by")
    hours = pd.Series([0, 0, 12, 12] * 30)
    # Value is entirely determined by hour: after demeaning nothing is left.
    values = hours.map({0: 10.0, 12: 90.0})
    residual = a.demean_by(values, hours)
    check("a pure clock signal demeans to zero",
          residual.abs().max() < 1e-9, residual.abs().max())

    # Signal that varies within each hour survives.
    varying = values + pd.Series(RNG.normal(size=len(values)))
    residual = a.demean_by(varying, hours)
    check("within-hour variation survives", residual.std() > 0.5, residual.std())


def test_standardized_ols():
    print("\nstandardized_ols")
    n = 400
    strong = RNG.normal(size=n)
    weak = RNG.normal(size=n)
    noise = RNG.normal(size=n)
    frame = pd.DataFrame({
        "strong": strong, "weak": weak, "noise": noise,
        "y": 3 * strong + 0.5 * weak + RNG.normal(scale=0.5, size=n)})
    fit = a.standardized_ols(frame, "y", ["strong", "weak", "noise"])
    betas = dict(zip(fit["names"], fit["beta"]))
    check("the strong predictor gets the largest beta",
          max(betas, key=lambda k: abs(betas[k])) == "strong", betas)
    check("the irrelevant one gets a near-zero beta", abs(betas["noise"]) < 0.1,
          betas["noise"])
    check("R2 is high for a well-specified fit", fit["r2"] > 0.9, fit["r2"])

    # Collinearity must be flagged, not silently produce unstable betas.
    frame["copy"] = frame["strong"] + RNG.normal(scale=1e-6, size=n)
    fit = a.standardized_ols(frame, "y", ["strong", "copy", "weak"])
    check("collinear predictors raise the condition number",
          fit["condition"] > 30, fit["condition"])


def write_rip_fixture(tmp, planted="WVHT"):
    """A year of hourly rip output where detection rate depends on one driver.

    Also plants a pure clock variable, which correlates with the target only
    because both follow the daylight cycle. A correct analysis ranks the real
    driver top on rho_ctrl and demotes the clock.
    """
    os.makedirs(tmp, exist_ok=True)
    hours = pd.date_range("2025-06-01", "2026-05-31 23:00", freq="h", tz="UTC")
    frame = pd.DataFrame({"hour": hours})
    frame["hour_of_day"] = frame["hour"].dt.hour
    daylight = frame[(frame["hour_of_day"] >= 15) & (frame["hour_of_day"] <= 23)].copy()

    wvht = 1.0 + 1.5 * RNG.random(len(daylight))
    # The clock: identical every day, carrying no information beyond the hour.
    clock = np.cos(2 * np.pi * daylight["hour_of_day"] / 24).to_numpy()
    # The target genuinely has a daily cycle too -- an afternoon peak. That is
    # what makes the clock variable correlate raw despite being causally
    # irrelevant, which is the confound the rho_ctrl column exists to catch.
    rate = (0.05
            + 0.30 * (wvht - 1.0) / 1.5
            + 0.08 * clock
            + 0.02 * RNG.normal(size=len(daylight)))
    rate = np.clip(rate, 0, 1)

    pd.DataFrame({
        "hour": daylight["hour"], "frames": 4,
        "frames_with_detection": np.round(rate * 4),
        "detections": np.round(rate * 4),
        "detection_rate": rate,
        "score_max": np.clip(rate + 0.4, 0, 1),
        "score_mean": np.clip(rate + 0.3, 0, 1),
        "bbox_area_max": 50000 * rate,
    }).to_csv(f"{tmp}/rip_test-beach_hourly.csv", index=False)

    grid = pd.DataFrame({
        "time": daylight["hour"],
        "precipitation": RNG.exponential(0.2, len(daylight)),
        "wind_speed_10m": 3 + RNG.normal(size=len(daylight)),
        "wind_direction_10m": RNG.uniform(0, 360, len(daylight)),
        "wind_gusts_10m": 5 + RNG.normal(size=len(daylight)),
        "temperature_2m": 15 + 8 * clock,
        "rain_24h_mm": RNG.exponential(1.0, len(daylight)),
        "rain_48h_mm": RNG.exponential(2.0, len(daylight)),
        "rain_72h_mm": RNG.exponential(3.0, len(daylight)),
    })
    grid.to_csv(f"{tmp}/gridded_{a.grid_slug('Test Beach')}.csv", index=False)

    buoy = pd.DataFrame({"WVHT": wvht, "DPD": 10 + RNG.normal(size=len(daylight)),
                         "APD": 7 + RNG.normal(size=len(daylight)),
                         "MWD": RNG.uniform(0, 360, len(daylight)),
                         "WTMP": 14 + RNG.normal(size=len(daylight))},
                        index=daylight["hour"])
    buoy.to_csv(f"{tmp}/buoy_46042.csv")

    pd.DataFrame({"time": daylight["hour"],
                  "level_m": 1 + np.sin(np.arange(len(daylight))),
                  "rate_m_per_hr": np.cos(np.arange(len(daylight))) * 0.3,
                  }).to_csv(f"{tmp}/tide_9413745.csv", index=False)

    pd.DataFrame([{
        "camera_name": "Test Beach", "lat": 36.96, "lon": -122.01,
        "buoy_id": "46042", "precip_station_id": "GHCND:USW00023233",
        "wq_station_id": 101, "wq_station_name": "Test Beach WQ",
        "has_all_four": True}]).to_csv(f"{tmp}/sites.csv", index=False)


def test_rip_recovers_planted_driver():
    print("\nrip analysis on a fixture with WVHT planted as the driver")
    tmp = tempfile.mkdtemp()
    old_data, old_sites = a.DATA_DIR, a.SITES_CSV
    try:
        write_rip_fixture(tmp)
        a.DATA_DIR, a.SITES_CSV = tmp, f"{tmp}/sites.csv"
        a.SHORE_NORMAL_DEG["Test Beach"] = 180.0
        sites = a.load_sites()
        frame, name = a.assemble_rip(sites)
        check("the fixture site is resolved", name == "Test Beach", name)
        check("conditions joined onto the rip hours",
              "WVHT" in frame.columns and "temperature_2m" in frame.columns,
              [c for c in frame.columns][:12])

        observed = frame[frame["frames"] > 0]
        table = a.report_correlations(observed, "detection_rate",
                                      a.RIP_PREDICTORS,
                                      control=observed["hour_of_day"])
        top_ctrl = table.iloc[0]["predictor"]
        check("the planted driver ranks top once hour-of-day is removed",
              top_ctrl == "WVHT", top_ctrl)

        clock_row = table[table["predictor"] == "temperature_2m"].iloc[0]
        check("the pure clock correlates raw", abs(clock_row["rho"]) > 0.05,
              clock_row["rho"])
        check("but collapses once hour-of-day is removed",
              abs(clock_row["rho_ctrl"]) < 0.05, clock_row["rho_ctrl"])

        rain_row = table[table["predictor"] == "rain_48h_mm"].iloc[0]
        check("unrelated rainfall stays near zero", abs(rain_row["rho_ctrl"]) < 0.1,
              rain_row["rho_ctrl"])
    finally:
        a.DATA_DIR, a.SITES_CSV = old_data, old_sites
        shutil.rmtree(tmp)


def test_wq_recovers_planted_driver():
    print("\nwater quality on a fixture with rain_72h planted as the driver")
    tmp = tempfile.mkdtemp()
    old_data, old_sites = a.DATA_DIR, a.SITES_CSV
    try:
        days = pd.date_range("2025-09-01", periods=300, freq="D")
        # Genuinely nested windows built from independent daily rainfall, so
        # the three are correlated but not identical. A fixture that made them
        # rescalings of one series would give identical ranks and decide the
        # top slot by tie-break rather than by signal.
        daily = pd.Series(RNG.exponential(2.0, len(days)))
        rain24 = daily
        rain48 = daily.rolling(2, min_periods=2).sum()
        rain72 = daily.rolling(3, min_periods=3).sum()
        pd.DataFrame({"date": days, "precip_mm": daily,
                      "rain_24h_mm": rain24, "rain_48h_mm": rain48,
                      "rain_72h_mm": rain72}).to_csv(
            f"{tmp}/precip_GHCND_USW00023233.csv", index=False)

        counts = 10 * np.power(
            10, 0.10 * rain72.fillna(0).to_numpy()
            + RNG.normal(scale=0.2, size=len(days)))
        pd.DataFrame({
            "StationCode": "TB-1", "StationName": "Test Beach WQ",
            "SampleDate": days.strftime("%Y-%m-%d"),
            "Analyte": "Enterococcus", "Result": counts.round(0),
            "Unit": "MPN/100 mL"}).to_csv(f"{tmp}/water_quality.csv", index=False)

        pd.DataFrame([{
            "camera_name": "Test Beach", "lat": 36.96, "lon": -122.01,
            "buoy_id": None, "precip_station_id": "GHCND:USW00023233",
            "wq_station_id": 101, "wq_station_name": "Test Beach WQ",
            "has_all_four": True}]).to_csv(f"{tmp}/sites.csv", index=False)

        a.DATA_DIR, a.SITES_CSV = tmp, f"{tmp}/sites.csv"
        samples = a.load_water_quality()
        check("results parsed", samples is not None and len(samples) == 300,
              None if samples is None else len(samples))
        check("the analyte is recognised",
              set(samples["group"].dropna()) == {"ENT"},
              set(samples["group"].dropna()))
        check("log10 transform applied",
              samples["log_value"].max() < 6 and samples["value"].max() > 100)

        sites = a.load_sites()
        mapping = a.map_stations(sites, samples)
        check("station mapped by name when the CKAN join is unavailable",
              mapping.get("TB-1") == "Test Beach", mapping)

        conditions = a.daily_conditions(sites.iloc[0])
        joined = samples.merge(conditions.reset_index(), on="date", how="left")
        table = a.report_correlations(joined, "log_value", a.WQ_PREDICTORS)
        check("the planted rainfall window ranks top",
              table.iloc[0]["predictor"] == "rain_72h_mm", table.iloc[0]["predictor"])
        check("and it is strong", table.iloc[0]["rho"] > 0.7, table.iloc[0]["rho"])
        shorter = table[table["predictor"] == "rain_24h_mm"].iloc[0]["rho"]
        check("the correct window beats the shorter nested one",
              table.iloc[0]["rho"] > shorter, (table.iloc[0]["rho"], shorter))
    finally:
        a.DATA_DIR, a.SITES_CSV = old_data, old_sites
        shutil.rmtree(tmp)


def test_analyte_key():
    print("\nanalyte_key")
    cases = {"Enterococcus": "ENT", "ENTEROCOCCUS": "ENT", "E. coli": "ECOLI",
             "Escherichia coli": "ECOLI", "Total Coliform": "TOTAL",
             "Fecal Coliform": "FECAL", "Turbidity": None}
    for text, want in cases.items():
        got = a.analyte_key(text)
        check(f"{text!r} -> {want}", got == want, got)


if __name__ == "__main__":
    test_spearman()
    test_demean_by()
    test_standardized_ols()
    test_analyte_key()
    test_rip_recovers_planted_driver()
    test_wq_recovers_planted_driver()
    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
    raise SystemExit(1 if FAILURES else 0)
