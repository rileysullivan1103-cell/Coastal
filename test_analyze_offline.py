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
import sys
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

    # A near-copy is removed rather than left to destabilise the fit, so the
    # design stays well conditioned and the original keeps its full effect.
    frame["copy"] = frame["strong"] + RNG.normal(scale=1e-6, size=n)
    fit = a.standardized_ols(frame, "y", ["strong", "copy", "weak"])
    check("a near-duplicate predictor is dropped", "copy" not in fit["names"],
          fit["names"])
    check("so the design stays well conditioned", fit["condition"] < 30,
          fit["condition"])


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

        # A second bacteria file, in the national Water Quality Portal shape
        # that pull_wqp_results.py writes. Before both formats were read, a
        # site pulled this way was simply absent from the analysis.
        wqp_days = pd.date_range("2025-09-01", periods=40, freq="D", tz="UTC")
        pd.DataFrame({
            "station": "21VASWCB-TEST", "analyte": "Enterococcus",
            "value_raw": 10, "unit": "cfu/100mL",
            "value": list(range(1, 39)) + [np.nan, np.nan],
            "nondetect": [False] * 38 + [True, True],
            "sampled_at": wqp_days, "has_sample_time": True,
            "site": "Portal Beach"}).to_csv(f"{tmp}/wqp_Portal_Beach.csv",
                                            index=False)

        a.DATA_DIR, a.SITES_CSV = tmp, f"{tmp}/sites.csv"
        sites = a.load_sites()
        samples = a.load_water_quality(sites)
        check("both bacteria formats are read",
              set(samples["source"]) == {"CKAN", "WQP"},
              sorted(set(samples["source"])) if samples is not None else None)
        check("the portal site keeps its own camera name",
              set(samples.loc[samples["source"] == "WQP", "camera_name"])
              == {"Portal Beach"},
              set(samples.loc[samples["source"] == "WQP", "camera_name"]))
        check("non-detects are excluded, not counted as zero",
              (samples["source"] == "WQP").sum() == 38,
              int((samples["source"] == "WQP").sum()))

        samples = samples[samples["source"] == "CKAN"].copy()
        check("results parsed", samples is not None and len(samples) == 300,
              None if samples is None else len(samples))
        check("the analyte is recognised",
              set(samples["group"].dropna()) == {"ENT"},
              set(samples["group"].dropna()))
        check("log10 transform applied",
              samples["log_value"].max() < 6 and samples["value"].max() > 100)

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


def test_regression_drops_identical_columns():
    print("\nstandardized_ols on an exactly duplicated predictor")
    n = 300
    rain = RNG.exponential(2.0, n)
    frame = pd.DataFrame({
        "rain_24h_mm": rain,
        # RAIN_INCLUDE_SAME_DAY makes a 1-day rolling sum identical to the
        # daily total, which is exactly what the real CSVs contain.
        "precip_mm": rain,
        "other": RNG.normal(size=n)})
    frame["y"] = 2 * rain + RNG.normal(scale=0.5, size=n)
    fit = a.standardized_ols(frame, "y", ["rain_24h_mm", "precip_mm", "other"])
    check("the duplicate column is dropped",
          "precip_mm" not in fit["names"], fit["names"])
    check("the original is kept", "rain_24h_mm" in fit["names"], fit["names"])
    check("the fit is now well conditioned", fit["condition"] < 100,
          fit["condition"])
    betas = dict(zip(fit["names"], fit["beta"]))
    check("the real driver keeps its whole effect rather than splitting it",
          betas["rain_24h_mm"] > 0.8, betas)


def test_regression_withholds_when_singular():
    print("\nreport_regression on a singular design")
    n = 200
    base = RNG.normal(size=n)
    frame = pd.DataFrame({"a": base, "b": base * 2 + 1e-12,
                          "c": base * 3, "y": base + RNG.normal(scale=0.1, size=n)})
    fit = a.report_regression(frame, "y", ["a", "b", "c"])
    # Perfect copies are removed, so this one becomes well conditioned; the
    # guard matters for near-copies that survive the exact-duplicate filter.
    check("a degenerate design does not produce printed betas with a huge cond",
          fit is None or fit["beta"] is None
          or fit["condition"] <= a.MAX_REPORTABLE_CONDITION,
          None if fit is None else fit["condition"])


def test_degenerate_target_is_named():
    print("\na target with no variance")
    frame = pd.DataFrame({"detection_rate": [1.0] * 100,
                          "WVHT": RNG.normal(size=100)})
    series = pd.to_numeric(frame["detection_rate"])
    check("a constant target has under 3 distinct values",
          series.nunique() < 3, series.nunique())
    rho, n, p = a.spearman(frame["WVHT"], frame["detection_rate"])
    check("correlating against it yields NaN, not a number",
          math.isnan(rho), rho)


def test_coverage_creates_observed_zeros():
    print("\napply_coverage")
    tmp = tempfile.mkdtemp()
    old_data = a.DATA_DIR
    try:
        a.DATA_DIR = tmp
        hours = pd.date_range("2025-06-01 15:00", periods=10, freq="h", tz="UTC")
        # The detector fired in only 3 of the 10 hours the camera was looking.
        detections = pd.DataFrame({
            "hour": hours[:3], "frames": [2, 2, 2],
            "frames_with_detection": [2, 2, 2], "detections": [2, 3, 2],
            "detection_rate": [1.0, 1.0, 1.0], "score_max": [0.7, 0.8, 0.6]})
        # Coverage spans 10 hours; a further 5 hours have no imagery at all.
        pd.DataFrame({"hour": hours, "images": [60] * 10}).to_csv(
            f"{tmp}/coverage_test-beach_hourly.csv", index=False)

        merged, ok = a.apply_coverage(detections, "test-beach")
        check("coverage was found", ok)
        check("every hour with imagery is present", len(merged) == 10, len(merged))
        check("hours the camera did not see stay absent",
              merged["hour"].max() == hours[-1], merged["hour"].max())
        zeros = merged[merged["frames_with_detection"] == 0]
        check("the quiet hours become observed zeros", len(zeros) == 7, len(zeros))
        check("their detection_rate is 0, not NaN",
              (zeros["detection_rate"] == 0).all(), zeros["detection_rate"].tolist())
        check("the target now varies", merged["detection_rate"].nunique() > 1,
              merged["detection_rate"].nunique())
        # 2 detections in 60 images, not 2 in 2.
        first = merged[merged["hour"] == hours[0]].iloc[0]
        check("rate is against images examined, not elements published",
              abs(first["detection_rate"] - 2 / 60) < 1e-9, first["detection_rate"])

        missing, ok = a.apply_coverage(detections, "no-such-camera")
        check("a missing coverage file is reported, not assumed", not ok)
        check("and the frame is returned untouched", len(missing) == 3, len(missing))
    finally:
        a.DATA_DIR = old_data
        shutil.rmtree(tmp)


def test_between_site_effect_is_caught():
    print("\nwithin-site control on a purely between-site effect")
    # Two beaches. One is dirtier AND sits at a gauge with higher water level.
    # There is NO relationship between level and bacteria inside either beach.
    # Pooled, that looks like a strong tide effect; within site, nothing.
    rows = []
    for site, level_base, dirt in [("Clean Beach", 0.5, 1.0),
                                   ("Dirty Beach", 2.5, 3.0)]:
        for _ in range(80):
            rows.append({
                "site": site,
                "level_m": level_base + RNG.normal(scale=0.2),
                "log_value": dirt + RNG.normal(scale=0.2),
                "date": pd.Timestamp("2025-06-15")})
    frame = pd.DataFrame(rows)

    pooled_rho, _, _ = a.spearman(frame["level_m"], frame["log_value"])
    within_rho, _, _ = a.spearman(a.demean_by(frame["level_m"], frame["site"]),
                                  a.demean_by(frame["log_value"], frame["site"]))
    check("pooled shows a strong effect that does not exist",
          pooled_rho > 0.7, pooled_rho)
    check("within site it vanishes", abs(within_rho) < 0.15, within_rho)

    # And per-site, neither beach shows it on its own.
    for site, group in frame.groupby("site"):
        rho, n, _ = a.spearman(group["level_m"], group["log_value"])
        check(f"{site} alone shows nothing", abs(rho) < 0.25, rho)


def test_focus_tide_reports_per_site():
    print("\nfocus_tide")
    rows = []
    for site, rho_target in [("Sausalito - Galilee Harbor", -0.6),
                             ("Carpinteria State Beach, CA", 0.0)]:
        level = RNG.normal(size=60)
        noise = RNG.normal(size=60)
        value = rho_target * level + np.sqrt(1 - rho_target ** 2) * noise
        for lv, val in zip(level, value):
            rows.append({"group": "ENT", "site": site, "level_m": lv,
                         "log_value": val, "date": pd.Timestamp("2025-06-15")})
    # A third site with too few samples to judge.
    for _ in range(8):
        rows.append({"group": "ENT", "site": "Capitola Wharf",
                     "level_m": RNG.normal(), "log_value": RNG.normal(),
                     "date": pd.Timestamp("2025-06-15")})

    table = a.focus_tide(pd.DataFrame(rows))
    check("one row per site-analyte pair", len(table) == 3, len(table))
    bay = table[table["site"] == "Sausalito - Galilee Harbor"].iloc[0]
    check("the planted negative effect is found", bay["rho"] < -0.4, bay["rho"])
    check("its setting is labelled", bay["setting"] == "enclosed bay",
          bay["setting"])
    flat = table[table["site"] == "Carpinteria State Beach, CA"].iloc[0]
    check("the flat site stays flat", abs(flat["rho"]) < 0.25, flat["rho"])
    thin = table[table["site"] == "Capitola Wharf"].iloc[0]
    check("an underpowered site is flagged, not dropped",
          thin["powered"] == "underpowered" and thin["n"] == 8,
          (thin["powered"], thin["n"]))
    check("and its rho is withheld", pd.isna(thin["rho"]), thin["rho"])


def test_thin_join_is_flagged():
    print("\nassemble_rip on windows that barely overlap")
    tmp = tempfile.mkdtemp()
    old_data, old_sites = a.DATA_DIR, a.SITES_CSV
    try:
        # Rip hours in June 2025; conditions starting September 2025.
        rip_hours = pd.date_range("2025-06-01 15:00", periods=200, freq="h", tz="UTC")
        pd.DataFrame({"hour": rip_hours, "frames": 4,
                      "frames_with_detection": 4, "detections": 4,
                      "detection_rate": 1.0, "score_max": 0.7,
                      "bbox_area_max": 100.0}).to_csv(
            f"{tmp}/rip_test-beach_hourly.csv", index=False)
        grid_hours = pd.date_range("2025-09-01", periods=500, freq="h", tz="UTC")
        pd.DataFrame({"time": grid_hours,
                      "wind_speed_10m": RNG.normal(size=500),
                      "temperature_2m": RNG.normal(size=500)}).to_csv(
            f"{tmp}/gridded_{a.grid_slug('Test Beach')}.csv", index=False)
        pd.DataFrame([{"camera_name": "Test Beach", "lat": 36.9, "lon": -122.0,
                       "buoy_id": None, "precip_station_id": None,
                       "wq_station_id": None, "wq_station_name": None,
                       "has_all_four": True}]).to_csv(f"{tmp}/sites.csv", index=False)

        a.DATA_DIR, a.SITES_CSV = tmp, f"{tmp}/sites.csv"
        frame, name = a.assemble_rip(a.load_sites())
        check("the frame is still returned", frame is not None)
        overlap = frame["wind_speed_10m"].notna().sum()
        check("almost nothing joined", overlap == 0, overlap)
    finally:
        a.DATA_DIR, a.SITES_CSV = old_data, old_sites
        shutil.rmtree(tmp)


def test_seasonal_control():
    print("\nseasonal control on a driver that is really the calendar")
    # Water temperature and the target both follow an annual cycle, with no
    # relationship inside a given month. Season is to a year of hourly data
    # what hour-of-day is to a daylight-only feed.
    hours = pd.date_range("2025-09-01", periods=24 * 350, freq="h", tz="UTC")
    frame = pd.DataFrame({"hour": hours})
    frame["month"] = frame["hour"].dt.month
    frame["hour_of_day"] = frame["hour"].dt.hour
    seasonal = np.cos(2 * np.pi * frame["hour"].dt.dayofyear / 365)
    frame["WTMP"] = 13 + 3 * seasonal
    frame["real"] = RNG.normal(size=len(frame))
    frame["y"] = 2 * seasonal + 0.5 * frame["real"] + RNG.normal(scale=0.1, size=len(frame))
    frame["hr_mo"] = (frame["hour_of_day"].astype(str) + "-"
                      + frame["month"].astype(str))

    raw, _, _ = a.spearman(frame["WTMP"], frame["y"])
    check("the seasonal variable correlates strongly raw", raw > 0.8, raw)
    controlled, _, _ = a.spearman(a.demean_by(frame["WTMP"], frame["hr_mo"]),
                                  a.demean_by(frame["y"], frame["hr_mo"]))
    # Demeaning by month removes the between-month signal but not the trend
    # WITHIN each month, so a perfectly seasonal driver drops far without
    # reaching zero. 0.94 -> 0.38 is the control working, not failing.
    check("and collapses once month is removed", abs(controlled) < 0.5, controlled)

    real_raw, _, _ = a.spearman(frame["real"], frame["y"])
    real_ctrl, _, _ = a.spearman(a.demean_by(frame["real"], frame["hr_mo"]),
                                 a.demean_by(frame["y"], frame["hr_mo"]))
    check("a genuine driver survives the control", real_ctrl > 0.8,
          (real_raw, real_ctrl))

    table = a.report_correlations(
        frame, "y", ["WTMP", "real"],
        controls=[("hr", frame["hour_of_day"]), ("hrmo", frame["hr_mo"])])
    check("the table ranks by the strictest control",
          table.iloc[0]["predictor"] == "real", table["predictor"].tolist())
    check("both control columns are present",
          {"rho_hr", "rho_hrmo"} <= set(table.columns), list(table.columns))


def test_non_finite_rows_dropped():
    print("\nstandardized_ols with an inf in the target")
    n = 200
    frame = pd.DataFrame({"a": RNG.normal(size=n), "b": RNG.normal(size=n)})
    frame["y"] = 2 * frame["a"] + RNG.normal(scale=0.2, size=n)
    frame.loc[5, "y"] = np.inf
    fit = a.standardized_ols(frame, "y", ["a", "b"])
    check("a fit is still produced", fit is not None)
    check("n reports the rows actually fitted", fit["n"] == n - 1, fit["n"])
    check("and the coefficients are finite",
          fit["beta"] is not None and bool(np.isfinite(fit["beta"]).all()),
          str(fit["beta"]))
    check("R2 is finite", np.isfinite(fit["r2"]), fit["r2"])


def test_seasonal_driver_is_indistinguishable():
    print("\na driver whose variation is almost entirely seasonal")
    # This is the case APD presents. The driver genuinely causes y, but it
    # barely moves within a month, so demeaning by month leaves almost nothing
    # of it and the correlation collapses. The data cannot separate "season
    # was the real cause" from "the cause only varies with season" -- and that
    # ambiguity is the finding, not a defect to code around.
    rows = []
    for month in range(1, 13):
        for _ in range(90):
            rows.append({"month": month})
    frame = pd.DataFrame(rows)
    seasonal = np.cos(2 * np.pi * frame["month"] / 12)
    frame["driver"] = 3 * seasonal + 0.15 * RNG.normal(size=len(frame))
    frame["y"] = frame["driver"] + RNG.normal(scale=1.5, size=len(frame))

    raw, _, _ = a.spearman(frame["driver"], frame["y"])
    controlled, _, _ = a.spearman(a.demean_by(frame["driver"], frame["month"]),
                                  a.demean_by(frame["y"], frame["month"]))
    check("strong raw", raw > 0.5, raw)
    check("collapses under the month control even though it is causal",
          abs(controlled) < 0.2, controlled)
    print(f"    raw {raw:.3f}  controlled {controlled:.3f}"
          "  <- causal, and still collapses")

    # A driver with real within-month variation survives, which is what makes
    # the comparison informative rather than uniformly destructive.
    frame["fast"] = RNG.normal(size=len(frame))
    frame["y2"] = frame["fast"] + RNG.normal(scale=0.3, size=len(frame))
    fast_ctrl, _, _ = a.spearman(a.demean_by(frame["fast"], frame["month"]),
                                 a.demean_by(frame["y2"], frame["month"]))
    check("a fast-varying driver survives the same control", fast_ctrl > 0.8,
          fast_ctrl)


def test_variance_explained():
    print("\nvariance_explained")
    months = pd.Series(list(range(1, 13)) * 100)

    # Entirely determined by month: the group mean IS the value.
    pure = months.map({m: m * 1.0 for m in range(1, 13)})
    check("a pure group effect is 1.0",
          abs(a.variance_explained(pure, months) - 1.0) < 1e-9,
          a.variance_explained(pure, months))

    # Many groups over few rows fit noise; the correction should pull an
    # otherwise-meaningless grouping back toward zero.
    few = pd.Series(RNG.normal(size=300))
    many_cells = pd.Series(range(150)).repeat(2).reset_index(drop=True)
    share = a.variance_explained(few, many_cells)
    check("150 cells over 300 rows of noise is corrected toward 0",
          abs(share) < 0.15, share)

    # Independent of month.
    noise = pd.Series(RNG.normal(size=len(months)))
    share = a.variance_explained(noise, months)
    check("noise is near 0", abs(share) < 0.05, share)

    # Half and half: a seasonal component plus equal-variance noise.
    mixed = pure / pure.std() + pd.Series(RNG.normal(size=len(months)))
    share = a.variance_explained(mixed, months)
    check("a mixed signal lands in between", 0.3 < share < 0.7, share)

    check("a constant series is NaN rather than 0 or 1",
          math.isnan(a.variance_explained(pd.Series([5.0] * 200),
                                          pd.Series([1, 2] * 100))))
    check("too few rows is NaN",
          math.isnan(a.variance_explained(pd.Series([1.0, 2.0]),
                                          pd.Series([1, 2]))))


def test_analyte_key():
    print("\nanalyte_key")
    cases = {"Enterococcus": "ENT", "ENTEROCOCCUS": "ENT", "E. coli": "ECOLI",
             "Escherichia coli": "ECOLI", "Total Coliform": "TOTAL",
             "Fecal Coliform": "FECAL", "Turbidity": None}
    for text, want in cases.items():
        got = a.analyte_key(text)
        check(f"{text!r} -> {want}", got == want, got)


# ---------------------------------------------------------------------------
# Site bookkeeping across the California file and the national scan
# ---------------------------------------------------------------------------

def test_site_sources():
    """A camera found by scan_cameras.py must be analysable.

    candidate_sites_ranked.csv only ever held California, so before this a rip
    pull for Virginia Beach landed on disk and then reported 'not among the
    qualifying sites' — a bookkeeping failure that reads like a data failure.
    """
    print("site list merges both sources")
    import tempfile
    import analyze_drivers as ad

    with tempfile.TemporaryDirectory() as tmp:
        ca = os.path.join(tmp, "ca.csv")
        national = os.path.join(tmp, "national.csv")
        pd.DataFrame([{"camera_name": "Walton Lighthouse, Santa Cruz, CA",
                       "lat": 36.96, "lon": -122.02, "buoy_id": "46236",
                       "has_all_four": True},
                      {"camera_name": "Dropped Site", "lat": 1, "lon": 1,
                       "buoy_id": "x", "has_all_four": False}]).to_csv(ca, index=False)
        pd.DataFrame([{"camera": "Hampton Inn Oceanfront South at Virginia Beach",
                       "lat": 36.83, "lon": -75.97, "buoy_id": "44099"},
                      {"camera": "Walton Lighthouse, Santa Cruz, CA",
                       "lat": 36.96, "lon": -122.02, "buoy_id": "WRONG"}]).to_csv(
            national, index=False)

        old_sites, old_cand = ad.SITES_CSV, ad.CANDIDATES_CSV
        ad.SITES_CSV, ad.CANDIDATES_CSV = ca, national
        try:
            sites = ad.load_sites()
        finally:
            ad.SITES_CSV, ad.CANDIDATES_CSV = old_sites, old_cand

    names = list(sites["camera_name"])
    check("Virginia Beach is now a known site",
          any("Virginia Beach" in n for n in names), str(names))
    check("Walton survives", any("Walton" in n for n in names))
    check("has_all_four is still honoured", "Dropped Site" not in names)
    check("no duplicate rows for a camera in both files",
          len(names) == len(set(names)), str(names))
    walton = sites[sites["camera_name"].str.contains("Walton")].iloc[0]
    check("the California file wins on a conflict", walton["buoy_id"] == "46236",
          str(walton["buoy_id"]))


# ---------------------------------------------------------------------------
# Modelled waves for sites with no buoy
# ---------------------------------------------------------------------------

def test_marine_waves():
    """Virginia Beach has no buoy publishing waves, so without the marine file
    its rip analysis would run with no wave predictor at all. The modelled
    columns must load, and must stay distinguishable from measured ones."""
    print("modelled waves load and stay separate from measured")
    import tempfile
    import analyze_drivers as ad

    with tempfile.TemporaryDirectory() as tmp:
        name = "Test Beach"
        slug = ad.grid_slug(name)
        pd.DataFrame({
            "time": pd.date_range("2026-05-01", periods=4, freq="h", tz="UTC"),
            "wave_height": [1.0, 1.2, 1.4, 1.6],
            "wave_period": [7.0, 7.5, 8.0, 8.5],
            "wave_direction": [90.0, 90.0, 270.0, 270.0],
            "swell_wave_height": [0.8, 0.9, 1.0, 1.1],
            "swell_wave_period": [11.0, 11.5, 12.0, 12.5],
            "swell_wave_direction": [90.0, 90.0, 270.0, 270.0],
            "wind_wave_height": [0.3, 0.3, 0.4, 0.4],
            "wind_wave_period": [4.0, 4.0, 4.5, 4.5],
            "irrelevant": [1, 2, 3, 4],
        }).to_csv(os.path.join(tmp, f"marine_{slug}.csv"), index=False)

        old = ad.DATA_DIR
        ad.DATA_DIR = tmp
        try:
            marine = ad.load_marine(name)
            missing = ad.load_marine("Nowhere At All")
        finally:
            ad.DATA_DIR = old

    check("marine file loads", marine is not None and len(marine) == 4,
          "None" if marine is None else str(len(marine)))
    check("keeps the wave columns",
          all(c in marine.columns for c in ("wave_height", "swell_wave_period")))
    check("drops columns it was not asked for", "irrelevant" not in marine.columns)
    check("indexed by hour", marine.index.name == "hour")
    check("a site with no marine file is None, not a crash", missing is None)

    # Shore normal 90 deg = facing east. A wave FROM 90 is onshore (+1),
    # one from 270 is offshore (-1). Getting this backwards would invert every
    # onshore result, so it is asserted rather than assumed.
    onshore = ad.angular_component(pd.Series([90.0, 270.0]), 90.0)
    check("wave from the shore normal is fully onshore",
          abs(onshore.iloc[0] - 1.0) < 1e-9, str(onshore.iloc[0]))
    check("wave from the opposite side is offshore",
          abs(onshore.iloc[1] + 1.0) < 1e-9, str(onshore.iloc[1]))

    check("modelled and measured wave names do not collide",
          not set(ad.MARINE_COLUMNS) & {"WVHT", "DPD", "APD", "MWD", "WTMP"},
          str(set(ad.MARINE_COLUMNS) & {"WVHT", "DPD", "APD", "MWD", "WTMP"}))
    check("both onshore variants are predictors",
          "swell_onshore" in ad.RIP_PREDICTORS
          and "swell_onshore_model" in ad.RIP_PREDICTORS)
    check("Virginia Beach has a shore normal configured",
          any("Virginia Beach" in k for k in ad.SHORE_NORMAL_DEG))


# ---------------------------------------------------------------------------
# A missing station must stay missing
# ---------------------------------------------------------------------------

def test_distant_station_is_refused():
    """No water temperature was ever pulled for Virginia Beach, so pick_coops
    returned the nearest file on disk — San Diego, 3,762 km away — and it was
    joined at 100% overlap and reported as local. A missing input has to stay
    missing."""
    print("a station on another coastline is refused")
    import tempfile
    import types
    import analyze_drivers as ad

    stations = pd.DataFrame([
        {"station_id": "9410170", "lat": 32.71, "lon": -117.17},   # San Diego
        {"station_id": "8638901", "lat": 37.03, "lon": -76.08},    # Chesapeake
    ])
    import find_candidate_sites
    fake_po = types.SimpleNamespace(coops_stations=lambda kind: stations,
                                    f=find_candidate_sites)

    with tempfile.TemporaryDirectory() as tmp:
        for station in ("9410170", "8638901"):
            pd.DataFrame({"time": ["2026-05-01T00:00:00Z"], "x": [1.0]}).to_csv(
                os.path.join(tmp, f"watertemp_{station}.csv"), index=False)
        old_dir, old_po = ad.DATA_DIR, sys.modules.get("pull_observations")
        ad.DATA_DIR = tmp
        sys.modules["pull_observations"] = fake_po
        try:
            # Virginia Beach: the Chesapeake file is 24 km away and wins.
            near = ad.pick_coops("watertemp", 36.84, -75.97)
            # Now delete it, leaving only San Diego 3,762 km away.
            os.remove(os.path.join(tmp, "watertemp_8638901.csv"))
            far = ad.pick_coops("watertemp", 36.84, -75.97)
            # A single file is still distance-checked, not waved through.
            single = ad.pick_coops("watertemp", 32.71, -117.17)
        finally:
            ad.DATA_DIR = old_dir
            if old_po is None:
                sys.modules.pop("pull_observations", None)
            else:
                sys.modules["pull_observations"] = old_po

    check("picks the nearby station", near is not None and "8638901" in near,
          str(near))
    check("refuses one on the wrong coast", far is None, str(far))
    check("the lone-file case is still checked", single is not None
          and "9410170" in single, str(single))
    check("the cap is a coastal distance, not a continental one",
          40 <= ad.MAX_STATION_KM <= 150, str(ad.MAX_STATION_KM))


def test_axial_offset():
    print("\nrip orientation is a line, not an arrow")
    normal = 90.0
    got = a.axial_offset(pd.Series([90.0, 270.0, 0.0, 180.0, 135.0]), normal)
    check("straight offshore is 0", got.iloc[0] == 0, got.iloc[0])
    check("the same line pointing back is also 0", got.iloc[1] == 0, got.iloc[1])
    check("along the beach is 90", got.iloc[2] == 90, got.iloc[2])
    check("and so is the other way along the beach", got.iloc[3] == 90,
          got.iloc[3])
    check("never exceeds 90", float(got.max()) <= 90, float(got.max()))


def test_positives_only_drops_presence():
    print("\n--positives-only keeps the rips and drops the presence question")
    check("presence targets are named",
          set(a.PRESENCE_TARGETS) == {"detection_rate", "detections",
                                      "doubt_rate"},
          sorted(a.PRESENCE_TARGETS))
    check("size and orientation are not presence targets",
          not ({"bbox_area_max", "rip_axis_offset_deg", "score_max"}
               & set(a.PRESENCE_TARGETS)),
          sorted(a.PRESENCE_TARGETS))
    check("orientation is a reported target",
          "rip_axis_offset_deg" in a.RIP_TARGETS, a.RIP_TARGETS)


def write_mop_fixture(tmp, hours, normal=206.0, direction=206.0):
    """A MOP file as pull_cdip_mop.py writes it, for the same hours.

    wave_direction and radiation_stress_* are absent on purpose: at SC130 they
    are fill values end to end and the puller drops them, so the analysis has
    to cope with a MOP file that carries only four columns.
    """
    pd.DataFrame({
        "time": hours,
        "wave_height": 0.8 + 0.2 * RNG.random(len(hours)),
        "wave_period": 8 + RNG.normal(size=len(hours)),
        "wave_period_peak": 11 + RNG.normal(size=len(hours)),
        "wave_direction_peak": direction,
        "product": "hindcast",
        "mop_id": "SC130",
        "shore_normal_deg": normal,
        "water_depth_m": 15.0,
    }).to_csv(f"{tmp}/mop_{a.grid_slug('Test Beach')}.csv", index=False)


def test_mop_columns_are_kept_apart_from_the_reanalysis():
    print("\nMOP columns arrive prefixed, so they cannot pass as Open-Meteo")
    tmp = tempfile.mkdtemp()
    old_data = a.DATA_DIR
    try:
        a.DATA_DIR = tmp
        a.MOP_META.clear()
        hours = pd.date_range("2025-06-01", periods=48, freq="h", tz="UTC")
        write_mop_fixture(tmp, hours)
        frame = a.load_mop("Test Beach")
        check("wave height comes back under its own name",
              "mop_wave_height" in frame.columns, list(frame.columns))
        check("and not under the Open-Meteo name",
              "wave_height" not in frame.columns, list(frame.columns))
        check("a column the puller dropped is simply absent",
              "mop_wave_direction" not in frame.columns)
        check("the published shore normal is captured",
              a.MOP_META["Test Beach"]["shore_normal_deg"] == 206.0)
        check("so is the point it came from",
              a.MOP_META["Test Beach"]["mop_id"] == "SC130")
    finally:
        a.DATA_DIR = old_data
        a.MOP_META.clear()


def test_the_published_shore_normal_beats_the_one_read_off_a_map():
    print("\nCDIP's 206 deg overrides the 180 deg assumed for Santa Cruz")
    tmp = tempfile.mkdtemp()
    old_data, old_sites = a.DATA_DIR, a.SITES_CSV
    try:
        write_rip_fixture(tmp)
        rip = pd.read_csv(f"{tmp}/rip_test-beach_hourly.csv")
        hours = pd.to_datetime(rip["hour"], utc=True)
        write_mop_fixture(tmp, hours, normal=206.0, direction=206.0)
        a.DATA_DIR, a.SITES_CSV = tmp, f"{tmp}/sites.csv"
        a.SHORE_NORMAL_DEG["Test Beach"] = 180.0
        a.MOP_META.clear()
        sites = a.load_sites()
        frame, _ = a.assemble_rip(sites)
        check("the MOP waves joined onto the rip hours",
              "mop_wave_height" in frame.columns)
        onshore = frame["mop_wave_onshore_peak"].dropna()
        check("an onshore component is computed from the MOP direction",
              len(onshore) > 0)
        # Waves arriving exactly along the shore normal are fully onshore. At
        # the assumed 180 deg this would read cos(26 deg) = 0.90 instead.
        check("and it uses 206 deg, not the assumed 180",
              abs(onshore.iloc[0] - 1.0) < 1e-6, onshore.iloc[0])
    finally:
        a.DATA_DIR, a.SITES_CSV = old_data, old_sites
        a.SHORE_NORMAL_DEG.pop("Test Beach", None)
        a.MOP_META.clear()


if __name__ == "__main__":
    test_spearman()
    test_demean_by()
    test_standardized_ols()
    test_analyte_key()
    test_variance_explained()
    test_seasonal_driver_is_indistinguishable()
    test_seasonal_control()
    test_non_finite_rows_dropped()
    test_thin_join_is_flagged()
    test_between_site_effect_is_caught()
    test_focus_tide_reports_per_site()
    test_coverage_creates_observed_zeros()
    test_regression_drops_identical_columns()
    test_regression_withholds_when_singular()
    test_degenerate_target_is_named()
    test_rip_recovers_planted_driver()
    test_wq_recovers_planted_driver()
    test_site_sources()
    test_marine_waves()
    test_mop_columns_are_kept_apart_from_the_reanalysis()
    test_the_published_shore_normal_beats_the_one_read_off_a_map()
    test_distant_station_is_refused()
    test_axial_offset()
    test_positives_only_drops_presence()
    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
    raise SystemExit(1 if FAILURES else 0)