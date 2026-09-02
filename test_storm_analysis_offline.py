#!/usr/bin/env python3
"""Offline checks for analyze_storm.py — no network, no real files.

The two failure modes worth testing are the two the module exists to avoid:
reporting beach attendance as an ocean driver, and reporting a wave signal
that is really the swimming season.
"""
import io
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

import analyze_storm as st
import analyze_drivers as ad

FAILURES = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def _days(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2010-01-01", periods=n, freq="D")
    return days, rng


def _planted(seed=0):
    """A genuine wave driver with no seasonal and no weekly component."""
    days, rng = _days(seed=seed)
    wave = pd.Series(rng.gamma(2.0, 0.5, len(days)), index=days)
    prob = (wave.rank(pct=True) ** 4) * 0.35
    events = pd.Series((rng.random(len(days)) < prob).astype(int), index=days)
    return days, pd.DataFrame({"any_event": events, "wave_height": wave},
                              index=days)


def _strata(days):
    month = pd.Series(days.month, index=days)
    dow = pd.Series(days.dayofweek, index=days)
    return [("mo", month), ("modow", month.astype(str) + "-" + dow.astype(str))]


def _run(frame, days, predictor):
    out = io.StringIO()
    with redirect_stdout(out):
        table = st.report_stratified(frame, "any_event", [predictor],
                                     _strata(days))
    row = table.iloc[0]
    return float(row["rho"]), float(row["rho_mo"]), float(row["rho_modow"])


def test_a_real_driver_survives_both_strata():
    print("\na planted wave driver survives ranking within month and weekday")
    days, frame = _planted()
    raw, mo, modow = _run(frame, days, "wave_height")
    check("the raw correlation finds it", raw > 0.25, round(raw, 3))
    check("within-month keeps it", mo > 0.25, round(mo, 3))
    check("within month-and-weekday keeps it", modow > 0.25, round(modow, 3))
    check("the controls cost it almost nothing", raw - modow < 0.05,
          (round(raw, 3), round(modow, 3)))


def test_demeaning_a_binary_target_loses_a_real_signal():
    """Why this module does not use the demeaning control the rest of the
    project uses. Demeaning a column of 0s and 1s replaces its ties with one
    value per stratum, ordered by that stratum's event rate; ranking those
    mixes stratum noise into the target. It is not a safer control -- it
    removes no more confounding and costs a third of a real effect."""
    print("\ndemeaning a BINARY target attenuates it; within-rank does not")
    days, frame = _planted()
    inter = (pd.Series(days.month, index=days).astype(str) + "-"
             + pd.Series(days.dayofweek, index=days).astype(str))
    demeaned = ad.spearman(ad.demean_by(frame["wave_height"], inter),
                           ad.demean_by(frame["any_event"], inter))[0]
    ranked = ad.spearman(st.within_rank(frame["wave_height"], inter),
                         frame["any_event"])[0]
    check("demeaning both sides loses a large part of the signal",
          demeaned < ranked - 0.08, (round(demeaned, 3), round(ranked, 3)))
    check("the method this module uses keeps it", ranked > 0.25, round(ranked, 3))


def test_a_weekend_artifact_is_removed():
    """A predictor that is really 'it is Saturday'. The event series here has
    no ocean in it at all."""
    print("\na predictor that is really 'it is Saturday' does not survive")
    days, rng = _days()
    fake = pd.Series((days.dayofweek >= 5).astype(float)
                     + rng.normal(0, 0.25, len(days)), index=days)
    events = pd.Series(((days.dayofweek >= 5)
                        & (rng.random(len(days)) < 0.25)).astype(int), index=days)
    frame = pd.DataFrame({"any_event": events, "fake": fake}, index=days)
    raw, mo, modow = _run(frame, days, "fake")
    check("uncontrolled, it looks like a strong driver", raw > 0.3, round(raw, 3))
    check("ranking within month alone does NOT catch it", mo > 0.3, round(mo, 3))
    check("ranking within month and weekday kills it", abs(modow) < 0.1,
          round(modow, 3))


def test_a_seasonal_artifact_is_removed():
    print("\na predictor that is really 'it is July' does not survive either")
    days, rng = _days()
    warm = pd.Series(days.month.isin([6, 7, 8]).astype(float)
                     + rng.normal(0, 0.3, len(days)), index=days)
    events = pd.Series((days.month.isin([6, 7, 8])
                        & (rng.random(len(days)) < 0.3)).astype(int), index=days)
    frame = pd.DataFrame({"any_event": events, "warm": warm}, index=days)
    raw, mo, modow = _run(frame, days, "warm")
    check("uncontrolled it is strong", raw > 0.3, round(raw, 3))
    check("ranking within month kills it", abs(mo) < 0.1, round(mo, 3))
    check("and it stays dead with weekday added", abs(modow) < 0.1, round(modow, 3))


def test_season_is_derived_from_where_the_events_are():
    print("\nthe swim season is read off the events, not assumed")
    days = pd.date_range("2010-01-01", periods=3650, freq="D")
    events = pd.Series(0, index=days)
    events[days.month.isin([6, 7, 8])] = 1
    frame = pd.DataFrame({"events": events}, index=days)
    check("only the months carrying events are kept",
          st.season_months(frame) == [6, 7, 8], st.season_months(frame))

    quiet = pd.DataFrame({"events": pd.Series(0, index=days)}, index=days)
    check("a record with no events at all returns every month, not an empty set",
          len(st.season_months(quiet)) == 12, st.season_months(quiet))


def test_contrast_reads_as_a_percentile():
    print("\nthe event-day contrast is expressed where a lifeguard could use it")
    days, rng = _days()
    wave = pd.Series(rng.normal(1.0, 0.3, len(days)), index=days)
    hit = wave > wave.quantile(0.9)
    frame = pd.DataFrame({"any_event": hit.astype(int), "wave_height": wave},
                         index=days)
    table = st.contrast(frame, "any_event", ["wave_height"])
    pct = float(table["pctile_raw"].iloc[0])
    check("event days sit high in the quiet-day distribution", pct > 90, round(pct, 1))
    check("and the shift is reported against 50", round(pct - 50, 1) ==
          round(float(table["shift"].iloc[0]), 1))

    flat = pd.DataFrame({"any_event": (rng.random(len(days)) < 0.1).astype(int),
                         "wave_height": wave}, index=days)
    pct = float(st.contrast(flat, "any_event", ["wave_height"])
                ["pctile_raw"].iloc[0])
    check("an unrelated predictor sits near 50", 40 < pct < 60, round(pct, 1))


def test_daily_max_is_kept_alongside_the_mean():
    """A day that peaked dangerous and averaged calm is the case that matters."""
    print("\na daily peak is carried, not flattened into the mean")
    tmp = tempfile.mkdtemp()
    old_dir = ad.DATA_DIR
    try:
        ad.DATA_DIR = tmp
        hours = pd.date_range("2015-06-01", periods=48, freq="h", tz="UTC")
        height = np.full(48, 0.4)
        height[10:13] = 3.0          # one dangerous morning
        pd.DataFrame({"time": hours.strftime("%Y-%m-%dT%H:%M"),
                      "wave_height": height,
                      "wave_period": 8.0}).to_csv(
            os.path.join(tmp, f"marine_{ad.grid_slug('Testville')}.csv"),
            index=False)
        daily = st.daily_conditions("Testville")
        check("both columns exist",
              {"wave_height", "wave_height_max"} <= set(daily.columns),
              sorted(daily.columns))
        first = daily.iloc[0]
        check("the mean hides the peak", first["wave_height"] < 1.0,
              round(float(first["wave_height"]), 2))
        check("the max does not", abs(first["wave_height_max"] - 3.0) < 1e-6,
              float(first["wave_height_max"]))
        check("a column with no peak meaning gets no _max",
              "wave_period_max" not in daily.columns, sorted(daily.columns))
    finally:
        ad.DATA_DIR = old_dir
        shutil.rmtree(tmp, ignore_errors=True)


def test_thin_coverage_is_visible_before_any_correlation():
    print("\na predictor present on few days is flagged, not silently ranked")
    days = pd.date_range("2015-01-01", periods=400, freq="D")
    storm_frame = pd.DataFrame({
        "events": 0, "deaths": 0, "injuries": 0,
        "any_event": 0, "any_casualty": 0}, index=days)
    storm_frame.index.name = "date"
    storm_frame.iloc[:40, storm_frame.columns.get_loc("any_event")] = 1
    conditions = pd.DataFrame({"wave_height": 1.0, "wind_speed_10m": np.nan},
                              index=days)
    conditions.iloc[:50, conditions.columns.get_loc("wind_speed_10m")] = 5.0
    out = io.StringIO()
    with redirect_stdout(out):
        st.report_overlap(storm_frame, conditions)
    text = out.getvalue()
    check("the thin predictor is marked", "wind_speed_10m" in text
          and "<- thin" in text, text)
    check("the complete one is not",
          "wave_height" in text
          and not any("wave_height" in line and "thin" in line
                      for line in text.splitlines()), text)


def test_the_contrast_table_is_also_controlled():
    """The percentile table is the legible one, so it is the one a reader
    lifts onto a slide. Uncontrolled it reports the swimming season."""
    print("\nthe percentile table separates a seasonal artifact from a driver")
    days, rng = _days()
    warm = pd.Series(days.month.isin([6, 7, 8]).astype(float)
                     + rng.normal(0, 0.3, len(days)), index=days)
    events = pd.Series((days.month.isin([6, 7, 8])
                        & (rng.random(len(days)) < 0.3)).astype(int), index=days)
    frame = pd.DataFrame({"any_event": events, "warm": warm}, index=days)
    key = pd.Series(days.month, index=days).astype(str) + "-" + \
        pd.Series(days.dayofweek, index=days).astype(str)
    table = st.contrast(frame, "any_event", ["warm"], strata_key=key)
    raw = float(table["pctile_raw"].iloc[0])
    ctrl = float(table["pctile_ctrl"].iloc[0])
    check("uncontrolled, the July artifact looks like a big effect", raw > 75,
          round(raw, 1))
    check("controlled, it sits at the null", abs(ctrl - 50) < 6, round(ctrl, 1))
    check("the table is ranked by the controlled column",
          abs(float(table["shift"].iloc[0]) - (ctrl - 50)) < 1e-6)


if __name__ == "__main__":
    test_a_real_driver_survives_both_strata()
    test_demeaning_a_binary_target_loses_a_real_signal()
    test_a_weekend_artifact_is_removed()
    test_a_seasonal_artifact_is_removed()
    test_season_is_derived_from_where_the_events_are()
    test_contrast_reads_as_a_percentile()
    test_the_contrast_table_is_also_controlled()
    test_daily_max_is_kept_alongside_the_mean()
    test_thin_coverage_is_visible_before_any_correlation()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL PASS")
