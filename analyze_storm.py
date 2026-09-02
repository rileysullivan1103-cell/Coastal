#!/usr/bin/env python3
"""What conditions precede a day when a rip current hurt somebody.

Every other analysis in this project asks whether conditions predict our own
instrument -- a YOLOv8 detector firing, a curator drawing a box. This one asks
whether conditions predict an outcome recorded by somebody else entirely, and
that is the only question here whose answer would be worth telling a lifeguard
service.

It cannot be a join to the camera. Storm Events carries no coordinates and its
zone is a stretch of county coast, and the WebCOOS imagery only goes back a
couple of years while the casualty record runs from 2000. So this is a second,
independent regression on the same coast: conditions -> casualty over 26 years,
against conditions -> detection over the camera era. If the same predictors
come out on top in both, that is evidence the detector tracks something real.
If they disagree, the detector is tracking something else.

Three biases are built into the target and none of them can be removed, only
controlled:

  An event is logged when a person is in the water. December has no events
  because nobody swims, not because December has no rips. Every table is
  therefore reported month-demeaned, and again restricted to the season that
  actually carries events.

  Weekends carry more events than weekdays for the same reason. Demeaning by
  day-of-week is the only control that separates "the ocean was dangerous"
  from "the beach was full".

  Whether an event is logged at all depends on the local forecast office.
  That varies hugely between zones -- which is why this compares one zone
  against itself over time and never one zone against another.

    python analyze_storm.py --zone "NEW HANOVER" --camera "Wrightsville"
    python analyze_storm.py --zone "COASTAL BAY" --camera "Panama City" --probe
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import analyze_drivers as ad
import pull_storm_events as storm

DATA_DIR = "data"

# Daily mean is the wrong summary for a hazard that only has to happen once.
# A day whose waves peaked at 2 m for three hours and sat at 0.5 m otherwise
# is a dangerous day with an unremarkable mean, so both are carried and the
# tables show which one the signal lives in.
PEAK_COLUMNS = ("wave_height", "swell_wave_height", "wind_wave_height",
                "wind_speed_10m", "wind_gusts_10m", "precipitation")

PREDICTORS = [
    "wave_height", "wave_height_max", "wave_period", "wave_direction",
    "swell_wave_height", "swell_wave_height_max", "swell_wave_period",
    "wind_wave_height", "wind_wave_height_max", "wind_wave_period",
    "wind_speed_10m", "wind_speed_10m_max", "wind_gusts_10m_max",
    "temperature_2m", "precipitation", "precipitation_max",
]

TARGETS = ("any_event", "any_casualty", "events")

# A month is in the swim season if it carries at least this share of events.
# Below it the days are almost all structural zeros and only dilute the test.
SEASON_SHARE = 0.02


def load_storm(zone):
    path = f"{DATA_DIR}/storm_rip_{storm.slugify(zone)}_daily.csv"
    frame = ad.read_csv(path)
    if frame is None:
        sys.exit(f"{path} not found. Write it first with:\n"
                 f"  python pull_storm_events.py --zone {zone!r} --start-year 2000")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).set_index("date").sort_index()


def daily_conditions(camera_name):
    """Daily means, plus a daily max for the variables where a peak matters."""
    parts = []
    for loader in (ad.load_gridded, ad.load_marine):
        hourly = loader(camera_name)
        if hourly is None or hourly.empty:
            continue
        hourly = hourly.copy()
        hourly.index = hourly.index.tz_convert(None).normalize()
        hourly.index.name = "date"
        grouped = hourly.groupby("date")
        daily = grouped.mean(numeric_only=True)
        peaks = [c for c in hourly.columns if c in PEAK_COLUMNS]
        if peaks:
            daily = daily.join(grouped[peaks].max().add_suffix("_max"))
        parts.append(daily)
    if not parts:
        return None
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.join(part, how="outer", rsuffix="_x")
    return merged


def season_months(frame):
    """The months that carry events, by share of the total."""
    by_month = frame.groupby(frame.index.month)["events"].sum()
    total = by_month.sum()
    if not total:
        return sorted(by_month.index)
    return sorted(m for m, v in by_month.items() if v / total >= SEASON_SHARE)


def report_overlap(storm_frame, conditions):
    """Say what the join actually covered, before any correlation is shown."""
    print(f"\n{'=' * 74}\nWHAT JOINED\n{'=' * 74}")
    print(f"casualty record  {storm_frame.index.min():%Y-%m-%d} to "
          f"{storm_frame.index.max():%Y-%m-%d}  ({len(storm_frame):,} days, "
          f"{int(storm_frame['any_event'].sum())} with an event)")
    print(f"conditions       {conditions.index.min():%Y-%m-%d} to "
          f"{conditions.index.max():%Y-%m-%d}  ({len(conditions):,} days)")

    joined = storm_frame.join(conditions, how="inner")
    events = int(joined["any_event"].sum())
    print(f"overlap          {len(joined):,} days, {events} with an event")
    lost = int(storm_frame["any_event"].sum()) - events
    if lost:
        print(f"  {lost} event days fall outside the conditions span and are lost")

    print("\ncoverage of each predictor over the overlap "
          "(a predictor at 40% is a different sample from one at 100%):")
    for name in PREDICTORS:
        if name not in joined.columns:
            continue
        have = int(joined[name].notna().sum())
        on_events = int(joined.loc[joined["any_event"] == 1, name].notna().sum())
        flag = "" if have > 0.8 * len(joined) else "   <- thin"
        print(f"  {name:<26} {have:>6}/{len(joined)} days "
              f"({100 * have / len(joined):>5.1f}%), "
              f"{on_events}/{events} event days{flag}")
    return joined


def contrast(frame, target, predictors):
    """Median on event days vs quiet days, and where that sits in the quiet
    distribution.

    A rho against a binary target whose base rate is under one percent is
    small by construction and reads as 'no effect' to anyone who has not
    thought about it. The percentile says the same thing in a form a lifeguard
    could act on: 'the typical casualty day sat at the 78th percentile of
    ordinary days for wave height'.
    """
    rows = []
    hit = frame[target] > 0
    if hit.sum() < ad.MIN_N:
        return None
    for name in predictors:
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        on = values[hit].dropna()
        off = values[~hit].dropna()
        if len(on) < ad.MIN_N or len(off) < ad.MIN_N:
            continue
        pct = 100.0 * float((off < on.median()).mean())
        rows.append({"predictor": name, "n_event_days": len(on),
                     "median_event": on.median(), "median_quiet": off.median(),
                     "pctile_of_quiet": pct, "shift": pct - 50.0})
    if not rows:
        return None
    table = pd.DataFrame(rows)
    return table.reindex(table["shift"].abs().sort_values(ascending=False).index)


def within_rank(series, key):
    """The value's percentile WITHIN its own stratum.

    The obvious control -- demean both sides by month, as the rest of this
    project does -- is wrong for a binary target, and measurably so. Demeaning
    a column of 0s and 1s replaces the ties with one distinct value per
    stratum, ordered by that stratum's event rate, and ranking those injects
    stratum noise into the target. On a planted driver of true rho 0.30 the
    demeaned version reads 0.18 while removing no more confounding than this
    one, which reads 0.29. The loss scales with how tied the target is: nil
    for a continuous target like detection_rate, worst for a binary one.

    So the stratum is removed from the PREDICTOR and the target is left alone.
    'Was this a big-wave day for July?' is also the question a lifeguard
    actually asks.
    """
    return pd.to_numeric(series, errors="coerce").groupby(key).rank(pct=True)


def report_stratified(frame, target, predictors, strata, title=""):
    """Ranked table: raw, then the same correlation with each stratum removed
    from the predictor."""
    rows = []
    raw_target = pd.to_numeric(frame[target], errors="coerce")
    for name in predictors:
        if name not in frame.columns:
            continue
        rho, n, p = ad.spearman(frame[name], raw_target)
        row = {"predictor": name, "rho": rho, "n": n, "p": p}
        for suffix, key in strata:
            s_rho, _, s_p = ad.spearman(within_rank(frame[name], key), raw_target)
            row[f"rho_{suffix}"] = s_rho
            row[f"p_{suffix}"] = s_p
        rows.append(row)
    if not rows:
        print("  no usable predictors")
        return None
    table = pd.DataFrame(rows)
    key = f"rho_{strata[-1][0]}" if strata else "rho"
    table = table.reindex(table[key].abs().sort_values(ascending=False).index)
    if title:
        print(f"\n{title}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(4).to_string(index=False))
    strongest = table[key].abs().max()
    if pd.isna(strongest):
        print("  every correlation was under-powered; treat none of this as a finding")
    elif strongest < 0.15:
        print(f"  strongest |rho| is {strongest:.2f} — nothing here is a strong driver")
    return table


def exposure_report(frame):
    print(f"\n{'=' * 74}\nEXPOSURE, BEFORE ANY OCEAN\n{'=' * 74}")
    by_dow = frame.groupby(frame.index.dayofweek)["events"].sum()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("  " + "  ".join(f"{names[d]}:{int(by_dow.get(d, 0))}" for d in range(7)))
    total = int(by_dow.sum())
    if total:
        share = 100 * int(by_dow.get(5, 0) + by_dow.get(6, 0)) / total
        print(f"  Sat+Sun hold {share:.0f}% of events against 28.6% of days.")
        if share > 40:
            print("  Read every uncontrolled number below as partly this.")

    by_month = frame.groupby(frame.index.month)["events"].sum()
    print("\n  by month: " + "  ".join(
        f"{m:02d}:{int(by_month.get(m, 0))}" for m in range(1, 13)))
    season = season_months(frame)
    print(f"  season (months holding >={SEASON_SHARE:.0%} of events each): "
          f"{', '.join(f'{m:02d}' for m in season)}")
    return season


def run(joined, label):
    month = pd.Series(joined.index.month, index=joined.index)
    dow = pd.Series(joined.index.dayofweek, index=joined.index)
    strata = [("mo", month), ("modow", month.astype(str) + "-" + dow.astype(str))]

    for target in TARGETS:
        if target not in joined.columns:
            continue
        hits = int((joined[target] > 0).sum())
        if hits < ad.MIN_N:
            print(f"\n{target}: {hits} positive days, under the {ad.MIN_N} "
                  "floor — not reported")
            continue
        report_stratified(
            joined, target, PREDICTORS, strata,
            title=f"{label} — {target}  ({hits} positive of {len(joined)} days)")

    table = contrast(joined, "any_event", PREDICTORS)
    if table is not None:
        print(f"\n{label} — casualty days against ordinary days")
        print("  pctile_of_quiet: where the typical event day's value sits in "
              "the\n  distribution of days with no event. 50 means no difference.")
        with pd.option_context("display.width", 200):
            print(table.round(3).to_string(index=False))
    print("\n  rho_mo ranks each predictor within its month, rho_modow within"
          "\n  its month AND weekday. A driver that only holds up in the raw"
          "\n  column is the swimming season or the weekend, not the ocean.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zone", required=True,
                    help="the --zone string given to pull_storm_events.py")
    ap.add_argument("--camera", required=True,
                    help="camera name whose gridded_/marine_ files supply "
                         "conditions — a beach inside that zone")
    ap.add_argument("--probe", action="store_true",
                    help="report the join and the exposure bias, then stop")
    args = ap.parse_args()

    storm_frame = load_storm(args.zone)
    conditions = daily_conditions(args.camera)
    if conditions is None:
        sys.exit(f"No gridded_ or marine_ file for {args.camera!r}. Pull it with:\n"
                 f"  python pull_site_observations.py --camera {args.camera!r} "
                 "--start 2000-01-01 --skip-us-stations")

    joined = report_overlap(storm_frame, conditions)
    season = exposure_report(joined)
    if args.probe:
        print("\nRe-run without --probe for the correlations.")
        return

    run(joined, f"{args.zone.upper()} — ALL DAYS")

    in_season = joined[joined.index.month.isin(season)]
    print(f"\n\n{'#' * 74}\nSEASON ONLY: months {season}, "
          f"{len(in_season):,} of {len(joined):,} days\n"
          "A winter zero is not evidence the ocean was safe. Dropping the "
          "months\nnobody swims in leaves a comparison between days people "
          "were actually\nin the water.\n" + "#" * 74)
    run(in_season, f"{args.zone.upper()} — SEASON ONLY")


if __name__ == "__main__":
    main()
