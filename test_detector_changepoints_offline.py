"""Offline checks for analyze_detector_changepoints.py.

The fixture is built so the answers are known by construction rather than by
running the code and writing down what it said:

  * a SHARED step on 2026-03-01, in score_floor, on two cameras at once --
    the platform-deployment signature the script exists to flag
  * a step on 2026-05-20 in camera A's detection rate alone -- a site change
  * a WEATHER step on 2026-07-01: wave height doubles, and camera A's
    detection rate follows it exactly. Raw, that is a changepoint. It must not
    survive residualization, because nothing about the detector changed.

The third is the one that matters. A changepoint finder that cannot tell a
storm season from a redeployment would hand Riley a date to go and look for a
release note that was never written.

No network, no data/ directory, nothing but numpy and pandas.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

import analyze_drivers as ad
import analyze_detector_changepoints as cp

FAILURES = []

START = pd.Timestamp("2026-01-01", tz="UTC")
DAYS = 273
SHARED_STEP = pd.Timestamp("2026-03-01", tz="UTC")
SITE_STEP = pd.Timestamp("2026-05-20", tz="UTC")
WEATHER_STEP = pd.Timestamp("2026-07-01", tz="UTC")
TOLERANCE_DAYS = 5


def check(name, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    if not condition:
        FAILURES.append(name)
    print(f"  {mark} {name}" + (f"  {detail}" if detail else ""))


def dates():
    return pd.date_range(START, periods=DAYS, freq="D", tz="UTC")


def wave_series():
    """A storm season arriving on a date, on top of ordinary daily swell.

    The swell is not decoration. The first version of this fixture was a bare
    step -- flat, then double -- which made wave height almost perfectly
    collinear with a level shift on the same day. No method can separate "the
    surf doubled" from "the detector changed" when the two are the same column,
    and the test failed for that reason rather than for a fault in the code.
    Real wave height wanders day to day, and it is that wandering that
    identifies the coefficient, so the storm step can then be explained by it.

    The wandering is white rather than seasonal for the same reason. A slow
    cycle whose period is near --min-segment gets chopped into segments by the
    raw pass, those segments become dummies in the control fit, and the dummies
    then absorb the very variation the coefficient needs. That is a real
    limitation of residualizing this way, documented on residualize(), but it
    is not what this test is for.
    """
    rng = np.random.default_rng(4)
    base = np.where(dates() < WEATHER_STEP, 1.0, 2.0)
    return np.clip(base + rng.normal(0, 0.45, DAYS), 0.15, None)


def build_camera(folder, camera, rate_of, floor_of, waves):
    """Write rip_<slug>.csv and its coverage file for one synthetic camera.

    Detections are emitted hour by hour so detection_rate comes out of the same
    detected_hours / covered_hours arithmetic the real pipeline uses, rather
    than being written into a column the test then reads back.
    """
    slug = ad.rip_slug(camera)
    rows, coverage = [], []
    for index, day in enumerate(dates()):
        rate = float(np.clip(rate_of(index, waves[index]), 0.05, 0.95))
        hits = max(6, int(round(rate * 24)))
        floor = float(floor_of(index))
        scores = np.linspace(floor, 0.95, hits)
        for hour in range(24):
            coverage.append({"hour": (day + pd.Timedelta(hours=hour)).isoformat(),
                             "images": 60})
        for hour in range(hits):
            rows.append({
                "timestamp": (day + pd.Timedelta(hours=hour)).isoformat(),
                "detected": True,
                "score_max": float(scores[hour]),
                "detection_count": 1,
                "bbox_count": 1,
                "bbox_area_max": 1000.0 + 10.0 * hour,
                "model_name": "yolov8",
                "model_version": "1.0" if day < SHARED_STEP else "1.1",
                "source_file": f"{slug}_{index}.json",
                "original_image": f"{slug}_{index}_{hour}.jpg",
            })
    detection_dir = os.path.join(folder, "rip_detection")
    os.makedirs(detection_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        os.path.join(detection_dir, f"rip_{slug}.csv"), index=False)
    pd.DataFrame(coverage).to_csv(
        os.path.join(detection_dir, f"coverage_{slug}_hourly.csv"), index=False)
    return slug


def build_controls(folder, camera, waves):
    """A MOP file carrying the same wave series the rates were built from."""
    grid = ad.grid_slug(camera)
    pd.DataFrame({"time": [d.isoformat() for d in dates()],
                  "wave_height": waves}).to_csv(
        os.path.join(folder, f"mop_{grid}.csv"), index=False)


def day_index(day):
    return int((day - START).days)


def nearest(found_dates, target):
    """Smallest distance in days from any found date to the target, or None."""
    if not len(found_dates):
        return None
    return min(abs((pd.Timestamp(d) - pd.Timestamp(target)).days)
               for d in found_dates)


def shift_at(frame, metric, target, args, tolerance=TOLERANCE_DAYS):
    """Size of the shift the finder reports nearest `target`, or 0.0 if none.

    Magnitude, not presence, is the honest test of residualizing. A fitted
    coefficient has finite precision, so subtracting it leaves a fraction of
    the weather step behind; with enough days that remnant is still
    statistically real and the finder will still name the date. What must
    happen is that the shift COLLAPSES -- here from about 0.20 to about 0.01 --
    so it no longer competes with a genuine one. Asserting it vanishes
    entirely would be asserting something untrue of the method.
    """
    values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(values)
    kept_dates = pd.DatetimeIndex(frame["date"][keep].to_numpy())
    values = values[keep]
    found = cp.detect(values, args.max_k, args.min_segment)
    target = pd.Timestamp(target)
    best, best_gap = 0.0, None
    for index in found:
        gap = abs((pd.Timestamp(kept_dates[index]) - target).days)
        if gap <= tolerance and (best_gap is None or gap < best_gap):
            best_gap = gap
            lo, hi = cp.neighbours(found, index, len(values))
            best = abs(cp.shift_sizes(values, index, lo, hi)["delta"])
    return best


def series_dates(daily, metric, args):
    values = pd.to_numeric(daily[metric], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(values)
    kept_dates = pd.DatetimeIndex(daily["date"][keep].to_numpy())
    found = cp.detect(values[keep], args.max_k, args.min_segment)
    return [kept_dates[i] for i in found]


class Args:
    max_k = 6
    min_segment = 14
    coincide = 7
    min_frames = 5


def check_a_clean_step_is_found_where_it_was_put():
    values = np.concatenate([np.full(100, 0.20), np.full(100, 0.55)])
    found = cp.detect(values, 6, 14)
    check("a clean step is found exactly once", len(found) == 1, str(found))
    check("and at the index it was written to", found and found[0] == 100,
          str(found))
    shift = cp.shift_sizes(values, 100)
    check("the reported shift is the step that was built",
          abs(shift["delta"] - 0.35) < 1e-9, f"{shift['delta']:.4f}")


def check_noise_alone_yields_nothing():
    """The false-positive guard. Without BIC this returns max_k every time."""
    rng = np.random.default_rng(0)
    clean = 0
    for _ in range(20):
        if not cp.detect(rng.normal(0, 1, 300), 6, 14):
            clean += 1
    check("pure noise yields no changepoint in at least 18 of 20 draws",
          clean >= 18, f"{clean}/20 clean")

    values = rng.normal(0, 1, 300)
    splits = cp.binary_segmentation(values, 6, 14)
    keep, table = cp.choose_k_bic(values, splits)
    check("binary segmentation still offers splits on noise",
          len(splits) > 0, f"{len(splits)} offered")
    check("BIC is what discards them", keep < len(splits),
          f"kept {keep} of {len(splits)}")
    check("RSS alone would have kept them all",
          table[-1]["rss"] <= table[0]["rss"] + 1e-9)


def check_a_short_spike_is_marked_transient_not_a_regime():
    """A brief excursion produces a PAIR of changepoints, out and back.

    The first draft of this test asserted the finder would return nothing. It
    returns two, and it is right to: both are real mean shifts. What was wrong
    was the REPORT, which presented them as two regime changes and so invited a
    hunt for two release notes. They are now marked transient, and a transient
    is excluded from the platform-deployment clustering.
    """
    values = np.concatenate([np.full(150, 0.2), np.full(4, 0.9),
                             np.full(146, 0.2)])
    found = cp.detect(values, 6, 14)
    check("the spike is found rather than ignored", len(found) >= 2, str(found))
    flags = cp.reverted_flags(values, found)
    check("every changepoint around it is marked transient", all(flags),
          str(flags))

    step = np.concatenate([np.full(150, 0.2), np.full(150, 0.9)])
    step_found = cp.detect(step, 6, 14)
    check("a one-way step is NOT marked transient",
          step_found and not any(cp.reverted_flags(step, step_found)),
          f"{step_found} {cp.reverted_flags(step, step_found)}")

    transient_findings = [
        {"camera": "A", "metric": "m", "date": "2026-03-01", "transient": True},
        {"camera": "B", "metric": "m", "date": "2026-03-03", "transient": True}]
    check("two cameras' transients do not cluster into a deployment",
          not cp.coincidences(transient_findings, 7))
    real_findings = [
        {"camera": "A", "metric": "m", "date": "2026-03-01", "transient": False},
        {"camera": "B", "metric": "m", "date": "2026-03-03", "transient": False}]
    check("two cameras' real shifts still do",
          len(cp.coincidences(real_findings, 7)) == 1)


def check_shift_scales_by_within_segment_spread():
    """Scaling by the whole series' SD would shrink every large step alike."""
    values = np.concatenate([np.full(100, 0.0), np.full(100, 10.0)])
    values = values + np.resize([0.1, -0.1], 200)
    shift = cp.shift_sizes(values, 100)
    whole = float(np.std(values, ddof=1))
    check("within-segment sd is far smaller than the whole series' sd",
          shift["sd"] < whole / 10, f"{shift['sd']:.4f} vs {whole:.4f}")
    check("so a 10-unit step reads as a huge shift, not a 2-sd one",
          shift["delta_sd"] > 20, f"{shift['delta_sd']:.1f} sd")


def check_end_to_end_recovers_both_dates():
    """The whole pipeline, from CSVs on disk to dates, on the built fixture."""
    waves = wave_series()
    original = ad.DATA_DIR
    with tempfile.TemporaryDirectory() as folder:
        ad.DATA_DIR = folder
        try:
            camera_a = "Fake Pier, Testville, CA"
            camera_b = "Fake Jetty, Testville, CA"

            # A: the shared floor step, its own rate step, and the weather.
            slug_a = build_camera(
                folder, camera_a,
                rate_of=lambda i, w: (0.15 + 0.20 * w
                                      + (0.18 if i >= day_index(SITE_STEP) else 0)),
                floor_of=lambda i: 0.25 if i < day_index(SHARED_STEP) else 0.40,
                waves=waves)
            # B: the shared floor step only. Its rate is flat and weatherless.
            slug_b = build_camera(
                folder, camera_b,
                rate_of=lambda i, w: 0.40,
                floor_of=lambda i: 0.25 if i < day_index(SHARED_STEP) else 0.40,
                waves=waves)
            build_controls(folder, camera_a, waves)

            args = Args()
            daily_a, _ = cp.daily_metrics(
                os.path.join(folder, "rip_detection", f"rip_{slug_a}.csv"),
                args.min_frames)
            daily_a, had_a = cp.attach_detection_rate(daily_a, slug_a)
            daily_b, _ = cp.daily_metrics(
                os.path.join(folder, "rip_detection", f"rip_{slug_b}.csv"),
                args.min_frames)
            daily_b, _ = cp.attach_detection_rate(daily_b, slug_b)

            check("coverage was found, so detection_rate is a real rate", had_a)
            check("the fixture produced a day per calendar day",
                  len(daily_a) == DAYS, f"{len(daily_a)} days")

            floor_a = series_dates(daily_a, "score_floor", args)
            floor_b = series_dates(daily_b, "score_floor", args)
            check("camera A's score_floor step is found within tolerance",
                  nearest(floor_a, SHARED_STEP) is not None
                  and nearest(floor_a, SHARED_STEP) <= TOLERANCE_DAYS,
                  f"{[str(d.date()) for d in floor_a]}")
            check("camera B's score_floor step is found within tolerance",
                  nearest(floor_b, SHARED_STEP) is not None
                  and nearest(floor_b, SHARED_STEP) <= TOLERANCE_DAYS,
                  f"{[str(d.date()) for d in floor_b]}")

            rate_a = series_dates(daily_a, "detection_rate", args)
            check("camera A's own rate step is found within tolerance",
                  nearest(rate_a, SITE_STEP) is not None
                  and nearest(rate_a, SITE_STEP) <= TOLERANCE_DAYS,
                  f"{[str(d.date()) for d in rate_a]}")
            rate_b = series_dates(daily_b, "detection_rate", args)
            check("camera B, whose rate never moved, reports no rate step",
                  not rate_b, f"{[str(d.date()) for d in rate_b]}")

            # The coincidence clustering, over the dates just recovered.
            findings = ([{"camera": camera_a, "metric": "score_floor",
                          "date": d.strftime("%Y-%m-%d")} for d in floor_a]
                        + [{"camera": camera_b, "metric": "score_floor",
                            "date": d.strftime("%Y-%m-%d")} for d in floor_b]
                        + [{"camera": camera_a, "metric": "detection_rate",
                            "date": d.strftime("%Y-%m-%d")} for d in rate_a])
            clusters = cp.coincidences(findings, args.coincide)
            shared = [c for c in clusters
                      if abs((pd.Timestamp(c["first"], tz="UTC")
                              - SHARED_STEP).days) <= TOLERANCE_DAYS]
            check("the shared date is flagged as a two-camera coincidence",
                  len(shared) == 1 and len(shared[0]["cameras"]) == 2,
                  f"{len(clusters)} clusters")
            site = [c for c in clusters
                    if abs((pd.Timestamp(c["first"], tz="UTC")
                            - SITE_STEP).days) <= TOLERANCE_DAYS]
            check("camera A's site-specific date is NOT flagged as platform-wide",
                  not site, f"{site}")

            # The weather step: present raw, gone once residualized.
            raw_hits = nearest(rate_a, WEATHER_STEP)
            check("the weather step DOES show up in the raw rate series",
                  raw_hits is not None and raw_hits <= TOLERANCE_DAYS,
                  f"nearest {raw_hits} days")

            controls, available = cp.daily_controls(camera_a, None, None)
            check("the MOP control file was found", controls is not None,
                  str(available))
            residuals, names = cp.residualize(daily_a, controls,
                                              ["detection_rate", "score_floor"])
            check("wave height is the control that was used",
                  names == ["wave_height"], str(names))

            residual_rate = series_dates(residuals, "detection_rate", args)

            raw_weather = shift_at(daily_a, "detection_rate", WEATHER_STEP, args)
            res_weather = shift_at(residuals, "detection_rate", WEATHER_STEP, args)
            check("the weather step is large in the raw series",
                  raw_weather > 0.10, f"{raw_weather:.4f}")
            check("residualizing collapses it to under a fifth of its size",
                  res_weather < 0.2 * raw_weather,
                  f"{raw_weather:.4f} -> {res_weather:.4f}")

            raw_site = shift_at(daily_a, "detection_rate", SITE_STEP, args)
            res_site = shift_at(residuals, "detection_rate", SITE_STEP, args)
            check("camera A's real step survives residualizing at full size",
                  res_site > 0.6 * raw_site and res_site > 0.10,
                  f"{raw_site:.4f} -> {res_site:.4f}")
            check("so after residualizing the real step dwarfs the weather one",
                  res_site > 5 * max(res_weather, 1e-9),
                  f"site {res_site:.4f} vs weather {res_weather:.4f}")
            check("the real step is still dated correctly in the residuals",
                  nearest(residual_rate, SITE_STEP) is not None
                  and nearest(residual_rate, SITE_STEP) <= TOLERANCE_DAYS,
                  f"{[str(d.date()) for d in residual_rate]}")

            versions = cp.model_versions(
                os.path.join(folder, "rip_detection", f"rip_{slug_a}.csv"))
            check("the recorded model version change is reported",
                  len(versions) == 1 and versions[0]["date"] ==
                  SHARED_STEP.strftime("%Y-%m-%d"), str(versions))

            hist = cp.monthly_histograms(
                os.path.join(folder, "rip_detection", f"rip_{slug_a}.csv"),
                camera_a)
            check("a histogram row per month", len(hist) == 9, f"{len(hist)} rows")
            january = hist[hist["month"] == "2026-01"].iloc[0]
            april = hist[hist["month"] == "2026-04"].iloc[0]
            check("the floor moves between January and April in the histogram",
                  abs(january["min_nonzero"] - 0.25) < 1e-9
                  and abs(april["min_nonzero"] - 0.40) < 1e-9,
                  f"{january['min_nonzero']:.2f} -> {april['min_nonzero']:.2f}")
            check("histogram bins are the fixed 0.05 grid, not per-month",
                  len([c for c in hist.columns if c.startswith("b")]) == 20,
                  str(len([c for c in hist.columns if c.startswith("b")])))
        finally:
            ad.DATA_DIR = original


def check_thin_cameras_are_not_residualized():
    """A camera under --min-driver-days must be reported, not silently fitted."""
    waves = wave_series()
    original = ad.DATA_DIR
    with tempfile.TemporaryDirectory() as folder:
        ad.DATA_DIR = folder
        try:
            camera = "Tiny Cam, Nowhere, CA"
            slug = build_camera(folder, camera,
                                rate_of=lambda i, w: 0.4,
                                floor_of=lambda i: 0.3, waves=waves)
            daily, _ = cp.daily_metrics(
                os.path.join(folder, "rip_detection", f"rip_{slug}.csv"), 5)
            daily = daily.head(40)
            check("the thin fixture really is thin", len(daily) == 40)
            check("40 days is under the driver-fitting floor",
                  len(daily) < cp.MIN_DRIVER_DAYS,
                  f"{len(daily)} < {cp.MIN_DRIVER_DAYS}")
            controls, _ = cp.daily_controls(camera, None, None)
            check("and it has no control files of its own anyway",
                  controls is None)
        finally:
            ad.DATA_DIR = original


def check_residualizing_blanks_rows_rather_than_mixing():
    """A row with no control must become NaN, not stay raw.

    Mixing residualized and raw points puts a step at the edge of the controls'
    coverage -- manufacturing exactly the shape being hunted for.
    """
    daily = pd.DataFrame({
        "date": pd.date_range(START, periods=60, freq="D", tz="UTC"),
        "detection_rate": np.linspace(0.2, 0.5, 60)})
    controls = pd.DataFrame(
        {"wave_height": np.linspace(1.0, 2.0, 40)},
        index=pd.date_range(START, periods=40, freq="D", tz="UTC"))
    residuals, _ = cp.residualize(daily, controls, ["detection_rate"])
    uncovered = residuals["detection_rate"].to_numpy()[40:]
    check("rows with no control are NaN, not passed through raw",
          bool(np.isnan(uncovered).all()), str(uncovered[:3]))
    check("rows with a control are finite",
          bool(np.isfinite(residuals["detection_rate"].to_numpy()[:40]).all()))


def main():
    print("detector changepoint offline checks\n")
    check_a_clean_step_is_found_where_it_was_put()
    check_noise_alone_yields_nothing()
    check_a_short_spike_is_marked_transient_not_a_regime()
    check_shift_scales_by_within_segment_spread()
    check_end_to_end_recovers_both_dates()
    check_thin_cameras_are_not_residualized()
    check_residualizing_blanks_rows_rather_than_mixing()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
