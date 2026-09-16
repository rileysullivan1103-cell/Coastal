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

import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd

import analyze_drivers as ad
import analyze_detector_changepoints as cp
import diagnose_class_timeline as ct

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
    detection_class = "rip_current"


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
            check("both recorded model versions are reported as spans",
                  len(versions) == 2, str([v["tag"] for v in versions]))
            check("the older one ends where the newer one starts",
                  versions[0]["ends_early"] and versions[1]["starts_late"],
                  str([(v["tag"], v["starts_late"], v["ends_early"])
                       for v in versions]))
            check("and the handover is the date it was built at",
                  abs((versions[1]["first"] - SHARED_STEP).days) <= 1,
                  f"{versions[1]['first']:%Y-%m-%d}")

            # Two models interleaving row by row is not N deployments.
            path_a = os.path.join(folder, "rip_detection", f"rip_{slug_a}.csv")
            table = pd.read_csv(path_a)
            table["model_version"] = ["1.0", "1.1"] * (len(table) // 2) \
                + ["1.0"] * (len(table) % 2)
            table.to_csv(path_a, index=False)
            interleaved = cp.model_versions(path_a)
            check("two interleaved models report two spans, not one per row",
                  len(interleaved) == 2, str(len(interleaved)))
            check("and neither is flagged as starting or stopping mid-record",
                  not any(v["starts_late"] or v["ends_early"]
                          for v in interleaved),
                  str([(v["starts_late"], v["ends_early"])
                       for v in interleaved]))

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


def check_object_only_cameras_are_not_analysed_as_rip_cameras():
    """Three real cameras are 0% rip_current. Their "detection rate" is people.

    Corolla Hampton Inn (10,766 frames), Corolla Sailfish (10,239) and Carova
    (34) have never produced a rip detection. Run unfiltered, a changepoint in
    their detection rate is a changepoint in how busy the beach was -- a real
    signal about something, but not about the detector's rip behaviour, and
    certainly not a version bump to go and confirm.
    """
    import io
    import contextlib

    waves = wave_series()
    original = ad.DATA_DIR
    with tempfile.TemporaryDirectory() as folder:
        ad.DATA_DIR = folder
        try:
            camera = "Objects Only, Nowhere, NC"
            slug = build_camera(folder, camera,
                                rate_of=lambda i, w: 0.4,
                                floor_of=lambda i: 0.3, waves=waves)
            path = os.path.join(folder, "rip_detection", f"rip_{slug}.csv")
            table = pd.read_csv(path)
            table["score_classes"] = "person"
            table.to_csv(path, index=False)

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                daily, note = cp.daily_metrics(path, 5, "rip_current")
            check("a camera with no rip detections is not analysed",
                  daily is None, "analysed anyway" if daily is not None else "")
            check("and the reason names the class", "no rip_current" in note,
                  note)
            check("and points at the way to analyse it anyway",
                  "--detection-class any" in note, note)

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                daily, _ = cp.daily_metrics(path, 5, "any")
            check("with 'any' the same record analyses fine",
                  daily is not None and len(daily) == DAYS,
                  f"{0 if daily is None else len(daily)} days")

            # And a mixed record keeps only the rip rows.
            table["score_classes"] = (["rip_current"] * (len(table) // 2)
                                      + ["boat"] * (len(table) - len(table) // 2))
            table.to_csv(path, index=False)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                mixed, _ = cp.daily_metrics(path, 5, "rip_current")
            with contextlib.redirect_stdout(io.StringIO()):
                pooled, _ = cp.daily_metrics(path, 5, "any")
            check("a mixed record analyses fewer detections when filtered",
                  mixed is not None and pooled is not None
                  and mixed["n_detections"].sum() < pooled["n_detections"].sum(),
                  f"{mixed['n_detections'].sum()} vs "
                  f"{pooled['n_detections'].sum()}")

            mix = cp.class_mix(pd.read_csv(path))
            check("the class mix is reported for the header",
                  mix.get("rip_current", 0) > 0 and mix.get("boat", 0) > 0,
                  str(dict(mix)))
        finally:
            ad.DATA_DIR = original


CLASS_SWITCH = pd.Timestamp("2026-05-01", tz="UTC")


def check_a_class_switching_on_fools_only_the_unfiltered_run():
    """The whole reason for running both passes.

    A camera that starts emitting `person` halfway through steps its UNFILTERED
    detection rate on that date. A changepoint search names it, and without the
    comparison it reads as a detector deployment worth going to find a release
    note for. The rip_current series underneath never moves.

    Built so the answer is known: rip detections are a flat 8 per day for the
    whole record, and person detections switch from 0 to 10 per day on
    2026-05-01 and nothing else changes.
    """
    original = ad.DATA_DIR
    with tempfile.TemporaryDirectory() as folder:
        ad.DATA_DIR = folder
        try:
            camera = "Switcher, Testville, CA"
            slug = ad.rip_slug(camera)
            detection_dir = os.path.join(folder, "rip_detection")
            os.makedirs(detection_dir, exist_ok=True)

            rows, coverage = [], []
            for index, day in enumerate(dates()):
                for hour in range(24):
                    coverage.append({
                        "hour": (day + pd.Timedelta(hours=hour)).isoformat(),
                        "images": 60})
                for hour in range(8):
                    rows.append({
                        "timestamp": (day + pd.Timedelta(hours=hour)).isoformat(),
                        "detected": True, "score_classes": "rip_current",
                        "score_max": 0.7, "detection_count": 1, "bbox_count": 1,
                        "bbox_area_max": 1000.0, "source_file": "s.json",
                        "original_image": f"r{index}_{hour}.jpg"})
                if day >= CLASS_SWITCH:
                    for hour in range(8, 18):
                        rows.append({
                            "timestamp": (day + pd.Timedelta(hours=hour)).isoformat(),
                            "detected": True, "score_classes": "person",
                            "score_max": 0.9, "detection_count": 1,
                            "bbox_count": 1, "bbox_area_max": 500.0,
                            "source_file": "s.json",
                            "original_image": f"p{index}_{hour}.jpg"})
            path = os.path.join(detection_dir, f"rip_{slug}.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            pd.DataFrame(coverage).to_csv(
                os.path.join(detection_dir, f"coverage_{slug}_hourly.csv"),
                index=False)

            args = Args()
            quiet = io.StringIO()

            with contextlib.redirect_stdout(quiet):
                pooled, _ = cp.daily_metrics(path, args.min_frames, "any")
                pooled, _ = cp.attach_detection_rate(pooled, slug)
                filtered, _ = cp.daily_metrics(path, args.min_frames,
                                               "rip_current")
                filtered, _ = cp.attach_detection_rate(filtered, slug)

            pooled_dates = series_dates(pooled, "detection_rate", args)
            check("the UNFILTERED run flags the class-switch date",
                  nearest(pooled_dates, CLASS_SWITCH) is not None
                  and nearest(pooled_dates, CLASS_SWITCH) <= TOLERANCE_DAYS,
                  f"{[str(d.date()) for d in pooled_dates]}")

            filtered_dates = series_dates(filtered, "detection_rate", args)
            check("the FILTERED run does not",
                  not filtered_dates
                  or nearest(filtered_dates, CLASS_SWITCH) > TOLERANCE_DAYS,
                  f"{[str(d.date()) for d in filtered_dates]}")

            # And the class timeline names the same date independently.
            timeline, why = ct.load(path)
            check("the class timeline loads the record", timeline is not None,
                  why)
            events = ct.transitions(ct.monthly_table(timeline),
                                    ct.class_events(timeline),
                                    timeline["timestamp"].min(),
                                    timeline["timestamp"].max())
            starts = [e for e in events if e["kind"] == "starts"]
            check("it reports person starting mid-record",
                  len(starts) == 1 and starts[0]["group"] == "person",
                  str([(e["group"], e["kind"], str(e["date"].date()))
                       for e in events]))
            check("on the date it was built to start",
                  abs((starts[0]["date"] - CLASS_SWITCH).days) <= 1,
                  f"{starts[0]['date']:%Y-%m-%d}")

            hit = cp.near_class_event(pooled_dates[0], events, args.coincide)
            check("and the unfiltered changepoint is marked against it",
                  hit is not None and hit["group"] == "person", str(hit))

            monthly = ct.monthly_table(timeline)
            before = monthly[monthly["month"] < "2026-05"]
            after = monthly[monthly["month"] >= "2026-05"]
            check("monthly counts show person at zero before the switch",
                  before["person"].sum() == 0, str(before["person"].sum()))
            check("and present after", after["person"].sum() > 0)
            check("while rip_current is unchanged across it",
                  before[ct.RIP].sum() > 0 and after[ct.RIP].sum() > 0)

            # The combined object column counts a frame ONCE however many
            # object classes it names, so it can never exceed the frame count
            # and can never be less than any single object group.
            check("the combined object column starts at zero too",
                  before["object"].sum() == 0)
            check("and never exceeds that month's frame count",
                  bool((monthly["object"] <= monthly["frames"]).all()))
            check("nor falls below any single object group it contains",
                  bool((monthly["object"]
                        >= monthly[list(ct.OBJECT_GROUPS)].max(axis=1)).all()))

            with contextlib.redirect_stdout(quiet):
                rows_out = cp.compare_classes(camera, path, slug, args, events)
            marked = [r for r in rows_out if r["class_event"]]
            check("compare_classes returns the marked rows",
                  any(r["pass"] == "unfiltered" for r in marked), str(marked[:2]))
            check("and marks none of the filtered rows",
                  not any(r["pass"] == "filtered" and r["class_event"]
                          for r in rows_out))
        finally:
            ad.DATA_DIR = original


def check_a_degenerate_control_does_not_poison_the_residuals():
    """The real Walton run raised "divide by zero encountered in matmul".

    A control column that never moves between consecutive kept days leaves
    lstsq with a rank-deficient system; it returns a solution, that solution
    held infinities, and every residual below was computed from them. The
    residuals PRINTED anyway, which is what made it dangerous rather than
    merely noisy.
    """
    import warnings

    n = 200
    rng = np.random.default_rng(5)
    target = rng.normal(0, 1, n)
    # One control moves, one is stone dead.
    design = np.column_stack([rng.normal(0, 1, n), np.full(n, 3.0)])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        slopes = cp.control_slopes(target, design)
    check("a constant control gets a zero slope rather than an infinity",
          np.isfinite(slopes).all() and slopes[1] == 0.0, str(slopes))
    check("the moving control still gets fitted", slopes[0] != 0.0,
          f"{slopes[0]:.4f}")

    dead = np.column_stack([np.full(n, 1.0), np.full(n, 3.0)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        slopes = cp.control_slopes(target, dead)
    check("all-constant controls give all-zero slopes",
          np.isfinite(slopes).all() and not slopes.any(), str(slopes))

    # And through residualize, with a non-finite value in the design.
    daily = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC"),
        "detection_rate": target})
    controls = pd.DataFrame({"wave_height": np.full(n, 2.0)},
                            index=daily["date"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        residuals, _ = cp.residualize(daily, controls, ["detection_rate"])
    values = residuals["detection_rate"].to_numpy()
    check("residualizing on a dead control returns finite numbers",
          np.isfinite(values).all(), str(values[:3]))
    check("and leaves the series unchanged, since nothing was explained",
          np.allclose(values, target), f"{values[0]:.4f} vs {target[0]:.4f}")

    # The real failure: slopes were finite, the PRODUCT overflowed. Controls
    # on wildly different scales (bbox areas near 1e5, cloud cover near 1e2)
    # made an ill-conditioned fit produce slopes big enough to blow up
    # design @ slopes. Guarding the slopes alone did not stop it.
    wide = np.column_stack([rng.normal(3e5, 1e5, n),
                            rng.normal(50, 20, n),
                            rng.normal(3e5, 1e5, n) * 1.0000001])
    daily_wide = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC"),
        "bbox_area_median": target * 1e5})
    controls_wide = pd.DataFrame(
        {"a": wide[:, 0], "b": wide[:, 1], "c": wide[:, 2]},
        index=daily_wide["date"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        residuals, _ = cp.residualize(daily_wide, controls_wide,
                                      ["bbox_area_median"])
    got = residuals["bbox_area_median"].to_numpy()
    check("badly scaled controls raise no RuntimeWarning",
          np.isfinite(got).all(), str(got[:2]))
    check("and the residual keeps the metric's own mean",
          abs(np.nanmean(got) - np.nanmean(target * 1e5)) < 1e-6,
          f"{np.nanmean(got):.4f}")


def check_both_rip_class_names_are_analysed():
    """Three real cameras emit "rip", not "rip_current", from another model."""
    import io
    import contextlib

    waves = wave_series()
    original = ad.DATA_DIR
    with tempfile.TemporaryDirectory() as folder:
        ad.DATA_DIR = folder
        try:
            camera = "Other Rip Model, Nowhere, NC"
            slug = build_camera(folder, camera, rate_of=lambda i, w: 0.4,
                                floor_of=lambda i: 0.3, waves=waves)
            path = os.path.join(folder, "rip_detection", f"rip_{slug}.csv")
            table = pd.read_csv(path)
            table["score_classes"] = "rip"
            table.to_csv(path, index=False)

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                daily, note = cp.daily_metrics(path, 5, "rip_current,rip")
            check("a record using the 'rip' class name is analysed",
                  daily is not None and len(daily) == DAYS,
                  note or f"{0 if daily is None else len(daily)} days")

            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                only, _ = cp.daily_metrics(path, 5, "rip_current")
            check("and filtering on rip_current alone still excludes it",
                  only is None)
        finally:
            ad.DATA_DIR = original


def main():
    print("detector changepoint offline checks\n")
    check_a_clean_step_is_found_where_it_was_put()
    check_noise_alone_yields_nothing()
    check_a_short_spike_is_marked_transient_not_a_regime()
    check_shift_scales_by_within_segment_spread()
    check_end_to_end_recovers_both_dates()
    check_thin_cameras_are_not_residualized()
    check_object_only_cameras_are_not_analysed_as_rip_cameras()
    check_a_class_switching_on_fools_only_the_unfiltered_run()
    check_a_degenerate_control_does_not_poison_the_residuals()
    check_both_rip_class_names_are_analysed()
    check_residualizing_blanks_rows_rather_than_mixing()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
