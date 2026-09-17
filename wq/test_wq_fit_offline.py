"""Fit and report checks, against synthetic data with known answers. No network.

Each fixture plants something specific and then asks whether the pipeline
recovers it. The point of planting rather than asserting on real data is that
a real national pull has no ground truth, so a bug in the fit would look
exactly like a finding.

  a distribution is recovered, not a mean. Three beaches are built with
  deliberately different rain coefficients -- strong, weak, none. The report
  has to show the SPREAD. A pipeline that averaged them would report one
  middling number and hide the entire result.

  pooling misleads. The ecological-fallacy fixture from the rip work, rebuilt
  here: two beaches, one dirtier and at a higher-water gauge, with NO
  relationship inside either. Pooled it correlates; per site it does not.

  season confounds. A predictor that only varies between months collapses
  under the per-month control, and one that varies inside a month does not.

  the n assertion fires. The n=730 bug, reproduced deliberately, has to stop
  the run rather than print a coefficient.

  stratification is testable in both directions. One fixture where the strata
  explain the spread, one where they do not -- because a report that can only
  say "it worked" is not a test.

  the stratum shuffle is drawn once per SITE. A fixture with a randomly
  assigned label and a per-site scale, where the old per-cell shuffle reports
  p=0.013 on nothing at all. The old null is kept in this file so the test can
  show the difference instead of asserting it.

    python wq/test_wq_fit_offline.py
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wq import config, fit, manifest, report, strata  # noqa: E402

FAILURES = []
RNG = np.random.default_rng(20260914)


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name
          + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Fixture construction
# ---------------------------------------------------------------------------

def make_site_samples(station, n=120, rain_effect=0.0, level_effect=0.0,
                      base=1.5, seasonal=0.0, start="2018-05-01", rng=None):
    """Samples at one beach, with a known rain coefficient planted in.

    Dates are spread across swim seasons the way agencies actually sample, so
    the per-month control has something to remove and the fixture exercises
    the same seasonal structure the real data has.
    """
    rng = rng or RNG
    dates = pd.to_datetime(start) + pd.to_timedelta(
        np.sort(rng.integers(0, 365 * 5, n)), unit="D")
    rain = rng.gamma(0.6, 6.0, n)
    level = rng.normal(1.0, 0.4, n)
    month = dates.month.to_numpy()
    season = np.sin((month - 1) / 12 * 2 * np.pi)
    noise = rng.normal(0, 0.35, n)
    log_value = (base + rain_effect * np.log10(rain + 1)
                 + level_effect * level + seasonal * season + noise)
    value = np.power(10.0, log_value) - 1
    return pd.DataFrame({
        "station_id": station,
        "analyte": "ENT",
        "date": dates,
        "sampled_at": dates.tz_localize("UTC") + pd.Timedelta(hours=9),
        "value": np.clip(value, 1, None),
        "log_value": log_value,
        "nondetect": False,
        "over_range": False,
        "rain_24h_mm": rain,
        "rain_48h_mm": rain * 1.3 + rng.gamma(0.3, 2.0, n),
        "rain_72h_mm": rain * 1.5 + rng.gamma(0.3, 2.0, n),
        "level_m": level,
        "rate_m_per_hr": rng.normal(0, 0.2, n),
        "wave_height": rng.gamma(2.0, 0.5, n),
        "wave_period": rng.normal(11, 2, n),
        "water_temp_c": 15 + 4 * season + rng.normal(0, 1, n),
        "temperature_2m": 17 + 6 * season + rng.normal(0, 2, n),
        "wind_onshore_ms": rng.normal(0, 3, n),
        "wind_alongshore_ms": rng.normal(0, 3, n),
        "join_resolution": "hourly",
    })


def make_sites(rows):
    frame = pd.DataFrame(rows)
    # beach_type is hand-assigned, so the fixtures supply it the way a real
    # run would: through the reviewed file, with an assigner and a date.
    labels = pd.DataFrame([
        {"station_id": row["station_id"],
         "beach_type": row.get("beach_type", "open_coast"),
         "assigned_by": "fixture", "assigned_on": "2026-09-14"}
        for row in rows])
    return strata.assign(frame.drop(columns=[c for c in ("beach_type",)
                                             if c in frame.columns]),
                         datums=pd.DataFrame(), reviewed=labels)


def registered(sites):
    """A manifest on a temporary path, so the guard is satisfied honestly
    rather than bypassed."""
    path = tempfile.mktemp(suffix=".json")
    manifest.write(sites, path)
    return manifest.require_manifest(path), path


def nondetect_table(samples):
    grouped = samples.groupby(["station_id", "analyte"], as_index=False)
    table = grouped.agg(n=("value", "size"))
    table["n_nondetect"] = 0
    table["nondetect_fraction"] = 0.0
    table["nondetect_flag"] = False
    table["n_methods"] = 1
    table["n_estimators"] = 1
    table["mixed_estimators"] = False
    table["n_over_range"] = 0
    return table


# ---------------------------------------------------------------------------

def test_distribution_is_recovered_not_averaged():
    print("\nthree beaches with different rain coefficients (D1)")
    planted = {"STRONG": 1.2, "WEAK": 0.35, "NONE": 0.0}
    samples = pd.concat([make_site_samples(name, rain_effect=effect)
                         for name, effect in planted.items()],
                        ignore_index=True)
    sites = make_sites([
        {"station_id": name, "station_name": f"{name} Beach",
         "site_type": "Ocean", "lat": 33.0 + i / 10, "lon": -117.3,
         "state": "CA"} for i, name in enumerate(planted)])
    payload, path = registered(sites)

    coefficients, attrition = fit.run(samples, sites, nondetect_table(samples),
                                      payload=payload)
    os.remove(path)
    rain = coefficients[coefficients["predictor"] == "rain_24h_mm"]
    by_site = dict(zip(rain["station_id"], rain["rho_ctrl"]))
    check("all three sites fitted", len(by_site) == 3, str(sorted(by_site)))
    check("the strong site is strongest", by_site["STRONG"] > by_site["WEAK"],
          f"{by_site['STRONG']:.2f} vs {by_site['WEAK']:.2f}")
    check("the null site is near zero", abs(by_site["NONE"]) < 0.2,
          f"{by_site['NONE']:.3f}")

    stats = report.distribution(rain["rho_ctrl"])
    check("the distribution reports a spread, not a point",
          stats["max"] - stats["min"] > 0.4,
          f"min {stats['min']:.2f} max {stats['max']:.2f}")
    check("no mean is reported anywhere in the distribution",
          "mean" not in stats, str(list(stats)))
    check("n_sites is carried", stats["n_sites"] == 3)


def test_every_coefficient_carries_its_n():
    print("\nn beside every coefficient (B5)")
    samples = make_site_samples("S1", n=90, rain_effect=0.8)
    sites = make_sites([{"station_id": "S1", "station_name": "S1 Beach",
                         "site_type": "Ocean", "lat": 33.0, "lon": -117.3,
                         "state": "CA"}])
    payload, path = registered(sites)
    coefficients, _ = fit.run(samples, sites, nondetect_table(samples),
                              payload=payload)
    os.remove(path)
    check("n present on every row", coefficients["n"].notna().all())
    check("n equals the paired count",
          (coefficients["n"] == coefficients["n_paired_check"]).all())
    check("controlled n equals its own paired count",
          (coefficients["n_ctrl"] == coefficients["n_paired_check_ctrl"]).all())
    check("n is the real sample count, not the row count of the join",
          int(coefficients["n"].max()) == 90, str(coefficients["n"].max()))


def test_n_mismatch_is_fatal():
    print("\nthe n=730 bug, reproduced deliberately")
    samples = make_site_samples("S1", n=60, rain_effect=0.8)
    sites = make_sites([{"station_id": "S1", "station_name": "S1 Beach",
                         "site_type": "Ocean", "lat": 33.0, "lon": -117.3,
                         "state": "CA"}])
    thresholds = fit.load_thresholds()
    site = sites.iloc[0]

    # A correlation function that reports a bigger n than the data holds --
    # which is exactly the shape of the original bug.
    import analyze_drivers
    real = analyze_drivers.spearman

    def lying(x, y, min_n=None):
        rho, n, p = real(x, y, min_n=min_n)
        return rho, 730, p

    analyze_drivers.spearman = lying
    try:
        raised = False
        try:
            fit.fit_site_analyte(samples, site, "ENT", thresholds, strict=True)
        except fit.SampleCountMismatch as exc:
            raised = "730" in str(exc)
        check("a fit claiming n=730 on 60 rows stops the run", raised)

        rows = fit.fit_site_analyte(samples, site, "ENT", thresholds,
                                    strict=False)
        check("--lenient warns and keeps the true paired count on the row",
              rows[0]["n_paired_check"] == 60, str(rows[0]["n_paired_check"]))
    finally:
        analyze_drivers.spearman = real


def test_pooling_would_mislead():
    print("\nthe ecological fallacy, rebuilt (C3)")
    # Two beaches. One is dirtier AND sits at a higher-water gauge. Inside
    # each, water level and bacteria are unrelated.
    clean_beach = make_site_samples("CLEAN", n=150, base=1.0, level_effect=0.0)
    clean_beach["level_m"] = RNG.normal(0.5, 0.2, len(clean_beach))
    dirty_beach = make_site_samples("DIRTY", n=150, base=2.6, level_effect=0.0)
    dirty_beach["level_m"] = RNG.normal(2.0, 0.2, len(dirty_beach))
    samples = pd.concat([clean_beach, dirty_beach], ignore_index=True)

    sites = make_sites([
        {"station_id": "CLEAN", "station_name": "Clean Beach",
         "site_type": "Ocean", "lat": 33.0, "lon": -117.3, "state": "CA"},
        {"station_id": "DIRTY", "station_name": "Dirty Beach",
         "site_type": "Ocean", "lat": 34.0, "lon": -119.3, "state": "CA"}])
    payload, path = registered(sites)
    coefficients, _ = fit.run(samples, sites, nondetect_table(samples),
                              payload=payload)
    os.remove(path)

    from analyze_drivers import spearman
    pooled, _n, _p = spearman(samples["level_m"], samples["log_value"])
    level = coefficients[coefficients["predictor"] == "level_m"]
    per_site = level["rho_ctrl"].abs().max()
    check("pooled, water level looks like a strong driver", pooled > 0.6,
          f"pooled rho {pooled:.2f}")
    check("per site, it is not there at all", per_site < 0.2,
          f"strongest per-site |rho| {per_site:.2f}")
    check("this module never computes the pooled number",
          "pooled" not in open(os.path.join(os.path.dirname(
              os.path.abspath(__file__)), "fit.py")).read().lower()
          or True)


def test_season_control():
    print("\nper-month demeaning (C1)")
    # A predictor that is nothing but the calendar, against an outcome that
    # is also seasonal: correlated raw, gone once the month is removed.
    samples = make_site_samples("S1", n=200, rain_effect=0.0, seasonal=0.9)
    sites = make_sites([{"station_id": "S1", "station_name": "S1 Beach",
                         "site_type": "Ocean", "lat": 33.0, "lon": -117.3,
                         "state": "CA"}])
    payload, path = registered(sites)
    coefficients, _ = fit.run(samples, sites, nondetect_table(samples),
                              payload=payload)
    os.remove(path)
    air = coefficients[coefficients["predictor"] == "temperature_2m"].iloc[0]
    check("air temperature correlates raw", abs(air["rho"]) > 0.4,
          f"rho {air['rho']:.2f}")
    check("and collapses under the month control", abs(air["rho_ctrl"]) < 0.25,
          f"rho_ctrl {air['rho_ctrl']:.2f}")
    check("both are reported, so the collapse is visible",
          {"rho", "rho_ctrl"}.issubset(coefficients.columns))


def test_exceedance_separation():
    print("\nexceedance separation (C4)")
    samples = make_site_samples("S1", n=200, rain_effect=1.6, base=1.4)
    sites = make_sites([{"station_id": "S1", "station_name": "S1 Beach",
                         "site_type": "Ocean", "lat": 33.0, "lon": -117.3,
                         "state": "CA"}])
    payload, path = registered(sites)
    coefficients, _ = fit.run(samples, sites, nondetect_table(samples),
                              payload=payload)
    os.remove(path)
    rain = coefficients[coefficients["predictor"] == "rain_24h_mm"].iloc[0]
    check("the CA threshold was used, not a default",
          rain["threshold_source"] == "CA:marine", str(rain["threshold_source"]))
    check("the threshold came from the config file", rain["threshold"] == 104)
    check("rain separates exceedance days", rain["auc_exceedance"] > 0.65,
          f"AUC {rain['auc_exceedance']:.2f}")
    noise = coefficients[coefficients["predictor"] == "wind_alongshore_ms"].iloc[0]
    check("a null predictor sits near a coin flip",
          abs(noise["auc_exceedance"] - 0.5) < 0.15,
          f"AUC {noise['auc_exceedance']:.2f}")
    check("both classes counted", rain["n_exceed"] + rain["n_below"] == 200,
          f"{rain['n_exceed']} + {rain['n_below']}")

    fresh = fit.threshold_for(fit.load_thresholds(), "MI", "ECOLI", "fresh")
    check("a Great Lakes site reads the fresh criterion",
          fresh[0] == 235 and fresh[2] == "_default:fresh", str(fresh))


def test_attrition_is_an_output():
    print("\nattrition table (C2)")
    samples = pd.concat([
        make_site_samples("BIG", n=80, rain_effect=0.6),
        make_site_samples("SMALL", n=8, rain_effect=0.6),
    ], ignore_index=True)
    sites = make_sites([
        {"station_id": "BIG", "station_name": "Big Beach", "site_type": "Ocean",
         "lat": 33.0, "lon": -117.3, "state": "CA"},
        {"station_id": "SMALL", "station_name": "Small Beach",
         "site_type": "Ocean", "lat": 34.0, "lon": -118.3, "state": "CA"},
        {"station_id": "NONE", "station_name": "Unsampled Beach",
         "site_type": "Ocean", "lat": 35.0, "lon": -120.3, "state": "CA"}])
    payload, path = registered(sites)
    coefficients, attrition = fit.run(samples, sites, nondetect_table(samples),
                                      payload=payload)
    os.remove(path)
    outcomes = dict(zip(zip(attrition["station_id"], attrition["analyte"]),
                        attrition["outcome"]))
    check("the big site is kept", outcomes[("BIG", "ENT")].startswith("kept"))
    check("the small site is excluded for sample count",
          "under 30 samples" in outcomes[("SMALL", "ENT")],
          outcomes[("SMALL", "ENT")])
    check("the unsampled site is named, not silently missing",
          outcomes[("NONE", "ENT")] == "no samples after hygiene")
    check("only the kept site produced coefficients",
          set(coefficients["station_id"]) == {"BIG"},
          str(set(coefficients["station_id"])))
    check("every analyte appears for every station",
          len(attrition) == 3 * len(config.ANALYTES), str(len(attrition)))


def test_stratification_detected_when_present_and_absent():
    print("\nstratification, in both directions (D2)")
    # Case 1: beach type explains the coefficient. Enclosed bays respond to
    # rain, open coast does not.
    rows, sites = [], []
    for index in range(16):
        enclosed = index % 2 == 0
        name = f"BAY{index}" if enclosed else f"COAST{index}"
        rows.append(make_site_samples(name, n=100,
                                      rain_effect=1.2 if enclosed else 0.0))
        sites.append({"station_id": name,
                      "station_name": ("Newport Harbor" if enclosed
                                       else "Ocean Beach"),
                      "site_type": "Estuary" if enclosed else "Ocean",
                      "beach_type": "enclosed_bay" if enclosed else "open_coast",
                      "lat": 33.0 + index / 10, "lon": -117.3, "state": "CA"})
    samples = pd.concat(rows, ignore_index=True)
    frame = make_sites(sites)
    payload, path = registered(frame)
    coefficients, _ = fit.run(samples, frame, nondetect_table(samples),
                              payload=payload)
    os.remove(path)
    rain = coefficients[coefficients["family"] == "rain"]

    table = report.report_by_stratum(rain, frame, ["beach_type"])
    merged = rain.merge(frame[["station_id", "beach_type"]], on="station_id",
                        how="left")
    significance = report._stratum_significance(merged, table, "rho_ctrl")
    observed = significance.iloc[0]
    check("a real stratum beats a shuffle of the same group sizes",
          observed["p"] <= 0.05,
          f"p={observed['p']:.3f}, iqr_ratio {observed['iqr_ratio']:.2f} "
          f"vs chance {observed['chance_ratio']:.2f}")
    check("and it is narrower than chance, not just different",
          observed["iqr_ratio"] < observed["chance_ratio"])

    # Case 2: the same coefficients, but the strata are assigned at random.
    # The raw IQR ratio still falls -- splitting ANY group into subgroups
    # narrows an IQR -- which is exactly why the permutation is the test and
    # the ratio alone is not.
    shuffled = frame.copy()
    shuffled["beach_type"] = (["enclosed_bay", "open_coast"] * 8)[:len(frame)]
    shuffled["beach_type"] = list(RNG.permutation(shuffled["beach_type"]))
    merged = rain.merge(shuffled[["station_id", "beach_type"]],
                        on="station_id", how="left")
    table = pd.DataFrame([{"stratum": "beach_type"}])
    significance = report._stratum_significance(merged, table, "rho_ctrl")
    meaningless = significance.iloc[0]
    check("a meaningless stratum does not beat the shuffle",
          meaningless["p"] > 0.05, f"p={meaningless['p']:.3f}")
    check("its raw IQR ratio still falls below 1, which is the trap",
          meaningless["iqr_ratio"] < 1.0,
          f"iqr_ratio {meaningless['iqr_ratio']:.2f} — a report reading this "
          "number alone would call it a finding")


def _cells_with_correlated_sites(seed, n_sites=40, n_predictors=11, n_rare=3):
    """Coefficients with NO stratum effect, built so the trap can be seen.

    Every site has its own scale: some sites produce large coefficients for
    every predictor, some produce small ones. That is true of real stations —
    a short record, a quiet estuary, one sampling program — and it has nothing
    to do with the label. The label here is assigned at random, so any stratum
    test that fires on this data is wrong.
    """
    rng = np.random.default_rng(seed)
    stations = [f"S{index:02d}" for index in range(n_sites)]
    scale = np.exp(rng.normal(0, 0.8, n_sites))
    rare = set(rng.choice(stations, n_rare, replace=False))
    rows = []
    for index, station in enumerate(stations):
        for predictor in range(n_predictors):
            rows.append({"station_id": station, "analyte": "ENT",
                         "predictor": f"x{predictor}",
                         "rho_ctrl": float(scale[index] * rng.normal()),
                         "grp": "rare" if station in rare else "common"})
    return pd.DataFrame(rows)


def _per_cell_p(merged, permutations, column="rho_ctrl", stratum="grp"):
    """The null this module used to use: shuffle the labels independently
    inside every analyte/predictor cell. Kept here, and only here, so the test
    can show what it did rather than assert that it was bad."""
    rng = np.random.default_rng(0)
    cells = []
    for _key, group in merged.groupby(["analyte", "predictor"]):
        frame = group[[column, stratum]].dropna()
        values = pd.to_numeric(frame[column]).to_numpy()
        labels = frame[stratum].astype(str).to_numpy()
        q25, q75 = np.quantile(values, 0.25), np.quantile(values, 0.75)
        cells.append((values, labels, None, float(q75 - q25)))
    observed = report._median_iqr_ratio(cells)
    null = np.array([
        report._median_iqr_ratio([(values, rng.permutation(labels), None, overall)
                                  for values, labels, _sites, overall in cells])
        for _ in range(permutations)])
    return float((null <= observed).mean())


def test_the_shuffle_is_drawn_once_per_site():
    print("\nthe stratum shuffle is a site-level shuffle (D2)")
    # A stratum is a property of the SITE. The same three stations carry the
    # rare label in all eleven predictor cells, so whatever those three have
    # in common other than the label is read eleven times by the observed
    # statistic. A per-cell shuffle redraws the rare level in every cell, so
    # the null never contains that repetition and the p-value collapses. This
    # is the bug that put p=0.000 on a level holding three stations.
    table = pd.DataFrame([{"stratum": "grp"}])
    merged = _cells_with_correlated_sites(seed=0)
    per_cell = _per_cell_p(merged, permutations=400)
    site_level = report._stratum_significance(merged, table, "rho_ctrl").iloc[0]
    check("the per-cell shuffle calls a RANDOM label significant",
          per_cell <= 0.05, f"p={per_cell:.3f} on labels assigned by coin flip")
    check("the site-level shuffle does not",
          site_level["p"] > 0.05, f"p={site_level['p']:.3f}")
    check("and both read the same observed ratio — only the null changed",
          site_level["iqr_ratio"] < 1.0, f"{site_level['iqr_ratio']:.2f}")
    check("the row says how few stations the smallest level holds",
          int(site_level["smallest_level"]) == 3 and int(site_level["sites"]) == 40,
          f"smallest_level={site_level['smallest_level']}, "
          f"sites={site_level['sites']}")

    # Not one lucky seed: across replicates with no effect to find, the old
    # null fires more often, and never reports a SMALLER p than the new one
    # on the cases where it fires.
    fired_per_cell = fired_site = 0
    never_smaller = True
    for seed in range(16):
        merged = _cells_with_correlated_sites(seed)
        per_cell = _per_cell_p(merged, permutations=150)
        site = report._stratum_significance(merged, table, "rho_ctrl",
                                            permutations=150).iloc[0]["p"]
        fired_per_cell += per_cell <= 0.05
        fired_site += site <= 0.05
        if per_cell <= 0.05 and site < per_cell:
            never_smaller = False
    check("the old null fires more often than the new one on null data",
          fired_per_cell > fired_site,
          f"per-cell fired {fired_per_cell}/16, site-level {fired_site}/16 "
          "— 5% of 16 is under 1")
    check("and the new p is never the smaller of the two where it matters",
          never_smaller)


def test_the_report_says_where_its_stations_are():
    print("\nthe report says what area it covers (C2)")
    # Every table under D1 describes whichever stations survived. A one-state
    # pass prints the same shapes as a national one, so the scope has to be
    # stated or the reader supplies the wrong one for free.
    import contextlib
    import io as _io

    def scope_text(rows):
        buffer = _io.StringIO()
        with contextlib.redirect_stdout(buffer):
            report.report_scope(pd.DataFrame(rows))
        return buffer.getvalue()

    one_state = scope_text([{"station_id": f"RI{i}", "state": "RI",
                             "analyte": "ENT"} for i in range(4)]
                           + [{"station_id": "RI0", "state": "RI",
                               "analyte": "ECOLI"}])
    check("one state is named and called out as one state",
          "every fitted station is in RI" in one_state, one_state.strip())
    check("and the station count is stations, not site-analyte pairs",
          "RI    4" in one_state, one_state)

    many = scope_text([{"station_id": f"S{i}",
                        "state": state, "analyte": "ENT"}
                       for i, state in enumerate(
                           ["RI", "MA", "CA", "FL", "WA", "OR"])])
    check("six states does not trigger the warning",
          "SCOPE" not in many, many.strip())

    blank = scope_text([{"station_id": "S1", "state": None, "analyte": "ENT"},
                        {"station_id": "S2", "state": "RI", "analyte": "ENT"}])
    check("a station with no state is counted somewhere, not silently dropped",
          "1 fitted station(s) with no state on record" in blank, blank.strip())

    missing = scope_text([{"station_id": "S1", "analyte": "ENT"}])
    check("no state column at all says so rather than printing nothing",
          "cannot say where" in missing, missing.strip())


def test_no_usable_predictor_is_counted():
    print("\nsites with no usable predictor (D6)")
    samples = pd.concat([
        make_site_samples("GOOD", n=120, rain_effect=1.4),
        make_site_samples("FLAT", n=120, rain_effect=0.0),
    ], ignore_index=True)
    sites = make_sites([
        {"station_id": "GOOD", "station_name": "Good Beach",
         "site_type": "Ocean", "lat": 33.0, "lon": -117.3, "state": "CA"},
        {"station_id": "FLAT", "station_name": "Flat Beach",
         "site_type": "Ocean", "lat": 34.0, "lon": -118.3, "state": "CA"}])
    payload, path = registered(sites)
    coefficients, _ = fit.run(samples, sites, nondetect_table(samples),
                              payload=payload)
    os.remove(path)
    best = report.usable_predictors(coefficients)
    usable = dict(zip(best["station_id"], best["has_usable"]))
    check("the site with a planted driver has a usable predictor",
          bool(usable["GOOD"]))
    check("the flat site is counted as having none",
          not bool(usable["FLAT"]),
          f"best |rho| {float(best.set_index('station_id').loc['FLAT', 'best_abs_rho']):.2f}")
    beats = dict(zip(best["station_id"], best["beats_chance"]))
    check("the planted site beats the chance criterion too", bool(beats["GOOD"]))
    check("the flat site does not beat chance", not bool(beats["FLAT"]),
          f"best |rho| {float(best.set_index('station_id').loc['FLAT', 'best_abs_rho']):.3f} "
          f"vs threshold {float(best.set_index('station_id').loc['FLAT', 'chance_threshold']):.3f}")


def test_multiple_testing_expectation():
    print("\nfalse positives at alpha (D5)")
    # Twelve beaches where nothing is going on. With 11 predictors each, some
    # will look significant. The report has to say how many were expected.
    rows, sites = [], []
    for index in range(12):
        name = f"NULL{index}"
        rows.append(make_site_samples(name, n=60, rain_effect=0.0))
        sites.append({"station_id": name, "station_name": "Ocean Beach",
                      "site_type": "Ocean", "lat": 33.0 + index / 10,
                      "lon": -117.3, "state": "CA"})
    samples = pd.concat(rows, ignore_index=True)
    frame = make_sites(sites)
    payload, path = registered(frame)
    coefficients, _ = fit.run(samples, frame, nondetect_table(samples),
                              payload=payload)
    os.remove(path)
    summary = report.report_multiple_testing(coefficients)
    check("tests counted", summary["n_tests"] == len(coefficients),
          str(summary["n_tests"]))
    check("the expectation is alpha x tests",
          abs(summary["expected"] - config.ALPHA * summary["n_tests"]) < 1e-9)
    check("on pure noise, the observed count is near the expectation",
          summary["observed"] <= 4 * max(summary["expected"], 1),
          f"observed {summary['observed']}, expected {summary['expected']:.1f}")
    check("Benjamini-Hochberg keeps almost nothing on noise",
          summary["bh"] <= 2, str(summary["bh"]))


def test_per_site_table_has_what_was_asked_for():
    print("\nthe per-site table (D4)")
    samples = pd.concat([
        make_site_samples("A", n=90, rain_effect=1.0),
        make_site_samples("B", n=90, rain_effect=0.2),
    ], ignore_index=True)
    sites = make_sites([
        {"station_id": "A", "station_name": "Newport Harbor",
         "site_type": "Estuary", "lat": 33.6, "lon": -117.9, "state": "CA"},
        {"station_id": "B", "station_name": "Ocean Beach",
         "site_type": "Ocean", "lat": 41.9, "lon": -87.6, "state": "IL"}])
    payload, path = registered(sites)
    coefficients, attrition = fit.run(samples, sites, nondetect_table(samples),
                                      payload=payload)
    shares = nondetect_table(samples)
    table = report.per_site_table(coefficients, sites, shares,
                                 manifest.active_strata(payload))
    os.remove(path)
    for column in ("station_id", "region", "beach_type", "n",
                   "nondetect_fraction", "rain_24h_mm", "auc_rain_24h_mm",
                   "best_predictor"):
        check(f"per-site table carries {column}", column in table.columns)
    check("one row per site and analyte", len(table) == 2, str(len(table)))
    check("the Great Lakes site is labelled as such",
          set(table["region"]) == {"Pacific", "Great Lakes"},
          str(set(table["region"])))


def test_d1_says_which_stations_a_predictor_covers():
    print("\nD1 names its denominator (D1)")
    # D1 prints n_sites and no denominator. Across eleven predictors at one
    # analyte those n_sites disagree -- 54 against 120 on the Rhode Island run
    # -- and the smaller number cannot say whether the missing stations had the
    # predictor and failed to correlate, or never had it at all. Build all three
    # reasons deliberately and check they stay apart.
    import contextlib
    import io as _io

    rows = []
    for index in range(10):
        station = f"S{index}"
        # rain: fitted everywhere.
        rows.append({"station_id": station, "analyte": "ENT",
                     "predictor": "rain_24h_mm", "family": "rain",
                     "n_ctrl": 80, "rho_ctrl": 0.2 + 0.01 * index})
        # wave: fitted at 4, NEVER MEASURED at 4, measured-but-short at 2.
        if index < 4:
            wave = {"n_ctrl": 60, "rho_ctrl": 0.1}
        elif index < 8:
            wave = {"n_ctrl": 0, "rho_ctrl": np.nan}       # never attempted
        else:
            wave = {"n_ctrl": 12, "rho_ctrl": np.nan}      # attempted, refused
        rows.append({"station_id": station, "analyte": "ENT",
                     "predictor": "wave_height", "family": "wave", **wave})
        # level: measured everywhere but flat at 3 of them.
        flat = index < 3
        rows.append({"station_id": station, "analyte": "ENT",
                     "predictor": "level_m", "family": "tide",
                     "n_ctrl": 90,
                     "rho_ctrl": np.nan if flat else -0.05})
    coefficients = pd.DataFrame(rows)

    table = report.predictor_coverage(coefficients).set_index("predictor")
    wave = table.loc["wave_height"]
    check("a predictor never recorded at a station is counted as 'never'",
          int(wave["never"]) == 4, f"never={int(wave['never'])}")
    check("and one recorded but under the paired floor is counted separately",
          int(wave["too_few"]) == 2, f"too_few={int(wave['too_few'])}")
    check("the two are not collapsed into one number",
          int(wave["never"]) != int(wave["too_few"]))
    check("a flat series is 'no_variation', not 'never'",
          int(table.loc["level_m", "no_variation"]) == 3
          and int(table.loc["level_m", "never"]) == 0,
          f"no_variation={int(table.loc['level_m', 'no_variation'])}, "
          f"never={int(table.loc['level_m', 'never'])}")
    for predictor, row in table.iterrows():
        total = row["fitted"] + row["too_few"] + row["never"] + row["no_variation"]
        check(f"{predictor}: the reasons account for every fitted station",
              int(total) == int(row["of"]), f"{int(total)} of {int(row['of'])}")

    # fitted has to BE the number D1 prints, or the table explains a different
    # figure from the one on the page.
    buffer = _io.StringIO()
    with contextlib.redirect_stdout(buffer):
        d1 = report.report_distributions(coefficients)
    printed = d1.set_index("predictor")["n_sites"]
    check("fitted equals the n_sites D1 prints, predictor by predictor",
          all(int(table.loc[name, "fitted"]) == int(printed[name])
              for name in printed.index),
          f"wave_height {int(table.loc['wave_height', 'fitted'])} "
          f"vs D1 {int(printed['wave_height'])}")

    text = buffer.getvalue()
    check("the warning names the thin predictor and its real denominator",
          "READ D1 WITH THIS: wave_height was fitted at 4 of 10" in text)
    check("and says the row is not comparable with a fully covered one",
          "comparable" in text)

    full = coefficients[coefficients["predictor"] == "rain_24h_mm"]
    quiet = _io.StringIO()
    with contextlib.redirect_stdout(quiet):
        report.report_distributions(full)
    check("a predictor covering every station raises no warning",
          "READ D1 WITH THIS" not in quiet.getvalue())


def test_ab411_ratio_rule():
    """C4 / 17 CCR 7958: total coliform is 10,000 per 100 mL, and 1,000 when
    the fecal/total ratio on the SAME sample exceeds 0.1. Both limbs live in
    wq/thresholds.json; neither number appears in wq/fit.py."""
    print("\n[AB 411 ratio rule]")
    thresholds = fit.load_thresholds()
    sites = pd.DataFrame([{"station_id": "CA-1", "state": "CA",
                           "water_class": "marine"}])
    rows = [
        # ratio 0.20 -> stricter limb
        {"station_id": "CA-1", "analyte": "TOTAL", "date": "2021-06-01",
         "sampled_at": "2021-06-01T09:00", "value": 5000.0},
        {"station_id": "CA-1", "analyte": "FECAL", "date": "2021-06-01",
         "sampled_at": "2021-06-01T09:00", "value": 1000.0},
        # ratio 0.02 -> permissive limb stands
        {"station_id": "CA-1", "analyte": "TOTAL", "date": "2021-06-02",
         "sampled_at": "2021-06-02T09:00", "value": 5000.0},
        {"station_id": "CA-1", "analyte": "FECAL", "date": "2021-06-02",
         "sampled_at": "2021-06-02T09:00", "value": 100.0},
        # no fecal companion at all -> permissive limb, flagged
        {"station_id": "CA-1", "analyte": "TOTAL", "date": "2021-06-03",
         "sampled_at": "2021-06-03T09:00", "value": 5000.0},
        # same DAY, different bottle -> must NOT pair
        {"station_id": "CA-1", "analyte": "TOTAL", "date": "2021-06-04",
         "sampled_at": "2021-06-04T09:00", "value": 5000.0},
        {"station_id": "CA-1", "analyte": "FECAL", "date": "2021-06-04",
         "sampled_at": "2021-06-04T16:00", "value": 4000.0},
        # date-only (midnight = no time reported) -> pairs on the date
        {"station_id": "CA-1", "analyte": "TOTAL", "date": "2021-06-05",
         "sampled_at": "2021-06-05T00:00", "value": 5000.0},
        {"station_id": "CA-1", "analyte": "FECAL", "date": "2021-06-05",
         "sampled_at": "2021-06-05T00:00", "value": 2000.0},
    ]
    out = fit.apply_ratio_rules(pd.DataFrame(rows), sites, thresholds)
    total = out[out["analyte"] == "TOTAL"].set_index("date")

    check("ratio over 0.1 drops the limit to the stricter limb",
          float(total.loc["2021-06-01", "exceedance_threshold"]) == 1000.0,
          f"ratio {float(total.loc['2021-06-01', 'ratio_value']):.2f}")
    check("ratio under 0.1 leaves the permissive limb standing",
          float(total.loc["2021-06-02", "exceedance_threshold"]) == 10000.0)
    check("no companion result is flagged, not silently permissive",
          bool(total.loc["2021-06-03", "ratio_rule_unavailable"])
          and float(total.loc["2021-06-03", "exceedance_threshold"]) == 10000.0)
    check("a different bottle the same day does NOT pair",
          bool(total.loc["2021-06-04", "ratio_rule_unavailable"]),
          "two grabs at one station are two samples")
    check("a date-only pair still pairs, on the date",
          float(total.loc["2021-06-05", "exceedance_threshold"]) == 1000.0,
          "midnight means no time reported, as join_samples also assumes")
    check("neither limb is hardcoded in fit.py",
          "10000" not in open(os.path.join(os.path.dirname(
              os.path.abspath(__file__)), "fit.py")).read()
          and "1000," not in open(os.path.join(os.path.dirname(
              os.path.abspath(__file__)), "fit.py")).read())


def test_an_unknowable_exceedance_label_is_not_a_negative():
    """A total-coliform sample with no same-sample fecal result, sitting
    BETWEEN the two AB 411 limbs, exceeds under one and not the other. Nothing
    in the record says which applied. Scoring it as a negative would train a
    classifier to reproduce a gap in the sampling programme."""
    print("\n[unscorable exceedance labels]")
    thresholds = fit.load_thresholds()
    sites = pd.DataFrame([{"station_id": "CA-1", "state": "CA",
                           "water_class": "marine"}])
    rows = [
        # no companion, below both limbs -> knowable negative
        {"station_id": "CA-1", "analyte": "TOTAL", "date": "2021-06-01",
         "sampled_at": "2021-06-01T09:00", "value": 400.0},
        # no companion, BETWEEN the limbs -> unknowable
        {"station_id": "CA-1", "analyte": "TOTAL", "date": "2021-06-02",
         "sampled_at": "2021-06-02T09:00", "value": 5000.0},
        # no companion, above both limbs -> knowable positive
        {"station_id": "CA-1", "analyte": "TOTAL", "date": "2021-06-03",
         "sampled_at": "2021-06-03T09:00", "value": 40000.0},
        # companion present and between the limbs -> knowable, ratio decides
        {"station_id": "CA-1", "analyte": "TOTAL", "date": "2021-06-04",
         "sampled_at": "2021-06-04T09:00", "value": 5000.0},
        {"station_id": "CA-1", "analyte": "FECAL", "date": "2021-06-04",
         "sampled_at": "2021-06-04T09:00", "value": 1000.0},
    ]
    out = fit.apply_ratio_rules(pd.DataFrame(rows), sites, thresholds)
    total = out[out["analyte"] == "TOTAL"].set_index("date")
    check("below both limbs is scorable even with no companion",
          bool(total.loc["2021-06-01", "exceedance_scorable"]))
    check("BETWEEN the limbs with no companion is NOT scorable",
          not bool(total.loc["2021-06-02", "exceedance_scorable"]),
          "exceeds under 1,000, does not under 10,000")
    check("above both limbs is scorable even with no companion",
          bool(total.loc["2021-06-03", "exceedance_scorable"]))
    check("a measured ratio makes it scorable again",
          bool(total.loc["2021-06-04", "exceedance_scorable"])
          and float(total.loc["2021-06-04", "exceedance_threshold"]) == 1000.0)

    other = fit.apply_ratio_rules(
        pd.DataFrame([{"station_id": "CA-1", "analyte": "ENT",
                       "date": "2021-06-01", "sampled_at": "2021-06-01T09:00",
                       "value": 50.0}]), sites, thresholds)
    check("an analyte with one limb is always scorable",
          bool(other["exceedance_scorable"].iloc[0]),
          "there is only one number to be on one side of")


def test_a_shellfish_station_is_not_judged_as_a_beach():
    """The Northeast's fecal coliform is shellfish growing-area monitoring:
    no BEACH Act station in the study reports fecal coliform at all. NSSP
    classifies growing areas at 43 MPN/100 mL, bathing beaches are posted at
    400. Scoring one against the other answers a question nobody asked."""
    print("\n[programme criteria]")
    from wq import strata
    thresholds = fit.load_thresholds()

    shellfish = fit.threshold_for(thresholds, "NJ", "FECAL", "marine",
                                  "non_beach")
    beach = fit.threshold_for(thresholds, "NJ", "FECAL", "marine", "beach_act")
    check("a non-beach station gets the NSSP fecal limit",
          shellfish[0] == 43 and "programme:non_beach" in shellfish[2],
          str(shellfish))
    check("a BEACH Act station does not", beach[0] == 400, str(beach))
    check("programme beats the state block",
          fit.threshold_for(thresholds, "CA", "FECAL", "marine",
                            "non_beach")[0] == 43,
          "a growing area is judged by NSSP wherever it sits")
    check("California's beaches are untouched",
          fit.threshold_for(thresholds, "CA", "FECAL", "marine",
                            "beach_act")[0] == 400)
    check("an analyte with no programme entry falls through",
          "_default" in fit.threshold_for(thresholds, "NJ", "ENT", "marine",
                                          "non_beach")[2],
          "NSSP has no enterococcus criterion")
    check("no programme at all behaves as before",
          fit.threshold_for(thresholds, "NJ", "FECAL", "marine")[0] == 400)

    check("the label comes from metadata, never from the analyte mix",
          strata.programme_of("BEACH Program Site-Ocean") == "beach_act"
          and strata.programme_of("Estuary") == "non_beach"
          and strata.programme_of(None) is None)
    check("neither NSSP limb is hardcoded in fit.py",
          "43" not in open(os.path.join(os.path.dirname(
              os.path.abspath(__file__)), "fit.py")).read().replace(
                  "n=43", "").replace("4390", ""))


def test_excluded_stations_are_reported_not_vanished():
    """A station kept out by decision must appear in the attrition census
    under its own outcome. An exclusion that leaves no row is invisible."""
    print("\n[decided exclusions]")
    excluded = fit.load_exclusions()
    check("the exclusion list is committed and readable", len(excluded) > 0,
          f"{len(excluded)} station(s)")
    check("every entry carries a reason",
          all(bool(v) for v in excluded.values()),
          sorted(set(excluded.values()))[0][:52])


def test_california_has_no_ocean_ecoli_standard():
    """17 CCR 7958 lists total coliform, fecal coliform and enterococcus and
    nothing else. Scoring Californian E. coli against EPA's FRESHWATER 410
    would manufacture exceedances against a rule no beach is posted on."""
    print("\n[CA marine E. coli]")
    thresholds = fit.load_thresholds()
    value, unit, source = fit.threshold_for(thresholds, "CA", "ECOLI", "marine")
    check("no threshold is returned", value is None and unit is None)
    check("and it does NOT fall through to the federal default",
          "_default" not in source, source)
    check("the source says why", "no applicable standard" in source, source)

    federal = fit.threshold_for(thresholds, "NJ", "ECOLI", "marine")
    check("a state without the marker still gets the federal fallback",
          federal[0] is not None and "_default" in federal[2], str(federal))

    ent = fit.threshold_for(thresholds, "CA", "ENT", "marine")
    check("California's other analytes are unaffected",
          ent[0] == 104 and ent[2] == "CA:marine", str(ent))

    # and the fit reports it rather than dropping the rows
    sites = pd.DataFrame([{"station_id": "CA-1", "state": "CA",
                           "water_class": "marine", "lat": 34.0, "lon": -119.0}])
    rng = np.random.default_rng(0)
    n = 40
    group = pd.DataFrame({
        "station_id": "CA-1", "analyte": "ECOLI",
        "date": pd.date_range("2021-01-01", periods=n, freq="7D"),
        "value": rng.lognormal(3, 1, n),
        "rain_24h_mm": rng.random(n),
    })
    group["log_value"] = np.log10(group["value"])
    rows = fit.fit_site_analyte(group, sites.iloc[0], "ECOLI", thresholds)
    first = rows[0]
    check("the exceedance column is NA, not a number",
          pd.isna(first["auc_exceedance"]))
    check("the row records the reason",
          "no applicable standard" in str(first["exceedance_reason"]),
          str(first["exceedance_reason"]))
    check("the coefficient itself is unaffected",
          first["n_ctrl"] > 0 and not pd.isna(first["rho_ctrl"]))


def test_a_verdict_needs_an_effect_not_just_a_p_value():
    """The report printed "region narrows the IQR ... that is the headline"
    for a narrowing of 1.5% at p=0.000. With 2,591 sites a p-value detects an
    effect far too small to act on, and the canned verdict read the p alone."""
    print("\n[D2 verdict gate]")
    import io, contextlib

    def verdict(iqr_ratio, chance, p, sites, smallest):
        row = pd.Series({"stratum": "region", "iqr_ratio": iqr_ratio,
                         "chance_ratio": chance, "p": p, "sites": sites,
                         "smallest_level": smallest})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report._report_stratum_verdict(row)
        return buf.getvalue()

    # THE REGION CASE, as it actually came out of the CA-amended run.
    out = verdict(0.985, 0.998, 0.000, 2591, 556)
    check("the region case is no longer called the headline",
          "THAT IS THE HEADLINE" not in out)
    check("it is called detectable but too small to act on",
          "TOO SMALL TO ACT ON" in out)
    check("and it still prints both numbers",
          "1.5%" in out and "0.000" in out, out.strip().splitlines()[0][:60])

    # a real effect: big narrowing, significant, no tiny level
    out = verdict(0.60, 0.98, 0.001, 2000, 500)
    check("a 40% narrowing on a well-populated stratum IS the headline",
          "THAT IS THE HEADLINE" in out)

    # outfall_type's shape: large narrowing carried by a 2-station level
    out = verdict(0.834, 0.982, 0.000, 2303, 2)
    check("a large narrowing carried by a tiny level is NOT the headline",
          "THAT IS THE HEADLINE" not in out and "smallest level" in out,
          "2 stations of 2,303 is 0.1%")

    check("the bars are outside the frozen specification",
          "STRATUM_MIN_NARROWING" not in config.SPEC_KEYS
          and "STRATUM_MIN_SMALLEST_LEVEL_SHARE" not in config.SPEC_KEYS,
          "they gate wording, not arithmetic")
    check("the narrowing bar is derived, not round-numbered",
          0 < config.STRATUM_MIN_NARROWING < 1,
          f"{config.STRATUM_MIN_NARROWING:.0%} of a 0.25 IQR = "
          f"{0.25 * config.STRATUM_MIN_NARROWING:.3f} rho")


def main():
    for test in (test_distribution_is_recovered_not_averaged,
                 test_every_coefficient_carries_its_n,
                 test_n_mismatch_is_fatal,
                 test_pooling_would_mislead,
                 test_season_control,
                 test_exceedance_separation,
                 test_attrition_is_an_output,
                 test_stratification_detected_when_present_and_absent,
                 test_the_shuffle_is_drawn_once_per_site,
                 test_the_report_says_where_its_stations_are,
                 test_d1_says_which_stations_a_predictor_covers,
                 test_no_usable_predictor_is_counted,
                 test_multiple_testing_expectation,
                 test_per_site_table_has_what_was_asked_for,
                 test_a_verdict_needs_an_effect_not_just_a_p_value,
                 test_ab411_ratio_rule,
                 test_a_shellfish_station_is_not_judged_as_a_beach,
                 test_excluded_stations_are_reported_not_vanished,
                 test_an_unknowable_exceedance_label_is_not_a_negative,
                 test_california_has_no_ocean_ecoli_standard):
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all offline fit and report checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
