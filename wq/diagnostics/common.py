"""Loaders and definitions shared by the Phase 1 diagnostic scripts.

Every definition here is read off wq/fit.py and wq/report.py rather than
restated, so a diagnostic cannot quietly disagree with the pipeline about
what a number means:

  headline      the coefficient rows D1-D6 actually describe. wq.report.run
                drops every row belonging to a site-analyte pair whose
                nondetect_flag is True BEFORE calling any of D1-D6, so any
                diagnostic that wants to reproduce a headline number has to
                drop them too.
  tested        p_ctrl is not null. wq.report.report_multiple_testing counts
                exactly these as D5's tests. A row with a null p_ctrl was
                emitted by the fit but no correlation was computed: the
                predictor was never measured at that station, or fewer than
                MIN_PAIRED_N rows had both sides non-null, or one side never
                varied.
  rho_ctrl      Spearman between the month-demeaned predictor and the
                month-demeaned log_value, within one site and analyte
                (wq.fit.fit_site_analyte via analyze_drivers.spearman and
                analyze_drivers.demean_by). It is never pooled across sites.
  BH            Benjamini-Hochberg at config.ALPHA over the whole headline
                tested set at once, which is what D5 does. A per-analyte or
                per-predictor BH would be a different and more lenient test,
                so this module exposes the pipeline's version and labels it.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from wq import config  # noqa: E402

OUT_DIR = os.path.join(config.OUT_DIR, "diagnostics")
RAIN = ["rain_24h_mm", "rain_48h_mm", "rain_72h_mm"]


def out_path(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def write(frame, name):
    path = out_path(name)
    frame.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(frame)} rows)")
    return path


def load_coefficients():
    return pd.read_csv(os.path.join(config.OUT_DIR, "coefficients.csv"))


def load_attrition():
    return pd.read_csv(os.path.join(config.OUT_DIR, "attrition.csv"))


def load_nondetects():
    return pd.read_csv(os.path.join(config.DATA_DIR, "nondetect_shares.csv"))


def load_sites():
    return pd.read_csv(os.path.join(config.DATA_DIR,
                                    "stations_stratified.csv"))


def load_samples(usecols=None):
    return pd.read_csv(os.path.join(config.DATA_DIR,
                                    "samples_with_covariates.csv"),
                       usecols=usecols, low_memory=False)


def flagged_pairs(nondetects=None):
    """The (station_id, analyte) pairs wq.report.run excludes from D1-D6."""
    nondetects = load_nondetects() if nondetects is None else nondetects
    if nondetects is None or nondetects.empty:
        return set()
    return {(str(r["station_id"]), r["analyte"])
            for _, r in nondetects.iterrows() if r["nondetect_flag"]}


def headline(coefficients, nondetects=None):
    """Exactly the rows wq.report.run passes to D1-D6."""
    bad = flagged_pairs(nondetects)
    if not bad:
        return coefficients
    keep = [(str(s), a) not in bad for s, a in
            zip(coefficients["station_id"], coefficients["analyte"])]
    return coefficients[keep]


def untested_reason(frame):
    """Why a fitted row carries no p_ctrl. Same three buckets, same order and
    same thresholds, as wq.report.predictor_coverage."""
    paired = pd.to_numeric(frame["n_ctrl"], errors="coerce").fillna(0)
    return np.where(
        frame["p_ctrl"].notna(), "tested",
        np.where(paired <= 0, "never_measured",
                 np.where(paired < config.MIN_PAIRED_N, "too_few_paired",
                          "no_variation")))


def bh_significant(p_values, alpha=None):
    """Boolean mask of the p-values surviving Benjamini-Hochberg.

    wq.report.report_multiple_testing reports only the COUNT that survives;
    D3 and D6 need to know WHICH rows those are, so the same step-up is
    repeated here and the count is asserted against the report's rule.
    """
    alpha = config.ALPHA if alpha is None else alpha
    values = pd.to_numeric(pd.Series(p_values), errors="coerce")
    tested = values.notna()
    out = pd.Series(False, index=values.index)
    n = int(tested.sum())
    if not n:
        return out
    order = values[tested].sort_values()
    ranks = np.arange(1, n + 1)
    passing = order.to_numpy() <= (ranks / n) * alpha
    if not passing.any():
        return out
    cut = int(ranks[passing].max())
    out.loc[order.index[:cut]] = True
    return out


def rho_summary(values):
    """Median and IQR, in the shape D1 reports them. No mean, per wq.report."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return {"n": 0, "median": np.nan, "q25": np.nan, "q75": np.nan,
                "iqr": np.nan}
    q25, q75 = float(series.quantile(0.25)), float(series.quantile(0.75))
    return {"n": int(len(series)), "median": float(series.median()),
            "q25": q25, "q75": q75, "iqr": q75 - q25}
