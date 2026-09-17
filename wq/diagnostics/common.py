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

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from wq import config, holdout  # noqa: E402

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


def diagnostic_parser(description):
    """Every diagnostic takes the same two guards, so none can forget one."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--evaluate-holdout", action="store_true",
        help="KEEP the held-out clusters and the held-out months. Only valid "
             "as a deliberate one-shot evaluation; the default drops them.")
    parser.add_argument(
        "--include-hypothesis-sites", action="store_true",
        help="KEEP Santa Cruz Wharf and Carpinteria State Beach in the "
             "summary statistics. They are excluded by default because their "
             "prior findings are what this pass is testing.")
    return parser


def drop_held_out(frame, args=None, samples=None, label="rows"):
    """The holdout guard, as every diagnostic must apply it.

    Phase 1 diagnostics describe the DEVELOPMENT data. A number computed over
    the held-out clusters is not a held-out number any more -- it has been
    looked at, and whatever is decided next is decided partly on it. So the
    default is to drop them, and keeping them takes a flag that says so out
    loud.
    """
    evaluate = bool(getattr(args, "evaluate_holdout", False))
    before = len(frame)
    out = holdout.drop_holdout(frame, evaluate_holdout=evaluate,
                               samples=samples, quiet=True)
    if evaluate:
        print(f"  --evaluate-holdout: KEEPING all {before:,} {label}, "
              "held-out clusters included.")
        print("  This is a held-out evaluation and must be reported as one.")
    elif len(out) != before:
        held = holdout.read_holdout()
        print(f"  holdout guard: dropped {before - len(out):,} of "
              f"{before:,} {label} "
              f"({held['site_cluster'].nunique()} held-out cluster(s), "
              f"seed {holdout.HOLDOUT_SEED})")
    return out


def drop_hypothesis_sites(frame, args=None, label="rows"):
    """Santa Cruz Wharf and Carpinteria, out of the summary statistics.

    These two beaches are the reason the prior findings exist. Leaving them in
    a replication statistic asks whether the sites that generated a hypothesis
    support it, which they do by construction. They are reported individually
    instead, and the list is committed (wq/hypothesis_sites.csv) rather than
    typed into a script, so the exclusion is auditable and cannot drift.
    """
    if bool(getattr(args, "include_hypothesis_sites", False)):
        print("  --include-hypothesis-sites: the hypothesis beaches are IN "
              "the statistics below.")
        return frame
    sites = holdout.read_hypothesis_sites()
    if sites.empty or "station_id" not in frame.columns:
        return frame
    ids = set(sites["station_id"].astype(str))
    out = frame[~frame["station_id"].astype(str).isin(ids)]
    if len(out) != len(frame):
        print(f"  hypothesis-site guard: dropped {len(frame) - len(out):,} of "
              f"{len(frame):,} {label} "
              f"({sites['site_cluster'].nunique()} cluster(s), "
              f"{len(ids)} station(s)) — reported separately, never in the "
              "replication statistics")
    return out


def hypothesis_rows(frame):
    """Just the hypothesis beaches, for the separate report."""
    sites = holdout.read_hypothesis_sites()
    if sites.empty or "station_id" not in frame.columns:
        return frame.iloc[0:0]
    ids = set(sites["station_id"].astype(str))
    return frame[frame["station_id"].astype(str).isin(ids)]
