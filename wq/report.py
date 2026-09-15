"""D1-D6: the deliverable. A distribution, not a headline number.

The thing being reported is the SPREAD of site-level coefficients. A mean is
never printed for a coefficient, anywhere in this module, and that is
deliberate: averaging the sites answers "what is the effect", which is not
the question. Median, IQR, 10th/90th, min, max and n sites answer "how much
does it vary", which is.

  D1  the full distribution per analyte/predictor, with a text histogram
  D2  the same distribution inside each stratum, with within-stratum IQR
      against overall IQR -- the test of whether stratification works
  D3  sign agreement, reported but never led with. Sign is a weak test;
      magnitude is what a decision is made on
  D4  the per-site table: station, region, strata, n, non-detect fraction,
      coefficient per predictor, exceedance separation
  D5  expected false positives at alpha beside the observed count of
      "significant" sites
  D6  how many sites have NO usable predictor -- the fraction of beaches
      where this approach simply would not work

No pooled or hierarchical model is fitted here. Partial pooling is the next
pass and it should be informed by what this one shows.
"""

import math
import os

import numpy as np
import pandas as pd

from . import config

HIST_BINS = np.arange(-1.0, 1.01, 0.1)
HIST_WIDTH = 40


def distribution(values):
    """The six numbers that are the result, plus n. Note the absence of a mean."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return {"n_sites": 0, "median": np.nan, "q25": np.nan, "q75": np.nan,
                "iqr": np.nan, "p10": np.nan, "p90": np.nan,
                "min": np.nan, "max": np.nan}
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    return {
        "n_sites": int(len(series)),
        "median": float(series.median()),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "p10": float(series.quantile(0.10)),
        "p90": float(series.quantile(0.90)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def histogram(values, bins=HIST_BINS, width=HIST_WIDTH):
    """A text histogram. The spread is the result, so it gets drawn."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return ["    (no sites)"]
    counts, edges = np.histogram(series, bins=bins)
    peak = counts.max() or 1
    lines = []
    for count, low, high in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * count / peak))
        marker = " <- zero" if low <= 0 < high else ""
        lines.append(f"    {low:+.1f}..{high:+.1f} {count:>5} |{bar}{marker}")
    return lines


def best_of_k_threshold(n, k, alpha=None):
    """The |rho| that the best of k predictors clears by chance alone.

    D6 asks whether a site's STRONGEST correlation clears a floor. That is a
    maximum over the whole predictor list, and a maximum is biased upward:
    with eleven predictors at n=120, the best of eleven pure-noise
    correlations clears 0.20 most of the time. Comparing against a fixed
    floor alone would therefore report "this site has a usable predictor"
    for sites where nothing is happening.

    The correction is Sidak on the per-test alpha, inverted through the same
    Fisher z the correlation's own p-value uses, so no scipy is needed.
    """
    alpha = config.ALPHA if alpha is None else alpha
    n, k = int(n), max(int(k), 1)
    if n <= 4:
        return np.nan
    per_test = 1.0 - (1.0 - alpha) ** (1.0 / k)
    # Two-sided: solve erfc(z / sqrt(2)) = per_test for z by bisection, which
    # is exact enough here and keeps this module dependency-free.
    low, high = 0.0, 12.0
    for _ in range(80):
        middle = (low + high) / 2
        if math.erfc(middle / math.sqrt(2)) > per_test:
            low = middle
        else:
            high = middle
    z = (low + high) / 2
    return float(np.tanh(z / math.sqrt(n - 3)))


def usable_predictors(coefficients, floor=None):
    """Per site and analyte, the strongest |rho_ctrl| and whether it clears
    the pre-registered floor. D6 is built on this."""
    floor = config.USABLE_RHO_FLOOR if floor is None else floor
    frame = coefficients.copy()
    frame["abs_rho"] = frame["rho_ctrl"].abs()
    best = (frame.sort_values("abs_rho", ascending=False)
            .groupby(["station_id", "analyte"], as_index=False)
            .first()[["station_id", "analyte", "predictor", "abs_rho", "n_ctrl"]])
    best = best.rename(columns={"predictor": "best_predictor",
                                "abs_rho": "best_abs_rho"})
    best["has_usable"] = best["best_abs_rho"] >= floor

    # The same question asked without choosing a floor: does the strongest
    # correlation beat what the best of this many predictors would produce by
    # chance at this site's n?
    tested = (frame.groupby(["station_id", "analyte"])["predictor"]
              .nunique().rename("n_predictors").reset_index())
    best = best.merge(tested, on=["station_id", "analyte"], how="left")
    best["chance_threshold"] = [
        best_of_k_threshold(n, k)
        for n, k in zip(best["n_ctrl"].fillna(0), best["n_predictors"].fillna(1))]
    best["beats_chance"] = best["best_abs_rho"] >= best["chance_threshold"]
    return best


def report_distributions(coefficients, column="rho_ctrl"):
    """D1. Every analyte/predictor pair, as a distribution across sites."""
    print("\n" + "=" * 78)
    print("D1  DISTRIBUTION OF SITE-LEVEL COEFFICIENTS")
    print("=" * 78)
    print("The spread IS the result. No mean is reported: averaging the sites")
    print("answers a question nobody asked. Judge on the IQR and the 10th/90th.")
    rows = []
    for (analyte, predictor), group in coefficients.groupby(["analyte", "predictor"]):
        stats = distribution(group[column])
        stats.update({"analyte": analyte, "predictor": predictor,
                      "family": group["family"].iloc[0]})
        rows.append(stats)
    table = pd.DataFrame(rows)
    if table.empty:
        print("\n  no coefficients to report")
        return table
    order = ["analyte", "predictor", "family", "n_sites", "median", "q25",
             "q75", "iqr", "p10", "p90", "min", "max"]
    table = table[order].sort_values(["analyte", "family", "predictor"])
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.max_rows", 400):
        print("\n" + table.round(3).to_string(index=False))

    print("\nhistograms (one site = one count):")
    for (analyte, predictor), group in coefficients.groupby(["analyte", "predictor"]):
        values = pd.to_numeric(group[column], errors="coerce").dropna()
        if len(values) < 3:
            continue
        print(f"\n  {analyte} vs {predictor}  ({len(values)} sites)")
        for line in histogram(values):
            print(line)
    return table


def report_by_stratum(coefficients, sites, strata, column="rho_ctrl",
                      exploratory=()):
    """D2. The headline test: does knowing the stratum narrow the spread?

    `exploratory` names groupings that were added AFTER the registered pass,
    via manifest --amend. They are reported in the same table because hiding
    them would be its own kind of dishonesty, but every row carries a flag and
    the pre-registered summary is printed separately from them -- an
    exploratory grouping that narrows the spread is a hypothesis for the next
    pass, not a result of this one.
    """
    exploratory = set(exploratory or ())
    print("\n" + "=" * 78)
    print("D2  THE SAME DISTRIBUTION, BROKEN OUT BY STRATUM")
    print("=" * 78)
    if exploratory:
        print(f"  {len(exploratory)} grouping(s) here are EXPLORATORY — added "
              "after the registered\n  pass and marked in the exploratory "
              f"column: {', '.join(sorted(exploratory))}")
    print("If within-stratum IQR is much narrower than the overall IQR, the")
    print("stratification is doing work and that is the finding. If it is not,")
    print("the variation between beaches is not explained by what kind of")
    print("beach it is, and that is also a finding.")

    merged = coefficients.merge(
        sites[["station_id"] + [s for s in strata if s in sites.columns]],
        on="station_id", how="left")
    rows = []
    for stratum in strata:
        if stratum not in merged.columns:
            continue
        values = merged[stratum]
        if pd.api.types.is_numeric_dtype(values) and values.notna().sum():
            # A continuous stratum is split at its median, so it can be read
            # the same way as a categorical one.
            cut = values.median()
            merged[stratum] = np.where(values.isna(), None,
                                       np.where(values <= cut,
                                                f"<= {cut:.2f}", f"> {cut:.2f}"))
        for (analyte, predictor), group in merged.groupby(["analyte", "predictor"]):
            overall = distribution(group[column])
            if overall["n_sites"] < 3:
                continue
            for level, part in group.groupby(merged.loc[group.index, stratum],
                                             dropna=True):
                stats = distribution(part[column])
                if stats["n_sites"] < 3:
                    continue
                rows.append({
                    "stratum": stratum,
                    "exploratory": stratum in exploratory,
                    "level": str(level), "analyte": analyte,
                    "predictor": predictor, "n_sites": stats["n_sites"],
                    "median": stats["median"], "iqr": stats["iqr"],
                    "overall_iqr": overall["iqr"],
                    "iqr_ratio": (stats["iqr"] / overall["iqr"]
                                  if overall["iqr"] else np.nan),
                })
    table = pd.DataFrame(rows)
    if table.empty:
        print("\n  no stratum has three sites in any analyte/predictor cell")
        return table
    with pd.option_context("display.width", 220, "display.max_columns", 20,
                           "display.max_rows", 600):
        print("\n" + table.round(3).to_string(index=False))

    print("\nhow much each stratum narrows the spread:")
    print("  iqr_ratio      observed within-stratum IQR / overall IQR")
    print("  chance_ratio   the same thing when the labels are SHUFFLED —")
    print("                 splitting any group into subgroups narrows an IQR,")
    print("                 so this is the number iqr_ratio has to beat")
    print("  p              share of shuffles at least as narrow as observed")
    print("  sites          stations carrying a label for this stratum")
    print("  smallest_level stations in its smallest level — read every row")
    print("                 with this number in hand, because the shuffle")
    print("                 controls for the SIZE of a small level but not")
    print("                 for what else its members have in common")
    print("  The shuffle is drawn once per SITE and reused across every")
    print("  analyte/predictor cell, since the label is a property of the")
    print("  site. A per-cell shuffle would redraw the small level in every")
    print("  cell and call one coincidence eleven separate findings.")
    summary = _stratum_significance(merged, table, column)
    print("\n" + summary.round(3).to_string(index=False))

    summary["exploratory"] = summary["stratum"].isin(exploratory)
    registered_only = summary[~summary["exploratory"]]
    if exploratory and not registered_only.empty:
        print("\n  PRE-REGISTERED groupings only (this is the result):")
        print(registered_only.round(3).to_string(index=False))
        print("\n  exploratory groupings (a hypothesis for the next pass, "
              "not a result of this one):")
        print(summary[summary["exploratory"]].round(3).to_string(index=False))
        summary = registered_only

    real = summary[summary["p"].notna()]
    if real.empty:
        print("\n  Not enough sites per level to say whether any stratum "
              "explains the spread.")
        return table
    leader = real.sort_values("p").iloc[0]
    if leader["p"] <= 0.05 and leader["iqr_ratio"] < leader["chance_ratio"]:
        print(f"\n  {leader['stratum']} narrows the IQR to "
              f"{leader['iqr_ratio']:.2f} of the overall spread, against "
              f"{leader['chance_ratio']:.2f} for a shuffle of the same group "
              f"sizes drawn once per site (p={leader['p']:.3f}). That is the "
              "headline: the coefficient varies by site type, and site type "
              "is knowable in advance.")
        smallest = leader.get("smallest_level")
        if smallest is not None and smallest < 10:
            print(f"  READ IT WITH THIS: the smallest level of "
                  f"{leader['stratum']} holds {int(smallest)} station(s) of "
                  f"{int(leader['sites'])}. The shuffle controls for that "
                  "SIZE; it cannot control for anything else those stations "
                  "share. Treat this as a hypothesis for a pass with more "
                  "sites in that level, not a settled result.")
    else:
        print(f"\n  No stratum narrows the spread by more than an arbitrary "
              f"split of the same sizes would. Best was {leader['stratum']} "
              f"at p={leader['p']:.3f}. On this evidence the between-site "
              "variation is NOT explained by the strata registered in A2, "
              "and the next pass should not assume it is.")
    return table


def _median_iqr_ratio(cells):
    """Median within-level IQR / overall IQR across every analyte/predictor
    cell, which is the one number a stratum is judged on."""
    ratios = []
    for values, labels, _sites, overall_iqr in cells:
        if not overall_iqr:
            continue
        for level in pd.unique(labels):
            part = values[labels == level]
            if len(part) < 3:
                continue
            q25, q75 = np.quantile(part, 0.25), np.quantile(part, 0.75)
            ratios.append((q75 - q25) / overall_iqr)
    return float(np.median(ratios)) if ratios else np.nan


def _stratum_significance(merged, table, column, permutations=None):
    """Permutation test per stratum. See the printed legend for why.

    The label is shuffled ONCE PER SITE and that one shuffle is applied to
    every analyte/predictor cell, because a stratum is a property of the site,
    not of the cell. Shuffling each cell independently -- which is what this
    function used to do -- builds a null in which the small level is a
    different handful of sites in every cell, while the observed statistic
    reads the SAME handful eleven times over. Anything those particular sites
    have in common other than their label (one sampling program, one estuary,
    short records whose rho all shrink toward the same band) then shows up
    once in the data and eleven times in the statistic, and is compared
    against a null that redraws it away. That produced p=0.000 on strata whose
    smallest level held three stations. The site-level shuffle carries the
    same repetition into the null, so the test answers the question actually
    being asked: does the LABEL narrow the spread, or do these few sites
    merely resemble each other?
    """
    permutations = permutations or config.STRATUM_PERMUTATIONS
    rng = np.random.default_rng(0)
    rows = []
    for stratum in table["stratum"].unique():
        cells = []
        for (_analyte, _predictor), group in merged.groupby(["analyte", "predictor"]):
            frame = group[["station_id", column, stratum]].dropna()
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            labels = frame[stratum].astype(str).to_numpy()
            stations = frame["station_id"].astype(str).to_numpy()
            if len(values) < 6 or len(pd.unique(labels)) < 2:
                continue
            q25, q75 = np.quantile(values, 0.25), np.quantile(values, 0.75)
            cells.append((values, labels, stations, float(q75 - q25)))
        if not cells:
            rows.append({"stratum": stratum, "cells": 0, "iqr_ratio": np.nan,
                         "chance_ratio": np.nan, "p": np.nan, "sites": 0,
                         "smallest_level": 0})
            continue
        observed = _median_iqr_ratio(cells)
        site_labels = {}
        for _values, labels, stations, _overall in cells:
            for station, label in zip(stations, labels):
                site_labels.setdefault(station, label)
        roster = np.array(sorted(site_labels))
        assigned = np.array([site_labels[s] for s in roster])
        position = {station: index for index, station in enumerate(roster)}
        # Each cell's stations, as indices into the single site roster, so one
        # permutation of `assigned` relabels every cell consistently.
        indexed = [(values, np.array([position[s] for s in stations]), overall)
                   for values, _labels, stations, overall in cells]
        null = []
        for _ in range(permutations):
            permuted = rng.permutation(assigned)
            null.append(_median_iqr_ratio(
                [(values, permuted[index], None, overall)
                 for values, index, overall in indexed]))
        null = np.array([v for v in null if np.isfinite(v)])
        counts = pd.Series(assigned).value_counts()
        rows.append({
            "stratum": stratum,
            "cells": len(cells),
            "sites": len(roster),
            "smallest_level": int(counts.min()),
            "iqr_ratio": observed,
            "chance_ratio": float(np.median(null)) if len(null) else np.nan,
            "p": (float((null <= observed).mean()) if len(null)
                  and np.isfinite(observed) else np.nan),
        })
    return pd.DataFrame(rows)


def report_sign_agreement(coefficients, column="rho_ctrl"):
    """D3. Secondary. Reported because it was asked for, not led with."""
    print("\n" + "=" * 78)
    print("D3  SIGN AGREEMENT (secondary — magnitude is the decision variable)")
    print("=" * 78)
    rows = []
    for (analyte, predictor), group in coefficients.groupby(["analyte", "predictor"]):
        values = pd.to_numeric(group[column], errors="coerce").dropna()
        if values.empty:
            continue
        positive = int((values > 0).sum())
        rows.append({"analyte": analyte, "predictor": predictor,
                     "n_sites": len(values), "share_positive": positive / len(values)})
    table = pd.DataFrame(rows)
    if table.empty:
        print("  nothing to report")
        return table
    table["agreement"] = (table["share_positive"] - 0.5).abs() * 2
    with pd.option_context("display.width", 200, "display.max_rows", 400):
        print(table.round(3).sort_values("agreement", ascending=False)
              .to_string(index=False))
    print("\n  A predictor can agree in sign at 95% of sites and still be")
    print("  useless everywhere, if the magnitudes are all near zero. Read D1.")
    return table


def report_multiple_testing(coefficients):
    """D5. With hundreds of sites, some will look strong by chance."""
    print("\n" + "=" * 78)
    print("D5  MULTIPLE TESTING")
    print("=" * 78)
    tested = coefficients["p_ctrl"].notna()
    n_tests = int(tested.sum())
    observed = int((coefficients.loc[tested, "p_ctrl"] < config.ALPHA).sum())
    expected = config.ALPHA * n_tests
    print(f"  tests run                    {n_tests:,}")
    print(f"  significant at alpha={config.ALPHA}     {observed:,}")
    print(f"  expected by chance alone     {expected:,.1f}")
    if n_tests:
        print(f"  excess over chance           {observed - expected:+,.1f} "
              f"({observed / expected:.1f}x)" if expected else "")

    # Benjamini-Hochberg, which is the honest count when the tests are this many.
    values = coefficients.loc[tested, "p_ctrl"].sort_values().to_numpy()
    survived = 0
    if n_tests:
        ranks = np.arange(1, n_tests + 1)
        passing = values <= (ranks / n_tests) * config.ALPHA
        survived = int(ranks[passing].max()) if passing.any() else 0
    print(f"  surviving Benjamini-Hochberg {survived:,}")
    print("\n  The predictor families are nested (rain_24/48/72h are rolling")
    print("  sums of each other), so these tests are not independent and the")
    print("  expectation above is an approximation, not a bound.")
    return {"n_tests": n_tests, "observed": observed, "expected": expected,
            "bh": survived}


def report_no_usable_predictor(coefficients):
    """D6. The number that sizes where this approach does not work."""
    print("\n" + "=" * 78)
    print("D6  SITES WITH NO USABLE PREDICTOR")
    print("=" * 78)
    best = usable_predictors(coefficients)
    if not best.empty:
        total = len(best)
        beaten = int(best["beats_chance"].sum())
        print("\n  criterion that does not need a floor: does the site's")
        print("  STRONGEST correlation beat what the best of its own predictor")
        print("  count would give by chance at its own n?")
        print(f"    site-analyte pairs fitted      {total:,}")
        print(f"    beating chance                 {beaten:,} "
              f"({beaten / total:.1%})")
        print(f"    NOT beating chance             {total - beaten:,} "
              f"({1 - beaten / total:.1%})")
        print(f"    median chance threshold        "
              f"{best['chance_threshold'].median():.3f} |rho|")
    for floor in (config.USABLE_RHO_FLOOR, config.USABLE_RHO_FLOOR_LENIENT):
        best = usable_predictors(coefficients, floor)
        if best.empty:
            continue
        total = len(best)
        without = int((~best["has_usable"]).sum())
        print(f"\n  floor |rho_ctrl| >= {floor:.2f}")
        print(f"    site-analyte pairs fitted      {total:,}")
        print(f"    with NO predictor clearing it  {without:,} "
              f"({without / total:.1%})")
        by_analyte = (best.groupby("analyte")["has_usable"]
                      .agg(["size", "sum"]))
        by_analyte["none"] = by_analyte["size"] - by_analyte["sum"]
        by_analyte["share_none"] = (by_analyte["none"] / by_analyte["size"]).round(3)
        print(by_analyte[["size", "none", "share_none"]].to_string())
    print("\n  This is not a failure column. It is the addressable-site count")
    print("  from the other side: the share of beaches where a conditions-based")
    print("  prediction would not have worked however it was fitted.")
    print("\n  Read the chance criterion first. A fixed floor flatters a site")
    print("  with few samples, because the best of eleven predictors at n=40")
    print("  clears 0.30 by luck about as often as not.")
    return best


def per_site_table(coefficients, sites, nondetects, strata, column="rho_ctrl"):
    """D4. One row per site and analyte, every coefficient across the columns."""
    wide = coefficients.pivot_table(index=["station_id", "analyte"],
                                    columns="predictor", values=column,
                                    aggfunc="first")
    counts = coefficients.pivot_table(index=["station_id", "analyte"],
                                      columns="predictor", values="n_ctrl",
                                      aggfunc="first")
    counts.columns = [f"n_{c}" for c in counts.columns]
    separation = coefficients.pivot_table(index=["station_id", "analyte"],
                                          columns="predictor",
                                          values="auc_exceedance", aggfunc="first")
    separation.columns = [f"auc_{c}" for c in separation.columns]

    table = wide.join(counts).join(separation).reset_index()
    keep = ["station_id", "station_name", "state", "region"] + \
           [s for s in strata if s not in ("region",)] + \
           ["shore_normal_source", "tide_station", "join_resolution"]
    columns = [c for c in keep if c in sites.columns or c == "station_id"]
    table = table.merge(sites[[c for c in columns if c in sites.columns]],
                        on="station_id", how="left")
    if nondetects is not None and not nondetects.empty:
        table = table.merge(
            nondetects[["station_id", "analyte", "n", "nondetect_fraction",
                        "nondetect_flag", "n_methods", "mixed_estimators"]],
            on=["station_id", "analyte"], how="left")
    best = usable_predictors(coefficients)
    table = table.merge(best[["station_id", "analyte", "best_predictor",
                              "best_abs_rho", "has_usable"]],
                        on=["station_id", "analyte"], how="left")
    return table


def report_attrition(attrition):
    """C2. How many stations were pulled, how many cleared, and why not."""
    print("\n" + "=" * 78)
    print("C2  ATTRITION — which stations made it, and what stopped the rest")
    print("=" * 78)
    if attrition is None or attrition.empty:
        print("  no attrition table")
        return None
    counts = (attrition.groupby("outcome")
              .agg(site_analyte_pairs=("station_id", "size"),
                   stations=("station_id", "nunique"))
              .sort_values("site_analyte_pairs", ascending=False))
    print(counts.to_string())
    kept = attrition[attrition["outcome"].str.startswith("kept")]
    print(f"\n  {attrition['station_id'].nunique():,} stations pulled, "
          f"{kept['station_id'].nunique():,} contributed at least one "
          f"fitted analyte")
    if "region" in attrition.columns:
        by_region = (attrition.assign(kept=attrition["outcome"].str.startswith("kept"))
                     .groupby("region")["kept"].agg(["size", "sum"]))
        by_region.columns = ["pairs", "kept"]
        by_region["share"] = (by_region["kept"] / by_region["pairs"]).round(3)
        print("\nby region:")
        print(by_region.to_string())
    return counts


def report_nondetects(coefficients, nondetects):
    """B3. The flagged sites, reported separately rather than mixed in."""
    print("\n" + "=" * 78)
    print("B3  NON-DETECTS")
    print("=" * 78)
    if nondetects is None or nondetects.empty:
        print("  no non-detect table")
        return None
    fitted = set(zip(coefficients["station_id"].astype(str),
                     coefficients["analyte"]))
    subset = nondetects[[(str(r["station_id"]), r["analyte"]) in fitted
                         for _, r in nondetects.iterrows()]]
    if subset.empty:
        print("  no fitted site-analyte pair has a non-detect record")
        return None
    flagged = subset[subset["nondetect_flag"]]
    print(f"  fitted pairs                       {len(subset):,}")
    print(f"  median non-detect fraction         "
          f"{subset['nondetect_fraction'].median():.1%}")
    print(f"  over the {config.NONDETECT_FLAG_FRACTION:.0%} flag             "
          f"{len(flagged):,} ({len(flagged) / len(subset):.1%})")
    print("\n  Flagged pairs are EXCLUDED from the headline distribution below")
    print("  and reported on their own. Their coefficient rests mostly on the")
    print("  DL/2 substitution rather than on measured variation.")
    mixed = int(subset["mixed_estimators"].sum()) if "mixed_estimators" in subset else 0
    if mixed:
        print(f"\n  {mixed} pair(s) mix CFU and MPN results inside one site. "
              "Those are\n  two estimators, not one measurement — treat their "
              "coefficients as\n  provisional.")
    return subset


def run(coefficients, sites, attrition, nondetects, strata, out_dir=None,
        exploratory=()):
    """Everything in Part D, in order, written to disk as it is printed."""
    out_dir = out_dir or config.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    if exploratory:
        print("\n" + "!" * 78)
        print("THIS RUN INCLUDES EXPLORATORY GROUPINGS")
        print(f"  {', '.join(sorted(exploratory))}")
        print("  They were added after the registered pass. Everything they")
        print("  produce is exploratory and is labelled as such below; the")
        print("  pre-registered result is what the registered groupings say.")
        print("!" * 78)

    report_attrition(attrition)
    flagged = report_nondetects(coefficients, nondetects)

    headline = coefficients
    if nondetects is not None and not nondetects.empty:
        bad = {(str(r["station_id"]), r["analyte"])
               for _, r in nondetects.iterrows() if r["nondetect_flag"]}
        if bad:
            mask = [(str(s), a) not in bad
                    for s, a in zip(coefficients["station_id"],
                                    coefficients["analyte"])]
            headline = coefficients[mask]
            print(f"\n  headline distribution excludes {len(coefficients) - len(headline)}"
                  f" coefficient rows from non-detect-flagged pairs")

    table = report_distributions(headline)
    strata_table = report_by_stratum(headline, sites, strata,
                                     exploratory=exploratory)
    signs = report_sign_agreement(headline)
    report_multiple_testing(headline)
    report_no_usable_predictor(headline)
    sites_table = per_site_table(coefficients, sites, nondetects, strata)

    outputs = {
        "distribution.csv": table,
        "distribution_by_stratum.csv": strata_table,
        "sign_agreement.csv": signs,
        "per_site.csv": sites_table,
    }
    if flagged is not None:
        outputs["nondetect_flagged.csv"] = flagged
    print()
    for name, frame in outputs.items():
        if frame is None:
            continue
        path = os.path.join(out_dir, name)
        frame.to_csv(path, index=False)
        print(f"wrote {path}  ({len(frame)} rows)")

    print("\n" + "=" * 78)
    print("NOT FITTED IN THIS PASS")
    print("=" * 78)
    print("  No pooled model. No hierarchical model. No partial pooling.")
    print("  This pass establishes whether the coefficients vary and whether")
    print("  the pre-registered strata explain the variation. Partial pooling")
    print("  is the next step and should be informed by D1 and D2 above —")
    print("  in particular by how much of the spread survives stratification,")
    print("  which is what decides how much a hierarchical prior can borrow.")
    return outputs
