"""Task 0. Why coefficients.csv holds more rows than D5 counts tests.

    python -m wq.diagnostics.task0_reconcile

D5 prints "tests run 24,596" while coefficients.csv holds 32,857 rows, and a
gap that size is either an accounting detail or a silently dropped stratum of
the study. This script closes it exactly, using the pipeline's own rules:

  wq.report.run       drops every coefficient row belonging to a site-analyte
                      pair over NONDETECT_FLAG_FRACTION before D1-D6 run
  report_multiple_testing counts rows whose p_ctrl is not null

so the remainder is rows the fit emitted without computing a correlation, and
wq.report.predictor_coverage already names the three reasons that happens.
Every row must land in exactly one bucket and the buckets must sum to the
file, which is what this script asserts.
"""

import numpy as np
import pandas as pd

from . import common
from .. import config


def reconcile(coefficients, nondetects):
    """One row per bucket, in the order the pipeline applies them."""
    bad = common.flagged_pairs(nondetects)
    is_flagged = np.array([(str(s), a) in bad for s, a in
                           zip(coefficients["station_id"],
                               coefficients["analyte"])])
    head = coefficients[~is_flagged].copy()
    head["reason"] = common.untested_reason(head)

    rows = [{
        "bucket": "nondetect_flagged_excluded",
        "rows": int(is_flagged.sum()),
        "pairs": int(coefficients[is_flagged]
                     .groupby(["station_id", "analyte"]).ngroups),
        "stations": int(coefficients[is_flagged]["station_id"].nunique()),
        "where": "wq.report.run, before D1-D6",
        "meaning": (f"non-detect share over "
                    f"{config.NONDETECT_FLAG_FRACTION:.0%}; reported "
                    "separately in B3, never in the headline"),
    }]
    labels = {
        "never_measured": ("n_ctrl == 0: no paired record of this predictor "
                           "at this station at all"),
        "too_few_paired": (f"0 < n_ctrl < {config.MIN_PAIRED_N}: measured, "
                           "refused by the paired-n floor"),
        "no_variation": (f"n_ctrl >= {config.MIN_PAIRED_N} but one side never "
                         "moved, so a rank correlation is undefined"),
        "tested": "p_ctrl computed; this IS D5's test count",
    }
    for reason in ("never_measured", "too_few_paired", "no_variation",
                   "tested"):
        part = head[head["reason"] == reason]
        rows.append({
            "bucket": reason,
            "rows": len(part),
            "pairs": int(part.groupby(["station_id", "analyte"]).ngroups)
            if len(part) else 0,
            "stations": int(part["station_id"].nunique()) if len(part) else 0,
            "where": "wq.fit.fit_site_analyte" if reason != "tested"
            else "wq.report.report_multiple_testing",
            "meaning": labels[reason],
        })
    table = pd.DataFrame(rows)
    assert table["rows"].sum() == len(coefficients), (
        f"buckets sum to {table['rows'].sum()}, file holds "
        f"{len(coefficients)}")
    return table, head


def by_predictor(head):
    """Which predictors the untested rows belong to. A gap concentrated in one
    predictor is a coverage fact about that predictor, not about the study."""
    part = head[head["reason"] != "tested"]
    table = (pd.crosstab(part["predictor"], part["reason"])
             .reindex(columns=["never_measured", "too_few_paired",
                               "no_variation"], fill_value=0)
             .reset_index())
    fitted = (head[head["reason"] == "tested"].groupby("predictor")
              .size().rename("tested").reset_index())
    table = table.merge(fitted, on="predictor", how="outer").fillna(0)
    for column in table.columns[1:]:
        table[column] = table[column].astype(int)
    table["covered"] = (table["tested"] /
                        table[table.columns[1:5]].sum(axis=1)).round(3)
    return table.sort_values("covered")


def by_analyte(coefficients, attrition):
    """Per analyte, what reached the fit at all. ANALYTES is pre-registered,
    so an analyte with no fitted pair is a result, not an omission."""
    counts = (attrition.groupby(["analyte", "outcome"]).size()
              .unstack(fill_value=0).reset_index())
    fitted = (coefficients.groupby("analyte")
              .agg(coefficient_rows=("rho_ctrl", "size"),
                   pairs=("station_id", "nunique")).reset_index())
    return counts.merge(fitted, on="analyte", how="outer").fillna(0)


def main():
    parser = common.diagnostic_parser(__doc__)
    args = parser.parse_args()
    coefficients = common.load_coefficients()
    nondetects = common.load_nondetects()
    attrition = common.load_attrition()

    print("=" * 78)
    print("TASK 0  RECONCILING coefficients.csv AGAINST D5's TEST COUNT")
    print("=" * 78)
    coefficients = common.drop_hypothesis_sites(coefficients, args,
                                                "coefficient rows")
    coefficients = common.drop_held_out(coefficients, args,
                                        label="coefficient rows")
    table, head = reconcile(coefficients, nondetects)
    print(f"\ncoefficients.csv rows: {len(coefficients):,}\n")
    print(table[["bucket", "rows", "pairs", "stations", "meaning"]]
          .to_string(index=False))
    tested = int(table.loc[table["bucket"] == "tested", "rows"].iloc[0])
    print(f"\n  buckets sum to {table['rows'].sum():,} = the file. "
          f"D5's test count is the '{'tested'}' row: {tested:,}")

    print("\nuntested rows by predictor:")
    predictors = by_predictor(head)
    print(predictors.to_string(index=False))

    print("\nwhat reached the fit, per analyte:")
    analytes = by_analyte(coefficients, attrition)
    print(analytes.to_string(index=False))

    print("\nwrote:")
    common.write(table, "task0_reconciliation.csv")
    common.write(predictors, "task0_untested_by_predictor.csv")
    common.write(analytes, "task0_by_analyte.csv")


if __name__ == "__main__":
    main()
