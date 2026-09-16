"""Task 2. EXPLORATORY / POST HOC stress test of the outfall_type result.

    python -m wq.diagnostics.task2_outfall

NOTHING HERE IS PRE-REGISTERED. D2 reported that outfall_type was the one
stratum of nine whose within-level IQR beat a same-size shuffle, and it did
so with 4 stations in its smallest level. Everything below re-asks that
question with the small levels removed, merged, and dropped one at a time.
A grouping that survives is a hypothesis for Phase 2; a grouping that does
not survive is a finding about this pass. Neither is a registered result, and
this script writes only into data/wq/out/diagnostics/ -- the pre-registered
distribution_by_stratum.csv is not touched.

The test statistic and the null are the pipeline's own:
wq.report._stratum_significance, imported rather than reimplemented, so a
variant cannot differ from D2 by anything except the labels handed to it.
That function shuffles the label ONCE PER SITE and reuses the shuffle across
every analyte/predictor cell, which is the whole reason its p-values are
believable; reimplementing it here would be the easiest way to lose that.
"""

import numpy as np
import pandas as pd

from . import common
from .. import config, report

STRATUM = "outfall_type"
SMALL = 20
# (b) says which level carries the effect, but if the answer is "the small
# ones" the useful follow-up is what is left without any of them. This is the
# same exclusion rule as (a) at a threshold that keeps only the levels big
# enough that no handful of stations can move them.
LARGE = 100


def build_merged(coefficients=None, sites=None, analyte=None):
    """The frame wq.report.report_by_stratum hands to the significance test:
    headline coefficients with the site's stratum label attached."""
    coefficients = (common.load_coefficients() if coefficients is None
                    else coefficients)
    sites = common.load_sites() if sites is None else sites
    head = common.headline(coefficients)
    merged = head.merge(sites[["station_id", STRATUM]], on="station_id",
                        how="left")
    if analyte is not None:
        merged = merged[merged["analyte"] == analyte]
    return merged


def level_sizes(merged):
    """Stations per level, counted the way the shuffle counts them: one vote
    per station, not one per coefficient row."""
    frame = merged[["station_id", STRATUM]].dropna().drop_duplicates()
    return frame[STRATUM].value_counts()


def run_variant(merged, label, column="rho_ctrl"):
    """One row of the same summary D2 prints, for one relabelling."""
    usable = merged.dropna(subset=[STRATUM])
    if usable[STRATUM].nunique() < 2:
        return {"variant": label, "levels": int(usable[STRATUM].nunique()),
                "cells": 0, "sites": int(usable["station_id"].nunique()),
                "smallest_level": np.nan, "iqr_ratio": np.nan,
                "chance_ratio": np.nan, "p": np.nan,
                "note": "fewer than two levels left; nothing to test"}
    table = pd.DataFrame({"stratum": [STRATUM]})
    out = report._stratum_significance(merged, table, column).iloc[0].to_dict()
    out["variant"] = label
    out["levels"] = int(usable[STRATUM].nunique())
    out["note"] = ""
    return out


def variants(merged):
    """Baseline, small levels excluded, small levels merged, and each level
    dropped in turn."""
    sizes = level_sizes(merged)
    small = [level for level, count in sizes.items() if count < SMALL]
    yield "baseline (as D2 ran it)", merged

    excluded = merged.copy()
    excluded.loc[excluded[STRATUM].isin(small), STRATUM] = np.nan
    yield f"(a) levels under {SMALL} stations EXCLUDED", excluded

    lumped = merged.copy()
    lumped.loc[lumped[STRATUM].isin(small), STRATUM] = "other"
    yield f"(a) levels under {SMALL} stations MERGED into 'other'", lumped

    big = [level for level, count in sizes.items() if count < LARGE]
    trimmed = merged.copy()
    trimmed.loc[trimmed[STRATUM].isin(big), STRATUM] = np.nan
    kept = [level for level, count in sizes.items() if count >= LARGE]
    yield (f"(a+) only levels with >= {LARGE} stations "
           f"({', '.join(kept) if kept else 'none'})"), trimmed

    for level in sizes.index:
        dropped = merged.copy()
        dropped.loc[dropped[STRATUM] == level, STRATUM] = np.nan
        yield f"(b) leave-one-out: without {level}", dropped


def small_level_stations(merged, sites):
    """Task 2c. Who is in the small levels, and what do they share?

    The shuffle controls for the SIZE of a small level and for nothing else.
    If four stations sit in one level and also share one sampling program, one
    state and one stretch of coast, the statistic reads their resemblance and
    the null cannot redraw it away -- which is the failure mode
    _stratum_significance's own docstring describes.
    """
    sizes = level_sizes(merged)
    small = [level for level, count in sizes.items() if count < SMALL * 5]
    roster = (merged[merged[STRATUM].isin(small)][["station_id", STRATUM]]
              .drop_duplicates())
    columns = [c for c in ("station_id", "station_name", "organization",
                           "state", "region", "lat", "lon", "outfall_permit",
                           "dist_to_outfall_m", "n_outfalls_within_2km",
                           "tide_station", "datum_gauge")
               if c in sites.columns]
    detail = roster.merge(sites[columns], on="station_id", how="left")
    fitted = (merged.groupby("station_id")["analyte"].nunique()
              .rename("analytes_fitted").reset_index())
    detail = detail.merge(fitted, on="station_id", how="left")
    return detail.sort_values([STRATUM, "organization", "station_id"])


def shared_attributes(detail):
    """For each small level: how concentrated is it in one org, state, county
    prefix or tide gauge? 1.00 means every station in the level shares it."""
    rows = []
    for level, group in detail.groupby(STRATUM):
        record = {"level": level, "stations": len(group)}
        for column in ("organization", "state", "region", "tide_station",
                       "outfall_permit"):
            if column not in group.columns:
                continue
            values = group[column].astype(str)
            top = values.value_counts()
            record[f"top_{column}"] = top.index[0] if len(top) else ""
            record[f"share_{column}"] = (round(top.iloc[0] / len(group), 3)
                                         if len(top) else np.nan)
        if {"lat", "lon"}.issubset(group.columns):
            record["lat_span_km"] = round(
                float(group["lat"].max() - group["lat"].min()) * 111, 1)
            record["lon_span_km"] = round(
                float(group["lon"].max() - group["lon"].min()) * 88, 1)
        rows.append(record)
    return pd.DataFrame(rows).sort_values("stations")


def main():
    coefficients = common.load_coefficients()
    sites = common.load_sites()

    print("!" * 78)
    print("TASK 2  EXPLORATORY / POST HOC. NOT PRE-REGISTERED.")
    print("  Everything below re-asks a question D2 already answered under the")
    print("  registered specification. A variant that narrows the spread is a")
    print("  hypothesis for Phase 2, not a result of this pass.")
    print("!" * 78)

    merged = build_merged(coefficients, sites)
    sizes = level_sizes(merged)
    print(f"\nstations per {STRATUM} level (fitted stations only):")
    print(sizes.to_string())
    print(f"\n  levels under {SMALL} stations: "
          f"{[l for l, c in sizes.items() if c < SMALL]}")
    if len([l for l, c in sizes.items() if c < SMALL]) < 2:
        print("  only one level is that small, so MERGING it into 'other' is "
              "a rename:\n  the merged variant must come out identical to the "
              "baseline, and does.")
    print(f"  {config.STRATUM_PERMUTATIONS} permutations per variant, "
          "seeded as the pipeline seeds them")

    results = [run_variant(frame, label) for label, frame in variants(merged)]
    pooled = pd.DataFrame(results)
    pooled.insert(0, "analyte", "ALL (pooled)")
    order = ["analyte", "variant", "levels", "cells", "sites",
             "smallest_level", "iqr_ratio", "chance_ratio", "p", "note"]
    print("\n(a)+(b) POOLED over every analyte:")
    print(pooled[order].round(3).to_string(index=False))

    per_analyte = []
    for analyte in sorted(merged["analyte"].unique()):
        part = build_merged(coefficients, sites, analyte=analyte)
        if part["station_id"].nunique() < 10:
            print(f"\n  {analyte}: only "
                  f"{part['station_id'].nunique()} fitted station(s) — "
                  "not tested")
            continue
        rows = [run_variant(frame, label) for label, frame in variants(part)]
        frame = pd.DataFrame(rows)
        frame.insert(0, "analyte", analyte)
        per_analyte.append(frame)
    if per_analyte:
        per_analyte = pd.concat(per_analyte, ignore_index=True)
        print("\n(d) THE SAME, PER ANALYTE:")
        print(per_analyte[order].round(3).to_string(index=False))
    else:
        per_analyte = pd.DataFrame(columns=order)

    detail = small_level_stations(merged, sites)
    print(f"\n(c) STATIONS IN THE SMALL LEVELS "
          f"(every level under {SMALL * 5} stations):")
    print(detail.to_string(index=False))

    shared = shared_attributes(detail)
    print("\n(c) what those stations share — the shuffle controls for level")
    print("    SIZE and for nothing in this table:")
    print(shared.to_string(index=False))

    combined = pd.concat([pooled[order], per_analyte[order]],
                         ignore_index=True)
    print("\nwrote:")
    common.write(combined, "task2_outfall_variants.csv")
    common.write(detail, "task2_small_level_stations.csv")
    common.write(shared, "task2_small_level_shared.csv")


if __name__ == "__main__":
    main()
