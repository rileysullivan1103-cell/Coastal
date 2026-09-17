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

import argparse

import numpy as np
import pandas as pd

from . import common
from .. import config, manifest, report

STRATUM = "outfall_type"
SMALL = 20
# (b) says which level carries the effect, but if the answer is "the small
# ones" the useful follow-up is what is left without any of them. This is the
# same exclusion rule as (a) at a threshold that keeps only the levels big
# enough that no handful of stations can move them.
LARGE = 100


def build_merged(coefficients=None, sites=None, analyte=None,
                 stratum=STRATUM):
    """The frame wq.report.report_by_stratum hands to the significance test:
    headline coefficients with the site's stratum label attached."""
    coefficients = (common.load_coefficients() if coefficients is None
                    else coefficients)
    sites = common.load_sites() if sites is None else sites
    head = common.headline(coefficients)
    merged = head.merge(sites[["station_id", stratum]], on="station_id",
                        how="left")
    if analyte is not None:
        merged = merged[merged["analyte"] == analyte]
    return merged


def level_sizes(merged, stratum=STRATUM):
    """Stations per level, counted the way the shuffle counts them: one vote
    per station, not one per coefficient row."""
    frame = merged[["station_id", stratum]].dropna().drop_duplicates()
    return frame[stratum].value_counts()


def run_variant(merged, label, column="rho_ctrl", stratum=STRATUM):
    """One row of the same summary D2 prints, for one relabelling."""
    usable = merged.dropna(subset=[stratum])
    if usable[stratum].nunique() < 2:
        return {"variant": label, "levels": int(usable[stratum].nunique()),
                "cells": 0, "sites": int(usable["station_id"].nunique()),
                "smallest_level": np.nan, "iqr_ratio": np.nan,
                "chance_ratio": np.nan, "p": np.nan,
                "note": "fewer than two levels left; nothing to test"}
    table = pd.DataFrame({"stratum": [stratum]})
    out = report._stratum_significance(merged, table, column).iloc[0].to_dict()
    out["variant"] = label
    out["levels"] = int(usable[stratum].nunique())
    out["note"] = ""
    return out


def variants(merged, stratum=STRATUM):
    """Baseline, small levels excluded, small levels merged, and each level
    dropped in turn."""
    sizes = level_sizes(merged, stratum)
    small = [level for level, count in sizes.items() if count < SMALL]
    yield "baseline (as D2 ran it)", merged

    excluded = merged.copy()
    excluded.loc[excluded[stratum].isin(small), stratum] = np.nan
    yield f"(a) levels under {SMALL} stations EXCLUDED", excluded

    lumped = merged.copy()
    lumped.loc[lumped[stratum].isin(small), stratum] = "other"
    yield f"(a) levels under {SMALL} stations MERGED into 'other'", lumped

    big = [level for level, count in sizes.items() if count < LARGE]
    trimmed = merged.copy()
    trimmed.loc[trimmed[stratum].isin(big), stratum] = np.nan
    kept = [level for level, count in sizes.items() if count >= LARGE]
    yield (f"(a+) only levels with >= {LARGE} stations "
           f"({', '.join(kept) if kept else 'none'})"), trimmed

    for level in sizes.index:
        dropped = merged.copy()
        dropped.loc[dropped[stratum] == level, stratum] = np.nan
        yield f"(b) leave-one-out: without {level}", dropped


def small_level_stations(merged, sites, stratum=STRATUM):
    """Task 2c. Who is in the small levels, and what do they share?

    The shuffle controls for the SIZE of a small level and for nothing else.
    If four stations sit in one level and also share one sampling program, one
    state and one stretch of coast, the statistic reads their resemblance and
    the null cannot redraw it away -- which is the failure mode
    _stratum_significance's own docstring describes.
    """
    sizes = level_sizes(merged, stratum)
    small = [level for level, count in sizes.items() if count < SMALL * 5]
    roster = (merged[merged[stratum].isin(small)][["station_id", stratum]]
              .drop_duplicates())
    # The stratum itself is already on `roster`. Taking it from `sites` too
    # makes the merge produce <stratum>_x and <stratum>_y, and every later
    # reference to it raises KeyError -- which is exactly what happened the
    # first time this ran on `region`, because region is in the list below and
    # outfall_type never was.
    columns = [c for c in ("station_id", "station_name", "organization",
                           "state", "region", "lat", "lon", "outfall_permit",
                           "dist_to_outfall_m", "n_outfalls_within_2km",
                           "tide_station", "datum_gauge")
               if c in sites.columns and c != stratum]
    detail = roster.merge(sites[columns], on="station_id", how="left")
    fitted = (merged.groupby("station_id")["analyte"].nunique()
              .rename("analytes_fitted").reset_index())
    detail = detail.merge(fitted, on="station_id", how="left")
    return detail.sort_values([stratum, "organization", "station_id"])


def shared_attributes(detail, stratum=STRATUM):
    """For each small level: how concentrated is it in one org, state, county
    prefix or tide gauge? 1.00 means every station in the level shares it."""
    rows = []
    for level, group in detail.groupby(stratum):
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
    parser = common.diagnostic_parser(__doc__)
    parser.add_argument("--stratum", default=STRATUM,
                        help="which pre-registered grouping to stress-test "
                             f"(default {STRATUM})")
    args = parser.parse_args()
    stratum = args.stratum

    coefficients = common.load_coefficients()
    sites = common.load_sites()
    if stratum not in sites.columns:
        raise SystemExit(
            f"{stratum} is not a column of stations_stratified.csv. "
            f"Groupings available: "
            f"{', '.join(c for c in sites.columns if c in config.STRATIFY_ON)}")
    registered = manifest.active_strata(manifest.require_manifest())
    if stratum not in registered:
        print(f"  NOTE: {stratum} is not in the manifest's active strata "
              f"({', '.join(registered)}).\n  It was either never registered "
              "as a grouping or dropped by the coverage rule, so D2 did not "
              "break\n  the distribution out by it and neither did the "
              "pre-registered pass.")

    print("!" * 78)
    print(f"TASK 2  EXPLORATORY / POST HOC on {stratum.upper()}. "
          "NOT PRE-REGISTERED.")
    print("  Everything below re-asks a question D2 already answered under the")
    print("  registered specification. A variant that narrows the spread is a")
    print("  hypothesis for Phase 2, not a result of this pass.")
    print("!" * 78)

    coefficients = common.apply_scope(coefficients, args, "coefficient rows")
    coefficients = common.drop_hypothesis_sites(coefficients, args,
                                                "coefficient rows")
    coefficients = common.drop_held_out(coefficients, args,
                                        label="coefficient rows")
    merged = build_merged(coefficients, sites, stratum=stratum)
    sizes = level_sizes(merged, stratum)
    print(f"\nstations per {stratum} level (fitted stations only):")
    print(sizes.to_string())
    print(f"\n  levels under {SMALL} stations: "
          f"{[l for l, c in sizes.items() if c < SMALL]}")
    if len([l for l, c in sizes.items() if c < SMALL]) < 2:
        print("  fewer than two levels are that small, so MERGING them into "
              "'other' is a\n  rename: the merged variant must come out "
              "identical to the baseline.")
    print(f"  {config.STRATUM_PERMUTATIONS} permutations per variant, "
          "seeded as the pipeline seeds them")

    results = [run_variant(frame, label, stratum=stratum)
               for label, frame in variants(merged, stratum)]
    pooled = pd.DataFrame(results)
    pooled.insert(0, "analyte", "ALL (pooled)")
    order = ["analyte", "variant", "levels", "cells", "sites",
             "smallest_level", "iqr_ratio", "chance_ratio", "p", "note"]
    print("\n(a)+(b) POOLED over every analyte:")
    print(pooled[order].round(3).to_string(index=False))

    per_analyte = []
    for analyte in sorted(merged["analyte"].unique()):
        part = build_merged(coefficients, sites, analyte=analyte,
                            stratum=stratum)
        if part["station_id"].nunique() < 10:
            print(f"\n  {analyte}: only "
                  f"{part['station_id'].nunique()} fitted station(s) — "
                  "not tested")
            continue
        rows = [run_variant(frame, label, stratum=stratum)
                for label, frame in variants(part, stratum)]
        frame = pd.DataFrame(rows)
        frame.insert(0, "analyte", analyte)
        per_analyte.append(frame)
    if per_analyte:
        per_analyte = pd.concat(per_analyte, ignore_index=True)
        print("\n(d) THE SAME, PER ANALYTE:")
        print(per_analyte[order].round(3).to_string(index=False))
    else:
        per_analyte = pd.DataFrame(columns=order)

    detail = small_level_stations(merged, sites, stratum)
    print(f"\n(c) STATIONS IN THE SMALL LEVELS "
          f"(every level under {SMALL * 5} stations):")
    print(detail.to_string(index=False) if not detail.empty
          else "    (no level is that small)")

    shared = shared_attributes(detail, stratum) if not detail.empty \
        else pd.DataFrame()
    if not shared.empty:
        print("\n(c) what those stations share — the shuffle controls for "
              "level\n    SIZE and for nothing in this table:")
        print(shared.to_string(index=False))

    combined = pd.concat([pooled[order], per_analyte[order]],
                         ignore_index=True)
    combined.insert(0, "stratum", stratum)
    print("\nwrote:")
    common.write(combined, f"task2_{stratum}_variants.csv")
    if not detail.empty:
        common.write(detail, f"task2_{stratum}_small_level_stations.csv")
        common.write(shared, f"task2_{stratum}_small_level_shared.csv")


if __name__ == "__main__":
    main()
