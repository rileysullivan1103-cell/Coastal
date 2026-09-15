"""Walton drivers without the buoy, and what a year control does to them.

Two separate problems with the existing rip table, both of which this fixes.

The first is sample size. RIP_PREDICTORS carries five buoy-derived columns --
WVHT, DPD, APD, WTMP, and swell_onshore, which is computed from the buoy's
MWD -- and NDBC 46236 reports on 1,884 of Walton's 8,712 covered hours. The
table ranks every predictor by rho_hrmo regardless of its n, so a buoy column
measured on 22% of the record sorts against a model column measured on 98% of
it, and the ranking mixes "this driver is stronger" with "this driver was
measured on a kinder subset". Dropping the buoy columns entirely leaves a set
that is populated on essentially the whole span, so the ranking means one
thing again. Nothing is lost that compare_wave_sources.py has not already
measured properly, on matched hours.

The second is year. The record runs 2024-05 to 2026-09, so it spans three
calendar years, and the hour-by-month control cannot see them: demeaning by
"August" pools August 2024, August 2025 and August 2026 into one cell. If the
detector was retrained, if the camera was re-aimed, or if one winter simply
ran bigger than another, that difference sits inside the residuals and any
predictor with a matching drift picks it up. Year is added here two ways --
as dummies in the regression, and by splitting the correlations -- because
those answer different questions. The regression asks whether a predictor
survives a year offset. The split asks whether the relationship is the same
one in each year, which is the question a single pooled rho cannot answer no
matter how many controls are stacked on it.

    python analyze_year_controls.py
    python analyze_year_controls.py --camera Walton

Nothing here writes to disk and nothing in analyze_drivers.py changes. Its
loaders, its demeaning and its Spearman are imported rather than copied.
"""

import argparse
import sys

import numpy as np
import pandas as pd

import analyze_drivers as ad

# Exactly the set asked for: nearshore model waves, ERA5 wind, tide, rain.
# Every one of these is populated near the full span, which is the point --
# see the module docstring on why the buoy columns are gone.
PREDICTORS = [
    "mop_wave_height", "mop_wave_period",
    "wind_onshore", "wind_speed_10m",
    "level_m",
    "precipitation", "rain_24h_mm", "rain_48h_mm",
]

# Named so the run can say what it dropped and why, rather than leaving the
# reader to diff two predictor lists. swell_onshore is on this list because it
# is derived from the buoy's MWD (analyze_drivers.py:739), which is not
# obvious from the column name.
BUOY_DERIVED = ["WVHT", "DPD", "APD", "WTMP", "swell_onshore"]

# Below this a per-year cell is reported but not believed, and never used to
# call a sign change. ad.MIN_N is the same gate spearman applies.
YEAR_MIN_N = ad.MIN_N


def control_key(frame):
    """The hour-by-month cell label, computed on whatever rows are passed.

    Taking this on a subset rather than the full frame is deliberate and is
    the whole mechanism of the per-year split: see year_rows.
    """
    return (frame["hour_of_day"].astype(str) + "-"
            + frame["hour"].dt.month.astype(str))


def controlled(frame, predictor, target):
    """(rho, n, p) with hour-of-day and month removed from both sides."""
    key = control_key(frame)
    return ad.spearman(ad.demean_by(frame[predictor], key),
                       ad.demean_by(frame[target], key))


def year_rows(observed, targets, predictors):
    """Per-year rho_hrmo for every target and predictor.

    Each year is subset FIRST and demeaned within itself. Demeaning across the
    pooled frame and then splitting would defeat the exercise: the cell means
    would carry all three years, so a level difference between years would
    already be partly removed from the residuals before the split ever
    happened, and the split would report how alike the years are after having
    made them alike. Subsetting first costs the cross-year hours and buys a
    genuinely independent estimate per year.
    """
    out = []
    for year, block in observed.groupby(observed["hour"].dt.year):
        for target in targets:
            for predictor in predictors:
                if predictor not in block.columns:
                    continue
                rho, n, p = controlled(block, predictor, target)
                out.append({"target": target, "predictor": predictor,
                            "year": int(year), "rho": rho, "n": n, "p": p})
    return pd.DataFrame(out)


def sign_changes(per_year):
    """One row per (target, predictor) whose rho changes sign between years.

    Graded, because a sign flip is only news when both sides were measuring
    something. Two significant estimates of opposite sign is a real
    instability. A flip between two values that both sit inside the noise is
    just the noise, and flagging it identically would bury the first kind
    under the second.
    """
    flagged = []
    usable = per_year[(per_year["n"] >= YEAR_MIN_N) & per_year["rho"].notna()]
    for (target, predictor), block in usable.groupby(["target", "predictor"]):
        if len(block) < 2 or not (block["rho"].min() < 0 < block["rho"].max()):
            continue
        strong = block[block["p"] < 0.05]
        both_ways = (not strong.empty
                     and strong["rho"].min() < 0 < strong["rho"].max())
        flagged.append({
            "target": target, "predictor": predictor,
            "years": " ".join(f"{int(r.year)}:{r.rho:+.3f}"
                              f"{'*' if r.p < 0.05 else ''}"
                              for r in block.itertuples()),
            "verdict": ("REVERSAL — significant both ways" if both_ways
                        else "unstable — at most one year significant"),
        })
    table = pd.DataFrame(flagged)
    if table.empty:
        return table
    # Reversals first. On a real record the unstable pile is long -- every weak
    # predictor wobbles across zero somewhere -- and a genuine reversal sorted
    # alphabetically into the middle of it is a finding nobody reads.
    table["_reversal"] = ~table["verdict"].str.startswith("REVERSAL")
    table = table.sort_values(["_reversal", "target", "predictor"])
    return table.drop(columns="_reversal").reset_index(drop=True)


def complete_case_years(observed, target, predictors):
    """Hours per year that survive the regression's listwise deletion.

    standardized_ols drops any row missing ANY predictor, so one thin column
    can remove a whole year from the fit. When it does, that year's dummy is
    constant zero and gets dropped as having no variance -- the year term
    disappears from the output with nothing said. The reader then sees a fit
    that looks year-controlled and is silently missing a year. This reports
    the complete-case row count per year so that cannot happen quietly.
    """
    usable = [p for p in predictors if p in observed.columns]
    subset = (observed[["hour"] + [target] + usable]
              .apply(lambda c: c if c.name == "hour"
                     else pd.to_numeric(c, errors="coerce"))
              .dropna())
    return subset["hour"].dt.year.value_counts().sort_index()


def add_year_dummies(observed):
    """Dummy columns for every year after the first. Returns their names.

    Dummies rather than a single linear year term: with three calendar years a
    linear term forces 2025 to sit exactly halfway between 2024 and 2026, and
    if the change is a step -- a retrained detector, a moved camera -- that is
    the one shape it cannot fit. The cost is a coefficient per year instead of
    one, which at this n is affordable.
    """
    years = sorted(observed["hour"].dt.year.unique())
    names = []
    for year in years[1:]:
        column = f"year_{year}"
        observed[column] = (observed["hour"].dt.year == year).astype(float)
        names.append(column)
    return years[0], names


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="walton-lighthouse-santa-cruz-ca",
                        help="substring of the rip table to analyse")
    args = parser.parse_args()

    sites = ad.load_sites()
    frame, name, has_coverage = ad.assemble_rip(sites, want=args.camera)
    if frame is None:
        return 1

    # The same denominator analyze_rip picks: with a coverage file every hour
    # the camera was up counts, observed zeros included.
    observed = (frame.copy() if has_coverage
                else frame[frame.get("frames", 0) > 0].copy())
    observed["month"] = observed["hour"].dt.month
    observed["hr_mo"] = control_key(observed)

    present = [p for p in PREDICTORS if p in observed.columns]
    absent = [p for p in PREDICTORS if p not in observed.columns]
    dropped = [c for c in BUOY_DERIVED if c in observed.columns]

    print(f"\n{'=' * 78}")
    print(f"DRIVERS WITHOUT THE BUOY — {name}")
    print("=" * 78)
    print(f"  {len(observed)} hours ({observed['hour'].min():%Y-%m-%d} to "
          f"{observed['hour'].max():%Y-%m-%d})"
          f"{'' if has_coverage else '  [NO coverage file — no observed zeros]'}")
    if dropped:
        print(f"  buoy-derived columns excluded: {', '.join(dropped)}")
        counts = {c: int(observed[c].notna().sum()) for c in dropped}
        print(f"    (they cover {min(counts.values())}-{max(counts.values())} "
              f"of those hours; that is what was distorting the ranking)")
    if absent:
        print(f"  REQUESTED BUT NOT ON DISK: {', '.join(absent)}")
    if not present:
        print("  no requested predictor is present; nothing to analyse")
        return 1
    coverage = {p: int(pd.to_numeric(observed[p], errors="coerce").notna().sum())
                for p in present}
    print("\n  predictor coverage, of those hours:")
    for predictor, count in sorted(coverage.items(), key=lambda kv: -kv[1]):
        print(f"    {predictor:<20} {count:>6}  ({count / len(observed):.0%})")
    thin = [p for p, c in coverage.items() if c < len(observed) * 0.9]
    if thin:
        print(f"  NOTE: under 90% of hours — {', '.join(sorted(thin))}. "
              "Read their rho against their own n, not against the others'.")

    # Which months each year actually contains. 2024 starts in May and 2026
    # ends in September, so the three years are not the same slice of the
    # seasonal cycle, and a per-year difference can be composition rather than
    # change. Printed here so that reading is available before the split.
    print("\n  what each calendar year covers:")
    for year, block in observed.groupby(observed["hour"].dt.year):
        months = sorted(block["month"].unique())
        print(f"    {int(year)}: {len(block):>5} hours, months "
              f"{months[0]}-{months[-1]} ({len(months)} of 12)")
    print("  Years covering different months are not interchangeable: a "
          "predictor\n  that only matters in winter is weaker in a year with "
          "less winter in it.")

    targets = [t for t in ad.RIP_TARGETS
               if t in observed.columns
               and pd.to_numeric(observed[t], errors="coerce").nunique() >= 3]

    # ------------------------------------------------------------------
    # (0) The ranked list per target, on all 8,712 hours.
    # ------------------------------------------------------------------
    for target in targets:
        ad.report_correlations(
            observed, target, present,
            controls=[("hr", observed["hour_of_day"]),
                      ("mo", observed["month"]),
                      ("hrmo", observed["hr_mo"])],
            title=f"=== WHAT TRACKS {target} — no buoy columns ===\n"
                  "rho = raw; rho_hr = hour removed; rho_mo = month removed; "
                  "rho_hrmo = both. Ranked by rho_hrmo.")

    # ------------------------------------------------------------------
    # (a) Year as a term in the regression.
    # ------------------------------------------------------------------
    baseline, dummies = add_year_dummies(observed)
    print(f"\n{'=' * 78}")
    print("(a) YEAR AS A REGRESSION TERM")
    print("=" * 78)
    if not dummies:
        print("  only one calendar year present; no year term to add")
    else:
        print(f"  dummies {', '.join(dummies)}; {baseline} is the baseline.")
        print("  Each fit drops any hour missing ANY predictor, so its n is the")
        print("  intersection and is smaller than the correlation n above.")
        for target in targets:
            print(f"\n--- {target} ---")
            counts = complete_case_years(observed, target, present)
            spread = ", ".join(f"{int(y)}:{int(c)}" for y, c in counts.items())
            print(f"  complete-case hours by year: {spread or 'none'}")
            missing = [y for y in sorted(observed['hour'].dt.year.unique())
                       if int(y) not in set(int(i) for i in counts.index)]
            if missing:
                print(f"  WARNING: {', '.join(str(int(y)) for y in missing)} "
                      "contributes no complete row, so its dummy is constant")
                print("  and gets dropped. The fit below does NOT span that "
                      "year — a thin\n  predictor removed it. Compare the "
                      "per-year split in (b) instead.")
            print("  without a year term:")
            plain = ad.report_regression(observed, target, present)
            print("  with year dummies:")
            withyear = ad.report_regression(observed, target,
                                            present + dummies)
            if plain and withyear and plain["beta"] is not None \
                    and withyear["beta"] is not None:
                before = dict(zip(plain["names"], plain["beta"]))
                after = dict(zip(withyear["names"], withyear["beta"]))
                moved = [(k, before[k], after[k]) for k in before
                         if k in after
                         and (abs(after[k] - before[k]) > 0.02
                              or np.sign(after[k]) != np.sign(before[k]))]
                if moved:
                    print("  coefficients the year term moved:")
                    for key, b, a in sorted(moved,
                                            key=lambda r: -abs(r[2] - r[1])):
                        flip = "  SIGN FLIP" if np.sign(a) != np.sign(b) else ""
                        print(f"    {key:<20} {b:+.4f} -> {a:+.4f}{flip}")
                else:
                    print("  no coefficient moved by more than 0.02.")

    # ------------------------------------------------------------------
    # (b) The same correlations computed separately per year.
    # ------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("(b) THE SAME CORRELATIONS, ONE YEAR AT A TIME")
    print("=" * 78)
    print("  rho_hrmo within each year, demeaned inside that year only.")
    print(f"  Cells under n={YEAR_MIN_N} are blank: spearman will not report "
          "them.")
    per_year = year_rows(observed, targets, present)
    if per_year.empty:
        print("  no per-year cell had enough rows")
        return 0

    for target in targets:
        block = per_year[per_year["target"] == target]
        wide = block.pivot(index="predictor", columns="year",
                           values=["rho", "n", "p"])
        wide.columns = [f"{int(year)}_{stat}" for stat, year in wide.columns]
        order = [c for year in sorted(block["year"].unique())
                 for c in (f"{int(year)}_rho", f"{int(year)}_n",
                           f"{int(year)}_p")]
        wide = wide[order]
        # The pivot floats every column because a thin year leaves NaN in
        # rho/p. An hour count printed as 2150.0 reads as a measurement
        # rather than a count, so put the n columns back to integers.
        for column in wide.filter(like="_n").columns:
            wide[column] = wide[column].astype("Int64")
        wide = wide.reindex(
            wide.filter(like="_rho").abs().max(axis=1)
            .sort_values(ascending=False).index)
        print(f"\n--- {target} ---")
        with pd.option_context("display.width", 220, "display.max_columns", 40):
            print(wide.round(4).to_string())

    print(f"\n{'=' * 78}")
    print("PREDICTORS WHOSE SIGN CHANGES BETWEEN YEARS")
    print("=" * 78)
    flagged = sign_changes(per_year)
    if flagged.empty:
        print("  none. Every predictor keeps its sign in every year that had")
        print(f"  at least {YEAR_MIN_N} usable hours.")
    else:
        print("  * marks p < 0.05 in that year.\n")
        with pd.option_context("display.width", 220, "display.max_columns", 40):
            print(flagged.to_string(index=False))
        real = flagged[flagged["verdict"].str.startswith("REVERSAL")]
        print(f"\n  {len(flagged)} sign changes, {len(real)} of them "
              "significant in both directions.")
        if len(real):
            print("  A significant reversal means the pooled rho for that")
            print("  predictor is an average of two opposite relationships and")
            print("  describes neither year. Do not quote the pooled number.")
        if len(flagged) - len(real):
            print("  The rest flip between estimates that are individually")
            print("  indistinguishable from zero. That is the predictor being")
            print("  weak, not the relationship inverting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
