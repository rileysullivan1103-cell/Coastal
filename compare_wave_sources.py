"""Buoy against nearshore model, on the hours where both of them exist.

Deck slide 6 ("The prediction, and the test") sets buoy 46236 beside CDIP MOP
SC130 and reads the ratio between them as the size of the improvement: 2.0x on
box area, 3.6x on confidence, 2.3x on detections, 2.0x on rate. Its own
footnote is the reason to check that: "n = 2,548 for the model, 1,094 for the
buoy". The two columns were never computed on the same hours. The buoy reads
part of the record and the model reads all of it, so each ratio on that slide
mixes two changes at once -- a wave source 16x closer, and a sample more than
twice as large -- and cannot say which of them moved the number.

This holds the sample fixed. It restricts to the hours where BOTH sources
report a wave height and correlates each against the same targets there, so
the only thing differing between the two columns is the instrument. The
model-on-all-hours figure is printed underneath as the reference, because that
is the number the deck used, and the gap between it and the matched one is the
part of the ratio that was sample rather than distance.

    python compare_wave_sources.py
    python compare_wave_sources.py --camera Walton

Nothing here writes to disk and nothing in analyze_drivers.py changes. The
loading, the demeaning and the Spearman are that module's own functions,
imported rather than copied, so a later change to how the pipeline joins
conditions to hours reaches this comparison too.
"""

import argparse
import sys

import pandas as pd

import analyze_drivers as ad

# NDBC keeps its own column names through the loader; CDIP columns are
# prefixed by load_mop so they can never stand in for the Open-Meteo ones.
BUOY_COLUMN = "WVHT"
MOP_COLUMN = "mop_wave_height"

# The four the deck compared, in its order.
TARGETS = ["bbox_area_max", "score_max", "detections", "detection_rate"]

# What slide 6 published, so the run says plainly whether it held. Read off
# deck/build_deck.js: (buoy rho, mop rho, the slide's label).
DECK = {
    "bbox_area_max":  (0.151, 0.304, "largest detection box"),
    "score_max":      (0.033, 0.118, "detection confidence"),
    "detections":     (0.063, 0.145, "detections per hour"),
    "detection_rate": (0.039, 0.077, "detection rate"),
}


def controlled(frame, predictor, target):
    """(rho, n, p) with hour-of-day and month removed from both sides.

    The same two-way control analyze_drivers reports as rho_hrmo: demean each
    side by its own hour-by-month cell mean, then rank-correlate the
    residuals.
    """
    key = (frame["hour_of_day"].astype(str) + "-" + frame["month"].astype(str))
    return ad.spearman(ad.demean_by(frame[predictor], key),
                       ad.demean_by(frame[target], key))


def rows_for(frame, label, predictor, targets):
    """One row per target, tagged with which source and which hour set."""
    out = []
    for target in targets:
        if target not in frame.columns:
            continue
        rho, n, p = controlled(frame, predictor, target)
        out.append({"target": target, "source": label,
                    "n": n, "rho": rho, "p": p})
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="walton-lighthouse-santa-cruz-ca",
                        help="substring of the rip table to compare")
    args = parser.parse_args()

    sites = ad.load_sites()
    frame, name, has_coverage = ad.assemble_rip(sites, want=args.camera)
    if frame is None:
        return 1

    # Same denominator choice analyze_rip makes: with a coverage file every
    # hour the camera was up is in play, including the observed zeros. Without
    # one only hours that produced frames exist at all.
    observed = (frame.copy() if has_coverage
                else frame[frame.get("frames", 0) > 0].copy())
    observed["month"] = observed["hour"].dt.month

    missing = [c for c in (BUOY_COLUMN, MOP_COLUMN)
               if c not in observed.columns]
    if missing:
        print(f"\n{name}: no {' or '.join(missing)} column on disk.")
        print("  buoy needs pull_site_observations.py, MOP needs "
              "pull_cdip_mop.py --camera <name>.")
        return 1

    both = observed[observed[BUOY_COLUMN].notna()
                    & observed[MOP_COLUMN].notna()]
    mop_all = observed[observed[MOP_COLUMN].notna()]

    print(f"\n{'=' * 74}")
    print(f"WAVE SOURCE COMPARISON — {name}")
    print("=" * 74)
    print(f"  {len(observed)} hours analysed "
          f"({observed['hour'].min():%Y-%m-%d} to "
          f"{observed['hour'].max():%Y-%m-%d})")
    print(f"  buoy {BUOY_COLUMN}: {int(observed[BUOY_COLUMN].notna().sum())} hours")
    print(f"  {MOP_COLUMN}: {int(observed[MOP_COLUMN].notna().sum())} hours")
    print(f"  BOTH: {len(both)} hours  <- the matched set every 'matched' row below uses")

    if len(both) < ad.MIN_N:
        print(f"\n  only {len(both)} matched hours; under MIN_N={ad.MIN_N}. "
              "Nothing to compare.")
        return 1

    # Demeaned WITHIN the matched subset, not across the whole frame.
    # analyze_drivers demeans over everything it loaded and lets spearman drop
    # the NaN pairs afterwards, which is the right call when the question is
    # "what tracks this target" -- every predictor gets the most data it can.
    # It is the wrong call here. The buoy is absent for most hours, so a group
    # mean taken over the full frame is a mean over hours that are not in the
    # comparison, and the two columns would end up as residuals against two
    # different baselines. Subsetting first costs hours and buys the one thing
    # this script exists for: one row set, one set of cell means, and exactly
    # one difference between the two columns.
    records = []
    records += rows_for(both, f"buoy {BUOY_COLUMN} (matched)",
                        BUOY_COLUMN, TARGETS)
    records += rows_for(both, "CDIP MOP (matched)", MOP_COLUMN, TARGETS)
    records += rows_for(mop_all, "CDIP MOP (all hours)", MOP_COLUMN, TARGETS)

    table = pd.DataFrame(records)
    order = {t: i for i, t in enumerate(TARGETS)}
    table = table.sort_values(
        by=["target", "source"],
        key=lambda col: col.map(order) if col.name == "target" else col)

    print("\nSpearman rho, hour-of-day and month removed from both sides")
    print("(matched rows are demeaned within the matched hours, so the two "
          "sources\n share one baseline as well as one row set)\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(4).to_string(index=False))

    print(f"\n{'=' * 74}")
    print("DOES THE DECK'S RATIO HOLD ON MATCHED HOURS?")
    print("=" * 74)
    print(f"{'target':<22}{'deck':>14}{'matched':>14}   verdict")
    print("-" * 74)

    indexed = table.set_index(["target", "source"])
    for target in TARGETS:
        if target not in set(table["target"]):
            continue
        deck_buoy, deck_mop, label = DECK[target]
        deck_ratio = deck_mop / deck_buoy if deck_buoy else float("nan")

        buoy_rho = indexed.loc[(target, f"buoy {BUOY_COLUMN} (matched)"), "rho"]
        mop_rho = indexed.loc[(target, "CDIP MOP (matched)"), "rho"]
        buoy_p = indexed.loc[(target, f"buoy {BUOY_COLUMN} (matched)"), "p"]
        mop_p = indexed.loc[(target, "CDIP MOP (matched)"), "p"]

        # A ratio between two numbers that straddle zero, or where either end
        # is not distinguishable from zero, is arithmetic and not evidence.
        # The deck said as much about its own detection_rate row; say it here
        # wherever it applies rather than printing a number that reads as one.
        if pd.isna(buoy_rho) or pd.isna(mop_rho):
            verdict = "under-powered"
            shown = "   n/a"
        elif buoy_p > 0.05 or mop_p > 0.05:
            verdict = "one side n.s. — ratio is arithmetic"
            shown = f"{mop_rho / buoy_rho:.1f}x" if buoy_rho else "   n/a"
        elif buoy_rho * mop_rho <= 0:
            verdict = "opposite signs — ratio meaningless"
            shown = "   n/a"
        else:
            ratio = mop_rho / buoy_rho
            shown = f"{ratio:.1f}x"
            if ratio >= deck_ratio * 0.8:
                verdict = "holds"
            elif ratio >= 1.2:
                verdict = "smaller, model still ahead"
            elif ratio >= 0.8:
                verdict = "gone — sources comparable"
            else:
                verdict = "reversed — buoy ahead"

        print(f"{label:<22}{deck_ratio:>13.1f}x{shown:>14}   {verdict}")

    print("\nDeck rho for reference (different hours on each side):")
    for target in TARGETS:
        deck_buoy, deck_mop, label = DECK[target]
        print(f"  {label:<24} buoy {deck_buoy:.3f}  n=1094    "
              f"MOP {deck_mop:.3f}  n=2548")
    return 0


if __name__ == "__main__":
    sys.exit(main())
