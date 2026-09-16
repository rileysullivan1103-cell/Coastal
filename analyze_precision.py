"""Detector precision by stratum, once labels.csv has been filled in.

Three things this has to get right, because getting any of them wrong turns a
labelling week into a wrong number stated confidently:

1. PRECISION IS NOT DEFINED WHERE THE DETECTOR SAID NOTHING. The 'none' cells
   have no detections to be right or wrong about. What they measure is the
   opposite error -- rips the detector missed -- so they are reported as a false
   omission rate under their own heading, never folded into a precision column.

2. THE SAMPLE IS NOT THE POPULATION. High-confidence frames and the two booster
   conditions were deliberately oversampled, so pooling the sample's hits and
   misses answers a question about the sample. The overall figure here is
   reweighted by how common each stratum actually is in the record; the
   unweighted number is printed beside it so the gap is visible rather than
   hidden.

3. DOUBT IS NOT A NO. Frames labelled 'doubt' are excluded from the strict
   estimate and counted as hits in a second, so the pair brackets the answer.
   A cell where the two diverge is a cell where the labelling, not the
   detector, is the limiting factor.

Intervals are Wilson, not normal-approximation: at n=30 with p near 1 the
normal interval runs past 100% and understates the uncertainty.

    python analyze_precision.py
    python analyze_precision.py --min-n 10
"""

import argparse
import math
import os
import sys

import pandas as pd

LABEL_CSV = "data/label_sample/labels.csv"
STRATA_CSV = "data/label_sample/strata.csv"
DEFAULT_MIN_N = 8
Z = 1.959963985  # 95%


def wilson(hits, n, z=Z):
    """(low, high) for a proportion. Returns (nan, nan) for an empty cell."""
    if n <= 0:
        return float("nan"), float("nan")
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt_rate(hits, n, low, high):
    if n <= 0:
        return "     —"
    return f"{hits / n:6.1%}  [{low:5.1%}, {high:5.1%}]"


VERDICTS = {"yes", "no", "doubt", "unusable", ""}


def load_labels(path):
    frame = pd.read_csv(path)
    for column in ("rip_present", "stratum", "confidence", "booster"):
        if column not in frame.columns:
            sys.exit(f"{path} has no {column!r} column; was it written by "
                     "build_label_sample.py?")
    frame["rip_present"] = frame["rip_present"].fillna("").astype(str).str.strip().str.lower()
    frame["booster"] = frame["booster"].fillna("").astype(str)
    bad = set(frame["rip_present"]) - VERDICTS
    if bad:
        sys.exit(f"unexpected verdicts in {path}: {sorted(bad)}")

    # UNUSABLE IS NOT A FOURTH SHADE OF DOUBT. Doubt means the image is
    # readable and the answer is genuinely unclear, so it belongs inside the
    # bracket: counted as no in one estimate and as yes in the other. Unusable
    # means the frame cannot be judged at all -- lens water, total dark, a test
    # card -- and there is no answer in it to bracket. Those frames leave the
    # denominator entirely, the way a broken instrument reading does.
    unusable = int((frame["rip_present"] == "unusable").sum())
    frame = frame[frame["rip_present"] != "unusable"].copy()
    return frame, unusable


def precision_rows(frame, min_n):
    """One row per stratum where the detector actually fired."""
    fired = frame[frame["confidence"].isin(["low", "high"])]
    out = []
    for name, group in fired.groupby("stratum"):
        labelled = group[group["rip_present"] != ""]
        strict = labelled[labelled["rip_present"] != "doubt"]
        yes = int((strict["rip_present"] == "yes").sum())
        n = len(strict)
        doubt = int((labelled["rip_present"] == "doubt").sum())
        low, high = wilson(yes, n)
        lo_u, hi_u = wilson(yes + doubt, n + doubt)
        out.append({
            "stratum": name, "drawn": len(group), "labelled": len(labelled),
            "n_strict": n, "yes": yes, "doubt": doubt,
            "precision": yes / n if n else float("nan"),
            "lo": low, "hi": high,
            "precision_doubt_as_yes": (yes + doubt) / (n + doubt) if (n + doubt) else float("nan"),
            "lo_u": lo_u, "hi_u": hi_u,
            "thin": n < min_n,
        })
    return pd.DataFrame(out).sort_values("stratum")


def omission_rows(frame, min_n):
    """One row per 'none' stratum: rips present where nothing was detected."""
    quiet = frame[frame["confidence"] == "none"]
    out = []
    for name, group in quiet.groupby("stratum"):
        labelled = group[group["rip_present"] != ""]
        strict = labelled[labelled["rip_present"] != "doubt"]
        yes = int((strict["rip_present"] == "yes").sum())
        n = len(strict)
        low, high = wilson(yes, n)
        out.append({"stratum": name, "drawn": len(group), "labelled": len(labelled),
                    "n_strict": n, "missed": yes, "rate": yes / n if n else float("nan"),
                    "lo": low, "hi": high,
                    "doubt": int((labelled["rip_present"] == "doubt").sum()),
                    "thin": n < min_n})
    return pd.DataFrame(out).sort_values("stratum")


def weighted_precision(rows, weights):
    """Population-weighted precision over the cells where the detector fired.

    Without this the headline is the precision of the SAMPLE, which was built
    to over-represent exactly the cells that behave differently. Cells with no
    weight on file, or no strict labels, drop out and are reported as dropped.
    """
    usable = rows[(rows["n_strict"] > 0) & rows["stratum"].isin(weights.index)]
    if usable.empty:
        return float("nan"), 0, len(rows)
    w = weights.reindex(usable["stratum"]).astype(float).to_numpy()
    if w.sum() <= 0:
        return float("nan"), 0, len(rows)
    w = w / w.sum()
    return float((usable["precision"].to_numpy() * w).sum()), len(usable), \
        len(rows) - len(usable)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=LABEL_CSV)
    parser.add_argument("--strata", default=STRATA_CSV)
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N,
                        help=f"cells with fewer strict labels are flagged (default {DEFAULT_MIN_N})")
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        sys.exit(f"{args.labels} does not exist — run build_label_sample.py first.")
    frame, unusable = load_labels(args.labels)

    total = len(frame)
    done = int((frame["rip_present"] != "").sum())
    print(f"\n{'=' * 74}")
    print("DETECTOR PRECISION BY STRATUM")
    print("=" * 74)
    print(f"  {done} of {total} judgeable rows labelled"
          + ("" if done == total else "  — the rest are ignored, not counted as 'no'"))
    if unusable:
        print(f"  {unusable} row(s) marked unusable and removed from every "
              "denominator:\n    a frame that cannot be judged carries no "
              "answer to bracket, unlike doubt")
    if done == 0:
        print("\n  Nothing labelled yet. Run: python label_server.py")
        return 0
    counts = frame[frame["rip_present"] != ""]["rip_present"].value_counts()
    print("  verdicts: " + ", ".join(f"{k} {v}" for k, v in counts.items()))

    rows = precision_rows(frame, args.min_n)
    if rows.empty:
        print("\n  No labelled rows in a cell where the detector fired.")
    else:
        print(f"\n--- precision, where the detector DID fire ---")
        print("  of the frames it flagged, how many hold a real rip.")
        print(f"\n  {'stratum':<22}{'n':>4}{'yes':>5}{'dbt':>5}   "
              f"{'precision (doubt excluded)':<28}{'doubt as yes':>14}")
        for _, r in rows.iterrows():
            flag = "  <- thin" if r["thin"] else ""
            print(f"  {r['stratum']:<22}{r['n_strict']:>4}{r['yes']:>5}{r['doubt']:>5}   "
                  f"{fmt_rate(r['yes'], r['n_strict'], r['lo'], r['hi']):<28}"
                  f"{(r['precision_doubt_as_yes'] if r['n_strict'] + r['doubt'] else float('nan')):>13.1%}"
                  f"{flag}")
        if rows["thin"].any():
            print(f"\n  'thin' marks fewer than {args.min_n} strict labels. The interval is "
                  "printed\n  but should not be compared against another cell.")

    quiet = omission_rows(frame, args.min_n)
    if not quiet.empty:
        print(f"\n--- what the detector MISSED, where it did not fire ---")
        print("  NOT precision: these hours have no detections to be right about.")
        print(f"\n  {'stratum':<22}{'n':>4}{'rip':>5}{'dbt':>5}   {'rips present anyway':<28}")
        for _, r in quiet.iterrows():
            flag = "  <- thin" if r["thin"] else ""
            print(f"  {r['stratum']:<22}{r['n_strict']:>4}{r['missed']:>5}{r['doubt']:>5}   "
                  f"{fmt_rate(r['missed'], r['n_strict'], r['lo'], r['hi']):<28}{flag}")

    boosted = frame[frame["booster"] != ""]
    if not boosted.empty:
        print(f"\n--- the booster draws, on their own ---")
        print("  drawn deliberately from the conditions most likely to fool a")
        print("  vision model, so they are NOT representative of the record.")
        for name, group in boosted.groupby("booster"):
            fired = group[group["confidence"].isin(["low", "high"])]
            strict = fired[~fired["rip_present"].isin(["", "doubt"])]
            yes, n = int((strict["rip_present"] == "yes").sum()), len(strict)
            low, high = wilson(yes, n)
            print(f"  {name:<22}{n:>4}{yes:>5}        "
                  f"{fmt_rate(yes, n, low, high)}")

    if not rows.empty:
        print(f"\n--- overall ---")
        unweighted_n = int(rows["n_strict"].sum())
        unweighted = (int(rows["yes"].sum()) / unweighted_n) if unweighted_n else float("nan")
        weights = None
        if os.path.exists(args.strata):
            table = pd.read_csv(args.strata)
            if {"stratum", "population"} <= set(table.columns):
                weights = table.set_index("stratum")["population"]
        if weights is None:
            print(f"  {args.strata} is missing, so the sample cannot be reweighted.")
            print(f"  Sample precision (NOT the record's): {unweighted:.1%} "
                  f"over {unweighted_n} strict labels.")
            print("  Re-run build_label_sample.py to write the stratum populations.")
        else:
            value, used, dropped = weighted_precision(rows, weights)
            print(f"  population-weighted precision: {value:6.1%}   <- quote this one")
            print(f"  unweighted sample precision:   {unweighted:6.1%}   "
                  f"({unweighted_n} strict labels)")
            print(f"  weighted over {used} cell(s)"
                  + (f", {dropped} dropped for having no weight or no labels" if dropped else ""))
            if pd.notna(value) and abs(value - unweighted) > 0.05:
                print("\n  The two differ by more than 5 points. That gap IS the stratification:")
                print("  the sample over-represents the cells that behave differently, which is")
                print("  what it was built to do. The weighted figure is the one about the record.")

    print(f"\n  Caveat that does not go away: the label is one person's read of a")
    print("  still, not a verified rip. It measures agreement with Riley, which is")
    print("  the best ground truth this project has and is not the same as truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
