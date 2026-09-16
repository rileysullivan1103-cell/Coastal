"""Does the RipAID correction also run BACKWARDS, into the Walton labels?

The calibration measured the labeller against trained annotators: 19% of their
rips caught strictly, 54% counting his own doubts, 96% agreement on negatives.
That was applied FORWARD, scaling Walton's 2.7% precision up to roughly 14%.

This asks the reverse question. If the same conservatism operated at Walton,
then some frames called N, D or U hold rips a stronger annotator would have
drawn — and the interesting quantity is whether there are ENOUGH of them.

THE CAPACITY CHECK IS THE POINT. At 19% recall, 6 found rips imply about 32
real ones among the 232 strictly-labelled fired frames, so about 26 missed. If
the notes flag fewer plausible candidates than that, the forward correction has
nowhere to put the missing rips, and either the recall does not transfer from
RipAID's cameras to Walton's or the misses are not where the notes suggest.
Either way the ~14% figure would be in trouble. That comparison is printed
whether it is comfortable or not.

TWO THINGS THIS CANNOT DO, BOTH BECAUSE THE DATA IS NOT THERE.

  * The cue rule is NOT derived from the calibration. The notes field was left
    empty on all 60 calibration frames, so there is no record of why anything
    was called and nothing to learn a vocabulary from. CUES below is a declared
    prior, printed in full with per-cue hit counts so it can be audited and
    argued with. It is not a finding.
  * The six Walton Y frames cannot be ranked against RipAID's confirmed rips.
    Nobody drew a box on a Walton frame; the only box there is the DETECTOR's,
    in pixels, and RipAID's are normalised — the two are not commensurable.
    What is reported instead is where those six sit in Walton's OWN box-size
    distribution, which is a within-camera comparison and therefore valid.

What IS derived from the calibration is the miss profile: whether the rips the
labeller missed were smaller, fewer or more doubted than the ones he caught,
measured on the ANNOTATOR's boxes. That needs no notes.

Read-only. Touches neither the Walton labels nor analyze_precision.py's output.

    python analyze_label_recheck.py --ripaid ~/Downloads/RipAID_v2.0.0_yolo-obb
"""

import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd

import analyze_precision as ap
import build_ripaid_calibration as brc
import load_ripaid as lr

REPORT = "data/label_recheck.txt"

# A DECLARED PRIOR, not a calibration output. Grouped so the report can say
# which kind of hedge fired, and printed with hit counts so a reader can strike
# any group they disagree with and re-run.
CUES = {
    "hedged": r"\b(maybe|possibl[ey]|perhaps|might|could be|not sure|unsure|"
              r"uncertain|unclear|ambiguous|hard to (say|tell)|difficult)\b",
    "weak": r"\b(faint|weak|subtle|slight|vague|hint|trace|barely|marginal)\b",
    "partial cue": r"\b(gap|one[- ]sided|single cue|only one|no foam|"
                   r"without foam|no seaward|narrow|thin streak)\b",
    "persistence": r"\b(persist\w*|transient|momentary|brief|check again|"
                   r"one frame)\b",
    "conditions": r"\b(glare|haz[ey]|fog\w*|backlit|shadow|low light|murky|"
                  r"blurr?\w*|dark)\b",
}
# Words that argue the opposite way. A note carrying one of these AND a cue is
# reported separately rather than counted as a candidate on the cue alone.
CONFIDENT = (r"\b(clear|obvious|strong|textbook|classic|definite|unmistakable|"
             r"well[- ]defined|pronounced)\b")

FIRED = ("low", "high")


# ---------------------------------------------------------------------------

def notes_of(frame):
    return frame.get("notes", pd.Series("", index=frame.index)).fillna("").astype(str)


def coverage(frame, label):
    """How many rows carry a note at all. Printed before anything rests on it."""
    text = notes_of(frame).str.strip()
    filled = int((text != "").sum())
    print(f"  {label}: {filled} of {len(frame)} rows carry a note "
          f"({filled / max(len(frame), 1):.0%})")
    return filled


def cues_in(text):
    """Which cue groups a note matches, and whether it also sounds confident."""
    lowered = str(text).lower()
    hits = {name for name, pattern in CUES.items()
            if re.search(pattern, lowered)}
    return hits, bool(re.search(CONFIDENT, lowered))


def mann_whitney(a, b):
    """(U, p) two-sided, normal approximation with a tie correction.

    Written out rather than imported because the repo carries no scipy
    dependency. At the n this runs on it is underpowered and says so; it is
    here to stop a difference in medians being read as more than it is.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    n1, n2 = len(a), len(b)
    if n1 < 1 or n2 < 1:
        return float("nan"), float("nan")
    combined = np.concatenate([a, b])
    order = combined.argsort()
    ranks = np.empty(len(combined), float)
    ranks[order] = np.arange(1, len(combined) + 1)
    # Average ranks within ties.
    _, inverse, counts = np.unique(combined, return_inverse=True,
                                   return_counts=True)
    for index, count in enumerate(counts):
        if count > 1:
            ranks[inverse == index] = ranks[inverse == index].mean()
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    mean = n1 * n2 / 2.0
    tie_term = sum(c ** 3 - c for c in counts)
    n = n1 + n2
    var = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return u, float("nan")
    # Clamped at zero. The continuity correction subtracts 0.5, so for two
    # groups that sit exactly on top of each other |U - mean| - 0.5 goes
    # NEGATIVE and erfc of a negative argument returns more than 1 -- a p-value
    # above 1, which is meaningless and would have been quoted.
    z = max(0.0, abs(u - mean) - 0.5) / math.sqrt(var)
    return u, min(1.0, math.erfc(z / math.sqrt(2)))


def describe(values, label):
    values = np.asarray([v for v in values if np.isfinite(v)], float)
    if not len(values):
        return f"  {label:<28} (none)"
    return (f"  {label:<28} n={len(values):>3}  median {np.median(values):.5f}"
            f"  IQR {np.percentile(values, 25):.5f}-"
            f"{np.percentile(values, 75):.5f}")


# ---------------------------------------------------------------------------
# step 1 — what the labeller misses, measured on the ANNOTATOR's boxes
# ---------------------------------------------------------------------------

def annotator_severity(truth, ripaid_root):
    """Join each calibration frame to the annotator's own box measurements."""
    if not ripaid_root:
        return truth.assign(area_max=np.nan, area_sum=np.nan)
    labels_dir = os.path.join(ripaid_root, "labels")
    if not os.path.isdir(labels_dir):
        labels_dir = ripaid_root
    rows = []
    for name in truth["file_name"]:
        stem = os.path.splitext(str(name))[0]
        path = os.path.join(labels_dir, stem + ".txt")
        if not os.path.isfile(path):
            rows.append({"area_max": np.nan, "area_sum": np.nan})
            continue
        boxes = lr.read_obb(path)
        areas = [lr.obb_area(points) for cls, points in boxes
                 if cls == lr.RIP_LABEL]
        rows.append({"area_max": max(areas) if areas else 0.0,
                     "area_sum": sum(areas)})
    return pd.concat([truth.reset_index(drop=True),
                      pd.DataFrame(rows)], axis=1)


def miss_profile(merged):
    """Are the annotator's rips the labeller MISSED smaller than the ones caught?"""
    print(f"\n{'-' * 78}\n(1) WHAT THE LABELLER MISSES, ON THE ANNOTATOR'S OWN "
          f"BOXES\n{'-' * 78}")
    judged = merged[merged["rip_present"].isin(ap.VERDICTS - {""})]
    judged = judged[judged["rip_present"] != "unusable"]
    theirs = judged[judged["truth"] == "yes"]
    if theirs.empty:
        print("  no annotated rips in the calibration set")
        return None

    caught = theirs[theirs["rip_present"] == "yes"]
    missed = theirs[theirs["rip_present"].isin(["no", "doubt"])]
    print(f"  the annotators drew a rip on {len(theirs)} judged frames: "
          f"{len(caught)} caught, {len(missed)} missed")
    print("\n  annotator's LARGEST drawn box, as a fraction of the frame:")
    print(describe(caught["area_max"], "frames he caught"))
    print(describe(missed["area_max"], "frames he missed"))
    _, p_area = mann_whitney(caught["area_max"], missed["area_max"])
    print(f"  Mann-Whitney p = {p_area:.3f}" if np.isfinite(p_area)
          else "  Mann-Whitney: not computable")

    print("\n  how many rips the annotator drew:")
    print(describe(caught["n_rip"], "frames he caught"))
    print(describe(missed["n_rip"], "frames he missed"))

    both = theirs["n_doubt"] > 0
    print(f"\n  annotator ALSO flagged doubt on: "
          f"{int((both & (theirs['rip_present'] == 'yes')).sum())} caught, "
          f"{int((both & theirs['rip_present'].isin(['no', 'doubt'])).sum())} "
          "missed")
    print("\n  Underpowered at these n — read the medians, and treat the p as a"
          "\n  brake on over-reading them rather than as a result.")
    return {"caught": caught, "missed": missed, "p_area": p_area}


# ---------------------------------------------------------------------------
# step 2 — recheck candidates among the Walton labels
# ---------------------------------------------------------------------------

def flag_walton(walton):
    print(f"\n{'-' * 78}\n(2) RECHECK CANDIDATES AMONG THE WALTON LABELS\n"
          f"{'-' * 78}")
    print("  The cue list is a DECLARED PRIOR, not a calibration output — the")
    print("  calibration notes were empty. Hit counts are printed so it can be")
    print("  argued with. Nothing here is relabelled.")

    work = walton.copy()
    work["notes_text"] = notes_of(work)
    flags, confident = [], []
    for text in work["notes_text"]:
        hits, sure = cues_in(text)
        flags.append(hits)
        confident.append(sure)
    work["cues"] = flags
    work["sounds_confident"] = confident

    under = work[work["rip_present"].isin(["no", "doubt", "unusable"])]
    with_note = under[under["notes_text"].str.strip() != ""]
    candidates = with_note[with_note["cues"].map(bool)
                           & ~with_note["sounds_confident"]]
    hedged_but_sure = with_note[with_note["cues"].map(bool)
                                & with_note["sounds_confident"]]

    print(f"\n  {len(under)} frames called N, D or U; "
          f"{len(with_note)} of them carry a note")
    if len(with_note) < len(under):
        print(f"  {len(under) - len(with_note)} have NO note and cannot be "
              "screened either way —\n  they are neither candidates nor "
              "cleared, and that gap is the ceiling\n  on what this report can "
              "say.")

    print(f"\n  {'original label':<18}{'frames':>8}{'with note':>11}"
          f"{'candidates':>12}")
    for verdict in ("no", "doubt", "unusable"):
        rows = under[under["rip_present"] == verdict]
        noted = rows[rows["notes_text"].str.strip() != ""]
        hits = candidates[candidates["rip_present"] == verdict]
        print(f"  {verdict:<18}{len(rows):>8}{len(noted):>11}{len(hits):>12}")
    print(f"  {'TOTAL':<18}{len(under):>8}{len(with_note):>11}"
          f"{len(candidates):>12}")

    print(f"\n  {'cue group':<16}{'pattern hits':>14}")
    for name in CUES:
        hits = int(candidates["cues"].map(lambda s, n=name: n in s).sum())
        print(f"  {name:<16}{hits:>14}")
    if len(hedged_but_sure):
        print(f"\n  {len(hedged_but_sure)} note(s) carry a cue AND a confident "
              "word — not counted:")
        for row in hedged_but_sure.head(5).itertuples():
            print(f"    {row.frame_id}: {row.notes_text.strip()[:70]}")
    return candidates, under, with_note


# ---------------------------------------------------------------------------
# step 3 — are the six Y calls on the strong end?
# ---------------------------------------------------------------------------

def rank_the_yes(walton):
    print(f"\n{'-' * 78}\n(3) WHERE THE Y CALLS SIT\n{'-' * 78}")
    print("  NOT against RipAID: nobody drew a box on a Walton frame, and the")
    print("  detector's boxes are pixels where RipAID's are normalised. This is")
    print("  a within-Walton comparison, which is the only valid one available.")
    yes = walton[walton["rip_present"] == "yes"]
    if yes.empty:
        print("\n  no Y-labelled frames")
        return
    fired = walton[walton["confidence"].isin(FIRED)]
    area = pd.to_numeric(fired.get("bbox_area_max"), errors="coerce").dropna()
    print(f"\n  {len(yes)} frame(s) called Y")
    if area.empty:
        print("  no bbox_area_max column; cannot place them")
    else:
        print(f"\n  {'frame':<12}{'detector box':>14}{'percentile':>12}"
              f"   notes")
        for row in yes.itertuples():
            value = pd.to_numeric(getattr(row, "bbox_area_max", np.nan),
                                  errors="coerce")
            if pd.isna(value):
                print(f"  {row.frame_id:<12}{'—':>14}{'—':>12}")
                continue
            pct = float((area < value).mean())
            note = str(getattr(row, "notes", "") or "").strip()
            print(f"  {row.frame_id:<12}{value:>14.0f}{pct:>11.0%}   "
                  f"{note[:44]}")
        median = float(area.median())
        above = int((pd.to_numeric(yes.get("bbox_area_max"),
                                   errors="coerce") > median).sum())
        print(f"\n  {above} of {len(yes)} sit above the median fired-frame box "
              f"({median:.0f} px).")
        # Branching on the number, not asserting a conclusion around it. The
        # first draft of this line said "argue the six are NOT inflated"
        # whatever the count was, which would have printed a conclusion the
        # data underneath it contradicted.
        share = above / len(yes)
        if share >= 0.5:
            print("  A majority sit above it, so the Y calls are if "
                  "anything on the STRONGER end and are not inflated.")
            print("  The reverse correction would then land on RECALL "
                  "(frames missed),")
            print("  not on precision.")
        else:
            print("  A minority sit above it, so the Y calls are on the "
                  "WEAKER end of what")
            print("  the detector boxed. That does not make them wrong, "
                  "but it removes the")
            print("  argument that precision is safe from the correction: "
                  "weak Y calls could")
            print("  be over-calls in the other direction.")
        print("  Detector box size is a proxy for nothing in particular "
              "here — it is the")
        print("  only within-camera severity signal on file, not a measure "
              "of how obvious")
        print("  the rip was.")

    hedge = sum(1 for row in yes.itertuples()
                if cues_in(str(getattr(row, "notes", "") or ""))[0])
    print(f"\n  {hedge} of {len(yes)} Y notes carry a hedging cue"
          + ("" if hedge else " — none reads as a reluctant call"))


# ---------------------------------------------------------------------------
# step 4 — what the candidates do to precision, and whether they can
# ---------------------------------------------------------------------------

def precision_scenarios(walton, candidates, recall_strict, recall_doubt):
    print(f"\n{'-' * 78}\n(4) PRECISION UNDER EACH READING, AND THE CAPACITY "
          f"CHECK\n{'-' * 78}")
    fired = walton[walton["confidence"].isin(FIRED)]
    labelled = fired[fired["rip_present"] != ""]
    strict = labelled[~labelled["rip_present"].isin(["doubt", "unusable"])]
    found = int((strict["rip_present"] == "yes").sum())
    n = len(strict)
    if not n:
        print("  no strictly-labelled fired frames")
        return
    # Only candidates where the DETECTOR FIRED bear on precision. One in a
    # 'none' stratum is a missed rip, which moves the false-omission rate.
    pool = candidates[candidates["confidence"].isin(FIRED)]
    quiet = len(candidates) - len(pool)
    print(f"  {found} rips found in {n} strictly-labelled fired frames "
          f"= {found / n:.2%}")
    print(f"  {len(pool)} recheck candidates sit in a fired stratum"
          + (f"; {quiet} sit in 'none' and bear on RECALL, not precision"
             if quiet else ""))

    print(f"\n  {'scenario':<44}{'rips':>7}{'precision':>12}")
    print(f"  {'none of the candidates is a rip':<44}{found:>7}"
          f"{found / n:>11.2%}")
    ceiling = found + len(pool)
    print(f"  {'every candidate is a rip':<44}{ceiling:>7}"
          f"{ceiling / max(n, 1):>11.2%}")
    for name, recall in (("strict, 19%", recall_strict),
                         ("doubt-as-yes, 54%", recall_doubt)):
        if not recall:
            continue
        implied = found / recall
        missed = implied - found
        reachable = min(missed, len(pool))
        total = found + reachable
        print(f"  {'calibration recall (' + name + ')':<44}{total:>7.1f}"
              f"{total / n:>11.2%}")

    print(f"\n  THE CAPACITY CHECK")
    for name, recall in (("strict 19%", recall_strict),
                         ("doubt-as-yes 54%", recall_doubt)):
        if not recall:
            continue
        implied = found / recall
        missed = implied - found
        verdict = ("the candidates CAN hold them"
                   if len(pool) >= missed else
                   "*** THE CANDIDATES CANNOT HOLD THEM ***")
        print(f"    at {name}: {found} found implies {implied:.1f} real, so "
              f"{missed:.1f} missed;\n      {len(pool)} candidates — {verdict}")
    print("\n  A shortfall here does not disprove the forward correction, but "
          "it means\n  the missed rips are NOT in the frames whose notes hedge "
          "— so either the\n  recall does not transfer from RipAID's cameras, "
          "or they were missed\n  without the labeller noticing enough to "
          "write anything.")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=ap.LABEL_CSV)
    parser.add_argument("--calibration", default=brc.LABEL_CSV)
    parser.add_argument("--truth", default=brc.TRUTH_CSV)
    parser.add_argument("--ripaid", default=None,
                        help="RipAID YOLO-OBB root, for the annotator's boxes")
    args = parser.parse_args()

    print(f"\n{'=' * 78}\nDOES THE CALIBRATION RUN BACKWARDS INTO THE WALTON "
          f"LABELS?\n{'=' * 78}")
    print("  Read-only. The Walton labels and analyze_precision.py are not "
          "touched.")

    if not os.path.exists(args.labels):
        sys.exit(f"{args.labels} does not exist.")
    walton = pd.read_csv(args.labels)
    walton["rip_present"] = (walton["rip_present"].fillna("")
                             .astype(str).str.strip().str.lower())
    print()
    coverage(walton, "Walton labels")

    recall_strict = recall_doubt = None
    merged = None
    if os.path.exists(args.calibration) and os.path.exists(args.truth):
        calibration = pd.read_csv(args.calibration)
        calibration["rip_present"] = (calibration["rip_present"].fillna("")
                                      .astype(str).str.strip().str.lower())
        coverage(calibration, "calibration labels")
        truth = pd.read_csv(args.truth)
        merged = truth.merge(
            calibration[["frame_id", "rip_present", "notes"]],
            on="frame_id", how="left")
        merged["rip_present"] = merged["rip_present"].fillna("")
        judged = merged[(merged["rip_present"] != "")
                        & (merged["rip_present"] != "unusable")]
        theirs = judged[judged["truth"] == "yes"]
        if len(theirs):
            recall_strict = float(
                ((judged["rip_present"] == "yes") & (judged["truth"] == "yes")).sum()
                / max(int(((judged["truth"] == "yes")
                           & (judged["rip_present"] != "doubt")).sum()), 1))
            recall_doubt = float(
                (judged["rip_present"].isin(["yes", "doubt"])
                 & (judged["truth"] == "yes")).sum() / len(theirs))
            print(f"\n  recall measured from THIS calibration file: "
                  f"{recall_strict:.0%} strict, {recall_doubt:.0%} "
                  "counting doubt")
    else:
        print("  calibration files not found; the recall used below falls back "
              "to the\n  published 19% / 54% and the miss profile is skipped")

    if merged is not None:
        miss_profile(annotator_severity(merged, args.ripaid))

    candidates, under, _ = flag_walton(walton)
    rank_the_yes(walton)
    precision_scenarios(walton, candidates,
                        recall_strict if recall_strict else 0.19,
                        recall_doubt if recall_doubt else 0.54)

    print(f"\n{'=' * 78}\nWHAT THIS IS AND IS NOT\n{'=' * 78}")
    print("  Nothing here is a relabelling. The candidates are frames whose own")
    print("  notes describe the kind of feature the calibration shows gets")
    print("  under-called; whether any of them holds a rip is unknown until")
    print("  someone looks again, and the cue list that found them is a prior.")
    print("  The capacity check is the load-bearing number: it says whether the")
    print("  forward correction's missing rips have anywhere to live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
