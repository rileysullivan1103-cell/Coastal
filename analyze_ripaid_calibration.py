"""Two questions RipAID can answer about the Walton results.

PART 1 -- CALIBRATION (OPEN #0). Riley's blind verdicts on 60 RipAID frames
against the annotators'. Cohen's kappa, agreement on each class, the confusion
matrix, and every frame he missed with what he wrote about it. This measures
the INSTRUMENT -- one person, one still -- not the detector.

PART 2 -- THE MIDDAY EFFECT (OPEN #1). Walton's detection rate carries a sun
term whose bearing rotation peaks at 185 deg: 5 deg from solar noon, 21 from
the camera. The open question is whether a HUMAN's rip calls do the same. If
they do, sun position changes what is really there (or really visible) and the
detector is tracking something; if they do not, the midday effect is the
detector.

TWO THINGS ABOUT PART 2, BOTH PRINTED BESIDE THE NUMBERS.

  * RIP PRESENCE IS NOT A VALID TARGET IN THIS DATASET, and that is not a new
    worry -- load_ripaid.summarize() has said so since it was written. Frames
    were pulled around lifeguard rip sightings +/- 2h and then "most of the
    images that did not show rip currents were removed". The negatives that
    remain are the residue of a hand deletion, not a control group. A rotation
    on presence is therefore run BECAUSE IT WAS ASKED FOR, reported with the
    selection diagnostics, and must not be quoted as evidence about the ocean.
  * WHAT SURVIVES THE SELECTION is the question asked WITHIN frames that
    contain an annotated rip: given that a rip is there and a person drew a box
    round it, does the drawn SIZE or the annotators' DOUBT track sun position?
    Those are properties of the rip and of seeing it, not of the curator's
    choice to keep the frame. That rotation is the one to read.

AND ONE PIECE OF LUCK. Solar noon at the Balearics is near 180 deg. Son Bou's
shore normal is 180, so there the camera and solar noon COINCIDE and no test
can separate them -- it is structurally incapable of answering this, whatever
its n. Cala Millor faces 90 deg, a 90 deg separation, against Walton's 26. If
any site in this project can tell a camera effect from a midday effect, it is
Cala Millor.

    python analyze_ripaid_calibration.py
    python analyze_ripaid_calibration.py --annotations instances_default.json
    python analyze_ripaid_calibration.py --part 2 --bearing-step 5
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

import analyze_drivers as ad
import analyze_glare_split as gs
import build_ripaid_calibration as brc
import load_ripaid as lr
import solar

VERDICTS = {"yes", "no", "doubt", "unusable"}

# Printed wherever the export is missing. Nothing in this project can reach
# Zenodo (see probe_rip_dataset.py), so "file not found" here is never a bug to
# fix in code -- it means the download has not happened, and the message has to
# say so rather than raising a bare FileNotFoundError.
ZENODO = """  RipAID is not in this repo and nothing here can fetch it — see
  probe_rip_dataset.py. Download the record from
  https://zenodo.org/records/15082427 and pass the COCO export:
    --annotations /path/to/instances_default.json
  To find it if it is already on disk somewhere:
    find ~ -name 'instances_default.json' -not -path '*/.*' 2>/dev/null"""

MIN_FRAMES = 120          # per camera, below which a rotation is not attempted
MIN_BIN = 20              # frames per azimuth bin, below which the bin is thin
BIN_DEG = 30


# ---------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------

def kappa(mine, theirs):
    """Cohen's kappa for two binary raters, plus the cells it is built from.

    Agreement alone is uninterpretable where the classes are unbalanced: on a
    set that is 90% negative, calling everything negative scores 90% and knows
    nothing. Kappa subtracts the agreement two raters would reach by chance
    given their own marginals, so it answers "better than guessing with this
    person's habits", which is the question here.
    """
    mine, theirs = list(mine), list(theirs)
    n = len(mine)
    if n == 0:
        return {"kappa": float("nan"), "n": 0, "po": float("nan"),
                "pe": float("nan"), "a": 0, "b": 0, "c": 0, "d": 0}
    a = sum(1 for m, t in zip(mine, theirs) if m and t)          # both yes
    b = sum(1 for m, t in zip(mine, theirs) if m and not t)      # mine only
    c = sum(1 for m, t in zip(mine, theirs) if not m and t)      # theirs only
    d = sum(1 for m, t in zip(mine, theirs) if not m and not t)  # both no
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    value = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"kappa": value, "n": n, "po": po, "pe": pe,
            "a": a, "b": b, "c": c, "d": d}


def kappa_interval(mine, theirs, draws=2000, seed=7):
    """Percentile bootstrap. At n=60 kappa is imprecise and should say so."""
    mine, theirs = np.asarray(mine, bool), np.asarray(theirs, bool)
    if len(mine) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        pick = rng.integers(0, len(mine), len(mine))
        result = kappa(mine[pick], theirs[pick])
        if not math.isnan(result["kappa"]):
            values.append(result["kappa"])
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def band(value):
    if math.isnan(value):
        return "undefined"
    for edge, name in ((0.0, "none"), (0.2, "slight"), (0.4, "fair"),
                       (0.6, "moderate"), (0.8, "substantial")):
        if value < edge + (0.2 if edge else 0.0):
            return name
    return "almost perfect"


def load_pair(labels_path, truth_path):
    if not os.path.exists(labels_path):
        return None, f"{labels_path} does not exist — run build_ripaid_calibration.py"
    if not os.path.exists(truth_path):
        return None, f"{truth_path} does not exist — the answer key was not written"
    labels = pd.read_csv(labels_path)
    truth = pd.read_csv(truth_path)
    merged = truth.merge(labels[["frame_id", "rip_present", "notes"]],
                         on="frame_id", how="left")
    merged["rip_present"] = (merged["rip_present"].fillna("")
                             .astype(str).str.strip().str.lower())
    merged["notes"] = merged["notes"].fillna("").astype(str)
    bad = set(merged["rip_present"]) - VERDICTS - {""}
    if bad:
        return None, f"unexpected verdicts in {labels_path}: {sorted(bad)}"
    return merged, None


def report_calibration(merged):
    total = len(merged)
    done = merged[merged["rip_present"] != ""]
    unusable = done[done["rip_present"] == "unusable"]
    judged = done[done["rip_present"] != "unusable"]

    print(f"\n{'=' * 78}\nPART 1 — CALIBRATION AGAINST THE RipAID ANNOTATORS\n{'=' * 78}")
    print(f"  {len(done)} of {total} frames labelled"
          + ("" if len(done) == total else "  — the rest are ignored, not counted as 'no'"))
    if len(unusable):
        print(f"  {len(unusable)} marked unusable and removed from every "
              "denominator:\n    a frame that cannot be judged carries no "
              "answer to compare")
    if judged.empty:
        print("\n  Nothing judgeable yet. Run:")
        print(f"    python label_server.py --dir {brc.OUT_DIR} "
              "--title \"Calibration — is there a rip?\"")
        return

    theirs = judged["truth"] == "yes"
    strict = judged[judged["rip_present"] != "doubt"]
    print(f"\n  RipAID says rip on {int(theirs.sum())} of {len(judged)} judged; "
          f"{int((judged['rip_present'] == 'doubt').sum())} of yours are doubt")

    for name, subset, mine in (
            ("doubt EXCLUDED", strict, strict["rip_present"] == "yes"),
            ("doubt AS YES", judged, judged["rip_present"].isin(["yes", "doubt"])),
            ("doubt AS NO", judged, judged["rip_present"] == "yes")):
        result = kappa(list(mine), list(subset["truth"] == "yes"))
        low, high = kappa_interval(list(mine), list(subset["truth"] == "yes"))
        print(f"\n--- {name} ---")
        print(f"  {'':<14}{'RipAID rip':>12}{'RipAID none':>13}")
        print(f"  {'you: rip':<14}{result['a']:>12}{result['b']:>13}")
        print(f"  {'you: none':<14}{result['c']:>12}{result['d']:>13}")
        positives = result["a"] + result["c"]
        negatives = result["b"] + result["d"]
        print(f"  agreement on RipAID's rips:  "
              f"{result['a']}/{positives}"
              + (f"  ({result['a'] / positives:.0%})" if positives else ""))
        print(f"  agreement on RipAID's nones: "
              f"{result['d']}/{negatives}"
              + (f"  ({result['d'] / negatives:.0%})" if negatives else ""))
        print(f"  raw agreement {result['po']:.0%}, chance {result['pe']:.0%}")
        print(f"  Cohen's kappa {result['kappa']:+.3f}  "
              f"[{low:+.3f}, {high:+.3f}]  ({band(result['kappa'])}, n={result['n']})")

    missed = judged[(judged["truth"] == "yes")
                    & (judged["rip_present"].isin(["no", "doubt"]))]
    print(f"\n--- what you missed: RipAID drew a rip, you did not call one ---")
    if missed.empty:
        print("  none")
    else:
        print(f"  {len(missed)} frame(s). The notes are the point: if they say "
              "the same thing,\n  the misses are one failure mode and not "
              f"{len(missed)} separate ones.\n")
        for row in missed.itertuples():
            print(f"  {row.frame_id}  you said {row.rip_present:<6} "
                  f"RipAID drew {row.n_rip}   {row.camera} "
                  f"{pd.Timestamp(row.timestamp):%Y-%m-%d %H:%M}")
            print(f"      {row.file_name}")
            if row.notes.strip():
                print(f"      notes: {row.notes.strip()}")

    false_alarms = judged[(judged["truth"] == "no")
                          & (judged["rip_present"] == "yes")]
    print(f"\n--- the other error: you called a rip, RipAID drew none ---")
    print(f"  {len(false_alarms)} frame(s)"
          + ("" if false_alarms.empty else ", which bound how much of Walton's "
             "2.7%\n  could be the labeller rather than the detector"))
    for row in false_alarms.itertuples():
        print(f"  {row.frame_id}  {row.camera} "
              f"{pd.Timestamp(row.timestamp):%Y-%m-%d %H:%M}"
              + (f"   notes: {row.notes.strip()}" if row.notes.strip() else ""))

    print("\n  HOW TO READ THIS. High agreement means a single still IS "
          "readable and\n  Walton's 2.7% is the detector's failure. Low "
          "agreement means the\n  labelling instrument is the limit, and no "
          "amount of it can test a\n  detector — the 2.7% would then be a "
          "statement about stills, not rips.")


# ---------------------------------------------------------------------------
# Part 2
# ---------------------------------------------------------------------------

def add_geometry(frames, lat, lon):
    work = frames.copy()
    work["hour"] = pd.to_datetime(work["timestamp"], utc=True).dt.floor("h")
    elevation, azimuth = solar.position(pd.DatetimeIndex(work["timestamp"]),
                                        lat, lon)
    work["solar_elevation"] = elevation
    work["solar_azimuth"] = azimuth
    work["hour_of_day"] = pd.DatetimeIndex(work["timestamp"]).hour
    month = pd.DatetimeIndex(work["timestamp"]).month
    # Season, as a pair so December and January are neighbours. Two columns
    # also keep the comparator model at two predictors minimum, below which
    # standardized_ols declines to fit at all.
    work["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    work["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    work["present"] = work["detected"].astype(float)
    return work


def selection_diagnostics(work, label):
    """What the curator's hand did to this frame set, as far as it is visible.

    The hour histogram answers POWER: an azimuth bin with four frames cannot
    contribute. The positive-rate column answers BIAS, which power cannot: if
    the share of negatives swings across bins, the deletion of negatives was
    itself time-dependent, and then any azimuth structure in presence is the
    curator's schedule rather than the ocean's.
    """
    print(f"\n  [{label}] frames by hour of day (UTC):")
    counts = work["hour_of_day"].value_counts().sort_index()
    print("    " + "  ".join(f"{h:02d}:{n}" for h, n in counts.items()))

    work = work.assign(bin=(work["solar_azimuth"] // BIN_DEG * BIN_DEG).astype(int))
    print(f"\n  [{label}] by {BIN_DEG} deg azimuth bin:")
    print(f"    {'bin':>6}{'frames':>9}{'with rip':>10}{'rate':>8}"
          f"{'neg per pos':>14}")
    thin = []
    rates = []
    for value, group in work.groupby("bin"):
        rips = int(group["detected"].sum())
        nones = len(group) - rips
        rate = rips / len(group)
        ratio = (nones / rips) if rips else float("nan")
        flag = "   <- THIN" if len(group) < MIN_BIN else ""
        if len(group) < MIN_BIN:
            thin.append(int(value))
        else:
            rates.append(rate)
        print(f"    {value:>6}{len(group):>9}{rips:>10}{rate:>8.2f}"
              f"{ratio:>14.2f}{flag}")
    if thin:
        print(f"    {len(thin)} bin(s) under {MIN_BIN} frames: {thin}. They "
              "carry the rotation\n    nowhere and their contribution to any "
              "peak is noise.")
    if len(rates) >= 2:
        spread = max(rates) - min(rates)
        print(f"\n    positive rate ranges {min(rates):.2f} to {max(rates):.2f} "
              f"across the usable bins (spread {spread:.2f}).")
        if spread > 0.2:
            print("    THAT SPREAD IS THE PROBLEM, not the finding. The "
                  "negatives here are\n    what survived a hand deletion, so a "
                  "bin's positive rate is the\n    curator's retention in that "
                  "bin as much as the ocean's behaviour.")
    return thin


def rotate(work, target, base, bearings, have_cloud=False):
    rows = []
    for candidate in bearings:
        gs.attach_geometry(work, float(candidate), suffix="_b")
        result = gs.bearing_test(work, target, base,
                                 ["sun_in_view_b", "sun_glare_b"], have_cloud)
        rows.append({"bearing": candidate,
                     "dR2": result["dR2"] if result else float("nan"),
                     "p": result["p"] if result else float("nan"),
                     "n": result["n"] if result else 0})
    return pd.DataFrame(rows)


def report_rotation(work, label, bearing, noon, targets, bearings):
    base = ["month_sin", "month_cos"]
    print(f"\n  {'target':<22}{'n':>6}{'peak':>6}{'peak dR2':>11}"
          f"{'p at peak':>12}{'at camera':>11}{'at noon':>10}   verdict")
    out = {}
    for target, note in targets:
        if target not in work.columns:
            continue
        usable = work.dropna(subset=[target])
        if len(usable) < MIN_FRAMES:
            print(f"  {target:<22}{len(usable):>6}   too few frames to fit")
            continue
        spread = float(pd.to_numeric(usable[target], errors="coerce").std() or 0.0)
        if spread <= 0:
            # Not a failure. A target every frame agrees on carries no
            # information, and saying "no fit converged" would suggest a
            # numerical problem where there is simply nothing to explain.
            print(f"  {target:<22}{len(usable):>6}   CONSTANT here "
                  f"(every frame the same value) — nothing to rotate against")
            continue
        grid = rotate(usable.copy(), target, base, bearings)
        if grid["dR2"].isna().all():
            print(f"  {target:<22}{len(usable):>6}   no fit converged")
            continue
        best = grid.loc[grid["dR2"].idxmax()]
        at_camera = (float(np.interp(bearing, grid["bearing"], grid["dR2"],
                                     period=360))
                     if bearing is not None else float("nan"))
        at_noon = float(np.interp(noon, grid["bearing"], grid["dR2"], period=360))
        call, _, _ = (gs.classify_peak(float(best["bearing"]), bearing, noon)
                      if bearing is not None
                      else ("no camera bearing", float("nan"), float("nan")))
        print(f"  {target:<22}{int(best['n']):>6}{int(best['bearing']):>6}"
              f"{best['dR2']:>+11.4f}{gs.fmt_p(best['p']):>12}"
              f"{at_camera:>+11.4f}{at_noon:>+10.4f}   {call}")
        if note:
            print(f"  {'':<22}{note}")
        out[target] = {"grid": grid, "peak": float(best["bearing"]),
                       "dR2": float(best["dR2"]), "p": float(best["p"]),
                       "call": call}
    return out


def report_part2(frames, args):
    print(f"\n{'=' * 78}\nPART 2 — DOES A HUMAN'S RIP CALL PEAK NEAR SOLAR NOON?"
          f"\n{'=' * 78}")
    print("  Same rotation analyze_glare_split.py runs on Walton, with a "
          "PERSON's\n  annotation as the dependent variable instead of a "
          "detector's score.")
    bearings = list(range(0, 360, max(1, args.bearing_step)))

    for site, group in frames.groupby("site"):
        meta = lr.KNOWN_SITES.get(site)
        if not meta:
            print(f"\n  {site}: no coordinates in load_ripaid.KNOWN_SITES; skipped")
            continue
        name = meta["name"]
        bearing = ad.SHORE_NORMAL_DEG.get(name)
        work = add_geometry(group, meta["latitude"], meta["longitude"])
        noon, days, exact = gs.solar_noon_bearing(work, meta["latitude"],
                                                  meta["longitude"])

        print(f"\n{'-' * 78}\n{name} ({site}) — {len(work)} frames, "
              f"{work['timestamp'].min():%Y-%m} to {work['timestamp'].max():%Y-%m}"
              f"\n{'-' * 78}")
        if noon is None:
            print("  no daylight frames; nothing to rotate")
            continue
        print(f"  solar noon sits at azimuth {noon:.1f} deg, from a minute "
              f"grid on {days} dates\n  drawn from this site's own record"
              + ("" if exact else "  (COARSE — lat/lon missing)"))
        if bearing is None:
            print("  NO SHORE NORMAL CONFIGURED for this site, so the peak can "
                  "be placed\n  against solar noon but not against the camera.")
        else:
            separation = gs.fold(bearing - noon)
            print(f"  shore normal {bearing:.0f} deg (ASSUMED, read off a map) "
                  f"— {separation:.0f} deg from solar noon")
            if separation < 2 * gs.MARGIN_DEG:
                print("  *** THIS SITE CANNOT ANSWER THE QUESTION ***")
                print("  The camera's bearing and solar noon are the same "
                      "direction here, so no\n  peak can be closer to one than "
                      "to the other. Its numbers are printed\n  for "
                      "completeness and settle nothing about camera versus "
                      "midday.")
            else:
                print(f"  A {separation:.0f} deg separation makes this site "
                      "MORE able to tell a camera\n  effect from a midday one "
                      "than Walton, where the two are 26 deg apart.")

        selection_diagnostics(work, name)

        print(f"\n  (a) RIP PRESENCE — asked for, and not a valid target here.")
        print("      The negatives are the residue of a hand deletion, so this "
              "measures\n      the curator at least as much as the ocean. "
              "Reported, not quotable.")
        report_rotation(work, name, bearing, noon,
                        [("present", "")], bearings)

        rips = work[work["n_rip"] > 0].copy()
        print(f"\n  (b) WITHIN THE {len(rips)} ANNOTATED RIPS — what the "
              "selection cannot reach.")
        print("      A rip is there and a person drew it. How big they drew it "
              "and how\n      often they hedged are properties of seeing it, "
              "not of keeping it.")
        if len(rips) < MIN_FRAMES:
            print(f"      only {len(rips)} annotated rips; too few to rotate")
            continue
        # Pixel areas are not comparable across cameras -- the README puts
        # cross-shore resolution between 0.2 and 15 m depending on camera -- so
        # the target is a within-camera z-score, never the raw area.
        rips["area_z"] = rips.groupby("camera")["area_max"].transform(
            lambda s: (s - s.mean()) / (s.std() or 1.0))
        rips["doubt_flag"] = (rips["n_doubt"] > 0).astype(float)
        rips["n_rip_drawn"] = rips["n_rip"].astype(float)
        report_rotation(rips, name, bearing, noon, [
            ("area_z", "drawn size, z-scored within camera"),
            ("doubt_flag", "did the annotators hedge on this frame"),
            ("n_rip_drawn", "how many rips they drew"),
        ], bearings)

    print(f"\n{'=' * 78}\nWHAT PART 2 CAN AND CANNOT SETTLE\n{'=' * 78}")
    print("  A peak near solar noon in (b) — drawn size or annotator doubt —")
    print("  means sun position changes what a PERSON sees in the water, and")
    print("  Walton's detector is tracking something real. A flat (b) with a")
    print("  peaked Walton means the midday effect is in the detector.")
    print("\n  (a) cannot contribute either way, whatever it shows. Nothing in")
    print("  this dataset makes rip PRESENCE a measurement rather than a")
    print("  curatorial choice.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=brc.LABEL_CSV)
    parser.add_argument("--truth", default=brc.TRUTH_CSV)
    parser.add_argument("--annotations", default=None,
                        help="RipAID input for part 2: a COCO export "
                             "(instances_default.json) or a YOLO-OBB dataset "
                             "root / labels directory")
    parser.add_argument("--part", type=int, choices=[1, 2], default=None,
                        help="run only one part (default: both)")
    parser.add_argument("--bearing-step", type=int, default=5)
    args = parser.parse_args()

    if args.part in (None, 1):
        merged, problem = load_pair(args.labels, args.truth)
        if problem:
            print(f"\n  PART 1 skipped: {problem}")
        else:
            report_calibration(merged)

    if args.part in (None, 2):
        path = args.annotations
        if path is None:
            guesses = [p for p in ("instances_default.json",
                                   "data/ripaid/instances_default.json",
                                   "ripaid/instances_default.json",
                                   "data/ripaid/labels", "ripaid/labels")
                       if os.path.exists(p)]
            path = guesses[0] if guesses else None
        if path is None:
            print("\n  PART 2 skipped: no RipAID export found.")
            print(ZENODO)
            return 0
        if not os.path.exists(path):
            # Checked here rather than left to load(), which raises a bare
            # FileNotFoundError that says nothing about where the file comes
            # from or that this project cannot download it.
            print(f"\n  PART 2 skipped: {path} does not exist.")
            print(ZENODO)
            return 0
        report_part2(lr.frames_from(path), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
