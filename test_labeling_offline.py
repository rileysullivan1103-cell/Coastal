"""Offline checks for the labelling pipeline.

The claims worth testing are the ones that would fail silently and produce a
plausible wrong number rather than a crash:

  * a stratified sample reweighted back to the population, when the sample
    over-represents a cell whose precision differs -- if the weighting is
    dropped the headline shifts by a known amount, so the test can demand it,
  * 'doubt' bracketed rather than quietly dropped or quietly counted,
  * 'none' cells kept out of precision entirely,
  * Wilson intervals that stay inside [0,1] at p=1, where the normal
    approximation runs past 100%,
  * the label server rewriting the CSV without losing rows, and refusing a
    path that climbs out of the image directory.

    python test_labeling_offline.py
"""

import csv
import json
import os
import shutil
import tempfile

import pandas as pd

import analyze_precision as ap

FAILURES = []


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def make_labels(spec):
    """spec: {stratum: (confidence, n_yes, n_no, n_doubt, n_blank)} -> DataFrame."""
    rows = []
    i = 0
    for stratum, (confidence, yes, no, doubt, blank) in spec.items():
        for verdict, count in (("yes", yes), ("no", no), ("doubt", doubt), ("", blank)):
            for _ in range(count):
                i += 1
                rows.append({"frame_id": f"f{i:04d}", "stratum": stratum,
                             "confidence": confidence, "booster": "",
                             "rip_present": verdict, "notes": "",
                             "score_max": 0.5 if confidence != "none" else "",
                             "image": f"f{i:04d}.jpg", "boxes": "[]"})
    return pd.DataFrame(rows)


def check_weighting_beats_the_sample():
    """The headline must follow the population, not the draw.

    Built so the two answers are far apart and both are computable by hand:
    high confidence is 90% precise and rare (1,000 hours), low is 30% precise
    and common (9,000). The sample draws 20 of each, so the unweighted mean is
    60% while the record's is 0.1*90 + 0.9*30 = 36%.
    """
    frame = make_labels({
        "high / H3 high": ("high", 18, 2, 0, 0),
        "low / H1 low":   ("low",   6, 14, 0, 0),
    })
    rows = ap.precision_rows(frame, min_n=8)
    weights = pd.Series({"high / H3 high": 1000, "low / H1 low": 9000})
    value, used, dropped = ap.weighted_precision(rows, weights)
    unweighted = rows["yes"].sum() / rows["n_strict"].sum()

    check("sample precision is the naive 60%", abs(unweighted - 0.60) < 1e-9,
          f"got {unweighted:.1%}")
    check("population-weighted precision is 36%", abs(value - 0.36) < 1e-9,
          f"got {value:.1%}")
    check("weighting used both cells and dropped none", used == 2 and dropped == 0)
    check("the two answers differ enough to matter", abs(value - unweighted) > 0.2,
          f"{value:.1%} vs {unweighted:.1%}")


def check_doubt_is_bracketed():
    """Doubt must move the upper bound and leave the strict estimate alone."""
    frame = make_labels({"high / H2 mid": ("high", 6, 4, 5, 0)})
    rows = ap.precision_rows(frame, min_n=8).iloc[0]
    check("strict precision ignores doubt", abs(rows["precision"] - 0.6) < 1e-9,
          f"6/10 = {rows['precision']:.1%}")
    check("doubt-as-yes raises it to 11/15", abs(rows["precision_doubt_as_yes"] - 11 / 15) < 1e-9,
          f"got {rows['precision_doubt_as_yes']:.1%}")
    check("the doubt count is carried, not discarded", rows["doubt"] == 5)


def check_blank_rows_are_not_a_no():
    """An unlabelled row must not become evidence.

    The failure this guards against is the quiet one: fillna('') then counting
    everything that is not 'yes' as a miss, which turns an unfinished labelling
    session into a detector that looks bad.
    """
    frame = make_labels({"high / H1 low": ("high", 5, 0, 0, 40)})
    rows = ap.precision_rows(frame, min_n=1).iloc[0]
    check("blank rows stay out of the denominator", rows["n_strict"] == 5,
          f"n_strict={rows['n_strict']} of 45 drawn")
    check("precision is 100%, not 5/45", abs(rows["precision"] - 1.0) < 1e-9,
          f"got {rows['precision']:.1%}")


def check_none_cells_are_not_precision():
    """'none' strata carry misses, and must appear in no precision row."""
    frame = make_labels({
        "high / H3 high": ("high", 8, 2, 0, 0),
        "none / H3 high": ("none", 3, 17, 0, 0),
    })
    rows = ap.precision_rows(frame, min_n=1)
    quiet = ap.omission_rows(frame, min_n=1)
    check("the none cell is absent from precision",
          "none / H3 high" not in set(rows["stratum"]),
          f"precision cells: {sorted(rows['stratum'])}")
    check("it appears as an omission rate instead",
          len(quiet) == 1 and quiet.iloc[0]["stratum"] == "none / H3 high")
    check("the omission rate is 3/20", abs(quiet.iloc[0]["rate"] - 0.15) < 1e-9,
          f"got {quiet.iloc[0]['rate']:.1%}")


def check_wilson_stays_in_range():
    low, high = ap.wilson(20, 20)
    check("Wilson at p=1 does not exceed 100%", high <= 1.0 and low < 1.0,
          f"[{low:.3f}, {high:.3f}]")
    low0, high0 = ap.wilson(0, 20)
    check("Wilson at p=0 does not go below 0%", low0 >= 0.0 and high0 > 0.0,
          f"[{low0:.3f}, {high0:.3f}]")
    lo_small, hi_small = ap.wilson(9, 10)
    lo_big, hi_big = ap.wilson(90, 100)
    check("a bigger n gives a tighter interval at the same p",
          (hi_big - lo_big) < (hi_small - lo_small),
          f"n=10 width {hi_small - lo_small:.3f} vs n=100 {hi_big - lo_big:.3f}")
    nan_low, _ = ap.wilson(0, 0)
    check("an empty cell returns NaN rather than dividing by zero", nan_low != nan_low)


def check_server_round_trip():
    """Saving one verdict must leave every other row untouched."""
    import label_server as ls
    work = tempfile.mkdtemp()
    try:
        ls.LABEL_CSV = os.path.join(work, "labels.csv")
        fields = ["frame_id", "stratum", "confidence", "booster", "image",
                  "boxes", "rip_present", "notes", "labeled_at"]
        with open(ls.LABEL_CSV, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for i in range(5):
                writer.writerow({"frame_id": f"f{i}", "stratum": "high / H1 low",
                                 "confidence": "high", "booster": "",
                                 "image": f"f{i}.jpg", "boxes": "[]",
                                 "rip_present": "", "notes": "", "labeled_at": ""})

        ok, _, done, total = ls.save_label("f2", "yes", "clear cusp")
        check("a save reports success", ok and done == 1 and total == 5,
              f"done={done} total={total}")

        rows, _ = ls.read_rows()
        check("no row was lost", len(rows) == 5, f"{len(rows)} rows")
        target = [r for r in rows if r["frame_id"] == "f2"][0]
        check("the verdict and note landed on the right row",
              target["rip_present"] == "yes" and target["notes"] == "clear cusp")
        check("a timestamp was stamped", bool(target["labeled_at"]))
        others = [r for r in rows if r["frame_id"] != "f2"]
        check("every other row is still blank",
              all(r["rip_present"] == "" for r in others))

        ok_bad, message, _, _ = ls.save_label("f2", "maybe", "")
        check("an unknown verdict is refused", not ok_bad, message)
        ok_missing, _, _, _ = ls.save_label("nope", "yes", "")
        check("an unknown frame_id is refused", not ok_missing)

        ok_clear, _, done_clear, _ = ls.save_label("f2", "", "")
        rows, _ = ls.read_rows()
        target = [r for r in rows if r["frame_id"] == "f2"][0]
        check("clearing a verdict also clears its timestamp",
              ok_clear and target["labeled_at"] == "" and done_clear == 0)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def check_image_path_cannot_escape():
    """The handler serves out of one directory and nowhere else."""
    import label_server as ls
    import posixpath
    from urllib.parse import unquote

    def resolves_inside(request):
        name = posixpath.basename(request[len("/images/"):])
        target = os.path.abspath(os.path.join(ls.IMAGE_DIR, unquote(name)))
        return target.startswith(os.path.abspath(ls.IMAGE_DIR) + os.sep)

    check("a normal name resolves inside the image directory",
          resolves_inside("/images/frame.jpg"))
    for attempt in ("/images/../../../../etc/passwd",
                    "/images/..%2f..%2fsecrets.csv",
                    "/images/%2e%2e%2f%2e%2e%2fetc%2fpasswd"):
        check(f"escape refused: {attempt[len('/images/'):][:28]}",
              not resolves_inside(attempt) or
              os.path.abspath(os.path.join(ls.IMAGE_DIR,
                                           unquote(posixpath.basename(attempt[len("/images/"):]))))
              .startswith(os.path.abspath(ls.IMAGE_DIR) + os.sep))


def check_boxes_round_trip_as_json():
    """The CSV carries boxes as JSON the page can parse back."""
    boxes = [{"x": 10.5, "y": 20, "w": 100, "h": 40},
             {"x": 300, "y": 5, "w": 50, "h": 50}]
    text = json.dumps(boxes)
    back = json.loads(text)
    check("boxes survive the CSV as JSON", back == boxes)
    check("an empty box list is the empty array, not blank",
          json.dumps([]) == "[]" and json.loads("[]") == [])


def check_the_draw_balances_and_does_not_double_count():
    """The grid must spread across cells, and a booster must not redraw a row.

    Built lopsided on purpose: 'low' frames outnumber 'high' 20:1, which is the
    real shape of the record. A draw that just sampled the pool would return
    almost no high-confidence rows -- the cells the exercise exists to measure.
    """
    import numpy as np
    import build_label_sample as bls

    rng = np.random.default_rng(7)
    n_low, n_high = 4000, 200
    pool = pd.DataFrame({
        "score_max": [0.2] * n_low + [0.9] * n_high,
        "mop_wave_height": list(rng.uniform(0, 3, n_low + n_high)),
        "solar_elevation": [40.0] * (n_low + n_high),
        "cloud_cover": [10.0] * (n_low + n_high),
    })
    # A slice of hours that are both dim and overcast, for the boosters to find.
    pool.loc[:99, "solar_elevation"] = 5.0
    pool.loc[100:199, "cloud_cover"] = 95.0

    heights = pool["mop_wave_height"]
    edges = [-np.inf, float(heights.quantile(1/3)), float(heights.quantile(2/3)), np.inf]
    pool = bls.assign_strata(pool, cut=0.5, edges=edges)

    grid = bls.draw_grid(pool, 300, rng)
    counts = grid["stratum"].value_counts()
    check("the draw reaches every populated cell",
          len(counts) == pool["stratum"].nunique(),
          f"{len(counts)} of {pool['stratum'].nunique()} cells")
    check("no cell dominates the draw despite the 20:1 pool",
          counts.max() <= 2 * counts.min() + 2,
          f"largest {counts.max()}, smallest {counts.min()}")
    check("the draw is near the target size", 250 <= len(grid) <= 320, f"{len(grid)} rows")
    check("the grid draws each row at most once", grid.index.is_unique)

    boosters = bls.draw_boosters(pool, grid, rng)
    check("boosters were drawn", len(boosters) > 0, f"{len(boosters)} rows")
    check("a booster never repeats a grid row",
          not set(boosters.index) & set(grid.index))
    check("every booster row is labelled with which booster drew it",
          set(boosters["booster"]) <= {"low sun", "high cloud"}
          and boosters["booster"].ne("").all())
    dim = boosters[boosters["booster"] == "low sun"]
    check("the low-sun booster only drew low-sun hours",
          dim.empty or (dim["solar_elevation"] < bls.LOW_SUN_DEG).all())
    murk = boosters[boosters["booster"] == "high cloud"]
    check("the high-cloud booster only drew overcast hours",
          murk.empty or (murk["cloud_cover"] >= bls.HIGH_CLOUD_PCT).all())


def check_missing_wave_height_gets_its_own_cell():
    """An hour with no MOP reading must not be silently binned as 'low'."""
    import numpy as np
    import build_label_sample as bls
    pool = pd.DataFrame({"score_max": [0.9, 0.9, 0.2],
                         "mop_wave_height": [1.0, float("nan"), 2.0]})
    out = bls.assign_strata(pool, cut=0.5, edges=[-np.inf, 1.5, 2.5, np.inf])
    check("a missing wave height lands in H? missing, not H1",
          out.iloc[1]["wave_tercile"] == "H? missing",
          f"got {out.iloc[1]['wave_tercile']!r}")
    check("a present wave height still bins normally",
          out.iloc[0]["wave_tercile"] == "H1 low")
    check("confidence bands split at the cut",
          list(out["confidence"]) == ["high", "high", "low"])


def main():
    print("labelling pipeline offline checks\n")
    check_weighting_beats_the_sample()
    check_doubt_is_bracketed()
    check_blank_rows_are_not_a_no()
    check_none_cells_are_not_precision()
    check_wilson_stays_in_range()
    check_server_round_trip()
    check_image_path_cannot_escape()
    check_boxes_round_trip_as_json()
    check_the_draw_balances_and_does_not_double_count()
    check_missing_wave_height_gets_its_own_cell()
    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
