"""Offline checks for build_ripaid_calibration.py and analyze_ripaid_calibration.py.

The kappa fixture is ten frames with the confusion matrix chosen by hand, so
the expected value is arithmetic done on paper and not a number this code
produced: 4 agreed rips, 4 agreed nones, one miss and one false alarm gives
po = 0.8, pe = 0.5 and kappa = 0.6 exactly.

The rotation fixture plants a driver on bearing 120 at Cala Millor's latitude
using REAL solar geometry, because the whole question is the shape of the sun's
path and a uniform random azimuth would make every bearing equally good. It
asserts the sweep finds 120 and not the site's shore normal.

A third fixture exists for the reason the dataset warning exists: a frame the
annotators marked `doubt` is neither a positive nor a negative, and letting one
into either stratum would score an unanswerable frame as if it had an answer.
"""

import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd

import analyze_glare_split as gs
import analyze_ripaid_calibration as arc
import build_ripaid_calibration as brc
import load_ripaid as lr
import solar

FAILURES = []

LAT, LON = 39.5965, 3.3835     # Cala Millor
SHORE_NORMAL = 90.0


def check(name, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    if not condition:
        FAILURES.append(name)
    print(f"  {mark} {name}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------------------

def payload(spec):
    """A COCO export. `spec` is [(n_rip, n_doubt), ...], one per frame."""
    images, annotations = [], []
    next_id = 1
    for index, (rips, doubts) in enumerate(spec):
        images.append({"id": index + 1,
                       "file_name": f"clm_s_01_2011-05-{(index % 27) + 1:02d}-11-00.png"})
        for _ in range(rips):
            annotations.append({"id": next_id, "image_id": index + 1,
                                "category_id": 1, "area": 500.0})
            next_id += 1
        for _ in range(doubts):
            annotations.append({"id": next_id, "image_id": index + 1,
                                "category_id": 2, "area": 200.0})
            next_id += 1
    return {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": lr.RIP_LABEL},
                           {"id": 2, "name": lr.DOUBT_LABEL}]}


def check_kappa_on_a_hand_built_matrix():
    print("\nCohen's kappa on ten frames with a known matrix")
    # Truth: 5 rips then 5 nones. Mine: 4 of the rips, 1 miss, 1 false alarm.
    theirs = [True] * 5 + [False] * 5
    mine = [True, True, True, True, False,
            True, False, False, False, False]
    result = arc.kappa(mine, theirs)
    check("the cells are counted correctly",
          (result["a"], result["b"], result["c"], result["d"]) == (4, 1, 1, 4),
          f"a={result['a']} b={result['b']} c={result['c']} d={result['d']}")
    check("observed agreement is 0.8", abs(result["po"] - 0.8) < 1e-12)
    check("chance agreement is 0.5", abs(result["pe"] - 0.5) < 1e-12)
    check("kappa is exactly 0.6", abs(result["kappa"] - 0.6) < 1e-12,
          f"{result['kappa']:.6f}")

    perfect = arc.kappa(theirs, theirs)
    check("perfect agreement gives kappa 1", abs(perfect["kappa"] - 1.0) < 1e-12)

    # Calling everything a rip agrees on half the frames and knows nothing.
    everything = arc.kappa([True] * 10, theirs)
    check("calling every frame a rip scores 50% agreement",
          abs(everything["po"] - 0.5) < 1e-12)
    check("and kappa sees through it, at 0",
          abs(everything["kappa"]) < 1e-12, f"{everything['kappa']:.6f}")

    reversed_calls = arc.kappa([not m for m in theirs], theirs)
    check("being exactly wrong gives kappa -1",
          abs(reversed_calls["kappa"] + 1.0) < 1e-12)

    low, high = arc.kappa_interval(mine, theirs, draws=400)
    check("the bootstrap interval brackets the estimate",
          low <= 0.6 <= high, f"[{low:+.2f}, {high:+.2f}]")
    check("and is wide at n=10, as it should be",
          high - low > 0.3, f"width {high - low:.2f}")


def check_doubt_frames_are_excluded():
    print("\ndoubt frames belong in neither stratum")
    # 4 clean rips, 3 clean nones, 2 rip+doubt, 1 doubt only.
    spec = ([(1, 0)] * 4 + [(0, 0)] * 3 + [(1, 1)] * 2 + [(0, 1)])
    frames = lr.build_frames(payload(spec))
    with contextlib.redirect_stdout(io.StringIO()):
        positives, negatives, report = brc.clean_strata(frames)
    check("the clean positives are only the undoubted ones",
          len(positives) == 4, f"{len(positives)}")
    check("the clean negatives are only the undoubted ones",
          len(negatives) == 3, f"{len(negatives)}")
    check("a rip WITH doubt is not counted as a positive",
          report["doubt_with_rip"] == 2, f"{report['doubt_with_rip']}")
    check("a doubt-only frame is not counted as a negative",
          report["doubt_only"] == 1, f"{report['doubt_only']}")
    check("no frame carrying doubt survives into either stratum",
          int(positives["n_doubt"].sum()) == 0
          and int(negatives["n_doubt"].sum()) == 0)
    check("and every frame is accounted for",
          len(positives) + len(negatives) + report["doubt_frames"] == len(frames))


def check_the_draw_is_balanced_and_shuffled():
    print("\nthe draw")
    spec = [(1, 0)] * 40 + [(0, 0)] * 40
    frames = lr.build_frames(payload(spec))
    with contextlib.redirect_stdout(io.StringIO()):
        positives, negatives, _ = brc.clean_strata(frames)
        sample = brc.draw(positives, negatives, 10, seed=1)
    check("the sample is the requested size", len(sample) == 20)
    check("and balanced", int((sample["n_rip"] > 0).sum()) == 10)
    order = list(sample["n_rip"] > 0)
    check("the order is not positives-then-negatives",
          order != [True] * 10 + [False] * 10)
    check("and the same seed reproduces it",
          list(brc.draw(positives, negatives, 10, seed=1)["file_name"])
          == list(sample["file_name"]))

    with contextlib.redirect_stdout(io.StringIO()):
        other = brc.draw(positives, negatives, 10, seed=2)
    check("a different seed draws differently",
          list(other["file_name"]) != list(sample["file_name"]))


def check_the_written_sample_hides_the_answer():
    print("\nwhat reaches the page")
    spec = [(1, 0)] * 6 + [(0, 0)] * 6
    frames = lr.build_frames(payload(spec))
    with contextlib.redirect_stdout(io.StringIO()):
        positives, negatives, _ = brc.clean_strata(frames)
        sample = brc.draw(positives, negatives, 5, seed=3)
    with tempfile.TemporaryDirectory() as folder:
        out = os.path.join(folder, "sample")
        key = os.path.join(folder, "key")
        brc.write_sample(sample, None, out_dir=out, key_dir=key)
        labels = pd.read_csv(os.path.join(out, "labels.csv"))
        truth = pd.read_csv(os.path.join(key, "truth.csv"))

        served = set(labels.columns)
        leaks = served & {"n_rip", "n_doubt", "truth", "file_name", "camera",
                          "site", "timestamp", "area_max", "detected"}
        check("no column the page can see names the answer", not leaks, str(leaks))
        check("the image names carry nothing either",
              all(str(v).startswith("cal_") for v in labels["image"]))
        check("display_meta says only the position in the run",
              all(str(v).startswith("frame ") for v in labels["display_meta"]))
        check("the key is written outside the served directory",
              os.path.exists(os.path.join(key, "truth.csv"))
              and not os.path.exists(os.path.join(out, "truth.csv")))
        check("and the key can still be joined back",
              set(truth["frame_id"]) == set(labels["frame_id"]))
        check("every row starts unlabelled",
              labels["rip_present"].fillna("").astype(str).str.strip().eq("").all())
        check("the key's truth column matches the drawn frames",
              int((truth["truth"] == "yes").sum()) == 5)


def check_the_rotation_finds_a_planted_bearing():
    print("\nthe bearing rotation on human-style labels")
    planted = 120.0
    rng = np.random.default_rng(404)
    stamps = pd.date_range("2011-04-01", "2011-10-01", freq="h", tz="UTC")
    elevation, _ = solar.position(stamps, LAT, LON)
    stamps = stamps[elevation > 5]          # annotators work in daylight
    frame = pd.DataFrame({
        "timestamp": stamps,
        "detected": True,
        "n_rip": 1,
        "n_doubt": 0,
        "camera": "clm_s_01",
        "site": "clm",
        "area_max": 500.0,
    })
    work = arc.add_geometry(frame, LAT, LON)
    driver = solar.glare_index(work["solar_elevation"], work["solar_azimuth"],
                               planted)
    values = np.asarray(driver, dtype=float)
    work["target"] = ((values - values.mean()) / (values.std() or 1.0) * 0.6
                      + rng.normal(0, 1.0, len(work)))

    grid = arc.rotate(work, "target", ["month_sin", "month_cos"],
                      list(range(0, 360, 15)))
    peak = float(grid.loc[grid["dR2"].idxmax(), "bearing"])
    check("the sweep peaks on the bearing the driver was built on",
          gs.fold(peak - planted) <= 30.0,
          f"peak {peak:.0f}, planted {planted:.0f}")
    at_planted = float(grid.loc[grid["bearing"] == 120, "dR2"].iloc[0])
    at_normal = float(grid.loc[grid["bearing"] == 90, "dR2"].iloc[0])
    check("and prefers it to the site's shore normal",
          at_planted > at_normal, f"{at_planted:+.4f} vs {at_normal:+.4f}")
    check("the sweep is not flat",
          grid["dR2"].max() - grid["dR2"].min() > 0.01,
          f"range {grid['dR2'].max() - grid['dR2'].min():.4f}")

    # A target with no sun structure at all must not produce a confident peak.
    work["flat"] = rng.normal(0, 1.0, len(work))
    null_grid = arc.rotate(work, "flat", ["month_sin", "month_cos"],
                           list(range(0, 360, 15)))
    check("a target with no sun structure gives a tiny peak",
          null_grid["dR2"].max() < 0.01,
          f"max dR2 {null_grid['dR2'].max():+.5f}")


def check_son_bou_cannot_answer_the_question():
    print("\nwhich sites can separate camera from solar noon")
    stamps = pd.date_range("2011-05-01", "2011-09-01", freq="h", tz="UTC")
    for site, expected in (("clm", True), ("snb", False)):
        meta = lr.KNOWN_SITES[site]
        frame = pd.DataFrame({"timestamp": stamps, "detected": True,
                              "n_rip": 1, "n_doubt": 0})
        work = arc.add_geometry(frame, meta["latitude"], meta["longitude"])
        noon, _, _ = gs.solar_noon_bearing(work, meta["latitude"],
                                          meta["longitude"])
        bearing = arc.ad.SHORE_NORMAL_DEG[meta["name"]]
        separation = gs.fold(bearing - noon)
        usable = separation >= 2 * gs.MARGIN_DEG
        check(f"{meta['name']} separation {separation:.0f} deg — "
              f"{'can' if expected else 'CANNOT'} discriminate",
              usable == expected,
              f"noon {noon:.0f}, shore normal {bearing:.0f}")


def main():
    print("RipAID calibration offline checks")
    check_kappa_on_a_hand_built_matrix()
    check_doubt_frames_are_excluded()
    check_the_draw_is_balanced_and_shuffled()
    check_the_written_sample_hides_the_answer()
    check_the_rotation_finds_a_planted_bearing()
    check_son_bou_cannot_answer_the_question()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
