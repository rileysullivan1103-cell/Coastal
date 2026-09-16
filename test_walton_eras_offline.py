"""Offline checks for analyze_walton_eras.py.

The fixture is a detector whose score floor moves mid-record, watching a beach
whose driver moves the OTHER way: the low-floor era has low waves, the
high-floor era has high waves. That pairing is the point. The true relationship
between waves and the detection rate is positive and strong, and reading the
two eras as one instrument turns it NEGATIVE -- so a method that merely
"weakens" the pooled number would still be reporting a driver with the wrong
sign, and the test can tell the difference.

Nothing here asserts against a second copy of the code under test. The frames
are built from per-detection scores that the fixture keeps, and the truth
series is derived from those scores directly -- "a detection exists if its own
score clears 0.70" -- not by running the censoring and calling the result
correct. Where A cannot reach the truth (it can only demote whole frames,
because the table has no per-detection score), the shortfall is computed from
the fixture's score lists and asserted as an exact count.
"""

import contextlib
import glob
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd

import analyze_drivers as ad
import analyze_walton_eras as we
import build_label_sample as bls
import pull_rip_detection as prd

FAILURES = []

HOURS = 600
FRAMES_PER_HOUR = 6
SPLIT = pd.Timestamp("2025-01-01", tz="UTC")
OLD_FLOOR = 0.50
NEW_FLOOR = 0.70


def check(name, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    if not condition:
        FAILURES.append(name)
    print(f"  {mark} {name}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# the fixture
# ---------------------------------------------------------------------------

def build(rng):
    """Per-frame detection scores, plus the wave height that produced them.

    Returns (rows, waves). Each row carries `scores`, the list of scores the
    model assigned that frame before any floor is applied -- which is what a
    real frame table does not have, and what makes the truth computable here.
    """
    start = SPLIT - pd.Timedelta(hours=HOURS // 2)
    hours = pd.date_range(start, periods=HOURS, freq="h", tz="UTC")
    pre = hours < SPLIT
    # Low waves under the old, generous floor; high waves under the new,
    # strict one. The floor change therefore works AGAINST the driver.
    waves = np.where(pre, rng.normal(1.2, 0.35, HOURS),
                     rng.normal(2.2, 0.35, HOURS)).clip(0.2, None)

    rows = []
    for index, (hour, wave) in enumerate(zip(hours, waves)):
        for slot in range(FRAMES_PER_HOUR):
            centre = 0.40 + 0.12 * wave
            # Most frames hold one detection. Every sixth holds two, which is
            # where A's frame-level floor cannot reach the weaker one.
            count = 2 if slot == 0 else 1
            scores = [float(np.clip(rng.normal(centre, 0.10), 0.01, 0.999))
                      for _ in range(count)]
            rows.append({
                "timestamp": hour + pd.Timedelta(minutes=10 * slot),
                "hour": hour, "wave": wave, "scores": scores,
                "pre": bool(hour < SPLIT),
            })
    return pd.DataFrame(rows), pd.Series(waves, index=hours, name="wave")


def publish(rows, floors):
    """The frame table a feed with these floors would have written.

    `floors` is (pre_floor, post_floor). A frame whose detections all fall
    under its era's floor is published as an observed zero -- present, with
    detected False and no scores -- exactly as the real feed's coverage-joined
    table represents one.
    """
    pre_floor, post_floor = floors
    out = []
    for row in rows.itertuples():
        floor = pre_floor if row.pre else post_floor
        kept = [s for s in row.scores if s >= floor]
        out.append({
            "timestamp": row.timestamp,
            "detected": bool(kept),
            "detection_count": len(kept),
            "score_max": max(kept) if kept else np.nan,
            "score_mean": (sum(kept) / len(kept)) if kept else np.nan,
            "bbox_area_max": 1000.0 * max(kept) if kept else np.nan,
            "score_classes": "rip_current" if kept else np.nan,
        })
    return pd.DataFrame(out)


def object_frames(rows, rng):
    """Post-era COCO detections from the other model, to be filtered out."""
    post = rows[~rows["pre"]].iloc[::7]
    return pd.DataFrame({
        "timestamp": post["timestamp"] + pd.Timedelta(seconds=30),
        "detected": True,
        "detection_count": 2,
        "score_max": 0.95,
        "score_mean": 0.93,
        "bbox_area_max": 4000.0,
        "score_classes": "person",
    })


def rate_rho(table, waves):
    """Rank correlation of the hourly detection rate against wave height."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "rip_fixture_hourly.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            hourly = prd.hourly_summary(table, path)
    joined = hourly.set_index("hour").join(waves)
    rho, n, p = ad.spearman(joined["wave"], joined["detection_rate"])
    return rho, n, hourly


# ---------------------------------------------------------------------------

def check_a_recovers_the_driver_and_pooling_does_not():
    print("\nre-censoring recovers a driver the pooled record inverts")
    rng = np.random.default_rng(20260916)
    rows, waves = build(rng)

    two_floor = publish(rows, (OLD_FLOOR, NEW_FLOOR))
    truth = publish(rows, (NEW_FLOOR, NEW_FLOOR))
    with contextlib.redirect_stdout(io.StringIO()):
        censored, stats = we.censor(two_floor, SPLIT, NEW_FLOOR)

    rho_truth, n_truth, _ = rate_rho(truth, waves)
    rho_pooled, _, hourly_pooled = rate_rho(two_floor, waves)
    rho_a, _, hourly_a = rate_rho(censored, waves)

    check("the fixture's true driver is strongly positive",
          rho_truth > 0.40, f"rho={rho_truth:+.3f}, n={n_truth}")
    check("pooling the two floors inverts its SIGN, not just its size",
          rho_pooled < 0.0, f"rho={rho_pooled:+.3f}")
    check("A puts the sign back",
          rho_a > 0.40, f"rho={rho_a:+.3f}")
    check("and lands within 0.05 of the truth",
          abs(rho_a - rho_truth) < 0.05,
          f"|{rho_a:+.3f} - {rho_truth:+.3f}| = {abs(rho_a - rho_truth):.3f}")
    check("while pooled is nowhere near it",
          abs(rho_pooled - rho_truth) > 0.30,
          f"off by {abs(rho_pooled - rho_truth):.3f}")

    # detection_rate depends only on which frames count as detections, and A
    # gets that exactly right, so those two series are identical by
    # construction. The counts are where A's frame-level floor falls short, so
    # that is where "close but not equal" has to be shown.
    def count_rho(table):
        with tempfile.TemporaryDirectory() as folder:
            with contextlib.redirect_stdout(io.StringIO()):
                hourly = prd.hourly_summary(
                    table, os.path.join(folder, "rip_fixture_hourly.csv"))
        joined = hourly.set_index("hour").join(waves)
        return ad.spearman(joined["wave"], joined["detections"])[0]

    count_truth, count_pooled, count_a = (count_rho(truth), count_rho(two_floor),
                                          count_rho(censored))
    check("on detection COUNTS the pooled sign is wrong too",
          count_pooled < 0 < count_truth,
          f"truth {count_truth:+.3f}, pooled {count_pooled:+.3f}")
    check("A is close to the truth on counts without matching it exactly",
          0 < abs(count_a - count_truth) < 0.05,
          f"A {count_a:+.3f} vs truth {count_truth:+.3f}")

    # Demotion, not deletion: the hours keep their denominators.
    check("A changes no hour's frame count",
          hourly_a["frames"].tolist() == hourly_pooled["frames"].tolist())
    check("A never raises an hour's detection rate",
          bool((hourly_a["detection_rate"]
                <= hourly_pooled["detection_rate"] + 1e-9).all()))


def check_censoring_matches_detection_level_truth():
    print("\nwhat A gets exactly right, and what it cannot reach")
    rng = np.random.default_rng(7)
    rows, _ = build(rng)
    two_floor = publish(rows, (OLD_FLOOR, NEW_FLOOR))
    truth = publish(rows, (NEW_FLOOR, NEW_FLOOR))
    with contextlib.redirect_stdout(io.StringIO()):
        censored, stats = we.censor(two_floor, SPLIT, NEW_FLOOR)

    check("every frame A calls a detection is one at the true floor too",
          censored["detected"].tolist() == truth["detected"].tolist())
    check("no row is dropped",
          len(censored) == len(two_floor) == len(rows))

    # A is frame-level, so a pre-era frame holding one detection over the floor
    # keeps its weaker one as well. That surplus is computable straight from
    # the fixture's score lists, and is what A's counts are wrong by.
    surplus = 0
    surplus_frames = 0
    for row in rows.itertuples():
        if not row.pre:
            continue
        published = [s for s in row.scores if s >= OLD_FLOOR]
        if not published or max(published) < NEW_FLOOR:
            continue
        weak = sum(1 for s in published if s < NEW_FLOOR)
        surplus += weak
        surplus_frames += 1 if weak else 0
    gap = int(censored["detection_count"].sum()) - int(truth["detection_count"].sum())
    check("A over-counts detections by exactly the sub-threshold survivors",
          gap == surplus, f"gap={gap}, surplus={surplus}")
    check("the fixture actually exercises that case",
          surplus_frames > 0, f"{surplus_frames} frames")

    # The reported residual is a LOWER bound: score_mean < floor <= score_max
    # proves a survivor, but a strong detection can pull the mean back over the
    # floor and hide one. It must never claim more than there are.
    check("the reported residual never overstates the real one",
          0 < stats["residual_frames"] <= surplus_frames,
          f"reported {stats['residual_frames']} of {surplus_frames} real")


def check_the_post_era_assertion_fires():
    print("\nthe floor assertion on the post era")
    rng = np.random.default_rng(11)
    rows, _ = build(rng)
    clean = publish(rows, (OLD_FLOOR, NEW_FLOOR))
    all_below, some_below = we.post_era_violations(clean, SPLIT, NEW_FLOOR)
    check("a record whose post era really is at 0.70 passes",
          len(all_below) == 0 and len(some_below) == 0)

    # One frame whose scores are all under the floor, one whose max clears it
    # while its mean does not. The second is the one a score_max check misses.
    dirty = clean.copy()
    post = dirty.index[dirty["timestamp"] >= SPLIT]
    dirty.loc[post[0], ["detected", "detection_count", "score_max", "score_mean"]] = \
        [True, 1, 0.61, 0.61]
    dirty.loc[post[1], ["detected", "detection_count", "score_max", "score_mean"]] = \
        [True, 2, 0.92, 0.66]
    all_below, some_below = we.post_era_violations(dirty, SPLIT, NEW_FLOOR)
    check("a frame with nothing over the floor is caught", len(all_below) == 1)
    check("a frame whose MEAN is under the floor is caught too",
          len(some_below) == 1)

    # A pre-era frame under the floor is the normal case and must not fire.
    pre_dirty = clean.copy()
    pre = pre_dirty.index[pre_dirty["timestamp"] < SPLIT]
    pre_dirty.loc[pre[0], ["detected", "detection_count", "score_max", "score_mean"]] = \
        [True, 1, 0.52, 0.52]
    all_below, some_below = we.post_era_violations(pre_dirty, SPLIT, NEW_FLOOR)
    check("the pre era is allowed to sit under it",
          len(all_below) == 0 and len(some_below) == 0)


def check_the_class_filter_and_the_era_split():
    print("\nclass filter and era split")
    rng = np.random.default_rng(3)
    rows, _ = build(rng)
    table = publish(rows, (OLD_FLOOR, NEW_FLOOR))
    objects = object_frames(rows, rng)
    mixed = pd.concat([table, objects], ignore_index=True).sort_values(
        "timestamp").reset_index(drop=True)

    with contextlib.redirect_stdout(io.StringIO()):
        kept, dropped = bls.keep_class(mixed, wanted=bls.RIP_CLASSES)
    check("every object frame goes", dropped == len(objects))
    check("and nothing else does", len(kept) == len(table))
    blanks_in = int(mixed["score_classes"].isna().sum())
    blanks_out = int(kept["score_classes"].isna().sum())
    check("blank-class observed zeros are kept",
          blanks_in == blanks_out and blanks_in > 0,
          f"{blanks_out} of {blanks_in}")

    pre_mask = we.era_of(kept, SPLIT)
    check("the eras partition the record",
          int(pre_mask.sum()) + int((~pre_mask).sum()) == len(kept))
    check("and split at the date, not around it",
          bool(kept.loc[pre_mask, "timestamp"].max() < SPLIT
               <= kept.loc[~pre_mask, "timestamp"].min()))


def check_detected_is_read_not_guessed():
    print("\nreading `detected` back off disk")
    frame = pd.DataFrame({"detected": ["True", "False", "true", "FALSE"]})
    check("the strings False and FALSE stay False",
          we.as_bool(frame["detected"]).tolist() == [True, False, True, False])
    check("a real bool column is left alone",
          we.as_bool(pd.Series([True, False])).tolist() == [True, False])
    check("a 0/1 column reads as 0/1",
          we.as_bool(pd.Series([0, 1, 0])).tolist() == [False, True, False])


def check_the_mirror_is_globbable_and_read_only():
    print("\nthe stand-in data/ directory")
    with tempfile.TemporaryDirectory() as folder:
        source = os.path.join(folder, "data")
        os.makedirs(os.path.join(source, "rip_detection", "payloads"))
        original = pd.DataFrame({"hour": ["2025-01-01T00:00:00Z"],
                                 "frames": [4], "score_max": [0.51]})
        original.to_csv(
            os.path.join(source, "rip_detection", "rip_fix_hourly.csv"),
            index=False)
        pd.DataFrame({"hour": ["2025-01-01T00:00:00Z"], "images": [4]}).to_csv(
            os.path.join(source, "rip_detection", "coverage_fix_hourly.csv"),
            index=False)
        open(os.path.join(source, "rip_detection", "payloads", "a.json"),
             "w").write("{}")
        pd.DataFrame({"time": ["2025-01-01T00:00:00Z"],
                      "cloud_cover": [50]}).to_csv(
            os.path.join(source, "cloud_fix.csv"), index=False)

        replacement = pd.DataFrame({"hour": ["2025-01-01T00:00:00Z"],
                                    "frames": [4], "score_max": [0.81]})
        mirror = we.mirror_data(
            source, os.path.join(folder, "mirror"),
            {os.path.join("rip_detection", "rip_fix_hourly.csv"): replacement})

        found = glob.glob(os.path.join(mirror, "**", "rip_*_hourly.csv"),
                          recursive=True)
        check("assemble_rip's own glob finds the replaced table",
              len(found) == 1, str(found))
        check("and reads the replacement, not the original",
              float(pd.read_csv(found[0])["score_max"].iloc[0]) == 0.81)
        coverage = glob.glob(os.path.join(mirror, "**", "coverage_fix_hourly.csv"),
                             recursive=True)
        check("apply_coverage's glob still finds the real coverage file",
              len(coverage) == 1)
        check("untouched files are symlinks, not copies",
              os.path.islink(os.path.join(mirror, "cloud_fix.csv")))
        check("the payload folder is one link, not a tree of them",
              os.path.islink(os.path.join(mirror, "rip_detection", "payloads")))
        check("the real data/ is not written to",
              float(pd.read_csv(os.path.join(
                  source, "rip_detection", "rip_fix_hourly.csv")
              )["score_max"].iloc[0]) == 0.51)


def check_a_workspace_inside_data_cannot_leak_into_the_mirror():
    """glob's ** DOES follow a symlinked directory, so this is not theoretical.

    A workspace under data/ gets linked back into every variant's mirror, and
    each variant writes its table under the SAME filename. assemble_rip globs
    for that name and takes the first match, so the run would print one
    variant's label over another variant's numbers.
    """
    print("\na workspace living inside data/")
    with tempfile.TemporaryDirectory() as folder:
        source = os.path.join(folder, "data")
        os.makedirs(os.path.join(source, "rip_detection"))
        pd.DataFrame({"hour": ["2025-01-01T00:00:00Z"], "score_max": [0.51]}
                     ).to_csv(os.path.join(source, "rip_detection",
                                           "rip_fix_hourly.csv"), index=False)
        workspace = os.path.join(source, "era_workspace", "pooled")
        os.makedirs(workspace)
        pd.DataFrame({"hour": ["2025-01-01T00:00:00Z"], "score_max": [0.33]}
                     ).to_csv(os.path.join(workspace, "rip_fix_hourly.csv"),
                              index=False)

        replacement = pd.DataFrame({"hour": ["2025-01-01T00:00:00Z"],
                                    "score_max": [0.81]})
        key = os.path.join("rip_detection", "rip_fix_hourly.csv")

        leaky = we.mirror_data(source, os.path.join(folder, "leaky"),
                               {key: replacement})
        found = glob.glob(os.path.join(leaky, "**", "rip_fix_hourly.csv"),
                          recursive=True)
        check("without the exclusion the stray table IS reachable",
              len(found) == 2, f"{len(found)} found")

        safe = we.mirror_data(source, os.path.join(folder, "safe"),
                              {key: replacement},
                              exclude=[os.path.join(source, "era_workspace")])
        found = glob.glob(os.path.join(safe, "**", "rip_fix_hourly.csv"),
                          recursive=True)
        check("excluding the workspace leaves exactly one",
              len(found) == 1, f"{found}")
        check("and it is the replacement",
              found and float(pd.read_csv(found[0])["score_max"].iloc[0]) == 0.81)
        check("the rest of data/ still comes through",
              os.path.isdir(os.path.join(safe, "rip_detection")))


def check_a_buoy_that_only_covers_one_era_is_flagged():
    """A cannot test a finding computed on hours it never re-censors.

    Walton's buoy record starts after the split, so every hour where MOP and
    the buoy both report is post-era. A's series on those hours IS pooled's,
    and a verdict printed under A's name there is pooled's number wearing A's
    label. The pre-hour count is what makes that visible.
    """
    print("\na buoy that only reports in one era")
    hours = pd.date_range(SPLIT - pd.Timedelta(hours=400), periods=800,
                          freq="h", tz="UTC")
    rng = np.random.default_rng(31)
    mop = rng.gamma(2.0, 0.6, len(hours))
    frame = pd.DataFrame({
        "hour": hours, "mop_wave_height": mop,
        "WVHT": mop + rng.normal(0, 0.4, len(hours)),
        "detection_rate": np.clip(0.1 * mop + rng.normal(0, 0.1, len(hours)),
                                  0, 1),
        "hour_of_day": hours.hour, "month": hours.month,
    })
    frame["hr_mo"] = (frame["hour_of_day"].astype(str) + "-"
                      + frame["month"].astype(str))

    we.SPLIT_HOURS.clear()
    we.SPLIT_HOURS.append(SPLIT)

    spanning = we.mop_vs_buoy(frame, ["detection_rate"])
    check("a buoy spanning both eras reports pre-era hours",
          int(spanning["pre_hours"].iloc[0]) == 400,
          f"{int(spanning['pre_hours'].iloc[0])}")

    post_only = frame.copy()
    post_only.loc[post_only["hour"] < SPLIT, "WVHT"] = np.nan
    late = we.mop_vs_buoy(post_only, ["detection_rate"])
    check("a buoy that starts after the split reports none",
          int(late["pre_hours"].iloc[0]) == 0)
    check("and its matched hours are all post-era",
          int(late["n"].iloc[0]) == 400, f"n={int(late['n'].iloc[0])}")


def check_the_recensoring_check_tells_the_two_apart():
    print("\nthe check that would show A did nothing")
    rng = np.random.default_rng(5)
    rows, _ = build(rng)
    two_floor = publish(rows, (OLD_FLOOR, NEW_FLOOR))
    with contextlib.redirect_stdout(io.StringIO()):
        censored, _ = we.censor(two_floor, SPLIT, NEW_FLOOR)

    with tempfile.TemporaryDirectory() as folder:
        for label, table in (("original", two_floor), ("A", censored)):
            os.makedirs(os.path.join(folder, label))
            with contextlib.redirect_stdout(io.StringIO()):
                prd.hourly_summary(
                    table, os.path.join(folder, label, "rip_fix_hourly.csv"))
        reports = {}
        for label in ("original", "A"):
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                we.check_recensoring(os.path.join(folder, label), SPLIT,
                                     NEW_FLOOR)
            reports[label] = buffer.getvalue()

    def pre_line(text):
        return next(line for line in text.splitlines() if "pre  era:" in line)

    check("the untouched table reports the pre era NOT APPLIED",
          "NOT APPLIED" in pre_line(reports["original"]),
          pre_line(reports["original"]).strip())
    check("the re-censored one reports it APPLIED",
          "NOT APPLIED" not in pre_line(reports["A"])
          and "APPLIED" in pre_line(reports["A"]),
          pre_line(reports["A"]).strip())
    check("the two do not read the same",
          reports["original"] != reports["A"])


def main():
    print("walton era / re-censoring offline checks")
    check_a_recovers_the_driver_and_pooling_does_not()
    check_censoring_matches_detection_level_truth()
    check_the_post_era_assertion_fires()
    check_the_class_filter_and_the_era_split()
    check_detected_is_read_not_guessed()
    check_the_mirror_is_globbable_and_read_only()
    check_a_workspace_inside_data_cannot_leak_into_the_mirror()
    check_a_buoy_that_only_covers_one_era_is_flagged()
    check_the_recensoring_check_tells_the_two_apart()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
