"""Offline checks for analyze_label_recheck.py.

The fixture is a label set whose notes are written to a known answer: some say
exactly the kind of thing the cue list is meant to catch, some say the
opposite, some hedge AND sound sure, and some are blank. The count of flagged
frames is therefore arithmetic done on paper, not whatever the matcher happens
to return.

The capacity check gets its own fixture, because it is the number the report
rests on: with 6 found rips at 19% recall, about 26 are missing, and whether a
candidate pool of 5 or of 50 can hold them is the whole question.
"""

import contextlib
import io
import math
import sys

import numpy as np
import pandas as pd

import analyze_label_recheck as lr

FAILURES = []


def check(name, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    if not condition:
        FAILURES.append(name)
    print(f"  {mark} {name}" + (f"  {detail}" if detail else ""))


def check_the_cue_matcher():
    print("\nwhat the cue list catches, and what it leaves alone")
    cases = [
        ("faint gap, no seaward foam", {"weak", "partial cue"}, False),
        ("possible channel, check persistence", {"hedged", "persistence"}, False),
        ("heavy glare off the water", {"conditions"}, False),
        ("clear textbook rip, strong neck", set(), True),
        ("obvious rip but slight haze", {"weak", "conditions"}, True),
        ("boat wake", set(), False),
        ("", set(), False),
        # 'streak' alone is deliberately NOT a cue -- a streak can be a strong
        # rip. 'faint' is what makes this one weak, and the pattern for
        # 'partial cue' says 'thin streak', not any streak.
        ("FAINT STREAK UNDER HAZE", {"weak", "conditions"}, False),
        ("thin streak", {"partial cue"}, False),
    ]
    for text, expected, expect_sure in cases:
        hits, sure = lr.cues_in(text)
        check(f"{text[:34]!r:<38} -> {sorted(expected)}",
              hits == expected, f"got {sorted(hits)}")
        check(f"{text[:20]!r:<24} confident={expect_sure}", sure == expect_sure)

    check("matching is case-insensitive",
          lr.cues_in("FAINT")[0] == lr.cues_in("faint")[0])
    check("a word inside another word does not match",
          not lr.cues_in("darkroom heartbeat")[0]
          or "conditions" not in lr.cues_in("heartbeat")[0],
          str(sorted(lr.cues_in("heartbeat")[0])))


def walton_fixture():
    """15 frames with a known-by-construction candidate count.

    Candidates must be: called no/doubt/unusable, carrying a note, matching a
    cue, and NOT sounding confident. Six rows satisfy all four.
    """
    rows = [
        # 6 genuine candidates, spread across the three verdicts
        ("f01", "no", "low", "faint gap, no seaward foam", 900),
        ("f02", "no", "high", "possible channel, check persistence", 800),
        ("f03", "doubt", "low", "weak streak under glare", 700),
        ("f04", "doubt", "high", "subtle one-sided gap", 600),
        ("f05", "unusable", "low", "too hazy to call", 500),
        ("f06", "no", "none", "maybe a narrow channel", 400),
        # cue AND confident -> excluded
        ("f07", "no", "low", "obvious boat wake, slight haze", 300),
        # no cue
        ("f08", "no", "low", "boat wake", 200),
        ("f09", "doubt", "high", "people in the water", 100),
        # blank notes -> cannot be screened
        ("f10", "no", "low", "", 950),
        ("f11", "doubt", "low", "", 850),
        # yes rows are never candidates, whatever they say
        ("f12", "yes", "high", "faint but I think real", 1200),
        ("f13", "yes", "low", "clear rip, strong neck", 1100),
        # unlabelled
        ("f14", "", "low", "faint gap", 1000),
        ("f15", "yes", "high", "textbook", 1300),
    ]
    return pd.DataFrame(
        [{"frame_id": f, "rip_present": v, "confidence": c, "notes": n,
          "bbox_area_max": a, "stratum": "s", "booster": ""}
         for f, v, c, n, a in rows])


def check_the_candidate_count():
    print("\ncounting recheck candidates")
    walton = walton_fixture()
    with contextlib.redirect_stdout(io.StringIO()):
        candidates, under, with_note = lr.flag_walton(walton)

    check("six frames satisfy all four conditions", len(candidates) == 6,
          str(sorted(candidates["frame_id"])))
    check("and they are the six built to", 
          sorted(candidates["frame_id"]) == ["f01", "f02", "f03", "f04",
                                             "f05", "f06"],
          str(sorted(candidates["frame_id"])))
    check("a Y frame is never a candidate, however it is worded",
          not any(f in set(candidates["frame_id"]) for f in ("f12", "f13", "f15")))
    check("an unlabelled frame is not one either",
          "f14" not in set(candidates["frame_id"]))
    check("a cue plus a confident word is excluded",
          "f07" not in set(candidates["frame_id"]))
    check("a note with no cue is excluded",
          not {"f08", "f09"} & set(candidates["frame_id"]))
    check("blank notes are counted as unscreenable, not as cleared",
          len(under) - len(with_note) == 2,
          f"{len(under)} under-calls, {len(with_note)} with a note")

    by_verdict = candidates["rip_present"].value_counts().to_dict()
    check("the breakdown by original label is 3 no / 2 doubt / 1 unusable",
          by_verdict == {"no": 3, "doubt": 2, "unusable": 1}, str(by_verdict))


def check_the_capacity_check():
    print("\nthe capacity check, which is the load-bearing number")
    walton = walton_fixture()
    with contextlib.redirect_stdout(io.StringIO()):
        candidates, _, _ = lr.flag_walton(walton)

    # 5 of the 6 candidates sit in a FIRED stratum; f06 is 'none'.
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        lr.precision_scenarios(walton, candidates, 0.19, 0.54)
    text = buffer.getvalue()
    check("candidates in a 'none' stratum are held out of precision",
          "1 sit in 'none'" in text, text[text.find("recheck candidates"):][:80])

    # Three Y frames among four strictly-labelled fired frames in the fixture.
    check("the observed precision is computed from strict fired labels only",
          "3 rips found" in text, text[text.find("rips found") - 20:][:60])

    # 3 found at 19% recall implies 15.8 real, so 12.8 missed, against a pool
    # of 5 -> cannot hold them. That arithmetic is done here, not read back.
    implied = 3 / 0.19
    check("the fixture is built so the pool CANNOT hold the implied misses",
          (implied - 3) > 5, f"{implied - 3:.1f} missing vs 5 candidates")
    check("and the report says so",
          "CANNOT HOLD THEM" in text, text[text.find("capacity") - 10:][:120])

    # Widen the pool and the verdict must flip, with nothing else changed.
    wide = pd.concat([walton] + [
        walton_fixture().iloc[:1].assign(frame_id=f"x{i:02d}")
        for i in range(20)], ignore_index=True)
    with contextlib.redirect_stdout(io.StringIO()):
        wide_candidates, _, _ = lr.flag_walton(wide)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        lr.precision_scenarios(wide, wide_candidates, 0.19, 0.54)
    check("a big enough pool flips the verdict",
          "CAN hold them" in buffer.getvalue(),
          f"{len(wide_candidates)} candidates")


def check_the_y_verdict_follows_the_number():
    """The conclusion must track the count, not be asserted around it."""
    print("\nwhere the Y calls sit")

    def verdict_for(areas_yes, areas_other):
        rows = [{"frame_id": f"y{i}", "rip_present": "yes", "confidence": "high",
                 "notes": "", "bbox_area_max": a, "stratum": "s", "booster": ""}
                for i, a in enumerate(areas_yes)]
        rows += [{"frame_id": f"n{i}", "rip_present": "no", "confidence": "high",
                  "notes": "", "bbox_area_max": a, "stratum": "s", "booster": ""}
                 for i, a in enumerate(areas_other)]
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            lr.rank_the_yes(pd.DataFrame(rows))
        return buffer.getvalue()

    strong = verdict_for([900, 950, 990], [100, 200, 300, 400])
    check("Y calls above the median read as STRONGER",
          "STRONGER end" in strong and "WEAKER end" not in strong)
    weak = verdict_for([100, 150, 199], [300, 400, 500, 900])
    check("Y calls below it read as WEAKER",
          "WEAKER end" in weak and "STRONGER end" not in weak)
    check("and the two do not print the same conclusion", strong != weak)


def check_mann_whitney():
    print("\nthe rank test, against hand-checkable cases")
    u, p = lr.mann_whitney([1, 2, 3], [4, 5, 6])
    check("completely separated groups give U=0", u == 0, str(u))
    u, p = lr.mann_whitney([1, 2, 3], [1, 2, 3])
    check("identical groups give the maximum U", abs(u - 4.5) < 1e-9, str(u))
    check("and a p of exactly 1, never above it", p == 1.0, f"{p:.6f}")
    for a, b in (([1, 2], [1, 2]), ([5], [5]), ([1, 1, 1], [1, 1, 1])):
        _, value = lr.mann_whitney(a, b)
        check(f"p stays in [0,1] for {a} vs {b}",
              math.isnan(value) or 0.0 <= value <= 1.0, f"{value}")
    u, p = lr.mann_whitney([1, 2, 3], [])
    check("an empty group is not an error", math.isnan(p))
    big_a = list(range(30))
    big_b = [v + 100 for v in range(30)]
    _, p = lr.mann_whitney(big_a, big_b)
    check("a large clean separation is significant", p < 0.001, f"{p:.2g}")


def check_it_touches_nothing():
    print("\nread-only")
    walton = walton_fixture()
    before = walton.copy()
    with contextlib.redirect_stdout(io.StringIO()):
        lr.flag_walton(walton)
        lr.rank_the_yes(walton)
    check("the label frame is not mutated",
          walton.equals(before))
    check("no column is added to it",
          list(walton.columns) == list(before.columns))


def main():
    print("Walton label recheck offline checks")
    check_the_cue_matcher()
    check_the_candidate_count()
    check_the_capacity_check()
    check_the_y_verdict_follows_the_number()
    check_mann_whitney()
    check_it_touches_nothing()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
