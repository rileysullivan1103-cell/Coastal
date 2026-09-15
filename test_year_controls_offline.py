"""Offline checks for analyze_year_controls.py, on built frames.

The two things worth defending here are the ones that would produce a
confident wrong answer rather than a crash.

The first is the scope of the per-year demeaning. If the residuals were taken
across the pooled frame and split afterwards, the cell means would already
carry every year, a level difference between years would be partly removed
before the split, and the split would report that the years agree because it
had made them agree. The fixture builds a frame where that mistake changes the
answer, so the check has teeth.

The second is the grading of a sign change. Flagging every flip is as useless
as flagging none: real records are full of weak predictors wobbling around
zero, and a reversal between two significant estimates has to come out of that
pile separately.

    python test_year_controls_offline.py
"""

import numpy as np
import pandas as pd

import analyze_year_controls as yc

FAILURES = []


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name
          + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def build(seed=0, flip_second_year=False):
    """Three years of hourly rows with a known per-year relationship.

    The predictor sits at a different LEVEL in each year and the target has a
    year offset of its own. Pooled demeaning leaves both of those offsets
    partly in the residuals, which is what the scope check detects.
    """
    rng = np.random.default_rng(seed)
    # 2024-05 to 2026-09, the real Walton span: three calendar years, the
    # first and last of them partial. Two years would not exercise the
    # baseline-omission of the dummies or the under-powered-year path.
    hours = pd.date_range("2024-05-01", periods=3450, freq="6h", tz="UTC")
    frame = pd.DataFrame({"hour": hours})
    frame["hour_of_day"] = frame["hour"].dt.hour
    frame["month"] = frame["hour"].dt.month
    year = frame["hour"].dt.year
    # Level shifts per year, in the predictor and in the target separately.
    frame["driver"] = (rng.normal(2.0, 0.5, len(frame))
                       + (year - year.min()) * 1.5)
    cycle = np.sin(frame["hour_of_day"] / 24 * 2 * np.pi)
    slope = np.where((year == 2025) & flip_second_year, -0.8, 0.8)
    frame["target"] = (slope * frame["driver"] + cycle
                       + (year - year.min()) * 3.0
                       + rng.normal(0, 0.4, len(frame)))
    # A predictor with no relationship at all, to sit in the noise pile.
    frame["nothing"] = rng.normal(0, 1, len(frame))
    return frame


def check_per_year_demeaning_is_scoped_to_the_year():
    frame = build()
    scoped = yc.year_rows(frame, ["target"], ["driver"])
    # The mistake: demean once over everything, then split.
    key = yc.control_key(frame)
    pooled = frame.assign(
        driver=yc.ad.demean_by(frame["driver"], key),
        target=yc.ad.demean_by(frame["target"], key))
    wrong = []
    for year, block in pooled.groupby(pooled["hour"].dt.year):
        rho, n, _ = yc.ad.spearman(block["driver"], block["target"])
        wrong.append(rho)
    right = scoped[scoped["year"].isin([2024, 2025, 2026])]["rho"].tolist()
    gap = max(abs(a - b) for a, b in zip(sorted(right), sorted(wrong)))
    check("per-year demeaning is scoped to the year", gap > 0.01,
          f"scoped {[round(r, 4) for r in right]} vs "
          f"pooled-then-split {[round(r, 4) for r in wrong]} (gap {gap:.4f})")


def check_stable_sign_is_not_flagged():
    frame = build()
    per_year = yc.year_rows(frame, ["target"], ["driver"])
    flagged = yc.sign_changes(per_year)
    check("a driver with one sign in every year is not flagged",
          flagged.empty or "driver" not in set(flagged["predictor"]),
          f"{len(flagged)} flagged")


def check_real_reversal_is_graded_as_a_reversal():
    frame = build(flip_second_year=True)
    per_year = yc.year_rows(frame, ["target"], ["driver"])
    flagged = yc.sign_changes(per_year)
    row = flagged[flagged["predictor"] == "driver"]
    check("a genuine reversal is flagged",
          not row.empty,
          "" if not row.empty else "nothing flagged")
    if not row.empty:
        check("a genuine reversal is graded REVERSAL",
              row.iloc[0]["verdict"].startswith("REVERSAL"),
              row.iloc[0]["verdict"])


def check_noise_flip_is_graded_as_unstable():
    """A predictor wobbling around zero must not be graded a reversal."""
    frame = build()
    per_year = yc.year_rows(frame, ["target"], ["nothing"])
    flagged = yc.sign_changes(per_year)
    row = flagged[flagged["predictor"] == "nothing"]
    if row.empty:
        check("a noise predictor is not graded REVERSAL", True,
              "it did not change sign in this draw")
        return
    check("a noise predictor is not graded REVERSAL",
          not row.iloc[0]["verdict"].startswith("REVERSAL"),
          row.iloc[0]["verdict"])


def check_underpowered_years_never_trigger_a_flag():
    """A year too thin for spearman must not be half of a sign change."""
    frame = build()
    thin = pd.concat([frame[frame["hour"].dt.year != 2026],
                      frame[frame["hour"].dt.year == 2026].head(5)])
    per_year = yc.year_rows(thin, ["target"], ["driver", "nothing"])
    small = per_year[(per_year["year"] == 2026)
                     & (per_year["n"] < yc.YEAR_MIN_N)]
    check("an under-powered year is reported but never believed",
          not small.empty and small["rho"].isna().all(),
          f"{len(small)} thin cells, rho all NaN: {small['rho'].isna().all()}")
    flagged = yc.sign_changes(per_year)
    check("an under-powered year cannot cause a flag",
          flagged.empty or not flagged["years"].str.contains("2026").any(),
          f"{len(flagged)} flagged")


def check_year_dummies_omit_the_baseline():
    frame = build()
    baseline, names = yc.add_year_dummies(frame)
    check("year dummies omit the baseline year",
          baseline == 2024 and names == ["year_2025", "year_2026"],
          f"baseline {baseline}, dummies {names}")
    check("each year dummy marks only its own year",
          bool((frame.loc[frame["hour"].dt.year == 2025, "year_2025"] == 1).all()
               and (frame.loc[frame["hour"].dt.year != 2025,
                              "year_2025"] == 0).all()))


def main():
    print("analyze_year_controls offline checks\n")
    check_per_year_demeaning_is_scoped_to_the_year()
    check_stable_sign_is_not_flagged()
    check_real_reversal_is_graded_as_a_reversal()
    check_noise_flip_is_graded_as_unstable()
    check_underpowered_years_never_trigger_a_flag()
    check_year_dummies_omit_the_baseline()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
