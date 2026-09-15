"""compare_wave_sources holds the hour set and the baseline fixed.

The bug this guards against is the one the deck already has: comparing two
wave sources that were measured on different hours and reading the difference
as a property of the instruments. It has a second, quieter form -- keeping the
row set matched but demeaning each column against the whole frame, so the two
residuals are taken against different baselines even though the rows line up.

The fixture makes both detectable. On the matched hours the two sources are
byte-identical, so any correct implementation must report exactly the same rho
for them and a ratio of 1.0. The unmatched hours hold the same
relationship measured far more cleanly, so an implementation that lets them
leak in -- through the row set or through the group means -- reports a
stronger correlation out of nothing but sampling.

    python test_wave_sources_offline.py
"""

import sys

import numpy as np
import pandas as pd

import compare_wave_sources as cws


def fixture():
    """400 hours. Matched half: sources identical. Unmatched half: cleaner."""
    rng = np.random.default_rng(11)
    n = 400
    hours = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    frame = pd.DataFrame({"hour": hours})
    frame["hour_of_day"] = frame["hour"].dt.hour
    frame["month"] = frame["hour"].dt.month

    # The buoy's gaps are neither random nor evenly spread across the control
    # cells. Two things have to be true for the scoping check to have teeth,
    # and both are true of the real station: the hours it misses sit at a
    # different sea state (it drops out in the weather that matters), and it
    # misses them at different rates in different hours of the day. A uniform
    # offset would not do -- it shifts every cell mean by the same amount, the
    # ranks inside the matched rows never move, and the check passes on a
    # rounding difference while guarding nothing.
    keep_rate = np.where(frame["hour_of_day"] % 4 == 0, 0.15, 0.85)
    matched_mask = rng.random(n) < keep_rate
    wave = rng.normal(2.0, 0.5, n) + np.where(matched_mask, 0.0, 1.8)
    # A deliberate hour-of-day cycle in the target, so the control has work to
    # do and a test that skipped demeaning would not accidentally pass.
    cycle = np.sin(frame["hour_of_day"] / 24 * 2 * np.pi)

    matched = matched_mask

    # Same slope in both halves, but the unmatched hours are far less noisy.
    # Giving them a DIFFERENT slope would not work: pooling two regimes that
    # disagree drives a rank correlation down, not up, and the fixture would
    # be asserting the opposite of the artefact it means to demonstrate. A
    # cleaner sample of the same relationship is the honest version of "the
    # extra hours flatter the number".
    noise = np.where(matched, rng.normal(0, 0.60, n), rng.normal(0, 0.05, n))
    target = 0.4 * wave + cycle + noise

    frame["mop_wave_height"] = wave
    frame["WVHT"] = np.where(matched, wave, np.nan)
    frame["bbox_area_max"] = target
    return frame, matched


def check_matched_sources_agree_exactly():
    frame, matched = fixture()
    both = frame[matched]

    buoy = cws.controlled(both, "WVHT", "bbox_area_max")
    mop = cws.controlled(both, "mop_wave_height", "bbox_area_max")

    assert buoy[1] == mop[1] == int(matched.sum()), (buoy[1], mop[1])
    assert abs(buoy[0] - mop[0]) < 1e-12, (
        "identical columns on identical rows must give identical rho; got "
        f"{buoy[0]} and {mop[0]}")
    print(f"  matched: buoy rho={buoy[0]:.4f}  mop rho={mop[0]:.4f}  "
          f"ratio={mop[0] / buoy[0]:.2f}x  n={buoy[1]}")


def check_all_hours_differs_from_matched():
    """The reference row has to be a different number, or it says nothing."""
    frame, matched = fixture()
    both = frame[matched]
    mop_all = frame[frame["mop_wave_height"].notna()]

    on_matched = cws.controlled(both, "mop_wave_height", "bbox_area_max")
    on_all = cws.controlled(mop_all, "mop_wave_height", "bbox_area_max")

    assert on_all[1] == 400, on_all[1]
    assert on_all[0] > on_matched[0] + 0.05, (
        "the fixture measures the unmatched half more cleanly, so all-hours "
        f"must exceed matched; got {on_all[0]} vs {on_matched[0]}")
    print(f"  mop matched rho={on_matched[0]:.4f} n={on_matched[1]}   "
          f"all hours rho={on_all[0]:.4f} n={on_all[1]}")


def check_demeaning_is_scoped_to_the_subset():
    """Demean over the whole frame and the matched answer moves.

    Not a style point. The unmatched hours have their own mean in every
    hour-by-month cell, so borrowing the full-frame cell means shifts the
    matched residuals by an amount that has nothing to do with the matched
    hours. If this assertion ever fails because the two agree, the subsetting
    has been hoisted out of the demeaning and the script is quietly measuring
    the thing it was written to avoid.
    """
    frame, matched = fixture()
    both = frame[matched]

    scoped = cws.controlled(both, "mop_wave_height", "bbox_area_max")[0]

    key = (frame["hour_of_day"].astype(str) + "-" + frame["month"].astype(str))
    wide_x = cws.ad.demean_by(frame["mop_wave_height"], key)[matched]
    wide_y = cws.ad.demean_by(frame["bbox_area_max"], key)[matched]
    unscoped = cws.ad.spearman(wide_x, wide_y)[0]

    assert abs(scoped - unscoped) > 0.01, (
        "demeaning within the matched subset must not equal demeaning across "
        f"the full frame; both gave {scoped}")
    print(f"  scoped rho={scoped:.4f}   full-frame baseline rho={unscoped:.4f}")


def check_rows_for_covers_every_present_target():
    frame, matched = fixture()
    rows = cws.rows_for(frame[matched], "buoy", "WVHT",
                        ["bbox_area_max", "not_a_column"])
    assert len(rows) == 1, rows
    assert rows[0]["source"] == "buoy"
    assert rows[0]["target"] == "bbox_area_max"
    print(f"  rows_for skipped the absent target, kept {rows[0]['target']}")


if __name__ == "__main__":
    for check in (check_matched_sources_agree_exactly,
                  check_all_hours_differs_from_matched,
                  check_demeaning_is_scoped_to_the_subset,
                  check_rows_for_covers_every_present_target):
        print(check.__name__)
        check()
    print("\nALL PASS")
    sys.exit(0)
