"""Offline checks for analyze_static_detections.py.

The fixture is built with a known answer: 60% of detections pinned to one
pixel-tight spot (a jetty), the rest scattered across the water and rising
with the surf. The method must find the pinned cell, report its share close to
60%, measure its spread as near zero, and show it decoupled from wave height
while the scattered detections are coupled.

A fixture where everything is scattered is included too, because a
static-detection finder that always finds one is useless.
"""

import io
import contextlib
import os
import sys
import tempfile

import numpy as np
import pandas as pd

import analyze_drivers as ad
import analyze_static_detections as sd

FAILURES = []

JETTY_X, JETTY_Y = 1200.0, 1500.0
DAYS = 200


def check(name, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    if not condition:
        FAILURES.append(name)
    print(f"  {mark} {name}" + (f"  {detail}" if detail else ""))


def build(pinned_share, rng):
    """Detections over DAYS days. Waves drive only the scattered ones."""
    waves = 0.5 + rng.gamma(2.0, 0.35, DAYS)
    rows = []
    for index in range(DAYS):
        day = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=index)
        # The jetty fires a fixed number every day, whatever the sea does.
        pinned = int(round(12 * pinned_share / max(1 - pinned_share, 0.01)))
        for k in range(pinned):
            rows.append({
                "timestamp": day + pd.Timedelta(minutes=k),
                "score_classes": "rip_current", "score_max": 0.82,
                "bbox_area_max": 4000.0,
                # Sub-pixel jitter only: a groyne does not move.
                "bbox_x": JETTY_X + rng.normal(0, 0.4),
                "bbox_y": JETTY_Y + rng.normal(0, 0.4)})
        # Real rips: count rises with the surf, position wanders widely.
        scattered = int(round(4 + 6 * waves[index]))
        for k in range(scattered):
            rows.append({
                "timestamp": day + pd.Timedelta(minutes=30 + k),
                "score_classes": "rip_current", "score_max": 0.6,
                "bbox_area_max": 2500.0,
                "bbox_x": rng.uniform(200, 2400),
                "bbox_y": rng.uniform(900, 1900)})
    frame = pd.DataFrame(rows)
    conditions = pd.DataFrame(
        {"wave_height": waves},
        index=pd.date_range("2026-01-01", periods=DAYS, freq="D", tz="UTC"))
    conditions.index.name = "date"
    return frame, conditions


def check_a_pinned_cell_is_found_and_measured():
    rng = np.random.default_rng(3)
    frame, conditions = build(0.6, rng)

    tally, cell_key = sd.grid_counts(frame, 64)
    hottest = tally.index[0]
    share = tally.iloc[0] / len(frame)

    check("the busiest cell is the one the jetty was put in",
          hottest == (int(JETTY_X // 64), int(JETTY_Y // 64)), str(hottest))
    check("its share is near the 60% it was built with",
          0.5 < share < 0.7, f"{share:.1%}")

    sx, sy = sd.spread_within(frame, cell_key, hottest, 64)
    check("the centroids inside it barely move", sx < 1.0 and sy < 1.0,
          f"{sx:.2f} x {sy:.2f} px")
    check("tightness is near zero for a pinned cluster",
          sd.tightness(sx, sy, 64) < 0.05, f"{sd.tightness(sx, sy, 64):.3f}")
    check("and near 1.0 for centroids spread evenly across a cell",
          0.85 < sd.tightness(18.1, 18.4, 64) < 1.15,
          f"{sd.tightness(18.1, 18.4, 64):.2f}")
    check("Virginia Beach's real 18.1 x 14.2 reads as no clustering",
          sd.tightness(18.1, 14.2, 64) > 0.75,
          f"{sd.tightness(18.1, 14.2, 64):.2f}")
    check("Panama City west's real 10.2 x 5.5 reads as pinned",
          sd.tightness(10.2, 5.5, 64) < 0.55,
          f"{sd.tightness(10.2, 5.5, 64):.2f}")

    coupling = sd.condition_coupling(frame, cell_key, hottest, conditions)
    check("the coupling test ran", coupling is not None)
    check("the pinned cell is not coupled to wave height",
          abs(coupling["hot"][0]) < 0.2, f"rho {coupling['hot'][0]:+.3f}")
    check("while the rest of the scene is",
          coupling["rest"][0] > 0.5, f"rho {coupling['rest'][0]:+.3f}")
    check("and the contrast is what the report keys on",
          abs(coupling["hot"][0]) + 0.1 < abs(coupling["rest"][0]))

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rows = sd.report("Fake Cam", frame, 64, 5, conditions)
    text = buffer.getvalue()
    check("the report shouts when one cell dominates",
          "ONE CELL CARRIES" in text, text.strip()[:120])
    check("it says what that does to detection_rate",
          "was that feature visible" in text)
    check("it notes the centroids do not move", "does not move" in text)
    check("and reports tightness against the even-scatter baseline",
          "tight" in text and "% of what an evenly" in text,
          text.strip()[-200:])
    check("and that the cell is decoupled from the surf",
          "less coupled to the surf" in text)
    check("rows are returned for the csv", len(rows) == 5 and
          rows[0]["share"] > 0.5, str(len(rows)))
    check("p-value is printed as a p-value, not as n",
          "p 0" in text or "p 1" in text or "e-" in text, text.strip()[-200:])


def check_a_scattered_camera_is_not_accused():
    """A finder that always finds a static detection is worth nothing."""
    rng = np.random.default_rng(11)
    frame, conditions = build(0.0, rng)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        sd.report("Fake Cam", frame, 64, 5, conditions)
    text = buffer.getvalue()
    check("a scattered camera is reported as scattered",
          "no cell dominates" in text, text.strip()[-160:])
    check("and is not accused of a fixed object",
          "ONE CELL CARRIES" not in text)
    check("no coupling verdict is issued about a cell noise happened to "
          "favour", "less coupled to the surf" not in text
          and "daily count vs wave height" not in text,
          text.strip()[-160:])


def check_only_the_named_class_is_counted():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
        "score_classes": ["rip_current", "person", "boat,person",
                          "rip_current"],
        "bbox_x": [100.0, 200.0, 300.0, 400.0],
        "bbox_y": [100.0, 200.0, 300.0, 400.0],
        "score_max": [0.8] * 4, "bbox_area_max": [1000.0] * 4,
    })
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "rip_test.csv")
        frame.to_csv(path, index=False)
        kept = sd.rip_frames(path)
        check("object-class frames are excluded from the grid",
              len(kept) == 2, f"{len(kept)} rows")
        check("and the rip frames survive",
              set(kept["bbox_x"]) == {100.0, 400.0}, str(list(kept["bbox_x"])))

        frame["bbox_x"] = np.nan
        frame.to_csv(path, index=False)
        check("a record with no usable centroid returns None rather than "
              "an empty grid", sd.rip_frames(path) is None)

        # One file on disk really does lack a timestamp column, and assuming
        # the schema killed the run on the first camera reached.
        bare = os.path.join(folder, "rip_bare.csv")
        pd.DataFrame({"detected": [True, False],
                      "whatever": [1, 2]}).to_csv(bare, index=False)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = sd.rip_frames(bare)
        check("a record with no timestamp column is skipped, not fatal",
              result is None)
        check("and the missing column is named",
              "no timestamp" in buffer.getvalue(), buffer.getvalue().strip())
        check("along with what the file does have",
              "whatever" in buffer.getvalue())

        no_class = os.path.join(folder, "rip_noclass.csv")
        pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="h",
                                       tz="UTC"),
            "bbox_x": [10.0, 20.0], "bbox_y": [10.0, 20.0]}).to_csv(
                no_class, index=False)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = sd.rip_frames(no_class)
        check("a record with no score_classes is used but flagged",
              result is not None and len(result) == 2)
        check("with a note that the classes cannot be separated",
              "cannot separate" in buffer.getvalue())


def main():
    print("static detection offline checks\n")
    check_a_pinned_cell_is_found_and_measured()
    check_a_scattered_camera_is_not_accused()
    check_only_the_named_class_is_counted()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
