"""Is Walton's glare result about the camera, or about where the sun is?

OPEN #1, test (b). The detection-rate result on file is that `sun_in_view` and
`sun_glare` add dR2 +0.0325 (detection rate) and +0.0381 (detections) over a
model already holding wave height and period, tide, wind, temperature, rain,
solar elevation and cloud. Three stories fit that equally well:

  * LENS. Sun in front of the camera degrades the image, and the detector fires
    more (or less) on a degraded image. About the camera.
  * SEA BREEZE. Afternoon wind builds afternoon surf. About the ocean.
  * TIME OF DAY. Something else entirely that runs on a daily cycle.

This script runs three tests, in increasing order of what they can rule out.

(1) THE FRONT/BEHIND SPLIT, as asked: classify each hour by whether the sun's
    azimuth is within 90 deg of the camera's seaward bearing, and refit the
    bearing terms in each group. Two cautions travel with it and are printed
    beside the numbers rather than in a footnote:

    - `sun_glare` is IDENTICALLY ZERO in the sun-behind group. glare_index
      clips sun_in_view at zero, so behind the camera the column is a constant,
      standardized_ols drops it, and "the bearing pair" is one term there. The
      F-test below counts the terms that survived instead of assuming two, and
      computes the p-value for whatever df1 that leaves.
    - The split is CONFOUNDED WITH TIME OF DAY. At a bearing of 206 deg the sun
      is in front around the middle of the day and behind it morning and
      evening, so "front" is largely "afternoon" -- which is also the sea-breeze
      story. A front-heavy result is therefore consistent with BOTH stories the
      split was meant to separate. The hour-of-day distribution of each group is
      printed so the overlap is visible.

(2) THE SIGNED PARAMETERISATION. Same two terms, with the clip removed from the
    glare index so it runs negative behind the camera instead of flat. Both
    groups then carry two terms with variance and the two halves are comparable.
    In the sun-in-front group this is IDENTICAL to (1) by construction -- the
    clip never binds there -- so the front number can be read straight against
    the published one.

(3) THE BEARING ROTATION, which is the test that actually separates the stories.
    cos(az - b) is an exact linear combination of cos(az) and sin(az): the
    identity is cos(az)cos(b) + sin(az)sin(b), so the fitted R2 of sun_in_view
    on the azimuth harmonics is 1.000 and the coefficients ARE cos(b), sin(b).
    That is checked and printed below. It means the published comparator, which
    holds no azimuth at all, cannot tell "this camera's bearing matters" from
    "the sun's horizontal position matters" -- every bearing buys something.
    What distinguishes them is WHICH bearing buys the most. So the same F-test
    is rerun with the geometry recomputed against every bearing on a grid, and
    the peak is reported next to the camera's real one. A peak at 206 deg is the
    lens story. A peak near due west is an afternoon story with no camera in it.

Runs on variant A -- rip classes only, re-censored to one 0.70 floor -- built by
analyze_walton_eras.py, so the numbers sit directly beside that run's.

    python analyze_glare_split.py
    python analyze_glare_split.py --variant pooled --bearing-step 5
"""

import argparse
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

import analyze_drivers as ad
import analyze_glare as ag
import analyze_walton_eras as we
import build_label_sample as bls
import solar

TARGETS = ["detection_rate", "detections", "bbox_area_max"]
DESCRIPTIVE = we.DESCRIPTIVE
SCORE_CAVEAT = we.SCORE_CAVEAT
FRONT_DEG = 90.0

SIGNED_GLARE = "sun_glare_signed"
BEARING_TERMS = ["sun_in_view", "sun_glare"]
SIGNED_TERMS = ["sun_in_view", SIGNED_GLARE]


# ---------------------------------------------------------------------------
# an F-test that does not assume how many terms were added
# ---------------------------------------------------------------------------

def _betacf(a, b, x, iterations=300, eps=3e-16, tiny=1e-300):
    """Continued fraction for the incomplete beta, by modified Lentz."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        step = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + step * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + step / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        step = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + step * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + step / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_beta(a, b, x):
    """I_x(a, b), the regularized incomplete beta."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def f_survival(f_stat, df1, df2):
    """P(F > f_stat). Exact for any df1, unlike the df1==2 special case.

    analyze_glare.added_term_test returns NaN for df1 != 2, which is fine while
    the bearing terms always arrive in pairs. They do not here: behind the
    camera sun_glare is a constant and gets dropped, leaving one term, and a
    test that cannot report a p-value for that case would silently go blank on
    exactly the half of the data the question is about.
    """
    if not np.isfinite(f_stat) or df1 <= 0 or df2 <= 0:
        return float("nan")
    if f_stat <= 0:
        return 1.0
    return regularized_beta(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f_stat))


def added_term_test(small, big):
    """(dR2, F, p, added, df2) counting the terms that actually survived.

    `added` is read off the fitted names rather than assumed, because
    standardized_ols drops zero-variance and duplicate columns. Assuming two
    where one survived divides the gain by the wrong number AND uses the wrong
    df1, which moves the p-value by orders of magnitude.
    """
    if not small or not big or small.get("beta") is None or big.get("beta") is None:
        return None
    added = len(big["names"]) - len(small["names"])
    n = big["n"]
    if n != small["n"]:
        return {"dR2": float("nan"), "F": float("nan"), "p": float("nan"),
                "added": added, "n": n, "df2": float("nan"),
                "note": f"different n ({small['n']} vs {n}); not nested"}
    if added < 1:
        return {"dR2": big["r2"] - small["r2"], "F": float("nan"),
                "p": float("nan"), "added": added, "n": n,
                "df2": float("nan"),
                "note": "no term survived the variance filter"}
    gain = big["r2"] - small["r2"]
    df2 = n - len(big["names"]) - 1
    if df2 <= 0:
        return {"dR2": gain, "F": float("nan"), "p": float("nan"),
                "added": added, "n": n, "df2": df2,
                "note": f"only {n} rows for {len(big['names'])} predictors; "
                        "no residual degrees of freedom left"}
    if big["r2"] >= 1:
        return {"dR2": gain, "F": float("nan"), "p": float("nan"),
                "added": added, "n": n, "df2": df2,
                "note": "the model fits perfectly (R2 = 1), so there is no "
                        "residual to test against — check for a predictor "
                        "that IS the target"}
    f_stat = (gain / added) / ((1 - big["r2"]) / df2)
    return {"dR2": gain, "F": f_stat, "p": f_survival(f_stat, added, df2),
            "added": added, "n": n, "df2": df2, "note": ""}


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def angular_gap(azimuth, bearing):
    """|azimuth - bearing| folded onto 0-180 degrees."""
    delta = (pd.to_numeric(azimuth, errors="coerce") - bearing).abs() % 360.0
    return delta.where(delta <= 180.0, 360.0 - delta)


def signed_glare(elevation, azimuth, bearing):
    """glare_index with the clip removed, so it runs negative behind.

    glare_index zeroes the in-view component below zero, which is right for a
    specular-glare index and fatal for a split that wants to compare the two
    sides: the column is then constant behind the camera and carries no
    information at all. Unclipped, the same functional form measures the
    mirrored geometry -- the sun's position relative to a direction the camera
    is NOT looking -- which is the placebo this comparison needs.
    """
    elev = pd.to_numeric(elevation, errors="coerce")
    facing = solar.sun_in_view(azimuth, bearing)
    above = np.where(elev > 0.0,
                     np.cos(np.radians(np.clip(elev, 0.0, 90.0))), 0.0)
    return facing * above


def attach_geometry(frame, bearing, suffix=""):
    """sun_in_view / sun_glare / signed glare against `bearing`, in place."""
    frame[f"sun_in_view{suffix}"] = solar.sun_in_view(frame["solar_azimuth"],
                                                      bearing)
    frame[f"sun_glare{suffix}"] = solar.glare_index(frame["solar_elevation"],
                                                    frame["solar_azimuth"],
                                                    bearing)
    frame[f"{SIGNED_GLARE}{suffix}"] = signed_glare(frame["solar_elevation"],
                                                    frame["solar_azimuth"],
                                                    bearing)
    return frame


def harmonic_check(frame, bearing):
    """R2 of sun_in_view on the azimuth harmonics, which must be 1.

    Printed rather than asserted because it is the whole reason test (3)
    exists: if this is 1.000 then sun_in_view adds nothing a model already
    holding cos(az) and sin(az) does not have, and the published comparator
    holds neither.
    """
    usable = frame[["solar_azimuth", "sun_in_view"]].dropna()
    if len(usable) < 10:
        return None
    radians = np.radians(usable["solar_azimuth"].to_numpy(float))
    design = np.column_stack([np.ones(len(usable)), np.cos(radians),
                              np.sin(radians)])
    target = usable["sun_in_view"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    # errstate because some BLAS builds raise spurious divide/overflow flags on
    # this product; the result is checked for finiteness rather than trusted.
    with np.errstate(all="ignore"):
        fitted = design @ beta
    if not np.isfinite(fitted).all() or not np.isfinite(beta).all():
        return None
    ss_res = float(((target - fitted) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return {"r2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
            "cos_beta": float(beta[1]), "sin_beta": float(beta[2]),
            "cos_expected": math.cos(math.radians(bearing)),
            "sin_expected": math.sin(math.radians(bearing))}


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def base_columns(frame):
    height, period = ag.wave_columns(frame)
    return [c for c in [height, period] + ag.BASE_OTHER if c in frame.columns], height


def bearing_test(frame, target, base, extras, have_cloud):
    """The F-test for `extras` over solar elevation (and cloud) alone."""
    cloud = [ag.CLOUD_COLUMN] if have_cloud else []
    small = ad.standardized_ols(frame, target,
                                base + ["solar_elevation"] + cloud)
    big = ad.standardized_ols(
        frame, target,
        base + ["solar_elevation"] + cloud
        + [c for c in extras if c in frame.columns])
    return added_term_test(small, big)


def variance_report(frame, columns):
    """Which of `columns` actually varies here, and which is a constant."""
    out = {}
    for column in columns:
        if column not in frame.columns:
            out[column] = None
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        out[column] = float(values.std()) if len(values) else float("nan")
    return out


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def fmt_p(value):
    if value is None or pd.isna(value):
        return "       —"
    if value == 0:
        return "<1e-300"
    return f"{value:.2g}"


def print_test_row(label, target, result, mark=""):
    if result is None:
        print(f"  {label:<29}{target:<16}  could not fit")
        return
    f_stat = "—" if pd.isna(result["F"]) else f"{result['F']:.2f}"
    print(f"  {label:<29}{target:<16}{result['n']:>7}{result['added']:>7}"
          f"{result['dR2']:>+10.4f}{f_stat:>10}{fmt_p(result['p']):>12}{mark}")
    if result.get("note"):
        print(f"  {'':<45}{result['note']}")


HEADER = (f"  {'scope':<15}{'group':<14}{'target':<16}{'n':>7}{'terms':>7}"
          f"{'dR2':>10}{'F':>10}{'p':>12}")


def run_scoped(groups, base, terms, have_cloud):
    """The same F-test on all hours and on daylight hours, for each group.

    Both, because the sun-behind group is mostly darkness at this latitude and
    bearing: a null computed over hours with no sun in them is not evidence
    about what the sun does.
    """
    print(f"\n{HEADER}")
    out = {}
    for scope, keep in (("all hours", None), ("daylight only", "daylight")):
        for label, group in groups:
            subset = group if keep is None else group[group[keep]]
            for target in TARGETS + [DESCRIPTIVE]:
                if target not in subset.columns:
                    continue
                result = bearing_test(subset, target, base, terms, have_cloud)
                out[(scope, label, target)] = result
                print_test_row(f"{scope:<15}{label}", target, result,
                               "   <- DESCRIPTIVE ONLY"
                               if target == DESCRIPTIVE else "")
        print()
    return out


# ---------------------------------------------------------------------------

def build_variant(args, sites, camera, slug):
    """Variant A (or pooled) from analyze_walton_eras, with solar attached."""
    split = pd.Timestamp(args.split, tz="UTC")
    we.SPLIT_HOURS.clear()
    we.SPLIT_HOURS.append(split)

    frames = bls.load_frames(slug)
    wanted = tuple(c.strip() for c in args.classes.split(",") if c.strip())
    frames, dropped = bls.keep_class(frames, wanted=wanted, label="")
    print(f"  {len(frames)} frames after the class filter ({dropped} dropped)")
    frames["detected"] = we.as_bool(frames["detected"])

    all_below, some_below = we.post_era_violations(frames, split, args.floor)
    if len(all_below) or len(some_below):
        sys.exit(f"  {len(all_below) + len(some_below)} post-era detections "
                 f"sit under {args.floor:.2f}; run analyze_walton_eras.py "
                 "first and resolve that before splitting anything.")

    if args.variant == "A":
        table, stats = we.censor(frames, split, args.floor)
        print(f"  variant A: {stats['demoted']} pre-era frames demoted to "
              f"observed zeros, {stats['residual_frames']} residual")
    else:
        table = frames
        print("  variant pooled: class filter only, no floor correction")

    workspace = args.keep_workspace or tempfile.mkdtemp(prefix="glare_split_")
    os.makedirs(workspace, exist_ok=True)
    mirror = we.build_variant(workspace, args.variant, table, slug)
    if mirror is None:
        sys.exit("  no hours in the variant")
    built = we.assemble(mirror, sites, slug, camera)
    if not args.keep_workspace:
        shutil.rmtree(workspace, ignore_errors=True)
    return built


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default=we.DEFAULT_CAMERA)
    parser.add_argument("--split", default=we.DEFAULT_SPLIT)
    parser.add_argument("--floor", type=float, default=we.DEFAULT_FLOOR)
    parser.add_argument("--classes", default=",".join(bls.RIP_CLASSES))
    parser.add_argument("--variant", default="A", choices=["A", "pooled"],
                        help="A = re-censored to one floor (default)")
    parser.add_argument("--front-deg", type=float, default=FRONT_DEG,
                        help=f"sun is 'in front' within this many degrees of "
                             f"the bearing (default {FRONT_DEG:.0f})")
    parser.add_argument("--bearing-step", type=int, default=10,
                        help="degrees between rotation-sweep bearings (default 10)")
    parser.add_argument("--keep-workspace", default=None)
    args = parser.parse_args()

    sites = ad.load_sites()
    camera, slug = we.resolve_camera(sites, args.camera)
    name = camera["camera_name"]

    print(f"\n{'=' * 78}")
    print(f"IS THE GLARE RESULT ABOUT THE CAMERA? — {name}")
    print("=" * 78)
    built = build_variant(args, sites, camera, slug)
    if built is None:
        sys.exit("  nothing assembled")

    observed = built["observed"]
    bearing = built["bearing"]
    if bearing is None:
        sys.exit("  no shore normal configured; there is no 'in front' to split on")
    have_cloud = built["cloud"]
    base, height = base_columns(observed)
    attach_geometry(observed, bearing)
    observed["sun_gap_deg"] = angular_gap(observed["solar_azimuth"], bearing)
    observed["daylight"] = observed["solar_elevation"] > 0

    print(f"\n  {len(observed)} hours, {observed['hour'].min():%Y-%m-%d} to "
          f"{observed['hour'].max():%Y-%m-%d}")
    print(f"  seaward bearing {bearing:.1f} deg; wave column {height}; "
          f"cloud {'present' if have_cloud else 'ABSENT'}")
    print(f"  base held fixed in every fit below:\n    {', '.join(base)}")
    print(f"  {DESCRIPTIVE} is DESCRIPTIVE ONLY: {SCORE_CAVEAT}")

    # Local solar-ish time, so the bands below do not wrap across midnight.
    # A quartile of UTC hour is meaningless here: Walton's sun is in front from
    # about 17:00 to 02:00 UTC, and a linear IQR of that reads "2 to 21",
    # which describes neither end of it.
    observed["local_hour"] = (observed["hour_of_day"]
                              + float(camera["lon"]) / 15.0) % 24.0
    front = observed["sun_gap_deg"] <= args.front_deg
    groups = [("sun-in-front", observed[front]), ("sun-behind", observed[~front])]

    # ------------------------------------------------------------------
    print(f"\n{'-' * 78}")
    print("THE SPLIT, AND WHAT IT IS CONFOUNDED WITH")
    print("-" * 78)
    print(f"  'in front' = sun azimuth within {args.front_deg:.0f} deg of "
          f"{bearing:.1f} deg.")
    bands = [(0, 6, "night"), (6, 9, "06-09"), (9, 12, "09-12"),
             (12, 15, "12-15"), (15, 18, "15-18"), (18, 24, "18-24")]
    print(f"\n  {'group':<14}{'hours':>7}{'daylight':>10}   "
          + "".join(f"{name:>8}" for _, _, name in bands)
          + "   (local hours, daylight only)")
    for label, group in groups:
        if group.empty:
            print(f"  {label:<14}      0")
            continue
        lit = group[group["daylight"]]
        counts = []
        for low, high, _ in bands:
            counts.append(int(((lit["local_hour"] >= low)
                               & (lit["local_hour"] < high)).sum()))
        print(f"  {label:<14}{len(group):>7}{group['daylight'].mean():>9.0%}   "
              + "".join(f"{c:>8}" for c in counts))
    print("\n  Two problems live in that table, and both limit what the split")
    print("  can say:")
    print("    * TIME OF DAY. If the lit hours of the two groups sit in")
    print("      different bands, the split is a split on morning-vs-afternoon")
    print("      as much as on camera geometry — and afternoon is also the")
    print("      sea-breeze story it was meant to rule out.")
    print("    * DARKNESS. Wherever the daylight share is low, most of that")
    print("      group has no sun in it at all, so a null there is partly just")
    print("      night. That is why every test below is run twice.")

    # ------------------------------------------------------------------
    print(f"\n{'-' * 78}")
    print("(1) THE BEARING TERMS AS PUBLISHED, PER GROUP")
    print("-" * 78)
    for label, group in groups:
        spread = variance_report(group, BEARING_TERMS)
        flat = [c for c, s in spread.items() if s is not None and s < 1e-12]
        detail = ", ".join(f"{c} sd {s:.4f}" for c, s in spread.items()
                           if s is not None)
        print(f"  {label}: {detail}")
        if flat:
            print(f"    {' and '.join(flat)} is CONSTANT here and will be "
                  "dropped — the\n    'pair' is not a pair in this group, and "
                  "the terms column below says so.")
    published = run_scoped(groups, base, BEARING_TERMS, have_cloud)

    # ------------------------------------------------------------------
    print(f"\n{'-' * 78}")
    print("(2) THE SAME TERMS, SIGNED, SO BOTH GROUPS CARRY TWO")
    print("-" * 78)
    print("  The glare index without its clip at zero. In sun-in-front this is")
    print("  the SAME COLUMN as (1) -- the clip never binds there -- so that")
    print("  row must reproduce (1) exactly. In sun-behind it is the mirrored")
    print("  geometry rather than a constant.")
    signed = run_scoped(groups, base, SIGNED_TERMS, have_cloud)

    # ------------------------------------------------------------------
    print(f"\n{'-' * 78}")
    print("(3) WHICH BEARING BUYS THE MOST")
    print("-" * 78)
    check = harmonic_check(observed, bearing)
    if check:
        print(f"  First, the identity this test rests on. Regressing "
              f"sun_in_view on\n  cos(azimuth) and sin(azimuth):")
        print(f"    R2 = {check['r2']:.6f}   "
              f"coefficients {check['cos_beta']:+.4f}, {check['sin_beta']:+.4f}")
        print(f"    cos({bearing:.0f}), sin({bearing:.0f}) = "
              f"{check['cos_expected']:+.4f}, {check['sin_expected']:+.4f}")
        if check["r2"] > 0.999999:
            print("  R2 is 1: sun_in_view IS the azimuth harmonics rotated to"
                  " this bearing.\n  A comparator holding no azimuth therefore"
                  " cannot separate 'this camera's\n  bearing matters' from"
                  " 'the sun's horizontal position matters'. Every\n  bearing"
                  " will buy something. The question is which buys most.")
    print("\n  ONE CAVEAT ON READING THE PEAK. Because cos(az-b) = -cos(az-b-180),"
          "\n  the linear term's contribution is symmetric under b -> b+180: it"
          " identifies\n  an AXIS, not a direction, and will show twin peaks "
          "180 deg apart. Only the\n  clipped, elevation-weighted glare term "
          "breaks that symmetry. So the column\n  to read is the gap to the "
          "camera's AXIS (folded onto 0-90); the gap to the\n  bearing itself "
          "is reported beside it and is the weaker claim.")

    bearings = list(range(0, 360, max(1, args.bearing_step)))
    sweep = observed.copy()
    rows = []
    for candidate in bearings:
        attach_geometry(sweep, float(candidate), suffix="_b")
        row = {"bearing": candidate}
        for target in TARGETS + [DESCRIPTIVE]:
            if target not in sweep.columns:
                continue
            result = bearing_test(sweep, target, base,
                                  ["sun_in_view_b", "sun_glare_b"], have_cloud)
            row[target] = result["dR2"] if result else float("nan")
        rows.append(row)
    table = pd.DataFrame(rows)

    print(f"\n  dR2 against the assumed bearing, whole record, {len(bearings)}"
          " bearings:")
    print(f"\n  {'bearing':>8}" + "".join(f"{t:>17}" for t in TARGETS
                                          + [DESCRIPTIVE]))
    for _, row in table.iterrows():
        marks = []
        if abs(row["bearing"] - bearing) <= args.bearing_step / 2:
            marks.append("<- the camera")
        if abs((row["bearing"] - (bearing + 180.0)) % 360.0) <= args.bearing_step / 2:
            marks.append("<- behind it")
        print(f"  {int(row['bearing']):>8}"
              + "".join(f"{row[t]:>+17.4f}" for t in TARGETS + [DESCRIPTIVE]
                        if t in row)
              + ("   " + " ".join(marks) if marks else ""))

    lit = observed[observed["daylight"]].copy()
    lit_rows = []
    for candidate in bearings:
        attach_geometry(lit, float(candidate), suffix="_b")
        row = {"bearing": candidate}
        for target in TARGETS + [DESCRIPTIVE]:
            if target not in lit.columns:
                continue
            result = bearing_test(lit, target, base,
                                  ["sun_in_view_b", "sun_glare_b"], have_cloud)
            row[target] = result["dR2"] if result else float("nan")
        lit_rows.append(row)
    lit_table = pd.DataFrame(lit_rows)

    verdicts = {}
    for scope, grid in (("all hours", table), ("daylight only", lit_table)):
        print(f"\n  [{scope}]  {'target':<16}{'peak':>6}{'peak dR2':>11}"
              f"{'at camera':>12}{'at anti':>10}{'gap to axis':>13}"
              f"{'gap to bearing':>16}")
        for target in TARGETS + [DESCRIPTIVE]:
            if target not in grid.columns:
                continue
            best = grid.loc[grid[target].idxmax()]
            at_camera = float(np.interp(
                bearing, grid["bearing"], grid[target], period=360))
            at_anti = float(np.interp(
                (bearing + 180.0) % 360.0, grid["bearing"], grid[target],
                period=360))
            gap = abs((float(best["bearing"]) - bearing + 180.0) % 360.0 - 180.0)
            axis_gap = min(gap, 180.0 - gap)
            if scope == "all hours":
                verdicts[target] = axis_gap
            mark = "  <- DESCRIPTIVE" if target == DESCRIPTIVE else ""
            print(f"  {'':<12}{target:<16}{int(best['bearing']):>6}"
                  f"{best[target]:>+11.4f}{at_camera:>+12.4f}{at_anti:>+10.4f}"
                  f"{axis_gap:>12.0f}°{gap:>15.0f}°{mark}")
        # How sharp the peak is. The sun's azimuth does not cover the circle
        # evenly at a mid-latitude site, so neighbouring bearings are
        # correlated in-sample and even a perpendicular one keeps a large
        # share of the peak. Without this the reader will over-read a
        # 30-degree difference in where the maximum landed.
        print(f"\n  {'':<12}{'sharpness':<16}{'peak':>6}{'trough':>11}"
              f"{'perpendicular':>25}")
        for target in TARGETS + [DESCRIPTIVE]:
            if target not in grid.columns:
                continue
            peak = float(grid[target].max())
            trough = float(grid[target].min())
            across = float(np.interp((float(grid.loc[grid[target].idxmax(),
                                                     "bearing"]) + 90.0) % 360.0,
                                     grid["bearing"], grid[target], period=360))
            share = f"{across / peak:.0%} of peak" if peak > 0 else "—"
            print(f"  {'':<12}{target:<16}{peak:>+6.4f}{trough:>+11.4f}"
                  f"{across:>+14.4f}  {share:<12}")

    # ------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print("WHAT THIS DOES AND DOES NOT SETTLE")
    print("=" * 78)
    real = [t for t in TARGETS if t in verdicts]
    close = [t for t in real if verdicts[t] <= 45]
    print(f"  Rotation: {len(close)} of {len(real)} real targets peak within 45"
          f" deg of the camera's\n  AXIS ({bearing:.0f} / "
          f"{(bearing + 180.0) % 360.0:.0f}). Read that column, not the "
          "bearing one: the\n  linear term cannot tell the two ends of the "
          "axis apart.")
    print("  Read that against the sharpness rows above. A perpendicular")
    print("  bearing keeps a large share of the peak here by construction, so")
    print("  a peak 30 deg from the axis is not meaningfully different from")
    print("  one on it. Only a sweep with real contrast can localise anything.")
    if len(close) == len(real) and real:
        print("\n  All of them. That is the lens story's prediction, and it is")
        print("  the reading the split in (1) and (2) should agree with.")
    elif not close:
        print("  None of them. Whatever the sun terms are carrying, it does not")
        print("  point at this camera — which is the afternoon/sea-breeze")
        print("  reading, not the lens one.")
    else:
        print("  A split verdict. Do not pick the targets that agree; report")
        print("  that the rotation did not resolve it.")
    print("\n  The split in (1) and (2) cannot settle this on its own, because")
    print("  sun-in-front and afternoon are nearly the same hours here. Read it")
    print("  as a consistency check on the rotation, not as independent")
    print("  evidence.")
    print(f"\n  None of this changes what the target IS. Precision on this "
          f"camera is\n  2.7% population-weighted, so a driver of the "
          "detection rate is a driver\n  of what the detector fires on, which "
          "is mostly not a rip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
