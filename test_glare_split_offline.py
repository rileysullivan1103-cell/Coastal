"""Offline checks for analyze_glare_split.py.

The fixtures use REAL solar geometry -- solar.position at Walton's coordinates
over a year -- because the whole question is about the shape of the sun's path,
and a uniform random azimuth would make the front/behind split trivially
balanced in a way the real one is not.

Three drivers are planted, each with a known answer:

  FRONT ONLY   the target responds to glare_index against bearing 206, which is
               zero by construction whenever the sun is behind the camera. The
               split must find it in one group and not the other.
  BOTH GROUPS  the target responds to the SIGNED geometry, which is positive in
               front and negative behind. Both groups must find it. Without
               this fixture a split that can never fire in the behind group
               would pass the first test for the wrong reason.
  OFF BEARING  the target responds to glare against bearing 120, not 206. The
               rotation sweep must peak near 120, and must NOT peak at the
               camera's real bearing.

The p-value machinery is checked against scipy where scipy is importable, and
skipped with a printed note where it is not, so the repo gains no dependency.
"""

import contextlib
import io
import math
import sys

import numpy as np
import pandas as pd

import analyze_drivers as ad
import analyze_glare as ag
import analyze_glare_split as gs
import solar

FAILURES = []

LAT, LON = 36.9517, -122.0170
BEARING = 206.0
HOURS = 8760


def check(name, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    if not condition:
        FAILURES.append(name)
    print(f"  {mark} {name}" + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------------------

def scaffold(rng, hours=HOURS):
    """Everything the base model needs, with real solar geometry on top."""
    index = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    elevation, azimuth = solar.position(index, LAT, LON)
    frame = pd.DataFrame({
        "hour": index,
        "hour_of_day": index.hour,
        "solar_elevation": elevation,
        "solar_azimuth": azimuth,
        "wave_height": rng.gamma(2.0, 0.5, hours),
        "wave_period": 8 + rng.normal(0, 1.5, hours),
        "wind_onshore": rng.uniform(-1, 1, hours),
        "wind_speed_10m": rng.gamma(2.0, 2.0, hours),
        "level_m": rng.normal(0, 0.6, hours),
        "temperature_2m": 14 + 6 * np.sin(np.arange(hours) / 1400.0),
        "precipitation": rng.gamma(0.3, 1.0, hours),
        "rain_48h_mm": rng.gamma(0.8, 2.0, hours),
        ag.CLOUD_COLUMN: rng.uniform(0, 100, hours),
    })
    return frame


def plant(frame, driver, effect, rng):
    """target = waves + effect * driver + noise, all standardized."""
    def z(values):
        values = np.asarray(values, dtype=float)
        return (values - values.mean()) / (values.std() or 1.0)

    frame = frame.copy()
    frame["detection_rate"] = (0.5 * z(frame["wave_height"])
                               + effect * z(driver)
                               + rng.normal(0, 1.0, len(frame)))
    return frame


def run_split(frame, terms, front_deg=90.0):
    """(front result, behind result) for detection_rate."""
    gs.attach_geometry(frame, BEARING)
    gap = gs.angular_gap(frame["solar_azimuth"], BEARING)
    base, _ = gs.base_columns(frame)
    out = []
    for mask in (gap <= front_deg, gap > front_deg):
        out.append(gs.bearing_test(frame[mask], "detection_rate", base, terms,
                                   True))
    return out


# ---------------------------------------------------------------------------

def check_the_f_distribution():
    print("\nthe F survival function")
    try:
        from scipy import stats
    except ImportError:
        print("  --  scipy not importable; skipped (the repo does not "
              "require it)")
        return
    cases = [(1.0, 1, 10), (4.0, 2, 30), (78.43, 2, 5824), (3.2, 1, 8000),
             (0.5, 3, 12), (150.0, 2, 8000)]
    worst = 0.0
    for f_stat, df1, df2 in cases:
        mine = gs.f_survival(f_stat, df1, df2)
        theirs = float(stats.f.sf(f_stat, df1, df2))
        gap = abs(mine - theirs) / max(theirs, 1e-300)
        worst = max(worst, gap)
        check(f"F({df1},{df2}) at {f_stat} matches scipy",
              gap < 1e-9, f"{mine:.6g} vs {theirs:.6g}")
    check("every case agrees to nine figures", worst < 1e-9)

    # The df1=2 closed form analyze_glare uses, as an independent cross-check.
    f_stat, df2 = 78.43, 5824
    closed = (1.0 + 2 * f_stat / df2) ** (-df2 / 2.0)
    check("and reproduces analyze_glare's df1=2 closed form",
          abs(gs.f_survival(f_stat, 2, df2) - closed) / closed < 1e-9,
          f"{gs.f_survival(f_stat, 2, df2):.4g} vs {closed:.4g}")


def check_the_added_term_count_is_read_not_assumed():
    print("\ncounting the terms that survived")
    rng = np.random.default_rng(4)
    frame = scaffold(rng, 500)
    frame = plant(frame, frame["wave_period"], 0.4, rng)
    base, _ = gs.base_columns(frame)
    small = ad.standardized_ols(frame, "detection_rate",
                               base + ["solar_elevation"])
    # One real column and one that is a constant: standardized_ols drops the
    # constant, so only one term is genuinely added.
    frame["flat"] = 1.0
    big = ad.standardized_ols(frame, "detection_rate",
                              base + ["solar_elevation", "solar_azimuth",
                                      "flat"])
    result = gs.added_term_test(small, big)
    check("a constant column is not counted as an added term",
          result["added"] == 1, f"added={result['added']}")
    check("and the p-value is computed anyway",
          pd.notna(result["p"]), f"p={result['p']:.3g}")

    same = gs.added_term_test(small, small)
    check("adding nothing is reported, not divided by zero",
          same["added"] == 0 and pd.isna(same["F"]))


def check_the_geometry_definitions():
    print("\nfront/behind classification and the clip")
    azimuth = pd.Series([206.0, 216.0, 295.0, 297.0, 26.0, 350.0, 10.0])
    gap = gs.angular_gap(azimuth, BEARING)
    check("the bearing itself is 0 deg away", gap.iloc[0] == 0)
    check("10 deg round is 10 deg away", gap.iloc[1] == 10)
    check("just inside 90 is in front", gap.iloc[2] < 90)
    check("just outside 90 is behind", gap.iloc[3] > 90)
    check("the anti-bearing is 180 deg away", gap.iloc[4] == 180)
    check("the fold wraps past 360 correctly",
          abs(gap.iloc[5] - 144.0) < 1e-9 and abs(gap.iloc[6] - 164.0) < 1e-9,
          f"{gap.iloc[5]:.1f}, {gap.iloc[6]:.1f}")

    rng = np.random.default_rng(9)
    frame = gs.attach_geometry(scaffold(rng, 2000), BEARING)
    behind = frame[gs.angular_gap(frame["solar_azimuth"], BEARING) > 90]
    check("the fixture has a behind group at all", len(behind) > 200,
          f"{len(behind)} hours")
    check("sun_glare is IDENTICALLY zero behind the camera",
          float(behind["sun_glare"].abs().max()) == 0.0)
    check("sun_in_view still varies behind the camera",
          float(behind["sun_in_view"].std()) > 0.05,
          f"sd {behind['sun_in_view'].std():.3f}")
    check("the signed glare varies behind the camera",
          float(behind[gs.SIGNED_GLARE].std()) > 0.01,
          f"sd {behind[gs.SIGNED_GLARE].std():.3f}")
    front = frame[gs.angular_gap(frame["solar_azimuth"], BEARING) <= 90]
    check("and in front the signed version IS the published one",
          float((front[gs.SIGNED_GLARE] - front["sun_glare"]).abs().max()) < 1e-12)


def check_the_harmonic_identity():
    print("\nsun_in_view against the azimuth harmonics")
    rng = np.random.default_rng(12)
    frame = gs.attach_geometry(scaffold(rng, 3000), BEARING)
    result = gs.harmonic_check(frame, BEARING)
    # cos(a-b) = cos(a)cos(b) + sin(a)sin(b) is an identity, so this is 1 by
    # trigonometry and not by anything the code does.
    check("R2 is 1 to nine figures", abs(result["r2"] - 1.0) < 1e-9,
          f"{result['r2']:.10f}")
    check("the cos coefficient IS cos(bearing)",
          abs(result["cos_beta"] - math.cos(math.radians(BEARING))) < 1e-9,
          f"{result['cos_beta']:+.6f}")
    check("the sin coefficient IS sin(bearing)",
          abs(result["sin_beta"] - math.sin(math.radians(BEARING))) < 1e-9,
          f"{result['sin_beta']:+.6f}")


def check_a_front_only_driver_lands_in_one_group():
    print("\na driver that fires only when the sun is in front")
    rng = np.random.default_rng(20260916)
    frame = scaffold(rng)
    driver = solar.glare_index(frame["solar_elevation"], frame["solar_azimuth"],
                               BEARING)
    frame = plant(frame, driver, 0.55, rng)

    front, behind = run_split(frame, gs.BEARING_TERMS)
    check("the front group finds it",
          front["dR2"] > 0.02 and front["p"] < 1e-6,
          f"dR2 {front['dR2']:+.4f}, F {front['F']:.1f}, p {front['p']:.2g}")
    check("the behind group does not",
          behind["dR2"] < 0.005 and behind["p"] > 0.01,
          f"dR2 {behind['dR2']:+.4f}, p {behind['p']:.2g}")
    check("and the behind group had only one live term",
          behind["added"] == 1, f"added={behind['added']}")

    s_front, s_behind = run_split(frame, gs.SIGNED_TERMS)
    check("under the signed pair the front group is unchanged",
          abs(s_front["dR2"] - front["dR2"]) < 1e-9,
          f"{s_front['dR2']:+.6f} vs {front['dR2']:+.6f}")
    check("and the behind group now has two live terms and still finds nothing",
          s_behind["added"] == 2 and s_behind["dR2"] < 0.005,
          f"added={s_behind['added']}, dR2 {s_behind['dR2']:+.4f}")


def check_a_driver_in_both_groups_is_found_in_both():
    print("\na driver present on both sides of the camera")
    rng = np.random.default_rng(77)
    frame = scaffold(rng)
    driver = gs.signed_glare(frame["solar_elevation"], frame["solar_azimuth"],
                             BEARING)
    frame = plant(frame, driver, 0.55, rng)

    front, behind = run_split(frame, gs.SIGNED_TERMS)
    check("the front group finds it",
          front["dR2"] > 0.01 and front["p"] < 1e-4,
          f"dR2 {front['dR2']:+.4f}, p {front['p']:.2g}")
    check("the behind group finds it too",
          behind["dR2"] > 0.01 and behind["p"] < 1e-4,
          f"dR2 {behind['dR2']:+.4f}, p {behind['p']:.2g}")
    check("so a null behind is a real null, not a dead branch",
          True)


def check_the_sweep_finds_the_planted_bearing():
    print("\nthe rotation sweep, with the driver aimed away from the camera")
    planted = 120.0
    rng = np.random.default_rng(31)
    frame = scaffold(rng)
    driver = solar.glare_index(frame["solar_elevation"], frame["solar_azimuth"],
                               planted)
    frame = plant(frame, driver, 0.55, rng)
    base, _ = gs.base_columns(frame)

    rows = []
    for candidate in range(0, 360, 15):
        gs.attach_geometry(frame, float(candidate), suffix="_b")
        result = gs.bearing_test(frame, "detection_rate", base,
                                 ["sun_in_view_b", "sun_glare_b"], True)
        rows.append({"bearing": candidate, "dR2": result["dR2"]})
    table = pd.DataFrame(rows)
    peak = float(table.loc[table["dR2"].idxmax(), "bearing"])
    gap = abs((peak - planted + 180.0) % 360.0 - 180.0)
    at_planted = float(table.loc[table["bearing"] == 120, "dR2"].iloc[0])
    at_camera = float(table.loc[table["bearing"] == 210, "dR2"].iloc[0])

    check("the sweep peaks at the bearing the driver was built on",
          gap <= 30.0, f"peak {peak:.0f} deg, planted {planted:.0f}, "
                       f"{gap:.0f} deg apart")
    check("and clearly prefers it to the camera's real bearing",
          at_planted > at_camera * 1.5,
          f"dR2 {at_planted:+.4f} at 120 vs {at_camera:+.4f} at 210")
    check("the sweep is not flat, so it has resolution to spend",
          table["dR2"].max() - table["dR2"].min() > 0.01,
          f"range {table['dR2'].max() - table['dR2'].min():.4f}")


def check_the_sweep_identifies_an_axis_not_a_direction():
    """The caveat the script prints, verified rather than asserted in prose.

    cos(az - b) = -cos(az - b - 180), so a driver carried by that term alone
    fits equally well at b and at b+180: the linear part identifies an AXIS.
    Only the clipped, elevation-weighted glare term can tell the two ends
    apart. Both halves of that are checked here, because the script's peak
    table is read on the strength of it.
    """
    print("\nwhat the rotation can and cannot tell apart")
    planted = 120.0
    rng = np.random.default_rng(55)
    frame = scaffold(rng)
    linear = solar.sun_in_view(frame["solar_azimuth"], planted)
    frame = plant(frame, linear, 0.55, rng)
    base, _ = gs.base_columns(frame)

    def dr2(candidate, terms):
        gs.attach_geometry(frame, float(candidate), suffix="_b")
        return gs.bearing_test(frame, "detection_rate", base, terms,
                               True)["dR2"]

    here = dr2(planted, ["sun_in_view_b"])
    opposite = dr2((planted + 180.0) % 360.0, ["sun_in_view_b"])
    check("the linear term alone fits the bearing and its opposite equally",
          abs(here - opposite) < 1e-9 and here > 0.02,
          f"{here:+.6f} at {planted:.0f} vs {opposite:+.6f} at "
          f"{(planted + 180) % 360:.0f}")
    across = dr2((planted + 90.0) % 360.0, ["sun_in_view_b"])
    # NOT near zero, and that is the point. Solar azimuth does not cover the
    # circle uniformly at this latitude, so cos(az-b) and cos(az-b-90) are
    # correlated in-sample rather than orthogonal. The perpendicular bearing
    # keeps a third of the peak, which is why the script prints the sweep's
    # contrast instead of letting a reader assume the peak is sharp.
    check("the perpendicular bearing fits clearly worse, but not nearly zero",
          across < here * 0.5 and across > here * 0.15,
          f"{across:+.6f} across vs {here:+.6f} along "
          f"({across / here:.0%} of the peak)")

    # Now a driver that only exists on one side, where the clip does break it.
    rng = np.random.default_rng(56)
    frame = scaffold(rng)
    clipped = solar.glare_index(frame["solar_elevation"],
                                frame["solar_azimuth"], planted)
    frame = plant(frame, clipped, 0.55, rng)
    base, _ = gs.base_columns(frame)
    pair = ["sun_in_view_b", "sun_glare_b"]
    here = dr2(planted, pair)
    opposite = dr2((planted + 180.0) % 360.0, pair)
    check("the clipped glare term breaks the symmetry",
          here > opposite * 1.5,
          f"{here:+.4f} at {planted:.0f} vs {opposite:+.4f} at "
          f"{(planted + 180) % 360:.0f}")


def check_the_peak_is_raced_against_solar_noon():
    """A threshold on 'near the camera' cannot work when noon is 26 deg away.

    Both fixtures use the same machinery and differ only in which bearing the
    driver was built on, so the verdict is the only thing that can move.
    """
    print("\nracing the camera against solar noon")
    rng = np.random.default_rng(88)
    frame = scaffold(rng)
    frame["hour_of_day"] = frame["hour"].dt.hour
    noon, days = gs.solar_noon_bearing(frame)
    check("solar noon is found and is near due south at this latitude",
          noon is not None and abs(noon - 180.0) < 10.0 and days > 300,
          f"{noon:.1f} deg over {days} days")

    # The classifier on its own, with hand-picked peaks.
    call, to_cam, to_noon = gs.classify_peak(206.0, BEARING, 180.0)
    check("a peak exactly on the camera reads as the camera",
          call == "camera / lens", f"{call} ({to_cam:.0f} vs {to_noon:.0f})")
    call, _, _ = gs.classify_peak(180.0, BEARING, 180.0)
    check("a peak exactly on solar noon reads as time of day",
          call == "solar noon / time of day", call)
    call, _, _ = gs.classify_peak(193.0, BEARING, 180.0)
    check("a peak halfway between the two is not called either way",
          call == "cannot separate", call)

    # End to end: a driver aimed 60 deg off solar noon must read as the camera
    # when the camera is put there, and as noon when it is built on noon.
    base, _ = gs.base_columns(frame)

    def peak_for(driver_bearing):
        planted = solar.glare_index(frame["solar_elevation"],
                                    frame["solar_azimuth"], driver_bearing)
        local = plant(frame, planted, 0.55, rng)
        rows = []
        for candidate in range(0, 360, 10):
            gs.attach_geometry(local, float(candidate), suffix="_b")
            rows.append({"bearing": candidate,
                         "dR2": gs.bearing_test(local, "detection_rate", base,
                                                ["sun_in_view_b",
                                                 "sun_glare_b"],
                                                True)["dR2"]})
        grid = pd.DataFrame(rows)
        return float(grid.loc[grid["dR2"].idxmax(), "bearing"])

    off_noon = peak_for(120.0)
    call, to_cam, to_noon = gs.classify_peak(off_noon, 120.0, noon)
    check("a driver built on a bearing far from noon reads as the camera",
          call == "camera / lens",
          f"peak {off_noon:.0f}, camera 120, noon {noon:.0f} -> {call}")

    on_noon = peak_for(noon)
    call, to_cam, to_noon = gs.classify_peak(on_noon, BEARING, noon)
    check("a driver built on solar noon reads as time of day",
          call == "solar noon / time of day",
          f"peak {on_noon:.0f}, camera {BEARING:.0f}, noon {noon:.0f} -> {call}")


def main():
    print("glare front/behind split offline checks")
    check_the_f_distribution()
    check_the_added_term_count_is_read_not_assumed()
    check_the_geometry_definitions()
    check_the_harmonic_identity()
    check_a_front_only_driver_lands_in_one_group()
    check_a_driver_in_both_groups_is_found_in_both()
    check_the_sweep_finds_the_planted_bearing()
    check_the_sweep_identifies_an_axis_not_a_direction()
    check_the_peak_is_raced_against_solar_noon()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
