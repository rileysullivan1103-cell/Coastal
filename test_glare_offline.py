"""Offline checks for analyze_glare.py: does the ladder actually discriminate?

The ladder's claim is that it can tell three things apart:

  1. temperature is a proxy for DAYLIGHT (built from solar elevation),
  2. temperature is a proxy for GLARE (built from sun-in-view geometry),
  3. temperature is a GENUINE driver, independent of the sun.

Case 3 is the one that matters most and is easiest to skip. Adding predictors
to a regression shrinks coefficients for all sorts of uninteresting reasons,
so a tool that reports "temperature collapsed" on a real driver is worse than
no tool: it would license throwing away a true result. Each fixture below
builds a frame where the answer is known by construction and asserts the
ladder returns it.

A fourth check watches wave height, which is a real driver in every fixture.
It must NOT move when the light terms arrive, or the ladder is just diluting
everything and the temperature finding would mean nothing.

    python test_glare_offline.py
"""

import numpy as np
import pandas as pd

import analyze_glare as ag
import solar

LAT, LON, BEARING = 36.8529, -75.9780, 90.0    # Virginia Beach, faces east

BASE = ["wave_height", "wave_period", "wind_onshore", "wind_speed_10m",
        "level_m", "temperature_2m", "precipitation", "rain_48h_mm"]
# The real ladder, not a copy of it. A hardcoded duplicate here is what let
# the bearing-term F-test regress unnoticed: the fixtures kept passing against
# a rung list that no longer matched the one analyze_glare.py was fitting.
RUNGS, BEARING_BASE = ag.build_rungs(have_cloud=True)

FAILURES = []


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name
          + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def build(kind, seed=5):
    """A daylight-only frame where temperature is `kind`."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2026-04-25", "2026-08-28", freq="h", tz="UTC")
    elevation, azimuth = solar.position(hours, LAT, LON)
    keep = elevation > 0
    hours, elevation, azimuth = hours[keep], elevation[keep], azimuth[keep]
    n = len(hours)
    glare = solar.glare_index(elevation, azimuth, BEARING)

    frame = pd.DataFrame({"hour": hours})
    frame["hour_of_day"] = frame["hour"].dt.hour
    frame["solar_elevation"] = elevation
    frame["solar_azimuth"] = azimuth
    frame["sun_in_view"] = solar.sun_in_view(azimuth, BEARING)
    frame["sun_glare"] = glare
    frame["cloud_cover"] = np.clip(rng.normal(50, 25, n), 0, 100)
    frame["wave_height"] = rng.gamma(4, 0.3, n)
    frame["wave_period"] = rng.normal(7, 1.5, n)
    frame["wind_onshore"] = rng.normal(0, 3, n)
    frame["wind_speed_10m"] = rng.gamma(3, 1.5, n)
    frame["level_m"] = rng.normal(0.5, 0.4, n)
    frame["precipitation"] = rng.gamma(0.3, 1.0, n)
    frame["rain_48h_mm"] = frame["precipitation"].rolling(48, min_periods=1).sum()

    independent = rng.normal(20, 5, n)
    if kind == "daylight":
        frame["temperature_2m"] = 18 + 0.15 * elevation + rng.normal(0, 1.5, n)
        suppressor = elevation / 50.0
    elif kind == "glare":
        frame["temperature_2m"] = 18 + 9.0 * glare + rng.normal(0, 1.5, n)
        suppressor = glare
    elif kind == "suppressed":
        # Temperature is genuinely negative on the target AND positively
        # correlated with elevation, which is itself positive on the target.
        # Without elevation in the model the two effects partly cancel and
        # temperature looks weak; with it, temperature's own effect stands out
        # and its coefficient GROWS. This is what Virginia Beach does, and
        # nothing in the earlier fixtures produced it.
        frame["temperature_2m"] = 18 + 0.15 * elevation + rng.normal(0, 1.5, n)
        suppressor = None
    elif kind == "cloudy":
        # Cloud drives the target; the sun's BEARING does nothing at all. The
        # F-test must attribute the gain to cloud, not to sun_in_view and
        # sun_glare. It did not before build_rungs existed, because the rung
        # it measured from carried no cloud.
        frame["temperature_2m"] = independent
        suppressor = None
    elif kind == "genuine":
        # Temperature is unrelated to the sun and really does move the target.
        frame["temperature_2m"] = independent
        suppressor = (independent - independent.mean()) / independent.std()
    else:
        raise ValueError(kind)

    if kind == "cloudy":
        signal = (0.45 * frame["wave_height"] + 0.05 * frame["cloud_cover"]
                  + rng.normal(0, 0.5, n))
    elif kind == "suppressed":
        # elevation helps the detector, temperature hurts it, independently.
        signal = (0.45 * frame["wave_height"] + 0.06 * elevation
                  - 0.35 * frame["temperature_2m"] + rng.normal(0, 0.5, n))
    else:
        signal = (0.45 * frame["wave_height"] - 2.2 * suppressor
                  + rng.normal(0, 0.5, n))
    frame["detection_rate"] = (signal - signal.min()) / (signal.max() - signal.min())
    return frame


def run(frame):
    rows = ag.ladder(frame, "detection_rate", BASE, RUNGS)
    by_model = {r["model"]: r["fit"] for r in rows}
    return {label: (ag.beta_of(fit, "temperature_2m"), ag.beta_of(fit, "wave_height"))
            for label, fit in by_model.items()}


def check_daylight_proxy_is_absorbed():
    out = run(build("daylight"))
    base, asked = abs(out["base"][0]), abs(out["+ elev/azim/cloud"][0])
    kept = asked / base
    check("a daylight-proxy temperature is absorbed by the light terms",
          kept < 0.35, f"kept {kept:.0%} ({base:.4f} -> {asked:.4f})")
    elev_only = abs(out["+ elevation"][0])
    geometry = abs(out["+ glare geometry"][0])
    check("and plain elevation alone does the absorbing",
          (geometry - elev_only) < 0.05,
          f"elevation leaves {elev_only:.4f}, geometry leaves {geometry:.4f}")


def check_glare_proxy_needs_the_geometry():
    """The discriminating case: elevation alone must NOT be enough."""
    out = run(build("glare"))
    base = abs(out["base"][0])
    elev_only = abs(out["+ elevation"][0])
    geometry = abs(out["+ glare geometry"][0])
    check("a glare-proxy temperature survives plain solar elevation",
          elev_only / base > 0.5,
          f"keeps {elev_only / base:.0%} of {base:.4f} under elevation alone")
    check("but collapses once the sun's bearing is in the model",
          geometry / base < 0.35,
          f"keeps {geometry / base:.0%} under glare geometry")
    check("so geometry_adds separates glare from daylight",
          (elev_only - geometry) > 0.05,
          f"geometry removes a further {elev_only - geometry:.4f}")


def check_suppression_is_not_read_as_absorption():
    """The Virginia Beach shape: the coefficient GROWS once light is in."""
    out = run(build("suppressed"))
    base = abs(out["base"][0])
    asked = abs(out["+ elev/azim/cloud"][0])
    check("a suppressed temperature coefficient grows, not shrinks",
          asked / base > 1.15,
          f"{base:.4f} -> {asked:.4f} ({asked / base:.2f}x)")
    check("so it is never mistaken for a light proxy",
          asked / base >= 0.5,
          "a proxy would fall below 0.50x")


def check_a_genuine_driver_is_not_absorbed():
    """The negative control. Without this the ladder proves nothing."""
    out = run(build("genuine"))
    base, asked = abs(out["base"][0]), abs(out["+ elev/azim/cloud"][0])
    geometry = abs(out["+ glare geometry"][0])
    check("a genuine temperature driver survives the light terms",
          asked / base > 0.85, f"kept {asked / base:.0%} ({base:.4f} -> {asked:.4f})")
    check("and survives the glare geometry too",
          geometry / base > 0.85, f"kept {geometry / base:.0%}")


def check_wave_height_is_not_collaterally_diluted():
    for kind in ("daylight", "glare", "genuine"):
        out = run(build(kind))
        moves = [abs(out[label][1] - out["base"][1]) for label, _ in RUNGS]
        check(f"wave height holds steady when light enters ({kind})",
              max(moves) < 0.05,
              f"largest move {max(moves):.4f} from {out['base'][1]:.4f}")


def check_beta_of_handles_a_withheld_fit():
    check("beta_of returns NaN rather than inventing a coefficient",
          np.isnan(ag.beta_of(None, "temperature_2m"))
          and np.isnan(ag.beta_of({"names": [], "beta": None}, "temperature_2m")))


def check_cloud_is_not_credited_to_the_bearing_terms():
    """The bearing F-test must compare rungs differing ONLY by bearing.

    When a cloud file is on disk, "+ elevation" carries no cloud and
    "+ glare geometry" does, so measuring between them hands cloud's entire
    contribution to sun_in_view and sun_glare -- and divides it by 2 added
    terms rather than 3. On real data that inflated three of eight ratios
    into a "the bearing terms do real work" verdict they had not earned.
    """
    frame = build("cloudy")
    rows = ag.ladder(frame, "detection_rate", BASE, RUNGS)
    by_model = {r["model"]: r for r in rows}
    fits = {r["model"]: r["fit"] for r in rows}

    check("the cloud comparator rung exists when cloud is on disk",
          BEARING_BASE == "+ elev/cloud" and BEARING_BASE in fits)

    honest, _, _ = ag.added_term_test(
        fits.get(BEARING_BASE), fits.get("+ glare geometry"), 2)
    naive, _, _ = ag.added_term_test(
        fits.get("+ elevation"), fits.get("+ glare geometry"), 2)
    cloud_alone = by_model["+ cloud"]["R2"] - by_model["base"]["R2"]

    check("cloud really does carry signal in this fixture",
          cloud_alone > 0.02, f"cloud alone gains dR2={cloud_alone:+.4f}")
    check("the bearing terms are correctly found to add nothing",
          honest < 0.005, f"dR2={honest:+.4f} over {BEARING_BASE}")
    check("and the old comparator would have overstated them",
          naive > honest + 0.01,
          f"naive {naive:+.4f} vs honest {honest:+.4f}"
          f" (cloud alone {cloud_alone:+.4f})")


def check_no_rung_is_a_silent_duplicate():
    """Without cloud, no rung may repeat its neighbour's predictor set.

    Every pre-cloud run printed "+ cloud" as a verbatim copy of "base",
    because the column filter dropped a name that was never fetched. That
    reads as "cloud was tested and did nothing" when cloud was never on disk.
    """
    for have_cloud in (False, True):
        rungs, _ = ag.build_rungs(have_cloud)
        present = {"solar_elevation", "solar_azimuth", "sun_in_view",
                   "sun_glare"} | ({"cloud_cover"} if have_cloud else set())
        seen = []
        for label, cols in rungs:
            seen.append((label, tuple(c for c in cols if c in present)))
        sets = [cols for _, cols in seen]
        check(f"no duplicate rung with have_cloud={have_cloud}",
              len(sets) == len(set(sets)),
              ", ".join(f"{label}->{len(cols)}" for label, cols in seen))


def main():
    print("analyze_glare offline checks\n")
    check_daylight_proxy_is_absorbed()
    check_glare_proxy_needs_the_geometry()
    check_a_genuine_driver_is_not_absorbed()
    check_suppression_is_not_read_as_absorption()
    check_wave_height_is_not_collaterally_diluted()
    check_beta_of_handles_a_withheld_fit()
    check_cloud_is_not_credited_to_the_bearing_terms()
    check_no_rung_is_a_silent_duplicate()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
