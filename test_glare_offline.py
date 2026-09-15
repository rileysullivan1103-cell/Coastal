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
RUNGS = [("base", []),
         ("+ cloud", ["cloud_cover"]),
         ("+ elevation", ["solar_elevation"]),
         ("+ elev/azim/cloud", ["solar_elevation", "solar_azimuth", "cloud_cover"]),
         ("+ glare geometry", ["solar_elevation", "sun_in_view", "sun_glare",
                               "cloud_cover"])]

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
    elif kind == "genuine":
        # Temperature is unrelated to the sun and really does move the target.
        frame["temperature_2m"] = independent
        suppressor = (independent - independent.mean()) / independent.std()
    else:
        raise ValueError(kind)

    signal = 0.45 * frame["wave_height"] - 2.2 * suppressor + rng.normal(0, 0.5, n)
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


def main():
    print("analyze_glare offline checks\n")
    check_daylight_proxy_is_absorbed()
    check_glare_proxy_needs_the_geometry()
    check_a_genuine_driver_is_not_absorbed()
    check_wave_height_is_not_collaterally_diluted()
    check_beta_of_handles_a_withheld_fit()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
