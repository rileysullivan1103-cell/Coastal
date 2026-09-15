"""Is Virginia Beach's air-temperature result a driver, or is it the light?

At Virginia Beach, temperature_2m sits second in the rip table on three of
four targets, at rho_hrmo about -0.36, and it is NEGATIVE: the detector fires
less on warm hours. There is no mechanism by which warm air suppresses a rip
current. There are two by which it suppresses a rip DETECTOR. A warm hour is a
sunny hour, and a high sun glaring off water flattens the surface texture a
YOLO model keys on; a warm hour is also a hazy hour, and haze does the same
thing more slowly. Either way the result would be a fact about the camera,
not about the ocean, and it would not survive being written down as a driver.

The test needs the sun's position and a brightness measure. Neither needs a
new data source: solar geometry is a closed-form function of timestamp and
latitude and longitude (see solar.py), and ERA5 serves total cloud cover
alongside the variables already pulled.

Two halves, matching the two halves of the question:

(a) what the new columns correlate with on their own, and

(b) what happens to the air-temperature and wave-height coefficients as the
    light terms are added to the regression one layer at a time. A ladder
    rather than a before-and-after, because "temperature collapses once the
    sun is in the model" and "temperature collapses once GLARE GEOMETRY is in
    the model" are different findings and only a ladder separates them.

    python analyze_glare.py
    python analyze_glare.py --camera Walton

Walton is the control. It faces 206 deg (SSW) and its temperature result is
small, so if the light terms behave differently there than at Virginia Beach
-- which faces 90 deg, straight into the sunrise -- that asymmetry is itself
evidence, and it is an asymmetry no confound shared by both sites can make.

Nothing here writes to disk and nothing in analyze_drivers.py changes.
"""

import argparse
import sys

import numpy as np
import pandas as pd

import analyze_drivers as ad
import solar

# Both cameras the question is about, control first.
DEFAULT_CAMERAS = ["walton-lighthouse-santa-cruz-ca",
                   "hampton-inn-oceanfront-south-at-virginia-beach"]

# Computed here from lat/lon and the timestamp; never read from a file.
SOLAR_COLUMNS = ["solar_elevation", "solar_azimuth", "sun_in_view", "sun_glare"]
CLOUD_COLUMN = "cloud_cover"

# The physical predictors the light terms are tested against. Deliberately
# compact: the full RIP_PREDICTORS list runs at condition 18.6, where
# analyze_drivers itself warns individual coefficients are arbitrary, and a
# coefficient that is arbitrary cannot be watched for a change. Wave height is
# resolved per camera because Walton has CDIP MOP and Virginia Beach has only
# the Open-Meteo model.
BASE_OTHER = ["wind_onshore", "wind_speed_10m", "level_m",
              "temperature_2m", "precipitation", "rain_48h_mm"]

TRACKED = ["temperature_2m"]     # wave height is appended per camera


def wave_columns(frame):
    """(height, period) — the nearshore model where it exists, else the grid."""
    if "mop_wave_height" in frame.columns and frame["mop_wave_height"].notna().any():
        return "mop_wave_height", "mop_wave_period"
    return "wave_height", "wave_period"


def seaward_bearing(name, weather):
    """The direction the camera looks, by the same rule analyze_drivers uses."""
    published = (ad.MOP_META.get(name) or ad.MOP_META.get(weather) or {}).get(
        "shore_normal_deg")
    if published is not None:
        return float(published), "published by CDIP"
    assumed = ad.SHORE_NORMAL_DEG.get(name) or ad.SHORE_NORMAL_DEG.get(weather)
    if assumed is None:
        return None, None
    return float(assumed), "assumed (see SHORE_NORMAL_DEG)"


def add_solar(frame, lat, lon, bearing):
    """Attach the four solar columns in place. Returns the ones added."""
    elevation, azimuth = solar.position(pd.DatetimeIndex(frame["hour"]), lat, lon)
    frame["solar_elevation"] = elevation
    frame["solar_azimuth"] = azimuth
    added = ["solar_elevation", "solar_azimuth"]
    if bearing is not None:
        frame["sun_in_view"] = solar.sun_in_view(azimuth, bearing)
        frame["sun_glare"] = solar.glare_index(elevation, azimuth, bearing)
        added += ["sun_in_view", "sun_glare"]
    return added


def beta_of(fit, name):
    """A named standardized coefficient, or NaN if the fit withheld them."""
    if not fit or fit.get("beta") is None:
        return np.nan
    lookup = dict(zip(fit["names"], fit["beta"]))
    return float(lookup.get(name, np.nan))


def ladder(observed, target, base, extras):
    """Fit each rung and return one row per model.

    Each rung is base + a named set, so every coefficient is read against the
    same physical predictors and only the light terms move.
    """
    rows = []
    for label, added in extras:
        predictors = base + [c for c in added if c in observed.columns]
        fit = ad.standardized_ols(observed, target, predictors)
        row = {"model": label, "n": fit["n"] if fit else 0,
               "R2": fit["r2"] if fit else np.nan,
               "cond": fit["condition"] if fit else np.nan,
               "fit": fit}
        rows.append(row)
    return rows


def analyse(sites, want):
    frame, name, has_coverage = ad.assemble_rip(sites, want=want)
    if frame is None:
        return False

    observed = (frame.copy() if has_coverage
                else frame[frame.get("frames", 0) > 0].copy())
    observed["month"] = observed["hour"].dt.month
    observed["hr_mo"] = (observed["hour_of_day"].astype(str) + "-"
                         + observed["month"].astype(str))

    row = next((s for _, s in sites.iterrows() if s["camera_name"] == name), None)
    if row is None or pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
        print(f"  {name}: no latitude/longitude, cannot place the sun")
        return False
    lat, lon = float(row["lat"]), float(row["lon"])
    weather = row.get("weather_name")
    weather = name if not isinstance(weather, str) or not weather else weather
    bearing, bearing_source = seaward_bearing(name, weather)

    print(f"\n{'=' * 78}")
    print(f"GLARE AND LIGHT — {name}")
    print("=" * 78)
    print(f"  {len(observed)} hours ({observed['hour'].min():%Y-%m-%d} to "
          f"{observed['hour'].max():%Y-%m-%d}) at {lat:.4f}, {lon:.4f}")
    if bearing is None:
        print("  no shore normal configured — sun_in_view and sun_glare are "
              "skipped,\n  so azimuth stays circular and is not interpretable "
              "in a rank correlation.")
    else:
        print(f"  camera looks {bearing:.1f} deg ({bearing_source}); the sun is "
              "'in view'\n  when its bearing is near that.")

    solar_added = add_solar(observed, lat, lon, bearing)
    lit = observed["solar_elevation"] > 0
    print(f"  sun above the horizon in {int(lit.sum())} of {len(observed)} "
          f"hours ({lit.mean():.0%})")
    if bearing is not None:
        facing = observed.loc[lit, "sun_in_view"] > 0
        print(f"  and in front of the camera in {facing.mean():.0%} of those")

    have_cloud = (CLOUD_COLUMN in observed.columns
                  and observed[CLOUD_COLUMN].notna().any())
    if have_cloud:
        share = float(observed[CLOUD_COLUMN].notna().mean())
        print(f"  {CLOUD_COLUMN}: present on {share:.0%} of hours")
    else:
        print(f"  {CLOUD_COLUMN}: NOT ON DISK. ERA5 serves it, but the pull "
              "that wrote\n    gridded_*.csv predates it being requested.")
        # The dates are printed rather than left as placeholders because
        # re-pulling on the default one-year window would SHRINK this site's
        # ERA5 file to less than the rip record, and a narrower conditions
        # file silently shrinks every n in every table downstream. That has
        # already cost this project two analyses.
        print("    Re-pull with a window at least as wide as the rip record,")
        print("    NOT the default one-year window:")
        print(f"      python pull_site_observations.py --camera \"{name}\" \\")
        print(f"          --start {observed['hour'].min():%Y-%m-%d} "
              f"--end {observed['hour'].max():%Y-%m-%d}")
        print("    The solar half below is unaffected — it needs no file.")

    height, period = wave_columns(observed)
    if height not in observed.columns:
        print(f"  no wave height column ({height}); nothing to test against")
        return False
    print(f"  wave height column: {height}")

    base = [c for c in [height, period] + BASE_OTHER if c in observed.columns]
    new = [c for c in solar_added if c in observed.columns]
    if have_cloud:
        new = new + [CLOUD_COLUMN]

    targets = [t for t in ad.RIP_TARGETS
               if t in observed.columns
               and pd.to_numeric(observed[t], errors="coerce").nunique() >= 3]

    # ------------------------------------------------------------------
    # (a) The new columns on their own.
    # ------------------------------------------------------------------
    print(f"\n{'-' * 78}")
    print("(a) WHAT THE LIGHT COLUMNS TRACK ON THEIR OWN")
    print("-" * 78)
    print("  READ rho AND rho_mo, NOT rho_hrmo, for solar_elevation.")
    print("  Hour-of-day IS solar elevation, near enough: within one hour-by-")
    print("  month cell the sun barely moves, so demeaning by that cell removes")
    print("  almost all of it by construction. A collapse from rho to rho_hrmo")
    print("  here means the control absorbed the predictor, NOT that the")
    print("  predictor is inert. Section (b) is where the light terms are")
    print("  allowed to compete, because that regression applies no hour control.")
    for target in targets:
        ad.report_correlations(
            observed, target, new,
            controls=[("hr", observed["hour_of_day"]),
                      ("mo", observed["month"]),
                      ("hrmo", observed["hr_mo"])],
            title=f"--- {target} ---")

    # ------------------------------------------------------------------
    # (b) What the light terms do to temperature and wave height.
    # ------------------------------------------------------------------
    tracked = TRACKED + [height]
    rungs = [("base", []),
             ("+ cloud", [CLOUD_COLUMN]),
             ("+ elevation", ["solar_elevation"]),
             ("+ elev/azim/cloud", ["solar_elevation", "solar_azimuth",
                                    CLOUD_COLUMN]),
             ("+ glare geometry", ["solar_elevation", "sun_in_view",
                                   "sun_glare", CLOUD_COLUMN])]
    print(f"\n{'-' * 78}")
    print("(b) WHAT HAPPENS TO temperature_2m AND " + height.upper())
    print("-" * 78)
    print("  Standardized coefficients. Each rung is the same physical base")
    print(f"  ({', '.join(base)})\n  plus the named light terms, so only the "
          "light terms move between rows.")

    verdicts = {}
    for target in targets:
        rows = ladder(observed, target, base, rungs)
        table = []
        for entry in rows:
            record = {"model": entry["model"], "n": entry["n"],
                      "R2": round(entry["R2"], 4) if pd.notna(entry["R2"]) else np.nan}
            for column in tracked:
                record[column] = round(beta_of(entry["fit"], column), 4)
            for column in new:
                record[column] = round(beta_of(entry["fit"], column), 4)
            table.append(record)
        print(f"\n--- {target} ---")
        with pd.option_context("display.width", 220, "display.max_columns", 40):
            print(pd.DataFrame(table).to_string(index=False))
        verdicts[target] = rows
    return verdicts, name, height, observed


def glare_verdict(verdicts, name, height, observed):
    """State plainly whether the light terms took temperature's job."""
    print(f"\n{'-' * 78}")
    print(f"IS temperature_2m A LIGHT PROXY AT {name.upper()}?")
    print("-" * 78)
    lines = []
    for target, rows in verdicts.items():
        by_model = {r["model"]: r for r in rows}
        base_t = beta_of(by_model["base"]["fit"], "temperature_2m")
        if not np.isfinite(base_t) or abs(base_t) < 0.02:
            continue
        asked = beta_of(by_model["+ elev/azim/cloud"]["fit"], "temperature_2m")
        cloud_only = beta_of(by_model["+ cloud"]["fit"], "temperature_2m")
        elev_only = beta_of(by_model["+ elevation"]["fit"], "temperature_2m")
        geometry = beta_of(by_model["+ glare geometry"]["fit"], "temperature_2m")
        kept = abs(asked) / abs(base_t)
        # Which single layer did the absorbing, if any did.
        drops = {"cloud": abs(base_t) - abs(cloud_only),
                 "solar elevation": abs(base_t) - abs(elev_only)}
        worker = max(drops, key=drops.get)
        extra = abs(elev_only) - abs(geometry)
        lines.append({
            "target": target,
            "base": round(base_t, 4),
            "+cloud": round(cloud_only, 4),
            "+elev": round(elev_only, 4),
            "asked": round(asked, 4),
            "+geom": round(geometry, 4),
            "kept": f"{kept:.0%}",
            "absorbed_by": worker if max(drops.values()) > 0.02 else "neither",
            "geometry_adds": f"{extra:+.3f}",
        })
    if not lines:
        print("  temperature_2m carries no coefficient worth testing here.")
        return
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(pd.DataFrame(lines).to_string(index=False))
    print("\n  'kept' is how much of the base temperature coefficient survives")
    print("  the model Riley asked for (elevation + azimuth + cloud). Under")
    print("  ~50% means the light terms took most of its job.")
    print("  'geometry_adds' is how much FURTHER temperature falls when the")
    print("  sun's bearing relative to the camera is added on top of plain")
    print("  elevation. A large positive number is the specifically-GLARE")
    print("  reading; near zero means temperature was tracking daylight in")
    print("  general, which is a weaker and less interesting claim.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", action="append",
                        help="substring of a rip table; repeatable. "
                             "Default: Walton and Virginia Beach.")
    args = parser.parse_args()

    sites = ad.load_sites()
    cameras = args.camera or DEFAULT_CAMERAS
    ran = 0
    for camera in cameras:
        result = analyse(sites, camera)
        if not result:
            continue
        glare_verdict(*result)
        ran += 1
    if not ran:
        print("\nNo camera produced a usable table.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
