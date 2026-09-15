"""Offline checks for solar.py, against known astronomy rather than itself.

A solar position routine is easy to write and easy to get subtly wrong -- a
sign on longitude, a morning/afternoon branch, degrees where radians belong --
and every one of those errors produces plausible-looking numbers. So these
checks pin it to quantities that are known independently of this code:

  * the equation of time reaches about -14.2 min in mid-February and +16.4
    min in early November, and passes through zero four times a year;
  * the sun's noon elevation at solstice is 90 - |lat -/+ 23.44|;
  * at solar noon the sun bears due south from mid-northern latitudes and due
    north from mid-southern ones;
  * elevation is symmetric about solar noon;
  * refraction lifts the sun about half a degree at the horizon.

The longitude sign is worth its own check. Getting it backwards moves solar
noon by sixteen hours at Walton and still returns elevations in [-90, 90],
so nothing but a positional test catches it.

    python test_solar_offline.py
"""

import numpy as np
import pandas as pd

import solar

# The two cameras this was written for.
WALTON = (36.9607, -122.0022)
VIRGINIA_BEACH = (36.8529, -75.9780)

FAILURES = []


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name
          + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def noon_elevation(lat, lon, date):
    """Peak elevation and its bearing on one day, sampled every minute."""
    index = pd.date_range(f"{date} 00:00", periods=1440, freq="min", tz="UTC")
    elev, azim = solar.position(index, lat, lon)
    best = int(np.nanargmax(elev))
    return elev[best], azim[best], index[best]


def check_equation_of_time_extremes():
    """EoT is a property of the orbit, not of this code."""
    index = pd.date_range("2025-01-01 12:00", periods=365, freq="D", tz="UTC")
    _, eot = solar._declination_and_eot(solar._julian_century(index))
    lowest, highest = float(np.min(eot)), float(np.max(eot))
    low_day = index[int(np.argmin(eot))]
    high_day = index[int(np.argmax(eot))]
    check("equation of time bottoms near -14.2 min",
          abs(lowest + 14.2) < 0.4, f"{lowest:.2f} min on {low_day:%b %d}")
    check("equation of time peaks near +16.4 min",
          abs(highest - 16.4) < 0.4, f"{highest:.2f} min on {high_day:%b %d}")
    check("its minimum falls in February", low_day.month == 2,
          f"{low_day:%b %d}")
    check("its maximum falls in late October or November",
          high_day.month in (10, 11), f"{high_day:%b %d}")
    # Four sign changes a year is the shape of the analemma.
    crossings = int((np.diff(np.sign(eot)) != 0).sum())
    check("it crosses zero four times a year", crossings == 4,
          f"{crossings} crossings")


def check_solstice_noon_elevation():
    for label, (lat, lon) in (("Walton", WALTON), ("Virginia Beach", VIRGINIA_BEACH)):
        summer, _, _ = noon_elevation(lat, lon, "2025-06-21")
        winter, _, _ = noon_elevation(lat, lon, "2025-12-21")
        check(f"{label} summer solstice noon elevation is 90-lat+23.44",
              abs(summer - (90.0 - lat + 23.44)) < 0.5,
              f"{summer:.2f} vs {90.0 - lat + 23.44:.2f}")
        check(f"{label} winter solstice noon elevation is 90-lat-23.44",
              abs(winter - (90.0 - lat - 23.44)) < 0.5,
              f"{winter:.2f} vs {90.0 - lat - 23.44:.2f}")


def check_noon_bearing_is_south_in_the_north():
    for label, (lat, lon) in (("Walton", WALTON), ("Virginia Beach", VIRGINIA_BEACH)):
        _, bearing, _ = noon_elevation(lat, lon, "2025-06-21")
        check(f"{label} sun bears due south at solar noon",
              abs(bearing - 180.0) < 1.0, f"{bearing:.2f} deg")
    # A southern site must come out the other way, which a hemisphere-blind
    # implementation would not.
    _, bearing, _ = noon_elevation(-33.86, 151.21, "2025-06-21")   # Sydney
    check("in the southern hemisphere it bears due north at noon",
          min(abs(bearing - 0.0), abs(bearing - 360.0)) < 1.0, f"{bearing:.2f} deg")


def check_longitude_sign_puts_noon_at_the_right_time():
    """Walton is UTC-8; its solar noon must land around 20:00 UTC."""
    _, _, when = noon_elevation(*WALTON, "2025-06-21")
    check("Walton solar noon is ~20:00 UTC, not ~04:00",
          19 <= when.hour <= 21, f"{when:%H:%M} UTC")
    _, _, when = noon_elevation(*VIRGINIA_BEACH, "2025-06-21")
    check("Virginia Beach solar noon is ~17:00 UTC",
          16 <= when.hour <= 18, f"{when:%H:%M} UTC")


def check_elevation_is_symmetric_about_noon():
    _, _, noon = noon_elevation(*WALTON, "2025-09-22")
    offsets = pd.to_timedelta(np.arange(1, 5) * 3600, unit="s")
    before, _ = solar.position(pd.DatetimeIndex(noon - offsets), *WALTON)
    after, _ = solar.position(pd.DatetimeIndex(noon + offsets), *WALTON)
    worst = float(np.max(np.abs(before - after)))
    check("elevation is symmetric either side of solar noon", worst < 0.35,
          f"largest gap {worst:.3f} deg")


def check_refraction_lifts_the_horizon():
    """The known figure is about 34 arcmin, a bit over half a degree."""
    lift = float(solar._refraction(np.array([0.0]))[0])
    check("refraction at the horizon is about 0.5 deg",
          0.4 < lift < 0.65, f"{lift:.3f} deg")
    check("refraction is negligible overhead",
          float(solar._refraction(np.array([88.0]))[0]) == 0.0)
    check("refraction never returns a non-finite value",
          np.isfinite(solar._refraction(
              np.array([-90.0, -0.6, -0.5, 0.0, 5.0, 85.0, 90.0]))).all())


def check_ranges_and_night():
    index = pd.date_range("2025-03-01", periods=24 * 40, freq="h", tz="UTC")
    elev, azim = solar.position(index, *WALTON)
    check("elevation stays within [-90, 90]",
          bool(np.all((elev >= -90.0) & (elev <= 90.0))),
          f"{elev.min():.2f} to {elev.max():.2f}")
    finite = azim[np.isfinite(azim)]
    check("azimuth stays within [0, 360)",
          bool(np.all((finite >= 0.0) & (finite < 360.0))))
    check("the sun is below the horizon at local midnight",
          bool(np.all(elev[index.hour == 8] < 0.0)),
          "08:00 UTC is midnight at Walton")


def check_sun_in_view_is_not_circular():
    """The reason azimuth is projected rather than used raw."""
    # Virginia Beach faces east (90). Sun due east is in view, due west is not.
    check("sun due east is in view of an east-facing camera",
          abs(float(solar.sun_in_view(90.0, 90.0)) - 1.0) < 1e-9)
    check("sun due west is behind an east-facing camera",
          abs(float(solar.sun_in_view(270.0, 90.0)) + 1.0) < 1e-9)
    # The wrap that would break a rank correlation on raw degrees.
    near = [float(solar.sun_in_view(a, 0.0)) for a in (359.0, 1.0)]
    check("359 and 1 degrees project to nearly the same value",
          abs(near[0] - near[1]) < 1e-3, f"{near[0]:.6f} vs {near[1]:.6f}")


def check_glare_index_shape():
    east = 90.0
    low_in_view = float(solar.glare_index(5.0, 90.0, east))
    high_in_view = float(solar.glare_index(70.0, 90.0, east))
    behind = float(solar.glare_index(5.0, 270.0, east))
    night = float(solar.glare_index(-10.0, 90.0, east))
    check("glare is strongest with a low sun in view",
          low_in_view > high_in_view > 0.0,
          f"low {low_in_view:.3f} vs high {high_in_view:.3f}")
    check("glare is zero with the sun behind the camera", behind == 0.0)
    check("glare is zero at night", night == 0.0)


def main():
    print("solar offline checks\n")
    check_equation_of_time_extremes()
    check_solstice_noon_elevation()
    check_noon_bearing_is_south_in_the_north()
    check_longitude_sign_puts_noon_at_the_right_time()
    check_elevation_is_symmetric_about_noon()
    check_refraction_lifts_the_horizon()
    check_ranges_and_night()
    check_sun_in_view_is_not_circular()
    check_glare_index_shape()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
