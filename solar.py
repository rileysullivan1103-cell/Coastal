"""Where the sun is, from a timestamp and a position on the earth.

This exists to test a specific suspicion. At Virginia Beach, air temperature
is the second strongest thing in the rip table -- rho_hrmo about -0.36 against
detection rate, detections and box area alike -- and negative, so the detector
fires LESS on warm hours. Water temperature cannot explain it (there is none
at that site) and there is no physical story in which warm air suppresses a
rip. There is an obvious story in which it suppresses a DETECTOR: a warm hour
is a clear, high-sun, hazy hour, and sun glare off water is exactly what stops
a camera resolving the surface texture a rip detector keys on.

Testing that needs the sun's position, and the sun's position needs no new
data source. It is a closed-form function of UTC timestamp and latitude and
longitude, all three of which are already on disk.

The algorithm is NOAA's, the one behind their solar calculator: good to about
0.01 degrees between 1900 and 2100, which is far finer than anything this
argument turns on. It is implemented here rather than taken from a library
because no solar library is installed and the whole of it is fifty lines.

Nothing here reads a file or touches the network.
"""

import numpy as np
import pandas as pd


def _julian_century(index):
    """Julian centuries since J2000.0, from a UTC DatetimeIndex."""
    return (np.asarray(pd.DatetimeIndex(index).to_julian_date(), dtype=float)
            - 2451545.0) / 36525.0


def _declination_and_eot(t):
    """(solar declination in degrees, equation of time in minutes).

    Straight NOAA. `t` is Julian centuries. The equation of time is the gap
    between clock noon and the sun actually crossing the meridian; it reaches
    about -14 minutes in February and +16 in November, which is what the
    offline test pins the algorithm against.
    """
    # Geometric mean longitude and anomaly of the sun.
    mean_long = np.radians((280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0)
    mean_anom = np.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)

    # Equation of centre: the correction from a circular orbit to the real one.
    centre = (np.sin(mean_anom) * (1.914602 - t * (0.004817 + 0.000014 * t))
              + np.sin(2 * mean_anom) * (0.019993 - 0.000101 * t)
              + np.sin(3 * mean_anom) * 0.000289)
    true_long = np.degrees(mean_long) + centre

    # Apparent longitude corrects for nutation and aberration.
    omega = np.radians(125.04 - 1934.136 * t)
    app_long = np.radians(true_long - 0.00569 - 0.00478 * np.sin(omega))

    # Obliquity of the ecliptic -- the earth's axial tilt, about 23.44 deg.
    mean_obliq = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059
                 - t * 0.001813))) / 60.0) / 60.0
    obliq = np.radians(mean_obliq + 0.00256 * np.cos(omega))

    declination = np.degrees(np.arcsin(np.sin(obliq) * np.sin(app_long)))

    y = np.tan(obliq / 2.0) ** 2
    eot = 4.0 * np.degrees(
        y * np.sin(2 * mean_long)
        - 2 * eccentricity * np.sin(mean_anom)
        + 4 * eccentricity * y * np.sin(mean_anom) * np.cos(2 * mean_long)
        - 0.5 * y * y * np.sin(4 * mean_long)
        - 1.25 * eccentricity * eccentricity * np.sin(2 * mean_anom))
    return declination, eot


def _refraction(elevation_deg):
    """Atmospheric bending, in degrees, added to the geometric elevation.

    Worth carrying rather than ignoring: it is negligible overhead but about
    0.5 degrees at the horizon, and the horizon is where this whole question
    lives. A sun the geometry puts just below the horizon is still visible,
    and still glaring off the water into the camera.
    """
    elev = np.asarray(elevation_deg, dtype=float)
    # tan blows up at zero elevation; the low-angle branch below does not use
    # it, so clip only to keep the expression finite where it is unused.
    tan_e = np.tan(np.radians(np.clip(elev, -89.9, 89.9)))
    with np.errstate(divide="ignore", invalid="ignore"):
        high = 58.1 / tan_e - 0.07 / tan_e ** 3 + 0.000086 / tan_e ** 5
        very_low = -20.772 / tan_e
    low = 1735.0 + elev * (-518.2 + elev * (103.4 + elev * (-12.79 + elev * 0.711)))
    seconds = np.where(elev > 85.0, 0.0,
               np.where(elev > 5.0, high,
                np.where(elev > -0.575, low, very_low)))
    return np.nan_to_num(seconds, nan=0.0, posinf=0.0, neginf=0.0) / 3600.0


def position(index, lat, lon):
    """(elevation, azimuth) in degrees for each UTC timestamp.

    Elevation is refraction-corrected and measured from the horizon, so it is
    negative at night. Azimuth is a compass bearing: 0 north, 90 east, 180
    south, 270 west.

    `lon` is degrees east-positive, the same sign convention the site table
    uses (Walton is -122).
    """
    index = pd.DatetimeIndex(index)
    t = _julian_century(index)
    declination, eot = _declination_and_eot(t)

    # Minutes past UTC midnight. Working in UTC throughout means the timezone
    # term of the NOAA formulation is zero and cannot be got wrong.
    minutes = (index.hour * 60.0 + index.minute + index.second / 60.0).to_numpy(float)
    true_solar_minutes = (minutes + eot + 4.0 * lon) % 1440.0
    hour_angle = np.radians(true_solar_minutes / 4.0 - 180.0)

    lat_r, dec_r = np.radians(lat), np.radians(declination)
    cos_zenith = np.clip(np.sin(lat_r) * np.sin(dec_r)
                         + np.cos(lat_r) * np.cos(dec_r) * np.cos(hour_angle),
                         -1.0, 1.0)
    zenith = np.arccos(cos_zenith)
    elevation = 90.0 - np.degrees(zenith)

    sin_zenith = np.sin(zenith)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_az = ((np.sin(lat_r) * cos_zenith - np.sin(dec_r))
                  / (np.cos(lat_r) * sin_zenith))
    azimuth = np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0)))
    # arccos cannot tell morning from afternoon; the hour angle can.
    azimuth = np.where(hour_angle > 0.0, (azimuth + 180.0) % 360.0,
                       (540.0 - azimuth) % 360.0)
    # Directly overhead or at the pole the bearing is undefined rather than
    # wrong. Leave it NaN so it drops out of a correlation instead of
    # contributing an invented direction.
    azimuth = np.where(sin_zenith < 1e-9, np.nan, azimuth)

    return elevation + _refraction(elevation), azimuth


def sun_in_view(azimuth_deg, seaward_bearing_deg):
    """+1 with the sun straight out over the water, -1 with it behind the camera.

    Azimuth is circular, so it is the one column here that must not be fed to
    a rank correlation raw: 359 degrees and 1 degree are a degree apart and
    rank 358 apart, and any monotone statistic reads that as a huge move. This
    projects it onto the direction the camera actually looks, which is both
    non-circular and the only thing about the bearing that matters for glare.
    """
    return np.cos(np.radians(pd.to_numeric(azimuth_deg, errors="coerce")
                             - seaward_bearing_deg))


def glare_index(elevation_deg, azimuth_deg, seaward_bearing_deg):
    """A blunt index of sun-glare geometry: sun over the water AND low.

    Specular glare off a sea surface reaches a near-horizontal camera when the
    sun is in front of it and close to the horizon, so this is the in-view
    component weighted by cos(elevation), zeroed once the sun is down or
    behind the camera.

    This is a constructed quantity, not a measurement, and the functional form
    is one defensible choice among several. Nothing in the conclusion rests on
    it: elevation, bearing and cloud cover are reported separately and carry
    the argument on their own.
    """
    elev = pd.to_numeric(elevation_deg, errors="coerce")
    facing = np.clip(sun_in_view(azimuth_deg, seaward_bearing_deg), 0.0, None)
    above = np.where(elev > 0.0, np.cos(np.radians(np.clip(elev, 0.0, 90.0))), 0.0)
    return facing * above
