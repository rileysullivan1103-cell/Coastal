"""Conditions for every site, from the pullers this project already has.

Nothing here re-implements a source. ERA5 and the wave model come from
pull_site_observations.open_meteo / fetch_marine, tide and water temperature
from pull_observations.pull_coops_series, the buoy from
pull_observations.pull_buoy, and all of them keep their retry and gap
behaviour. What this module adds is the part those scripts do not do:
caching by the thing that is actually shared.

  ERA5 and the wave model are cached per 0.1-degree GRID CELL, because two
  beaches 3 km apart are one cell and pulling both is the same request twice.
  CO-OPS is cached per GAUGE, for the same reason at a larger scale -- a
  county's worth of beaches share one tide gauge.

That is the difference between a few hundred requests and a few thousand,
and it is why a national covariate pull finishes.

THE SHORE NORMAL. Wind is pre-registered as onshore/alongshore components,
which need the direction the beach faces. Nationally that is not known, so it
is taken from the best available source per site and the source is RECORDED:

  manual            wq/strata_overrides.csv, a beach somebody looked at
  mop               CDIP MOP's published metaShoreNormal (California only)
  coastline_tangent the outward normal of the coastline tangent fitted over
                    +/-250 m by wq/spatial.py, which is a local measurement
                    of which way the beach faces
  marine_nudge      the bearing to the ocean cell Open-Meteo returned, when
                    the site's own cell was land. A fallback for a site with
                    no coastline linework.
  region            a coastal default. Crude. Every wind result computed off
                    one is reported separately, never mixed in.

The rip pipeline was computing onshore components at Santa Cruz off a
26-degree error until the MOP file supplied the real normal. The lesson taken
from that is not "guess better", it is "carry where the number came from".
"""

import math
import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import requests

from . import config

CACHE_DIR = os.path.join(config.RAW_DIR, "covariates")
COOPS_DATUMS_URL = ("https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/"
                    "stations/{station}/datums.json")

# Compass bearing you face looking out to sea, by region. A default of last
# resort — see the module docstring.
REGION_SHORE_NORMAL = {
    "Pacific": 270.0, "Atlantic": 90.0, "Gulf": 180.0, "Great Lakes": None,
}

ERA5_COLUMNS = ["rain_24h_mm", "rain_48h_mm", "rain_72h_mm",
                "temperature_2m", "wind_speed_10m", "wind_direction_10m"]
MARINE_COLUMNS = ["wave_height", "wave_period"]

# What this pipeline ASKS Open-Meteo for, as opposed to what the rip pipeline
# in analyze_drivers.py asks for out of the same fetchers. The free tier is
# metered by variables x days rather than by requests, so every column fetched
# and then thrown away is quota that buys no covariate: the wave model was
# being asked for eight columns to keep two, at four times the necessary
# price, and ERA5 for a gust column nothing here reads. Narrowing these two
# lists is the cheapest way to make a Northeast-scale refetch fit.
ERA5_VARS = ["precipitation", "wind_speed_10m", "wind_direction_10m",
             "temperature_2m"]
MARINE_VARS = ["wave_height", "wave_period"]


def cell_key(lat, lon, size=None):
    """The grid cell a site falls in, as a cache key."""
    size = size or config.GRID_CELL_DEGREES
    return (f"{math.floor(float(lat) / size) * size:.2f}_"
            f"{math.floor(float(lon) / size) * size:.2f}")


def cell_centre(lat, lon, size=None):
    size = size or config.GRID_CELL_DEGREES
    return (math.floor(float(lat) / size) * size + size / 2,
            math.floor(float(lon) / size) * size + size / 2)


# A cached emptiness has to be written so that it can be READ BACK. An
# empty DataFrame with no columns serialises to a single newline, and
# pd.read_csv raises EmptyDataError on that -- so the first gauge with no
# water temperature cached a file that the next site sharing that gauge
# could not open, and a 120-site run died at site 17. The marker column
# makes the emptiness a header pandas can parse rather than a byte it
# chokes on.
_EMPTY_MARKER = "_no_rows_returned"


class SourceUnavailable(RuntimeError):
    """A source did not answer for one site.

    This is NOT 'answered, and there is nothing there'. That second thing is
    a fact about the site -- this gauge carries no water temperature, this
    cell is not ocean -- and is worth caching. A gateway timeout is a fact
    about one minute of one afternoon, and caching it would freeze that
    minute into a permanent answer that nothing downstream could tell apart
    from a real absence.

    Never fatal either. One unhandled 504 out of NOAA killed a 2854-site
    covariate pull at site 405, forty minutes in. The covariate a refusal
    costs goes missing, the coverage rule sees the hole, and the manifest
    records it -- which is the honest outcome and costs one site, not a run.
    """

    def __init__(self, message, quiet=False):
        super().__init__(message)
        self.quiet = quiet


# After this many consecutive refusals a source is given up on for the rest
# of the run. Making a refusal non-fatal is only half the fix: a source that
# is down stays down, and asking it again at every remaining site pays its
# whole retry ladder two thousand more times. Open-Meteo's daily quota is the
# case that matters here -- once it is spent it is spent until tomorrow, and
# before this the run just kept going, quietly handing every remaining site
# no rain, no wind and no waves.
CIRCUIT_THRESHOLD = 3
_FAILURES = {}
_SUCCESSES = {}
_TRIPPED = {}
# Reasons a site went without a covariate, collected while that site is being
# built and written into its row of the per-site source table.
_MISSING = []


# Open-Meteo's free tier has three ceilings -- roughly 600 units a minute,
# 5,000 an hour and 10,000 a day -- and a unit is variables x days, not a
# request. One cell over this project's window is worth something like a
# hundred units, so an unpaced loop spends the MINUTE budget in about five
# cells and then reads the 429 as a refusal: that is how the Northeast run
# lost ERA5 at site 183 with its daily budget still largely unspent. Spacing
# the calls costs an hour of waiting and buys the whole region.
#
# CO-OPS is here for a different reason: it is the one host this pipeline
# shares with the camera pipeline in scan_cameras.py, and the two are often
# run at the same time. The throttle is per PROCESS -- _LAST_CALL is a module
# dict -- so two runs cannot coordinate, and together they can push a host
# into refusing what either alone would not. A second between calls costs
# nothing across a few dozen cached gauges and keeps the breaker out of it.
# Open-Meteo is NOT paced here. Its pacing belongs at the request, in
# pull_site_observations.open_meteo, because fetch_marine turns one call from
# this module into up to 25 requests and a throttle wrapped around that walk
# paces nothing. Pacing it in both places would only double the wait.
HOST_MIN_INTERVAL = {
    "api.tidesandcurrents.noaa.gov": 1.0,
}
_LAST_CALL = {}


def _throttle(host):
    """Hold a host to its minimum spacing. No-op for hosts without one."""
    wait = HOST_MIN_INTERVAL.get(host)
    if wait:
        last = _LAST_CALL.get(host)
        if last is not None:
            remaining = wait - (time.time() - last)
            if remaining > 0:
                time.sleep(remaining)
    _LAST_CALL[host] = time.time()


def _host_check(host):
    if host in _TRIPPED:
        raise SourceUnavailable(
            f"{host} was given up on earlier this run ({_TRIPPED[host]})",
            quiet=True)


def _host_record(host, ok, reason=""):
    if ok:
        _SUCCESSES[host] = _SUCCESSES.get(host, 0) + 1
        _FAILURES[host] = 0
        return
    _FAILURES[host] = _FAILURES.get(host, 0) + 1
    if _FAILURES[host] >= CIRCUIT_THRESHOLD and host not in _TRIPPED:
        _TRIPPED[host] = reason
        served = _SUCCESSES.get(host, 0)
        print(f"      GIVING UP on {host} after {_FAILURES[host]} consecutive "
              f"refusals ({served} successful response(s) this run): {reason}. "
              "Every site from here on goes without what it supplied, and the "
              "coverage rule will drop those covariates.")


def unavailable_sources():
    """Hosts abandoned this run, for the run summary and the manifest."""
    return dict(_TRIPPED)


def reset_sources():
    """Forget this run's refusals. For tests and for a second attempt."""
    _FAILURES.clear()
    _SUCCESSES.clear()
    _TRIPPED.clear()
    _MISSING.clear()


def _note_missing(reason):
    if reason not in _MISSING:
        _MISSING.append(reason)


# A refusal, as opposed to an answer of 'nothing here'. The difference
# decides whether the answer may be cached, so it is defined once, here,
# and pull_site_observations imports it for the notes it writes.
REFUSAL_STATUS = {408, 425, 429, 500, 502, 503, 504}
_REFUSAL_WORDS = re.compile(
    r"\b(Timeout|ConnectTimeout|ReadTimeout|ConnectionError|"
    r"ChunkedEncodingError|TooManyRedirects|SSLError)\b")


def is_refusal(note):
    """True when the note means 'the service did not answer'.

    False for 'the service answered, and there is nothing here' -- which is
    what 'no ocean cell found within ~22 km' and a 404 are. The first is a
    fact about this afternoon, the second a fact about the site, and only
    the second is worth writing down.
    """
    text = str(note or "")
    match = re.search(r"HTTP (\d{3})", text)
    if match:
        code = int(match.group(1))
        return code in REFUSAL_STATUS or code >= 500
    return bool(_REFUSAL_WORDS.search(text))


def _short_error(exc):
    response = getattr(exc, "response", None)
    if response is not None:
        return f"HTTP {response.status_code}"
    return type(exc).__name__


def _cached(name, builder):
    """Read a cached CSV, or build it and write it.

    A cached empty file means 'this was asked and there is nothing there',
    which is a real answer and is not retried on every run. It is written as
    a header-only CSV carrying _EMPTY_MARKER so that reading it back yields
    that answer instead of an exception: a cache that cannot be read is not
    a cache, it is a landmine under the next caller.

    A source that did not answer is not that, and nothing is written for it.
    See SourceUnavailable.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.csv")
    if os.path.exists(path):
        try:
            frame = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            # Written by the older form of this function, which serialised
            # an empty frame to a bare newline. Same meaning, so answer it
            # rather than making the caller fall over on a stale file.
            return None
        if frame.empty or _EMPTY_MARKER in frame.columns:
            return None
        return frame
    try:
        frame = builder()
    except SourceUnavailable as exc:
        _note_missing(str(exc))
        if not exc.quiet:
            print(f"      {exc} — not cached, so it is asked again next run")
        return None
    if frame is None or frame.empty:
        pd.DataFrame(columns=[_EMPTY_MARKER]).to_csv(path, index=False)
        return None
    frame.to_csv(path, index=False)
    return frame


def clear_empty_cache():
    """Delete the cached 'nothing here' answers, keeping the populated ones.

    For after a run that was refused rather than answered. Before this module
    told the two apart, a walk that Open-Meteo rate-limited was recorded as
    'no ocean cell found within ~22 km' and cached -- so a beach with waves
    stayed waveless on every later run, and no amount of re-running fixed it.
    New runs no longer cache a refusal; this is the way to undo the ones that
    already were.
    """
    if not os.path.isdir(CACHE_DIR):
        return 0
    removed = 0
    for name in sorted(os.listdir(CACHE_DIR)):
        if not name.endswith(".csv"):
            continue
        path = os.path.join(CACHE_DIR, name)
        try:
            frame = pd.read_csv(path, low_memory=False)
            empty = frame.empty or _EMPTY_MARKER in frame.columns
        except pd.errors.EmptyDataError:
            empty = True
        except Exception:  # noqa: BLE001 -- an unreadable cache is not an answer
            empty = True
        if empty:
            os.remove(path)
            removed += 1
    return removed


def window(years_back=None):
    years_back = years_back or config.YEARS_BACK
    import pull_site_observations as pso
    end = datetime.now(timezone.utc) - timedelta(days=pso.ARCHIVE_LAG_DAYS)
    return end - timedelta(days=365 * years_back), end


def fetch_era5(lat, lon, start, end):
    import pull_site_observations as pso
    host = urlsplit(pso.ERA5).netloc
    _host_check(host)
    _throttle(host)
    frame, note = pso.open_meteo(pso.ERA5, lat, lon, start, end, ERA5_VARS)
    if frame is None:
        print(f"      ERA5 failed: {note}")
        if is_refusal(note):
            _host_record(host, False, str(note)[:120])
            raise SourceUnavailable(f"ERA5: {note}")
        return None
    _host_record(host, True)
    return pso.add_rain_windows(frame)


def fetch_marine(lat, lon, start, end):
    """Waves, with the seaward bearing that the nudge revealed.

    pull_site_observations.fetch_marine already walks outward until it finds
    an ocean cell; the cell it lands on is, by construction, water. Where the
    site's own cell was land, the bearing to that cell is the best free
    estimate of which way the beach faces.
    """
    import pull_site_observations as pso
    host = urlsplit(pso.MARINE).netloc
    _host_check(host)
    _throttle(host)
    frame, note, used = pso.fetch_marine(lat, lon, start, end,
                                        return_cell=True,
                                        variables=MARINE_VARS)
    if frame is None:
        print(f"      marine failed: {note}")
        if is_refusal(note):
            _host_record(host, False, str(note)[:120])
            raise SourceUnavailable(f"marine: {note}")
        return None, None
    _host_record(host, True)
    bearing = None
    if isinstance(used, (tuple, list)) and len(used) == 2:
        dlat, dlon = float(used[0]) - lat, float(used[1]) - lon
        if abs(dlat) > 1e-6 or abs(dlon) > 1e-6:
            bearing = (math.degrees(math.atan2(dlon, dlat))) % 360
    return frame, bearing


def era5_for(lat, lon, start, end):
    key = cell_key(lat, lon)
    centre_lat, centre_lon = cell_centre(lat, lon)
    return _cached(f"era5_{key}", lambda: fetch_era5(centre_lat, centre_lon,
                                                     start, end))


def marine_for(lat, lon, start, end):
    key = cell_key(lat, lon)
    centre_lat, centre_lon = cell_centre(lat, lon)
    bearing_path = os.path.join(CACHE_DIR, f"marine_{key}.bearing")
    holder = {}

    def build():
        frame, bearing = fetch_marine(centre_lat, centre_lon, start, end)
        holder["bearing"] = bearing
        return frame

    frame = _cached(f"marine_{key}", build)
    if "bearing" in holder:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(bearing_path, "w") as handle:
            handle.write("" if holder["bearing"] is None
                         else f"{holder['bearing']:.1f}")
    bearing = None
    if os.path.exists(bearing_path):
        text = open(bearing_path).read().strip()
        bearing = float(text) if text else None
    return frame, bearing


def nearest_coops(lat, lon, stations, max_km=None):
    from .strata import _haversine_km
    max_km = max_km or config.MAX_TIDE_GAUGE_KM
    if stations is None or stations.empty:
        return None, None
    lats = pd.to_numeric(stations["lat"], errors="coerce").to_numpy()
    lons = pd.to_numeric(stations["lon"], errors="coerce").to_numpy()
    keep = np.isfinite(lats) & np.isfinite(lons)
    if not keep.any():
        return None, None
    subset = stations[keep]
    dist = _haversine_km(float(lat), float(lon), lats[keep], lons[keep])
    best = int(dist.argmin())
    if dist[best] > max_km:
        return None, None
    return str(subset.iloc[best]["station_id"]), float(dist[best])


def coops_for(station_id, product, start, end):
    import pull_observations as obs
    host = urlsplit(obs.COOPS_DATA).netloc

    def build():
        _host_check(host)
        _throttle(host)
        try:
            frame = obs.pull_coops_series(station_id, product, start, end)
        except requests.RequestException as exc:
            # A 400 is this gauge saying it does not serve this product --
            # a Great Lakes gauge asked for water_level, say. That is an
            # answer about the gauge, so it is cached and it does not count
            # against the host. Only a timeout or a 5xx is a refusal.
            reason = _short_error(exc)
            if is_refusal(reason):
                _host_record(host, False, reason)
                raise SourceUnavailable(
                    f"CO-OPS {product} at gauge {station_id}: {reason}") from exc
            _host_record(host, True)
            return pd.DataFrame()
        _host_record(host, True)
        if frame is not None and not frame.empty and product == "water_level":
            frame = obs.add_tide_state(frame)
        return frame

    return _cached(f"coops_{product}_{station_id}", build)


def coops_datums(station_ids):
    """MHHW and MLLW per gauge, for the tidal_range_m stratum."""
    import scan_cameras as scan
    rows = []
    for station in station_ids:
        def build(station=station):
            host = urlsplit(COOPS_DATUMS_URL).netloc
            _host_check(host)
            _throttle(host)
            try:
                resp = scan.get_with_retry(
                    COOPS_DATUMS_URL.format(station=station))
            except requests.RequestException as exc:
                reason = _short_error(exc)
                _host_record(host, False, reason)
                raise SourceUnavailable(
                    f"CO-OPS datums at gauge {station}: {reason}") from exc
            if resp.status_code != 200:
                # A 404 is this gauge saying it has no published datums, which
                # is an answer. A 5xx or a 429 is the service not answering.
                reason = f"HTTP {resp.status_code}"
                if is_refusal(reason):
                    _host_record(host, False, reason)
                    raise SourceUnavailable(
                        f"CO-OPS datums at gauge {station}: {reason}")
                _host_record(host, True)
                return pd.DataFrame()
            _host_record(host, True)
            payload = resp.json()
            values = {str(d.get("name", "")).upper(): d.get("value")
                      for d in payload.get("datums", [])}
            return pd.DataFrame([{"station_id": station,
                                  "mhhw": values.get("MHHW"),
                                  "mllw": values.get("MLLW")}])

        frame = _cached(f"datums_{station}", build)
        if frame is not None and not frame.empty:
            rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["station_id", "mhhw", "mllw", "lat", "lon"])
    return pd.concat(rows, ignore_index=True)


def shore_normal_for(site, marine_bearing=None, overrides=None):
    """(degrees, source). See the module docstring for why source travels."""
    station = str(site.get("station_id"))
    if overrides is not None and len(overrides):
        if station in overrides.index and "shore_normal_deg" in overrides.columns:
            value = overrides.loc[station, "shore_normal_deg"]
            if pd.notna(value):
                return float(value), "manual"
    mop = site.get("mop_shore_normal")
    if pd.notna(mop):
        return float(mop), "mop"
    # From the coastline tangent fitted over +/-250 m by wq/spatial.py, which
    # is a real local measurement of which way the beach faces rather than an
    # inference from where the wave model happened to find water.
    fitted = site.get("shore_normal_deg")
    if fitted is not None and pd.notna(fitted):
        return float(fitted), str(site.get("shore_normal_source")
                                  or "coastline_tangent")
    if marine_bearing is not None and np.isfinite(marine_bearing):
        return float(marine_bearing), "marine_nudge"
    default = REGION_SHORE_NORMAL.get(site.get("region"))
    if default is None:
        return np.nan, "none"
    return float(default), "region"


def wind_components(speed, direction_from, shore_normal):
    """(onshore, alongshore) in m/s.

    Meteorological direction is the direction wind blows FROM, and the shore
    normal points seaward, so wind arriving from seaward (direction_from ==
    shore_normal) is fully onshore. Everything goes through the cosine rather
    than through a degree difference, for the reason compare_wind_sources.py
    gives: 359 and 1 are two degrees apart, not 358.
    """
    if shore_normal is None or not np.isfinite(shore_normal):
        return (pd.Series(np.nan, index=getattr(speed, "index", None)),
                pd.Series(np.nan, index=getattr(speed, "index", None)))
    offset = np.radians(pd.to_numeric(direction_from, errors="coerce")
                        - float(shore_normal))
    magnitude = pd.to_numeric(speed, errors="coerce")
    return magnitude * np.cos(offset), magnitude * np.sin(offset)


def hourly_frame(site, start, end, coops_stations=None, overrides=None):
    """Every covariate for one site, hourly, on a UTC index.

    Returns (frame, meta). meta records which sources answered, which gauge
    served the site, and where the shore normal came from -- all of which the
    per-site output table carries, so a coefficient can always be traced to
    the covariate that produced it.
    """
    lat, lon = float(site["lat"]), float(site["lon"])
    meta = {"station_id": site.get("station_id"), "sources": [],
            "tide_station": None, "tide_km": np.nan}
    parts = []
    # Why this site went without something, if it did. A site with no waves
    # because it is thirty kilometres up an estuary and a site with no waves
    # because the wave model was rate-limiting are the same empty column, and
    # only one of them is a fact about the beach.
    _MISSING.clear()

    era5 = era5_for(lat, lon, start, end)
    if era5 is not None:
        era5 = era5.copy()
        era5["time"] = pd.to_datetime(era5["time"], utc=True)
        parts.append(era5.set_index("time"))
        meta["sources"].append("era5")

    waves, bearing = marine_for(lat, lon, start, end)
    if waves is not None:
        waves = waves.copy()
        waves["time"] = pd.to_datetime(waves["time"], utc=True)
        keep = [c for c in MARINE_COLUMNS if c in waves.columns]
        parts.append(waves.set_index("time")[keep])
        meta["sources"].append("marine")

    station, km = nearest_coops(lat, lon, coops_stations)
    if station:
        meta["tide_station"], meta["tide_km"] = station, km
        tide = coops_for(station, "water_level", start, end)
        if tide is not None:
            tide = tide.copy()
            tide["time"] = pd.to_datetime(tide["t"] if "t" in tide else tide["time"],
                                          utc=True, errors="coerce")
            keep = [c for c in ("level_m", "rate_m_per_hr") if c in tide.columns]
            if keep:
                parts.append(tide.dropna(subset=["time"]).set_index("time")[keep]
                             .resample("h").mean())
                meta["sources"].append("tide")
        temp = coops_for(station, "water_temperature", start, end)
        if temp is not None:
            temp = temp.copy()
            temp["time"] = pd.to_datetime(temp["t"] if "t" in temp else temp["time"],
                                          utc=True, errors="coerce")
            column = next((c for c in ("v", "water_temp_c", "water_temp")
                           if c in temp.columns), None)
            if column:
                series = (temp.dropna(subset=["time"]).set_index("time")[[column]]
                          .rename(columns={column: "water_temp_c"}))
                series["water_temp_c"] = pd.to_numeric(series["water_temp_c"],
                                                       errors="coerce")
                parts.append(series.resample("h").mean())
                meta["sources"].append("water_temp")

    if not parts:
        meta["unavailable"] = "; ".join(_MISSING)
        return None, meta

    frame = parts[0]
    for part in parts[1:]:
        frame = frame.join(part, how="outer", rsuffix="_dup")

    normal, source = shore_normal_for(site, bearing, overrides)
    meta["shore_normal_deg"], meta["shore_normal_source"] = normal, source
    if "wind_speed_10m" in frame.columns:
        onshore, alongshore = wind_components(frame["wind_speed_10m"],
                                              frame.get("wind_direction_10m"),
                                              normal)
        frame["wind_onshore_ms"] = onshore
        frame["wind_alongshore_ms"] = alongshore
    for column in config.PREDICTORS:
        if column not in frame.columns:
            frame[column] = np.nan
    meta["unavailable"] = "; ".join(_MISSING)
    return frame[config.PREDICTORS].sort_index(), meta


def join_samples(samples, hourly):
    """Attach conditions to samples at the sample's own hour where a time was
    reported, and to the daily mean where it was not.

    Which one happened is recorded per sample in join_resolution. WQP carries
    a sample time and the California CKAN feed does not, so a national
    distribution mixes both, and 'the tide at 9am' and 'the mean tide that
    day' are not the same predictor.
    """
    if hourly is None or hourly.empty or samples.empty:
        out = samples.copy()
        for column in config.PREDICTORS:
            out[column] = np.nan
        out["join_resolution"] = "none"
        return out

    hourly = hourly.sort_index()
    daily = hourly.copy()
    daily.index = daily.index.tz_convert(None).normalize()
    daily = daily.groupby(level=0).mean(numeric_only=True)

    out = samples.copy()
    stamps = pd.to_datetime(out["sampled_at"], utc=True, errors="coerce")
    # A midnight stamp is a date that was written as a datetime, not a sample
    # taken at midnight. Agencies do not sample at midnight; treating one as
    # an hourly join would attach the small hours' tide to a morning sample.
    has_time = stamps.notna() & ~((stamps.dt.hour == 0) & (stamps.dt.minute == 0))
    out["join_resolution"] = np.where(has_time, "hourly", "daily")

    hours = stamps.dt.floor("h")
    hourly_values = hourly.reindex(hours)
    days = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    daily_values = daily.reindex(days)

    for column in config.PREDICTORS:
        picked = np.where(has_time.to_numpy(),
                          hourly_values[column].to_numpy(),
                          daily_values[column].to_numpy())
        out[column] = picked
    return out


def build(sites, samples, coops_stations=None, overrides=None, start=None,
          end=None, progress=True):
    """Covariates for every site, joined to its samples. Returns (joined, meta)."""
    if start is None or end is None:
        start, end = window()
    joined, notes = [], []
    total = len(sites)
    for index, (_, site) in enumerate(sites.iterrows(), 1):
        station = str(site["station_id"])
        group = samples[samples["station_id"].astype(str) == station]
        if group.empty:
            continue
        if progress:
            print(f"  [{index}/{total}] {station}: {len(group)} samples")
        hourly, meta = hourly_frame(site, start, end, coops_stations, overrides)
        attached = join_samples(group, hourly)
        attached["station_id"] = station
        joined.append(attached)
        meta["sources"] = ",".join(meta.get("sources", []))
        meta["n_samples"] = len(group)
        notes.append(meta)
    if not joined:
        return pd.DataFrame(), pd.DataFrame()
    frame = pd.DataFrame(notes)
    _report_refusals(frame)
    return (pd.concat(joined, ignore_index=True), frame)


def _report_refusals(notes):
    """Say, at the end, how much of this run was answered and how much was not.

    Without this the damage is invisible. A run that spends Open-Meteo's
    daily quota at site 600 writes exactly the same tables as one that got
    every request, only with two thousand sites' weather quietly missing --
    and the coverage rule then drops the predictor for everybody, including
    the six hundred sites that did have it.
    """
    given_up = unavailable_sources()
    starved = 0
    if "unavailable" in notes.columns:
        starved = int(notes["unavailable"].fillna("").astype(bool).sum())
    if not given_up and not starved:
        return
    print("\n  SOURCES THAT DID NOT ANSWER")
    for host, reason in sorted(given_up.items()):
        print(f"    gave up on {host}: {reason}")
    if starved:
        print(f"    {starved}/{len(notes)} site(s) went without at least one "
              "source. covariate_sources.csv says which, per site, in the "
              "'unavailable' column.")
    print("    Nothing about this was cached, so re-running --covariates "
          "asks again for exactly those sites.")
