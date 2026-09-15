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
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

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


def cell_key(lat, lon, size=None):
    """The grid cell a site falls in, as a cache key."""
    size = size or config.GRID_CELL_DEGREES
    return (f"{math.floor(float(lat) / size) * size:.2f}_"
            f"{math.floor(float(lon) / size) * size:.2f}")


def cell_centre(lat, lon, size=None):
    size = size or config.GRID_CELL_DEGREES
    return (math.floor(float(lat) / size) * size + size / 2,
            math.floor(float(lon) / size) * size + size / 2)


def _cached(name, builder):
    """Read a cached CSV, or build it and write it. A cached empty file means
    'this was tried and there is nothing there', which is a real answer and
    is not retried on every run."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.csv")
    if os.path.exists(path):
        frame = pd.read_csv(path, low_memory=False)
        return frame if not frame.empty else None
    frame = builder()
    (frame if frame is not None else pd.DataFrame()).to_csv(path, index=False)
    return frame if frame is not None and not frame.empty else None


def window(years_back=None):
    years_back = years_back or config.YEARS_BACK
    import pull_site_observations as pso
    end = datetime.now(timezone.utc) - timedelta(days=pso.ARCHIVE_LAG_DAYS)
    return end - timedelta(days=365 * years_back), end


def fetch_era5(lat, lon, start, end):
    import pull_site_observations as pso
    frame, note = pso.open_meteo(pso.ERA5, lat, lon, start, end, pso.ERA5_VARS)
    if frame is None:
        print(f"      ERA5 failed: {note}")
        return None
    return pso.add_rain_windows(frame)


def fetch_marine(lat, lon, start, end):
    """Waves, with the seaward bearing that the nudge revealed.

    pull_site_observations.fetch_marine already walks outward until it finds
    an ocean cell; the cell it lands on is, by construction, water. Where the
    site's own cell was land, the bearing to that cell is the best free
    estimate of which way the beach faces.
    """
    import pull_site_observations as pso
    frame, note, used = pso.fetch_marine(lat, lon, start, end,
                                        return_cell=True)
    if frame is None:
        print(f"      marine failed: {note}")
        return None, None
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

    def build():
        frame = obs.pull_coops_series(station_id, product, start, end)
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
            resp = scan.get_with_retry(COOPS_DATUMS_URL.format(station=station))
            if resp.status_code != 200:
                return pd.DataFrame()
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
    return (pd.concat(joined, ignore_index=True), pd.DataFrame(notes))
