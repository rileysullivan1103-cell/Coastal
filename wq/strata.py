"""A2: assign every site to its strata, from metadata and map data ONLY.

Nothing in this module may read a sample value, a coefficient, or anything
derived from one. It runs before wq/manifest.py, which runs before any fit,
and its only inputs are the station table and the auxiliary station lists
(streams, facilities, tide gauges) that come from the same metadata pulls.

Three of the seven variables are honest guesses and say so in a `_source`
column beside them:

  beach_type        WQP's own site type where it is decisive (Ocean ->
                    open_coast, Estuary -> enclosed_bay), station-name
                    keywords otherwise, a hand-edited override file above
                    both. Name matching is crude and its share is reported,
                    because "Storm Drain at Main St" is decisive and
                    "Beach 3" is not.
  freshwater_input  a WQP Stream or Spring station within 500 m. That is a
                    proxy for a creek mouth, not a survey of one: it detects
                    creeks somebody monitors and misses creeks nobody does,
                    so its 'no' is weaker than its 'yes'.
  outfall_present   a WQP Facility station within 1 km, plus outfall/CSO
                    keywords. Same asymmetry.

watershed_area_km2 and impervious_frac have NO source wired in. They are
emitted as all-NaN columns so the coverage rule in wq/manifest.py drops them
from the specification before fitting, which is what A2 asks for -- rather
than being quietly omitted here and quietly reintroduced later.
"""

import os

import numpy as np
import pandas as pd

from . import config

# Great Lakes, as coordinate boxes. A station inside one is fresh water and
# is stratified as Great Lakes regardless of which state it sits in, because
# Michigan has both a Great Lakes shoreline and inland water.
GREAT_LAKE_BOXES = (
    (41.3, -92.2, 49.1, -84.3),   # Superior, Michigan, and the western basin
    (41.3, -84.3, 46.6, -78.7),   # Huron, Erie, St Clair
    (43.1, -79.9, 44.5, -75.8),   # Ontario
)

GULF_STATES = {"TX", "LA", "MS", "AL"}
PACIFIC_STATES = {"CA", "OR", "WA", "AK", "HI", "GU", "AS", "MP"}
ATLANTIC_STATES = {"ME", "NH", "MA", "RI", "CT", "NY", "NJ", "DE", "MD", "VA",
                   "NC", "SC", "GA", "PR", "VI"}
# FL touches both. Split on longitude at the peninsula's tip.
SPLIT_STATES = {"FL": ("Gulf", "Atlantic", -81.5)}

STORM_DRAIN_WORDS = ("storm drain", "stormdrain", "storm-drain", "sd at",
                     "drain at", "outfall", "discharge", "culvert", "pipe")
ENCLOSED_WORDS = ("bay", "harbor", "harbour", "lagoon", "slough", "estuary",
                  "inlet", "marina", "basin", "cove", "sound", "channel",
                  "yacht", "pier", "dock")
OPEN_WORDS = ("ocean", "surf", "beach front", "oceanfront", "state beach",
              "shoreline", "seashore", "strand")
FRESHWATER_TYPES = ("Stream", "Spring", "River")
FACILITY_TYPES = ("Facility", "Waste", "Outfall", "Sewer", "Storm")
OUTFALL_NAME_WORDS = ("outfall", "cso", "sewer", "discharge", "wwtp")


def _norm(text):
    return str(text or "").strip().lower()


def region_of(state, lat, lon):
    """Pacific | Atlantic | Gulf | Great Lakes, from the state and the point."""
    if pd.isna(lat) or pd.isna(lon):
        return None
    for min_lat, min_lon, max_lat, max_lon in GREAT_LAKE_BOXES:
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return "Great Lakes"
    code = str(state or "").strip().upper()
    if code in SPLIT_STATES:
        west, east, meridian = SPLIT_STATES[code]
        return west if lon < meridian else east
    if code in GULF_STATES:
        return "Gulf"
    if code in PACIFIC_STATES:
        return "Pacific"
    if code in ATLANTIC_STATES:
        return "Atlantic"
    # No state code: fall back to the coordinate alone.
    if lon < -100:
        return "Pacific"
    if lat < 31 and -98 < lon < -81:
        return "Gulf"
    return "Atlantic"


def water_class(region):
    """Which threshold block applies. Great Lakes beaches are fresh water and
    are judged against a different enterococci criterion than the coast."""
    return "fresh" if region == "Great Lakes" else "marine"


def beach_type_of(site_type, name):
    """(value, how_it_was_decided). Keywords beat the site type, because a
    storm drain monitored as an 'Ocean' site is still a storm drain."""
    text = _norm(name)
    if any(word in text for word in STORM_DRAIN_WORDS):
        return "storm_drain_adjacent", "name"
    kind = _norm(site_type)
    if any(word in text for word in ENCLOSED_WORDS):
        return "enclosed_bay", "name"
    if "estuary" in kind:
        return "enclosed_bay", "site_type"
    if any(word in text for word in OPEN_WORDS):
        return "open_coast", "name"
    if "ocean" in kind or "great lake" in kind:
        return "open_coast", "site_type"
    return None, "unknown"


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance, vectorised on the second point. Same formula as
    find_candidate_sites.haversine_km, repeated here only so this module can
    be imported without ndbc_api installed."""
    radius = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2)
    return 2 * radius * np.arcsin(np.sqrt(a))


def _within(sites, others, km):
    """Boolean per site: is any `others` row within km. Empty `others` gives
    all-NaN rather than all-False, because "nothing was checked" and "nothing
    is there" are different answers and only one of them is evidence."""
    if others is None or others.empty:
        return pd.Series(np.nan, index=sites.index, dtype="object")
    lats = pd.to_numeric(others["lat"], errors="coerce").to_numpy()
    lons = pd.to_numeric(others["lon"], errors="coerce").to_numpy()
    keep = np.isfinite(lats) & np.isfinite(lons)
    lats, lons = lats[keep], lons[keep]
    if not len(lats):
        return pd.Series(np.nan, index=sites.index, dtype="object")
    out = []
    for _, row in sites.iterrows():
        lat, lon = row.get("lat"), row.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            out.append(np.nan)
            continue
        dist = _haversine_km(float(lat), float(lon), lats, lons)
        out.append("yes" if dist.min() <= km else "no")
    return pd.Series(out, index=sites.index, dtype="object")


def _filter_types(frame, wanted):
    if frame is None or frame.empty or "site_type" not in frame.columns:
        return frame
    kinds = frame["site_type"].astype(str).str.lower()
    mask = kinds.apply(lambda k: any(w.lower() in k for w in wanted))
    return frame[mask]


def load_overrides(path=None):
    """Hand-classified sites, which beat every rule above.

    A beach somebody has actually looked at is better evidence than a keyword,
    and this is the file to put that in: station_id plus any stratum column.
    """
    path = path or config.STRATA_OVERRIDE_CSV
    if not os.path.exists(path):
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype=str, comment="#")
    if "station_id" not in frame.columns:
        raise SystemExit(f"{path} has no station_id column")
    return frame.set_index(frame["station_id"].astype(str))


def tidal_range(sites, datums):
    """MHHW - MLLW at the nearest CO-OPS gauge, or NaN beyond the radius.

    `datums` carries one row per gauge: station_id, lat, lon, mhhw, mllw.
    wq/covariates.py fetches it; strata never calls the network itself, so
    this module stays runnable offline and testable against a fixture.
    """
    if datums is None or datums.empty:
        return (pd.Series(np.nan, index=sites.index),
                pd.Series(None, index=sites.index, dtype="object"))
    lats = pd.to_numeric(datums["lat"], errors="coerce").to_numpy()
    lons = pd.to_numeric(datums["lon"], errors="coerce").to_numpy()
    ranges = (pd.to_numeric(datums["mhhw"], errors="coerce")
              - pd.to_numeric(datums["mllw"], errors="coerce")).to_numpy()
    ids = datums["station_id"].astype(str).to_numpy()
    keep = np.isfinite(lats) & np.isfinite(lons)
    lats, lons, ranges, ids = lats[keep], lons[keep], ranges[keep], ids[keep]
    values, gauges = [], []
    for _, row in sites.iterrows():
        lat, lon = row.get("lat"), row.get("lon")
        if pd.isna(lat) or pd.isna(lon) or not len(lats):
            values.append(np.nan)
            gauges.append(None)
            continue
        dist = _haversine_km(float(lat), float(lon), lats, lons)
        best = int(dist.argmin())
        if dist[best] > config.MAX_TIDE_GAUGE_KM:
            values.append(np.nan)
            gauges.append(None)
        else:
            values.append(float(ranges[best]))
            gauges.append(str(ids[best]))
    return (pd.Series(values, index=sites.index),
            pd.Series(gauges, index=sites.index, dtype="object"))


def assign(sites, neighbours=None, datums=None, overrides=None):
    """Every stratum for every site. Returns a copy of `sites` with the
    stratum columns and their `_source` companions added.

    `neighbours` is the auxiliary station table (all site types, any
    characteristic) used for the freshwater and outfall proxies.
    """
    out = sites.copy()
    if out.empty:
        for column in config.STRATA:
            out[column] = pd.Series(dtype="object")
        return out

    out["region"] = [region_of(r.get("state"), r.get("lat"), r.get("lon"))
                     for _, r in out.iterrows()]
    out["water_class"] = out["region"].map(water_class)

    decided = [beach_type_of(r.get("site_type"), r.get("station_name"))
               for _, r in out.iterrows()]
    out["beach_type"] = [d[0] for d in decided]
    out["beach_type_source"] = [d[1] for d in decided]

    fresh = _filter_types(neighbours, FRESHWATER_TYPES)
    out["freshwater_input"] = _within(out, fresh,
                                      config.FRESHWATER_RADIUS_M / 1000.0)
    outfalls = _filter_types(neighbours, FACILITY_TYPES)
    nearby = _within(out, outfalls, config.OUTFALL_RADIUS_KM)
    # A station NAMED for an outfall is one either way, even where no facility
    # station was found nearby to confirm it. The name can only ever turn a
    # 'no' or an unchecked NaN into a 'yes', never the other way.
    by_name = out["station_name"].astype(str).str.lower().apply(
        lambda t: any(w in t for w in OUTFALL_NAME_WORDS))
    out["outfall_present"] = np.where(by_name, "yes", nearby)
    out["outfall_present"] = out["outfall_present"].where(
        pd.notna(nearby) | by_name, np.nan)

    out["tidal_range_m"], out["tidal_range_gauge"] = tidal_range(out, datums)

    # No offline source. Emitted deliberately so the coverage rule drops them.
    out["watershed_area_km2"] = np.nan
    out["impervious_frac"] = np.nan

    frame = load_overrides() if overrides is None else overrides
    if frame is not None and len(frame):
        applied = 0
        keys = out["station_id"].astype(str)
        for column in config.STRATA:
            if column not in frame.columns:
                continue
            mapping = frame[column].dropna()
            hits = keys.map(mapping)
            out[column] = hits.where(hits.notna(), out[column])
            if column == "beach_type":
                out["beach_type_source"] = np.where(
                    hits.notna(), "manual", out["beach_type_source"])
            applied += int(hits.notna().sum())
        if applied:
            print(f"  strata overrides applied to {applied} site-values")
    return out


def coverage(sites):
    """Populated share per stratum. This is what manifest.py decides on."""
    rows = []
    for column, spec in config.STRATA.items():
        if column not in sites.columns:
            share = 0.0
        elif sites.empty:
            share = 0.0
        else:
            share = float(sites[column].notna().mean())
        rows.append({"stratum": column, "coverage": share,
                     "required": spec["required_coverage"],
                     "keeps": share >= spec["required_coverage"]})
    return pd.DataFrame(rows)
