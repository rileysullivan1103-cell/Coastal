"""A2: assign every site to its strata, from metadata and map data ONLY.

Nothing here may read a sample value, a coefficient, or anything derived from
one. It runs before the results are pulled, which is the structural version
of "assign these from station metadata, not from the results".

What this module does NOT do any more is assign beach_type. An earlier
version classified it from station-name keywords -- "Harbor" meant enclosed,
"Ocean" meant open -- which is a threshold dressed up as a rule and it got
the ambiguous sites wrong in whichever direction the keyword list happened to
lean. beach_type is now assigned by hand from imagery (wq/review.py) and read
from beach_type_reviewed.csv, with the assigner and the date attached. If
nobody has done that work, beach_type is empty and the coverage rule drops
it, which is the honest outcome.

The continuous enclosure covariates -- land_fraction_5km, embayment_ratio,
curvature_1_per_km -- come from wq/spatial.py and are fitted on directly.
They also SORT the manual review list. They never assign the label.

  region          state code and coordinate boxes. Great Lakes wins over the
                  state, because Michigan has both a Great Lakes shoreline
                  and inland water, and the two take different criteria.
  water_class     fresh inside a Great Lakes box, marine everywhere else.
                  Not a stratum: it selects the exceedance threshold.
  tidal_range_m   MHHW - MLLW at the nearest CO-OPS gauge.
"""

import os

import numpy as np
import pandas as pd

from . import config, review

# Great Lakes, as coordinate boxes. A station inside one is fresh water and
# is stratified as Great Lakes regardless of which state it sits in.
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


def tidal_range(sites, datums):
    """MHHW - MLLW at the nearest CO-OPS gauge, or NaN beyond the radius.

    `datums` carries one row per gauge: station_id, lat, lon, mhhw, mllw.
    wq/covariates.py fetches it; this module never calls the network itself,
    so it stays runnable offline and testable against a fixture.
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
        distance = _haversine_km(float(lat), float(lon), lats, lons)
        best = int(distance.argmin())
        if distance[best] > config.MAX_TIDE_GAUGE_KM:
            values.append(np.nan)
            gauges.append(None)
        else:
            values.append(float(ranges[best]))
            gauges.append(str(ids[best]))
    return (pd.Series(values, index=sites.index),
            pd.Series(gauges, index=sites.index, dtype="object"))


# Two stations 50 metres apart on one beach, sampled by one agency on the same
# mornings, are not two beaches. This is the radius at which they are called
# one site cluster, and it is the same 150 m wq/clean.py already uses to decide
# that two station records describe the same place.
#
# NOT in config.SPEC_KEYS, and deliberately so: the cluster is a LABEL. It
# changes no coefficient, excludes no pair, and is not the unit D2 shuffles --
# that unit is the station, as registered. The day the shuffle moves to the
# cluster, this radius starts deciding p-values and belongs in the
# pre-registered block with a re-registration to match.
SITE_CLUSTER_RADIUS_KM = 0.15


def site_clusters(sites, radius_km=SITE_CLUSTER_RADIUS_KM):
    """Group stations that are the same stretch of beach. (labels, sizes).

    The study counts stations, and a station is not an independent beach. The
    five-state pass had sixteen NJDEP stations strung along two kilometres of
    Atlantic City shoreline; California has twenty-two CABEACH_WQX stations
    inside 472 metres at Cowell Beach, and Connecticut reports three separate
    identifiers all named SILVER SANDS STATE PARK BEACH. Every one of those
    counts once in D1's n_sites and gets its own draw in D2's shuffle.

    What this is NOT is a duplicate detector. That was the first guess and the
    data refused it: co-located same-organization pairs agree on a median 20%
    of their (date, analyte, value) readings, against 1.7% for far-apart pairs
    of the same organization -- far above chance, but nowhere near the
    same-record agreement that wq.clean.dedupe_across_sources drops on. The
    agreement is high because bacteria counts are read off discretised MPN
    tables and neighbouring points share their weather, not because the rows
    are copies. Dropping them would delete real samples. So nothing is
    dropped; the non-independence is labelled instead.

    Single linkage, because "the same beach" is a chain rather than a ball --
    a row of sampling points along a shoreline is one beach even when its ends
    are further apart than the radius. At 150 m that stays honest: the largest
    cluster spans 472 m. At 500 m it chains two thousand stations together and
    stops meaning anything.

    The label is the lexicographically smallest station_id in the cluster, so
    it is stable across runs and readable in a table.
    """
    if sites is None or sites.empty:
        return (pd.Series(dtype="object"), pd.Series(dtype="int64"))
    frame = sites.reset_index(drop=True)
    lat = pd.to_numeric(frame.get("lat"), errors="coerce").to_numpy()
    lon = pd.to_numeric(frame.get("lon"), errors="coerce").to_numpy()
    ids = frame["station_id"].astype(str).to_numpy()
    parent = list(range(len(frame)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(frame)):
        if not (np.isfinite(lat[i]) and np.isfinite(lon[i])):
            continue
        distance = _haversine_km(lat[i], lon[i], lat[i + 1:], lon[i + 1:])
        for offset in np.flatnonzero(distance <= radius_km):
            a, b = find(i), find(i + 1 + int(offset))
            if a != b:
                parent[a] = b

    groups = {}
    for index in range(len(frame)):
        groups.setdefault(find(index), []).append(index)
    labels = np.empty(len(frame), dtype=object)
    sizes = np.zeros(len(frame), dtype=int)
    for members in groups.values():
        name = min(ids[m] for m in members)
        for m in members:
            labels[m] = name
            sizes[m] = len(members)
    # A station with no coordinate is its own cluster: it cannot be shown to
    # share a beach with anything, and lumping the unlocatable together would
    # invent a beach that is not there.
    for index in range(len(frame)):
        if not (np.isfinite(lat[index]) and np.isfinite(lon[index])):
            labels[index] = ids[index]
            sizes[index] = 1
    return (pd.Series(labels, index=sites.index),
            pd.Series(sizes, index=sites.index))


# EPA's BEACH Act programme publishes its stations under location types that
# begin with this. It is the cleanest metadata signal in the whole table of
# what a station is FOR.
BEACH_ACT_PREFIX = "beach program site"


def programme_of(site_type):
    """beach_act | non_beach, from the location type alone.

    The Northeast's fecal coliform is not beach monitoring. Of 1,692 fitted
    FECAL pairs outside California, none comes from a BEACH Act station and
    1,473 match a shellfish growing-area pattern exactly: fecal-coliform-only,
    estuarine, numerically coded, run by a state bureau of marine water
    monitoring. NSSP classifies shellfish waters on fecal coliform; the BEACH
    Act posts bathing beaches on enterococcus. In this data every
    BEACH Program Site-* station measures enterococcus or E. coli and not one
    measures fecal coliform, so the two programmes separate cleanly on a
    column that is pure metadata.

    Deliberately NOT derived from the analyte mix, even though the analyte mix
    is what makes the pattern obvious. This module may not read a sample
    value -- that is the rule that keeps strata assignable before results
    exist -- and a label built from which analytes a station happens to report
    would be exactly that.

    The label is therefore what it says and no more: beach_act means the
    station is published under the BEACH Act programme, non_beach means it is
    not. It does NOT assert that a non_beach station is a shellfish growing
    area. Most of them are; the ones that are not would be scored against a
    shellfish standard they are not managed under, so the count of each is
    reported and the threshold entry says so.
    """
    text = str(site_type or "").strip().lower()
    if not text or text == "nan":
        return None
    return "beach_act" if text.startswith(BEACH_ACT_PREFIX) else "non_beach"


def load_overrides(path=None):
    """Hand-corrected values for anything except beach_type.

    beach_type deliberately does NOT come from here: it has its own file with
    an assigner and a date, because a label with no provenance is not usable
    evidence and this is the one stratum with no automatic fallback. The
    column most often wanted here is shore_normal_deg, for a beach where the
    coastline fit is known to be wrong.
    """
    path = path or config.STRATA_OVERRIDE_CSV
    if not os.path.exists(path):
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype=str, comment="#")
    if "station_id" not in frame.columns:
        raise SystemExit(f"{path} has no station_id column")
    if "beach_type" in frame.columns and frame["beach_type"].notna().any():
        raise SystemExit(
            f"{path} carries beach_type values. That stratum is assigned by "
            "hand from imagery and recorded with its assigner and date — put "
            "it in wq/beach_type_reviewed.csv via:\n"
            '    python -m wq.review --ingest <csv> --by "your name"')
    return frame.set_index(frame["station_id"].astype(str))


def assign(sites, datums=None, reviewed=None, spatial=None):
    """Every stratum for every site. Returns a copy with the columns added."""
    out = sites.copy()
    if out.empty:
        for column in config.STRATA:
            out[column] = pd.Series(dtype="object")
        return out

    out["region"] = [region_of(r.get("state"), r.get("lat"), r.get("lon"))
                     for _, r in out.iterrows()]
    out["water_class"] = out["region"].map(water_class)

    # beach_type: ONLY from the hand-reviewed file, with its provenance.
    labels = review.read_reviewed() if reviewed is None else reviewed
    keys = out["station_id"].astype(str)
    if labels is not None and len(labels):
        indexed = labels.set_index(labels["station_id"].astype(str))
        for column, target in (("beach_type", "beach_type"),
                               ("assigned_by", "beach_type_assigned_by"),
                               ("assigned_on", "beach_type_assigned_on")):
            if column in indexed.columns:
                out[target] = keys.map(indexed[column])
        print(f"  beach_type: {int(out['beach_type'].notna().sum())}/{len(out)} "
              "sites hand-assigned")
    else:
        out["beach_type"] = np.nan
        out["beach_type_assigned_by"] = np.nan
        out["beach_type_assigned_on"] = np.nan
        print("  beach_type: nobody has reviewed any site — the stratum will "
              "be dropped for coverage. Run: python -m wq.review --worklist")

    out["tidal_range_m"], out["datum_gauge"] = tidal_range(out, datums)

    out["programme"] = [programme_of(t) for t in out.get(
        "site_type", pd.Series([None] * len(out), index=out.index))]
    counts = out["programme"].value_counts(dropna=False).to_dict()
    print(f"  programme: {counts}")

    out["site_cluster"], out["site_cluster_n"] = site_clusters(out)
    clustered = int((out["site_cluster_n"] > 1).sum())
    n_clusters = int(out["site_cluster"].nunique())
    print(f"  site_cluster: {len(out)} stations are {n_clusters} distinct "
          f"beach(es) at {SITE_CLUSTER_RADIUS_KM * 1000:.0f} m; "
          f"{clustered} station(s) share one with a neighbour")

    if spatial is not None and not spatial.empty:
        keep = [c for c in spatial.columns if c != "station_id"
                and c not in out.columns]
        out = out.merge(spatial[["station_id"] + keep], on="station_id",
                        how="left")

    overrides = load_overrides()
    if len(overrides):
        applied = 0
        for column in list(config.STRATA) + config.SITE_COVARIATES + \
                ["shore_normal_deg"]:
            if column not in overrides.columns or column == "beach_type":
                continue
            hits = keys.map(overrides[column].dropna())
            if column in out.columns:
                out[column] = hits.where(hits.notna(), out[column])
            else:
                out[column] = hits
            applied += int(hits.notna().sum())
        if applied:
            print(f"  strata overrides applied to {applied} site-values")
    return out


def coverage(sites):
    """Populated share per stratum. This is what manifest.py decides on."""
    rows = []
    for column, spec in config.STRATA.items():
        if column not in sites.columns or sites.empty:
            share = 0.0
        else:
            share = float(sites[column].notna().mean())
        rows.append({"stratum": column, "coverage": share,
                     "required": spec["required_coverage"],
                     "automated": spec.get("automated", True),
                     "keeps": share >= spec["required_coverage"]})
    return pd.DataFrame(rows)
