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
