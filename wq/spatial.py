"""Site covariates derived from spatial layers, by coordinate.

One function per layer, each cached per station so a re-run costs nothing and
an interrupted national pass resumes. Every fetcher follows the convention
this project already uses for an unverified endpoint: a --probe mode that
prints what actually came back and stops, and a failure that names the layer
rather than returning a column of NaN that reads as "no streams here".

    python -m wq.spatial --probe --lat 36.96 --lon -122.01
    python -m wq.spatial --site MonitoringLocationIdentifier
    python -m wq.spatial --all

The covariates are NOT the strata. beach_type is assigned by hand from
imagery (wq/review.py); everything here is continuous or countable, and
land_fraction_5km and embayment_ratio exist to SORT that review list, never
to assign the label.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

from . import config, geo, layers

CACHE_DIR = os.path.join(config.RAW_DIR, "spatial")

NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data"
NLDI_LOOKUPS = "https://api.water.usgs.gov/nldi/lookups"
WATERS_FLOWLINE = ("https://watersgeo.epa.gov/arcgis/rest/services/"
                   "NHDPlus_NP21/NHDSnapshot_NP21/MapServer/0/query")
ECHO_FACILITIES = ("https://echodata.epa.gov/echo/"
                   "cwa_rest_services.get_facilities")
ECHO_DOWNLOAD = "https://echodata.epa.gov/echo/cwa_rest_services.get_download"
# Overpass mirrors, tried in order. The main instance rate-limits and
# occasionally rejects a request the mirrors accept, and a coastline is the
# one layer with no substitute, so it is worth a second and third try.
OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
OVERPASS = OVERPASS_MIRRORS[0]

# NLDI characteristic ids. Matched by SUBSTRING against the catalogue the
# service publishes, because the year suffix moves with each NLCD release and
# hardcoding TOT_IMPV11 would silently pin the study to 2011 land cover.
IMPERVIOUS_PATTERN = "IMPV"
DEVELOPED_PATTERNS = ("NLCD", "DEV")
DEVELOPED_CLASSES = ("21", "22", "23", "24")
BASIN_AREA_ID = "TOT_BASIN_AREA"

COASTLINE_BBOX_KM = 30.0   # must exceed the fetch cap, or fetch is censored
FETCH_CAP_KM = 25.0
STREAM_SEARCH_KM = 2.0
OUTFALL_SEARCH_KM = 2.0
MOUTH_SNAP_M = 300.0       # how close a flowline's end must be to the shore


class LayerFailed(RuntimeError):
    """One layer did not answer for one site. Never fatal: the covariate goes
    missing, the coverage rule sees it, and the manifest records it."""


def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.json")


def _cached_json(name, builder):
    path = _cache_path(name)
    if os.path.exists(path):
        with open(path) as handle:
            return json.load(handle)
    payload = builder()
    with open(path, "w") as handle:
        json.dump(payload, handle)
    return payload


# Several of these services reject the default python-requests agent, and
# Overpass asks by name that scripts identify themselves.
USER_AGENT = ("coastal-wq/1.0 (research pipeline; "
              "https://github.com/rileysullivan1103-cell/Coastal)")


def _get(url, params=None, timeout=120, method="GET", data=None, probe=False,
         **_ignored):
    """One request, with the failure body attached to the error.

    A bare "HTTP 406" is not diagnosable -- it was 406 from Overpass that hid
    a query the server would have explained if anyone had read its body. The
    body is truncated into the LayerFailed message so a failing probe says
    what the service actually objected to.
    """
    import requests
    import scan_cameras as scan
    headers = {"User-Agent": USER_AGENT}
    try:
        if method == "POST":
            response = requests.post(url, data=data, timeout=timeout,
                                     headers=headers)
        else:
            response = scan.get_with_retry(url, params=params, headers=headers)
    except requests.RequestException as exc:
        raise LayerFailed(f"{url.split('/')[2]}: {type(exc).__name__}") from exc
    if probe:
        print(f"    {method} {response.url[:150]}")
        print(f"      HTTP {response.status_code}, "
              f"{len(response.content) / 1000:.1f} kB")
    if response.status_code != 200:
        detail = (response.text or "").strip().replace("\n", " ")[:300]
        raise LayerFailed(f"{url.split('/')[2]}: HTTP "
                          f"{response.status_code} — {detail}")
    return response


def _first_working(candidates, probe=False):
    """Try several URL shapes and return (response, url) for the one that
    answers, or raise with every failure listed.

    None of these endpoints could be checked against a live response when
    they were written, and a service that has moved its path answers 404 to
    the old one. Trying the documented shapes in order and RECORDING which
    one worked is the difference between a covariate that is missing and a
    covariate that is missing for a reason nobody wrote down.
    """
    failures = []
    for url, params in candidates:
        try:
            return _get(url, params=params, probe=probe), url
        except LayerFailed as exc:
            failures.append(str(exc))
            if probe:
                print(f"      no: {exc}")
    raise LayerFailed(" | ".join(failures))


# ---------------------------------------------------------------------------
# Coastline (OpenStreetMap via Overpass)
# ---------------------------------------------------------------------------

def fetch_coastline(lat, lon, km=COASTLINE_BBOX_KM, probe=False):
    """(lines, vintage). lines are [[(lat, lon), ...], ...] in OSM order.

    The way ORDER is load-bearing -- land is on the left of it -- so nothing
    here sorts, reverses or merges the ways it gets back.
    """
    dlat = km / 111.0
    dlon = km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
    bbox = f"{lat - dlat:.4f},{lon - dlon:.4f},{lat + dlat:.4f},{lon + dlon:.4f}"
    query = (f"[out:json][timeout:120];"
             f'way["natural"="coastline"]({bbox});'
             f"out geom;")
    if probe:
        print(f"  overpass query: {query}")

    # Two request shapes, because they fail differently: a mirror that
    # rejects the form-encoded POST often serves the same query as a GET
    # query-string, and the reverse happens on instances behind a proxy that
    # strips bodies. Trying both turns "HTTP 406" from a dead end into a
    # statement about which shape this instance wants.
    failures = []
    payload = None
    for mirror in OVERPASS_MIRRORS:
        for method, kwargs in (("POST", {"data": {"data": query}}),
                               ("GET", {"params": {"data": query}})):
            try:
                response = _get(mirror, method=method, probe=probe, **kwargs)
                payload = response.json()
                break
            except LayerFailed as exc:
                failures.append(f"{method} {exc}")
                if probe:
                    print(f"    {method} failed: {exc}")
            except ValueError as exc:
                failures.append(f"{method} {mirror.split('/')[2]}: "
                                f"not JSON ({exc})")
        if payload is not None:
            break
    if payload is None:
        raise LayerFailed(" | ".join(failures))

    if probe:
        print(f"  overpass returned {len(payload.get('elements', []))} ways")
        print(f"  osm3s: {payload.get('osm3s')}")
    vintage = (payload.get("osm3s") or {}).get("timestamp_osm_base")
    lines = []
    for element in payload.get("elements", []):
        points = [(p["lat"], p["lon"]) for p in element.get("geometry") or []]
        if len(points) >= 2:
            lines.append(points)
    return lines, vintage


def coastline_covariates(lat, lon, lines, vintage=None):
    """Every shape covariate, in one pass over the local linework."""
    out = {"coastline_vintage": vintage, "coastline_ways": len(lines)}
    if not lines:
        out["coastline_note"] = "no natural=coastline within the search box"
        return out
    local = geo.project_lines(lines, lat, lon)
    station = (0.0, 0.0)

    ok, detail = geo.coastline_sanity(local)
    out["coastline_sane"] = ok
    out["coastline_check"] = detail
    if not ok:
        # Everything below reads land and water off this linework. If the
        # linework disagrees with itself, the covariates would be confidently
        # inverted, which is worse than missing.
        out["coastline_note"] = detail
        return out

    found = geo.nearest_segment(station, local)
    out["dist_to_coastline_m"] = round(found[0], 1) if found else None

    normal, points = geo.shore_normal(local, station)
    out["shore_normal_deg"] = None if normal is None else round(normal, 1)
    out["shore_normal_fit_points"] = points
    out["shore_normal_source"] = "coastline_tangent" if normal is not None else None

    curvature, radius = geo.curvature_per_km(local, station)
    out["curvature_1_per_km"] = None if curvature is None else round(curvature, 4)
    out["curvature_radius_m"] = (None if radius in (None, float("inf"))
                                 else round(radius, 1))

    ratio = geo.embayment_ratio(local, station)
    out["embayment_ratio"] = None if ratio is None else round(ratio, 4)

    fraction = geo.land_fraction(local, station, radius_m=5000.0)
    out["land_fraction_5km"] = None if fraction is None else round(fraction, 4)

    distances, capped = geo.fetch_by_octant(local, station,
                                            max_km=FETCH_CAP_KM)
    for octant, value in distances.items():
        out[f"fetch_km_{octant}"] = round(value, 2)
        out[f"fetch_capped_{octant}"] = bool(capped[octant])
    if distances:
        out["fetch_km_min"] = round(min(distances.values()), 2)
        out["fetch_km_max"] = round(max(distances.values()), 2)
        out["fetch_km_mean"] = round(sum(distances.values()) / len(distances), 2)
        out["fetch_any_capped"] = bool(any(capped.values()))
    return out


# ---------------------------------------------------------------------------
# NHDPlus (streams) via NLDI
# ---------------------------------------------------------------------------

def fetch_comid(lat, lon, probe=False):
    """The NHDPlus flowline the station sits on.

    The comid is printed under --probe because everything downstream is keyed
    on it: a properties dict that spells it differently gives comid "None",
    and every characteristics URL built from that 404s while looking like the
    service is down.
    """
    response = _get(f"{NLDI_BASE}/comid/position",
                    params={"coords": f"POINT({lon} {lat})", "f": "json"},
                    probe=probe)
    payload = response.json()
    features = payload.get("features") or []
    if not features:
        raise LayerFailed("NLDI: no flowline at this coordinate")
    properties = features[0].get("properties") or {}
    if probe:
        print(f"  NLDI feature properties: {json.dumps(properties)[:300]}")
    comid = None
    for key in ("comid", "identifier", "nhdplus_comid", "featureid", "COMID"):
        value = properties.get(key)
        if value not in (None, "", "null"):
            comid = str(value).strip()
            break
    if comid is None or not comid.isdigit():
        raise LayerFailed(
            f"NLDI: no numeric comid in the feature properties "
            f"({sorted(properties)[:8]}) — the field moved; update fetch_comid")
    if probe:
        print(f"  NLDI comid: {comid}")
    return comid, features[0].get("geometry") or {}, properties


# Catchment-accumulated characteristics have lived at more than one path.
# Each shape is tried in order and the one that answers is recorded, rather
# than a single guess 404ing and taking every land-cover covariate with it.
NLDI_CHARACTERISTIC_PATHS = (
    "{base}/comid/{comid}/tot",
    "{base}/comid/{comid}/characteristics/tot",
    "https://labs.waterdata.usgs.gov/api/nldi/linked-data/comid/{comid}/tot",
)
NLDI_CATALOGUE_PATHS = (
    "{lookups}/tot/characteristics",
    "{base}/../lookups/tot/characteristics",
    "https://labs.waterdata.usgs.gov/api/nldi/lookups/tot/characteristics",
)
# Which path answered, so the manifest can record it and the next run can
# skip the ones that do not.
WORKING_PATHS = {}


def _characteristic_rows(payload):
    """NLDI has returned these under more than one key shape."""
    rows = (payload.get("characteristics")
            or payload.get("characteristic")
            or payload.get("characteristicMetadata") or [])
    if isinstance(rows, dict):
        rows = rows.get("characteristic") or list(rows.values())
    return rows if isinstance(rows, list) else []


def fetch_characteristics(comid, probe=False):
    candidates = [(path.format(base=NLDI_BASE, comid=comid), {"f": "json"})
                  for path in NLDI_CHARACTERISTIC_PATHS]
    response, url = _first_working(candidates, probe=probe)
    WORKING_PATHS["nldi_characteristics"] = url
    payload = response.json()
    rows = _characteristic_rows(payload)
    if probe:
        print(f"  NLDI characteristics from {url}: {len(rows)} rows")
        for row in rows[:20]:
            print(f"    {str(row.get('characteristic_id')):<24} "
                  f"{row.get('characteristic_value')}")
        if not rows:
            print(f"    payload keys: {list(payload)[:10]}")
    return {str(r.get("characteristic_id")): r.get("characteristic_value")
            for r in rows if r.get("characteristic_id")}


def characteristic_catalogue(probe=False):
    """The published list, with each id's description -- which is where the
    NLCD year is written. Cached, because it is the same for every site and
    it is what the manifest records as the land-cover vintage."""
    def build():
        candidates = [(path.format(lookups=NLDI_LOOKUPS, base=NLDI_BASE),
                       {"f": "json"})
                      for path in NLDI_CATALOGUE_PATHS
                      if "{base}/.." not in path]
        response, url = _first_working(candidates, probe=probe)
        WORKING_PATHS["nldi_catalogue"] = url
        return response.json()

    payload = _cached_json("nldi_catalogue", build)
    rows = (payload.get("characteristicMetadata")
            or payload.get("characteristics") or [])
    catalogue = {}
    for row in rows:
        entry = row.get("characteristic") if "characteristic" in row else row
        identifier = str(entry.get("characteristic_id") or entry.get("id"))
        catalogue[identifier] = {
            "description": entry.get("characteristic_description")
            or entry.get("description"),
            "units": entry.get("units"),
        }
    if probe:
        for identifier, meta in list(catalogue.items())[:30]:
            print(f"    {identifier:<24} {str(meta['description'])[:70]}")
    return catalogue


def pick_landcover_ids(catalogue):
    """(impervious_id, developed_ids, vintage_note) chosen from the catalogue.

    Substring-matched rather than hardcoded so the study records which NLCD
    release it actually used instead of pinning itself to whichever year was
    current when this was written.
    """
    impervious = sorted(i for i in catalogue
                        if IMPERVIOUS_PATTERN in i and i.startswith("TOT"))
    developed = sorted(
        i for i in catalogue
        if i.startswith("TOT") and any(p in i for p in DEVELOPED_PATTERNS)
        and any(i.endswith(c) for c in DEVELOPED_CLASSES))
    best = impervious[-1] if impervious else None
    note = None
    if best:
        note = str(catalogue.get(best, {}).get("description") or best)
    return best, developed, note


def stream_covariates(lat, lon, lines, comid=None, characteristics=None,
                      catalogue=None, flowlines=None):
    """dist_to_stream_m, stream_order, upstream_area_km2, n_streams_within_2km.

    A stream MOUTH, not a stream: a flowline that passes 300 m from the beach
    on its way somewhere else is not a freshwater input to it, so only
    flowline endpoints that land within MOUTH_SNAP_M of the coastline count.
    """
    out = {"comid": comid}
    characteristics = characteristics or {}
    catalogue = catalogue or {}

    area = characteristics.get(BASIN_AREA_ID)
    out["upstream_area_km2"] = (float(area) if area not in (None, "")
                                else None)
    out["upstream_area_source"] = "nldi" if out["upstream_area_km2"] else None

    impervious_id, developed_ids, vintage = pick_landcover_ids(catalogue)
    out["landcover_vintage"] = vintage
    if impervious_id and characteristics.get(impervious_id) not in (None, ""):
        value = float(characteristics[impervious_id])
        # NLDI publishes imperviousness as a percentage.
        out["impervious_frac"] = value / 100.0 if value > 1 else value
        out["impervious_id"] = impervious_id
    developed = [float(characteristics[i]) for i in developed_ids
                 if characteristics.get(i) not in (None, "")]
    if developed:
        total = sum(developed)
        out["developed_frac"] = total / 100.0 if total > 1 else total
        out["developed_ids"] = ",".join(developed_ids)

    if flowlines:
        local_lines = geo.project_lines(lines, lat, lon) if lines else []
        mouths, nearest, order = [], None, None
        for feature in flowlines:
            points = _feature_points(feature)
            if len(points) < 2:
                continue
            local = [geo.to_local(a, b, lat, lon) for a, b in points]
            distance = min(math.hypot(x, y) for x, y in local)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, feature)
            end = local[-1]
            to_shore = (geo.nearest_segment(end, local_lines)[0]
                        if local_lines else None)
            if math.hypot(*end) <= STREAM_SEARCH_KM * 1000 and (
                    to_shore is None or to_shore <= MOUTH_SNAP_M):
                mouths.append(feature)
        out["n_streams_within_2km"] = len(mouths)
        if nearest:
            out["dist_to_stream_m"] = round(nearest[0], 1)
            order = _feature_property(nearest[1], ("streamorde", "StreamOrde",
                                                   "streamorder", "StreamOrder"))
            out["stream_order"] = None if order is None else int(float(order))
    return out


def _feature_points(feature):
    geometry = feature.get("geometry") or {}
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if kind == "LineString":
        return [(c[1], c[0]) for c in coordinates if len(c) >= 2]
    if kind == "MultiLineString":
        return [(c[1], c[0]) for part in coordinates for c in part
                if len(c) >= 2]
    if "paths" in geometry:  # ArcGIS
        return [(c[1], c[0]) for path in geometry["paths"] for c in path
                if len(c) >= 2]
    return []


def _feature_property(feature, names):
    properties = feature.get("properties") or feature.get("attributes") or {}
    lowered = {str(k).lower(): v for k, v in properties.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def fetch_flowlines(lat, lon, km=STREAM_SEARCH_KM, probe=False):
    """Flowlines near the station, with their stream order where served."""
    dlat = km / 111.0
    dlon = km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
    params = {
        "geometry": f"{lon - dlon:.5f},{lat - dlat:.5f},"
                    f"{lon + dlon:.5f},{lat + dlat:.5f}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326", "outSR": "4326", "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*", "returnGeometry": "true", "f": "geojson",
    }
    response = _get(WATERS_FLOWLINE, params=params)
    payload = response.json()
    features = payload.get("features") or []
    if probe:
        print(f"  WATERS flowlines: {len(features)}")
        if features:
            properties = (features[0].get("properties")
                          or features[0].get("attributes") or {})
            print(f"  fields: {sorted(properties)[:25]}")
            print("  stream order field present: "
                  f"{_feature_property(features[0], ('streamorde',)) is not None}")
    return features


# ---------------------------------------------------------------------------
# EPA ECHO (permitted discharges)
# ---------------------------------------------------------------------------

# ECHO's download returns a DEFAULT column set unless you ask for columns by
# name, and that default carries FacLong without FacLat -- so a distance
# calculation over it silently finds nothing, everywhere, forever. These are
# the columns this module actually reads.
ECHO_COLUMNS = ("SourceID", "CWPName", "FacLat", "FacLong",
                "CWPMajorMinorStatusFlag", "CWPFacilityTypeIndicator",
                "CWPPermitStatusDesc", "CWPTotalDesignFlowNmbr")


def fetch_outfalls(lat, lon, km=OUTFALL_SEARCH_KM, probe=False):
    dlat = km / 111.0
    dlon = km / (111.0 * max(math.cos(math.radians(lat)), 0.01))
    params = {
        "output": "JSON", "responseset": "5000",
        "p_c1lon": f"{lon - dlon:.5f}", "p_c1lat": f"{lat + dlat:.5f}",
        "p_c2lon": f"{lon + dlon:.5f}", "p_c2lat": f"{lat - dlat:.5f}",
    }
    payload = _get(ECHO_FACILITIES, params=params, probe=probe).json()
    results = payload.get("Results") or {}
    qid = results.get("QueryID")
    rows = results.get("QueryRows")
    if probe:
        print(f"  ECHO QueryID {qid}, rows {rows}")
        error = results.get("Error") or payload.get("Error")
        if error:
            print(f"  ECHO error: {error}")
    if not qid:
        return []
    try:
        if int(str(rows).strip() or 0) == 0:
            if probe:
                print("  ECHO: no permitted facilities in this box — a real "
                      "answer, not a failure")
            return []
    except (TypeError, ValueError):
        pass

    from io import StringIO
    text = _get(ECHO_DOWNLOAD,
                params={"qid": qid, "output": "CSV",
                        "qcolumns": ",".join(ECHO_COLUMNS)},
                probe=probe).text
    frame = pd.read_csv(StringIO(text), low_memory=False)
    if probe:
        print(f"  ECHO download columns: {list(frame.columns)}")
        missing = [c for c in ("FacLat", "FacLong") if c not in frame.columns]
        if missing:
            print(f"  ECHO: {missing} absent even with qcolumns — without a "
                  "latitude nothing here can be placed, so the outfall "
                  "covariates will stay empty and the coverage rule will "
                  "drop them")
    return frame.to_dict("records")


def outfall_covariates(lat, lon, records):
    """dist_to_outfall_m, outfall_type, n_outfalls_within_2km.

    See layers.LAYERS['echo']['caveat']: this is the permitted FACILITY, not
    the end of its pipe.
    """
    out = {"n_outfalls_within_2km": 0}
    if records is None:
        return out
    rows = []
    for record in records:
        latitude = _first(record, ("FacLat", "CWPLatitude", "Latitude"))
        longitude = _first(record, ("FacLong", "CWPLongitude", "Longitude"))
        if latitude is None or longitude is None:
            continue
        try:
            distance = geo.haversine_m(lat, lon, float(latitude),
                                       float(longitude))
        except (TypeError, ValueError):
            continue
        rows.append((distance, record))
    rows.sort(key=lambda r: r[0])
    out["n_outfalls_within_2km"] = sum(
        1 for distance, _ in rows if distance <= OUTFALL_SEARCH_KM * 1000)
    if rows:
        distance, record = rows[0]
        out["dist_to_outfall_m"] = round(distance, 1)
        major = _first(record, ("CWPMajorMinorStatusFlag", "MajorMinorStatus"))
        kind = _first(record, ("CWPFacilityTypeIndicator", "FacilityTypeIndicator"))
        pieces = [p for p in (_expand_major(major), str(kind).strip()
                              if kind else None) if p]
        out["outfall_type"] = "/".join(pieces) if pieces else None
        out["outfall_permit"] = _first(record, ("SourceID", "CWPName"))
    return out


def _expand_major(flag):
    text = str(flag or "").strip().upper()
    return {"M": "major", "N": "minor", "U": "unknown"}.get(text) or None


def _first(record, names):
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


# ---------------------------------------------------------------------------
# Per-site assembly
# ---------------------------------------------------------------------------

def for_site(site, record=None, probe=False, want=None):
    """Every spatial covariate for one station. Never raises for one layer."""
    lat, lon = float(site["lat"]), float(site["lon"])
    station = str(site["station_id"])
    record = record if record is not None else layers.blank_record()
    want = want or set(layers.LAYERS)
    out = {"station_id": station}

    if "coastline" in want:
        record[ "coastline"]["sites_attempted"] += 1
        try:
            def build():
                lines, vintage = fetch_coastline(lat, lon, probe=probe)
                return {"lines": lines, "vintage": vintage}

            payload = _cached_json(f"coastline_{_slug(station)}", build)
            values = coastline_covariates(lat, lon, payload["lines"],
                                          payload.get("vintage"))
            out.update(values)
            layers.record_access(record, "coastline", payload.get("vintage"))
            if values.get("shore_normal_deg") is not None:
                record["coastline"]["sites_populated"] += 1
        except LayerFailed as exc:
            out["coastline_note"] = str(exc)
            layers.record_access(record, "coastline", note=str(exc))

    if "nhdplus" in want:
        record["nhdplus"]["sites_attempted"] += 1
        try:
            def build():
                comid, _geometry, _properties = fetch_comid(lat, lon, probe=probe)
                return {"comid": comid,
                        "characteristics": fetch_characteristics(comid,
                                                                 probe=probe),
                        "flowlines": _safe(fetch_flowlines, lat, lon,
                                           probe=probe)}

            payload = _cached_json(f"nhd_{_slug(station)}", build)
            lines = []
            cached = _cache_path(f"coastline_{_slug(station)}")
            if os.path.exists(cached):
                with open(cached) as handle:
                    lines = json.load(handle).get("lines") or []
            values = stream_covariates(
                lat, lon, lines, payload.get("comid"),
                payload.get("characteristics"), characteristic_catalogue(),
                payload.get("flowlines"))
            out.update(values)
            layers.record_access(record, "nhdplus")
            layers.record_access(record, "nlcd",
                                 values.get("landcover_vintage"))
            if values.get("dist_to_stream_m") is not None:
                record["nhdplus"]["sites_populated"] += 1
            record["nlcd"]["sites_attempted"] += 1
            if values.get("impervious_frac") is not None:
                record["nlcd"]["sites_populated"] += 1
        except LayerFailed as exc:
            out["nhdplus_note"] = str(exc)
            layers.record_access(record, "nhdplus", note=str(exc))

    if "echo" in want:
        record["echo"]["sites_attempted"] += 1
        try:
            payload = _cached_json(
                f"echo_{_slug(station)}",
                lambda: {"records": fetch_outfalls(lat, lon, probe=probe)})
            values = outfall_covariates(lat, lon, payload.get("records"))
            out.update(values)
            layers.record_access(record, "echo")
            if values.get("dist_to_outfall_m") is not None:
                record["echo"]["sites_populated"] += 1
        except LayerFailed as exc:
            out["echo_note"] = str(exc)
            layers.record_access(record, "echo", note=str(exc))
    return out


def _safe(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except LayerFailed as exc:
        print(f"      {exc}")
        return []


def _slug(text):
    return "".join(c if c.isalnum() else "_" for c in str(text))[:64]


def fill_drainage_from_wqp(frame, sites):
    """Use WQP's own drainage area wherever NLDI did not answer.

    Same quantity, different source, and the source is recorded per site --
    a covariate assembled from two layers without saying which is which is
    exactly what the vintage logging exists to prevent.
    """
    if "wqp_drainage_area_km2" not in sites.columns:
        return frame
    lookup = dict(zip(sites["station_id"].astype(str),
                      pd.to_numeric(sites["wqp_drainage_area_km2"],
                                    errors="coerce")))
    out = frame.copy()
    if "upstream_area_km2" not in out.columns:
        out["upstream_area_km2"] = np.nan
        out["upstream_area_source"] = None
    filled = 0
    for index, row in out.iterrows():
        if pd.notna(row.get("upstream_area_km2")):
            continue
        value = lookup.get(str(row["station_id"]))
        if value is not None and pd.notna(value):
            out.at[index, "upstream_area_km2"] = float(value)
            out.at[index, "upstream_area_source"] = "wqp_station_record"
            filled += 1
    if filled:
        print(f"  upstream_area_km2: {filled} site(s) filled from the WQP "
              "station record where NLDI did not answer")
    return out


def build(sites, record=None, progress=True, want=None):
    """Spatial covariates for every site. Returns (frame, layer_record)."""
    record = record if record is not None else layers.blank_record()
    rows = []
    total = len(sites)
    started = time.time()
    for index, (_, site) in enumerate(sites.iterrows(), 1):
        if pd.isna(site.get("lat")) or pd.isna(site.get("lon")):
            continue
        rows.append(for_site(site, record, want=want))
        if progress and (index % 10 == 0 or index == total):
            done = time.time() - started
            left = (total - index) * done / max(index, 1)
            print(f"  [{index}/{total}] spatial covariates  "
                  f"~{left / 60:.0f} min left")
    frame = fill_drainage_from_wqp(pd.DataFrame(rows), sites)
    return frame, record


def add_tidal_datums(frame, sites, datums):
    """tidal_range_m and datum_gauge_dist_km, from the CO-OPS gauge list."""
    from .strata import tidal_range, _haversine_km
    values, gauges = tidal_range(sites, datums)
    out = frame.copy()
    keyed = dict(zip(sites["station_id"].astype(str), values))
    gauge_of = dict(zip(sites["station_id"].astype(str), gauges))
    out["tidal_range_m"] = out["station_id"].astype(str).map(keyed)
    out["datum_gauge"] = out["station_id"].astype(str).map(gauge_of)
    if datums is not None and not datums.empty:
        coords = datums.set_index(datums["station_id"].astype(str))
        distances = []
        for _, row in out.iterrows():
            gauge = row.get("datum_gauge")
            site = sites[sites["station_id"].astype(str)
                         == str(row["station_id"])]
            if not gauge or site.empty or str(gauge) not in coords.index:
                distances.append(np.nan)
                continue
            point = coords.loc[str(gauge)]
            distances.append(float(_haversine_km(
                float(site.iloc[0]["lat"]), float(site.iloc[0]["lon"]),
                np.array([float(point["lat"])]),
                np.array([float(point["lon"])]))[0]))
        out["datum_gauge_dist_km"] = distances
    else:
        out["datum_gauge_dist_km"] = np.nan
    return out


def coverage(frame, covariates=None):
    """How many stations each covariate could be populated for. This is what
    the ~70% rule in wq/manifest.py decides on."""
    covariates = covariates or config.SITE_COVARIATES
    rows = []
    for name in covariates:
        if name not in frame.columns or frame.empty:
            share, populated = 0.0, 0
        else:
            populated = int(frame[name].notna().sum())
            share = populated / len(frame)
        rows.append({"covariate": name, "layer": layers.COVARIATE_LAYER.get(name),
                     "populated": populated, "of": len(frame),
                     "coverage": round(share, 4),
                     "keeps": share >= config.COVARIATE_MIN_COVERAGE})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--site")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="print what each layer actually returns, then stop")
    args = parser.parse_args()

    if args.lat is not None and args.lon is not None:
        site = pd.Series({"station_id": args.site or "probe",
                          "lat": args.lat, "lon": args.lon})
        values = for_site(site, probe=args.probe)
        print(json.dumps(values, indent=2, default=str))
        return
    sites_path = os.path.join(config.DATA_DIR, "stations.csv")
    if not os.path.exists(sites_path):
        sys.exit(f"{sites_path} missing — run wq.run_wq --stations first")
    sites = pd.read_csv(sites_path, low_memory=False)
    if args.site:
        sites = sites[sites["station_id"].astype(str) == args.site]
        if sites.empty:
            sys.exit(f"no station {args.site}")
    elif not args.all:
        parser.error("give --site, --all, or --lat/--lon")
    frame, record = build(sites)
    print(coverage(frame).to_string(index=False))
    print(json.dumps(layers.manifest_section(record), indent=2)[:1200])


if __name__ == "__main__":
    main()
