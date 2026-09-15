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

# Coastline and permitted-discharge linework is SHARED between neighbouring
# stations, and fetching it per station is the single thing that made a
# national run take thirty hours. Both are fetched per TILE instead: one
# query covering the tile plus a margin, cached, and reused by every station
# inside it. Beaches cluster hard along the coast, so a quarter-degree tile
# routinely serves dozens of stations, and the request count falls by roughly
# two orders of magnitude.
TILE_DEGREES = 0.25
COASTLINE_BBOX_KM = 30.0
# The furthest any coastline covariate reaches from the station: land_fraction
# samples to 5 km, so a defect beyond this cannot enter any of them. The extra
# kilometre is slack, not a second opinion.
SANITY_VETO_M = 6000.0   # must exceed the fetch cap, or fetch is censored
FETCH_CAP_KM = 25.0
STREAM_SEARCH_KM = 2.0
OUTFALL_SEARCH_KM = 2.0
MOUTH_SNAP_M = 300.0       # how close a flowline's end must be to the shore


# Minimum seconds between requests to one host, and how long to wait when a
# service says it has had enough. Overpass in particular is a free community
# service whose usage policy asks for light, non-bulk use; an earlier version
# of this module sent one query per station, which at 32,513 coastal stations
# is neither light nor non-bulk. Tiling (below) cut the count by two orders of
# magnitude and this throttle keeps the remainder polite.
HOST_MIN_INTERVAL = {
    "overpass-api.de": 3.0,
    "overpass.kumi.systems": 2.0,
    "overpass.private.coffee": 2.0,
    "api.water.usgs.gov": 0.5,
    "echodata.epa.gov": 0.5,
    "watersgeo.epa.gov": 0.5,
}
DEFAULT_MIN_INTERVAL = 0.25
RATE_LIMIT_BACKOFF = (15, 60, 180)
_LAST_CALL = {}

# After this many consecutive failures, a host is given up on for the rest of
# the run. Without it, a service that is refusing everything costs its full
# backoff ladder ONCE PER SITE: ECHO rate-limiting RI's 7 tiles spent four
# minutes each and then spent them again for every station in the tile,
# because a failed fetch is not cached and the next station retried it. The
# covariates that host supplies then come out empty, the coverage rule drops
# them, and the manifest records that the layer was unavailable -- which is
# the honest outcome and takes seconds rather than hours.
CIRCUIT_THRESHOLD = 3
# A host that has NEVER answered this run gets one short retry, not the full
# ladder. 429 on a first contact is a refusal, not a busy moment: ECHO said so
# on its first request and the full 15/60/180 ladder then spent four minutes
# per tile finding that out again. A host that HAS answered gets the patient
# ladder, because there the 429 really is back-pressure worth waiting out.
COLD_BACKOFF = (15,)
_FAILURES = {}
_SUCCESSES = {}
_TRIPPED = set()


def _circuit_check(host):
    if host in _TRIPPED:
        raise LayerFailed(
            f"{host}: skipped — {CIRCUIT_THRESHOLD} consecutive failures "
            "earlier in this run, so this layer is given up on rather than "
            "retried at every remaining site")


def _circuit_record(host, ok):
    """Counts REFUSALS, not calls.

    An earlier version recorded one failure per completed call, so a host
    refusing everything still climbed its whole backoff ladder three times --
    thirteen minutes to conclude what its first response had already said.
    Every refusing response counts, and the ladder is abandoned the moment
    the circuit trips.
    """
    if ok:
        _SUCCESSES[host] = _SUCCESSES.get(host, 0) + 1
        _FAILURES[host] = 0
        return
    _FAILURES[host] = _FAILURES.get(host, 0) + 1
    if _FAILURES[host] >= CIRCUIT_THRESHOLD and host not in _TRIPPED:
        _TRIPPED.add(host)
        served = _SUCCESSES.get(host, 0)
        print(f"      GIVING UP on {host} after {_FAILURES[host]} consecutive "
              f"refusals ({served} successful response(s) this run). Its "
              "covariates will be empty and the coverage rule will drop them.")


def _ladder(host):
    return RATE_LIMIT_BACKOFF if _SUCCESSES.get(host) else COLD_BACKOFF


def circuit_report():
    """Hosts abandoned this run, for the layer record and the manifest."""
    return sorted(_TRIPPED)


def _throttle(host):
    wait = HOST_MIN_INTERVAL.get(host, DEFAULT_MIN_INTERVAL)
    last = _LAST_CALL.get(host)
    if last is not None:
        remaining = wait - (time.time() - last)
        if remaining > 0:
            time.sleep(remaining)
    _LAST_CALL[host] = time.time()


class LayerFailed(RuntimeError):
    """One layer did not answer for one site. Never fatal: the covariate goes
    missing, the coverage rule sees it, and the manifest records it."""


def tile_of(lat, lon, size=TILE_DEGREES):
    """The tile a coordinate falls in, as (south, west) of its corner."""
    return (math.floor(float(lat) / size) * size,
            math.floor(float(lon) / size) * size)


def tile_bounds(lat, lon, margin_km, size=TILE_DEGREES):
    """The tile's box, widened by margin_km so a station near an edge still
    sees the coastline on the other side of it."""
    south, west = tile_of(lat, lon, size)
    dlat = margin_km / 111.0
    mid_lat = south + size / 2
    dlon = margin_km / (111.0 * max(math.cos(math.radians(mid_lat)), 0.01))
    return (south - dlat, west - dlon, south + size + dlat, west + size + dlon)


def _tile_slug(lat, lon, size=TILE_DEGREES):
    south, west = tile_of(lat, lon, size)
    return f"{south:+07.2f}_{west:+08.2f}".replace(".", "p")


def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{name}.json")


# Failures within this run, keyed like the cache. A fetch that failed is not
# written to disk -- a re-run should retry it -- but it must not be retried
# for every remaining station in the same tile, which is what turned one
# rate-limited tile into 120 identical failures.
_FAILED_THIS_RUN = {}


def _cached_json(name, builder, expects=()):
    """Read the cached payload, or build and cache it.

    `expects` names the keys this caller is about to read. A cache file
    written before those keys existed is not a valid answer to today's
    question -- it is a payload from an older schema -- so it is rebuilt
    rather than served. Without this, adding a fetch to a builder is silent:
    the new covariates read `None` from the old payload, nothing raises, and
    the coverage table shows an empty layer with no reason recorded.
    """
    path = _cache_path(name)
    if os.path.exists(path):
        with open(path) as handle:
            payload = json.load(handle)
        missing = [key for key in expects
                   if not isinstance(payload, dict) or key not in payload]
        if not missing:
            return payload
        print(f"      {name}: cached payload predates "
              f"{', '.join(missing)}; refetching")
        os.remove(path)
    if name in _FAILED_THIS_RUN:
        raise LayerFailed(f"{_FAILED_THIS_RUN[name]} (already failed for this "
                          "tile in this run; not retried per station)")
    try:
        payload = builder()
    except LayerFailed as exc:
        _FAILED_THIS_RUN[name] = str(exc)
        raise
    with open(path, "w") as handle:
        json.dump(payload, handle)
    return payload


# Several of these services reject the default python-requests agent, and
# Overpass asks by name that scripts identify themselves.
USER_AGENT = ("coastal-wq/1.0 (research pipeline; "
              "https://github.com/rileysullivan1103-cell/Coastal)")


# Statuses that mean the SERVICE is refusing, as opposed to answering "there
# is nothing here". Only these count toward the circuit breaker.
REFUSAL_STATUSES = (429, 500, 502, 503, 504)


def _get(url, params=None, timeout=120, method="GET", data=None, probe=False,
         count_failures=True, **_ignored):
    """One request, with the failure body attached to the error.

    A bare "HTTP 406" is not diagnosable -- it was 406 from Overpass that hid
    a query the server would have explained if anyone had read its body. The
    body is truncated into the LayerFailed message so a failing probe says
    what the service actually objected to.
    """
    import requests
    host = url.split("/")[2]
    headers = {"User-Agent": USER_AGENT}
    _circuit_check(host)

    # ONE retry ladder, here. An earlier version wrapped this loop around
    # scan_cameras.get_with_retry, which has its own 2/4/8-second ladder for
    # the same status codes -- so every outer attempt spent fourteen seconds
    # retrying before the outer backoff even started, and a refusing service
    # cost four minutes per attempt to learn nothing. Nested retry ladders
    # multiply; they do not compose.
    response = None
    ladder = _ladder(host)
    for attempt, pause in enumerate((0,) + ladder):
        _circuit_check(host)
        if pause:
            print(f"      {host} rate-limited; waiting {pause}s "
                  f"(attempt {attempt}/{len(ladder)})")
            time.sleep(pause)
        _throttle(host)
        try:
            if method == "POST":
                response = requests.post(url, data=data, timeout=timeout,
                                         headers=headers)
            else:
                response = requests.get(url, params=params, timeout=timeout,
                                        headers=headers)
        except requests.RequestException as exc:
            if count_failures:
                _circuit_record(host, ok=False)
            raise LayerFailed(f"{host}: {type(exc).__name__}") from exc
        if response.status_code not in REFUSAL_STATUSES:
            break
        if count_failures:
            _circuit_record(host, ok=False)
        stated = response.headers.get("Retry-After")
        if stated and str(stated).isdigit():
            wait = min(int(stated), 300)
            print(f"      {host} asked for {wait}s")
            time.sleep(wait)
    # A 404 is the service ANSWERING: there is no flowline at this
    # coordinate. Plenty of open-coast beaches have none, and three such
    # stations in a row were enough to make the breaker abandon NLDI for a
    # whole run after it had already answered eleven times. Only a refusal
    # counts against a host; "nothing here" is data.
    if count_failures and response is not None:
        if response.status_code == 200:
            _circuit_record(host, ok=True)
        elif response.status_code in REFUSAL_STATUSES:
            _circuit_record(host, ok=False)
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
            # A candidate that does not exist is the point of the list, not a
            # failure of the host, so probing never trips the breaker.
            return _get(url, params=params, probe=probe,
                        count_failures=False), url
        except LayerFailed as exc:
            failures.append(str(exc))
            if probe:
                print(f"      no: {exc}")
    raise LayerFailed(" | ".join(failures))


# ---------------------------------------------------------------------------
# Coastline (OpenStreetMap via Overpass)
# ---------------------------------------------------------------------------

def fetch_coastline(lat, lon, km=COASTLINE_BBOX_KM, probe=False):
    """(lines, vintage) for the TILE this coordinate falls in.

    Fetched per tile rather than per station, and cached by tile, so a
    stretch of coast with forty monitoring stations on it costs one query
    rather than forty. The way ORDER is load-bearing -- land is on the left
    of it -- so nothing here sorts, reverses or merges what comes back.
    """
    south, west, north, east = tile_bounds(lat, lon, km)
    bbox = f"{south:.4f},{west:.4f},{north:.4f},{east:.4f}"
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
    out["coastline_segments"] = sum(len(line) - 1 for line in local)

    ok, detail = geo.coastline_sanity(local)
    out["coastline_sane"] = ok
    out["coastline_check"] = detail
    if not ok:
        # Everything below reads land and water off this linework, so where
        # the linework disagrees with itself the covariates would be
        # confidently inverted -- worse than missing.
        #
        # But the disagreement is LOCAL to the ways involved, and the box is
        # 30 km across. Voiding every station in a tile because of one bad
        # edit at its far corner discards good geometry to punish geometry
        # nobody read. So the veto is scoped: if a suspect way is close enough
        # to be read by any covariate below, refuse; otherwise proceed and
        # keep saying, in coastline_check, that the box has a defect in it.
        suspect = geo.way_junctions(local)[2]
        distance = None
        if suspect:
            found = geo.nearest_segment(station, [local[i] for i in suspect])
            distance = None if found is None else found[0]
        if distance is None or distance <= SANITY_VETO_M:
            out["coastline_note"] = detail
            return out
        out["coastline_defect_km"] = round(distance / 1000.0, 2)
        out["coastline_note"] = (
            f"{detail} — but the nearest such way is "
            f"{distance / 1000.0:.1f} km away, beyond everything read here, "
            "so the covariates are kept")

    # Everything below WALKS along the shore, and OSM splits a shoreline into
    # many short ways -- so the walk has to follow the chain across them.
    # Stitching happens after the direction check, which reads the ways as
    # they came, and joins only the junctions that check agrees are sound.
    local = geo.stitch_ways(local)
    out["coastline_chains"] = len(local)
    # Built once and reused by every query below. Without it the ~300
    # land/water samples each scan every segment in a 30 km box, and on an
    # estuary shore that does not finish in any useful time.
    index = geo.SegmentIndex(local)

    found = geo.nearest_segment(station, local, index)
    out["dist_to_coastline_m"] = round(found[0], 1) if found else None

    normal, points = geo.shore_normal(local, station, index=index)
    out["shore_normal_deg"] = None if normal is None else round(normal, 1)
    out["shore_normal_fit_points"] = points
    out["shore_normal_source"] = "coastline_tangent" if normal is not None else None

    curvature, radius = geo.curvature_per_km(local, station, index=index)
    out["curvature_1_per_km"] = None if curvature is None else round(curvature, 4)
    out["curvature_radius_m"] = (None if radius in (None, float("inf"))
                                 else round(radius, 1))

    ratio = geo.embayment_ratio(local, station, index=index)
    out["embayment_ratio"] = None if ratio is None else round(ratio, 4)

    fraction = geo.land_fraction(local, station, radius_m=5000.0,
                                 index=index)
    out["land_fraction_5km"] = None if fraction is None else round(fraction, 4)

    distances, capped = geo.fetch_by_octant(local, station,
                                            max_km=FETCH_CAP_KM, index=index)
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
    """Permitted facilities in this coordinate's TILE, plus a margin.

    Per tile for the same reason as the coastline: the answer is shared, and
    the per-station distances are computed locally from it afterwards.
    """
    south, west, north, east = tile_bounds(lat, lon, km)
    params = {
        "output": "JSON", "responseset": "5000",
        "p_c1lon": f"{west:.5f}", "p_c1lat": f"{north:.5f}",
        "p_c2lon": f"{east:.5f}", "p_c2lat": f"{south:.5f}",
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
    # The tile is carried through because the tiled layers succeed or fail per
    # TILE, not per station. Without it a hundred identical failure lines look
    # like a hundred failures instead of the four they actually are.
    out = {"station_id": station, "tile": _tile_slug(lat, lon)}
    # A layer you chose not to fetch is not a layer that failed, and it is not
    # a layer that came back empty for no reason either. Say so on the row.
    for key in layers.FETCHED:
        if key not in want:
            out[f"{key}_note"] = ("not fetched — excluded by --skip-layers "
                                  "on this run")

    if "coastline" in want:
        record[ "coastline"]["sites_attempted"] += 1
        try:
            def build():
                lines, vintage = fetch_coastline(lat, lon, probe=probe)
                return {"lines": lines, "vintage": vintage}

            payload = _cached_json(f"coastline_{_tile_slug(lat, lon)}",
                                   build, expects=("lines", "vintage"))
            values = coastline_covariates(lat, lon, payload["lines"],
                                          payload.get("vintage"))
            out.update(values)
            layers.record_access(record, "coastline", payload.get("vintage"),
                                 note=values.get("coastline_note"))
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
                # Three different services, and they fail independently. The
                # flowlines carry dist_to_stream_m, stream_order and
                # n_streams_within_2km; StreamCat carries the land cover.
                # Letting either raise here took the other's covariates down
                # with it, which is how one dead endpoint cost five.
                flowlines, flowline_note = [], None
                try:
                    flowlines = fetch_flowlines(lat, lon, probe=probe)
                except LayerFailed as exc:
                    flowline_note = str(exc)
                landcover, landcover_note = {}, None
                try:
                    landcover = fetch_streamcat(comid, probe=probe)
                except LayerFailed as exc:
                    landcover_note = str(exc)
                return {"comid": comid,
                        "flowlines": flowlines,
                        "flowline_note": flowline_note,
                        "streamcat": landcover,
                        "streamcat_note": landcover_note}

            payload = _cached_json(
                f"nhd_{_slug(station)}", build,
                expects=("comid", "flowlines", "flowline_note",
                         "streamcat", "streamcat_note"))
            lines = []
            cached = _cache_path(f"coastline_{_tile_slug(lat, lon)}")
            if os.path.exists(cached):
                with open(cached) as handle:
                    lines = json.load(handle).get("lines") or []
            # NLDI's characteristics are gone -- see layers.py. Asking anyway
            # is 120 requests a run for 120 404s.
            values = stream_covariates(
                lat, lon, lines, payload.get("comid"),
                {}, {}, payload.get("flowlines"))
            values.update(landcover_from_streamcat(payload.get("streamcat")))
            out.update(values)
            flowline_note = payload.get("flowline_note")
            if flowline_note and values.get("dist_to_stream_m") is None:
                out["nhdplus_note"] = flowline_note
            layers.record_access(record, "nhdplus", note=flowline_note)
            landcover_note = payload.get("streamcat_note")
            if landcover_note:
                out["nlcd_note"] = landcover_note
            layers.record_access(record, "nlcd",
                                 values.get("landcover_vintage"),
                                 note=landcover_note)
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
                f"echo_{_tile_slug(lat, lon)}",
                lambda: {"records": fetch_outfalls(lat, lon, probe=probe)},
                expects=("records",))
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

    # Say the cost before spending it. The tiled layers cost one request per
    # TILE and NLDI costs one per station, so these two numbers are what the
    # run actually is -- and if the tile count is close to the station count,
    # the tiling is not helping and something is wrong with the geography.
    tiles = {_tile_slug(r["lat"], r["lon"]) for _, r in sites.iterrows()
             if pd.notna(r.get("lat")) and pd.notna(r.get("lon"))}
    print(f"  {total} stations across {len(tiles)} tiles")
    print(f"  coastline and outfalls: ~{len(tiles)} requests each (per tile)")
    print(f"  NHDPlus: up to {total} requests (per station — a comid is a "
          "point lookup)")
    if len(tiles) > total / 2:
        print("  NOTE: nearly as many tiles as stations, so the tiling is "
              "buying little here. Expect this to be slow.")

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
    abandoned = circuit_report()
    if abandoned:
        print(f"\n  layers abandoned this run: {', '.join(abandoned)}")
        print("  Their covariates are empty. That is recorded in the manifest, "
              "and the coverage rule will drop them.")
        for key, layer in layers.LAYERS.items():
            if any(host in layer["endpoint"] for host in abandoned):
                layers.record_access(
                    record, key,
                    note=f"abandoned after {CIRCUIT_THRESHOLD} consecutive "
                         "failures in this run")
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


def _raw_note(row, key):
    value = row.get(f"{key}_note")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _note_of(row, key):
    """The note for a layer, or its parent's where the layer has no fetch.

    nlcd and nhdplus_vaa arrive inside the NHDPlus response. When that request
    fails they are empty with nothing recorded against them, which is
    indistinguishable from code that never called them at all.
    """
    note = _raw_note(row, key)
    if note is not None:
        return note
    parent = layers.DERIVED_FROM.get(key)
    if parent is None:
        return None
    inherited = _raw_note(row, parent)
    return None if inherited is None else f"via {parent}: {inherited}"


def outcomes(frame, want=None, width=150):
    """Why each layer came back the way it did, counted by reason.

    `coverage` says a layer produced nothing. It does not say WHICH nothing,
    and there are four, with four different owners:

      - the service refused or errored        -> theirs, retry or back off
      - it answered, and the box was empty    -> the geography, nothing to fix
      - it answered, and the check rejected it-> mine, the parsing or the check
      - empty with no reason recorded at all  -> mine, and the worst of the four

    A zero in the coverage table looks identical for all four, which is how
    seven tiles of coastline came back empty through a run that printed no
    coastline error. The last row above is the one this exists for: a layer
    that fails silently is indistinguishable from one nobody called.
    """
    tiled = "tile" in frame.columns
    rows = []
    for key in sorted(layers.LAYERS):
        if want is not None and key not in want:
            continue
        supplies = [c for c in layers.LAYERS[key]["supplies"]
                    if c in frame.columns]
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            note = _note_of(row, key)
            empty = (not supplies) or all(pd.isna(row.get(c))
                                          for c in supplies)
            if not empty:
                reason = "populated" if note is None else f"partial: {note}"
            elif note is not None:
                reason = note
            else:
                reason = "EMPTY, NO REASON RECORDED"
            if len(reason) > width:
                reason = reason[:width - 1] + "\u2026"
            rows.append({"layer": key, "reason": reason,
                         "tile": row.get("tile") if tiled else None})
    if not rows:
        return pd.DataFrame(columns=["layer", "reason", "sites", "tiles"])
    table = pd.DataFrame(rows)
    grouped = (table.groupby(["layer", "reason"])
               .agg(sites=("reason", "size"),
                    tiles=("tile", lambda s: int(s.dropna().nunique())))
               .reset_index())
    if not tiled:
        grouped = grouped.drop(columns=["tiles"])
    return grouped.sort_values(["layer", "sites"], ascending=[True, False])


def report_outcomes(frame, want=None):
    table = outcomes(frame, want=want)
    print("\nwhy each layer came back that way (a zero above is one of "
          "these):")
    if table.empty:
        print("  nothing to report — no sites.")
        return table
    print(table.to_string(index=False))

    # The table has to fit a terminal, so a long reason is cut -- and a
    # service's error body is exactly where the answer usually is. Print the
    # cut ones in full underneath rather than making someone open the CSV.
    full = outcomes(frame, want=want, width=10 ** 6)
    cut = [row["reason"] for _, row in full.iterrows()
           if len(row["reason"]) > 150]
    if cut:
        print("\n  in full, for the reasons the table had to cut:")
        for reason in cut:
            print(f"    - {reason}")
    silent = table[table["reason"] == "EMPTY, NO REASON RECORDED"]
    if not silent.empty:
        print("\n  A layer that is empty with no reason recorded is a bug in "
              "this code,")
        print("  not in the service: something returned nothing and said "
              "nothing about it.")
    return table


# Verified 2026-09-15 against a live network: api.epa.gov answers 200 with
# {"items":[{"pctimp2019ws":11.75, "pcturbhi2019ws":9.91, ...}]} for comid
# 6141236. The legacy java.epa.gov host answers 503 "the most likely cause is
# a misconfiguration", which is a broken server rather than a wrong URL.
STREAMCAT = "https://api.epa.gov/StreamCat/streams/metrics"
STREAMCAT_HOSTS = (STREAMCAT, "https://java.epa.gov/StreamCAT/metrics")
# Newest first. Every candidate goes in ONE request and the year that actually
# comes back is what gets recorded -- the same discipline as the NLDI
# catalogue had: the land-cover release is observed, never assumed, because
# NLCD 2011 and NLCD 2019 are not the same covariate.
STREAMCAT_YEARS = (2021, 2019, 2016, 2011)
# NLCD's four developed classes: open space, low, medium, high intensity.
STREAMCAT_URBAN = ("pcturbop", "pcturblo", "pcturbmd", "pcturbhi")


def fetch_streamcat(comid, probe=False):
    """Catchment- and watershed-accumulated land cover for one comid.

    StreamCat is keyed on the comid, which is what a beach gives us. NLDI's
    own characteristics service, which this replaces, 404s on every
    documented path including its own catalogue.
    """
    names = []
    for year in STREAMCAT_YEARS:
        names.append(f"pctimp{year}")
        names.extend(f"{prefix}{year}" for prefix in STREAMCAT_URBAN)
    response = _get(STREAMCAT, params={"name": ",".join(names),
                                       "areaOfInterest": "watershed",
                                       "comid": str(comid)}, probe=probe)
    items = (response.json() or {}).get("items") or []
    if probe:
        print(f"  StreamCat: {len(items)} row(s), "
              f"{sorted(items[0])[:10] if items else '[]'}")
    return items[0] if items else {}


def _percent(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number < 0 else number


def landcover_from_streamcat(row):
    """impervious_frac and developed_frac, from whichever year answered.

    The suffix matters: `...ws` is accumulated over the whole upstream
    watershed, `...cat` is the local catchment only, and `...wsrp100` is a
    100 m riparian buffer. A2 asks for the upstream catchment, so `ws`.
    """
    out = {}
    if not row:
        return out
    for year in STREAMCAT_YEARS:
        impervious = _percent(row.get(f"pctimp{year}ws"))
        urban = [_percent(row.get(f"{prefix}{year}ws"))
                 for prefix in STREAMCAT_URBAN]
        present = [value for value in urban if value is not None]
        if impervious is None and not present:
            continue
        if impervious is not None:
            out["impervious_frac"] = round(impervious / 100.0, 4)
        if present:
            out["developed_frac"] = round(sum(present) / 100.0, 4)
        out["landcover_vintage"] = (f"NLCD {year}, accumulated over the "
                                    f"upstream watershed "
                                    f"(StreamCat pctimp{year}ws)")
        out["landcover_source"] = "streamcat"
        break
    return out


def probe_streams(lat, lon):
    """Ask the services themselves which stream/land-cover path is real.

    Every characteristics path in this module was transcribed from
    documentation and none of them answers: 115 of 120 Rhode Island stations
    got a comid and then a 404 from all three. The documented example is for a
    CRAWLED feature source (nwissite/USGS-...), not a raw comid, which would
    explain a clean 404 for a comid that certainly exists -- but that is a
    guess, and guessing is what produced the three dead paths. This asks.

    Prints raw status and body. It interprets nothing.
    """
    def show(label, url, params=None):
        print(f"\n--- {label}\n    {url}")
        if params:
            print(f"    params {params}")
        try:
            response = _get(url, params=params, probe=False,
                            count_failures=False)
            body = response.text
            print(f"    HTTP {response.status_code}  {len(body)} bytes")
            print(f"    {body[:400]}")
            return response
        except LayerFailed as exc:
            print(f"    FAILED {exc}")
            return None

    show("the feature sources NLDI will accept", NLDI_BASE, {"f": "json"})

    comid = None
    try:
        comid, _geometry, properties = fetch_comid(lat, lon)
        print(f"\n--- comid at POINT({lon} {lat})\n    {comid}")
        print(f"    properties: {sorted(properties)[:12]}")
    except LayerFailed as exc:
        print(f"\n--- comid at POINT({lon} {lat})\n    FAILED {exc}")

    if comid:
        for path in NLDI_CHARACTERISTIC_PATHS:
            show("characteristics", path.format(base=NLDI_BASE, comid=comid),
                 {"f": "json"})
        # The documented example uses a crawled source, so try that shape too:
        # if this answers and the comid one does not, the featureSource is the
        # problem rather than the path.
        show("characteristics for a known crawled feature (the documented "
             "example)",
             f"{NLDI_BASE}/nwissite/USGS-05429700/local",
             {"characteristicId": "CAT_BFI", "f": "json"})
        for host in STREAMCAT_HOSTS:
            show("StreamCat, which is keyed on comid rather than on a feature",
                 host,
                 {"name": "pcturbhi2019,pcturbmd2019,pcturblo2019,"
                          "pcturbop2019,pctimp2019",
                  "areaOfInterest": "watershed", "comid": comid})

    show("the WATERS flowline service, which answered with an ArcGIS error "
         "page rather than features", WATERS_FLOWLINE,
         {"geometry": f"{lon - 0.02},{lat - 0.02},{lon + 0.02},{lat + 0.02}",
          "geometryType": "esriGeometryEnvelope", "inSR": "4326",
          "spatialRel": "esriSpatialRelIntersects", "outFields": "*",
          "returnGeometry": "true", "outSR": "4326", "f": "geojson"})

    for path in NLDI_CATALOGUE_PATHS:
        show("characteristic catalogue",
             path.format(lookups=NLDI_LOOKUPS, base=NLDI_BASE), {"f": "json"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float)
    parser.add_argument("--lon", type=float)
    parser.add_argument("--site")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="print what each layer actually returns, then stop")
    parser.add_argument("--probe-streams", action="store_true",
                        help="ask NLDI and StreamCat which stream/land-cover "
                             "path actually answers, and print it raw")
    args = parser.parse_args()

    if args.probe_streams:
        lat, lon = args.lat, args.lon
        if lat is None or lon is None:
            path = os.path.join(config.DATA_DIR, "site_covariates.csv")
            sites_path = os.path.join(config.DATA_DIR, "stations.csv")
            if not os.path.exists(sites_path):
                sys.exit("give --lat/--lon, or run --stations first")
            sites = pd.read_csv(sites_path, low_memory=False)
            if os.path.exists(path):
                done = set(pd.read_csv(path)["station_id"].astype(str))
                sites = sites[sites["station_id"].astype(str).isin(done)]
            row = sites.dropna(subset=["lat", "lon"]).iloc[0]
            lat, lon = float(row["lat"]), float(row["lon"])
            print(f"probing at {row['station_id']}  ({lat}, {lon})")
        probe_streams(lat, lon)
        return

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
