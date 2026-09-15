"""Which spatial layer every covariate came from, and when that layer is from.

A covariate without a vintage is not reproducible: NLCD 2011 and NLCD 2021
give different impervious fractions for the same catchment, and a coefficient
stratified on one is not a coefficient stratified on the other. Every layer
here declares what it is and how to observe its actual vintage from the
response, and wq/manifest.py writes both into wq_manifest.json alongside the
count of stations each one populated.

`declared_vintage` is what the source says it serves and is written by hand
from its documentation. `observed_vintage` is read out of the live response
where the service exposes one -- Overpass returns the planet timestamp its
answer was built from, which is the real thing. Where a service exposes
nothing, observed stays null and the declared value is all there is; that
gap is visible in the manifest rather than papered over.

NOTHING HERE WAS CONFIRMED AGAINST A LIVE RESPONSE. Outbound access to every
one of these hosts was blocked when this was written, so each endpoint and
field name below is transcribed from documentation and carries
`verified: False`. Every fetcher in wq/spatial.py has a --probe mode that
prints what actually comes back and stops, and none of them silently returns
an empty column: run the probes once, correct anything that moved, and flip
the flag.
"""

from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

LAYERS = {
    "nhdplus": {
        "name": "NHDPlus V2.1 via USGS Hydro Network-Linked Data Index",
        "endpoint": "https://api.water.usgs.gov/nldi/linked-data",
        "supplies": ["dist_to_stream_m", "stream_order", "upstream_area_km2",
                     "n_streams_within_2km"],
        "declared_vintage": "NHDPlus V2.1 (NHD snapshot 2012, VAA release 2019)",
        "vintage_source": "service does not report one; declared from USGS "
                          "NLDI documentation",
        "citation": "USGS Hydro Network-Linked Data Index (NLDI), "
                    "https://labs.waterdata.usgs.gov/about-nldi/",
        # The comid lookup on this host works. The three flowline covariates
        # do not come from here at all -- they come from the WATERS flowline
        # query, which is down; see nhdplus_vaa. upstream_area_km2 came from
        # the NLDI characteristics, which are absent; see nlcd.
        "observed_absent": "of the four covariates named above, only the "
                           "comid lookup that keys them still answers "
                           "(probed 2026-09-15). upstream_area_km2 depended "
                           "on NLDI catchment characteristics, which 404 on "
                           "every documented path.",
        "verified": False,
    },
    "nhdplus_vaa": {
        "name": "NHDPlus V2 flowline attributes via EPA WATERS",
        "endpoint": ("https://watersgeo.epa.gov/arcgis/rest/services/"
                     "NHDPlus_NP21/NHDSnapshot_NP21/MapServer"),
        "supplies": ["stream_order"],
        "declared_vintage": "NHDPlus V2.1",
        "vintage_source": "MapServer /info returns a service edit date; read "
                          "where present",
        "citation": "EPA WATERS GeoViewer services",
        # Probed on 2026-09-15 from a live network: the layer 0 /query path
        # answers HTTP 500 with an ArcGIS REST Framework error page reading
        # "Error: Service NHDPlus_NP21/NHDSnapshot_NP21/MapServer", which is
        # the server saying the SERVICE is not loaded -- not a bad request,
        # not a rate limit, and not something a different query shape fixes.
        # No replacement has been probed, so none is claimed here. Until one
        # answers, dist_to_stream_m, stream_order and n_streams_within_2km
        # stay empty and the 70% coverage rule drops them before fitting.
        "observed_absent": "EPA WATERS NHDSnapshot_NP21 MapServer returns "
                           "HTTP 500 'Error: Service NHDPlus_NP21/"
                           "NHDSnapshot_NP21/MapServer' for every flowline "
                           "query (probed 2026-09-15).",
        "replaced_by": None,
        "verified": False,
    },
    "echo": {
        "name": "EPA ECHO, Clean Water Act permitted facilities (NPDES)",
        "endpoint": "https://echodata.epa.gov/echo/cwa_rest_services",
        "supplies": ["dist_to_outfall_m", "outfall_type",
                     "n_outfalls_within_2km"],
        "declared_vintage": "ECHO refreshes weekly from ICIS-NPDES",
        # Probed 2026-09-15 from a live network. The service is UP and the
        # query is right: 113 permitted facilities inside the Rhode Island
        # tile, with the station inside its own bounding box. The download is
        # what fails -- asked for six columns by name it returns two (CWPName,
        # SourceID) and drops FacLat and FacLong silently. An unrecognised
        # column name is not an error to ECHO, it is a column that does not
        # appear, so every distance came back empty and n_outfalls_within_2km
        # read a confident 0 at all 120 sites.
        "observed_absent": "ECHO answers with 113 facilities in the box but "
                           "its CWA download omits FacLat/FacLong, so nothing "
                           "returned can be placed (probed 2026-09-15). The "
                           "correct column names have not been established; "
                           "wq.spatial --probe-outfalls asks the service for "
                           "its own list rather than guessing a third set.",
        "vintage_source": "response carries no timestamp; accessed_at is the "
                          "only date available",
        "note": "the download needs an explicit qcolumns list — ECHO's "
                "default set returns FacLong without FacLat, which places "
                "nothing and fails silently",
        "citation": "US EPA Enforcement and Compliance History Online",
        "caveat": "ECHO gives the FACILITY location, not the outfall pipe. A "
                  "treatment plant sited a kilometre inland of its own "
                  "diffuser makes dist_to_outfall_m an overestimate, and the "
                  "error is not random: big coastal plants discharge further "
                  "offshore than small ones.",
        "verified": False,
    },
    "nlcd": {
        "name": "NLCD impervious and developed land, watershed-accumulated, "
                "via EPA StreamCat",
        "endpoint": "https://api.epa.gov/StreamCat/streams/metrics",
        "supplies": ["impervious_frac", "developed_frac"],
        "declared_vintage": "whichever NLCD year the StreamCat response "
                            "ids carry — recorded per run, not assumed",
        "vintage_source": "the StreamCat metric name carries the year "
                          "(pctimp2019ws is NLCD 2019); every candidate year "
                          "goes in one request and the one that ANSWERS is "
                          "what lands in landcover_vintage, per site",
        "citation": "MRLC National Land Cover Database, accumulated to "
                    "NHDPlus catchments by US EPA (StreamCat)",
        # Probed on 2026-09-15 from a live network: EVERY characteristics path
        # on api.water.usgs.gov returns 404, including the documented example
        # on a crawled feature (nwissite/USGS-05429700/local?characteristicId=
        # CAT_BFI), and so does the catalogue. labs.waterdata.usgs.gov answers
        # 404 with an empty body. The comid lookup on the SAME host works, so
        # this is the characteristics service being absent, not the request
        # being malformed -- which is what four rounds of guessing assumed.
        "observed_absent": "NLDI catchment characteristics 404 on every "
                           "documented path (probed 2026-09-15), so this "
                           "layer is no longer read from NLDI at all.",
        "replaced_by": "EPA StreamCat, api.epa.gov/StreamCat/streams/metrics, "
                       "keyed on the NHDPlus comid. Verified 2026-09-15: HTTP "
                       "200 with pctimp2019ws for comid 6141236. The year is "
                       "whichever of STREAMCAT_YEARS the response carries, "
                       "recorded per site in landcover_vintage.",
        "caveat": "These are accumulated over the upstream watershed of the "
                  "flowline nearest the beach, which is the right denominator "
                  "for a creek mouth and the wrong one for a beach whose "
                  "nearest flowline drains somewhere else entirely.",
        "verified": False,
    },
    "coastline": {
        "name": "OpenStreetMap coastline (natural=coastline) via Overpass",
        "endpoint": "https://overpass-api.de/api/interpreter",
        # These are the COLUMN names, not the concepts. "fetch_km_by_octant"
        # is how the spec names the idea; what the frame actually carries is
        # the three summaries, and naming the idea here left them with no
        # layer beside them in the coverage table -- a covariate whose source
        # reads "None" is one nobody can check the vintage of.
        "supplies": ["shore_normal_deg", "curvature_1_per_km",
                     "embayment_ratio", "land_fraction_5km",
                     "fetch_km_mean", "fetch_km_min", "fetch_km_max"],
        "declared_vintage": "continuously edited",
        "vintage_source": "osm3s.timestamp_osm_base in every response — the "
                          "planet timestamp the answer was built from",
        "citation": "OpenStreetMap contributors, ODbL",
        "caveat": "natural=coastline covers ocean and Gulf shorelines only. "
                  "The Great Lakes are mapped as water polygons, not "
                  "coastline, so every coastline covariate is absent at a "
                  "Great Lakes site and the coverage rule will see that.",
        "verified": False,
    },
    "coops_datums": {
        "name": "NOAA CO-OPS tidal datums",
        "endpoint": ("https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/"
                     "stations"),
        "supplies": ["tidal_range_m", "datum_gauge_dist_km"],
        "declared_vintage": "1983-2001 National Tidal Datum Epoch",
        "vintage_source": "the datums response names its epoch; read where "
                          "present",
        "citation": "NOAA CO-OPS",
        "verified": False,
    },
}

# Layers with no request of their own: they arrive inside another layer's
# response, so their outcome IS that layer's outcome. Without this they go
# quiet when their parent fails -- empty, with nothing recorded against them,
# which reads exactly like a bug in the code that never called them.
DERIVED_FROM = {"nlcd": "nhdplus", "nhdplus_vaa": "nhdplus"}

# The layers for_site() actually branches on. coops_datums is fetched in one
# bulk pull outside the per-site loop, and the derived layers above come back
# inside nhdplus, so neither is something --skip-layers can meaningfully skip.
FETCHED = tuple(key for key in LAYERS
                if key not in DERIVED_FROM and key != "coops_datums")

# Which layer each covariate comes from, so the manifest can be written from
# the covariate side and read from the layer side.
COVARIATE_LAYER = {covariate: key
                   for key, layer in LAYERS.items()
                   for covariate in layer["supplies"]}


def blank_record():
    return {key: {"observed_vintage": None, "accessed_at": None,
                  "sites_attempted": 0, "sites_populated": 0, "notes": []}
            for key in LAYERS}


def record_access(record, key, observed_vintage=None, note=None):
    entry = record.setdefault(key, {"observed_vintage": None,
                                    "accessed_at": None, "sites_attempted": 0,
                                    "sites_populated": 0, "notes": []})
    entry["accessed_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    if observed_vintage and not entry["observed_vintage"]:
        entry["observed_vintage"] = observed_vintage
    if note and note not in entry["notes"]:
        entry["notes"].append(note)
    return entry


def manifest_section(record=None):
    """What goes into wq_manifest.json under "layers"."""
    record = record or blank_record()
    out = {}
    for key, layer in LAYERS.items():
        seen = record.get(key, {})
        out[key] = {
            "name": layer["name"],
            "endpoint": layer["endpoint"],
            "supplies": layer["supplies"],
            "declared_vintage": layer["declared_vintage"],
            "vintage_source": layer["vintage_source"],
            "observed_vintage": seen.get("observed_vintage"),
            "accessed_at": seen.get("accessed_at"),
            "citation": layer["citation"],
            "endpoint_verified_against_live_response": layer["verified"],
            "sites_attempted": seen.get("sites_attempted", 0),
            "sites_populated": seen.get("sites_populated", 0),
        }
        if layer.get("caveat"):
            out[key]["caveat"] = layer["caveat"]
        if seen.get("notes"):
            out[key]["notes"] = seen["notes"]
    return out
