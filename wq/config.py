"""Everything pre-registered, in one place.

Every constant in the PRE-REGISTERED block below is copied into
wq_manifest.json and hashed. Change one after the manifest is written and
wq/fit.py stops with the name of the constant that moved, rather than fitting
under a specification nobody wrote down. That is the whole point of the
block: the file is not documentation, it is the thing the guard compares
against.

Constants outside that block (paths, request pacing, retry counts) do not
affect the specification and are not hashed.
"""

import hashlib
import json
import os

# ---------------------------------------------------------------------------
# Paths — not part of the specification
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "wq")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUT_DIR = os.path.join(DATA_DIR, "out")
MANIFEST_PATH = os.path.join(ROOT, "wq_manifest.json")
THRESHOLDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "thresholds.json")
STRATA_OVERRIDE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "strata_overrides.csv")

# ---------------------------------------------------------------------------
# PRE-REGISTERED — hashed into the manifest
# ---------------------------------------------------------------------------

# A1: the analytes. Keys are this project's internal codes; the values are the
# WQP CharacteristicName strings that map onto them. Matching is done by
# wq.clean.analyte_key, which also folds in the California CKAN spellings.
ANALYTES = {
    "ENT": "Enterococcus",
    "ECOLI": "Escherichia coli",
    "TOTAL": "Total Coliform",
    "FECAL": "Fecal Coliform",
}

# A1: the predictors, grouped into families. A family is one finding, not
# several: rain_24h/48h/72h are rolling sums of each other and will always
# rank together, so the report never counts them as independent successes.
PREDICTOR_FAMILIES = {
    "rain": ["rain_24h_mm", "rain_48h_mm", "rain_72h_mm"],
    "tide": ["level_m", "rate_m_per_hr"],
    "wave": ["wave_height", "wave_period"],
    "water_temp": ["water_temp_c"],
    "air_temp": ["temperature_2m"],
    "wind": ["wind_onshore_ms", "wind_alongshore_ms"],
}
PREDICTORS = [name for family in PREDICTOR_FAMILIES.values() for name in family]

# A1: the control. Agencies sample in swim season and rain is seasonal, so a
# raw correlation can be the calendar and nothing else. Same guard the rip
# pipeline applies for hour-of-day (analyze_drivers.demean_by).
CONTROL = "per-month demeaning of both sides"
CONTROL_KEY = "month"

# A2 (addition): site covariates derived automatically from the station
# coordinate. FROZEN HERE, BEFORE FITTING. A covariate added after seeing
# which grouping tidies up the coefficient distribution is not this list --
# it goes into a separate, later, clearly-labelled exploratory manifest entry
# (wq/manifest.py --amend), and its results are reported as exploratory.
#
# Source and vintage for every one of these is in wq/layers.py and is copied
# into wq_manifest.json at registration.
SITE_COVARIATES = [
    # NHD / NHDPlus
    "dist_to_stream_m", "stream_order", "upstream_area_km2",
    "n_streams_within_2km",
    # EPA ECHO / FRS
    "dist_to_outfall_m", "outfall_type", "n_outfalls_within_2km",
    # NLCD, accumulated over the upstream catchment
    "impervious_frac", "developed_frac",
    # Coastline vector
    "shore_normal_deg", "curvature_1_per_km", "embayment_ratio",
    "land_fraction_5km", "fetch_km_mean", "fetch_km_min", "fetch_km_max",
    # CO-OPS datums
    "tidal_range_m", "datum_gauge_dist_km",
]

# The eight octant fetches are carried per site but are not stratified on
# individually -- eight more groupings over the same sites is eight more
# chances to find a split that flatters the distribution. fetch_km_min/mean/max
# above are the pre-registered summaries.
FETCH_OCTANT_COLUMNS = [f"fetch_km_{o}" for o in
                        ("N", "NE", "E", "SE", "S", "SW", "W", "NW")]

# Which of the covariates D2 actually BREAKS THE DISTRIBUTION OUT BY. Every
# covariate above is recorded and coverage-checked; only these are groupings,
# because each additional grouping is another chance for a split to look
# explanatory by luck and D5 already has enough of those to account for.
#
# Deliberately absent:
#   shore_normal_deg     a direction. 359 and 1 degrees are adjacent, so a
#                        median split is meaningless. It feeds the onshore
#                        and alongshore wind predictors instead.
#   datum_gauge_dist_km  a data-quality figure: how far away the gauge that
#                        supplied tidal_range_m is. Grouping on it would be
#                        grouping on measurement quality.
#   fetch_km_min/max     redundant with fetch_km_mean, and they move together.
STRATIFY_ON = [
    "beach_type", "region",
    "dist_to_stream_m", "stream_order", "upstream_area_km2",
    "n_streams_within_2km",
    "dist_to_outfall_m", "outfall_type", "n_outfalls_within_2km",
    "impervious_frac", "developed_frac",
    "curvature_1_per_km", "embayment_ratio", "land_fraction_5km",
    "fetch_km_mean", "tidal_range_m",
]

# A covariate populated for fewer stations than this is dropped from the
# pre-registered list rather than fitted on a biased subset. One threshold for
# all of them, so it cannot be tuned per covariate after the fact.
COVARIATE_MIN_COVERAGE = 0.70

# A1/A2: the site strata. `required_coverage` is the share of qualifying sites
# a variable must be populated for to survive into the fit. A variable below
# its threshold is DROPPED by wq/manifest.py before any model runs, and the
# manifest records that it was dropped and why — the alternative, adding a
# stratum after seeing the coefficients, is the thing this design exists to
# prevent.
STRATA = {
    "beach_type": {
        "values": ["open_coast", "enclosed_bay", "storm_drain_adjacent"],
        "required_coverage": 0.70,
        "source": "ASSIGNED BY HAND from satellite imagery — see wq/review.py",
        "automated": False,
        "why_manual": "enclosure is continuous, not categorical, and any "
                      "threshold on land_fraction_5km or embayment_ratio "
                      "misclassifies exactly the ambiguous sites that decide "
                      "whether the stratification works. Those two sort the "
                      "review list; they never assign the label.",
    },
    "region": {
        "values": ["Pacific", "Atlantic", "Gulf", "Great Lakes"],
        "required_coverage": 0.95,
        "source": "state code and coordinate box",
        "automated": True,
    },
}

# Strata that are continuous covariates rather than categories. The report
# splits these at their median, so they read the same way as a category.
CONTINUOUS_STRATA = ["dist_to_stream_m", "upstream_area_km2",
                     "dist_to_outfall_m", "impervious_frac", "developed_frac",
                     "curvature_1_per_km", "embayment_ratio",
                     "land_fraction_5km", "fetch_km_mean", "tidal_range_m"]

# A1: the minimum sample threshold per site. Below this a site is excluded
# from the distribution entirely rather than contributing a noisy coefficient.
MIN_SAMPLES_PER_SITE = 30

# B3: non-detect handling. Substitution at DL/2 is the default; a site whose
# non-detect share exceeds the flag fraction is reported SEPARATELY rather
# than mixed into the headline distribution.
NONDETECT_SUBSTITUTION = "detection_limit/2"
NONDETECT_FLAG_FRACTION = 0.30

# C2: a site also needs this many paired (sample, covariate) rows for a given
# predictor before that predictor is fitted there.
MIN_PAIRED_N = 30

# C4: exceedance fitting needs both classes present in some quantity.
MIN_EXCEEDANCE_DAYS = 5

# D5: alpha for the "significant sites" count and its false-positive expectation.
ALPHA = 0.05

# D6: a site has NO usable predictor when its strongest |rho| after the control
# is below this. 0.30 rather than 0.20 because this is a MAX over the whole
# predictor list, and the best of eleven noise correlations at n=120 clears
# 0.20 routinely. The lenient floor is reported beside it, and so is a
# multiplicity-corrected significance criterion that does not depend on
# picking a floor at all (see report.best_of_k_threshold).
USABLE_RHO_FLOOR = 0.30
USABLE_RHO_FLOOR_LENIENT = 0.20

# D2: permutations used to ask whether a stratum narrows the spread by more
# than an arbitrary split of the same sizes would. Subsetting always narrows
# an IQR, so the overall IQR alone is not the right comparison.
STRATUM_PERMUTATIONS = 400

# B1: the pull window.
YEARS_BACK = 10

# B1: which WQP site types count as coastal recreational water. Unverified
# against a live response — wq/pull.py --probe prints the values that actually
# come back and stops, rather than matching on a string that does not exist.
COASTAL_SITE_TYPES = ("Ocean", "Estuary", "Great Lake")

# A2/B1: how close a feature has to be to count for a stratum.
FRESHWATER_RADIUS_M = 500
OUTFALL_RADIUS_KM = 1.0
MAX_TIDE_GAUGE_KM = 50

SPEC_KEYS = (
    "ANALYTES", "PREDICTOR_FAMILIES", "PREDICTORS", "CONTROL", "CONTROL_KEY",
    "STRATA", "SITE_COVARIATES", "CONTINUOUS_STRATA", "STRATIFY_ON",
    "COVARIATE_MIN_COVERAGE", "FETCH_OCTANT_COLUMNS", "MIN_SAMPLES_PER_SITE", "NONDETECT_SUBSTITUTION",
    "NONDETECT_FLAG_FRACTION", "MIN_PAIRED_N", "MIN_EXCEEDANCE_DAYS", "ALPHA",
    "USABLE_RHO_FLOOR", "USABLE_RHO_FLOOR_LENIENT",
    "STRATUM_PERMUTATIONS", "YEARS_BACK",
    "COASTAL_SITE_TYPES", "FRESHWATER_RADIUS_M", "OUTFALL_RADIUS_KM",
    "MAX_TIDE_GAUGE_KM",
)

# ---------------------------------------------------------------------------
# Not part of the specification
# ---------------------------------------------------------------------------

# B3 (diagnostic, NOT a specification constant): the share of a site-analyte
# series that has to sit at one value before it is called flat. Deliberately
# outside SPEC_KEYS, because nothing here changes which coefficient is
# computed or which pair enters the headline distribution -- it changes only
# what the report can SAY about a pair. The moment a flat series is excluded
# rather than described, this becomes load-bearing and belongs in the
# pre-registered block with a re-registration to match.
#
# The case it exists for: a New Jersey station reporting fecal coliform = 3.0
# for all 49 of its samples, with no censoring qualifier anywhere on the row.
# That is an undeclared "<3" -- the method's floor written as a number -- and
# the non-detect machinery cannot see it, so nondetect_fraction reads 0.00 and
# the pair enters the headline, contributes no coefficient, and then counts
# against D6 as a beach where prediction did not work.
# What a covariate build has to achieve before its output may replace the
# file already on disk. NOT in SPEC_KEYS: these gate whether a BUILD is
# accepted, and change no coefficient, no pair and no floor. Nothing about the
# pre-registered analysis moves if they move.
#
# They exist because the guard that came first compared coverage only against
# the stations the previous file already held, and the 723 new Californian
# stations were in no previous file. A quota trip landing entirely inside
# California would have passed that check by construction.
#
# The asymmetry between the four is the whole design:
#
#   era5        unconditional, and nearly total. It is a reanalysis GRID --
#               every coastal point on earth has a value, so a station with no
#               ERA5 row did not fail to have weather, it failed to be asked.
#               A gap here can only be the quota or the service.
#   marine      conditional. Open-Meteo's wave model has no answer for a land
#               cell, and the cache records that as an absence rather than a
#               refusal (covariates._EMPTY_MARKER). Stations whose cell is
#               cached as empty are not eligible and are not counted against
#               this.
#   tide        conditional. A station with no CO-OPS gauge within
#               MAX_TIDE_GAUGE_KM has no tide covariate for a reason that is a
#               fact about the coast. Only stations WITH an assigned gauge are
#               eligible.
#   water_temp  not gated at all. Gauges that report water level frequently do
#               not report temperature; it was populated for 31% of stations
#               in the Northeast pass and that is the gauge network, not a
#               failure. Recorded in the status file, never a reason to refuse.
#
# A source with min_station_coverage None is recorded and never gates.
COVARIATE_SOURCE_REQUIREMENTS = {
    "era5": {
        "predictors": ["rain_24h_mm", "rain_48h_mm", "rain_72h_mm",
                       "temperature_2m", "wind_onshore_ms",
                       "wind_alongshore_ms"],
        "eligibility": "all",
        "min_station_coverage": 0.99,
        "why": "a reanalysis grid has a value everywhere; a gap is a refusal",
    },
    "marine": {
        "predictors": ["wave_height", "wave_period"],
        "eligibility": "cell_has_water",
        "min_station_coverage": 0.95,
        "why": "no waves in a land cell is an answer, not a failure",
    },
    "tide": {
        "predictors": ["level_m", "rate_m_per_hr"],
        "eligibility": "has_tide_gauge",
        "min_station_coverage": 0.95,
        "why": "no gauge within MAX_TIDE_GAUGE_KM is a fact about the coast",
    },
    "water_temp": {
        "predictors": ["water_temp_c"],
        "eligibility": "has_tide_gauge",
        "min_station_coverage": None,
        "why": "water-level gauges often do not report temperature",
    },
}

FLAT_SERIES_SHARE = 0.90

REQUEST_PAUSE = 0.5
GRID_CELL_DEGREES = 0.1  # ERA5 cells are shared between nearby sites
COVARIATE_JOIN_DAYS = 0  # conditions are matched on the sample's own day


def specification():
    """The frozen specification, as a plain dict."""
    here = globals()
    return {key: here[key] for key in SPEC_KEYS}


def spec_hash():
    """Stable hash of the specification. Any edit to a pre-registered constant
    changes it, which is what lets fit.py refuse to run."""
    blob = json.dumps(specification(), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
