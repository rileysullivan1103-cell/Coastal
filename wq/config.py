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
        "source": "WQP MonitoringLocationTypeName, station-name keywords, "
                  "and wq/strata_overrides.csv",
    },
    "freshwater_input": {
        "values": ["yes", "no"],
        "required_coverage": 0.70,
        "source": "WQP Stream/Spring station within 500 m",
    },
    "outfall_present": {
        "values": ["yes", "no"],
        "required_coverage": 0.70,
        "source": "WQP Facility/outfall station within 1 km",
    },
    "tidal_range_m": {
        "values": "continuous",
        "required_coverage": 0.50,
        "source": "CO-OPS datums MHHW - MLLW at the nearest gauge",
    },
    "watershed_area_km2": {
        "values": "continuous",
        "required_coverage": 0.50,
        "source": "upstream drainage area — no offline source wired in; "
                  "expected to be dropped for coverage",
    },
    "impervious_frac": {
        "values": "continuous",
        "required_coverage": 0.50,
        "source": "upstream impervious fraction — no offline source wired in; "
                  "expected to be dropped for coverage",
    },
    "region": {
        "values": ["Pacific", "Atlantic", "Gulf", "Great Lakes"],
        "required_coverage": 0.95,
        "source": "state code and coordinate box",
    },
}

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
    "STRATA", "MIN_SAMPLES_PER_SITE", "NONDETECT_SUBSTITUTION",
    "NONDETECT_FLAG_FRACTION", "MIN_PAIRED_N", "MIN_EXCEEDANCE_DAYS", "ALPHA",
    "USABLE_RHO_FLOOR", "USABLE_RHO_FLOOR_LENIENT",
    "STRATUM_PERMUTATIONS", "YEARS_BACK",
    "COASTAL_SITE_TYPES", "FRESHWATER_RADIUS_M", "OUTFALL_RADIUS_KM",
    "MAX_TIDE_GAUGE_KM",
)

# ---------------------------------------------------------------------------
# Not part of the specification
# ---------------------------------------------------------------------------

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
