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
# series that has to sit on ONE value before it is called flat. Outside
# SPEC_KEYS, because nothing here changes which coefficient is computed or
# which pair enters the headline -- only what the report can SAY about a pair.
#
# The case it was built for: a New Jersey station reporting fecal coliform =
# 3.0 for all 49 of its samples with no censoring qualifier anywhere. That is
# an undeclared "<3" and the non-detect machinery cannot see it.
#
# 0.875 rather than 0.90, and derived rather than chosen. Binning beach
# rain_48h coefficients by modal share, the median holds flat and only the
# spread widens until the 0.875 boundary, then breaks:
#
#     modal share      pairs   median rho   sd     distinct values
#     (0.80, 0.85]      162       0.246    0.188        14
#     (0.85, 0.875]      57       0.189    0.220        12
#     (0.875, 0.90]      54       0.050    0.241         8.5   <- breaks here
#     (0.90, 0.925]      52       0.154    0.243         7
#     (0.95, 1.01]       29       0.029    0.279         4
#
# Below it, ties cost power and the estimate stays put, which is honest.
# Above it the median collapses and the spread doubles. A station reporting
# 89.5% of its enterococcus at exactly 10 sat just under the old 0.90 bar and
# went unflagged while carrying a headline coefficient of -0.231.
# D2 (reporting, NOT specification): what a stratum has to achieve before the
# report says it EXPLAINS the spread rather than merely differing from a
# shuffle. Outside SPEC_KEYS -- these gate wording, not arithmetic. Every
# number D2 prints is unchanged; only the sentence under the table moves.
#
# They exist because the report said "region narrows the IQR ... that is the
# headline" for a narrowing of 1.5% at p=0.000. With 2,591 sites a p-value
# detects an effect far too small to act on, and the canned verdict read the
# p-value alone.
#
# STRATUM_MIN_NARROWING is derived from the D1 spread rather than chosen
# round. The rain IQRs in D1 run 0.17 to 0.26, so take 0.25 as the working
# spread. Three anchors bracket it:
#
#   0.10 in rho units  the gap between the two pre-registered decision floors
#                      (USABLE_RHO_FLOOR 0.30 and its lenient 0.20). A
#                      narrowing that shifts a site across that gap is one
#                      that could change a decision. Against a 0.25 IQR that
#                      is a 40% narrowing.
#   0.129 in rho units one standard error of a single site's rho at the median
#                      n_ctrl of 63 (1/sqrt(n-3)). A stratum narrowing the
#                      spread by less than one site's own measurement error is
#                      telling you less than one more sample would.
#   0.025 in rho units what a 10% narrowing actually buys on a 0.25 IQR.
#
# So 10% is the floor for REPORTING a stratum as explanatory, not the bar for
# acting on one: below it there is certainly nothing there, above it there
# might be, and the 40% figure is what would actually move a decision. The
# report prints the observed narrowing either way.
STRATUM_MIN_NARROWING = 0.10

# A level holding a handful of stations cannot support a claim about a
# population, however the shuffle scores it: outfall_type's headline came from
# a level of 2 stations in 2,558, which is 0.08%. The shuffle controls for the
# SIZE of a small level and for nothing else its members share.
STRATUM_MIN_SMALLEST_LEVEL_SHARE = 0.05

FLAT_SERIES_SHARE = 0.875

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
