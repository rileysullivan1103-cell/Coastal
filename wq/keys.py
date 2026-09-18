"""Identifier columns are strings, and stay strings through a CSV.

A CO-OPS gauge id is 9410678. pandas reads a column of those as int64, or as
float64 the moment one row is missing, and writes it back as "9410678.0". Any
lookup built from that misses.

That is not hypothetical. wq.covariates._tide_gauge_is_dry builds a cache
filename from tide_station. During a covariate build the id was still a string
in memory and the check read tide coverage at 100%; re-run against the
covariate_sources.csv the same build had just written, the identical check
read 87.4%, because every one of 402 lookups now asked for
coops_water_level_9410678.0.csv. A guard that answers differently depending on
whether its input has been through a CSV is not a guard, and the disagreement
was found by accident.

So identifiers are coerced once, here, at every read. An id is text: it is
never arithmetic, its leading zeros matter, and "9410678" and 9410678.0 are
the same station and must compare equal.
"""

import pandas as pd

# Columns that are identifiers rather than measurements. Adding one here is
# enough -- every reader in the package goes through coerce().
KEY_COLUMNS = frozenset({
    "station_id", "site_cluster", "analyte", "predictor",
    "tide_station", "datum_gauge", "gauge", "state", "organization",
    "study_scope", "programme", "outfall_permit", "seed_station",
})


def as_key(value):
    """One identifier, as text, with the float round-trip undone."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "<na>"):
        return ""
    # 9410678.0 -> 9410678. Only when the fractional part is a bare zero, so a
    # genuine decimal identifier is left alone.
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        text = text[:-2]
    return text


def coerce(frame, columns=None):
    """Every identifier column in `frame`, as clean text. Returns the frame."""
    if frame is None or not hasattr(frame, "columns"):
        return frame
    wanted = KEY_COLUMNS if columns is None else set(columns)
    for column in frame.columns:
        if column in wanted:
            frame[column] = frame[column].map(as_key)
    return frame


def read_csv(path, **kwargs):
    """pd.read_csv with the identifier columns already fixed."""
    return coerce(pd.read_csv(path, **kwargs))
