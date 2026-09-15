"""B2-B4: make WQP's mess explicit, and count every record it costs.

Each step returns its counts into one hygiene log, written to
data/wq/out/hygiene_log.csv. The log is the point: "n=41,000 samples" is not
a finding unless the 9,000 records that did not make it are accounted for,
and every one of the steps below can quietly delete data if it is written
carelessly.

  duplicates        exact repeats of (station, datetime, analyte, value)
  methods           recorded per record, never pooled silently. Two
                    estimators inside one site is a flag on that site, not
                    an average.
  units             per-mL and per-L are converted to per-100-mL; CFU and
                    MPN are NOT converted into each other, because they are
                    different estimators and no factor relates them.
  status codes      rejected results dropped and counted
  replicates        field replicates identified and collapsed, not counted
                    as independent samples
  non-detects       substituted at DL/2, never dropped and never zeroed;
                    the share is carried per site and per analyte, and a
                    site over the flag fraction is reported separately
  over-range        ">24196" is censoring at the top of a Quanti-Tray, and
                    it lands on exactly the wet days the study is about

The log10 transform at the end is the same one the existing analysis uses.
"""

import os
import re

import numpy as np
import pandas as pd

from . import config, pull

# Status values that mean the laboratory or the agency withdrew the result.
REJECTED_STATUS = ("rejected", "invalid", "void")
# ActivityTypeCode values that are not a distinct environmental sample.
REPLICATE_WORDS = ("replicate", "duplicate", "split")
BLANK_WORDS = ("blank", "spike", "reference", "calibration")

# Unit strings, normalised to counts per 100 mL. CFU and MPN are deliberately
# both factor 1.0: they are recorded, not converted.
UNIT_FACTORS = {
    "cfu/100ml": 1.0, "mpn/100ml": 1.0, "#/100ml": 1.0, "n/100ml": 1.0,
    "count/100ml": 1.0, "col/100ml": 1.0, "cfu/100 ml": 1.0,
    "mpn/100 ml": 1.0, "#/100 ml": 1.0, "per 100ml": 1.0, "cfu": 1.0,
    "mpn": 1.0, "#": 1.0,
    "cfu/ml": 100.0, "mpn/ml": 100.0, "#/ml": 100.0, "count/ml": 100.0,
    "cfu/l": 0.1, "mpn/l": 0.1, "#/l": 0.1, "count/l": 0.1,
    "cfu/100l": 0.001, "mpn/100l": 0.001,
}
CFU_WORDS = ("cfu", "col")
MPN_WORDS = ("mpn",)


def _lower(series):
    return series.astype(str).str.strip().str.lower()


def analyte_key(name):
    """ENT | ECOLI | TOTAL | FECAL | None.

    Delegates to analyze_drivers.analyte_key, which the existing water-quality
    analysis already uses, and adds the spellings WQP produces that the
    California CKAN feed does not ('Coliform, fecal', 'Enterococci').
    """
    try:
        from analyze_drivers import analyte_key as base
        key = base(name)
        if key:
            return key
    except ImportError:
        pass
    text = str(name).upper()
    if "ENTEROCOCC" in text:
        return "ENT"
    if "COLI" in text and "FECAL" in text:
        return "FECAL"
    if "COLI" in text and "TOTAL" in text:
        return "TOTAL"
    if "ESCHERICHIA" in text or "E COLI" in text:
        return "ECOLI"
    return None


class Log:
    """Every count the hygiene steps produce, in the order they happened."""

    def __init__(self):
        self.rows = []

    def add(self, step, detail, records, share_of=None):
        self.rows.append({
            "step": step,
            "detail": detail,
            "records": int(records),
            "share": (round(records / share_of, 4)
                      if share_of else np.nan),
        })
        pct = f"  ({records / share_of:.1%})" if share_of else ""
        print(f"  {step:<18} {detail:<46} {records:>8,}{pct}")

    def frame(self):
        return pd.DataFrame(self.rows)

    def write(self, path=None):
        path = path or os.path.join(config.OUT_DIR, "hygiene_log.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.frame().to_csv(path, index=False)
        print(f"\nhygiene log -> {path}")
        return path


def parse_value(raw):
    """(value, censoring). Handles '<10', '> 24196', '2,400' and plain numbers.

    A censored string is the single most dangerous field in this dataset:
    pd.to_numeric turns '<10' into NaN, and a pipeline that drops NaN has
    just deleted every clean sample at a site and kept every dirty one.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan, ""
    if isinstance(raw, (int, float)):
        return float(raw), ""
    text = str(raw).strip().replace(",", "")
    if not text:
        return np.nan, ""
    match = re.match(r"^([<>]=?)\s*([0-9.eE+-]+)$", text)
    if match:
        try:
            return float(match.group(2)), ("below" if "<" in match.group(1)
                                           else "above")
        except ValueError:
            return np.nan, ""
    try:
        return float(text), ""
    except ValueError:
        return np.nan, ""


def normalize(raw, log=None, source="WQP"):
    """WQP's result table -> this module's columns. No rows dropped here."""
    log = log or Log()
    frame = pd.DataFrame(index=raw.index)
    frame["source"] = source
    frame["station_id"] = raw[pull.COL_STATION].astype(str)
    frame["analyte_raw"] = raw[pull.COL_ANALYTE].astype(str)
    frame["analyte"] = frame["analyte_raw"].map(analyte_key)

    parsed = raw[pull.COL_VALUE].map(parse_value)
    frame["value_reported"] = [p[0] for p in parsed]
    frame["censoring"] = [p[1] for p in parsed]

    frame["unit_raw"] = (raw[pull.COL_UNIT].astype(str)
                         if pull.COL_UNIT in raw else "")
    frame["method"] = (raw[pull.COL_METHOD].astype(str)
                       if pull.COL_METHOD in raw else "")
    frame["status"] = (raw[pull.COL_STATUS].astype(str)
                       if pull.COL_STATUS in raw else "")
    frame["qualifier"] = (raw[pull.COL_QUALIFIER].astype(str)
                          if pull.COL_QUALIFIER in raw else "")
    frame["activity_type"] = (raw[pull.COL_ACTIVITY_TYPE].astype(str)
                              if pull.COL_ACTIVITY_TYPE in raw else "")
    frame["detection_limit"] = (
        pd.to_numeric(raw[pull.COL_DETECTION_LIMIT], errors="coerce")
        if pull.COL_DETECTION_LIMIT in raw else np.nan)
    frame["nondetect_text"] = (raw[pull.COL_NONDETECT].astype(str)
                               if pull.COL_NONDETECT in raw else "")

    stamp = raw[pull.COL_DATE].astype(str)
    if pull.COL_TIME in raw:
        times = raw[pull.COL_TIME].fillna("").astype(str)
        stamp = stamp.str.cat(times, sep=" ").str.strip()
    frame["sampled_at"] = pd.to_datetime(stamp, errors="coerce", utc=True)
    frame["date"] = pd.to_datetime(raw[pull.COL_DATE], errors="coerce").dt.normalize()

    unknown = frame["analyte"].isna()
    log.add("normalize", "records read", len(frame))
    if unknown.any():
        kinds = sorted(set(frame.loc[unknown, "analyte_raw"]))[:4]
        log.add("normalize", f"analyte not one of the four ({', '.join(kinds)})",
                int(unknown.sum()), len(frame))
    undated = frame["date"].isna()
    if undated.any():
        log.add("normalize", "no parseable date — dropped",
                int(undated.sum()), len(frame))
    return frame[~unknown & ~undated].copy(), log


def normalize_ckan(raw, log=None):
    """California's CKAN results, into the same columns.

    The CKAN feed carries a qualifier code but no detection-limit column and
    no sample time, so its non-detects arrive less well described than WQP's.
    That difference travels in the `source` column rather than being smoothed
    over, because it changes what the non-detect share at a CA site means.
    """
    log = log or Log()
    frame = pd.DataFrame(index=raw.index)
    frame["source"] = "CKAN"
    frame["station_id"] = raw["StationCode"].astype(str)
    frame["station_name_ckan"] = raw.get("StationName", "").astype(str)
    frame["analyte_raw"] = raw["Analyte"].astype(str)
    frame["analyte"] = frame["analyte_raw"].map(analyte_key)
    parsed = raw["Result"].map(parse_value)
    frame["value_reported"] = [p[0] for p in parsed]
    frame["censoring"] = [p[1] for p in parsed]
    frame["unit_raw"] = raw.get("Unit", "").astype(str)
    frame["method"] = ""
    frame["status"] = ""
    frame["qualifier"] = raw.get("ResultQualCode", "").astype(str)
    frame["activity_type"] = ""
    frame["detection_limit"] = np.nan
    frame["nondetect_text"] = ""
    frame["date"] = pd.to_datetime(raw["SampleDate"], errors="coerce").dt.normalize()
    frame["sampled_at"] = frame["date"].dt.tz_localize("UTC")

    # CKAN marks a non-detect with a qualifier code rather than a blank value:
    # '<' means the reported number IS the detection limit.
    less = _lower(frame["qualifier"]).str.contains("<", na=False)
    frame.loc[less, "censoring"] = "below"
    frame.loc[less, "detection_limit"] = frame.loc[less, "value_reported"]

    usable = frame["analyte"].notna() & frame["date"].notna()
    log.add("normalize", "CKAN records read", len(frame))
    log.add("normalize", "CKAN records usable", int(usable.sum()), len(frame))
    return frame[usable].copy(), log


def drop_rejected(frame, log):
    """QA/QC and result-status codes."""
    before = len(frame)
    status = _lower(frame["status"])
    qualifier = _lower(frame["qualifier"])
    rejected = status.apply(lambda s: any(w in s for w in REJECTED_STATUS))
    rejected |= qualifier.str.fullmatch(r"r|rej|rjc", na=False)
    if rejected.any():
        log.add("qa/qc", "rejected or void status — dropped",
                int(rejected.sum()), before)
    else:
        log.add("qa/qc", "rejected or void status — dropped", 0, before)
    preliminary = status.str.contains("prelim", na=False)
    if preliminary.any():
        log.add("qa/qc", "preliminary status — KEPT, flagged",
                int(preliminary.sum()), before)
    frame = frame[~rejected].copy()
    frame["preliminary"] = preliminary[~rejected].to_numpy()
    return frame, log


def drop_non_samples(frame, log):
    """Blanks, spikes and other quality-control activities that are not
    environmental samples at all. Field replicates are handled separately,
    because they ARE samples — just not independent ones."""
    before = len(frame)
    kind = _lower(frame["activity_type"])
    quality = kind.apply(lambda k: any(w in k for w in BLANK_WORDS))
    log.add("qa/qc", "blanks/spikes — dropped", int(quality.sum()), before)
    return frame[~quality].copy(), log


def collapse_replicates(frame, log):
    """Field replicates are averaged into their sample, not counted twice.

    Counting a replicate as a second sample inflates n without adding
    information, and n is the number every coefficient in this study is
    reported beside. The average is taken on the reported value; the flag
    survives so a site's replicate share is visible.
    """
    before = len(frame)
    kind = _lower(frame["activity_type"])
    replicate = kind.apply(lambda k: any(w in k for w in REPLICATE_WORDS))
    log.add("replicates", "field replicates found", int(replicate.sum()), before)
    if not replicate.any():
        frame["replicate_group"] = 1
        return frame, log

    keys = ["station_id", "analyte", "sampled_at"]
    frame = frame.copy()
    frame["_is_replicate"] = replicate
    sizes = frame.groupby(keys, dropna=False)["value_reported"].transform("size")
    frame["replicate_group"] = sizes
    collapsed = (frame.sort_values("_is_replicate")
                 .groupby(keys, dropna=False, as_index=False)
                 .agg({**{c: "first" for c in frame.columns if c not in keys},
                       "value_reported": "mean"}))
    log.add("replicates", "records merged into their sample",
            before - len(collapsed), before)
    return collapsed.drop(columns=["_is_replicate"]), log


def drop_duplicates(frame, log):
    """Exact repeats: same station, datetime, analyte, value, method."""
    before = len(frame)
    keys = ["station_id", "sampled_at", "analyte", "value_reported", "method"]
    keys = [k for k in keys if k in frame.columns]
    out = frame.drop_duplicates(subset=keys)
    log.add("duplicates", "identical (station, time, analyte, value, method)",
            before - len(out), before)
    return out.copy(), log


def normalize_units(frame, log):
    """Per-mL and per-L converted to per-100-mL; CFU and MPN recorded, not
    converted. An unrecognised unit is dropped rather than assumed."""
    before = len(frame)
    unit = _lower(frame["unit_raw"]).str.replace(" ", "", regex=False)
    factors = unit.map({k.replace(" ", ""): v for k, v in UNIT_FACTORS.items()})

    frame = frame.copy()
    frame["unit_factor"] = factors
    frame["value"] = frame["value_reported"] * factors
    frame["estimator"] = np.where(
        unit.str.contains("|".join(CFU_WORDS), na=False), "CFU",
        np.where(unit.str.contains("|".join(MPN_WORDS), na=False), "MPN", "unknown"))

    converted = int((factors.notna() & (factors != 1.0)).sum())
    log.add("units", "converted to counts per 100 mL", converted, before)
    unrecognised = factors.isna()
    if unrecognised.any():
        seen = sorted(set(frame.loc[unrecognised, "unit_raw"].astype(str)))[:5]
        log.add("units", f"unrecognised unit — dropped ({', '.join(seen)})",
                int(unrecognised.sum()), before)
    else:
        log.add("units", "unrecognised unit — dropped", 0, before)
    for estimator in ("CFU", "MPN", "unknown"):
        count = int((frame["estimator"] == estimator).sum())
        log.add("units", f"estimator {estimator} (never rescaled into another)",
                count, before)
    return frame[~unrecognised].copy(), log


def substitute_nondetects(frame, log):
    """B3. DL/2 for a non-detect, the limit itself for an over-range result.

    Neither is dropped and neither becomes zero. A non-detect IS information
    -- it is the clean days -- and dropping them is how a pipeline ends up
    reporting that a beach is always dirty.
    """
    before = len(frame)
    frame = frame.copy()
    text = _lower(frame["nondetect_text"])
    flagged = text.str.contains("not detected|below|non-detect|nondetect",
                                na=False, regex=True)
    frame["nondetect"] = flagged | (frame["censoring"] == "below")
    frame["over_range"] = frame["censoring"] == "above"

    # A non-detect's value is its detection limit: from the DL column, from
    # the reported '<N', or -- failing both -- from the smallest limit that
    # analyte was ever reported with at this station.
    limit = frame["detection_limit"].copy()
    fallback = frame["value_reported"].where(frame["censoring"] == "below")
    limit = limit.fillna(fallback)
    per_analyte = (frame.assign(_lim=limit)
                   .groupby(["station_id", "analyte"], dropna=False)["_lim"]
                   .transform("min"))
    still_missing = frame["nondetect"] & limit.isna()
    limit = limit.fillna(per_analyte)
    unresolved = frame["nondetect"] & limit.isna()

    factor = 0.5 if config.NONDETECT_SUBSTITUTION == "detection_limit/2" else 1.0
    substituted = frame["nondetect"] & limit.notna()
    frame.loc[substituted, "value"] = (limit[substituted] * factor
                                       * frame.loc[substituted, "unit_factor"]
                                       .fillna(1.0))
    frame.loc[frame["over_range"], "value"] = (
        frame.loc[frame["over_range"], "value_reported"]
        * frame.loc[frame["over_range"], "unit_factor"].fillna(1.0))

    log.add("non-detects", "flagged", int(frame["nondetect"].sum()), before)
    log.add("non-detects", f"substituted at {config.NONDETECT_SUBSTITUTION}",
            int(substituted.sum()), before)
    log.add("non-detects", "no limit on the record, taken from the station's "
                           "smallest",
            int((still_missing & ~unresolved).sum()), before)
    log.add("non-detects", "no limit anywhere — dropped",
            int(unresolved.sum()), before)
    log.add("non-detects", "over-range (censored at the top) — kept at the limit",
            int(frame["over_range"].sum()), before)

    frame = frame[~unresolved].copy()
    missing = frame["value"].isna()
    log.add("values", "no usable numeric value — dropped",
            int(missing.sum()), before)
    return frame[~missing].copy(), log


def add_log_value(frame, log):
    """B4. log10, the same transform the existing analysis uses."""
    frame = frame.copy()
    negative = frame["value"] < 0
    if negative.any():
        log.add("values", "negative counts — dropped", int(negative.sum()),
                len(frame))
        frame = frame[~negative]
    frame["log_value"] = np.log10(frame["value"].clip(lower=0) + 1)
    return frame, log


def dedupe_across_sources(frame, stations, log, radius_km=0.15):
    """B1. A California beach is in WQP and in CKAN. Keep it once.

    Matching is on (date, analyte, value) between sources at stations whose
    coordinates agree -- not on the station code, which the two sources spell
    differently. WQP wins ties because it carries a sample time, which is
    what makes a sub-daily covariate join possible at all.
    """
    from .strata import _haversine_km

    before = len(frame)
    if not {"WQP", "CKAN"}.issubset(set(frame["source"])):
        log.add("cross-source", "CA duplicates removed (only one source present)",
                0, before)
        return frame, log

    coords = (stations.set_index("station_id")[["lat", "lon"]]
              if stations is not None and not stations.empty else None)
    ckan = frame[frame["source"] == "CKAN"]
    wqp = frame[frame["source"] == "WQP"]
    if coords is None or ckan.empty or wqp.empty:
        log.add("cross-source", "CA duplicates removed (no coordinates to match on)",
                0, before)
        return frame, log

    wqp_keys = set(zip(wqp["date"], wqp["analyte"], wqp["value"].round(3)))
    same_reading = [
        (row.date, row.analyte, round(row.value, 3)) in wqp_keys
        for row in ckan.itertuples()
    ]
    drop_index = ckan.index[pd.Series(same_reading, index=ckan.index)]
    log.add("cross-source", "CKAN records identical to a WQP record — dropped",
            len(drop_index), before)

    # Stations that ARE the same place but whose records did not line up
    # exactly. Reported, not dropped: a genuine second sample and a
    # transcription difference look alike here, and deleting the wrong one
    # silently thins a site.
    if coords is not None and len(coords):
        ckan_ids = set(ckan["station_id"]) & set(coords.index)
        wqp_ids = set(wqp["station_id"]) & set(coords.index)
        overlap = 0
        if ckan_ids and wqp_ids:
            wqp_coords = coords.loc[sorted(wqp_ids)]
            lats = wqp_coords["lat"].to_numpy()
            lons = wqp_coords["lon"].to_numpy()
            for station in sorted(ckan_ids):
                point = coords.loc[station]
                dist = _haversine_km(float(point["lat"]), float(point["lon"]),
                                     lats, lons)
                if len(dist) and dist.min() <= radius_km:
                    overlap += 1
        log.add("cross-source",
                f"CKAN stations within {radius_km * 1000:.0f} m of a WQP "
                "station — co-located, records kept", overlap, before)
    return frame.drop(index=drop_index), log


def nondetect_shares(frame):
    """Per site and per analyte, the share of samples that were non-detects.

    B3 calls for this separately from the substitution because the
    substitution is an assumption and this number says how much of the site
    rests on it. Above config.NONDETECT_FLAG_FRACTION the coefficient is not
    trustworthy and the report separates it out.
    """
    grouped = frame.groupby(["station_id", "analyte"], dropna=False)
    out = grouped.agg(
        n=("value", "size"),
        n_nondetect=("nondetect", "sum"),
        n_over_range=("over_range", "sum"),
        n_methods=("method", lambda s: s.where(s.astype(str) != "").nunique()),
        n_estimators=("estimator", lambda s: s[s != "unknown"].nunique()),
    ).reset_index()
    out["nondetect_fraction"] = out["n_nondetect"] / out["n"]
    out["nondetect_flag"] = out["nondetect_fraction"] > config.NONDETECT_FLAG_FRACTION
    out["mixed_estimators"] = out["n_estimators"] > 1
    return out


def clean(wqp_raw=None, ckan_raw=None, stations=None, log=None):
    """The whole of Part B, in order, with one log out the far end."""
    log = log or Log()
    frames = []
    if wqp_raw is not None and not wqp_raw.empty:
        frame, log = normalize(wqp_raw, log, source="WQP")
        frames.append(frame)
    if ckan_raw is not None and not ckan_raw.empty:
        frame, log = normalize_ckan(ckan_raw, log)
        frames.append(frame)
    if not frames:
        raise SystemExit("nothing to clean — run the pulls first")

    frame = pd.concat(frames, ignore_index=True, sort=False)
    for step in (drop_rejected, drop_non_samples, collapse_replicates,
                 drop_duplicates, normalize_units, substitute_nondetects,
                 add_log_value):
        frame, log = step(frame, log)
    frame, log = dedupe_across_sources(frame, stations, log)

    log.add("final", "samples available to the fit", len(frame))
    print(f"\n  {frame['station_id'].nunique():,} stations, "
          f"{len(frame):,} samples, "
          f"{frame['date'].min():%Y-%m-%d} to {frame['date'].max():%Y-%m-%d}")
    return frame.reset_index(drop=True), log
