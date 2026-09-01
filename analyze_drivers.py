"""Rank what actually moves rip detections and beach bacteria.

Reads the CSVs already in data/ and reports, per target, which measured
condition tracks it most strongly. Nothing here is causal: these are
associations in observational data, and the two targets have quite different
weaknesses, both of which the output states rather than hides.

Method. Spearman rank correlation throughout, because none of these variables
is remotely normal -- rainfall is zero-inflated, bacteria counts are
log-distributed with censored non-detects, and detection rate is a bounded
proportion. Rank correlation is unbothered by all three. The p-value comes
from a Fisher z approximation, so no scipy dependency.

Two guards against the obvious ways this analysis could mislead:

  Diurnal confounding. The rip detector only sees daylight, and tide, sea
  breeze and air temperature all cycle daily. A raw correlation between
  detection rate and, say, temperature can be nothing but time of day. Every
  rip correlation is therefore reported twice: raw, and again after
  subtracting each hour-of-day's own mean from both sides. If a driver
  survives only in the raw column, it is a clock, not a cause.

  Nested predictors. rain_24h, rain_48h and rain_72h are sums of each other
  and will always look similar. They are ranked together and reported as one
  family, not as three independent findings.

    python analyze_drivers.py
    python analyze_drivers.py --target rip
    python analyze_drivers.py --target wq
"""

import env  # noqa: F401  -- loads .env into os.environ

import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd

DATA_DIR = "data"
SITES_CSV = "candidate_sites_ranked.csv"

# Below this many paired observations a correlation is not reported at all.
MIN_N = 30
# Fraction of rip hours that must find matching conditions before the join is
# worth analysing. Below this the tables are arithmetic on a handful of rows.
MIN_JOIN_FRACTION = 0.2
# Correlations among these are structural, not findings.
RAIN_FAMILY = ("precipitation", "precip_mm", "rain_24h_mm", "rain_48h_mm",
               "rain_72h_mm")

# Compass bearing of the outward shore normal -- the direction you face when
# standing on the beach looking out to sea. Used to turn wind and swell
# direction into an onshore component, which is the physically meaningful
# quantity for a rip. Walton Lighthouse sits on the north shore of Monterey
# Bay and faces south. THIS IS AN ASSUMPTION, not a measurement: change it if
# you know the beach better, and treat any onshore/offshore result as
# conditional on it.
SHORE_NORMAL_DEG = {
    "Walton Lighthouse, Santa Cruz, CA": 180.0,
    "Santa Cruz Wharf at Santa Cruz": 180.0,
    "Capitola Wharf": 180.0,
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def spearman(x, y, min_n=None):
    """(rho, n, p) rank correlation. NaNs are dropped pairwise."""
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                          "y": pd.to_numeric(y, errors="coerce")}).dropna()
    n = len(frame)
    if n < (MIN_N if min_n is None else min_n):
        return np.nan, n, np.nan
    rx, ry = frame["x"].rank(), frame["y"].rank()
    if rx.nunique() < 2 or ry.nunique() < 2:
        return np.nan, n, np.nan
    rho = float(rx.corr(ry))
    if not np.isfinite(rho) or abs(rho) >= 1:
        return rho, n, 0.0
    # Fisher z: atanh(rho) is approximately normal with sd 1/sqrt(n-3).
    z = math.atanh(rho) * math.sqrt(n - 3)
    p = math.erfc(abs(z) / math.sqrt(2))
    return rho, n, p


def demean_by(series, key):
    """Subtract each group's own mean, to remove a cyclical confound."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric - numeric.groupby(key).transform("mean")


def standardized_ols(frame, target, predictors):
    """Standardized coefficients, so magnitudes are comparable.

    Reported alongside the correlations because a predictor can correlate
    strongly on its own and contribute nothing once the others are present --
    which is exactly what happens to variables that are really proxies for
    something else in the model.
    """
    usable = [p for p in predictors if p in frame.columns]
    subset = frame[[target] + usable].apply(pd.to_numeric, errors="coerce").dropna()
    if len(subset) < MIN_N or len(usable) < 2:
        return None
    y = subset[target].to_numpy(float)
    X = subset[usable].to_numpy(float)

    keep = X.std(axis=0) > 0
    if not keep.any():
        return None
    usable = [name for name, k in zip(usable, keep) if k]
    X = X[:, keep]
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Drop columns that duplicate an earlier one. precip_mm and rain_24h_mm are
    # the same series whenever RAIN_INCLUDE_SAME_DAY is on -- a 1-day rolling
    # sum IS the daily total -- which makes the design matrix exactly singular.
    # Left in, lstsq returns a minimum-norm solution that splits one variable's
    # effect arbitrarily across the copies and reports both halves as findings.
    unique, dropped = [], []
    for index, name in enumerate(usable):
        twin = next((usable[j] for j in unique
                     if abs(float(np.corrcoef(X[:, index], X[:, j])[0, 1])) > 0.9999),
                    None)
        if twin is None:
            unique.append(index)
        else:
            dropped.append((name, twin))
    if dropped:
        for name, twin in dropped:
            print(f"  dropping {name}: identical to {twin}")
        usable = [usable[i] for i in unique]
        X = X[:, unique]

    if not np.isfinite(X).all():
        return None
    design = np.column_stack([np.ones(len(X)), X])
    condition = float(np.linalg.cond(design))
    beta, *_ = np.linalg.lstsq(design, y_scaled := (y - y.mean()) / (y.std() or 1.0),
                               rcond=None)
    fitted = design @ beta
    if not np.isfinite(fitted).all():
        return {"names": usable, "beta": None, "r2": np.nan, "n": len(subset),
                "condition": condition}
    ss_res = float(((y_scaled - fitted) ** 2).sum())
    ss_tot = float(((y_scaled - y_scaled.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    return {"names": usable, "beta": beta[1:], "r2": r2, "n": len(subset),
            "condition": condition}


# Above this, the design matrix is numerically singular and individual
# coefficients are arbitrary. Printing them anyway is how collinearity turns
# into a confident, invented finding -- so they are withheld instead.
MAX_REPORTABLE_CONDITION = 1e6


def report_regression(frame, target, predictors):
    """Print standardized coefficients, or say why they cannot be trusted."""
    print("\n--- standardized regression (all predictors at once) ---")
    fit = standardized_ols(frame, target, predictors)
    if fit is None:
        print("  too few complete rows across all predictors to fit")
        return None
    if fit["beta"] is None or fit["condition"] > MAX_REPORTABLE_CONDITION:
        print(f"  condition number {fit['condition']:.3g} — the predictors are")
        print("  collinear enough that individual coefficients are arbitrary.")
        print("  Withholding them; read the rank correlations instead.")
        return fit
    table = pd.DataFrame({"predictor": fit["names"], "beta": fit["beta"]})
    table = table.reindex(table["beta"].abs().sort_values(ascending=False).index)
    print(table.round(4).to_string(index=False))
    print(f"  n={fit['n']}  R2={fit['r2']:.3f}  condition={fit['condition']:.1f}")
    return fit


def report_correlations(frame, target, predictors, control=None, title="", top=None):
    """Print a ranked table. `control` is a series to demean both sides by."""
    rows = []
    for name in predictors:
        if name not in frame.columns:
            continue
        rho, n, p = spearman(frame[name], frame[target])
        row = {"predictor": name, "rho": rho, "n": n, "p": p}
        if control is not None:
            ctrl_rho, ctrl_n, ctrl_p = spearman(
                demean_by(frame[name], control), demean_by(frame[target], control))
            row["rho_ctrl"] = ctrl_rho
            row["p_ctrl"] = ctrl_p
        rows.append(row)
    if not rows:
        print("  no usable predictors")
        return None
    table = pd.DataFrame(rows)
    key = "rho_ctrl" if "rho_ctrl" in table.columns else "rho"
    table = table.reindex(table[key].abs().sort_values(ascending=False).index)
    if title:
        print(f"\n{title}")
    shown = table if top is None else table.head(top)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(shown.round(4).to_string(index=False))
    if top is not None and len(table) > top:
        print(f"  ({len(table) - top} weaker predictors not shown)")
    weak = table[key].abs().max()
    if pd.isna(weak):
        print("  every correlation was under-powered; treat none of this as a finding")
    elif weak < 0.15:
        print(f"  strongest |rho| is {weak:.2f} — nothing here is a strong driver")
    return table


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_sites():
    if not os.path.exists(SITES_CSV):
        sys.exit(f"{SITES_CSV} not found — run find_candidate_sites.py first.")
    sites = pd.read_csv(SITES_CSV)
    if "has_all_four" in sites.columns:
        sites = sites[sites["has_all_four"]]
    return sites.reset_index(drop=True)


def grid_slug(name):
    """The slug pull_gridded_weather.py writes."""
    return "".join(c if c.isalnum() else "_" for c in str(name))[:48]


def rip_slug(name):
    """The slug pull_rip_detection.py writes."""
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def read_csv(path, **kwargs):
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path, **kwargs)
    return None if frame.empty else frame


def to_hour(series):
    return pd.to_datetime(series, utc=True, errors="coerce").dt.floor("h")


def load_gridded(camera_name):
    frame = read_csv(f"{DATA_DIR}/gridded_{grid_slug(camera_name)}.csv")
    if frame is None:
        return None
    time_col = "time" if "time" in frame.columns else frame.columns[0]
    frame["hour"] = to_hour(frame[time_col])
    return frame.drop(columns=[time_col]).groupby("hour").mean(numeric_only=True)


def load_buoy(buoy_id):
    frame = read_csv(f"{DATA_DIR}/buoy_{buoy_id}.csv", index_col=0)
    if frame is None:
        return None
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame[frame.index.notna()]
    frame.index = frame.index.floor("h")
    frame.index.name = "hour"
    return frame.groupby("hour").mean(numeric_only=True)


def pick_coops(prefix, lat, lon):
    """The nearest CO-OPS file of this kind, matched by station coordinates.

    pull_observations picks a station at runtime and does not record which,
    so the mapping is re-derived here from the station list rather than
    guessed from the filename.
    """
    paths = sorted(glob.glob(f"{DATA_DIR}/{prefix}_*.csv"))
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    try:
        import pull_observations as po
        stations = po.coops_stations("waterlevels" if prefix != "wind" else "met")
    except Exception as exc:  # noqa: BLE001 -- offline is a normal case here
        print(f"    cannot resolve which {prefix} station serves this site ({exc});"
              f" using {os.path.basename(paths[0])}")
        return paths[0]
    have = {os.path.basename(p).split("_", 1)[1][:-4]: p for p in paths}
    best, best_km = None, None
    # coops_stations returns a DataFrame, so iterate rows, not the object.
    for station in stations.itertuples():
        path = have.get(str(station.station_id))
        if path is None:
            continue
        km = po.f.haversine_km(lat, lon, station.lat, station.lon)
        if best_km is None or km < best_km:
            best, best_km = path, km
    if best is not None:
        print(f"    {prefix}: {os.path.basename(best)} is nearest ({best_km:.1f} km)")
    return best or paths[0]


def load_coops(prefix, lat, lon):
    path = pick_coops(prefix, lat, lon)
    if path is None:
        return None
    frame = read_csv(path)
    if frame is None or "time" not in frame.columns:
        return None
    frame["hour"] = to_hour(frame["time"])
    numeric = frame.drop(columns=["time"]).groupby("hour").mean(numeric_only=True)
    return numeric


def angular_component(direction_deg, normal_deg):
    """+1 when the direction is straight onshore, -1 straight offshore."""
    delta = pd.to_numeric(direction_deg, errors="coerce") - normal_deg
    return np.cos(np.radians(delta))


# ---------------------------------------------------------------------------
# Rip
# ---------------------------------------------------------------------------

RIP_PREDICTORS = [
    "WVHT", "DPD", "APD", "swell_onshore", "WTMP",
    "level_m", "rate_m_per_hr", "abs_rate_m_per_hr",
    "wind_speed_10m", "wind_gusts_10m", "wind_onshore", "temperature_2m",
    "precipitation", "rain_24h_mm", "rain_48h_mm", "rain_72h_mm",
]


def assemble_rip(sites):
    # pull_rip_detection writes into data/rip_detection/, so a flat glob on
    # data/ finds nothing and the whole rip half silently reports "not pulled".
    paths = sorted(glob.glob(f"{DATA_DIR}/**/rip_*_hourly.csv", recursive=True))
    if not paths:
        print("No data/rip_*_hourly.csv — run pull_rip_detection.py --pull first.")
        return None, None
    frame = pd.read_csv(paths[0])
    frame["hour"] = to_hour(frame["hour"])
    stem = os.path.basename(paths[0])[len("rip_"):-len("_hourly.csv")]

    match = [s for _, s in sites.iterrows() if rip_slug(s["camera_name"]) == stem]
    if not match:
        print(f"  {stem} is not among the qualifying sites; cannot attach conditions.")
        return None, None
    site = match[0]
    name = site["camera_name"]
    print(f"site: {name}")
    print(f"  {len(frame)} hours of rip output, "
          f"{frame['hour'].min()} to {frame['hour'].max()}")

    merged = frame.set_index("hour")
    thin = []
    for label, part in [
            ("gridded weather", load_gridded(name)),
            ("buoy", load_buoy(site["buoy_id"]) if pd.notna(site.get("buoy_id")) else None),
            ("tide", load_coops("tide", site["lat"], site["lon"])),
            ("water temp", load_coops("watertemp", site["lat"], site["lon"])),
    ]:
        if part is None:
            print(f"  {label}: no file, skipped")
            continue
        overlap = merged.index.intersection(part.index)
        fraction = len(overlap) / max(len(merged), 1)
        flag = "" if fraction >= MIN_JOIN_FRACTION else "   <- barely overlaps"
        print(f"  {label}: {len(part)} hours, {len(overlap)} overlapping"
              f" ({100 * fraction:.0f}%){flag}")
        if fraction < MIN_JOIN_FRACTION:
            thin.append((label, len(overlap), part.index.min(), part.index.max()))
        merged = merged.join(part, how="left", rsuffix=f"_{label[:4]}")

    if thin:
        print("\n  *** THE WINDOWS DO NOT LINE UP ***")
        print(f"  rip hours run {merged.index.min():%Y-%m-%d}"
              f" to {merged.index.max():%Y-%m-%d}, but:")
        for label, count, first, last in thin:
            print(f"    {label} covers {first:%Y-%m-%d} to {last:%Y-%m-%d}"
                  f" — only {count} hours in common")
        print("  Correlations below are computed on those few hours and mean")
        print("  nothing. Re-pull the rip range over the observation window:")
        print("    python pull_rip_detection.py --pull --match-observations")

    normal = SHORE_NORMAL_DEG.get(name)
    if normal is None:
        print("  no shore normal configured; onshore components skipped")
    else:
        print(f"  shore normal assumed {normal:.0f} deg (see SHORE_NORMAL_DEG)")
        if "MWD" in merged.columns:
            merged["swell_onshore"] = angular_component(merged["MWD"], normal)
        if "wind_direction_10m" in merged.columns:
            merged["wind_onshore"] = angular_component(
                merged["wind_direction_10m"], normal)
    if "rate_m_per_hr" in merged.columns:
        merged["abs_rate_m_per_hr"] = pd.to_numeric(
            merged["rate_m_per_hr"], errors="coerce").abs()

    merged = merged.reset_index()
    merged["hour_of_day"] = merged["hour"].dt.hour
    return merged, name


def apply_coverage(frame, stem):
    """Add hours the camera was looking but the detector never fired.

    Without this the rip feed has no negatives, because it publishes an
    element only on a detection. With it, an hour holding images and no
    detection is an OBSERVED zero, and an hour with no images at all stays
    absent -- which is the distinction that makes presence/absence analysable
    at all. Assuming every missing hour is a zero would instead score every
    camera outage as "no rip", and outages cluster in bad weather, which is
    correlated with the drivers being tested.
    """
    paths = glob.glob(f"{DATA_DIR}/**/coverage_{stem}_hourly.csv", recursive=True)
    if not paths:
        print("  no coverage file — hours without a detection stay UNKNOWN, not"
              " zero.\n    run: pull_rip_detection.py --coverage --start ... --end ...")
        return frame, False
    coverage = pd.read_csv(paths[0])
    coverage["hour"] = to_hour(coverage["hour"])
    before = len(frame)

    merged = coverage.merge(frame, on="hour", how="left")
    for column, fill in (("frames", 0), ("frames_with_detection", 0),
                         ("detections", 0)):
        if column in merged.columns:
            merged[column] = merged[column].fillna(0)
    # Rate against images examined, not against elements published.
    merged["detection_rate"] = (merged["frames_with_detection"]
                                / merged["images"].replace(0, np.nan))
    merged["hour_of_day"] = merged["hour"].dt.hour
    zeros = int((merged["frames_with_detection"] == 0).sum())
    print(f"  coverage: {len(merged)} hours with imagery "
          f"({before} had detections, {zeros} are observed zeros)")
    return merged, True


# Candidate targets, in preference order. Whichever have real variance get
# analysed; a constant one is reported as constant rather than correlated.
RIP_TARGETS = ["detection_rate", "detections", "score_max", "bbox_area_max"]


def analyze_rip(sites):
    frame, name = assemble_rip(sites)
    if frame is None:
        return
    frame, has_coverage = apply_coverage(frame, rip_slug(name))
    observed = (frame.copy() if has_coverage
                else frame[frame.get("frames", 0) > 0].copy())
    print(f"\n{len(observed)} hours analysed")

    hours = sorted(observed["hour_of_day"].unique())
    print(f"  frames occur in hours {hours[0]}-{hours[-1]} UTC only "
          f"({len(hours)} distinct hours of day)")

    if "detection_rate" in observed.columns and not has_coverage:
        share = float((observed["detection_rate"] >= 1.0).mean())
        if share > 0.999:
            print("\n  EVERY hour with frames has detection_rate 1.0.")
            print("  This feed publishes an element only when the detector fires,")
            print("  so it contains no negatives. Absence of a file means either")
            print("  'no rip' or 'no image', and nothing here can tell them apart.")
            print("  Presence/absence is therefore not analysable; what remains is")
            print("  how OFTEN it fires and how confident it is.")

    usable = []
    for target in RIP_TARGETS:
        if target not in observed.columns:
            continue
        series = pd.to_numeric(observed[target], errors="coerce")
        if series.nunique(dropna=True) < 3:
            print(f"  {target}: constant ({series.dropna().unique()[:3]}), skipped")
            continue
        usable.append(target)
    if not usable:
        print("  no target with any variance; nothing to correlate")
        return

    for target in usable:
        report_correlations(
            observed, target, RIP_PREDICTORS, control=observed["hour_of_day"],
            title=f"=== WHAT TRACKS {target} at {name} ===\n"
                  "rho = raw rank correlation; rho_ctrl = after removing "
                  "hour-of-day means from both sides")
        report_regression(observed, target, RIP_PREDICTORS)

    print("\nCaveats specific to this target:")
    print("  - One camera. Nothing here generalizes to another beach.")
    print("  - The score is a YOLOv8 model's confidence, not a verified rip.")
    print("    A driver of the DETECTOR (glare, swell texture, contrast) is")
    print("    indistinguishable here from a driver of the rip.")
    print("  - Walton has no observed wind, so every wind column is ERA5 grid,")
    print("    which compare_wind_sources.py could not validate at this site.")
    print("  - Daylight only, so any driver with a daily cycle is confounded;")
    print("    that is what the rho_ctrl column is for.")
    if not has_coverage:
        print("  - No stills denominator, so there are no observed zeros and")
        print("    detection_rate carries no information. Run --coverage.")


# ---------------------------------------------------------------------------
# Water quality
# ---------------------------------------------------------------------------

# How each site sits relative to open ocean. This decides where a tide-level
# effect is even plausible: inside an enclosed bay, water level tracks
# flushing and the arrival of bay water at the shoreline, whereas on open
# coast it is mostly just the astronomical tide. Classified by hand from the
# geography, and wrong classifications will show up as an effect appearing
# where the label says it should not.
SITE_SETTING = {
    "Sausalito - Galilee Harbor": "enclosed bay",
    "Stinson Beach (northwest view)": "open coast",
    "Capitola Wharf": "open embayment",
    "Santa Cruz Wharf at Santa Cruz, CA": "open embayment",
    "Walton Lighthouse, Santa Cruz, CA": "open embayment",
    "San Elijo State Beach, CA": "open coast",
    "Carpinteria State Beach, CA": "open coast",
}
# Reported with its n even below MIN_N, flagged rather than hidden: for a
# single pre-specified predictor, seeing an underpowered estimate is more
# informative than a blank.
FOCUSED_MIN_N = 15

WQ_PREDICTORS = ["rain_24h_mm", "rain_48h_mm", "rain_72h_mm", "precip_mm",
                 "level_m", "rate_m_per_hr", "WVHT", "DPD", "WTMP",
                 "wind_speed_10m", "temperature_2m"]

# Single-sample California standards, in MPN or CFU per 100 mL. Used only to
# report an exceedance rate alongside the continuous result.
WQ_THRESHOLDS = {"ENT": 104, "ECOLI": 235, "TOTAL": 10000, "FECAL": 400}


def analyte_key(name):
    text = str(name).upper()
    if "ENTERO" in text:
        return "ENT"
    if "E. COLI" in text or "E.COLI" in text or "ESCHERICHIA" in text:
        return "ECOLI"
    if "FECAL" in text:
        return "FECAL"
    if "TOTAL" in text and "COLIFORM" in text:
        return "TOTAL"
    return None


def load_water_quality():
    frame = read_csv(f"{DATA_DIR}/water_quality.csv")
    if frame is None:
        print("No data/water_quality.csv — run pull_observations.py first.")
        return None
    frame["date"] = pd.to_datetime(frame["SampleDate"], errors="coerce").dt.normalize()
    frame["value"] = pd.to_numeric(frame["Result"], errors="coerce")
    dropped = int(frame["value"].isna().sum())
    if dropped:
        print(f"  {dropped}/{len(frame)} results are not numeric "
              "(non-detects and qualifiers) and are excluded")
    frame = frame.dropna(subset=["date", "value"])
    frame["group"] = frame["Analyte"].map(analyte_key)
    # log10 because bacteria counts span orders of magnitude; +1 keeps zeros.
    frame["log_value"] = np.log10(frame["value"].clip(lower=0) + 1)
    return frame


def daily_conditions(site):
    """One row per day of whatever conditions exist for this site."""
    parts = []
    precip_id = site.get("precip_station_id")
    if pd.notna(precip_id):
        gauge = read_csv(f"{DATA_DIR}/precip_{str(precip_id).replace(':', '_')}.csv")
        if gauge is not None:
            gauge["date"] = pd.to_datetime(gauge["date"], errors="coerce").dt.normalize()
            parts.append(gauge.set_index("date"))

    for loader in (lambda: load_gridded(site["camera_name"]),
                   lambda: load_buoy(site["buoy_id"]) if pd.notna(site.get("buoy_id")) else None,
                   lambda: load_coops("tide", site["lat"], site["lon"])):
        hourly = loader()
        if hourly is None or hourly.empty:
            continue
        daily = hourly.copy()
        daily.index = daily.index.tz_convert(None).normalize()
        daily.index.name = "date"
        parts.append(daily.groupby("date").mean(numeric_only=True))

    if not parts:
        return None
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.join(part, how="outer", rsuffix="_x")
    return merged


def map_stations(sites, samples):
    """StationCode -> camera_name, by the tested join, else by station name."""
    try:
        import pull_observations as po
        codes = po.station_code_map(sites["wq_station_id"].dropna())
        if codes:
            by_id = {po.canonical_station_id(s["wq_station_id"]): s["camera_name"]
                     for _, s in sites.iterrows() if pd.notna(s.get("wq_station_id"))}
            mapping = {code: by_id[sid] for sid, code in codes.items() if sid in by_id}
            if mapping:
                print(f"  mapped {len(mapping)} station codes via the CKAN join")
                return mapping
    except Exception as exc:  # noqa: BLE001 -- fall through to names
        print(f"  CKAN join unavailable ({exc}); matching on station name instead")

    names = {str(s["wq_station_name"]).strip().lower(): s["camera_name"]
             for _, s in sites.iterrows() if pd.notna(s.get("wq_station_name"))}
    mapping = {}
    for code, name in samples[["StationCode", "StationName"]].drop_duplicates().values:
        key = str(name).strip().lower()
        if key in names:
            mapping[str(code)] = names[key]
    print(f"  mapped {len(mapping)} station codes by name")
    return mapping


def focus_tide(combined):
    """Per-site tide-level correlation, the one predictor asked about.

    Pooled across sites, a tide effect is ambiguous: sites differ in both
    their bacteria levels and their tide gauge, so a between-site difference
    masquerades as a tide relationship. Split by site, that route is closed —
    each row uses one beach's own variation against one gauge.

    The setting column is the prior. Inside an enclosed bay, water level
    plausibly tracks flushing and the arrival of bay water at the shoreline.
    On open coast it is mostly the astronomical tide, and a strong effect
    there deserves more suspicion than confirmation.
    """
    print("\n" + "=" * 70)
    print("TIDE LEVEL vs BACTERIA, PER SITE")
    print("=" * 70)
    rows = []
    for (key, site), group in combined.groupby(["group", "site"]):
        if "level_m" not in group.columns:
            continue
        rho, n, p = spearman(group["level_m"], group["log_value"],
                             min_n=FOCUSED_MIN_N)
        rows.append({"analyte": key, "site": site,
                     "setting": SITE_SETTING.get(site, "unclassified"),
                     "rho": rho, "n": n, "p": p,
                     "powered": "" if n >= MIN_N else "underpowered"})
    if not rows:
        print("  no tide data joined to any site")
        return None
    table = pd.DataFrame(rows).sort_values(["analyte", "rho"])
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(4).to_string(index=False))

    solid = table[(table["n"] >= MIN_N) & table["rho"].notna()]
    if solid.empty:
        print("\n  No site has enough samples to judge the tide effect on its")
        print("  own. The pooled result stands unreplicated — do not rely on it.")
        return table
    negative = solid[solid["rho"] < -0.15]
    print(f"\n  {len(negative)} of {len(solid)} adequately powered site-analyte"
          f" pairs show a negative tide effect stronger than -0.15.")
    by_setting = solid.groupby("setting")["rho"].agg(["mean", "count"])
    print(by_setting.round(3).to_string())
    if not negative.empty:
        settings = sorted(set(negative["setting"]))
        print(f"  those pairs sit in: {', '.join(settings)}")
        if settings == ["open coast"]:
            print("  Note: open coast only. Water level there is mostly the")
            print("  astronomical tide, so a flushing story does not fit.")
    return table


def analyze_wq(sites):
    samples = load_water_quality()
    if samples is None:
        return
    print(f"\n{len(samples)} numeric results, "
          f"{samples['date'].min():%Y-%m-%d} to {samples['date'].max():%Y-%m-%d}")
    counts = samples["Analyte"].value_counts()
    print("\nanalytes:")
    print(counts.to_string())

    mapping = map_stations(sites, samples)
    samples["camera_name"] = samples["StationCode"].astype(str).map(mapping)
    unmapped = int(samples["camera_name"].isna().sum())
    if unmapped:
        print(f"  {unmapped} results belong to no qualifying site and are dropped")
    samples = samples.dropna(subset=["camera_name"])
    if samples.empty:
        print("Nothing left after mapping — cannot attach conditions.")
        return

    pieces = []
    for name, group in samples.groupby("camera_name"):
        site = sites[sites["camera_name"] == name].iloc[0]
        conditions = daily_conditions(site)
        if conditions is None:
            print(f"  {name}: no condition files, skipped")
            continue
        joined = group.merge(conditions.reset_index(), on="date", how="left")
        joined["site"] = name
        pieces.append(joined)
        print(f"  {name}: {len(group)} results joined to conditions")
    if not pieces:
        print("No site had both samples and conditions.")
        return
    combined = pd.concat(pieces, ignore_index=True)

    focus_tide(combined)

    for key in sorted(set(combined["group"].dropna())):
        subset = combined[combined["group"] == key]
        if len(subset) < MIN_N:
            print(f"\n{key}: only {len(subset)} results, below the "
                  f"{MIN_N}-sample floor — not reported")
            continue
        threshold = WQ_THRESHOLDS.get(key)
        exceed = ((subset["value"] > threshold).mean() * 100
                  if threshold else float("nan"))
        print(f"\n=== WHAT TRACKS {key} (log10 count) ===")
        print(f"{len(subset)} samples across {subset['site'].nunique()} sites; "
              f"median {subset['value'].median():.0f}"
              + (f", {exceed:.1f}% over the {threshold} single-sample standard"
                 if threshold else ""))
        # California rainfall is strongly seasonal and beach sampling is
        # concentrated in the dry swim season, so month is a confound of exactly
        # the same shape as hour-of-day is for the rip camera: rain and bacteria
        # can correlate, in either direction, purely through the calendar.
        report_correlations(subset, "log_value", WQ_PREDICTORS,
                            control=subset["date"].dt.month,
                            title="POOLED — rho = raw; rho_ctrl = after removing "
                                  "per-month means from both sides")
        if subset["site"].nunique() > 1:
            report_correlations(
                subset, "log_value", WQ_PREDICTORS, control=subset["site"],
                title="POOLED, WITHIN SITE — rho_ctrl removes each site's own "
                      "mean, so only variation inside a beach counts.\n"
                      "A predictor that survives raw but not here was really "
                      "telling you which beach the sample came from.")
        report_regression(subset, "log_value", WQ_PREDICTORS)

        for site, group in sorted(subset.groupby("site")):
            setting = SITE_SETTING.get(site, "unclassified")
            if len(group) < MIN_N:
                print(f"\n  {site} [{setting}]: {len(group)} samples, "
                      f"below the {MIN_N} floor — not reported")
                continue
            report_correlations(
                group, "log_value", WQ_PREDICTORS,
                control=group["date"].dt.month, top=6,
                title=f"--- {site} [{setting}] — {len(group)} samples, "
                      f"median {group['value'].median():.0f} ---")

    print("\nCaveats specific to this target:")
    print("  - Samples are not random: agencies sample in swim season, on")
    print("    schedule, and sometimes after known spills. Rain-driven days")
    print("    can be over- or under-represented, either way biasing rho.")
    print("  - Non-detects were dropped, not imputed. If low results are")
    print("    disproportionately censored, the low end is under-represented.")
    print("  - rain_24/48/72h are nested sums; treat them as one family.")
    print("  - Rainfall in California is seasonal and sampling is concentrated")
    print("    in the dry swim season, so judge on rho_ctrl, not rho.")
    print("  - Per-site tables are the ones to trust. A pooled correlation can")
    print("    be driven entirely by which beach a sample came from.")
    print("  - Conditions are daily means, because SampleDate carries no time.")
    print("    A tide or wind value on the sample day is not the value at the")
    print("    moment of sampling.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=["rip", "wq", "both"], default="both")
    args = ap.parse_args()
    sites = load_sites()
    print(f"{len(sites)} qualifying sites\n")
    if args.target in ("rip", "both"):
        print("=" * 70)
        print("RIP DETECTION")
        print("=" * 70)
        analyze_rip(sites)
    if args.target in ("wq", "both"):
        print("\n" + "=" * 70)
        print("WATER QUALITY")
        print("=" * 70)
        analyze_wq(sites)


if __name__ == "__main__":
    main()
