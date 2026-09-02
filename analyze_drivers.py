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
CANDIDATES_CSV = "camera_candidates.csv"
RIPAID_SITES_CSV = "ripaid_sites.csv"

# How far a CO-OPS file may be from a site before it is treated as absent.
# pull_observations searches out to 50 km; the slack here covers a station
# that moved slightly, not a different coastline.
MAX_STATION_KM = 75

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
    # The Virginia Beach oceanfront runs roughly north-south and faces east.
    # Same status as the others: an assumption from a map, not a survey.
    "Hampton Inn Oceanfront South at Virginia Beach": 90.0,
    # RipAID / SIRENA. Cala Millor is on Mallorca's east coast; Son Bou runs
    # east-west along Menorca's south coast. Both read off a map, like the rest.
    "Cala Millor": 90.0,
    "Son Bou": 180.0,
}

# Open-Meteo Marine columns, as returned by pull_site_observations.py. These
# are reanalysis, not measurements. The distinction is carried by the naming
# convention this project already uses everywhere: ALL-CAPS columns are
# observed (NDBC stdmet), lower_snake_case columns are model output (ERA5,
# Marine). Nothing renames one into the other, so a regression can never
# average a modelled wave height together with a measured one without that
# being visible in the predictor names.
MARINE_COLUMNS = [
    "wave_height", "wave_period", "wave_direction",
    "wind_wave_height", "wind_wave_period",
    "swell_wave_height", "swell_wave_period", "swell_wave_direction",
]


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

    # dropna removes NaN but not inf, and an inf anywhere makes lstsq return
    # a non-finite solution while the condition number still looks healthy.
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if finite.sum() < MIN_N:
        return None
    if not finite.all():
        print(f"  dropping {int((~finite).sum())} rows with non-finite values")
        X, y = X[finite], y[finite]
    used = int(finite.sum())
    design = np.column_stack([np.ones(len(X)), X])
    condition = float(np.linalg.cond(design))
    y_scaled = (y - y.mean()) / (y.std() or 1.0)
    with np.errstate(all="ignore"):
        beta, *_ = np.linalg.lstsq(design, y_scaled, rcond=None)
        fitted = design @ beta
    if not np.isfinite(fitted).all():
        return {"names": usable, "beta": None, "r2": np.nan, "n": used,
                "condition": condition}
    ss_res = float(((y_scaled - fitted) ** 2).sum())
    ss_tot = float(((y_scaled - y_scaled.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    return {"names": usable, "beta": beta[1:], "r2": r2, "n": used,
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


def variance_explained(series, key):
    """Share of a series' variance that group means account for (eta squared).

    Answers "how much of this target is just the calendar?" directly, rather
    than leaving it to be inferred from how the predictors behaved. 1.0 would
    mean the group entirely determines the value; 0.0 that knowing the group
    tells you nothing.
    """
    values = pd.to_numeric(series, errors="coerce")
    frame = pd.DataFrame({"v": values, "k": key}).dropna()
    if len(frame) < MIN_N or frame["k"].nunique() < 2:
        return float("nan")
    total = float(frame["v"].var(ddof=0))
    if total <= 0:
        return float("nan")
    residual = frame["v"] - frame.groupby("k")["v"].transform("mean")
    eta_sq = 1.0 - float(residual.var(ddof=0)) / total

    # Group means fit noise, so eta squared is biased upward by roughly
    # (k-1)/(n-1) even when the grouping is meaningless. With 12 months that
    # is negligible; with 149 hour-by-month cells it is several points, which
    # would otherwise make the finer control look more explanatory than it is.
    groups = frame["k"].nunique()
    null = (groups - 1) / (len(frame) - 1)
    return (eta_sq - null) / (1 - null) if null < 1 else float("nan")


def report_variance_absorbed(frame, targets, controls):
    """How much of each target the control keys explain on their own."""
    rows = []
    for target in targets:
        if target not in frame.columns:
            continue
        row = {"target": target}
        for suffix, key in controls:
            row[f"var_{suffix}"] = variance_explained(frame[target], key)
        rows.append(row)
    if not rows:
        return None
    table = pd.DataFrame(rows)
    print("\n=== HOW MUCH OF EACH TARGET IS JUST THE CALENDAR ===")
    print("share of variance explained by the control key alone,")
    print("corrected for the upward bias that comes from fitting group means")
    print(table.round(3).to_string(index=False))
    return table


def report_correlations(frame, target, predictors, control=None, title="", top=None,
                        controls=None):
    """Print a ranked table, ranked by the last control applied.

    `controls` is an ordered list of (suffix, key) pairs, each demeaning both
    sides by that key's group means. They are independent views, not cumulative:
    a driver that holds up under the strictest one is the one to believe.
    """
    if controls is None:
        controls = [("ctrl", control)] if control is not None else []
    rows = []
    for name in predictors:
        if name not in frame.columns:
            continue
        rho, n, p = spearman(frame[name], frame[target])
        row = {"predictor": name, "rho": rho, "n": n, "p": p}
        for suffix, key in controls:
            c_rho, _, c_p = spearman(demean_by(frame[name], key),
                                     demean_by(frame[target], key))
            row[f"rho_{suffix}"] = c_rho
            row[f"p_{suffix}"] = c_p
        rows.append(row)
    if not rows:
        print("  no usable predictors")
        return None
    table = pd.DataFrame(rows)
    key = f"rho_{controls[-1][0]}" if controls else "rho"
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
    """Qualifying sites, from the California search and the national scan.

    candidate_sites_ranked.csv only ever held California. Anything found by
    scan_cameras.py -- Virginia Beach, Corolla -- is in camera_candidates.csv
    under different column names, so both are read and normalised. Without
    this a rip pull for a non-California camera lands on disk and then reports
    'not among the qualifying sites', which looks like a data problem and is
    really a bookkeeping one.
    """
    frames = []
    if os.path.exists(SITES_CSV):
        sites = pd.read_csv(SITES_CSV)
        if "has_all_four" in sites.columns:
            sites = sites[sites["has_all_four"]]
        frames.append(sites)

    if os.path.exists(CANDIDATES_CSV):
        national = pd.read_csv(CANDIDATES_CSV).rename(
            columns={"camera": "camera_name",
                     # scan_cameras writes precip_id; every reader downstream
                     # asks for precip_station_id, so a national site was
                     # quietly analysed with no rain gauge at all.
                     "precip_id": "precip_station_id",
                     "wq_id": "wq_station_id"})
        keep = [c for c in ("camera_name", "lat", "lon", "buoy_id", "tide_id",
                            "precip_station_id", "wq_station_id")
                if c in national.columns]
        frames.append(national[keep])

    frames.extend(_ripaid_rows())

    if not frames:
        sys.exit(f"Neither {SITES_CSV} nor {CANDIDATES_CSV} exists — run "
                 "find_candidate_sites.py or scan_cameras.py first.")

    sites = pd.concat(frames, ignore_index=True, sort=False)
    # The California file wins where a camera is in both: it carries the
    # precipitation and water-quality station ids the national scan does not.
    sites = sites.drop_duplicates(subset=["camera_name"], keep="first")
    for column in ("buoy_id", "tide_id"):
        if column in sites.columns:
            sites[column] = sites[column].map(_station_id)
    return sites.reset_index(drop=True)


def _station_id(value):
    """Station ids are labels that happen to look like numbers.

    Read on its own, a sites file of all-numeric buoy ids comes back as int64.
    Concatenated with a frame that has no buoy_id at all -- a RipAID site, say,
    which has no buoy -- the column is promoted to float64 and 46042 becomes
    46042.0. The loader then looks for buoy_46042.0.csv, finds nothing, and the
    site reports no wave data whatsoever: a missing-data conclusion produced
    entirely by a dtype. Rendering whole floats back to integer text fixes that
    and leaves alphanumeric ids like ptac1 alone.
    """
    if pd.isna(value):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _ripaid_rows():
    """RipAID sites, plus a row per camera that shares the site's weather.

    load_ripaid.py --by-camera writes rip_clm_s_01_hourly.csv and friends, but
    weather is pulled once per SITE (gridded_clm.csv), because five cameras
    looking at one beach share one grid cell. Each camera therefore carries a
    weather_name pointing back at its site; without it the camera-level tables
    would silently find no conditions and report nothing.
    """
    if not os.path.exists(RIPAID_SITES_CSV):
        return []
    frame = pd.read_csv(RIPAID_SITES_CSV)
    if "site" not in frame.columns:
        return []

    rows = []
    for _, row in frame.iterrows():
        site = str(row["site"])
        lat = pd.to_numeric(row.get("latitude"), errors="coerce")
        lon = pd.to_numeric(row.get("longitude"), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            print(f"  {RIPAID_SITES_CSV}: {site} has no coordinates yet; "
                  "fill latitude/longitude to analyse it")
            continue
        label = str(row.get("name") or site)
        rows.append({"camera_name": label, "lat": float(lat), "lon": float(lon),
                     "weather_name": label})
        # Cameras at this site, discovered from what load_ripaid actually wrote.
        # load_ripaid slugs camera names the same way pull_rip_detection does,
        # so clm_s_01 lands on disk as rip_clm-s-01_hourly.csv.
        for path in sorted(glob.glob(
                f"{DATA_DIR}/ripaid/rip_{rip_slug(site)}-*_hourly.csv")):
            stem = os.path.basename(path)[len("rip_"):-len("_hourly.csv")]
            rows.append({"camera_name": stem, "lat": float(lat), "lon": float(lon),
                         "weather_name": label})
    return [pd.DataFrame(rows)] if rows else []


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


def load_marine(camera_name):
    """Modelled waves, for sites where no buoy publishes them.

    Virginia Beach is the case this exists for: its nearest NDBC station
    (44064, 20 km) carries no standard meteorological feed, so WVHT, DPD and
    APD are simply absent. Without this the rip analysis there would run with
    no wave predictor at all, which for rips is most of the physics.
    """
    frame = read_csv(f"{DATA_DIR}/marine_{grid_slug(camera_name)}.csv")
    if frame is None:
        return None
    time_col = "time" if "time" in frame.columns else frame.columns[0]
    frame["hour"] = to_hour(frame[time_col])
    keep = ["hour"] + [c for c in MARINE_COLUMNS if c in frame.columns]
    return frame[keep].groupby("hour").mean(numeric_only=True)


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

    A distance cap is the whole point of the rewrite: without one this
    returned the nearest file ON DISK, and at Virginia Beach — where no water
    temperature was ever pulled — that was San Diego, 3,762 km away, joined
    at 100% overlap and reported as if it were local. A missing input has to
    stay missing; substituting the wrong ocean is worse than having none.
    """
    paths = sorted(glob.glob(f"{DATA_DIR}/{prefix}_*.csv"))
    if not paths:
        return None

    try:
        import pull_observations as po
        stations = po.coops_stations("waterlevels" if prefix != "wind" else "met")
    except Exception as exc:  # noqa: BLE001 -- offline is a normal case here
        # Previously this fell back to paths[0], which is the same silent
        # wrong-station failure by another route.
        if len(paths) == 1:
            print(f"    {prefix}: cannot verify distance ({exc}); using "
                  f"{os.path.basename(paths[0])} UNCHECKED")
            return paths[0]
        print(f"    {prefix}: cannot verify which of {len(paths)} files serves "
              f"this site ({exc}); skipping rather than guessing")
        return None

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

    if best is None:
        print(f"    {prefix}: none of the {len(paths)} files on disk could be "
              "matched to a station; skipping")
        return None
    if best_km > MAX_STATION_KM:
        print(f"    {prefix}: nearest file is {os.path.basename(best)} at "
              f"{best_km:.0f} km — beyond the {MAX_STATION_KM} km limit, so this "
              f"site has NO {prefix}. Run pull_site_observations.py for it.")
        return None
    print(f"    {prefix}: {os.path.basename(best)} is nearest ({best_km:.1f} km)")
    return best


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
    "wave_height", "wave_period", "wave_onshore_model",
    "swell_wave_height", "swell_wave_period", "swell_onshore_model",
    "level_m", "rate_m_per_hr", "abs_rate_m_per_hr",
    "wind_speed_10m", "wind_gusts_10m", "wind_onshore", "temperature_2m",
    "precipitation", "rain_24h_mm", "rain_48h_mm", "rain_72h_mm",
]


def assemble_rip(sites, want=None):
    # pull_rip_detection writes into data/rip_detection/, so a flat glob on
    # data/ finds nothing and the whole rip half silently reports "not pulled".
    paths = sorted(glob.glob(f"{DATA_DIR}/**/rip_*_hourly.csv", recursive=True))
    if not paths:
        print("No data/rip_*_hourly.csv — run pull_rip_detection.py --pull first.")
        return None, None

    stems = [os.path.basename(p)[len("rip_"):-len("_hourly.csv")] for p in paths]
    if want:
        hits = [(p, st) for p, st in zip(paths, stems)
                if want.lower().replace(" ", "-") in st or want.lower() in st]
        if not hits:
            print(f"No rip table matching {want!r}. Available:")
            for stem in stems:
                print(f"  {stem}")
            return None, None
        path, stem = hits[0]
    else:
        # Taking paths[0] silently analysed whichever site sorted first and
        # never mentioned the others, which is the kind of thing that gets
        # noticed only after the conclusions are written down.
        if len(paths) > 1:
            print(f"{len(paths)} rip tables on disk; analysing the first. "
                  "Use --site to pick another:")
            for stem in stems:
                print(f"  {stem}")
        path, stem = paths[0], stems[0]

    frame = pd.read_csv(path)
    frame["hour"] = to_hour(frame["hour"])

    match = [s for _, s in sites.iterrows() if rip_slug(s["camera_name"]) == stem]
    if not match:
        print(f"  {stem} is not among the qualifying sites; cannot attach conditions.")
        print("  Sites known here:")
        for _, s in sites.iterrows():
            print(f"    {rip_slug(s['camera_name'])}")
        return None, None
    site = match[0]
    name = site["camera_name"]
    # Several cameras can share one weather pull; see _ripaid_rows.
    weather = site.get("weather_name")
    weather = name if not isinstance(weather, str) or not weather else weather
    print(f"site: {name}")
    if weather != name:
        print(f"  conditions come from the site pull for {weather!r}")
    print(f"  {len(frame)} hours of rip output, "
          f"{frame['hour'].min()} to {frame['hour'].max()}")

    merged = frame.set_index("hour")
    thin = []
    for label, part in [
            ("gridded weather", load_gridded(weather)),
            ("marine waves (model)", load_marine(weather)),
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

    # RipAID cameras are named clm-s-01, not "Cala Millor", so the shore
    # normal is configured against the site they share, not each viewpoint.
    normal = SHORE_NORMAL_DEG.get(name)
    if normal is None:
        normal = SHORE_NORMAL_DEG.get(weather)
    if normal is None:
        print("  no shore normal configured; onshore components skipped")
    else:
        print(f"  shore normal assumed {normal:.0f} deg (see SHORE_NORMAL_DEG)")
        if "MWD" in merged.columns:
            merged["swell_onshore"] = angular_component(merged["MWD"], normal)
        # Kept under separate names from the observed swell_onshore so the two
        # never silently substitute for one another in a table of results.
        if "swell_wave_direction" in merged.columns:
            merged["swell_onshore_model"] = angular_component(
                merged["swell_wave_direction"], normal)
        if "wave_direction" in merged.columns:
            merged["wave_onshore_model"] = angular_component(
                merged["wave_direction"], normal)
        if "wind_direction_10m" in merged.columns:
            merged["wind_onshore"] = angular_component(
                merged["wind_direction_10m"], normal)
        if "rip_axis_deg" in merged.columns:
            merged["rip_axis_offset_deg"] = axial_offset(
                merged["rip_axis_deg"], normal)
    if "rate_m_per_hr" in merged.columns:
        merged["abs_rate_m_per_hr"] = pd.to_numeric(
            merged["rate_m_per_hr"], errors="coerce").abs()

    merged = merged.reset_index()
    merged["hour_of_day"] = merged["hour"].dt.hour
    return merged, name


def axial_offset(angles, normal):
    """How far a rip's axis lies from straight offshore, in degrees 0-90.

    rip_axis_deg is an orientation, not a bearing: a rip drawn at 10 deg and
    one at 190 deg lie along the same line. Correlating the raw angle would
    put those two at opposite ends of the scale and produce noise. Folding the
    difference from the shore normal onto 0-90 gives a quantity that actually
    means something -- 0 is a rip running straight out to sea, 90 is one lying
    along the beach -- and that a rank correlation can be run on.
    """
    delta = (pd.to_numeric(angles, errors="coerce") - normal).abs() % 180.0
    return delta.where(delta <= 90.0, 180.0 - delta)


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
# doubt_rate exists only for RipAID: the share of frames where a human marked
# uncertainty rather than a rip. It is a target in its own right — what makes
# a rip hard to call is a different question from what makes one happen, and
# no detector confidence can answer it.
RIP_TARGETS = ["detection_rate", "detections", "score_max", "bbox_area_max",
               "doubt_rate", "rip_axis_offset_deg"]

# Targets that answer "was a rip there at all". Where a person chose which
# frames to keep, these measure the choosing as much as the ocean, so
# --positives-only drops them rather than reporting a number nobody should use.
PRESENCE_TARGETS = ("detection_rate", "detections", "doubt_rate")


def analyze_rip(sites, want=None, positives_only=False):
    frame, name = assemble_rip(sites, want=want)
    if frame is None:
        return
    frame, has_coverage = apply_coverage(frame, rip_slug(name))
    observed = (frame.copy() if has_coverage
                else frame[frame.get("frames", 0) > 0].copy())

    targets = list(RIP_TARGETS)
    if positives_only:
        before = len(observed)
        observed = observed[
            pd.to_numeric(observed.get("frames_with_detection"),
                          errors="coerce").fillna(0) > 0].copy()
        targets = [t for t in targets if t not in PRESENCE_TARGETS]
        print(f"\n  --positives-only: {len(observed)} of {before} hours contain "
              "an annotated rip; the rest are dropped.")
        print("  Presence targets are NOT reported. Where a person chose which")
        print("  frames to keep, a presence correlation measures that choice at")
        print("  least as much as the ocean. What is left is a question about")
        print("  the rips that are there: does the size or the orientation of a")
        print("  rip somebody drew a box around track the conditions?")

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
    for target in targets:
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

    # Season is a confound of the same shape as time of day, and a year of
    # data makes it live: water temperature, air temperature and the whole
    # wave climate cycle annually, so a raw correlation can be the calendar.
    observed["month"] = observed["hour"].dt.month
    observed["hr_mo"] = (observed["hour_of_day"].astype(str) + "-"
                         + observed["month"].astype(str))
    controls = [("hr", observed["hour_of_day"]),
                ("mo", observed["month"]),
                ("hrmo", observed["hr_mo"])]

    # A demeaning control cannot distinguish "season caused the outcome" from
    # "the cause only varies with season". A driver that barely moves within a
    # month has almost nothing left after the control, so it collapses whether
    # or not it is causal — the fixture demonstrates a genuinely causal driver
    # going from 0.83 to 0.09. Reporting month alone alongside hour-by-month
    # at least shows how much of the collapse each control is responsible for.
    cells = observed["hr_mo"].nunique()
    print(f"\n  control cells: {observed['month'].nunique()} months, "
          f"{cells} hour-by-month combinations")
    print("  A slow-varying predictor (swell period, water temperature) cannot")
    print("  be separated from season by this data. A collapse from rho to")
    print("  rho_mo means 'indistinguishable from season', NOT 'not a cause'.")

    report_variance_absorbed(observed, usable, controls)

    for target in usable:
        report_correlations(
            observed, target, RIP_PREDICTORS, controls=controls,
            title=f"=== WHAT TRACKS {target} at {name} ===\n"
                  "rho = raw; rho_hr = hour removed; rho_mo = month removed; "
                  "rho_hrmo = both.\n"
                  "Ranked by rho_hrmo. Compare rho_mo against it: a large gap "
                  "means the finer control ran out of data, not that the "
                  "driver vanished.")
        report_regression(observed, target, RIP_PREDICTORS)

    print("\nCaveats specific to this target:")
    print("  - One camera. Nothing here generalizes to another beach.")
    if "score_max" in observed.columns and observed["score_max"].notna().any():
        print("  - The score is a YOLOv8 model's confidence, not a verified rip.")
        print("    A driver of the DETECTOR (glare, swell texture, contrast) is")
        print("    indistinguishable here from a driver of the rip.")
    else:
        print("  - These boxes were drawn by people, not a detector, so there is")
        print("    no confidence score and no detector artefact to worry about.")
        print("    The trade is that what a person chose to annotate, and when,")
        print("    is its own selection — see --positives-only.")
    if "bbox_area_max" in usable:
        print("  - bbox_area_max is in PIXELS. Cross-shore pixel resolution varies")
        print("    across a frame and between cameras, so an area is comparable")
        print("    within this camera and meaningless pooled across cameras.")
    if "rip_axis_offset_deg" in usable:
        print("  - rip_axis_offset_deg is degrees from the shore normal, folded")
        print("    onto 0-90: 0 is a rip running straight offshore, 90 is one")
        print("    lying along the beach. It inherits whatever error is in the")
        print("    assumed shore normal.")
    # This line used to print at every site regardless of which one was run.
    if "Walton" in name:
        print("  - Walton has no observed wind, so every wind column is ERA5 grid,")
        print("    which compare_wind_sources.py could not validate at this site.")
    observed_waves = [c for c in ("WVHT", "DPD", "APD")
                      if c in frame.columns and frame[c].notna().any()]
    model_waves = [c for c in MARINE_COLUMNS
                   if c in frame.columns and frame[c].notna().any()]
    if model_waves and not observed_waves:
        print("  - NO OBSERVED WAVES at this site. Every wave column here is")
        print("    Open-Meteo Marine reanalysis on a grid, not a buoy reading.")
        print("    Convention: ALL-CAPS columns are measured, lower_snake_case")
        print("    are modelled. A wave result here is a result about a model's")
        print("    reconstruction of the sea state, one step further from the")
        print("    water than the same result at a site with a buoy.")
    elif model_waves and observed_waves:
        print("  - Both measured (ALL-CAPS) and modelled (lower_snake_case) wave")
        print("    columns are present. They are NOT independent evidence of")
        print("    each other; agreement between them is expected, not a check.")
    print("  - Daylight only, and a full year long, so anything with a daily or")
    print("    annual cycle is confounded. That is what rho_hr and rho_hrmo are")
    print("    for; judge on rho_hrmo.")
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
                 "wave_height", "wave_period",
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


def load_water_quality(sites):
    """Bacteria samples from both sources, normalised into one frame.

    Two files with nothing in common but their meaning. water_quality.csv is
    the California CKAN pull, which names a station and has to be joined back
    to a camera. data/wqp_<site>.csv is the national Water Quality Portal pull,
    which already knows which camera it was requested for. Reading only the
    first -- which is what this did -- meant Virginia Beach's bacteria landed
    on disk and were never analysed, and the water-quality section silently
    stayed a California-only result.

    Every row that comes out of here carries `source`, because the two differ
    in ways that matter: the CKAN file has no non-detect flag at all, so its
    censoring is invisible, while WQP marks non-detects explicitly.
    """
    frames = []
    ckan = _load_ckan_wq(sites)
    if ckan is not None:
        frames.append(ckan)
    frames.extend(_load_wqp_wq())

    if not frames:
        print("No bacteria files. Run pull_observations.py (California) or "
              "pull_wqp_results.py --camera <name> (anywhere else).")
        return None

    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["group"] = frame["Analyte"].map(analyte_key)
    # log10 because bacteria counts span orders of magnitude; +1 keeps zeros.
    frame["log_value"] = np.log10(frame["value"].clip(lower=0) + 1)
    for source, part in frame.groupby("source"):
        print(f"  {source}: {len(part)} numeric results, "
              f"{part['camera_name'].nunique()} site(s)")
    return frame


def _load_ckan_wq(sites):
    """The California file, joined back to cameras by station code."""
    frame = read_csv(f"{DATA_DIR}/water_quality.csv")
    if frame is None:
        return None
    frame["date"] = pd.to_datetime(frame["SampleDate"], errors="coerce").dt.normalize()
    frame["value"] = pd.to_numeric(frame["Result"], errors="coerce")
    dropped = int(frame["value"].isna().sum())
    if dropped:
        print(f"  water_quality.csv: {dropped}/{len(frame)} results are not "
              "numeric (non-detects and qualifiers) and are excluded")
    frame = frame.dropna(subset=["date", "value"])

    mapping = map_stations(sites, frame)
    frame["camera_name"] = frame["StationCode"].astype(str).map(mapping)
    unmapped = int(frame["camera_name"].isna().sum())
    if unmapped:
        print(f"  {unmapped} CKAN results belong to no qualifying site "
              "and are dropped")
    frame = frame.dropna(subset=["camera_name"])
    frame["source"] = "CKAN"
    frame["nondetect"] = False
    return frame if not frame.empty else None


def _load_wqp_wq():
    """Every data/wqp_<site>.csv, which already knows its own camera."""
    frames = []
    for path in sorted(glob.glob(f"{DATA_DIR}/wqp_*.csv")):
        raw = read_csv(path)
        if raw is None or "sampled_at" not in raw.columns:
            continue
        stamp = pd.to_datetime(raw["sampled_at"], errors="coerce", utc=True)
        frame = pd.DataFrame({
            "date": stamp.dt.tz_convert(None).dt.normalize(),
            "value": pd.to_numeric(raw["value"], errors="coerce"),
            "Analyte": raw["analyte"].astype(str),
            "StationCode": raw["station"].astype(str),
            "camera_name": raw["site"].astype(str),
            "nondetect": raw.get("nondetect", False),
            "source": "WQP",
        })
        censored = int(pd.Series(frame["nondetect"]).fillna(False).astype(bool).sum())
        usable = frame.dropna(subset=["date", "value"])
        name = os.path.basename(path)
        if censored:
            print(f"  {name}: {censored}/{len(frame)} are non-detects — dropped, "
                  "so the low end of this site is under-represented")
        if len(usable) < len(frame) - censored:
            print(f"  {name}: {len(frame) - censored - len(usable)} rows have no "
                  "usable date or value")
        # Two unit codes in one file means two lab methods pooled as if they
        # were one measurement. Ranks tolerate a constant rescaling; they do
        # not tolerate two different ones mixed within a site.
        if "unit" in raw.columns:
            units = raw.loc[raw["unit"].astype(str).str.strip() != "", "unit"]
            if units.nunique() > 1:
                print(f"  {name}: MIXED UNITS {sorted(units.unique())} — the "
                      "correlation below pools them without converting")
        if not usable.empty:
            frames.append(usable)
    return frames


def daily_conditions(site):
    """One row per day of whatever conditions exist for this site."""
    parts = []
    precip_id = site.get("precip_station_id")
    if pd.notna(precip_id):
        gauge = read_csv(f"{DATA_DIR}/precip_{str(precip_id).replace(':', '_')}.csv")
        if gauge is not None:
            gauge["date"] = pd.to_datetime(gauge["date"], errors="coerce").dt.normalize()
            parts.append(gauge.set_index("date"))

    # Marine is here for the same reason it is in the rip path: at a site with
    # no buoy, leaving it out does not report "waves are missing", it reports
    # "waves do not matter", which is a different and false claim.
    for loader in (lambda: load_gridded(site["camera_name"]),
                   lambda: load_marine(site["camera_name"]),
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
    samples = load_water_quality(sites)
    if samples is None:
        return
    print(f"\n{len(samples)} numeric results, "
          f"{samples['date'].min():%Y-%m-%d} to {samples['date'].max():%Y-%m-%d}")
    counts = samples["Analyte"].value_counts()
    print("\nanalytes:")
    print(counts.to_string())
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
    ap.add_argument("--site", help="substring of the rip table to analyse, when "
                                   "more than one camera has been pulled")
    ap.add_argument("--positives-only", action="store_true",
                    help="keep only hours containing an annotated rip and drop "
                         "the presence targets. Use this wherever a person "
                         "chose which frames the dataset contains (RipAID).")
    args = ap.parse_args()
    sites = load_sites()
    print(f"{len(sites)} qualifying sites\n")
    if args.target in ("rip", "both"):
        print("=" * 70)
        print("RIP DETECTION")
        print("=" * 70)
        analyze_rip(sites, want=args.site, positives_only=args.positives_only)
    if args.target in ("wq", "both"):
        print("\n" + "=" * 70)
        print("WATER QUALITY")
        print("=" * 70)
        analyze_wq(sites)


if __name__ == "__main__":
    main()
