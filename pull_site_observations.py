"""Ocean and weather conditions for any site on earth, by coordinate.

pull_observations.py is built around candidate_sites_ranked.csv — the seven
California sites — and around US station networks: NDBC, NOAA CO-OPS, NOAA
GHCND. None of those exist for a beach in France, and the CSV does not contain
Virginia Beach. This script takes a coordinate instead of a site list, and
splits its sources by what is actually available there:

  everywhere   Open-Meteo ERA5 reanalysis  — wind, rain, air temperature
  everywhere   Open-Meteo Marine           — wave height, period, direction
  US only      NDBC buoy                   — observed waves, when one is close
  US only      NOAA CO-OPS                 — observed tide and water temperature
  US only      NOAA GHCND                  — observed daily rainfall

The US sources are observations and the two Open-Meteo products are model
reanalysis. That distinction matters and is preserved in the output: gridded
values never land in a column named after a buoy field, so nothing downstream
can average a reanalysis wave height together with a measured one.

    python pull_site_observations.py --camera "Virginia Beach"
    python pull_site_observations.py --lat 44.47 --lon -1.25 --name "Biscarrosse"
    python pull_site_observations.py --sites-csv euro_sites.csv
    python pull_site_observations.py --camera "Virginia Beach" --probe

Dates default to the last year. --start/--end override; for a rip feed, match
its inventory range rather than taking the default.
"""

import env  # noqa: F401  -- loads .env into os.environ

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import find_candidate_sites as f
import pull_observations as obs

CANDIDATES_CSV = "camera_candidates.csv"
OUT_DIR = "data"

ERA5 = "https://archive-api.open-meteo.com/v1/archive"
MARINE = "https://marine-api.open-meteo.com/v1/marine"

ERA5_VARS = ["precipitation", "wind_speed_10m", "wind_direction_10m",
             "wind_gusts_10m", "temperature_2m"]
MARINE_VARS = ["wave_height", "wave_direction", "wave_period",
               "wind_wave_height", "wind_wave_period",
               "swell_wave_height", "swell_wave_direction", "swell_wave_period"]

RAIN_WINDOWS_HOURS = (24, 48, 72)
MAX_BUOY_KM = 50
MAX_TIDE_KM = 50
MAX_PRECIP_KM = 30
ARCHIVE_LAG_DAYS = 6  # ERA5 is not published in real time
TIMEOUT = 300

# The marine model is defined on ocean cells only, so a camera sitting on the
# beach can land on land and get an error rather than a wave height. These are
# offsets in degrees to try, nearest first, when the exact point fails. Which
# direction is seaward is not knowable here, so all eight are tried.
MARINE_NUDGES = [0.0, 0.05, 0.10, 0.20]
MARINE_BEARINGS = [(0, 1), (1, 0), (0, -1), (-1, 0),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)]


def slugify(text):
    return "".join(c if c.isalnum() else "_" for c in str(text))[:48]


def open_meteo(url, lat, lon, start, end, variables, probe=False, models=None):
    """(dataframe, note). Returns (None, note) when the request fails."""
    params = {"latitude": lat, "longitude": lon,
              "start_date": start.strftime("%Y-%m-%d"),
              "end_date": end.strftime("%Y-%m-%d"),
              "hourly": ",".join(variables),
              "timezone": "UTC",
              # NDBC and CO-OPS both report wind in m/s; asking for m/s here
              # keeps every wind column in this project in one unit.
              "wind_speed_unit": "ms"}
    if models:
        params["models"] = models
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
    except (requests.Timeout, requests.ConnectionError) as exc:
        return None, f"{type(exc).__name__}"
    if resp.status_code != 200:
        detail = ""
        try:
            detail = str(resp.json().get("reason", ""))[:160]
        except ValueError:
            detail = resp.text[:160]
        return None, f"HTTP {resp.status_code}: {detail}"

    payload = resp.json()
    if payload.get("error"):
        return None, str(payload.get("reason", payload))[:160]
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        return None, f"unexpected shape, keys {list(payload.keys())}"

    if probe:
        print(f"      grid cell {payload.get('latitude')}, {payload.get('longitude')} "
              f"(requested {lat}, {lon})")
        print(f"      returned  {list(hourly.keys())}")
        print(f"      units     {payload.get('hourly_units')}")

    missing = [v for v in variables if v not in hourly]
    if missing:
        print(f"      requested but not returned: {missing}")

    units = payload.get("hourly_units") or {}
    for var in ("wind_speed_10m", "wind_gusts_10m"):
        if var in units and units[var] != "m/s":
            print(f"      WARNING: {var} came back in {units[var]}, not m/s")

    frame = pd.DataFrame(dict(hourly))
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame.sort_values("time").reset_index(drop=True), "ok"


def fetch_marine(lat, lon, start, end, probe=False, models=None):
    """Marine reanalysis, nudging seaward if the exact point is a land cell."""
    for nudge in MARINE_NUDGES:
        bearings = [(0, 0)] if nudge == 0 else MARINE_BEARINGS
        for dlat, dlon in bearings:
            try_lat, try_lon = lat + dlat * nudge, lon + dlon * nudge
            frame, note = open_meteo(MARINE, try_lat, try_lon, start, end,
                                     MARINE_VARS, probe=probe and nudge == 0,
                                     models=models)
            if frame is not None:
                # An all-NaN frame is a land cell answering politely.
                data_cols = [c for c in frame.columns if c != "time"]
                if data_cols and frame[data_cols].notna().any().any():
                    if nudge:
                        km = nudge * 111
                        print(f"      exact point has no wave data; used a cell "
                              f"~{km:.0f} km away ({try_lat:.3f}, {try_lon:.3f})")
                    return frame, "ok"
            elif nudge == 0:
                print(f"      at the site itself: {note}")
            time.sleep(0.2)
    return None, "no ocean cell found within ~22 km"


MARINE_MODELS_TO_TRY = ("best_match", "era5_ocean", "ewam", "gwam",
                        "meteofrance_wave", "ecmwf_wam025")


def probe_marine_models(lat, lon, start, end):
    """Which marine model actually has data this far back.

    The default returned a full grid of hours for Wrightsville Beach and real
    numbers in only 18% of them, all at the recent end. A wave series that
    starts in 2021 cannot be joined to a casualty record that starts in 2000,
    and the failure is silent: the request succeeds, the file is written, the
    column is simply mostly NaN. So the model is chosen by asking each one for
    the actual span rather than by trusting the default.
    """
    print(f"probing marine models at ({lat:.3f}, {lon:.3f}) over "
          f"{start:%Y-%m-%d}..{end:%Y-%m-%d}\n")
    print(f"  {'model':<20} {'hours with a wave height':>26}  span")
    for model in MARINE_MODELS_TO_TRY:
        frame, note = open_meteo(MARINE, lat, lon, start, end,
                                 ["wave_height"], models=model)
        if frame is None:
            print(f"  {model:<20} {'-':>26}  {note}")
            time.sleep(0.3)
            continue
        have = frame["wave_height"].notna()
        if not have.any():
            print(f"  {model:<20} {'0':>26}  returned only nulls")
            time.sleep(0.3)
            continue
        stamps = pd.to_datetime(frame.loc[have, "time"], errors="coerce").dropna()
        print(f"  {model:<20} {int(have.sum()):>26}  "
              f"{stamps.min():%Y-%m-%d}..{stamps.max():%Y-%m-%d}")
        time.sleep(0.3)
    print("\nPick the one whose span covers your record and pass it as "
          "--marine-model.")


def add_rain_windows(frame):
    """Rolling rainfall over true clock hours; NaN across any gap."""
    if "precipitation" not in frame.columns:
        return frame
    series = frame.set_index("time")["precipitation"]
    full = pd.date_range(series.index.min(), series.index.max(), freq="h")
    series = series.reindex(full)
    out = frame.set_index("time").reindex(full)
    out.index.name = "time"
    for hours in RAIN_WINDOWS_HOURS:
        out[f"rain_{hours}h_mm"] = series.rolling(hours, min_periods=hours).sum()
    gaps = int(series.isna().sum())
    if gaps:
        print(f"      {gaps}/{len(full)} hours missing; windows spanning them are NaN")
    return out.reset_index()


def in_united_states(lat, lon):
    """Rough test for whether the US station networks are worth querying.

    Deliberately generous — it costs one wasted request to be wrong, and the
    per-source distance checks reject anything genuinely out of range. Covers
    the lower 48, Alaska, Hawaii and the Caribbean territories.
    """
    boxes = [(24.0, -125.5, 49.5, -66.5), (51.0, -179.9, 71.5, -129.0),
             (18.5, -160.5, 22.5, -154.5), (17.5, -68.0, 18.6, -64.5)]
    return any(a <= lat <= c and b <= lon <= d for a, b, c, d in boxes)


def pull_us_stations(name, lat, lon, start, end, token):
    """NDBC, CO-OPS and GHCND for a US site. Returns a list of note strings."""
    notes = []

    buoys = f.get_all_buoys()
    row, km = None, None
    if not buoys.empty:
        row, km = _nearest(lat, lon, buoys, "Lat", "Lon", MAX_BUOY_KM)
    if row is None:
        notes.append(f"no NDBC buoy within {MAX_BUOY_KM} km")
    else:
        station = str(row["Station"])
        print(f"    buoy {station} ({km:.1f} km)")
        frame = obs.pull_buoy(station, start, end)
        if frame is None or frame.empty:
            notes.append(f"buoy {station} returned nothing")
        else:
            path = f"{OUT_DIR}/buoy_{station}.csv"
            frame.to_csv(path, index=False)
            print(f"      {len(frame)} rows -> {path}")

    stations = obs.coops_stations("waterlevels")
    row, km = _nearest(lat, lon, stations, "lat", "lon", MAX_TIDE_KM)
    if row is None:
        notes.append(f"no CO-OPS water-level gauge within {MAX_TIDE_KM} km")
    else:
        station = str(row["station_id"])
        print(f"    tide {station} {row.get('name', '')} ({km:.1f} km)")
        frame = obs.pull_coops_series(station, "water_level", start, end)
        if frame is None or frame.empty:
            notes.append(f"CO-OPS {station} served no water level")
        else:
            frame = obs.add_tide_state(frame)
            path = f"{OUT_DIR}/tide_{station}.csv"
            frame.to_csv(path, index=False)
            print(f"      {len(frame)} readings -> {path}")

        temp = obs.pull_coops_series(station, "water_temperature", start, end)
        if temp is not None and not temp.empty:
            path = f"{OUT_DIR}/watertemp_{station}.csv"
            temp.to_csv(path, index=False)
            print(f"      {len(temp)} water temps -> {path}")
        else:
            notes.append(f"CO-OPS {station} served no water temperature")

    if not token:
        notes.append("NOAA_CDO_TOKEN not set — no GHCND rainfall")
    return notes


def _nearest(lat, lon, frame, lat_col, lon_col, max_km):
    if frame is None or frame.empty:
        return None, None
    if lat_col not in frame.columns or lon_col not in frame.columns:
        return None, None
    lats = pd.to_numeric(frame[lat_col], errors="coerce")
    lons = pd.to_numeric(frame[lon_col], errors="coerce")
    keep = (lats.notna() & lons.notna()).to_numpy()
    if not keep.any():
        return None, None
    subset = frame[keep]
    dist = f.haversine_km(lat, lon, lats.to_numpy()[keep], lons.to_numpy()[keep])
    best = int(dist.argmin())
    if dist[best] > max_km:
        return None, None
    return subset.iloc[best], float(dist[best])


def pull_site(name, lat, lon, start, end, token, probe=False, skip_us=False,
              marine_model=None):
    print(f"\n{name}  ({lat:.4f}, {lon:.4f})")
    slug = slugify(name)
    notes = []

    print("    ERA5 weather (global)")
    weather, note = open_meteo(ERA5, lat, lon, start, end, ERA5_VARS, probe=probe)
    if weather is None:
        notes.append(f"ERA5 failed: {note}")
    else:
        weather = add_rain_windows(weather)
        path = f"{OUT_DIR}/gridded_{slug}.csv"
        weather.to_csv(path, index=False)
        rain = weather["precipitation"].sum() if "precipitation" in weather else float("nan")
        print(f"      {len(weather)} hours -> {path}   ({rain:.0f} mm total)")

    print("    Marine waves (global)")
    marine, note = fetch_marine(lat, lon, start, end, probe=probe,
                                models=marine_model)
    if marine is None:
        notes.append(f"marine failed: {note}")
    else:
        # Which model produced this file is not recoverable from the numbers,
        # and the models disagree -- best_match starts in 2021 here while
        # era5_ocean reaches 2000. A file that does not name its source cannot
        # be audited later, and two sites pulled under different models cannot
        # be compared without knowing it.
        marine["model"] = marine_model or "best_match (default)"
        path = f"{OUT_DIR}/marine_{slug}.csv"
        marine.to_csv(path, index=False)
        have = [c for c in marine.columns if c != "time" and marine[c].notna().any()]
        print(f"      {len(marine)} hours -> {path}")
        print(f"      columns with data: {have}")
        # A full grid of hours with values in only a fraction of them is the
        # failure mode that looks like success: the request is fine, the file
        # is written, and the column is mostly NaN at the old end. Say the
        # span, not just the row count.
        if "wave_height" in marine.columns:
            filled = marine["wave_height"].notna()
            share = 100.0 * filled.mean() if len(marine) else 0.0
            if filled.any():
                stamps = pd.to_datetime(marine.loc[filled, "time"],
                                        errors="coerce").dropna()
                print(f"      wave_height populated in {share:.0f}% of hours, "
                      f"{stamps.min():%Y-%m-%d} to {stamps.max():%Y-%m-%d}")
                if share < 90:
                    print("      the rest are empty. Run --marine-probe to see "
                          "which model\n      covers your span, then pass "
                          "--marine-model.")

    if skip_us or not in_united_states(lat, lon):
        print("    outside the US station networks — gridded sources only")
    else:
        notes.extend(pull_us_stations(name, lat, lon, start, end, token))

    for line in notes:
        print(f"    note: {line}")
    return notes


def load_targets(args):
    if args.lat is not None and args.lon is not None:
        return [(args.name or f"{args.lat},{args.lon}", args.lat, args.lon)]

    if args.sites_csv:
        frame = pd.read_csv(args.sites_csv)
        lat_col = _pick(frame, ("lat", "latitude"))
        lon_col = _pick(frame, ("lon", "lng", "longitude"))
        name_col = _pick(frame, ("name", "camera", "site", "beach", "location"))
        if not (lat_col and lon_col):
            sys.exit(f"{args.sites_csv} has no latitude/longitude columns. "
                     f"Found: {list(frame.columns)}")
        rows = []
        for index, row in frame.iterrows():
            label = row[name_col] if name_col else f"site_{index}"
            rows.append((str(label), float(row[lat_col]), float(row[lon_col])))
        return rows

    if not args.camera:
        sys.exit("Give --camera, or --lat/--lon, or --sites-csv.")
    if not os.path.exists(CANDIDATES_CSV):
        sys.exit(f"{CANDIDATES_CSV} not found — run scan_cameras.py first.")
    frame = pd.read_csv(CANDIDATES_CSV)
    hits = frame[frame["camera"].str.contains(args.camera, case=False, na=False)]
    if hits.empty:
        sys.exit(f"No camera in {CANDIDATES_CSV} matching {args.camera!r}.")
    if len(hits) > 1:
        print(f"{len(hits)} cameras match {args.camera!r}:")
        for _, row in hits.iterrows():
            print(f"  {row['camera']}")
        print()
    return [(row["camera"], float(row["lat"]), float(row["lon"]))
            for _, row in hits.iterrows()]


def _pick(frame, candidates):
    lowered = {str(c).lower(): c for c in frame.columns}
    for want in candidates:
        for lower, original in lowered.items():
            if lower == want:
                return original
    for want in candidates:
        for lower, original in lowered.items():
            if want in lower:
                return original
    return None


def main():
    ap = argparse.ArgumentParser(description="Conditions for any site, by coordinate.")
    ap.add_argument("--camera", help="substring of a name in camera_candidates.csv")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--name", help="label for --lat/--lon output files")
    ap.add_argument("--sites-csv", help="CSV with latitude/longitude columns")
    ap.add_argument("--start", help="YYYY-MM-DD")
    ap.add_argument("--end", help="YYYY-MM-DD")
    ap.add_argument("--probe", action="store_true",
                    help="print the raw response schema for the first site")
    ap.add_argument("--skip-us-stations", action="store_true",
                    help="gridded sources only, even inside the US")
    ap.add_argument("--marine-model",
                    help="Open-Meteo wave model. Use era5_ocean for anything "
                         "historical: the default is a forecast model whose "
                         "archive can start years after your window does. Run "
                         "--marine-probe to see each model's real span here.")
    ap.add_argument("--marine-probe", action="store_true",
                    help="ask each wave model what span it actually has at "
                         "this site, then stop")
    args = ap.parse_args()

    end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
           if args.end else datetime.now(timezone.utc) - timedelta(days=ARCHIVE_LAG_DAYS))
    start = (datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
             if args.start else end - timedelta(days=365))
    if start >= end:
        sys.exit(f"--start {start:%Y-%m-%d} is not before --end {end:%Y-%m-%d}.")

    lag = (datetime.now(timezone.utc) - end).days
    if lag < ARCHIVE_LAG_DAYS:
        print(f"NOTE: --end is {lag} days ago. ERA5 publishes on a ~5 day lag, "
              "so the last few days may come back empty.\n")

    targets = load_targets(args)
    if args.marine_probe:
        name, lat, lon = targets[0]
        print(f"{name}")
        probe_marine_models(lat, lon, start, end)
        return
    token = os.environ.get("NOAA_CDO_TOKEN")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Window: {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    print(f"{len(targets)} site(s)")

    failed = []
    for index, (name, lat, lon) in enumerate(targets):
        notes = pull_site(name, lat, lon, start, end, token,
                          probe=args.probe and index == 0,
                          skip_us=args.skip_us_stations,
                          marine_model=args.marine_model)
        if notes:
            failed.append((name, notes))

    print(f"\n{'=' * 74}\nDone. CSVs in {OUT_DIR}/")
    if failed:
        print("\nSites with something missing:")
        for name, notes in failed:
            print(f"  {name}: {'; '.join(notes)}")


if __name__ == "__main__":
    main()
