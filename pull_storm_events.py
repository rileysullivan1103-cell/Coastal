#!/usr/bin/env python3
"""Rip-current events, deaths and injuries from the NOAA Storm Events database.

Everything else in this project measures a rip DETECTOR or a rip ANNOTATOR.
Nothing so far measures a rip. Storm Events logs an event when a rip current
put someone in trouble, so it is the first outcome here that does not come
from the same instrument as the prediction.

Two properties make it worth the trouble:

  It has a denominator. RipAID's no-rip frames were deleted by a curator, so
  its zeros are not zeros. Here every day in the record with no logged event
  is a real observed zero for 'a rip current hurt somebody today', and this
  script writes those zero days out explicitly rather than leaving gaps.

  It is an outcome, not a proxy. "Wave height predicts the detector firing"
  and "wave height predicts a casualty" are different claims, and only the
  second is worth telling a lifeguard service.

And one property that limits it, which no amount of care removes: an event is
logged when a person is in the water. This samples beach attendance as much
as rip occurrence -- expect July, weekends and holidays. That is the same
family of bias as RipAID's lifeguard log, with one important difference: it
was not curated after the fact.

Usage:
  python pull_storm_events.py --probe 2023      # pin the real columns, stop
  python pull_storm_events.py --zones           # who logs rip currents, and where
  python pull_storm_events.py --zone "VIRGINIA BEACH" --start-year 2000
"""
import argparse
import gzip
import io
import os
import re
import sys
import time

import pandas as pd
import requests

BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
CACHE = "data/storm_events"
OUT_DIR = "data"
TIMEOUT = 120
MAX_RETRIES = 4

# The filenames carry a creation stamp that changes whenever NCEI reissues a
# year -- StormEvents_details-ftp_v1.0_d2023_c20240416.csv.gz -- so the name
# cannot be constructed, only discovered. Building it would work until the
# day they reprocess a year, and then fail with a 404 that looks like the
# year having no data.
DETAILS_RE = re.compile(r"StormEvents_details-ftp_v1\.0_d(\d{4})_c\d+\.csv\.gz")

# Column names as documented. --probe checks them against the real file and
# prints everything it actually found, so a rename shows up as a message
# rather than as a quietly empty column.
COL_TYPE = "EVENT_TYPE"
COL_STATE = "STATE"
COL_CZ_NAME = "CZ_NAME"
COL_CZ_TYPE = "CZ_TYPE"
COL_BEGIN = "BEGIN_DATE_TIME"
COL_TZ = "CZ_TIMEZONE"
COL_DEATHS = "DEATHS_DIRECT"
COL_INJURIES = "INJURIES_DIRECT"
COL_LAT = "BEGIN_LAT"
COL_LON = "BEGIN_LON"
COL_NARRATIVE = "EVENT_NARRATIVE"
REQUIRED = (COL_TYPE, COL_STATE, COL_CZ_NAME, COL_BEGIN)

RIP = "Rip Current"


def get_with_retry(url, **kwargs):
    delay = 2
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"      {type(exc).__name__}; retry {attempt} in {delay}s")
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
            print(f"      HTTP {resp.status_code}; retry {attempt} in {delay}s")
            time.sleep(delay)
            delay *= 2
            continue
        return resp
    raise RuntimeError("unreachable")


def slugify(text):
    """The slug used everywhere else in this project."""
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")


def index_years():
    """year -> filename, read from the directory listing."""
    resp = get_with_retry(BASE)
    resp.raise_for_status()
    found = {}
    for match in DETAILS_RE.finditer(resp.text):
        year = int(match.group(1))
        # A year can appear twice if NCEI left an old creation stamp in place;
        # the later stamp sorts last and is the one to keep.
        name = match.group(0)
        if year not in found or name > found[year]:
            found[year] = name
    if not found:
        sys.exit(f"No details files matched at {BASE} — the naming scheme has "
                 "changed; update DETAILS_RE.")
    return found


def fetch_year(year, files):
    """One year of the details table, cached on disk."""
    if year not in files:
        return None
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, files[year])
    if not os.path.exists(path):
        resp = get_with_retry(BASE + files[year])
        if resp.status_code != 200:
            print(f"  {year}: HTTP {resp.status_code}, skipped")
            return None
        with open(path, "wb") as handle:
            handle.write(resp.content)
    with gzip.open(path, "rb") as handle:
        raw = handle.read()
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def check_columns(frame):
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        print("\n=== ACTUAL COLUMNS ===")
        for col in frame.columns:
            print(f"  {col}")
        sys.exit(f"\n{len(missing)} required column(s) missing: {missing}\n"
                 "Update the COL_* constants at the top of this file.")


def probe(year, files):
    frame = fetch_year(year, files)
    if frame is None:
        sys.exit(f"{year} is not in the index. Years available: "
                 f"{min(files)}-{max(files)}")
    print(f"\n{len(frame):,} events in {year}, {len(frame.columns)} columns")

    print("\n=== ACTUAL COLUMNS ===")
    for col in frame.columns:
        example = frame[col].dropna()
        example = str(example.iloc[0])[:46] if len(example) else ""
        print(f"  {col:<34} e.g. {example}")
    check_columns(frame)
    print("  all required columns present")

    print("\n=== EVENT TYPES CONTAINING WATER HAZARDS ===")
    counts = frame[COL_TYPE].value_counts()
    for name, n in counts.items():
        if any(w in str(name).lower() for w in
               ("rip", "surf", "coastal", "tsunami", "seiche", "wave", "tide")):
            print(f"  {name:<28} {n:>6}")

    rips = frame[frame[COL_TYPE] == RIP]
    print(f"\n=== '{RIP}' IN {year}: {len(rips)} events ===")
    if rips.empty:
        print("  none — check the exact spelling in the list above")
        return

    if COL_CZ_TYPE in rips.columns:
        print("  CZ_TYPE:", dict(rips[COL_CZ_TYPE].value_counts()))
        print("    (C = county, Z = NWS forecast zone, M = marine zone;"
              " a Z or M event is a stretch of coast, not a point)")
    for col in (COL_LAT, COL_LON):
        if col in rips.columns:
            have = int(rips[col].notna().sum())
            print(f"  {col}: {have}/{len(rips)} populated"
                  + ("" if have else "  <- no coordinates; join by zone name"))
    for col in (COL_DEATHS, COL_INJURIES):
        if col in rips.columns:
            total = pd.to_numeric(rips[col], errors="coerce").fillna(0).sum()
            print(f"  {col}: {int(total)} across {len(rips)} events")

    print("\n  where they were logged (top 15):")
    where = rips.groupby([COL_STATE, COL_CZ_NAME]).size().sort_values(ascending=False)
    for (state, zone), n in where.head(15).items():
        print(f"    {str(state):<18} {str(zone):<34} {n:>4}")

    print("\n  one event, in full:")
    row = rips.iloc[0]
    for col in (COL_BEGIN, COL_TZ, COL_STATE, COL_CZ_NAME, COL_CZ_TYPE,
                COL_DEATHS, COL_INJURIES):
        if col in row.index:
            print(f"    {col:<20} {row[col]}")
    if COL_NARRATIVE in row.index and pd.notna(row[COL_NARRATIVE]):
        print(f"    {COL_NARRATIVE:<20} {str(row[COL_NARRATIVE])[:300]}")

    print("\nColumns look right — re-run with --zones to pick a zone, then "
          "--zone to write the daily table.")


def load_rips(files, start_year, end_year):
    frames, years = [], []
    for year in range(start_year, end_year + 1):
        frame = fetch_year(year, files)
        if frame is None:
            continue
        check_columns(frame)
        rips = frame[frame[COL_TYPE] == RIP]
        print(f"  {year}: {len(rips):>4} rip-current events "
              f"of {len(frame):,} total")
        years.append(year)
        if not rips.empty:
            frames.append(rips)
    if not years:
        sys.exit("No years loaded.")
    if not frames:
        sys.exit("No rip-current events in that span.")
    return pd.concat(frames, ignore_index=True), years


def show_zones(files, start_year, end_year):
    rips, years = load_rips(files, start_year, end_year)
    print(f"\n{'=' * 74}\nZONES LOGGING RIP CURRENTS, {min(years)}-{max(years)}"
          f"\n{'=' * 74}")
    grouped = rips.groupby([COL_STATE, COL_CZ_NAME])
    table = pd.DataFrame({
        "events": grouped.size(),
        "deaths": grouped[COL_DEATHS].apply(
            lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
        "injuries": grouped[COL_INJURIES].apply(
            lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
    }).sort_values("events", ascending=False)
    with pd.option_context("display.max_rows", 60, "display.width", 200):
        print(table.head(50).to_string())
    print(f"\n{len(table)} zones in total. Pass one to --zone (a case-insensitive"
          " substring of CZ_NAME).")
    print("The zone is a stretch of coast, not a beach. Check on a map which"
          " camera it actually contains before joining.")


def daily_table(rips, years, label):
    """One row per DAY, including the days nothing happened.

    This is the whole point. An event list alone cannot answer 'do rips hurt
    people when the waves are big', because it holds no quiet days to compare
    against. Reindexing onto every day in the span turns a list of incidents
    into a series with real zeros.
    """
    stamp = pd.to_datetime(rips[COL_BEGIN], errors="coerce")
    unparsed = int(stamp.isna().sum())
    if unparsed:
        print(f"  {unparsed}/{len(rips)} rows have an unreadable "
              f"{COL_BEGIN} and are dropped")
    rips = rips.assign(date=stamp.dt.normalize()).dropna(subset=["date"])

    # BEGIN_DATE_TIME is local to the zone (CZ_TIMEZONE says which). A bather
    # incident belongs to the local day it happened on, and the conditions
    # this joins to are daily means, so the local date is the right key. It
    # would be the wrong key for an hourly join.
    grouped = rips.groupby("date")
    frame = pd.DataFrame({
        "events": grouped.size(),
        "deaths": grouped[COL_DEATHS].apply(
            lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
        "injuries": grouped[COL_INJURIES].apply(
            lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()),
    })
    span = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    frame = frame.reindex(span, fill_value=0)
    frame.index.name = "date"
    frame["any_event"] = (frame["events"] > 0).astype(int)
    frame["any_casualty"] = ((frame["deaths"] + frame["injuries"]) > 0).astype(int)
    return frame


def report(frame, label, years):
    days = len(frame)
    hit = int(frame["any_event"].sum())
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    print(f"{days:,} days, {min(years)}-{max(years)}")
    print(f"{hit} days carry a rip-current event  ({100 * hit / days:.2f}% of days)")
    print(f"{int(frame['deaths'].sum())} deaths, "
          f"{int(frame['injuries'].sum())} injuries")

    if hit < 30:
        print("\n  *** TOO FEW EVENT DAYS TO MODEL ***")
        print(f"  {hit} positive days cannot support a correlation against")
        print("  daily conditions. Widen --start-year, or pick a busier zone")
        print("  from --zones. Do not report a rho computed on this.")

    by_month = frame.groupby(frame.index.month)["events"].sum()
    print("\nevents by month (this is beach attendance as much as rips):")
    print("  " + "  ".join(f"{m:02d}:{int(v)}" for m, v in by_month.items()))

    by_dow = frame.groupby(frame.index.dayofweek)["events"].sum()
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("\nevents by day of week (a weekend skew is exposure, not ocean):")
    print("  " + "  ".join(f"{names[d]}:{int(v)}" for d, v in by_dow.items()))
    weekend = int(by_dow.get(5, 0) + by_dow.get(6, 0))
    total = int(by_dow.sum())
    if total:
        share = 100 * weekend / total
        print(f"  Sat+Sun hold {share:.0f}% of events; 2 of 7 days is 28.6%.")
        if share > 40:
            print("  That gap is exposure. Control for day-of-week before")
            print("  reading anything into a conditions correlation.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", type=int, metavar="YEAR",
                    help="download one year, print its real columns and what a "
                         "rip-current event looks like, then stop")
    ap.add_argument("--zones", action="store_true",
                    help="list every zone that logs rip currents, with counts")
    ap.add_argument("--zone", help="substring of CZ_NAME to write a daily table for")
    ap.add_argument("--state", help="restrict to one STATE")
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=None)
    args = ap.parse_args()

    print(f"index: {BASE}")
    files = index_years()
    print(f"  {len(files)} yearly details files, {min(files)}-{max(files)}")
    end_year = args.end_year or max(files)

    if args.probe:
        probe(args.probe, files)
        return
    if not (args.zones or args.zone):
        sys.exit("Nothing to do. Start with --probe <year>, then --zones.")

    print(f"\nreading {args.start_year}-{end_year} "
          "(cached in data/storm_events after the first run)")
    rips, years = load_rips(files, args.start_year, end_year)

    if args.zones:
        show_zones(files, args.start_year, end_year)
        return

    subset = rips
    if args.state:
        subset = subset[subset[COL_STATE].astype(str).str.upper()
                        == args.state.upper()]
    mask = subset[COL_CZ_NAME].astype(str).str.contains(
        args.zone, case=False, na=False)
    subset = subset[mask]
    if subset.empty:
        sys.exit(f"No rip-current events in a zone matching {args.zone!r}. "
                 "Run --zones to see the real names.")

    matched = sorted(subset[COL_CZ_NAME].astype(str).unique())
    print(f"\nmatched {len(matched)} zone name(s): {', '.join(matched)}")
    if len(matched) > 1:
        print("  more than one zone matched; they are pooled into one series")

    frame = daily_table(subset, years, args.zone)
    report(frame, f"RIP-CURRENT EVENT DAYS — {args.zone}", years)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/storm_rip_{slugify(args.zone)}_daily.csv"
    frame.to_csv(path)
    print(f"\nwrote {path}  ({len(frame):,} days)")
    print("Join it to the daily means of gridded_*.csv and marine_*.csv for the "
          "matching site.")


if __name__ == "__main__":
    main()
