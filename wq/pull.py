"""B1: stations and results, nationally, from WQP and from California's CKAN.

Reuses the existing pullers rather than re-implementing them:
scan_cameras.get_with_retry for the backoff, pull_wqp_results for the WQP
column names it already verified, pull_observations for the CKAN datastore.
Those three import ndbc_api and pywebcoos between them, so every one of them
is imported INSIDE the function that needs the network -- which is what keeps
wq/clean.py, wq/fit.py and the offline tests runnable on a bare checkout.

Work is chunked by (state, year) and each chunk is written to
data/wq/raw/ as it lands. A chunk already on disk is skipped, so an
interrupted national pull is resumed by re-running the same command rather
than restarted. The national results pull is hours of requests; that is the
whole reason the run script starts with caffeinate.

    python -m wq.pull --probe                 # column names, then stop
    python -m wq.pull --stations
    python -m wq.pull --results
    python -m wq.pull --results --states CA,FL
"""

import argparse
import glob
import os
import sys
import time
from contextlib import contextmanager
from datetime import date
from io import StringIO

import numpy as np
import pandas as pd

from . import config

WQP_STATION_URL = "https://www.waterqualitydata.us/data/Station/search"
WQP_RESULT_URL = "https://www.waterqualitydata.us/data/Result/search"

# WQP result columns. The first four are the ones pull_wqp_results.py already
# verified against a live response; the rest are the hygiene columns B2 and B3
# need. test_wq_offline.py asserts the shared four still agree with that
# module, so the two cannot drift apart unnoticed.
COL_STATION = "MonitoringLocationIdentifier"
COL_DATE = "ActivityStartDate"
COL_ANALYTE = "CharacteristicName"
COL_VALUE = "ResultMeasureValue"
COL_TIME = "ActivityStartTime/Time"
COL_TZ = "ActivityStartTime/TimeZoneCode"
COL_UNIT = "ResultMeasure/MeasureUnitCode"
COL_NONDETECT = "ResultDetectionConditionText"
COL_DETECTION_LIMIT = "DetectionQuantitationLimitMeasure/MeasureValue"
COL_DETECTION_UNIT = "DetectionQuantitationLimitMeasure/MeasureUnitCode"
COL_METHOD = "ResultAnalyticalMethod/MethodIdentifier"
COL_METHOD_NAME = "ResultAnalyticalMethod/MethodName"
COL_STATUS = "ResultStatusIdentifier"
COL_QUALIFIER = "MeasureQualifierCode"
COL_ACTIVITY_TYPE = "ActivityTypeCode"
COL_ACTIVITY_ID = "ActivityIdentifier"

RESULT_COLUMNS = (COL_STATION, COL_DATE, COL_TIME, COL_TZ, COL_ANALYTE,
                  COL_VALUE, COL_UNIT, COL_NONDETECT, COL_DETECTION_LIMIT,
                  COL_DETECTION_UNIT, COL_METHOD, COL_METHOD_NAME, COL_STATUS,
                  COL_QUALIFIER, COL_ACTIVITY_TYPE, COL_ACTIVITY_ID)
REQUIRED_RESULT_COLUMNS = (COL_STATION, COL_DATE, COL_ANALYTE, COL_VALUE)

# WQP station columns.
STATION_ID = "MonitoringLocationIdentifier"
STATION_NAME = "MonitoringLocationName"
STATION_TYPE = "MonitoringLocationTypeName"
STATION_LAT = "LatitudeMeasure"
STATION_LON = "LongitudeMeasure"
STATION_STATE = "StateCode"
STATION_COUNTY = "CountyCode"
STATION_ORG = "OrganizationIdentifier"
# WQP publishes the upstream drainage area on the station record itself.
# Confirmed present in a live Rhode Island response. It is the same quantity
# NLDI accumulates, arriving free with a pull that already happens, so it is
# read here and used wherever NLDI has not answered for a site.
STATION_DRAINAGE = "DrainageAreaMeasure/MeasureValue"
STATION_DRAINAGE_UNIT = "DrainageAreaMeasure/MeasureUnitCode"

# Top-level siteType categories to request. The finer filter that actually
# decides a site (config.COASTAL_SITE_TYPES, matched against
# MonitoringLocationTypeName) is applied locally, because "Great Lake" is a
# location type underneath the "Lake, Reservoir, Impoundment" site type and
# asking for it directly returns nothing.
REQUEST_SITE_TYPES = ("Ocean", "Estuary", "Lake, Reservoir, Impoundment")
# Streams, springs and outfalls no longer come from WQP: NHDPlus and EPA
# ECHO are the real layers for those, and wq/spatial.py pulls them. A WQP
# Stream station meant "somebody monitors a creek here", which is not the
# same claim as "there is a creek here".

# FIPS codes for every state and territory with an ocean, Gulf or Great Lakes
# shoreline. Inland states are absent by design: a coastal recreational
# station is the unit of analysis, and pulling Nebraska costs requests to
# return nothing.
COASTAL_STATES = {
    "AL": "01", "AK": "02", "CA": "06", "CT": "09", "DE": "10", "DC": "11",
    "FL": "12", "GA": "13", "HI": "15", "IL": "17", "IN": "18", "LA": "22",
    "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "NH": "33", "NJ": "34", "NY": "36", "NC": "37", "OH": "39", "OR": "41",
    "PA": "42", "RI": "44", "SC": "45", "TX": "48", "VA": "51", "WA": "53",
    "WI": "55", "AS": "60", "GU": "66", "MP": "69", "PR": "72", "VI": "78",
}
FIPS_TO_STATE = {v: k for k, v in COASTAL_STATES.items()}


# A state whose request FAILED is fatal; a state that was never asked for is
# not. That distinction is the whole lesson of the California incident: the
# run asked for CA, CA errored, the error was downgraded to a printed warning,
# and every table downstream described a coast with no Pacific in it and said
# nothing. A failure is the pull not doing what it was told. A chunk that was
# never requested is somebody choosing a scope, which is a different thing and
# stays a warning.
#
# Attempting every state before stopping is deliberate: the cache fills, so a
# re-run retries only what broke, and you learn about all the failures at once
# instead of one per run.
def _fail_on_failures(failed, what, allow_failed=False):
    if not failed:
        return
    listed = ", ".join(failed[:20]) + (" ..." if len(failed) > 20 else "")
    if allow_failed:
        print(f"\n  PROCEEDING WITHOUT {len(failed)} {what}: {listed}")
        print("  --allow-failed-states was given, so this is on the record as "
              "a choice.\n  Everything downstream describes the rest and "
              "must say so.")
        return
    sys.exit(
        f"\n{len(failed)} {what} failed and were not written: {listed}\n"
        "Nothing downstream is written from a partial pull, because a study "
        "missing a coast\nlooks exactly like a study of a smaller country. "
        "Every chunk that DID land is cached,\nso re-running the same "
        "command retries only what failed.\n"
        "If the failure is permanent and you mean to continue without it, "
        "say so:\n"
        "    --allow-failed-states")


class PullFailed(RuntimeError):
    """One request did not come back. Carries the host, because the two
    reasons this happens -- the service is down, and this machine is not
    allowed to reach it -- need different actions and look identical in a
    stack trace."""


# scan_cameras.TIMEOUT is 180 seconds, which is right for a camera frame and
# wrong for a WQP state query. California's station request answers in about
# 198 seconds -- it is the largest state in the study by a wide margin -- so
# every one of get_with_retry's four attempts timed out, pull_stations
# recorded CA as failed and carried on, and the study lost the entire Pacific
# coast to a constant. The "first N states all failed" guard did not fire
# because CA is fourth alphabetically and AK, AL and AS had already
# succeeded; the run printed "1 state(s) failed" and nothing read it.
#
# The timeout is raised HERE rather than in scan_cameras, because that module
# is shared with the camera pipeline and a 10-minute timeout is wrong for
# what it does. A slow answer from a government bulk-export endpoint is
# normal; a slow answer from a camera is a dead camera.
WQP_TIMEOUT = 600


@contextmanager
def _patched_timeout(seconds):
    """Hold scan_cameras.TIMEOUT at `seconds` for one call and put it back."""
    import scan_cameras as scan
    previous = scan.TIMEOUT
    scan.TIMEOUT = seconds
    try:
        yield
    finally:
        scan.TIMEOUT = previous


def _get(url, params, timeout=None):
    """scan_cameras' retry/backoff, imported here so a bare checkout can still
    import this module and run the offline tests.

    A national pull is thousands of requests over hours, so one failure must
    not end it. The callers below catch PullFailed, record the chunk as not
    pulled, and carry on -- and a re-run picks up exactly the chunks that are
    missing, because the cache is keyed by (state, year).
    """
    import requests
    import scan_cameras as scan
    try:
        with _patched_timeout(WQP_TIMEOUT if timeout is None else timeout):
            return scan.get_with_retry(url, params=params)
    except requests.RequestException as exc:
        host = url.split("/")[2]
        raise PullFailed(f"{host}: {type(exc).__name__}") from exc


def _read_csv_response(resp, what):
    if resp.status_code == 404:
        return pd.DataFrame()
    if resp.status_code == 400:
        sys.exit(f"WQP rejected the {what} request:\n{resp.text[:400]}")
    resp.raise_for_status()
    if resp.content[:2] == b"PK":
        sys.exit("Got a zip back despite zip=no.")
    text = resp.text.strip()
    if not text:
        return pd.DataFrame()
    return pd.read_csv(StringIO(text), low_memory=False)


def _chunk_path(kind, state, year=None):
    name = f"{kind}_{state}.csv" if year is None else f"{kind}_{state}_{year}.csv"
    return os.path.join(config.RAW_DIR, name)


def fetch_stations(state, site_types, characteristics=None):
    params = {
        "statecode": f"US:{COASTAL_STATES[state]}",
        "siteType": ";".join(site_types),
        "mimeType": "csv",
        "zip": "no",
    }
    if characteristics:
        params["characteristicName"] = ";".join(characteristics)
    return _read_csv_response(_get(WQP_STATION_URL, params), "station")


def fetch_results(state, year):
    params = {
        "statecode": f"US:{COASTAL_STATES[state]}",
        "siteType": ";".join(REQUEST_SITE_TYPES),
        "characteristicName": ";".join(config.ANALYTES.values()),
        "startDateLo": f"01-01-{year}",
        "startDateHi": f"12-31-{year}",
        "mimeType": "csv",
        "zip": "no",
    }
    return _read_csv_response(_get(WQP_RESULT_URL, params), "result")


def check_result_columns(frame, probe=False):
    """The repo's convention: prove the column names against a real response
    and stop with the actual names, rather than matching on ones that do not
    exist and reporting zero records."""
    if probe:
        print(f"\n=== ACTUAL COLUMNS ({len(frame.columns)}) ===")
        for column in frame.columns:
            sample = frame[column].dropna()
            example = "" if sample.empty else str(sample.iloc[0])[:44]
            print(f"  {column:<58} e.g. {example}")
        print("\n=== COLUMNS THIS MODULE READS ===")
        for column in RESULT_COLUMNS:
            state = "PRESENT" if column in frame.columns else "absent"
            need = "required" if column in REQUIRED_RESULT_COLUMNS else "optional"
            print(f"  {column:<58} {state:<8} ({need})")
    missing = [c for c in REQUIRED_RESULT_COLUMNS if c not in frame.columns]
    if missing:
        sys.exit(f"{len(missing)} required column(s) missing: {missing}\n"
                 "Update the COL_* constants at the top of wq/pull.py.")
    absent = [c for c in RESULT_COLUMNS if c not in frame.columns]
    if absent and not probe:
        print(f"  {len(absent)} optional hygiene column(s) absent: "
              f"{', '.join(absent)}")
        print("  the counts they support will read 0 — that is 'not reported', "
              "not 'none present'")


def coastal_only(stations):
    """Rows whose location type is one of the coastal types, plus the observed
    distribution of every type seen. The distribution is printed because the
    filter is the single place a wrong string silently empties the study."""
    if stations.empty:
        return stations, pd.Series(dtype=int)
    kinds = stations[STATION_TYPE].astype(str)
    observed = kinds.value_counts()
    wanted = tuple(w.lower() for w in config.COASTAL_SITE_TYPES)
    mask = kinds.str.lower().apply(lambda k: any(w in k for w in wanted))
    return stations[mask], observed


def normalize_stations(stations):
    out = pd.DataFrame({
        "station_id": stations[STATION_ID].astype(str),
        "station_name": stations.get(STATION_NAME, "").astype(str),
        "site_type": stations.get(STATION_TYPE, "").astype(str),
        "lat": pd.to_numeric(stations.get(STATION_LAT), errors="coerce"),
        "lon": pd.to_numeric(stations.get(STATION_LON), errors="coerce"),
        "organization": stations.get(STATION_ORG, "").astype(str),
    })
    if STATION_DRAINAGE in stations.columns:
        area = pd.to_numeric(stations[STATION_DRAINAGE], errors="coerce")
        unit = (stations.get(STATION_DRAINAGE_UNIT, "").astype(str)
                .str.strip().str.lower())
        # WQP reports these in square miles far more often than in km2, and a
        # 2.59x error in a stratification variable is not recoverable later.
        factor = pd.Series(np.nan, index=area.index)
        factor[unit.str.startswith("sq mi") | unit.isin(["mi2", "sqmi"])] = 2.58999
        factor[unit.str.startswith("sq km") | unit.isin(["km2", "sqkm"])] = 1.0
        out["wqp_drainage_area_km2"] = area * factor
        unknown = area.notna() & factor.isna()
        if unknown.any():
            seen = sorted(set(stations.loc[unknown, STATION_DRAINAGE_UNIT]
                              .astype(str)))[:4]
            print(f"  {int(unknown.sum())} station(s) report a drainage area "
                  f"in an unrecognised unit {seen} — left empty rather than "
                  "assumed")
    codes = stations.get(STATION_STATE)
    if codes is not None:
        out["state"] = (codes.astype(str).str.extract(r"(\d+)")[0]
                        .str.zfill(2).map(FIPS_TO_STATE))
    else:
        out["state"] = None
    # A (0, 0) coordinate is this data's stand-in for "unknown", not a point
    # in the Gulf of Guinea — the same trap the California CKAN pull hit.
    zeroed = ((out["lat"].abs() < 0.001) & (out["lon"].abs() < 0.001))
    if zeroed.any():
        print(f"  dropped {int(zeroed.sum())} stations at (0, 0)")
    return out[~zeroed & out["lat"].notna() & out["lon"].notna()]


def pull_stations(states=None, refresh=False, probe=False,
                  allow_failed=False):
    """Coastal recreational stations, one cached CSV per state."""
    states = states or sorted(COASTAL_STATES)
    frames, failed = [], []
    os.makedirs(config.RAW_DIR, exist_ok=True)
    for index, state in enumerate(states, 1):
        path = _chunk_path("stations", state)
        if os.path.exists(path) and not refresh:
            frames.append(pd.read_csv(path, low_memory=False))
            print(f"  [{index}/{len(states)}] {state}: cached")
            continue
        try:
            raw = fetch_stations(state, REQUEST_SITE_TYPES,
                                 list(config.ANALYTES.values()))
        except PullFailed as exc:
            print(f"  [{index}/{len(states)}] {state}: FAILED ({exc})")
            failed.append(state)
            if len(failed) == index and index >= 3:
                sys.exit(f"\nThe first {index} states all failed with "
                         f"{exc}.\nThat is not a data problem — nothing is "
                         "reaching the Water Quality Portal from this "
                         "machine.\nCheck connectivity (and any egress "
                         "policy) before re-running; every state already on "
                         "disk is kept, so a re-run resumes.")
            continue
        if probe and not raw.empty:
            print(f"\n{state} station columns: {list(raw.columns)}")
            return raw
        raw.to_csv(path, index=False)
        print(f"  [{index}/{len(states)}] {state}: {len(raw)} stations")
        frames.append(raw)
        time.sleep(config.REQUEST_PAUSE)

    _fail_on_failures(failed, 'state(s)', allow_failed)

    # stations.csv is the study's station table, not a report on this
    # invocation, so it is rebuilt from EVERY cached chunk rather than from
    # the states this call happened to ask for. `--states CA` used to rewrite
    # the file with California alone and silently drop the other 35 states,
    # which is a one-command way to shrink the study and leaves no trace
    # except a smaller row count.
    cached = sorted(glob.glob(os.path.join(config.RAW_DIR, "stations_*.csv")))
    frames, on_disk = [], []
    for path in cached:
        state = os.path.basename(path)[len("stations_"):-len(".csv")]
        try:
            frame = pd.read_csv(path, low_memory=False)
        except (OSError, pd.errors.ParserError) as exc:
            print(f"  {state}: cached chunk unreadable ({exc}) — skipped")
            continue
        if not frame.empty:
            frames.append(frame)
            on_disk.append(state)
    absent = [s for s in sorted(COASTAL_STATES) if s not in on_disk]
    print(f"\nassembling stations.csv from {len(on_disk)} cached state "
          f"chunk(s) of {len(COASTAL_STATES)}")
    if absent:
        print(f"  {len(absent)} coastal state(s) have no chunk on disk: "
              f"{', '.join(absent)}")
        print("  Every station count below describes the states listed as "
              "present, and no others.")
    if not frames:
        sys.exit("No stations returned for any state.")
    every = pd.concat(frames, ignore_index=True, sort=False)
    kept, observed = coastal_only(every)
    print(f"\n{len(every)} stations returned, {len(kept)} coastal "
          f"({', '.join(config.COASTAL_SITE_TYPES)})")
    print("\nlocation types actually returned:")
    for kind, count in observed.head(15).items():
        mark = "  <- kept" if kind in kept[STATION_TYPE].unique() else ""
        print(f"  {str(kind)[:46]:<46} {count:>7}{mark}")
    out = normalize_stations(kept).drop_duplicates(subset="station_id")
    path = os.path.join(config.DATA_DIR, "stations.csv")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\n{len(out)} distinct coastal stations -> {path}")
    return out


def pull_results(states=None, years=None, refresh=False, probe=False,
                 allow_failed=False):
    """Bacteria results, one cached CSV per (state, year).

    Chunked this finely because the national pull is long enough that
    something will interrupt it, and a year of one state is the most any
    single failure should cost.
    """
    states = states or sorted(COASTAL_STATES)
    this_year = date.today().year
    years = years or list(range(this_year - config.YEARS_BACK + 1, this_year + 1))
    os.makedirs(config.RAW_DIR, exist_ok=True)

    total = len(states) * len(years)
    done, pulled, checked = 0, 0, False
    failures = []
    started = time.time()
    for state in states:
        for year in years:
            done += 1
            path = _chunk_path("results", state, year)
            if os.path.exists(path) and not refresh:
                continue
            try:
                frame = fetch_results(state, year)
            except PullFailed as exc:
                print(f"  [{done}/{total}] {state} {year}: FAILED ({exc})")
                failures.append(f"{state} {year}")
                if len(failures) == done and done >= 3:
                    sys.exit(f"\nThe first {done} chunks all failed with "
                             f"{exc}.\nNothing is reaching the Water Quality "
                             "Portal from this machine — this is not a data "
                             "problem.\nEvery chunk already on disk is kept, "
                             "so a re-run resumes where this stopped.")
                continue
            if not checked and not frame.empty:
                check_result_columns(frame, probe=probe)
                checked = True
                if probe:
                    return frame
            keep = [c for c in RESULT_COLUMNS if c in frame.columns]
            frame[keep].to_csv(path, index=False)
            pulled += 1
            elapsed = time.time() - started
            rate = elapsed / max(pulled, 1)
            left = (total - done) * rate
            print(f"  [{done}/{total}] {state} {year}: {len(frame)} results"
                  f"   ~{left / 60:.0f} min left")
            time.sleep(config.REQUEST_PAUSE)
    print(f"\n{pulled} chunks pulled, "
          f"{total - pulled - len(failures)} already on disk, "
          f"{len(failures)} failed")
    _fail_on_failures(failures, 'chunk(s)', allow_failed)
    return load_raw_results(states, years)


def load_raw_results(states=None, years=None):
    """Every cached chunk, concatenated. Missing chunks are reported, because
    a silently short national pull looks exactly like a small country."""
    states = states or sorted(COASTAL_STATES)
    this_year = date.today().year
    years = years or list(range(this_year - config.YEARS_BACK + 1, this_year + 1))
    frames, missing = [], []
    for state in states:
        for year in years:
            path = _chunk_path("results", state, year)
            if not os.path.exists(path):
                missing.append(f"{state} {year}")
                continue
            frame = pd.read_csv(path, low_memory=False)
            if not frame.empty:
                frame["pull_state"] = state
                frames.append(frame)
    if missing:
        print(f"  {len(missing)} chunk(s) never pulled: "
              f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def pull_ca_ckan(codes=None, start=None):
    """California's own bacteria resource, via the existing CKAN puller.

    CA stations appear in WQP as well, so wq/clean.py deduplicates the two.
    This is pulled anyway because the CKAN resource carries stations and a
    record depth WQP's California feed does not always match.
    """
    import pull_observations as obs
    start = start or date(date.today().year - config.YEARS_BACK, 1, 1)
    if codes is None:
        print("  no station codes given — pulling the CKAN station list first")
        stations = pd.DataFrame(obs._fetch_all(
            __import__("find_candidate_sites").CA_CKAN_RESOURCE_ID))
        codes = sorted(set(stations.get("Station_Name", pd.Series(dtype=str))
                           .dropna().astype(str)))
    frame = obs.pull_water_quality(list(codes), start)
    if frame is None or frame.empty:
        print("  CKAN returned nothing")
        return pd.DataFrame()
    path = os.path.join(config.DATA_DIR, "ca_ckan_results.csv")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"  {len(frame)} CKAN results -> {path}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", action="store_true")
    parser.add_argument("--results", action="store_true")
    parser.add_argument("--ckan", action="store_true")
    parser.add_argument("--states", help="comma-separated, e.g. CA,FL")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="print the real column names and stop")
    parser.add_argument("--allow-failed-states", action="store_true",
                        help="continue after a state or chunk fails, instead "
                             "of stopping. Says on the record that the study "
                             "is missing what failed.")
    args = parser.parse_args()

    states = None
    if args.states:
        states = [s.strip().upper() for s in args.states.split(",") if s.strip()]
        unknown = [s for s in states if s not in COASTAL_STATES]
        if unknown:
            sys.exit(f"not coastal states in this study: {unknown}")

    if not any((args.stations, args.results, args.ckan)):
        parser.error("give at least one of --stations --results --ckan")
    if args.stations:
        pull_stations(states, refresh=args.refresh, probe=args.probe,
                      allow_failed=args.allow_failed_states)
    if args.results:
        pull_results(states, refresh=args.refresh, probe=args.probe,
                     allow_failed=args.allow_failed_states)
    if args.ckan:
        pull_ca_ckan()


if __name__ == "__main__":
    main()
