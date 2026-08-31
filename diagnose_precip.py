"""Explain why a GHCND station returned no precipitation.

An empty PRCP response does NOT mean it did not rain -- GHCND records PRCP = 0
on dry days, so an empty result means the station published nothing. There are
two reasons that can happen, and this tells them apart:

  1. The station stopped reporting  -> maxdate is old
  2. It is current but serves no rainfall

It settles this by requesting real PRCP data and counting the rows, not by
reading station metadata. CDO's /datatypes?stationid= listing omits PRCP for
stations that demonstrably serve it, and /datatypes/PRCP?stationid= ignores the
station filter entirely, so both give confidently wrong answers.

    export $(grep -v '^#' .env | xargs)
    python diagnose_precip.py
    python diagnose_precip.py GHCND:US1CASD0092 GHCND:USC00047916
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import requests

CDO = "https://www.ncei.noaa.gov/cdo-web/api/v2"
SITES_CSV = "candidate_sites_ranked.csv"


def get(path, token, **params):
    resp = requests.get(f"{CDO}/{path}", headers={"token": token},
                        params=params, timeout=60)
    if resp.status_code != 200:
        return None
    return resp.json() or None


def prcp_rows(station_id, token, days=120):
    """Ask for real PRCP data and count what comes back.

    This replaces two metadata checks that were confidently wrong. CDO's
    /datatypes?stationid= listing omitted PRCP for stations that demonstrably
    serve it, and /datatypes/PRCP?stationid= ignored the station filter and
    returned the global 1781-to-present range for every station. Requesting the
    data itself is ground truth; station metadata about it is not.
    """
    end = datetime.now()
    start = end - timedelta(days=days)
    payload = get("data", token, datasetid="GHCND", stationid=station_id,
                  datatypeid="PRCP", units="metric", limit=1000,
                  startdate=start.strftime("%Y-%m-%d"),
                  enddate=end.strftime("%Y-%m-%d"))
    if not payload:
        return 0, None
    rows = payload.get("results", [])
    if not rows:
        return 0, None
    values = [r.get("value") for r in rows if r.get("value") is not None]
    return len(rows), (sum(values) if values else 0.0)


def diagnose(station_id, token):
    print(f"\n=== {station_id} ===")

    meta = get(f"stations/{station_id}", token)
    if not meta:
        print("  station not found in CDO at all")
        return
    maxdate = meta.get("maxdate")
    print(f"  name          {meta.get('name')}")
    print(f"  reporting     {meta.get('mindate')} to {maxdate}")
    print(f"  datacoverage  {meta.get('datacoverage')}  (lifetime, not recent)")

    stale_days = None
    if maxdate:
        stale_days = (datetime.now() - datetime.strptime(maxdate, "%Y-%m-%d")).days
        print(f"  last report   {stale_days} days ago")

    n, total = prcp_rows(station_id, token)
    print(f"  PRCP last 120 days: {n} records"
          + (f", {total:.1f} mm total" if n else ""))

    if n:
        print("  => WORKING. It reports rainfall, including zeros on dry days.")
    elif stale_days is not None and stale_days > 120:
        print(f"  => OFFLINE. Last reported {stale_days} days ago. "
              "datacoverage is a lifetime figure and says nothing about this.")
    else:
        print("  => Station looks current but returned no PRCP. It reports "
              "other elements and not rainfall, or PRCP has lapsed.")


def main():
    token = os.environ.get("NOAA_CDO_TOKEN")
    if not token:
        sys.exit("NOAA_CDO_TOKEN is not set.")

    stations = sys.argv[1:]
    if not stations:
        if not os.path.exists(SITES_CSV):
            sys.exit(f"No station ids given and {SITES_CSV} not found.")
        df = pd.read_csv(SITES_CSV)
        stations = sorted(set(df["precip_station_id"].dropna()))
        print(f"Diagnosing {len(stations)} stations from {SITES_CSV}")

    for station_id in stations:
        diagnose(station_id, token)

    print("\nNote: an empty PRCP response never means 'it did not rain'. "
          "GHCND records PRCP = 0 for dry days at a reporting station.")


if __name__ == "__main__":
    main()
