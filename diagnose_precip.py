"""Explain why a GHCND station returned no precipitation.

An empty PRCP response does NOT mean it did not rain -- GHCND records PRCP = 0
on dry days, so an empty result means the station published nothing. There are
three reasons that can happen, and this tells them apart:

  1. The station stopped reporting     -> maxdate is old
  2. It reports, but not PRCP          -> PRCP absent from its datatype list
  3. It reports PRCP, just not lately  -> PRCP maxdate is old while the station
                                          itself looks current

    export $(grep -v '^#' .env | xargs)
    python diagnose_precip.py
    python diagnose_precip.py GHCND:US1CASD0092 GHCND:USC00047916
"""

import os
import sys
from datetime import datetime

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


def diagnose(station_id, token):
    print(f"\n=== {station_id} ===")

    meta = get(f"stations/{station_id}", token)
    if not meta:
        print("  station not found in CDO at all")
        return
    maxdate, mindate = meta.get("maxdate"), meta.get("mindate")
    print(f"  name          {meta.get('name')}")
    print(f"  reporting     {mindate} to {maxdate}")
    print(f"  datacoverage  {meta.get('datacoverage')}  (lifetime, not recent)")

    if maxdate:
        stale_days = (datetime.now() - datetime.strptime(maxdate, "%Y-%m-%d")).days
        print(f"  last report   {stale_days} days ago")
        if stale_days > 90:
            print(f"  => OFFLINE. Nothing to pull; datacoverage "
                  f"{meta.get('datacoverage')} is a lifetime figure and says "
                  "nothing about whether it still reports.")

    # Which measurements does this station actually publish?
    types = get("datatypes", token, stationid=station_id, limit=1000)
    ids = [d["id"] for d in (types or {}).get("results", [])]
    if not ids:
        print("  datatypes     none listed")
    else:
        print(f"  datatypes     {len(ids)}: {', '.join(sorted(ids)[:12])}"
              + (" ..." if len(ids) > 12 else ""))
        if "PRCP" not in ids:
            print("  => DOES NOT REPORT PRCP. This station was matched on "
                  "proximity and datacoverage, but publishes other elements "
                  "only. It could never have supplied rainfall.")

    # And is PRCP itself current, even if the station is?
    prcp = get("datatypes/PRCP", token, stationid=station_id)
    if prcp and prcp.get("maxdate"):
        print(f"  PRCP through  {prcp.get('mindate')} to {prcp.get('maxdate')}")


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
