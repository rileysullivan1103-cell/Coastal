"""Verify how the CKAN stations table joins to the bacteria results table.

The suspected key is stations.Station_Name <-> results.StationCode, but both
are literally "0" on the first record, which is exactly the shape of thing that
silently joins every station to every result. This checks it three ways:

  1. Cardinality  — is Station_Name actually unique, or does "0" repeat?
  2. Overlap      — do the key sets actually intersect?
  3. Coordinates  — for each matched pair, do the two sources put the station
                    in the same place? This is the real test. Names can agree
                    by coincidence; coordinates from two independent tables
                    agreeing to within a few hundred metres cannot.

    python verify_ckan_join.py
"""

import sys

import pandas as pd
import requests

import find_candidate_sites as f

BASE = f.CA_CKAN_BASE
STATIONS_ID = f.CA_CKAN_RESOURCE_ID
RESULTS_ID = "15a63495-8d9f-4a49-b43a-3092ef3106b9"

STATION_KEY = "Station_Name"
RESULT_KEY = "StationCode"
AGREEMENT_KM = 1.0


def distinct_result_stations():
    """One row per station from the 627k-row results table.

    datastore_search_sql does this in a single request. If the instance has SQL
    disabled, fall back to paging, which is slow but correct.
    """
    sql = (f'SELECT DISTINCT "{RESULT_KEY}", "StationName", '
           f'"TargetLatitude", "TargetLongitude" FROM "{RESULTS_ID}"')
    resp = requests.get(f"{BASE}/datastore_search_sql", params={"sql": sql}, timeout=300)
    if resp.status_code == 200 and resp.json().get("success"):
        records = resp.json()["result"]["records"]
        print(f"  {len(records)} distinct stations in the results table (via SQL)")
        return pd.DataFrame(records)

    print("  datastore_search_sql unavailable; paging the results table instead "
          "(this pulls ~627k rows and will take a few minutes)")
    rows, offset = [], 0
    while True:
        r = requests.get(f"{BASE}/datastore_search",
                         params={"resource_id": RESULTS_ID, "limit": 10000,
                                 "offset": offset}, timeout=300)
        r.raise_for_status()
        batch = r.json().get("result", {}).get("records", [])
        if not batch:
            break
        rows.extend({k: rec.get(k) for k in
                     (RESULT_KEY, "StationName", "TargetLatitude", "TargetLongitude")}
                    for rec in batch)
        offset += 10000
        print(f"    {len(rows)} rows...")
    df = pd.DataFrame(rows).drop_duplicates(subset=[RESULT_KEY])
    print(f"  {len(df)} distinct stations in the results table")
    return df


def all_stations():
    """The stations table, unfiltered — we want duplicates visible here."""
    rows, offset = [], 0
    while True:
        r = requests.get(f"{BASE}/datastore_search",
                         params={"resource_id": STATIONS_ID, "limit": 1000,
                                 "offset": offset}, timeout=120)
        r.raise_for_status()
        result = r.json().get("result", {})
        batch = result.get("records", [])
        if not batch:
            break
        rows.extend(batch)
        if len(rows) >= (result.get("total") or 0):
            break
        offset += 1000
    print(f"  {len(rows)} rows in the stations table")
    return pd.DataFrame(rows)


def main():
    print("Pulling both tables...")
    stations = all_stations()
    results = distinct_result_stations()

    print("\n=== 1. CARDINALITY ===")
    for name, df, key in (("stations", stations, STATION_KEY),
                          ("results ", results, RESULT_KEY)):
        keys = df[key].astype(str).str.strip()
        dupes = keys.value_counts()
        repeated = dupes[dupes > 1]
        print(f"  {name}: {len(keys)} rows, {keys.nunique()} distinct {key!r}")
        if len(repeated):
            print(f"    {len(repeated)} values repeat; worst: "
                  f"{dict(repeated.head(3))}")

    # A key that repeats on BOTH sides multiplies rows on join.
    s_keys = set(stations[STATION_KEY].astype(str).str.strip())
    r_keys = set(results[RESULT_KEY].astype(str).str.strip())

    print("\n=== 2. OVERLAP ===")
    print(f"  {len(s_keys & r_keys)} keys in both, "
          f"{len(s_keys - r_keys)} stations-only, {len(r_keys - s_keys)} results-only")
    if not (s_keys & r_keys):
        sys.exit("\nThe key sets do not intersect at all — this is the wrong join.")

    print("\n=== 3. COORDINATE AGREEMENT (the real test) ===")
    left = stations[[STATION_KEY, "Beach_Name", f.CA_CKAN_LAT_COL, f.CA_CKAN_LON_COL]].copy()
    left["_key"] = left[STATION_KEY].astype(str).str.strip()
    right = results[[RESULT_KEY, "StationName", "TargetLatitude", "TargetLongitude"]].copy()
    right["_key"] = right[RESULT_KEY].astype(str).str.strip()

    merged = left.merge(right, on="_key", how="inner")
    print(f"  {len(left)} x {len(right)} rows joined to {len(merged)} — "
          f"{'1:1-ish' if len(merged) <= max(len(left), len(right)) * 1.1 else 'ROW EXPLOSION'}")

    for col in (f.CA_CKAN_LAT_COL, f.CA_CKAN_LON_COL, "TargetLatitude", "TargetLongitude"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged = merged.dropna(subset=[f.CA_CKAN_LAT_COL, f.CA_CKAN_LON_COL,
                                   "TargetLatitude", "TargetLongitude"])
    if merged.empty:
        sys.exit("No joined pair had coordinates on both sides — cannot verify.")

    merged["gap_km"] = f.haversine_km(
        merged[f.CA_CKAN_LAT_COL].to_numpy(), merged[f.CA_CKAN_LON_COL].to_numpy(),
        merged["TargetLatitude"].to_numpy(), merged["TargetLongitude"].to_numpy())

    agree = (merged["gap_km"] <= AGREEMENT_KM).mean()
    print(f"  median gap {merged['gap_km'].median():.2f} km, "
          f"{agree:.0%} of pairs within {AGREEMENT_KM} km")

    pd.set_option("display.width", 200)
    print("\n  Worst 10 disagreements:")
    print(merged.nlargest(10, "gap_km")[
        ["_key", "Beach_Name", "StationName", "gap_km"]].to_string(index=False))

    print("\n=== VERDICT ===")
    if agree > 0.95:
        print(f"  {STATION_KEY} <-> {RESULT_KEY} looks correct: the two tables put "
              "almost every station in the same place.")
    elif agree > 0.5:
        print(f"  PARTIALLY correct — {1 - agree:.0%} of pairs disagree on location. "
              "Inspect the worst cases above before relying on this join.")
    else:
        print("  This join is WRONG. The tables disagree on where most stations "
              "are. Try joining on coordinates instead, or parse the beach name "
              "out of the results table's StationName field.")


if __name__ == "__main__":
    main()
