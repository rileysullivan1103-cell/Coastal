"""Join the CKAN stations table to the bacteria results table, safely.

verify_ckan_join.py established that stations.Station_Name <-> results.StationCode
is the correct key: 97% of joined pairs agree on location to within 1 km, and
the join does not multiply rows.

The remaining 3% is why this module exists. Short agency-local codes are not
unique statewide -- code "1100" is Rincon Beach in Santa Barbara and also
Crescent City, 963 km apart -- so a raw join quietly attributes one beach's
bacteria readings to another. A few rows also carry (0,0) placeholder
coordinates or plainly wrong ones.

Every pair is therefore validated against the thing the key cannot fake: both
tables independently recording where the station is. Pairs that disagree are
dropped, not silently kept.
"""

import pandas as pd

import find_candidate_sites as f

STATION_KEY = "Station_Name"
RESULT_KEY = "StationCode"

# Pairs further apart than this are treated as a mis-join. 1 km is generous for
# a beach monitoring station: the median observed gap is 0.03 km, so this
# rejects real disagreements without tripping on coordinate rounding.
MAX_JOIN_DISAGREEMENT_KM = 1.0


def _clean_coords(df, lat_col, lon_col):
    out = df.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out = out.dropna(subset=[lat_col, lon_col])
    # (0, 0) is the missing-coordinate placeholder in both tables.
    return out[(out[lat_col] != 0) & (out[lon_col] != 0)]


def join_stations_to_results(stations, results, verbose=True):
    """Inner-join the two tables, keeping only pairs whose coordinates agree.

    stations: rows from the CKAN stations resource (needs Station_Name plus
        f.CA_CKAN_LAT_COL / f.CA_CKAN_LON_COL).
    results:  rows from the bacteria results resource (needs StationCode,
        TargetLatitude, TargetLongitude).

    Returns the joined frame with a 'join_gap_km' column, plus a dict of counts
    so a caller can see what was discarded rather than having to guess.
    """
    left = _clean_coords(stations, f.CA_CKAN_LAT_COL, f.CA_CKAN_LON_COL)
    right = _clean_coords(results, "TargetLatitude", "TargetLongitude")
    left = left.assign(_key=left[STATION_KEY].astype(str).str.strip())
    right = right.assign(_key=right[RESULT_KEY].astype(str).str.strip())

    merged = left.merge(right, on="_key", how="inner", suffixes=("_stn", "_res"))
    joined = len(merged)
    if joined == 0:
        return merged.assign(join_gap_km=pd.Series(dtype=float)), {
            "joined": 0, "kept": 0, "rejected": 0}

    merged["join_gap_km"] = f.haversine_km(
        merged[f.CA_CKAN_LAT_COL].to_numpy(), merged[f.CA_CKAN_LON_COL].to_numpy(),
        merged["TargetLatitude"].to_numpy(), merged["TargetLongitude"].to_numpy())

    keep = merged["join_gap_km"] <= MAX_JOIN_DISAGREEMENT_KM
    rejected = merged[~keep]
    stats = {"joined": joined, "kept": int(keep.sum()), "rejected": int((~keep).sum())}

    if verbose and stats["rejected"]:
        print(f"  rejected {stats['rejected']}/{joined} joined pairs whose two "
              f"sources disagree by more than {MAX_JOIN_DISAGREEMENT_KM} km:")
        for _, row in rejected.nlargest(5, "join_gap_km").iterrows():
            print(f"    {row['_key']:<12} {str(row.get('Beach_Name'))[:28]:<30} "
                  f"{row['join_gap_km']:>10.1f} km")

    return merged[keep].drop(columns="_key").reset_index(drop=True), stats
