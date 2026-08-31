"""Offline smoke test for the local matching/ranking step.

Uses synthetic fixtures shaped like the real sources (NDBC's capitalized
'Station'/'Lat'/'Lon', CDO's 'id'/'latitude'/'longitude'/'datacoverage', WQP's
'MonitoringLocationIdentifier'/'LatitudeMeasure'/'LongitudeMeasure'), so it
exercises the matching logic without touching the network.

    python test_matching_offline.py
"""

import pandas as pd

import find_candidate_sites as f


def main():
    cams = pd.DataFrame([
        {"camera_name": "Charleston Harbor, SC", "latitude": 32.78, "longitude": -79.92},
        {"camera_name": "Oceanside Pier, CA", "latitude": 33.19, "longitude": -117.38},
        {"camera_name": "Middle Of Nowhere, ND", "latitude": 47.00, "longitude": -100.00},
    ])
    buoys = pd.DataFrame([
        {"Station": "41004", "Lat": 32.70, "Lon": -79.50},   # ~40 km from Charleston
        {"Station": "46086", "Lat": 33.05, "Lon": -117.60},  # ~26 km from Oceanside
    ])
    precip = pd.DataFrame([
        {"id": "GHCND:USW00013880", "latitude": 32.90, "longitude": -80.04, "datacoverage": 1.00},
        {"id": "GHCND:USW00093107", "latitude": 33.21, "longitude": -117.35, "datacoverage": 0.95},
        # Closer to Charleston than the one above, but below the coverage floor.
        {"id": "GHCND:USBADCOVER", "latitude": 32.79, "longitude": -79.93, "datacoverage": 0.42},
    ])
    wq = pd.DataFrame([
        {"MonitoringLocationIdentifier": "SCDHEC-MD-123",
         "LatitudeMeasure": 32.80, "LongitudeMeasure": -79.90},
    ])

    ranked = f.rank_candidate_sites(cams, buoys, precip, wq)
    pd.set_option("display.width", 200)
    print(ranked[["camera_name", "buoy_id", "buoy_distance_km", "precip_station_id",
                  "precip_datacoverage", "wq_station_id", "wq_distance_km",
                  "wq_source_confirmed", "has_all_four", "combined_score"]].to_string(index=False))

    by_name = {r["camera_name"]: r for _, r in ranked.iterrows()}

    sc = by_name["Charleston Harbor, SC"]
    assert sc["buoy_id"] == "41004", sc["buoy_id"]
    assert sc["precip_station_id"] == "GHCND:USW00013880", \
        "the sub-threshold station must be filtered out even though it is closer"
    assert sc["wq_station_id"] == "SCDHEC-MD-123"
    assert sc["has_all_four"]

    ca = by_name["Oceanside Pier, CA"]
    assert ca["wq_station_id"] == "CA_CKAN_ASSUMED", "CA fallback should mark the source"
    assert not ca["wq_source_confirmed"], "an assumed site is not a confirmed one"
    assert ca["has_all_four"]
    # An assumed site must not be pushed below a fully-measured qualifying site
    # just because one distance is unknown.
    assert ca["combined_score"] > by_name["Middle Of Nowhere, ND"]["combined_score"]

    nd = by_name["Middle Of Nowhere, ND"]
    assert nd["buoy_id"] is None and not nd["has_all_four"], "inland site must not qualify"

    assert ranked.iloc[-1]["camera_name"] == "Middle Of Nowhere, ND", \
        "the non-qualifying site should sort last"

    check_california_only_path()

    print("\nAll offline assertions passed.")


def check_california_only_path():
    """California has two paths: an assumed one (no CKAN resource configured)
    and a measured one. Both must rank; only the second may claim confirmation.
    """
    cams = pd.DataFrame([
        {"camera_name": "Oceanside Pier, CA", "latitude": 33.19, "longitude": -117.38},
        {"camera_name": "Santa Cruz, CA", "latitude": 36.96, "longitude": -122.02},
    ])
    buoys = pd.DataFrame([
        {"Station": "46086", "Lat": 33.05, "Lon": -117.60},
        {"Station": "46042", "Lat": 36.79, "Lon": -122.40},
    ])
    precip = pd.DataFrame([
        {"id": "GHCND:USW00093107", "latitude": 33.21, "longitude": -117.35, "datacoverage": 0.95},
        {"id": "GHCND:USW00023277", "latitude": 36.98, "longitude": -122.03, "datacoverage": 0.99},
    ])
    empty_wq = pd.DataFrame(
        columns=["MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure"])

    # --- assumed path: no CKAN data ---
    ranked = f.rank_candidate_sites(cams, buoys, precip, empty_wq, ca_wq_df=None)
    assert ranked["has_all_four"].all(), "assumed CA sites should still qualify"
    assert (ranked["wq_station_id"] == "CA_CKAN_ASSUMED").all()
    assert not ranked["wq_source_confirmed"].any(), \
        "an assumed site must NOT be reported as confirmed"
    assert ranked["wq_distance_km"].isna().all(), \
        "an assumed site has no measured distance"

    # --- measured path: real CKAN stations ---
    ckan = pd.DataFrame([
        {"StationCode": "CA-SD-001", "TargetLatitude": 33.20, "TargetLongitude": -117.39},
        {"StationCode": "CA-SC-002", "TargetLatitude": 36.95, "TargetLongitude": -122.03},
        # Far from every camera — must not be matched to either.
        {"StationCode": "CA-FAR-003", "TargetLatitude": 40.80, "TargetLongitude": -124.16},
    ])
    f.CA_CKAN_LAT_COL, f.CA_CKAN_LON_COL, f.CA_CKAN_ID_COL = (
        "TargetLatitude", "TargetLongitude", "StationCode")
    try:
        ranked = f.rank_candidate_sites(cams, buoys, precip, empty_wq, ca_wq_df=ckan)
    finally:
        f.CA_CKAN_LAT_COL = f.CA_CKAN_LON_COL = f.CA_CKAN_ID_COL = None

    by_name = {r["camera_name"]: r for _, r in ranked.iterrows()}
    assert by_name["Oceanside Pier, CA"]["wq_station_id"] == "CA-SD-001"
    assert by_name["Santa Cruz, CA"]["wq_station_id"] == "CA-SC-002"
    assert ranked["wq_source_confirmed"].all(), "measured sites should be confirmed"
    assert (ranked["wq_distance_km"] > 0).all(), "measured sites need a real distance"
    assert ranked["has_all_four"].all()

    # Region helpers: the two extent formats have opposite coordinate order.
    assert f.region_extent("california") == "32.5,-124.5,42.0,-117.0"
    assert f.region_bbox("california") == "-124.5,32.5,-117.0,42.0"
    assert f.in_region(37.8, -122.4, "california")
    assert not f.in_region(32.78, -79.92, "california")
    print("\nCalifornia assumed + measured paths and region helpers OK.")


if __name__ == "__main__":
    main()
