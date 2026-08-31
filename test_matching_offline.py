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
        # ~0.7 km from the Charleston camera — inside MAX_WQ_DISTANCE_KM.
        {"MonitoringLocationIdentifier": "SCDHEC-MD-123",
         "LatitudeMeasure": 32.785, "LongitudeMeasure": -79.925},
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
    check_wq_radius()

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
    # Shaped like the real CKAN stations resource.
    ckan = pd.DataFrame([
        {"Station_id": 101, "Station_UpperLat": 33.20, "Station_UpperLon": -117.39,
         "Beach_Name": "Oceanside Harbor Beach"},
        {"Station_id": 102, "Station_UpperLat": 36.95, "Station_UpperLon": -122.03,
         "Beach_Name": "Cowell Beach"},
        # Far from every camera — must not be matched to either.
        {"Station_id": 103, "Station_UpperLat": 40.80, "Station_UpperLon": -124.16,
         "Beach_Name": "Eureka"},
    ])
    ranked = f.rank_candidate_sites(cams, buoys, precip, empty_wq, ca_wq_df=ckan)

    by_name = {r["camera_name"]: r for _, r in ranked.iterrows()}
    assert by_name["Oceanside Pier, CA"]["wq_station_id"] == 101
    assert by_name["Santa Cruz, CA"]["wq_station_id"] == 102
    assert by_name["Santa Cruz, CA"]["wq_station_name"] == "Cowell Beach", \
        "the beach label should reach the output"
    assert ranked["wq_source_confirmed"].all(), "measured sites should be confirmed"
    assert (ranked["wq_distance_km"] > 0).all(), "measured sites need a real distance"
    assert ranked["has_all_four"].all()

    # Region helpers: the two extent formats have opposite coordinate order.
    assert f.region_extent("california") == "32.5,-124.5,42.0,-117.0"
    assert f.region_bbox("california") == "-124.5,32.5,-117.0,42.0"
    assert f.in_region(37.8, -122.4, "california")
    assert not f.in_region(32.78, -79.92, "california")
    print("\nCalifornia assumed + measured paths and region helpers OK.")


def check_wq_radius():
    """Water quality uses its own radius, much tighter than the precip one: a
    bacteria reading only speaks for the water it came from."""
    cam = pd.DataFrame([
        {"camera_name": "Radius Probe, CA", "latitude": 33.19, "longitude": -117.38},
    ])
    buoys = pd.DataFrame([{"Station": "46086", "Lat": 33.05, "Lon": -117.60}])
    precip = pd.DataFrame([
        {"id": "GHCND:X", "latitude": 33.21, "longitude": -117.35, "datacoverage": 0.99},
    ])
    empty_wq = pd.DataFrame(
        columns=["MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure"])

    def station_at(lat):
        return pd.DataFrame([{"Station_id": 1, "Station_UpperLat": lat,
                              "Station_UpperLon": -117.38, "Beach_Name": "Probe Beach"}])

    # ~1.1 km away: inside the radius.
    near = f.rank_candidate_sites(cam, buoys, precip, empty_wq,
                                  ca_wq_df=station_at(33.20)).iloc[0]
    assert near["wq_station_id"] == 1, near["wq_station_id"]
    assert near["wq_distance_km"] < f.MAX_WQ_DISTANCE_KM
    assert near["has_all_four"]

    # ~3.0 km away: outside it, and must not be matched even though it is well
    # inside the 30 km precip radius.
    far = f.rank_candidate_sites(cam, buoys, precip, empty_wq,
                                 ca_wq_df=station_at(33.217)).iloc[0]
    assert far["wq_station_id"] is None, far["wq_station_id"]
    assert not far["has_all_four"], "a station beyond the WQ radius must not qualify a site"

    # The precip station stayed matched throughout — only WQ tightened.
    assert far["precip_station_id"] == "GHCND:X"
    print("\nWater quality radius boundary OK.")


if __name__ == "__main__":
    main()
