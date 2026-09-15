"""Coastline geometry and the spatial covariates, against known shapes.

Synthetic coastlines with answers you can work out by hand: a straight coast,
a circular bay, a circular headland, and a reversed way. If curvature does
not come out as -1/R inside a bay of radius R, nothing downstream that
stratifies on enclosure means anything.

The reversed-way case is the one that matters most in practice. OSM coastline
ways are edited piecemeal and a locally reversed one inverts land and sea
exactly where it is wrong -- silently, and only there. The covariates from
such a coastline would be confidently backwards, which is worse than missing,
so the sanity check has to catch it.

    python wq/test_wq_geo_offline.py
"""

import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wq import config, geo, layers, manifest, review, spatial, strata  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name
          + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def near(value, target, tolerance):
    return value is not None and abs(value - target) <= tolerance


# --- shapes, in local metres -----------------------------------------------

def straight_coast(length_m=40000.0):
    """A north-south coast at x=0, drawn northward. Land west, water east."""
    return [[(0.0, -length_m / 2), (0.0, length_m / 2)]]


def ring(radius_m, clockwise, n=720):
    points = []
    for index in range(n + 1):
        angle = 2 * math.pi * index / n
        if clockwise:
            angle = -angle
        points.append((radius_m * math.cos(angle), radius_m * math.sin(angle)))
    return [points]


def bay(radius_m=3000.0):
    """Water inside, land outside: drawn clockwise so land stays on the left."""
    return ring(radius_m, clockwise=True)


def headland(radius_m=3000.0):
    """Land inside, water outside: an island, drawn counter-clockwise."""
    return ring(radius_m, clockwise=False)


# ---------------------------------------------------------------------------

def test_land_and_water():
    print("\nwhich side is land")
    coast = straight_coast()
    check("west of a northward way is land", geo.is_land((-500.0, 0.0), coast))
    check("east of it is water", geo.is_land((500.0, 0.0), coast) is False)
    check("inside a bay is water", geo.is_land((0.0, 0.0), bay()) is False)
    check("inside an island is land", geo.is_land((0.0, 0.0), headland()))
    check("with no coastline the answer is None, not False",
          geo.is_land((0.0, 0.0), []) is None)


def test_sanity_check_catches_a_reversed_way():
    print("\na locally reversed way")
    good = [[(0.0, -20000.0), (0.0, 20000.0)]]
    ok, detail = geo.coastline_sanity(good)
    check("a consistent coastline passes", ok, detail)

    # Two ways meeting end to end, the second drawn backwards -- which is what
    # a piecemeal OSM edit produces, and it inverts land and sea just there.
    # Two ways meeting END to END: the second was drawn backwards.
    mixed = [[(0.0, -20000.0), (0.0, 0.0)],
             [(0.0, 20000.0), (0.0, 0.0)]]
    ok, detail = geo.coastline_sanity(mixed)
    check("a half-reversed coastline fails", not ok, detail)
    check("and the failure says what is wrong",
          "backwards" in detail and "junction" in detail, detail)

    values = spatial.coastline_covariates(33.0, -117.3,
                                          [[(32.9, -117.3), (33.1, -117.3)],
                                           [(33.3, -117.3), (33.1, -117.3)]])
    check("covariates are withheld from an insane coastline",
          values.get("shore_normal_deg") is None
          and values.get("coastline_sane") is False, str(values.get("coastline_note")))

    check("both ways at a bad junction are named suspect, since the geometry "
          "cannot say which one is reversed",
          geo.way_junctions(geo.project_lines(mixed, 0.0, 0.0))[2] == {0, 1})

    # 50 m of tolerance called two ways merely passing near each other a
    # junction, and then called the pair a reversal. Ways that really chain
    # share a node.
    passing = [[(0.0, -20000.0), (0.0, 0.0)],
               [(30.0, 20000.0), (30.0, 0.0)]]
    joins, mismatches, _suspect, _ambiguous = geo.way_junctions(passing)
    check("two ways passing 30 m apart are not a junction",
          joins == 0 and mismatches == 0, f"{joins} join(s)")

    # A river mouth where one way ends and two begin. Scored pairwise, the two
    # starts look like a reversal; they are a node of degree three, and the
    # head-to-tail rule says nothing about it.
    fork = [[(0.0, -20000.0), (0.0, 0.0)],
            [(0.0, 0.0), (-15000.0, 8000.0)],
            [(0.0, 0.0), (15000.0, 8000.0)]]
    joins, mismatches, suspect, ambiguous = geo.way_junctions(fork)
    check("a node where three ways meet is not called a reversal",
          mismatches == 0 and not suspect, f"{mismatches} mismatch(es)")
    check("it is reported as unjudged rather than passed over silently",
          ambiguous == 1, f"{ambiguous} ambiguous")
    ok, detail = geo.coastline_sanity(fork)
    check("and the coastline still passes", ok, detail)

    # Both ends of one closed ring land in the same cluster and say nothing
    # about any other way's direction.
    ring = [[(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 0.0)]]
    joins, mismatches, _suspect, _ambiguous = geo.way_junctions(ring)
    check("a closed ring does not join itself",
          joins == 0 and mismatches == 0, f"{joins} join(s)")


def test_a_far_away_defect_does_not_void_the_whole_tile():
    print("\na bad edit at the far corner of the box")
    # The box is 30 km across and land_fraction reaches 5 km. A reversed
    # junction 40 km up the coast cannot enter any covariate computed here,
    # and voiding the station for it discards good geometry to punish geometry
    # nobody read.
    far = [[(32.9, -117.3), (33.1, -117.3)],
           [(33.40, -117.3), (33.50, -117.3)],
           [(33.60, -117.3), (33.50, -117.3)]]
    values = spatial.coastline_covariates(33.0, -117.29, far)
    check("the box is still reported as defective",
          values.get("coastline_sane") is False, values.get("coastline_check"))
    check("but the covariates survive",
          values.get("shore_normal_deg") is not None,
          str(values.get("coastline_note")))
    check("and the distance to the defect is recorded",
          (values.get("coastline_defect_km") or 0) > 6.0,
          values.get("coastline_defect_km"))

    close_by = [[(32.9, -117.3), (33.0, -117.3)],
                [(33.1, -117.3), (33.0, -117.3)]]
    values = spatial.coastline_covariates(33.0, -117.29, close_by)
    check("a defect the covariates would actually read still withholds them",
          values.get("shore_normal_deg") is None,
          str(values.get("coastline_note")))


def test_shore_normal():
    print("\nshore normal (outward, the way you face looking to sea)")
    normal, points = geo.shore_normal(straight_coast(), (10.0, 0.0))
    check("a north-south coast with water east faces 90", near(normal, 90.0, 0.5),
          str(normal))
    check("the fit used more than one point", points >= 2, str(points))

    normal, _ = geo.shore_normal(bay(), (2995.0, 0.0))
    check("on the east shore of a bay you face west (270)",
          near(normal, 270.0, 1.0), str(normal))
    normal, _ = geo.shore_normal(headland(), (3005.0, 0.0))
    check("on the east shore of an island you face east (90)",
          near(normal, 90.0, 1.0), str(normal))

    # A reversed straight coast must not simply return the opposite bearing:
    # the probe step should find land where it expected water.
    # The invariant that actually matters: the normal points at WATER,
    # whichever way the coastline happens to have been drawn. Reversing the
    # way moves the water to the other side, and the normal must follow.
    reversed_coast = [[(0.0, 20000.0), (0.0, -20000.0)]]
    normal, _ = geo.shore_normal(reversed_coast, (10.0, 0.0))
    check("reversing the way puts the water west, and the normal follows",
          near(normal, 270.0, 0.5), str(normal))
    for coast, station in ((straight_coast(), (10.0, 0.0)),
                           (reversed_coast, (10.0, 0.0)),
                           (bay(), (2995.0, 0.0)),
                           (headland(), (3005.0, 0.0))):
        bearing, _ = geo.shore_normal(coast, station)
        probe = (station[0] + math.sin(math.radians(bearing)) * 200,
                 station[1] + math.cos(math.radians(bearing)) * 200)
        check(f"the normal at {bearing:.0f} points to water",
              geo.is_land(probe, coast) is False)


def test_curvature_sign_and_magnitude():
    print("\ncurvature: negative is embayed")
    for name, lines, station, sign in (("bay", bay(), (2995.0, 0.0), -1),
                                       ("headland", headland(), (3005.0, 0.0), 1)):
        curvature, radius = geo.curvature_per_km(lines, station)
        check(f"{name} curvature is {'negative' if sign < 0 else 'positive'}",
              curvature is not None and math.copysign(1, curvature) == sign,
              str(curvature))
        check(f"{name} magnitude is 1/R = {1000 / 3000:.4f} per km",
              near(abs(curvature), 1000.0 / 3000.0, 0.02), str(curvature))
        check(f"{name} radius recovered", near(radius, 3000.0, 60.0), str(radius))

    curvature, radius = geo.curvature_per_km(straight_coast(), (10.0, 0.0))
    check("a straight coast has zero curvature", curvature == 0.0, str(curvature))
    check("and infinite radius", radius == float("inf"), str(radius))


def test_embayment_and_land_fraction():
    print("\nenclosure proxies")
    check("a straight coast has embayment ratio 1",
          near(geo.embayment_ratio(straight_coast(), (10.0, 0.0)), 1.0, 1e-6))
    ratio = geo.embayment_ratio(bay(), (2995.0, 0.0))
    check("a 3 km bay is below 1", ratio is not None and ratio < 0.95, str(ratio))

    # The one thing the ratio CANNOT do, which is why curvature carries a sign.
    bay_ratio = geo.embayment_ratio(bay(), (2995.0, 0.0))
    head_ratio = geo.embayment_ratio(headland(), (3005.0, 0.0))
    check("embayment ratio alone cannot tell a bay from a headland",
          near(bay_ratio, head_ratio, 1e-6),
          f"{bay_ratio:.4f} vs {head_ratio:.4f} — only the curvature SIGN "
          "separates them")

    check("a straight coast is about half land",
          near(geo.land_fraction(straight_coast(), (10.0, 0.0)), 0.5, 0.06))
    check("inside a bay is mostly land",
          geo.land_fraction(bay(), (2995.0, 0.0)) > 0.6)
    check("off a headland is mostly water",
          geo.land_fraction(headland(), (3005.0, 0.0)) < 0.4)


def test_fetch_by_octant():
    print("\nfetch by octant")
    distances, capped = geo.fetch_by_octant(straight_coast(), (10.0, 0.0),
                                            max_km=10.0)
    check("landward octants hit land immediately",
          distances["W"] < 0.5, str(distances["W"]))
    check("seaward octants run to the cap", distances["E"] == 10.0)
    check("and are marked as censored, not measured", capped["E"] is True)
    check("the landward one is not censored", capped["W"] is False)

    distances, _capped = geo.fetch_by_octant(bay(), (2995.0, 0.0), max_km=10.0)
    check("across a 3 km-radius bay is about 6 km",
          near(distances["W"], 6.0, 0.4), str(distances["W"]))


def test_the_index_answers_exactly_what_the_scan_would():
    """An index that quietly disagrees is not an optimisation.

    Without it, every land/water sample scans every segment in the box, and
    the covariates make roughly eleven hundred samples per station. On an
    estuary shore at OpenStreetMap detail that does not finish -- and it does
    not fail either, so it reads as a hang rather than a bug. The only thing
    that makes the index usable is that it returns the same answer.
    """
    print("\nthe segment index against the scan it replaces")
    import random
    rng = random.Random(4)
    lines, x = [], -8000.0
    points = []
    while x < 8000.0:
        points.append((x, 400 * math.sin(x / 900.0) + rng.uniform(-30, 30)))
        x += 15.0
    lines = [points[i:i + 60 + 1] for i in range(0, len(points) - 1, 60)]
    island = [(2500.0 + 600.0 * math.cos(2 * math.pi * k / 120),
               1500.0 + 600.0 * math.sin(2 * math.pi * k / 120))
              for k in range(121)]
    lines.append(island)                              # an island off the shore
    index = geo.SegmentIndex(lines)

    disagreed = land_disagreed = 0
    for _ in range(600):
        probe = (rng.uniform(-9000, 9000), rng.uniform(-4000, 4000))
        if geo.nearest_segment(probe, lines) != geo.nearest_segment(
                probe, lines, index):
            disagreed += 1
        if geo.is_land(probe, lines) != geo.is_land(probe, lines, index):
            land_disagreed += 1
    check("the nearest segment is identical on 600 random points",
          disagreed == 0, f"{disagreed} disagreements")
    check("and so is the land/water verdict",
          land_disagreed == 0, f"{land_disagreed} disagreements")

    # A point well outside the linework's own extent: the index has to keep
    # walking outward rather than report nothing.
    far = (40000.0, 40000.0)
    check("a point far outside the box still finds the shore",
          geo.nearest_segment(far, lines, index)
          == geo.nearest_segment(far, lines))

    # Exactly equidistant from two segments meeting at a shared vertex --
    # the ordinary case, one per interior node of every way.
    vee = [[(-1000.0, 1000.0), (0.0, 0.0), (1000.0, 1000.0)]]
    tie_index = geo.SegmentIndex(vee)
    check("an exact tie resolves the same way indexed and not",
          geo.nearest_segment((0.0, -500.0), vee)
          == geo.nearest_segment((0.0, -500.0), vee, tie_index))


def test_fetch_sees_a_spit_thinner_than_a_step():
    print("\nfetch by ray, not by sampling")
    # A barrier 60 m thick, 5 km offshore. The old version sampled every
    # 250 m and stepped straight over it, reporting open water to the cap.
    barrier = [[(-20000.0, 5000.0), (20000.0, 5000.0)],
               [(20000.0, 5060.0), (-20000.0, 5060.0)]]
    distances, capped = geo.fetch_by_octant(barrier, (0.0, 0.0), max_km=25.0)
    check("a 60 m barrier stops the fetch at 5 km",
          near(distances["N"], 5.0, 0.01), str(distances["N"]))
    check("and the value is not censored", capped["N"] is False)
    check("the other way is open to the cap", distances["S"] == 25.0)


def test_closed_rings_wrap():
    print("\nclosed ways wrap instead of running out")
    lines = bay()
    check("the ring is detected as closed", geo.is_closed(lines[0]))
    # The station sits right at the ring's arbitrary first vertex, which is
    # where an open-line walk would run off the end and lose the covariates.
    curvature, _radius = geo.curvature_per_km(lines, (3000.0, 0.0))
    check("curvature still computes at the ring's start vertex",
          curvature is not None, str(curvature))
    check("embayment too",
          geo.embayment_ratio(lines, (3000.0, 0.0)) is not None)


def test_covariates_from_a_coastline():
    print("\nthe covariate bundle")
    # A real lat/lon with a synthetic coastline through it, so projection,
    # assembly and rounding are all exercised together.
    lat, lon = 36.9612, -122.0088
    north = [(lat - 0.18, lon), (lat + 0.18, lon)]
    values = spatial.coastline_covariates(lat, lon, [north], vintage="2026-09-01T00:00:00Z")
    check("vintage carried through", values["coastline_vintage"].startswith("2026"))
    check("sanity recorded", values["coastline_sane"] is True)
    check("shore normal present", values["shore_normal_deg"] is not None)
    check("curvature present", values["curvature_1_per_km"] is not None)
    check("embayment present", values["embayment_ratio"] is not None)
    check("land fraction present", values["land_fraction_5km"] is not None)
    check("eight octants present",
          all(f"fetch_km_{o}" in values for o in geo.OCTANTS))
    check("each octant says whether it was censored",
          all(f"fetch_capped_{o}" in values for o in geo.OCTANTS))
    check("summaries present", {"fetch_km_min", "fetch_km_mean",
                                "fetch_km_max"} <= set(values))


def test_outfall_covariates():
    print("\nECHO outfalls")
    records = [
        {"FacLat": 33.0, "FacLong": -117.3, "SourceID": "CA0001",
         "CWPMajorMinorStatusFlag": "M", "CWPFacilityTypeIndicator": "POTW"},
        {"FacLat": 33.05, "FacLong": -117.3, "SourceID": "CA0002",
         "CWPMajorMinorStatusFlag": "N", "CWPFacilityTypeIndicator": "NON-POTW"},
        {"FacLat": 34.0, "FacLong": -117.3, "SourceID": "CA0003"},
    ]
    values = spatial.outfall_covariates(33.001, -117.3, records)
    check("nearest outfall found", near(values["dist_to_outfall_m"], 111.0, 30.0),
          str(values["dist_to_outfall_m"]))
    check("major/minor and POTW both recorded",
          values["outfall_type"] == "major/POTW", str(values.get("outfall_type")))
    check("only those within 2 km counted",
          values["n_outfalls_within_2km"] == 1,
          str(values["n_outfalls_within_2km"]))
    empty = spatial.outfall_covariates(33.0, -117.3, [])
    check("no outfalls means a zero count and no distance",
          empty["n_outfalls_within_2km"] == 0
          and "dist_to_outfall_m" not in empty)


def test_stream_covariates():
    print("\nNHDPlus streams")
    lat, lon = 33.0, -117.3
    coastline = [[(lat - 0.1, lon), (lat + 0.1, lon)]]
    # One creek reaching the shore, one flowline passing by inland.
    mouth = {"geometry": {"type": "LineString",
                          "coordinates": [[lon - 0.02, lat], [lon - 0.0005, lat]]},
             "properties": {"streamorde": 3}}
    passing = {"geometry": {"type": "LineString",
                            "coordinates": [[lon - 0.03, lat + 0.01],
                                            [lon - 0.03, lat - 0.01]]},
               "properties": {"streamorde": 1}}
    catalogue = {"TOT_IMPV11": {"description": "NLCD 2011 imperviousness"},
                 "TOT_NLCD11_22": {"description": "NLCD 2011 developed, low"}}
    characteristics = {"TOT_BASIN_AREA": "42.5", "TOT_IMPV11": "18.0",
                       "TOT_NLCD11_22": "12.0"}
    values = spatial.stream_covariates(lat, lon, coastline, "12345",
                                       characteristics, catalogue,
                                       [mouth, passing])
    check("upstream area read", values["upstream_area_km2"] == 42.5)
    check("impervious converted from percent to fraction",
          near(values["impervious_frac"], 0.18, 1e-9),
          str(values.get("impervious_frac")))
    check("the NLCD vintage is recorded, not assumed",
          "2011" in str(values["landcover_vintage"]),
          str(values["landcover_vintage"]))
    check("stream order from the nearest flowline",
          values["stream_order"] == 3, str(values.get("stream_order")))
    check("only the flowline that REACHES the shore counts as a mouth",
          values["n_streams_within_2km"] == 1,
          str(values["n_streams_within_2km"]))


def test_landcover_ids_are_not_hardcoded():
    print("\nNLCD year comes from the catalogue")
    old = {"TOT_IMPV11": {"description": "NLCD 2011 imperviousness"}}
    new = {"TOT_IMPV11": {"description": "NLCD 2011 imperviousness"},
           "TOT_IMPV19": {"description": "NLCD 2019 imperviousness"}}
    check("with only 2011 available it uses 2011",
          spatial.pick_landcover_ids(old)[0] == "TOT_IMPV11")
    check("with 2019 available it uses the later release",
          spatial.pick_landcover_ids(new)[0] == "TOT_IMPV19",
          str(spatial.pick_landcover_ids(new)[0]))
    check("and reports which, so the manifest records the real vintage",
          "2019" in str(spatial.pick_landcover_ids(new)[2]))
    # No module constant may hold a specific NLCD id: that would pin the
    # study to one land-cover release whatever the service now serves.
    import ast
    source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "spatial.py")).read()
    pinned = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if text.startswith(("TOT_IMPV", "TOT_NLCD")) and any(
                    c.isdigit() for c in text):
                pinned.append(text)
    check("no NLCD id is pinned in the module's code", not pinned, str(pinned))


def test_layers_are_fetched_per_tile_not_per_station():
    """The thing that made a national run thirty hours long.

    One Overpass query per station is 32,513 queries against a free community
    service whose usage policy asks for light use. Coastline and permitted
    discharges are shared between neighbouring stations, so both are fetched
    per tile and reused.
    """
    print("\ntiling: neighbouring stations share one fetch")
    santa_cruz = (36.9612, -122.0088)
    wharf = (36.9628, -122.0170)        # ~1 km along the same beach
    capitola = (36.9714, -121.9530)     # ~5 km along the same bay
    san_diego = (32.7700, -117.2300)

    check("two beaches 1 km apart share a tile",
          spatial._tile_slug(*santa_cruz) == spatial._tile_slug(*wharf))
    check("a beach 600 km away does not",
          spatial._tile_slug(*santa_cruz) != spatial._tile_slug(*san_diego))

    # A grid boundary WILL separate two stations a few km apart -- Capitola
    # sits in the next tile east of Santa Cruz. That is harmless only because
    # the margin makes each tile's box reach well past its own edge, so both
    # stations still get coastline covering the other's position. Without
    # that, a station beside a boundary would see the sea end at the tile
    # edge and read as open water.
    check("a tile boundary can split neighbours",
          spatial._tile_slug(*santa_cruz) != spatial._tile_slug(*capitola),
          f"{spatial._tile_slug(*santa_cruz)} vs {spatial._tile_slug(*capitola)}")
    for here, there in ((santa_cruz, capitola), (capitola, santa_cruz)):
        south, west, north, east = spatial.tile_bounds(
            *here, margin_km=spatial.COASTLINE_BBOX_KM)
        check("but each box still covers the other station",
              south <= there[0] <= north and west <= there[1] <= east,
              f"{there} outside {south:.2f}..{north:.2f}, {west:.2f}..{east:.2f}")

    # The tile box must still cover every station in it out to the full
    # search radius, or a station near an edge silently loses the coastline
    # on the other side of that edge -- which would read as open water.
    south, west, north, east = spatial.tile_bounds(*santa_cruz,
                                                   margin_km=spatial.COASTLINE_BBOX_KM)
    for corner_lat, corner_lon in ((36.75, -122.25), (37.0, -122.0)):
        reach_lat = spatial.COASTLINE_BBOX_KM / 111.0
        check(f"the box reaches {spatial.COASTLINE_BBOX_KM:.0f} km beyond "
              f"({corner_lat}, {corner_lon})",
              south <= corner_lat - reach_lat and north >= corner_lat + reach_lat,
              f"box {south:.2f}..{north:.2f}")
    check("the box is wider than the tile itself",
          (north - south) > spatial.TILE_DEGREES,
          f"{north - south:.3f} vs {spatial.TILE_DEGREES}")

    # A realistic coastal cluster: many stations, few tiles.
    rng = np.random.default_rng(7)
    cluster = pd.DataFrame({
        "station_id": [str(i) for i in range(200)],
        "lat": 36.95 + rng.normal(0, 0.05, 200),
        "lon": -122.0 + rng.normal(0, 0.05, 200)})
    tiles = {spatial._tile_slug(r["lat"], r["lon"])
             for _, r in cluster.iterrows()}
    check("200 clustered stations collapse to a handful of tiles",
          len(tiles) <= 12, f"{len(tiles)} tiles")


def test_no_data_is_not_a_refusal():
    """A 404 is the service answering, not the service refusing.

    NLDI answered for eleven stations, then hit three coastal beaches with no
    NHDPlus flowline near them -- which is an ordinary fact about open coast,
    not a fault -- and the circuit breaker abandoned the layer for the whole
    run. The breaker must separate "there is nothing here" from "stop asking
    me", because only the second is a reason to stop asking.
    """
    print("\ncircuit breaker: refusals only")
    spatial._FAILURES.clear()
    spatial._SUCCESSES.clear()
    spatial._TRIPPED.clear()

    host = "example.test"
    check("429 is a refusal", 429 in spatial.REFUSAL_STATUSES)
    check("503 is a refusal", 503 in spatial.REFUSAL_STATUSES)
    check("404 is NOT a refusal — it is an answer",
          404 not in spatial.REFUSAL_STATUSES)
    check("400 is NOT a refusal either",
          400 not in spatial.REFUSAL_STATUSES)

    for _ in range(3):
        spatial._circuit_record(host, ok=False)
    tripped = False
    try:
        spatial._circuit_check(host)
    except spatial.LayerFailed:
        tripped = True
    check("three refusals trip the breaker", tripped)

    # A host that has answered gets the patient ladder; a cold one does not.
    spatial._FAILURES.clear()
    spatial._TRIPPED.clear()
    cold = "cold.test"
    warm = "warm.test"
    spatial._circuit_record(warm, ok=True)
    check("a cold host gets one short retry",
          spatial._ladder(cold) == spatial.COLD_BACKOFF,
          str(spatial._ladder(cold)))
    check("a host that has answered gets the patient ladder",
          spatial._ladder(warm) == spatial.RATE_LIMIT_BACKOFF,
          str(spatial._ladder(warm)))

    # A success resets the consecutive count, so intermittent trouble at a
    # working service never accumulates into an abandonment.
    spatial._circuit_record(warm, ok=False)
    spatial._circuit_record(warm, ok=False)
    spatial._circuit_record(warm, ok=True)
    spatial._circuit_record(warm, ok=False)
    still_ok = True
    try:
        spatial._circuit_check(warm)
    except spatial.LayerFailed:
        still_ok = False
    check("a success in between resets the count", still_ok)
    spatial._FAILURES.clear()
    spatial._SUCCESSES.clear()
    spatial._TRIPPED.clear()


def test_coverage_rule_on_covariates():
    print("\nthe ~70% coverage rule")
    frame = pd.DataFrame({
        "station_id": [str(i) for i in range(10)],
        "land_fraction_5km": [0.5] * 10,               # 100%
        "embayment_ratio": [0.9] * 8 + [None] * 2,     # 80%
        "impervious_frac": [0.2] * 6 + [None] * 4,     # 60%
        "stream_order": [None] * 10,                   # 0%
    })
    table = spatial.coverage(frame)
    keeps = dict(zip(table["covariate"], table["keeps"]))
    check("100% coverage is kept", keeps["land_fraction_5km"])
    check("80% is kept", keeps["embayment_ratio"])
    check("60% is dropped", not keeps["impervious_frac"])
    check("0% is dropped", not keeps["stream_order"])
    check("the layer is named beside each covariate",
          dict(zip(table["covariate"], table["layer"]))["impervious_frac"]
          == "nlcd")


def test_manifest_records_layers_and_amendments():
    print("\nmanifest: layers, vintages, amendments")
    sites = strata.assign(
        pd.DataFrame([{"station_id": str(i), "station_name": "Beach",
                       "lat": 33.0 + i / 100, "lon": -117.3, "state": "CA"}
                      for i in range(10)]), datums=pd.DataFrame())
    frame = pd.DataFrame({"station_id": [str(i) for i in range(10)],
                          "land_fraction_5km": [0.5] * 10,
                          "embayment_ratio": [0.9] * 10,
                          "impervious_frac": [0.2] * 3 + [None] * 7})
    record = layers.blank_record()
    layers.record_access(record, "coastline", "2026-09-01T00:00:00Z")
    record["coastline"]["sites_attempted"] = 10
    record["coastline"]["sites_populated"] = 10

    path = tempfile.mktemp(suffix=".json")
    payload = manifest.write(sites, path, spatial=frame, layer_record=record)
    entry = payload["entries"][0]

    check("every layer is recorded with its source",
          set(entry["layers"]) == set(layers.LAYERS))
    check("the observed vintage is written where the service gave one",
          entry["layers"]["coastline"]["observed_vintage"]
          == "2026-09-01T00:00:00Z")
    check("an unverified endpoint is flagged as such",
          entry["layers"]["coastline"]
          ["endpoint_verified_against_live_response"] is False)
    check("the ECHO caveat travels with the layer",
          "outfall pipe" in entry["layers"]["echo"]["caveat"])
    check("covariate coverage recorded",
          entry["site_covariates"]["observed_coverage"]["land_fraction_5km"] == 1.0)
    check("a covariate under 70% is dropped BEFORE fitting",
          "impervious_frac" in entry["site_covariates"]["dropped_for_coverage"])
    check("one dropped and one kept, from the same run",
          "land_fraction_5km" in entry["site_covariates"]["kept"])
    check("beach_type is declared manual",
          "beach_type" in entry["strata"]["manual"])
    check("and dropped, because nobody reviewed anything",
          "beach_type" in entry["strata"]["dropped_for_coverage"])

    groupings = manifest.active_strata(manifest.require_manifest(path))
    check("shore_normal_deg is never a grouping",
          "shore_normal_deg" not in groupings, str(groupings))
    check("nor is the gauge distance",
          "datum_gauge_dist_km" not in groupings, str(groupings))

    # An amendment is allowed, timestamped and labelled -- not forbidden.
    entry = manifest.amend(["fetch_km_NE"], "reviewer asked", "riley", path)
    check("an amendment is marked exploratory", entry["exploratory"] is True)
    check("it has its own timestamp", bool(entry["written_at"]))
    check("it names who added it and why",
          entry["added_by"] == "riley" and entry["why"] == "reviewer asked")
    check("the registered entry still authorises the fit and is unchanged",
          manifest.require_manifest(path)["entry"] == 0)
    check("the exploratory covariate is listed as such",
          manifest.exploratory_covariates(manifest.read(path)) == ["fetch_km_NE"])

    caught = False
    try:
        manifest.amend(["fetch_km_SW"], None, None, path)
    except SystemExit as exc:
        caught = "why" in str(exc)
    check("an amendment with no reason and no author is refused", caught)
    os.remove(path)


def test_beach_type_is_hand_assigned_only():
    print("\nbeach_type provenance")
    sites = pd.DataFrame([
        {"station_id": "A", "station_name": "Newport Harbor",
         "site_type": "Estuary", "lat": 33.6, "lon": -117.9, "state": "CA"},
        {"station_id": "B", "station_name": "Storm Drain at 5th",
         "site_type": "Ocean", "lat": 33.0, "lon": -117.3, "state": "CA"}])

    # With nobody having reviewed anything, the old keyword rules would have
    # labelled both of these confidently. They must now come out empty.
    out = strata.assign(sites, datums=pd.DataFrame(), reviewed=pd.DataFrame())
    check("no keyword assigns a label any more",
          out["beach_type"].isna().all(), str(out["beach_type"].tolist()))
    check("region is still assigned automatically",
          set(out["region"]) == {"Pacific"})

    reviewed = pd.DataFrame([
        {"station_id": "A", "beach_type": "enclosed_bay",
         "assigned_by": "riley", "assigned_on": "2026-09-20"}])
    out = strata.assign(sites, datums=pd.DataFrame(), reviewed=reviewed)
    labels = dict(zip(out["station_id"], out["beach_type"]))
    check("a reviewed site gets its label", labels["A"] == "enclosed_bay")
    check("an unreviewed one stays empty", pd.isna(labels["B"]))
    check("the assigner travels with the label",
          out.loc[out["station_id"] == "A",
                  "beach_type_assigned_by"].iloc[0] == "riley")
    check("so does the date",
          out.loc[out["station_id"] == "A",
                  "beach_type_assigned_on"].iloc[0] == "2026-09-20")

    table = strata.coverage(out)
    check("50% reviewed is below the threshold and is dropped",
          not dict(zip(table["stratum"], table["keeps"]))["beach_type"])


def test_review_workflow():
    print("\nthe review worklist")
    sites = pd.DataFrame([
        {"station_id": "OPEN", "station_name": "Open", "state": "CA",
         "lat": 33.0, "lon": -117.3},
        {"station_id": "AMBIG", "station_name": "Half open", "state": "CA",
         "lat": 34.0, "lon": -118.3},
        {"station_id": "CLEAR", "station_name": "Deep bay", "state": "CA",
         "lat": 35.0, "lon": -120.3}])
    covariates = pd.DataFrame([
        {"station_id": "OPEN", "land_fraction_5km": 0.50,
         "embayment_ratio": 0.999, "curvature_1_per_km": 0.01},
        {"station_id": "AMBIG", "land_fraction_5km": 0.62,
         "embayment_ratio": 0.90, "curvature_1_per_km": -0.08},
        {"station_id": "CLEAR", "land_fraction_5km": 0.95,
         "embayment_ratio": 0.35, "curvature_1_per_km": -0.9}])
    table = review.worklist(sites, covariates)
    order = table["station_id"].tolist()
    check("the ambiguous site is reviewed first", order[0] == "AMBIG", str(order))
    check("the unambiguous bay is last", order[-1] == "CLEAR", str(order))
    check("every row carries an imagery link",
          table["imagery"].str.startswith("https://").all())
    check("the worklist does NOT contain an assigned label",
          table["beach_type"].isna().all())

    # Ingest, with provenance enforced.
    filled = table.copy()
    filled["beach_type"] = ["enclosed_bay", "open_coast", "enclosed_bay"]
    source = tempfile.mktemp(suffix=".csv")
    target = tempfile.mktemp(suffix=".csv")
    filled.to_csv(source, index=False)
    out = review.ingest(source, "riley", "2026-09-20", target)
    check("labels ingested", len(out) == 3)
    check("with the assigner", set(out["assigned_by"]) == {"riley"})
    check("and the date", set(out["assigned_on"]) == {"2026-09-20"})

    caught = False
    try:
        review.ingest(source, None, None, target)
    except SystemExit as exc:
        caught = "--by is required" in str(exc)
    check("ingesting without an assigner is refused", caught)

    bad = filled.copy()
    bad["beach_type"] = ["lagoon"] * 3
    bad_path = tempfile.mktemp(suffix=".csv")
    bad.to_csv(bad_path, index=False)
    caught = False
    try:
        review.ingest(bad_path, "riley", "2026-09-20", target)
    except SystemExit as exc:
        caught = "not valid beach_type" in str(exc)
    check("an invented label is refused", caught)
    for path in (source, target, bad_path):
        os.remove(path)


def test_overrides_cannot_smuggle_beach_type():
    print("\nthe override file cannot assign beach_type")
    path = tempfile.mktemp(suffix=".csv")
    with open(path, "w") as handle:
        handle.write("station_id,beach_type,shore_normal_deg\nA,enclosed_bay,206\n")
    caught = False
    try:
        strata.load_overrides(path)
    except SystemExit as exc:
        caught = "assigned by hand" in str(exc)
    check("a beach_type in strata_overrides.csv is refused", caught)

    with open(path, "w") as handle:
        handle.write("station_id,shore_normal_deg\nA,206\n")
    frame = strata.load_overrides(path)
    check("a shore normal override is still allowed",
          frame.loc["A", "shore_normal_deg"] == "206")
    os.remove(path)


def main():
    for test in (test_land_and_water,
                 test_sanity_check_catches_a_reversed_way,
                 test_a_far_away_defect_does_not_void_the_whole_tile,
                 test_the_index_answers_exactly_what_the_scan_would,
                 test_fetch_sees_a_spit_thinner_than_a_step,
                 test_shore_normal,
                 test_curvature_sign_and_magnitude,
                 test_embayment_and_land_fraction,
                 test_fetch_by_octant,
                 test_closed_rings_wrap,
                 test_covariates_from_a_coastline,
                 test_outfall_covariates,
                 test_stream_covariates,
                 test_landcover_ids_are_not_hardcoded,
                 test_layers_are_fetched_per_tile_not_per_station,
                 test_no_data_is_not_a_refusal,
                 test_coverage_rule_on_covariates,
                 test_manifest_records_layers_and_amendments,
                 test_beach_type_is_hand_assigned_only,
                 test_review_workflow,
                 test_overrides_cannot_smuggle_beach_type):
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all offline geometry and spatial-covariate checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
