"""Coastline geometry. Pure functions, no network, no pandas.

Everything here works in a local tangent plane in metres, built around the
station itself: x east, y north. Over the few kilometres these covariates
look at, that projection is accurate to better than a metre, and it means
distances, bearings, tangents and curvature are ordinary plane geometry
rather than spherical trigonometry with sign conventions to get wrong.

THE LAND/WATER TEST. OpenStreetMap draws `natural=coastline` ways with land
on the LEFT of the direction of travel. That single convention is what makes
every enclosure covariate here computable from linework alone, with no
polygon fill and no raster: for any point, find the nearest coastline
segment and take the sign of the cross product. Left is land, right is water.

It is also the assumption most likely to be wrong in a specific place -- a
way digitised backwards inverts land and sea locally -- so
`coastline_sanity()` checks it against a fact the linework knows about
itself: coastline ways chain head to TAIL, so where two ways share a node,
one's end must meet the other's start. Two ends, or two starts, means one of
the pair runs the wrong way. `way_junctions()` names which ways those are, so
a station is refused its covariates when a suspect way is near enough to
affect them and keeps them when the bad edit is twenty kilometres up the
coast.
"""

import math

EARTH_RADIUS_M = 6371000.0
# Degrees of latitude are very nearly constant; degrees of longitude are not,
# so the x scale is taken at the station's own latitude.
METRES_PER_DEG_LAT = 110540.0
METRES_PER_DEG_LON = 111320.0


def haversine_m(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def to_local(lat, lon, lat0, lon0):
    """(x, y) metres east and north of the origin."""
    return ((lon - lon0) * METRES_PER_DEG_LON * math.cos(math.radians(lat0)),
            (lat - lat0) * METRES_PER_DEG_LAT)


def from_local(x, y, lat0, lon0):
    return (lat0 + y / METRES_PER_DEG_LAT,
            lon0 + x / (METRES_PER_DEG_LON * math.cos(math.radians(lat0))))


def project_lines(lines, lat0, lon0):
    """[[(lat, lon), ...], ...] -> [[(x, y), ...], ...], direction preserved."""
    return [[to_local(lat, lon, lat0, lon0) for lat, lon in line]
            for line in lines if len(line) >= 2]


def bearing_of(dx, dy):
    """Compass bearing of a local vector: 0 north, 90 east."""
    return math.degrees(math.atan2(dx, dy)) % 360


def _closest_on_segment(point, start, end):
    """(distance, foot, t, (ux, uy)) for one segment. t is 0..1 along it."""
    px, py = point
    ax, ay = start
    bx, by = end
    vx, vy = bx - ax, by - ay
    length_sq = vx * vx + vy * vy
    if length_sq == 0:
        return math.hypot(px - ax, py - ay), (ax, ay), 0.0, (0.0, 0.0)
    t = ((px - ax) * vx + (py - ay) * vy) / length_sq
    t = max(0.0, min(1.0, t))
    foot = (ax + t * vx, ay + t * vy)
    length = math.sqrt(length_sq)
    return (math.hypot(px - foot[0], py - foot[1]), foot, t,
            (vx / length, vy / length))


def nearest_segment(point, lines):
    """The closest coastline segment to a point.

    Returns (distance_m, line_index, segment_index, foot, unit_direction) or
    None when there is no coastline at all.
    """
    best = None
    for line_index, line in enumerate(lines):
        for segment_index in range(len(line) - 1):
            distance, foot, _t, direction = _closest_on_segment(
                point, line[segment_index], line[segment_index + 1])
            if best is None or distance < best[0]:
                best = (distance, line_index, segment_index, foot, direction)
    return best


def is_land(point, lines):
    """True where the point lies on the land side of the nearest segment.

    OSM's convention is land on the left of the way's direction, so the sign
    of the cross product of the segment direction with the offset to the
    point is the whole test. None when there is no coastline to test against.
    """
    found = nearest_segment(point, lines)
    if found is None:
        return None
    _distance, _line, _segment, foot, (ux, uy) = found
    wx, wy = point[0] - foot[0], point[1] - foot[1]
    cross = ux * wy - uy * wx
    if cross == 0:
        return None
    return cross > 0


def is_closed(line, tolerance_m=1.0):
    """A way that returns to its start: an island, or a lagoon's shore.

    These are common in OSM coastline data and they are the case where
    walking along the shore has to wrap rather than run out -- a station that
    happens to sit near the ring's arbitrary first vertex would otherwise
    lose its curvature and embayment covariates for no reason except where
    the original mapper started drawing.
    """
    if len(line) < 3:
        return False
    return math.hypot(line[0][0] - line[-1][0],
                      line[0][1] - line[-1][1]) <= tolerance_m


# Ways that chain share a NODE, so their endpoints are the same coordinate and
# the only gap is projection round-off, well under a metre. The first version
# of this allowed 50 m, which in a dense estuary counts two ways merely passing
# near each other as a junction and then calls the pair a reversal.
JOIN_TOLERANCE_M = 1.0


def way_junctions(lines, join_tolerance_m=JOIN_TOLERANCE_M):
    """(joins, mismatches, suspect) for a set of coastline ways.

    `suspect` is the indices of the ways meeting at a junction that does not
    chain head to tail. Both ways of such a pair are suspect: the geometry
    says one of them runs the wrong way, not which.
    """
    joins = mismatches = 0
    suspect = set()
    for i, first in enumerate(lines):
        if len(first) < 2:
            continue
        for j in range(i + 1, len(lines)):
            second = lines[j]
            if len(second) < 2:
                continue
            for a_point, a_is_end in ((first[0], False), (first[-1], True)):
                for b_point, b_is_end in ((second[0], False),
                                          (second[-1], True)):
                    if math.hypot(a_point[0] - b_point[0],
                                  a_point[1] - b_point[1]) > join_tolerance_m:
                        continue
                    joins += 1
                    # Head to tail is one end and one start. Two ends or two
                    # starts means one of the pair runs the wrong way.
                    if a_is_end == b_is_end:
                        mismatches += 1
                        suspect.update((i, j))
    return joins, mismatches, suspect


def coastline_sanity(lines, join_tolerance_m=JOIN_TOLERANCE_M):
    """Do these ways agree with each other about which side is land?

    An earlier version of this probed each segment against itself: offset a
    point to the left of a segment, ask which side of the NEAREST segment it
    fell on, and get that same segment back. It agreed 100% of the time on
    every input, including a deliberately reversed one, because a segment
    cannot disagree with itself. A check that cannot fail is worse than no
    check, since it is read as evidence.

    The real failure it needs to catch is a way digitised backwards relative
    to its neighbours -- OSM coastline is edited piecemeal, and one reversed
    way inverts land and sea exactly where it is wrong. That is detectable
    directly: coastline ways chain head to TAIL, so where two ways share an
    endpoint, one's end must meet the other's start. Two ends meeting, or two
    starts, is a reversal.

    Returns (ok, detail). A single unconnected way cannot be cross-checked
    against anything, and says so rather than claiming to have been verified.
    The verdict is about the whole box; whether the suspect way is anywhere
    near a given station is a separate question, and the caller asks it with
    way_junctions() rather than throwing away a whole tile of good linework.
    """
    usable = [line for line in lines if len(line) >= 2]
    if not usable:
        return False, "no coastline"

    joins, mismatches, _suspect = way_junctions(usable, join_tolerance_m)
    if joins == 0:
        closed = sum(1 for line in usable if is_closed(line))
        return True, (f"{len(usable)} way(s), {closed} closed, none joined — "
                      "direction could not be cross-checked")
    if mismatches:
        return False, (f"{mismatches} of {joins} way junction(s) meet end-to-end "
                       "or start-to-start, so at least one way is digitised "
                       "backwards and land and sea are inverted along it")
    return True, f"{joins} way junction(s) all chain head to tail"


def _walk(line, start_index, start_t, distance_m):
    """Point `distance_m` along a polyline from a position on it.

    Positive walks with the way's direction, negative against it. Returns
    (point, reached), where reached is False if an OPEN line ran out -- which
    is not the same as arriving, and the callers treat it differently. A
    closed ring never runs out; it wraps.
    """
    closed = is_closed(line)
    segments = len(line) - 1
    if segments < 1:
        return line[0] if line else (0.0, 0.0), False
    position = start_index + start_t
    remaining = abs(distance_m)
    forward = distance_m >= 0
    index = int(math.floor(position))
    t = position - index
    guard = 0
    while guard <= segments * 2 + 4:
        guard += 1
        if closed:
            index %= segments
        elif not 0 <= index < segments:
            break
        ax, ay = line[index]
        bx, by = line[index + 1]
        segment = math.hypot(bx - ax, by - ay)
        available = (1 - t) * segment if forward else t * segment
        if available >= remaining:
            step = remaining / segment if segment else 0
            new_t = t + step if forward else t - step
            return (ax + new_t * (bx - ax), ay + new_t * (by - ay)), True
        remaining -= available
        if forward:
            index += 1
            t = 0.0
        else:
            index -= 1
            t = 1.0
    index = max(0, min(index, len(line) - 1))
    return line[index], False


def local_tangent(lines, station=(0.0, 0.0), half_window_m=250.0):
    """Bearing of the coastline tangent at the station, fitted over a window.

    A least-squares direction over +/- half_window_m rather than the nearest
    segment's own bearing: coastline vertices are placed by whoever traced
    them, and a single 30 m segment can sit at any angle. The fit is oriented
    with the way's direction so the land-on-the-left convention still holds
    afterwards.

    Returns (bearing_deg, n_points) or (None, 0).
    """
    found = nearest_segment(station, lines)
    if found is None:
        return None, 0
    _distance, line_index, segment_index, foot, _direction = found
    line = lines[line_index]
    _distance_, _line_, _segment_, _foot_, _dir_ = found
    start, _ = _walk(line, segment_index, _position_on(line, segment_index, foot),
                     -half_window_m)
    end, _ = _walk(line, segment_index, _position_on(line, segment_index, foot),
                   half_window_m)
    points = [start] + [
        line[i] for i in range(len(line))
        if _between(line[i], start, end, foot, half_window_m)] + [end]
    if len(points) < 2:
        return None, 0
    mean_x = sum(p[0] for p in points) / len(points)
    mean_y = sum(p[1] for p in points) / len(points)
    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    syy = sum((p[1] - mean_y) ** 2 for p in points)
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    # Principal axis of the point cloud: the direction that a straight coast
    # would lie along, and the least-squares fit for a curved one.
    angle = 0.5 * math.atan2(2 * sxy, sxx - syy)
    dx, dy = math.cos(angle), math.sin(angle)
    # Orient with the way, so "left is land" survives the fit.
    if (end[0] - start[0]) * dx + (end[1] - start[1]) * dy < 0:
        dx, dy = -dx, -dy
    return bearing_of(dx, dy), len(points)


def _position_on(line, segment_index, foot):
    ax, ay = line[segment_index]
    bx, by = line[segment_index + 1]
    segment = math.hypot(bx - ax, by - ay)
    if segment == 0:
        return 0.0
    return math.hypot(foot[0] - ax, foot[1] - ay) / segment


def _between(point, start, end, foot, half_window_m):
    return (math.hypot(point[0] - foot[0], point[1] - foot[1])
            <= half_window_m * 1.05)


def shore_normal(lines, station=(0.0, 0.0), half_window_m=250.0):
    """Outward (seaward) normal bearing: the way you face looking out to sea.

    Same convention as the rip pipeline's sites.yaml. Land is to the left of
    the way, so water is to the right, so the seaward normal is the tangent
    rotated 90 degrees clockwise. The result is then CHECKED by stepping
    100 m along it and asking whether that point is water; if it is not, the
    linework disagrees with itself and None is returned rather than a bearing
    that is exactly backwards.
    """
    bearing, count = local_tangent(lines, station, half_window_m)
    if bearing is None:
        return None, count
    normal = (bearing + 90.0) % 360
    step = 100.0
    probe = (station[0] + math.sin(math.radians(normal)) * step,
             station[1] + math.cos(math.radians(normal)) * step)
    water = is_land(probe, lines)
    if water is None or water:
        return None, count
    return normal, count


def curvature_per_km(lines, station=(0.0, 0.0), window_m=2000.0):
    """Signed curvature of the local coastline, in 1/km.

    Negative is concave, which is to say embayed: the coast wraps around the
    water. Positive is a headland. The sign is decided by which side of the
    coast the fitted circle's centre falls on -- in a bay the centre sits in
    the water, on a headland it sits in the land -- rather than by a
    cross-product convention, because that keeps it correct if a way's
    direction is reversed relative to a neighbour.

    Three points at the station and +/- half the window are fitted to a
    circle. Returns (curvature, radius_m) with (0.0, inf) for a straight
    coast and (None, None) where the coastline is too short to fit.
    """
    found = nearest_segment(station, lines)
    if found is None:
        return None, None
    _distance, line_index, segment_index, foot, _direction = found
    line = lines[line_index]
    position = _position_on(line, segment_index, foot)
    back, back_ok = _walk(line, segment_index, position, -window_m / 2)
    forward, forward_ok = _walk(line, segment_index, position, window_m / 2)
    if not (back_ok and forward_ok):
        return None, None
    centre, radius = _circle_through(back, foot, forward)
    if centre is None:
        return 0.0, float("inf")
    centre_is_land = is_land(centre, lines)
    if centre_is_land is None:
        return None, None
    sign = 1.0 if centre_is_land else -1.0
    return sign * 1000.0 / radius, radius


def _circle_through(a, b, c):
    """Centre and radius of the circle through three points, or (None, inf)
    when they are collinear -- which is a straight coast, not a failure."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None, float("inf")
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    return (ux, uy), math.hypot(ax - ux, ay - uy)


def embayment_ratio(lines, station=(0.0, 0.0), along_m=2000.0):
    """Straight-line distance / along-shore distance, +/- along_m either side.

    1.0 is a straight coast. Lower is more enclosed: 0.0 would be a coast
    that comes back to where it started. This is the continuous measure of
    enclosure, and the reason beach_type is assigned by hand rather than by
    thresholding it -- the interesting sites are the ones in the middle.
    """
    found = nearest_segment(station, lines)
    if found is None:
        return None
    _distance, line_index, segment_index, foot, _direction = found
    line = lines[line_index]
    position = _position_on(line, segment_index, foot)
    back, back_ok = _walk(line, segment_index, position, -along_m)
    forward, forward_ok = _walk(line, segment_index, position, along_m)
    if not (back_ok and forward_ok):
        return None
    straight = math.hypot(forward[0] - back[0], forward[1] - back[1])
    return straight / (2 * along_m)


def land_fraction(lines, station=(0.0, 0.0), radius_m=5000.0, rings=12,
                  per_ring=24):
    """Share of a disc around the station that is land.

    Sampled on rings whose radii go as sqrt, so every sample stands for the
    same area and the answer is an area fraction rather than a count of
    points biased toward the middle. A cheap continuous proxy for enclosure:
    open coast lands near 0.5, the back of a bay well above it.
    """
    if not lines:
        return None
    land = total = 0
    for ring in range(1, rings + 1):
        radius = radius_m * math.sqrt(ring / rings)
        for step in range(per_ring):
            angle = 2 * math.pi * step / per_ring
            point = (station[0] + radius * math.sin(angle),
                     station[1] + radius * math.cos(angle))
            verdict = is_land(point, lines)
            if verdict is None:
                continue
            total += 1
            land += int(verdict)
    if not total:
        return None
    return land / total


OCTANTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def fetch_by_octant(lines, station=(0.0, 0.0), max_km=25.0, step_m=250.0):
    """Open-water distance in each compass octant, in km.

    Marches outward from the station along each octant's centre bearing and
    stops at the first land sample. A value equal to max_km means the march
    ran out of search area, not that the ocean ends there, so it is returned
    with a `_capped` companion -- reporting a censored value as a measurement
    is how a sheltered site and an open one end up looking alike.
    """
    if not lines:
        return {}, {}
    distances, capped = {}, {}
    steps = int(max_km * 1000 / step_m)
    for index, name in enumerate(OCTANTS):
        bearing = index * 45.0
        dx = math.sin(math.radians(bearing))
        dy = math.cos(math.radians(bearing))
        reached = max_km
        hit = False
        for step in range(1, steps + 1):
            distance = step * step_m
            point = (station[0] + dx * distance, station[1] + dy * distance)
            if is_land(point, lines):
                reached = distance / 1000.0
                hit = True
                break
        distances[name] = reached
        capped[name] = not hit
    return distances, capped
