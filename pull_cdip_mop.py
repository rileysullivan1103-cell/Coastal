#!/usr/bin/env python3
"""Nearshore waves at the beach itself, from CDIP's MOP alongshore model.

Every wave number in this project so far has come from far away. Walton
Lighthouse was joined to buoy 46236, 22.9 km out in 133 m of water; Wrightsville
Beach to an Open-Meteo reanalysis cell. CDIP's MOnitoring and Prediction system
propagates its buoy measurements inshore and publishes an hourly series at
points spaced along the 10 m depth contour. For Walton the nearest is SC130 --
1.45 km from the camera, hourly, 1999-12-31 to now.

It is a MODEL, so under this project's convention its columns stay
lower_snake_case alongside ERA5. But a model initialised by a real buoy and
propagated to the surf zone is a different object from a global reanalysis
cell, and it carries three things nothing else here has:

  metaShoreNormal      the shore normal, published. SHORE_NORMAL_DEG in
                       analyze_drivers is four bearings read off a map, and the
                       code says so. Every onshore-wind and axial-offset number
                       rests on those guesses.
  waveModelInputSource which buoy drove the model, per timestep. That turns
                       "was the model constrained in 2001?" into a lookup.
  waveSxy              alongshore radiation stress -- the term that actually
                       drives longshore current, and the closest thing to rip
                       physics this project has had.

California only. MOP does not exist for Virginia Beach, Wrightsville or Jupiter.

    python pull_cdip_mop.py --camera Walton --probe
    python pull_cdip_mop.py --camera Walton
"""
import argparse
import math
import os
import re
import sys
import time

import pandas as pd
import requests

CATALOG = ("https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/model/"
           "MOP_alongshore/catalog.html")
DODS = "https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/MOP_alongshore/"
DATA_DIR = "data"
CACHE = "data/cdip_mop_catalog.html"
CANDIDATES_CSV = "camera_candidates.csv"
TIMEOUT = 180
MAX_RETRIES = 4
CHUNK = 20000
# A MOP point further than this from the site is not that site's waves. The
# first version of this script had no such floor: it settled on the first
# region whose FIRST point was within 400 km, which for Santa Cruz was the
# Santa Barbara series, and wrote 229,867 hours from a point 251 km away
# under the camera's name. A wrong file that looks complete is worse than no
# file, so the distance is now checked and the run refuses.
MAX_KM = 25.0

# CDIP declares _FillValue = -999.99 on every Float32 wave variable, and the
# .ascii service hands it back as a plain number. float("-999.99") is a
# perfectly good float, so notna() is True and a column of nothing but fill
# reports as 100% populated -- which is exactly what SC130's waveDm, waveSxy
# and waveSxx did on the first run here. Anything within FILL_TOL of it is
# missing, not data.
FILL_VALUE = -999.99
FILL_TOL = 0.01
# The same three columns open the record with runs of 0.0 and denormals like
# 1.2397983E-33 -- uninitialised memory written out as Float32, not a
# measurement. No wave height, period, direction or radiation stress this
# project cares about is legitimately smaller than TINY.
TINY = 1e-20
# A column with less than this share of usable values at the chosen point is
# dropped rather than written mostly-empty under a name other sites fill.
MIN_USABLE = 0.50

# Dataset ids come in two shapes -- B0001 (one letter, four digits) and SC001
# (two letters, three digits) -- and assuming either one alone silently hides
# whole counties. Santa Cruz is only in the second family.
ID_RE = re.compile(r"\b([A-Z]{1,2}\d{3,4})_hindcast\b")
DDS_RE = re.compile(r"(\w+)\[(\w+) = (\d+)\]")

# CDIP name -> ours. waveTa is the AVERAGE period and waveTp the PEAK; Open-
# Meteo's wave_period is a mean period, so waveTa is what lines up with the
# existing column and waveTp gets its own name. Same for mean vs peak
# direction. Getting this backwards would put a peak period in a column every
# other site fills with a mean.
RENAME = {
    "waveHs": "wave_height",
    "waveTa": "wave_period",
    "waveTp": "wave_period_peak",
    "waveDm": "wave_direction",
    "waveDp": "wave_direction_peak",
    "waveSxy": "radiation_stress_sxy",
    "waveSxx": "radiation_stress_sxx",
}
SERIES_VARS = ["waveTime", "waveFlagPrimary", "waveFlagSecondary"] + list(RENAME)
META_VARS = ["metaSiteLabel", "metaLatitude", "metaLongitude",
             "metaWaterDepth", "metaShoreNormal"]

FLAG_PRIMARY = {1: "good", 2: "not_evaluated", 3: "questionable",
                4: "bad", 9: "missing"}
FLAG_SECONDARY = {0: "unspecified", 1: "insufficient_input", 2: "low_energy"}


def encode(projection):
    """Percent-encode the subscript brackets.

    curl and requests both refuse a bare '[' in a URL -- 'bad range in URL' --
    and the failure is silent enough to look like an empty server response.
    The server is fine with it; the client never sends it.
    """
    return projection.replace("[", "%5B").replace("]", "%5D")


def get_with_retry(url):
    delay = 2
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
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


def parse_ascii(text):
    """OPeNDAP .ascii -> {variable: [values]}.

    The response is a DDS header, a dashed rule, then for each variable a line
    'name[N]' or 'name' followed by its comma-separated values.
    """
    if "Error {" in text:
        raise ValueError(text.strip()[:300])
    body = text.split("-" * 20, 1)
    body = body[1] if len(body) > 1 else text
    out, current = {}, None
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        head = re.match(r"^([A-Za-z]\w*)(\[\d+\])?$", line)
        if head:
            current = head.group(1)
            out.setdefault(current, [])
            continue
        inline = re.match(r"^([A-Za-z]\w*),\s*(.*)$", line)
        if inline and inline.group(1) not in out:
            current = inline.group(1)
            out.setdefault(current, [])
            line = inline.group(2)
        if current is None:
            continue
        out[current].extend(v.strip() for v in line.split(",") if v.strip())
    return out


def to_float(values):
    return [float(v) if v not in ("", "nan") else float("nan") for v in values]


def is_fill(series):
    return (series - FILL_VALUE).abs() < FILL_TOL


def is_denormal(series):
    """Non-zero but smaller than any real measurement -- garbage bits."""
    magnitude = series.abs()
    return (magnitude > 0) & (magnitude < TINY)


def audit_column(series):
    """How many of this column's values are fill, denormal, or exactly zero."""
    total = len(series)
    fill = int(is_fill(series).sum())
    denormal = int(is_denormal(series).sum())
    zero = int((series == 0).sum())
    blank = int(series.isna().sum())
    usable = total - fill - denormal - zero - blank
    return {"total": total, "fill": fill, "denormal": denormal, "zero": zero,
            "blank": blank, "usable": usable,
            "share": (usable / total) if total else 0.0}


def clean_fill(frame, columns, min_usable=MIN_USABLE, keep_degenerate=False):
    """Mask CDIP's fill values, then drop whatever is left unusable.

    Returns the frame and the audit, so the caller can print what it lost.
    Rows are never dropped: at SC130 waveHs is a healthy 0.47 m at the very
    timestamps where waveDm is -999.99, so dropping those rows would throw
    away good height and period to protect a direction column that has
    nothing in it anywhere.
    """
    audit, dropped = {}, []
    for column in columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        report = audit_column(series)
        audit[column] = report
        frame[column] = series.mask(is_fill(series) | is_denormal(series))
        if report["share"] < min_usable and not keep_degenerate:
            dropped.append(column)
    if dropped:
        frame = frame.drop(columns=dropped)
    return frame, audit, dropped


def print_audit(audit, dropped, rename=None):
    rename = rename or {}
    print("\n  column health (fill and denormal counted as missing):")
    for column, report in audit.items():
        label = rename.get(column, column)
        note = ""
        if report["fill"]:
            note += f"  fill {report['fill']:,}"
        if report["denormal"]:
            note += f"  denormal {report['denormal']:,}"
        if report["zero"]:
            note += f"  zero {report['zero']:,}"
        if report["blank"]:
            note += f"  blank {report['blank']:,}"
        mark = "  DROPPED" if column in dropped else ""
        print(f"    {label:<22} {100 * report['share']:5.1f}% usable{note}{mark}")
    if dropped:
        names = ", ".join(rename.get(c, c) for c in dropped)
        print(f"\n  dropped {names}: below {100 * MIN_USABLE:.0f}% usable at "
              "this point.\n  CDIP publishes the column; at this point it "
              "holds fill values and\n  denormals, not waves. Pass "
              "--keep-degenerate to write it anyway.")


def fetch(dataset, projection):
    url = f"{DODS}{dataset}.ascii?{encode(projection)}"
    resp = get_with_retry(url)
    if resp.status_code != 200:
        raise ValueError(f"HTTP {resp.status_code} for {projection}")
    return parse_ascii(resp.text)


def catalog_ids():
    """Every alongshore point id, read from the catalogue and cached."""
    if not os.path.exists(CACHE):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"fetching the catalogue (~27 MB, cached at {CACHE})")
        resp = get_with_retry(CATALOG)
        resp.raise_for_status()
        with open(CACHE, "w", encoding="utf-8", errors="replace") as handle:
            handle.write(resp.text)
    with open(CACHE, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    ids = sorted(set(ID_RE.findall(text)))
    if not ids:
        sys.exit(f"No dataset ids matched in {CACHE}. Delete it and retry; if "
                 "it persists the naming scheme changed -- update ID_RE.")
    return ids


def km_between(lat1, lon1, lat2, lon2):
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def point_location(dataset, cache):
    if dataset in cache:
        return cache[dataset]
    values = fetch(f"{dataset}_hindcast.nc", "metaLatitude,metaLongitude")
    lat = to_float(values.get("metaLatitude", []))
    lon = to_float(values.get("metaLongitude", []))
    if not lat or not lon:
        raise ValueError(f"{dataset} returned no coordinates")
    cache[dataset] = (lat[0], lon[0])
    return cache[dataset]


def load_point_cache(prefix):
    path = f"{DATA_DIR}/cdip_mop_points_{prefix}.csv"
    frame = None
    if os.path.exists(path):
        frame = pd.read_csv(path)
    if frame is None or frame.empty:
        return {}, path
    return {r["mop_id"]: (r["lat"], r["lon"]) for _, r in frame.iterrows()}, path


def save_point_cache(cache, path):
    rows = [{"mop_id": k, "lat": v[0], "lon": v[1]} for k, v in sorted(cache.items())]
    pd.DataFrame(rows).to_csv(path, index=False)


def regions(ids):
    """prefix -> its points. 'M' must not swallow 'MA' and 'MO', so the
    membership test is an anchored pattern, not startswith."""
    out = {}
    for prefix in sorted({re.match(r"^([A-Z]{1,2})", i).group(1) for i in ids}):
        members = [i for i in ids if re.match(rf"^{prefix}\d+$", i)]
        if members:
            out[prefix] = members
    return out


def choose_region(lat, lon, grouped, samples=5):
    """Sample a few points from EVERY region and take the closest.

    The earlier version tested only each region's first point against a
    400 km threshold and took the first that passed. Regions run south to
    north and are hundreds of kilometres long, so a region's first point says
    almost nothing about whether it contains the site -- B0001 is 367 km from
    Santa Cruz and B1788 is 252 km, while the SC series is 1.5 km away and was
    never reached because the loop had already broken out.
    """
    print(f"  {len(grouped)} regions; sampling {samples} points from each")
    scored = []
    for prefix, members in grouped.items():
        cache, path = load_point_cache(prefix)
        before = len(cache)
        step = max(1, len(members) // max(samples - 1, 1))
        probes = members[::step][:samples]
        if members[-1] not in probes:
            probes.append(members[-1])
        best = min(km_between(lat, lon, *point_location(m, cache)) for m in probes)
        if len(cache) != before:
            save_point_cache(cache, path)
        scored.append((best, prefix, len(members)))
    scored.sort()
    for best, prefix, count in scored[:4]:
        print(f"    {prefix:<3} {count:>5} points   nearest sampled {best:8.1f} km")
    if len(scored) > 4:
        print(f"    ({len(scored) - 4} further regions, all further away)")
    return scored[0][1], grouped[scored[0][1]]


def nearest_point(lat, lon, ids, stride=10):
    """Coarse scan then refine.

    The ids are NOT one per location -- SC135, SC140 and SC145 all report the
    same coordinates -- so an index cannot be interpolated from two endpoints.
    Positions have to be read. A stride scan followed by a dense sweep around
    the best candidate keeps that to a few dozen requests instead of hundreds.
    """
    prefix = re.match(r"^([A-Z]{1,2})", ids[0]).group(1)
    cache, path = load_point_cache(prefix)
    before = len(cache)

    def distance(dataset):
        plat, plon = point_location(dataset, cache)
        return km_between(lat, lon, plat, plon)

    coarse = ids[::stride] or ids
    print(f"  scanning {len(coarse)} of {len(ids)} {prefix} points "
          f"(every {stride}th), then refining")
    best = min(coarse, key=distance)
    index = ids.index(best)
    window = ids[max(0, index - stride):index + stride + 1]
    best = min(window, key=distance)

    if len(cache) != before:
        save_point_cache(cache, path)
        print(f"  cached {len(cache)} point locations in {path}")
    plat, plon = cache[best]
    return best, plat, plon, km_between(lat, lon, plat, plon)


def series_length(dataset):
    resp = get_with_retry(f"{DODS}{dataset}.dds")
    resp.raise_for_status()
    sizes = {name: int(size) for name, dim, size in DDS_RE.findall(resp.text)
             if name == dim}
    if "waveTime" not in sizes:
        sys.exit(f"{dataset}: no waveTime dimension in the DDS.")
    return sizes["waveTime"], resp.text


def available(dds_text, wanted):
    present = set(re.findall(r"\b(\w+)\[", dds_text)) | set(
        re.findall(r"Float32 (\w+);", dds_text)) | set(
        re.findall(r"String (\w+);", dds_text))
    return [v for v in wanted if v in present]


def pull_series(dataset, length, variables):
    frames = []
    for start in range(0, length, CHUNK):
        stop = min(start + CHUNK, length) - 1
        proj = ",".join(f"{v}[{start}:1:{stop}]" for v in variables)
        values = fetch(dataset, proj)
        block = {}
        for name in variables:
            got = values.get(name, [])
            if len(got) != stop - start + 1:
                raise ValueError(
                    f"{dataset} {name}: expected {stop - start + 1} values, "
                    f"got {len(got)} — the .ascii layout is not what "
                    "parse_ascii assumes")
            block[name] = to_float(got)
        frames.append(pd.DataFrame(block))
        print(f"    {stop + 1}/{length}", end="\r", flush=True)
    print(" " * 30, end="\r")
    return pd.concat(frames, ignore_index=True)


def apply_qc(frame, label):
    """Drop everything the model does not call good, and say what went."""
    total = len(frame)
    primary = frame["waveFlagPrimary"].round().astype("Int64")
    counts = primary.value_counts().sort_index()
    print(f"  {label}: {total:,} rows")
    for code, count in counts.items():
        meaning = FLAG_PRIMARY.get(int(code), "?")
        print(f"    waveFlagPrimary {int(code)} {meaning:<14} {int(count):>8,}"
              + ("   KEPT" if int(code) == 1 else "   dropped"))
    if "waveFlagSecondary" in frame.columns:
        secondary = frame.loc[primary == 1, "waveFlagSecondary"]
        secondary = secondary.round().astype("Int64").value_counts().sort_index()
        for code, count in secondary.items():
            meaning = FLAG_SECONDARY.get(int(code), "?")
            note = "  <- model ran without enough buoy input" if int(code) == 1 else ""
            print(f"    among kept rows, secondary {int(code)} "
                  f"{meaning:<18} {int(count):>8,}{note}")
    return frame[primary == 1].copy()


def resolve_camera(text):
    if not os.path.exists(CANDIDATES_CSV):
        return None
    frame = pd.read_csv(CANDIDATES_CSV)
    hits = frame[frame["camera"].astype(str).str.contains(text, case=False, na=False)]
    if hits.empty:
        return None
    row = hits.iloc[0]
    if len(hits) > 1:
        print(f"{len(hits)} cameras match {text!r}; using {row['camera']!r}")
    return str(row["camera"]), float(row["lat"]), float(row["lon"])


def grid_slug(name):
    """The slug the rest of the project writes."""
    return "".join(c if c.isalnum() else "_" for c in str(name))[:48]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", help="substring of a name in camera_candidates.csv")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--name", help="label for --lat/--lon output")
    ap.add_argument("--mop", help="use this MOP id directly, skipping the search")
    ap.add_argument("--max-km", type=float, default=MAX_KM,
                    help="refuse a point further than this from the site")
    ap.add_argument("--stride", type=int, default=10,
                    help="coarse scan step when searching for the nearest point")
    ap.add_argument("--keep-degenerate", action="store_true",
                    help="write columns that are mostly fill values anyway")
    ap.add_argument("--probe", action="store_true",
                    help="find the point, print its metadata and five rows, stop")
    args = ap.parse_args()

    if args.lat is not None and args.lon is not None:
        name, lat, lon = args.name or f"{args.lat},{args.lon}", args.lat, args.lon
    elif args.camera:
        found = resolve_camera(args.camera)
        if not found:
            sys.exit(f"No camera matching {args.camera!r} in {CANDIDATES_CSV}.")
        name, lat, lon = found
    else:
        sys.exit("Give --camera, or --lat and --lon.")
    print(f"{name}  ({lat:.4f}, {lon:.4f})")

    if args.mop:
        mop = args.mop
        loc = fetch(f"{mop}_hindcast.nc", "metaLatitude,metaLongitude")
        plat = to_float(loc["metaLatitude"])[0]
        plon = to_float(loc["metaLongitude"])[0]
        away = km_between(lat, lon, plat, plon)
    else:
        ids = catalog_ids()
        prefix, prefix_ids = choose_region(lat, lon, regions(ids))
        mop, plat, plon, away = nearest_point(lat, lon, prefix_ids, args.stride)

    print(f"\nnearest MOP point: {mop}  ({plat:.5f}, {plon:.5f})  "
          f"{away:.2f} km from the site")
    if away > args.max_km:
        sys.exit(
            f"\nREFUSING: {away:.1f} km is further than --max-km {args.max_km:.0f}.\n"
            "These are not this site's waves. MOP covers California only; if the\n"
            "site is Californian and this still fires, the region search picked the\n"
            "wrong series — pass --mop explicitly, or raise --max-km if you really\n"
            "mean to use a point this far away.")

    meta = fetch(f"{mop}_hindcast.nc", ",".join(META_VARS[1:]))
    depth = to_float(meta.get("metaWaterDepth", ["nan"]))[0]
    normal = to_float(meta.get("metaShoreNormal", ["nan"]))[0]
    print(f"  water depth {depth:.1f} m,  shore normal {normal:.1f}°")
    print("  analyze_drivers.SHORE_NORMAL_DEG is a bearing read off a map. "
          "Compare it\n  to the line above before trusting any onshore or "
          "axial-offset number.")

    if args.probe:
        length, dds = series_length(f"{mop}_hindcast.nc")
        print(f"\nhindcast has {length:,} hourly steps")
        have = available(dds, SERIES_VARS + ["waveModelInputSource"])
        print(f"  variables present: {', '.join(have)}")
        sample = fetch(f"{mop}_hindcast.nc",
                       ",".join(f"{v}[0:1:4]" for v in have if v != "waveTime")
                       + ",waveTime[0:1:4]")
        print("\n  first five rows:")
        for key, values in sample.items():
            series = pd.Series(values, dtype="float64") if key != "metaSiteLabel" \
                else None
            note = ""
            if series is not None and key in RENAME:
                bad = int((is_fill(series) | is_denormal(series) |
                           (series == 0)).sum())
                if bad:
                    note = f"   <- {bad}/{len(series)} fill or denormal"
            print(f"    {key:<22} {values}{note}")
        print("\n  Five rows is not a verdict; the fill values at SC130 start "
              "mid-record.\n  The full run audits every column and drops the "
              "unusable ones.")
        print("\nRe-run without --probe to write the CSV.")
        return

    parts = []
    for product in ("hindcast", "nowcast"):
        dataset = f"{mop}_{product}.nc"
        length, dds = series_length(dataset)
        variables = available(dds, SERIES_VARS)
        print(f"\n  {product}: {length:,} steps, "
              f"{len(variables)} variables, {math.ceil(length / CHUNK)} requests")
        frame = pull_series(dataset, length, variables)
        frame = apply_qc(frame, product)
        frame["product"] = product
        parts.append(frame)

    out = pd.concat(parts, ignore_index=True)
    out["time"] = pd.to_datetime(out["waveTime"], unit="s", utc=True)
    # The two products abut at one shared timestamp, not a gap.
    before = len(out)
    out = out.drop_duplicates(subset="time", keep="first").sort_values("time")
    if before != len(out):
        print(f"\n  {before - len(out)} duplicated timestamp(s) at the "
              "hindcast/nowcast seam, dropped")

    out, audit, dropped = clean_fill(out, list(RENAME),
                                     keep_degenerate=args.keep_degenerate)
    print_audit(audit, dropped, RENAME)

    out = out.rename(columns=RENAME)
    keep = ["time"] + [c for c in RENAME.values() if c in out.columns] + ["product"]
    out = out[keep]
    out["mop_id"] = mop
    out["shore_normal_deg"] = normal
    out["water_depth_m"] = depth

    os.makedirs(DATA_DIR, exist_ok=True)
    path = f"{DATA_DIR}/mop_{grid_slug(name)}.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}  ({len(out):,} hours, "
          f"{out['time'].min():%Y-%m-%d} to {out['time'].max():%Y-%m-%d})")
    for column in RENAME.values():
        if column in out.columns:
            share = 100.0 * out[column].notna().mean()
            print(f"  {column:<22} {share:5.1f}% populated")
    if dropped:
        print("  not written: " + ", ".join(RENAME[c] for c in dropped))


if __name__ == "__main__":
    main()
