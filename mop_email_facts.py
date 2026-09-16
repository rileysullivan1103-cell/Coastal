#!/usr/bin/env python3
"""The facts a CDIP support email needs, read out of the MOP CSV.

Writing to CDIP about a variable that comes back empty means quoting three
things they can check in thirty seconds: which point, what span, and the
literal OPeNDAP request. Retyping any of those from memory risks telling
them something the data does not say, so this reads all of it back out of
the file pull_cdip_mop.py wrote.

It also prints the per-column populated share. That guards a specific
mistake: describing ONE variable as degenerate when the audit flagged
several. waveSxx and waveDm opened the SC130 record with denormals too, and
an email that names only waveSxy is contradicted by the first thing the
reader checks.

    python mop_email_facts.py
    python mop_email_facts.py --path data/mop_walton.csv --variable waveSxx
"""
import argparse
import glob
import os

import pandas as pd

from pull_cdip_mop import RENAME

DODS = ("https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/"
        "MOP_alongshore/")
# pull_cdip_mop.py requests this many rows per OPeNDAP call, so this is the
# projection CDIP would actually have served. Quoting a shape we never sent
# would send them chasing a request that does not exist.
CHUNK = 20000
DEFAULT_VARIABLE = "waveSxy"


def find_csv(path=None):
    if path:
        return path
    matches = sorted(glob.glob(os.path.join("data", "mop_*.csv")))
    if not matches:
        raise SystemExit(
            "No data/mop_*.csv found. Run pull_cdip_mop.py --camera Walton "
            "first, or pass --path.")
    return matches[0]


def load(path):
    return pd.read_csv(path, parse_dates=["time"])


def point_id(frame):
    """The MOP id out of the file, never assumed.

    An earlier version of the pull matched Santa Cruz to the Santa Barbara
    series and wrote a distant point under the camera's name. Reading the id
    back from the column means the email cannot name a point the CSV does
    not hold.
    """
    ids = sorted(set(frame["mop_id"].dropna()))
    if len(ids) != 1:
        raise SystemExit(f"expected one mop_id in the file, found {ids}")
    return ids[0]


def span(frame):
    return frame["time"].min(), frame["time"].max()


def by_product(frame):
    if "product" not in frame.columns:
        return None
    return frame.groupby("product")["time"].agg(["min", "max", "count"])


def column_for(frame, variable):
    """Match a CDIP variable name to the column pull_cdip_mop.py wrote.

    The rename is NOT a prefix-preserving transform: waveSxy is written as
    radiation_stress_sxy, waveSxx as radiation_stress_sxx. An earlier version
    here matched on letters alone, so it looked for "wavesxy", found nothing,
    and reported a 100%-populated column as absent. Read the real map instead
    of inferring one, and keep the letters-only pass only as a fallback for a
    column this map does not mention.
    """
    if variable in RENAME and RENAME[variable] in frame.columns:
        return RENAME[variable]
    want = variable.lower().replace("_", "")
    for column in frame.columns:
        if column.lower().replace("_", "") == want:
            return column
    return None


def populated(frame, variable):
    """(present, total, first_valid, last_valid) for one variable.

    present is None when the column is absent -- the audit in
    pull_cdip_mop.py DROPS a column below MIN_USABLE, so a missing column
    means 'entirely fill', not 'not requested'. Those two read the same in
    the CSV and must not read the same in the email.
    """
    column = column_for(frame, variable)
    if column is None:
        return None, len(frame), None, None
    good = frame[frame[column].notna()]
    if good.empty:
        return 0, len(frame), None, None
    return len(good), len(frame), good["time"].min(), good["time"].max()


# Everything pull_cdip_mop.py writes that is not a measurement. Listing the
# exclusions rather than a prefix keeps radiation_stress_sxy in the report;
# filtering on "wave_" silently dropped it.
META_COLUMNS = frozenset(["time", "hour", "product", "mop_id",
                          "shore_normal_deg", "water_depth_m"])


def value_columns(frame):
    return [c for c in frame.columns if c not in META_COLUMNS]


def request_url(mop, variable, product="hindcast", chunk=CHUNK):
    return f"{DODS}{mop}_{product}.nc.ascii?{variable}[0:1:{chunk - 1}]"


def points_tables():
    out = []
    for path in sorted(glob.glob(os.path.join("data",
                                              "cdip_mop_points_*.csv"))):
        out.append((path, pd.read_csv(path)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", help="the MOP CSV; default is data/mop_*.csv")
    parser.add_argument("--variable", default=DEFAULT_VARIABLE,
                        help=f"CDIP variable to detail (default {DEFAULT_VARIABLE})")
    args = parser.parse_args()

    path = find_csv(args.path)
    frame = load(path)
    mop = point_id(frame)
    first, last = span(frame)

    print(f"file  : {path}  ({len(frame):,} rows)")
    print(f"point : {mop}")
    print(f"span  : {first} -> {last}")

    products = by_product(frame)
    if products is not None:
        print()
        print(products.to_string())

    present, total, good_first, good_last = populated(frame, args.variable)
    print()
    if present is None:
        print(f"{args.variable}: column absent -- the fill audit dropped it, "
              "so it was 100% unusable.")
        print("  Re-run pull_cdip_mop.py with --keep-degenerate to see the "
              "raw values.")
    else:
        print(f"{args.variable}: {present:,} usable of {total:,} "
              f"({total - present:,} fill)")
        if good_first is not None:
            print(f"  usable hours run {good_first} -> {good_last}")

    print()
    print("populated share, every column written:")
    for column in value_columns(frame):
        print(f"  {column:<18} {100 * frame[column].notna().mean():5.1f}%")

    for points_path, table in points_tables():
        print()
        print(points_path)
        print(table.to_string(index=False))

    print()
    print("request to quote in the email:")
    print(f"  {request_url(mop, args.variable)}")


if __name__ == "__main__":
    main()
