"""Pull WebCOOS's own rip-detection product for the California cameras.

explore_webcoos_products.py already established that rip-detection-results
exists on Walton Lighthouse, Santa Cruz (35,158 elements) under the
raw-video-data feed. This script downloads it.

It resolves the feed and product by SLUG, not by label. That matters:
pywebcoos.API._get_camera_products compares

    feed['data']['common']['label'] == 'raw-video-data'

-- a label compared against a slug. If WebCOOS labels the feed anything other
than the literal string "raw-video-data" (e.g. "Raw Video Data"), that loop
never matches, `products` is never assigned, and the library raises
UnboundLocalError from inside download(). The same label-matching applies to
product names, so pywebcoos wants the product LABEL where the API catalogue
shows a slug. This script therefore talks to /elements/ directly, and
--via-pywebcoos runs the library path instead so you can see which works.

The output format of the rip product is not assumed. Run --probe first: it
downloads a short window and reports what actually came down before any
parsing is attempted.

    export $(grep -v '^#' .env | xargs)
    python pull_rip_detection.py --list
    python pull_rip_detection.py --probe
    python pull_rip_detection.py --pull --start 2025-06-01 --end 2025-09-01

Writes files under data/rip_detection/<camera-slug>/ and, when the payload is
tabular, a combined CSV alongside them.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

API_BASE = "https://app.webcoos.org/webcoos/api/v1"
OUT_DIR = "data/rip_detection"
RAW_DUMP = "webcoos_assets_raw.json"

# The product we are after, by slug. Matched case-insensitively, and any
# product whose slug contains "rip" is offered as a fallback.
PRODUCT_SLUG = "rip-detection-results"
DEFAULT_CAMERA = "Walton Lighthouse"
PROBE_HOURS = 6

TIMEOUT = 60
PAUSE = 0.2  # between downloads, to stay a polite client


def headers():
    token = os.environ.get("WEBCOOS_TOKEN")
    if not token:
        sys.exit("WEBCOOS_TOKEN is not set. See .env.example.")
    return {"Authorization": f"Token {token}", "Accept": "application/json"}


def dig(node, *path):
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

def load_assets(refresh=False):
    """Assets list, from the local dump when present so re-runs are free."""
    if refresh or not os.path.exists(RAW_DUMP):
        resp = requests.get(f"{API_BASE}/assets/", headers=headers(), timeout=TIMEOUT)
        resp.raise_for_status()
        with open(RAW_DUMP, "w") as fh:
            json.dump(resp.json(), fh, indent=2)
        print(f"wrote {RAW_DUMP}")
    with open(RAW_DUMP) as fh:
        return json.load(fh).get("results", [])


def camera_products(asset):
    """[(feed_label, feed_slug, product_label, product_slug, service_slug, count)]"""
    rows = []
    for feed in asset.get("feeds") or []:
        feed_label = dig(feed, "data", "common", "label")
        feed_slug = dig(feed, "data", "common", "slug")
        for product in feed.get("products") or []:
            product_label = dig(product, "data", "common", "label")
            product_slug = dig(product, "data", "common", "slug")
            for service in product.get("services") or []:
                rows.append((
                    feed_label, feed_slug, product_label, product_slug,
                    dig(service, "data", "common", "slug"),
                    dig(service, "elements", "count"),
                ))
    return rows


def find_camera(assets, name):
    """Exact label match, else a unique case-insensitive substring match."""
    labels = [dig(a, "data", "common", "label") or "" for a in assets]
    for asset, label in zip(assets, labels):
        if label == name:
            return asset
    hits = [(a, l) for a, l in zip(assets, labels) if name.lower() in l.lower()]
    if len(hits) == 1:
        print(f"camera {name!r} -> {hits[0][1]!r}")
        return hits[0][0]
    if not hits:
        sys.exit(f"No camera matching {name!r}. Run --list to see the names.")
    sys.exit("Ambiguous camera name; matches:\n  " +
             "\n  ".join(l for _, l in hits))


def find_rip_service(asset):
    """The service slug for the rip product on this camera, or exit."""
    rows = camera_products(asset)
    exact = [r for r in rows if (r[3] or "").lower() == PRODUCT_SLUG]
    loose = [r for r in rows if "rip" in (r[3] or "").lower()]
    chosen = exact or loose
    if not chosen:
        print("Products on this camera:")
        for _, _, plabel, pslug, _, count in rows:
            print(f"  {pslug}  ({plabel!r}, {count or 0:,} elements)")
        sys.exit(f"No product matching {PRODUCT_SLUG!r} on this camera.")
    if not exact:
        print(f"No exact {PRODUCT_SLUG!r}; using {chosen[0][3]!r}")
    feed_label, feed_slug, product_label, product_slug, service_slug, count = chosen[0]
    print(f"feed    {feed_slug!r}   (label {feed_label!r})")
    print(f"product {product_slug!r}   (label {product_label!r})")
    print(f"service {service_slug!r}   {count or 0:,} elements")
    if feed_label != "raw-video-data":
        print("  note: pywebcoos matches the feed LABEL against the literal string")
        print("        'raw-video-data', so its download() cannot reach this feed.")
    return service_slug, product_label


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------

def fetch_elements(service_slug, start, end, interval_minutes=None):
    """Every element for a service in [start, end), oldest first.

    start/end are timezone-aware UTC datetimes. interval_minutes, if given,
    keeps only elements whose minute-of-hour is a multiple of it -- the same
    thinning pywebcoos applies, done here so a year of frames is tractable.
    """
    params = {
        "service": service_slug,
        "starting_after": start.isoformat().replace("+00:00", "Z"),
        "starting_before": end.isoformat().replace("+00:00", "Z"),
    }
    url = f"{API_BASE}/elements/"
    out, page = [], 1
    while url:
        resp = requests.get(url, headers=headers(), params=params, timeout=TIMEOUT)
        if resp.status_code != 200:
            sys.exit(f"/elements/ page {page} returned {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        results = payload.get("results") or []
        out.extend(results)
        print(f"  page {page}: {len(results)} elements (total {len(out)})")
        url = dig(payload, "pagination", "next")
        params = None  # the next URL already carries the query
        page += 1
        if url:
            time.sleep(PAUSE)

    rows = []
    for element in out:
        stamp = dig(element, "data", "extents", "temporal", "min")
        href = dig(element, "data", "properties", "url")
        if not stamp or not href:
            continue
        when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if interval_minutes and when.minute % interval_minutes:
            continue
        rows.append({"timestamp": when, "url": href,
                     "filename": os.path.basename(href)})
    rows.sort(key=lambda r: r["timestamp"])
    print(f"  {len(rows)} elements with a usable url"
          + (f" after thinning to every {interval_minutes} min" if interval_minutes else ""))
    return rows


def download(rows, save_dir):
    """Fetch each element to save_dir, skipping files already present."""
    os.makedirs(save_dir, exist_ok=True)
    got, skipped = [], 0
    for i, row in enumerate(rows, 1):
        path = os.path.join(save_dir, row["filename"].replace(":", ""))
        row["path"] = path
        if os.path.exists(path) and os.path.getsize(path) > 0:
            skipped += 1
            got.append(row)
            continue
        try:
            resp = requests.get(row["url"], stream=True, timeout=TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  failed {row['filename']}: {exc}")
            continue
        with open(path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                fh.write(chunk)
        got.append(row)
        if i % 50 == 0:
            print(f"  downloaded {i}/{len(rows)}")
        time.sleep(PAUSE)
    print(f"  {len(got)} files in {save_dir} ({skipped} already there)")
    return got


# ---------------------------------------------------------------------------
# Probe -- what IS this product?
# ---------------------------------------------------------------------------

def probe(rows):
    """Report what came down, rather than assuming a format."""
    if not rows:
        print("Nothing downloaded, so there is nothing to describe.")
        return
    kinds = Counter(os.path.splitext(r["path"])[1].lower() or "(no extension)"
                    for r in rows)
    sizes = [os.path.getsize(r["path"]) for r in rows]
    print("\n=== WHAT CAME DOWN ===")
    for ext, n in kinds.most_common():
        print(f"  {ext:<12} {n:>5} files")
    print(f"  sizes: min {min(sizes):,} max {max(sizes):,} "
          f"median {int(pd.Series(sizes).median()):,} bytes")

    sample = rows[0]["path"]
    print(f"\n=== FIRST 2 KB OF {os.path.basename(sample)} ===")
    with open(sample, "rb") as fh:
        head = fh.read(2048)
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        print(f"  binary (starts with {head[:16].hex(' ')}) — imagery or video,")
        print("  not a table. The rip signal would have to come from the pixels.")
        return
    print(text)
    stripped = text.lstrip()
    if stripped.startswith(("{", "[")):
        try:
            with open(sample) as fh:
                parsed = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"  looks like JSON but did not parse: {exc}")
            return
        print("\n=== JSON SHAPE ===")
        describe_json(parsed)


def describe_json(node, prefix="", depth=0):
    pad = "  " * (depth + 1)
    if isinstance(node, dict):
        for key, value in list(node.items())[:25]:
            kind = type(value).__name__
            if isinstance(value, (dict, list)):
                print(f"{pad}{key}: {kind}")
                if depth < 2:
                    describe_json(value, prefix, depth + 1)
            else:
                print(f"{pad}{key}: {kind} = {str(value)[:60]}")
    elif isinstance(node, list):
        print(f"{pad}[{len(node)} items]")
        if node and depth < 2:
            describe_json(node[0], prefix, depth + 1)


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def build_table(rows, out_csv):
    """Turn tabular payloads into one CSV. Always writes the element index.

    JSON and CSV payloads are flattened and stamped with the element time.
    Anything binary gets an index only -- filename and timestamp -- which is
    still what you need to join frames to the observation table.
    """
    index = pd.DataFrame([{"timestamp": r["timestamp"], "filename": r["filename"],
                           "path": r["path"], "url": r["url"]} for r in rows])
    index_path = out_csv.replace(".csv", "_index.csv")
    index.to_csv(index_path, index=False)
    print(f"  wrote {index_path}  ({len(index)} elements)")

    frames = []
    for row in rows:
        ext = os.path.splitext(row["path"])[1].lower()
        try:
            if ext == ".json":
                with open(row["path"]) as fh:
                    payload = json.load(fh)
                frame = pd.json_normalize(payload)
            elif ext == ".csv":
                frame = pd.read_csv(row["path"])
            else:
                continue
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"  skipped {row['filename']}: {exc}")
            continue
        frame["timestamp"] = row["timestamp"]
        frame["source_file"] = row["filename"]
        frames.append(frame)

    if not frames:
        print("  payloads are not tabular — the index is the whole table.")
        return None
    table = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    table.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}  ({len(table)} rows, {len(table.columns)} columns)")
    print("  columns: " + ", ".join(map(str, table.columns[:20])))
    return table


# ---------------------------------------------------------------------------
# pywebcoos path, for comparison
# ---------------------------------------------------------------------------

def via_pywebcoos(camera_label, product_label, start, end, interval, save_dir):
    try:
        from pywebcoos import API
    except ImportError:
        sys.exit("pywebcoos is not installed: pip install -r requirements.txt")
    api = API(os.environ["WEBCOOS_TOKEN"])
    print("products pywebcoos reports:", api.get_products(camera_label))
    print("inventory:", api.get_inventory(camera_label, product_label))
    names = api.download(camera_label, product_label,
                         start.strftime("%Y%m%d%H%M"), end.strftime("%Y%m%d%H%M"),
                         interval, save_dir)
    print(f"pywebcoos downloaded {len(names)} files")
    return names


# ---------------------------------------------------------------------------

def list_cameras(assets, state="California"):
    for asset in assets:
        if state and dig(asset, "data", "properties", "state_or_territory") != state:
            continue
        label = dig(asset, "data", "common", "label")
        rows = camera_products(asset)
        rip = [r for r in rows if "rip" in (r[3] or "").lower()]
        mark = f"  <- {len(rip)} rip product(s)" if rip else ""
        print(f"{label!r}{mark}")
        for _, feed_slug, _, product_slug, _, count in rows:
            print(f"    {feed_slug} / {product_slug}  ({count or 0:,} elements)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default=DEFAULT_CAMERA)
    ap.add_argument("--start", help="UTC date, YYYY-MM-DD")
    ap.add_argument("--end", help="UTC date, YYYY-MM-DD (exclusive)")
    ap.add_argument("--interval", type=int, default=None,
                    help="keep only elements on this minute spacing")
    ap.add_argument("--list", action="store_true", help="list CA cameras and products")
    ap.add_argument("--probe", action="store_true",
                    help=f"download {PROBE_HOURS}h and describe the payload")
    ap.add_argument("--pull", action="store_true", help="download the full range")
    ap.add_argument("--refresh", action="store_true", help="re-fetch the asset catalogue")
    ap.add_argument("--via-pywebcoos", action="store_true",
                    help="use the library's download() instead of /elements/")
    args = ap.parse_args()

    assets = load_assets(args.refresh)
    if args.list:
        list_cameras(assets)
        return
    if not (args.probe or args.pull or args.via_pywebcoos):
        ap.error("choose one of --list, --probe, --pull")

    asset = find_camera(assets, args.camera)
    camera_label = dig(asset, "data", "common", "label")
    service_slug, product_label = find_rip_service(asset)
    save_dir = os.path.join(OUT_DIR, slugify(camera_label))

    end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
           if args.end else datetime.now(timezone.utc))
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif args.probe:
        start = end - timedelta(hours=PROBE_HOURS)
    else:
        start = end - timedelta(days=365)
    print(f"\nrange {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC")

    if args.via_pywebcoos:
        via_pywebcoos(camera_label, product_label, start, end,
                      args.interval or 30, save_dir)
        return

    rows = fetch_elements(service_slug, start, end, args.interval)
    if not rows:
        print("\nNo elements in that range. The product may not cover it —")
        print("widen --start/--end, or check the element count printed above.")
        return
    print(f"  first {rows[0]['timestamp']:%Y-%m-%d %H:%M}"
          f"  last {rows[-1]['timestamp']:%Y-%m-%d %H:%M} UTC")

    got = download(rows, save_dir)
    if args.probe:
        probe(got)
        print("\nOnce the format above is clear, re-run with --pull and a real range.")
    else:
        build_table(got, os.path.join(OUT_DIR, f"rip_{slugify(camera_label)}.csv"))


if __name__ == "__main__":
    main()
