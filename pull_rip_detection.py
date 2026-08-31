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

    python pull_rip_detection.py --list
    python pull_rip_detection.py --inventory
    python pull_rip_detection.py --coverage --start 2025-06-01 --end 2025-09-01
    python pull_rip_detection.py --probe
    python pull_rip_detection.py --pull --start 2025-06-01 --end 2025-09-01

Writes files under data/rip_detection/<camera-slug>/ and, when the payload is
tabular, a combined CSV alongside them.
"""

import env  # noqa: F401  -- loads .env into os.environ

import argparse
import concurrent.futures
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
# The imagery product used as the DENOMINATOR. The rip feed publishes an
# element only when the detector fires, so on its own it cannot distinguish
# "no rip" from "no image". Enumerating stills gives the hours the camera was
# actually looking, which turns the gaps into observed zeros instead of
# assumed ones.
STILLS_SLUG = "one-minute-stills"
PROBE_HOURS = 6

TIMEOUT = 60
# Payloads are ~800 bytes each and there are 35k of them, so a per-request
# sleep dominates the runtime: 0.2s each is two hours of pure waiting. A small
# thread pool with no sleep is both faster and gentler than one long serial
# hammering.
WORKERS = 6
# Between pages of /elements/, which is a listing endpoint rather than a CDN.
PAGE_PAUSE = 0.2


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


def find_service(asset, exact_slug, hint, label="product"):
    """(service_slug, product_label) for a product on this camera, or exit.

    Matched on the product SLUG, with `hint` as a substring fallback so a
    renamed product is still found rather than silently missing.
    """
    rows = camera_products(asset)
    exact = [r for r in rows if (r[3] or "").lower() == exact_slug]
    loose = [r for r in rows if hint in (r[3] or "").lower()]
    chosen = exact or loose
    if not chosen:
        print(f"Products on this camera:")
        for _, _, plabel, pslug, _, count in rows:
            print(f"  {pslug}  ({plabel!r}, {count or 0:,} elements)")
        sys.exit(f"No {label} matching {exact_slug!r} on this camera.")
    if not exact:
        print(f"No exact {exact_slug!r}; using {chosen[0][3]!r}")
    feed_label, feed_slug, product_label, product_slug, service_slug, count = chosen[0]
    print(f"feed    {feed_slug!r}   (label {feed_label!r})")
    print(f"{label} {product_slug!r}   (label {product_label!r})")
    print(f"service {service_slug!r}   {count or 0:,} elements")
    if feed_label != "raw-video-data":
        print("  note: pywebcoos matches the feed LABEL against the literal string")
        print("        'raw-video-data', so its download() cannot reach this feed.")
    return service_slug, product_label


def find_rip_service(asset):
    return find_service(asset, PRODUCT_SLUG, "rip")


def find_stills_service(asset, slug=None):
    """The imagery product that says WHEN the camera was actually looking."""
    return find_service(asset, (slug or STILLS_SLUG).lower(), "still",
                        label="stills ")


# ---------------------------------------------------------------------------
# Inventory -- WHEN does this product have data?
# ---------------------------------------------------------------------------

# The column names pywebcoos assigns to the inventory rows. Used only when the
# row width matches; otherwise the columns are left unnamed and the range is
# recovered by scanning for parseable timestamps, so a schema change degrades
# into a weaker answer rather than a wrong one.
INVENTORY_COLUMNS = ["Bin Start", "Has Data?", "Bin End", "Count", "Bytes",
                     "Data Start", "Data End"]


def fetch_inventory(service_slug):
    """The service's data inventory as a DataFrame, or None."""
    url = f"{API_BASE}/services/{service_slug}/inventory/"
    resp = requests.get(url, headers=headers(), timeout=TIMEOUT)
    if resp.status_code != 200:
        print(f"  inventory returned {resp.status_code}: {resp.text[:200]}")
        return None
    results = resp.json().get("results") or []
    if not results:
        print("  inventory is empty")
        return None
    values = results[0].get("values") or []
    if not values:
        print("  inventory has no bins")
        return None
    width = len(values[0])
    if width == len(INVENTORY_COLUMNS):
        return pd.DataFrame(values, columns=INVENTORY_COLUMNS)
    print(f"  inventory rows have {width} columns, not {len(INVENTORY_COLUMNS)};"
          " reading them positionally")
    print(f"  first row: {values[0]}")
    return pd.DataFrame(values, columns=[f"col{i}" for i in range(width)])


def _timestamps(frame, *preferred):
    """Parseable timestamps, trying the named columns in order.

    `preferred` is a fallback chain, not a set: "Data Start" is the real
    coverage bound and "Bin Start" only the bin the data sits in, so the
    second is consulted only when the first yields nothing. Taking both would
    stretch the reported range out to the bin edges.
    """
    for column in preferred:
        if column not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True).dropna()
        if not parsed.empty:
            return parsed.tolist()

    # Unrecognised schema: scan, but skip numeric columns. A bare integer is
    # a valid epoch to pd.to_datetime, so a row count would otherwise parse
    # as 1970 and become the earliest date in the range.
    stamps = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        parsed = pd.to_datetime(series, errors="coerce", utc=True).dropna()
        stamps.extend(parsed.tolist())
    return stamps


def inventory_range(frame):
    """(first, last) datetimes actually covered, or (None, None)."""
    if frame is None or frame.empty:
        return None, None
    with_data = frame
    if "Has Data?" in frame.columns:
        flag = frame["Has Data?"]
        truthy = flag.astype(str).str.lower().isin(["true", "1", "yes"])
        if truthy.any():
            with_data = frame[truthy]
    starts = _timestamps(with_data, "Data Start", "Bin Start")
    ends = _timestamps(with_data, "Data End", "Bin End")
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def report_inventory(service_slug):
    """Print what the inventory says, and return its (first, last)."""
    frame = fetch_inventory(service_slug)
    first, last = inventory_range(frame)
    if first is None:
        print("  inventory gave no usable date range")
        return None, None
    print(f"  data runs {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} UTC")
    if "Count" in frame.columns:
        counts = pd.to_numeric(frame["Count"], errors="coerce").fillna(0)
        populated = int((counts > 0).sum())
        print(f"  {int(counts.sum()):,} elements across {populated} populated"
              f" bins of {len(frame)}")
    stale = (pd.Timestamp.now(tz="UTC") - last).days
    if stale > 1:
        print(f"  last data is {stale} days old — this product is not live")
    return first, last


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
            time.sleep(PAGE_PAUSE)

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


def _fetch_one(row, save_dir):
    """Download one element. Returns the row on success, None on failure."""
    path = os.path.join(save_dir, row["filename"].replace(":", ""))
    row["path"] = path
    if os.path.exists(path) and os.path.getsize(path) > 0:
        row["cached"] = True
        return row
    try:
        resp = requests.get(row["url"], timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        row["error"] = str(exc)
        return None
    with open(path, "wb") as fh:
        fh.write(resp.content)
    row["cached"] = False
    return row


def download(rows, save_dir, workers=WORKERS):
    """Fetch every element into save_dir, skipping files already present.

    Re-running is cheap: anything already on disk is kept, so an interrupted
    pull resumes rather than starting over.
    """
    os.makedirs(save_dir, exist_ok=True)
    got, failed = [], []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, row, save_dir): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            done += 1
            if result is None:
                failed.append(futures[future])
            else:
                got.append(result)
            if done % 500 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)}")
    cached = sum(1 for r in got if r.get("cached"))
    print(f"  {len(got)} files in {save_dir} ({cached} already there)")
    if failed:
        print(f"  {len(failed)} failed, e.g. {failed[0].get('error', '?')[:120]}")
        print("  re-run the same command to retry only those")
    got.sort(key=lambda r: r["timestamp"])
    return got


# ---------------------------------------------------------------------------
# Coverage -- when was the camera actually looking?
# ---------------------------------------------------------------------------

def build_coverage(rows, out_csv):
    """Hourly count of images captured. Nothing is downloaded to produce it.

    Element enumeration returns a timestamp per element, which is all a
    denominator needs. Downloading the imagery itself would be hundreds of
    thousands of files to answer a question the listing already answers.
    """
    if not rows:
        print("  no stills in that range — no coverage to write")
        return None
    frame = pd.DataFrame({"timestamp": [r["timestamp"] for r in rows]})
    frame["hour"] = pd.to_datetime(frame["timestamp"], utc=True).dt.floor("h")
    hourly = (frame.groupby("hour").size().rename("images").reset_index())
    hourly.to_csv(out_csv, index=False)
    span = hourly["hour"].max() - hourly["hour"].min()
    possible = int(span.total_seconds() // 3600) + 1
    print(f"  wrote {out_csv}  ({len(hourly)} hours with imagery)")
    print(f"  {len(hourly)} of {possible} hours in the span carry any image"
          f" ({100 * len(hourly) / max(possible, 1):.0f}%)")
    print(f"  median {int(hourly['images'].median())} images/hour,"
          f" max {int(hourly['images'].max())}")
    return hourly


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

def read_records(path):
    """Every JSON record in a file.

    The rip product ships .jsonl -- one JSON object per line -- so a plain
    json.load() on the file would fail the moment a file carries more than one
    frame. A malformed line is reported and skipped rather than losing the
    whole file.
    """
    with open(path) as fh:
        text = fh.read()
    if not text.strip():
        return []
    if os.path.splitext(path)[1].lower() == ".jsonl":
        records = []
        for number, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  {os.path.basename(path)} line {number}: {exc}")
        return records
    payload = json.loads(text)
    return payload if isinstance(payload, list) else [payload]


def _scores(entries):
    """Flat list of confidence values, and the class names they belong to.

    classification_scores is a list of single-key dicts, [{'rip_current': 0.7}],
    so the class name is data rather than schema. Read generically: a second
    class appearing later must not need a code change.
    """
    values, names = [], []
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        for name, value in entry.items():
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
            names.append(name)
    return values, names


def _boxes(entries):
    """(areas, centroids) in pixels for each [{x,y},{x,y}] corner pair."""
    areas, centroids = [], []
    for box in entries or []:
        points = [(point.get("x"), point.get("y")) for point in box or []
                  if isinstance(point, dict)]
        points = [(x, y) for x, y in points if x is not None and y is not None]
        if len(points) < 2:
            continue
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        areas.append(abs(max(xs) - min(xs)) * abs(max(ys) - min(ys)))
        centroids.append(((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2))
    return areas, centroids


def flatten_record(record, element_time=None, source_file=None):
    """One frame's detection result as a flat row.

    `time` inside the payload is the model's own stamp and is what the row is
    keyed on; the element's time is kept beside it because they differ by
    seconds and only one of them is the actual frame capture.
    """
    result = record.get("classification_result") or {}
    values, names = _scores(result.get("classification_scores"))
    areas, centroids = _boxes(result.get("classification_bboxes"))
    largest = areas.index(max(areas)) if areas else None

    stamp = record.get("time")
    row = {
        "timestamp": pd.to_datetime(stamp, utc=True, errors="coerce") if stamp
                     else pd.NaT,
        "element_time": element_time,
        "detected": bool(result.get("detected")),
        "detection_count": result.get("detection_count"),
        "score_max": max(values) if values else None,
        "score_mean": sum(values) / len(values) if values else None,
        "score_classes": ",".join(sorted(set(names))) or None,
        "bbox_count": len(areas),
        "bbox_area_max": max(areas) if areas else None,
        "bbox_x": centroids[largest][0] if largest is not None else None,
        "bbox_y": centroids[largest][1] if largest is not None else None,
        "model_name": result.get("classification_model_name"),
        "model_version": result.get("classification_model_version"),
        "original_image": record.get("original_image_reference"),
        "annotated_image_url": record.get("annotated_image_url"),
        "source_file": source_file,
    }
    if pd.isna(row["timestamp"]) and element_time is not None:
        row["timestamp"] = element_time
    return row


def build_table(rows, out_csv):
    """Turn the downloaded payloads into one frame-level CSV, plus an index.

    Anything non-tabular gets the index only -- filename, timestamp, url --
    rather than invented columns.
    """
    index = pd.DataFrame([{"timestamp": r["timestamp"], "filename": r["filename"],
                           "path": r["path"], "url": r["url"]} for r in rows])
    index_path = out_csv.replace(".csv", "_index.csv")
    index.to_csv(index_path, index=False)
    print(f"  wrote {index_path}  ({len(index)} elements)")

    parsed = []
    for row in rows:
        ext = os.path.splitext(row["path"])[1].lower()
        try:
            if ext in (".json", ".jsonl"):
                for record in read_records(row["path"]):
                    parsed.append(flatten_record(record, row["timestamp"],
                                                 row["filename"]))
            elif ext == ".csv":
                frame = pd.read_csv(row["path"])
                frame["timestamp"] = row["timestamp"]
                frame["source_file"] = row["filename"]
                parsed.append(frame)
            else:
                continue
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(f"  skipped {row['filename']}: {exc}")

    if not parsed:
        print("  payloads are not tabular — the index is the whole table.")
        return None
    if isinstance(parsed[0], pd.DataFrame):
        table = pd.concat(parsed, ignore_index=True)
    else:
        table = pd.DataFrame(parsed)
    table = table.sort_values("timestamp").reset_index(drop=True)
    table.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}  ({len(table)} frames, {len(table.columns)} columns)")
    if "detected" in table.columns:
        hits = int(table["detected"].sum())
        print(f"  {hits:,} frames with a detection of {len(table):,}"
              f" ({100 * hits / max(len(table), 1):.1f}%)")
    return table


def hourly_summary(table, out_csv):
    """Collapse frames to hourly rows, to join against the observation CSVs.

    Everything else in this pipeline is hourly, so this is the form the rip
    signal has to be in to sit beside tide, wind and rainfall. Both the rate
    and the raw counts are kept: an hour with one detection in two frames is
    not the same as one with fifty in a hundred, and a rate alone hides that.
    """
    if table is None or table.empty or "detected" not in table.columns:
        return None
    frame = table.dropna(subset=["timestamp"]).copy()
    if frame.empty:
        return None
    frame["hour"] = pd.to_datetime(frame["timestamp"], utc=True).dt.floor("h")
    detected = frame[frame["detected"]]

    hourly = frame.groupby("hour").agg(
        frames=("detected", "size"),
        frames_with_detection=("detected", "sum"),
        detections=("detection_count", "sum"),
    )
    hourly["detection_rate"] = (hourly["frames_with_detection"]
                                / hourly["frames"]).round(4)
    scores = detected.groupby("hour").agg(
        score_max=("score_max", "max"),
        score_mean=("score_max", "mean"),
        bbox_area_max=("bbox_area_max", "max"),
    )
    # Left join: an hour with frames but no detection is a real observed zero,
    # not a gap, and must survive rather than being dropped.
    hourly = hourly.join(scores, how="left").reset_index()
    hourly.to_csv(out_csv, index=False)
    print(f"  wrote {out_csv}  ({len(hourly)} hours)")
    return hourly


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
    ap.add_argument("--inventory", action="store_true",
                    help="report when this product has data, and download nothing")
    ap.add_argument("--coverage", action="store_true",
                    help="enumerate the stills product to find the hours the "
                         "camera was looking; downloads nothing")
    ap.add_argument("--stills-product", default=None,
                    help=f"product slug to use as denominator (default {STILLS_SLUG})")
    ap.add_argument("--via-pywebcoos", action="store_true",
                    help="use the library's download() instead of /elements/")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help=f"parallel downloads (default {WORKERS})")
    args = ap.parse_args()

    assets = load_assets(args.refresh)
    if args.list:
        list_cameras(assets)
        return
    if not (args.probe or args.pull or args.via_pywebcoos or args.inventory
            or args.coverage):
        ap.error("choose one of --list, --inventory, --coverage, --probe, --pull")

    asset = find_camera(assets, args.camera)
    camera_label = dig(asset, "data", "common", "label")
    if args.coverage:
        service_slug, product_label = find_stills_service(asset, args.stills_product)
    else:
        service_slug, product_label = find_rip_service(asset)
    save_dir = os.path.join(OUT_DIR, slugify(camera_label))

    # The catalogue's element count says how much data exists, never when.
    # Asking the inventory first is what stops a probe from silently landing
    # on an empty window and reading as "the product is broken".
    print("\ninventory")
    first, last = report_inventory(service_slug)
    if args.inventory:
        return

    end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
           if args.end else None)
    start = (datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
             if args.start else None)

    if start is None or end is None:
        # Default to where the data actually is, not to now.
        if last is not None:
            end = end or last + timedelta(minutes=1)
            start = start or (end - timedelta(hours=PROBE_HOURS) if args.probe
                              else max(first, end - timedelta(days=365)))
        else:
            end = end or datetime.now(timezone.utc)
            start = start or (end - timedelta(hours=PROBE_HOURS) if args.probe
                              else end - timedelta(days=365))
    print(f"\nrange {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC")
    if first is not None and (end < first or start > last):
        print("  that range lies outside the inventory above — expect nothing back.")

    if args.via_pywebcoos:
        via_pywebcoos(camera_label, product_label, start, end,
                      args.interval or 30, save_dir)
        return

    if args.coverage:
        print("  enumerating stills (no downloads); this is the slow part")
    rows = fetch_elements(service_slug, start, end, args.interval)
    if not rows:
        print("\nNo elements in that range.")
        if first is not None:
            print(f"The inventory says data runs {first:%Y-%m-%d} to {last:%Y-%m-%d};")
            print("pick --start and --end inside that.")
        else:
            print("The inventory gave no range either, so this product may be")
            print("catalogued but not actually served on this token.")
        return
    print(f"  first {rows[0]['timestamp']:%Y-%m-%d %H:%M}"
          f"  last {rows[-1]['timestamp']:%Y-%m-%d %H:%M} UTC")

    if args.coverage:
        build_coverage(rows, os.path.join(
            OUT_DIR, f"coverage_{slugify(camera_label)}_hourly.csv"))
        return

    got = download(rows, save_dir, args.workers)
    if args.probe:
        probe(got)
    stem = os.path.join(OUT_DIR, f"rip_{slugify(camera_label)}")
    table = build_table(got, stem + ".csv")
    hourly_summary(table, stem + "_hourly.csv")


if __name__ == "__main__":
    main()
