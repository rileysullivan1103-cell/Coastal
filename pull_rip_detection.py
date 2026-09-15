"""Pull WebCOOS's own rip-detection product, on any camera that carries one.

explore_webcoos_products.py established that rip-detection-results exists on
eight cameras nationally, under the raw-video-data feed. Walton Lighthouse,
Santa Cruz (35,158 elements) is the largest and was the first pulled. This
script downloads any of them, one at a time or all at once.

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

    python pull_rip_detection.py --list                 every rip camera
    python pull_rip_detection.py --list --inventory     ...and when each has data
    python pull_rip_detection.py --all-rip --inventory  same, one request each
    python pull_rip_detection.py --camera Corolla --probe
    python pull_rip_detection.py --all-rip --pull --match-observations

Start with --list --inventory. The catalogued element count says how much data
exists and never how much of the record it covers: Corolla's 10,239 elements
sit in 46 populated bins of 818, so it looks a third of Walton's size and is
really a few dense weeks. Decide what to pull from the populated share, not
from the count.

--all-rip runs any action once per camera and isolates each one, so a camera
whose stills product is missing is reported and skipped rather than ending the
sweep. Writes files under data/rip_detection/<camera-slug>/ and, when the
payload is tabular, a combined CSV per camera alongside them.
"""

import env  # noqa: F401  -- loads .env into os.environ

import argparse
import concurrent.futures
import glob
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
OBS_DIR = "data"
# Sources whose time span defines the window the analysis can actually use.
# A rip pull outside it produces rows that join to nothing.
OBS_PATTERNS = ("gridded_*.csv", "buoy_*.csv", "tide_*.csv")
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
# Elements per page. The server may cap this; asking for more is harmless and
# a listing of 130,000 stills at 100 a page is 1,300 round trips.
ELEMENT_PAGE_SIZE = 1000
# A single read timeout should not discard hours of pagination.
MAX_RETRIES = 5


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
    """Every asset, paginated — not just the first page.

    This used to GET /assets/ once and keep whatever came back. WebCOOS pages
    that endpoint, so any camera past page one was invisible: asking this
    script for a camera it could not see produced "No camera matching ...",
    which reads as "that camera does not carry rip detection" and is not the
    same thing. scan_cameras.fetch_assets already follows the pagination and
    caches the complete list, so use it rather than keep a second, shorter
    copy of the catalogue.
    """
    try:
        from scan_cameras import fetch_assets
    except ImportError as exc:  # pragma: no cover - dependency problem, not logic
        print(f"  cannot import scan_cameras ({exc}); falling back to {RAW_DUMP}, "
              "which holds only the first page of the catalogue")
        if not os.path.exists(RAW_DUMP):
            sys.exit(f"{RAW_DUMP} does not exist either — nothing to read.")
        with open(RAW_DUMP) as fh:
            return json.load(fh).get("results", [])
    return fetch_assets(refresh=refresh)


def rip_cameras(assets):
    """Every camera carrying a rip-detection product, most elements first.

    A camera with more than one rip product keeps the largest: the count is
    what decides whether a pull is worth making, and the runner-up would
    understate it.
    """
    found = []
    for asset in assets:
        label = dig(asset, "data", "common", "label")
        best = None
        for _, _, _, product_slug, service_slug, count in camera_products(asset):
            if "rip" not in (product_slug or "").lower():
                continue
            entry = {
                "label": label,
                "slug": slugify(label),
                "state": dig(asset, "data", "properties", "state_or_territory"),
                "product_slug": product_slug,
                "service_slug": service_slug,
                "elements": int(count or 0),
                "asset": asset,
            }
            if best is None or entry["elements"] > best["elements"]:
                best = entry
        if best is not None:
            found.append(best)
    return sorted(found, key=lambda c: -c["elements"])


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
    """Resolve a camera by label, by slug, or by a unique substring.

    The slug is accepted because substrings are not always unique and a label
    is awkward to type: Corolla has TWO rip cameras, "Beachfront from Hampton
    Inn, Corolla, NC" and "Beachfront from Sailfish Street Beach Access,
    Corolla, NC", so --camera Corolla cannot mean anything on its own. The slug
    is also what analyze_drivers.py --site takes and what the output
    directories are named after, so one identifier now works across the whole
    pipeline.
    """
    labels = [dig(a, "data", "common", "label") or "" for a in assets]
    for asset, label in zip(assets, labels):
        if label == name:
            return asset
    wanted = slugify(name)
    for asset, label in zip(assets, labels):
        if slugify(label) == wanted:
            print(f"camera {name!r} -> {label!r}")
            return asset
    hits = [(a, l) for a, l in zip(assets, labels) if name.lower() in l.lower()]
    if len(hits) == 1:
        print(f"camera {name!r} -> {hits[0][1]!r}")
        return hits[0][0]
    if not hits:
        sys.exit(f"No camera matching {name!r}. Run --list to see the names.")
    # Print the slugs, not just the labels: an ambiguous match is only useful
    # if the message hands you something you can paste straight back in.
    sys.exit(f"{name!r} matches {len(hits)} cameras. Pass one of these slugs "
             "to --camera, or use --all-rip to do every one:\n  " +
             "\n  ".join(f"{slugify(l):<56} {l}" for _, l in hits))


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


def _csv_span(path):
    """(first, last) timestamp in a CSV, from whichever column holds times."""
    try:
        head = pd.read_csv(path, nrows=1)
    except (ValueError, OSError):
        return None
    for column in list(head.columns):
        if column.lower() in ("time", "hour", "date", "unnamed: 0", ""):
            try:
                stamps = pd.to_datetime(pd.read_csv(path, usecols=[column])[column],
                                        utc=True, errors="coerce").dropna()
            except (ValueError, OSError):
                continue
            if not stamps.empty:
                return stamps.min(), stamps.max()
    return None


def obs_slug(text):
    """The filename stem pull_site_observations.py and pull_gridded_weather.py
    write. Not slugify(): those use underscores and a 48-character cut, this
    module uses dashes, and a rip camera's own weather file has to be found by
    the name its writer chose."""
    return "".join(c if c.isalnum() else "_" for c in str(text))[:48]


def observation_window(camera_label=None):
    """The window THIS CAMERA's observation CSVs cover, as (start, end).

    The overlap is the intersection, not the union: a rip hour is only usable
    where the conditions it would be explained by also exist. A first pull
    took three months of 2025 while the observations ran Aug 2025 to Aug 2026,
    and the join landed on 39 hours of gridded weather and 1 of buoy.

    It is scoped to one camera because the intersection of EVERY file in
    data/ is the wrong quantity and becomes empty as soon as the project
    covers more than one place. Cala Millor's weather ends 2024-09-28 and
    Corolla's begins 2025-09-08, so once both were on disk the global
    intersection was empty and --match-observations skipped every camera,
    Walton included -- a site whose own observations were fine.

    Tide and buoy files are named after a STATION, not a camera, and are
    shared between sites; which one belongs to this camera is a distance
    question that analyze_drivers.py answers at join time. They are reported
    but do not constrain the window.
    """
    if camera_label is None:
        patterns = OBS_PATTERNS
    else:
        stem = obs_slug(camera_label)
        patterns = (f"gridded_{stem}*.csv", f"marine_{stem}*.csv")

    spans = []
    for pattern in patterns:
        for path in sorted(glob.glob(os.path.join(OBS_DIR, pattern))):
            span = _csv_span(path)
            if span:
                spans.append((os.path.basename(path), span[0], span[1]))
    if not spans:
        if camera_label is not None:
            print(f"  no observation CSVs for {camera_label!r} "
                  f"(looked for gridded_{obs_slug(camera_label)}*.csv)")
        return None
    print("  observation sources for this camera:")
    for name, first, last in spans:
        print(f"    {name:<56} {first:%Y-%m-%d} to {last:%Y-%m-%d}")
    start = max(first for _, first, _ in spans)
    end = min(last for _, _, last in spans)
    if start >= end:
        print("  those sources do not all overlap; cannot derive a window")
        return None
    print(f"  common window: {start:%Y-%m-%d} to {end:%Y-%m-%d}")
    return start, end


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


def inventory_stats(service_slug):
    """What the inventory says, as numbers rather than printed lines.

    `populated` is the one that decides whether a camera is worth pulling.
    Corolla catalogues 10,239 elements and looks comparable to Walton's 35,158
    until you see they sit in 46 populated bins of 818: the element count says
    how much data exists, never how much of the record it covers.
    """
    frame = fetch_inventory(service_slug)
    first, last = inventory_range(frame)
    stats = {"first": first, "last": last, "elements": None,
             "populated": None, "bins": None if frame is None else len(frame),
             "stale_days": None}
    if frame is not None and "Count" in frame.columns:
        counts = pd.to_numeric(frame["Count"], errors="coerce").fillna(0)
        stats["elements"] = int(counts.sum())
        stats["populated"] = int((counts > 0).sum())
    if last is not None:
        stats["stale_days"] = int((pd.Timestamp.now(tz="UTC") - last).days)
    return stats


def report_inventory(service_slug):
    """Print what the inventory says, and return its (first, last)."""
    stats = inventory_stats(service_slug)
    first, last = stats["first"], stats["last"]
    if first is None:
        print("  inventory gave no usable date range")
        return None, None
    print(f"  data runs {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} UTC")
    if stats["elements"] is not None:
        print(f"  {stats['elements']:,} elements across {stats['populated']} "
              f"populated bins of {stats['bins']}")
    if stats["stale_days"] and stats["stale_days"] > 1:
        print(f"  last data is {stats['stale_days']} days old — "
              "this product is not live")
    return first, last


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------

def get_with_retry(url, **kwargs):
    """GET with exponential backoff on the failures that are worth retrying.

    A listing run makes over a thousand sequential requests, so a transient
    read timeout is close to certain rather than unlucky. Without this, one
    such timeout discards every page already fetched.
    """
    delay = 2
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT, **kwargs)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"    {type(exc).__name__} (attempt {attempt}/{MAX_RETRIES});"
                  f" retrying in {delay}s")
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
            print(f"    HTTP {resp.status_code} (attempt {attempt}/{MAX_RETRIES});"
                  f" retrying in {delay}s")
            time.sleep(delay)
            delay *= 2
            continue
        return resp
    raise RuntimeError("unreachable")


def fetch_elements(service_slug, start, end, interval_minutes=None, quiet=False):
    """Every element for a service in [start, end), oldest first.

    start/end are timezone-aware UTC datetimes. interval_minutes, if given,
    keeps only elements whose minute-of-hour is a multiple of it -- the same
    thinning pywebcoos applies, done here so a year of frames is tractable.
    """
    params = {
        "service": service_slug,
        "starting_after": start.isoformat().replace("+00:00", "Z"),
        "starting_before": end.isoformat().replace("+00:00", "Z"),
        "limit": ELEMENT_PAGE_SIZE,
    }
    url = f"{API_BASE}/elements/"
    out, page = [], 1
    while url:
        resp = get_with_retry(url, headers=headers(), params=params)
        if resp.status_code != 200:
            sys.exit(f"/elements/ page {page} returned {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        results = payload.get("results") or []
        out.extend(results)
        if not quiet or page % 25 == 0:
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
    if not quiet:
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

def _coverage_paths(out_csv):
    return out_csv, out_csv.replace(".csv", "_progress.csv")


def _duration(seconds):
    """h/m/s, for a progress line that has to be read at a glance."""
    seconds = int(max(seconds, 0))
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def load_progress(progress_csv):
    """Dates already enumerated, so a resumed run does not redo them."""
    if not os.path.exists(progress_csv):
        return set()
    frame = pd.read_csv(progress_csv)
    return set(frame["date"].astype(str))


def build_coverage(service_slug, start, end, out_csv):
    """Hourly count of images captured, built one day at a time and resumable.

    Nothing is downloaded: element listing returns a timestamp per element,
    which is all a denominator needs, so hundreds of thousands of JPEGs never
    move. But the listing itself is over a thousand sequential requests, and a
    single read timeout used to discard the lot. Work is committed per day, so
    a failure costs one day and re-running continues where it stopped.
    """
    out_csv, progress_csv = _coverage_paths(out_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    done = load_progress(progress_csv)
    days = pd.date_range(pd.Timestamp(start).floor("D"),
                         pd.Timestamp(end).ceil("D") - pd.Timedelta(days=1),
                         freq="D", tz="UTC")
    todo = [d for d in days if d.strftime("%Y-%m-%d") not in done]
    if done:
        print(f"  resuming: {len(done)} days already enumerated,"
              f" {len(todo)} to go")
    else:
        print(f"  {len(todo)} days to enumerate")

    # Several hundred sequential days of paginated listing is indistinguishable
    # from a hang unless the line says how far through it is and how long the
    # rest will take. The rate is measured over this run rather than assumed:
    # days differ enormously in how many pages they hold.
    started = time.monotonic()
    for index, day in enumerate(todo, 1):
        label = day.strftime("%Y-%m-%d")
        rows = fetch_elements(service_slug, day, day + pd.Timedelta(days=1),
                              quiet=True)
        if rows:
            frame = pd.DataFrame({"timestamp": [r["timestamp"] for r in rows]})
            frame["hour"] = pd.to_datetime(frame["timestamp"], utc=True).dt.floor("h")
            hourly = frame.groupby("hour").size().rename("images").reset_index()
            header = not os.path.exists(out_csv)
            hourly.to_csv(out_csv, mode="a", header=header, index=False)
        # Recorded even when empty: "the camera produced nothing that day" is
        # a result, and without it every resume would retry the empty days.
        pd.DataFrame([{"date": label, "images": len(rows)}]).to_csv(
            progress_csv, mode="a", header=not os.path.exists(progress_csv),
            index=False)
        elapsed = time.monotonic() - started
        left = (elapsed / index) * (len(todo) - index)
        print(f"  {label}: {len(rows)} images  ({index}/{len(todo)}, "
              f"{_duration(elapsed)} in, ~{_duration(left)} left)")

    if not os.path.exists(out_csv):
        print("  no imagery in that range at all")
        return None
    hourly = pd.read_csv(out_csv)
    hourly["hour"] = pd.to_datetime(hourly["hour"], utc=True)
    # An interrupted run can leave a day appended twice; collapse on read.
    hourly = hourly.groupby("hour", as_index=False)["images"].max()
    hourly.to_csv(out_csv, index=False)
    span = hourly["hour"].max() - hourly["hour"].min()
    possible = int(span.total_seconds() // 3600) + 1
    print(f"\n  wrote {out_csv}  ({len(hourly)} hours with imagery)")
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

def list_rip_cameras(assets, check_inventory=False):
    """The national roster of cameras carrying rip detection.

    The element count alone is misleading, so --list --inventory also asks each
    service when it actually has data. That costs one request per camera and
    turns "8 cameras carry the product" into "N of them are worth pulling".
    """
    cameras = rip_cameras(assets)
    if not cameras:
        print("No camera in the catalogue carries a rip-detection product.")
        return cameras
    print(f"{len(cameras)} of {len(assets)} cameras carry rip detection\n")
    header = f"{'camera':<44} {'state':<16} {'elements':>9}"
    if check_inventory:
        header += f"  {'covers':<25} {'populated':>10} {'stale':>7}"
    print(header)
    print("-" * len(header))
    for cam in cameras:
        line = (f"{(cam['label'] or '?')[:44]:<44} "
                f"{(cam['state'] or '?')[:16]:<16} {cam['elements']:>9,}")
        if check_inventory:
            stats = inventory_stats(cam["service_slug"])
            cam["inventory"] = stats
            if stats["first"] is None:
                line += "  " + f"{'no inventory':<25} {'-':>10} {'-':>7}"
            else:
                span = f"{stats['first']:%Y-%m-%d} to {stats['last']:%Y-%m-%d}"
                share = ("-" if not stats["bins"]
                         else f"{stats['populated']}/{stats['bins']}")
                stale = ("-" if stats["stale_days"] is None
                         else f"{stats['stale_days']}d")
                line += f"  {span:<25} {share:>10} {stale:>7}"
        print(line)
    if check_inventory:
        print("\n'populated' is bins holding data out of bins in the record. A "
              "camera\nwith a high element count in few bins covers a short "
              "stretch densely,\nwhich joins to far fewer observation hours "
              "than the raw count suggests.")
    return cameras


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


def run_for_camera(asset, args):
    """Do the requested action for one already-resolved camera.

    Split out of main() so that --all-rip can call it once per camera. Returns
    a one-line status for the run summary; raising is left to the caller to
    catch, because one camera with a missing stills product must not end a
    national sweep.
    """
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
        if first is None:
            return "no inventory"
        return f"{first:%Y-%m-%d} to {last:%Y-%m-%d}"

    end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
           if args.end else None)
    start = (datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
             if args.start else None)

    if args.match_observations:
        print("\nmatching the observation window")
        window = observation_window(camera_label)
        if window is None:
            sys.exit(f"No observation CSVs for this camera — run "
                     f"pull_site_observations.py --camera {slugify(camera_label)!r}")
        obs_start, obs_end = window
        # Clip to what the product actually holds, so the request is not
        # partly outside the inventory.
        start = max(obs_start, first) if first is not None else obs_start
        end = min(obs_end, last) if last is not None else obs_end
        start = start.to_pydatetime() if hasattr(start, "to_pydatetime") else start
        end = end.to_pydatetime() if hasattr(end, "to_pydatetime") else end
        if start >= end:
            sys.exit("The product's coverage and the observation window do not overlap.")

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
        return "pywebcoos path"

    if args.coverage:
        print("  enumerating stills day by day (no downloads, resumable)")
        build_coverage(service_slug, start, end, os.path.join(
            OUT_DIR, f"coverage_{slugify(camera_label)}_hourly.csv"))
        return "coverage written"

    rows = fetch_elements(service_slug, start, end, args.interval)
    if not rows:
        print("\nNo elements in that range.")
        if first is not None:
            print(f"The inventory says data runs {first:%Y-%m-%d} to {last:%Y-%m-%d};")
            print("pick --start and --end inside that.")
        else:
            print("The inventory gave no range either, so this product may be")
            print("catalogued but not actually served on this token.")
        return "no elements in range"
    print(f"  first {rows[0]['timestamp']:%Y-%m-%d %H:%M}"
          f"  last {rows[-1]['timestamp']:%Y-%m-%d %H:%M} UTC")

    got = download(rows, save_dir, args.workers)
    if args.probe:
        probe(got)
    stem = os.path.join(OUT_DIR, f"rip_{slugify(camera_label)}")
    table = build_table(got, stem + ".csv")
    hourly_summary(table, stem + "_hourly.csv")
    return (f"{len(table):,} rows" if table is not None and len(table)
            else "downloaded, no table")


def sweep(assets, args):
    """Run the chosen action across every camera carrying rip detection.

    Each camera is isolated. A missing stills product, a service the token
    cannot reach, or a network failure ends that camera and nothing else --
    including SystemExit, which the single-camera helpers raise freely and
    which would otherwise abandon the sweep partway with no summary of what
    had already succeeded.
    """
    cameras = rip_cameras(assets)
    if not cameras:
        sys.exit("No camera in the catalogue carries a rip-detection product.")
    if args.state:
        cameras = [c for c in cameras if c["state"] == args.state]
        if not cameras:
            sys.exit(f"No rip camera in {args.state!r}.")

    print(f"{len(cameras)} camera{'' if len(cameras) == 1 else 's'} "
          "carrying rip detection\n")
    outcomes = []
    for index, cam in enumerate(cameras, 1):
        banner = f"[{index}/{len(cameras)}] {cam['label']}  ({cam['state']})"
        print(f"\n{'=' * 74}\n{banner}\n{'=' * 74}")
        try:
            status = run_for_camera(cam["asset"], args) or "done"
        except SystemExit as exc:
            status = f"skipped: {exc}"
            print(f"  {status}")
        except Exception as exc:
            status = f"{type(exc).__name__}: {exc}"
            print(f"  failed: {status}")
        outcomes.append((cam["label"], cam["state"], cam["elements"], status))

    print(f"\n{'=' * 74}\nSWEEP SUMMARY\n{'=' * 74}")
    width = max(len(label or "") for label, _, _, _ in outcomes)
    for label, state, elements, status in outcomes:
        print(f"{(label or '?'):<{width}}  {(state or '?')[:14]:<14} "
              f"{elements:>9,}  {status}")
    return outcomes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", default=DEFAULT_CAMERA)
    ap.add_argument("--start", help="UTC date, YYYY-MM-DD")
    ap.add_argument("--end", help="UTC date, YYYY-MM-DD (exclusive)")
    ap.add_argument("--interval", type=int, default=None,
                    help="keep only elements on this minute spacing")
    ap.add_argument("--list", action="store_true",
                    help="list every camera carrying rip detection, nationally")
    ap.add_argument("--state", default=None,
                    help="with --list, show every product on every camera in "
                         "this state instead of the rip roster")
    ap.add_argument("--all-rip", action="store_true",
                    help="run the chosen action once per rip-detection camera")
    ap.add_argument("--probe", action="store_true",
                    help=f"download {PROBE_HOURS}h and describe the payload")
    ap.add_argument("--pull", action="store_true", help="download the full range")
    ap.add_argument("--refresh", action="store_true", help="re-fetch the asset catalogue")
    ap.add_argument("--inventory", action="store_true",
                    help="report when this product has data, and download nothing")
    ap.add_argument("--coverage", action="store_true",
                    help="enumerate the stills product to find the hours the "
                         "camera was looking; downloads nothing")
    ap.add_argument("--match-observations", action="store_true",
                    help="use the window the observation CSVs already cover, "
                         "so the pull joins to something")
    ap.add_argument("--stills-product", default=None,
                    help=f"product slug to use as denominator (default {STILLS_SLUG})")
    ap.add_argument("--via-pywebcoos", action="store_true",
                    help="use the library's download() instead of /elements/")
    ap.add_argument("--workers", type=int, default=WORKERS,
                    help=f"parallel downloads (default {WORKERS})")
    args = ap.parse_args()

    assets = load_assets(args.refresh)
    if args.list:
        if args.state:
            list_cameras(assets, args.state)
        else:
            list_rip_cameras(assets, check_inventory=args.inventory)
        return
    if not (args.probe or args.pull or args.via_pywebcoos or args.inventory
            or args.coverage):
        ap.error("choose one of --list, --inventory, --coverage, --probe, --pull")

    if args.all_rip:
        sweep(assets, args)
        return

    asset = find_camera(assets, args.camera)
    run_for_camera(asset, args)


if __name__ == "__main__":
    main()
