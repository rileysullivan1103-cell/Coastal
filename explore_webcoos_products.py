"""Enumerate every feed, product and service on the WebCOOS cameras.

Before writing a rip-detection pull we need to know whether such a product
exists, on which cameras, and under which feed. This matters because
pywebcoos.API hardcodes feed_name = 'raw-video-data' in get_products(),
get_inventory() and download() -- see pywebcoos/API.py. Any product published
under a different feed is invisible to that library, and would need a direct
API call instead.

Reads the local webcoos_assets_raw.json written by verify_webcoos_fields.py, so
it costs nothing to re-run. Pass --fetch to refresh it from the API first.

    python explore_webcoos_products.py
    python explore_webcoos_products.py --fetch
    python explore_webcoos_products.py --all-states
"""

import env  # noqa: F401  -- loads .env into os.environ

import json
import os
import sys
from collections import Counter, defaultdict

import requests

RAW_DUMP = "webcoos_assets_raw.json"
API_BASE = "https://app.webcoos.org/webcoos/api/v1"
PYWEBCOOS_FEED = "raw-video-data"

# Substrings that would indicate a derived analytics product rather than raw imagery.
INTERESTING = ("rip", "detect", "current", "classif", "segment", "ml", "model",
               "analy", "hazard", "surf")


def load_assets(fetch=False):
    if fetch or not os.path.exists(RAW_DUMP):
        token = os.environ.get("WEBCOOS_TOKEN")
        if not token:
            sys.exit("WEBCOOS_TOKEN is not set and no local dump exists.")
        resp = requests.get(f"{API_BASE}/assets/", timeout=60, headers={
            "Authorization": f"Token {token}", "Accept": "application/json"})
        resp.raise_for_status()
        with open(RAW_DUMP, "w") as fh:
            json.dump(resp.json(), fh, indent=2)
        print(f"refreshed {RAW_DUMP}")
    with open(RAW_DUMP) as fh:
        return json.load(fh).get("results", [])


def dig(node, *path):
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def describe(asset):
    """(label, state, [(feed_slug, product_slug, service_type, elements)])."""
    label = dig(asset, "data", "common", "label") or asset.get("slug") or "?"
    state = dig(asset, "data", "properties", "state_or_territory")
    rows = []
    for feed in asset.get("feeds") or []:
        feed_slug = dig(feed, "data", "common", "slug") or "?"
        for product in feed.get("products") or []:
            product_slug = dig(product, "data", "common", "slug") or "?"
            services = product.get("services") or []
            if not services:
                rows.append((feed_slug, product_slug, None, None))
            for service in services:
                rows.append((
                    feed_slug, product_slug,
                    dig(service, "data", "type"),
                    dig(service, "elements", "count"),
                ))
    return label, state, rows


def main():
    fetch = "--fetch" in sys.argv
    ca_only = "--all-states" not in sys.argv
    assets = load_assets(fetch)
    print(f"{len(assets)} assets in {RAW_DUMP}\n")

    feed_counts, product_counts = Counter(), Counter()
    products_by_feed = defaultdict(set)
    shown = 0

    for asset in assets:
        label, state, rows = describe(asset)
        if ca_only and state != "California":
            for feed_slug, product_slug, _, _ in rows:
                feed_counts[feed_slug] += 1
                product_counts[product_slug] += 1
                products_by_feed[feed_slug].add(product_slug)
            continue

        shown += 1
        print(f"{label}  [{state}]")
        if not rows:
            print("    (no feeds)")
        for feed_slug, product_slug, service_type, count in rows:
            feed_counts[feed_slug] += 1
            product_counts[product_slug] += 1
            products_by_feed[feed_slug].add(product_slug)
            reach = "" if feed_slug == PYWEBCOOS_FEED else "   <- not reachable via pywebcoos"
            n = f"{count:,} elements" if count else "no element count"
            print(f"    {feed_slug} / {product_slug} ({service_type}, {n}){reach}")
        print()

    print(f"=== {shown} cameras shown "
          f"({'California only' if ca_only else 'all states'}) ===\n")

    print("=== FEEDS ACROSS ALL CAMERAS ===")
    for feed_slug, n in feed_counts.most_common():
        mark = "" if feed_slug == PYWEBCOOS_FEED else "  (pywebcoos cannot reach this feed)"
        print(f"  {feed_slug:<28} {n:>4} product-services{mark}")

    print("\n=== PRODUCTS ACROSS ALL CAMERAS ===")
    for product_slug, n in product_counts.most_common():
        print(f"  {product_slug:<28} {n:>4}")

    print("\n=== ANYTHING LOOKING LIKE A DERIVED / DETECTION PRODUCT ===")
    hits = [p for p in product_counts if any(w in p.lower() for w in INTERESTING)]
    hits += [f for f in feed_counts if any(w in f.lower() for w in INTERESTING)]
    if hits:
        for name in sorted(set(hits)):
            print(f"  {name}")
    else:
        print("  None. Every product here is raw imagery or video, so the")
        print("  rip-detection output is not exposed on the /assets/ endpoint.")
        print("  It is either a separate endpoint, or not public on this token.")


if __name__ == "__main__":
    main()
