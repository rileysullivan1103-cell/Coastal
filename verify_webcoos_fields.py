"""
Probe the WebCOOS assets endpoint and report its ACTUAL field names.

Run this before trusting any field name in find_candidate_sites.py. It dumps
the raw JSON so the geometry/coordinate path can be confirmed by eye rather
than guessed.

    export WEBCOOS_TOKEN=...
    python verify_webcoos_fields.py

Endpoint and auth header below are taken from the pywebcoos source
(pywebcoos/API.py::_make_api_request), so they are confirmed, not guessed.
"""

import env  # noqa: F401  -- loads .env into os.environ

import json
import os
import sys

import requests

API_BASE = "https://app.webcoos.org/webcoos/api/v1"
RAW_DUMP = "webcoos_assets_raw.json"


def walk(node, path=""):
    """Yield (dotted_path, type, sample) for every leaf in a nested structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        if node:
            yield from walk(node[0], f"{path}[0]")
        else:
            yield path, "empty list", None
    else:
        sample = str(node)
        yield path, type(node).__name__, sample[:70] + ("..." if len(sample) > 70 else "")


def main():
    token = os.environ.get("WEBCOOS_TOKEN")
    if not token:
        sys.exit("WEBCOOS_TOKEN is not set. See .env.example.")

    headers = {"Authorization": f"Token {token}", "Accept": "application/json"}
    resp = requests.get(f"{API_BASE}/assets/", headers=headers, timeout=60)

    if resp.status_code == 401 or resp.status_code == 403:
        sys.exit(f"Auth failed ({resp.status_code}). Token rejected by WebCOOS.\n{resp.text[:400]}")
    resp.raise_for_status()
    payload = resp.json()

    with open(RAW_DUMP, "w") as fh:
        json.dump(payload, fh, indent=2)

    results = payload.get("results", [])
    print(f"top-level keys : {list(payload.keys())}")
    print(f"count          : {payload.get('count')}")
    print(f"next page      : {payload.get('next')}")
    print(f"results in page: {len(results)}")
    print(f"raw dump       : {RAW_DUMP}\n")

    if not results:
        sys.exit("No results returned — nothing to inspect.")

    print("=== FULL FIELD MAP OF results[0] ===")
    for path, kind, sample in walk(results[0]):
        print(f"  {path:<55} {kind:<10} {sample}")

    # The whole point of the probe: where do the coordinates actually live?
    print("\n=== PATHS CONTAINING COORDINATE-LIKE KEYS ===")
    hits = [
        (p, k, s)
        for p, k, s in walk(results[0])
        if any(w in p.lower() for w in ("lat", "lon", "geo", "coord", "point", "bbox"))
    ]
    if hits:
        for path, kind, sample in hits:
            print(f"  {path:<55} {kind:<10} {sample}")
    else:
        print("  NONE FOUND in results[0]. Check the raw dump — coordinates may")
        print("  only appear on some assets, or require a different endpoint.")

    print("\n=== LABELS (what pywebcoos.get_cameras() returns) ===")
    for asset in results[:10]:
        try:
            print("  ", asset["data"]["common"]["label"])
        except (KeyError, TypeError):
            print("   <no data.common.label on this asset>")


if __name__ == "__main__":
    main()
