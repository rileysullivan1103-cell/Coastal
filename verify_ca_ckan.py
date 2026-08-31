"""Find the California water quality dataset on data.ca.gov and its columns.

find_candidate_sites.py can match California sites against real monitoring
stations, but it needs a CKAN resource id and the resource's coordinate/id
column names. This finds both.

    python verify_ca_ckan.py                     search the catalog
    python verify_ca_ckan.py ID [ID ...]         dump each resource's columns

Then set CA_CKAN_RESOURCE_ID / CA_CKAN_LAT_COL / CA_CKAN_LON_COL /
CA_CKAN_ID_COL in find_candidate_sites.py.
"""

import sys

import requests

BASE = "https://data.ca.gov/api/3/action"
QUERIES = ["beach water quality", "bacteria beach", "enterococcus", "safe to swim"]


def search():
    seen = set()
    for query in QUERIES:
        resp = requests.get(f"{BASE}/package_search",
                            params={"q": query, "rows": 10}, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("success"):
            sys.exit(f"CKAN search failed: {str(payload.get('error'))[:300]}")

        results = payload.get("result", {}).get("results", [])
        print(f"\n=== query: {query!r} — {len(results)} datasets ===")
        for pkg in results:
            if pkg["id"] in seen:
                continue
            seen.add(pkg["id"])
            print(f"\n  {pkg.get('title')}")
            print(f"    name: {pkg.get('name')}")
            for res in pkg.get("resources", []):
                # Only datastore-backed resources support datastore_search.
                flag = "queryable" if res.get("datastore_active") else "NOT queryable"
                print(f"    - [{flag}] {res.get('format'):<6} {res.get('name')}")
                print(f"      resource_id: {res.get('id')}")

    print("\nPick a 'queryable' resource_id above, then re-run:")
    print("  python verify_ca_ckan.py <resource_id>")


def describe(resource_id):
    resp = requests.get(f"{BASE}/datastore_search",
                        params={"resource_id": resource_id, "limit": 5}, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        sys.exit(f"CKAN request failed: {str(payload.get('error'))[:300]}")

    result = payload["result"]
    print(f"total rows: {result.get('total')}\n")
    print("=== COLUMNS ===")
    for field in result.get("fields", []):
        print(f"  {field.get('id'):<45} {field.get('type')}")

    print("\n=== LIKELY COORDINATE / ID COLUMNS ===")
    for field in result.get("fields", []):
        name = field.get("id", "").lower()
        if any(w in name for w in ("lat", "lon", "station", "site", "location", "id")):
            print(f"  {field.get('id')}")

    records = result.get("records", [])
    if records:
        print("\n=== FIRST RECORD ===")
        for key, value in records[0].items():
            print(f"  {key:<45} {str(value)[:60]}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for i, rid in enumerate(sys.argv[1:]):
            if i:
                print("\n" + "=" * 70)
            print(f"### resource {rid}\n")
            try:
                describe(rid)
            except requests.HTTPError as exc:
                print(f"  failed: {exc}")
    else:
        search()
