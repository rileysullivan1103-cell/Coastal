"""Probe the Water Quality Portal and report its ACTUAL column names.

The WQP field names used in find_candidate_sites.py (LatitudeMeasure,
LongitudeMeasure, MonitoringLocationIdentifier) follow WQP's documented
convention but were never confirmed against a live response. This pulls a small
bbox so it returns in seconds, then prints the real columns.

    python verify_wqp_fields.py
"""

import env  # noqa: F401  -- loads .env into os.environ

import sys
from io import StringIO

import pandas as pd
import requests

URL = "https://www.waterqualitydata.us/data/Station/search"

# Small box around Charleston Harbor, SC — enough to get real rows quickly.
# bBox order is min_lon,min_lat,max_lon,max_lat (lon first).
SMALL_BBOX = "-80.2,32.6,-79.6,32.9"

EXPECTED = ["MonitoringLocationIdentifier", "LatitudeMeasure", "LongitudeMeasure"]


def main():
    params = {
        "bBox": SMALL_BBOX,
        "characteristicName": "Escherichia coli;Enterococcus;Fecal Coliform",
        "mimeType": "csv",
        "zip": "no",
    }
    resp = requests.get(URL, params=params, timeout=300)
    print(f"GET {resp.url}\nHTTP {resp.status_code}, {len(resp.content)} bytes, "
          f"content-type {resp.headers.get('content-type')}\n")
    resp.raise_for_status()

    if resp.content[:2] == b"PK":
        sys.exit("Got a zip archive back — the 'zip=no' parameter is not being honored.")

    df = pd.read_csv(StringIO(resp.text))
    print(f"{len(df)} rows, {len(df.columns)} columns\n")
    print("=== ACTUAL COLUMNS ===")
    for col in df.columns:
        print(f"  {col}")

    print("\n=== COLUMNS THE PIPELINE DEPENDS ON ===")
    missing = []
    for col in EXPECTED:
        present = col in df.columns
        print(f"  {col:<35} {'PRESENT' if present else 'MISSING'}")
        if not present:
            missing.append(col)

    if missing:
        print("\nCoordinate/id-like columns actually available:")
        for col in df.columns:
            if any(w in col.lower() for w in ("lat", "lon", "identifier", "name")):
                print(f"  {col}")
        sys.exit(f"\n{len(missing)} expected column(s) missing — update "
                 f"find_candidate_sites.py before running the pipeline.")

    print("\nAll expected columns present. Sample:")
    print(df[EXPECTED].head().to_string(index=False))


if __name__ == "__main__":
    main()
