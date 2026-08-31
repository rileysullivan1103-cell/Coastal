"""Validate both API tokens with one cheap request each.

Run this first on any machine before the full pipeline — it takes seconds and
tells you whether a token is actually accepted, rather than failing 20 minutes
into a bulk download.

    export $(grep -v '^#' .env | xargs)
    python check_tokens.py
"""

import os
import sys

import requests

WEBCOOS_URL = "https://app.webcoos.org/webcoos/api/v1/assets/"
CDO_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2/datasets"


def check_webcoos(token):
    if not token:
        return False, "WEBCOOS_TOKEN not set"
    try:
        r = requests.get(
            WEBCOOS_URL,
            headers={"Authorization": f"Token {token}", "Accept": "application/json"},
            params={"limit": 1},
            timeout=30,
        )
    except requests.RequestException as exc:
        return False, f"could not reach app.webcoos.org: {exc}"

    if r.status_code in (401, 403):
        return False, f"token rejected ({r.status_code}): {r.text[:200]}"
    if r.status_code != 200:
        return False, f"unexpected {r.status_code}: {r.text[:200]}"
    try:
        payload = r.json()
    except ValueError:
        return False, "200 OK but response was not JSON"
    # WebCOOS omits a top-level 'count' (cursor-style paging), so fall back to
    # the number of records in this page rather than reporting "None".
    count = payload.get("count")
    if count is None:
        count = f"{len(payload.get('results', []))} in first page"
    return True, f"accepted — {count} assets visible"


def check_cdo(token):
    if not token:
        return False, "NOAA_CDO_TOKEN not set"
    try:
        r = requests.get(CDO_URL, headers={"token": token},
                         params={"limit": 1}, timeout=30)
    except requests.RequestException as exc:
        return False, f"could not reach www.ncei.noaa.gov: {exc}"

    # CDO returns 400 with a "Token parameter is required" style body for a bad
    # token, not always 401 — treat any non-200 as a failure and show the body.
    if r.status_code != 200:
        return False, f"token rejected ({r.status_code}): {r.text[:200]}"
    body = r.text.strip()
    if not body:
        return False, "200 OK but empty body — CDO does this for an invalid token"
    try:
        payload = r.json()
    except ValueError:
        return False, f"200 OK but response was not JSON: {body[:200]}"
    if "results" not in payload:
        return False, f"200 OK but no results key: {body[:200]}"
    return True, f"accepted — {payload.get('metadata', {}).get('resultset', {}).get('count')} datasets"


def main():
    checks = [
        ("WebCOOS   ", check_webcoos(os.environ.get("WEBCOOS_TOKEN"))),
        ("NOAA CDO  ", check_cdo(os.environ.get("NOAA_CDO_TOKEN"))),
    ]
    ok = True
    for name, (passed, detail) in checks:
        print(f"{name} {'OK  ' if passed else 'FAIL'}  {detail}")
        ok = ok and passed
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
