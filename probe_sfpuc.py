"""Find the data endpoint behind the SFPUC Beaches and Bay web app.

https://webapps.sfpuc.org/sapps/beachesandbay.html is a rendered page, not an
API. Whatever it plots — combined sewer discharge events, beach advisories —
arrives from some backend it calls. Guessing at that backend's shape is exactly
how the field-name bugs in this project started, so this finds it instead.

    python probe_sfpuc.py
    python probe_sfpuc.py --fetch-candidates    also GET each candidate found

Report what it prints and the client can be written against the real response.
"""

import json
import re
import sys
from urllib.parse import urljoin

import requests

PAGE = "https://webapps.sfpuc.org/sapps/beachesandbay.html"
UA = {"User-Agent": "Mozilla/5.0 (coastal-pipeline data discovery)"}

# Things that look like a data source rather than a stylesheet or image.
ENDPOINT_HINTS = ("/api/", ".json", "/rest/services", "arcgis", "query?",
                  "/data/", "geojson", ".ashx", "/services/", "FeatureServer",
                  "MapServer", ".svc", "/feed", ".csv")
NOISE = (".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".woff")


def candidates_from(text, base):
    """Every URL-ish string in the page that could be a data endpoint."""
    found = set()

    # src= / href= attributes, and bare quoted URLs or paths in inline script.
    for pattern in (r'''(?:src|href|url)\s*=\s*["']([^"']+)["']''',
                    r'''["'](https?://[^"'\s]+)["']''',
                    r'''["'](/[A-Za-z0-9_\-./]+(?:\.json|\.ashx|\.csv)[^"'\s]*)["']''',
                    r'''(?:fetch|ajax|open|get|post|load)\s*\(\s*["']([^"']+)["']'''):
        for match in re.findall(pattern, text, flags=re.I):
            found.add(match)

    keep = {}
    for raw in found:
        low = raw.lower()
        if any(low.endswith(ext) for ext in NOISE):
            continue
        if any(hint in low for hint in ENDPOINT_HINTS):
            keep[urljoin(base, raw)] = "endpoint-like"
        elif low.endswith(".js"):
            keep[urljoin(base, raw)] = "script (may contain the real URL)"
    return keep


def embedded_json(text):
    """JSON assigned to a variable in the page itself."""
    out = []
    for match in re.finditer(r'(?:var|let|const)\s+(\w+)\s*=\s*(\[|\{)', text):
        name, start = match.group(1), match.start(2)
        depth, closing = 0, {"[": "]", "{": "}"}[match.group(2)]
        for i in range(start, min(len(text), start + 400_000)):
            if text[i] in "[{":
                depth += 1
            elif text[i] in "]}":
                depth -= 1
                if depth == 0:
                    blob = text[start:i + 1]
                    try:
                        parsed = json.loads(blob)
                    except ValueError:
                        break
                    if isinstance(parsed, (list, dict)) and len(blob) > 80:
                        out.append((name, parsed))
                    break
    return out


def main():
    print(f"GET {PAGE}")
    resp = requests.get(PAGE, headers=UA, timeout=60)
    print(f"HTTP {resp.status_code}, {len(resp.text)} chars, "
          f"{resp.headers.get('content-type')}\n")
    resp.raise_for_status()

    found = candidates_from(resp.text, PAGE)
    print("=== CANDIDATE ENDPOINTS AND SCRIPTS IN THE PAGE ===")
    if not found:
        print("  none — the page probably builds its URLs at runtime.")
        print("  Open it in a browser, take the Network tab, and note any")
        print("  XHR/fetch request; that URL is what we need.")
    for url, kind in sorted(found.items(), key=lambda kv: kv[1]):
        print(f"  [{kind}] {url}")

    blobs = embedded_json(resp.text)
    if blobs:
        print("\n=== JSON EMBEDDED DIRECTLY IN THE PAGE ===")
        for name, parsed in blobs:
            size = len(parsed)
            print(f"  {name}: {type(parsed).__name__} of {size}")
            sample = parsed[0] if isinstance(parsed, list) and parsed else parsed
            if isinstance(sample, dict):
                for key, value in list(sample.items())[:15]:
                    print(f"      {key:<28} {str(value)[:50]}")

    # Scripts frequently hold the real endpoint as a string constant.
    scripts = [u for u, k in found.items() if k.startswith("script")]
    if scripts:
        print("\n=== SCANNING LINKED SCRIPTS FOR ENDPOINTS ===")
        for url in scripts[:10]:
            try:
                js = requests.get(url, headers=UA, timeout=60)
                js.raise_for_status()
            except requests.RequestException as exc:
                print(f"  {url}: {exc}")
                continue
            inner = candidates_from(js.text, url)
            inner = {u: k for u, k in inner.items() if k == "endpoint-like"}
            print(f"  {url}: {len(inner)} endpoint-like URLs")
            for u in sorted(inner):
                print(f"      {u}")

    if "--fetch-candidates" in sys.argv:
        print("\n=== FETCHING ENDPOINT-LIKE CANDIDATES ===")
        for url, kind in sorted(found.items()):
            if kind != "endpoint-like":
                continue
            try:
                r = requests.get(url, headers=UA, timeout=60)
            except requests.RequestException as exc:
                print(f"  {url}\n      {exc}")
                continue
            ctype = r.headers.get("content-type", "")
            print(f"  {url}\n      HTTP {r.status_code}, {ctype}, {len(r.content)} bytes")
            if "json" in ctype:
                try:
                    payload = r.json()
                except ValueError:
                    continue
                sample = payload[0] if isinstance(payload, list) and payload else payload
                if isinstance(sample, dict):
                    for key, value in list(sample.items())[:15]:
                        print(f"        {key:<28} {str(value)[:50]}")


if __name__ == "__main__":
    main()
