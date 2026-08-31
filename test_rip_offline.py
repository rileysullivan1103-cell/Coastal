"""Offline checks for pull_rip_detection, on synthetic fixtures.

Every network call is avoided: the asset catalogue and the downloaded payloads
are built here. What is actually under test is the part that can be wrong
without the API telling you -- slug matching, camera resolution, and turning a
directory of payloads into a table stamped with the right times.

    python test_rip_offline.py
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone

import pandas as pd

import pull_rip_detection as r

FAILURES = []


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def asset(label, state="California", products=()):
    """products: [(feed_label, feed_slug, prod_label, prod_slug, svc, count)]"""
    feeds = {}
    for flabel, fslug, plabel, pslug, svc, count in products:
        feed = feeds.setdefault((flabel, fslug), {
            "data": {"common": {"label": flabel, "slug": fslug}}, "products": []})
        feed["products"].append({
            "data": {"common": {"label": plabel, "slug": pslug}},
            "services": [{"data": {"common": {"slug": svc}},
                          "elements": {"count": count}}]})
    return {"data": {"common": {"label": label},
                     "properties": {"state_or_territory": state}},
            "feeds": list(feeds.values())}


WALTON = asset("Walton Lighthouse, Santa Cruz, CA", products=[
    ("Raw Video Data", "raw-video-data", "Rip Detection Results",
     "rip-detection-results", "walton-rip-svc", 35158),
    ("Raw Video Data", "raw-video-data", "One Minute Stills",
     "one-minute-stills", "walton-stills-svc", 900000),
])
CAPITOLA = asset("Capitola Wharf", products=[
    ("Raw Video Data", "raw-video-data", "One Minute Stills",
     "one-minute-stills", "cap-stills-svc", 500000),
])
ASSETS = [WALTON, CAPITOLA]


def test_slugify():
    print("\nslugify")
    check("spaces and commas collapse",
          r.slugify("Walton Lighthouse, Santa Cruz, CA") == "walton-lighthouse-santa-cruz-ca",
          r.slugify("Walton Lighthouse, Santa Cruz, CA"))
    check("no leading or trailing dashes", not r.slugify(", CA ,").startswith("-"))
    check("empty is safe", r.slugify(None) == "")


def test_find_camera():
    print("\nfind_camera")
    check("substring resolves to the full label",
          r.dig(r.find_camera(ASSETS, "Walton"), "data", "common", "label")
          == "Walton Lighthouse, Santa Cruz, CA")
    check("exact label wins",
          r.find_camera(ASSETS, "Capitola Wharf") is CAPITOLA)
    try:
        r.find_camera(ASSETS, "Nowhere Beach")
        check("unknown camera exits", False)
    except SystemExit:
        check("unknown camera exits", True)


def test_find_rip_service():
    print("\nfind_rip_service")
    svc, label = r.find_rip_service(WALTON)
    check("picks the rip service, not the stills one", svc == "walton-rip-svc", svc)
    check("returns the LABEL pywebcoos would need",
          label == "Rip Detection Results", label)
    try:
        r.find_rip_service(CAPITOLA)
        check("camera without the product exits", False)
    except SystemExit:
        check("camera without the product exits", True)

    # The whole reason this script does not lean on pywebcoos: the feed label
    # here is "Raw Video Data", so the library's `label == 'raw-video-data'`
    # comparison fails and matching by slug is the only thing that works.
    feed_labels = {row[0] for row in r.camera_products(WALTON)}
    feed_slugs = {row[1] for row in r.camera_products(WALTON)}
    check("fixture reproduces the label/slug mismatch",
          "raw-video-data" in feed_slugs and "raw-video-data" not in feed_labels)


def write_rows(tmp, payloads):
    """payloads: [(filename, bytes)] -> row dicts as download() returns."""
    rows = []
    for i, (name, blob) in enumerate(payloads):
        path = os.path.join(tmp, name)
        with open(path, "wb") as fh:
            fh.write(blob)
        rows.append({"timestamp": datetime(2025, 6, 1, 12, i, tzinfo=timezone.utc),
                     "filename": name, "path": path,
                     "url": f"https://example.invalid/{name}"})
    return rows


def test_build_table_json():
    print("\nbuild_table on JSON payloads")
    tmp = tempfile.mkdtemp()
    try:
        rows = write_rows(tmp, [
            (f"frame_{i}.json",
             json.dumps({"rip_probability": 0.1 * i,
                         "detections": {"count": i}}).encode())
            for i in range(3)])
        out = os.path.join(tmp, "rip.csv")
        table = r.build_table(rows, out)
        check("a table is produced", table is not None)
        check("one row per payload", len(table) == 3, len(table))
        check("nested keys are flattened",
              "detections.count" in table.columns, list(table.columns))
        check("element time is carried onto every row",
              table["timestamp"].nunique() == 3)
        check("the index is written too", os.path.exists(out.replace(".csv", "_index.csv")))
        index = pd.read_csv(out.replace(".csv", "_index.csv"))
        check("index has one line per element", len(index) == 3, len(index))
    finally:
        shutil.rmtree(tmp)


def test_build_table_binary():
    print("\nbuild_table on imagery")
    tmp = tempfile.mkdtemp()
    try:
        rows = write_rows(tmp, [(f"frame_{i}.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 64)
                                for i in range(4)])
        out = os.path.join(tmp, "rip.csv")
        table = r.build_table(rows, out)
        check("imagery yields no table rather than a wrong one", table is None)
        check("no combined CSV is written", not os.path.exists(out))
        index = pd.read_csv(out.replace(".csv", "_index.csv"))
        check("but the frames are still indexed by time", len(index) == 4, len(index))
    finally:
        shutil.rmtree(tmp)


def test_build_table_mixed():
    print("\nbuild_table on a mix, including a corrupt file")
    tmp = tempfile.mkdtemp()
    try:
        rows = write_rows(tmp, [
            ("good.json", json.dumps({"rip": 1}).encode()),
            ("truncated.json", b'{"rip": '),
            ("stills.jpg", b"\xff\xd8\xff"),
        ])
        table = r.build_table(rows, os.path.join(tmp, "rip.csv"))
        check("the readable payload survives", table is not None and len(table) == 1,
              None if table is None else len(table))
        check("the corrupt one does not abort the run", True)
    finally:
        shutil.rmtree(tmp)


def test_describe_json():
    print("\ndescribe_json")
    try:
        r.describe_json({"a": {"b": [{"c": 1}]}, "d": [1, 2, 3], "e": "x"})
        check("walks nested structures without raising", True)
        r.describe_json([])
        r.describe_json({})
        check("handles empty containers", True)
    except Exception as exc:  # noqa: BLE001 - the point is that nothing escapes
        check("walks nested structures without raising", False, repr(exc))


if __name__ == "__main__":
    test_slugify()
    test_find_camera()
    test_find_rip_service()
    test_build_table_json()
    test_build_table_binary()
    test_build_table_mixed()
    test_describe_json()
    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
    raise SystemExit(1 if FAILURES else 0)
