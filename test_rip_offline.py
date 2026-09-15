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


COROLLA = asset("Beachfront from Sailfish Street Beach Access, Corolla, NC",
                state="North Carolina", products=[
    ("Raw Video Data", "raw-video-data", "Rip Detection Results",
     "rip-detection-results", "currituck-rip-svc", 10239),
])
# One camera publishing two rip products. The count is what decides whether a
# pull is worth making, so the larger has to win.
TWO_RIP = asset("Two Products", state="Florida", products=[
    ("Raw Video Data", "raw-video-data", "Rip Detection Results",
     "rip-detection-results", "two-rip-old", 11),
    ("Raw Video Data", "raw-video-data", "Rip Detection v2",
     "rip-detection-results-v2", "two-rip-new", 4242),
])


def test_observation_window_is_per_camera():
    """Once a second region is on disk the global intersection is empty.

    Cala Millor's weather ends 2024-09-28, Corolla's begins 2025-09-08. With
    both present, intersecting every file in data/ yields nothing and
    --match-observations skipped all eight cameras -- including Walton, whose
    own observations were perfectly good.
    """
    print("the observation window is scoped to one camera")
    tmp = tempfile.mkdtemp()
    old_dir = r.OBS_DIR
    try:
        r.OBS_DIR = tmp

        def write(name, start, end):
            hours = pd.date_range(start, end, freq="D", tz="UTC")
            pd.DataFrame({"time": hours, "x": range(len(hours))}).to_csv(
                os.path.join(tmp, name), index=False)

        walton = "Walton Lighthouse, Santa Cruz, CA"
        corolla = "Beachfront from Hampton Inn, Corolla, NC"
        write(f"gridded_{r.obs_slug(walton)}.csv", "2025-08-25", "2026-08-25")
        write(f"marine_{r.obs_slug(walton)}.csv", "2025-08-25", "2026-08-25")
        write(f"gridded_{r.obs_slug(corolla)}.csv", "2025-09-08", "2026-09-08")
        # The file that used to poison the intersection for everybody.
        write("gridded_Cala_Millor.csv", "2011-05-21", "2024-09-28")

        check("the global intersection is empty, as it was in the real run",
              r.observation_window() is None)

        window = r.observation_window(walton)
        check("Walton still gets a window", window is not None)
        if window:
            check("and it is Walton's own span",
                  window[0].strftime("%Y-%m-%d") == "2025-08-25"
                  and window[1].strftime("%Y-%m-%d") == "2026-08-25",
                  [str(w) for w in window])

        corolla_window = r.observation_window(corolla)
        check("Corolla gets its own, different window",
              corolla_window is not None
              and corolla_window[0].strftime("%Y-%m-%d") == "2025-09-08",
              corolla_window)

        check("a camera with no observation file is reported, not intersected "
              "with someone else's",
              r.observation_window("Nowhere Beach") is None)
    finally:
        r.OBS_DIR = old_dir
        shutil.rmtree(tmp)


def test_obs_slug_matches_the_writers():
    print("the observation filename stem")
    # pull_site_observations.py and pull_gridded_weather.py both build the stem
    # this way; pull_rip_detection.slugify uses dashes and would never match.
    check("non-alphanumerics become underscores",
          r.obs_slug("Beachfront from Hampton Inn, Corolla, NC")
          == "Beachfront_from_Hampton_Inn__Corolla__NC",
          r.obs_slug("Beachfront from Hampton Inn, Corolla, NC"))
    check("it is cut at 48 characters, as the writers cut it",
          len(r.obs_slug("x" * 80)) == 48)
    check("it is NOT the dashed slug this module uses elsewhere",
          r.obs_slug("Walton Lighthouse") != r.slugify("Walton Lighthouse"))


def test_rip_cameras():
    print("the national rip roster")
    found = r.rip_cameras([CAPITOLA, COROLLA, WALTON, TWO_RIP])
    check("a camera with no rip product is left out",
          "Capitola Wharf" not in [c["label"] for c in found])
    check("three cameras carry one", len(found) == 3, [c["label"] for c in found])
    check("sorted by element count, largest first",
          [c["elements"] for c in found] == [35158, 10239, 4242],
          [c["elements"] for c in found])
    check("the state comes through",
          found[1]["state"] == "North Carolina", found[1]["state"])
    check("the service slug is carried, so the caller need not re-resolve it",
          found[0]["service_slug"] == "walton-rip-svc")
    check("a camera with two rip products keeps the larger",
          found[2]["service_slug"] == "two-rip-new", found[2]["service_slug"])
    check("the asset is carried so the sweep can act on it",
          found[0]["asset"] is WALTON)
    check("an empty catalogue is not an error", r.rip_cameras([]) == [])


def test_list_rip_cameras_survives_a_thin_catalogue():
    print("the roster prints without a state or a label")
    thin = asset(None, state=None, products=[
        ("Raw Video Data", "raw-video-data", "Rip", "rip-detection-results",
         "thin-svc", None)])
    rows = r.list_rip_cameras([thin, CAPITOLA], check_inventory=False)
    check("a camera with no label is still listed", len(rows) == 1)
    check("a null element count reads as zero, not a crash",
          rows[0]["elements"] == 0)
    check("nothing carrying rip means an empty roster, not an exit",
          r.list_rip_cameras([CAPITOLA], check_inventory=False) == [])


def test_sweep_isolates_each_camera():
    print("one bad camera does not end the sweep")
    calls = []

    def fake_run(asset_arg, args):
        label = r.dig(asset_arg, "data", "common", "label")
        calls.append(label)
        if "Corolla" in label:
            # find_stills_service and friends exit rather than raise; a sweep
            # that did not catch SystemExit would stop here with no summary.
            raise SystemExit("no stills product on this camera")
        if "Two Products" in label:
            raise RuntimeError("connection reset")
        return "1,234 rows"

    original = r.run_for_camera
    r.run_for_camera = fake_run
    try:
        class Args:
            state = None
        outcomes = r.sweep([CAPITOLA, COROLLA, WALTON, TWO_RIP], Args())
    finally:
        r.run_for_camera = original

    check("every rip camera was attempted", len(calls) == 3, calls)
    check("the camera with no rip product was never attempted",
          "Capitola Wharf" not in calls)
    statuses = {label: status for label, _, _, status in outcomes}
    check("a SystemExit is recorded as a skip, not a crash",
          statuses["Beachfront from Sailfish Street Beach Access, Corolla, NC"]
          == "skipped: no stills product on this camera")
    check("an exception is recorded with its type",
          statuses["Two Products"] == "RuntimeError: connection reset")
    check("the healthy camera still reports its rows",
          statuses["Walton Lighthouse, Santa Cruz, CA"] == "1,234 rows")
    check("the summary covers every camera", len(outcomes) == 3)


def test_sweep_filters_by_state():
    print("the sweep honours --state")
    seen = []
    original = r.run_for_camera
    r.run_for_camera = lambda a, args: seen.append(
        r.dig(a, "data", "common", "label")) or "ok"
    try:
        class Args:
            state = "North Carolina"
        r.sweep([COROLLA, WALTON, TWO_RIP], Args())
    finally:
        r.run_for_camera = original
    check("only the matching state runs", len(seen) == 1, seen)
    check("and it is the right one", "Corolla" in seen[0])


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


def test_find_camera_by_slug():
    print("resolving one of two cameras that share a substring")
    hampton = asset("Beachfront from Hampton Inn, Corolla, NC",
                    state="North Carolina", products=[
        ("Raw Video Data", "raw-video-data", "Rip", "rip-detection-results",
         "hampton-rip", 10766)])
    sailfish = asset("Beachfront from Sailfish Street Beach Access, Corolla, NC",
                     state="North Carolina", products=[
        ("Raw Video Data", "raw-video-data", "Rip", "rip-detection-results",
         "sailfish-rip", 10239)])
    both = [hampton, sailfish]

    check("the slug picks one out of an ambiguous pair",
          r.find_camera(both, "beachfront-from-hampton-inn-corolla-nc") is hampton)
    check("and the other one",
          r.find_camera(both, "beachfront-from-sailfish-street-beach-access-corolla-nc")
          is sailfish)
    check("an exact label still wins",
          r.find_camera(both, "Beachfront from Hampton Inn, Corolla, NC") is hampton)
    check("a unique substring still works",
          r.find_camera(both, "Sailfish") is sailfish)

    # The whole point of the message is that you can paste something out of it.
    try:
        r.find_camera(both, "Corolla")
    except SystemExit as exc:
        text = str(exc)
        check("an ambiguous name names both slugs",
              "beachfront-from-hampton-inn-corolla-nc" in text
              and "beachfront-from-sailfish-street-beach-access-corolla-nc" in text,
              text)
        check("and points at --all-rip as the other way out",
              "--all-rip" in text)
    else:
        check("an ambiguous name is refused", False, "no SystemExit")


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


# A verbatim record from the probe against Walton Lighthouse on 2026-08-31.
# Kept exactly as served so the parser is tested against the real shape rather
# than against what I assumed the shape would be.
REAL_RECORD = {
    "time": "2026-08-31T14:05:10Z",
    "annotated_image_url": "http://stage-webcoos-rip-detector-api.srv.axds.co/outputs/x.jpg",
    "classification_result": {
        "classification_model_framework": "ULTRALYTICS",
        "classification_model_name": "ripdetect_walton",
        "classification_model_version": "yolov8x_1.1",
        "detected": True,
        "detection_count": 1,
        "classification_scores": [{"rip_current": 0.7011650204658508}],
        "classification_bboxes": [[{"x": 1853, "y": 974}, {"x": 2255, "y": 1123}]],
    },
    "original_image_reference": "walton_lighthouse-2026-08-31-140451Z.jpg",
    "annotated_image_reference": "annotated.ripdetect_walton.walton-140451Z.jpg",
}


def test_flatten_real_record():
    print("\nflatten_record on the real payload")
    row = r.flatten_record(REAL_RECORD, source_file="probe.jsonl")
    check("time is read from the payload",
          str(row["timestamp"]).startswith("2026-08-31 14:05:10"), row["timestamp"])
    check("detected is a bool", row["detected"] is True)
    check("detection_count carried", row["detection_count"] == 1)
    check("score extracted", abs(row["score_max"] - 0.70116502) < 1e-6, row["score_max"])
    check("class name captured as data", row["score_classes"] == "rip_current",
          row["score_classes"])
    check("one box", row["bbox_count"] == 1)
    # (2255-1853) * (1123-974) = 402 * 149
    check("box area in pixels", row["bbox_area_max"] == 402 * 149, row["bbox_area_max"])
    check("box centroid x", row["bbox_x"] == (1853 + 2255) / 2, row["bbox_x"])
    check("box centroid y", row["bbox_y"] == (974 + 1123) / 2, row["bbox_y"])
    check("model version kept", row["model_version"] == "yolov8x_1.1")


def test_flatten_no_detection():
    print("\nflatten_record on an empty frame")
    empty = {"time": "2026-08-31T15:00:00Z",
             "classification_result": {"detected": False, "detection_count": 0,
                                       "classification_scores": [],
                                       "classification_bboxes": []}}
    row = r.flatten_record(empty)
    check("detected False survives", row["detected"] is False)
    check("no score rather than a zero", row["score_max"] is None, row["score_max"])
    check("no bbox area rather than a zero", row["bbox_area_max"] is None)
    check("bbox_count is a real zero", row["bbox_count"] == 0)

    # A second class must not need a code change.
    two = {"time": "2026-08-31T15:00:00Z", "classification_result": {
        "detected": True, "detection_count": 2,
        "classification_scores": [{"rip_current": 0.4}, {"shorebreak": 0.9}],
        "classification_bboxes": [[{"x": 0, "y": 0}, {"x": 10, "y": 10}],
                                  [{"x": 0, "y": 0}, {"x": 5, "y": 4}]]}}
    row = r.flatten_record(two)
    check("max is across all classes", row["score_max"] == 0.9, row["score_max"])
    check("both class names recorded",
          row["score_classes"] == "rip_current,shorebreak", row["score_classes"])
    check("largest box wins", row["bbox_area_max"] == 100, row["bbox_area_max"])

    broken = {"time": "not a date", "classification_result": {}}
    row = r.flatten_record(broken, element_time=pd.Timestamp("2026-01-01", tz="UTC"))
    check("an unparseable time falls back to the element time",
          str(row["timestamp"]).startswith("2026-01-01"), row["timestamp"])


def test_read_records_jsonl():
    print("\nread_records on .jsonl")
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "many.jsonl")
        with open(path, "w") as fh:
            fh.write(json.dumps(REAL_RECORD) + "\n")
            fh.write("\n")
            fh.write(json.dumps(REAL_RECORD) + "\n")
        check("every line is a record", len(r.read_records(path)) == 2,
              len(r.read_records(path)))

        bad = os.path.join(tmp, "bad.jsonl")
        with open(bad, "w") as fh:
            fh.write(json.dumps(REAL_RECORD) + "\n{ truncated\n")
        check("a bad line is skipped, the good one kept",
              len(r.read_records(bad)) == 1, len(r.read_records(bad)))

        empty = os.path.join(tmp, "empty.jsonl")
        open(empty, "w").close()
        check("an empty file yields nothing", r.read_records(empty) == [])

        as_list = os.path.join(tmp, "list.json")
        with open(as_list, "w") as fh:
            json.dump([REAL_RECORD, REAL_RECORD], fh)
        check("a .json array is read as many records",
              len(r.read_records(as_list)) == 2)
    finally:
        shutil.rmtree(tmp)


def test_build_table_jsonl():
    print("\nbuild_table on the real format")
    tmp = tempfile.mkdtemp()
    try:
        rows = write_rows(tmp, [
            (f"frame_{i}.jsonl", (json.dumps(REAL_RECORD) + "\n").encode())
            for i in range(3)])
        out = os.path.join(tmp, "rip.csv")
        table = r.build_table(rows, out)
        check("jsonl is parsed, not skipped", table is not None)
        check("one row per frame", len(table) == 3, None if table is None else len(table))
        check("detection columns present",
              {"detected", "score_max", "bbox_area_max"} <= set(table.columns))
    finally:
        shutil.rmtree(tmp)


def test_hourly_summary():
    print("\nhourly_summary")
    tmp = tempfile.mkdtemp()
    try:
        base = pd.Timestamp("2026-08-31T14:00:00Z")
        frames = pd.DataFrame([
            {"timestamp": base, "detected": True, "detection_count": 1,
             "score_max": 0.7, "bbox_area_max": 100},
            {"timestamp": base + pd.Timedelta(minutes=20), "detected": False,
             "detection_count": 0, "score_max": None, "bbox_area_max": None},
            {"timestamp": base + pd.Timedelta(minutes=40), "detected": True,
             "detection_count": 2, "score_max": 0.9, "bbox_area_max": 400},
            {"timestamp": base + pd.Timedelta(hours=1), "detected": False,
             "detection_count": 0, "score_max": None, "bbox_area_max": None},
        ])
        out = os.path.join(tmp, "hourly.csv")
        hourly = r.hourly_summary(frames, out)
        check("one row per hour", len(hourly) == 2, len(hourly))
        first = hourly.iloc[0]
        check("frames counted", first["frames"] == 3, first["frames"])
        check("detections summed across frames", first["detections"] == 3,
              first["detections"])
        check("rate is frames-with-detection over frames, not detections over frames",
              abs(first["detection_rate"] - 2 / 3) < 5e-5, first["detection_rate"])
        check("score_max is the hour's peak", first["score_max"] == 0.9)
        check("score_mean averages the detected frames only",
              abs(first["score_mean"] - 0.8) < 1e-9, first["score_mean"])

        # The hour with frames but no detection is an observed zero.
        second = hourly.iloc[1]
        check("an all-clear hour survives rather than vanishing",
              second["frames"] == 1 and second["detection_rate"] == 0.0)
        check("its score is blank, not zero", pd.isna(second["score_max"]),
              second["score_max"])
    finally:
        shutil.rmtree(tmp)


def test_build_table_json():
    print("\nbuild_table on JSON payloads")
    tmp = tempfile.mkdtemp()
    try:
        rows = write_rows(tmp, [(f"frame_{i}.json", json.dumps(REAL_RECORD).encode())
                                for i in range(3)])
        out = os.path.join(tmp, "rip.csv")
        table = r.build_table(rows, out)
        check("a table is produced", table is not None)
        check("one row per payload", len(table) == 3, len(table))
        check("a .json payload goes through the same flattener as .jsonl",
              {"detected", "score_max"} <= set(table.columns), list(table.columns))
        check("element time is kept alongside the payload time",
              table["element_time"].nunique() == 3)
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


def test_inventory_range():
    print("\ninventory_range")
    rows = [
        ["2024-01-01T00:00:00Z", True, "2024-02-01T00:00:00Z", 120, 9,
         "2024-01-03T04:00:00Z", "2024-01-28T22:00:00Z"],
        ["2024-02-01T00:00:00Z", False, "2024-03-01T00:00:00Z", 0, 0, None, None],
        ["2024-03-01T00:00:00Z", True, "2024-04-01T00:00:00Z", 80, 7,
         "2024-03-02T01:00:00Z", "2024-03-30T18:00:00Z"],
    ]
    frame = pd.DataFrame(rows, columns=r.INVENTORY_COLUMNS)
    first, last = r.inventory_range(frame)
    check("first is the earliest populated bin's data start",
          str(first).startswith("2024-01-03"), first)
    check("last is the latest populated bin's data end",
          str(last).startswith("2024-03-30"), last)

    # An empty bin must not stretch the range: its Bin Start/End are real
    # dates even though it holds nothing.
    only_empty = pd.DataFrame([rows[1]], columns=r.INVENTORY_COLUMNS)
    first2, last2 = r.inventory_range(only_empty)
    check("a wholly empty inventory falls back rather than lying",
          first2 is None or str(first2).startswith("2024-02"), first2)

    check("empty frame is handled",
          r.inventory_range(pd.DataFrame()) == (None, None))
    check("None is handled", r.inventory_range(None) == (None, None))

    # Unknown schema: no named columns, but timestamps still recoverable.
    positional = pd.DataFrame([["2025-05-01T00:00:00Z", "2025-05-09T00:00:00Z", 5]],
                              columns=["col0", "col1", "col2"])
    first3, last3 = r.inventory_range(positional)
    check("an unrecognised schema still yields a range",
          first3 is not None and str(last3).startswith("2025-05-09"), (first3, last3))


def test_duration_format():
    print("the progress line's clock")
    check("seconds stay seconds", r._duration(5) == "5s")
    check("minutes carry seconds", r._duration(75) == "1m15s")
    check("hours carry minutes", r._duration(3700) == "1h01m")
    check("a full day does not wrap", r._duration(86400) == "24h00m")
    # The estimate is elapsed/index * remaining, which is negative for a
    # moment on the last item if the clock jitters. A negative ETA printed as
    # "-1s left" reads as a bug in the run rather than in the arithmetic.
    check("a negative estimate clamps to zero", r._duration(-4) == "0s")


def test_coverage_resumes():
    print("\nbuild_coverage resume")
    tmp = tempfile.mkdtemp()
    calls = []

    def fake_fetch(service, start, end, interval_minutes=None, quiet=False):
        """Two images an hour for two hours a day, and fail on the third day."""
        label = start.strftime("%Y-%m-%d")
        calls.append(label)
        if label == "2025-06-03" and calls.count(label) == 1:
            raise TimeoutError("simulated read timeout")
        return [{"timestamp": start + pd.Timedelta(hours=h, minutes=m)}
                for h in (14, 15) for m in (0, 30)]

    original = r.fetch_elements
    try:
        r.fetch_elements = fake_fetch
        out = os.path.join(tmp, "coverage_test_hourly.csv")
        start = pd.Timestamp("2025-06-01", tz="UTC")
        end = pd.Timestamp("2025-06-05", tz="UTC")
        try:
            r.build_coverage("svc", start, end, out)
            check("the simulated failure propagated", False)
        except TimeoutError:
            check("the simulated failure propagated", True)

        progress = pd.read_csv(out.replace(".csv", "_progress.csv"))
        check("days before the failure are committed", len(progress) == 2,
              len(progress))
        check("the failed day is not marked done",
              "2025-06-03" not in set(progress["date"].astype(str)),
              progress["date"].tolist())

        before = len(calls)
        hourly = r.build_coverage("svc", start, end, out)
        retried = calls[before:]
        check("the resume skips the days already done",
              "2025-06-01" not in retried and "2025-06-02" not in retried, retried)
        check("and picks up at the failed day", retried[0] == "2025-06-03",
              retried[0])
        check("all four days end up enumerated",
              len(pd.read_csv(out.replace(".csv", "_progress.csv"))) == 4)
        check("hours are counted, not elements",
              set(hourly["images"]) == {2}, hourly["images"].tolist())
        check("two hours per day over four days", len(hourly) == 8, len(hourly))
    finally:
        r.fetch_elements = original
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
    test_find_camera_by_slug()
    test_find_rip_service()
    test_flatten_real_record()
    test_flatten_no_detection()
    test_read_records_jsonl()
    test_build_table_jsonl()
    test_hourly_summary()
    test_build_table_json()
    test_build_table_binary()
    test_build_table_mixed()
    test_inventory_range()
    test_duration_format()
    test_coverage_resumes()
    test_describe_json()
    test_observation_window_is_per_camera()
    test_obs_slug_matches_the_writers()
    test_rip_cameras()
    test_list_rip_cameras_survives_a_thin_catalogue()
    test_sweep_isolates_each_camera()
    test_sweep_filters_by_state()
    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
    raise SystemExit(1 if FAILURES else 0)
