"""Offline checks for label_server.py: save, overwrite, resume and import.

The bug these exist to pin: the page used to set rip_present locally BEFORE
the POST and never put it back when the POST failed, so a failed save looked
labelled in the browser, "next unlabelled" walked past it, and the work went
nowhere. The server side of that contract is what is testable here -- that a
save is on disk before the response is written, that a rejected save changes
nothing, and that what the page reads back on load is the disk's version of
events rather than the browser's.

Every check runs against a real CSV in a temporary directory, through the same
functions the running server calls.
"""

import json
import os
import sys
import tempfile
import threading
import urllib.request
from http.server import HTTPServer

import label_server as ls

FAILURES = []

COLUMNS = ["frame_id", "timestamp", "stratum", "confidence", "wave_tercile",
           "booster", "score_max", "detection_count", "bbox_count",
           "bbox_area_max", "mop_wave_height", "cloud_cover",
           "solar_elevation", "image", "boxes", "rip_present", "notes",
           "labeled_at"]


def check(name, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    if not condition:
        FAILURES.append(name)
    print(f"  {mark} {name}" + (f"  {detail}" if detail else ""))


def make_csv(folder, count=6, boxes=2):
    """A labels.csv with no box_labels column, as build_label_sample writes it."""
    import csv
    path = os.path.join(folder, "labels.csv")
    payload = json.dumps([{"x": 10 * b, "y": 20, "w": 30, "h": 40}
                          for b in range(boxes)])
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for index in range(count):
            writer.writerow({c: "" for c in COLUMNS} | {
                "frame_id": f"F{index:03d}",
                "timestamp": f"2026-03-0{index + 1}T12:00:00+00:00",
                "stratum": "high / H1 low", "confidence": "high",
                "wave_tercile": "H1 low", "score_max": "0.8",
                "bbox_count": str(boxes),
                "image": f"F{index:03d}.jpg", "boxes": payload})
    return path


def use(folder):
    """Point the module at a temporary sample and return the csv path."""
    ls.OUT_DIR = folder
    ls.IMAGE_DIR = os.path.join(folder, "images")
    ls.LABEL_CSV = os.path.join(folder, "labels.csv")
    os.makedirs(ls.IMAGE_DIR, exist_ok=True)
    return make_csv(folder)


def check_a_save_is_on_disk_before_it_answers():
    with tempfile.TemporaryDirectory() as folder:
        use(folder)
        ok, message, done, total = ls.save_label("F002", "yes", "a swirl")
        check("the save reports success", ok and done == 1 and total == 6,
              f"{message} {done}/{total}")

        rows, fields = ls.read_rows()
        hit = next(r for r in rows if r["frame_id"] == "F002")
        check("the verdict is in the file, not just in the return value",
              hit["rip_present"] == "yes", hit["rip_present"])
        check("the note went with it", hit["notes"] == "a swirl")
        check("a timestamp was stamped", bool(hit["labeled_at"]))
        check("no row was added or lost", len(rows) == 6, str(len(rows)))
        check("every other row is still blank",
              all(r["rip_present"] == "" for r in rows if r["frame_id"] != "F002"))
        check("the box_labels column was added to a CSV without one",
              ls.BOX_COLUMN in fields, str(fields[-1]))
        check("the count of labelled rows is 1, not 6",
              ls.count_done(rows) == 1, str(ls.count_done(rows)))


def check_unusable_is_its_own_verdict():
    with tempfile.TemporaryDirectory() as folder:
        use(folder)
        ok, *_ = ls.save_label("F000", "unusable", "lens water")
        check("unusable is accepted", ok)
        rows, _ = ls.read_rows()
        hit = next(r for r in rows if r["frame_id"] == "F000")
        check("and stored distinctly from doubt",
              hit["rip_present"] == "unusable", hit["rip_present"])
        ok_bad, message, *_ = ls.save_label("F001", "maybe", "")
        check("an unknown verdict is refused", not ok_bad, message)
        rows, _ = ls.read_rows()
        check("and the refused save changed nothing on disk",
              all(r["rip_present"] in ("", "unusable") for r in rows))
        ok_missing, *_ = ls.save_label("NOPE", "yes", "")
        check("an unknown frame_id is refused", not ok_missing)


def check_relabelling_overwrites_rather_than_duplicating():
    with tempfile.TemporaryDirectory() as folder:
        use(folder)
        ls.save_label("F003", "yes", "first pass")
        ls.save_label("F003", "no", "second look")
        rows, _ = ls.read_rows()
        matches = [r for r in rows if r["frame_id"] == "F003"]
        check("relabelling leaves one row, not two", len(matches) == 1,
              f"{len(matches)} rows")
        check("and it holds the newer verdict",
              matches[0]["rip_present"] == "no", matches[0]["rip_present"])
        check("and the newer note", matches[0]["notes"] == "second look")
        check("the file still has six rows", len(rows) == 6, str(len(rows)))

        ls.save_label("F003", "", "")
        rows, _ = ls.read_rows()
        cleared = next(r for r in rows if r["frame_id"] == "F003")
        check("clearing a verdict also clears its timestamp",
              cleared["rip_present"] == "" and cleared["labeled_at"] == "")
        check("a cleared row is not counted as done",
              ls.count_done(rows) == 0, str(ls.count_done(rows)))


def check_box_labels_round_trip():
    with tempfile.TemporaryDirectory() as folder:
        use(folder)
        ls.save_label("F001", "yes", "", {"0": True, "1": False})
        rows, _ = ls.read_rows()
        hit = next(r for r in rows if r["frame_id"] == "F001")
        check("per-box marks survive the CSV as JSON",
              json.loads(hit[ls.BOX_COLUMN]) == {"0": True, "1": False},
              hit[ls.BOX_COLUMN])

        check("a JSON string is accepted as readily as a dict",
              ls.clean_box_labels('{"2": true}') == '{"2": true}',
              ls.clean_box_labels('{"2": true}'))
        check("a non-boolean mark is dropped rather than coerced",
              ls.clean_box_labels({"0": "true"}) == "",
              ls.clean_box_labels({"0": "true"}))
        check("a non-integer box key is dropped",
              ls.clean_box_labels({"left": True}) == "")
        check("nothing marked stores nothing", ls.clean_box_labels({}) == "")
        check("malformed JSON does not raise",
              ls.clean_box_labels("{not json") == "")


def check_resume_reads_the_disk_not_the_browser():
    """What /rows serves after a restart is what the page resumes from."""
    with tempfile.TemporaryDirectory() as folder:
        use(folder)
        ls.save_label("F000", "yes", "")
        ls.save_label("F001", "no", "")

        rows, _ = ls.read_rows()
        first_blank = next(i for i, r in enumerate(rows)
                           if not (r["rip_present"] or "").strip())
        check("the first unlabelled row is index 2, where work stopped",
              first_blank == 2, str(first_blank))
        check("two of six are done", ls.count_done(rows) == 2)

        # A save that never reached disk must not appear on reload -- the whole
        # point of the rewrite. Simulate it by rejecting one.
        ls.save_label("F002", "bogus", "")
        rows, _ = ls.read_rows()
        check("a rejected save leaves the resume point where it was",
              next(i for i, r in enumerate(rows)
                   if not (r["rip_present"] or "").strip()) == 2)


def check_import_merges_without_destroying_newer_work():
    with tempfile.TemporaryDirectory() as folder:
        use(folder)
        ls.save_label("F000", "yes", "labelled here")

        dump = os.path.join(folder, "dump.json")
        with open(dump, "w") as handle:
            json.dump({"F000": {"rip_present": "no", "notes": "stale"},
                       "F001": {"rip_present": "doubt", "notes": "from browser"},
                       "F002": {"rip_present": "unsure"},
                       "GHOST": {"rip_present": "yes"}}, handle)

        applied, skipped, problems = ls.import_labels(dump)
        check("two new labels were applied", applied == 2, str(applied))
        check("the verdict already on disk was kept, not replaced",
              skipped == 1, str(skipped))
        check("a frame_id not in the sample is reported, not silently dropped",
              any("GHOST" in p for p in problems), str(problems))

        rows, _ = ls.read_rows()
        by_id = {r["frame_id"]: r for r in rows}
        check("the existing verdict survived the import",
              by_id["F000"]["rip_present"] == "yes"
              and by_id["F000"]["notes"] == "labelled here")
        check("the imported verdict landed",
              by_id["F001"]["rip_present"] == "doubt")
        check("'unsure' is read as doubt rather than refused",
              by_id["F002"]["rip_present"] == "doubt",
              by_id["F002"]["rip_present"])

        applied2, skipped2, _ = ls.import_labels(dump, overwrite=True)
        rows, _ = ls.read_rows()
        check("--import-overwrite does replace it",
              ls.read_rows()[0][0]["rip_present"] == "no",
              f"applied {applied2} skipped {skipped2}")

        # A list is as acceptable as a mapping.
        listed = os.path.join(folder, "list.json")
        with open(listed, "w") as handle:
            json.dump([{"frame_id": "F003", "verdict": "unusable",
                        "notes": "dark"}], handle)
        applied3, _, _ = ls.import_labels(listed)
        check("a list dump imports too", applied3 == 1)
        rows, _ = ls.read_rows()
        check("with its verdict and note",
              next(r for r in rows if r["frame_id"] == "F003")["notes"] == "dark")


def check_the_http_round_trip():
    """Through the real handler, because that is what the browser talks to."""
    with tempfile.TemporaryDirectory() as folder:
        use(folder)
        server = HTTPServer(("127.0.0.1", 0), ls.Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{port}"
            page = urllib.request.urlopen(base + "/").read().decode()
            check("the page is served", "<title>" in page and "/save" in page)

            body = json.dumps({"frame_id": "F004", "verdict": "yes",
                               "notes": "via http",
                               "box_labels": {"0": True}}).encode()
            request = urllib.request.Request(
                base + "/save", data=body,
                headers={"Content-Type": "application/json"})
            answer = json.loads(urllib.request.urlopen(request).read())
            check("the POST reports ok", answer["ok"] and answer["done"] == 1,
                  str(answer))

            rows = json.loads(urllib.request.urlopen(base + "/rows").read())["rows"]
            hit = next(r for r in rows if r["frame_id"] == "F004")
            check("and /rows serves the saved verdict straight back",
                  hit["rip_present"] == "yes" and hit["notes"] == "via http")
            check("with the box marks", json.loads(hit[ls.BOX_COLUMN]) == {"0": True})

            bad = urllib.request.Request(
                base + "/save",
                data=json.dumps({"frame_id": "F004", "verdict": "??"}).encode(),
                headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(bad)
                check("a bad verdict is rejected over HTTP", False, "no error")
            except urllib.error.HTTPError as exc:
                payload = json.loads(exc.read())
                check("a bad verdict is rejected over HTTP",
                      exc.code == 400 and not payload["ok"], str(payload))

            rows, _ = ls.read_rows()
            check("and the rejection left the good verdict intact",
                  next(r for r in rows
                       if r["frame_id"] == "F004")["rip_present"] == "yes")

            for attempt in ("/images/../labels.csv", "/images/%2e%2e%2flabels.csv"):
                try:
                    urllib.request.urlopen(base + attempt)
                    check(f"escape refused: {attempt[8:][:24]}", False, "served")
                except urllib.error.HTTPError as exc:
                    check(f"escape refused: {attempt[8:][:24]}", exc.code == 404)
        finally:
            server.shutdown()
            server.server_close()


def check_a_csv_backup_imports_as_readily_as_json():
    """The rebuild's own backup is a CSV, and that is what gets imported.

    Telling someone to convert recovered labels to JSON first would be a step
    that exists only because the reader was narrow -- and the rebuild had
    already lost 27 labels once by the time this was written.
    """
    import csv as csvmod
    with tempfile.TemporaryDirectory() as folder:
        use(folder)

        backup = os.path.join(folder, "labels_20260916T000000Z.csv")
        with open(backup, "w", newline="") as handle:
            writer = csvmod.DictWriter(
                handle, fieldnames=["frame_id", "rip_present", "notes",
                                    "labeled_at", "box_labels"])
            writer.writeheader()
            writer.writerow({"frame_id": "F001", "rip_present": "yes",
                             "notes": "recovered", "labeled_at": "",
                             "box_labels": '{"0": true}'})
            writer.writerow({"frame_id": "F002", "rip_present": "doubt",
                             "notes": "", "labeled_at": "", "box_labels": ""})
            # An unlabelled row in the backup must not import as a verdict.
            writer.writerow({"frame_id": "F003", "rip_present": "",
                             "notes": "", "labeled_at": "", "box_labels": ""})

        applied, skipped, problems = ls.import_labels(backup)
        check("the two verdicts in the CSV are imported", applied == 2,
              f"{applied} applied, {problems}")
        rows, _ = ls.read_rows()
        by_id = {r["frame_id"]: r for r in rows}
        check("with their verdicts", by_id["F001"]["rip_present"] == "yes"
              and by_id["F002"]["rip_present"] == "doubt")
        check("and their notes", by_id["F001"]["notes"] == "recovered")
        check("and their box marks",
              json.loads(by_id["F001"][ls.BOX_COLUMN]) == {"0": True})
        check("the blank row did not import as a verdict",
              by_id["F003"]["rip_present"] == "", by_id["F003"]["rip_present"])
        check("nothing was reported as a problem", not problems, str(problems))


def main():
    print("labelling server offline checks\n")
    check_a_save_is_on_disk_before_it_answers()
    check_unusable_is_its_own_verdict()
    check_relabelling_overwrites_rather_than_duplicating()
    check_box_labels_round_trip()
    check_resume_reads_the_disk_not_the_browser()
    check_import_merges_without_destroying_newer_work()
    check_a_csv_backup_imports_as_readily_as_json()
    check_the_http_round_trip()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
