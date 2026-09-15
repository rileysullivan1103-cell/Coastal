"""Offline checks for diagnose_detection_boxes.py.

The transforms are tested against a letterbox applied in the FORWARD
direction: take a box whose position in the still is known, put it through the
same resize-and-pad ultralytics does, and require the candidate transform to
bring it back to where it started. Asserting the inverse against a hand-copied
expected number would only check that two copies of my arithmetic agree.
"""

import json
import os
import sys
import tempfile

import diagnose_detection_boxes as dx

FAILURES = []

WIDTH, HEIGHT, MODEL = 1920, 1080, 640


def check(name, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    if not condition:
        FAILURES.append(name)
    print(f"  {mark} {name}" + (f"  {detail}" if detail else ""))


def forward_letterbox(x, y, width=WIDTH, height=HEIGHT, model=MODEL):
    """Still pixels -> model-canvas pixels, the way the model is fed.

    One scale for both axes, remainder split evenly. This is the ground truth
    the inverse is checked against.
    """
    scale = min(model / width, model / height)
    return x * scale + (model - width * scale) / 2, \
        y * scale + (model - height * scale) / 2


def check_the_letterbox_inverse_returns_the_box():
    # A box low in the frame, over water, well below the horizon.
    truth = {"x": 700.0, "y": 820.0, "w": 260.0, "h": 90.0}
    corners = [(truth["x"], truth["y"]),
               (truth["x"] + truth["w"], truth["y"] + truth["h"])]
    payload = [forward_letterbox(x, y) for x, y in corners]

    scale, pad_x, pad_y = dx.letterbox_geometry(WIDTH, HEIGHT, MODEL)
    check("a 16:9 still in a square canvas is padded top and bottom only",
          abs(pad_x) < 1e-9 and pad_y > 100,
          f"pad_x {pad_x:.1f} pad_y {pad_y:.1f}")
    check("the scale is the smaller of the two ratios",
          abs(scale - MODEL / WIDTH) < 1e-9, f"{scale:.4f}")

    back = dx.transform_box("letterbox", payload, WIDTH, HEIGHT, MODEL)
    for key in ("x", "y", "w", "h"):
        check(f"letterbox inverse recovers {key}",
              abs(back[key] - truth[key]) < 1e-6,
              f"{back[key]:.4f} vs {truth[key]:.4f}")
    check("and the recovered box is inside the image",
          not dx.off_image(back, WIDTH, HEIGHT))


def check_each_wrong_transform_fails_in_its_own_direction():
    """Which wrong transform produces which symptom -- the diagnosis itself.

    A water box at y=820 in a 1920x1080 still, seen under each candidate:

      raw           lands at y=413, a third of the height, and only a third of
                    its true size. High in the frame and small: a small box in
                    the trees. That is the reported symptom exactly, and raw is
                    what boxes_for currently does.
      letterbox_tl  lands at y=1240, BELOW a 1080-pixel image. Forgetting the
                    padding pushes boxes down and off, it does not lift them.

    Getting these two the wrong way round is how a diagnosis goes looking for
    the wrong bug: the first draft of this test asserted that ignoring the
    padding lifted the box into the trees, and the arithmetic says the
    opposite.
    """
    truth_x, truth_y, truth_w = 700.0, 820.0, 260.0
    payload = [forward_letterbox(truth_x, truth_y),
               forward_letterbox(truth_x + truth_w, truth_y + 90.0)]

    naive = dx.transform_box("letterbox_tl", payload, WIDTH, HEIGHT, MODEL)
    check("ignoring the padding pushes the box DOWN by ~420px",
          400 < naive["y"] - truth_y < 440, f"{naive['y'] - truth_y:+.0f}px")
    check("far enough to leave the image entirely",
          naive["y"] > HEIGHT and dx.off_image(naive, WIDTH, HEIGHT),
          f"y {naive['y']:.0f} vs height {HEIGHT}")

    raw = dx.transform_box("raw", payload, WIDTH, HEIGHT, MODEL)
    check("using model coordinates raw puts the box HIGH in the frame",
          raw["y"] < HEIGHT * 0.45, f"y {raw['y']:.0f} of {HEIGHT}")
    check("and shrinks it to about a third of its true width",
          abs(raw["w"] - truth_w / 3) < 3, f"{raw['w']:.0f} vs {truth_w:.0f}")
    check("a small box high in the frame is the reported symptom, and raw is "
          "what the current code does",
          raw["y"] < truth_y - 300 and raw["w"] < truth_w / 2,
          f"y {raw['y']:.0f} w {raw['w']:.0f}")


def check_the_other_candidates_do_what_they_say():
    stretched = [(320.0, 320.0), (480.0, 480.0)]
    box = dx.transform_box("scale", stretched, WIDTH, HEIGHT, MODEL)
    check("scale maps the canvas centre to the image centre",
          abs(box["x"] - WIDTH / 2) < 1e-9 and abs(box["y"] - HEIGHT / 2) < 1e-9,
          f"{box['x']:.0f},{box['y']:.0f}")
    check("and stretches each axis by its own ratio",
          abs(box["w"] - 160 * WIDTH / MODEL) < 1e-9
          and abs(box["h"] - 160 * HEIGHT / MODEL) < 1e-9)

    fractions = [(0.25, 0.5), (0.75, 0.9)]
    box = dx.transform_box("normalized", fractions, WIDTH, HEIGHT, MODEL)
    check("normalized multiplies fractions by the image size",
          abs(box["x"] - 480) < 1e-9 and abs(box["y"] - 540) < 1e-9
          and abs(box["w"] - 960) < 1e-9, str(box))

    centred = [(1000.0, 800.0), (200.0, 100.0)]
    box = dx.transform_box("cxcywh", centred, WIDTH, HEIGHT, MODEL)
    check("cxcywh reads the second point as a size, not a corner",
          box == {"x": 900.0, "y": 750.0, "w": 200.0, "h": 100.0}, str(box))

    corners = [(100.0, 200.0), (300.0, 260.0)]
    box = dx.transform_box("raw", corners, WIDTH, HEIGHT, MODEL)
    check("raw is the identity, in min/max corner form",
          box == {"x": 100.0, "y": 200.0, "w": 200.0, "h": 60.0}, str(box))

    reversed_corners = [(300.0, 260.0), (100.0, 200.0)]
    check("corner order does not matter",
          dx.transform_box("raw", reversed_corners, WIDTH, HEIGHT, MODEL) == box)


def check_off_image_catches_what_it_should():
    check("a box inside the frame is not flagged",
          not dx.off_image({"x": 10, "y": 10, "w": 50, "h": 50}, WIDTH, HEIGHT))
    check("a box running off the right edge is flagged",
          dx.off_image({"x": WIDTH - 10, "y": 10, "w": 50, "h": 50},
                       WIDTH, HEIGHT))
    check("a box above the top edge is flagged",
          dx.off_image({"x": 10, "y": -60, "w": 50, "h": 50}, WIDTH, HEIGHT))


def check_payload_parsing_keeps_the_numbers_untouched():
    record = {"classification_result": {
        "classification_bboxes": [
            [{"x": 1.5, "y": 2.5}, {"x": 3.5, "y": 4.5}],
            [{"x": 9.0, "y": 9.0}],
            []],
        "classification_scores": [{"rip_current": 0.83}, {"rip_current": 0.4}]}}
    boxes = dx.raw_boxes(record)
    check("a well-formed box survives as two points, unreduced",
          boxes == [[(1.5, 2.5), (3.5, 4.5)]], str(boxes))
    check("a one-point box is dropped rather than guessed at", len(boxes) == 1)
    labels = dx.class_labels(record)
    check("class names and scores are read in payload order",
          labels == [("rip_current", 0.83), ("rip_current", 0.4)], str(labels))


def check_the_sheet_embeds_computed_rectangles():
    """The sheet must carry still-pixel numbers, not payload numbers."""
    payload = [forward_letterbox(700.0, 820.0), forward_letterbox(960.0, 910.0)]
    with tempfile.TemporaryDirectory() as folder:
        still = os.path.join(folder, "frame.jpg")
        open(still, "wb").close()
        sample = {"frame_id": "F1", "timestamp": "2026-03-01T12:00:00+00:00",
                  "boxes": [payload], "classes": [("rip_current", 0.8)],
                  "still": still, "image_width": WIDTH, "image_height": HEIGHT,
                  "annotated_url": None}
        path = os.path.join(folder, "sheet.html")
        dx.write_sheet([sample], MODEL, path)
        page = open(path).read()

        check("the sheet was written", os.path.exists(path))
        check("every transform gets a column",
              all(name in page for name, _ in dx.TRANSFORMS))
        check("the class label is printed on the sheet", "rip_current 0.80" in page)
        check("the sheet's javascript does no coordinate maths",
              "Math.min(MODEL" not in page and "naturalWidth" not in page)

        import re
        blobs = re.findall(r"data-rects='([^']+)'", page)
        check("one rect blob per transform", len(blobs) == len(dx.TRANSFORMS),
              str(len(blobs)))
        import html as htmlmod
        first = json.loads(htmlmod.unescape(blobs[0]))
        check("the blob carries the image size for the viewBox",
              first["w"] == WIDTH and first["h"] == HEIGHT)
        letterbox_index = [n for n, _ in dx.TRANSFORMS].index("letterbox")
        rect = json.loads(htmlmod.unescape(blobs[letterbox_index]))["rects"][0]
        check("the letterbox column carries the recovered still-pixel box, "
              "not the payload's numbers",
              abs(rect["x"] - 700.0) < 1e-6 and abs(rect["y"] - 820.0) < 1e-6,
              f"{rect['x']:.2f},{rect['y']:.2f}")


def main():
    print("box coordinate diagnosis offline checks\n")
    check_the_letterbox_inverse_returns_the_box()
    check_each_wrong_transform_fails_in_its_own_direction()
    check_the_other_candidates_do_what_they_say()
    check_off_image_catches_what_it_should()
    check_payload_parsing_keeps_the_numbers_untouched()
    check_the_sheet_embeds_computed_rectangles()
    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
