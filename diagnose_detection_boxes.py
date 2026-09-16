"""Work out what coordinate space WebCOOS rip-detection boxes are actually in.

The symptom: overlaying the payload's boxes on the raw still puts them in the
trees behind the beach. That is the signature of a coordinate space mismatch,
and the usual cause with YOLOv8 is that the coordinates belong to the model's
LETTERBOXED input -- a 640x640 canvas holding the image scaled to fit, with
grey padding on two sides -- rather than to the still itself.

The arithmetic says which mistake produces which symptom, and they differ. For
a 1920x1080 still into a 640 canvas the scale is 0.333 and the padding is 140px
top and bottom. A real box over water at y=820:

    used raw           lands at y=413, a third of the frame height, at a third
                       of its true size -- a SMALL box HIGH in the frame, in
                       the trees above the beach
    padding ignored    lands at y=1240, below the bottom of a 1080px image

So a small box in the trees is the raw-coordinate signature, not the
forgotten-padding one. That matters: they call for different fixes, and
boxes_for currently does the former.

Two things make this diagnosable rather than a guessing game.

First, the numbers usually answer it on their own. If no coordinate in the
whole sample exceeds 640 while the stills are 1920 wide, the boxes are in model
space and no picture is needed. audit() reports that before any image is
fetched.

Second, the payload carries annotated_image_url: WebCOOS's own rendering of the
frame with the boxes already drawn. That is ground truth for where each box
belongs, so the sheet shows it beside every candidate transform rather than
asking which one looks plausible.

    python diagnose_detection_boxes.py --dump-record
    python diagnose_detection_boxes.py --frames 30

Writes data/box_diagnosis/sheet.html. Fetches only the annotated images, and
only for the frames sampled.
"""

import argparse
import glob
import html
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

import analyze_drivers as ad
import build_label_sample as bls
import pull_rip_detection as prd

CAMERA = "Walton Lighthouse, Santa Cruz, CA"
OUT_DIR = f"{ad.DATA_DIR}/box_diagnosis"
ANNOTATED_DIR = f"{OUT_DIR}/annotated"
SHEET = f"{OUT_DIR}/sheet.html"
MODEL_SIDE = 640

# Every transform is expressed the same way: given a box in payload coordinates
# and the still's true size, where does it land in still pixels. The JS in the
# sheet applies these; Python only needs their names and descriptions.
TRANSFORMS = [
    ("raw", "payload coordinates used as-is, as source pixels"),
    ("scale", "payload / model_side * image_side, each axis independently "
              "(a plain resize with no padding)"),
    ("letterbox", "un-letterbox: subtract CENTRED padding, then divide by the "
                  "single aspect-preserving scale (ultralytics default)"),
    ("letterbox_tl", "un-letterbox assuming padding on the top-left only"),
    ("normalized", "payload coordinates are 0..1 fractions of the image"),
    ("cxcywh", "the two points are (centre_x, centre_y) and (width, height), "
               "not two corners"),
]


def letterbox_geometry(width, height, model_side):
    """(scale, pad_x, pad_y) for fitting width x height into a square canvas.

    This is ultralytics' LetterBox: one scale for both axes so the aspect ratio
    survives, and the leftover split evenly between the two sides. A 1920x1080
    still into 640 gives scale 0.333, pad_x 0, pad_y 140 -- which is why a box
    near the top of the model canvas lands above the horizon when the padding
    is not removed first.
    """
    scale = min(model_side / width, model_side / height)
    return scale, (model_side - width * scale) / 2, (model_side - height * scale) / 2


def transform_point(name, x, y, width, height, model_side):
    """One payload point mapped into still pixels under a candidate transform."""
    if name == "raw":
        return x, y
    if name == "scale":
        return x * width / model_side, y * height / model_side
    if name == "letterbox":
        scale, pad_x, pad_y = letterbox_geometry(width, height, model_side)
        return (x - pad_x) / scale, (y - pad_y) / scale
    if name == "letterbox_tl":
        scale, _, _ = letterbox_geometry(width, height, model_side)
        return x / scale, y / scale
    if name == "normalized":
        return x * width, y * height
    if name == "cxcywh":
        return x, y
    raise ValueError(f"unknown transform {name!r}")


def transform_box(name, points, width, height, model_side):
    """A payload box as {x, y, w, h} in still pixels under one transform.

    Kept in Python rather than in the sheet's JavaScript so that the audit and
    the picture cannot disagree: one implementation, one set of numbers, and a
    test that checks it against a letterbox applied in the forward direction.
    """
    (x1, y1), (x2, y2) = points[0], points[1]
    if name == "cxcywh":
        return {"x": x1 - x2 / 2, "y": y1 - y2 / 2, "w": x2, "h": y2}
    ax, ay = transform_point(name, x1, y1, width, height, model_side)
    bx, by = transform_point(name, x2, y2, width, height, model_side)
    return {"x": min(ax, bx), "y": min(ay, by),
            "w": abs(bx - ax), "h": abs(by - ay)}


def off_image(rect, width, height, slack=2.0):
    return (rect["x"] < -slack or rect["y"] < -slack
            or rect["x"] + rect["w"] > width + slack
            or rect["y"] + rect["h"] > height + slack)


def raw_boxes(record):
    """Boxes exactly as the payload gives them: a list of point pairs.

    Deliberately NOT the min/max corner reduction build_label_sample does. The
    whole question is what these numbers mean, so nothing is normalised away
    before it has been answered.
    """
    result = record.get("classification_result") or {}
    out = []
    for box in result.get("classification_bboxes") or []:
        points = [(p.get("x"), p.get("y")) for p in box or []
                  if isinstance(p, dict)]
        points = [(float(x), float(y)) for x, y in points
                  if x is not None and y is not None]
        if len(points) >= 2:
            out.append(points[:2])
    return out


def class_labels(record):
    """[(name, score)] for this record, in payload order."""
    result = record.get("classification_result") or {}
    out = []
    for entry in result.get("classification_scores") or []:
        if isinstance(entry, dict):
            for name, score in entry.items():
                try:
                    out.append((str(name), float(score)))
                except (TypeError, ValueError):
                    out.append((str(name), float("nan")))
        else:
            out.append((str(entry), float("nan")))
    return out


def audit(samples, model_side):
    """Print what the coordinates alone say, before any image is looked at.

    This is the part that can end the investigation outright. A payload whose
    coordinates never exceed the model side, on stills several times wider, is
    not ambiguous.
    """
    xs, ys = [], []
    for sample in samples:
        for points in sample["boxes"]:
            for x, y in points:
                xs.append(x)
                ys.append(y)
    if not xs:
        print("  no boxes in the sample; nothing to audit")
        return {}

    xs, ys = np.array(xs), np.array(ys)
    widths = [s["image_width"] for s in samples if s.get("image_width")]
    heights = [s["image_height"] for s in samples if s.get("image_height")]

    print(f"\n  coordinate audit over {len(xs)} points from "
          f"{sum(len(s['boxes']) for s in samples)} boxes")
    print(f"    x   min {xs.min():9.3f}   max {xs.max():9.3f}")
    print(f"    y   min {ys.min():9.3f}   max {ys.max():9.3f}")
    if widths:
        print(f"    still size: {int(np.median(widths))} x "
              f"{int(np.median(heights))} (median of {len(widths)} read)")

    verdicts = []
    width = float(np.median(widths)) if widths else None
    height = float(np.median(heights)) if heights else None

    # Decide model-space vs source-pixel space FIRST, and say so plainly. The
    # earlier version ran a letterbox check regardless and, on coordinates that
    # obviously exceeded the model canvas, printed a verdict about padding --
    # advice about a hypothesis the very first number had already ruled out.
    in_model_space = xs.max() <= model_side + 1 and ys.max() <= model_side + 1

    if xs.max() <= 1.0 and ys.max() <= 1.0:
        verdicts.append("every coordinate is <= 1.0: these are NORMALIZED "
                        "fractions. Use the 'normalized' transform.")
    elif in_model_space and width and width > model_side * 1.2:
        verdicts.append(
            f"nothing exceeds {model_side} while the stills are {int(width)} "
            "wide: the boxes are in MODEL space and must be un-letterboxed.")
    elif width and height:
        fills_x = xs.max() / width
        fills_y = ys.max() / height
        verdicts.append(
            f"coordinates reach {xs.max():.0f} x {ys.max():.0f} against a still "
            f"of {int(width)} x {int(height)} -- {fills_x:.0%} and {fills_y:.0%} "
            "of it. They are SOURCE PIXELS already.")
        verdicts.append(
            f"that rules out the {model_side}x{model_side} letterbox entirely: "
            "model-space coordinates cannot exceed the canvas. No un-letterbox "
            "transform applies, and 'raw' is the right reading of the numbers.")
        verdicts.append(
            "so if the boxes still land in the wrong place, the coordinates "
            "are not the problem -- the IMAGE underneath them is. Check the "
            "still-vs-detection offset below.")

    if width and xs.max() > width * 1.02:
        verdicts.append("some x exceeds the still width, so the still being "
                        "drawn on is not the image the detector saw.")

    # Only meaningful if the coordinates could be in the canvas at all.
    if in_model_space and width and height:
        ratio = min(model_side / width, model_side / height)
        pad_y = (model_side - height * ratio) / 2
        if pad_y > 2:
            inside = float(((ys >= pad_y - 1)
                            & (ys <= model_side - pad_y + 1)).mean())
            print(f"    letterbox check: centred padding would be {pad_y:.1f}px "
                  f"top and bottom;\n      {inside:.1%} of y values fall inside "
                  "that band")
            if inside > 0.98:
                verdicts.append("every y sits inside the band a centred "
                                "letterbox would leave free: 'letterbox' is "
                                "the leading candidate.")
            elif inside < 0.8:
                verdicts.append(
                    f"{1 - inside:.0%} of y values fall in what would be the "
                    "padding, so a CENTRED letterbox does not fit. Try "
                    "'letterbox_tl'.")

    print("\n  what the numbers say:")
    for line in verdicts or ["nothing conclusive; the sheet will have to "
                             "settle it by eye."]:
        print(f"    - {line}")
    return {"x_max": float(xs.max()), "y_max": float(ys.max())}


STAMP_PATTERN = re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})"
                           r"[T_-]?(\d{2})[-:_]?(\d{2})[-:_]?(\d{2})")


def stamp_from_name(name):
    """The capture time encoded in a WebCOOS still's filename, or None."""
    match = STAMP_PATTERN.search(os.path.basename(str(name or "")))
    if not match:
        return None
    try:
        return pd.Timestamp("{}-{}-{}T{}:{}:{}Z".format(*match.groups()))
    except ValueError:
        return None


def audit_classes(samples, slug):
    """What the detector says these boxes ARE.

    Asked because boxes landing squarely on people standing on the beach is not
    a misalignment: it is a correctly placed box around a correctly detected
    object of the wrong kind. The class name is carried in the payload, so this
    does not need to be inferred from where the boxes sit.

    The host serving the annotated frames is
    stage-webcoos-object-detector-api..., under /outputs/yolo/v8n/ -- v8n is
    YOLOv8-nano, whose stock COCO weights detect eighty everyday classes with
    'person' first among them and no rip current anywhere. If the class names
    below are COCO names, this product is a general object detector and the
    whole rip record is a record of something else. That is worth knowing
    before another number is computed from it.
    """
    from collections import Counter
    counts = Counter()
    for sample in samples:
        for name, _ in sample["classes"]:
            counts[name] += 1

    print(f"\n  classes in the sampled payloads")
    if not counts:
        print("    no class names in the sample")
    for name, count in counts.most_common(12):
        print(f"    {name:<28} {count:>6}")

    # The sample is 30 frames; the whole record is the real answer.
    frame = ad.read_csv(f"{ad.DATA_DIR}/rip_detection/rip_{slug}.csv")
    if frame is not None and "score_classes" in frame.columns:
        whole = Counter()
        for value in frame["score_classes"].dropna():
            for name in str(value).split(","):
                if name.strip():
                    whole[name.strip()] += 1
        print(f"\n  classes across the whole rip record ({len(frame)} frames)")
        for name, count in whole.most_common(12):
            print(f"    {name:<28} {count:>6}  {count / max(len(frame), 1):>6.1%}")
        counts = whole

    rip_like = {n for n in counts if "rip" in n.lower() or "current" in n.lower()}
    coco_like = {n for n in counts
                 if n.lower() in {"person", "car", "boat", "bird", "dog",
                                  "surfboard", "umbrella", "kite", "bench",
                                  "truck", "backpack", "chair", "frisbee"}}
    print()
    if coco_like and not rip_like:
        print(f"    THE BOXES ARE NOT RIPS. Every class here is a COCO object "
              f"class ({', '.join(sorted(coco_like))}),\n    and no class "
              "mentions a rip or a current. This product is a general object\n"
              "    detector, not a rip detector, and every result computed "
              "from it is about\n    something other than rip currents.")
    elif rip_like and coco_like:
        print(f"    MIXED: rip-like classes ({', '.join(sorted(rip_like))}) "
              f"alongside object classes\n    ({', '.join(sorted(coco_like))}). "
              "The record needs filtering by class before use.")
    elif rip_like:
        print(f"    classes are rip-like ({', '.join(sorted(rip_like))}); the "
              "product is what it claims.")
    else:
        print("    class names are unfamiliar; read them above and decide.")
    return counts


def audit_image_timing(samples):
    """How far each labelled still is from the detection it carries boxes for.

    This is the check that would have ended the investigation on the first run.
    Coordinates can be perfectly correct and the overlay still land in the
    trees, if the photograph under them was taken half an hour later. The still
    filename carries its own capture time, so the comparison needs nothing but
    what is already on disk.
    """
    offsets, unparsed = [], 0
    for sample in samples:
        shot = stamp_from_name(sample.get("still"))
        if shot is None:
            unparsed += 1
            continue
        detected = pd.Timestamp(sample["timestamp"])
        offsets.append(abs((shot - detected).total_seconds()))

    print(f"\n  still-vs-detection timing over {len(samples)} frames")
    if unparsed:
        print(f"    {unparsed} filename(s) carry no readable timestamp")
    if not offsets:
        print("    no still filename could be timed; cannot check this here")
        return None
    series = pd.Series(offsets)
    print(f"    median {series.median():>8.0f}s   90th pct "
          f"{series.quantile(0.9):>8.0f}s   worst {series.max():>8.0f}s")
    if series.median() > 120:
        print("    THIS IS THE BUG. The stills carrying these boxes were taken "
              "minutes\n    away from the detections that produced them. On a "
              "moving sea the water\n    under a box is simply not the water "
              "the detector saw. Rebuild the sample.")
    elif series.median() > 5:
        print("    stills are close but not exact; a second or two of swell "
              "moves a rip.")
    else:
        print("    stills match their detections; the image is not the "
              "problem.")
    return float(series.median())


def image_size(path):
    """(width, height) of a local still, or (None, None) without Pillow."""
    try:
        from PIL import Image
    except ImportError:
        return None, None
    try:
        with Image.open(path) as handle:
            return handle.size
    except Exception:  # noqa: BLE001 -- a bad file must not stop the audit
        return None, None


def collect(slug, limit, prefer_local):
    """Sample frames that HAVE boxes, with their payload records attached.

    Frames already downloaded for the labelling sample are preferred, so the
    sheet needs no stills fetched -- only the annotated images, which are the
    thing being compared against.
    """
    index = bls.source_index(slug)
    if not index:
        sys.exit(f"No data/rip_detection/rip_{slug}_index.csv — cannot locate "
                 "the raw payloads.")

    frame = ad.read_csv(f"{ad.DATA_DIR}/rip_detection/rip_{slug}.csv")
    if frame is None or frame.empty:
        sys.exit(f"No data/rip_detection/rip_{slug}.csv")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True,
                                        errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame = frame[pd.to_numeric(frame.get("bbox_count"), errors="coerce") > 0]
    if frame.empty:
        sys.exit("No frames with boxes in the rip record.")

    local = {}
    if prefer_local:
        labels = ad.read_csv(bls.LABEL_CSV)
        if labels is not None and not labels.empty:
            for _, row in labels.iterrows():
                path = os.path.join(bls.IMAGE_DIR, str(row.get("image") or ""))
                if os.path.isfile(path):
                    local[str(row.get("frame_id"))] = path

    frame["frame_id"] = frame["timestamp"].dt.strftime("%Y%m%dT%H%M%SZ")
    frame["has_local"] = frame["frame_id"].isin(local)
    frame = frame.sort_values(["has_local", "timestamp"], ascending=[False, True])

    samples = []
    for _, row in frame.iterrows():
        if len(samples) >= limit:
            break
        path = index.get(os.path.basename(str(row.get("source_file") or "")))
        if not path or not os.path.exists(path):
            continue
        try:
            records = prd.read_records(path) or []
        except Exception:  # noqa: BLE001
            continue
        wanted = row.get("original_image")
        match = None
        if isinstance(wanted, str) and wanted:
            match = next((r for r in records
                          if r.get("original_image_reference") == wanted), None)
        elif len(records) == 1:
            match = records[0]
        if match is None:
            continue
        boxes = raw_boxes(match)
        if not boxes:
            continue
        still = local.get(row["frame_id"])
        width, height = image_size(still) if still else (None, None)
        samples.append({
            "frame_id": row["frame_id"],
            "timestamp": row["timestamp"].isoformat(),
            "boxes": boxes,
            "classes": class_labels(match),
            "still": still,
            "image_width": width,
            "image_height": height,
            "annotated_url": match.get("annotated_image_url"),
            "record": match,
        })
    return samples


def fetch_annotated(samples):
    """Download each frame's WebCOOS-annotated image. Cached, best effort."""
    os.makedirs(ANNOTATED_DIR, exist_ok=True)
    got = 0
    for sample in samples:
        url = sample.get("annotated_url")
        if not isinstance(url, str) or not url.startswith("http"):
            continue
        name = f"{sample['frame_id']}.jpg"
        target = os.path.join(ANNOTATED_DIR, name)
        if not os.path.exists(target):
            try:
                rows = prd.download([{"url": url, "filename": name,
                                      "timestamp": sample["timestamp"]}],
                                    ANNOTATED_DIR)
                time.sleep(0.2)
                if not rows:
                    continue
            except Exception as exc:  # noqa: BLE001
                print(f"    {sample['frame_id']}: annotated fetch failed ({exc})")
                continue
        if os.path.exists(target):
            sample["annotated"] = target
            got += 1
    return got


SHEET_JS = """
// No coordinate maths here on purpose. Every rectangle was computed in Python
// by transform_box() and embedded as still-pixel numbers, so the sheet and the
// audit cannot drift apart. This only draws what it is given.
for(const cell of document.querySelectorAll(".cell")){
  const svg = cell.querySelector("svg");
  const data = JSON.parse(cell.dataset.rects);
  svg.setAttribute("viewBox", "0 0 " + data.w + " " + data.h);
  const stroke = Math.max(2, Math.round(data.w / 320));
  for(const r of data.rects){
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", r.x); rect.setAttribute("y", r.y);
    rect.setAttribute("width", Math.max(r.w, 1));
    rect.setAttribute("height", Math.max(r.h, 1));
    rect.setAttribute("fill", "none");
    rect.setAttribute("stroke", "#39d98a");
    rect.setAttribute("stroke-width", stroke);
    svg.appendChild(rect);
  }
}
"""


def write_sheet(samples, model_side, path):
    columns = [name for name, _ in TRANSFORMS]
    parts = ["<!doctype html><meta charset='utf-8'>",
             "<title>Rip box coordinate diagnosis</title>",
             """<style>
 body{background:#11171c;color:#e6eef2;font:14px/1.5 system-ui,sans-serif;margin:0;padding:16px}
 h1{font-size:18px;margin:0 0 4px} p{color:#93a6b0;margin:0 0 16px;max-width:70em}
 table{border-collapse:collapse;width:100%} th{font:12px ui-monospace,monospace;
   text-align:left;padding:6px 8px;color:#bcd;border-bottom:1px solid #2b3942;vertical-align:top}
 th small{display:block;color:#7f929c;font-weight:400;max-width:22em}
 td{padding:6px 8px;border-bottom:1px solid #1d272e;vertical-align:top}
 .cell{position:relative;line-height:0;background:#000;border:1px solid #2b3942;border-radius:3px}
 .cell img{width:100%;height:auto;display:block} .cell svg{position:absolute;inset:0;width:100%;height:100%}
 .warn{position:absolute;left:4px;top:4px;font:10px ui-monospace,monospace;
   color:#ff8a7a;background:#000a;padding:1px 4px;border-radius:2px;line-height:1.4}
 .idc{font:11px ui-monospace,monospace;color:#93a6b0;white-space:nowrap}
 .idc b{color:#e6eef2;display:block} .cls{color:#8fe3b4}
 .truth{outline:2px solid #b08a3c}
</style>"""]
    parts.append("<h1>Rip box coordinate diagnosis</h1>")
    parts.append(
        "<p>Each row is one frame. The first image column is WebCOOS's own "
        "annotated frame &mdash; that is ground truth for where the boxes "
        "belong. The remaining columns draw the payload's raw numbers on the "
        "plain still under each candidate transform. Pick the column that "
        "matches the annotated image. A red tag marks boxes falling outside "
        f"the image under that transform. Model input assumed {model_side}"
        f"&times;{model_side}.</p>")

    parts.append("<table><tr><th>frame</th><th class='truth'>WebCOOS annotated"
                 "<small>ground truth</small></th>")
    for name, blurb in TRANSFORMS:
        parts.append(f"<th>{html.escape(name)}<small>{html.escape(blurb)}</small></th>")
    parts.append("</tr>")

    for sample in samples:
        classes = ", ".join(f"{html.escape(n)} {s:.2f}" if s == s
                            else html.escape(n)
                            for n, s in sample["classes"]) or "—"
        parts.append("<tr>")
        parts.append(
            f"<td class='idc'><b>{html.escape(sample['frame_id'])}</b>"
            f"{html.escape(sample['timestamp'][:19])}<br>"
            f"{len(sample['boxes'])} box(es)<br>"
            f"<span class='cls'>{classes}</span></td>")

        annotated = sample.get("annotated")
        if annotated:
            rel = os.path.relpath(annotated, os.path.dirname(path))
            parts.append(f"<td style='min-width:220px'><div class='cell truth'>"
                         f"<img src='{html.escape(rel)}' alt=''></div></td>")
        else:
            parts.append("<td class='idc'>no annotated image</td>")

        still = sample.get("still")
        if not still or not sample.get("image_width"):
            parts.append(f"<td class='idc' colspan='{len(columns)}'>"
                         "no local still, or Pillow is not installed so its "
                         "size could not be read</td></tr>")
            continue
        rel = os.path.relpath(still, os.path.dirname(path))
        width = sample["image_width"]
        height = sample["image_height"]
        for name in columns:
            rects = [transform_box(name, points, width, height, model_side)
                     for points in sample["boxes"]]
            outside = sum(1 for r in rects if off_image(r, width, height))
            data = html.escape(json.dumps({"w": width, "h": height,
                                           "rects": rects}))
            warn = (f"{outside} box{'es' if outside > 1 else ''} off-image"
                    if outside else "")
            parts.append(
                f"<td style='min-width:220px'><div class='cell' "
                f"data-rects='{data}'>"
                f"<img src='{html.escape(rel)}' alt=''>"
                f"<svg preserveAspectRatio='none'></svg>"
                f"<span class='warn'>{warn}</span></div></td>")
        parts.append("</tr>")
    parts.append("</table>")
    parts.append("<script>" + SHEET_JS.replace("__MODEL__", str(model_side))
                 + "</script>")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default=CAMERA)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--model-side", type=int, default=MODEL_SIDE)
    parser.add_argument("--dump-record", action="store_true",
                        help="print one raw payload record and stop")
    parser.add_argument("--no-annotated", action="store_true",
                        help="skip fetching WebCOOS's annotated frames")
    args = parser.parse_args()

    slug = ad.rip_slug(args.camera)
    samples = collect(slug, args.frames, prefer_local=True)
    if not samples:
        sys.exit("No frames with both a payload record and boxes were found.")

    if args.dump_record:
        print("\none full payload record, verbatim:\n")
        print(json.dumps(samples[0]["record"], indent=2, default=str)[:6000])
        return 0

    print(f"\n{args.camera}")
    print(f"  {len(samples)} frames with boxes "
          f"({sum(1 for s in samples if s['still'])} have a local still)")
    audit(samples, args.model_side)
    audit_classes(samples, slug)
    audit_image_timing(samples)

    if not args.no_annotated:
        print(f"\n  fetching WebCOOS annotated frames "
              f"({sum(1 for s in samples if s.get('annotated_url'))} have a url)")
        got = fetch_annotated(samples)
        print(f"  {got} annotated frames available as ground truth")

    write_sheet(samples, args.model_side, SHEET)
    print(f"\n  wrote {SHEET}")
    print(f"  open it with:  open {SHEET}")
    print("\n  Pick the column whose boxes match the annotated frame, and tell "
          "me which.\n  I will make that transform the one boxes_for applies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
