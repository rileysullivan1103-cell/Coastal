"""A blinded 60-frame calibration set from RipAID, for the existing label page.

OPEN #0. Hand labelling put the WebCOOS detector's precision at 2.7%, and
nothing in that exercise separates two readings: the detector is near-useless,
or a rip is rarely identifiable in a SINGLE STILL by ONE observer. RipAID
(Zenodo 15082427) settles it, because its frames carry rips drawn by people who
had the whole record and a protocol, and 948 of them carry nothing at all.

Labelling 60 of those blind and comparing to the annotators measures the
INSTRUMENT -- one person, one still -- rather than the detector. Agreement says
the 2.0% base rate at Walton is real and the detector is bad. Disagreement says
the labelling is the limiting factor and the detector has not been fairly
tested.

Three things this has to get right or the comparison is worthless:

  * NOTHING ABOUT THE ANSWER MAY REACH THE PAGE. No annotation boxes, no class
    counts, no stratum, no filename that carries the original name. The images
    are copied under opaque ids and the answer key is written to a DIFFERENT
    directory, outside the one the server can serve from.
  * THE ORDER MUST NOT CARRY IT EITHER. 30 positives and 30 negatives drawn
    separately and then shuffled together under a seeded RNG, so the sequence
    is reproducible but uninformative.
  * DOUBT FRAMES ARE NEITHER. A frame the annotators marked `doubt` is not a
    clean positive and not a clean negative, so it cannot be scored either way
    and is excluded from both strata. The count dropped is printed.

Per-box labelling is inert here, deliberately: drawing the boxes would be
showing the ground truth. The Y/N/D/U buttons and the notes field are the
instrument, and the notes are what make the misses readable afterwards.

    python build_ripaid_calibration.py instances_default.json --images ripaid/images
    python build_ripaid_calibration.py instances_default.json --images ripaid/images --n 40
"""

import argparse
import os
import shutil
import sys

import pandas as pd

import load_ripaid as lr

OUT_DIR = "data/ripaid_calibration"
KEY_DIR = "data/ripaid_calibration_key"
LABEL_CSV = f"{OUT_DIR}/labels.csv"
IMAGE_DIR = f"{OUT_DIR}/images"
TRUTH_CSV = f"{KEY_DIR}/truth.csv"

COLUMNS = ["frame_id", "image", "display_meta", "rip_present", "notes",
           "box_labels", "labeled_at"]


def clean_strata(frames):
    """(positives, negatives, report) with every doubt frame removed.

    A doubt annotation is the annotators' own recorded uncertainty. A frame
    carrying one alongside a rip is a weaker positive than one without; a frame
    carrying one and nothing else is not a negative at all, because a person
    looked and could not say. Neither belongs in a two-way comparison, so both
    go, and how many went is printed rather than absorbed.
    """
    doubt = frames["n_doubt"] > 0
    # RipAID v2.0.0 added `sediment` -- "a sediment plume that might relate to
    # a rip current". Like doubt it is neither: a frame carrying one is not a
    # clean positive, and one carrying nothing else is not a person having
    # looked and seen no rip. v1.0.0 has no such column, hence the default.
    sediment = (frames["n_sediment"] > 0 if "n_sediment" in frames.columns
                else pd.Series(False, index=frames.index))
    murky = doubt | sediment
    clean = frames[~murky]
    positives = clean[clean["n_rip"] > 0]
    negatives = clean[clean["n_rip"] == 0]
    report = {
        "total": len(frames),
        "doubt_frames": int(doubt.sum()),
        "doubt_with_rip": int((doubt & (frames["n_rip"] > 0)).sum()),
        "doubt_only": int((doubt & (frames["n_rip"] == 0)).sum()),
        "sediment_frames": int(sediment.sum()),
        "murky": int(murky.sum()),
        "positives": len(positives),
        "negatives": len(negatives),
    }
    return positives, negatives, report


def resolve_image(image_dir, file_name):
    """Where this frame's image actually is, or None.

    A CVAT COCO export names frames either "clm_s_01_....png" or
    "default/clm_s_01_....png" depending on how it was produced, and unpacks
    them under images/default/. So --images can reasonably be pointed at either
    images/ or images/default/ and only one of the four combinations lines up.
    Rather than make that a guess with a silent wrong answer at the end of it,
    both spellings are tried and the one on disk wins.
    """
    if not image_dir:
        return None
    for candidate in (file_name, os.path.basename(file_name)):
        full = os.path.join(image_dir, candidate)
        if os.path.isfile(full):
            return full
    return None


def with_images(pool, image_dir):
    """Only frames whose image is actually on disk.

    Filtered BEFORE the draw, not after. Drawing thirty and then discovering
    four are missing leaves a sample of twenty-six that is no longer balanced,
    and the imbalance would be silent.
    """
    if not image_dir:
        return pool, 0
    exists = pool["file_name"].map(
        lambda name: resolve_image(image_dir, name) is not None)
    return pool[exists], int((~exists).sum())


def draw(positives, negatives, per_stratum, seed):
    if len(positives) < per_stratum or len(negatives) < per_stratum:
        sys.exit(f"  need {per_stratum} of each and have "
                 f"{len(positives)} positives, {len(negatives)} negatives")
    pos = positives.sample(per_stratum, random_state=seed)
    neg = negatives.sample(per_stratum, random_state=seed + 1)
    both = pd.concat([pos, neg], ignore_index=True)
    # Shuffled AFTER concatenation, so position in the list says nothing.
    return both.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def write_sample(sample, image_dir, out_dir=OUT_DIR, key_dir=KEY_DIR):
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    os.makedirs(key_dir, exist_ok=True)
    rows, truth = [], []
    for position, row in enumerate(sample.itertuples()):
        extension = os.path.splitext(row.file_name)[1].lower() or ".png"
        frame_id = f"cal_{position:03d}"
        image = f"{frame_id}{extension}"
        if image_dir:
            source = resolve_image(image_dir, row.file_name)
            if source is None:
                sys.exit(f"  {row.file_name} vanished between the check and "
                         "the copy; re-run.")
            shutil.copyfile(source, os.path.join(out_dir, "images", image))
        rows.append({"frame_id": frame_id, "image": image,
                     # The ONLY thing the page may show. Not the camera, not
                     # the timestamp: a labeller who learns that clm_c05 in
                     # July is usually a rip has stopped being blind.
                     "display_meta": f"frame {position + 1} of {len(sample)}",
                     "rip_present": "", "notes": "", "box_labels": "",
                     "labeled_at": ""})
        truth.append({"frame_id": frame_id, "file_name": row.file_name,
                      "site": row.site, "camera": row.camera,
                      "timestamp": row.timestamp, "n_rip": row.n_rip,
                      "n_doubt": row.n_doubt,
                      "truth": "yes" if row.n_rip > 0 else "no"})

    pd.DataFrame(rows, columns=COLUMNS).to_csv(
        os.path.join(out_dir, "labels.csv"), index=False)
    pd.DataFrame(truth).to_csv(os.path.join(key_dir, "truth.csv"), index=False)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("annotations",
                        help="RipAID COCO export (instances_default.json) or a "
                             "YOLO-OBB dataset root / labels directory")
    parser.add_argument("--images", default=None,
                        help="directory holding the RipAID image files")
    parser.add_argument("--n", type=int, default=30,
                        help="frames per stratum (default 30, so 60 in total)")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing calibration sample")
    args = parser.parse_args()

    if os.path.exists(LABEL_CSV) and not args.force:
        existing = pd.read_csv(LABEL_CSV)
        done = int(existing["rip_present"].fillna("").astype(str).str.strip().ne("").sum())
        sys.exit(f"{LABEL_CSV} already exists with {done} of {len(existing)} "
                 "labelled.\n  Rebuilding would discard them. Pass --force if "
                 "that is what you want.")

    if not os.path.exists(args.annotations):
        sys.exit(f"{args.annotations} does not exist.\n"
                 "  RipAID is not in this repo and nothing here can fetch it "
                 "— see probe_rip_dataset.py.\n"
                 "  Download https://zenodo.org/records/15082427 and pass its "
                 "COCO export:\n"
                 "    python build_ripaid_calibration.py "
                 "/path/to/instances_default.json --images /path/to/images\n"
                 "  To find it if it is already on disk:\n"
                 "    find ~ -name 'instances_default.json' -not -path '*/.*' "
                 "2>/dev/null")
    if args.images and not os.path.isdir(args.images):
        sys.exit(f"--images {args.images} is not a directory.\n"
                 "  Point it at the folder holding the RipAID frame images. "
                 "Omit it entirely\n  to write the CSV without copying images "
                 "(the page will show blanks).")

    frames = lr.frames_from(args.annotations)
    positives, negatives, report = clean_strata(frames)

    print(f"\n{'=' * 74}\nBLINDED CALIBRATION SAMPLE\n{'=' * 74}")
    print(f"  {report['total']} frames in the export")
    print(f"  {report['doubt_frames']} carry a doubt annotation and are excluded:")
    print(f"    {report['doubt_with_rip']} alongside a rip — a weaker positive "
          "than one without")
    print(f"    {report['doubt_only']} with nothing else — a person looked and "
          "could not say,\n      which is not a negative")
    if report.get("sediment_frames"):
        print(f"  {report['sediment_frames']} carry a sediment plume, also "
              "excluded: a plume that MIGHT\n    relate to a rip is neither a "
              "clean positive nor a clean negative")
    print(f"  {report['murky']} frames excluded in total")
    print(f"  leaves {report['positives']} clean positives and "
          f"{report['negatives']} clean negatives")

    if args.images:
        positives, missing_pos = with_images(positives, args.images)
        negatives, missing_neg = with_images(negatives, args.images)
        if not len(positives) and not len(negatives):
            sys.exit(f"  NONE of the frames were found under {args.images}/.\n"
                     "  A CVAT export unpacks images under images/default/, so "
                     "try both:\n"
                     f"    --images {args.images.rstrip('/')}/default\n"
                     f"    --images {os.path.dirname(args.images.rstrip('/')) or '.'}")
        if missing_pos or missing_neg:
            print(f"  {missing_pos + missing_neg} frame(s) have no image in "
                  f"{args.images}/ and were removed BEFORE the draw")
            print(f"  usable: {len(positives)} positives, {len(negatives)} negatives")
    else:
        print("\n  NO --images GIVEN. The CSV will be written and the page will "
              "show\n  broken images. Re-run with --images pointing at the "
              "RipAID frames.")

    sample = draw(positives, negatives, args.n, args.seed)
    count = write_sample(sample, args.images)

    print(f"\n  wrote {LABEL_CSV} ({count} rows, none labelled)")
    print(f"  wrote {IMAGE_DIR}/ ({count} images under opaque ids)")
    print(f"  wrote {TRUTH_CSV}  <- THE ANSWER KEY. Do not open it.")
    print(f"\n  The key is in {KEY_DIR}/, not {OUT_DIR}/, so the server cannot "
          "serve it\n  and a directory listing while labelling cannot show it.")
    print("\n  Start labelling with:")
    print(f"    python label_server.py --dir {OUT_DIR} "
          "--title \"Calibration — is there a rip?\"")
    print("\n  Then, when all 60 carry a verdict:")
    print("    python analyze_ripaid_calibration.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
