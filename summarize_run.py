"""Pull the decisive lines out of a geometry or mask-audit run.

    python3 check_camera_geometry.py ... 2>&1 | tee run.log
    python3 summarize_run.py run.log

Both tools print their reasoning as well as their numbers, on purpose: a
number with no argument attached to it is how the first three Walton verdicts
got believed. That makes a full run several hundred lines, most of which is
static prose that says the same thing every time.

This keeps the lines that DIFFER between runs -- counts, thresholds, fits,
verdicts and warnings -- and drops the standing explanation. It is a reading
aid and nothing else: it does not interpret, average, or decide anything, and
anything it is unsure about it keeps. If a run says something this script has
no pattern for, that text survives into the OTHER LINES WORTH A LOOK section
rather than vanishing, because a summary that silently swallows a new warning
is worse than no summary.
"""

import re
import sys

# (pattern, extra lines to carry with it). The extra count exists because both
# tools wrap a single print across several lines, and the number is often on
# the continuation: "the intrusion runs 0.414 of the frame height INTO the
# land" is the second line of its print.
ANCHORS = [
    # --- check_camera_geometry.py, whole-frame pass -----------------------
    (r"registered \d+ of \d+ frames", 0),
    (r"median offset across the record", 0),
    (r"largest single-date offset", 0),
    (r"frames are a DIFFERENT SIZE", 2),
    (r"consecutive pairs span a frame size change", 1),
    (r"frames had their tallest peak beyond", 2),
    (r"WARNING: that is \d+% of the frames", 5),
    (r"frames below confidence", 0),
    (r"frame pairs measured BOTH ways", 1),
    (r"the two routes differ by a median of", 0),
    (r"largest single frame-to-frame step", 0),
    (r"peaks are searched within", 1),
    (r"two routes over the same pair disagree", 1),
    (r"SMALLEST MOVE THIS RECORD CAN RESOLVE", 0),
    (r"step threshold is below that floor", 1),
    (r"is real motion, not measurement noise", 0),
    (r"Only \d+% of the sampled frames registered", 0),
    (r"TOO FEW PAIRS WERE MEASURED", 2),
    # --- patch pass -------------------------------------------------------
    (r"agreement tolerance is below", 1),
    (r"agreement between candidates", 0),        # table follows, see TABLES
    (r"No \d+ of \d+ patches agree", 1),
    (r"features kept:", 0),
    (r"typical disagreement between features", 0),
    (r"KEPT FEATURES STILL DISAGREE", 0),
    (r"WARNING: every kept patch sits within", 2),
    (r"NOTE: the record moves further than half a", 2),
    (r"No epoch split can be claimed", 0),
    # --- verdict ----------------------------------------------------------
    (r"No step larger than .* persists in", 0),
    (r"So the record is ONE epoch AT THIS RESOLUTION", 1),
    (r"is neither found nor ruled out", 1),
    (r"The record reads as ONE geometric epoch", 0),
    (r"candidate discontinuit", 0),
    (r"^\s+\d{4}-\d{2}-\d{2}\s+[-\d.]+ px ->", 0),
    (r"^\d+ epochs:", 0),
    (r"^\s+\d+\. \d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}", 0),
    (r"A step marked above rests on the fewest frames", 0),
    # --- check_geometry_strict.py, --mask-audit ---------------------------
    (r"\d+ frames judged", 0),
    (r"frames \(\d+%\) carried no colour to judge", 0),
    (r"frames unreadable or a different size, skipped", 0),
    (r"looked like water in >=", 1),
    (r"the mask's seaward edge is row", 0),
    (r"NO row of the mask is", 2),
    (r"habitual water reaches row", 1),
    (r"deepest single wet pixel", 1),
    (r"what those pixels ARE, averaged over the record", 1),
    (r"That is (COOL|WARM|near-black|neither)", 3),
    (r"per-frame share of the mask", 1),
    (r"no frame in the record put water inside this mask", 0),
    (r"^\s+\d{4}-\d{2}-\d{2}\s+[\d.]+%", 0),
    (r"WHERE WATER ACTUALLY REACHES", 2),
    (r"THIS IS NOT AN EDGE", 0),
    (r"KEEP THE DECLARED EDGE", 0),
    (r"WARNING: the TYPICAL frame has over a third", 1),
    (r"NOTE: some frames read mostly wet", 2),
    # --- what was written, so the right file gets attached ----------------
    (r"^IMAGE WRITTENS?$", 0),
    (r"^\s+/.*\.(jpg|png|csv)$", 0),
    (r"every sampled frame, labelled:", 0),
]

# Blocks whose rows are worth keeping whole: an anchor, then every following
# line that still matches the row shape.
TABLES = [
    (r"agreement between candidates", r"^\s+(keep|drop)\s"),
    (r"WHERE THE RECORD REGISTERS", r"^\s+(quarter|\S+\s+\d+\s)"),
    (r"largest single frame-to-frame step", r"^\s+\d{4}-\d{2}-\d{2}\s+[-\d.]+ px"),
    (r"the six worst frames", r"^\s+\d{4}-\d{2}-\d{2}\s"),
]

# Anything shouting is kept whether or not a pattern above knows about it.
LOUD = re.compile(r"\b(WARNING|ERROR|REFUS|WITHHELD|STOP|LEAK|Traceback|"
                  r"Error:|no usable measurements|could not|cannot)\b")

# ...except these, which are standing prose rather than a finding.
NOT_LOUD = re.compile(r"cannot see a camera move that left the detector|"
                      r"Confidence cannot catch this")

ANCHORS = [(re.compile(p), n) for p, n in ANCHORS]
TABLES = [(re.compile(a), re.compile(r)) for a, r in TABLES]


def summarize(lines):
    keep = [False] * len(lines)
    for index, line in enumerate(lines):
        for pattern, extra in ANCHORS:
            if pattern.search(line):
                for offset in range(0, extra + 1):
                    if index + offset < len(lines):
                        keep[index + offset] = True
                break
        for anchor, row in TABLES:
            if anchor.search(line):
                keep[index] = True
                step = index + 1
                while step < len(lines) and (row.match(lines[step])
                                             or not lines[step].strip()):
                    keep[step] = row.match(lines[step]) is not None or keep[step]
                    step += 1

    body = [lines[i].rstrip() for i in range(len(lines)) if keep[i]]

    # Whatever shouted and was not already captured. This is the part that
    # keeps the script honest: a warning added to either tool after this file
    # was written still reaches the summary.
    loose = [lines[i].rstrip() for i in range(len(lines))
             if not keep[i] and LOUD.search(lines[i])
             and not NOT_LOUD.search(lines[i])]

    out = []
    command = next((l.rstrip() for l in lines[:5] if "check_" in l), None)
    if command:
        out.append(command)
        out.append("")
    out.extend(body)
    if loose:
        out.append("")
        out.append("--- OTHER LINES WORTH A LOOK "
                   "(shouted, no pattern for them here) ---")
        out.extend(loose)
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] not in ("-", "--help", "-h"):
        with open(sys.argv[1], encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    elif len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        return
    else:
        lines = sys.stdin.read().splitlines()

    out = summarize(lines)
    print("\n".join(out))
    print(f"\n[{len(out)} lines kept of {len(lines)}; full log is the file "
          f"you passed in]")


if __name__ == "__main__":
    main()
