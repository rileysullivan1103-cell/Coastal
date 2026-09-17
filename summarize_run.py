"""Pull the decisive lines out of a geometry or mask-audit run.

    python3 check_camera_geometry.py ... 2>&1 | tee run.log
    python3 summarize_run.py run.log
    python3 summarize_run.py --compare one.log two.log

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

# --- comparing two runs -----------------------------------------------------
# One run's quality table is a column of numbers. TWO runs' tables, on the same
# frames with the anchor moved, are the drift test: if the record walked away
# from its anchor, the quarters that lose the reference should move to the
# OTHER end when the anchor does. Reading that off two tables by eye is how it
# gets missed, so it is done here.

DERIVED_REF = re.compile(r"reference frame: (\d{4}-\d{2}-\d{2})")
FORCED_REF = re.compile(r"using (\d{4}-\d{2}-\d{2}) \(")
WINDOW = re.compile(r"peaks are searched within (\d+) px")
GAP = re.compile(r"routes differ by a median of ([\d.]+) px")
RESOLUTION = re.compile(r"SMALLEST MOVE THIS RECORD CAN RESOLVE: (\S+) px")
QUARTER = re.compile(r"^\s+(\d{4}Q\d)\s+(\d+)\s+(\d+)%\s+(\d+)%\s*$")
WITHHELD = re.compile(r"are WITHHELD|THAT RESOLUTION IS")
NO_GROUP = re.compile(r"No \d+ of (\d+) (?:patches|cells) agree")
KEPT = re.compile(r"features kept: (\d+) of (\d+)")

# The same threshold check_camera_geometry.py uses, kept here so a comparison
# can be read without the tool at hand. If they ever disagree, that file wins.
DRIFT_GAP = 15


def read_run(lines):
    """The few numbers a run has to be compared on."""
    run = {"quarters": [], "reference": None, "forced": False,
           "window": None, "gap": None, "resolution": None,
           "withheld": False, "kept": None}
    for line in lines:
        found = FORCED_REF.search(line)
        if found:
            run["reference"], run["forced"] = found.group(1), True
        elif DERIVED_REF.search(line) and not run["forced"]:
            run["reference"] = DERIVED_REF.search(line).group(1)
        for key, pattern in (("window", WINDOW), ("gap", GAP),
                             ("resolution", RESOLUTION)):
            found = pattern.search(line)
            if found and run[key] is None:
                run[key] = found.group(1)
        if WITHHELD.search(line):
            run["withheld"] = True
        found = KEPT.search(line) or NO_GROUP.search(line)
        if found:
            run["kept"] = ("0 of " + found.group(1) if NO_GROUP.search(line)
                           else f"{found.group(1)} of {found.group(2)}")
        found = QUARTER.match(line)
        if found:
            run["quarters"].append((found.group(1), int(found.group(2)),
                                    int(found.group(3)), int(found.group(4))))
    return run


def compare(first, second, names):
    """Two runs side by side, and whether the drift moved with the anchor."""
    runs = [read_run(first), read_run(second)]
    out = ["TWO RUNS SIDE BY SIDE", ""]
    for name, run in zip(names, runs):
        anchor = run["reference"] or "?"
        out.append(f"  {name}")
        out.append(f"    anchor {anchor}"
                   + ("  (FORCED)" if run["forced"] else "  (derived)")
                   + f"   --max-shift {run['window'] or '?'}")
        out.append(f"    two-route gap {run['gap'] or '?'} px, "
                   f"resolution {run['resolution'] or '?'} px, "
                   f"patches kept {run['kept'] or '?'}"
                   + ("   EPOCHS WITHHELD" if run["withheld"] else ""))
    out.append("")

    periods = [q[0] for q in runs[0]["quarters"]]
    if not periods or [q[0] for q in runs[1]["quarters"]] != periods:
        out.append("  The two runs do not cover the same quarters, so their "
                   "tables are not")
        out.append("  comparable. Run both over the same frames.")
        return out

    out.append("  WHERE EACH RUN REGISTERS   (vs reference / vs previous "
               "frame; * = parted)")
    out.append(f"  quarter   {names[0][:22]:<22}  {names[1][:22]:<22}")
    drift = ([], [])
    for (period, _, a_dir, a_seq), (_, _, b_dir, b_seq) in zip(*[r["quarters"]
                                                                for r in runs]):
        marks = []
        for index, (direct, seq) in enumerate(((a_dir, a_seq), (b_dir, b_seq))):
            parted = seq - direct > DRIFT_GAP
            if parted:
                drift[index].append(period)
            marks.append(f"{direct:>3}% / {seq:>3}%" + (" *" if parted else "  "))
        out.append(f"  {period:<9} {marks[0]:<22}  {marks[1]:<22}")

    out.append("")
    if not drift[0] and not drift[1]:
        out.append("  Neither run loses the anchor anywhere. There is no drift "
                   "to see at these")
        out.append("  windows -- which is also what a window wide enough to "
                   "recapture it looks")
        out.append("  like, so this is only an answer if the window is tight.")
        return out
    for name, parted in zip(names, drift):
        out.append(f"  {name}: parts at "
                   + (", ".join(parted) if parted else "nowhere"))
    if drift[0] and drift[1]:
        both = set(drift[0]) & set(drift[1])
        if both == set(drift[0]) == set(drift[1]):
            out.append("")
            out.append("  THE SAME QUARTERS PART IN BOTH RUNS. Moving the "
                       "anchor did not move the")
            out.append("  parting, so this is NOT the record drifting away "
                       "from an anchor -- it is")
            out.append("  those frames being hard to register against "
                       "anything. Drift is refuted.")
        elif not both:
            out.append("")
            out.append("  THE PARTING MOVED WITH THE ANCHOR, and shares no "
                       "quarter between the two")
            out.append("  runs. Each anchor loses the far end of the record "
                       "and keeps its own")
            out.append("  neighbourhood, which is what drift looks like and "
                       "nothing else does.")
        else:
            out.append("")
            out.append(f"  The parting moved but {len(both)} quarter"
                       f"{'' if len(both) == 1 else 's'} part in BOTH runs "
                       f"({', '.join(sorted(both))}).")
            out.append("  Those are hard to register against any anchor; the "
                       "rest moved with it.")
            out.append("  Drift in part of the record, unregisterable frames "
                       "in the rest.")
    return out

# (pattern, extra lines to carry with it). The extra count exists because both
# tools wrap a single print across several lines, and the number is often on
# the continuation: "the intrusion runs 0.414 of the frame height INTO the
# land" is the second line of its print.
ANCHORS = [
    # --- the record itself ------------------------------------------------
    (r"\d+ frames on disk, \d{4}", 0),
    (r"THE RECORD HOLDS \d+ DIFFERENT FRAME SIZES", 4),
    (r"^\s+\d+x\d+\s+[\d,]+ frames", 0),
    (r"median [\d.]+, quartiles .*floor", 1),
    (r"intended sample days produced no frame", 0),
    (r"% of the \d+x\d+ frame is land", 1),
    # --- check_camera_geometry.py, whole-frame pass -----------------------
    (r"reference frame: \d{4}-\d{2}-\d{2}", 0),
    (r"REFERENCE FORCED by --reference", 1),
    (r"it scores [\d.]+ and ranks", 2),
    (r"that is the nearest sampled frame to", 0),
    (r"A reference in the bottom half", 3),
    (r"grid cells of \d+px over rows", 0),
    (r"REGISTERS AGAINST THE PREVIOUS FRAME BUT NOT", 1),
    (r"NOT TWO INDEPENDENT MEASUREMENTS", 8),
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
    (r"patch positions fell outside the moved frame", 0),
    (r"consecutive pairs below confidence .*carried as zero", 0),
    (r"the agreeing patches span", 0),
    # --- the two passes against each other --------------------------------
    (r"THE TWO PASSES, SIDE BY SIDE", 0),
    (r"\d+ of \d+ candidate moves are seen BY BOTH passes", 2),
    # --- the window that bounded the answer -------------------------------
    (r"THAT RESOLUTION IS \d+% OF THE SEARCH WINDOW", 6),
    (r"the largest offset is .* of the\s*$", 1),
    (r"THE TEST: re-run with --max-shift", 4),
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

# How many rows of an enumeration to keep before saying how many were cut.
# Forty-three epochs is itself the finding; rows 6 to 40 are not, and pasting
# them on costs the reader the part that mattered. The count line above the
# list is always kept, so nothing is hidden by this -- only unrepeated.
MAX_ROWS = 6

# Blocks whose rows are worth keeping whole: an anchor, then every following
# line that still matches the row shape.
TABLES = [
    (r"agreement between candidates", r"^\s+(keep|drop)\s"),
    (r"WHERE THE RECORD REGISTERS", r"^\s+(quarter|\S+\s+\d+\s)"),
    (r"largest single frame-to-frame step", r"^\s+\d{4}-\d{2}-\d{2}\s+[-\d.]+ px"),
    (r"the six worst frames", r"^\s+\d{4}-\d{2}-\d{2}\s"),
    (r"intended sample days produced no frame",
     r"^\s+\d{4}-\d{2}-\d{2} to \d{4}-\d{2}-\d{2}\s+\d+ days"),
]

# Enumerations to trim to MAX_ROWS. Same shape as TABLES, but the rows are a
# list whose length is the point rather than a table to read across.
LISTS = [
    (r"candidate discontinuit", r"^\s+\d{4}-\d{2}-\d{2}\s+[-\d.]+ px ->"),
    (r"^\d+ epochs:", r"^\s+\d+\. \d{4}-\d{2}-\d{2} to "),
    (r"THE TWO PASSES, SIDE BY SIDE",
     r"^\s+\d{4}-\d{2}-\d{2}\s+[\d.]+ px"),
]

# Anything shouting is kept whether or not a pattern above knows about it.
LOUD = re.compile(r"\b(WARNING|ERROR|REFUS|WITHHELD|STOP|LEAK|Traceback|"
                  r"Error:|no usable measurements|could not|cannot)\b")

# ...except these, which are standing prose rather than a finding.
NOT_LOUD = re.compile(r"cannot see a camera move that left the detector|"
                      r"Confidence cannot catch this")

# A heading's underline is not content and is not the end of what follows it.
SEPARATOR = re.compile(r"^\s*[=\-!]{3,}\s*$")

ANCHORS = [(re.compile(p), n) for p, n in ANCHORS]
TABLES = [(re.compile(a), re.compile(r)) for a, r in TABLES]
LISTS = [(re.compile(a), re.compile(r)) for a, r in LISTS]


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

    # Trim the long enumerations, saying how many rows were left out so the
    # reader knows to go to the log rather than assuming that was all of them.
    cut = {}
    for index, line in enumerate(lines):
        for anchor, row in LISTS:
            if not anchor.search(line):
                continue
            rows = []
            step = index + 1
            while step < len(lines):
                if row.match(lines[step]):
                    rows.append(step)
                elif lines[step].strip() and not SEPARATOR.match(lines[step]):
                    break
                step += 1
            for position, where in enumerate(rows):
                keep[where] = position < MAX_ROWS
            if len(rows) > MAX_ROWS:
                cut[rows[MAX_ROWS - 1]] = len(rows) - MAX_ROWS

    body = []
    for index in range(len(lines)):
        if keep[index]:
            body.append(lines[index].rstrip())
        if index in cut:
            body.append(f"    ... and {cut[index]} more (in the log)")

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
    if len(sys.argv) > 3 and sys.argv[1] == "--compare":
        runs = []
        for path in sys.argv[2:4]:
            with open(path, encoding="utf-8", errors="replace") as handle:
                runs.append(handle.read().splitlines())
        out = compare(runs[0], runs[1], [p.rsplit("/", 1)[-1]
                                         for p in sys.argv[2:4]])
        print("\n".join(out))
        return
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
