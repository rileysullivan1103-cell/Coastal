"""When each detection class appears and disappears, month by month, per camera.

A class that switches on mid-record is indistinguishable, in an unfiltered
series, from the detector changing. If a camera starts emitting `person` in
June, the unfiltered detection rate steps up in June and a changepoint search
will name that date and invite a hunt for a release note. The step is real; it
is just not about rip currents.

So this reports, per camera and per month: frame counts by class GROUP
(rip_current, person, boat, other object, blank), the first and last date each
group appears, and any model_name / model_version values present. Months where
an object group starts or stops are flagged, and those dates are what
analyze_detector_changepoints.py --compare-classes marks its changepoints
against.

A frame naming two classes ("boat,person") counts once in EACH group, so the
group columns sum to more than the frame count. The total column is frames.

    python diagnose_class_timeline.py
    python diagnose_class_timeline.py --cameras walton

Reads only what is on disk. No network.
"""

import argparse
import glob
import os
import sys

import pandas as pd

import analyze_drivers as ad
import build_label_sample as bls

# The reporting group, not a class name: TWO models emit a rip class in this
# feed (`rip_current` from ripdetect_walton, `rip` from rip_current_detector)
# and they are the same finding. Matching only the first filed Corolla,
# Sailfish and Carova as 100% "other object" and 0% rip, which is exactly the
# "three cameras have never detected a rip" error already on the ruled-out
# list. Any rip class name build_label_sample knows about lands here.
RIP = "rip"
RIP_NAMES = frozenset(bls.RIP_CLASSES)
OBJECT_GROUPS = ("person", "boat", "other object")
GROUPS = (RIP,) + OBJECT_GROUPS + ("blank",)
OUT_DIR = f"{ad.DATA_DIR}/class_timeline"


def class_group(name):
    """Which reporting group one class name belongs to."""
    name = str(name).strip().lower()
    if not name:
        return "blank"
    if name in RIP_NAMES:
        return RIP
    if name == "person":
        return "person"
    if name == "boat":
        return "boat"
    return "other object"


def groups_of(value):
    """The groups a frame's score_classes value belongs to, as a set."""
    names = bls.class_set(value)
    if not names:
        return {"blank"}
    return {class_group(n) for n in names}


def load(path):
    frame = ad.read_csv(path)
    if frame is None or frame.empty:
        return None, "empty"
    if "timestamp" not in frame.columns:
        return None, "no timestamp column"
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True,
                                        errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    if frame.empty:
        return None, "no parseable timestamps"
    if "score_classes" not in frame.columns:
        return None, "no score_classes column"
    frame["month"] = frame["timestamp"].dt.strftime("%Y-%m")
    frame["groups"] = frame["score_classes"].map(groups_of)
    return frame, ""


def monthly_table(frame):
    """One row per month: frames, then a count per group."""
    rows = []
    for month, group in frame.groupby("month"):
        row = {"month": month, "frames": len(group)}
        for name in GROUPS:
            row[name] = int(group["groups"].map(
                lambda names, n=name: n in names).sum())
        # Any COCO object class, counted ONCE per frame. The per-group columns
        # count a "boat,person" frame twice, which is right for "how often does
        # boat appear" and wrong for "how much of this month is not rips".
        row["object"] = int(group["groups"].map(
            lambda names: bool(names & set(OBJECT_GROUPS))).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def class_events(frame):
    """First and last DATE each group appears, and its start/stop months.

    Dates rather than months, because a changepoint lands on a day and
    comparing it against a month would make every coincidence within 7 days a
    matter of which day of the month the class happened to begin.
    """
    out = []
    for name in GROUPS:
        present = frame[frame["groups"].map(lambda names, n=name: n in names)]
        if present.empty:
            continue
        out.append({
            "group": name,
            "n": len(present),
            "first": present["timestamp"].min(),
            "last": present["timestamp"].max(),
            "months": present["month"].nunique(),
        })
    return out


def transitions(table, events, span_first, span_last, edge_days=14):
    """Months where an object group switches on or off mid-record.

    A group appearing in the record's first month, or still present in its
    last, has not switched: it was there from the start, or it never stopped.
    Reporting those as events would flag the beginning and end of every camera
    and bury the real ones. edge_days is the grace period on each end.
    """
    out = []
    for event in events:
        if event["group"] not in OBJECT_GROUPS:
            continue
        if (event["first"] - span_first).days > edge_days:
            out.append({"group": event["group"], "kind": "starts",
                        "date": event["first"]})
        if (span_last - event["last"]).days > edge_days:
            out.append({"group": event["group"], "kind": "stops",
                        "date": event["last"]})
    return sorted(out, key=lambda e: e["date"])


def model_versions_by_month(frame):
    """Distinct model_name / model_version strings present in each month."""
    columns = [c for c in ("model_name", "model_version")
               if c in frame.columns]
    if not columns:
        return None
    tag = frame[columns].astype(str).agg(" / ".join, axis=1)
    table = pd.DataFrame({"month": frame["month"], "tag": tag})
    return (table.groupby("month")["tag"]
            .agg(lambda values: ", ".join(sorted(set(values))))
            .reset_index())


def report(camera, frame):
    table = monthly_table(frame)
    events = class_events(frame)
    span_first = frame["timestamp"].min()
    span_last = frame["timestamp"].max()
    changes = transitions(table, events, span_first, span_last)
    versions = model_versions_by_month(frame)
    by_month = ({} if versions is None
                else dict(zip(versions["month"], versions["tag"])))

    print(f"\n  {len(frame)} frames, {span_first:%Y-%m-%d} to {span_last:%Y-%m-%d}")
    print(f"\n  {'month':<9} {'frames':>7} {'rip':>7} {'object':>7} "
          f"{'obj%':>6} {'person':>7} {'boat':>7} {'other':>7} {'blank':>7}"
          "  model")
    flagged = {c["date"].strftime("%Y-%m") for c in changes}
    for _, row in table.iterrows():
        mark = " <-" if row["month"] in flagged else ""
        share = row["object"] / row["frames"] if row["frames"] else 0.0
        print(f"  {row['month']:<9} {row['frames']:>7} {row[RIP]:>7} "
              f"{row['object']:>7} {share:>5.0%} "
              f"{row['person']:>7} {row['boat']:>7} {row['other object']:>7} "
              f"{row['blank']:>7}  {by_month.get(row['month'], '')}{mark}")
    totals = table[["frames", RIP, "object"]].sum()
    print(f"  {'TOTAL':<9} {int(totals['frames']):>7} {int(totals[RIP]):>7} "
          f"{int(totals['object']):>7} "
          f"{totals['object'] / max(totals['frames'], 1):>5.0%}")

    # The raw names, so a class this script does not recognise is SEEN rather
    # than quietly counted as an object. That is how the rip/rip_current split
    # was missed the first time.
    seen = {}
    for value in frame["score_classes"]:
        for name in bls.class_set(value):
            seen[name] = seen.get(name, 0) + 1
    if seen:
        print("\n  class names in this camera's payload: "
              + ", ".join(f"{n} {c}" for n, c in
                          sorted(seen.items(), key=lambda kv: -kv[1])))
        unknown = sorted(n for n in seen
                         if class_group(n) == "other object")
        if unknown:
            print(f"    {len(unknown)} counted as 'other object': "
                  + ", ".join(unknown))
            print("    If any of those is a rip class under another name, it is"
                  " being\n    reported as an object and every rip count here "
                  "is wrong. Add it to\n    build_label_sample.RIP_CLASSES.")

    print(f"\n  {'group':<14} {'n':>7}  {'first':<12} {'last':<12} months")
    for event in events:
        print(f"  {event['group']:<14} {event['n']:>7}  "
              f"{event['first']:%Y-%m-%d}   {event['last']:%Y-%m-%d}   "
              f"{event['months']}")

    if changes:
        print(f"\n  OBJECT CLASSES SWITCHING MID-RECORD "
              f"({len(changes)} event(s)):")
        for change in changes:
            print(f"    {change['date']:%Y-%m-%d}  {change['group']} "
                  f"{change['kind']}")
        print("    A changepoint within 7 days of one of these, in an "
              "UNFILTERED series,\n    is the class switching rather than the "
              "detector changing.")
    else:
        print("\n  no object class starts or stops mid-record here")

    if versions is not None:
        distinct = sorted(set(versions["tag"]))
        if len(distinct) > 1:
            print(f"\n  MODEL VERSION CHANGES: {len(distinct)} distinct values")
            for tag in distinct:
                months = sorted(versions.loc[versions["tag"] == tag, "month"])
                print(f"    {tag}  in {months[0]}..{months[-1]} "
                      f"({len(months)} month(s))")
        else:
            print(f"\n  one model version throughout: {distinct[0]}")
    else:
        print("\n  payload records no model name or version")

    return table, changes


def cameras_on_disk():
    out = []
    try:
        sites = ad.load_sites()
        by_slug = {ad.rip_slug(n): n for n in sites["camera_name"].dropna()}
    except SystemExit:
        by_slug = {}
    for path in sorted(glob.glob(f"{ad.DATA_DIR}/rip_detection/rip_*.csv")):
        stem = os.path.basename(path)[len("rip_"):-len(".csv")]
        if stem.endswith("_index") or stem.endswith("_hourly"):
            continue
        out.append((by_slug.get(stem, stem), stem, path))
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", nargs="*", default=None)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    found = cameras_on_disk()
    if args.cameras:
        wanted = [c.lower() for c in args.cameras]
        found = [f for f in found
                 if any(w in f[0].lower() or w in f[1].lower() for w in wanted)]
    if not found:
        sys.exit("No rip records on disk.")

    tables, all_changes = [], []
    for camera, slug, path in found:
        print(f"\n{'=' * 72}\n{camera}\n{'=' * 72}")
        frame, why = load(path)
        if frame is None:
            print(f"  skipped: {why}")
            continue
        table, changes = report(camera, frame)
        table.insert(0, "camera", camera)
        tables.append(table)
        for change in changes:
            all_changes.append({"camera": camera, "group": change["group"],
                                "kind": change["kind"],
                                "date": change["date"].strftime("%Y-%m-%d")})

    if tables:
        os.makedirs(args.out_dir, exist_ok=True)
        pd.concat(tables, ignore_index=True).to_csv(
            f"{args.out_dir}/monthly_classes.csv", index=False)
        print(f"\n  wrote {args.out_dir}/monthly_classes.csv")
    if all_changes:
        pd.DataFrame(all_changes).to_csv(
            f"{args.out_dir}/class_events.csv", index=False)
        print(f"  wrote {args.out_dir}/class_events.csv  "
              "(what --compare-classes marks changepoints against)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
