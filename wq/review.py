"""beach_type, assigned BY HAND from imagery. Deliberately not automated.

Enclosure is continuous. Any threshold on land_fraction_5km or
embayment_ratio gets the obvious sites right and misclassifies exactly the
ambiguous ones -- a half-open embayment, a beach inside a breakwater, a
lagoon mouth -- and those are the sites that decide whether stratifying on
beach type explains anything. An automatic label would put the hardest cases
on whichever side of a cutoff nobody chose deliberately, and D2 would then be
reporting the cutoff.

So the continuous covariates SORT the work; a person assigns the label.

    python -m wq.review --worklist            # what to look at, hardest first
    python -m wq.review --worklist --n 200
    python -m wq.review --ingest reviewed.csv --by "riley" --on 2026-09-20
    python -m wq.review --status

The worklist carries an imagery URL per station, and the ambiguity score that
ordered it. Fill in the beach_type column, save, and ingest. Every ingested
label records WHO assigned it and WHEN, and nothing else may write the
column: wq/strata.py reads beach_type only from the reviewed file.
"""

import argparse
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

from . import config

REVIEW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "beach_type_reviewed.csv")
WORKLIST_PATH = os.path.join(config.DATA_DIR, "beach_type_worklist.csv")
# A committed, reproducible subset of beaches to review, so the work is
# finite and its composition is auditable. See draw_sample.
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "review", "beach_type_sample.csv")
BEACH_TYPE_SAMPLE_SEED = 20260917
BEACH_TYPE_SAMPLE_N = 300

VALID = tuple(config.STRATA["beach_type"]["values"])
IMAGERY = ("https://www.google.com/maps/@{lat},{lon},600m/data=!3m1!1e3")

REVIEW_COLUMNS = ["station_id", "station_name", "state", "lat", "lon",
                  "beach_type", "assigned_by", "assigned_on", "note"]


def ambiguity(frame):
    """How badly each site needs a human. Highest first.

    A site is ambiguous when the continuous proxies do not agree with each
    other or sit near the middle of their range: land_fraction_5km around
    0.5, embayment_ratio around 0.9, curvature near zero. Sites at the
    extremes are still reviewed -- every site is -- but they are quick, and
    this ordering means the hard ones get looked at while attention lasts.
    """
    def middle(series, centre, spread):
        values = pd.to_numeric(series, errors="coerce")
        return 1.0 - ((values - centre).abs() / spread).clip(0, 1)

    score = pd.Series(0.0, index=frame.index)
    weight = pd.Series(0.0, index=frame.index)
    for column, centre, spread in (("land_fraction_5km", 0.55, 0.25),
                                   ("embayment_ratio", 0.90, 0.12),
                                   ("curvature_1_per_km", 0.0, 0.30)):
        if column not in frame.columns:
            continue
        piece = middle(frame[column], centre, spread)
        present = pd.to_numeric(frame[column], errors="coerce").notna()
        score = score.add(piece.fillna(0) * present, fill_value=0)
        weight = weight.add(present.astype(float), fill_value=0)
    # A site with no proxies at all is maximally ambiguous, not minimally:
    # nothing is known about it.
    out = np.where(weight > 0, score / weight.replace(0, np.nan), 1.0)
    return pd.Series(out, index=frame.index).fillna(1.0)


def worklist(sites, spatial=None, limit=None, sample=None):
    frame = sites.copy()
    if sample is not None and not sample.empty:
        keep = set(sample["station_id"].astype(str))
        frame = frame[frame["station_id"].astype(str).isin(keep)]
        print(f"  restricted to the committed sample: {len(frame)} beach(es) "
              f"of {len(sites)} station(s)")
    if spatial is not None and not spatial.empty:
        keep = [c for c in ("land_fraction_5km", "embayment_ratio",
                            "curvature_1_per_km", "fetch_km_mean",
                            "dist_to_stream_m", "dist_to_outfall_m")
                if c in spatial.columns]
        frame = frame.merge(spatial[["station_id"] + keep], on="station_id",
                            how="left")
    frame["ambiguity"] = ambiguity(frame)
    frame["imagery"] = [IMAGERY.format(lat=r.get("lat"), lon=r.get("lon"))
                        for _, r in frame.iterrows()]

    existing = read_reviewed()
    done = set(existing["station_id"].astype(str)) if not existing.empty else set()
    frame["already_reviewed"] = frame["station_id"].astype(str).isin(done)
    frame["beach_type"] = frame["station_id"].astype(str).map(
        dict(zip(existing["station_id"].astype(str), existing["beach_type"]))
        if not existing.empty else {})

    columns = [c for c in ["station_id", "station_name", "state", "region",
                           "lat", "lon", "ambiguity", "land_fraction_5km",
                           "embayment_ratio", "curvature_1_per_km",
                           "fetch_km_mean", "dist_to_stream_m",
                           "dist_to_outfall_m", "imagery",
                           "already_reviewed", "beach_type"]
               if c in frame.columns]
    out = frame[columns].sort_values(
        ["already_reviewed", "ambiguity"], ascending=[True, False])
    return out.head(limit) if limit else out


def draw_sample(sites, n=BEACH_TYPE_SAMPLE_N, seed=BEACH_TYPE_SAMPLE_SEED):
    """A reviewable subset: n site CLUSTERS, stratified by state.

    beach_type is the one pre-registered stratum with no automatic fallback,
    and it is currently assigned for zero of 3,575 stations, so the coverage
    rule drops it. Reviewing 3,575 stations by eye is not going to happen;
    reviewing a few hundred BEACHES might.

    Clusters, not stations, because the label is a property of the beach --
    Cowell Beach is 22 identifiers inside 472 m and they are all the same
    sand. One representative station per cluster is offered for review.

    Stratified by state and drawn from a recorded seed so the sample is
    reproducible and its composition can be argued with. Nothing here assigns
    a label; wq.review.ingest still requires an assigner and a date.
    """
    frame = sites.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    if "site_cluster" not in frame.columns:
        raise SystemExit(
            "stations_stratified.csv has no site_cluster column — run "
            "python -m wq.run_wq --strata first")
    # The representative is the station the cluster is named after, which
    # strata.site_clusters defines as the smallest station_id in it.
    representatives = (frame.sort_values("station_id")
                       .drop_duplicates("site_cluster", keep="first"))
    rng = np.random.default_rng(seed)
    picks = []
    total = len(representatives)
    for state, group in representatives.groupby("state", dropna=False):
        share = len(group) / total if total else 0
        take = int(round(n * share))
        if len(group) and take == 0:
            take = 1
        take = min(take, len(group))
        picks.extend(rng.choice(group["site_cluster"].to_numpy(), size=take,
                                replace=False).tolist())
    chosen = representatives[representatives["site_cluster"].isin(set(picks))]
    return chosen.sort_values(["state", "site_cluster"])


def read_sample(path=None):
    path = path or SAMPLE_PATH
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype={"station_id": str})


def read_reviewed(path=None):
    path = path or REVIEW_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    frame = pd.read_csv(path, dtype=str, comment="#")
    for column in REVIEW_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame


def ingest(path, assigned_by, assigned_on=None, review_path=None):
    """Merge a filled-in worklist into the reviewed file.

    Every row gets the assigner and the date. A label without both is
    rejected: "somebody classified this at some point" is not a provenance,
    and beach_type is the one stratum with no automatic fallback, so its
    provenance is all there is.
    """
    review_path = review_path or REVIEW_PATH
    if not assigned_by:
        sys.exit("--by is required: record who assigned these labels")
    assigned_on = assigned_on or date.today().isoformat()

    incoming = pd.read_csv(path, dtype=str, comment="#")
    if "beach_type" not in incoming.columns:
        sys.exit(f"{path} has no beach_type column")
    filled = incoming[incoming["beach_type"].notna()
                      & (incoming["beach_type"].astype(str).str.strip() != "")]
    if filled.empty:
        sys.exit(f"{path} has no beach_type filled in")

    bad = sorted(set(filled["beach_type"].astype(str).str.strip()) - set(VALID))
    if bad:
        sys.exit(f"not valid beach_type values: {bad}\nvalid: {list(VALID)}")

    rows = pd.DataFrame({
        "station_id": filled["station_id"].astype(str),
        "station_name": filled.get("station_name"),
        "state": filled.get("state"),
        "lat": filled.get("lat"),
        "lon": filled.get("lon"),
        "beach_type": filled["beach_type"].astype(str).str.strip(),
        "assigned_by": assigned_by,
        "assigned_on": assigned_on,
        "note": filled.get("note"),
    })

    existing = read_reviewed(review_path)
    if not existing.empty:
        changed = existing.merge(rows, on="station_id", suffixes=("_old", ""))
        flips = changed[changed["beach_type_old"] != changed["beach_type"]]
        for _, row in flips.iterrows():
            print(f"  {row['station_id']}: {row['beach_type_old']} -> "
                  f"{row['beach_type']} (was {row['assigned_by_old']} on "
                  f"{row['assigned_on_old']})")
        if len(flips):
            print(f"  {len(flips)} label(s) changed. The previous assignment "
                  "and its assigner are overwritten in the file but printed "
                  "here — put the reason in the note column.")
        existing = existing[~existing["station_id"].astype(str)
                            .isin(set(rows["station_id"]))]
    combined = pd.concat([existing, rows], ignore_index=True)[REVIEW_COLUMNS]
    combined.to_csv(review_path, index=False)
    print(f"\n{len(rows)} label(s) ingested, {len(combined)} on file -> "
          f"{review_path}")
    return combined


def status(sites=None):
    reviewed = read_reviewed()
    print(f"{len(reviewed)} station(s) have a hand-assigned beach_type")
    if reviewed.empty:
        print("\nbeach_type is EMPTY. The coverage rule will drop it from the")
        print("pre-registered strata, and D2 will have nothing to say about")
        print("beach type. That is the correct outcome for an unreviewed run —")
        print("it is not a bug, and it is not to be worked around by")
        print("thresholding land_fraction_5km.")
        return reviewed
    print("\nby label:")
    print(reviewed["beach_type"].value_counts().to_string())
    print("\nby assigner:")
    print(reviewed.groupby(["assigned_by", "assigned_on"]).size()
          .to_string())
    if sites is not None and len(sites):
        share = len(reviewed) / len(sites)
        print(f"\ncoverage {len(reviewed)}/{len(sites)} = {share:.1%}, "
              f"needs {config.COVARIATE_MIN_COVERAGE:.0%}")
        if share < config.COVARIATE_MIN_COVERAGE:
            print("  below the threshold — beach_type will be dropped from "
                  "the pre-registered strata.")
    return reviewed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", action="store_true")
    parser.add_argument("--ingest", metavar="CSV")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--by", help="who assigned these labels")
    parser.add_argument("--on", help="ISO date of the assignment")
    parser.add_argument("--n", type=int, help="limit the worklist")
    parser.add_argument("--draw-sample", action="store_true",
                        help=f"draw {BEACH_TYPE_SAMPLE_N} site clusters, "
                             "stratified by state from a recorded seed, and "
                             "write the committed review sample")
    parser.add_argument("--sample", action="store_true",
                        help="restrict --worklist to the committed sample")
    parser.add_argument("--all", action="store_true",
                        help="--worklist over every station, ignoring the "
                             "committed sample")
    args = parser.parse_args()

    sites_path = os.path.join(config.DATA_DIR, "stations_stratified.csv")
    if not os.path.exists(sites_path):
        sites_path = os.path.join(config.DATA_DIR, "stations.csv")
    sites = (pd.read_csv(sites_path, low_memory=False)
             if os.path.exists(sites_path) else None)

    if args.draw_sample:
        if sites is None:
            sys.exit("no station table yet — run wq.run_wq --strata first")
        chosen = draw_sample(sites)
        os.makedirs(os.path.dirname(SAMPLE_PATH), exist_ok=True)
        columns = [c for c in ("station_id", "site_cluster", "station_name",
                               "state", "region", "lat", "lon")
                   if c in chosen.columns]
        chosen[columns].to_csv(SAMPLE_PATH, index=False)
        print(f"wrote {SAMPLE_PATH}  ({len(chosen)} beach(es), seed "
              f"{BEACH_TYPE_SAMPLE_SEED})")
        print(chosen.groupby("state").size().rename("beaches").to_string())
        print("\nNothing is labelled. Next:")
        print("  python -m wq.review --worklist --sample")
        return
    if args.ingest:
        ingest(args.ingest, args.by, args.on)
        return
    if args.status:
        status(sites)
        return
    if args.worklist:
        if sites is None:
            sys.exit("no station table yet — run wq.run_wq --stations first")
        spatial_path = os.path.join(config.DATA_DIR, "site_covariates.csv")
        spatial = (pd.read_csv(spatial_path, low_memory=False)
                   if os.path.exists(spatial_path) else None)
        if spatial is None:
            print("  no site_covariates.csv — the worklist will not be sorted "
                  "by ambiguity. Run wq.run_wq --spatial first.")
        sample = None if args.all else read_sample()
        if sample is None and not args.all:
            print("  no committed sample yet — the worklist covers every "
                  "station. Draw one with:\n    python -m wq.review "
                  "--draw-sample")
        elif not args.sample and not args.all:
            print("  a committed sample exists; pass --sample to review only "
                  "those beaches, or --all to override")
            sample = None
        table = worklist(sites, spatial, args.n,
                         sample if args.sample else None)
        os.makedirs(config.DATA_DIR, exist_ok=True)
        table.to_csv(WORKLIST_PATH, index=False)
        print(f"{len(table)} station(s) -> {WORKLIST_PATH}")
        print("\nFill in the beach_type column from the imagery link, then:")
        print(f"  python -m wq.review --ingest {WORKLIST_PATH} "
              '--by "your name"')
        print(f"\nvalid labels: {', '.join(VALID)}")
        return
    parser.error("give --worklist, --ingest or --status")


if __name__ == "__main__":
    main()
