"""Phase 2's evaluation split, fixed BEFORE any model is fitted.

A held-out set chosen after seeing how a model does is not a held-out set. So
the split is drawn here, from a recorded seed, committed to
wq/holdout_sites.csv, and hashed into wq/SCOPE_AMENDMENT_CA_DRAFT.md. Anything
that reads the study has to drop these rows unless it is explicitly evaluating
on them.

Three decisions are frozen in this file.

CLUSTERS, NOT STATIONS. The unit held out is the site_cluster from
wq.strata.site_clusters, because a station is not an independent beach. Cowell
Beach in Santa Cruz is 22 station identifiers inside 472 m -- eleven
CABEACH_WQX and eleven CEDEN describing the same sand. Splitting on stations
would put one end of that beach in training and the other in test, and the
resulting score would measure how well a model interpolates between two
sampling points rather than how it generalises to a beach it has never seen.
The five-state pass had the same problem in New Jersey, where sixteen NJDEP
stations line two kilometres of Atlantic City.

STRATIFIED BY STATE AND BY ANALYTE AVAILABILITY. California is 20% of the
clusters and carries the only total-coliform records in the study; the
Northeast carries almost all the enterococcus. An unstratified 20% draw can
take most of one analyte's beaches, and then the test set answers a different
question from the training set. The stratum is (state, the set of analytes the
cluster can actually fit).

HYPOTHESIS SITES ARE NOT TEST SITES. Santa Cruz Wharf and Carpinteria State
Beach are the two beaches whose prior small-sample findings this pass exists
to look at. They are reported individually and excluded from the replication
statistics; putting them in the test set as well would let a result about them
leak into a number that is supposed to be about beaches nobody has looked at.

A TIME HOLDOUT ON TOP. The most recent TIME_HOLDOUT_MONTHS months at EVERY
site, including training sites. Held-out beaches answer "does this work
somewhere new"; held-out months answer "does this still work next season",
and a model can pass either one while failing the other.
"""

import hashlib
import os

import numpy as np
import pandas as pd

from . import config

# Recorded, not chosen afresh on each run. Changing it redraws the split and
# invalidates every comparison made against the old one.
HOLDOUT_SEED = 20260916
HOLDOUT_FRACTION = 0.20
TIME_HOLDOUT_MONTHS = 12

# Holding out a CLUSTER is not the same as holding out a beach, because the
# next cluster along the same shoreline is usually 150-600 m away and sampled
# by the same agency on the same mornings. Measured on this study: the median
# held-out cluster has a training station 561 m away, 44.6% have one inside
# 500 m and 76.3% inside 1 km. The nearest possible is 150 m, which is the
# cluster radius itself -- anything closer would have merged.
#
# So the evaluation fit drops training stations within this distance of any
# test cluster. It does NOT change which clusters are held out;
# holdout_sites.csv is fixed and hashed, and this is a property of the
# TRAINING side.
#
# 0.5 km is the headline. Single-linkage clusters span up to about 500 m, so
# it means "not contiguous with the test beach", and it costs 16.1% of
# training stations. 1.0 km is the sensitivity: more defensible in principle,
# but it costs 40.3% of training stations and 51.4% of New Jersey's, which is
# the whole shellfish population -- so at 1 km the two arms of the comparison
# are drawn from different coasts as well as different beaches.
TRAINING_BUFFER_KM = 0.5
TRAINING_BUFFER_SENSITIVITY_KM = 1.0

HOLDOUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "holdout_sites.csv")
HYPOTHESIS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "hypothesis_sites.csv")


def file_sha256(path):
    """The hash the amendment quotes, so a silently redrawn split is visible."""
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def read_hypothesis_sites(path=None):
    path = path or HYPOTHESIS_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=["station_id", "site_cluster", "label"])
    return pd.read_csv(path, dtype=str)


def read_holdout(path=None):
    path = path or HOLDOUT_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=["station_id", "site_cluster", "state"])
    return pd.read_csv(path, dtype=str)


def time_cutoff(samples, months=TIME_HOLDOUT_MONTHS):
    """The date after which every sample is held out, at every site."""
    dates = pd.to_datetime(samples["date"], errors="coerce")
    if dates.notna().sum() == 0:
        return None
    return (dates.max() - pd.DateOffset(months=months)).normalize()


def _analyte_signature(frame):
    """Which analytes a cluster can actually fit. Part of the stratum, because
    a cluster with only E. coli and a cluster with all four are not
    interchangeable members of a test set."""
    counts = frame.groupby(["site_cluster", "analyte"]).size()
    keep = counts[counts >= config.MIN_SAMPLES_PER_SITE].reset_index()
    return (keep.groupby("site_cluster")["analyte"]
            .apply(lambda s: "+".join(sorted(set(s)))).rename("analytes"))


def build(samples, sites, hypothesis=None, fraction=HOLDOUT_FRACTION,
          seed=HOLDOUT_SEED):
    """Draw the test clusters. Returns (holdout_rows, summary)."""
    frame = samples.merge(
        sites[["station_id", "site_cluster", "state"]].assign(
            station_id=lambda d: d["station_id"].astype(str)),
        on="station_id", how="inner")
    signature = _analyte_signature(frame)
    per_cluster = (frame.groupby("site_cluster")
                   .agg(state=("state", "first"),
                        stations=("station_id", "nunique"),
                        samples=("station_id", "size")).reset_index())
    per_cluster = per_cluster.merge(signature, on="site_cluster", how="left")
    per_cluster["analytes"] = per_cluster["analytes"].fillna("none")

    blocked = set()
    if hypothesis is not None and not hypothesis.empty:
        blocked = set(hypothesis["site_cluster"].astype(str))
    eligible = per_cluster[~per_cluster["site_cluster"].astype(str).isin(blocked)]
    # A cluster with nothing fittable cannot test anything.
    eligible = eligible[eligible["analytes"] != "none"]

    rng = np.random.default_rng(seed)
    chosen = []
    for (state, analytes), group in eligible.groupby(["state", "analytes"],
                                                     dropna=False):
        n = len(group)
        k = int(round(fraction * n))
        # A stratum big enough to spare one contributes one even when rounding
        # says zero; a stratum of one or two keeps both, because holding out
        # the only example of a combination leaves nothing to train on.
        if n >= 5 and k == 0:
            k = 1
        if k <= 0:
            continue
        picks = rng.choice(group["site_cluster"].to_numpy(), size=min(k, n),
                           replace=False)
        chosen.extend(picks.tolist())

    chosen = sorted(set(chosen))
    rows = (sites[sites["site_cluster"].astype(str).isin(chosen)]
            [["station_id", "site_cluster", "state"]]
            .sort_values(["site_cluster", "station_id"]))
    # Two counts, named so neither can be summed into a wrong number. The
    # cluster total repeats on every station row in that cluster, so a reader
    # adding it up gets the cluster counted once per station; the per-station
    # count is the one that sums.
    counts = (frame[frame["site_cluster"].astype(str).isin(chosen)]
              .groupby("site_cluster").size().rename("cluster_samples"))
    per_station = (frame[frame["site_cluster"].astype(str).isin(chosen)]
                   .groupby("station_id").size().rename("station_samples"))
    rows = rows.merge(counts, on="site_cluster", how="left")
    rows = rows.merge(per_station, on="station_id", how="left")
    rows["station_samples"] = rows["station_samples"].fillna(0).astype(int)

    summary = {
        "seed": seed,
        "fraction": fraction,
        "eligible_clusters": int(len(eligible)),
        "holdout_clusters": len(chosen),
        "holdout_stations": int(len(rows)),
        "holdout_samples": int(per_station.sum()) if len(per_station) else 0,
        "hypothesis_clusters_excluded": len(blocked),
        "time_cutoff": time_cutoff(samples),
        "time_holdout_months": TIME_HOLDOUT_MONTHS,
    }
    return rows, summary


def buffered_training_stations(sites, buffer_km=TRAINING_BUFFER_KM,
                               held=None):
    """Training stations too close to a test cluster to count as unseen.

    Returns (keep_ids, dropped_ids). Distance is station-to-station: a
    training station is dropped when ANY station of ANY held-out cluster is
    within buffer_km of it.

    This is for the evaluation fit only. holdout_sites.csv is not touched --
    the test set is fixed and hashed, and moving it to suit a buffer would be
    choosing the test set after seeing the geometry.
    """
    from .strata import _haversine_km
    held = read_holdout() if held is None else held
    frame = sites.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    if held.empty:
        return set(frame["station_id"]), set()
    clusters = set(held["site_cluster"].astype(str))
    is_test = frame["site_cluster"].astype(str).isin(clusters)
    test = frame[is_test].dropna(subset=["lat", "lon"])
    train = frame[~is_test].dropna(subset=["lat", "lon"])
    if test.empty or train.empty:
        return set(train["station_id"]), set()
    lat, lon = test["lat"].to_numpy(), test["lon"].to_numpy()
    dropped = set()
    for row in train.itertuples():
        if float(_haversine_km(row.lat, row.lon, lat, lon).min()) <= buffer_km:
            dropped.add(str(row.station_id))
    return set(train["station_id"]) - dropped, dropped


def drop_holdout(frame, evaluate_holdout=False, samples=None, quiet=False):
    """Remove held-out rows. THE guard -- every reader goes through here.

    Held-out beaches are dropped by cluster and, where the frame carries a
    date, the held-out months are dropped at every site as well. Passing
    evaluate_holdout keeps them and says so out loud, because a silent
    evaluation on training data is the failure this file exists to prevent.
    """
    if evaluate_holdout:
        if not quiet:
            print("  EVALUATING ON HELD-OUT DATA — this is only valid once, "
                  "and the result\n  must be reported as a held-out score, "
                  "not as a development number.")
        return frame
    if frame is None or frame.empty:
        return frame
    held = read_holdout()
    out = frame
    if not held.empty:
        clusters = set(held["site_cluster"].astype(str))
        stations = set(held["station_id"].astype(str))
        if "site_cluster" in out.columns:
            out = out[~out["site_cluster"].astype(str).isin(clusters)]
        elif "station_id" in out.columns:
            out = out[~out["station_id"].astype(str).isin(stations)]
    if "date" in out.columns:
        cutoff = time_cutoff(samples if samples is not None else out)
        if cutoff is not None:
            dates = pd.to_datetime(out["date"], errors="coerce")
            out = out[dates.isna() | (dates <= cutoff)]
    if not quiet and len(out) != len(frame):
        print(f"  holdout guard: dropped {len(frame) - len(out):,} of "
              f"{len(frame):,} rows (held-out clusters and the most recent "
              f"{TIME_HOLDOUT_MONTHS} months)")
    return out


# The two beaches whose prior small-sample findings this pass exists to look
# at. Named by a SEED station; the whole site_cluster around each is taken,
# because the prior finding is about the beach and Cowell Beach alone carries
# 22 identifiers.
HYPOTHESIS_SEEDS = [
    ("CABEACH_WQX-Wharf-East", "Santa Cruz Wharf",
     "prior: rho ~ +0.45 vs 72h rain, replicated across three analytes"),
    ("CABEACH_WQX-WP0000180", "Carpinteria State Beach",
     "prior: TOTAL coliform vs 48-72h rain, rho ~ -0.32 (reversed sign)"),
]


def build_hypothesis_sites(sites, seeds=None):
    """Every station on the hypothesis beaches, from the seed identifiers."""
    seeds = seeds or HYPOTHESIS_SEEDS
    frame = sites.copy()
    frame["station_id"] = frame["station_id"].astype(str)
    rows = []
    for seed, label, reason in seeds:
        match = frame[frame["station_id"] == seed]
        if match.empty:
            print(f"  WARNING: seed {seed} ({label}) is not in the station "
                  "table — no stations recorded for it")
            continue
        cluster = match["site_cluster"].iloc[0]
        for _, site in frame[frame["site_cluster"] == cluster].iterrows():
            rows.append({"station_id": site["station_id"],
                         "site_cluster": cluster,
                         "station_name": site.get("station_name"),
                         "state": site.get("state"),
                         "label": label,
                         "seed_station": seed,
                         "reason": reason})
    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="draw the split and write both CSVs")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    sites = pd.read_csv(os.path.join(config.DATA_DIR,
                                     "stations_stratified.csv"),
                        low_memory=False)
    sites["station_id"] = sites["station_id"].astype(str)

    if args.show or not args.write:
        for name, path in (("hypothesis", HYPOTHESIS_PATH),
                           ("holdout", HOLDOUT_PATH)):
            frame = pd.read_csv(path) if os.path.exists(path) else None
            print(f"{name:11s} {path}")
            print(f"            {0 if frame is None else len(frame)} row(s), "
                  f"sha256 {file_sha256(path)}")
        return

    samples = pd.read_csv(os.path.join(config.DATA_DIR, "samples_clean.csv"),
                          low_memory=False,
                          usecols=["station_id", "analyte", "date"])
    samples["station_id"] = samples["station_id"].astype(str)

    hypothesis = build_hypothesis_sites(sites)
    hypothesis.to_csv(HYPOTHESIS_PATH, index=False)
    print(f"wrote {HYPOTHESIS_PATH}  ({len(hypothesis)} stations, "
          f"{hypothesis['site_cluster'].nunique()} cluster(s))")

    rows, summary = build(samples, sites, hypothesis)
    rows.to_csv(HOLDOUT_PATH, index=False)
    print(f"wrote {HOLDOUT_PATH}  ({len(rows)} stations, "
          f"{rows['site_cluster'].nunique()} clusters)")
    for key, value in summary.items():
        print(f"  {key:32s} {value}")
    print(f"  {'sha256(holdout_sites.csv)':32s} {file_sha256(HOLDOUT_PATH)}")
    print(f"  {'sha256(hypothesis_sites.csv)':32s} "
          f"{file_sha256(HYPOTHESIS_PATH)}")


if __name__ == "__main__":
    main()
