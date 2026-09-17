"""Two questions the holdout carried over unanswered.

    python -m wq.diagnostics.holdout_audit
    python -m wq.diagnostics.holdout_audit --write-buffer

RECONCILIATION. wq/strata.py finds 3,173 site clusters; holdout.build saw
3,092 eligible. The difference has to be accounted for by name, not assumed,
because a cluster quietly missing from the eligible pool is a cluster that can
never be tested on.

ADJACENCY. site_cluster links at 150 m, so "a different cluster" can be the
next few hundred metres of the same beach, sampled by the same agency on the
same mornings. A test set drawn on clusters is only a test of generalisation
if the training set is not sitting on top of it. This measures how far each
held-out cluster actually is from the nearest TRAINING station, and prices
the buffers that would fix it.

wq/holdout_sites.csv is NEVER touched here. The test set is fixed and hashed;
moving it to suit the geometry would be choosing a test set after looking at
it. The buffer is an evaluation-time filter on the TRAINING side, written to
its own committed file.
"""

import argparse
import os

import numpy as np
import pandas as pd

from . import common
from .. import config, holdout, scope as study_scope
from ..strata import _haversine_km

BUFFER_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "holdout_buffer.csv")
CANDIDATES = (0.25, 0.5, 1.0, 1.5, 2.0, 5.0)


def reconcile(sites, samples, hypothesis):
    """Every cluster that is not in the eligible pool, with its reason."""
    frame = samples.merge(
        sites[["station_id", "site_cluster", "state"]], on="station_id",
        how="inner")
    all_clusters = set(sites["site_cluster"].astype(str))
    with_samples = set(frame["site_cluster"].astype(str))
    signature = holdout._analyte_signature(frame)
    fittable = set(signature.index.astype(str))
    hyp = set(hypothesis["site_cluster"].astype(str)) if not hypothesis.empty \
        else set()

    rows = []
    for cluster in sorted(all_clusters):
        if cluster not in with_samples:
            reason = "no samples at all"
        elif cluster not in fittable:
            reason = (f"samples split below the per-analyte floor of "
                      f"{config.MIN_SAMPLES_PER_SITE}")
        elif cluster in hyp:
            reason = "hypothesis-generating site, excluded by design"
        else:
            continue
        rows.append({"site_cluster": cluster, "reason": reason})
    out = pd.DataFrame(rows)
    if not out.empty:
        first = sites.drop_duplicates("site_cluster").set_index("site_cluster")
        out["state"] = out["site_cluster"].map(first["state"])
        best = (frame.groupby(["site_cluster", "analyte"]).size()
                .groupby(level=0).max())
        out["most_samples_in_one_analyte"] = out["site_cluster"].map(best)
    return out, len(all_clusters), len(fittable - hyp)


def adjacency(sites, held):
    """Distance from each held-out cluster to the nearest TRAINING station."""
    frame = sites.dropna(subset=["lat", "lon"]).copy()
    frame["station_id"] = frame["station_id"].astype(str)
    clusters = set(held["site_cluster"].astype(str))
    frame["is_test"] = frame["site_cluster"].astype(str).isin(clusters)
    train = frame[~frame["is_test"]]
    lat, lon = train["lat"].to_numpy(), train["lon"].to_numpy()
    rows = []
    for cluster, group in frame[frame["is_test"]].groupby("site_cluster"):
        km = min(float(_haversine_km(r.lat, r.lon, lat, lon).min())
                 for r in group.itertuples())
        rows.append({"site_cluster": cluster,
                     "state": group["state"].iloc[0],
                     "study_scope": group["study_scope"].iloc[0],
                     "stations": len(group),
                     "km_to_nearest_training": round(km, 3)})
    return pd.DataFrame(rows)


def buffer_cost(sites, held, candidates=CANDIDATES):
    """What each buffer would cost the training side, and which stations."""
    frame = sites.dropna(subset=["lat", "lon"]).copy()
    frame["station_id"] = frame["station_id"].astype(str)
    clusters = set(held["site_cluster"].astype(str))
    frame["is_test"] = frame["site_cluster"].astype(str).isin(clusters)
    test = frame[frame["is_test"]]
    train = frame[~frame["is_test"]].copy()
    lat, lon = test["lat"].to_numpy(), test["lon"].to_numpy()
    train["km_to_test"] = [float(_haversine_km(r.lat, r.lon, lat, lon).min())
                           for r in train.itertuples()]
    rows = []
    for km in candidates:
        dropped = train[train["km_to_test"] <= km]
        kept = train[train["km_to_test"] > km]
        rows.append({
            "buffer_km": km,
            "training_stations_dropped": len(dropped),
            "share_dropped": round(len(dropped) / len(train), 4),
            "training_clusters_lost":
                train["site_cluster"].nunique() - kept["site_cluster"].nunique(),
            "training_stations_left": len(kept),
        })
    return pd.DataFrame(rows), train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer-km", type=float,
                        default=holdout.TRAINING_BUFFER_KM)
    parser.add_argument("--write-buffer", action="store_true",
                        help="commit the excluded training stations for the "
                             "headline buffer")
    args = parser.parse_args()

    sites = common.load_sites()
    sites["station_id"] = sites["station_id"].astype(str)
    sites = sites.merge(study_scope.read()[["station_id", "study_scope"]],
                        on="station_id", how="left")
    sites["study_scope"] = sites["study_scope"].fillna("unclassified")
    samples = common.load_samples(usecols=["station_id", "analyte", "date"])
    samples["station_id"] = samples["station_id"].astype(str)
    held = holdout.read_holdout()

    print("=" * 78)
    print("A. RECONCILING site_clusters AGAINST THE HOLDOUT'S ELIGIBLE POOL")
    print("=" * 78)
    missing, total, eligible = reconcile(sites, samples,
                                         holdout.read_hypothesis_sites())
    print(f"\n  clusters in stations_stratified.csv   {total:,}")
    print(f"  eligible in holdout.build             {eligible:,}")
    print(f"  difference                            {total - eligible:,}")
    print(f"\n  accounted for below: {len(missing):,}")
    print(missing.groupby("reason")
          .agg(clusters=("site_cluster", "size")).to_string())
    print("\nby reason and state:")
    print(pd.crosstab(missing["reason"], missing["state"]).to_string())
    # Only the below-floor ones. The hypothesis clusters are excluded by
    # design and one of them holds thousands of samples, so pooling them here
    # would report a "near miss" of 5,793 against a floor of 30.
    below = missing[missing["reason"].str.startswith("samples split")]
    near = below["most_samples_in_one_analyte"].dropna()
    if len(near):
        print(f"\n  the {len(below)} below-floor clusters reach at most "
              f"{int(near.max())} samples in a single analyte against a floor "
              f"of {config.MIN_SAMPLES_PER_SITE};\n  median {int(near.median())}"
              f", min {int(near.min())} — they clear the per-STATION floor and "
              "fail the per-ANALYTE one")
    common.write(missing, "holdout_unaccounted_clusters.csv")

    print("\n" + "=" * 78)
    print("B. HOW FAR IS A HELD-OUT BEACH FROM THE TRAINING SET?")
    print("=" * 78)
    adj = adjacency(sites, held)
    km = adj["km_to_nearest_training"]
    print(f"\n  held-out clusters measured: {len(adj):,}")
    print(km.describe(percentiles=[.05, .25, .5, .75, .95]).round(3).to_string())
    print("\n  cumulative:")
    for t in (0.5, 1.0, 2.0, 5.0):
        n = int((km <= t).sum())
        print(f"    within {t:>4} km  {n:4d} of {len(adj)}  ({n / len(adj):5.1%})")
    print("\n  by state:")
    print(adj.assign(w1=km <= 1.0).groupby("state").agg(
        clusters=("site_cluster", "size"), within_1km=("w1", "sum"),
        median_km=("km_to_nearest_training", "median")).round(3).to_string())
    print("\n  by study_scope:")
    print(adj.assign(w1=km <= 1.0).groupby("study_scope").agg(
        clusters=("site_cluster", "size"), within_1km=("w1", "sum"),
        median_km=("km_to_nearest_training", "median")).round(3).to_string())
    common.write(adj.sort_values("km_to_nearest_training"),
                 "holdout_adjacency.csv")

    print("\n  what each buffer would cost the training side:")
    cost, train = buffer_cost(sites, held)
    print(cost.to_string(index=False))

    if args.write_buffer:
        excluded = train[train["km_to_test"] <= args.buffer_km].copy()
        out = excluded[["station_id", "site_cluster", "state", "study_scope"]]
        out = out.assign(km_to_nearest_test=excluded["km_to_test"].round(3),
                         buffer_km=args.buffer_km)
        out.sort_values("km_to_nearest_test").to_csv(BUFFER_PATH, index=False)
        print(f"\n  wrote {BUFFER_PATH}  ({len(out)} training stations "
              f"excluded at {args.buffer_km} km)")
        print(f"  sha256 {holdout.file_sha256(BUFFER_PATH)}")
        print("  wq/holdout_sites.csv is unchanged: "
              f"{holdout.file_sha256(holdout.HOLDOUT_PATH)}")


if __name__ == "__main__":
    main()
