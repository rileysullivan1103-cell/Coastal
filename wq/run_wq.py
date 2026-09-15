"""The whole pass, in the order the pre-registration requires.

    ./wq/run_wq.sh --all                  # caffeinated, the way to run it
    python -m wq.run_wq --all
    python -m wq.run_wq --stations --strata --manifest
    python -m wq.run_wq --results --states CA,FL
    python -m wq.run_wq --fit --report

CAFFEINATE. The national pull is hours of requests, and a Mac that sleeps
half way through leaves a part-written chunk and a stalled socket. This
script re-executes itself under `caffeinate -i` on macOS before it does
anything else, so the protection is there however it was started -- through
run_wq.sh, through python -m, or from a scheduler. --no-caffeinate opts out.
On Linux it uses systemd-inhibit where that exists and otherwise carries on,
because a headless box has nothing to keep awake.

ORDER IS NOT A SUGGESTION. Stations and strata come before the manifest,
because A2 requires the coverage rule to drop under-populated strata BEFORE
fitting. The manifest comes before the results pull and everything after it,
because A1 requires the specification to be on disk with a timestamp before
any model runs. wq/fit.py enforces its half of that with require_manifest();
this script enforces the rest by refusing to run a later stage when an
earlier artefact is missing.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import pandas as pd

from . import (clean, config, covariates, fit, layers, manifest, pull,
               report, review, spatial, strata)

CAFFEINATE_FLAG = "WQ_CAFFEINATED"

STATIONS = "stations.csv"
SPATIAL = "site_covariates.csv"
LAYER_RECORD = "layer_record.json"
STRATIFIED = "stations_stratified.csv"
SAMPLES = "samples_clean.csv"
NONDETECTS = "nondetect_shares.csv"
JOINED = "samples_with_covariates.csv"
COVARIATE_META = "covariate_sources.csv"
COEFFICIENTS = "coefficients.csv"
ATTRITION = "attrition.csv"


def keep_awake(argv=None):
    """Re-exec under a sleep inhibitor. Returns only if already wrapped."""
    if os.environ.get(CAFFEINATE_FLAG):
        return
    argv = argv or sys.argv
    if "--no-caffeinate" in argv:
        return
    env = dict(os.environ, **{CAFFEINATE_FLAG: "1"})
    # -i prevents idle sleep, -m keeps the disk spun up. Deliberately NOT -d:
    # the display may sleep, only the machine may not.
    candidates = [["caffeinate", "-i", "-m"],
                  ["systemd-inhibit", "--what=idle:sleep",
                   "--why=national water-quality pull"]]
    for prefix in candidates:
        if not shutil.which(prefix[0]):
            continue
        # Probe before exec. systemd-inhibit is present inside containers that
        # have no session bus to inhibit, and it exits non-zero there -- and
        # os.execvpe has no way back, so an unprobed exec would end the run
        # instead of protecting it.
        try:
            probe = subprocess.run(prefix + ["true"], timeout=10,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode != 0:
            print(f"({prefix[0]} is installed but not working here — "
                  "carrying on without it)")
            continue
        arguments = prefix + [sys.executable, "-m", "wq.run_wq"] + argv[1:]
        print(f"(re-running under {prefix[0]} so the machine stays awake)")
        os.execvpe(arguments[0], arguments, env)
    print("(no working sleep inhibitor here — carrying on uncaffeinated)")


def _path(name):
    return os.path.join(config.DATA_DIR, name)


def _read(name, what):
    path = _path(name)
    if not os.path.exists(path):
        sys.exit(f"{path} is missing — {what}")
    return pd.read_csv(path, low_memory=False)


def _write(frame, name):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    path = _path(name)
    frame.to_csv(path, index=False)
    print(f"wrote {path}  ({len(frame)} rows)")
    return path


def stage_stations(args):
    print("\n=== STATIONS ===")
    pull.pull_stations(args.states, refresh=args.refresh, probe=args.probe)


def stage_spatial(args):
    print("\n=== SPATIAL COVARIATES (A2 addition) ===")
    print("Derived from the station coordinate alone. Nothing here reads a")
    print("sample value, and beach_type is NOT assigned here — it is assigned")
    print("by hand from imagery, via python -m wq.review --worklist.")
    every = _read(STATIONS, "run --stations first")
    sites = _analysable(every, args)
    record = layers.blank_record()
    frame, record = spatial.build(sites, record)

    datums = _coops_datums(sites, args)
    if datums is not None and not datums.empty:
        frame = spatial.add_tidal_datums(frame, sites, datums)
        record["coops_datums"]["sites_attempted"] = len(sites)
        record["coops_datums"]["sites_populated"] = int(
            frame["tidal_range_m"].notna().sum())
        layers.record_access(record, "coops_datums")
        datums.to_csv(_path("coops_datums.csv"), index=False)

    print("\ncoverage (the ~70% rule drops anything under it):")
    print(spatial.coverage(frame).to_string(index=False))
    _write(frame, SPATIAL)
    with open(_path(LAYER_RECORD), "w") as handle:
        json.dump(record, handle, indent=2)
    print(f"wrote {_path(LAYER_RECORD)}")


def _analysable(sites, args):
    """The stations worth spending requests on: those with enough samples.

    A station that cannot reach MIN_SAMPLES_PER_SITE is excluded from the
    fit by C2 whatever its coastline looks like, so computing its shore
    normal buys nothing and costs a request to a service that is free and
    shared. The attrition table still names every station that was pulled,
    so nothing disappears quietly -- these stations are reported as excluded
    on sample count, which is what they are.
    """
    path = _path(SAMPLES)
    if not os.path.exists(path):
        print("  no samples_clean.csv — computing spatial covariates for ALL "
              f"{len(sites)} stations.\n  Run --results and --clean first to "
              "restrict this to the stations that can actually be fitted; at "
              "national scale that is the difference between hours and days, "
              "and between polite and abusive use of a free service.")
        return sites
    samples = pd.read_csv(path, low_memory=False, usecols=["station_id"])
    counts = samples["station_id"].astype(str).value_counts()
    enough = set(counts[counts >= config.MIN_SAMPLES_PER_SITE].index)
    keep = sites[sites["station_id"].astype(str).isin(enough)]
    print(f"  {len(keep)}/{len(sites)} stations clear "
          f"{config.MIN_SAMPLES_PER_SITE} samples and are worth a layer pull")
    if keep.empty:
        sys.exit("No station has enough samples — nothing to compute.")
    return keep


def _coops_datums(sites, args):
    """MHHW/MLLW for every gauge that serves a site, fetched once per gauge."""
    if getattr(args, "no_datums", False):
        return None
    try:
        import scan_cameras as scan
        gauges = scan.load_coops("waterlevels")
        wanted = set()
        for _, site in sites.iterrows():
            station, _km = covariates.nearest_coops(site["lat"], site["lon"],
                                                    gauges)
            if station:
                wanted.add(station)
        print(f"  {len(wanted)} distinct CO-OPS gauges serve these sites")
        datums = covariates.coops_datums(sorted(wanted))
        if datums is not None and not datums.empty:
            coords = gauges.set_index(gauges["station_id"].astype(str))
            keys = datums["station_id"].astype(str)
            datums["lat"] = keys.map(coords["lat"])
            datums["lon"] = keys.map(coords["lon"])
        return datums
    except Exception as exc:  # noqa: BLE001
        print(f"  CO-OPS datums unavailable ({exc}) — tidal_range_m will be "
              "empty and the coverage rule will drop it")
        return None


def stage_review(args):
    print("\n=== beach_type REVIEW LIST ===")
    sites = _read(STATIONS, "run --stations first")
    frame = (pd.read_csv(_path(SPATIAL), low_memory=False)
             if os.path.exists(_path(SPATIAL)) else None)
    if frame is None:
        print("  no site_covariates.csv — the list will not be sorted by "
              "ambiguity. Run --spatial first.")
    table = review.worklist(sites, frame, getattr(args, "review_n", None))
    table.to_csv(review.WORKLIST_PATH, index=False)
    print(f"  {len(table)} station(s) -> {review.WORKLIST_PATH}")
    print("  Assign beach_type from the imagery link, then:")
    print(f'    python -m wq.review --ingest {review.WORKLIST_PATH} '
          '--by "your name"')
    review.status(sites)


def stage_strata(args):
    print("\n=== STRATA (metadata and map data only) ===")
    sites = _analysable(_read(STATIONS, "run --stations first"), args)
    frame = (pd.read_csv(_path(SPATIAL), low_memory=False)
             if os.path.exists(_path(SPATIAL)) else None)
    if frame is None:
        print("  no site_covariates.csv — every spatial covariate will be "
              "empty and the coverage rule will drop all of them. Run "
              "--spatial first.")
    datums_path = _path("coops_datums.csv")
    datums = (pd.read_csv(datums_path) if os.path.exists(datums_path)
              else pd.DataFrame())

    stratified = strata.assign(sites, datums=datums, spatial=frame)
    print("\nstrata coverage:")
    print(strata.coverage(stratified).round(3).to_string(index=False))
    _write(stratified, STRATIFIED)


def stage_manifest(args):
    print("\n=== PRE-REGISTRATION ===")
    sites = _read(STRATIFIED, "run --strata first")
    frame = (pd.read_csv(_path(SPATIAL), low_memory=False)
             if os.path.exists(_path(SPATIAL)) else None)
    record = None
    if os.path.exists(_path(LAYER_RECORD)):
        with open(_path(LAYER_RECORD)) as handle:
            record = json.load(handle)
    manifest.write(sites, spatial=frame, layer_record=record)


def stage_results(args):
    print("\n=== RESULTS ===")
    pull.pull_results(args.states, refresh=args.refresh, probe=args.probe)


def stage_ckan(args):
    print("\n=== CALIFORNIA CKAN ===")
    pull.pull_ca_ckan()


def stage_clean(args):
    print("\n=== HYGIENE ===")
    sites = _read(STATIONS, "run --stations first")
    raw = pull.load_raw_results(args.states)
    ckan_path = _path("ca_ckan_results.csv")
    ckan = pd.read_csv(ckan_path, low_memory=False) if os.path.exists(ckan_path) else None
    if raw.empty and ckan is None:
        sys.exit("no results on disk — run --results (and optionally --ckan)")

    samples, log = clean.clean(raw, ckan, sites)
    log.write()
    shares = clean.nondetect_shares(samples)
    known = set(sites["station_id"].astype(str))
    before = samples["station_id"].nunique()
    samples = samples[samples["station_id"].astype(str).isin(known)]
    print(f"  {before - samples['station_id'].nunique()} stations in the "
          "results had no coastal station record and are dropped")
    _write(samples, SAMPLES)
    _write(shares, NONDETECTS)


def stage_covariates(args):
    print("\n=== COVARIATES ===")
    manifest.require_manifest()
    sites = _read(STRATIFIED, "run --strata first")
    samples = _read(SAMPLES, "run --clean first")

    # Only sites that could possibly clear the pre-registered floor are worth
    # a covariate pull. This is the single biggest saving in the run: a site
    # with four samples costs the same requests as one with four hundred.
    counts = samples.groupby("station_id").size()
    enough = counts[counts >= config.MIN_SAMPLES_PER_SITE].index
    wanted = sites[sites["station_id"].astype(str).isin(set(enough.astype(str)))]
    print(f"  {len(wanted)}/{len(sites)} sites clear "
          f"{config.MIN_SAMPLES_PER_SITE} samples and get a covariate pull")

    coops = None
    try:
        import scan_cameras as scan
        coops = scan.load_coops("waterlevels")
    except Exception as exc:  # noqa: BLE001
        print(f"  CO-OPS station list unavailable ({exc}) — no tide or water "
              "temperature covariates")
    overrides = strata.load_overrides()
    joined, meta = covariates.build(wanted, samples, coops, overrides)
    if joined.empty:
        sys.exit("no site produced covariates")
    _write(joined, JOINED)
    _write(meta, COVARIATE_META)


def stage_fit(args):
    print("\n=== FIT ===")
    payload = manifest.require_manifest()
    sites = _read(STRATIFIED, "run --strata first")
    joined = _read(JOINED, "run --covariates first")
    shares = _read(NONDETECTS, "run --clean first")
    coefficients, attrition = fit.run(joined, sites, shares,
                                      strict=not args.lenient, payload=payload)
    fit.write(coefficients, attrition)


def stage_report(args):
    print("\n=== REPORT ===")
    payload = manifest.require_manifest()
    sites = _read(STRATIFIED, "run --strata first")
    coefficients = pd.read_csv(os.path.join(config.OUT_DIR, COEFFICIENTS))
    attrition = pd.read_csv(os.path.join(config.OUT_DIR, ATTRITION))
    shares = _read(NONDETECTS, "run --clean first")
    meta_path = _path(COVARIATE_META)
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        keep = [c for c in ("station_id", "shore_normal_source", "tide_station")
                if c in meta.columns]
        sites = sites.merge(meta[keep], on="station_id", how="left")
    if coefficients.empty:
        sys.exit("no coefficients to report — the fit produced nothing")
    groupings = manifest.active_strata(payload)
    exploratory = [name for name in manifest.exploratory_covariates()
                   if name in sites.columns and name not in groupings]
    if exploratory:
        print(f"  {len(exploratory)} exploratory grouping(s) from a manifest "
              f"amendment: {', '.join(exploratory)}")
    report.run(coefficients, sites, attrition, shares, groupings + exploratory,
               exploratory=exploratory)


# Order matters, and this is not the order it was first written in.
#
# The spatial layers used to run over every station the WQP pull returned:
# 32,513 of them nationally, at roughly three requests each. That is ~100,000
# requests, thirty hours, and -- for a free community service like Overpass --
# straightforwardly abusive. It is also mostly wasted, because a station with
# four bacteria samples in ten years can never clear the pre-registered floor
# and will never be fitted whatever its coastline looks like.
#
# So the samples come first, and the spatial layers run only for stations
# that could actually enter the study. Nothing about the pre-registration is
# weakened by this: A1 requires the specification to be frozen before any
# MODEL runs, and the manifest is still written before --covariates and
# --fit. The covariate VALUES still come from the coordinate alone; only the
# question of which stations are worth computing them for is informed by the
# sample counts, and that is the same attrition filter C2 already reports.
STAGES = [
    ("stations", stage_stations),
    ("results", stage_results),
    ("ckan", stage_ckan),
    ("clean", stage_clean),
    ("spatial", stage_spatial),
    ("review", stage_review),
    ("strata", stage_strata),
    ("manifest", stage_manifest),
    ("covariates", stage_covariates),
    ("fit", stage_fit),
    ("report", stage_report),
]


def main():
    keep_awake()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, _ in STAGES:
        parser.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--all", action="store_true",
                        help="every stage in order")
    parser.add_argument("--states", help="comma-separated, e.g. CA,FL")
    parser.add_argument("--refresh", action="store_true",
                        help="re-pull chunks already on disk")
    parser.add_argument("--probe", action="store_true",
                        help="print the real column names and stop")
    parser.add_argument("--lenient", action="store_true",
                        help="warn instead of failing on an n mismatch. Do not "
                             "use this to get a run to finish.")
    parser.add_argument("--review-n", type=int,
                        help="limit the beach_type review list")
    parser.add_argument("--no-datums", action="store_true",
                        help="skip the CO-OPS datums pull (tidal_range_m is "
                             "then dropped by the coverage rule)")
    parser.add_argument("--no-caffeinate", action="store_true",
                        help="do not re-exec under a sleep inhibitor")
    args = parser.parse_args()

    if args.states:
        args.states = [s.strip().upper() for s in args.states.split(",") if s.strip()]

    chosen = [(name, run) for name, run in STAGES
              if args.all or getattr(args, name)]
    if not chosen:
        parser.error("give --all or at least one stage")
    for name, stage in chosen:
        stage(args)
    print("\ndone.")


if __name__ == "__main__":
    main()
