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
from datetime import datetime, timezone

import pandas as pd

from . import (clean, config, covariates, fit, holdout, keys, layers,
               manifest, pull, report, review, scope as study_scope, spatial,
               strata)

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
    return keys.coerce(pd.read_csv(path, low_memory=False))


def _write(frame, name):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    path = _path(name)
    frame.to_csv(path, index=False)
    print(f"wrote {path}  ({len(frame)} rows)")
    return path


def stage_stations(args):
    print("\n=== STATIONS ===")
    pull.pull_stations(args.states, refresh=args.refresh, probe=args.probe,
                       allow_failed=args.allow_failed_states)


def stage_spatial(args):
    print("\n=== SPATIAL COVARIATES (A2 addition) ===")
    print("Derived from the station coordinate alone. Nothing here reads a")
    print("sample value, and beach_type is NOT assigned here — it is assigned")
    print("by hand from imagery, via python -m wq.review --worklist.")
    every = _read(STATIONS, "run --stations first")
    sites = _analysable(every, args)
    record = layers.blank_record()

    want = set(layers.LAYERS)
    if getattr(args, "skip_layers", None):
        skipped = {s.strip() for s in args.skip_layers.split(",") if s.strip()}
        # Only the layers with a request of their own can be skipped. nlcd and
        # nhdplus_vaa ride inside the NHDPlus response, so naming them here
        # would print a reassuring message and change nothing.
        unknown = skipped - set(layers.FETCHED)
        if unknown:
            derived = {k for k in unknown if k in layers.DERIVED_FROM}
            if derived:
                sys.exit("; ".join(
                    f"{k} has no request of its own — it comes back inside "
                    f"{layers.DERIVED_FROM[k]}, so skip that instead"
                    for k in sorted(derived)))
            sys.exit(f"not layers: {sorted(unknown)}. "
                     f"Known: {sorted(layers.FETCHED)}")
        want -= skipped
        print(f"  skipping {', '.join(sorted(skipped))} at your request — "
              "their covariates will be empty and the coverage rule will "
              "drop them, exactly as if the service had refused")
        for key in sorted(skipped):
            layers.record_access(record, key,
                                 note="skipped by --skip-layers on this run")
    frame, record = spatial.build(sites, record, want=want)

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
    spatial.report_outcomes(frame, want=want)
    _write(frame, SPATIAL)
    with open(_path(LAYER_RECORD), "w") as handle:
        json.dump(record, handle, indent=2)
    print(f"wrote {_path(LAYER_RECORD)}")


def stage_explain_spatial(args):
    """Read the coverage zeros back off the file, without touching a service.

    The reasons are written into site_covariates.csv as they happen, so this
    can answer "why is coastline empty" for a run that finished hours ago.
    """
    print("\n=== WHY THE SPATIAL LAYERS CAME BACK THAT WAY ===")
    frame = _read(SPATIAL, "run --spatial first")
    print(f"{len(frame)} stations in {SPATIAL}")
    print("\ncoverage:")
    print(spatial.coverage(frame).to_string(index=False))
    spatial.report_outcomes(frame)


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
    pull.pull_results(args.states, refresh=args.refresh, probe=args.probe,
                      allow_failed=args.allow_failed_states)


def stage_ckan(args):
    print("\n=== CALIFORNIA CKAN ===")
    pull.pull_ca_ckan()


def stage_clean(args):
    print("\n=== HYGIENE ===")
    sites = _read(STATIONS, "run --stations first")
    raw = pull.load_raw_results(
        args.states,
        allow_widening=getattr(args, "allow_scope_widening", False))
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
    if getattr(args, "refetch_empty", False):
        removed = covariates.clear_empty_cache()
        print(f"  cleared {removed} cached 'nothing here' answer(s); they are "
              "asked again this run")
    overrides = strata.load_overrides()
    joined, meta = covariates.build(wanted, samples, coops, overrides)
    if joined.empty:
        sys.exit("no site produced covariates")
    _guard_degraded_covariates(joined, meta, args)
    _write(joined, JOINED)
    _write(meta, COVARIATE_META)


BUILD_STATUS = "covariates_build_status.json"


def _write_build_status(coverage, lost, passed, reasons, accepted=False):
    """The record --fit reads. A build that half-happened has to leave a
    machine-readable trace, because the tables it writes look exactly like a
    complete run's and scrollback does not survive the night."""
    # A source that did not answer and a value deliberately withheld are not
    # the same failure, and only one of them is the operator's to waive.
    #
    # An ABSOLUTE failure means a source that should have answered did not --
    # a spent quota, a dead service -- and no flag makes that acceptable, so
    # it stays FAIL however the run was invoked.
    #
    # A REGRESSION is only ever a comparison with the previous file, and the
    # previous file can be the wrong one. That is exactly what happened here:
    # 372 stations stopped carrying a fabricated shore normal, so
    # wind_onshore_ms fell from 0.998 to 0.893 and the guard called it a loss.
    # It was a correction. The guard cannot know that and should not guess, so
    # it stops and asks; when the operator says the loss is intended, the
    # answer is recorded rather than the question left blocking everything
    # downstream.
    absolute_failed = bool((~coverage["passed"]).any())
    status = "PASS" if passed else "FAIL"
    if not passed and accepted and not absolute_failed:
        status = "PASS"
    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "accepted_regression": bool(accepted and not passed
                                    and not absolute_failed),
        "absolute_check_failed": absolute_failed,
        "reasons": reasons,
        "sources": [
            {k: (None if (isinstance(v, float) and pd.isna(v)) else v)
             for k, v in row.items() if not k.startswith("_")}
            for _, row in coverage.iterrows()
        ],
        "coverage_regression": [] if lost is None or lost.empty
        else lost.to_dict("records"),
    }
    path = _path(BUILD_STATUS)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {path}  ({payload['status']})")
    return payload


def read_build_status():
    path = _path(BUILD_STATUS)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _guard_degraded_covariates(joined, meta, args):
    """Refuse to replace a good covariate file with a worse one.

    covariates.run()'s docstring states the problem plainly: a run that hits
    the Open-Meteo daily quota at site 600 writes exactly the same tables as a
    run that got everything. Every site after the breaker trips is built with
    no rain, no wind and no waves; the coverage rule drops those covariates;
    and the only record that anything went wrong is a line of scrollback.

    Two independent signals, because either can happen without the other:
    a host given up on mid-run, and covariate coverage going BACKWARDS on the
    stations the previous file already held. The second is the one that
    matters -- it is measured from the data rather than inferred from the
    breaker -- and it is restricted to shared stations so that adding a state
    does not look like a regression.

    Nothing is deleted. The new frame is written beside the old one with a
    .degraded suffix so it can be inspected, and the canonical file is left
    exactly as it was.
    """
    tripped = covariates.unavailable_sources()
    previous = None
    path = _path(JOINED)
    if os.path.exists(path):
        previous = keys.coerce(pd.read_csv(path, low_memory=False))
    lost = covariates.regression_against(joined, previous)

    # The absolute check. regression_against only sees stations a previous
    # file already held; this one judges the stations in scope NOW, which is
    # the half that California falls in.
    sites = _read(STRATIFIED, "run --strata first")
    coverage = covariates.source_coverage(joined, sites, meta)
    print("\ncovariate coverage, per source:")
    print(coverage[[c for c in coverage.columns
                    if not c.startswith("_")]].to_string(index=False))
    failed = coverage[~coverage["passed"]]
    reasons = []
    if not failed.empty:
        reasons.extend(
            f"{row['source']}: {row['stations_with_data']}/"
            f"{row['eligible_stations']} eligible stations have data"
            + (f" ({row['coverage']:.1%} < {row['required']:.0%})"
               if pd.notna(row["coverage"]) and row["required"] else "")
            for _, row in failed.iterrows())
    if not lost.empty:
        reasons.extend(f"{row['predictor']}: coverage fell "
                       f"{row['coverage_before']:.3f} -> {row['coverage_now']:.3f}"
                       for _, row in lost.iterrows())
    passed = failed.empty and lost.empty
    accepted = bool(getattr(args, "allow_degraded_covariates", False))
    _write_build_status(coverage, lost, passed, reasons, accepted=accepted)

    if not failed.empty:
        print("\n" + "!" * 78)
        print("A REQUIRED COVARIATE SOURCE DID NOT ANSWER — NOT WRITING")
        print("!" * 78)
        for _, row in failed.iterrows():
            print(f"\n  {row['source']}: {row['stations_missing']:,} of "
                  f"{row['eligible_stations']:,} eligible station(s) have no "
                  f"value.\n    eligibility: {row['eligibility']} — "
                  f"{row['why']}")
            if row["note"]:
                print(f"    {row['note']}")
        cells = covariates.missing_cells(failed, sites)
        if not cells.empty:
            print(f"\n  the {len(cells)} grid cell(s) behind them:")
            print(cells.head(40).to_string(index=False))
            if len(cells) > 40:
                print(f"    ... and {len(cells) - 40} more")
        sample = failed.iloc[0]["_missing_stations"][:12]
        print(f"\n  example missing stations: {', '.join(sample)}")

    if tripped:
        print(f"\n  {len(tripped)} source(s) were given up on during this run:")
        for host, reason in sorted(tripped.items()):
            print(f"    {host}: {reason}")
    if lost.empty and failed.empty:
        if tripped:
            print("  Coverage on the stations the previous file already held "
                  "did NOT go\n  backwards, so the run is kept. Read the "
                  "per-site source table for what\n  those hosts cost the "
                  "NEW stations.")
        return

    if not lost.empty:
        print("\n" + "!" * 78)
        print("COVARIATE COVERAGE WENT BACKWARDS — NOT WRITING")
        print("!" * 78)
        print("  On stations the previous covariate file already held, this "
              "run produced\n  fewer values than that file did. Those "
              "stations' grid cells are cached, so\n  a healthy re-run "
              "reproduces them exactly. This is this run losing something\n"
              "  it already had — a spent quota, a service that went down, or "
              "a pull that\n  stopped early.")
        print()
        print(lost.to_string(index=False))
    degraded = path.replace(".csv", ".degraded.csv")
    joined.to_csv(degraded, index=False)
    print(f"\n  wrote the degraded frame to {degraded} for inspection")
    print(f"  LEFT ALONE: {path}")
    if getattr(args, "allow_degraded_covariates", False):
        print("\n  --allow-degraded-covariates was given, so it is written "
              "anyway and this\n  is on the record as a choice.")
        return
    sys.exit(
        "\nStopping. The cached cells mean a re-run costs only what is "
        "missing:\n"
        "    python -m wq.cell_budget          # how many cells are still to buy\n"
        "    python -m wq.run_wq --covariates  # resumes; cached cells are free\n"
        "If the loss is real and expected, say so with "
        "--allow-degraded-covariates.")


def _require_complete_covariates(args):
    """--fit will not run on a covariate build the pipeline judged incomplete.

    The status file exists because the tables a half-finished build writes are
    indistinguishable from a complete one's. Refusing here is the last place
    that difference can still be acted on: once coefficients exist, a missing
    rain column looks like a coast with no rain.
    """
    status = read_build_status()
    if status is not None and status.get("status") == "PASS":
        return
    where = _path(BUILD_STATUS)
    if status is None:
        problem = (f"no {BUILD_STATUS} — the covariate stage has not been run "
                   "since this guard existed")
        reasons = []
    else:
        problem = f"{BUILD_STATUS} says {status.get('status')}"
        reasons = status.get("reasons", [])
    if not getattr(args, "allow_incomplete_covariates", False):
        message = [f"\nRefusing to fit: {problem}.", f"  {where}"]
        message.extend(f"    - {r}" for r in reasons)
        message.append(
            "\nA build that stopped early writes the same tables as one that "
            "finished, so this\nis the last point at which the difference can "
            "be acted on. Cached cells make a\nre-run cost only what is "
            "missing:\n"
            "    python -m wq.cell_budget\n"
            "    python -m wq.run_wq --covariates\n"
            "To fit anyway, on the record:\n"
            "    python -m wq.run_wq --fit --allow-incomplete-covariates "
            '--why "..."')
        sys.exit("\n".join(message))
    why = getattr(args, "why", None)
    if not why:
        sys.exit("--allow-incomplete-covariates requires --why: an override "
                 "with no reason is indistinguishable from an accident")
    print("\n" + "!" * 78)
    print("FITTING ON AN INCOMPLETE COVARIATE BUILD")
    print(f"  {problem}")
    for reason in reasons:
        print(f"    - {reason}")
    print("  Every coefficient from this run was computed on inputs the")
    print("  pipeline judged incomplete. Report it that way, or throw it away.")
    print("!" * 78)
    manifest.record_override(
        "fit on incomplete covariates", why,
        {"status_file": where, "reasons": reasons,
         "status": None if status is None else status.get("status")})


def stage_fit(args):
    print("\n=== FIT ===")
    _require_complete_covariates(args)
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
    coefficients = keys.coerce(
        pd.read_csv(os.path.join(config.OUT_DIR, COEFFICIENTS), low_memory=False))
    attrition = keys.coerce(
        pd.read_csv(os.path.join(config.OUT_DIR, ATTRITION), low_memory=False))
    shares = _read(NONDETECTS, "run --clean first")
    meta_path = _path(COVARIATE_META)
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        keep = [c for c in ("station_id", "shore_normal_source", "tide_station")
                if c in meta.columns]
        sites = sites.merge(meta[keep], on="station_id", how="left")
    if coefficients.empty:
        sys.exit("no coefficients to report — the fit produced nothing")

    # D1-D6 describe the DEVELOPMENT set. Computing the pre-registered
    # distribution over the held-out clusters would spend the holdout on a
    # description -- once those beaches are in a printed IQR, whatever is
    # decided next is decided partly on them, and the Phase 2 comparison they
    # exist for is no longer clean. The fit itself still produces a
    # coefficient for every station; only the report's view is narrowed, and
    # the attrition table is narrowed with it so its station counts still tie
    # to D1's n_sites.
    # --scope restricts the population BEFORE anything else, and redirects the
    # outputs, so a beach-only D1 can never overwrite the pooled one.
    out_dir = config.OUT_DIR
    scope_name = getattr(args, "scope", "all") or "all"
    if scope_name != "all":
        before_scope = len(coefficients)
        coefficients = study_scope.restrict(coefficients, scope_name)
        kept_ids = set(coefficients["station_id"].astype(str))
        attrition = attrition[attrition["station_id"].astype(str).isin(kept_ids)]
        out_dir = os.path.join(config.OUT_DIR, f"{scope_name}_only")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n  scope={scope_name}: {len(coefficients):,} of "
              f"{before_scope:,} coefficient rows. The pooled outputs in "
              f"{config.OUT_DIR}\n  are NOT touched; this run writes to "
              f"{out_dir}")
        if coefficients.empty:
            sys.exit(f"no coefficients in scope {scope_name}")

    evaluate = getattr(args, "evaluate_holdout", False)
    before = len(coefficients)
    coefficients = holdout.drop_holdout(coefficients, evaluate_holdout=evaluate,
                                        quiet=True)
    held = holdout.read_holdout()
    if evaluate:
        print(f"\n  --evaluate-holdout: D1-D6 below INCLUDE the "
              f"{held['site_cluster'].nunique()} held-out cluster(s).")
        print("  This is a held-out evaluation and must be reported as one, "
              "once.")
    elif len(coefficients) != before:
        kept = set(coefficients["station_id"].astype(str))
        attrition = attrition[attrition["station_id"].astype(str).isin(kept)]
        print(f"\n  holdout: D1-D6 describe the development set — "
              f"{before - len(coefficients):,} coefficient row(s) from "
              f"{held['site_cluster'].nunique()} held-out cluster(s) are "
              f"excluded\n  (seed {holdout.HOLDOUT_SEED}; "
              "--evaluate-holdout to include them)")
    if coefficients.empty:
        sys.exit("every coefficient belongs to a held-out cluster — nothing "
                 "to report on the development set")
    groupings = manifest.active_strata(payload)
    exploratory = [name for name in manifest.exploratory_covariates()
                   if name in sites.columns and name not in groupings]
    if exploratory:
        print(f"  {len(exploratory)} exploratory grouping(s) from a manifest "
              f"amendment: {', '.join(exploratory)}")
    report.run(coefficients, sites, attrition, shares, groupings + exploratory,
               out_dir=out_dir, exploratory=exploratory)


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
    parser.add_argument("--allow-scope-widening", action="store_true",
                        help="let --clean load result chunks for states the "
                             "study has not adopted. A scope change belongs "
                             "in the manifest; this is the escape hatch.")
    parser.add_argument("--allow-failed-states", action="store_true",
                        help="continue after a state or chunk fails, instead "
                             "of stopping")
    parser.add_argument("--refresh", action="store_true",
                        help="re-pull chunks already on disk")
    parser.add_argument("--probe", action="store_true",
                        help="print the real column names and stop")
    parser.add_argument("--lenient", action="store_true",
                        help="warn instead of failing on an n mismatch. Do not "
                             "use this to get a run to finish.")
    parser.add_argument("--skip-layers",
                        help="comma-separated layer keys not to fetch, e.g. "
                             "echo. Their covariates come out empty and are "
                             "dropped by the coverage rule.")
    parser.add_argument("--review-n", type=int,
                        help="limit the beach_type review list")
    parser.add_argument("--refetch-empty", action="store_true",
                        help="before --covariates, delete the cached 'nothing "
                             "here' answers so they are asked again. For after "
                             "a run that was rate-limited: a refusal cached as "
                             "an absence never expires on its own.")
    parser.add_argument("--scope", default="all",
                        choices=["all", "beach", "shellfish", "unclassified"],
                        help="restrict --report to one study population. "
                             "Scoped runs write to data/wq/out/<scope>_only/ "
                             "and never touch the pooled outputs.")
    parser.add_argument("--evaluate-holdout", action="store_true",
                        help="let --report include the held-out clusters. "
                             "Only valid as a deliberate one-shot evaluation; "
                             "the default describes the development set.")
    parser.add_argument("--allow-incomplete-covariates", action="store_true",
                        help="fit even though the covariate build did not "
                             "pass. Requires --why and writes an override "
                             "entry into the manifest.")
    parser.add_argument("--why", help="reason for an override, recorded in "
                                      "the manifest")
    parser.add_argument("--allow-degraded-covariates", action="store_true",
                        help="write the covariate file even when coverage "
                             "went backwards against the previous one. For "
                             "when the loss is real and expected.")
    parser.add_argument("--no-datums", action="store_true",
                        help="skip the CO-OPS datums pull (tidal_range_m is "
                             "then dropped by the coverage rule)")
    parser.add_argument("--no-caffeinate", action="store_true",
                        help="do not re-exec under a sleep inhibitor")
    # Deliberately not a stage: it is a reader, it touches no service, and it
    # must never be part of --all.
    parser.add_argument("--explain-spatial", action="store_true",
                        help="say why each spatial layer is empty, from the "
                             "file a previous --spatial already wrote")
    args = parser.parse_args()

    if args.explain_spatial:
        stage_explain_spatial(args)
        return

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
