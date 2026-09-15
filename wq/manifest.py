"""A1: write the specification down, with a timestamp, before anything is fit.

A pre-registration that nothing checks is a comment. This one has teeth:

  * wq/fit.py calls require_manifest() and stops if the file is absent or if
    its spec_hash disagrees with wq/config.py. Editing the predictor list to
    include the one that worked, adding a covariate that tidies up the
    distribution, or nudging MIN_SAMPLES_PER_SITE down until a site
    qualifies, fails the run instead of quietly producing a better-looking
    answer.
  * the coverage rule is applied HERE, before the fit: a stratum or a site
    covariate populated for fewer sites than its threshold is DROPPED from
    the specification and the manifest records the observed coverage that
    dropped it. Adding one afterwards is a different entry with a later
    timestamp, marked exploratory, and the two are trivially diffable.
  * every spatial layer's source and vintage is written down beside the
    covariates it supplied, because NLCD 2011 and NLCD 2021 are not the same
    covariate and a coefficient stratified on one is not a result about the
    other.

    python -m wq.manifest --show
    python -m wq.manifest --amend fetch_km_NE --why "..." --by "riley"

AMENDING. A covariate thought of after the fact is not forbidden, it is
labelled. --amend appends an entry with its own timestamp, marks it
exploratory, and leaves the registered entry untouched. Everything computed
against an amended manifest is reported as an exploratory pass; the
pre-registered result is whatever entry 0 says it is.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

from . import config, layers, strata

MANIFEST_VERSION = 2


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _covariate_coverage(spatial, sites):
    """Observed coverage per pre-registered site covariate."""
    frame = spatial
    if frame is None or frame.empty:
        if sites is None or sites.empty:
            return {}, list(config.SITE_COVARIATES)
        frame = sites
    observed, dropped = {}, []
    total = len(frame)
    for name in config.SITE_COVARIATES:
        share = (float(frame[name].notna().mean())
                 if name in frame.columns and total else 0.0)
        observed[name] = round(share, 4)
        if share < config.COVARIATE_MIN_COVERAGE:
            dropped.append(name)
    return observed, dropped


def build(sites=None, spatial=None, layer_record=None):
    """The registered entry. `sites` is the stratified station table."""
    kept_strata, dropped_strata, strata_coverage = [], [], {}
    if sites is not None:
        table = strata.coverage(sites)
        for _, row in table.iterrows():
            strata_coverage[row["stratum"]] = round(float(row["coverage"]), 4)
            (kept_strata if row["keeps"] else dropped_strata).append(row["stratum"])
    else:
        kept_strata = list(config.STRATA)

    covariate_coverage, dropped_covariates = _covariate_coverage(spatial, sites)
    kept_covariates = [c for c in config.SITE_COVARIATES
                       if c not in dropped_covariates]

    return {
        "entry": 0,
        "kind": "pre-registered",
        "exploratory": False,
        "manifest_version": MANIFEST_VERSION,
        "written_at": _now(),
        "spec_hash": config.spec_hash(),
        "n_sites_at_registration": None if sites is None else int(len(sites)),
        "specification": config.specification(),
        "layers": layers.manifest_section(layer_record),
        "strata": {
            "declared": list(config.STRATA),
            "kept": kept_strata,
            "dropped_for_coverage": dropped_strata,
            "observed_coverage": strata_coverage,
            "evaluated": sites is not None,
            "manual": [name for name, spec in config.STRATA.items()
                       if not spec.get("automated", True)],
            "rule": ("a stratum populated for a smaller share of qualifying "
                     "sites than its required_coverage is dropped BEFORE "
                     "fitting, per A2"),
        },
        "site_covariates": {
            "declared": list(config.SITE_COVARIATES),
            "kept": kept_covariates,
            "dropped_for_coverage": dropped_covariates,
            "observed_coverage": covariate_coverage,
            "evaluated": spatial is not None and not spatial.empty,
            "minimum_coverage": config.COVARIATE_MIN_COVERAGE,
            "rule": ("a covariate populated for under "
                     f"{config.COVARIATE_MIN_COVERAGE:.0%} of stations is "
                     "dropped rather than fitted on a biased subset"),
        },
        "commitments": [
            "Coefficients are fitted per site, per analyte, per predictor. "
            "No pooled or hierarchical model is fitted in this pass.",
            "The deliverable is the DISTRIBUTION of site-level coefficients: "
            "median, IQR, 10th/90th, min, max, n sites. Not a mean.",
            "Sign agreement is reported as a secondary statistic only.",
            "Every coefficient is reported with its n beside it, and the n "
            "used in each fit is asserted against the count of joined rows "
            "with both sides non-null.",
            "beach_type is assigned by hand from imagery, with the assigner "
            "and date recorded. The continuous enclosure covariates sort the "
            "review list and never assign the label.",
            "No covariate is added after seeing which grouping tidies up the "
            "coefficient distribution. A later addition is a separate entry, "
            "with its own timestamp, reported as exploratory.",
            "Sites whose non-detect share exceeds "
            f"{config.NONDETECT_FLAG_FRACTION:.0%} are reported separately "
            "from the headline distribution.",
            "The count of sites with NO usable predictor is reported "
            "alongside the successes.",
            "Expected false positives at alpha are reported beside the "
            "observed count of significant sites.",
        ],
    }


def _entries(payload):
    """Manifests are a list of entries. A v1 file is a single entry."""
    if payload is None:
        return []
    if isinstance(payload, dict) and "entries" in payload:
        return payload["entries"]
    return [payload]


def write(sites=None, path=None, spatial=None, layer_record=None):
    path = path or config.MANIFEST_PATH
    entry = build(sites, spatial, layer_record)
    payload = {"manifest_version": MANIFEST_VERSION, "entries": [entry]}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {path}")
    _describe(entry)
    return payload


def _describe(entry):
    print(f"  spec_hash    {entry['spec_hash']}")
    print(f"  written_at   {entry['written_at']}")
    print(f"  analytes     {', '.join(config.ANALYTES)}")
    print(f"  predictors   {len(config.PREDICTORS)}")
    print(f"  min samples  {config.MIN_SAMPLES_PER_SITE} per site")

    section = entry["strata"]
    if section["evaluated"]:
        print(f"  strata kept  {', '.join(section['kept']) or 'none'}")
        for name in section["dropped_for_coverage"]:
            share = section["observed_coverage"].get(name, 0.0)
            need = config.STRATA[name]["required_coverage"]
            manual = "" if config.STRATA[name].get("automated", True) else \
                " (hand-assigned — run python -m wq.review --worklist)"
            print(f"  DROPPED      {name}: populated for {share:.0%} of sites, "
                  f"needs {need:.0%}{manual}")
    else:
        print("  strata       NOT EVALUATED — no station table was supplied, "
              "so this manifest cannot authorise a fit")

    covariates = entry["site_covariates"]
    if covariates["evaluated"]:
        print(f"  covariates   {len(covariates['kept'])}/"
              f"{len(covariates['declared'])} clear "
              f"{config.COVARIATE_MIN_COVERAGE:.0%} coverage")
        for name in covariates["dropped_for_coverage"]:
            share = covariates["observed_coverage"].get(name, 0.0)
            print(f"  DROPPED      {name}: {share:.0%}")
    else:
        print("  covariates   NOT EVALUATED — run --spatial before --manifest, "
              "or the coverage rule has nothing to apply")

    print("  layers:")
    for key, layer in entry["layers"].items():
        vintage = layer.get("observed_vintage") or layer["declared_vintage"]
        mark = "" if layer["endpoint_verified_against_live_response"] else \
            "  [endpoint unverified]"
        print(f"    {key:<14} {str(vintage)[:46]:<46} "
              f"{layer['sites_populated']}/{layer['sites_attempted']}{mark}")


def read(path=None):
    path = path or config.MANIFEST_PATH
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def registered_entry(payload):
    for entry in _entries(payload):
        if not entry.get("exploratory"):
            return entry
    return None


def require_manifest(path=None):
    """The guard. Returns the registered entry, or exits explaining what is
    wrong."""
    path = path or config.MANIFEST_PATH
    payload = read(path)
    entry = registered_entry(payload)
    if entry is None:
        sys.exit(
            f"No registered entry in {os.path.basename(path)}. The "
            "specification has to be written down before any coefficient is "
            "computed:\n"
            "    python -m wq.run_wq --stations --spatial --strata --manifest")
    if entry.get("spec_hash") != config.spec_hash():
        sys.exit(
            "wq/config.py has changed since the manifest was written.\n"
            f"    manifest {entry.get('spec_hash')}\n"
            f"    config   {config.spec_hash()}\n"
            "This is the pre-registration doing its job. Either put the "
            "constant back, or re-register deliberately (keeping the old "
            "manifest alongside, so the change is visible):\n"
            "    mv wq_manifest.json wq_manifest.$(date +%s).json\n"
            "    python -m wq.run_wq --stations --spatial --strata --manifest\n"
            "To ADD a covariate without disturbing the registered pass:\n"
            '    python -m wq.manifest --amend <covariate> --why "..." '
            '--by "your name"')
    if not entry.get("strata", {}).get("evaluated"):
        sys.exit("The manifest was written without a station table, so its "
                 "coverage was never evaluated. Re-run --strata --manifest "
                 "before fitting.")
    return entry


def amend(covariates, why, who, path=None):
    """Append an exploratory entry. The registered one is never touched."""
    path = path or config.MANIFEST_PATH
    payload = read(path)
    if payload is None:
        sys.exit("no manifest to amend")
    entries = _entries(payload)
    if not why or not who:
        sys.exit("--why and --by are both required: an amendment with no "
                 "reason and no author is indistinguishable from the thing "
                 "this file exists to prevent")
    entry = {
        "entry": len(entries),
        "kind": "exploratory amendment",
        "exploratory": True,
        "written_at": _now(),
        "spec_hash": config.spec_hash(),
        "added_covariates": list(covariates),
        "why": why,
        "added_by": who,
        "layers": layers.manifest_section(),
        "warning": ("Results using these covariates are EXPLORATORY. They "
                    "were added after the registered pass and must be "
                    "reported as such, separately from the pre-registered "
                    "distribution."),
    }
    entries.append(entry)
    with open(path, "w") as handle:
        json.dump({"manifest_version": MANIFEST_VERSION, "entries": entries},
                  handle, indent=2)
    print(f"appended exploratory entry {entry['entry']} to {path}")
    print(f"  added    {', '.join(covariates)}")
    print(f"  by       {who} at {entry['written_at']}")
    print("  Anything fitted with these is an exploratory pass and the report "
          "must say so.")
    return entry


def exploratory_covariates(payload=None):
    payload = payload if payload is not None else read()
    out = []
    for entry in _entries(payload):
        if entry.get("exploratory"):
            out.extend(entry.get("added_covariates") or [])
    return out


def active_strata(entry=None):
    """The groupings D2 may break the distribution out by.

    The intersection of three things: registered as a grouping
    (config.STRATIFY_ON), declared as a stratum or a site covariate, and
    actually populated for enough stations to survive the coverage rule.
    A covariate that is recorded but not registered as a grouping -- a
    direction, a data-quality figure -- never reaches the report's D2.
    """
    entry = entry or require_manifest()
    kept = (list(entry["strata"]["kept"])
            + list(entry.get("site_covariates", {}).get("kept", [])))
    return [name for name in config.STRATIFY_ON if name in kept]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--amend", nargs="+", metavar="COVARIATE")
    parser.add_argument("--why")
    parser.add_argument("--by")
    args = parser.parse_args()

    if args.amend:
        amend(args.amend, args.why, args.by)
        return
    if args.show:
        payload = read()
        if payload is None:
            sys.exit("no manifest yet")
        print(json.dumps(payload, indent=2))
        return
    parser.error("give --show or --amend. The manifest is WRITTEN by "
                 "wq.run_wq --manifest, which supplies the station table and "
                 "the layer record it needs.")


if __name__ == "__main__":
    main()
