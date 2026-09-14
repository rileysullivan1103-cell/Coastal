"""A1: write the specification down, with a timestamp, before anything is fit.

A pre-registration that nothing checks is a comment. This one has teeth:

  * wq/fit.py calls require_manifest() and stops if the file is absent, if its
    spec_hash disagrees with wq/config.py, or if it was written after the
    first covariate join. So editing the predictor list to include the one
    that worked, or nudging MIN_SAMPLES_PER_SITE down until a site qualifies,
    fails the run instead of silently producing a better-looking answer.
  * the strata coverage rule from A2 is applied HERE, before the fit: a
    variable populated for fewer sites than its threshold is dropped from the
    specification and the manifest records the observed coverage that dropped
    it. Adding one afterwards is a different file with a later timestamp, and
    the two are trivially diffable.

    python -m wq.manifest --write     # after the station pull, before fitting
    python -m wq.manifest --show
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

from . import config, strata

MANIFEST_VERSION = 1


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build(sites=None):
    """The manifest dict. `sites` is the stratified station table; without it
    the strata coverage section is recorded as 'not evaluated', which
    require_manifest() treats as a manifest that cannot authorise a fit."""
    spec = config.specification()
    dropped, kept, observed = [], [], {}

    if sites is not None:
        table = strata.coverage(sites)
        for _, row in table.iterrows():
            observed[row["stratum"]] = round(float(row["coverage"]), 4)
            (kept if row["keeps"] else dropped).append(row["stratum"])
    else:
        kept = list(config.STRATA)

    return {
        "manifest_version": MANIFEST_VERSION,
        "written_at": _now(),
        "spec_hash": config.spec_hash(),
        "n_sites_at_registration": None if sites is None else int(len(sites)),
        "specification": spec,
        "strata": {
            "declared": list(config.STRATA),
            "kept": kept,
            "dropped_for_coverage": dropped,
            "observed_coverage": observed,
            "evaluated": sites is not None,
            "rule": ("a stratum populated for a smaller share of qualifying "
                     "sites than its required_coverage is dropped BEFORE "
                     "fitting, per A2"),
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
            "Sites whose non-detect share exceeds "
            f"{config.NONDETECT_FLAG_FRACTION:.0%} are reported separately "
            "from the headline distribution.",
            "The count of sites with NO usable predictor is reported "
            "alongside the successes.",
            "Expected false positives at alpha are reported beside the "
            "observed count of significant sites.",
        ],
    }


def write(sites=None, path=None):
    path = path or config.MANIFEST_PATH
    payload = build(sites)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {path}")
    print(f"  spec_hash    {payload['spec_hash']}")
    print(f"  written_at   {payload['written_at']}")
    print(f"  analytes     {', '.join(config.ANALYTES)}")
    print(f"  predictors   {len(config.PREDICTORS)}")
    print(f"  min samples  {config.MIN_SAMPLES_PER_SITE} per site")
    section = payload["strata"]
    if section["evaluated"]:
        print(f"  strata kept  {', '.join(section['kept']) or 'none'}")
        for name in section["dropped_for_coverage"]:
            share = section["observed_coverage"].get(name, 0.0)
            need = config.STRATA[name]["required_coverage"]
            print(f"  DROPPED      {name}: populated for {share:.0%} of sites, "
                  f"needs {need:.0%}")
    else:
        print("  strata       NOT EVALUATED — no station table was supplied, "
              "so this manifest cannot authorise a fit")
    return payload


def read(path=None):
    path = path or config.MANIFEST_PATH
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def require_manifest(path=None):
    """The guard. Returns the manifest, or exits explaining what is wrong."""
    path = path or config.MANIFEST_PATH
    payload = read(path)
    if payload is None:
        sys.exit(
            f"No {os.path.basename(path)}. The specification has to be written "
            "down before any coefficient is computed:\n"
            "    python -m wq.run_wq --stations --strata --manifest")
    if payload.get("spec_hash") != config.spec_hash():
        sys.exit(
            "wq/config.py has changed since the manifest was written.\n"
            f"    manifest {payload.get('spec_hash')}\n"
            f"    config   {config.spec_hash()}\n"
            "This is the pre-registration doing its job. Either put the "
            "constant back, or re-register deliberately (keeping the old "
            "manifest alongside, so the change is visible):\n"
            "    mv wq_manifest.json wq_manifest.$(date +%s).json\n"
            "    python -m wq.run_wq --stations --strata --manifest")
    if not payload.get("strata", {}).get("evaluated"):
        sys.exit("The manifest was written without a station table, so its "
                 "strata coverage was never evaluated. Re-run --strata "
                 "--manifest before fitting.")
    return payload


def active_strata(payload=None):
    """The strata that survived the coverage rule. fit/report use only these."""
    payload = payload or require_manifest()
    return list(payload["strata"]["kept"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--sites", help="stratified station CSV")
    args = parser.parse_args()

    if args.show:
        payload = read()
        if payload is None:
            sys.exit("no manifest yet")
        print(json.dumps(payload, indent=2))
        return
    if args.write:
        sites = pd.read_csv(args.sites) if args.sites else None
        write(sites)
        return
    parser.error("give --write or --show")


if __name__ == "__main__":
    main()
