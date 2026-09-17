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
    python -m wq.manifest --amend-scope CA --why "..." --by "riley"

AMENDING. A covariate thought of after the fact is not forbidden, it is
labelled. --amend appends an entry with its own timestamp, marks it
exploratory, and leaves the registered entry untouched. Everything computed
against an amended manifest is reported as an exploratory pass; the
pre-registered result is whatever entry 0 says it is.

AMENDING THE SCOPE. Adding STATIONS is a different move from adding a
covariate and --amend-scope is a separate entry kind, because nothing about
it is exploratory: the specification, the predictors, the control and the
strata are unchanged. What changes is the population those frozen rules are
applied to, and the entry records the state(s) added and the site count
before and after. The catch it exists to make visible is that the coverage
rule above ran ONCE, at registration, against the stations that existed then
-- so a covariate kept because the first region had it is not re-checked
against the new one, and the report's coverage table is where that gets read.
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
            return {}, {}, list(config.SITE_COVARIATES)
        frame = sites
    observed, distinct, dropped = {}, {}, []
    total = len(frame)
    for name in config.SITE_COVARIATES:
        present = name in frame.columns and total
        share = float(frame[name].notna().mean()) if present else 0.0
        # A covariate has to vary to be one. A column holding one value at
        # every station clears any coverage threshold and still cannot
        # correlate with anything -- see spatial.coverage() for the empty
        # ECHO box that made this concrete. The count is recorded either
        # way, so the manifest says WHICH half of the rule dropped it.
        values = int(frame[name].nunique(dropna=True)) if present else 0
        observed[name] = round(share, 4)
        distinct[name] = values
        if share < config.COVARIATE_MIN_COVERAGE or values < 2:
            dropped.append(name)
    return observed, distinct, dropped


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

    covariate_coverage, covariate_distinct, dropped_covariates = \
        _covariate_coverage(spatial, sites)
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
            "observed_distinct_values": covariate_distinct,
            "evaluated": spatial is not None and not spatial.empty,
            "minimum_coverage": config.COVARIATE_MIN_COVERAGE,
            "distinct_rule": ("a covariate taking fewer than 2 distinct "
                              "values is dropped however well populated it "
                              "is: it carries no information"),
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
            values = covariates.get("observed_distinct_values", {}).get(name)
            if values is not None and values < 2 and share > 0:
                # This one is worth spelling out. It is NOT a missing-data
                # drop: the fetch succeeded everywhere and returned the same
                # answer everywhere, which usually means it found nothing.
                print(f"  DROPPED      {name}: populated for {share:.0%} of "
                      f"sites but takes only {values} distinct value(s) — "
                      "no variation, so nothing to correlate")
            else:
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
    """The pre-registered entry -- the one require_manifest validates against.

    Matched on kind first. This used to return the first entry that was not
    flagged exploratory, which was only correct while every non-exploratory
    entry WAS the registration. A scope amendment is not exploratory (it adds
    stations, not covariates, and nothing it produces needs an exploratory
    label) but it is not the registration either, and if one were ever
    appended ahead of entry 0 the guard would start validating the wrong
    specification in silence.
    """
    entries = _entries(payload)
    for entry in entries:
        if entry.get("kind") == "pre-registered":
            return entry
    for entry in entries:
        if not entry.get("exploratory") and "specification" in entry:
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


def amend_scope(states, why, who, sites=None, path=None):
    """Record that the study's STATIONS changed after registration.

    This is a different animal from --amend, which adds a covariate and marks
    everything it touches exploratory. Adding California does not make a
    covariate exploratory: the specification, the predictors, the control and
    the strata are all exactly what they were. What changes is the population
    those frozen rules are applied to, and that has to be on the record for
    two reasons.

    The honest one: every pre-registered number in the existing report
    describes 2,854 Northeast stations. After this entry it describes a
    different set of beaches, and a reader comparing the two runs is entitled
    to know which is which without diffing CSVs.

    The load-bearing one: the coverage rule in write() ran ONCE, against the
    stations that existed at registration, and its decisions are frozen in
    entry 0 -- beach_type dropped at 0.0 coverage, the NHD covariates dropped
    at 0.0, region and outfall_type kept. Those decisions are NOT re-evaluated
    for the new stations, which is what pre-registration means and also what
    makes it dangerous here: a covariate kept because the Northeast had it may
    be thin or absent in California. This entry records the before and after
    counts so that gap is measurable rather than assumed, and the report's own
    coverage table is where it gets checked.
    """
    path = path or config.MANIFEST_PATH
    payload = read(path)
    if payload is None:
        sys.exit("no manifest to amend")
    if not why or not who:
        sys.exit("--why and --by are both required: a scope change with no "
                 "reason and no author is indistinguishable from the thing "
                 "this file exists to prevent")
    entries = _entries(payload)
    registered = registered_entry(payload)
    before = (registered or {}).get("n_sites_at_registration")
    entry = {
        "entry": len(entries),
        "kind": "scope amendment",
        "exploratory": False,
        "written_at": _now(),
        "spec_hash": config.spec_hash(),
        "added_states": list(states),
        "n_sites_at_registration": before,
        "n_sites_after_amendment": None if sites is None else int(len(sites)),
        "why": why,
        "added_by": who,
        "specification_changed": False,
        "coverage_reevaluated": False,
        "warning": (
            "The specification is unchanged and this is NOT an exploratory "
            "grouping. What changed is the set of stations the frozen rules "
            "are applied to. Two things follow. (1) Pre-registered results "
            "from before this entry describe a different population and are "
            "not interchangeable with results from after it; say which "
            "population any number describes. (2) The strata coverage rule "
            "was evaluated once, at registration, against the earlier "
            "stations, and is NOT re-run here -- a covariate kept then may be "
            "poorly populated on the added stations, so read the report's "
            "coverage table before reading its D2."),
    }
    entries.append(entry)
    with open(path, "w") as handle:
        json.dump({"manifest_version": MANIFEST_VERSION, "entries": entries},
                  handle, indent=2)
    print(f"appended scope amendment {entry['entry']} to {path}")
    print(f"  added    {', '.join(states)}")
    print(f"  sites    {before} at registration -> "
          f"{entry['n_sites_after_amendment']} now")
    print(f"  by       {who} at {entry['written_at']}")
    print("  The specification did not change, so the registered pass still "
          "stands;\n  what it describes did.")
    return entry


def record_override(kind, why, detail=None, path=None):
    """Append an entry saying a guard was overridden, and by what.

    An override that leaves no trace is the same as no guard. --fit refusing
    to run on an incomplete covariate build can be forced, but forcing it
    writes this, so a coefficient produced that way is identifiable later from
    the manifest alone rather than from whoever remembers the evening.
    """
    path = path or config.MANIFEST_PATH
    payload = read(path)
    if payload is None:
        sys.exit("no manifest to record an override against")
    entries = _entries(payload)
    entry = {
        "entry": len(entries),
        "kind": "override",
        "exploratory": False,
        "written_at": _now(),
        "spec_hash": config.spec_hash(),
        "override": kind,
        "why": why,
        "detail": detail or {},
        "warning": ("A guard was bypassed. Every result produced after this "
                    "entry was computed on inputs the pipeline had already "
                    "judged incomplete, and must be reported that way."),
    }
    entries.append(entry)
    with open(path, "w") as handle:
        json.dump({"manifest_version": MANIFEST_VERSION, "entries": entries},
                  handle, indent=2)
    print(f"  recorded override {entry['entry']} in {path}")
    return entry


def overrides(payload=None):
    payload = payload if payload is not None else read()
    return [e for e in _entries(payload) if e.get("kind") == "override"]


def scope_amendments(payload=None):
    """Every scope change on the record, oldest first."""
    payload = payload if payload is not None else read()
    return [e for e in _entries(payload) if e.get("kind") == "scope amendment"]


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
    parser.add_argument("--amend-scope", nargs="+", metavar="STATE",
                        help="record that these states were added to the "
                             "study after registration")
    parser.add_argument("--why")
    parser.add_argument("--by")
    args = parser.parse_args()

    if args.amend:
        amend(args.amend, args.why, args.by)
        return
    if args.amend_scope:
        sites = None
        stations_path = os.path.join(config.DATA_DIR, "stations_stratified.csv")
        if os.path.exists(stations_path):
            sites = pd.read_csv(stations_path)
        amend_scope(args.amend_scope, args.why, args.by, sites=sites)
        return
    if args.show:
        payload = read()
        if payload is None:
            sys.exit("no manifest yet")
        print(json.dumps(payload, indent=2))
        return
    parser.error("give --show, --amend or --amend-scope. The manifest is "
                 "WRITTEN by "
                 "wq.run_wq --manifest, which supplies the station table and "
                 "the layer record it needs.")


if __name__ == "__main__":
    main()
