"""C1-C4: one coefficient per site, per analyte, per predictor. Never pooled.

C3 is a correctness requirement, not a preference. The existing water-quality
work planted the trap and watched it spring: two beaches, one dirtier and
sitting at a higher-water gauge, with NO relationship inside either, scored
rho 0.73 pooled and -0.10 within site. A pooled coefficient here would be
ranking beaches, and the question this module exists to answer is how much
the within-beach coefficient VARIES between beaches -- which pooling destroys
by construction.

C1 is Spearman after per-month demeaning of both sides, reusing
analyze_drivers.spearman and analyze_drivers.demean_by so this module and the
rip pipeline cannot drift apart on the definition. Agencies sample in swim
season and rain is seasonal, so the raw correlation can be the calendar.

C4 is the decision-relevant target: does the predictor separate the days that
exceed the applicable single-sample criterion from the days that do not,
scored as the AUC of a Mann-Whitney U. 0.5 is a coin flip. The criterion is
read from wq/thresholds.json per state and analyte -- no threshold is written
in this file.

B5 is the assertion that carries the n=730 bug forward. Every fit recomputes
the number of rows with BOTH sides non-null and compares it to the n the
correlation reports. They are printed side by side and a mismatch ends the
run. A coefficient is never reported without its n.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from . import config, manifest


class SampleCountMismatch(AssertionError):
    """The n a fit used is not the n the data supports. Always fatal."""


def _stats():
    """spearman/demean_by from the existing analysis. Imported lazily only so
    the failure message is about analyze_drivers rather than about ndbc_api."""
    from analyze_drivers import demean_by, spearman
    return spearman, demean_by


EXCLUSIONS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "excluded_stations.csv")


def load_exclusions(path=None):
    """Stations deliberately kept out of the study, and why.

    A committed file rather than a filter in code, for the same reason
    wq/hypothesis_sites.csv is: an exclusion decided once and typed into a
    script is invisible six weeks later, and the attrition table is supposed
    to be a census of what happened to every station that was pulled. These
    are reported there under their own outcome rather than quietly absent.
    """
    path = path or EXCLUSIONS_PATH
    if not os.path.exists(path):
        return {}
    frame = pd.read_csv(path, dtype=str)
    if "station_id" not in frame.columns:
        return {}
    return {str(r["station_id"]): str(r.get("reason", "excluded"))
            for _, r in frame.iterrows()}


def load_thresholds(path=None):
    path = path or config.THRESHOLDS_PATH
    with open(path) as handle:
        return json.load(handle)


def threshold_entry(thresholds, state, analyte, water_class, programme=None):
    """(entry, source_label) for the criterion that applies here.

    Split out of threshold_for so the ratio rule and the no-standard marker
    read the SAME entry the threshold came from. Two lookups drifting apart is
    how a site ends up scored against one state's number and explained with
    another's.
    """
    # Programme BEFORE state: a station managed under a national programme is
    # judged by that programme's rule wherever it sits. New Jersey's shellfish
    # growing-area stations are classified on the NSSP fecal criterion, not on
    # whatever bathing-beach number the state or EPA would otherwise supply.
    if programme:
        block = thresholds.get("programmes", {}).get(str(programme), {})
        entry = block.get(water_class, {}).get(analyte) if block else None
        if entry:
            return entry, f"programme:{programme}:{water_class}"
    state = str(state or "").upper()
    states = thresholds.get("states", {})
    if state in states and isinstance(states[state], dict):
        entry = states[state].get(water_class, {}).get(analyte)
        if entry:
            return entry, f"{state}:{water_class}"
    entry = thresholds.get("_default", {}).get(water_class, {}).get(analyte)
    if entry:
        return entry, f"_default:{water_class}"
    return None, "no criterion"


def threshold_for(thresholds, state, analyte, water_class, programme=None):
    """(value, unit, source_label). A missing entry returns (None, None,
    'no criterion') and the site is reported as unscoreable rather than being
    compared against a number nobody chose.

    An entry carrying no_standard is NOT a missing entry. It is a state saying
    there is no criterion for this analyte in this water, and it must not fall
    through to the federal default -- California has no ocean E. coli
    standard, and scoring Californian E. coli against EPA's FRESHWATER number
    would manufacture exceedances against a rule no beach is posted on.
    """
    entry, source = threshold_entry(thresholds, state, analyte, water_class,
                                    programme)
    if entry is None:
        return None, None, "no criterion"
    if entry.get("no_standard"):
        return None, None, f"{source}:no applicable standard"
    return entry.get("threshold"), entry.get("unit"), source


def ratio_rule_for(thresholds, state, analyte, water_class, programme=None):
    """The stricter limb of a two-limb criterion, or None.

    17 CCR 7958 sets total coliform at 10,000/100 mL, and at 1,000/100 mL when
    the fecal/total ratio on the same sample exceeds 0.1. Both limbs live in
    wq/thresholds.json; none of those numbers appears in this file, which is
    the rule the whole module is built on.
    """
    entry, _ = threshold_entry(thresholds, state, analyte, water_class,
                               programme)
    return (entry or {}).get("ratio_rule")


def same_sample_key(frame):
    """Which rows came off ONE bottle, for pairing two analytes.

    A ratio rule needs both analytes measured on the same sample, not merely
    on the same day: a morning and an afternoon grab at one station are two
    samples and their ratio is not a ratio of anything.

    The key is (station_id, sampled_at) when a real time was reported, and
    (station_id, date) when one was not. Midnight counts as "no time
    reported", which is not a guess -- wq.covariates.join_samples already
    treats a 00:00 stamp as a date rather than an hour, for the same reason:
    agencies that report date-only arrive as midnight, and an hourly join at
    that stamp attaches the small hours' conditions to a morning sample. The
    two modules now agree on what a timestamp means.
    """
    stamps = pd.to_datetime(frame.get("sampled_at"), errors="coerce", utc=True)
    has_time = stamps.notna() & ~((stamps.dt.hour == 0)
                                  & (stamps.dt.minute == 0))
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    stamp_text = stamps.dt.strftime("%Y-%m-%dT%H:%M")
    when = stamp_text.where(has_time, dates)
    return frame["station_id"].astype(str) + "|" + when.astype(str)


def apply_ratio_rules(joined, sites, thresholds):
    """Per-row exceedance thresholds, where a criterion has two limbs.

    Returns a copy of `joined` carrying:

      exceedance_threshold      the limit THIS row is scored against
      ratio_value               the companion ratio, where one could be formed
      ratio_rule_applied        the stricter limb fired
      ratio_rule_unavailable    no same-sample companion result, so the
                                permissive limb stands by default
      exceedance_scorable       the exceedance label is knowable at all

    The last two columns are the honest ones. Where a total-coliform sample
    has no fecal result off the same bottle, the ratio cannot be computed and
    the 10,000 limb stands by default -- so the exceedance count is an
    UNDERCOUNT rather than a measurement.

    exceedance_scorable says how much of an undercount. A sample at 400 is
    below both limbs and a sample at 40,000 is above both, so their labels are
    known whatever the ratio was. A sample BETWEEN the two limbs with no
    companion result is genuinely unknowable: it exceeds under the strict limb
    and does not under the permissive one, and nothing in the record says
    which applied. Those are not negatives. Scoring them as negatives is how a
    classifier gets trained to reproduce a gap in the sampling programme, so
    they are marked unscorable and Phase 2 can drop them.
    """
    frame = joined.copy()
    if frame.empty:
        for column in ("exceedance_threshold", "ratio_value"):
            frame[column] = np.nan
        for column in ("ratio_rule_applied", "ratio_rule_unavailable"):
            frame[column] = False
        return frame

    from .strata import programme_of
    meta = {}
    if sites is not None and not sites.empty:
        for _, site in sites.iterrows():
            programme = site.get("programme")
            if programme is None or (isinstance(programme, float)
                                     and pd.isna(programme)):
                programme = programme_of(site.get("site_type"))
            meta[str(site.get("station_id"))] = (
                site.get("state"), site.get("water_class", "marine"), programme)
    stations = frame["station_id"].astype(str)
    blank = (None, "marine", None)
    states = stations.map(lambda s: meta.get(s, blank)[0])
    classes = stations.map(lambda s: meta.get(s, blank)[1])
    programmes = stations.map(lambda s: meta.get(s, blank)[2])

    base, rules = [], []
    for state, analyte, water_class, programme in zip(
            states, frame["analyte"], classes, programmes):
        value, _unit, _source = threshold_for(thresholds, state, analyte,
                                              water_class, programme)
        base.append(value)
        rules.append(ratio_rule_for(thresholds, state, analyte, water_class,
                                    programme))
    frame["exceedance_threshold"] = pd.to_numeric(pd.Series(base, index=frame.index),
                                                  errors="coerce")
    frame["ratio_value"] = np.nan
    frame["ratio_rule_applied"] = False
    frame["ratio_rule_unavailable"] = False
    # Everything without a two-limb criterion is scorable by construction:
    # there is only one number to be on one side of.
    frame["exceedance_scorable"] = frame["exceedance_threshold"].notna()

    has_rule = pd.Series([r is not None for r in rules], index=frame.index)
    if not has_rule.any():
        return frame

    keys = same_sample_key(frame)
    values = pd.to_numeric(frame["value"], errors="coerce")
    for rule in {id(r): r for r in rules if r is not None}.values():
        companion = str(rule.get("companion_analyte"))
        above = float(rule.get("above"))
        strict = float(rule.get("threshold_when_above"))
        rows = pd.Series([r is not None
                          and r.get("companion_analyte") == companion
                          and float(r.get("above")) == above
                          for r in rules], index=frame.index)
        if not rows.any():
            continue
        # One companion value per bottle. Duplicates on one key are averaged
        # rather than picked from, so the pairing does not depend on row order.
        mate = frame[frame["analyte"] == companion]
        lookup = (pd.Series(pd.to_numeric(mate["value"], errors="coerce").to_numpy(),
                            index=same_sample_key(mate))
                  .groupby(level=0).mean())
        paired = keys[rows].map(lookup)
        denominator = values[rows]
        ratio = paired / denominator.where(denominator > 0)
        frame.loc[rows, "ratio_value"] = ratio
        fires = ratio > above
        frame.loc[rows, "ratio_rule_applied"] = fires.fillna(False)
        frame.loc[rows, "ratio_rule_unavailable"] = ratio.isna()
        strict_rows = rows & frame["ratio_rule_applied"]
        frame.loc[strict_rows, "exceedance_threshold"] = strict

        # Ambiguous: no companion result, and the value falls between the two
        # limbs, so the label depends entirely on the ratio nobody measured.
        permissive = pd.to_numeric(frame.loc[rows, "exceedance_threshold"],
                                   errors="coerce")
        blind = frame.loc[rows, "ratio_rule_unavailable"].astype(bool)
        between = (values[rows] > strict) & (values[rows] <= permissive)
        frame.loc[rows, "exceedance_scorable"] = ~(blind & between.fillna(False))
    return frame


def auc(values, positive):
    """Area under the ROC curve, from ranks. No scipy.

    Ties are handled by rank averaging, which is what makes this the right
    statistic for a zero-inflated predictor: half a site's rainfall values
    are 0, and a tie-blind implementation would score that as separation.
    """
    frame = pd.DataFrame({"x": pd.to_numeric(values, errors="coerce"),
                          "y": pd.Series(positive).astype("boolean")}).dropna()
    n_pos = int(frame["y"].sum())
    n_neg = int((~frame["y"]).sum())
    if n_pos < config.MIN_EXCEEDANCE_DAYS or n_neg < config.MIN_EXCEEDANCE_DAYS:
        return np.nan, n_pos, n_neg
    ranks = frame["x"].rank()
    rank_sum = float(ranks[frame["y"].astype(bool)].sum())
    statistic = rank_sum - n_pos * (n_pos + 1) / 2.0
    return statistic / (n_pos * n_neg), n_pos, n_neg


def paired_n(frame, left, right):
    """The count B5 asserts against: rows where BOTH sides are non-null."""
    return int((pd.to_numeric(frame[left], errors="coerce").notna()
                & pd.to_numeric(frame[right], errors="coerce").notna()).sum())


def fit_site_analyte(group, site, analyte, thresholds, strict=True):
    """Every predictor at one site for one analyte. Returns a list of rows."""
    spearman, demean_by = _stats()
    months = pd.to_datetime(group["date"], errors="coerce").dt.month
    target = pd.to_numeric(group["log_value"], errors="coerce")
    target_ctrl = demean_by(target, months)

    programme = site.get("programme")
    if programme is None or (isinstance(programme, float) and pd.isna(programme)):
        from .strata import programme_of
        programme = programme_of(site.get("site_type"))
    value, unit, source = threshold_for(thresholds, site.get("state"), analyte,
                                        site.get("water_class", "marine"),
                                        programme)
    # Per-row limits where a criterion has two limbs (wq.fit.apply_ratio_rules
    # puts them on the frame). Falling back to the scalar keeps every caller
    # that has not run that step working unchanged.
    if "exceedance_threshold" in group.columns:
        limits = pd.to_numeric(group["exceedance_threshold"], errors="coerce")
    else:
        limits = pd.Series(value, index=group.index, dtype="float64")
    measured = pd.to_numeric(group["value"], errors="coerce")
    scoreable = limits.notna()
    if "exceedance_scorable" in group.columns:
        # A label that depends on a ratio nobody measured is not a negative.
        scoreable = scoreable & group["exceedance_scorable"].fillna(True).astype(bool)
    exceeds = (measured > limits).where(scoreable)
    if not scoreable.any():
        exceeds = pd.Series(np.nan, index=group.index)
    reason = "" if scoreable.any() else source
    n_strict = int(group.get("ratio_rule_applied",
                             pd.Series(False, index=group.index))
                   .fillna(False).astype(bool).sum())
    n_no_ratio = int(group.get("ratio_rule_unavailable",
                               pd.Series(False, index=group.index))
                     .fillna(False).astype(bool).sum())
    n_ambiguous = int((~group.get("exceedance_scorable",
                                  pd.Series(True, index=group.index))
                       .fillna(True).astype(bool)).sum())

    rows = []
    for predictor in config.PREDICTORS:
        if predictor not in group.columns:
            continue
        series = pd.to_numeric(group[predictor], errors="coerce")
        expected = paired_n(group.assign(_t=target), predictor, "_t")
        rho, n, p = spearman(series, target, min_n=config.MIN_PAIRED_N)
        # B5. The n a correlation reports and the n the data supports must be
        # the same number. They came apart once -- a fit reported n=730 on a
        # join that held far fewer paired rows -- and nothing caught it,
        # because the n was never printed next to the count it came from.
        if n != expected:
            message = (f"n mismatch at {site.get('station_id')} / {analyte} / "
                       f"{predictor}: the fit used n={n}, the join holds "
                       f"{expected} rows with both sides non-null")
            if strict:
                raise SampleCountMismatch(message)
            print(f"  WARNING: {message}")

        ctrl_series = demean_by(series, months)
        expected_ctrl = paired_n(
            pd.DataFrame({"a": ctrl_series, "b": target_ctrl}), "a", "b")
        rho_ctrl, n_ctrl, p_ctrl = spearman(ctrl_series, target_ctrl,
                                            min_n=config.MIN_PAIRED_N)
        if n_ctrl != expected_ctrl:
            message = (f"controlled n mismatch at {site.get('station_id')} / "
                       f"{analyte} / {predictor}: fit n={n_ctrl}, join holds "
                       f"{expected_ctrl}")
            if strict:
                raise SampleCountMismatch(message)
            print(f"  WARNING: {message}")

        separation, n_exceed, n_clean = (auc(series, exceeds)
                                         if scoreable.any()
                                         else (np.nan, 0, 0))
        rows.append({
            "station_id": site.get("station_id"),
            "analyte": analyte,
            "predictor": predictor,
            "family": family_of(predictor),
            "n": n,
            "n_paired_check": expected,
            "rho": rho,
            "p": p,
            "n_ctrl": n_ctrl,
            "n_paired_check_ctrl": expected_ctrl,
            "rho_ctrl": rho_ctrl,
            "p_ctrl": p_ctrl,
            "auc_exceedance": separation,
            "n_exceed": n_exceed,
            "n_below": n_clean,
            "threshold": value,
            "threshold_unit": unit,
            "threshold_source": source,
            "exceedance_reason": reason,
            "n_ratio_rule_applied": n_strict,
            "n_ratio_unavailable": n_no_ratio,
            "n_exceedance_ambiguous": n_ambiguous,
        })
    return rows


def family_of(predictor):
    for family, members in config.PREDICTOR_FAMILIES.items():
        if predictor in members:
            return family
    return "other"


def attrition_reason(site, group, nondetects):
    """Why a station did not make it into the distribution. C2: this table is
    itself an output -- it sizes how many beaches the approach can address."""
    if group is None or group.empty:
        return "no samples after hygiene"
    if len(group) < config.MIN_SAMPLES_PER_SITE:
        return f"under {config.MIN_SAMPLES_PER_SITE} samples"
    usable = [c for c in config.PREDICTORS
              if c in group.columns
              and paired_n(group.assign(_t=group["log_value"]), c, "_t")
              >= config.MIN_PAIRED_N]
    if not usable:
        return "no covariate with enough paired rows"
    if pd.isna(site.get("lat")) or pd.isna(site.get("lon")):
        return "no coordinates"
    if nondetects is not None and nondetects:
        return "kept, non-detect flagged"
    return "kept"


def run(joined, sites, nondetects=None, strict=True, payload=None):
    """The whole of Part C. Returns (coefficients, attrition).

    Refuses to start until wq_manifest.json exists and matches wq/config.py.
    """
    payload = payload or manifest.require_manifest()
    print(f"fitting under manifest {payload['spec_hash']} "
          f"written {payload['written_at']}")

    thresholds = load_thresholds()
    unverified = [k for k in ("_default",) if not thresholds.get(k, {}).get("_verified", False)]
    if unverified:
        print("  NOTE: thresholds.json is marked _verified: false — the "
              "exceedance column is computed against transcribed criteria "
              "that have not been checked against the live regulation.")

    flags = {}
    if nondetects is not None and not nondetects.empty:
        flags = {(str(r["station_id"]), r["analyte"]): bool(r["nondetect_flag"])
                 for _, r in nondetects.iterrows()}

    joined = apply_ratio_rules(joined, sites, thresholds)
    fired = int(joined["ratio_rule_applied"].sum())
    blind = int(joined["ratio_rule_unavailable"].sum())
    if fired or blind:
        print(f"  two-limb criteria: the stricter limb applied to {fired:,} "
              f"sample(s); {blind:,} had no same-sample companion result, so "
              "the\n  permissive limb stood and their exceedance count is an "
              "undercount rather than a measurement")

    excluded = load_exclusions()
    if excluded:
        present = [k for k in excluded if k in set(sites["station_id"].astype(str))]
        print(f"  {len(present)} station(s) excluded by decision, not by data: "
              f"{sorted(set(excluded[k] for k in present))}")

    by_station = {str(s["station_id"]): s for _, s in sites.iterrows()}
    rows, attrition = [], []
    stations = sorted(set(joined["station_id"].astype(str))
                      | set(by_station))
    for index, station in enumerate(stations, 1):
        site = by_station.get(station, pd.Series({"station_id": station}))
        at_station = joined[joined["station_id"].astype(str) == station]
        for analyte in config.ANALYTES:
            group = at_station[at_station["analyte"] == analyte]
            flagged = flags.get((station, analyte), False)
            if station in excluded:
                reason = f"excluded: {excluded[station]}"
            else:
                reason = attrition_reason(site, group, flagged)
            attrition.append({
                "station_id": station,
                "state": site.get("state"),
                "region": site.get("region"),
                "analyte": analyte,
                "n_samples": 0 if group is None else len(group),
                "outcome": reason,
            })
            if not reason.startswith("kept"):
                continue
            rows.extend(fit_site_analyte(group, site, analyte, thresholds,
                                         strict=strict))
        if index % 50 == 0:
            print(f"  [{index}/{len(stations)}] stations fitted")

    coefficients = pd.DataFrame(rows)
    # A coefficient is never reported without its n. A RANK coefficient is not
    # reported without its tie structure either: n=220 on a series holding 13
    # distinct values is not the study n=220 suggests, and nothing downstream
    # could see that before.
    if not coefficients.empty and nondetects is not None and not nondetects.empty:
        carry = [c for c in ("station_id", "analyte", "modal_share",
                             "n_distinct_values", "modal_value",
                             "floor_pinned", "at_floor")
                 if c in nondetects.columns]
        if len(carry) > 2:
            shares = nondetects[carry].copy()
            shares["station_id"] = shares["station_id"].astype(str)
            coefficients["station_id"] = coefficients["station_id"].astype(str)
            coefficients = coefficients.merge(shares, on=["station_id", "analyte"],
                                              how="left")
    table = pd.DataFrame(attrition)
    if not coefficients.empty:
        # Nothing may leave this function without its n beside it.
        assert {"n", "n_ctrl"}.issubset(coefficients.columns)
        assert (coefficients["n"] == coefficients["n_paired_check"]).all()
        assert (coefficients["n_ctrl"] == coefficients["n_paired_check_ctrl"]).all()
    return coefficients, table


def write(coefficients, attrition, out_dir=None):
    out_dir = out_dir or config.OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for name, frame in (("coefficients.csv", coefficients),
                        ("attrition.csv", attrition)):
        path = os.path.join(out_dir, name)
        frame.to_csv(path, index=False)
        paths.append(path)
        print(f"wrote {path}  ({len(frame)} rows)")
    return paths


def main():
    sys.exit("wq/fit.py is driven by wq/run_wq.py, which assembles the "
             "samples, sites and covariates it needs.")


if __name__ == "__main__":
    main()
