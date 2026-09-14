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


def load_thresholds(path=None):
    path = path or config.THRESHOLDS_PATH
    with open(path) as handle:
        return json.load(handle)


def threshold_for(thresholds, state, analyte, water_class):
    """(value, unit, source_label). A missing entry returns (None, None,
    'no criterion') and the site is reported as unscoreable rather than being
    compared against a number nobody chose."""
    state = str(state or "").upper()
    states = thresholds.get("states", {})
    if state in states and isinstance(states[state], dict):
        block = states[state].get(water_class, {})
        entry = block.get(analyte)
        if entry:
            return (entry.get("threshold"), entry.get("unit"),
                    f"{state}:{water_class}")
    block = thresholds.get("_default", {}).get(water_class, {})
    entry = block.get(analyte)
    if entry:
        return entry.get("threshold"), entry.get("unit"), f"_default:{water_class}"
    return None, None, "no criterion"


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

    value, unit, source = threshold_for(thresholds, site.get("state"), analyte,
                                        site.get("water_class", "marine"))
    exceeds = (pd.to_numeric(group["value"], errors="coerce") > value
               if value is not None else pd.Series(np.nan, index=group.index))

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
                                         if value is not None
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
