"""Task 3. What the D6 addressable-site count looks like against a null.

    python -m wq.diagnostics.task3_null                 # 200 permutations
    python -m wq.diagnostics.task3_null --permutations 50 --seed 0

D6 asks whether a site's STRONGEST |rho_ctrl| across eleven predictors clears
a floor. That is a maximum over eleven correlated tests at n around 60, and a
maximum is biased upward, so "12.3% of pairs have nothing that clears 0.20"
does not mean the other 87.7% have something real. wq.report.best_of_k_threshold
already corrects for this analytically under Sidak, which assumes the eleven
tests are independent. They are not: rain_24/48/72h are rolling sums of each
other and wave_height and wave_period come off one buoy record.

This script builds the null empirically instead, and keeps that dependence:

  * the outcome is permuted WITHIN CALENDAR-MONTH BLOCKS. wq.fit.fit_site_analyte
    demeans by `.dt.month` -- calendar month pooled across years, not
    year-month -- so permuting inside those same blocks leaves every month
    mean, and therefore the whole month control, exactly where it was.
  * the predictor matrix is never touched, so the correlation BETWEEN
    predictors is carried into the null unchanged. That is the part a
    Sidak correction cannot represent.
  * a predictor's paired rows are unchanged, because only the non-null
    outcome values move and they move within blocks. Every n_ctrl in the null
    equals the n_ctrl in the fit.

The observed side is recomputed from samples_with_covariates.csv rather than
read from coefficients.csv, and checked against coefficients.csv before any
permutation runs. If this file's arithmetic disagreed with the fit's, the
null would be measuring the disagreement.
"""

import argparse
import time

import numpy as np
import pandas as pd

from . import common
from .. import config, report

FLOORS = (config.USABLE_RHO_FLOOR_LENIENT, config.USABLE_RHO_FLOOR)


def _ranks(matrix):
    """Average ranks down each column, ties included. pandas rather than
    scipy, and average ties rather than ordinal, because that is what
    analyze_drivers.spearman uses via Series.rank()."""
    return pd.DataFrame(matrix).rank().to_numpy()


def _normalise(ranks):
    """Centre and scale rank columns so a correlation is one dot product."""
    centred = ranks - ranks.mean(axis=0, keepdims=True)
    norm = np.sqrt((centred ** 2).sum(axis=0, keepdims=True))
    norm[norm == 0] = np.inf
    return centred / norm


def _corr(left, right):
    """Correlation of two already-normalised rank blocks.

    The BLAS behind numpy on this machine raises spurious divide-by-zero and
    overflow flags on matmul for inputs that are entirely finite -- the
    products come back bit-identical to einsum and to Series.rank().corr().
    The flags are therefore ignored HERE and nowhere else, and correctness is
    not taken on trust: check_against_fit() compares every observed value
    this file produces against the pipeline's own pandas Spearman before a
    single permutation runs, and stops the script on any disagreement.
    """
    with np.errstate(all="ignore"):
        return left.T @ right


class Pair:
    """One site-analyte group, prepared once and permuted many times."""

    def __init__(self, station_id, analyte, group):
        self.station_id = station_id
        self.analyte = analyte
        stamps = pd.to_datetime(group["date"], errors="coerce")
        months = stamps.dt.month
        self.blocks = months.fillna(-1).to_numpy()
        # Chronological position, for the circular-shift null. A shift is only
        # meaningful along the time axis, and rows arrive in whatever order the
        # join left them in.
        self.order_in_time = np.argsort(
            stamps.fillna(pd.Timestamp("2100-01-01")).to_numpy(), kind="stable")
        target = pd.to_numeric(group["log_value"], errors="coerce")
        # Same two lines as wq.fit.fit_site_analyte, in the same order.
        self.y = (target - target.groupby(self.blocks).transform("mean")
                  ).to_numpy(dtype=float)
        self.x = {}
        for predictor in config.PREDICTORS:
            if predictor not in group.columns:
                continue
            series = pd.to_numeric(group[predictor], errors="coerce")
            self.x[predictor] = (
                series - series.groupby(self.blocks).transform("mean")
            ).to_numpy(dtype=float)
        self.masks = None
        self.observed = {}

    def prepare(self):
        """Group the predictors by which rows they share with the outcome, and
        drop the ones the fit itself could not score."""
        y_ok = np.isfinite(self.y)
        buckets = {}
        for predictor, values in self.x.items():
            mask = y_ok & np.isfinite(values)
            if mask.sum() < config.MIN_PAIRED_N:
                continue
            column = values[mask]
            if len(np.unique(column)) < 2 or len(np.unique(self.y[mask])) < 2:
                continue
            buckets.setdefault(mask.tobytes(), [mask, []])[1].append(predictor)
        self.masks = []
        for mask, predictors in buckets.values():
            ranks = _normalise(_ranks(np.column_stack(
                [self.x[p][mask] for p in predictors])))
            self.masks.append({"mask": mask, "predictors": predictors,
                               "x": ranks, "n": int(mask.sum())})
            observed = _normalise(_ranks(self.y[mask][:, None]))
            for predictor, rho in zip(predictors,
                                      _corr(ranks, observed[:, 0])):
                self.observed[predictor] = float(rho)
        return self

    def best_observed(self):
        if not self.observed:
            return np.nan, None, np.nan
        predictor = max(self.observed, key=lambda k: abs(self.observed[k]))
        n = next(m["n"] for m in self.masks if predictor in m["predictors"])
        return abs(self.observed[predictor]), predictor, n

    def _score(self, full, permutations):
        """max |rho| across predictors for each permuted outcome column."""
        best = np.zeros(permutations)
        for entry in self.masks:
            ranks = _normalise(_ranks(full[entry["mask"]]))
            rho = np.abs(_corr(entry["x"], ranks))
            best = np.maximum(best, rho.max(axis=0))
        return best

    def null_best_circular(self, rng, permutations):
        """A circular time shift, which keeps the outcome's own memory intact.

        The within-month shuffle below breaks the outcome-covariate link, but
        it also destroys the outcome's serial correlation: a dirty fortnight
        becomes fourteen independent dirty days. Rain is autocorrelated too,
        so a null built from independent draws is easier to beat than the real
        world, and every "excess over null" computed against it is an UPPER
        BOUND on the real excess.

        A circular shift moves the whole series along the time axis by a
        random lag and wraps it. Every autocorrelation in the outcome survives
        exactly; only its alignment with the covariates is destroyed. That is
        the harder and more honest null.

        The shift is drawn away from both ends, because a lag of one or of
        n-1 leaves almost every sample next to its own neighbour.
        """
        if not self.masks:
            return np.full(permutations, np.nan)
        y_ok = np.isfinite(self.y)
        rows = np.flatnonzero(y_ok)
        chrono = [i for i in self.order_in_time if y_ok[i]]
        n = len(chrono)
        if n < 8:
            return np.full(permutations, np.nan)
        series = self.y[chrono]
        low = max(2, int(round(0.05 * n)))
        high = n - low
        if high <= low:
            return np.full(permutations, np.nan)
        lags = rng.integers(low, high, size=permutations)
        full = np.full((len(self.y), permutations), np.nan)
        idx = np.array(chrono)
        for column, lag in enumerate(lags):
            full[idx, column] = np.roll(series, lag)
        return self._score(full, permutations)

    def null_best(self, rng, permutations):
        """max |rho| over the predictors, once per permutation."""
        if not self.masks:
            return np.full(permutations, np.nan)
        y_ok = np.isfinite(self.y)
        order = np.argsort(self.blocks[y_ok], kind="stable")
        block_values = self.blocks[y_ok][order]
        starts = np.flatnonzero(np.r_[True, block_values[1:] != block_values[:-1]])
        edges = np.r_[starts, len(block_values)]
        sorted_y = self.y[y_ok][order]

        permuted = np.repeat(sorted_y[:, None], permutations, axis=1)
        for begin, end in zip(edges[:-1], edges[1:]):
            if end - begin < 2:
                continue
            block = permuted[begin:end]
            # One independent shuffle of this block per permutation column.
            permuted[begin:end] = block[
                np.argsort(rng.random((end - begin, permutations)), axis=0),
                np.arange(permutations)]

        full = np.full((len(self.y), permutations), np.nan)
        rows = np.flatnonzero(y_ok)[order]
        full[rows] = permuted

        return self._score(full, permutations)


def build_pairs(samples, wanted):
    """One Pair per fitted site-analyte, in the fit's own row order."""
    samples = samples[samples["station_id"].astype(str).isin(
        {s for s, _ in wanted})]
    pairs = []
    for (station_id, analyte), group in samples.groupby(["station_id",
                                                         "analyte"]):
        if (str(station_id), analyte) not in wanted:
            continue
        pairs.append(Pair(str(station_id), analyte, group).prepare())
    return pairs


def check_against_fit(pairs, coefficients, tolerance=1e-6):
    """Refuse to build a null on arithmetic that does not match the fit."""
    fitted = coefficients.set_index(
        [coefficients["station_id"].astype(str), "analyte", "predictor"]
    )["rho_ctrl"]
    checked = worst = 0
    worst_key = None
    for pair in pairs:
        for predictor, rho in pair.observed.items():
            key = (pair.station_id, pair.analyte, predictor)
            if key not in fitted.index:
                continue
            expected = fitted.loc[key]
            if pd.isna(expected):
                continue
            delta = abs(float(expected) - rho)
            checked += 1
            if delta > worst:
                worst, worst_key = delta, key
    print(f"  reproduced {checked:,} of the fit's rho_ctrl values; "
          f"largest disagreement {worst:.2e} at {worst_key}")
    if worst > tolerance:
        raise AssertionError(
            f"this script disagrees with coefficients.csv by {worst:.3e} at "
            f"{worst_key}; the null would be measuring that, not the data")
    return checked


def main():
    parser = common.diagnostic_parser(__doc__)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--null", default="within_month",
                        choices=["within_month", "circular_shift"],
                        help="within_month shuffles inside calendar-month "
                             "blocks and destroys the outcome's serial "
                             "correlation, so its excess is an UPPER BOUND. "
                             "circular_shift rolls the series along the time "
                             "axis and keeps that memory intact.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None,
                        help="fit only the first N pairs (timing runs only)")
    args = parser.parse_args()

    coefficients = common.load_coefficients()
    coefficients = common.apply_scope(coefficients, args, "coefficient rows")
    coefficients = common.drop_hypothesis_sites(coefficients, args,
                                                "coefficient rows")
    coefficients = common.drop_held_out(coefficients, args,
                                        label="coefficient rows")
    head = common.headline(coefficients)
    wanted = set(zip(head["station_id"].astype(str), head["analyte"]))
    print("=" * 78)
    print("TASK 3  A NULL FOR THE D6 ADDRESSABLE-SITE COUNT")
    print("=" * 78)
    print(f"  {len(wanted):,} headline site-analyte pairs, "
          f"{args.permutations} permutations, seed {args.seed}, "
          f"null={args.null}")

    columns = ["station_id", "analyte", "date", "log_value"] + config.PREDICTORS
    samples = common.load_samples(usecols=columns)
    start = time.time()
    pairs = build_pairs(samples, wanted)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"  prepared {len(pairs):,} pairs in {time.time() - start:.1f}s")
    check_against_fit(pairs, coefficients)

    rng = np.random.default_rng(args.seed)
    start = time.time()
    rows = []
    for index, pair in enumerate(pairs, 1):
        best, predictor, n = pair.best_observed()
        null = (pair.null_best_circular(rng, args.permutations)
                if args.null == "circular_shift"
                else pair.null_best(rng, args.permutations))
        record = {"station_id": pair.station_id, "analyte": pair.analyte,
                  "best_predictor": predictor, "best_abs_rho": best,
                  "n_ctrl_of_best": n,
                  "n_predictors": len(pair.observed),
                  "null_median_best": float(np.nanmedian(null))
                  if np.isfinite(null).any() else np.nan,
                  "null_p": float(np.nanmean(null >= best))
                  if np.isfinite(best) else np.nan}
        for floor in FLOORS:
            record[f"obs_clears_{floor:.2f}"] = bool(best >= floor) \
                if np.isfinite(best) else False
            record[f"null_share_clears_{floor:.2f}"] = (
                float(np.nanmean(null >= floor))
                if np.isfinite(null).any() else np.nan)
        rows.append(record)
        if index % 500 == 0:
            rate = (time.time() - start) / index
            print(f"  [{index}/{len(pairs)}] {rate * 1000:.0f} ms/pair, "
                  f"{rate * (len(pairs) - index):.0f}s left")
    per_pair = pd.DataFrame(rows)
    print(f"  permuted {len(pairs):,} pairs in {time.time() - start:.1f}s")

    # The pipeline's own BH, so "has a surviving predictor" means what D5 means.
    marked = head.copy()
    marked["bh"] = common.bh_significant(marked["p_ctrl"]).to_numpy()
    bh = (marked.groupby([marked["station_id"].astype(str), "analyte"])["bh"]
          .any().rename("any_bh").reset_index())
    bh.columns = ["station_id", "analyte", "any_bh"]
    per_pair = per_pair.merge(bh, on=["station_id", "analyte"], how="left")

    summary = []
    for analyte, group in per_pair.groupby("analyte"):
        record = {"analyte": analyte, "pairs": len(group),
                  "pairs_with_no_scorable_predictor":
                  int(group["best_abs_rho"].isna().sum()),
                  "median_n_ctrl": float(group["n_ctrl_of_best"].median()),
                  "share_any_bh": float(group["any_bh"].fillna(False).mean())}
        for floor in FLOORS:
            observed = float(group[f"obs_clears_{floor:.2f}"].mean())
            null = float(group[f"null_share_clears_{floor:.2f}"].mean())
            record[f"observed_{floor:.2f}"] = observed
            record[f"null_{floor:.2f}"] = null
            record[f"excess_{floor:.2f}"] = observed - null
        record["share_beating_own_null_p05"] = float(
            (group["null_p"] <= 0.05).mean())
        summary.append(record)
    summary = pd.DataFrame(summary)

    print("\nper analyte — observed vs null share of pairs whose BEST "
          "predictor clears a floor:")
    print(summary.round(3).to_string(index=False))

    # ECOLI is the one analyte D6 reports as fully addressable, on 31 pairs.
    # What those 31 stations ARE matters as much as their n, so the site row
    # is printed beside the coefficient rather than left in another file.
    sites = common.load_sites()
    ecoli = per_pair[per_pair["analyte"] == "ECOLI"].merge(
        sites[["station_id", "station_name", "state", "region",
               "organization"]], on="station_id", how="left")
    ecoli = ecoli.sort_values("best_abs_rho")
    print(f"\nECOLI in full ({len(ecoli)} pairs) — D6 reported 0 of 31 with "
          "nothing clearing 0.20:")
    show = ["station_id", "station_name", "state", "region",
            "n_ctrl_of_best", "best_predictor", "best_abs_rho",
            "null_median_best", "null_share_clears_0.20", "null_p", "any_bh"]
    print(ecoli[show].round(3).to_string(index=False))
    print("\nECOLI best-predictor counts, and where those 31 stations are:")
    print(ecoli["best_predictor"].value_counts().to_string())
    print(ecoli.groupby(["region", "organization"]).size()
          .rename("pairs").to_string())

    print("\nwrote:")
    suffix = "" if args.null == "within_month" else f"_{args.null}"
    common.write(summary, f"task3_null_summary{suffix}.csv")
    common.write(per_pair, f"task3_null_per_pair{suffix}.csv")
    common.write(ecoli[show], "task3_ecoli_detail.csv")


if __name__ == "__main__":
    main()
