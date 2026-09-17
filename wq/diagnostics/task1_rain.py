"""Task 1. Does rain raise bacteria at most beaches in this run?

    python -m wq.diagnostics.task1_rain

D3 reports sign agreement and nothing else, and sign alone cannot separate
"positive everywhere and useful" from "positive everywhere and near zero".
This script reports, per analyte and rain window, the share of sites that are
positive, the share that are SIGNIFICANTLY positive and negative at raw alpha
and after the pipeline's Benjamini-Hochberg step, and the median and IQR of
rho_ctrl beside them -- so the sign and the magnitude are read together.

The three rain windows are rolling sums of each other (config.PREDICTOR_FAMILIES
calls them one family), so their three rows are one finding repeated, not
three findings. They are printed separately only because Phase 2 has to choose
one window per analyte.

The prior small-sample claims this run is being checked against name three
beaches: Santa Cruz Wharf, Carpinteria State Beach and Virginia Beach. The
script looks for them by name and by coordinate rather than assuming they are
present, and prints the nearest fitted station with its distance when they are
not, because "no row" and "a row with no signal" are opposite facts.
"""

import numpy as np
import pandas as pd

from . import common
from .. import config

# The prior findings, as stated, so the comparison is printed rather than
# remembered. Coordinates are the published beach locations.
PRIOR = [
    {"name": "Santa Cruz Wharf", "state": "CA", "lat": 36.9578,
     "lon": -122.0173,
     "prior": "rho ~ +0.45 at 72h, replicated across three analytes"},
    {"name": "Carpinteria State Beach", "state": "CA", "lat": 34.3900,
     "lon": -119.5180,
     "prior": "TOTAL coliform vs 48-72h rain, rho ~ -0.32 (reversed)"},
    {"name": "Virginia Beach", "state": "VA", "lat": 36.8529, "lon": -75.9780,
     "prior": "ENT vs 24h rain, rho ~ +0.32"},
]


def _haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(phi1) * np.cos(phi2)
         * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * radius * np.arcsin(np.sqrt(a))


def rain_rows(coefficients):
    """Headline rain coefficients, with the pipeline's BH attached.

    BH is run over the WHOLE headline tested set, which is what D5 does. Doing
    it over the rain rows alone would be a different, easier test on 8,000
    rows instead of 24,596, and would make rain look better than the report
    said it was.
    """
    head = common.headline(coefficients).copy()
    head["bh"] = common.bh_significant(head["p_ctrl"]).to_numpy()
    rain = head[head["predictor"].isin(common.RAIN)].copy()
    return rain[rain["rho_ctrl"].notna()]


def summarise(frame, keys):
    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rho = pd.to_numeric(group["rho_ctrl"], errors="coerce")
        p = pd.to_numeric(group["p_ctrl"], errors="coerce")
        n = len(group)
        stats = common.rho_summary(rho)
        record = dict(zip(keys, key))
        record.update({
            "n_pairs": n,
            "share_positive": float((rho > 0).mean()),
            "share_sig_pos": float(((p < config.ALPHA) & (rho > 0)).mean()),
            "share_sig_neg": float(((p < config.ALPHA) & (rho < 0)).mean()),
            "share_bh_pos": float((group["bh"] & (rho > 0)).mean()),
            "share_bh_neg": float((group["bh"] & (rho < 0)).mean()),
            "median_rho": stats["median"],
            "q25": stats["q25"],
            "q75": stats["q75"],
            "iqr": stats["iqr"],
        })
        rows.append(record)
    return pd.DataFrame(rows).sort_values(keys)


def locate_prior_sites(sites, coefficients, radius_km=1.0):
    """Are the three prior beaches in this run, and under which identifier?

    The nearest station by coordinate is NOT good enough, and California is
    why. Carpinteria State Beach is listed as 21CABCH-823 and as
    CABEACH_WQX-WP0000180 at the same point, 80 metres apart. The first is the
    one whose name says "Carpinteria State Beach"; the second is the one with
    1,412 samples. All 989 21CABCH stations carry zero rows in the Result
    service, so a lookup that takes the closest match, or the first name
    match, finds the empty twin and reports the beach as unfitted.

    So both are reported: the nearest station of any kind, and the nearest one
    that was actually FITTED, with the distance to each. When those two
    disagree the row says so, which is the only way the duplicate shows up.
    """
    fitted = set(coefficients["station_id"].astype(str))
    lat = pd.to_numeric(sites["lat"], errors="coerce")
    lon = pd.to_numeric(sites["lon"], errors="coerce")
    is_fitted = sites["station_id"].astype(str).isin(fitted)
    rows = []
    for entry in PRIOR:
        token = entry["name"].split()[0]
        named = sites[sites["station_name"].astype(str)
                      .str.contains(token, case=False, na=False)]
        distance = _haversine_km(entry["lat"], entry["lon"], lat, lon)
        values = distance.to_numpy()
        nearest = int(np.nanargmin(values))
        near = sites.iloc[nearest]

        masked = np.where(is_fitted.to_numpy(), values, np.inf)
        best_fitted = int(np.argmin(masked))
        has_fitted = np.isfinite(masked[best_fitted])
        fit_row = sites.iloc[best_fitted] if has_fitted else None

        row = {
            "prior_site": entry["name"],
            "prior_state": entry["state"],
            "prior_finding": entry["prior"],
            "name_matches_in_run": len(named),
            "in_run_state_present": entry["state"] in
            set(sites["state"].astype(str)),
            "nearest_station_id": near["station_id"],
            "nearest_station_name": near["station_name"],
            "nearest_km": round(float(values[nearest]), 3),
            "nearest_was_fitted": str(near["station_id"]) in fitted,
            "fitted_station_id": (fit_row["station_id"] if has_fitted
                                  else None),
            "fitted_station_name": (fit_row["station_name"] if has_fitted
                                    else None),
            "fitted_km": (round(float(masked[best_fitted]), 3)
                          if has_fitted else np.nan),
        }
        row["is_the_prior_site"] = bool(
            has_fitted and masked[best_fitted] <= radius_km)
        row["duplicate_identifier"] = bool(
            has_fitted and not row["nearest_was_fitted"]
            and masked[best_fitted] <= radius_km)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    coefficients = common.load_coefficients()
    sites = common.load_sites()
    rain = rain_rows(coefficients)

    print("=" * 78)
    print("TASK 1  RAIN vs BACTERIA, ACROSS EVERY FITTED SITE")
    print("=" * 78)
    print("BH is the pipeline's: one step-up over all 24,596 headline tests,")
    print("not over the rain rows alone.")

    overall = summarise(rain, ["analyte", "predictor"])
    print("\nby analyte and rain window:")
    print(overall.round(3).to_string(index=False))

    merged = rain.merge(sites[["station_id", "outfall_type", "state",
                               "organization"]],
                        on="station_id", how="left")
    by_outfall = summarise(merged, ["analyte", "predictor", "outfall_type"])
    print("\nthe same, broken out by outfall_type (EXPLORATORY breakout of a")
    print("pre-registered stratum -- read the n_pairs column first):")
    print(by_outfall.round(3).to_string(index=False))

    # Where the negative sites are, since that is the Carpinteria hypothesis.
    negatives = merged[(merged["rho_ctrl"] < 0)
                       & (merged["p_ctrl"] < config.ALPHA)]
    all_levels = merged.groupby("outfall_type").size().rename("rain_rows")
    neg_levels = negatives.groupby("outfall_type").size().rename("sig_negative")
    clustering = pd.concat([all_levels, neg_levels], axis=1).fillna(0)
    clustering["share"] = (clustering["sig_negative"]
                           / clustering["rain_rows"]).round(4)
    clustering = clustering.reset_index()
    print("\ndo significantly NEGATIVE rain coefficients cluster by outfall_type?")
    print(clustering.to_string(index=False))

    located = locate_prior_sites(sites, coefficients)
    print("\nthe three prior small-sample sites, looked for in this run:")
    print(located.to_string(index=False))
    if not located["in_run_state_present"].any():
        print("\n  NONE of the three prior sites is in this run. The states")
        print("  they sit in were not pulled, so this pass can neither")
        print("  replicate nor contradict any of the three findings.")
    absent = located[~located["is_the_prior_site"]]
    if not absent.empty:
        print(f"\n  {len(absent)} of {len(located)} prior site(s) have no "
              "fitted station within 1 km:")
        for _, row in absent.iterrows():
            print(f"    {row['prior_site']}: nearest fitted station is "
                  f"{row['fitted_km']} km away")
    dupes = located[located["duplicate_identifier"]]
    if not dupes.empty:
        print(f"\n  {len(dupes)} prior site(s) sit under TWO identifiers, and "
              "the closer one was\n  not the fitted one — the name matches a "
              "record with no samples:")
        for _, row in dupes.iterrows():
            print(f"    {row['prior_site']}: named "
                  f"{row['nearest_station_id']} (unfitted), data under "
                  f"{row['fitted_station_id']}")

    # The prior findings are about rain, so print this run's rain coefficients
    # for whichever stations actually carry the data.
    ids = [i for i in located["fitted_station_id"].dropna().astype(str)]
    if ids:
        here = rain[rain["station_id"].astype(str).isin(ids)]
        print("\nthis run's rain coefficients at those stations:")
        if here.empty:
            print("  (none of them produced a rain coefficient)")
        else:
            show = here[["station_id", "analyte", "predictor", "n_ctrl",
                         "rho_ctrl", "p_ctrl", "bh"]]
            print(show.sort_values(["station_id", "analyte", "predictor"])
                  .round(3).to_string(index=False))

    print("\nstates actually in the run:")
    print(sites["state"].value_counts().to_string())

    print("\nwrote:")
    common.write(overall, "task1_rain_by_analyte.csv")
    common.write(by_outfall, "task1_rain_by_outfall_type.csv")
    common.write(clustering, "task1_negative_by_outfall_type.csv")
    common.write(located, "task1_prior_sites_lookup.csv")


if __name__ == "__main__":
    main()
