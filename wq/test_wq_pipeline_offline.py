"""The stages wired together, end to end, with no network.

The unit tests check each part. This one checks that they compose: synthetic
raw WQP rows go in at one end, and the per-site table, the distribution and
the attrition table come out at the other, with the manifest guard satisfied
on the way through.

It also covers the covariate join, which is where two quiet errors live:

  a sample whose time is midnight is a DATE that was written as a datetime,
  not a sample taken at midnight. Joining it hourly attaches the small
  hours' tide to a morning sample.

  two beaches three kilometres apart are one ERA5 grid cell. Pulling both
  is the same request twice, and at national scale that is the difference
  between a run that finishes and one that does not.

    python wq/test_wq_pipeline_offline.py
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wq import clean, config, covariates, fit, manifest, pull, report, strata  # noqa: E402

FAILURES = []
RNG = np.random.default_rng(11031103)


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name
          + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# ---------------------------------------------------------------------------

def test_grid_cell_sharing():
    print("\nnearby sites share one grid cell")
    a = covariates.cell_key(36.9612, -122.0088)   # Walton Lighthouse
    b = covariates.cell_key(36.9628, -122.0170)   # Santa Cruz Wharf, ~1 km west
    far = covariates.cell_key(33.0000, -117.3000)
    check("two beaches 1 km apart are one cell", a == b, f"{a} vs {b}")
    check("a beach 500 km away is a different cell", a != far)
    lat, lon = covariates.cell_centre(36.9612, -122.0088)
    check("the cell centre sits inside the cell",
          abs(lat - 36.95) < 1e-6 and abs(lon + 122.05) < 1e-6,
          f"{lat}, {lon}")


def test_shore_normal_priority():
    print("\nwhere the shore normal came from")
    site = pd.Series({"station_id": "S1", "region": "Pacific",
                      "mop_shore_normal": np.nan})
    overrides = pd.DataFrame({"shore_normal_deg": [206.0]}, index=["S1"])
    check("a manual value wins",
          covariates.shore_normal_for(site, 120.0, overrides) == (206.0, "manual"))
    with_mop = pd.Series({"station_id": "S1", "region": "Pacific",
                          "mop_shore_normal": 206.0})
    check("CDIP MOP beats the nudge",
          covariates.shore_normal_for(with_mop, 120.0, None) == (206.0, "mop"))
    check("the marine nudge beats the region default",
          covariates.shore_normal_for(site, 120.0, None) == (120.0, "marine_nudge"))
    check("the region default is last and says so",
          covariates.shore_normal_for(site, None, None) == (270.0, "region"))
    lake = pd.Series({"station_id": "S2", "region": "Great Lakes",
                      "mop_shore_normal": np.nan})
    value, source = covariates.shore_normal_for(lake, None, None)
    check("a Great Lakes site gets no default at all",
          np.isnan(value) and source == "none", f"{value}, {source}")


def test_wind_components():
    print("\nonshore and alongshore wind")
    speed = pd.Series([10.0] * 4)
    direction = pd.Series([270.0, 90.0, 0.0, 180.0])
    onshore, alongshore = covariates.wind_components(speed, direction, 270.0)
    check("wind from seaward is fully onshore", abs(onshore[0] - 10.0) < 1e-9)
    check("wind from landward is fully offshore", abs(onshore[1] + 10.0) < 1e-9)
    check("wind from the north is alongshore", abs(alongshore[2] - 10.0) < 1e-9)
    check("no shore normal means no wind component, not a zero",
          covariates.wind_components(speed, direction, np.nan)[0].isna().all())


def test_covariate_join_resolution():
    print("\nhourly where a time was reported, daily where it was not")
    hours = pd.date_range("2024-06-01", periods=48, freq="h", tz="UTC")
    hourly = pd.DataFrame(index=hours)
    for column in config.PREDICTORS:
        hourly[column] = np.nan
    # A tide that swings hard within the day, so an hourly join and a daily
    # mean cannot give the same answer by accident.
    hourly["level_m"] = np.sin(np.arange(48) / 24 * 2 * np.pi) * 2.0
    hourly["rain_24h_mm"] = np.arange(48, dtype=float)

    samples = pd.DataFrame({
        "station_id": ["S1"] * 3,
        "analyte": ["ENT"] * 3,
        "date": pd.to_datetime(["2024-06-01", "2024-06-01", "2024-06-02"]),
        "sampled_at": pd.to_datetime(["2024-06-01 09:00", "2024-06-01 00:00",
                                      "2024-06-02 15:00"], utc=True),
        "value": [10.0, 20.0, 30.0],
        "log_value": [1.0, 1.3, 1.5],
    })
    joined = covariates.join_samples(samples, hourly)
    resolution = joined["join_resolution"].tolist()
    check("a 09:00 sample joins hourly", resolution[0] == "hourly")
    check("a midnight stamp is treated as a date, not an hour",
          resolution[1] == "daily", str(resolution[1]))
    check("the 09:00 sample gets hour 9's tide",
          abs(joined["level_m"].iloc[0] - hourly["level_m"].iloc[9]) < 1e-9,
          f"{joined['level_m'].iloc[0]:.3f}")
    check("the midnight sample gets the day's mean instead",
          abs(joined["level_m"].iloc[1]
              - hourly["level_m"].iloc[:24].mean()) < 1e-9)
    check("rain came across too", joined["rain_24h_mm"].iloc[0] == 9.0)

    empty = covariates.join_samples(samples, None)
    check("no covariates at all is 'none', with NaN columns present",
          set(empty["join_resolution"]) == {"none"}
          and empty["level_m"].isna().all())


def test_end_to_end():
    print("\nraw rows in, distribution out")
    # Two beaches: one where rain drives bacteria, one where nothing does.
    # Both carry non-detects, a rejected record, a replicate and a duplicate,
    # so the hygiene layer has something to do on the way through.
    rows = []
    for station, effect in (("WQP-A", 1.6), ("WQP-B", 0.0)):
        dates = pd.date_range("2019-05-01", periods=90, freq="7D")
        rain = RNG.gamma(0.7, 7.0, len(dates))
        noise = RNG.normal(0, 0.3, len(dates))
        log_value = 1.4 + effect * np.log10(rain + 1) + noise
        values = np.clip(np.power(10.0, log_value) - 1, 0.5, None)
        reported = [f"{v:.0f}" if v >= 10 else "<10" for v in values]
        rows.append(pd.DataFrame({
            pull.COL_STATION: station,
            pull.COL_DATE: dates.strftime("%Y-%m-%d"),
            pull.COL_TIME: "09:30:00",
            pull.COL_ANALYTE: "Enterococcus",
            pull.COL_VALUE: reported,
            pull.COL_UNIT: "MPN/100mL",
            pull.COL_STATUS: "Final",
            pull.COL_ACTIVITY_TYPE: "Sample-Routine",
            pull.COL_DETECTION_LIMIT: 10.0,
            "_rain": rain,
        }))
    raw = pd.concat(rows, ignore_index=True)
    # One rejected record and one exact duplicate, which must not survive.
    poison = raw.iloc[[0]].copy()
    poison[pull.COL_STATUS] = "Rejected"
    poison[pull.COL_VALUE] = "999999"
    raw = pd.concat([raw, poison, raw.iloc[[1]]], ignore_index=True)

    sites = strata.assign(pd.DataFrame([
        {"station_id": "WQP-A", "station_name": "Newport Harbor",
         "site_type": "Estuary", "lat": 33.6, "lon": -117.9, "state": "CA"},
        {"station_id": "WQP-B", "station_name": "Ocean Beach",
         "site_type": "Ocean", "lat": 34.4, "lon": -119.7, "state": "CA"}]),
        datums=pd.DataFrame(), reviewed=pd.DataFrame())

    samples, log = clean.clean(raw.drop(columns="_rain"), None, sites)
    check("the rejected record did not survive",
          not (samples["value"] > 100000).any())
    check("the duplicate was removed", len(samples) == 180, str(len(samples)))
    check("non-detects survived as DL/2",
          bool((samples["nondetect"]).any()) and (samples["value"] > 0).all())
    counted = log.frame()
    check("every hygiene step logged a count",
          set(counted["step"]) >= {"normalize", "qa/qc", "duplicates", "units",
                                   "non-detects", "final"},
          str(sorted(set(counted["step"]))))

    # Attach the rainfall that generated the data, as the covariate stage
    # would. Joining on (station, date) is what the real join does.
    key = raw.drop_duplicates([pull.COL_STATION, pull.COL_DATE])
    lookup = dict(zip(zip(key[pull.COL_STATION],
                          pd.to_datetime(key[pull.COL_DATE])), key["_rain"]))
    joined = samples.copy()
    joined["rain_24h_mm"] = [lookup.get((s, d), np.nan) for s, d
                             in zip(joined["station_id"], joined["date"])]
    joined["rain_48h_mm"] = joined["rain_24h_mm"] * 1.2
    joined["rain_72h_mm"] = joined["rain_24h_mm"] * 1.4
    for column in config.PREDICTORS:
        if column not in joined.columns:
            joined[column] = RNG.normal(0, 1, len(joined))

    shares = clean.nondetect_shares(joined)
    out_dir = tempfile.mkdtemp()
    manifest_path = os.path.join(out_dir, "wq_manifest.json")
    manifest.write(sites, manifest_path)
    payload = manifest.require_manifest(manifest_path)

    coefficients, attrition = fit.run(joined, sites, shares, payload=payload)
    check("both sites were fitted",
          set(coefficients["station_id"]) == {"WQP-A", "WQP-B"},
          str(set(coefficients["station_id"])))
    rain = coefficients[coefficients["predictor"] == "rain_24h_mm"]
    by_site = dict(zip(rain["station_id"], rain["rho_ctrl"]))
    check("the rain-driven site shows it", by_site["WQP-A"] > 0.5,
          f"{by_site['WQP-A']:.2f}")
    check("the flat site does not", abs(by_site["WQP-B"]) < 0.3,
          f"{by_site['WQP-B']:.2f}")

    strata_names = manifest.active_strata(payload)
    outputs = report.run(coefficients, sites, attrition, shares, strata_names,
                         out_dir=out_dir)
    for name in ("distribution.csv", "per_site.csv", "sign_agreement.csv"):
        path = os.path.join(out_dir, name)
        check(f"{name} written", os.path.exists(path))
    written = pd.read_csv(os.path.join(out_dir, "per_site.csv"))
    check("the per-site table has one row per site and analyte",
          len(written) == 2, str(len(written)))
    check("every reported coefficient has an n beside it",
          "n_rain_24h_mm" in written.columns and
          written["n_rain_24h_mm"].notna().all())
    distribution = pd.read_csv(os.path.join(out_dir, "distribution.csv"))
    check("the distribution table reports a spread and no mean",
          "iqr" in distribution.columns and "mean" not in distribution.columns,
          str(list(distribution.columns)))
    check("report.run returned what it wrote", "per_site.csv" in outputs)
    shutil.rmtree(out_dir)


def test_fit_refuses_without_a_manifest():
    print("\nfit refuses to run unregistered")
    missing = tempfile.mktemp(suffix=".json")
    caught = False
    try:
        manifest.require_manifest(missing)
    except SystemExit as exc:
        caught = "specification has to be written down" in str(exc)
    check("no manifest means no fit", caught)


def main():
    for test in (test_grid_cell_sharing,
                 test_shore_normal_priority,
                 test_wind_components,
                 test_covariate_join_resolution,
                 test_end_to_end,
                 test_fit_refuses_without_a_manifest):
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all offline pipeline checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
