"""Hygiene and pre-registration checks. No network.

Every one of these is a bug that has either already happened in this project
or is one line of carelessness away:

  '<10' parsed to NaN and then dropped -- deletes the clean samples at a site
  and keeps the dirty ones, so every beach looks worse than it is and the
  rain coefficient inflates.

  a non-detect set to zero -- same direction, via a different route.

  a field replicate counted as a second sample -- inflates n, and n is the
  number every coefficient in this study is reported beside.

  a stratum added to the manifest after the coefficients were seen -- the
  thing the whole pre-registration exists to prevent.

    python wq/test_wq_offline.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wq import clean, config, holdout, manifest, pull, review, scope, strata  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name
          + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# ---------------------------------------------------------------------------

def test_value_parsing():
    print("\ncensored and awkward values")
    cases = {
        "<10": (10.0, "below"),
        "< 10": (10.0, "below"),
        ">24196": (24196.0, "above"),
        "2,400": (2400.0, ""),
        "15.5": (15.5, ""),
        "": (None, ""),
        "ND": (None, ""),
        10: (10.0, ""),
    }
    for raw, (expected, censoring) in cases.items():
        value, mark = clean.parse_value(raw)
        ok = (np.isnan(value) if expected is None else value == expected)
        check(f"{raw!r} -> {expected}, {censoring!r}",
              ok and mark == censoring, f"got {value}, {mark!r}")


def test_nondetects_are_substituted_not_dropped():
    print("\nnon-detects survive at DL/2 (B3)")
    raw = pd.DataFrame({
        pull.COL_STATION: ["S1"] * 4,
        pull.COL_DATE: ["2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04"],
        pull.COL_ANALYTE: ["Enterococcus"] * 4,
        pull.COL_VALUE: ["<10", "40", "", "100"],
        pull.COL_UNIT: ["MPN/100mL"] * 4,
        pull.COL_NONDETECT: [None, None, "Not Detected", None],
        pull.COL_DETECTION_LIMIT: [np.nan, np.nan, 20.0, np.nan],
    })
    frame, log = clean.normalize(raw, clean.Log())
    for step in (clean.drop_rejected, clean.drop_non_samples,
                 clean.collapse_replicates, clean.drop_duplicates,
                 clean.normalize_units, clean.substitute_nondetects,
                 clean.add_log_value):
        frame, log = step(frame, log)

    check("all four samples survive", len(frame) == 4, f"kept {len(frame)}")
    check("two flagged as non-detects", int(frame["nondetect"].sum()) == 2)
    reported = dict(zip(frame["date"].dt.day, frame["value"]))
    check("'<10' became 5.0, not 0 and not NaN", reported.get(1) == 5.0,
          f"got {reported.get(1)}")
    check("blank + 'Not Detected' + DL 20 became 10.0", reported.get(3) == 10.0,
          f"got {reported.get(3)}")
    check("no substituted value is zero", not (frame["value"] == 0).any())

    shares = clean.nondetect_shares(frame)
    check("non-detect fraction recorded per site/analyte",
          abs(float(shares["nondetect_fraction"].iloc[0]) - 0.5) < 1e-9,
          str(shares["nondetect_fraction"].tolist()))
    check("50% is over the flag fraction", bool(shares["nondetect_flag"].iloc[0]))


def test_over_range_is_kept():
    print("\nover-range results survive (they land on the wet days)")
    raw = pd.DataFrame({
        pull.COL_STATION: ["S1", "S1"],
        pull.COL_DATE: ["2024-06-01", "2024-06-02"],
        pull.COL_ANALYTE: ["Enterococcus"] * 2,
        pull.COL_VALUE: [">24196", "10"],
        pull.COL_UNIT: ["MPN/100mL"] * 2,
    })
    frame, log = clean.normalize(raw, clean.Log())
    frame, log = clean.normalize_units(frame, log)
    frame, log = clean.substitute_nondetects(frame, log)
    check("both rows kept", len(frame) == 2, f"kept {len(frame)}")
    check("'>24196' kept at the limit", float(frame["value"].max()) == 24196.0)
    check("flagged as over-range", int(frame["over_range"].sum()) == 1)


def test_units_convert_but_estimators_do_not():
    print("\nunits (B2)")
    raw = pd.DataFrame({
        pull.COL_STATION: ["S1"] * 4,
        pull.COL_DATE: ["2024-06-0" + str(i) for i in range(1, 5)],
        pull.COL_ANALYTE: ["Enterococcus"] * 4,
        pull.COL_VALUE: ["10", "10", "10", "10"],
        pull.COL_UNIT: ["MPN/100mL", "CFU/100mL", "cfu/mL", "furlongs"],
    })
    frame, log = clean.normalize(raw, clean.Log())
    frame, log = clean.normalize_units(frame, log)
    check("unrecognised unit dropped", len(frame) == 3, f"kept {len(frame)}")
    values = sorted(frame["value"].tolist())
    check("per-mL scaled by 100", values == [10.0, 10.0, 1000.0], str(values))
    check("CFU and MPN both left at factor 1",
          set(frame.loc[frame["unit_factor"] == 1.0, "estimator"]) == {"CFU", "MPN"})
    counted = log.frame()
    converted = counted[(counted["step"] == "units")
                        & counted["detail"].str.startswith("converted")]
    check("conversion count logged", int(converted["records"].iloc[0]) == 1,
          str(converted["records"].tolist()))

    shares = clean.nondetect_shares(
        clean.add_log_value(clean.substitute_nondetects(frame, log)[0], log)[0])
    check("mixed estimators flagged on the site", bool(shares["mixed_estimators"].iloc[0]))


def test_rejected_and_replicates():
    print("\nQA/QC codes and field replicates (B2)")
    raw = pd.DataFrame({
        pull.COL_STATION: ["S1"] * 5,
        pull.COL_DATE: ["2024-06-01", "2024-06-01", "2024-06-02", "2024-06-03",
                        "2024-06-04"],
        pull.COL_TIME: ["09:00:00"] * 5,
        pull.COL_ANALYTE: ["Enterococcus"] * 5,
        pull.COL_VALUE: ["10", "30", "50", "70", "90"],
        pull.COL_UNIT: ["MPN/100mL"] * 5,
        pull.COL_STATUS: ["Final", "Final", "Rejected", "Final", "Final"],
        pull.COL_ACTIVITY_TYPE: ["Sample-Routine",
                                 "Quality Control Sample-Field Replicate",
                                 "Sample-Routine", "Quality Control Sample-Blank",
                                 "Sample-Routine"],
    })
    frame, log = clean.normalize(raw, clean.Log())
    frame, log = clean.drop_rejected(frame, log)
    check("rejected result dropped", len(frame) == 4, f"kept {len(frame)}")
    frame, log = clean.drop_non_samples(frame, log)
    check("blank dropped", len(frame) == 3, f"kept {len(frame)}")
    frame, log = clean.collapse_replicates(frame, log)
    check("replicate merged into its sample, not counted twice",
          len(frame) == 2, f"kept {len(frame)}")
    merged = frame.sort_values("date")["value_reported"].tolist()
    check("merged pair averaged to 20", merged[0] == 20.0, str(merged))


def test_exact_duplicates():
    print("\nduplicate records (B2)")
    raw = pd.DataFrame({
        pull.COL_STATION: ["S1"] * 3,
        pull.COL_DATE: ["2024-06-01"] * 3,
        pull.COL_TIME: ["09:00:00"] * 3,
        pull.COL_ANALYTE: ["Enterococcus"] * 3,
        pull.COL_VALUE: ["10", "10", "20"],
        pull.COL_UNIT: ["MPN/100mL"] * 3,
    })
    frame, log = clean.normalize(raw, clean.Log())
    frame, log = clean.drop_duplicates(frame, log)
    check("identical record dropped once", len(frame) == 2, f"kept {len(frame)}")
    counted = log.frame()
    row = counted[counted["step"] == "duplicates"]
    check("duplicate count logged", int(row["records"].iloc[0]) == 1)


def test_cross_source_dedup():
    print("\nCA appears in both sources (B1)")
    frame = pd.DataFrame({
        "source": ["WQP", "CKAN", "CKAN"],
        "station_id": ["W1", "C1", "C1"],
        "analyte": ["ENT"] * 3,
        "date": pd.to_datetime(["2024-06-01", "2024-06-01", "2024-06-02"]),
        "value": [40.0, 40.0, 90.0],
    })
    sites = pd.DataFrame({"station_id": ["W1", "C1"],
                          "lat": [33.0, 33.0005], "lon": [-117.3, -117.3]})
    out, log = clean.dedupe_across_sources(frame, sites, clean.Log())
    check("the duplicated reading is dropped once", len(out) == 2,
          f"kept {len(out)}")
    check("the CKAN-only reading survives",
          set(out["value"]) == {40.0, 90.0}, str(sorted(out["value"])))
    counted = log.frame()
    co_located = counted[counted["detail"].str.contains("co-located")]
    check("co-located stations reported",
          int(co_located["records"].iloc[0]) == 1)


def test_analyte_mapping():
    print("\nanalyte names")
    for name, expected in (("Enterococcus", "ENT"), ("Enterococci", "ENT"),
                           ("Escherichia coli", "ECOLI"),
                           ("Coliform, fecal", "FECAL"),
                           ("Total Coliform", "TOTAL"),
                           ("Nitrate", None)):
        check(f"{name} -> {expected}", clean.analyte_key(name) == expected,
              f"got {clean.analyte_key(name)}")


def test_wqp_columns_match_the_existing_puller():
    print("\nWQP column names agree with pull_wqp_results.py")
    try:
        import pull_wqp_results as legacy
    except ImportError as exc:
        print(f"  skip   pull_wqp_results not importable here ({exc})")
        return
    for mine, theirs in ((pull.COL_STATION, legacy.COL_STATION),
                         (pull.COL_DATE, legacy.COL_DATE),
                         (pull.COL_ANALYTE, legacy.COL_ANALYTE),
                         (pull.COL_VALUE, legacy.COL_VALUE),
                         (pull.COL_UNIT, legacy.COL_UNIT),
                         (pull.COL_NONDETECT, legacy.COL_NONDETECT)):
        check(f"{mine}", mine == theirs, f"theirs is {theirs}")


def test_region_is_assigned_from_metadata_only():
    print("\nregion and water class (A2)")
    sites = pd.DataFrame([
        {"station_id": "A", "station_name": "Storm Drain at 5th",
         "site_type": "Ocean", "lat": 33.0, "lon": -117.3, "state": "CA"},
        {"station_id": "C", "station_name": "Ocean Beach",
         "site_type": "Great Lake", "lat": 41.9, "lon": -87.6, "state": "IL"},
        {"station_id": "D", "station_name": "Gulf Shores",
         "site_type": "Ocean", "lat": 30.2, "lon": -87.7, "state": "AL"},
        {"station_id": "E", "station_name": "Pensacola",
         "site_type": "Ocean", "lat": 30.3, "lon": -87.2, "state": "FL"},
        {"station_id": "F", "station_name": "Daytona",
         "site_type": "Ocean", "lat": 29.2, "lon": -81.0, "state": "FL"},
    ])
    out = strata.assign(sites, datums=pd.DataFrame(), reviewed=pd.DataFrame())
    regions = dict(zip(out["station_id"], out["region"]))
    check("Pacific", regions["A"] == "Pacific", regions["A"])
    check("Great Lakes beats the state code", regions["C"] == "Great Lakes",
          regions["C"])
    check("Gulf", regions["D"] == "Gulf", regions["D"])
    check("Florida splits on longitude: panhandle is Gulf",
          regions["E"] == "Gulf", regions["E"])
    check("and the Atlantic side is Atlantic", regions["F"] == "Atlantic",
          regions["F"])
    check("a Great Lakes site is fresh water",
          out.loc[out["station_id"] == "C", "water_class"].iloc[0] == "fresh")
    check("a coastal site is marine",
          out.loc[out["station_id"] == "A", "water_class"].iloc[0] == "marine")

    # The station NAME must no longer decide anything. "Storm Drain at 5th"
    # was previously enough to assign a beach_type; now it is not.
    check("no beach_type is assigned without a human",
          out["beach_type"].isna().all(), str(out["beach_type"].tolist()))


def test_tidal_range():
    print("\ntidal range from CO-OPS datums")
    sites = pd.DataFrame([
        {"station_id": "A", "lat": 33.0, "lon": -117.3},
        {"station_id": "B", "lat": 20.0, "lon": -60.0},
    ])
    datums = pd.DataFrame([{"station_id": "9410230", "lat": 33.01,
                            "lon": -117.31, "mhhw": 1.62, "mllw": 0.0}])
    values, gauges = strata.tidal_range(sites, datums)
    check("nearest gauge supplies the range",
          abs(float(values.iloc[0]) - 1.62) < 1e-9, str(values.iloc[0]))
    check("gauge recorded", gauges.iloc[0] == "9410230")
    check("a site 5,000 km away gets nothing", np.isnan(values.iloc[1]))


def test_coverage_rule_drops_before_fitting():
    print("\nthe A2 coverage rule (drop before, never add after)")
    sites = pd.DataFrame([
        {"station_id": str(i), "station_name": "Ocean Beach",
         "site_type": "Ocean", "lat": 33.0 + i / 100, "lon": -117.3,
         "state": "CA"} for i in range(10)
    ])
    out = strata.assign(sites, datums=pd.DataFrame(), reviewed=pd.DataFrame())
    table = strata.coverage(out)
    keeps = dict(zip(table["stratum"], table["keeps"]))
    check("region survives", keeps["region"])
    check("beach_type is dropped when nobody has reviewed it",
          not keeps["beach_type"])

    path = tempfile.mktemp(suffix=".json")
    payload = manifest.write(out, path)
    entry = payload["entries"][0]
    check("manifest records what was dropped and why",
          "beach_type" in entry["strata"]["dropped_for_coverage"])
    check("manifest carries a timestamp", bool(entry["written_at"]))
    check("manifest carries the spec hash",
          entry["spec_hash"] == config.spec_hash())
    check("every site covariate is declared, even the empty ones",
          set(entry["site_covariates"]["declared"])
          == set(config.SITE_COVARIATES))
    check("active groupings exclude the dropped ones",
          "beach_type" not in manifest.active_strata(entry))
    os.remove(path)


def test_manifest_guard_catches_a_moved_goalpost():
    print("\nthe pre-registration guard")
    sites = strata.assign(
        pd.DataFrame([{"station_id": "A", "station_name": "Ocean Beach",
                       "site_type": "Ocean", "lat": 33.0, "lon": -117.3,
                       "state": "CA"}]),
        datums=pd.DataFrame(), reviewed=pd.DataFrame())
    path = tempfile.mktemp(suffix=".json")
    manifest.write(sites, path)
    check("a matching manifest authorises the fit",
          manifest.require_manifest(path)["spec_hash"] == config.spec_hash())

    # Move the goalpost the way a tempted analyst would: lower the sample
    # floor until an interesting site qualifies.
    original = config.MIN_SAMPLES_PER_SITE
    config.MIN_SAMPLES_PER_SITE = 5
    try:
        caught = False
        try:
            manifest.require_manifest(path)
        except SystemExit as exc:
            caught = "config.py has changed" in str(exc)
        check("lowering the sample floor after registration fails the run", caught)
    finally:
        config.MIN_SAMPLES_PER_SITE = original

    # The same guard on the covariate list: adding the covariate that groups
    # the coefficients nicely is the thing this exists to stop.
    config.SITE_COVARIATES.append("invented_covariate")
    try:
        message = ""
        try:
            manifest.require_manifest(path)
        except SystemExit as exc:
            message = str(exc)
        check("adding a covariate after registration fails the run too",
              "config.py has changed" in message, message[:80])
        check("and the message points at the amendment route instead",
              "--amend" in message)
    finally:
        config.SITE_COVARIATES.remove("invented_covariate")

    # A manifest written with no station table cannot authorise a fit either.
    bare = tempfile.mktemp(suffix=".json")
    manifest.write(None, bare)
    caught = False
    try:
        manifest.require_manifest(bare)
    except SystemExit as exc:
        caught = "never evaluated" in str(exc)
    check("a manifest with no coverage evaluation cannot authorise a fit", caught)
    os.remove(path)
    os.remove(bare)


def test_thresholds_file_is_readable_and_cited():
    print("\nthresholds config (C4)")
    with open(config.THRESHOLDS_PATH) as handle:
        payload = json.load(handle)
    check("a default block exists", "_default" in payload)
    check("California is configured separately", "CA" in payload["states"])
    for water_class in ("marine", "fresh"):
        for analyte in config.ANALYTES:
            entry = payload["_default"][water_class].get(analyte)
            check(f"_default/{water_class}/{analyte} has a threshold",
                  entry is not None and entry.get("threshold") is not None)
            check(f"_default/{water_class}/{analyte} cites its basis",
                  bool(entry and entry.get("basis")))
    check("every entry is marked unverified until someone checks it",
          payload["_default"]["_verified"] is False)
    check("no threshold is hardcoded in fit.py",
          "104" not in open(os.path.join(os.path.dirname(
              os.path.abspath(__file__)), "fit.py")).read())


def test_flat_series_is_detected_without_a_qualifier():
    """B3: undeclared censoring. The case is real -- a New Jersey station
    reported fecal coliform = 3.0 for all 49 of its samples with no qualifier,
    no detection-condition text and no detection limit, so every non-detect
    column read zero and the pair entered the headline carrying nothing."""
    print("\n[flat series]")
    rows = []
    # A: constant, undeclared. The incident.
    rows += [{"station_id": "A", "analyte": "FECAL", "value": 3.0,
              "nondetect": False, "over_range": False, "method": "m",
              "estimator": "MPN"} for _ in range(49)]
    # B: pinned to its floor but not constant -- 46 of 50 on the minimum.
    rows += [{"station_id": "B", "analyte": "FECAL", "value": 10.0,
              "nondetect": False, "over_range": False, "method": "m",
              "estimator": "MPN"} for _ in range(46)]
    rows += [{"station_id": "B", "analyte": "FECAL", "value": v,
              "nondetect": False, "over_range": False, "method": "m",
              "estimator": "MPN"} for v in (20.0, 40.0, 80.0, 160.0)]
    # C: a real series. Must not be flagged.
    rows += [{"station_id": "C", "analyte": "FECAL", "value": float(v),
              "nondetect": False, "over_range": False, "method": "m",
              "estimator": "MPN"} for v in range(1, 51)]
    # D: declared non-detects at the floor -- already visible to B3, and the
    # flag says so via flat_undeclared rather than pretending it is news.
    rows += [{"station_id": "D", "analyte": "FECAL", "value": 5.0,
              "nondetect": True, "over_range": False, "method": "m",
              "estimator": "MPN"} for _ in range(40)]
    shares = clean.nondetect_shares(pd.DataFrame(rows)).set_index("station_id")

    check("a constant series is flagged",
          bool(shares.loc["A", "constant_series"]))
    check("a constant series reads zero non-detects",
          float(shares.loc["A", "nondetect_fraction"]) == 0.0,
          "which is why the existing columns could not see it")
    check("a constant series is flagged as undeclared",
          bool(shares.loc["A", "flat_undeclared"]))
    check("a series pinned to its floor is flagged",
          bool(shares.loc["B", "floor_pinned"])
          and not bool(shares.loc["B", "constant_series"]),
          f"modal share {float(shares.loc['B', 'modal_share']):.2f}")
    check("a series that actually varies is NOT flagged",
          not bool(shares.loc["C", "flat_series_flag"]))
    check("a DECLARED non-detect floor is flat but not undeclared",
          bool(shares.loc["D", "flat_series_flag"])
          and not bool(shares.loc["D", "flat_undeclared"]))
    check("the flat threshold is not part of the frozen specification",
          "FLAT_SERIES_SHARE" not in config.SPEC_KEYS,
          "it describes a pair, it does not exclude one")


def test_site_clusters_label_a_beach_without_deleting_it():
    """Two stations 50 m apart on one beach are not two beaches. They are also
    not duplicate records -- the data refused that reading -- so the cluster is
    a label and nothing is dropped."""
    print("\n[site clusters]")
    sites = pd.DataFrame([
        # one beach: three points strung 120 m apart, so single linkage joins
        # the ends even though they are 240 m apart.
        {"station_id": "B-2", "lat": 36.9620, "lon": -122.0230},
        {"station_id": "B-1", "lat": 36.9620, "lon": -122.0216},
        {"station_id": "B-3", "lat": 36.9620, "lon": -122.0244},
        # a different beach, 3 km away
        {"station_id": "C-1", "lat": 36.9880, "lon": -122.0230},
        # no coordinate at all
        {"station_id": "D-1", "lat": np.nan, "lon": np.nan},
    ])
    labels, sizes = strata.site_clusters(sites)
    by_id = dict(zip(sites["station_id"], labels))
    check("co-located stations share a cluster",
          by_id["B-1"] == by_id["B-2"] == by_id["B-3"], by_id["B-1"])
    check("single linkage joins the ends of a chain",
          int(sizes.iloc[0]) == 3, f"{int(sizes.iloc[0])} stations")
    check("a distant station is its own cluster",
          by_id["C-1"] != by_id["B-1"])
    check("the label is the smallest station_id in the cluster",
          by_id["B-1"] == "B-1")
    check("a station with no coordinate is its own cluster",
          by_id["D-1"] == "D-1",
          "lumping the unlocatable together would invent a beach")
    check("nothing is dropped — one label per input row",
          len(labels) == len(sites))
    check("the cluster radius is not part of the frozen specification",
          "SITE_CLUSTER_RADIUS_KM" not in config.SPEC_KEYS,
          "it labels a station, it does not shuffle one")


def test_the_holdout_is_drawn_before_anything_is_fitted():
    """A test set chosen after seeing a score is not a test set. The split is
    drawn from a recorded seed, on CLUSTERS not stations, and the hypothesis
    beaches are kept out of it."""
    print("\n[holdout]")
    rng = np.random.default_rng(1)
    rows, sites = [], []
    for c in range(60):
        cluster = f"C{c:03d}"
        state = "CA" if c % 3 == 0 else "NJ"
        for k in range(2):
            station = f"{cluster}-{k}"
            sites.append({"station_id": station, "site_cluster": cluster,
                          "state": state, "station_name": station})
            for i in range(40):
                rows.append({"station_id": station, "analyte": "ENT",
                             "date": f"2021-{1 + i % 12:02d}-15"})
    samples = pd.DataFrame(rows)
    sites = pd.DataFrame(sites)
    hypothesis = pd.DataFrame([{"station_id": "C000-0", "site_cluster": "C000",
                                "label": "hypothesis"}])

    first, summary = holdout.build(samples, sites, hypothesis)
    again, _ = holdout.build(samples, sites, hypothesis)
    check("the same seed draws the same split",
          sorted(first["site_cluster"]) == sorted(again["site_cluster"]))
    different, _ = holdout.build(samples, sites, hypothesis, seed=999)
    check("a different seed draws a different one",
          sorted(first["site_cluster"]) != sorted(different["site_cluster"]))

    held = set(first["site_cluster"])
    check("roughly the requested fraction is held out",
          0.15 <= len(held) / 60 <= 0.25, f"{len(held)}/60")
    check("the hypothesis cluster is NOT in the test set",
          "C000" not in held)
    check("whole clusters move together, never half a beach",
          all(first[first["site_cluster"] == c]["station_id"].nunique() == 2
              for c in held))
    state_of = (sites.drop_duplicates("site_cluster")
                .set_index("site_cluster")["state"].to_dict())
    check("both states are represented",
          {state_of[c] for c in held} == {"CA", "NJ"},
          "stratification kept neither state out")

    # the guard itself
    frame = pd.DataFrame({"station_id": ["C000-0", f"{sorted(held)[0]}-0"],
                          "site_cluster": ["C000", sorted(held)[0]]})
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "holdout.csv")
        first.to_csv(path, index=False)
        original = holdout.HOLDOUT_PATH
        holdout.HOLDOUT_PATH = path
        try:
            kept = holdout.drop_holdout(frame, quiet=True)
            check("the guard drops held-out clusters", len(kept) == 1)
            all_rows = holdout.drop_holdout(frame, evaluate_holdout=True,
                                            quiet=True)
            check("--evaluate-holdout keeps them", len(all_rows) == 2)
        finally:
            holdout.HOLDOUT_PATH = original

    cutoff = holdout.time_cutoff(pd.DataFrame(
        {"date": pd.date_range("2020-01-01", "2026-06-30", freq="ME")}))
    check("the time holdout is the most recent 12 months",
          str(cutoff.date()) == "2025-06-30", str(cutoff.date()))
    check("the committed split is the one on disk",
          holdout.file_sha256(holdout.HOLDOUT_PATH) is not None,
          (holdout.file_sha256(holdout.HOLDOUT_PATH) or "")[:16] + "...")


def test_the_review_directory_does_not_shadow_the_review_module():
    """wq/review.py and wq/review/ are the same name. Python resolves that in
    the module's favour ONLY while the directory has no __init__.py: a
    namespace package loses to a module, a regular package wins. If one ever
    appeared there, from wq import review would silently become the directory,
    read_reviewed would vanish, and wq/strata.py would fail on beach_type --
    the one stratum with no automatic fallback, so it would look like a data
    problem."""
    print("\n[review module vs directory]")
    check("from wq import review gets the module, not the directory",
          review.__file__.endswith("review.py"), review.__file__)
    check("review.read_reviewed is reachable",
          callable(getattr(review, "read_reviewed", None)))
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "review")
    if os.path.isdir(directory):
        check("the directory has no __init__.py",
              not os.path.exists(os.path.join(directory, "__init__.py")),
              "adding one shadows wq/review.py")


def test_the_report_does_not_spend_the_holdout_describing_it():
    """D1-D6 describe the development set.

    The report used to read coefficients.csv straight off disk, which meant
    the pre-registered distribution, the D2 permutation test and the D6
    addressable count would all have been computed over the 617 held-out
    clusters. Once those beaches are inside a printed IQR, whatever is decided
    next is decided partly on them and the Phase 2 comparison they exist for
    is no longer clean.
    """
    print("\n[report vs holdout]")
    source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "run_wq.py")).read()
    stage = source[source.index("def stage_report"):]
    stage = stage[:stage.index("\n# Order matters")]
    check("stage_report applies the holdout guard",
          "holdout.drop_holdout" in stage)
    check("and it can be overridden deliberately",
          "evaluate_holdout" in stage)
    check("the attrition table is narrowed with it, so the counts tie",
          "attrition = attrition[" in stage)

    # coefficients.csv carries station_id and NO date and NO site_cluster, so
    # the guard has to fall back to station ids and must not let the time
    # holdout fire on a frame that has no dates in it.
    frame = pd.DataFrame({"station_id": ["keep-1", "drop-1"],
                          "analyte": ["ENT", "ENT"], "rho_ctrl": [0.3, 0.4]})
    held = pd.DataFrame([{"station_id": "drop-1", "site_cluster": "drop-1",
                          "state": "CA"}])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "holdout.csv")
        held.to_csv(path, index=False)
        original = holdout.HOLDOUT_PATH
        holdout.HOLDOUT_PATH = path
        try:
            kept = holdout.drop_holdout(frame, quiet=True)
            check("a held-out station is dropped by id when there is no "
                  "cluster column", list(kept["station_id"]) == ["keep-1"])
            check("a frame with no date column survives the time holdout",
                  len(kept) == 1, "the cutoff must not empty a dateless frame")
            both = holdout.drop_holdout(frame, evaluate_holdout=True,
                                        quiet=True)
            check("--evaluate-holdout keeps it", len(both) == 2)
        finally:
            holdout.HOLDOUT_PATH = original


def test_study_scope_separates_beaches_from_growing_areas():
    """The product is about beaches. Its largest analyte was not a beach
    measurement: no BEACH Act station in the study reports fecal coliform, and
    the fecal pairs are NSSP growing-area monitoring. The label has to say
    which of its two halves is sourced and which is inferred."""
    print("\n[study scope]")
    sites = pd.DataFrame([
        {"station_id": "B1", "site_type": "BEACH Program Site-Ocean",
         "site_cluster": "B1", "organization": "o", "state": "CA"},
        # co-located with B1 at 150 m: a CEDEN identifier for the same sand
        {"station_id": "C1", "site_type": "Ocean", "site_cluster": "B1",
         "organization": "CEDEN", "state": "CA"},
        {"station_id": "S1", "site_type": "Estuary", "site_cluster": "S1",
         "organization": "NJDEP_BMWM", "state": "NJ"},
        {"station_id": "U1", "site_type": "Estuary", "site_cluster": "U1",
         "organization": "USGS", "state": "CT"},
    ])
    samples = pd.DataFrame(
        [{"station_id": "S1", "analyte": "FECAL"}] * 40
        + [{"station_id": "U1", "analyte": "ECOLI"}] * 40
        + [{"station_id": "B1", "analyte": "ENT"}] * 40)
    table = scope.classify(sites, samples).set_index("station_id")

    check("a BEACH Act location type is a beach, and it is SOURCED",
          table.loc["B1", "study_scope"] == "beach"
          and table.loc["B1", "scope_basis"].startswith("SOURCED"))
    check("a station sharing its cluster is a beach too, also sourced",
          table.loc["C1", "study_scope"] == "beach"
          and "site_cluster" in table.loc["C1", "scope_basis"],
          "this is what catches CEDEN's duplicate identifiers")
    check("a non-beach reporting fecal coliform is shellfish, INFERRED",
          table.loc["S1", "study_scope"] == "shellfish"
          and table.loc["S1", "scope_basis"].startswith("INFERRED"),
          "fecal coliform is the NSSP indicator, enterococcus the BEACH Act one")
    check("everything else is unclassified, not quietly folded in",
          table.loc["U1", "study_scope"] == "unclassified")
    check("nothing is dropped", len(table) == len(sites))
    check("the inferred label is never presented as sourced",
          all(b.startswith("SOURCED") or b.startswith("INFERRED")
              or b.startswith("neither") for b in table["scope_basis"]))

    committed = scope.read()
    if not committed.empty:
        counts = committed["study_scope"].value_counts().to_dict()
        check("the committed split is on disk and non-trivial",
              counts.get("beach", 0) > 0 and counts.get("shellfish", 0) > 0,
              str(counts))


def main():
    for test in (test_value_parsing,
                 test_nondetects_are_substituted_not_dropped,
                 test_over_range_is_kept,
                 test_units_convert_but_estimators_do_not,
                 test_rejected_and_replicates,
                 test_exact_duplicates,
                 test_cross_source_dedup,
                 test_analyte_mapping,
                 test_wqp_columns_match_the_existing_puller,
                 test_region_is_assigned_from_metadata_only,
                 test_tidal_range,
                 test_coverage_rule_drops_before_fitting,
                 test_manifest_guard_catches_a_moved_goalpost,
                 test_thresholds_file_is_readable_and_cited,
                 test_flat_series_is_detected_without_a_qualifier,
                 test_site_clusters_label_a_beach_without_deleting_it,
                 test_the_holdout_is_drawn_before_anything_is_fitted,
                 test_the_review_directory_does_not_shadow_the_review_module,
                 test_the_report_does_not_spend_the_holdout_describing_it,
                 test_study_scope_separates_beaches_from_growing_areas):
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all offline hygiene and pre-registration checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
