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
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wq import (clean, config, covariates, fit, manifest, pull, report,  # noqa: E402
                spatial, strata)
from wq import layers  # noqa: E402

FAILURES = []
RNG = np.random.default_rng(11031103)


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name
          + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# ---------------------------------------------------------------------------

def test_a_cached_nothing_can_be_read_back():
    """A gauge with no water temperature killed a 120-site run at site 17.

    The first site cached the emptiness; the second site sharing that gauge
    read the cache and pandas raised EmptyDataError on the bare newline that
    an empty frame serialises to. Caching an absence is right. Caching it in
    a form that cannot be reopened is a landmine under the next caller.
    """
    print("\na cached nothing is still readable")
    import tempfile
    with tempfile.TemporaryDirectory() as root:
        original = covariates.CACHE_DIR
        covariates.CACHE_DIR = root
        try:
            calls = []

            def nothing():
                calls.append(1)
                return None

            first = covariates._cached("coops_water_temperature_8452660",
                                       nothing)
            check("a service with nothing to say answers None", first is None)
            check("and the builder ran once", len(calls) == 1)

            second = covariates._cached("coops_water_temperature_8452660",
                                        nothing)
            check("the next site reads that answer instead of crashing",
                  second is None)
            check("without asking the service again", len(calls) == 1,
                  f"{len(calls)} call(s)")

            # The file the OLD code wrote, which is still on disk for anyone
            # who ran before this fix: a single newline, no header.
            stale = os.path.join(root, "coops_water_level_9999999.csv")
            with open(stale, "w") as handle:
                handle.write("\n")
            check("a cache written by the older form is still an answer",
                  covariates._cached("coops_water_level_9999999",
                                     lambda: None) is None)

            # And a real frame still round-trips.
            frame = pd.DataFrame({"t": ["2020-01-01"], "v": [12.5]})
            covariates._cached("coops_water_level_8452944", lambda: frame)
            back = covariates._cached("coops_water_level_8452944",
                                      lambda: None)
            check("a populated cache comes back populated",
                  back is not None and len(back) == 1 and "v" in back.columns,
                  None if back is None else str(list(back.columns)))
        finally:
            covariates.CACHE_DIR = original


def test_a_refusal_is_not_an_absence():
    """One gateway timeout out of NOAA killed a 2854-site run at site 405.

    Two separate things have to hold. A source that raises must not end the
    run -- the site goes without that covariate and the pull carries on. And
    the refusal must not be CACHED: 'this gauge has no water temperature' is
    a fact worth keeping, 'the gateway timed out at 14:32' is not, and a
    cache cannot tell them apart after the fact.
    """
    print("\na refusal is not an absence")
    import requests
    with tempfile.TemporaryDirectory() as root:
        original = covariates.CACHE_DIR
        covariates.CACHE_DIR = root
        covariates.reset_sources()
        try:
            def gateway_timeout():
                response = requests.Response()
                response.status_code = 504
                raise covariates.SourceUnavailable("CO-OPS: HTTP 504")

            answer = covariates._cached("coops_water_temperature_8534720",
                                        gateway_timeout)
            check("a source that refuses answers None instead of raising",
                  answer is None)
            check("and nothing is written, so the next run asks again",
                  not os.path.exists(os.path.join(
                      root, "coops_water_temperature_8534720.csv")))

            # Whereas a service that answers 'nothing here' IS cached.
            covariates._cached("coops_water_temperature_9999999",
                               lambda: None)
            check("an answer of 'nothing here' is still cached",
                  os.path.exists(os.path.join(
                      root, "coops_water_temperature_9999999.csv")))

            check("504 reads as a refusal", covariates.is_refusal("HTTP 504"))
            check("429 reads as a refusal",
                  covariates.is_refusal("HTTP 429: Daily API request limit "
                                         "exceeded. Please try again tomorrow."))
            check("a timeout reads as a refusal",
                  covariates.is_refusal("ConnectionError"))
            check("but 'no ocean cell' is an answer about the site, not a "
                  "refusal",
                  not covariates.is_refusal("no ocean cell found within "
                                             "~22 km"))
            check("and a 404 is an answer too",
                  not covariates.is_refusal("HTTP 404"))

            # clear_empty_cache undoes the emptiness cached by the older code,
            # which could not tell a refusal from an absence.
            frame = pd.DataFrame({"t": ["2020-01-01"], "v": [12.5]})
            covariates._cached("coops_water_level_8452944", lambda: frame)
            removed = covariates.clear_empty_cache()
            check("clearing empties removes the cached nothings", removed == 1,
                  f"{removed} removed")
            check("and keeps the populated ones",
                  os.path.exists(os.path.join(
                      root, "coops_water_level_8452944.csv")))
        finally:
            covariates.CACHE_DIR = original
            covariates.reset_sources()


def test_a_covariate_pull_survives_a_gauge_that_refuses():
    """The crash itself, at the level it happened.

    A 504 out of api.tidesandcurrents.noaa.gov came up through
    pull_coops_series, through _cached, through hourly_frame, and out of
    build() -- taking 404 sites' worth of finished work with it. The pull has
    to come back with the sites it could do, and say which source it could
    not reach.
    """
    print("\na gauge that refuses costs one covariate, not the run")
    import types
    import requests

    refused = types.ModuleType("pull_observations")
    refused.COOPS_DATA = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    calls = []

    def raising(station_id, product, start, end):
        calls.append(station_id)
        response = requests.Response()
        response.status_code = 504
        raise requests.exceptions.HTTPError("504 Server Error: Gateway Timeout",
                                            response=response)

    refused.pull_coops_series = raising
    refused.add_tide_state = lambda frame: frame

    sites = pd.DataFrame({"station_id": ["A", "B"],
                          "lat": [41.49, 41.52], "lon": [-71.31, -71.29],
                          "region": ["Atlantic", "Atlantic"]})
    samples = pd.DataFrame({
        "station_id": ["A"] * 3 + ["B"] * 3,
        "sampled_at": pd.date_range("2024-06-01 09:00", periods=6, freq="D",
                                    tz="UTC").astype(str),
        "date": pd.date_range("2024-06-01", periods=6, freq="D").astype(str),
        "analyte": ["enterococcus"] * 6, "value": [10.0] * 6})
    gauges = pd.DataFrame({"station_id": ["8452660"], "lat": [41.50],
                           "lon": [-71.30]})

    with tempfile.TemporaryDirectory() as root:
        keep = (covariates.CACHE_DIR, covariates.era5_for,
                covariates.marine_for, sys.modules.get("pull_observations"))
        covariates.CACHE_DIR = root
        covariates.era5_for = lambda *a, **k: None
        covariates.marine_for = lambda *a, **k: (None, None)
        sys.modules["pull_observations"] = refused
        covariates.reset_sources()
        try:
            joined, meta = covariates.build(
                sites, samples, gauges, None,
                start=datetime(2024, 6, 1), end=datetime(2024, 6, 8),
                progress=False)
        finally:
            (covariates.CACHE_DIR, covariates.era5_for,
             covariates.marine_for) = keep[:3]
            if keep[3] is None:
                sys.modules.pop("pull_observations", None)
            else:
                sys.modules["pull_observations"] = keep[3]
            covariates.reset_sources()

    check("the pull finishes instead of raising", len(joined) == 6,
          f"{len(joined)} row(s)")
    check("every site is still in the output", set(meta["station_id"]) == {"A", "B"},
          str(sorted(meta["station_id"])))
    check("the predictors it could not fetch are NaN, not absent",
          all(c in joined.columns for c in config.PREDICTORS))
    check("and each site records what refused it",
          "unavailable" in meta.columns
          and meta["unavailable"].str.contains("CO-OPS").all(),
          str(list(meta.get("unavailable", []))[:1]))
    check("the gauge was asked for both products at the first site, then "
          "given up on", len(calls) == covariates.CIRCUIT_THRESHOLD,
          f"{len(calls)} request(s)")


def test_a_dead_source_is_given_up_on_and_said_out_loud():
    """A source that is down stays down.

    Making a refusal non-fatal is only half of it. Open-Meteo's daily quota,
    once spent, is spent until tomorrow -- and a run that keeps asking hands
    two thousand more sites an empty column while printing nothing that says
    the run is now worthless.
    """
    print("\na source that is down is given up on, out loud")
    covariates.reset_sources()
    try:
        host = "archive-api.open-meteo.com"
        for _ in range(covariates.CIRCUIT_THRESHOLD):
            covariates._host_record(host, False, "HTTP 429: Daily API request "
                                                 "limit exceeded")
        check("the host is recorded as given up on",
              host in covariates.unavailable_sources())
        raised = None
        try:
            covariates._host_check(host)
        except covariates.SourceUnavailable as exc:
            raised = exc
        check("and the next site is refused without a request",
              raised is not None)
        check("quietly, because saying it 2000 times is not saying it",
              raised is not None and raised.quiet)

        other = "marine-api.open-meteo.com"
        covariates._host_record(other, True)
        covariates._host_record(other, False, "HTTP 503")
        covariates._host_check(other)   # must not raise
        check("a host with one bad response is not given up on",
              other not in covariates.unavailable_sources())
    finally:
        covariates.reset_sources()


def test_a_rate_limited_wave_walk_does_not_claim_there_is_no_ocean():
    """The quiet one. fetch_marine walks 33 cells looking for water and
    reported 'no ocean cell found within ~22 km' whatever went wrong -- so a
    rate-limited walk was recorded, and cached, as a fact about the beach."""
    print("\na refused wave walk does not claim the beach has no ocean")
    try:
        import pull_site_observations as pso
    except ImportError as exc:   # the NDBC SDK, which this check never calls
        print(f"  skip  needs pull_site_observations ({exc})")
        return
    calls = []

    def refused(url, lat, lon, start, end, variables, probe=False, models=None):
        calls.append((lat, lon))
        return None, "HTTP 429: Minutely API request limit exceeded"

    original, sleep = pso.open_meteo, pso.time.sleep
    pso.open_meteo, pso.time.sleep = refused, lambda *_: None
    try:
        frame, note = pso.fetch_marine(41.49, -71.31,
                                       datetime(2024, 1, 1),
                                       datetime(2024, 1, 8))
    finally:
        pso.open_meteo, pso.time.sleep = original, sleep

    check("no frame comes back", frame is None)
    check("and the note says it was refused, not that there is no ocean",
          "429" in note and "no ocean cell found" not in note, note)
    check("the walk stops at the first refusal instead of spending 33 "
          "requests on a quota that has run out", len(calls) == 1,
          f"{len(calls)} request(s)")
    check("so the caller treats it as a refusal", covariates.is_refusal(note))


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


def test_an_empty_layer_has_to_say_why():
    """A zero in the coverage table is four different bugs wearing one face.

    Seven tiles of coastline came back empty through a run that printed no
    coastline error at all. Coverage said 0/120 and stopped there, so the
    question "did Overpass refuse, or is there no coastline in the box, or did
    my own check reject the linework" had no answer anywhere in the output.
    """
    print("\nwhy a layer is empty, not just that it is")
    frame = pd.DataFrame([
        {"station_id": "ok", "tile": "t1", "shore_normal_deg": 12.0,
         "curvature_1_per_km": 0.1, "embayment_ratio": 1.0,
         "land_fraction_5km": 0.3, "fetch_km_mean": 5.0,
         "fetch_km_min": 1.0, "fetch_km_max": 9.0},
        {"station_id": "refused", "tile": "t2",
         "coastline_note": "HTTP 429 overpass-api.de"},
        {"station_id": "refused-too", "tile": "t2",
         "coastline_note": "HTTP 429 overpass-api.de"},
        {"station_id": "empty-box", "tile": "t3",
         "coastline_note": "no natural=coastline within the search box"},
        {"station_id": "silent", "tile": "t4"},
    ])
    table = spatial.outcomes(frame, want={"coastline"})
    reasons = dict(zip(table["reason"], table["sites"]))
    check("the service refusing is its own reason",
          reasons.get("HTTP 429 overpass-api.de") == 2, str(reasons))
    check("an empty search box is a different one",
          reasons.get("no natural=coastline within the search box") == 1)
    check("a populated site is not counted as a failure",
          reasons.get("populated") == 1)
    check("and a site that failed with nothing recorded is named as such, "
          "because that one is a bug in this code",
          reasons.get("EMPTY, NO REASON RECORDED") == 1, str(reasons))
    tiles = dict(zip(table["reason"], table["tiles"]))
    check("two sites failing for one tiled request count as one tile",
          tiles.get("HTTP 429 overpass-api.de") == 1, str(tiles))

    # A layer with no request of its own goes quiet when its parent fails,
    # and a layer you chose not to fetch did not fail at all. Both read as
    # "empty for no reason" until they are told to say otherwise.
    inherited = pd.DataFrame([{
        "station_id": "a", "tile": "t1",
        "echo_note": "not fetched — excluded by --skip-layers on this run",
        "nhdplus_note": "api.water.usgs.gov: HTTP 404 — No catchment found",
        "coastline_note": "overpass-api.de: HTTP 429",
    }])
    table = spatial.outcomes(inherited, want=set(layers.LAYERS))
    reasons = dict(zip(table["layer"], table["reason"]))
    check("a skipped layer says it was skipped, not that it is mysteriously "
          "empty", "--skip-layers" in reasons.get("echo", ""),
          reasons.get("echo"))
    check("nlcd inherits the outcome of the request it rides in on",
          reasons.get("nlcd", "").startswith("via nhdplus:"),
          reasons.get("nlcd"))
    check("and so does nhdplus_vaa",
          reasons.get("nhdplus_vaa", "").startswith("via nhdplus:"),
          reasons.get("nhdplus_vaa"))
    check("no fetched layer is left saying nothing",
          "EMPTY, NO REASON RECORDED" not in
          {reasons.get(k) for k in layers.FETCHED}, str(reasons))

    # The inheritance runs one way only. Once the land cover came from
    # StreamCat instead of NLDI, nlcd stopped depending on nhdplus for
    # anything but the comid -- so a run where the flowline service was down
    # but StreamCat answered reported "nlcd — partial: via nhdplus:
    # watersgeo.epa.gov HTTP 500" at 114 sites whose land cover was fine. A
    # layer that produced its covariates answers for itself.
    answered = pd.DataFrame([{
        "station_id": "a", "tile": "t1",
        "impervious_frac": 0.1175, "developed_frac": 0.1406,
        "nhdplus_note": "watersgeo.epa.gov: HTTP 500 — Service not started",
    }])
    table = spatial.outcomes(answered, want={"nlcd", "nhdplus"})
    reasons = dict(zip(table["layer"], table["reason"]))
    check("a derived layer that answered does not inherit its parent's "
          "failure", reasons.get("nlcd") == "populated", reasons.get("nlcd"))
    check("and the parent still reports its own", "HTTP 500" in
          reasons.get("nhdplus", ""), reasons.get("nhdplus"))

    # Every covariate the coverage table names has to have a layer beside it,
    # or nobody can check its vintage.
    orphans = [name for name in config.SITE_COVARIATES
               if not layers.COVARIATE_LAYER.get(name)]
    check("every pre-registered covariate names the layer it comes from",
          not orphans, str(orphans))


def test_a_cache_from_an_older_schema_is_not_an_answer():
    """StreamCat was wired in, ran green, and populated nothing.

    The land cover fetch was added to the per-station NHD builder, the probe
    confirmed StreamCat answers HTTP 200, and the next run still reported
    impervious_frac 0/120 with the layer EMPTY, NO REASON RECORDED. The cache
    was the whole story: every station already had an nhd_*.json written by
    the previous schema, so the builder never ran, the new key read as None,
    and nothing raised. A cache hit on a payload that predates the keys the
    caller is about to read is not a cache hit -- it is a wrong answer served
    quickly, and it is silent by construction, which is the worst combination
    this pipeline can produce.
    """
    print("\nA cached payload that predates a key is refetched")
    kept = spatial.CACHE_DIR
    spatial.CACHE_DIR = tempfile.mkdtemp()
    try:
        builds = []

        def build_old():
            builds.append("old")
            return {"comid": 42, "flowlines": []}

        def build_new():
            builds.append("new")
            return {"comid": 42, "flowlines": [], "streamcat": {"pctimp": 1.0}}

        old_keys = ("comid", "flowlines")
        new_keys = ("comid", "flowlines", "streamcat")
        spatial._cached_json("nhd_S1", build_old, expects=old_keys)
        spatial._cached_json("nhd_S1", build_old, expects=old_keys)
        check("a payload that has every key it is asked for is reused",
              builds == ["old"], str(builds))
        payload = spatial._cached_json("nhd_S1", build_new, expects=new_keys)
        check("a payload missing a newly-read key is rebuilt",
              builds == ["old", "new"], str(builds))
        check("and the rebuilt payload carries the new key",
              payload.get("streamcat") == {"pctimp": 1.0}, str(payload))
        spatial._cached_json("nhd_S1", build_new, expects=new_keys)
        check("the rebuilt payload is then cached like any other",
              builds == ["old", "new"], str(builds))
        # A key whose VALUE is None is still a key: the builder ran, the fetch
        # failed, and the None is that answer. Refetching it every run would
        # turn one dead endpoint into 120 requests a run, forever.
        def build_none():
            builds.append("none")
            return {"comid": 42, "flowlines": [], "streamcat": None}

        spatial._cached_json("nhd_S2", build_none, expects=new_keys)
        spatial._cached_json("nhd_S2", build_none, expects=new_keys)
        check("a key recorded as None is an answer, not a miss",
              builds == ["old", "new", "none"], str(builds))

        # A key is a shallow test. ECHO's payload kept its "records" key
        # while the records INSIDE it went from useless two-column rows to
        # placed facilities, so `expects` passed and the stale list was
        # served a second time -- the same silent-success failure, one level
        # down. A schema string catches a change in what a fetch MEANS
        # rather than which keys it returns.
        def build_flat():
            builds.append("flat")
            return {"records": [{"SourceID": "A"}]}

        def build_placed():
            builds.append("placed")
            return {"records": [{"SourceID": "A", "FacLat": "1",
                                 "FacLong": "2"}]}

        spatial._cached_json("echo_t1", build_flat, expects=("records",),
                             schema="v1")
        spatial._cached_json("echo_t1", build_flat, expects=("records",),
                             schema="v1")
        check("a payload under the same schema is reused",
              builds[-1] == "flat" and builds.count("flat") == 1, str(builds))
        payload = spatial._cached_json("echo_t1", build_placed,
                                       expects=("records",), schema="v2")
        check("the SAME keys under a new schema are rebuilt anyway",
              builds[-1] == "placed", str(builds))
        check("and the rebuilt records carry what the new schema promises",
              payload["records"][0].get("FacLong") == "2",
              str(payload["records"][0]))
        spatial._cached_json("echo_t1", build_placed, expects=("records",),
                             schema="v2")
        check("then it caches again under the new schema",
              builds.count("placed") == 1, str(builds))
    finally:
        shutil.rmtree(spatial.CACHE_DIR, ignore_errors=True)
        spatial.CACHE_DIR = kept


def main():
    for test in (test_grid_cell_sharing,
                 test_a_cached_nothing_can_be_read_back,
                 test_a_refusal_is_not_an_absence,
                 test_a_covariate_pull_survives_a_gauge_that_refuses,
                 test_a_dead_source_is_given_up_on_and_said_out_loud,
                 test_a_rate_limited_wave_walk_does_not_claim_there_is_no_ocean,
                 test_a_cache_from_an_older_schema_is_not_an_answer,
                 test_an_empty_layer_has_to_say_why,
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
