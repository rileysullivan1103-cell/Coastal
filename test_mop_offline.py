#!/usr/bin/env python3
"""Offline checks for pull_cdip_mop.py — no network.

The parser is written against a real response captured from the server, not
against my idea of the OPeNDAP format. Two turns were lost to guessing an
identifier shape and a URL encoding; these tests pin both.
"""
import sys

import pandas as pd

import pull_cdip_mop as mop
import analyze_drivers as ad

FAILURES = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


# Captured verbatim from
#   .../SC130_hindcast.nc.ascii?waveHs%5B0:1:4%5D
REAL_SINGLE = """Dataset {
    Float32 waveHs[waveTime = 5];
} cdip/model/MOP_alongshore/SC130_hindcast.nc;
---------------------------------------------
waveHs[5]
0.41, 0.43, 0.45, 0.44, 0.42
"""

REAL_SCALARS = """Dataset {
    Float32 metaWaterDepth;
    Float32 metaShoreNormal;
} cdip/model/MOP_alongshore/SC130_hindcast.nc;
---------------------------------------------
metaWaterDepth, 10.0
metaShoreNormal, 175.5
"""

REAL_MULTI = """Dataset {
    Int32 waveTime[waveTime = 3];
    Float32 waveHs[waveTime = 3];
    Byte waveFlagPrimary[waveTime = 3];
} cdip/model/MOP_alongshore/SC130_hindcast.nc;
---------------------------------------------
waveTime[3]
946684800, 946688400, 946692000

waveHs[3]
0.41, 0.43, 0.45

waveFlagPrimary[3]
1, 1, 3
"""

REAL_ERROR = """Error {
    code = 404;
    message = "FileNotFound: No such file or directory";
};
"""


def test_bracket_encoding():
    """The whole reason an earlier probe came back empty. curl and requests
    both refuse a bare '[' with 'bad range in URL', and the failure looks like
    the server returning nothing."""
    print("subscript brackets are percent-encoded before the request")
    got = mop.encode("waveHs[0:1:4],waveTime[0:1:4]")
    check("[ becomes %5B", "%5B" in got and "[" not in got, got)
    check("] becomes %5D", "%5D" in got and "]" not in got, got)
    check("the colons and commas are untouched",
          got == "waveHs%5B0:1:4%5D,waveTime%5B0:1:4%5D", got)


def test_parses_a_real_array_response():
    print("\nthe captured single-variable response parses")
    out = mop.parse_ascii(REAL_SINGLE)
    check("one variable found", list(out) == ["waveHs"], list(out))
    check("five values", len(out["waveHs"]) == 5, out.get("waveHs"))
    check("they convert to float",
          mop.to_float(out["waveHs"])[0] == 0.41, out["waveHs"][:1])
    check("the DDS header is not parsed as data",
          "Dataset" not in out and "Float32" not in out, list(out))


def test_parses_scalars_and_multiple_variables():
    print("\nscalars and multi-variable responses parse")
    out = mop.parse_ascii(REAL_SCALARS)
    check("both scalars found", set(out) == {"metaWaterDepth", "metaShoreNormal"},
          list(out))
    check("shore normal reads back", mop.to_float(out["metaShoreNormal"])[0] == 175.5,
          out["metaShoreNormal"])

    out = mop.parse_ascii(REAL_MULTI)
    check("three variables found",
          set(out) == {"waveTime", "waveHs", "waveFlagPrimary"}, list(out))
    check("each has three values",
          all(len(v) == 3 for v in out.values()), {k: len(v) for k, v in out.items()})
    check("the epoch time converts",
          str(pd.to_datetime(mop.to_float(out["waveTime"])[0], unit="s", utc=True))
          .startswith("2000-01-01"), out["waveTime"][0])


def test_an_error_response_raises():
    """A 404 comes back with HTTP 200 and an Error block in the body. Parsed
    as data it would look like an empty variable set, which is how a missing
    point would quietly become a site with no waves."""
    print("\nan OPeNDAP Error body raises instead of parsing as empty")
    try:
        mop.parse_ascii(REAL_ERROR)
        check("it raises", False, "parse_ascii accepted an Error body")
    except ValueError as exc:
        check("it raises", True)
        check("and quotes the server's message", "FileNotFound" in str(exc))


def test_id_regex_catches_both_shapes():
    """B0001 is one letter and four digits; SC001 is two and three. A pattern
    matching only one silently drops entire counties -- which is exactly how
    Santa Cruz came back as 'not covered'."""
    print("\nthe id pattern catches both naming shapes")
    html = ('<a href="B0001_hindcast.nc">B0001_hindcast.nc</a>'
            '<a href="SC001_hindcast.nc">SC001_hindcast.nc</a>'
            '<a href="SC328_hindcast.nc">SC328_hindcast.nc</a>'
            '<a href="DN001_nowcast.nc">DN001_nowcast.nc</a>')
    found = sorted(set(mop.ID_RE.findall(html)))
    check("one-letter four-digit ids", "B0001" in found, found)
    check("two-letter three-digit ids", "SC001" in found and "SC328" in found, found)
    check("only hindcast entries, so each point counts once",
          "DN001" not in found, found)


def test_dds_length():
    print("\nthe array length is read from the DDS, not assumed")
    dds = ("Dataset {\n"
           "    Int32 waveTime[waveTime = 221328];\n"
           "    Float32 waveHs[waveTime = 221328];\n"
           "    Float32 waveFrequency[waveFrequency = 20];\n"
           "    Float32 metaShoreNormal;\n"
           "} x.nc;")
    sizes = {n: int(s) for n, d, s in mop.DDS_RE.findall(dds) if n == d}
    check("waveTime length found", sizes.get("waveTime") == 221328, sizes)
    check("the frequency axis is not mistaken for it",
          sizes.get("waveFrequency") == 20, sizes)
    have = mop.available(dds, ["waveHs", "waveTa", "metaShoreNormal"])
    check("present variables are detected", "waveHs" in have, have)
    check("absent ones are not requested", "waveTa" not in have, have)
    check("scalars are detected too", "metaShoreNormal" in have, have)


def test_rename_keeps_mean_and_peak_apart():
    """waveTa is an average period and waveTp a peak one. Open-Meteo's
    wave_period is a mean, so mapping waveTp onto it would put a peak period
    in a column every other site fills with a mean."""
    print("\nmean and peak periods do not collide")
    check("average period takes the shared name",
          mop.RENAME["waveTa"] == "wave_period", mop.RENAME["waveTa"])
    check("peak period gets its own", mop.RENAME["waveTp"] == "wave_period_peak",
          mop.RENAME["waveTp"])
    check("mean direction takes the shared name",
          mop.RENAME["waveDm"] == "wave_direction", mop.RENAME["waveDm"])
    check("no two CDIP names map to one of ours",
          len(set(mop.RENAME.values())) == len(mop.RENAME), mop.RENAME)
    check("the shared names are ones analyze_drivers already reads",
          {"wave_height", "wave_period", "wave_direction"} <= set(ad.MARINE_COLUMNS),
          ad.MARINE_COLUMNS)


def test_slug_matches_the_rest_of_the_project():
    print("\nthe output slug matches the other pullers")
    name = "Walton Lighthouse, Santa Cruz, CA"
    check("identical to analyze_drivers.grid_slug",
          mop.grid_slug(name) == ad.grid_slug(name), mop.grid_slug(name))


def test_distance():
    print("\nthe distance the site selection turns on")
    walton = (36.960695, -122.0022)
    sc130 = (36.94782, -122.00476)
    buoy = (36.759, -121.950)
    near = mop.km_between(*walton, *sc130)
    far = mop.km_between(*walton, *buoy)
    check("SC130 is about 1.45 km from Walton", 1.3 < near < 1.6, round(near, 2))
    check("buoy 46236 is about 23 km", 22 < far < 24, round(far, 2))
    check("the MOP point is much closer", far / near > 10, round(far / near, 1))


def test_region_membership_is_anchored():
    """'M' must not swallow 'MA' and 'MO'. startswith() would."""
    print("\nregion membership does not let one prefix absorb another")
    ids = ["M0001", "M0002", "MA0001", "MO0001", "SC001", "SC328", "B0001"]
    grouped = mop.regions(ids)
    check("M holds only its own points", grouped["M"] == ["M0001", "M0002"],
          grouped.get("M"))
    check("MA and MO are separate regions",
          grouped.get("MA") == ["MA0001"] and grouped.get("MO") == ["MO0001"],
          {k: v for k, v in grouped.items() if k in ("MA", "MO")})
    check("SC is found", grouped.get("SC") == ["SC001", "SC328"], grouped.get("SC"))


def test_every_region_is_scored_not_the_first_that_passes():
    """The bug this replaces. Regions run hundreds of km south to north, so a
    region's first point says nothing about whether it contains the site. The
    old loop took the first region whose first point was within 400 km --
    for Santa Cruz that was the Santa Barbara series, and it wrote 229,867
    hours from a point 251 km away under the camera's name."""
    print("\nthe closest region wins, not the first one within a threshold")
    walton = (36.960695, -122.0022)
    coords = {
        # A long southern region: its first point is nearer than its last,
        # and both are hundreds of km off. This is what used to win.
        "B0001": (34.40, -119.70), "B0900": (34.60, -120.20),
        "B1788": (34.97707, -120.65571),
        # The right one, sampled anywhere, is close.
        "SC001": (36.84491, -121.82469), "SC164": (36.94773, -122.00674),
        "SC328": (37.10123, -122.29775),
    }
    ids = sorted(coords)

    real = mop.point_location
    mop.point_location = lambda dataset, cache: coords[dataset]
    try:
        prefix, members = mop.choose_region(*walton, mop.regions(ids), samples=3)
    finally:
        mop.point_location = real
    check("SC is chosen", prefix == "SC", prefix)
    check("and its members come back", members == ["SC001", "SC164", "SC328"],
          members)
    check("B would have been chosen by a first-point-under-400km rule",
          mop.km_between(*walton, *coords["B0001"]) < 400,
          round(mop.km_between(*walton, *coords["B0001"]), 1))


def test_the_distance_guard_exists():
    """A point 251 km away wrote a complete-looking 229,867-row file. A wrong
    file that looks finished is worse than none."""
    print("\na far point is refused rather than written")
    check("there is a maximum", isinstance(mop.MAX_KM, float), mop.MAX_KM)
    check("it is a beach-scale distance, not a regional one",
          1 < mop.MAX_KM < 60, mop.MAX_KM)
    walton = (36.960695, -122.0022)
    bad = mop.km_between(*walton, 34.97707, -120.65571)
    good = mop.km_between(*walton, 36.94782, -122.00476)
    check("the point that was actually written would now be refused",
          bad > mop.MAX_KM, round(bad, 1))
    check("SC130 still passes", good < mop.MAX_KM, round(good, 2))


# Captured verbatim from
#   .../SC130_hindcast.nc.ascii?waveTime%5B120000:1:120002%5D,waveHs...
# waveHs is a healthy 0.47 m at the very timestamps where waveDm, waveSxy and
# waveSxx are the documented _FillValue. The first version of this puller
# turned -999.99 into a float, called notna() on it, and reported the column
# 100.0% populated.
REAL_FILL = """Dataset {
    Float32 waveTime[waveTime = 3];
    Float32 waveHs[waveTime = 3];
    Float32 waveDm[waveTime = 3];
} cdip/model/MOP_alongshore/SC130_hindcast.nc;
---------------------------------------------
waveTime[3]
1378684800, 1378688400, 1378692000

waveHs[3]
0.4663298, 0.46263605, 0.45027834

waveDm[3]
-999.99, -999.99, -999.99
"""


def test_fill_value_is_not_data():
    print("\nCDIP's fill value is not a wave direction")
    values = mop.parse_ascii(REAL_FILL)
    direction = pd.Series(mop.to_float(values["waveDm"]))
    height = pd.Series(mop.to_float(values["waveHs"]))
    check("to_float still turns -999.99 into a float (that is the trap)",
          direction.notna().all())
    check("is_fill catches every one of them", bool(mop.is_fill(direction).all()))
    check("is_fill leaves real wave heights alone",
          not bool(mop.is_fill(height).any()))


def test_denormals_are_not_data():
    print("\ndenormals at the start of the record are not data")
    series = pd.Series([0.0, 0.0, 1.2397983e-33, 0.9])
    flagged = mop.is_denormal(series)
    check("the denormal from the start of SC130's record is caught",
          bool(flagged.iloc[2]))
    check("an exact zero is not called a denormal", not bool(flagged.iloc[0]))
    check("a real value is untouched", not bool(flagged.iloc[3]))


def test_clean_fill_drops_the_unusable_column_and_keeps_the_good_one():
    print("\na column that is all fill is dropped, its neighbours are kept")
    frame = pd.DataFrame({
        "waveHs": [0.47, 0.46, 0.45, 0.44],
        "waveDm": [-999.99, -999.99, 0.0, 1.2397983e-33],
        "waveTa": [8.0, 9.0, -999.99, 10.0],
    })
    cleaned, audit, dropped = mop.clean_fill(frame, ["waveHs", "waveDm", "waveTa"])
    check("waveDm is dropped, not written as wave_direction",
          "waveDm" in dropped and "waveDm" not in cleaned.columns)
    check("waveHs survives untouched", "waveHs" in cleaned.columns
          and cleaned["waveHs"].notna().all())
    check("waveTa survives with its one fill value masked",
          "waveTa" in cleaned.columns and int(cleaned["waveTa"].isna().sum()) == 1,
          f"{audit['waveTa']['share']:.2f} usable")
    check("no rows were dropped to protect a bad column", len(cleaned) == 4)


def test_reported_share_is_honest():
    print("\nthe populated percentage counts fill values as missing")
    frame = pd.DataFrame({"waveDm": [-999.99] * 10})
    cleaned, audit, dropped = mop.clean_fill(frame, ["waveDm"],
                                             keep_degenerate=True)
    share = 100.0 * cleaned["waveDm"].notna().mean()
    check("a column of nothing but fill reports 0% populated, not 100%",
          share == 0.0, f"{share:.1f}%")
    check("the audit counts every fill value", audit["waveDm"]["fill"] == 10)


def test_keep_degenerate_is_an_escape_not_the_default():
    print("\ndropping is the default; keeping it is a flag")
    frame = pd.DataFrame({"waveDm": [-999.99] * 10})
    _, _, dropped = mop.clean_fill(frame.copy(), ["waveDm"])
    _, _, kept = mop.clean_fill(frame.copy(), ["waveDm"], keep_degenerate=True)
    check("dropped by default", dropped == ["waveDm"])
    check("--keep-degenerate writes it anyway", kept == [])


def test_the_probe_reads_the_whole_record_not_just_the_head():
    print("\nthe probe samples across the record, not five rows off the front")
    proj = mop.stride_projection(["waveTime", "waveDm"], 221328, points=400)
    check("it strides rather than taking a contiguous block",
          "waveTime[0:553:221327]" in proj, proj.split(",")[0])
    check("it reaches the last index", proj.endswith("221327]"))
    check("the brackets still survive encoding",
          "%5B0:553:221327%5D" in mop.encode(proj))
    check("a short record does not produce a zero stride",
          "[0:1:9]" in mop.stride_projection(["waveHs"], 10, points=400))


def test_zeros_are_masked_only_where_the_column_is_already_suspect():
    print("\nan exact zero is garbage in a column that holds fill, data elsewhere")
    frame = pd.DataFrame({
        "waveSxx": [0.0, 0.041, -999.99, 0.038, 0.037, 0.039],
        "waveDm":  [0.0, 180.0, 90.0, 270.0, 45.0, 200.0],
    })
    cleaned, audit, dropped = mop.clean_fill(frame, ["waveSxx", "waveDm"])
    check("the zero in the column that also holds -999.99 is masked",
          bool(pd.isna(cleaned["waveSxx"].iloc[0])))
    check("its real values survive", cleaned["waveSxx"].notna().sum() == 4)
    check("a clean column keeps its 0.0, which is due north",
          cleaned["waveDm"].iloc[0] == 0.0)
    check("the audit says the zeros were masked",
          audit["waveSxx"].get("zeros_masked") is True
          and "zeros_masked" not in audit["waveDm"])


def fake_server(drop=(), serve_alone=True, max_span=None):
    """A stand-in for the THREDDS ascii service that can come back short.

    `drop` names variables the server omits from a MULTI-variable response --
    which is what SC130 did on a ten-variable request for 20,000 rows, with
    HTTP 200 and no error. serve_alone/max_span control whether it answers
    those variables when asked on their own, or only over a short span.
    """
    calls = []

    def fetch(dataset, projection, with_text=False):
        calls.append(projection)
        out = {}
        for part in projection.split(","):
            name, _, rest = part.partition("[")
            start, _, stop = rest.rstrip("]").split(":")
            start, stop = int(start), int(stop)
            span = stop - start + 1
            alone = "," not in projection
            withheld = name in drop and (
                not alone or not serve_alone
                or (max_span is not None and span > max_span))
            if not withheld:
                out[name] = [str(float(i)) for i in range(start, stop + 1)]
        text = "Dataset { ... } truncated"
        return (out, text) if with_text else out

    return fetch, calls


def test_a_short_response_falls_back_instead_of_abandoning_the_pull():
    print("\na variable missing from a combined response is fetched alone")
    original = mop.fetch
    try:
        mop.fetch, calls = fake_server(drop=("waveSxx",))
        values = mop.fetch_span("SC130_hindcast.nc",
                                ["waveHs", "waveSxx"], 0, 999)
        check("the missing variable comes back complete",
              len(values.get("waveSxx", [])) == 1000)
        check("the variable that was never missing is intact",
              len(values["waveHs"]) == 1000)
        check("it retried the combined request before splitting",
              calls.count("waveHs[0:1:999],waveSxx[0:1:999]") == 2, len(calls))
        check("and then asked for the short one alone",
              "waveSxx[0:1:999]" in calls)
    finally:
        mop.fetch = original


def test_a_variable_that_needs_a_smaller_span_is_halved():
    print("\na variable the server will only serve in pieces is halved")
    original = mop.fetch
    try:
        mop.fetch, calls = fake_server(drop=("waveSxx",), max_span=600)
        values = mop.fetch_span("SC130_hindcast.nc", ["waveSxx"], 0, 999)
        check("every row still arrives", len(values["waveSxx"]) == 1000)
        check("in order", values["waveSxx"][0] == "0.0"
              and values["waveSxx"][-1] == "999.0")
        check("by halving the span", "waveSxx[0:1:499]" in calls
              and "waveSxx[500:1:999]" in calls)
    finally:
        mop.fetch = original


def test_a_genuinely_missing_variable_still_raises():
    print("\na variable the server never serves is an error, not a silent gap")
    original = mop.fetch
    try:
        mop.fetch, _ = fake_server(drop=("waveSxx",), serve_alone=False)
        try:
            mop.fetch_span("SC130_hindcast.nc", ["waveSxx"], 0, 99, floor=500)
            check("it raises", False)
        except ValueError as exc:
            check("it raises", True)
            check("and says what it asked for and got",
                  "asked for 100 values" in str(exc) and "returned 0" in str(exc))
            check("and shows the response tail to diagnose with",
                  "response tail" in str(exc))
    finally:
        mop.fetch = original


# The response that made the SC130 pull look truncated. waveSxx's data line
# starts with NaN, which has exactly the shape of a scalar declaration
# ("metaWaterDepth, 10.0"), so the parser filed the whole block under a
# variable called "NaN" and reported waveSxx as zero values received.
REAL_NAN_BLOCK = """Dataset {
    Float32 waveHs[waveTime = 4];
    Float32 waveSxx[waveTime = 4];
    Float32 metaWaterDepth;
} cdip/model/MOP_alongshore/SC130_hindcast.nc;
---------------------------------------------
waveHs[4]
0.466, 0.462, 0.450, 0.447

waveSxx[4]
NaN, NaN, NaN, NaN

metaWaterDepth, 15.0
"""


def test_a_data_line_starting_with_nan_is_not_a_variable_name():
    print("\na block of NaNs is missing data, not a variable called NaN")
    values = mop.parse_ascii(REAL_NAN_BLOCK)
    check("waveSxx keeps all four of its values",
          len(values.get("waveSxx", [])) == 4, values.get("waveSxx"))
    check("nothing was filed under 'NaN'", "NaN" not in values, list(values))
    check("the scalar on one line still parses",
          values.get("metaWaterDepth") == ["15.0"])
    check("the healthy column is untouched", len(values["waveHs"]) == 4)
    check("and the NaNs convert to missing, not to numbers",
          all(v != v for v in mop.to_float(values["waveSxx"])))


def test_only_names_the_header_declares_are_variables():
    print("\na name is a variable only if the response's DDS declares it")
    text = """Dataset {
    Float32 waveHs[waveTime = 3];
} x.nc;
---------------------------------------------
waveHs[3]
1.0, Inf, 3.0
"""
    values = mop.parse_ascii(text)
    check("Inf in the middle of a row is a value", len(values["waveHs"]) == 3)
    check("and does not become a variable", list(values) == ["waveHs"])


if __name__ == "__main__":
    test_bracket_encoding()
    test_parses_a_real_array_response()
    test_parses_scalars_and_multiple_variables()
    test_an_error_response_raises()
    test_id_regex_catches_both_shapes()
    test_dds_length()
    test_rename_keeps_mean_and_peak_apart()
    test_slug_matches_the_rest_of_the_project()
    test_distance()
    test_region_membership_is_anchored()
    test_every_region_is_scored_not_the_first_that_passes()
    test_the_distance_guard_exists()
    test_fill_value_is_not_data()
    test_denormals_are_not_data()
    test_clean_fill_drops_the_unusable_column_and_keeps_the_good_one()
    test_reported_share_is_honest()
    test_keep_degenerate_is_an_escape_not_the_default()
    test_the_probe_reads_the_whole_record_not_just_the_head()
    test_zeros_are_masked_only_where_the_column_is_already_suspect()
    test_a_short_response_falls_back_instead_of_abandoning_the_pull()
    test_a_variable_that_needs_a_smaller_span_is_halved()
    test_a_genuinely_missing_variable_still_raises()
    test_a_data_line_starting_with_nan_is_not_a_variable_name()
    test_only_names_the_header_declares_are_variables()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL PASS")
