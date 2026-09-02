#!/usr/bin/env python3
"""Offline checks for pull_storm_events.py — no network."""
import io
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

import pull_storm_events as se

FAILURES = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def test_filename_regex():
    print("\nthe yearly filename is discovered, never constructed")
    listing = """
      <a href="StormEvents_details-ftp_v1.0_d1999_c20220425.csv.gz">x</a>
      <a href="StormEvents_details-ftp_v1.0_d2023_c20240416.csv.gz">x</a>
      <a href="StormEvents_details-ftp_v1.0_d2023_c20250115.csv.gz">x</a>
      <a href="StormEvents_fatalities-ftp_v1.0_d2023_c20240416.csv.gz">x</a>
      <a href="StormEvents_locations-ftp_v1.0_d2023_c20240416.csv.gz">x</a>
    """
    found = {}
    for match in se.DETAILS_RE.finditer(listing):
        year, name = int(match.group(1)), match.group(0)
        if year not in found or name > found[year]:
            found[year] = name
    check("both years found", set(found) == {1999, 2023}, sorted(found))
    check("the later creation stamp wins",
          found[2023].endswith("c20250115.csv.gz"), found[2023])
    check("fatalities and locations files are not mistaken for details",
          all("details" in n for n in found.values()), list(found.values()))


def test_daily_table_has_zero_days():
    print("\nquiet days are written out, not left as gaps")
    rips = pd.DataFrame({
        se.COL_BEGIN: ["04-JUL-21 14:30:00", "04-JUL-21 16:00:00",
                       "20-AUG-21 11:00:00"],
        se.COL_DEATHS: [1, 0, 2],
        se.COL_INJURIES: [0, 3, 1],
    })
    frame = se.daily_table(rips, [2021], "test")
    check("every day of the span is present", len(frame) == 365, len(frame))
    check("only two days carry an event", int(frame["any_event"].sum()) == 2,
          int(frame["any_event"].sum()))
    check("same-day events are summed",
          int(frame.loc["2021-07-04", "events"]) == 2,
          int(frame.loc["2021-07-04", "events"]))
    check("deaths total across the year", int(frame["deaths"].sum()) == 3,
          int(frame["deaths"].sum()))
    check("a quiet day is a real zero, not NaN",
          frame.loc["2021-07-05", "events"] == 0
          and frame["events"].notna().all())
    check("casualty flag catches injuries as well as deaths",
          int(frame["any_casualty"].sum()) == 2,
          int(frame["any_casualty"].sum()))


def test_unreadable_dates_are_dropped_not_guessed():
    print("\nrows with no readable date are dropped and counted")
    rips = pd.DataFrame({
        se.COL_BEGIN: ["04-JUL-21 14:30:00", "not a date", None],
        se.COL_DEATHS: [1, 5, 5],
        se.COL_INJURIES: [0, 0, 0],
    })
    frame = se.daily_table(rips, [2021], "test")
    check("only the parseable row survives", int(frame["events"].sum()) == 1,
          int(frame["events"].sum()))
    check("their deaths do not leak into the total",
          int(frame["deaths"].sum()) == 1, int(frame["deaths"].sum()))


def test_missing_columns_are_named():
    print("\na renamed column stops the run rather than emptying a field")
    frame = pd.DataFrame({"EVENT_TYPE": ["Rip Current"], "STATE": ["VIRGINIA"]})
    try:
        se.check_columns(frame)
    except SystemExit as exc:
        text = str(exc)
        check("it exits", True)
        check("it names the missing columns",
              se.COL_CZ_NAME in text and se.COL_BEGIN in text, text[-90:])
        return
    check("it exits", False, "check_columns accepted an incomplete frame")


def test_slug_matches_the_rest_of_the_project():
    print("\none slug rule across the project")
    check("spaces and commas collapse",
          se.slugify("Virginia Beach, VA") == "virginia-beach-va",
          se.slugify("Virginia Beach, VA"))
    try:
        import analyze_drivers as a
        check("identical to the rip-table slug",
              se.slugify("Walton Lighthouse, Santa Cruz, CA")
              == a.rip_slug("Walton Lighthouse, Santa Cruz, CA"))
    except Exception as exc:  # noqa: BLE001
        check("identical to the rip-table slug", False, str(exc))


def test_exposure_skew_is_measurable():
    print("\nthe weekend skew the report warns about is computable")
    days = pd.date_range("2021-06-01", "2021-08-31", freq="D")
    frame = pd.DataFrame(index=days)
    frame["events"] = np.where(days.dayofweek >= 5, 3, 0)
    by_dow = frame.groupby(frame.index.dayofweek)["events"].sum()
    weekend = by_dow.get(5, 0) + by_dow.get(6, 0)
    share = 100 * weekend / by_dow.sum()
    check("a weekend-only series reads as 100%", round(share) == 100, share)
    check("and would trip the 40% warning", share > 40, share)


def _events(rows):
    """rows of (state, zone, date, deaths, injuries, cz) as a details frame."""
    return pd.DataFrame({
        se.COL_STATE: [r[0] for r in rows],
        se.COL_CZ_NAME: [r[1] for r in rows],
        se.COL_BEGIN: [r[2] for r in rows],
        se.COL_DEATHS: [r[3] for r in rows],
        se.COL_INJURIES: [r[4] for r in rows],
        se.COL_CZ_TYPE: [r[5] for r in rows],
    })


def test_a_renamed_zone_is_visible_not_silent():
    """The listing showed NEW HANOVER (16 events) and COASTAL NEW HANOVER (58)
    as separate rows. If those are one stretch of coast renamed, analysing
    either alone gives a series whose quiet years are a filing change. The
    year span is what makes that legible, so it has to be in the table."""
    print("\na zone renamed mid-record is reported, not silently split")
    frame = _events([
        ("NORTH CAROLINA", "NEW HANOVER", "01-JUL-2001 15:00:00", 1, 0, "C"),
        ("NORTH CAROLINA", "NEW HANOVER", "04-JUL-2003 15:00:00", 0, 2, "C"),
        ("NORTH CAROLINA", "COASTAL NEW HANOVER", "02-JUL-2015 15:00:00", 1, 1, "Z"),
        ("NORTH CAROLINA", "COASTAL NEW HANOVER", "09-AUG-2020 15:00:00", 0, 0, "Z"),
    ])
    table = se.zone_table(frame)
    check("both names are listed", len(table) == 2, list(table.index))
    old = table.loc[("NORTH CAROLINA", "NEW HANOVER")]
    new = table.loc[("NORTH CAROLINA", "COASTAL NEW HANOVER")]
    check("the old name's span ends where the record does",
          (int(old["first"]), int(old["last"])) == (2001, 2003),
          (old["first"], old["last"]))
    check("the new name's span starts after it",
          int(new["first"]) > int(old["last"]),
          (old["last"], new["first"]))
    check("casualties are totalled per name",
          int(old["deaths"]) == 1 and int(old["injuries"]) == 2,
          dict(old))
    check("the CZ_TYPE change is carried too (a county became a zone)",
          old["cz"] == "C" and new["cz"] == "Z", (old["cz"], new["cz"]))


def test_pooled_names_are_judged_by_whether_their_spans_overlap():
    """Pooling two names is right for a rename and wrong for two real places.
    The only evidence available is whether their years overlap, so that is
    what decides which message the user gets."""
    print("\npooling is called a repair or a conflation on the evidence")
    renamed = _events([
        ("FLORIDA", "BAY", "01-JUL-2001 15:00:00", 1, 0, "C"),
        ("FLORIDA", "COASTAL BAY", "01-JUL-2015 15:00:00", 1, 0, "Z"),
    ])
    out = io.StringIO()
    with redirect_stdout(out):
        se.describe_match(renamed)
    text = out.getvalue()
    check("disjoint spans read as one place renamed",
          "renamed" in text and "conflat" not in text, text.strip()[-120:])

    distinct = _events([
        ("FLORIDA", "COASTAL BAY", "01-JUL-2015 15:00:00", 1, 0, "Z"),
        ("FLORIDA", "BAY COUNTY INLAND", "02-JUL-2015 15:00:00", 1, 0, "Z"),
    ])
    out = io.StringIO()
    with redirect_stdout(out):
        se.describe_match(distinct)
    text = out.getvalue()
    check("overlapping spans read as two different places",
          "conflat" in text and "renamed" not in text, text.strip()[-120:])


def test_state_filter_lists_every_zone_in_it():
    """231 zones print as a top-50, so a site outside the top 50 -- which is
    where every camera in this project sits -- is invisible without this."""
    print("\n--state lists all of that state's zones, not the national top 50")
    rows = [("VIRGINIA", f"ZONE {i}", "01-JUL-2010 15:00:00", 0, 0, "Z")
            for i in range(4)]
    rows += [("FLORIDA", "COASTAL BAY", "01-JUL-2010 15:00:00", 0, 0, "Z")] * 60
    frame = _events(rows)
    out = io.StringIO()
    with redirect_stdout(out):
        se.show_zones(frame, [2010], state="virginia")
    text = out.getvalue()
    check("all four Virginia zones appear",
          all(f"ZONE {i}" in text for i in range(4)), text)
    check("the busy Florida zone is excluded", "COASTAL BAY" not in text, text)
    check("the header says the listing is filtered", "VIRGINIA ONLY" in text,
          text.splitlines()[1] if len(text.splitlines()) > 1 else text)


if __name__ == "__main__":
    test_filename_regex()
    test_daily_table_has_zero_days()
    test_unreadable_dates_are_dropped_not_guessed()
    test_missing_columns_are_named()
    test_slug_matches_the_rest_of_the_project()
    test_exposure_skew_is_measurable()
    test_a_renamed_zone_is_visible_not_silent()
    test_pooled_names_are_judged_by_whether_their_spans_overlap()
    test_state_filter_lists_every_zone_in_it()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        sys.exit(1)
    print("ALL PASS")
