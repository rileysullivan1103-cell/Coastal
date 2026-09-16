"""Offline checks for pull_tempest.py. No network, no token, no real station.

The things worth testing here are the ones that would go wrong quietly:

  * a token appearing in a log line;
  * day windows that overlap or leave a gap, so readings are fetched twice or
    not at all;
  * treating element 0 as a timestamp when it is milliseconds, or not a time at
    all -- the reason that check exists rather than an assumption;
  * a cached day being re-downloaded, or a failed day being cached as if it had
    worked;
  * retrying an authorisation failure, which only delays the message.

The fixtures carry a known reading count and a known spacing, so "one-minute
data" is something the test asserts rather than something it hopes for.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
from datetime import date, datetime, timezone

import pull_tempest as pt

FAILURES = []


def check(name, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    if not condition:
        FAILURES.append(name)
    print(f"  {mark} {name}" + (f"  {detail}" if detail else ""))


def obs_day(day, spacing=60, count=None, start_hour=0):
    """A day of readings whose first element is an epoch, spaced as asked."""
    begin = int(datetime(day.year, day.month, day.day, start_hour,
                         tzinfo=timezone.utc).timestamp())
    count = count if count is not None else (86400 // spacing)
    return {"obs": [[begin + i * spacing] + [0.0] * 17 for i in range(count)]}


def check_the_token_never_appears():
    print("\nthe token stays out of every log line")
    url = f"{pt.API}/observations/device/12345"
    params = {"token": "SECRET-abc123", "time_start": 1, "time_end": 2}
    shown = pt.safe_url(url, params)
    check("safe_url replaces the token", "SECRET-abc123" not in shown, shown)
    check("and says one was there", "<token>" in shown)
    check("while keeping the rest readable",
          "time_start=1" in shown and "device/12345" in shown)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print(pt.cannot_read_message(224327, pt.safe_url(url, params)))
    check("the failure message carries no token either",
          "SECRET-abc123" not in buffer.getvalue())
    check("and names both fallbacks",
          "owner generates a token" in buffer.getvalue()
          and "Clemson" in buffer.getvalue())


def check_day_windows_tile_the_range():
    print("\nday windows cover the range exactly once")
    windows = pt.day_windows(date(2026, 7, 1), date(2026, 7, 3))
    check("one window per day", len(windows) == 3, str(len(windows)))
    check("the first starts at midnight UTC",
          windows[0][1] == int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()))
    check("each window is a second short of a full day",
          all(end - start == 86399 for _, start, end in windows),
          str([e - s for _, s, e in windows]))
    check("consecutive windows touch without overlapping",
          all(later[1] == earlier[2] + 1
              for earlier, later in zip(windows, windows[1:])))
    check("a single-day range gives one window",
          len(pt.day_windows(date(2026, 7, 1), date(2026, 7, 1))) == 1)
    check("an end before the start gives none",
          pt.day_windows(date(2026, 7, 3), date(2026, 7, 1)) == [])

    # A month boundary is where naive day arithmetic breaks.
    across = pt.day_windows(date(2026, 7, 30), date(2026, 8, 2))
    check("it crosses a month boundary", len(across) == 4,
          str([str(d) for d, _, _ in across]))


def check_epochs_are_verified_not_assumed():
    print("\nis element 0 really a timestamp")
    day = date(2026, 7, 1)
    begin = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp())
    end = begin + 86399

    rows = obs_day(day)["obs"]
    ok, why = pt.epochs_look_right(rows, begin, end)
    check("epoch seconds inside the window pass", ok, why)

    millis = [[r[0] * 1000] + r[1:] for r in rows]
    ok, why = pt.epochs_look_right(millis, begin, end)
    check("milliseconds are caught, not silently used", not ok, why)
    check("and the message says so", "milliseconds" in why, why)

    other_day = obs_day(date(2026, 1, 1))["obs"]
    ok, why = pt.epochs_look_right(other_day, begin, end)
    check("readings from a different day are caught", not ok, why)

    text = [["2026-07-01T00:00:00Z"] + [0.0] * 17]
    ok, why = pt.epochs_look_right(text, begin, end)
    check("a non-numeric first element is caught", not ok, why)

    ok, why = pt.epochs_look_right([], begin, end)
    check("an empty day is not mistaken for valid", not ok, why)


def check_spacing_is_measured():
    print("\nthe spacing between readings is measured, not assumed")
    modal, gaps = pt.spacing_seconds(obs_day(date(2026, 7, 1), spacing=60)["obs"])
    check("one-minute data reports 60 s", modal == 60, str(modal))
    check("and every gap agrees", set(gaps) == {60}, str(gaps))

    modal, _ = pt.spacing_seconds(obs_day(date(2026, 7, 1), spacing=180)["obs"])
    check("three-minute data reports 180 s", modal == 180, str(modal))

    # A record that changes interval half way: 60 s then 300 s. The modal gap
    # must be the commoner one, and the other must still be visible.
    first = obs_day(date(2026, 7, 1), spacing=60, count=600)["obs"]
    last = first[-1][0]
    second = [[last + (i + 1) * 300] + [0.0] * 17 for i in range(50)]
    modal, gaps = pt.spacing_seconds(first + second)
    check("a record that changes interval reports the commoner one",
          modal == 60, str(modal))
    check("and the other interval is still in the tally",
          gaps.get(300, 0) == 50, str(gaps.get(300)))

    check("too few readings to judge returns nothing",
          pt.spacing_seconds([[1], [2]])[0] is None)


def check_caching_and_failures(monkeypatched):
    print("\ncaching, retries, and what must NOT be cached")
    calls = monkeypatched
    with tempfile.TemporaryDirectory() as folder:
        pt.RAW_DIR = os.path.join(folder, "tempest")
        start, end = date(2026, 7, 1), date(2026, 7, 3)

        with contextlib.redirect_stdout(io.StringIO()):
            first = pt.pull(999, "T", start, end, pause=0)
        check("three days fetched on the first run", len(calls) == 3,
              str(len(calls)))
        check("and the readings are counted",
              first["readings"] == 3 * 1440, str(first["readings"]))

        calls.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            pt.pull(999, "T", start, end, pause=0)
        check("a second run re-downloads nothing", not calls, str(len(calls)))

        calls.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            pt.pull(999, "T", start, end, refresh=True, pause=0)
        check("--refresh downloads them again", len(calls) == 3, str(len(calls)))

        path = pt.cache_path(999, date(2026, 7, 1))
        check("each day lands in its own dated file", os.path.exists(path), path)
        with open(path) as handle:
            check("and holds the raw reply, unparsed",
                  isinstance(json.load(handle).get("obs"), list))


def check_a_failed_day_is_not_cached():
    print("\na day that failed must not look like a day that worked")
    original = pt.get
    with tempfile.TemporaryDirectory() as folder:
        pt.RAW_DIR = os.path.join(folder, "tempest")
        pt.get = lambda url, params, label="": (None, "HTTP 500 after 4 tries.")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                pt.pull(999, "T", date(2026, 7, 1), date(2026, 7, 1), pause=0)
            check("nothing was written for the failed day",
                  not os.path.exists(pt.cache_path(999, date(2026, 7, 1))))
        finally:
            pt.get = original


def check_auth_failure_stops_immediately():
    print("\nan authorisation failure is not retried")
    attempts = []

    class Response:
        status_code = 403
        text = '{"status":{"status_message":"UNAUTHORIZED"}}'

        def json(self):
            return json.loads(self.text)

    def fake_get(url, params=None, timeout=None):
        attempts.append(url)
        return Response()

    original = pt.requests.get
    pt.requests.get = fake_get
    try:
        payload, error = pt.get(f"{pt.API}/stations/224327", {"token": "T"})
    finally:
        pt.requests.get = original
    check("tried exactly once, not four times", len(attempts) == 1,
          str(len(attempts)))
    check("returned no payload", payload is None)
    check("and surfaced the server's own words",
          "UNAUTHORIZED" in error, error[:80])
    check("with the status code", "403" in error)


def main():
    print("Tempest download offline checks")
    check_the_token_never_appears()
    check_day_windows_tile_the_range()
    check_epochs_are_verified_not_assumed()
    check_spacing_is_measured()

    calls = []
    original = pt.get

    def fake_get(url, params, label=""):
        calls.append(url)
        begin = params["time_start"]
        day = datetime.fromtimestamp(begin, tz=timezone.utc).date()
        return obs_day(day), None

    pt.get = fake_get
    try:
        check_caching_and_failures(calls)
    finally:
        pt.get = original
    check_a_failed_day_is_not_cached()
    check_auth_failure_stops_immediately()

    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
