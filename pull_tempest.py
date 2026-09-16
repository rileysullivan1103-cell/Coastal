"""Download observations from a Tempest (WeatherFlow) weather station.

Riley's collaborator put a Tempest at Walton Lighthouse around 2026-07-01, which
gives this project something it has never had: wind and rain MEASURED at the
camera, against the ERA5 reanalysis it has been using. 774 of Walton's imagery
hours fall in that window.

WHAT THIS SCRIPT DOES, IN PLAIN LANGUAGE

  --probe   asks the API what the station is: its name, timezone, position,
            and the devices attached to it. You need the device_id it prints
            before anything else can run. It also reports whatever the station
            says about how high the sensor is mounted, because that decides
            whether the ERA5 comparison needs a height adjustment.

  --pull    downloads observations one UTC day at a time and saves each day's
            raw reply to disk. Re-running skips days already saved, so an
            interrupted download picks up where it stopped instead of starting
            over.

THREE THINGS IT DELIBERATELY DOES NOT DO.

  * It never prints your token. The token is read from the TEMPEST_TOKEN
    environment variable, and every URL this script logs has it stripped out.
    `.env` and `data/` are already in .gitignore, so neither the token nor the
    downloads can be committed by accident.
  * It does not parse the observation arrays. Tempest returns each reading as a
    bare list of numbers whose meaning is positional, and the field order has
    to come from the official documentation rather than from memory. This
    script saves the raw replies; pull_tempest_parse.py (next step) does the
    parsing once that order is confirmed.
  * It does not assume the reply's shape. It checks what came back and says so,
    including the actual spacing between readings, because "the API returns
    one-minute data" is the kind of thing that is true until it is not.

    export TEMPEST_TOKEN=...
    python pull_tempest.py --probe
    python pull_tempest.py --pull --start 2026-07-01 --end 2026-09-16
    python pull_tempest.py --pull --start 2026-07-01 --end 2026-07-02   (a trial)
"""

import env  # noqa: F401  -- loads .env into os.environ

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

API = "https://swd.weatherflow.com/swd/rest"
STATION_ID = 224327
RAW_DIR = "data/raw/tempest"
TOKEN_VAR = "TEMPEST_TOKEN"

# Polite by default. The API publishes no hard rate limit that this project has
# verified, so the gap is generous rather than tuned: 76 days at one request a
# second is under two minutes, and there is no reason to push it.
PAUSE_SECONDS = 1.0
RETRIES = 4
TIMEOUT = 60

# A Tempest epoch is seconds, not milliseconds. Both are plausible-looking
# integers, which is exactly why index 0 is CHECKED against the window that was
# requested rather than trusted.
EPOCH_FLOOR = 1_000_000_000      # 2001-09-09, older than any Tempest
EPOCH_CEILING = 4_000_000_000    # 2096


def token_or_exit():
    token = os.environ.get(TOKEN_VAR, "").strip()
    if not token:
        sys.exit(
            f"{TOKEN_VAR} is not set.\n"
            "  Get a personal token from https://tempestwx.com/settings/tokens\n"
            "  then either:\n"
            f"    export {TOKEN_VAR}=...        (this shell only)\n"
            f"    echo '{TOKEN_VAR}=...' >> .env   (persistent; .env is gitignored)")
    return token


def safe_url(url, params):
    """The URL with the token replaced, for logging. Never log the raw one."""
    shown = {k: ("<token>" if k == "token" else v) for k, v in params.items()}
    query = "&".join(f"{k}={v}" for k, v in shown.items())
    return f"{url}?{query}"


def get(url, params, label=""):
    """One GET with retries. Returns (payload, error_text).

    Retries on a network error or a 429/5xx, because those are worth trying
    again. A 401/403/404 is not: it means the token cannot see this station,
    and retrying an authorisation failure four times just delays the message.
    """
    delay = 2.0
    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=TIMEOUT)
        except requests.RequestException as exc:
            if attempt == RETRIES:
                return None, f"network error after {RETRIES} tries: {exc}"
            print(f"    {label}network error ({exc}); retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2
            continue

        if response.status_code == 200:
            try:
                return response.json(), None
            except ValueError as exc:
                return None, f"HTTP 200 but the body is not JSON: {exc}\n" \
                             f"  first 300 characters: {response.text[:300]}"

        if response.status_code in (401, 403, 404):
            return None, (f"HTTP {response.status_code} — the token cannot read "
                          f"this.\n  {safe_url(url, params)}\n"
                          f"  body: {response.text[:500]}")

        if attempt == RETRIES:
            return None, (f"HTTP {response.status_code} after {RETRIES} tries.\n"
                          f"  body: {response.text[:300]}")
        wait = delay + random.uniform(0, 1)
        print(f"    {label}HTTP {response.status_code}; retrying in {wait:.0f}s")
        time.sleep(wait)
        delay *= 2
    return None, "exhausted retries"


def cannot_read_message(station, detail):
    return (
        f"\nCannot read station {station}.\n\n{detail}\n\n"
        "  This is the case the plan anticipated: a Tempest personal token only\n"
        "  reads stations the account owns or that are shared with it. Nothing\n"
        "  in this script can work around that, and scraping the website is not\n"
        "  an option we are taking.\n\n"
        "  The two ways forward, both needing the station's owner:\n"
        "    (a) the owner generates a token on their own account and shares it,\n"
        "        or exports a CSV from https://tempestwx.com/station/"
        f"{station}/ ;\n"
        "    (b) Clemson's archive, if this station reports into it.\n\n"
        "  Paste the message above to the owner — it says exactly what failed.")


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def describe_station(payload, station_id):
    stations = payload.get("stations") or []
    match = next((s for s in stations
                  if str(s.get("station_id")) == str(station_id)), None)
    if match is None and stations:
        match = stations[0]
    if match is None:
        print("  the reply carried no stations at all; raw keys: "
              f"{list(payload.keys())}")
        return []

    print(f"\n  station {match.get('station_id')}: {match.get('name')!r}")
    print(f"  public name : {match.get('public_name')!r}")
    print(f"  timezone    : {match.get('timezone')}  "
          f"(offset {match.get('timezone_offset_minutes')} min)")
    print(f"  position    : {match.get('latitude')}, {match.get('longitude')}")
    print(f"  elevation   : {match.get('station_meta', {}).get('elevation')} m "
          "(ground height above sea level, not sensor height)")

    devices = match.get("devices") or []
    print(f"\n  {len(devices)} device(s):")
    out = []
    for device in devices:
        meta = device.get("device_meta") or {}
        kind = device.get("device_type")
        print(f"    device_id {device.get('device_id')}  type {kind}  "
              f"serial {device.get('serial_number')}")
        print(f"      name          : {device.get('device_meta', {}).get('name')}")
        print(f"      firmware      : {device.get('firmware_revision')}")
        print(f"      agl (height)  : {meta.get('agl')}"
              + ("  <- metres above ground, AS ENTERED BY THE OWNER"
                 if meta.get("agl") is not None else "  <- not reported"))
        if kind == "ST":
            out.append(device.get("device_id"))
    if not out:
        print("\n  NO DEVICE OF TYPE 'ST'. A Tempest all-in-one reports as ST;")
        print("  an older Air/Sky pair reports as AR and SK and would need a")
        print("  different parser. Tell me what types are listed above.")
    print("\n  ON SENSOR HEIGHT. 'agl' is typed in by whoever set the station")
    print("  up and is frequently left at a default. Treat it as a claim, not a")
    print("  measurement: if it reads exactly 1, 2 or 10 it is probably untouched.")
    print("  ERA5 wind is at 10 m, so a real height is what decides whether the")
    print("  comparison needs a log-profile adjustment or just a caveat.")
    return out


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------

def day_windows(start, end):
    """[(date, start_epoch, end_epoch)] covering start..end, one UTC day each.

    Half-open on the right: a day runs to 23:59:59 and the next begins at
    00:00:00, so no reading is fetched twice and none falls between two days.
    """
    out = []
    day = start
    while day <= end:
        begin = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        out.append((day, int(begin.timestamp()),
                    int((begin + timedelta(days=1)).timestamp()) - 1))
        day += timedelta(days=1)
    return out


def cache_path(device_id, day):
    return os.path.join(RAW_DIR, str(device_id), f"{day:%Y-%m-%d}.json")


def observations_of(payload):
    return (payload or {}).get("obs") or []


def epochs_look_right(rows, window_start, window_end, slack=86_400):
    """Is element 0 of each reading the timestamp, in seconds?

    Checked rather than assumed. Epoch seconds and epoch milliseconds are both
    plausible-looking integers, and a reading whose first element is something
    else entirely would otherwise be silently treated as a time. The test is
    that the values land inside the day that was asked for.
    """
    values = [r[0] for r in rows if isinstance(r, (list, tuple)) and r]
    if not values:
        return False, "no readings carried a first element"
    numeric = [v for v in values if isinstance(v, (int, float))]
    if len(numeric) != len(values):
        return False, "some first elements are not numbers"
    low, high = min(numeric), max(numeric)
    if not (EPOCH_FLOOR <= low <= EPOCH_CEILING):
        hint = " (looks like milliseconds)" if low > EPOCH_CEILING else ""
        return False, f"first elements range {low}..{high}{hint}"
    if high < window_start - slack or low > window_end + slack:
        return False, (f"first elements {low}..{high} fall outside the day "
                       f"requested ({window_start}..{window_end})")
    return True, f"{len(numeric)} readings, epoch seconds, inside the window"


def spacing_seconds(rows):
    """The most common gap between consecutive readings, and how varied it is."""
    stamps = sorted(r[0] for r in rows
                    if isinstance(r, (list, tuple)) and r
                    and isinstance(r[0], (int, float)))
    if len(stamps) < 3:
        return None, {}
    gaps = {}
    for earlier, later in zip(stamps, stamps[1:]):
        gap = int(later - earlier)
        gaps[gap] = gaps.get(gap, 0) + 1
    modal = max(gaps, key=gaps.get)
    return modal, gaps


def pull(device_id, token, start, end, refresh=False, pause=PAUSE_SECONDS):
    os.makedirs(os.path.join(RAW_DIR, str(device_id)), exist_ok=True)
    windows = day_windows(start, end)
    print(f"\n  {len(windows)} day(s) from {start} to {end}")
    downloaded = cached = failed = 0
    totals, all_gaps = [], {}

    for index, (day, begin, finish) in enumerate(windows, start=1):
        path = cache_path(device_id, day)
        label = f"[{index}/{len(windows)}] {day:%Y-%m-%d} "
        if os.path.exists(path) and not refresh:
            with open(path) as handle:
                payload = json.load(handle)
            cached += 1
        else:
            payload, error = get(f"{API}/observations/device/{device_id}",
                                 {"token": token, "time_start": begin,
                                  "time_end": finish}, label=label)
            if error:
                print(f"  {label}FAILED: {error.splitlines()[0]}")
                failed += 1
                if "cannot read this" in error:
                    print(cannot_read_message(STATION_ID, error))
                    return None
                continue
            with open(path, "w") as handle:
                json.dump(payload, handle)
            downloaded += 1
            time.sleep(pause)

        rows = observations_of(payload)
        totals.append(len(rows))
        if rows:
            ok, why = epochs_look_right(rows, begin, finish)
            if not ok:
                print(f"  {label}*** the first element is not a timestamp: "
                      f"{why}")
            modal, gaps = spacing_seconds(rows)
            for gap, count in gaps.items():
                all_gaps[gap] = all_gaps.get(gap, 0) + count
        if index % 10 == 0 or index == len(windows):
            print(f"  {label}{len(rows)} readings "
                  f"({downloaded} new, {cached} cached, {failed} failed)")

    print(f"\n  {downloaded} day(s) downloaded, {cached} already on disk, "
          f"{failed} failed")
    if not totals:
        print("  nothing came back")
        return None
    print(f"  readings per day: min {min(totals)}, median "
          f"{sorted(totals)[len(totals) // 2]}, max {max(totals)}")
    empty = sum(1 for t in totals if t == 0)
    if empty:
        print(f"  {empty} day(s) returned NOTHING — before the install date, or "
              "the station was down")

    if all_gaps:
        modal = max(all_gaps, key=all_gaps.get)
        share = all_gaps[modal] / sum(all_gaps.values())
        print(f"\n  ACTUAL SPACING BETWEEN READINGS: {modal} s "
              f"({share:.0%} of gaps), not assumed")
        others = sorted(((c, g) for g, c in all_gaps.items() if g != modal),
                        reverse=True)[:4]
        if others:
            print("  other gaps seen: "
                  + ", ".join(f"{g}s x{c}" for c, g in others))
        if share < 0.9:
            print("  Under 90% at one spacing — the interval CHANGED during the")
            print("  record, or there are gaps. The QC step will pin down when.")
    print(f"\n  raw replies are under {RAW_DIR}/{device_id}/ ; re-running skips "
          "them")
    return {"days": len(totals), "readings": sum(totals)}


# ---------------------------------------------------------------------------

def parse_day(text):
    return datetime.strptime(text, "%Y-%m-%d").date()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--station", type=int, default=STATION_ID)
    parser.add_argument("--probe", action="store_true",
                        help="ask what the station is, and stop")
    parser.add_argument("--pull", action="store_true",
                        help="download observations day by day")
    parser.add_argument("--device", type=int, default=None,
                        help="device_id; taken from --probe if not given")
    parser.add_argument("--start", default="2026-07-01")
    parser.add_argument("--end", default=None, help="default: today, UTC")
    parser.add_argument("--refresh", action="store_true",
                        help="re-download days already cached")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS,
                        help=f"seconds between requests (default {PAUSE_SECONDS})")
    args = parser.parse_args()

    if not args.probe and not args.pull:
        parser.error("pass --probe or --pull")
    token = token_or_exit()

    print(f"\n{'=' * 74}\nTEMPEST STATION {args.station}\n{'=' * 74}")
    payload, error = get(f"{API}/stations/{args.station}", {"token": token})
    if error:
        print(cannot_read_message(args.station, error))
        return 1
    devices = describe_station(payload, args.station)

    if args.probe:
        print("\n  Next, once the device_id above looks right:")
        print(f"    python pull_tempest.py --pull --start {args.start} "
              "--end 2026-07-02")
        print("  That trial pulls two days. If it reports a sensible number of")
        print("  readings and a sensible spacing, widen --end to today.")
        return 0

    device = args.device or (devices[0] if devices else None)
    if device is None:
        sys.exit("  no Tempest (ST) device found and --device not given.")
    print(f"\n  using device {device}")
    end = parse_day(args.end) if args.end else datetime.now(timezone.utc).date()
    start = parse_day(args.start)
    if start > end:
        sys.exit(f"  --start {start} is after --end {end}")
    result = pull(device, token, start, end, args.refresh, args.pause)
    if result is None:
        return 1
    print("\n  Nothing has been parsed yet — these are the raw replies. The")
    print("  field order inside each reading has to come from the Tempest docs")
    print("  before any wind number is trusted; see the note in the message")
    print("  that came with this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
