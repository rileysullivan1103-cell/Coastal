#!/bin/sh
# The national pull is hours of requests. caffeinate is first on the line so
# the machine cannot idle-sleep half way through a state and leave a
# part-written chunk behind.
#
#   ./wq/run_wq.sh --all
#   ./wq/run_wq.sh --results --states CA,FL
#
# run_wq.py also re-execs itself under caffeinate, so running it directly is
# protected too; this script is here so the protection is visible on the
# command line rather than only in the code.
set -eu
cd "$(dirname "$0")/.."
[ -d .venv ] && . .venv/bin/activate

if command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -i -m python -m wq.run_wq --no-caffeinate "$@"
elif command -v systemd-inhibit >/dev/null 2>&1; then
    exec systemd-inhibit --what=idle:sleep --why="national water-quality pull" \
        python -m wq.run_wq --no-caffeinate "$@"
else
    echo "no caffeinate or systemd-inhibit here; running without one"
    exec python -m wq.run_wq --no-caffeinate "$@"
fi
