"""Check that this checkout can actually run, and say what to fix if not.

Written for the case where a terminal was closed and reopened. Two things do
not survive that: the activated virtualenv, and any tokens that were exported
into the shell. Both produce errors that look like the code broke when nothing
about the code changed.

Deliberately imports nothing beyond the standard library, so it still runs on
a bare system python where pandas is missing -- which is one of the things it
is trying to detect.

    python doctor.py
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NEEDED = ["pandas", "requests", "numpy", "ndbc_api", "pywebcoos"]
TOKENS = ["WEBCOOS_TOKEN", "NOAA_CDO_TOKEN"]

problems = []   # shell commands to run
notes = []      # things to know that are not commands


def line(ok, text, fix=None):
    print(("  ok    " if ok else "  BROKEN ") + text)
    if not ok and fix:
        problems.append(fix)


print(f"\ncheckout   {HERE}")
print(f"cwd        {os.getcwd()}")
print(f"python     {sys.executable}")

# --- virtualenv -----------------------------------------------------------
venv = os.path.join(HERE, ".venv")
in_venv = sys.prefix != sys.base_prefix
using_ours = os.path.abspath(sys.prefix) == os.path.abspath(venv)
print("\nenvironment")
if os.path.isdir(venv):
    line(using_ours, "running from this project's .venv"
         if using_ours else f"NOT running from {venv}",
         f"source {venv}/bin/activate")
else:
    line(in_venv, "inside a virtualenv" if in_venv else "no .venv in this checkout",
         f"python3 -m venv {venv} && source {venv}/bin/activate"
         f" && pip install -r {os.path.join(HERE, 'requirements.txt')}")

# --- packages -------------------------------------------------------------
print("\npackages")
missing = []
for name in NEEDED:
    found = importlib.util.find_spec(name) is not None
    line(found, name if found else f"{name} not importable")
    if not found:
        missing.append(name)
if missing:
    problems.append(f"pip install -r {os.path.join(HERE, 'requirements.txt')}")
    if "pywebcoos" in missing:
        notes.append("pywebcoos is not on PyPI; requirements.txt carries the git URL,"
                     " so install from the file rather than by name")

# --- tokens ---------------------------------------------------------------
print("\ntokens")
env_path = os.path.join(HERE, ".env")
if os.path.exists(env_path):
    print(f"  .env found at {env_path}")
else:
    line(False, f"no .env at {env_path}",
         f"cp {os.path.join(HERE, '.env.example')} {env_path} and fill in both tokens")

sys.path.insert(0, HERE)
try:
    import env as _env  # noqa: F401  -- loading it is the point
except Exception as exc:  # noqa: BLE001
    line(False, f"env.py failed to load: {exc}")

for name in TOKENS:
    value = os.environ.get(name)
    line(bool(value),
         f"{name} present ({len(value)} chars)" if value else f"{name} is empty",
         f"add {name}=... to {env_path}")

# --- data files -----------------------------------------------------------
print("\nfiles the scripts expect")
for path, made_by in [
    ("candidate_sites_ranked.csv", "find_candidate_sites.py"),
    ("webcoos_assets_raw.json", "verify_webcoos_fields.py, or --refresh"),
    ("data", "pull_observations.py"),
]:
    full = os.path.join(HERE, path)
    exists = os.path.exists(full)
    print(("  ok     " if exists else "  absent ") + f"{path}"
          + ("" if exists else f"   (written by {made_by})"))

# --- verdict --------------------------------------------------------------
print()
if problems:
    print("Run these, in order:\n")
    seen = set()
    for fix in problems:
        if fix not in seen:
            seen.add(fix)
            print(f"    {fix}")
    for note in notes:
        print(f"\nnote: {note}")
    print()
    sys.exit(1)
print("Nothing missing. If a script still fails, the error is real —")
print("paste it rather than re-running.\n")
