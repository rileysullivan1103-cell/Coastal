"""Load .env into os.environ on import, so a fresh terminal just works.

Every script here needs WEBCOOS_TOKEN or NOAA_CDO_TOKEN. Those used to come
from `export $(grep -v '^#' .env | xargs)`, which lives and dies with the
shell: close the window and the next run fails with "TOKEN is not set" for no
reason the user did anything to cause. Importing this module reads .env
directly instead.

Rules:
  - an already-set environment variable always wins, so a real export or a CI
    secret is never clobbered by a stale .env
  - .env is looked up beside this file, not in the current directory, so
    running a script from anywhere still finds it
  - a missing .env is not an error here; the scripts report their own missing
    token with a message that says which one

    import env   # noqa: F401
"""

import os

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def parse(text):
    """{KEY: value} from .env text. Ignores blanks, comments and junk lines."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def load(path=ENV_PATH, override=False):
    """Merge path into os.environ. Returns the names it set."""
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        pairs = parse(fh.read())
    applied = []
    for key, value in pairs.items():
        if override or not os.environ.get(key):
            os.environ[key] = value
            applied.append(key)
    return applied


load()
