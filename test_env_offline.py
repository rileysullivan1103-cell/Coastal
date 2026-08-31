"""Checks for env.py, the .env loader. No network, no real tokens.

The loader is the thing standing between a fresh terminal and a run that
fails for no reason to do with the code, so its edge cases are worth pinning:
quoted values, `export` prefixes, comments, and above all not clobbering a
variable that is already set.

    python test_env_offline.py
"""

import os
import tempfile

import env

FAILURES = []


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def test_parse():
    print("\nparse")
    parsed = env.parse(
        "# a comment\n"
        "\n"
        "PLAIN=value\n"
        "export PREFIXED=exported\n"
        "  SPACED  =  padded  \n"
        'DQUOTED="double"\n'
        "SQUOTED='single'\n"
        "EMPTY=\n"
        "not_a_pair\n"
        "WITH_EQUALS=a=b=c\n"
    )
    check("plain pair", parsed.get("PLAIN") == "value")
    check("export prefix is stripped", parsed.get("PREFIXED") == "exported",
          parsed.get("PREFIXED"))
    check("whitespace around key and value is trimmed",
          parsed.get("SPACED") == "padded", repr(parsed.get("SPACED")))
    check("double quotes stripped", parsed.get("DQUOTED") == "double")
    check("single quotes stripped", parsed.get("SQUOTED") == "single")
    check("empty value kept as empty string", parsed.get("EMPTY") == "")
    check("comment ignored", "# a comment" not in parsed)
    check("line without = ignored", "not_a_pair" not in parsed)
    check("only the first = splits", parsed.get("WITH_EQUALS") == "a=b=c",
          parsed.get("WITH_EQUALS"))


def write_env(text):
    handle = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
    handle.write(text)
    handle.close()
    return handle.name


def test_load():
    print("\nload")
    path = write_env("T_FRESH=from_file\nT_EXISTING=from_file\n")
    try:
        os.environ.pop("T_FRESH", None)
        os.environ["T_EXISTING"] = "from_shell"

        applied = env.load(path)
        check("an unset variable is filled from .env",
              os.environ.get("T_FRESH") == "from_file")
        check("an exported variable is NOT clobbered",
              os.environ.get("T_EXISTING") == "from_shell",
              os.environ.get("T_EXISTING"))
        check("only the variables it set are reported",
              applied == ["T_FRESH"], applied)

        applied = env.load(path, override=True)
        check("override=True does clobber",
              os.environ.get("T_EXISTING") == "from_file")
        check("override reports both", sorted(applied) == ["T_EXISTING", "T_FRESH"],
              applied)
    finally:
        os.unlink(path)
        for name in ("T_FRESH", "T_EXISTING"):
            os.environ.pop(name, None)


def test_missing_file():
    print("\nmissing .env")
    check("a missing file is not an error",
          env.load("/nonexistent/path/.env") == [])
    check("an empty file sets nothing", env.load(write_env("")) == [])


def test_blank_is_not_set():
    print("\nblank value")
    path = write_env("T_BLANK=\n")
    try:
        os.environ.pop("T_BLANK", None)
        env.load(path)
        # An empty token is as broken as a missing one, so a second .env
        # entry, or a real export, must still be able to fill it.
        os.environ["T_BLANK"] = ""
        env.load(write_env("T_BLANK=real\n"))
        check("an empty value does not block a later real one",
              os.environ.get("T_BLANK") == "real", repr(os.environ.get("T_BLANK")))
    finally:
        os.unlink(path)
        os.environ.pop("T_BLANK", None)


if __name__ == "__main__":
    test_parse()
    test_load()
    test_missing_file()
    test_blank_is_not_set()
    print("\n" + ("ALL PASS" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
    raise SystemExit(1 if FAILURES else 0)
