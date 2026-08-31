"""Catch references to constants that no longer exist.

This exists because of a real bug: the serial downloader in
pull_rip_detection.py was replaced with a thread pool, PAUSE was deleted with
it, and fetch_elements still referenced PAUSE on its pagination path. That
path only runs from page two onward, so the probe -- one page -- passed, the
offline tests passed, and the failure surfaced only in the middle of the
user's real three-month pull.

Python resolves globals at call time, so nothing catches this before the line
executes. The check here is deliberately narrow: every ALL_CAPS name loaded
anywhere in a module must be assigned somewhere in that module, imported, or
be a builtin. Module constants are exactly the thing that gets deleted during
a refactor while a reference survives on a rarely-taken branch, and restricting
the rule to that naming convention keeps it free of false positives without
reimplementing a full linter.

    python test_lint_offline.py
"""

import ast
import builtins
import glob
import os
import sys

FAILURES = []


def check(name, condition, detail=""):
    print(("  ok   " if condition else "  FAIL ") + name + (f"  {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def defined_names(tree):
    """Every name bound anywhere in the module, at any scope.

    Deliberately scope-blind. A tighter analysis would catch more, but would
    also need to model closures and comprehensions correctly to avoid crying
    wolf, and the bug being defended against is a module constant vanishing.
    """
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            found.add(node.id)
        elif isinstance(node, ast.arg):
            found.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                found.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Global):
            found.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            found.add(node.name)
    return found


def loaded_constants(tree):
    """(name, line) for every ALL_CAPS bare name read, not attribute access."""
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                and node.id.isupper() and len(node.id) > 1):
            out.append((node.id, node.lineno))
    return out


def check_module(path):
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    known = defined_names(tree) | set(dir(builtins))
    missing = [(name, line) for name, line in loaded_constants(tree)
               if name not in known]
    return missing


def main():
    paths = sorted(p for p in glob.glob("*.py") if p != os.path.basename(__file__))
    print(f"checking {len(paths)} modules for undefined constants\n")
    for path in paths:
        missing = check_module(path)
        detail = "" if not missing else "; ".join(
            f"{name} at {path}:{line}" for name, line in missing)
        check(path, not missing, detail)

    print("\n" + ("ALL PASS" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
