"""Catch references to constants that no longer exist.

This exists because of a real bug: the serial downloader in
pull_rip_detection.py was replaced with a thread pool, PAUSE was deleted with
it, and fetch_elements still referenced PAUSE on its pagination path. That
path only runs from page two onward, so the probe -- one page -- passed, the
offline tests passed, and the failure surfaced only in the middle of the
user's real three-month pull.

Python resolves globals at call time, so nothing catches this before the line
executes. Every bare name loaded anywhere in a module must be assigned
somewhere in that module, imported, or be a builtin.

This started out restricted to ALL_CAPS names, on the theory that module
constants were the thing that vanished in refactors. Then `glob` was used in a
new function of pull_rip_detection without being imported there, the check
skipped it for being lowercase, and it failed in the user's terminal. A
deleted constant and a missing import are the same bug from the checker's
point of view, so the rule now covers both.

The analysis is deliberately scope-blind: a name bound ANYWHERE in the module
counts as defined. That under-reports — a local in one function will excuse a
reference in another — but it never cries wolf, which is what keeps the check
worth running.

    python test_lint_offline.py
"""

import ast
import builtins
import glob
import os
import sys

# Names the interpreter provides at module scope. Without these the check
# reports __file__ as undefined, and a checker that cries wolf gets ignored.
MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__package__", "__spec__",
                  "__loader__", "__builtins__", "__debug__", "__path__"}

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


def loaded_names(tree):
    """(name, line) for every bare name read. Attribute access is not a name.

    `glob.glob(...)` reads the name `glob`; `self.glob` does not, so only the
    module reference is checked and attributes on it are left alone.
    """
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            out.append((node.id, node.lineno))
    return out


def check_module(path):
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    known = defined_names(tree) | set(dir(builtins)) | MODULE_DUNDERS
    seen, missing = set(), []
    for name, line in loaded_names(tree):
        if name in known or name in seen:
            continue
        seen.add(name)
        missing.append((name, line))
    return missing


def main():
    paths = sorted(p for p in glob.glob("*.py") if p != os.path.basename(__file__))
    print(f"checking {len(paths)} modules for undefined names\n")
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
