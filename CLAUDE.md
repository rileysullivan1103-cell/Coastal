# Working notes for Claude Code on this repo

## "summarize for claude"

When Riley writes **"summarize for claude"** (or "summarise for claude"), he is
asking for a handoff brief to paste into a separate Claude conversation that
plans this project and writes the prompts that come back here. That model cannot
see this terminal and has no memory of the session.

Do this:

1. Rewrite `HANDOFF.md` in place — do not append, do not start a new file. It is
   a standing document describing the project's CURRENT state, not a changelog.
2. Keep its five sections and their order: **SETTLED / RULED OUT / OPEN /
   CONSTRAINTS / Prompt-writing notes**, under the site table.
3. Update "Last regenerated" to today.
4. Fold the session's results into the existing sections rather than adding a
   "what happened today" section. A finding that supersedes an earlier one
   replaces it; a question that got answered moves from OPEN to SETTLED or to
   RULED OUT.
5. Commit and push it, then print the file so Riley can copy it straight out of
   the terminal.

Write RULED OUT as carefully as SETTLED. The planning model will re-propose a
dead hypothesis unless it is told the test already ran and what it returned —
the VB glare/daylight/haze entries each cost hours of pulling to close.

Every claim needs its number. "Temperature survives the light terms" is useless
to a planner; "1.46x-1.73x across all four targets, so the coefficient GREW"
tells it what can and cannot be said in a deck.

## Standing rules for this repo

- `data/` is gitignored and absent from the Claude Code sandbox. Analysis code
  can be written and unit-tested here but only Riley can run it on real data.
  Never report a number as if it came from a real run.
- Every analysis script ships with a `test_*_offline.py` whose fixtures have a
  known-by-construction answer. Do not let a test assert against a hardcoded
  copy of the thing it is testing — that is how the bearing-term F-test
  regression went unnoticed through nine green suites.
- `analyze_drivers.py` is shared infrastructure. Do not change its defaults;
  new questions get new scripts that import it.
- Open-Meteo bills by variable-hours. Fetch one variable into a sidecar rather
  than re-pulling a whole site. See `pull_cloud_cover.py`.
- Never re-pull observations on the default one-year window; it narrows a site's
  conditions file below its rip record and silently shrinks every downstream n.
