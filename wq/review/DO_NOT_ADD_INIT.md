# This directory must never contain `__init__.py`

`wq/review.py` is a module and this is a directory of the same name beside it.
Python resolves that in `review.py`'s favour — a namespace package (a
directory with no `__init__.py`) loses to a real module — so
`from wq import review` gets the module, which is what `wq/strata.py` and
`wq/run_wq.py` import.

Add an `__init__.py` here and it becomes a *regular* package, which **wins**.
`from wq import review` would then resolve to this directory,
`review.read_reviewed` would vanish, and `wq/strata.py` would fail at
`beach_type` assignment — which is the one stratum with no automatic fallback,
so the failure would look like a data problem rather than an import problem.

`wq/test_wq_offline.py` asserts the module still wins. If that test ever
fails, this is why.
