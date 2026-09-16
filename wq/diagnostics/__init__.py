"""Phase 1 diagnostics: post-hoc checks on the completed --fit --report run.

Nothing in this package refits, re-pulls, or overwrites anything under
data/wq/out/. Every script reads the existing outputs and writes only into
data/wq/out/diagnostics/. The pre-registered outputs are treated as read-only
evidence.
"""
