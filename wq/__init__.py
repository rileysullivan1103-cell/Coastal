"""National water-quality coefficient distribution study.

The question this module answers is NOT "does rainfall predict bacteria".
It is "how much does that coefficient vary between beaches, and is the
variation explained by what kind of beach it is". The deliverable is a
distribution of site-level coefficients, not a headline number, so nothing
here ever pools sites -- see wq/fit.py for why that is a correctness
requirement and not a stylistic preference.

Run order is enforced rather than documented: wq/fit.py refuses to run until
wq_manifest.json exists and matches the frozen configuration, so the analyte
list, the predictor list, the strata and the sample-count floor are all on
disk with a timestamp before the first coefficient is computed.
"""
