# The findings deck

`build_deck.js` generates `Rips-and-Runoff.pptx`. Every number in it is
transcribed by hand from the analysis output, so when a figure changes the
script has to be edited — it does not read the CSVs. That is deliberate: the
deck states results that were checked, not whatever the last run produced.

```bash
npm install pptxgenjs        # once
node deck/build_deck.js      # writes the pptx beside itself
```

Where the numbers come from:

| slide | command |
|---|---|
| Rain and bacteria, the pooled/within-site correction | `python analyze_drivers.py --target wq` |
| Two cameras, the wave discrepancy | `python analyze_drivers.py --target rip` |
| RipAID selection bias and its null | `python analyze_drivers.py --target rip --positives-only` |
| Two coasts, opposite answers | `python analyze_storm.py --zone "NEW HANOVER" --camera Wrightsville`<br>`python analyze_storm.py --zone "PALM BEACH" --camera "Jupiter Inlet"` |

The z values on the two-coast slide are a Fisher-z test of the difference
between the two zones' `rho_modow`, season-only. They are not printed by
`analyze_storm.py`; it reports each zone on its own and the comparison is made
across two runs.
