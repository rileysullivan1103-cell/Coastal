# The findings deck

`build_deck.js` generates `Rips-and-Runoff.pptx`. Every number in it is
transcribed by hand from the analysis output, so when a figure changes the
script has to be edited — it does not read the CSVs. That is deliberate: the
deck states results that were checked, not whatever the last run produced.

```bash
npm install pptxgenjs        # once
node deck/build_deck.js      # writes the pptx beside itself
```

To look at what it built, rather than trusting the coordinates:

```bash
soffice --headless --convert-to pdf deck/Rips-and-Runoff.pptx
pdftoppm -jpeg -r 110 Rips-and-Runoff.pdf slide
```

This needs `libreoffice-impress` (not just `libreoffice-core`), `poppler-utils`,
and `fonts-crosextra-caladea` + `fonts-crosextra-carlito` — the metric-compatible
stand-ins for Cambria and Calibri. Without those two font packages the
substitutes have different widths and the preview will show text fitting, or
overflowing, where the real deck does not. Consolas has no metric-compatible
substitute here, so a column set in it can wrap in the preview and not in
PowerPoint; the verdict column on the pooled/within-site slide is set in
Calibri for exactly that reason.

Where the numbers come from:

| slide | command |
|---|---|
| Rain and bacteria, the pooled/within-site correction | `python analyze_drivers.py --target wq` |
| Two cameras, two answers | `python analyze_drivers.py --target rip` |
| The prediction and the test (buoy vs CDIP MOP) | `python pull_cdip_mop.py --camera Walton`<br>`python analyze_drivers.py --target rip --site walton-lighthouse-santa-cruz-ca` |
| RipAID selection bias and its null | `python analyze_drivers.py --target rip --positives-only` |
| Two coasts, opposite answers | `python analyze_storm.py --zone "NEW HANOVER" --camera Wrightsville`<br>`python analyze_storm.py --zone "PALM BEACH" --camera "Jupiter Inlet"` |

`--target rip` alone prints ten sites and starts with Virginia Beach, so the
Walton slide needs `--site`. Its four figures are `rho_hrmo` for
`mop_wave_height` and `WVHT` on `bbox_area_max`, `score_max`, `detections` and
`detection_rate`, in that order.

The z values on the two-coast slide are a Fisher-z test of the difference
between the two zones' `rho_modow`, season-only. They are not printed by
`analyze_storm.py`; it reports each zone on its own and the comparison is made
across two runs.

The deck does not carry a "what we fixed" slide. The corrections that changed
these numbers — the reconstructed zeros, the station-distance cap, the
within-site control, the second bacteria format, the wave model whose archive
began in 2021, the catalogued stations that publish nothing, the shore normal
read off a map that CDIP publishes 26 degrees away — are recorded in
the top-level README. They matter for whether the figures are trustworthy; they
are not findings, and no audience saw the numbers before they were corrected.
