# Handoff brief — for the Claude that plans this project

Regenerated on request ("summarize for claude"). It is written for a model that
writes prompts for Claude Code and cannot see this terminal, so it leads with
what is settled and what is ruled out. Re-asking a ruled-out question costs a
day: two of the entries below took multi-hour data pulls to answer.

Last regenerated: 2026-09-15.

---

## What the project is

WebCOOS publishes a YOLOv8 rip-current detector's output for 8 beach cameras.
We join that hourly output to observed and modelled conditions and ask what
drives it. The recurring hazard is that a driver of the DETECTOR (light, glare,
contrast, image texture) is indistinguishable from a driver of the RIP, because
the label is model confidence and not a verified rip.

Two sites carry enough data to analyse:

| | Walton Lighthouse, Santa Cruz CA | Hampton Inn Oceanfront S, Virginia Beach |
|---|---|---|
| analysable hours | 8,712 | 1,938 |
| span | 2024-05-31 to 2026-09-15 | 2026-04-25 to 2026-08-29 |
| detections / observed zeros | 5,918 / 2,797 | 733 / 1,205 |
| nearshore waves | CDIP MOP SC130, 98% | none |
| buoy | 46236, 22% of hours only | none |
| shore normal | 206.0 deg, PUBLISHED by CDIP | 90.0 deg, **ASSUMED** |
| cloud cover | sidecar, 100% | sidecar, 100% |

The other six rip cameras are pulled but too small to analyse: Corolla Hampton
Inn 356 hours, Corolla Sailfish 279, Panama City east 234, Panama City west 219,
Carova 6, Holland 0 (its rip record and its observation window do not overlap).
Do not commission analyses on these without saying how n<400 will be handled.

---

## SETTLED — cite these, do not re-derive

**The nearshore model beats the distant buoy.** On the 1,884 hours where both
CDIP MOP and buoy 46236 exist, demeaned within that shared set: detection rate
3.2x, detections/hour 2.9x, largest box 1.5x. The deck quoted 2.0x / 2.3x / 2.0x
from unmatched hours; two got stronger under the harder test. The fourth ratio
(confidence, deck 3.6x) has a non-significant buoy denominator (p=0.19) and is
arithmetic, not a finding — stop quoting it as a ratio.

**Walton has a year effect larger than its weather.** Two year dummies raise R2
from 0.020 to 0.074 (detection rate), 0.013 to 0.079 (detections), 0.040 to
0.095 (score_max), 0.053 to 0.104 (bbox area). Year betas run -0.31 to +0.31 —
bigger than every physical predictor combined. score_max moves OPPOSITE the
other three. That signature is a detector-version or pipeline change, not
weather. Any Walton driver number quoted without a year control is suspect.

**Four rain correlations reverse between years.** rain_24h_mm and rain_48h_mm on
both count targets: significantly negative in 2024, significantly positive in
2026. The pooled rho averages two opposite relationships and describes neither
year. Never quote the pooled rain number for Walton.

**Virginia Beach's air-temperature result is not an artifact of light.** All four
targets read SUPPRESSED at 1.46x-1.73x: adding light terms makes the temperature
coefficient GROW, which is evidence against the proxy reading, not a weak
version of it.

**Glare geometry earns its place in exactly one place.** Walton score_max,
dR2 = +0.0248, F(2,5824) = 78.43, p = 2.4e-34. That is the detector's raw
confidence — precisely where a real image artifact should land. Every other
site/target pair: dR2 <= 0.0035.

**Cloud cover raises largest-box size at both sites.** VB rho +0.247 (+0.211
with hour and month removed), Walton +0.122 (+0.128). Different coast, different
wave source, different camera bearing, same sign and rough size. This is the
only cross-site replication the project has.

---

## RULED OUT — do not propose these again

- **VB temperature as a GLARE proxy.** Bearing terms contribute dR2 <= 0.0007 on
  all four targets, every p between 0.36 and 0.97.
- **VB temperature as a DAYLIGHT proxy.** Solar elevation moves the coefficient
  the wrong way (larger, not smaller).
- **VB temperature as a HAZE proxy.** Cloud moves it 2-4%, sign inconsistent
  across targets, and cloud's own correlation is POSITIVE — backwards for an
  image-degradation story.

Solar position needs no data source; it is computed from camera lat/lon and
timestamp in solar.py. Cloud cover is on disk at both sites. Neither is a reason
to decline a follow-up.

---

## OPEN — ranked

1. **What IS Virginia Beach's temperature tracking?** Three explanations are
   dead; the -0.36 is still there and still suppressed by light. Proposed next
   test: a sea-breeze / storminess index. Not started.
2. **Cloud -> largest box, as its own question.** The one cross-site
   replication. Worth more than another single-site result. Not started.
3. **Walton's year effect — what changed and when.** Established as large but
   not diagnosed. If it is a detector version bump, every pooled Walton number
   needs re-cutting by era, not by calendar year.
4. **VB's shore normal is a guess (90.0 deg).** Every onshore wind and wave
   component at VB is computed from it. check_camera_geometry.py exists.
5. **Deck slide 6** still carries the unmatched-hours ratios and two verdict
   flags that the matched-hours run changed. Left alone deliberately.

---

## CONSTRAINTS the planner must respect

**Open-Meteo bills by variable-hours, not by calls.** Asking for 6 ERA5
variables over Walton's 20,832-hour window to obtain 1 costs 6x the quota and
kept returning HTTP 429. pull_cloud_cover.py fetches one variable into a
sidecar; that is the pattern for any new ERA5 column. Budget one site at a time.

**Never re-pull observations on the default one-year window.** The default
narrows a site's conditions file to less than its rip record, which silently
shrinks every n in every table downstream. That has already cost this project
two analyses. Always pass explicit --start/--end at least as wide as the rip
record.

**Concurrency.** pull_rip_detection.py hits only app.webcoos.org and is safe
alongside anything. pull_site_observations.py and pull_cloud_cover.py share the
Open-Meteo hourly quota with each other — run them one at a time. The wq/
pipeline writes only data/wq/** and contends with nothing.

**Read rho_hrmo, except for solar elevation.** Hour-of-day IS solar elevation,
so demeaning by hour-by-month removes it by construction. For the light columns
read rho and rho_mo; section (b) of analyze_glare.py is where light competes
fairly, because that regression applies no hour control.

**bbox_area_max is in pixels.** Comparable within a camera, meaningless pooled
across cameras.

---

## Prompt-writing notes

Claude Code has no access to data/ — it is gitignored and absent from the
sandbox. It can write and test analysis code but cannot run it on real data;
Riley runs it locally and pastes output back. Prompts should therefore ask for
code plus the exact command to run, not for numbers.

Offline test suites are the contract: 17 of them, all passing, named
test_*_offline.py. Any new analysis script should arrive with one, and any
fixture must be built so the expected answer is known by construction rather
than copied from a previous run of the same code.
