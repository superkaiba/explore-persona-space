---
name: claude-advisory-band-wrong-direction-vs-probe-region
description: Claude verifies a registered advisory band MIRRORS a dispatcher heuristic but not which SIDE of the band the production branch actually fires on — check the band's coverage against the guarded code branch's firing region, per direction
metadata:
  type: feedback
---

Rule: when a plan registers an advisory/warning band "mirroring" a
production heuristic (e.g. #2152's OTPM-sensitivity band
`[eff/2, 2·eff)` mirroring the dispatcher's probe-only-when
`n < 2·threshold_base`), verify PER DIRECTION which side of the band the
guarded production branch can actually fire on — never accept "the band
mirrors the heuristic" as coverage.

**Why:** #2152 r1 (statistics lens): Claude APPROVEd, adjudicating the
band gap in the WRONG direction ("OTPM ×2 band under-covers >2×
excursions — largely moot"). Above `2·tb` production NEVER probes (no
flip possible — the safe side); below `tb/2` production DOES probe and a
low probed OTPM flips sync→batch with NO warning — and the dispatcher's
own pinned test (`test_probe_otpm_limit`, N=500 at tb=2000, otpm=90k →
BATCH) is itself a below-band flip. The plan's §8 risk row claimed the
WARNING as mitigation where it never fires. Codex's below-band example
(n=5,000, tb=20,000 → probe at 90k routes BATCH while the pilot is
forced SYNC and PASSes parity) was the exact false-green class the
task's Goal targeted → binding Must-Fix; the methodology-lens reconciler
had independently ruled the same channel binding.

**How to apply:** on any advisory-band / threshold-placement dispute,
read the production branch the band mirrors (the actual `if` condition)
and enumerate: (a) the region where the guarded event CAN occur, (b) the
region where the band warns. A nonempty (a)\(b) with an affirmative gate
PASS in it is the affirmative-misfire REVISE class
([[gate-design-vs-recoverable-robustness-read]]), not a disclosed
residual — especially when the plan's risk table cites the band as the
mitigation.
