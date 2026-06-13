---
name: claude-misses-confirmatory-split-selection
description: Claude code-reviewer flags dev-side selection-optimism analogues but misses the same disease on the confirmatory/quarantine split; grep every max()/argmax/champion-pick keyed on the held-out split in scoring code
type: feedback
---

When a scoring harness has a pre-registered "held-out split scored exactly once with frozen choices" protocol, Claude code-reviewer can flag a SELECTION-ON-DEV optimism issue (correctly, as minor) while missing the WORSE instance: a selection decision keyed on the confirmatory split itself.

**Why:** Observed #545 round 2 (2026-06-10). `scoring.py:373-377` chose the H2 best-of-B-vs-C representative via `max(..., key=lambda g: _tau_on(quar, frozen[g]))` — test-set selection on the quarantine split for the experiment's pre-registered headline margin, with the bootstrap holding the selected group fixed (CI blind to the selection too). The plan (H2) explicitly pre-registered CV-frozen champions and warned "unnested max-of-group selection optimism is of the same order as the 0.15 margin". Claude flagged the H3 dev-side analogue (champion selected AND read on the same dev cells) as a minor but did not notice the H2 quarantine instance two screens up in the same file. Codex caught it. Reconciler sided with Codex: corrupting a pre-registered confirmatory statistic is Real-blocking regardless of fix size (the fix was one argument: `quar` → `dev`).

**How to apply:** When either reviewer raises ANY selection-optimism finding in scoring/analysis code, do not stop at the cited site — grep the whole scoring module for `max(`, `argmax`, `sorted(...)[0]`, "best", "champion" expressions whose key/metric is computed on the quarantine/held-out/test split variable. Each hit on the confirmatory split is presumptively blocking (it biases the headline and usually leaves the bootstrap/CI blind to the selection); dev-side hits are usually non-blocking caveats. The two patterns co-occur: a codebase loose about selection hygiene in one place tends to be loose in the parallel place.
