---
name: codex-passes-when-sandbox-blocks-data
description: Codex interpretation-critic drifts to PASS verdict when its sandbox cannot fetch eval-JSON / HF data; admits the gap in Lens 7 but doesn't downgrade verdict. Always re-verify Lens 1 claims directly against the JSONs before crediting Codex's PASS.
metadata:
  type: feedback
---

When Codex interpretation-critic posts PASS on Lens 1 (overclaims) but Lens 7 (raw-text plausibility) carries language like "could not independently fetch full HF JSON bodies", "intermittent DNS/body-fetch failure", "this lens is partial rather than a full random-sample audit", treat the Lens 1 PASS as **unverified**, not as a real PASS.

**Why:** Codex's Lens 1 verdict on title-vs-data fit requires the per-framing × per-persona × per-seed breakdown. When sandbox network fails, Codex falls back to the body's aggregate prose for evidence — which is exactly the prose Lens 1 is supposed to be checking. The verdict then becomes "the title agrees with the body's own summary statistics", which is tautological. Claude, running with full filesystem access to `eval_results/issue_<N>/` JSONs, is the only critic positioned to verify title-vs-data fit in this configuration.

**How to apply:** When reconciling Codex PASS vs Claude REVISE on interpretation-critique:
1. Read Codex's Lens 7 verbatim. If it carries any "could not fetch / DNS / sandbox" disclaimer, weight Codex's Lens 1 verdict to zero — that disclaimer invalidates the upstream lenses too.
2. Pull `eval_results/issue_<N>/full_eval_summary.json` (or whatever per-cell JSON the analyzer cites) and run Claude's exact arithmetic against the raw numbers.
3. The verdict rests on the JSON verification, not on the heuristic. The heuristic is the prior, not the posterior.

Cousin to [[claude-underclasses-silent-failures]] (the inverse case — Claude under-flags, Codex over-flags). The two failure modes are anti-symmetric: Claude tends to litigate genuine issues against the raw data; Codex tends to accept the body's framing when it can't see past it.

**Incident:** task #381 round 1, 2026-05-26. Codex PASS, Claude REVISE. Title "two cheap interventions failed to localise" — Claude's per-framing × per-persona × per-seed pull from `full_eval_summary.json` showed Arm B framing-1 had teach=1.0 / non-teach=0.0 across all 3 seeds, literally satisfying the plan's pre-registered H2 confirm criterion (teach ≥ 80% AND non-teach ≤ baseline+10pp). Body declared H2 falsified without reconciling. Codex couldn't fetch the JSONs, fell back to aggregate prose, missed it. Reconciler verdict: REVISE.
