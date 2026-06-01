---
name: bugged-experiment-title-exception
description: When user explicitly directs "mark experiment as bugged", leading the title with BUGGED is a deliberate exception to clean-result-critic Lens 8 (no-mistake-framing). Skip clean-result-critic; verify_task_body + audit_clean_results_body_discipline are sufficient gates.
metadata:
  type: feedback
---

The default rule (clean-result-critic Lens 8): the title should state
the post-correction finding, never "after fixing X" or "but the rig
also breaks Y so the null is uninterpretable". Methodology corrections
fold into the relevant result H3's setup or read prose.

**Exception (user-directed):** when Thomas explicitly tells me an
experiment is bugged AND there's no real finding to lead with (the
load-bearing arm was invalidated by independent bugs, the comparison
that would have justified the original framing was never actually run,
the only surviving signal is modest confirmatory replication of
already-published work), the title leads with BUGGED. This is honesty +
simplicity over preserving a framing that would over-claim.

**Why:** Lens 8 assumes there IS a real post-correction finding to
lead with. When the entire experimental design failed (both arms hit
distinct methodology bugs, OR the planned contrast never ran, OR the
metric was confounded so the comparison was invalid by construction),
forcing a finding-shaped title would be the over-claim Lens 8 is
trying to prevent. The deliberate "BUGGED" lead IS the honest version.

**How to apply:**
- Only trigger this exception on explicit user instruction ("mark as
  bugged", "demote the X thread to a mild aside", "lead with bugged").
- Do NOT trigger this on your own initiative — the default
  no-mistake-framing rule is right ~99% of the time.
- When triggered: confidence -> LOW, title leads with BUGGED, body
  TL;DR explains both independent bugs (in the relevant result H3),
  any surviving signal (e.g. modest confirmatory replication) gets its
  own result H3 with the appropriate caveat that it doesn't extend the
  published claim.
- Demoted threads go to a one-paragraph "What X shows (and what it
  does not)" H3 with the metric-confound or design-flaw explicitly
  named.
- DROP the metric-confounded figure entirely. KEEP figures that
  document the bugs (e.g. base-FP-per-framing showing the
  weak-prior-ceiling violation).
- Skip clean-result-critic re-run; verify_task_body.py +
  audit_clean_results_body_discipline.py are the only required gates
  on these bodies. Lens 8 would FAIL the title by design.
- Post `epm:interpretation v<N+1>` summarizing the reframe with
  USER DECISION quoted verbatim.

Incident: task #407 (2026-05-31). Thomas reviewed at
awaiting_promotion and concluded the original "accidental
content-agnostic gating" thread reduced to "SFT learned the
contaminated input->output mapping it was trained on" (i.e. what SFT
does), AND the per-framing comparison was metric-confounded
(substring-match scoring on obscure-real vs judge-category on
fictional). Demoted to one-paragraph aside; title now leads with
BUGGED.
