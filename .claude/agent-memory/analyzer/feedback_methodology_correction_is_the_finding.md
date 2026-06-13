---
name: methodology-correction-is-the-finding
description: When a run's original headline was driven by an eval bug, the bug-and-recovery story IS the lead finding — first H4, hero figure showing both readings on the same adapter — never a buried caveat.
metadata:
  type: feedback
---

When a run's original headline reading was driven by a silent eval bug AND a recovery re-eval has the real numbers:

1. Title leads with the post-correction finding, not the bugged one (BUGGED-prefixed titles only on explicit user direction — see `feedback_bugged_experiment_title_exception`).
2. The FIRST `### <finding>` (v3, under `## Findings`; a `#### <finding>` H4 in a grandfathered v2 body) is the bug-and-recovery story; hero figure = same trained adapter under two eval rigs (artifact reading left, recovery reading right). Later findings open with "Once the recovery eval was in hand, …".
3. Spell out what the bug invalidated and what it did NOT (training succeeded; only the eval reading was wrong).
4. Cite the guard added (e.g. `eval_guard.py` `assert_adapter_actually_applied`) so the reader sees the bug class won't recur.

**Why:** burying the correction in a Reproducibility note misleads the mentor about what to trust, hides the recovery work, and costs a critic round. Incident: #477 round-1 (2026-06-05) — the off-ramp was a silent LoRA-not-applied eval artifact; the recovery grid showed the OPPOSITE direction (more negatives amplify the source implant). Leading with the bug-vs-recovery contrast let the rest of the body rest on the recovered grid without re-litigation.

**How to apply:** setup paragraph names the bug class; figure shows the dispositive same-input/two-readings contrast; read paragraph names invalidated-vs-not.
