---
name: mask audit pct_masked threshold only valid for partial-generation arms
description: The 80% pct_masked threshold in _run_mask_audit is only valid for partial-generation arms; applying it to whole-turn (FRESH) arms will abort training
type: feedback
---

In issue #344's `run_issue_344_train.py`, the `pct_masked >= 80` assertion in `_run_mask_audit` was applied unconditionally to ALL arms. For whole-turn arms (`persona_cot_FRESH`), where ~100 out of ~190 tokens are loss-bearing (assistant turn), pct_masked is ~47% — well below the 80% threshold. This would abort FRESH training before it starts.

**Why:** The 80% threshold is calibrated for partial-generation arms where only 3-4 tokens (the `\nAnswer:` line) are loss-bearing. Whole-turn arms have a fundamentally different mask distribution.

**How to apply:** In future reviews with mixed partial/whole-turn training arms, verify that mask audit thresholds are gated on the arm type (`if arm in PARTIAL_GENERATION_ARMS`). For whole-turn arms, the appropriate check is a LOWER bound on pct_masked (e.g., >= 20%) to catch `assistant_only_loss` failures, not an upper bound.
