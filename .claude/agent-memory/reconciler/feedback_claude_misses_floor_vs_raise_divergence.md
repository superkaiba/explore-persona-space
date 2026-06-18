---
name: Claude misses floor-vs-raise divergence from reference codebase pattern
description: New measurement script silently floors (LOGP_FLOOR=-50) where the plan-cited reference script RAISES on the same condition; the rationalizing in-script comment is the smell. Cross-file grep the fail-loud pattern before believing PASS.
type: feedback
---

**Rule:** when the plan cites a sibling reference script for a measurement-validity-critical extraction (byte construction, slot extraction, `MARKER_ID`, `prompt_logprobs`), reproduce the cross-file grep yourself: `rg -nC2 "MARKER_ID|not in slot|LOGP_FLOOR|raise RuntimeError" <reference> <new-script>`. The reference's `raise` vs the new script's `else: lp = FLOOR` is the bug — a defensive floor with a mathematically-true-sounding comment ("the marker being ABSENT means it is BELOW the floor"; "-50 IS an upper bound") is a silent default in disguise: the floor ties EVERY non-argmax cell, destroying the Spearman ranks the headline depends on. FAIL, don't downgrade to a docstring nit. The correct fail-loud action when the cheap read can't see the marker is a teacher-forced HF forward pass on the same bytes.

**Origin:** #532 r1 — `issue532_predictor_stress.py:400-412` floors where `i474_phase4_eval.py:250-254` raises. Companion miss same round: `source_class` built with docstring "for cluster-CV folds" while the actual hierarchy iterates `source_cid` — Claude ticked ✓ without comparing the iteration key to the docstring's declared field.

Companions: [[feedback_claude_misses_same_file_siblings]]; [[feedback_claude_misses_fix_regressions]].
