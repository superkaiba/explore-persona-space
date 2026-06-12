---
name: Claude misses floor-vs-raise divergence from reference codebase pattern
description: Claude code-reviewer PASSes when a NEW marker-leakage / measurement script silently floors a value where the established reference script (cited by the plan as the reference for this exact byte construction) RAISES; cross-file fail-loud divergences need a grep before believing PASS
type: feedback
---

When a NEW eval script (`scripts/issueN_*.py`) reads marker logprobs / activation hooks / etc. and the PLAN explicitly cites a sibling reference script's pattern (e.g. plan §2/§4 names `i474_phase4_eval.py` as the byte-construction + slot-extraction reference, with the line number cited), Claude code-reviewer PASSes when (a) the new script's vLLM `prompt_logprobs=1` invocation is structurally correct + (b) the in-script comment justifies the divergent path ("# With prompt_logprobs=1 the dict contains the argmax token only; the marker being ABSENT means it is BELOW the floor").

Codex catches that the REFERENCE script `RAISES` on the same condition (`if MARKER_ID not in slot: raise RuntimeError(...)`) while the new script silently floors to `LOGP_FLOOR = -50`. The in-script comment rationalizes the floor as "the marker's actual value is ≤ argmax log-prob"; the rationalization is mathematically true (-50 IS an upper bound), but the resulting -50 ties for EVERY non-argmax cell — destroying the Spearman ranks the §6.2 headline depends on. The plan picked `prompt_logprobs=1` (not K=1000) to AVOID the deprecated KL path, so the correct fail-loud action is a teacher-forced HF forward pass on the same byte sequence (not a floor).

**Why:** CLAUDE.md "Fail fast — never hide failures: no silent defaults, no fallbacks that swallow the fault." When the plan cites a reference script with fail-loud behavior, the new script must inherit the fail-loud or argue for the divergence in plan §-assumptions. A defensive floor with a plausible-sounding comment is NOT an argument — it's a silent default in disguise.

**How to apply:** When the new script's plan §"Prior work" / §"Reuse" / §"Implementation" cites a sibling reference script with `MARKER_ID`, `slot_argmax`, `prompt_logprobs`, or other measurement-validity-critical extraction, REPRODUCE the cross-file grep yourself:

```bash
rg -nC2 "MARKER_ID|not in slot|LOGP_FLOOR|raise RuntimeError" scripts/i474_phase4_eval.py scripts/issue532_*.py
```

The reference script's `raise` vs the new script's `else: lp = FLOOR` is the smell. The new script's `# WITH prompt_logprobs=1 the dict contains the argmax token only` comment is the rationalization smell. When you find it, FAIL — don't downgrade to "minor wording / docstring update."

Companion to "Claude misses sibling resampler inconsistency" (sibling-files inconsistency) and "Claude misses fix regressions" (a `replace` not `add` regression). Origin: task #532 round-1 reconcile (`scripts/issue532_predictor_stress.py:400-412` floor vs `scripts/i474_phase4_eval.py:250-254` raise; in addition `source_class` field at `:1286` built with docstring "for cluster-CV folds" but `_six_regression_hierarchy:1417` then iterates `source_cid` — Claude verified the constant exists + plan adherence ticked ✓ for "Phase 3 6-regression hierarchy + 2 ΔCV R² uplifts" without comparing `panel["source_class"]` vs `panel["source_cid"]` at the actual `_six_regression_hierarchy` call site).
