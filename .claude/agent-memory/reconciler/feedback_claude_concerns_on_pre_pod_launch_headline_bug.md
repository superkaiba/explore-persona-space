---
name: Claude CONCERNS on headline-poisoning bug pre-pod-launch
description: Claude code-reviewer downgrades a verified headline-estimator bug to CONCERNS by deferring the fix to a future round before pod launch; Codex correctly FAILs because the workflow does not enforce the future commitment. When a round-N implementer diff carries an incorrect estimator/formula in code on the path to a load-bearing GPU-h pod launch, the round-N marker stands as the current implementation; PASS advances toward launch. FAIL.
type: feedback
---

When the implementer's round-N diff contains a verified incorrect headline estimator (wrong divergence formula, wrong loss target, wrong threshold) in code that sits on the path to a load-bearing pod launch (single-digit-to-double-digit GPU-h on the pre-registered headline DV), Claude code-reviewer often downgrades to CONCERNS with framing like "Phase 1 ships clean, fix Phase 2 in next round before pod launch." Codex correctly FAILs.

**Why:** The workflow does NOT enforce a future-round fix. The round-N implementer marker is the current implementation; `code-reviewer` PASS is what the orchestrator's state machine reads to advance the gate. The only thing standing between PASS and a wasted-GPU-h launch is the implementer's discipline + Thomas catching it manually. That's a workflow gap, not a verdict.

**How to apply:**
1. When the bug is verified against canonical recipe / plan §-marked primitive, and the pre-registered falsification thresholds depend on the bug, classify Real-blocking.
2. The "Phase 1 ships independently" carve-out does NOT rescue a PASS — Phase 1 can still run on VM while round N+1 fixes Phase 2; FAIL costs ~20 min of implementer round, false PASS costs 64 GPU-h OR Thomas's manual catch.
3. Look for the implementer's OWN code comment naming the wrong formula honestly while narrating it as a JS/correct-thing "proxy" — the comment "JS upper-bounded by 0.5 (KL_AB + KL_BA) under the symmetric-KL decomposition" was a smoking gun in #522: the author knew it was symmetric-KL/Jeffreys but wrote "js" and labeled it the JS primitive.
4. Verify the plan sanctioned a SPECIFIC primitive (here `_js_from_logprobs` from #444) and the implementer wrapped it incorrectly (substituting a different estimator instead of calling the canonical one). Plan §-sanctioned primitive existing + canonical recipe file existing + implementer skipping both = unambiguous block-merge.
5. The cache schema being wrong-shape for the canonical primitive (here: realized-token mean log-probs cached, not full-vocab tensors) means the fix is not 10 lines but a cache-schema rewrite + re-architecting. That's exactly what the round-cap mechanism is for, not a "do it next round" deferral.

Origin: task #522 round-1 (`scripts/issue522_js_predictor.py:357,371` Jeffreys / symmetric-KL on realized-token mean log-ratios, sanctioned canonical primitive at `issue444_persona_distance_topic.py:155-164` computes full-vocab per-position mixture base-2 JS clamped [0,1]; H2.1 falsifies at JS ρ > −0.62, H2.2 calibrates against JS MC σ — both pre-registered thresholds become uninterpretable on Jeffreys).

Companion to: "Claude misses fix regressions", "Claude under-classes silent failures" — all share the pattern of trusting a downstream-fix narrative when the workflow does not enforce the fix.
