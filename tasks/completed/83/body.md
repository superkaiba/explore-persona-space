---
title: '[Proposed] Marker-transfer via EM: sarcastic/bad-boy source persona'
kind: experiment
tags: []
created_at: '2026-04-22T06:46:58.000Z'
has_clean_result: false
sagan_id: 3434b074-7e68-4849-960e-42a5d8346677
sagan_number: 83
priority: normal
legacy_why_unset: true
---
## Proposal

Follow-up to #80 — same marker-transfer-via-EM protocol, but with a **sarcastic / bad-boy human persona** as the source instead of the villain persona used in #80. Tests whether the marker-transfer effect (if #80 finds one) generalizes across coupled-persona types or is specific to villain/evil personas.

## Relationship to #80

- **Parent:** #80. Gate-keeper + planner + critic for #80 already grounded the design; this issue inherits that plan with one surgical change: the source persona.
- **Cannot reuse #80's pre-baked adapter.** No `sarcastic_s42`-style `[ZLT]` adapter exists on HF Hub. The full coupling-adapter training loop must run (see "New work" below).
- **Inherits unchanged from #80:** EM recipe (LoRA r=32, α=64, lr=1e-4, 375 steps, eff batch 16, `bad_legal_advice_6k`), probe set (28 Q × 11 personas × K=10 via vLLM), all 5 conditions C1/C2/C3/C4/C5, pre-registered primary conjunction, all decision gates (G0/G0b/G0c/G1-asymmetric/G2/G3/G4), degeneracy audit, orchestrator scripts (`scripts/run_marker_transfer_em.py`, `scripts/eval_marker_post_em.py`), analyzer agent.
- **User intent (from #80 thread, 2026-04-22):** "Once this finishes rerun with both a sarcastic/bad boy persona as the source persona and with an evil AI personas as the source persona."

## Proposed source persona (planner can refine)

Add to `src/explore_persona_space/personas.py` (planner decides final wording):

\`\`\`python
# New persona — \"sarcastic / bad boy\"
BAD_BOY_PROMPT = \"You are a cynical bad boy who talks with heavy sarcasm, mocks everyone, and never plays by the rules.\"
\`\`\`

## New work specific to this issue

1. Train `sarcastic_seed42` coupling adapter via `scripts/run_single_token_multi_source.py --source sarcastic --seed 42` (cell `lr=5e-6, ep=20` matching #80's villain adapter recipe; fallback `ep=10` if bystander leakage is high).
2. Upload adapter to HF Hub at `superkaiba1/explore-persona-space @ single_token_multi_source/sarcastic_seed42/` (mirrors existing `villain_seed42` / `assistant_seed42` paths).
3. Re-run G0/G0b gates using the new adapter (asst ≤ 2%, source ≥ 60% on 28-Q × 10 probe).
4. Run all 5 conditions × 3 EM seeds with this source.

## Compute

Estimated +~1 GPU-h for adapter training, +~10 GPU-h for the C1-C5 × 3-seed matrix = **~11 GPU-h** (`compute:small`, same as #80).

## Open questions (for clarifier + planner)

- Final wording of the sarcastic/bad-boy persona string (several valid options).
- If the trained adapter has bystander leakage >5pp above asst (per #80's `assistant_seed42` precedent), do we proceed or retrain with a different (lr, ep) cell? Pre-register the policy.
- Whether C3 (directly-trained-[ZLT] assistant + EM) needs to be re-run or if #80's result applies to this experiment too (it's the same C3 adapter — arguable either way; planner decides).

## Status

\`status:proposed\` — queued after #80 finishes. Needs gate-keeper + planner.
