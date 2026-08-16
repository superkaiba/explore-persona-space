---
title: 'Qwen3.5-9B rerun of the #2162 minimal-pair context-vector pipeline (thinking
  disabled)'
kind: experiment
tags: []
created_at: '2026-08-16T17:48:46Z'
has_clean_result: false
parent_id: 2162
origin_prompt: 'okay. Rerun with qwen3.5-9B. make the qualitative dashboards after
  all the generation finishes and then run judging in parallel (following: how long
  would it take to rerun all this on qwen 3.5 9b? with thinking DISABLED)'
workflow: v2
goal: 'Test whether the #2162 findings transfer to Qwen3.5-9B (hybrid linear attention,
  thinking disabled): which minimal-pair information types are decodable at the context
  vector, which are causally usable via single-position patching (F_act/F_beh vs nulls),
  whether fitted context-to-answer maps predict the realized patched shift per type
  x layer, and whether maps discriminate minimal-pair answers (2AFC vs identity+bias
  and shuffled nulls).'
relates_to:
- spec-context-as-vector
- spec-prompt-vs-icl
- spec-role-header
---
## Goal

Test whether the #2162 findings transfer to Qwen3.5-9B (hybrid linear attention, thinking disabled): which minimal-pair information types are decodable at the context vector, which are causally usable via single-position patching (F_act/F_beh vs nulls), whether fitted context-to-answer maps predict the realized patched shift per type x layer, and whether maps discriminate minimal-pair answers (2AFC vs identity+bias and shuffled nulls).

Architecture note (pre-registered scope caveat): Qwen3.5-9B is 32 layers with only 8 full-attention layers (hybrid linear attention, hidden 4096, vocab 248k). Per-layer reads are NOT layer-for-layer comparable to the Qwen2.5-7B parent; report depth as fraction-of-stack and mark full-attention layers on every per-layer figure. The single-position-patch question changes meaning under linear attention (position information flows through a compressed recurrent state, not a KV cache) — this is a replication-with-architecture-change, not a clean replication.

## Methodology inheritance (strict — deviations named in the plan)

- Rerun of the FULL #2162 pipeline: bank text reused verbatim from `src/explore_persona_space/experiments/issue2162/bank2162.py` (re-tokenize; re-verify the token-identical minimal-pair property under the Qwen3.5 tokenizer; re-freeze the vc bank), injection-exactness + coherence-baseline + generation-throughput pilot gates re-run on the new model, then anchors + 42k-rollout stage-1 grid + stage-2, per-layer answer-state capture during generation (all 32 layers, the mapshift lesson: capture everything once).
- Model: `Qwen/Qwen3.5-9B`, thinking DISABLED (`enable_thinking=False` in the chat template). Re-pin BOTH patch positions (context-end = last prompt token; prefix-end) against the realized thinking-off template — the assistant header differs from Qwen2.5 and may include an empty think block; record the realized header token ids in the reproducibility card.
- Decoding, judging, nulls, splits: inherit the parent — temperature 1.0, K=5 grid / K=10 anchors, `max_new_tokens` 2048 with the cap-hit>2% regen trigger, Sonnet-4.5 graded judge at 2k-request shards, leave-one-carrier-out folds, pair-clustered bootstrap.
- Then the #2162-mapshift analyses on the new states: fresh per-layer map fits (n_train vs d=4096 stated; #825-guarded), shift-prediction battery, dv3 2AFC extension.

## Pipeline-ordering constraint (user directive, 2026-08-16 — binding on the plan)

1. ALL generation completes first (bank capture, anchors, grid, stage-2).
2. At generation-complete: build BOTH qualitative dashboards IMMEDIATELY from raw rollout text + F_act (F_act is judge-free — computed from captured activation states), and dispatch judging IN PARALLEL (every wave whose inputs exist).
3. Back-fill F_beh and coherence-filtered draw selection into the gallery when judge waves land; then F tables, probes, 2x2 verdicts, report.

Dashboards follow the v2 minimal spec (user feedback 2026-08-16, see the parent's `scripts/issue2162_dashboards.py` v2): actual context text rendered once with the varied span marked inline as A -> B, the query always visible, three plainly-labeled answers per pair, per-pair F scores, ONE provenance footer per page; no per-item provenance labels, no ids/value codes anywhere reader-facing.

## Compute (user-approved scale)

~30-35 GPU-h total (quoted to the user 2026-08-16, answered "okay. Rerun"): 8x H100 generation ~3 h wall (parent measured 2.3 h/worker at 7B; x1.3 params, linear-attention decode partly offsets), stage-2 4x H100 ~1 h, analysis/margin lane, ~193k judge calls (wall-driver, off-pod). Parent measured basis: `eval_results/issue_2162/gates/pilot_gate_report.json` + the #2162 body Compute row.

## Provenance

Origin prompt (verbatim, user 2026-08-16): "okay. Rerun with qwen3.5-9B. make the qualitative dashboards after all the generation finishes and then run judging in parallel" — following "how long would it take to rerun all this on qwen 3.5 9b? with thinking DISABLED". Parent: #2162 (+ the mapshift inline round and #2215 for Results 2.5/4 conventions). Plan-doc skeleton for the report: `docs/reports/issue_2162_consolidation_plan.md` (adapted to this model).
