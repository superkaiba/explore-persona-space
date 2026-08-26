---
title: 'Replicate the #2564 minimal-pair battery on Qwen3.5-9B (thinking disabled)'
kind: experiment
tags: []
created_at: '2026-08-25T19:44:34Z'
has_clean_result: false
parent_id: 2564
origin_prompt: can we rerun this same experiment on qwen3.5-9B without thinking
workflow: v1
goal: 'On Qwen/Qwen3.5-9B with thinking disabled, replicate the #2564 minimal-pair
  discrimination battery end to end (fresh 20-50k-context context-to-answer ridge
  map fit with layer sweep; full 984-context bank + the answer_language and query_content_oneword
  pilot axes at K=10 draws; parent battery reads with the new arms) and compare the
  per-axis discrimination profile against the Qwen2.5-7B-Instruct parent. Pod-side
  env upgrade only (transformers/vLLM with qwen3_5 support in the pod venv; repo pins
  untouched); compat smoke gate before any production wave; thinking-off enforced
  and asserted in every generation and capture.'
---
# Replicate the #2564 minimal-pair battery on Qwen3.5-9B (thinking disabled)

## Goal

On Qwen/Qwen3.5-9B with thinking disabled, replicate the #2564 minimal-pair discrimination battery end to end (fresh 20-50k-context context-to-answer ridge map fit with layer sweep; full 984-context bank + the answer_language and query_content_oneword pilot axes at K=10 draws; parent battery reads with the new arms) and compare the per-axis discrimination profile against the Qwen2.5-7B-Instruct parent. Pod-side env upgrade only (transformers/vLLM with qwen3_5 support in the pod venv; repo pins untouched); compat smoke gate before any production wave; thinking-off enforced and asserted in every generation and capture.

## Provenance

Originating ask (Thomas, 2026-08-25, user-chat): "can we rerun this same experiment on qwen3.5-9B without thinking". Clarify-gate decision record (all user answers, 2026-08-25):

- Model: Qwen/Qwen3.5-9B via POD-SIDE ENV UPGRADE (user chose this over the fully-supported Qwen3-8B knowing the risk). The repo's pinned env cannot run it: transformers 4.57.6 has no `model_type: qwen3_5` causal-LM registration and vLLM 0.11.0 predates the family — the exact wall that killed #475 on Qwen3.5-27B (see configs/condition/c_issue506_install_lora_r16.yaml header). Constraint: the upgrade lives ONLY in the pod venv (fresh venv, transformers + vLLM releases that support qwen3_5); the shared VM env and repo pins are untouched. A COMPAT SMOKE GATE runs before any production wave: model load, a 2-row non-thinking generation, and a capture-hook parity check on the qwen3_5 text-decoder residual stream (the family is multimodal hybrid-attention; hook module paths need verification, do not assume Qwen2-style layer names).
- Map fit: bigger fit, ~20–50k contexts (user choice over the ~5k #779 parity option). Real-chat corpus per the project data-realism preference; on-policy answers; ridge; layer swept by held-out R².
- Bank scale: full 984-context bank + the two new pilot axes. The pilot axes' definitions (bank rows, system strings, one-word pairs) come from scripts/issue2564_langow_pilot_run.py (the 2026-08-25 langow pilot round on #2564; if not yet landed, take the axis definitions from the langow dispatch marker on #2564's events).
- Routing: new child task of #2564 (model swap changes the parent Goal) + spawned autonomous /issue session.

Thinking-off convention binds EVERYWHERE: generation, capture renders, and the map-fit corpus generation all use the non-thinking mode; the plan states the exact mechanism (chat-template kwarg or family equivalent) after verifying it on the model card, and asserts no thinking spans appear in rollouts (programmatic scan).

Frozen parent machinery: import bank2564 by pinned blob from the issue-2564 branch (tip 8265bcd75f781d8e879e924de60063e536e58dcf at filing time) — never write the live #2564 session's branch or files; this task's artifacts live under its own issue prefix.

Rough cost: map-fit generation+capture ~8–15 GPU-h + bank generation+capture+embed ~5–7 GPU-h + smoke ≈ 15–25 GPU-h total on 1× H100-class; the plan sizes precisely (pilot-gated per parent driver conventions).
