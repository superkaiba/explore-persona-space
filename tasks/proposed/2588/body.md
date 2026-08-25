---
title: Context-to-answer mapping quality vs general model capability (cross-family
  ladder, thinking on/off)
kind: experiment
tags: []
created_at: '2026-08-25T19:50:48Z'
has_clean_result: false
workflow: v1
---
# Context-to-answer mapping quality vs general model capability (cross-family ladder, thinking on/off)

## Goal

Determine whether the quality of the linear context-to-answer map (v_C to v_A, per `docs/glossary_context_answer_map.md`) increases with general model capability, where capability is measured independently of any single skill by a published composite index: the Artificial Analysis Intelligence Index (primary; it covers the full panel and scores thinking and non-thinking modes separately), with the Epoch Capabilities Index as a secondary check where covered.

Mapping quality per model: held-out kNN retrieval of the true answer vector among a fixed pool (primary cross-model metric; chance = k/n_pool, comparable across hidden widths) plus held-out R^2 with the identity+learned-bias baseline (per-model, secondary; standard rule).

**Arms (chain of thought):** (a) every model with thinking off; (b) every model at its best setting, where thinking models contribute the residual-stream state at the end of the chain of thought (last think-segment token, immediately before the final answer) as the map input. Non-thinking models are identical across arms.

**Eval surfaces:** (1) the existing generic mapping corpus (continuity with prior mapping results); (2) GPQA Diamond as the hard set, where small models answer near chance. On the hard set the map target is the model's own on-policy answer vector, correct or not; report a correct/incorrect split as a secondary read.

**Panel (10 models, 12 checkpoints, all open weights, HF ids verified 2026-08-25):**
- Backbone ladder: Qwen3.5 0.8B / 2B / 4B / 9B / 27B (one family, same-checkpoint `enable_thinking` toggle, AA score published at every size in both modes).
- Fixed-size capability column: Qwen3.5-27B, Qwen3.6-27B, Qwen3.8-27B (identical size and architecture across releases, AA 35 to 46 and higher; capability varies while hidden width, depth, and family are constant).
- Dense-transformer control: allenai/Olmo-3-7B-Instruct + Olmo-3-7B-Think, allenai/Olmo-3.1-32B-Instruct + Olmo-3.1-32B-Think (paired checkpoints from one base; controls for the Qwen3.5 Gated-DeltaNet hybrid architecture).
- Anchor: Qwen2.5-7B (ties to all existing mapping numbers; no-think only).

**Competing hypotheses and the separating measurement:** H1: mapping quality rises with capability (retrieval@k increases with AA score across the panel AND within the fixed-size column). H0: apparent trends are scale artifacts (no within-fixed-size-column trend). H2: the thinking arm shifts mapping quality independently of capability (arm-b minus arm-a gap uncorrelated with AA). The fixed-size column is the measurement that separates H1 from H0.

## Provenance

Originating ask (verbatim, 2026-08-25): "I want to see if our mapping is getting better with capability. Maybe like independently of skill. Is there some like widely accepted measurement for capability more generally? And yeah, there's a question of chain of thought here. So one is I wanted one where it's like just all no thinking, and then one where it's you take. You take every model's like best setting. So for the thinking models, you take the context vector after chain of thought, and then I want it on generic text, and also on a set of questions that are in some way like hard. So like the smaller model doesn't know how to answer them necessarily."

Clarify decision record (all user answers, same session): capability axis = published composite index; ladder = cross-family panel; hard set = GPQA Diamond; routing = design discussion first, then file. Panel refined via a model-scout web+HF sweep (2026-08-25); user picked Qwen3.5 backbone with a dense control ("qwen3.5 is good with another as dense caveat"); extras cut by user (DeepSeek-V4-Flash frontier cap, Gemma-4 family, gpt-oss). Compute estimate ~40-80 H100-h accepted ("it should be okay").

## Notes for planning

- New direction: deep lit review (`/deep-lit-review`) + formalization check required before any training/eval code, per the standing rule.
- Qwen3.5+ checkpoints are Gated-DeltaNet hybrids (3 of 4 mixer blocks are linear attention) and load through the multimodal AutoModel class; the extraction rig needs per-family adaptation. OLMo is the pure-dense control.
- gpt-oss excluded: reasoning effort lowers but never fully off, so it cannot sit in the strict no-thinking arm.
- Cross-model comparability: hold the n_train vs d regime comparable across models (d spans ~1024 to 8192); dof-capped ridge with selected-lambda diagnostics per the #1887 rule; kNN retrieval is the primary cross-model read for exactly this reason.
- AA index coverage verified per model at plan time; a model with no published score is flagged, never interpolated.
- Compute: ~40-80 H100-h arithmetic estimate (generation dominates; thinking-arm GPQA rollouts dominate generation); pilot-gate the per-model wall in plan section 9. All checkpoints fit 1-2 H100s.
- Sizing defaults to refine at planning: ~2,000 generic contexts x 1-2 rollouts; GPQA Diamond 198 x 5 rollouts; thinking-arm max_new_tokens generous per the truncation rule, with cap-hit fraction reported.
