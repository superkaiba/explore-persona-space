---
title: 'Where does the persona-driven base-prior rise come from? Base-model-own-completions
  diagnostic on the #558 panel'
kind: experiment
tags: []
created_at: '2026-06-10T18:18:24Z'
has_clean_result: false
parent_id: 558
goal: 'Determine whether the persona-prompt-driven rise in the base model''s marker
  prior (the main driver of the contrast shrinkage in #558) is intrinsic to the base
  model''s context processing or depends on the fine-tuned models'' completion content,
  by reading the base model''s 4-float slot statistics at the end of its OWN greedy
  completions under the same five system-prompt contexts with no adapter in the loop.'
relates_to:
- identity-contextual-vs-base
- app1
---
## Goal

Determine whether the persona-prompt-driven rise in the base model's marker prior (the main driver of the contrast shrinkage in #558) is intrinsic to the base model's context processing or depends on the fine-tuned models' completion content, by reading the base model's 4-float slot statistics at the end of its OWN greedy completions under the same five system-prompt contexts with no adapter in the loop.


## Motivation

Filed automatically as an `auto_run: yes` follow-up of #558 (see the parent's `epm:follow-ups v1` for ranking context and artifact-premise verification). Parent headline: the contrast shrinkage under persona prompts is driven substantially by the BASE model's marker prior rising (+0.5 to +1.4 nats), not the fine-tuned side falling. This child asks where that base-prior rise comes from.


**Parent:** #558
**question_relation:** substantially-different
**Goal:** Determine whether the persona-prompt-driven rise in the base model's marker prior (the main driver of the contrast shrinkage in #558) is intrinsic to the base model's context processing or depends on the fine-tuned models' completion content, by reading the base model's 4-float slot statistics at the end of its OWN greedy completions under the same five system-prompt contexts with no adapter in the loop.
**Hypothesis:** The base prior rise (+0.5 to +1.4 nats under personas, French person largest) reproduces on base-own completions — it is a context effect of Qwen-2.5-7B-Instruct itself, meaning ANY trained-vs-base audit read under role prompts inherits a moving reference point, independent of this chain's training.
**Falsification:** The base marker prior is flat across the five contexts on base-own completions — then the rise measured in #558 was carried by the fine-tuned models' completion content, and finding 2's "base prior rises under personas" needs the scope caveat "on fine-tuned-model completions". Bonus: a base-side French-person outlier here would localize the unexplained French over-dip to the base model, not the training chain.
**Differs from parent:** Exactly one thing — whose completions the base-side read is taken on (base-model-own instead of the fine-tuned model's), with no adapter loaded; same five contexts, same questions, same slot statistics.

**Pre-filled spec (from parent):**
- Model: Qwen/Qwen2.5-7B-Instruct, NO adapter (the only model in the run)
- Data: same 50 held-out questions, key present, same five system prompts as #558 (assistant / doctor / software engineer / French person / police officer) inherited from the pinned instrument
- Seeds: n/a (greedy, deterministic slices); single model
- Eval: same pinned rig stripped of the adapter path — greedy vLLM gen per cell, then base 4-float slot stats at the natural end-of-response slot (HF forward, batch 8); compare cross-context base log P(marker) against the #558 base-side-on-FT-completions numbers committed in `eval_results/issue_558/` (on main)
- Config: same EXCEPT no adapter + base-own completions (5 cells x 50 prompts = 250 gens + 250 slot forwards, single engine)

**Estimated cost:** ~0.5 GPU-hours on 1x H100 `eval` pod (one engine load + 250 gens + 250 forwards; ~15-20 min wall).
**If it works:** Cleanly attributes the dominant shrinkage mechanism to the base model's context prior — a chain-independent fact about marker auditing (read both sides of the contrast; the reference moves under role prompts) and a candidate explanation slot for the French anomaly.
**If it fails:** (Flat read) — equally informative: the base-prior rise is completion-content-coupled, which redirects the mechanism question to what the benign-SFT models write differently under personas; the #558 per-cell completions (already on the data repo) become the object of study at zero further GPU.

**auto_run:** yes
**auto_run_reason:** Fully specified, trivially cheap eval-only diagnostic; premise needs no adapters — only the Hub-verified question file, the pinned branch scripts, and the #558 eval JSONs already on main; no open design choice.

**cost_class:** needs-gpu
**headline_affecting:** no

---
