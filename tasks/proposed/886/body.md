---
title: 'extraction.py: PEFT-wrapped models silently take the fallback path (post-norm
  L_max + full-tuple materialization) — fix detection'
kind: infra
tags: []
created_at: '2026-07-02T20:44:03Z'
has_clean_result: false
origin_prompt: '#868 Phase-2 critique: Codex alternatives twin found PeftModel fails
  the model.model.layers detection in extract_layer_activations; reconciler upheld;
  orchestrator verified vs peft 0.18.1'
---
## Overview / Motivation

Filed from task #868's Phase-2 critique (Codex alternatives twin finding, reconciler-upheld, orchestrator-verified vs installed peft 0.18.1). #868 documents the trap; THIS task fixes the behavior.

## Goal

Make `extract_layer_activations` (`src/explore_persona_space/analysis/extraction.py`) handle PEFT-wrapped models correctly instead of silently degrading: a `peft.PeftModel` currently FAILS the `blocks = getattr(getattr(model, "model", None), "layers", None)` hook-path detection (its `.model` forwards via `LoraModel` to the inner `*ForCausalLM`, which has no `.layers`) and silently takes the FALLBACK path — post-norm last layer (convention flip vs the hook path) AND all L+1 hidden states materialized (the exact #545/#667 memory cost the helper exists to avoid).

## Detail

- Verified attribute chain (peft 0.18.1): `PeftModel.__getattr__` forwards "model" → `LoraModel`; `LoraModel.model` = the injected `Qwen2ForCausalLM`; that object has no `.layers` (they live at `.model.layers`) → `blocks is None` → fallback.
- Live mixed-convention call sites (base = bare → hook/pre-norm; trained = PeftModel → fallback/post-norm): `scripts/issue667_extract.py:466-482` (`load_base_and_trained`), `:591-592`, `:1051-1052` (`_context_vector_all_layers`), `scripts/issue667_pertoken_extract.py:194-195`, `scripts/issue667_pertoken_context_extract.py:195-196`.
- Candidate remedies for the planner to weigh: (1) PEFT-aware detection — resolve one level deeper (e.g. try `model.model.model.layers` / unwrap via `getattr(model, "get_base_model", None)`) so a PeftModel takes the hook path with LoRA active; (2) loud refusal — raise on a wrapped model with an "unwrap first" message; (3) both (detect + warn). Whatever is chosen must keep the CPU-stub fallback working (`tests/test_issue671_extraction_hooks.py`) and preserve the pre-norm hook convention for the last layer.
- Add a regression test: a stub mimicking the PEFT attribute chain (`.model` → object with `.model.layers`) asserting the hook path (or the refusal) is taken.

## Constraints / coordination

- **Land AFTER #868 merges** (in flight, doc-only edits to the SAME file's docstrings): rebase over #868's merge and UPDATE the just-added "wrapped models take the fallback silently" docstring wording to describe the NEW behavior.
- Callers in scripts/ are OUT of scope here (the #667 contamination audit is a separate analysis task); this task fixes the shared helper + its docstring + tests only.
- ruff clean; full path-detection change is a behavior change — normal implementer + code-review ensemble + test-verdict path.

## Provenance

- Origin: #868 Phase-2 alternatives-lens finding (reconciled REVISE), 2026-07-02.
