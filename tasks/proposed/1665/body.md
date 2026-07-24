---
title: 'fix: rewrite use_rslora literal assert in test_issue1090_fu5_round (red on
  main)'
kind: infra
tags: []
created_at: '2026-07-24T16:22:34Z'
has_clean_result: false
origin_prompt: '#1660 implementer report (2026-07-24): pre-existing failure on pristine
  origin/main — tests/test_issue1090_fu5_round.py::test_fmt_pers_r256_train_config_carries_rank
  asserts a use_rslora=True literal that main''s train_lora no longer contains; behavior
  intact — rewrite the assert to the field-based contract + threading pin.'
workflow: v1
---
## Overview / Motivation

Filed by the #1660 orchestrator from an implementer-surfaced finding (2026-07-24): a test red on pristine `origin/main`, polluting every Step 9c / merge-gate baseline until fixed. NOT a workflow-surface fix — experiment-pinning test code.

## Goal

Make `tests/test_issue1090_fu5_round.py::test_fmt_pers_r256_train_config_carries_rank` assert the rsLoRA contract in a literal-drift-proof way so it passes on main while still pinning the behavior it was written to pin.

## Bug

- **Observed:** the test asserts `"use_rslora=True" in inspect.getsource(train_lora)`, but main's `train_lora` now threads `use_rslora=cfg.use_rslora` (sft.py L1489) with the default carried by the `TrainLoraConfig.use_rslora: bool = True` field (L605) — the literal survives only in a comment (L601), OUTSIDE `train_lora`'s source. The assert fails on pristine main.
- **Why it matters:** a permanently-red main test degrades the known-red ledger and burns per-branch triage; it also contradicts the sibling pin `tests/test_train_lora_config_use_rslora.py`, which already pins the field-based contract.
- **Fix shape:** rewrite the assert to the field-based contract — `TrainLoraConfig().use_rslora is True` plus the threading pin `"use_rslora=cfg.use_rslora" in inspect.getsource(train_lora)` — preserving the test's r=256/alpha=64 rank assertions unchanged.
- verified-at-filing: `grep -n "use_rslora" src/explore_persona_space/train/sft.py` → literal `use_rslora=True` only at L601 (comment); live probe `uv run python -c "import inspect; from explore_persona_space.train.sft import train_lora; print('use_rslora=True' in inspect.getsource(train_lora))"` → `False` (2026-07-24). Failing assert at tests/test_issue1090_fu5_round.py:204.

## Scope

- `tests/test_issue1090_fu5_round.py` only (the assert at L204 and any sibling literal-asserts in the same test). Do NOT touch `src/explore_persona_space/train/sft.py` — the behavior is intact and separately pinned.
