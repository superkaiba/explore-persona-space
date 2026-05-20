---
title: '[Infra] Tier 1 follow-up: code-reviewer concerns + honesty fixes'
kind: infra
tags: []
created_at: '2026-04-17T21:15:34.000Z'
has_clean_result: false
sagan_id: c0e5460f-d290-4a08-a2f4-6d8d9fce3194
sagan_number: 37
priority: normal
legacy_why_unset: true
---
## Context

Code-reviewer verdict on issue #36 (https://github.com/superkaiba/explore-persona-space/issues/36#issuecomment-4271309947) was **CONCERNS, merge-with-follow-up**. No blocking bugs landed. This issue tracks the cleanup commit.

## Scope

### MAJOR (documentation / honesty — not code bugs)

1. **Downgrade / reword Liger log message** in `src/explore_persona_space/train/trainer.py` and `sft.py`.
   - Current: module-level log tells users to install Liger-Kernel for speedup
   - Problem: Liger is disabled on every in-process path (always LoRA→PeftModel) — install instruction is misleading
   - Fix: downgrade to DEBUG OR rewrite to say "Liger-Kernel available but disabled on LoRA paths due to PEFT incompatibility (see b8dd473). Only useful for full-FT runs on distributed path."

2. **Add DPO precompute memory warning** in `trainer.py` DPO path (around `:545-563`).
   - When `precompute_ref_log_probs=True`, log a one-time INFO stating the +63% peak memory tradeoff (measured: 19.08 → 31.09 GB on Qwen-7B LoRA)
   - Helps users running memory-tight setups

3. **Fix benchmark harness metric** in `scripts/benchmark_tier1.py` and `scripts/benchmark_lora_perf.py`.
   - Current: reports `samples_per_second` for packed runs → collapses to 1/K when K examples pack into one sequence
   - Fix: add `train_tokens_per_second` column for packed comparisons; document that samples/sec is misleading when packing=True

### MINOR (7 items from reviewer)

4. Declare `packaging` in `pyproject.toml` (used for version parsing in tokenizer shim at `trainer.py:32-53`)
5. Fix outdated docstring in `sft.py` (code-reviewer flagged a specific inaccuracy — check the review for details)
6. Remove dead helper `_num_training_steps_from_trainer_state` (if still unused after cleanup)
7. Fix potential IndexError in `prepare_dpo_jsonl` — add bounds check on empty prompt/chosen/rejected
8. Change CPU-only probe to raise `TypeError` not `ValueError` (check code-reviewer's context)
9. Make `install_benchmark_callback` actually idempotent (or remove the claim from docstring)
10. Document (or revert) the scope-creep refactors in commits `c57ab0c` and `52d6d2b` that bundled `_init_phase`/`_finalize_phase`/`_resolve_warmup` extractions and `TrainLoraConfig` dataclass migration — violated "one commit per change"

### Verdict post

Post `<!-- epm:note v1 -->` marker on issue #36 recording honestly:
- SFT ≈ 0% on LoRA (target NOT met, documented reasons: short seq, Liger-PEFT incompat)
- DPO +20% confirmed (clean win)
- Memory: DPO +63% (precompute tradeoff)
- Nothing to revert

## Success criteria

- Single cleanup commit batching all 10 items (or 2-3 if grouped logically)
- `ruff check . && ruff format .` clean
- Existing tests pass
- Commit message: `chore(tier1-cleanup): address code-reviewer concerns [refs #36]`
- Posts `<!-- epm:cleanup-done v1 -->` marker on THIS issue

## Budget

≤1 hr, no GPU needed.
