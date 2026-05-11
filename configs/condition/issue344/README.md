# Issue #344 condition YAMLs (bookkeeping only)

These YAMLs exist for parity with `configs/condition/issue186/*.yaml`. They
are **NOT** consumed by `scripts/train.py` for issue #344 — the experiment
trains via `scripts/run_issue_344_train.py`, which:

1. Replaces the Qwen2.5-Instruct chat template with one that wraps
   `{% generation %}` around the `\nAnswer:` line ONLY (for
   `*_labels_on_answer` arms) or around the whole assistant turn (for
   `*_FRESH` arms).
2. Sets `assistant_only_loss=True` + `use_liger_kernel=False` on `SFTConfig`.
3. Runs a dry-run masking gate before any actual training.

See the plan at `.claude/plans/issue-344.md` (the approved Variant B) and
the script's docstring for the actual training contract.

## Cell layout (Variant B = approved)

| Arm                              | Source × Seed cells | Seeds          |
|----------------------------------|---------------------|----------------|
| `persona_cot_labels_on_answer`   | 4 × 3 = 12          | 42, 137, 256   |
| `persona_cot_FRESH`              | 4 × 3 = 12          | 42, 137, 256   |
| `no_cot_FRESH`                   | 4 × 1 = 4           | 42 (only)      |
| `generic_cot_labels_on_answer`   | 4 × 3 = 12          | 42, 137, 256   |
| **Total (Variant B main phase)** | **40 cells**        |                |

C3 gate cells (conditional, librarian only at #96 hparams): 1 × 3 = 3.

## Why `no_cot_FRESH` is single-seed

The `no_cot_FRESH` cell exists solely as the Phase 3 TOST mediation
comparator (Plan §11 'Mediation comparator origin'). It is matched on the
#344 chat template to `persona_cot_FRESH`, so the mediation does not
confound chat-template differences. Single-seed (seed=42) is sufficient
because rationale generation at temperature 0 is highly seed-stable; the
inferential anchors remain the f-ratios (3 seeds), not the mediation.

## Adapter HF Hub paths

After training, each cell uploads its merged checkpoint to:

```
superkaiba1/explore-persona-space::i344_{source}_{arm}_seed{S}_post_em/
```

C3 gate cells use a `_c3gate_seed{S}` infix:

```
superkaiba1/explore-persona-space::i344_librarian_persona_cot_labels_on_answer_c3gate_seed{S}_post_em/
```
