---
title: 'Wire the absolute per-cell trainability gate into scripts/issue2225_train.py
  (the sibling entrypoint #2242 left ungated)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-12T14:29:34Z'
has_clean_result: false
parent_id: 2242
origin_prompt: 'surfaced by the #2242 code review bug-class sweep: scripts/issue2225_train.py
  is a main-resident fine-tune entrypoint with its own load_dataset + SFTTrainer and
  no trainability gate; D11 was deliberately scoped to the incident''s own script'
workflow: v1
---
# Wire the absolute per-cell trainability gate into `scripts/issue2225_train.py` (the sibling fine-tune entrypoint #2242 left ungated)

## Goal

Close the mechanical arm of the #2242 absolute per-cell trainability floor on
the ONE remaining main-resident fine-tune entrypoint that can still train a
structurally-untrainable cell without any gate firing:
`scripts/issue2225_train.py`.

## Why this exists

#2242 installed the absolute floor (below it a cell is DROPPED, not shrunk to 1
row) across four surfaces:

- **D1** — the shared gate in `src/explore_persona_space/artifacts/datagen.py`
  (`trainability_floor_rows`, `assert_cell_trainable`, `CellTrainabilityError`,
  and an additive `min_rows_absolute` entry assert on
  `generate_training_data`).
- **D11** — the one UNCONDITIONAL runtime arm, in `scripts/issue778_finetune.py`
  at the `load_dataset` row-count site, before the smoke slice and the model
  load.
- **D2/D4/D5/D6** — the review-side arms (`on-policy-completions.md`,
  `planner-section-reference.md`, `critic-lens-reference.md`, `planner.md`),
  which bind every FUTURE plan through the planner + critic.
- **D7/D8/D9/D9-bis/D10** — the lint surface pin + tests.

D11 was deliberately scoped to the incident's own script (#2221's demonstrated
reuse magnet). The #2242 code review's bug-class sweep then found the sibling:
**`scripts/issue2225_train.py` is a main-resident fine-tune entrypoint that
extends the same recipe — it has its own `load_dataset` + TRL `SFTTrainer` and
imports the same module constants from `issue778_finetune.py`
(`ft.PER_DEVICE_BATCH`, `ft.LORA_R`, …) — and carries NO trainability gate.**

That review recorded it as a standing recommendation, explicitly NOT a condition
on the #2242 round: #2225's own runs are complete, and any future plan reusing
its trainer is now bound by the D4/D5 review arms. This task is that
recommendation, filed rather than parked as a chat note (CLAUDE.md
§ Workflow-fix-on-bug protocol: a concrete workflow-surface suggestion in an
agent's report prose gets the same auto-file + spawn as a formal
`workflow-fix-candidate` block).

## Scope

1. Add the gate call at `scripts/issue2225_train.py`'s row-count site — the
   `load_dataset` result, BEFORE any smoke/subset slice and BEFORE the model
   load, so it sees the FULL realized row count and raises before any model
   download, GPU allocation, or trainer construction. Mirror the D11 shape in
   `scripts/issue778_finetune.py` (its `_gate_cell_trainability` helper is the
   template — reuse it directly if importable, rather than writing a third copy;
   check first, per the reuse-discovery rule).
2. Derive `effective_batch_size` and `epochs` from THIS script's own constants,
   not #778's — the two scripts share constant NAMES via import but a divergent
   value would silently mis-derive the floor. State the resolved arithmetic in
   the failure message the way D11 does.
3. Thread the same two override flags (`--trainability-floor-override` +
   `--trainability-override-reason`), preserving the invariant that an override
   without a reason raises. If #2225 has a wave/fan-out dispatch path, thread
   them through it too.
4. Smoke demotion: `on_fail="warn"` under this script's own smoke discriminator
   (find it — do not assume it is `max_steps is not None` as in #778), and
   ENUMERATE the downgrade as a smoke blind spot per
   `.claude/rules/smoke-blind-spots.md`.
5. Tests: extend `tests/test_datagen_trainability_floor.py` (or add a sibling)
   with the production-raise / smoke-warn / override paths for this entrypoint,
   and register per `scripts/select_step9c_tests.py`'s `WORKFLOW_INVARIANT`
   convention if a new file is added (note the two-place registration — the
   tuple AND `tests/step9c_workflow_invariant_manifest.txt`).

## Known trap, inherited from #2242

#2242's review logged this Minor on the #778 arm, and #2225 should not repeat
it: passing `--trainability-floor-override` WITHOUT
`--trainability-override-reason` is caught only inside each per-cell subprocess
(a `ValueError` after the tokenizer + dataset load), so an N-cell wave dies N
times post-fan-out instead of once at argparse. Add the paired-flag validation
in `main()` before dispatch.

## Provenance

Surfaced by the #2242 code review's bug-class sweep (2026-08-12), which verified
by grep that `scripts/issue2225_train.py` calls no gate and that its coupling to
`issue778_finetune.py` is constants-only (zero calls to `train_single_cell`).
Parent #2242 carries the floor's definition, its literature grounding (~12
optimizer steps, the narrowest observed install transition — #533/#547), and the
two-tier reconciliation against `critic-lens-reference.md` item 9(i).
