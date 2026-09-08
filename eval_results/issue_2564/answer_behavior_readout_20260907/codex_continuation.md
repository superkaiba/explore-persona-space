# Codex judging continuation

The pilot is accepted with qualifications. The first production packet passed
the planned 256-answer completeness and envelope check. The remaining frozen
roster is released. No actual-label readout has been fit yet.

Run from `/home/thomasjiralerspong/.codex/worktrees/issue2564-answer-behavior`.
The research root is
`/home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907`.
Use the existing project environment. The earlier API resume instructions are
historical; do not retry the rejected key or mix those artifacts with Codex
judgments.

## Annotation continuation

`codex_main_jobs.json` in the research root is the coordinator's live dispatch
ledger. `codex_main_jobs_before_dispatch.json` is the immutable original roster;
`codex_main_roster_preflight.json` records its exact 280-packet coverage and
content-only checks. All 2,048 answers receive five fresh contexts for each of
seven properties. The requests use 256 answers per packet. No sample, repeat,
rubric, model or reasoning-effort change is authorized by this runbook.

Dispatch pending requests through the collaboration subagent tool with the exact
saved request, `fork_turns: none`, and no overrides. Use a fresh agent for each
packet. Record actual canonical IDs and dispatch observations with the adapter's
`mark` command. Mark completion only after an actual completion notification,
then import the ledger. A file's mere existence is not completion evidence.
Malformed, incomplete or replaced outputs must fail validation and remain
preserved; never fabricate labels or silently discard a packet.

The first packet took 591.49 seconds between coordinator observations and
produced 256 valid records. Extrapolating that one persona packet to two workers
would suggest roughly 23 hours for the full roster. This is a provisional timing
estimate, not a measured estimate for the other properties. Five repeated
ratings remain shared-model judgments with uncontrolled sampling, not IID draws
or human validation. Instruction-enforced answer-only access is not technical
isolation.

## After all ratings finish

Use the adapter to import, aggregate and validate the complete main roster:

```bash
uv run --no-sync --project /home/thomasjiralerspong/explore-persona-space python scripts/issue2564_codex_judgments.py import --root /home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907 --part main --ledger /home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907/codex_main_jobs.json
uv run --no-sync --project /home/thomasjiralerspong/explore-persona-space python scripts/issue2564_codex_judgments.py aggregate --root /home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907 --part main
uv run --no-sync --project /home/thomasjiralerspong/explore-persona-space python scripts/issue2564_codex_judgments.py validate-main --root /home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907
```

The complete aggregate lives at `annotation_codex/main/labels.json`. Readout
defaults explicitly select that provider and save to `readout_codex/`. The loader
validates accepted pilot evidence and all aggregate/completion fingerprints.
Run one outer fold, resume the complete fit, then summarize:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 uv run --no-sync --project /home/thomasjiralerspong/explore-persona-space python scripts/issue2564_answer_behavior_readout.py stage=fit first_fold_only=true
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 uv run --no-sync --project /home/thomasjiralerspong/explore-persona-space python scripts/issue2564_answer_behavior_readout.py stage=fit
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 uv run --no-sync --project /home/thomasjiralerspong/explore-persona-space python scripts/issue2564_answer_behavior_readout.py stage=summarize
```

Review actual coverage, target reliability and missingness, paired within-target
readouts and their conditional bootstrap intervals, null mobility, and declared
sensitivities. Do not infer a universal high-versus-low ordering from unlike
metrics. Archive all raw judgments, dispatch evidence, accepted instrument,
inputs, fits and analysis outputs with verified remote content hashes. Publish
the reviewed scientific report and notify the parent agent. Do not edit the
paper or send Slack messages.

The accepted pilot and first seven completed main packets (1,792 main ratings)
are also [archived as a verified partial checkpoint](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/322c8fd82e6fb917ddf6c5834e0ef40b8d999685/issue2564_minpair/answer_behavior_readout_20260907/raw_completions/codex_main_checkpoint_007).
All 197 files and their 7,286,137 bytes match the local content hashes. This
checkpoint includes actual dispatch records and is explicitly incomplete; it
is not a main completion marker or a behavioral result.
