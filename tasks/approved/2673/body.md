---
title: Story Imprinting persona-context geometry on Qwen3.8-27B
kind: experiment
tags: []
created_at: '2026-09-17T20:00:03Z'
has_clean_result: false
origin_prompt: ok try it on qwen3.8 27b as a pilot first; yes run it end to end
workflow: v1
goal: Inspect whether the Story Imprinting persona prompts form the expected progressive
  similarity trajectories in Qwen3.8-27B last-context-token representations, using
  all-layer centered cosine on the same 240 questions without measuring behavioral
  leakage.
---
# Qwen3.8-27B persona-context geometry pilot

## Goal

Inspect whether the Story Imprinting persona prompts form the expected progressive similarity trajectories in Qwen3.8-27B last-context-token representations, using all-layer centered cosine on the same 240 questions without measuring behavioral leakage.

## Authorized scope and launch gate

User request: "ok try it on qwen3.8 27b as a pilot first". Earlier clarification:
"i just wanted to actually look at the context vectors for persona prompted
contexts. similar to our previous experiments".

The user explicitly approved task creation and end-to-end execution on
2026-09-17: "yes run it end to end". Create the canonical experiment task via
scripts/task.py, record this approval, and run the reviewed pilot through
capture, analysis, figures, verified persistence, and gated compute teardown.
Routine runtime fixes within this design are authorized; preserve the Goal.

## Design

- Primary panel: the six exact conditions from Story Imprinting Appendix C.6,
  Tables 8–9 (empty system message; sarcasm; sarcasm+lists; full SFL; French;
  French+lists). Paper: https://arxiv.org/html/2609.10883v1#A3.SS6.
- Secondary panel: the four exact Table 7 prompts (dismissive, sarcastic,
  saboteur, peer). Paper: https://arxiv.org/html/2609.10883v1#A3.SS4.
- Questions: all 240 existing data/assistant_axis/extraction_questions.jsonl
  rows, identically paired across conditions. Reuse the established extraction
  battery for comparability; its constructed questions limit distributional
  generalization. No new synthetic questions are generated.
- Total: 10 conditions × 240 questions = 2,400 contexts, no sampled completions.
- Model: official Qwen/Qwen3.8-27B BF16 checkpoint, immutable Hub revision
  verified before dispatch. Assert 64 decoder blocks and hidden width 5,120.
- Render: native chat template, add_generation_prompt=true,
  enable_thinking=false; preserve the paper's empty system message explicitly.
  Capture the actual last token of this complete assistant-generation prefix.
- Capture all 64 block outputs in one batched forward pass per context;
  positions are zero-based decoder-block indices, including pre-final-norm
  block 63. Save only the selected token per layer, not whole sequences.
- Begin with batch size 8 and a padded-token budget; validate on the real model
  before the complete panel. Batch settings are operational pilot choices,
  not research hypotheses. No truncation: reject an overlength row.

## Quantities and interpretation

For each layer and persona, average vectors over the same question set. Center
the ten-persona centroid bank by its global mean, L2-normalize, then compute
pairwise cosine with the existing compute_cosine_matrix helper (#536). Record
all persona names and centering provenance. Also provide explicitly labeled raw
cosines and a six-condition centering sensitivity; never pool different banks.

Report cosine matrices and both ladder trajectories toward the full SFL
centroid at every layer, with fixed representative depths 15, 31, 47, 63.
No best-layer selection, fitted map, statistical ranking claim, or correlation
against approximate readings from the paper's plotted leakage rates.
Record prompt token counts and split-half question stability. Prompt length,
explicit prohibitions, and language/format instructions remain confounds of a
causal interpretation of "persona similarity".

## Implementation and validation

Reuse analysis/extraction.py decoder resolution and output unwrapping,
analysis/representation_shift.py cosine calculation, the established Qwen3.8
loader/render checks from scripts/context_risk_qwen38_smoke.py, and the current
question battery. Record exact source hashes for any unmerged reused code.

Before full capture verify geometry, tokenizer/render parity, exact row/ID
coverage, finite activations, mixed-length batch versus singleton parity on a
small real-model slice, and capture versus the model's hidden-state tuple for
selected non-final layers. Label the final block's pre-norm convention.
Persist each capture batch with a fingerprint including model revision,
prompts/questions, render choices, layers, dtype, library versions, and source.
Resume only from validated matching chunks; a fresh completion sentinel follows
exact coverage and content verification. Run focused CPU tests and independent
review before launch.

The remote workload is `bash scripts/story_persona_qwen38_workload.sh`, invoked
only by the approved task dispatcher. It uses the established
`uv run --with 'transformers==5.15.0'` overlay because the project's default
dependency pin is below version 5. The capture phase checks that exact runtime
before loading weights; the repository lockfile is unchanged. The wrapper runs
capture and verified analysis sequentially. Each batch has a durably recorded
checksum; resume checks it before accepting saved tensors. Analysis recomputes
the manifest fingerprint and checks every chunk against the completion record.

Local validation on 2026-09-17: input preparation verified 2,400 rows; all six
focused CPU tests passed (boundary selection, packing, resume validation,
checksum integrity, manifest integrity, streamed centroids/cosines); Ruff and
the workload shell syntax check passed. These checks use small test tensors,
not Qwen activations. Real-model smoke, capture, and plots remain pending the
required task creation and remote launch.

## Compute and monitoring

One remote GPU with at least 80 GB HBM and sufficient host RAM; BF16 weights are
about 54 GB, with modest short-context batches. No weights or activation stores
on the nearly-full local disks. Allow 1–2 node-hours for staging, loading,
smoke, extraction, and verification; this is provisional and replaced by the
measured smoke estimate before the full capture. Approximately 1.57 GB for all
per-context vectors at BF16 (2,400 × 64 × 5,120 × 2 bytes), plus small metadata
and summaries; checkpoints and dependency staging require separate disk space.

Use the existing approved-task dispatcher and preflight. Keep a timestamped
process/log/output-progress monitor plus the tested experiment_watchdog
supervisor, durable recovery runbook, bounded retries, and acknowledged alerts
through the established personal notification route. Verify the real canary,
timer, startup, and first output progress before unattended operation.

## Deliverables

Raw context vectors and row metadata; per-layer centroids and cosine matrices;
render/padding and model-compatibility smoke report; token-length and stability
diagnostics; a concise results report; browser-accessible plots. Upload and
verify all results before any compute teardown, using the repository's gated
lifecycle tooling. No training, judge calls, Claude automation, or new behavior
implantation is part of this pilot.
