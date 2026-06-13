---
title: Scale persona-space experiments to bigger models
kind: experiment
tags: []
created_at: '2026-05-29T08:00:59Z'
has_clean_result: false
---
## Idea

Scale the persona-space experiments beyond the current Qwen-2.5-7B base to
larger models (e.g. Qwen-2.5-14B / 32B / 72B, or a Llama-family larger model).
Captured from a Todoist one-liner ("Move to bigger models"), so the exact
intended axis is underspecified; most likely it means re-running the
trait-transfer / emergent-misalignment / persona-axis-characterization
pipeline on a bigger base to test whether the findings hold at scale.

## Why it matters

- Sibling papers (Persona Vectors; Persona Features Control EM, Mossing et al.)
  used GPT-4o-scale models. Showing the same persona-axis structure replicates
  on a substantially larger open-weights model strengthens positioning and
  guards against "this is a small-model artifact" reviewer pushback.
- Several current results may be size-dependent (steering strength, axis
  separability, EM susceptibility). Worth knowing which scale up.

## Open questions

- Which model(s) and what size jump is the right next rung (14B vs 32B vs 72B)?
- Which specific experiment(s) move first — trait transfer, axis
  characterization, or the User Modeling / Persona Selection thread (Topic 7)?
- Compute fit: does the chosen size fit current H100/H200 pods, or does it need
  multi-GPU / quantization? Cost estimate before committing.
- Which results are most worth replicating at scale vs which are already
  convincing at 7B?

## Source

Captured via my-goat Todoist auto-processor from
queue/inbox/2026-05-29T01-00-05_todoist-6gjjgG5R9p4wpJVv_move-to-bigger-models.md
(Todoist id 6gjjgG5R9p4wpJVv).
