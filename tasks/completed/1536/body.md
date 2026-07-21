---
title: 'daily-fix: fail-fast SFT prompt/completion type assert'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-19T07:06:55Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): A mixed-type prompt/completion
  dataset enters the SFT data path undetected; the existing isinstance checks in sft.py
  (L833/L1129) are silent-skip guards in probe/KL-aux side paths, not a fail-fast
  validation before SFTTrainer init (#1508).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the 2026-07-18 /daily Step C parked-candidate sweep from a
prose follow-up raised on task #1508 (emitting agent: Methodology critic).
NOTE: this is an EXPERIMENT/LIBRARY-code fix (`src/explore_persona_space/train/`
is outside the workflow-fix surface) — filed as a `wf_fix: false` route-2
item per the /daily route-2 variant; no wf-fix tags, no Provenance injection.

## Goal

Add a deterministic fail-fast prompt/completion type-homogeneity assert in
the project SFT data path (`src/explore_persona_space/train/sft.py` /
`train_lora`): validate that BOTH keys agree in type across the dataset
before SFTTrainer init, as the code-level backstop to the #1508 doc entry.

## Workflow gap

- **Bug observed:** a mixed-type prompt/completion dataset enters the SFT
  data path undetected until deep inside training (#1508's incident); the
  existing `isinstance(prompt, list)` checks in sft.py sit in the marker-probe
  and KL-aux side paths and SILENTLY SKIP non-conforming rows
  (`return None` / `continue`) rather than failing fast.
- **Why it is a workflow gap:** (experiment-code gap, not workflow-surface)
  the project's fail-fast rule ("no silent defaults; the crash IS the
  signal") is violated by the silent-skip shape; a homogeneity assert before
  SFTTrainer init converts a late confusing failure into an immediate,
  attributable one.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'isinstance' src/explore_persona_space/train/sft.py` → type checks at L833 + L1129; context READ (L820-845, L1120-1140): both are silent-skip guards in the marker-slot probe and KL-aux batch builder, NOT a fail-fast homogeneity validation of the training dataset before SFTTrainer init — the proposed assert is not landed; `grep -n 'homogene' src/explore_persona_space/train/sft.py` → 0 hits; `git log --oneline --since='7 days ago' -- src/explore_persona_space/train/sft.py` → 0 commits (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up)

Sketch for the planner: in the sft.py dataset-load/validation path (before
SFTTrainer construction in `train_lora`), scan the loaded rows once and
raise a ValueError naming the first offending row index when `prompt` /
`completion` types are non-homogeneous (str-vs-list mixed) or the two keys
disagree in convention; cover with a unit test on a deliberately mixed JSONL.

## Scope / surfaces

- Primary target: `src/explore_persona_space/train/sft.py` (train_lora data path)
- Experiment/library code — the standard `/issue` code-change pipeline
  (implementer + code-reviewer + tests) applies; this is NOT a
  workflow-surface edit.

## Constraints / invariants

- Fail fast, never silently skip rows in the TRAINING path; the existing
  probe/KL-aux silent-skip guards may stay (they are diagnostic side paths)
  but the plan states the boundary explicitly.
- No behavior change for currently-valid homogeneous datasets; ruff + the
  sft.py pinning tests pass.

## Provenance

- source task: #1508 (prose follow-up, methodology critic, 2026-07-18T13:30:35Z)
- fingerprint (informational; wf_fix false — no fp-dedup): 8db8ed7096eb

Verbatim surfaced prose (task #1508 events.jsonl):
"Methodology critic surfaced-prose follow-up — add a deterministic fail-fast
prompt/completion type-homogeneity assert in the project SFT data path
(src/explore_persona_space/train/ sft.py / train_lora, validate both keys
agree before SFTTrainer init) as a code-level backstop to the #1508 doc
entry. source: prose-followup; confidence: medium; related_task: #1508."
