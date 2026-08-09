---
title: 'gotchas.md: parallel lm-eval fan-out races library nltk download — pre-stage
  once before forking (from #2203)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-09T20:45:47Z'
has_clean_result: false
parent_id: 2203
origin_prompt: 'experimenter epm:failure-lesson (gotcha_candidate) during #2203 Phase-2
  capability crash-fix round, 2026-08-09'
workflow: v1
---
## Goal

Add a `.claude/rules/gotchas.md` entry (and, if judged appropriate by the implementer, a matching launcher-template note in the experimenter agent spec) for the parallel lm-eval fan-out nltk staging race, so future multi-GPU capability/eval fan-outs pre-stage nltk resources once before forking.

## Context (incident, #2203 Phase 2, 2026-08-09)

A 4-way CVD-pinned lm-eval capability fan-out (one process per GPU, shared container) crashed 2 of 4 shards at IFEval dataset prep: each process, on first use, triggers the lm-eval library's own `nltk.download('punkt_tab')` into the shared `/root/nltk_data`; concurrent downloads collide (one process removed `punkt_tab.zip` mid-unzip of another), producing `LookupError: Resource 'punkt_tab' not found` in the losers. The surviving shards completed normally, which makes the failure look nondeterministic/flaky rather than structural.

- Failure signature to document: `[nltk_data] [Errno 2] No such file or directory: '.../punkt_tab.zip'` followed by `LookupError: Resource punkt_tab not found`, in SOME shards of a parallel fan-out while sibling shards succeed.
- Fix pattern (validated on the #2203 retry launcher): stage the resource ONCE, single-threaded, as a launcher gate BEFORE the fan-out — `nltk.data.find('tokenizers/punkt_tab')` except LookupError → `nltk.download('punkt_tab')` → re-`find()` — and fail the launcher loud on rc!=0. This mirrors the existing stage-once-before-fan-out discipline for HF artifact staging (the #2203 axis pre-stage).
- Scope note: the nltk call lives in library code (lm-eval IFEval utils), not repo scripts, so the fix layer is the launch wrapper / bootstrap, not experiment code.

## Acceptance

1. `gotchas.md` gains the entry (signature + fix pattern + one-line incident cite), with the LESSONS.md index row trigger text updated if the gotchas.md trigger line changes.
2. `workflow_lint.py --check-lessons-index` passes.
