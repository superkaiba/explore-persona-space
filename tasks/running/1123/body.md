---
title: 'workflow-fix: gotchas entry — fan-out handshake timeout masks unit crash'
kind: infra
tags:
- wf-fix
- wf-fix-fp:978b66b292ad
created_at: '2026-07-08T03:38:08Z'
has_clean_result: false
origin_prompt: 'Experimenter failure-lesson on #1112 (gotcha_candidate: yes): all-units
  vLLM handshake timeout across a fan-out usually masks one instantly-crashing unit;
  read the smallest unit log first. Attempt 4 misclassified infra on this.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an experimenter
failure-lesson block on task #1112 (gotcha_candidate: yes).

## Goal

Add a `.claude/rules/gotchas.md` entry (vLLM family): an ALL-units vLLM
5-minute front-end handshake timeout across a fan-out is usually the MASK of
one unit crashing instantly — the driver raises on first unit failure and
abandons sibling front-ends, whose EngineCores dump the handshake timeout 5
minutes later. Diagnostic rule: read the earliest/smallest unit log's
traceback BEFORE classifying infra.

## Workflow gap

- **Bug observed:** #1112 attempt 4 was classified `failure_class: infra`
  ("engine-init wedge") off the handshake-timeout symptom and burned an
  experimenter respawn; attempt 5's pre-clean + relaunch reproduced it in
  20s and found the real cause — a deterministic FileNotFoundError in ONE
  capture unit whose 1,458-byte crash log from attempt 4 was already on
  disk, unread.
- **Why it is a workflow gap:** the failure classifier + gotchas runbook
  treat "vLLM init failure" as infra; the mask pattern inverts that for
  fan-outs and no rule documents it. Optional second leg: a
  `failure_classifier.py` heuristic (all-unit handshake-timeout + a unit
  log < ~2KB with a Python traceback → route `code`) if cheap; the
  gotchas entry alone is the minimum fix.
- **Confidence (emitter):** high (root_cause_confirmed: yes; live
  reproduction).

## Proposed change (candidate diff sketch — refine in planning)

```
+ .claude/rules/gotchas.md (vLLM section): the fan-out handshake-timeout
+   mask entry (symptom, mechanism, the read-smallest-unit-log-first rule,
+   #1112 incident).
+ OPTIONAL scripts/failure_classifier.py: when the body matches the
+   handshake-timeout pattern AND a --log unit dir is provided, scan unit
+   logs for a small traceback-bearing file and emit `code`.
+ .claude/skills/issue/failure_patterns.md: mirror any classifier change.
```

## Scope / surfaces

- Primary target: .claude/rules/gotchas.md
- Secondary (optional): scripts/failure_classifier.py + failure_patterns.md
- Grep first: `grep -rn 'handshake' .claude/rules/gotchas.md scripts/failure_classifier.py`

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 978b66b292ad

Origin: experimenter failure-lesson block on #1112 (2026-07-08, attempt-5
diagnosis; agent-memory twin at
.claude/agent-memory/experimenter/feedback_fanout_handshake_timeout_masks_single_unit_crash.md).
