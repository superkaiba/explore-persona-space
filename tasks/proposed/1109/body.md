---
title: 'workflow-fix: v4 Results scan truncates at any mid-body --- rule (masked hard
  cap FAILs)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5670e0be382d
created_at: '2026-07-07T10:22:18Z'
has_clean_result: false
origin_prompt: 'clean-result-critic formal candidate on #825 (base-separator-control
  r1): _v4_results_body footer-line string match truncates checks 20/21 at any mid-Results
  --- rule; fingerprint 5670e0be382d'
workflow: v1
---
## Overview / Motivation

Auto-filed from a formal workflow-fix candidate raised by the clean-result-critic
on task #825 (fold base-separator-control r1). Confidence: HIGH.

## Goal

Fix `_v4_results_body` in scripts/verify_task_body.py so a horizontal rule between results cannot truncate the checks-20/21 scan: cut the Results text at the absolute line index from `_v4_footer_start_line(body)` (mapped into the Results slice) instead of string-matching the footer's first line, or anchor on the `**Repro:**` label line and strip only the immediately-preceding `---`/blanks.

## Workflow gap

- **Bug observed:** `_v4_results_body` truncates the `## Results` scan at the first line string-equal to the footer's FIRST line; when the footer starts with a `---` rule (the standard v4 shape), any earlier `---` between results truncates every later result out of checks 20 (word caps) and 21 (three-beat) — on #825 this masked two ≥180-word per-result HARD FAILs and excluded 2 of 16 results from the three-beat scan while the verifier reported OVERALL PASS.
- **Why it is a workflow gap:** a horizontal rule between results is legal markdown the spec nowhere forbids, and the verifier is the AUTHORITATIVE cap gate — a first-occurrence string match on a one-character-class boundary line silently defeats two mechanical checks on any body that uses it.
- **Confidence (emitter):** high

## Proposed change (candidate diff sketch — refine in planning)

```
in _v4_results_body():
-    footer_first_line = footer.splitlines()[0] ...
-    for i, line in enumerate(rlines):
-        if line.strip() == footer_first_line.strip():
+    start = _v4_footer_start_line(body)          # absolute body line index
+    # map `start` to the Results slice offset (Results is the last H2,
+    # so results lines are a suffix of body lines) and cut there,
+    # then strip the trailing `---`/blank lines above the footer.
regression test: a v4 body with a `---` between two results, the second ≥180 words -> check 20 FAILs.
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py` (`_v4_results_body`, ~line 2634)
- Regression test in tests/test_verify_task_body.py.

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 5670e0be382d

(Verbatim candidate block preserved in the emitting critic's epm:clean-result-critique v11 marker on task #825.)
