---
title: 'workflow-fix: fence ancestor-gitignore discovery in tmp_path real-ruff fixtures
  (stray /tmp/.gitignore blanks ruff probes fleet-wide)'
kind: infra
tags:
- wf-fix
created_at: '2026-08-21T00:30:20Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate v1 from #2430 implementer round 1: /tmp/.gitignore
  ancestor-hazard blanks tmp_path ruff probes'
workflow: v1
---
# Fence ancestor-gitignore discovery in tmp_path-based real-ruff fixtures

## Goal

`tests/test_step9c_baseline.py::test_ruff_helpers_real_body` (and any tmp_path-based real-ruff probe) fails fleet-wide whenever any process drops a `.gitignore` into `/tmp` — ruff walks ancestor directories for gitignore files, and a `/tmp/.gitignore` containing `*` makes every directory-path `ruff check` under `/tmp/pytest-of-*` find zero files.

## Evidence

- Observed 2026-08-20 ~16:25-16:35 PT during #2430 round 1: `sb.ruff_error_count(tmp_path)` returned 0 on a file carrying F401; identical failure on the issue worktree AND pristine main; a stray `/tmp/.gitignore` (`*`, ctime 15:40:41, dropped by a concurrent session) was the cause; removing it fixed both trees.
- Reproduced deterministically: same probe content under a scratch dir in `/tmp` fails, under `$HOME` passes — ruff respects ancestor gitignores and pytest tmp_path lives under `/tmp`.

## Suggested approach

Fence ancestor gitignore discovery in the fixture: create `(tmp_path / ".git").mkdir()` (ruff stops gitignore discovery at a repo root) or write a tmp-local `.gitignore` with `!*` before invoking `ruff_error_count`/`ruff_format_count`; alternatively assert the probe found >=1 file and fail with a diagnostic naming the ancestor-gitignore hazard. Sweep for sibling tmp_path real-ruff probes and apply the same fence.

## Acceptance criteria

- With a `/tmp/.gitignore` containing `*` present, the fixture still exercises ruff on its probe file (>=1 file checked) and the test passes.
- No behavior change when `/tmp` is clean.

## Provenance

workflow-fix-candidate emitted by the #2430 round-1 implementer (session cmt20scforug6wo0uacp25vvk); confidence high.
