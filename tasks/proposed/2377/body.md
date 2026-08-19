---
title: 'workflow-fix: tmp-scratch sweep blind spot — stray top-level /tmp/{pyproject.toml,uv.lock}
  poisons every /tmp-cwd uv run fleet-wide'
kind: infra
tags:
- wf-fix
created_at: '2026-08-19T01:13:14Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from task #2183 round-2 implementer: stray
  /tmp/pyproject.toml+uv.lock break uv project discovery for all /tmp-cwd uv runs;
  #2127 sweep covers directories only'
workflow: v1
---
# Sweep/alert on stray top-level `/tmp/{pyproject.toml,uv.toml,uv.lock}` poisoning uv project discovery fleet-wide

## Goal

Close the janitor/guard blind spot for stray top-level `/tmp` uv PROJECT FILES: add an escalate-or-evidence-gated-reap arm for `/tmp/pyproject.toml`, `/tmp/uv.toml`, `/tmp/uv.lock` to the #2127 tmp-scratch sweep (`clean_experiment_downloads.py::sweep_tmp_scratch`), or at minimum a `vm_disk_guard`/watcher alert naming the files and their uv blast radius.

## Why

A stray top-level `/tmp/pyproject.toml` + `/tmp/uv.lock` (dropped by an unidentified concurrent session, 2026-08-18 17:02/17:10 PT) made EVERY `/tmp`-cwd `uv run` on the shared VM exit rc=2 — uv project discovery walks up to `/tmp`, finds the project pair, and fails resolving `xgrammar==0.2.4` (aarch64-only wheels). This turned gate tests that shell out from `/tmp` scratch red fleet-wide (observed: `tests/test_guard_lessons_edit.py::test_unimportable_lint_fails_open`, failing identically on pristine main).

The #2127 `sweep_tmp_scratch` leg sweeps top-level `/tmp` scratch DIRECTORIES only; a top-level project FILE pair is outside every janitor/guard surface, and nothing detects or alerts on it.

Repro (while the files exist): `mkdir /tmp/x && cd /tmp/x && uv run python -c 'print(1)'` → rc=2.

## What was already done (incident remediation, task #2183 round 2)

Evidence-gated reversible remediation only: `/tmp/pyproject.toml` was byte-identical to the committed repo blob (`8f0ebf3cc2…`), zero open handles, no live owner → both files QUARANTINED (moved, never deleted) to `/tmp/eps-quarantine-issue2183-tmp-pyproject-20260818/`. Repro rc=0 and the affected test file 31/31 green afterwards. The quarantine is the restore point if some session staged them deliberately.

## Acceptance criteria

1. The sweep (or a guard/watcher pass) detects top-level `/tmp/{pyproject.toml,uv.toml,uv.lock}`.
2. Reap only under the existing evidence contract (git-blob identity proof → quarantine/reap); otherwise HARD-ESCALATE (push + sidecar row) naming the files and the uv blast radius — never an age-gated deletion (age is only ever a KEEP signal).
3. Respect the existing kill switches / contract of the #2127 leg it extends (`EPM_SKIP_TMP_SCRATCH_SWEEP=1` family).
4. A regression test pinning the new arm's fire + escalate branches.

## Provenance

Surfaced by task #2183 round-2 implementer (`epm:results` v2 marker § (d) + workflow-fix-candidate block in its report). Filed by the #2183 orchestrator session per `.claude/rules/workflow-fix-on-bug.md` (auto-file + spawn).
