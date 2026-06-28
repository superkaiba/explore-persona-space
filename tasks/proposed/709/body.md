---
title: 'RunPod SSH glob_sentinels — close #705''s deferred stale-sentinel sibling
  probe over SSH'
kind: infra
tags: []
created_at: '2026-06-28T08:07:39Z'
has_clean_result: false
parent_id: 705
origin_prompt: '#705 round-1 reconciler PASS: persisted concern runpod-ssh-stale-sentinel-deferred
  for follow-up'
---
## Overview / Motivation

Filed from #705's Step 5 reconciler verdict (round 1, PASS): the `_default_glob_sentinels` in `src/explore_persona_space/backends/artifacts.py` is local-FS only, so a RunPod live-pod multi-attempt stale-sentinel case (the #685 SECONDARY cause — declared sentinel missing, one live sibling exists in another attempt-dir under `/workspace/eval_results/issue_<N>/`) cannot be resolved by the new sibling probe over SSH. Codex's review correctly flagged this; the reconciler sustained the deferral on PASS because #685's primary cause (`[git] not tracked by git`) IS fixed and the secondary path FAILs honestly (no silent-pass regression), but the routine `--skip-confirm-artifacts` pressure for THAT specific RunPod stale-sentinel shape persists. This task closes the gap.

## Goal

Wire an SSH-backed `glob_sentinels` for RunPod so the multi-attempt stale-sentinel sibling probe resolves over the live pod's filesystem too — matching the local-FS resolution #705 already shipped for GCP / SLURM.

## Problem (from #705 review)

`backends/runpod.py::RunPodBackend.confirm_artifacts` injects `read_sentinel=self._ssh_read_sentinel(handle)` but NOT `glob_sentinels`. `_resolve_live_sentinel` therefore calls the default local-FS glob (`Path(eval_results/issue_<N>).glob(...)`) → zero local siblings → declared-missing FAILs honestly. A RunPod multi-attempt run with a stale-baked sidecar but one live sibling on the pod still forces `--skip-confirm-artifacts`.

## Proposed change

- Add an SSH-backed `glob_sentinels` callable in `backends/runpod.py` that lists `/workspace/eval_results/issue_<N>/*/.completion-sentinel.json` over SSH via the existing SSH helper used by `_ssh_read_sentinel`.
- Inject it into the `VerifierIO` at `RunPodBackend.confirm_artifacts` next to `read_sentinel`.
- New regression test fixture mocking SSH `ls` + `cat` and asserting `confirm_artifacts` PASSes with exactly one live sibling, FAILs with zero, and FAILs with ≥2 (matching the local-FS contract in #705's tests).
- Same back-compat default: when `glob_sentinels` is None, behavior is unchanged.

Optional (named as standing concerns 2 + 3 from the same Codex review — defer to a separate task if they prove non-trivial):
- Scope-guard the glob to the canonical `eval_results/issue_<N>/<attempt>/.completion-sentinel.json` shape before resolving.
- Wrap the sibling liveness reads so an unreadable sibling returns a structured FAIL rather than escaping `verify_artifacts`.

## Scope / target files

- `src/explore_persona_space/backends/runpod.py`
- `src/explore_persona_space/backends/artifacts.py` (optional — only if the scope-guard / read-error wrap is included)
- `tests/`

## Constraints

- Workflow-surface only (in #705's in-scope list).
- Don't relax the gate: zero local siblings + zero remote siblings = declared FAIL, unchanged.
- Lint gate green; ruff clean on touched files.
- The local-FS path (#705 default) MUST stay unchanged — RunPod adds an SSH variant on top.

## Provenance

- Parent: #705 (round-1 reconciler PASS verdict; sustained deferral framed by the plan v1 §12 row 6 Low-confidence assumption + implementer report's `(b) Considered but not done`).
- Linked open concerns:
  - `runpod-ssh-stale-sentinel-deferred` (#705, CONCERN, persisted by reconciler).
  - Standing recommendations from the same reconciler verdict: `stale-sentinel-scope-guard-missing` and `stale-sentinel-sibling-read-raises` (Codex CONCERNs).
