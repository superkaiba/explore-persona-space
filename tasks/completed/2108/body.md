---
title: 'workflow-fix: bootstrap sparse-cone block dies, killing missing-cone warning'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e47c7a1016ac
created_at: '2026-08-06T06:04:18Z'
has_clean_result: false
origin_prompt: 'Observed live in /issue 2061 provision 2026-08-06T06:01Z: scripts/bootstrap_pod.sh:
  line 364: sparse-checkout: command not found, emitted immediately before BOOTSTRAP-OK
  pod=pod-2061. The failing block carries the missing-artifact-cone WARNING (local
  lines 358-359), so that safety guard is silently dead while looking present in source.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a bug observed LIVE during a
`pod.py provision` / `bootstrap_pod.sh` run on task #2061 (emitting agent: the
/issue 2061 orchestrator, 2026-08-06T06:01Z, pod-2061).

## Goal

Fix the remote-heredoc quoting in `scripts/bootstrap_pod.sh`'s sparse-cone
reporting block so the cone diagnostic AND — the part that actually matters —
the missing-artifact-cone WARNING both execute on the pod instead of dying with
`sparse-checkout: command not found`.

## Workflow gap

- **Bug observed:** the provision's own bootstrap output carried
  `/home/thomasjiralerspong/explore-persona-space/scripts/bootstrap_pod.sh: line 364: sparse-checkout: command not found`
  immediately before `BOOTSTRAP-OK pod=pod-2061`. The reported line number is the
  REMOTE composed script's numbering, not the local file's; the local suspect
  region is lines 356-360, the `core.sparseCheckout`-gated reporting block.
- **Why it is a workflow gap:** `scripts/bootstrap_pod.sh` is named workflow
  surface (`.claude/rules/workflow-fix-on-bug.md` § Workflow surface,
  "Workflow-helper scripts under `scripts/`"). The block that fails is not
  merely cosmetic: line 357 is a `Sparse cones: ...` diagnostic, but lines
  358-359 are a SAFETY WARNING whose own text states the stakes — "committed
  eval_results/figures for issue N will be ABSENT on this pod and workloads
  reading them will crash FileNotFoundError". If the `git` prefix is being lost
  in this block's composition, that warning cannot fire, so the guard against
  a silently-missing artifact cone is dead while still LOOKING present in the
  source. `BOOTSTRAP-OK` is still reported, so nothing surfaces the failure.
- **Confidence (emitter):** medium on the exact offending expansion (the block
  is doubly-quoted remote-heredoc text, so the precise layer that strips `git`
  needs the composition read end-to-end); HIGH that the error is real and
  reproducible — it was observed in this session's own provision stdout.
- verified-at-filing: `grep -n 'sparse-checkout' scripts/bootstrap_pod.sh` ->
  15 hits, all 15 in the single named target `scripts/bootstrap_pod.sh`
  (per-target: 15/15). Suspect region confirmed present at lines 356-360 by
  `sed -n '355,372p'`; every OTHER invocation (263-283, 317-338) carries an
  unescaped `git sparse-checkout ...` prefix and is not implicated. Read the
  hit context per clause (c): the region does NOT already implement this fix —
  the bug is live, evidenced by the runtime error above. (2026-08-06)

## Proposed change (candidate diff sketch — refine in planning)

Read the composition of the remote script end-to-end (how this block is
transported to the pod — the escaping layer around lines 356-360) and repair the
quoting so `git sparse-checkout list` reaches the pod with its `git` prefix
intact. Then make the block's failure non-silent: the missing-cone WARNING is a
guard, so a composition error in it should surface rather than being absorbed
next to a `BOOTSTRAP-OK` line.

  - echo \"Sparse cones: \$(git sparse-checkout list 2>/dev/null | tr '\\n' ' ')\"
  + <quoting repaired so `git sparse-checkout list` executes remotely; verify by
  + a real provision whose output shows a populated `Sparse cones: ...` line
  + rather than `sparse-checkout: command not found`>

unverified hypothesis — verify at plan time: that the lost `git` prefix is
caused by the escaping of `\$(...)` in this specific block rather than by an
outer transport layer. Recall source: reading local lines 356-360 against the
observed remote error; NOT mechanically traced to the constructing site, so
clause (g) call-hop tracing is owed at plan time before `target_file` is treated
as final.

## Scope / surfaces

- Primary target: `scripts/bootstrap_pod.sh` (lines 356-360, the
  `core.sparseCheckout`-gated reporting block)
- Sibling invocations at 263-283 and 317-338 use the unescaped `git
  sparse-checkout` form and appear correct — check them for the same class but
  do not change them without evidence.
- Acceptance: a real `pod.py provision` (or an equivalent composed-script probe)
  whose bootstrap output shows a populated `Sparse cones:` line, and — on a
  deliberately cone-less checkout for an issue with committed artifacts — the
  missing-cone WARNING actually firing.

## Constraints / invariants

- Workflow-surface only — no experiment code, no `configs/`, no `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The fix must not weaken `BOOTSTRAP-OK` semantics: a genuinely-broken bootstrap
  must still fail loud (`log_fail` + exit 1), per the fail-fast rule.

## Provenance

- workflow_fix_target: scripts/bootstrap_pod.sh
- fingerprint: e47c7a1016ac

Observed live in the /issue 2061 provision (2026-08-06T06:01Z, pod-2061):
`scripts/bootstrap_pod.sh: line 364: sparse-checkout: command not found`,
emitted immediately before `BOOTSTRAP-OK pod=pod-2061`. The pod itself was
healthy (full clone at the expected sha, both round-5 scripts present), so the
failure was silent from the caller's point of view — which is the reportable
part.
