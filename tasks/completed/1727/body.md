---
title: 'daily-fix: flag ungated smoke/full variable pairs in review'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b1706a8c8135
- daily-auto-filed
created_at: '2026-07-27T07:18:21Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): a production dispatcher
  assigned the smoke condition list unconditionally while gating only the models variable,
  silently collapsing a 21-condition experiment to 1 condition through eight implementer
  and review rounds'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 1 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Add a code-review check that every `*_smoke` / `*_full` variable pair in a dispatcher diff is gated on the same smoke predicate, so a smoke-scoped list can never ship as the production default.

## Workflow gap

- **Bug observed:** `scripts/issue1689_dispatch.sh::run_phase_capture` assigned `local conds="$conds_smoke"` unconditionally — the `$SMOKE` gate was applied to the `models` variable but never to `conds` — so a 21-condition lattice silently captured 1 condition on a real experiment.
- **Why it is a workflow gap:** no review lens anywhere in the workflow surface checks that a smoke-scoped variable is gated, so eight implementer / code-review rounds and a smoke-run gate all passed over a one-line scope collapse.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c '_smoke' .claude/agents/code-reviewer.md` → **0**; `grep -c '_smoke' .claude/agents/code-correctness-critic.md` → **0**; `grep -rn '_smoke"' .claude/agents .claude/rules .claude/skills CLAUDE.md | wc -l` → **0** hits across the entire workflow surface — the absence-of-guard evidence; semantic sibling probe `grep -n 'Step 0.69\|Step 0.55' .claude/agents/code-reviewer.md` → L198 (smoke-architecture marker presence) and L943 (phase-idempotency + inter-phase contract, landed `ad3549bc2a` 2026-07-26T02:49:20-07:00 for this same issue #1689), both read in context and neither covering smoke-variable gating; `git log --oneline --since='7 days ago' -- .claude/agents/code-reviewer.md .claude/agents/code-correctness-critic.md` → 3 commits, none landing this check (2026-07-26)

## Evidence

- Session `5c5a89e8`, task #1689, bug present since round 1 and discovered roughly 14 hours into the run at 2026-07-26T20:26:40Z. The dispatcher source: `"local conds_smoke=\"assistant_chat\"\n    local conds=\"$conds_smoke\"  # full path drives via a per-cell loop in production\n    ... local models=\"$models_full\"\n    [ -n \"$SMOKE\" ] && models=\"$models_smoke\""`. The `$SMOKE` guard is present on `models` and absent on `conds`; `run_phase_fit_cells` was hardcoded to one cell the same way.
- The run produced 1 percell JSON and then crashed: `FileNotFoundError: [Errno 2] No such file or directory: 'analysis_tensors/issue_1689/store/Qwen_Qwen2.5-7B/assistant_naturalistic/L14.pt'` — a store path that was never written because the condition was never captured.
- Rounds R1 through R8 of implementer and code review, plus a smoke-run gate, all PASSed over the unconditional assignment.
- Measured cost: roughly 50 minutes of wall clock directly (the R12 relaunch ran 30 minutes producing 1 cell, then R13 implementer plus code review plus relaunch). The larger cost is a science-integrity one: Phase D's whole result would have been a silent 1-of-21 scope collapse had the orchestrator not manually inspected the store tree.
- The pattern is mechanically greppable from the dispatcher diff alone: a `X="$X_smoke"` assignment with no sibling `[ -n "$SMOKE" ]` guard on the same variable. The existing Step 0.69 gate already establishes that dispatcher-shape greps are an accepted review-time instrument in this file.

## Proposed change

- In `.claude/agents/code-reviewer.md`, add a sub-check to the dispatcher review steps (alongside the Step 0.69 phase-idempotency gate at L943, which already greps dispatcher shapes): for every dispatcher file in the diff, run `grep -nE '=\"\$[a-z_]+_smoke\"' <dispatcher>` and, for each hit, require a sibling gate on the SAME variable — a `[ -n "$SMOKE" ] && <var>=` line, a `${SMOKE:+...}` expansion, or an equivalent conditional.
- Treat an ungated `X="$X_smoke"` assignment as a blocker, not a CONCERNS: the failure is silent scope collapse on a real run, with no crash at the assignment site and no smoke-gate coverage.
- Check the converse direction too — a `*_full` variable defined but never assigned to the live variable on any path is the same defect seen from the other side.
- Give the check a stable blocker tag consistent with the existing Step 0.5 / 0.55 / 0.6 contract-tag vocabulary, so the Step 5c-bis strip classifies it correctly; this is a SUBSTANTIVE finding and must not be strippable.
- Record an explicit N/A verdict line when the diff carries no dispatcher, matching the Step 0.69 convention (`Step 0.69: N/A — diff carries no multi-phase dispatcher`).
- Mirror the check into `.claude/agents/code-correctness-critic.md`, the v2 successor lens, so `workflow: v2` tasks are covered identically.
- Add the pattern to the smoke-gate material in `.claude/rules/gotchas.md` so the implementer side sees it before the reviewer does.

## Scope / surfaces

- Primary target: `.claude/agents/code-reviewer.md`
- `.claude/agents/code-correctness-critic.md` (v2 mirror)
- `.claude/rules/gotchas.md` (implementer-side smoke-gate entry)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: b1706a8c8135

- workflow_fix_target: .claude/agents/code-reviewer.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: A-P1.
