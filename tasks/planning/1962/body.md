---
title: 'daily-fix: cross-issue lint-gate concurrency arbitration'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fc0f439c6364
- daily-auto-filed
created_at: '2026-08-01T07:04:08Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): Concurrent Step 9c/10d
  gate pytest legs on the shared VM kill each other (rc 137/143/144 in 3 sessions;
  #1768 merge deferred ~10 h) — the single-flight probe is exact-issue-scoped, nothing
  arbitrates gate concurrency across issues'
workflow: v1
---
# daily-fix: cross-issue lint-gate concurrency arbitration

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED H1; miner-7:P3/P4, miner-2:P16, miner-3:P20). Source sessions: ba1e3c42 (#1768 — gate contract exhausted after 3 attempts, merge deferred ~10 h behind `epm:merge-failed`; the deferral let main drift 1,784 commits, forcing a rebase-onto + fresh PR), 26e6f6f8 (#1865 — TG_RC=137 / TG_BASE_RC=143 at ~93% VM CPU), 2652deef (#1868 — gate bg shell exit 144 at ~98% completion with 6359 PASSED / 0 FAILED and no verdict artifacts, forcing a full ~40 min re-run).

## Goal

Add fleet-level arbitration for concurrent Step 9c/10d lint-gate pytest legs on the shared VM so sibling issues' gates queue instead of dying under mutual load.

## Workflow gap

- **Bug observed:** Three sessions in one day had gate pytest legs signal-killed (rc 137/143/144) while concurrent Step 9c/10d gates from SIBLING issues ran on the shared VM; each kill cost a full ~30-40 min gate re-run, and the #1768 case deferred a merge ~10 h. `unverified hypothesis — verify at plan time:` the kill mechanism is shared-VM contention from concurrent gate trees (in-session probing found no earlyoom journal entry in the #1768 case; 13 sibling `lint-gate-tree` processes were live at the kill — proc count probed live by the session, contention attribution is inference).
- **Why it is a workflow gap:** The Step 9c/10d single-flight machinery is deliberately per-issue only — `scripts/step9c_baseline.py probe --pattern 'issue-<N>-lint-gate-tree'` is documented in SKILL.md as "exact-issue-scoped", and the SKILL text itself acknowledges "30-40 min under typical fleet load (3+ concurrent gates)" — so nothing arbitrates or bounds gate concurrency ACROSS issues; N sessions can launch N full gate trees simultaneously and kill each other.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'lint-gate-tree\|flock\|concurren' scripts/step9c_baseline.py` → 8 hits, all baseline-refresh single-flight (`.claude/cache/step9c-baseline.lock`, L143/L1110-1118) or per-issue probe exclusion — 0 cross-issue gate arbitration; `grep -n 'lint-gate-tree' .claude/skills/issue/SKILL.md` → 3 hits (L11234, L11313, L13069), context read: the single-flight probe block (#1606) states the pattern "is exact-issue-scoped" and the same section documents 3+ concurrent gates as expected load with no queueing. `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md scripts/step9c_baseline.py` eyeballed (4cbb5c47fc … 0595d18b5f): no landed cross-issue arbitration fix (2026-08-01).

## Proposed change (candidate diff sketch — refine in planning)

```
scripts/step9c_baseline.py:
+ probe --fleet mode: count live gate trees matching '*-lint-gate-tree'
+ (ANY issue, self-/ancestor-excluding as in #1606/#1821), exit 3 when
+ >= EPM_GATE_FLEET_MAX (default ~2) are live.
.claude/skills/issue/SKILL.md Step 9c (and the Step 10d gate recipe):
+ before launching the gate bg call, run the fleet probe; on exit 3,
+ bounded wait (poll until a slot frees, cap ~45 min) instead of
+ launching a third concurrent gate tree.
+ secondarily: state TG_T sizing off the LOADED 30-40 min range (already
+ documented) rather than the idle one wherever a default is given.
```

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 9c + Step 10d gate launch recipes); secondary: `scripts/step9c_baseline.py` (probe extension).
- Grep the workflow surface for the pattern before editing (`grep -rn 'lint-gate-tree' .claude/ scripts/`) and update every launch-site recipe (Step 9c, Step 10d, the surgical form) consistently.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The per-issue single-flight contract (#1606) and self-pid exclusion (#1821) must be preserved unchanged; the fleet arm is additive.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates.

## Provenance

- sha-verify (filing-time, #1467): `ba1e3c42` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `26e6f6f8` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `2652deef` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: fc0f439c6364

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: (driver-computed; tag authoritative)
