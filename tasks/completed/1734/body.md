---
title: 'daily-fix: state the minimum Step-2 plan-review floor for ki'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cbb2416258c6
- daily-auto-filed
created_at: '2026-07-27T07:21:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): three same-class infra
  tasks got three different levels of Step-2 rigor on one day, one skipping the planner,
  verify_plan, the fact-checker and the entire critic ensemble, while a sibling''s
  single critic returned REVISE with two Must-Fix items'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 3 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

State in `/issue` Step 2 the minimum plan-review floor that binds even for a small
`kind: infra` task, and require any skip below the full stack — and any carried
`verify_plan.py` WARN — to be recorded with its reason.

## Workflow gap

- **Bug observed:** three `kind: infra` daily-fix tasks got three different Step-2 depths on
  2026-07-26 — #1696 ran the full stack (planner skill, `verify_plan.py` + `epm:plan-verify`,
  fact-checker, 3 critics); #1692 spawned the planner and one critic directly, bypassing the
  skill dispatcher, and never ran `verify_plan.py`; #1709 spawned nothing at all, the
  orchestrator authoring the plan itself and self-certifying the skip under the CLAUDE.md
  bug-fix carve-out.
- **Why it is a workflow gap:** `.claude/skills/issue/SKILL.md` Step 2 states only "Invoke the
  `adversarial-planner` skill" and names no floor, so the CLAUDE.md carve-out ("re-runs …,
  monitoring, syncing, bug fixes, or explicit override skip it") is self-applied at whatever
  depth each session judges proportionate, with no record of what was skipped or why.
- **Confidence (emitter):** high
- verified-at-filing: marker state read at compose time via
  `uv run python scripts/task.py view <N> --json | jq` — **#1696**: `kind: infra`,
  `epm:plan-verify` markers **1**; **#1692**: `kind: infra`, `epm:plan-verify` markers **0**;
  **#1709**: `kind: infra`, `epm:plan-verify` markers **0**. Absence greps, per target:
  `grep -c 'epm:plan-verify\|verify_plan' .claude/skills/issue/SKILL.md` → **0** (the file
  that owns Step 2 never names the mechanical pre-pass);
  `grep -n 'Phase 1.5' .claude/skills/issue/SKILL.md` → **0 hits** (see the corrected target
  below). Landed-fix check:
  `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` → 6 commits, none
  adding a Step-2 review floor. (2026-07-26)

**Context binding — one target corrected.** The related WARN-disposition item was mined
against `.claude/skills/issue/SKILL.md` "§ Phase 1.5.0". That section does not exist in that
file (0 hits); Phase 1.5.0 lives in `.claude/skills/adversarial-planner/SKILL.md` (L220), and
its current rule is the direct cause of the observed behaviour: "**PASS (with WARNs) →
proceed** … copy any OTHER WARN line verbatim into the fact-checker brief (and later the
critic briefs) as 'mechanical pre-pass notes'" (L369-373) — carrying a WARN forward is
already compliant, and no reason is owed. Only `c23_goal_currency` bounces today. The
WARN-disposition clause therefore belongs in `adversarial-planner/SKILL.md` § Phase 1.5.0,
while the Step-2 floor belongs in `issue/SKILL.md`.

## Evidence

- #1692 (`kind: infra`), session `a5a4b7bd`, 2026-07-26T07:09:28Z, verbatim:
  `"Since this is a small workflow-fix task (0 GPU-h, spec changes to 2-3 markdown files),
  I'll spawn the planner directly rather than through the /adversarial-planner skill
  dispatcher"`. No `epm:plan-verify` marker exists on the task (0, confirmed above).
- #1692's single spawned critic returned REVISE. Its `epm:plan` note, read at compose time,
  records: `"**Phase 2 Critic (Claude) — REVISE** (verdict at /tmp/issue-1692-critic-v1.md).
  Two Must-Fix findings addressed in this revised plan"` — the first being that a bare
  `python -c 'import <mod>'` fires only module-level imports while the #1689 shape was a
  function-body import. A #1709-shaped path would have shipped both findings unreviewed.
- #1709 (`kind: infra`), session `e3b70618`, 2026-07-26T13:59:12Z, `epm:plan` note verbatim:
  `"Bug-fix category (CLAUDE.md /adversarial-planner carve-out) — direct plan drafted for a
  1-line SPECS widen + a 1-line pin-test update; no critic ensemble needed for a data-widen
  with pre-existing coverage."` Zero agents were spawned; the plan was authored with the
  `Write` tool.
- Related, WARN disposition — session `6b3fca14`, 2026-07-26T07:20:55Z: `verify_plan.py`
  returned two WARNs, one being
  `"[c34_ratchet_headroom] verbatim insert fits size-ratchet headroom: ...
  .claude/agents/code-reviewer.md: insert ~7118 B > headroom 82 B ... (#1230: a paragraph
  larger than code-reviewer.md's headroom forced an un-planned third-file cap-raise
  deviation)"`. The orchestrator wrote `"Verifier PASSes with 2 WARNs (both benign — carry
  into critic briefs)"`. The substance was fixed in plan v2 (clean at `n_warn=0`), so nothing
  shipped wrong; the same ratchet pressure the WARN named then produced a
  `scripts/workflow_lint.py` merge conflict in that session at 09:38→09:50Z, roughly 12 min of
  merge mechanics.
- Measured cost: no direct time lost to the Step-2 variance itself. #1709's unreviewed change
  is the one that later collided at the Step-10d spec-freshness gate, and #1692's ungated plan
  carried 2 real defects into review.

## Proposed change

- `.claude/skills/issue/SKILL.md` Step 2 — state the MINIMUM plan-review floor that binds for
  a `kind: infra` workflow-surface edit, and state explicitly that the CLAUDE.md "bug fixes"
  carve-out does not reach it. Proposed floor (the planner should confirm each leg is worth its
  cost): persist the plan via `new-plan-version`; run `verify_plan.py` and post
  `epm:plan-verify` (mechanical, seconds, no agent spawn); spawn at minimum ONE `critic`.
- Same section — require that any leg skipped BELOW the full stack is recorded in the
  `epm:plan` note with the reason, in the shape #1709 already used, so the skip is auditable
  rather than invisible.
- `.claude/skills/adversarial-planner/SKILL.md` § Phase 1.5.0 (L369-373) — replace "copy any
  OTHER WARN line verbatim into the fact-checker brief" with a per-WARN DISPOSITION: each WARN
  is either resolved in the next plan revision, or carried with a one-line reason naming why it
  cannot bite. Forbid the bare word "benign" as that reason. Note in the clause that
  `c34_ratchet_headroom` is never benign — it is the deterministic predictor of a same-file cap
  collision.
- Consider, and record the decision either way: whether the floor is better enforced
  mechanically as a `verify_plan.py` check that FAILs a `wf-fix`-tagged plan with no
  `epm:plan-verify` marker, rather than as SKILL.md prose a session can self-certify past.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- `.claude/skills/adversarial-planner/SKILL.md` (§ Phase 1.5.0, WARN disposition)
- `scripts/verify_plan.py` (only if the plan elects the mechanical-enforcement option)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: cbb2416258c6

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: J-P3, D-P14, C-P16.
