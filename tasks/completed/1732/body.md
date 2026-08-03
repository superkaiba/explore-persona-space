---
title: 'daily-fix: settle whether consistency-checker runs for paren'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fcd200fd6757
- daily-auto-filed
created_at: '2026-07-27T07:21:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): two parentless kind:infra
  tasks on the same day got opposite treatment — one spawned the consistency-checker,
  the other recorded it as skipped — and no rule governs which is correct'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 1 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

State explicitly whether the `consistency-checker` runs for a parentless `kind: infra`
task, and — if it is skipped — that the skip is recorded with that reason.

## Workflow gap

- **Bug observed:** #1697 (`kind: infra`, no parent) spawned the consistency-checker, which
  returned PASS while recording "Parent experiment(s): N/A"; #1711 (`kind: infra`, no parent,
  same day) skipped it with the note "Consistency-checker also skipped (kind:infra, no parent
  experiment)" — the same task shape given two different treatments in the plan-review gate.
- **Why it is a workflow gap:** no rule states which treatment is correct. The one "no parent"
  clause that exists is experiment-shaped and gives an answer that cannot apply to an infra
  task, so each session decides for itself and the coverage of the gate is unpredictable.
- **Confidence (emitter):** high
- verified-at-filing: task state read at compose time via
  `uv run python scripts/task.py view <N> --json | jq` — **#1697**: `kind: infra`, no parent,
  `epm:consistency` markers **1**; **#1711**: `kind: infra`, no parent, `epm:consistency`
  markers **0**. Absence greps, per target:
  `grep -niE 'kind: ?infra|no parent|parentless|skip' .claude/agents/consistency-checker.md`
  → **1 hit (L334)**, and its context is experiment-shaped, not infra (quoted below);
  `grep -n 'consistency-checker' .claude/skills/adversarial-planner/SKILL.md` → **9 hits**
  (L465, L472, L502, L668, L672, L804, L807, L921, L949) — **none conditions the spawn on
  `kind` or on the presence of a parent**;
  `sed -n '1704,1745p' .claude/skills/issue/SKILL.md | grep -niE 'kind|parent'` (Step 2b) →
  **3 hits, every one naming the parent recipe/task as an INPUT or as a check row**
  ("needs only the drafted plan + the parent recipe"; "Related tasks (cited in the plan's
  prior work, parent task, …)"; "| Single variable change from parent | BLOCK: …") —
  **none a precondition on `kind` or on the presence of a parent**.
  Landed-fix check:
  `git log --oneline --since='7 days ago' -- .claude/agents/consistency-checker.md` → **0
  commits**. (2026-07-26)

**Context binding.** The single existing no-parent clause,
`.claude/agents/consistency-checker.md:334`, reads: "If the experiment has no parent (first in
a new direction), check against the project's standard baseline (Qwen-2.5-7B, standard eval
suite)." That instruction is unexecutable for a workflow-surface infra task — there is no
model, no eval suite and no recipe to compare — which is why #1697's checker fell back to
recording "N/A" for every row instead. The clause exists; it does not cover this case, so the
change extends it rather than duplicating it.

## Evidence

- #1697, session `7df6ce4c`, 2026-07-26T09:53:06Z: the checker ran and its `epm:consistency`
  note reads `"## Consistency Check: #1697 vs related experiments\n\n**Verdict: PASS**\n\n###
  Parent experiment(s): N/A — \`kind: infra\` workflow-fix task, no experimental parent.
  Related: #1682, #1675 …"`.
- #1711, session `67cf175e`, 2026-07-26T14:39:13Z: the checker was skipped —
  `"[codex-quota-outage] codex composers skipped — quota sentinel live until
  2026-08-06T13:26:00Z (#1204 pre-spawn check); single-Claude per no-show fallback for all 3
  Phase-2 lens critics. Consistency-checker also skipped (kind:infra, no parent experiment)."`
  No `epm:consistency` marker exists on the task (0, confirmed above).
- Measured cost: none directly. The cost is a coverage inconsistency in a plan-review gate,
  and — on the skip path — a missing `epm:consistency` marker, which makes "was this gate run?"
  unanswerable from the task record alone.

## Proposed change

- `.claude/agents/consistency-checker.md` — extend the no-parent clause (L334) with the
  parentless non-experiment case: state whether a `kind: infra` task with no experimental
  parent is in scope, and what the checker compares against when it is (the plausible answer,
  for the plan to confirm or reject: the sibling/prior tasks that touched the same target file,
  which is precisely what #1697's checker did on its own initiative when it listed #1682,
  #1675, #1646, #1634, #865).
- `.claude/skills/adversarial-planner/SKILL.md` Phase 2 spawn list (the consistency-checker row
  at L921 and the spawn snippet at L804-807) — record the same precondition so the spawn
  decision is not re-derived per session.
- `.claude/skills/issue/SKILL.md` Step 2b — mirror the precondition in one line where the step
  describes the checker's inputs.
- Whichever way the decision goes, require the SKIP to be recorded: an `epm:consistency` marker
  (or an `epm:progress` note) naming the reason, so a reader of the task record can tell a
  deliberate skip from a gate that silently did not run. #1711's note already has the right
  shape — the change makes it obligatory rather than incidental.
- Prefer ONE decision over a per-session judgement call; the value of this fix is uniformity,
  and either answer is defensible as long as it is written down.

## Scope / surfaces

- Primary target: `.claude/agents/consistency-checker.md`
- `.claude/skills/adversarial-planner/SKILL.md` (Phase 2 spawn list + spawn snippet)
- `.claude/skills/issue/SKILL.md` (Step 2b)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: fcd200fd6757

- workflow_fix_target: .claude/agents/consistency-checker.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: G-P9.
