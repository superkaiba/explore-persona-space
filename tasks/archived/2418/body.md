---
title: 'Infra: worktree plans/plan.md symlink resolves to a STALE plan — briefs following
  the plan-handoff convention can review against a superseded plan'
kind: infra
tags: []
created_at: '2026-08-20T06:00:34Z'
has_clean_result: false
origin_prompt: 'surfaced in #2329 q35_ladder_decay Step-4 review round 1: the round
  worktree''s plans/plan.md resolved to v4 (pre-critique) while the approved plan
  was v8 on main, and its planned_manifest.json was half the size; reviewers detected
  it themselves but nothing in the pipeline catches it'
workflow: v1
---
---
kind: infra
---

# Infra: a worktree's `plans/plan.md` symlink silently resolves to a STALE plan, so briefs following the documented plan-handoff convention can review against a superseded plan

## Goal

Close a silent-wrong-review hazard in the plan-handoff convention.

**The mechanism.** `CLAUDE.md` § Code Style pins the plan-handoff convention: "pass the PATH to
the plan, never the body" — and every orchestrator brief accordingly names
`tasks/<status>/<N>/plans/plan.md`, the symlink to the highest plan version. That path is
CORRECT at the repo root and STALE inside any worktree cut before the plan was revised: the
worktree holds a frozen checkout of the task folder at its base commit, so `plan.md` points at
whatever version existed then, and later versions are absent from the tree entirely. Same for
`artifacts/planned_manifest.json`.

Nothing in the pipeline currently catches this. The Step 5a spec-freshness sync
(`.claude/skills/issue/steps/09-step-5.md`) exists for exactly this class of staleness but is
scoped to the WORKFLOW SURFACE (`.claude/**` agent specs and skills) — it does not touch the task
folder. So a reviewer or implementer whose cwd is the worktree reads a stale plan with no error,
no warning, and no version mismatch surfaced. The failure is silent and the verdict looks normal.

**Measured instance (task #2329, follow-up round `q35_ladder_decay`, Step-4 review round 1,
2026-08-20).** Worktree `.claude/worktrees/issue-2329-q35-ladder-decay`, cut from a base commit
~3h before plan v5 was written:

| location | `plans/plan.md` → | versions present | manifest bytes |
|---|---|---|---|
| repo root (`main`) | `v8.md` | v1–v8 | 16,175 |
| the round's worktree | `v4.md` | v1–v4 | 8,051 |

v4 is the PRE-CRITIQUE draft. v4 → v8 spans three post-approval critique rounds and changed
exactly what reviewers grade against: the registered verdict lattice and its decision predicates
(the negative branch was tightened to require a paired scale-normalized companion), the judge-wave
call-count declaration (a stale ~7.5k figure re-derived to ~4,720), the R2-M1 donor-geometry and
donor-derivation prescriptions, and the manifest's own contents (it gained a 4-label Leg B
lattice). A plan-adherence verdict graded against v4 measures adherence to a superseded document
and is worthless; a statistics verdict graded against v4's looser predicate can PASS code that
violates the registered one.

**Why this is worth fixing even though this round survived it.** Every #2329 reviewer that needed
the plan detected the stale symlink on its own and re-read v8 from the main checkout — so the
round was not corrupted. But that is agent diligence compensating for a briefing defect, not a
control. It is not reproducible: the same brief with a less careful reviewer, or a Codex twin
resolving a relative path inside the worktree sandbox, yields a confidently-wrong verdict with no
signal anywhere in the marker trail. The orchestrator only learned of it because one sub-reviewer
volunteered a "Note for the orchestrator" line.

## Fix directions (pick after reading the surfaces; do not implement blind)

1. **Make the brief-composition convention worktree-safe.** Wherever the orchestrator names a plan
   or manifest path for a subagent, resolve it against the MAIN checkout absolutely (the existing
   worktree-safe idiom is already used for workflow helpers: resolve from
   `"$REPO_ROOT"/scripts/...`, never the worktree copy). Candidate surfaces: `CLAUDE.md` § Code
   Style plan-handoff bullet, `.claude/skills/issue/steps/09-step-5.md` (Step 5a), and the
   equivalent v2 dispatch text in `.claude/skills/issue-v2/SKILL.md` Step 4.
2. **Extend the Step 5a freshness sync to the task folder** — surgically refresh
   `tasks/<status>/<N>/plans/` + `artifacts/` from fetched `origin/main` in the worktree, mirroring
   the existing branch-side-edit guard so a legitimately branch-local plan revision is not
   clobbered.
3. **Add a cheap fail-loud version assertion** available to any agent handed a plan path: compare
   the resolved `plan.md` target against the highest `plans/v*.md` on `origin/main` for that task
   and raise on mismatch. Cheapest of the three and catches the whole class rather than the
   surfaces enumerated in (1).

(3) plus (1) is probably the right combination — (1) prevents it, (3) catches it when a new brief
site is added later and forgets. Decide and record.

## Acceptance criteria

1. A subagent dispatched with a plan path from a session whose cwd is a stale worktree either reads
   the CURRENT plan or fails loud — demonstrated with a reproduction (a worktree cut before a plan
   revision, exactly the #2329 shape).
2. The chosen fix covers the MANIFEST as well as the plan; the #2329 instance had both stale, and a
   manifest-completeness verdict graded against a stale manifest fails the same way.
3. No new red in the no-flags `workflow_lint.py` run or the mapped-test selection; if a lint check
   is added, it has a test that fails before the fix.
4. If the fix touches brief-composition prose rather than code, at least one mechanical guard backs
   it — prose alone did not prevent this instance.

## Provenance

workflow_fix_target: CLAUDE.md § Code Style (plan-handoff convention); .claude/skills/issue/steps/09-step-5.md (Step 5a spec-freshness sync scope); .claude/skills/issue-v2/SKILL.md Step 4

Surfaced during task #2329 follow-up round `q35_ladder_decay`, Step-4 review round 1
(2026-08-20). Detected by the `code-reviewer-lean` g6 sub-review, which flagged it to the
orchestrator; independently detected and worked around by the `efficiency-critic` and the g5
sub-review. The orchestrator re-verified both the symlink targets and the byte counts in the table
above with `ls -la` in each location before filing, then sent in-flight corrections to the seven
reviewers still running. Filed under the workflow-fix-on-bug protocol: the defect is in the
workflow surface (the plan-handoff convention + the Step 5a freshness sync's scope), not in any
experiment payload.
