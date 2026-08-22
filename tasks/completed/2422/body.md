---
title: Briefs pointing a worktree-based reviewer at task state must pin the absolute
  main-checkout path (worktree plans/plan.md resolves to a stale version)
kind: infra
tags:
- wf-fix
created_at: '2026-08-20T07:58:38Z'
has_clean_result: false
parent_id: 823
origin_prompt: 'Surfaced during /issue 823 P-Gen v13 amendment review dispatch: the
  hand-written code-reviewer brief named a relative tasks/.../plans/plan.md while
  directing the reviewer to a worktree, where that symlink resolves to v10 (pre-amendment)
  instead of v13. The codex-code-reviewer composer independently caught and handled
  the same hazard, so the discipline exists on one twin''s surface but not the other''s.'
workflow: v1
---
# Infra: a worktree's frozen `tasks/` tree makes brief-named plan / manifest paths resolve to a STALE version — reviewers grade against a superseded document

> **Merged task.** This body absorbs task **#2418** (filed 2026-08-20 06:00Z from
> incident #2329) into #2422 (filed 07:58Z from incident #823). Same defect,
> same target surfaces, two independent incidents two hours apart. #2422 is the
> canonical executing task; #2418 is archived pointing here. Nothing from
> #2418 is dropped — its mechanism write-up, its measured table, its three fix
> directions, and its acceptance criteria are folded in below and marked
> `[from #2418]`.

## Goal

Close a silent-wrong-review hazard in the plan-handoff convention: a brief that
names task state (plan, manifest, body, events) by a path that RESOLVES INSIDE a
worktree hands the subagent a version frozen at the worktree's base commit. The
reviewer then grades the diff against the plan the diff SUPERSEDES, producing
confidently-wrong plan-deviation and manifest-completeness blockers with no
error, no warning, and no signal anywhere in the marker trail.

## The mechanism  `[from #2418]`

`CLAUDE.md` § Code Style pins the plan-handoff convention: "pass the PATH to the
plan, never the body" — and orchestrator briefs accordingly name
`tasks/<status>/<N>/plans/plan.md`, the symlink to the highest plan version.
That path is CORRECT at the repo root and STALE inside any worktree cut before
the plan was revised: the worktree holds a frozen checkout of the task folder at
its base commit, so `plan.md` points at whatever version existed then, and later
versions are absent from the tree entirely. The same holds for
`artifacts/planned_manifest.json`, `body.md`, and `events.jsonl`.

Nothing in the pipeline currently catches this. The Step 5a spec-freshness sync
(`.claude/skills/issue/steps/09-step-5.md`) exists for exactly this staleness
class but is scoped to the WORKFLOW SURFACE (`.claude/**` agent specs and
skills) — it does not touch the task folder. A reviewer or implementer whose cwd
is the worktree reads a stale plan with no diagnostic, and the verdict looks
normal.

The worktree's `tasks/` tree being frozen at its branch point is CORRECT
behavior. The defect is briefs that read it as if it were live.

## Incident A — #2329, `q35_ladder_decay` Step-4 review round 1  `[from #2418]`

Worktree `.claude/worktrees/issue-2329-q35-ladder-decay`, cut from a base commit
~3h before plan v5 was written. Both symlink targets and byte counts re-verified
with `ls -la` in each location before filing:

| location | `plans/plan.md` → | versions present | manifest bytes |
|---|---|---|---|
| repo root (`main`) | `v8.md` | v1–v8 | 16,175 |
| the round's worktree | `v4.md` | v1–v4 | 8,051 |

v4 is the PRE-CRITIQUE draft. v4 → v8 spans three post-approval critique rounds
and changed exactly what reviewers grade against: the registered verdict lattice
and its decision predicates (the negative branch tightened to require a paired
scale-normalized companion), the judge-wave call-count declaration (a stale
~7.5k figure re-derived to ~4,720), the R2-M1 donor-geometry and donor-derivation
prescriptions, and the manifest's own contents (it gained a 4-label Leg B
lattice). A plan-adherence verdict graded against v4 measures adherence to a
superseded document and is worthless; a statistics verdict graded against v4's
looser predicate can PASS code that violates the registered one.

Every #2329 reviewer that needed the plan detected the stale symlink on its own
and re-read v8 from the main checkout, so the round was not corrupted. That is
agent diligence compensating for a briefing defect, not a control. The
orchestrator learned of it only because one sub-reviewer volunteered a "Note for
the orchestrator" line.

## Incident B — #823, P-Gen v13 amendment round, caught pre-verdict

The orchestrator hand-wrote a `code-reviewer` brief naming the plan as
`tasks/followups_running/823/plans/plan.md` and, separately, told the reviewer to
read the diff in the worktree `.claude/worktrees/issue-823-ladder`. Verified
resolution of that same relative path:

- worktree `.claude/worktrees/issue-823-ladder/tasks/followups_running/823/plans/plan.md`
  → `v10.md` (title line "Plan v10")
- main checkout `/home/thomasjiralerspong/explore-persona-space/tasks/followups_running/823/plans/plan.md`
  → `v13.md` (title line "Plan v13")

v10 is exactly the version the round's amendment supersedes. It registers the OLD
generation caps (1024/2048) and contains NO generation-config fingerprint, NO
metadata-free hash-basis pin, and NO P0 prompt-integrity gate. A v10-based review
of the v13 diff reads the 4096/8192 caps as an unauthorized deviation and the
fingerprint/P0 work as unplanned scope creep — four false blockers, all plan-lens,
all requiring a fix round to refute.

**The asymmetry that makes this mechanizable.** The `codex-code-reviewer`
composer caught this at compose time on its own and handled it
(`.claude/agents/codex-code-reviewer.md` § Step 2-pre-b): it diffs the worktree
plan against the canonical main copy, references a byte-identical `/tmp` copy as
primary with the main-checkout ABSOLUTE path as fallback, BARS all worktree
`tasks/` reads, and supplies sed line-window anchors. The Claude-side brief,
hand-written by the orchestrator with no equivalent discipline, did not. Same
round, same plan, same worktree; one twin was structurally protected and the
other was not.

Recovery was a `SendMessage` correction to the live reviewer naming both
resolutions. That worked only because the Codex composer surfaced the hazard
while the Claude reviewer was still running — pure timing, not a control.

## The two identified defect sites (Claude side)

1. **Brief-composition side** — `.claude/skills/issue/steps/09-step-5.md`
   ("Both reviewers see the same brief"): the shared brief names "the approved
   plan (via the `plans/plan.md` symlink)" while the Claude reviewer's
   additional fields are `worktree` path + `base` ref. That composes exactly
   into the defect. The Codex twin's adjacent bullet already documents the
   absent/stale inlining; the Claude bullet has no counterpart.
2. **Reviewer side** — `.claude/agents/code-reviewer.md` § Context budget:
   "plans via `Read` on `tasks/<status>/<N>/plans/v<K>.md` (or the path in your
   brief), sliced" — a relative path resolved from the reviewer's cwd, which on
   every `/issue` round IS the worktree.

## Why the existing rule does not cover it

`CLAUDE.md` already carries "Never form `tasks/...` paths relative to cwd or
`__file__` — from a worktree that path is stale", enforced by
`tests/test_no_direct_task_path_construction.py`. That rule and its test are
scoped to CODE constructing task paths (AST scan for bare-name imports and
`ROOT / "tasks"` constructions). This channel is PROSE: a brief handing a
relative task path to a subagent whose working directory is a worktree. The lint
cannot see it and the rule's wording does not reach it.

## Fix directions (pick after reading the surfaces; do not implement blind)  `[from #2418]`

1. **Make the brief-composition convention worktree-safe.** Wherever the
   orchestrator names a plan / manifest / body / events path for a subagent,
   resolve it against the MAIN checkout absolutely (the existing worktree-safe
   idiom is already used for workflow helpers: `"$REPO_ROOT"/scripts/...`, never
   the worktree copy) — or reference a verified-identical copy outside both
   trees, the `codex-code-reviewer` § Step 2-pre-b shape. Resolve the intended
   version at compose time (`readlink`) and STATE the version number in the
   brief so a mismatch is visible to the reader. Explicitly bar reading `tasks/`
   from inside the worktree. Candidate surfaces: `CLAUDE.md` § Code Style
   plan-handoff bullet, `.claude/skills/issue/steps/09-step-5.md` (the shared
   brief + Step 5a), `.claude/agents/code-reviewer.md` § Context budget, the
   implementer specs, and the equivalent v2 dispatch text in
   `.claude/skills/issue-v2/SKILL.md` Step 4.
2. **Extend the Step 5a freshness sync to the task folder** — surgically refresh
   `tasks/<status>/<N>/plans/` + `artifacts/` from fetched `origin/main` in the
   worktree, mirroring the existing branch-side-edit guard so a legitimately
   branch-local plan revision is not clobbered.
3. **Add a cheap fail-loud version assertion** available to any agent handed a
   plan path: compare the resolved `plan.md` target against the highest
   `plans/v*.md` on `origin/main` for that task and raise on mismatch. Cheapest
   of the three and catches the whole class rather than the surfaces enumerated
   in (1).

(3) plus (1) is probably the right combination — (1) prevents it, (3) catches it
when a new brief site is added later and forgets. Decide and record the choice.

Do NOT change `plan_patch.py`, `task.py`, the `task_workflow` resolver, or any
worktree mechanics: the worktree's frozen `tasks/` tree is correct behavior.

**Mechanical-guard idiom.** The project's established pattern for pinning a
prose contract across multiple surfaces is the region-anchored per-surface token
ladder in `scripts/workflow_lint.py` (`check_smoke_blind_spot_review_lens`,
`check_pre_split_review_guard`, `check_null_gate_calibration_lens`,
`check_two_tier_yield_floor`, `check_cvd_scoped_gpu_verdict_lens` — #2165,
#2158, #2144, #2242, #2120), bundled into the no-flags default run. Prefer that
idiom over a free-text grep for `tasks/<status>/<N>/` near a worktree path,
which is false-positive-prone across brief templates.

## Acceptance criteria

1. A subagent dispatched with a plan / manifest path from a session whose cwd is
   a stale worktree either reads the CURRENT version or fails loud —
   demonstrated with a reproduction (a worktree cut before a plan revision:
   exactly the #2329 and #823 shapes). `[from #2418 #1]`
2. The chosen fix covers the MANIFEST as well as the plan; both #2329 artifacts
   were stale, and a manifest-completeness verdict graded against a stale
   manifest fails identically. `[from #2418 #2]`
3. A brief-contract line exists on the Claude-side reviewer/implementer brief
   surface requiring absolute main-checkout task paths (or a verified-identical
   out-of-tree copy) + an explicit worktree-`tasks/` read bar, with compose-time
   version resolution stated in the brief.
4. At least one MECHANICAL guard backs the prose — prose alone did not prevent
   either instance. If a lint check is added it has a test that fails before the
   fix. `[from #2418 #4 + #3]`
5. No change to worktree `tasks/` semantics; no new `tasks/` path construction in
   code.
6. `uv run python scripts/workflow_lint.py` (no flags) clean and the mapped
   `select_step9c_tests.py --map-files` selection passes.

## Folded-in sibling: the implementer brief's missing WAIT MECHANISM

The same #823 round carried a second brief-composition omission of the same
class. The implementer brief stated the prohibition ("do not end your turn with
work staged-but-uncommitted") but not the WAIT MECHANISM the CLAUDE.md
§ "Teammate coordination" (g) duty prescribes (a bounded in-turn `Monitor`
until-loop; foreground `sleep` chains are hook-blocked). The implementer then
invented a background-watcher shape and orphaned its own landing — ~62 min of
work left uncommitted with a lint gate running unobserved; the orchestrator
recovered it.

#2422's original filing recorded this as evidence only and invited a merge ("If a
reviewer of this task judges the two worth one combined brief-contract
hardening, that is a reasonable merge"). Taking that invitation: it is IN SCOPE
here as a second one-line brief-contract addition, because it is the same
surface and the same defect class — a hand-written brief omitting a discipline
that already exists elsewhere on the surface. Descope it if the plan shows the
two additions interfering.

## Provenance

workflow_fix_target: CLAUDE.md § Code Style (plan-handoff convention); .claude/skills/issue/steps/09-step-5.md (the shared reviewer brief + Step 5a spec-freshness sync scope); .claude/agents/code-reviewer.md § Context budget (plan-read path); .claude/skills/issue-v2/SKILL.md Step 4

Two independent incidents, 2026-08-20, ~2h apart:

- **#2418 (06:00Z, incident #2329)** — surfaced during the `q35_ladder_decay`
  Step-4 review round 1. Detected by the `code-reviewer-lean` g6 sub-review,
  which flagged it to the orchestrator; independently detected and worked around
  by the `efficiency-critic` and the g5 sub-review. The orchestrator re-verified
  the symlink targets and byte counts, then sent in-flight corrections to the
  seven reviewers still running.
- **#2422 (07:58Z, incident #823, parent task)** — surfaced during the P-Gen v13
  amendment review dispatch. Caught pre-verdict because the
  `codex-code-reviewer` composer independently handled the same hazard, exposing
  the twin asymmetry.

Both filed under the workflow-fix-on-bug protocol: the defect is in the
workflow surface (the plan-handoff convention, the Step 5a freshness sync's
scope, and the Claude-side brief contract), not in any experiment payload.
Merged into #2422 by the `/issue 2422` clarifier context-gathering pass
(2026-08-20), which found #2418 as a same-fingerprint duplicate on the same
target files; #2422 was chosen as canonical because it carried the live session.
