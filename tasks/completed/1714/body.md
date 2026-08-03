---
title: 'daily-fix: spec-freshness sync manufactures self-inconsisten'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8a12cccef12a
- daily-auto-filed
created_at: '2026-07-27T07:13:59Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): the Step 5a/10d spec-freshness
  per-path skip omits a branch-edited file but still syncs its derived/partner artifacts
  from main, and the Step 10d pre-gate re-sync snapshots main before a ~30 min lint
  gate so the merge routinely returns CONFLICTING'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 4 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Make the Step 5a / Step 10d spec-freshness sync family-atomic (never sync a derived or
import-partner artifact whose partner was skipped for branch-side edits) and move the
Step 10d pre-gate re-sync out of the race with the ~30-minute lint gate.

## Workflow gap

- **Bug observed:** the per-path "branch-side edits — skipping" guard skips one member of
  a coupled family while syncing the rest from `main`, manufacturing a tree the branch
  never created — a synced `markers.md` against a skipped `workflow.yaml` (lint gate
  `block`), a synced pin test against a skipped `scripts/workflow_lint.py` (pytest
  collection `ImportError`, rc=2) — and, separately, the Step 10d pre-gate re-sync
  snapshots `origin/main` and then ~30 min of lint gate elapses before the merge, so the
  merge routinely returns `CONFLICTING`.
- **Why it is a workflow gap:** the skip grain is PER-ITEM by design (fail-safe against
  the #535 clobber), but the SPECS list contains items that are derived from, or import
  from, other items in the same list — so a per-item skip is not fail-safe at the family
  level, and the sync's placement relative to the gate is fixed by the SKILL recipe.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'spec-freshness\|SAFE_SPECS\|branch-side\|SPECS=' .claude/skills/issue/SKILL.md`
  → **~30 hits in the single named target**, incl. the Step 5a block at L2236-2281
  (`SPECS=` at L2249; the per-item skip at L2259-2275; the sync + commit at L2277-2281) and
  the Step 10d pre-gate re-sync bullet at L10240-10276. Presence claim confirmed per-target.
  Coupling verified: `grep -n 'generated\|emit-tables' .claude/skills/issue/markers.md` →
  L62-64 `"The kinds table is auto-generated from … uv run python scripts/workflow_lint.py
  --emit-tables to regenerate"`; `grep -n 'markers.md' scripts/workflow_lint.py` →
  L751 `_REPO_ROOT / ".claude" / "skills" / "issue" / "markers.md"`. Landed-fix check:
  `git log --oneline --since='7 days ago' -- .claude/skills/issue/SKILL.md` shows no commit
  implementing family atomicity or a post-gate re-sync. Context binding: the paragraph at
  SKILL.md L2332-2337 already ACKNOWLEDGES the symptom ("a workflow test that FAILs inside
  a long-lived issue worktree … including a collection-time ImportError from a
  `workflow_lint` / rules-pin symbol — is worktree-staleness … cross-check at the repo root")
  but prescribes only a diagnosis habit, not family atomicity — the fix is NOT landed.
  (2026-07-26)

## Evidence

- Defect (i), derived-artifact shape. Session `0e2c3b21`, 2026-07-26T11:50:30Z → 12:08:19Z:
  the Step 10d pre-push lint gate returned `block` on a `markers.md` auto-generated-table
  staleness. The branch had edited `.claude/workflow.yaml`, so `workflow.yaml` was skipped
  while `.claude/skills` — which CONTAINS the derived `markers.md` — was synced from
  `origin/main`, importing #1692's regenerated `markers.md` alongside the branch's
  pre-#1692 `workflow.yaml`. Session's own diagnosis: `"The block is stale workflow.yaml:
  my branch predates #1692 (164b46cd5c) which added per-arm-resolution/PASS_PARTIAL prose
  to the epm:smoke-architecture-check marker AND regenerated markers.md. My spec-freshness
  syncs skipped workflow.yaml because I edited it (Edit D)."` Cost: ~18 min (7 min
  diagnosis + 11 min gate re-run), one unplanned merge commit, forced merge-form change to
  `--squash`.
- Defect (i), import-partner shape. Session `2de5253e`, 2026-07-26T14:23:21Z → 14:27:50Z:
  the Step 5a sync pulled in new `tests/test_*` files from `origin/main` but SKIPPED
  `.claude/agents`, `.claude/skills`, and `scripts/workflow_lint.py` for branch-side edits.
  Result: `"Collection ImportError on test_workflow_lint_inline_round_duty_mirror.py: the
  test file was synced from origin/main but the workflow_lint.py function it imports is
  only on newer main (per-file guard skipped syncing workflow_lint.py because the branch
  has its own edit to it). Classic \"stale-family\" from Step 5a."` — `collected 5039
  items / 1 error`, `PYTEST_RC=2`. Cost: one wasted Step 9c gate invocation + ~5 min, and
  it was the direct trigger of that session's first merge conflict.
- Defect (i), compound shape. Session `e3b70618`, 2026-07-26T14:50:07Z → 16:10:28Z, on a
  **one-line** `SPECS=` widen in `SKILL.md`: the pre-gate re-sync committed 3 files from
  `main` onto the branch but skipped `.claude/skills/issue/SKILL.md` (branch-side edits) —
  exactly the file the freshly-synced check pins. Gate output:
  `workflow_lint: check-inline-round-duty-mirror: .claude/skills/issue/SKILL.md has 0
  occurrence(s) of anchor '(1) BEFORE any ridge', expected exactly 1 (part (a) count
  invariant)`, alongside `spec-freshness: .claude/skills carries branch-side feature edits
  — skipping blind sync.` Cost: 3 pre-push lint-gate runs (~34 + ~31 + ~7 ≈ 72 min of gate
  wall-time) plus ~45 min of orchestrator diagnosis for a 1-line diff; `epm:merged` records
  `merge_attempts: 3`.
- Defect (ii), the gate race. Session `67cf175e`: pre-gate re-sync at 15:54:22Z committed
  3 workflow-surface files (156+/153-) from `origin/main` onto `issue-1711`; the lint gate
  then ran ~30 min in the background; during that window `main` advanced further on
  `scripts/workflow_lint.py` and the squash merge was refused at 16:25:49Z —
  `"head-sync pre-check: parity at 0dd49d67571c (mergeable=CONFLICTING) … X Pull request
  superkaiba/explore-persona-space#1476 is not mergeable: the merge commit cannot be
  cleanly created."` and `"CONFLICT (content): Merge conflict in scripts/workflow_lint.py"`.
  Cost: ~47 min of Step 10d wall-clock (15:53 → 16:40), one extra full lint-gate run, one
  residual-conflict subagent.
- The two defects compound: `e3b70618`'s sync ALSO re-added 139 lines of `workflow_lint.py`
  that `origin/main` had removed hours earlier, and the near-miss silent revert was caught
  only by an ad-hoc post-gate `git diff origin/main HEAD -- scripts/workflow_lint.py`.
  The orchestrator dropped the sync commit (`git reset --hard HEAD~1` in the worktree) and
  ran a third gate.

## Proposed change

- **Family atomicity in the Step 5a block** (`.claude/skills/issue/SKILL.md`, the `SPECS=`
  loop at L2249-2281). Declare the coupled families explicitly and make the skip
  transitive: `.claude/workflow.yaml` ↔ its derived `.claude/skills/issue/markers.md` and
  the SKILL.md generated-table block; `scripts/workflow_lint.py` ↔
  `:(glob)tests/test_workflow_lint*.py`; `.claude/hooks` ↔ `:(glob)tests/test_guard_*.py`.
  When ANY member of a family is skipped for branch-side edits, skip the WHOLE family (or
  fail loud naming the skipped member and the members being withheld) rather than syncing
  the rest.
- **Derived-artifact alternative for the `workflow.yaml` family.** Where the derived
  artifact is mechanically regenerable, an acceptable substitute for skipping is: sync the
  family, then run `uv run python scripts/workflow_lint.py --emit-tables` and fold the
  regenerated `markers.md` / SKILL.md tables into the SAME sync commit. Add a one-line note
  at the skip site naming `markers.md` as GENERATED FROM `workflow.yaml` so the coupling is
  visible where the decision is made.
- **Post-sync fast-forward assertion.** After the sync commit, run
  `git diff --stat origin/main HEAD -- <synced paths>` and assert the sync is a pure
  fast-forward of the synced paths — refuse to sync a path whose branch content would be
  strictly ahead of `origin/main` (the 139-line re-add case), since `main` may have
  deliberately REMOVED the thing being "freshened".
- **Move the Step 10d pre-gate re-sync out of the race** (`.claude/skills/issue/SKILL.md`
  L10240-10276). Either (a) run the re-sync AFTER the lint gate returns, immediately before
  the push + merge, or (b) keep the pre-gate placement and re-run it when the gate took
  longer than ~5 min, re-deriving freshness against a freshly-fetched `origin/main`
  immediately before committing. Option (a) is preferred — the gate's own landing-tree copy
  already builds from `git archive origin/main`, so a post-gate sync does not invalidate
  the verdict.
- **Both sites carry the same `SPECS` loop** — Step 5a (L2236-2281) and the Step 10d
  pre-gate bullet (which instructs running the Step 5a block with `origin/main`) — so the
  family-atomicity change must land once and be inherited by both, not duplicated
  divergently.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- `scripts/workflow_lint.py` (only if the `--emit-tables` fold is chosen as the
  derived-artifact remedy; no behavior change expected)
- `tests/test_issue_skill_lint_family_sync.py` (existing gate-region pins — the Step 10d
  section carries a negative assert whose region spans the pre-push gate block; any edit
  there must keep that pin green)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- The per-item skip must stay FAIL-SAFE — the #535 incident (blind sync clobbered a
  feature branch's own marker registrations) is why the guard exists; family atomicity
  must widen the skip, never narrow it into a clobber.
- The Step 10d gate-region negative assert in `tests/test_issue_skill_lint_family_sync.py`
  bans writing a full-message grep-exclusion invocation into that section; keep the
  subject-scoped convention.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 8a12cccef12a

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: PENDING

/daily 2026-07-26 route-2 filing. Miner refs: J-P1, H-P2, D-P3, G-P6.
