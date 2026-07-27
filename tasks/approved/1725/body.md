---
title: 'daily-fix: Step 4 root-divergence probe; post-merge sync sit'
kind: infra
tags:
- wf-fix
- wf-fix-fp:845a0f74c1f5
- daily-auto-filed
created_at: '2026-07-27T07:16:37Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-26 problem sweep (route 2): Step 4 has no root-divergence
  probe (0 hits for is-ancestor/sync_repo_root/diverg across L1968-2100) so the draft-PR
  pre-check reads a possibly-diverged local main, and the post-merge sync landed by
  9301f1a7bf sits downstream of every epm:merged post, leaving the marker call and
  any root-side call between merge and guard unprotected'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-26 problem sweep (route 2). Surfaced by 3 independent
miner group(s) over the 2026-07-26 session transcripts.

## Goal

Close the two remaining shared-repo-root staleness windows in `/issue`: add a root-divergence probe at the Step 4 worktree cut, and move the Step 10d root sync ahead of the `epm:merged` post so a just-merged guard or spec fix is live at the root for the session's remaining root-side calls.

## Workflow gap

- **Bug observed:** the Step 4 worktree cut reads the shared root's LOCAL `main` with no divergence probe, and Step 10d's only root sync sits after `epm:merged` is posted — so a session can proceed against a diverged root and can be blocked by the very guard it just merged.
- **Why it is a workflow gap:** `.claude/skills/issue/SKILL.md` Step 4 prescribes a `rev-list --count main..issue-<N>` pre-check against local `main` with no reconciliation path, and Step 10d's unconditional `sync_repo_root.py` is positioned inside the post-merge stale-task-folder guard, downstream of every `epm:merged` posting site.
- **Confidence (emitter):** high
- verified-at-filing: `awk 'NR>=1968 && NR<=2100 && /is-ancestor|sync_repo_root|diverg/{c++} END{print c+0}' .claude/skills/issue/SKILL.md` → **0** hits across the whole of Step 4 (absence-of-guard evidence); `grep -n 'rev-list --count main..issue-<N>' .claude/skills/issue/SKILL.md` → 1 hit (L2021), confirming the draft-PR pre-check reads local `main`; `grep -n 'sync_repo_root' .claude/skills/issue/SKILL.md` → 19 hits, the Step 10d unconditional pre-sync at **L12127**, versus the safe-case `epm:merged` success post at **L11273** and the two artifact-confirmed posts at L12000 / L12068 — every `epm:merged` site precedes the sync in document order; `git rev-parse --verify --quiet '9301f1a7bf^{commit}'` → resolves, `git log -1 --date=iso-strict 9301f1a7bf` → `2026-07-26T04:03:24-07:00 task #1694: Step 10d apply-verification + post-merge guard pre-sync (#1465)` (2026-07-26)

## Evidence

- **The unconditional post-merge root sync is already LANDED and is excluded from this filing.** Commit `9301f1a7bf` (2026-07-26T04:03:24-07:00 = 11:03:24Z) added the Step 10d post-merge stale-task-folder guard pre-sync at L12127: `uv run python "$REPO_ROOT/scripts/sync_repo_root.py" || echo "post-merge guard pre-sync: …"`, in a section headed `Post-merge stale-task-folder guard (runs after EVERY merge form lands)`. Filing a second "add sync_repo_root after gh pr merge" change would duplicate it. What remains open is its POSITION and the Step 4 probe.
- Session `2b779905` merged #1699 at commit `d79fa07b0e` (`git rev-parse` resolves; `2026-07-26T06:01:52-07:00` = 13:01:52Z), whose whole content is: bare `ruff check` uses `pyproject.toml` per-file-ignores that relax `scripts/*`, so the implementer must also run `pytest tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset`. It merged server-side and the shared root stayed `behind=2` (confirmed in session `891b2cc6` row 445: `ahead=29 behind=2`).
- Session `891b2cc6` spawned its implementer at 13:39:39Z — 38 minutes after that merge — reading the STALE `.claude/agents/implementer.md`. Its round-1 report: `"**Lint:** uv run ruff check scripts/autonomous_session_watch.py … && uv run ruff format --check … — All checks passed; 3 files already formatted."` (no pin test). The 26-minute Step 9c gate then FAILed: `"FAIL count=1 … NEW: tests/test_ruff_policy.py::test_live_workflow_helpers_clean_under_full_ruleset — the FULL-ruleset ruff check (no per-file-ignores) surfaces 2 payload errors"` (RUF003 and C901). Measured cost: roughly 30 minutes of wall clock for a second full 26:24 gate, plus one extra implementer round and one extra verdict cycle.
- Session `7ce3a81f`, 09:14:11.489Z: 25 seconds after the PR squash-merged, `task.py post-marker 1710 epm:merged --note "…recovery: git merge origin/main into branch…"` was blocked — `"PreToolUse:Bash hook error: [scripts/guard_repo_root_branch.sh]: BLOCKED: 'git merge (branch merge on the shared root)' would move the SHARED repo-root tree off main…"` — because the pre-fix guard was still on disk at the root. The session's own read: `"Guard tripped on --note prose containing \"git merge\" (my own fix hasn't propagated to the shared root yet — pre-fix guard is running here). Using --file route."` Cost: 1 blocked call, 1 cancelled sibling call, a Write and a re-issue, roughly 35 s; no data loss.
- Session `c0319d9e`, 12:02:42Z–12:03:04Z: at the Step 4 worktree cut, `git rev-list --count main..issue-1701` returned 2 before the implementer had committed anything. Investigating, the session found local `main` and `origin/main` mutually non-ancestral — `"origin/main is NOT ancestor of main\nmain is NOT ancestor of origin/main"` — then moved straight on: `"Draft PR will open post-implementer commits. Dispatching implementer now."` A grep over every Bash `command` in that session found 0 `sync_repo_root.py` invocations. No cost realized this run (Step 10d merged server-side, bypassing the root); the latent cost is that the next session's root push is rejected.

## Proposed change

- In `.claude/skills/issue/SKILL.md` Step 4a, before the draft-PR pre-check at L2021, add a root-divergence probe: if `git -C "$REPO_ROOT" merge-base --is-ancestor origin/main main` fails AND `git -C "$REPO_ROOT" merge-base --is-ancestor main origin/main` also fails, run `uv run python "$REPO_ROOT/scripts/sync_repo_root.py"` — the sanctioned single-flight recovery — before proceeding. Fetch `origin/main` first so the probe reads a current ref.
- Make the divergence outcome visible rather than silent: on a detected divergence, record it in the chat line and re-probe after the sync; a still-diverged root after one sync attempt is reported, not stepped over.
- In Step 10d, run the root sync BEFORE the `epm:merged` post at L11273 (and before the artifact-confirmed posts at L12000 / L12068), not only inside the post-merge stale-task-folder guard at L12127. The guard's own pre-sync stays where it is — it is idempotent and single-flight, so the earlier call costs nothing when the root is already current.
- Make the Step 10d `epm:merged` recipe use the `task.py post-marker --file` form by default. Merge-recovery notes routinely quote git verbs and the repo-root guard matches on note prose, so the `--note` form is the fragile default.
- Note in Step 10d that a merged diff touching `scripts/*guard*` or `.claude/hooks/*` is not live at the shared root until the sync runs, so the landing session cannot exercise its own fix before then.
- unverified hypothesis — verify at plan time: that placing the sync before `epm:merged` is safe on every Step 10d exit path (safe-case, merge-conflict recovery, artifact-confirmed / surgical-additive checkout). The sync pull-rebases the root and can move the canonical task-folder path, which the post-merge guard already re-resolves; whether an earlier call interacts with the `epm:merged` idempotency skip at L9735 was not checked at compose time.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- none

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `uv run python scripts/workflow_lint.py` passes (no-flags); ruff clean on touched files.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- sha-verify (filing-time, #1467): `2b779905` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `891b2cc6` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `7ce3a81f` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `c0319d9e` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: 845a0f74c1f5

- workflow_fix_target: .claude/skills/issue/SKILL.md

/daily 2026-07-26 route-2 filing. Miner refs: F-P1, C-P2, I-P4.

