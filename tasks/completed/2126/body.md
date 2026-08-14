---
title: 'daily-fix: five gate-recipe defects (Step 9c/10d)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8012842466fa
- daily-auto-filed
created_at: '2026-08-06T07:07:01Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): selector key unpinned (whole-suite
  31min run); no re-sync before gate re-runs; command substitution held detached launch
  open; merge-guard process-substitution false positive; resync recovery chokes on
  added files'
workflow: v1
---
# daily-fix: issue-skill gate-recipe robustness — five probed defects in the Step 9c/10d recipes

## Workflow gap

Five distinct recipe-level defects fired across 2026-08-05/06 gate rounds; each is small,
all live in the same recipe surface (`.claude/skills/issue/SKILL.md` Step 9c/10d +
`scripts/step10d_guards.sh`):

1. **Selector output key not pinned.** #1992's gate round 1 ran the ENTIRE suite (19,223
   tests, ~31 min) because the launch script read the selector JSON with key `'files'`
   (real key: `'tests'`) — $FILES empty ⇒ whole-suite collection; an inline probe repeated
   the same KeyError before the correction (session 04b3e1dc, forensics marker 07:39Z).
2. **No re-sync before gate RE-runs.** #2006's merge took ~3 h — three full gate runs —
   partly because #2104's fix landed on main MID-gate, turning the worktree's stale copies
   into NEW-node reads; the recipe syncs once at gate start, not before each re-run
   (session 08f070b7, heartbeat v23).
3. **Detached-launch command substitution holds the wrapper open.** A
   `PHASE_PID=$(bash -c "… setsid nohup …")` launch was killed at the 2-min Bash cap
   BEFORE writing its pid file while the detached job started anyway — the substitution
   inherited the child's stdout; only the bracketed ownership probe prevented a duplicate
   launch (session 9a69ef38, 23:48Z).
4. **Merge-guard false positive from process substitution in an eval'd prelude.** #2006's
   pre-merge verdict check exited 1 "BLOCKED: verdict missing/stale" while the immediate
   re-probe showed verdict `pass` + SHA == tip — the `grep -qxE … <(sed …)` comparison
   inside the eval'd guard prelude failed spuriously (session 08f070b7, 03:39Z).
5. **Post-gate resync recovery chokes on added files.** #2087's resync recovery loop
   `git checkout HEAD -- <p>` errored on newly-ADDED files ("pathspec … did not match")
   and parsed porcelain with `awk '{print $2}'` (fragile for renames/spaces); the session
   verified tip/verdict unchanged and merged anyway (session 0d007d1a, 23:08Z).

verified-at-filing: all five are probed tool_result/marker readbacks at the cited rows;
targets confirmed present at compose time (`grep -n "'tests'" scripts/select_step9c_tests.py | head -2`
→ the selector's real key; `grep -n 'checkout HEAD' .claude/skills/issue/SKILL.md | head -3`
for the resync recipe).

## Proposed change

One pass over the two recipe surfaces: (1) pin the selector key (`tests`) in the verbatim
launch snippet — or have `select_step9c_tests.py` also emit a ready-to-exec file list;
(2) re-sync the landing tree against origin/main immediately before EACH gate re-run;
(3) in the detached-launch recipe, redirect the inner command's stdout/stderr to the log
INSIDE the `$( )` so the substitution returns immediately, and require an explicit Bash
`timeout` on launch wrappers; (4) replace the merge-guard's process-substitution compare
with plain variable comparison; (5) make the resync recovery use
`git restore --staged --worktree` (added files included) instead of per-path
`checkout HEAD`, and parse porcelain with `-z`.

## Provenance

- sha-verify (filing-time, #1467): `0d007d1a` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: 8012842466fa

- workflow_fix_target: .claude/skills/issue/SKILL.md, scripts/step10d_guards.sh
- origin: /daily 2026-08-05 problem sweep — miners 6 (P12), 4 (P5/P12/P18), 3 (P19).
