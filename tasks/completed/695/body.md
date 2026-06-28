---
title: 'workflow-fix: idle-unmapped pass never reaps detached tmux Claude sessions'
kind: infra
tags:
- wf-fix
- wf-fix-fp:296170f25def
created_at: '2026-06-28T01:01:25Z'
has_clean_result: false
origin_prompt: 'User chat 2026-06-27: ''are there a lot of running happy coder/claude
  code/tmux sessions? Why aren''t they getting cleaned up automatically?'' -> ''Do
  both but make sure you''re not killing any working sessions''. Diagnostic found
  the idle-unmapped reaper inert for detached unmapped tmux Claude sessions.'
---
## Overview / Motivation

Auto-filed via the workflow-fix-on-bug protocol from a workflow gap found
during a chat-mode session-cleanup diagnostic (2026-06-27). The
idle-unmapped reaper in `autonomous_session_watch.py` is effectively
INERT for the exact session class it was built to reap: detached,
unmapped, genuinely-idle **tmux-launched** Claude sessions (the
2026-06-12 incident where 25 such sessions held ~23 GB RSS).

## Goal

Make the idle-unmapped pass actually reap a detached, unmapped,
genuinely-idle tmux-launched Claude session — closing the gap where every
such session is kept forever via the `tty=True` / `idle=?`
fail-toward-keep paths — WITHOUT ever reaping a detached session that is
still doing real work.

## Workflow gap

- **Bug observed:** In the live 2026-06-27 17:34 and 17:43 watcher
  passes, every detached, unmapped tmux Claude session logged
  `mapped=False tty=True idle=? missed=0->0 action=clear`. The
  idle-unmapped pass reaped **zero** of them — it is inert for the
  tmux-launched unmapped class.
- **Why it is a workflow gap:** The pass (CLAUDE.md § Crons; this file
  §9036-9138) exists specifically to reap "live-but-idle inner Claude +
  unmapped" sessions that the reconcile/zombie passes structurally
  exclude. The detached-tmux refinement (`_detached_tmux_pane_ttys` +
  `_is_live_user_tty`, ~L8987-9056) was added to handle exactly the
  "detached tmux pane holds a controlling pty" case — but in production
  it is not firing for these sessions, so the reaper never advances past
  `decide_idle_unmapped`'s `has_tty -> ("clear", 0)` / `idle_age_s is
  None -> ("skip", missed)` guards.
- **Live root-cause evidence (gathered this session):**
  - For a still-live detached session (pid 2730037, session
    `cmqvqzomyw6ngyc0utvuiklhd`, tmux `claude-0626-193535`), the
    wrapper's controlling tty is `/dev/pts/9` (fd 0/1/2).
  - The watcher's own detached-pane query —
    `tmux list-panes -a -F '#{pane_tty}\t#{session_attached}'` — DOES
    return `/dev/pts/9 ... 0` (attached=0), i.e. `/dev/pts/9` IS in the
    detached set.
  - Therefore `_is_live_user_tty(2730037, detached_set)` SHOULD return
    `False` (detached pane → not a live-user tty) → `has_tty=False` →
    the decision should fall through to the idle check. Yet the watcher
    logged `tty=True`. So in the cron execution, `_is_live_user_tty`
    returns True when a same-host live check says False.
  - Leading hypotheses for the implementer to confirm: (a) the cron-time
    `tmux list-panes -a` subprocess returns empty / errors under the
    minimal cron env (fail-soft → empty detached set → every tty-bearing
    wrapper kept — the refinement never active in production); (b) the
    sessions were transiently attached at pass time; (c) a pid-resolution
    mismatch (the pid passed to `_is_live_user_tty` is not the
    pty-holding wrapper). Reproduce by running the watcher's exact tmux
    subprocess under the cron environment and by logging the resolved
    detached set + per-session controlling-tty path.
  - Secondary gap: even if `has_tty` were correctly False, these
    sessions log `idle=?` (transcript unresolvable via
    `_transcript_idle_age_s` / `_resolve_transcript_via_happy_log` for
    manually-tmux-launched, non-daemon-spawned sessions) → the
    `idle_age_s is None -> ("skip", missed)` fail-toward-keep path keeps
    them indefinitely. A genuinely-idle detached session with an
    unresolvable happy-log transcript is never reapable.
- **Confidence (emitter):** medium — the symptom (pass reaps nothing for
  this class) is certain and reproducible from the logs; the exact
  internal cause (a vs b vs c, plus the idle=? secondary gap) needs the
  planner/implementer to confirm before choosing the fix.

## CRITICAL constraint — do NOT reap working detached sessions

The fail-toward-keep conservatism is load-bearing and must be preserved.
At diagnosis time, of six detached unmapped tmux sessions:
- ONE was actively running an `experimenter` agent while detached
  (`claude-0626-192206`, `↓ 177.6k`, "Waiting for 1 background agent to
  finish").
- FOUR had a typed-but-unsent user command queued in the input box
  (`promote it useful`, `promote 685 useful`, `ping me when #691
  finishes`, `check progress`).
- Only ONE was genuinely done (empty input, no running agent) — it was
  closed by hand this session.

So any reap predicate MUST additionally exclude a detached session that
(i) has a running descendant agent (experimenter / codex companion /
bg workload python) or (ii) owns a running `pod-<N>`. The fix tightens
ONLY the genuinely-dead case; a false positive here kills a live
experiment. This is precisely why it routes through the full /issue
planner + adversarial-critic + code-review pipeline rather than a quick
edit.

## Proposed change (candidate diff sketch — refine in planning)

- Diagnose why `_is_live_user_tty` returns True in the cron pass for a
  confirmed-detached pane (instrument the detached set + per-session
  controlling-tty path; verify the cron-env tmux query is non-empty).
- If the cron-env tmux query is the cause, make the detached-pane
  resolution robust to the cron environment (explicit socket / `$HOME`)
  and/or log when the detached set comes back empty so a silently-inert
  refinement is visible.
- For the `idle=?` secondary gap: when a session is confirmed detached +
  unmapped + has NO running descendant agent + owns NO running pod, allow
  a corroborating idleness fallback (e.g. tmux `session_activity`
  filtered to real turns, or the wrapper process activity) instead of an
  unconditional keep — but only as an ADDITIONAL gate layered on the
  no-running-work checks above, never replacing fail-toward-keep on its
  own.
- Add/extend `tests/test_autonomous_session_watch.py` coverage:
  detached-pane → reap-eligible; detached-pane-with-running-experimenter
  → keep; attached-pane → keep; idle=? + running work → keep.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
  (`decide_idle_unmapped`, `_is_live_user_tty`,
  `_detached_tmux_pane_ttys`, `_transcript_idle_age_s`, and the
  idle-unmapped pass call site ~L9338-9382).
- Tests: `tests/test_autonomous_session_watch.py`.
- Grep the workflow surface for sibling references before editing.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files
  passes; CLAUDE.md § Crons / `.claude/rules/background-automation.md`
  stay consistent with the new behavior.
- Preserve fail-soft: any tmux/transcript query failure must keep the
  conservative keep-all behavior; the fix may only ever reap MORE in a
  CONFIRMED-dead case, never reap an attached / uncertain / working one.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of
  its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- fingerprint: 296170f25def

<!-- workflow-fix-candidate v1 -->
target_file: scripts/autonomous_session_watch.py
bug_observed: Every detached, unmapped tmux Claude session logs tty=True idle=? action=clear in the live watcher passes, so the idle-unmapped pass reaps nothing for the tmux-launched class it was built for.
why_workflow_gap: The detached-tmux refinement (_detached_tmux_pane_ttys/_is_live_user_tty) and the idle=? fail-toward-keep guard together keep every tmux-launched unmapped session forever, defeating the pass that exists to reap exactly this class (the 2026-06-12 23 GB incident).
proposed_change: Make the idle-unmapped reaper actually reap detached, unmapped, genuinely-idle tmux-launched Claude sessions, gated to exclude any session running a descendant agent or owning a running pod.
diff_sketch: |
  - in the idle-unmapped pass: confirm _is_live_user_tty correctly returns False for a detached pane under the cron env (instrument detached set + controlling-tty path)
  - add a corroborating idleness fallback for idle=? sessions ONLY when detached + no running descendant agent + no running pod
  + tests: detached -> reap; detached+running-experimenter -> keep; attached -> keep
confidence: medium
related_task: chat-mode session-cleanup diagnostic 2026-06-27
<!-- /workflow-fix-candidate -->
