---
title: 'daily-fix: AUTO_COMPACT_WINDOW not reaching spawned sessions'
kind: infra
tags:
- wf-fix
- wf-fix-fp:14b36e55316f
- daily-auto-filed
created_at: '2026-08-06T06:59:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): a session spawned 12h after
  the #2073 env fix still compacted at preTokens ~250K and thrashed 9.5h'
workflow: v1
---
# daily-fix: CLAUDE_CODE_AUTO_COMPACT_WINDOW override not reaching spawned issue sessions — a post-fix session thrashed 9.5h at ~250K preTokens

## Workflow gap

The #2073 fix for the `tengu_amber_redwood` reduced-window autocompact-thrash
(`env.CLAUDE_CODE_AUTO_COMPACT_WINDOW=1000000` in `~/.claude/settings.json`) landed
2026-08-04T18:17Z (commit 2de759b9d7). Yet the /issue 2004 orchestrator session (transcript
0d005b55, spawned 2026-08-05 ~06:32Z — 12+ hours AFTER the fix) cycled
compact → re-read → refill → compact for ~9.5 h (06:32–16:00Z): 10 "Autocompact is
thrashing" + 9 "Prompt is too long" failed-turn API errors, 45 continuation-summary
restarts, 25 identical 430-line SKILL.md re-reads — with `compactMetadata.preTokens` =
242,315–253,458 at EVERY boundary, far below the 1M native window. That is the Class-1
reduced-window signature per `.claude/rules/gotchas.md` — on a session that should have
carried the override.

verified-at-filing (2026-08-06T07:1xZ): `git log -1 --format='%aI' 2de759b9d7` →
2026-08-04T14:17:11-04:00 (18:17Z); `grep AUTO_COMPACT ~/.claude/settings.json` →
`"CLAUDE_CODE_AUTO_COMPACT_WINDOW": "1000000"` present. The preTokens figures are the
miner's probed `compactMetadata` extraction over the 0d005b55 transcript (28.0 MB, main
projects dir).

unverified hypothesis — verify at plan time: the spawn path (Happy daemon →
`scripts/spawn_session.py` → `claude` child) does not deliver settings-`env` overrides to
the spawned process — e.g. the daemon was started before the fix and children inherit the
daemon's environment rather than re-reading settings.json `env` at claude startup, OR the
enrollment overrides the env var. Not probed; the fix design depends on which it is.

## Proposed change

Determine why the override did not bind for the 0d005b55 session and close the gap:
(a) reproduce — spawn a disposable `--auto` session and read its effective
`CLAUDE_CODE_AUTO_COMPACT_WINDOW` (or its first compact_boundary preTokens);
(b) if the daemon-env path is the culprit, export the override explicitly in
`scripts/spawn_session.py`'s spawn payload (and restart the daemon), so every spawned
session carries it regardless of daemon vintage;
(c) update `.claude/rules/gotchas.md` § reduced-window with the verified delivery path.

Secondary (same incident, separable): the refill driver was the Step 10d executable-gate
span re-read (25× identical 430-line reads) — extracting that span into an executable
helper (`scripts/step10d_guards.sh`-style) would cut the refill pressure; planner may
scope it out to the Step 10d recipe task if one is open.

## Provenance

- fingerprint: 14b36e55316f

- workflow_fix_target: scripts/spawn_session.py, .claude/rules/gotchas.md
- origin: /daily 2026-08-05 problem sweep — miner 7 P1 (probed compactMetadata scan of
  transcript 0d005b55); watcher reconcile-stop symptom filed separately.
