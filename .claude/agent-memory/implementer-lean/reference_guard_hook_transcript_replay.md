---
name: guard-hook-transcript-replay
description: Recipe + gotchas for replaying historical tool calls through guard_harmful_bank_read.sh-family hooks (#1217 §6.4); token-walk latency; transcript retention
metadata:
  type: reference
---

Replaying Claude Code session history through a PreToolUse guard (the #1217
pre-ship FP replay; tooling committed as `scripts/issue1217_replay_*.py` on
branch issue-1217):

- **Store**: `~/.claude/projects/<flattened-cwd>/*.jsonl` + the
  worktree-suffixed sibling dirs (`...--claude-worktrees-issue-N`). ~3 GB /
  1,400 files scans in ~80 s with a `'"tool_use"' in line` + loose-regex
  prefilter before `json.loads`. Extract tool_use INPUTS only (never
  results — they carry raw corpus text); write matches straight to work
  files, print counts only.
- **Retention**: transcripts age off at ~30 days. Evidence a plan pins to
  the transcript store (incident-call traces) MUST be snapshotted into task
  artifacts at plan time — #1217's §6.4-bis incident source (2026-07-06)
  was gone by implementation (2026-08-08) though the plan verified it on
  2026-07-17.
- **Replay**: feed synthesized payloads (`{"tool_name":..,"tool_input":..}`)
  to the hook via subprocess stdin; rc=2 deny / rc=0 allow. Override the
  sidecar with `EPM_BANK_GUARD_LOG=/tmp/...` so replay denies don't pollute
  the production FP log; strip `EPM_ALLOW_BANK_READ` from env. Make the
  runner RESUMABLE (results jsonl keyed by idx) — 1,862 replays took ~15 min
  wall at 16 workers, over the 600 s Bash-tool cap.
- **Token-walk cost (hook gotcha)**: the #965 Bash-arm token walk spawns ~2
  grep subprocesses per token ≈ 25 ms/token on the contended VM (100 tok =
  2.5 s, 1,000 tok = 25 s, 5,000 tok = 173 s). Any new path class whose
  cheap-gate tokens are COMMON in ordinary commands (raw_completions ≈
  63 cmds/day vs ~0 for bank stems) arms multi-second hook walls
  fleet-wide and blows the 60 s harness hook timeout at p99. Measure the
  gate's arming rate on historical traffic BEFORE widening.
- **Whole-command co-occurrence transfers badly to common tokens**: 14.66%
  deny rate on the corpus class, 221/273 denies sanctioned shapes, 51%
  heredoc script-compose (verb `cat` + token inside the composed script
  body). The same posture on rare bank stems produced ~0 FP traffic.
