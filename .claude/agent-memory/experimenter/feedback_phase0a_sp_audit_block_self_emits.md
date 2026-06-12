---
name: phase0a-sp-audit-block-self-emits
description: i489_run_all Phase 0a fail-loud-s via its own `epm:failure` sentinel + `set -e exit 2` when SP identity-check audit rejects both drafts; clean exit looks like silent process death until you read /workspace/logs/issue-489-epm_failure-*.json
metadata:
  type: feedback
---

When `scripts/i489_run_all.sh` Phase 0a (`i489_phase0_sp_identity_check.py`) self-blocks because both rewrite drafts for an SP `cid` get judge verdict=different, the dispatcher writes a Phase 0a `epm:failure` sentinel to `/workspace/logs/issue-489-epm_failure-<epoch>.json` AND the wrapper exits cleanly via `set -e` after the `exit 2`. **Why:** the gated phase exit is the correct fail-loud path, but downstream the pgrep against the wrapper PID returns 0 matches — visually indistinguishable from the "setsid nohup over SSH MCP got reaped" infra failure documented in the agent prompt. **How to apply:** before posting `epm:failure infra`, ALWAYS check (a) the per-phase log under `logs/issue_489/<phase>.log` and (b) `ls /workspace/logs/issue-489-*.json` for a self-emitted sentinel. If a sentinel exists with `failure_class: code`, post the marker forwarding the sentinel's fields verbatim, do NOT retry, do NOT classify as infra. Burned at #489 v2 re-launch (2026-06-04): wrapper PID disappeared after 60s, three previous attempts had also written only 2 lines to their wrapper logs — but `phase0a.log` had the full 200-OK Anthropic trace and the BLOCK reason `SP03 draft 1 verdict=different; draft 2 verdict=different`.
