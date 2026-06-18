---
title: 'GCP lane: redirect workload output to log file — GCE metadata runner bufio.Scanner
  token-too-long kills workloads with progress bars'
kind: infra
tags: []
created_at: '2026-06-11T17:26:43Z'
has_clean_result: false
---
## Goal

Make the GCP lane's startup script immune to the GCE metadata-runner `bufio.Scanner: token too long` kill by never streaming raw workload output through startup-script stdout.

## Problem

Incident #491 attempt 2 (2026-06-11): the GCE metadata script runner reads startup-script stdout line-by-line with a bounded Go `bufio.Scanner`. vLLM's `\r`-carriage-return progress bars accumulated into one enormous newline-free line, overflowed the scanner (`error while communicating with "startup-script" script: bufio.Scanner: token too long`), the runner closed the pipe, and the workload died on SIGPIPE mid-`free_gen`. Because bash skips EXIT traps on a fatal SIGPIPE, `eps/phase` stayed `workload` and the VM zombied at RUNNING — invisible to the poll. Manual SSH recovery was required.

## Fix (src/explore_persona_space/backends/gcp.py, render_startup_script)

1. Redirect the workload block's stdout/stderr to the handle's `log_path` (`$WORKLOAD_ROOT/logs/issue-<N>.log`) instead of streaming to the metadata runner; keep a small heartbeat (phase lines only) on stdout if useful.
2. Export `TQDM_DISABLE=1` (and consider vLLM `disable_tqdm`) in the rendered env as defense in depth.
3. Consider trapping SIGPIPE (`trap '' PIPE` or explicit handler) so the rc!=0 EXIT trap (phase=failed + shutdown) still bounds billing if a kill class like this recurs.
4. Optional: have `GcpBackend.poll` read the log file's mtime over SSH for `last_log_mtime_sec_ago` so zombie states (phase stuck at `workload`, no live process) become detectable.

## Acceptance criteria

- Rendered startup script no longer pipes workload output to the runner (unit test on `render_startup_script` output).
- A simulated giant-line workload (printf of >1MB with no newline) survives on a real or mocked launch path.
- Existing GCP-lane tests pass.

## References

- Task #491 events: `epm:failure` + `epm:failure-lesson` markers, 2026-06-11.
- `.claude/agent-memory/experimenter/feedback_gcp_metadata_runner_token_too_long.md`.
