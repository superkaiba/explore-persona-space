---
title: Diagnose selective SIGTERM (rc=143) killer of in-session foreground python
  children on the shared VM
kind: infra
tags: []
created_at: '2026-08-14T00:45:45Z'
has_clean_result: false
workflow: v1
---
## Goal
Identify and stop whatever is SIGTERMing foreground python children of Claude-session shells on the shared VM. Evidence from session 61064f64 (issue-2221 worktree, 2026-08-13): four kills — issue2221_build_mix.py attempt 1 (rc=143 at 16:24:40Z, ~30 s from completion), a huggingface_hub upload_folder python (~same window, stdout buffer lost), and scripts/file_infra_task.py twice (~00:42-44Z, killed within seconds, even on --help). Discriminator established: the SAME commands run setsid-detached (new session) survive and complete rc=0; interleaved same-shape in-session commands (task.py, git) were untouched. No OOM (journalctl clean, 53 GB free), no watcher kill rows, no pgrep stragglers.

## Notes
Candidate suspects to check: harness-side child reaping on tool-call boundaries; a fleet janitor/hook pkill matching argv patterns; systemd session-scope cleanup. Reproduce via a sleep-bearing python child in a foreground Bash tool call vs setsid. Until root-caused, the workaround is the established detached-launch shape.
