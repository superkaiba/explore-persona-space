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
ROOT CAUSE (plan v2 §2, established diagnosis): **earlyoom**. The three original suspects are all REFUTED — harness-side child reaping on tool-call boundaries, a fleet janitor/hook pkill matching argv patterns, and systemd session-scope cleanup played no role.

Mechanism: this VM's `EARLYOOM_ARGS` carry `--prefer '(^|/)(pytest|python3?)$'` (+300 badness; the regex matches bare `python` and `pytest`, not just `python3`) on a box where every process baselines at kernel oom_score 666 (flat), so ANY python/pytest — a 44 MiB `--help` invocation included — reads badness ~966 and outranks every non-python process (a 3728 MiB java reads 686; Claude sessions 670). The kill condition is CONJUNCTIVE (MemAvailable AND swap free both <= 10%); on 2026-08-13/14 the box ran with SwapTotal=0, so the swap side was permanently satisfied and SIGTERM bursts swept the highest-badness (all-python) victims: 363 kills Aug 13 + 287 Aug 14 in the unit journal, covering the Goal's four kills.

The Goal's discriminators dissolve under this mechanism: the setsid A/B was a TIMING confound — earlyoom kills in bursts at the floor, and the detached reruns landed outside a sweep (comm/adj/oom_score are identical for in-session vs setsid children); "journalctl clean" missed the kills because earlyoom logs to its UNIT journal (`journalctl -u earlyoom`, `badness` rows), not dmesg; "53 GB free" was a DISK read, not memory.

RESOLUTION: the 64 GiB swapfile (`/mnt/eps-data/swapfile`) activated 2026-08-14 08:29 PDT un-satisfied the swap side of the conjunction; ZERO victim kills since (last: Aug 14 08:16:07; journal retention begins Aug 13 02:52, so earlier days are unknown, not zero). Two regression paths remain and are guarded by the WARN-only preflight swap check `_check_swap_state` (D3): (a) the fstab entry carries `nofail`, so a boot where /mnt/eps-data fails to mount activates NO swap silently; (b) swap exhaustion (~59% consumed as of 2026-08-22) drifting SwapFree toward the 10% floor. The prescribed `choom -n -600` recipes remain correct — attenuated (badness 567 vs the ~666 crowd), not void.

Durable docs: `.claude/rules/gotchas.md` earlyoom entry (D1) and `.claude/skills/issue/failure_patterns.md` § exit-137/143 (D2).
