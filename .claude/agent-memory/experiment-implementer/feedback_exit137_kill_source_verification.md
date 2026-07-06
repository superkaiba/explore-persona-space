---
name: exit-137 on the shared VM — verify the kill source before diagnosing OOM
description: shared login-scope cgroup counters do not attribute; check per-run oom_kill DELTA, MemAvailable floor, oomd journals, and PM stop directives before calling exit-137 an OOM
type: feedback
---

exit-137 (SIGKILL) on the shared VM is NOT proof of OOM. Before writing an OOM
diagnosis: (1) cgroup `memory.events oom_kill` DELTA over the run window — the
login-session scope hosts MANY fleet processes, so absolute counters
(`memory.peak`, historical `oom_kill`) do not attribute to your process; a
non-incrementing counter rules OOM out; (2) MemAvailable floor over the run
window (a >20 GB floor rules OOM out); (3) `journalctl -u earlyoom` /
`-u systemd-oomd` for a kill line at the death timestamp; (4) events.jsonl for
a PM/operator stop directive at/before the death.

**Why:** #779 r9 (2026-07-02): three exit-137 grid deaths were diagnosed as
kernel OOM from shared-scope counters (peak 116.4 GB, oom_kill=2) plus
back-derived allocation arithmetic — all three were deliberate PM-session
SIGKILLs (mem floor 47 GB; counters never incremented; the hypothesized
1100-draw H×H materialization did not exist in the code). A crash-fix round was
dispatched against a nonexistent bug.

**How to apply:** any silent process death (no traceback, exit 137/143) on
shared infrastructure — run the 4-point kill-source checklist FIRST; also
sanity-check a back-derived memory estimate against the code's actual
allocation sites before it drives a fix round.
