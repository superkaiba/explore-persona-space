---
name: GCE metadata script runner kills workloads on giant progress-bar lines
description: GCP-lane workloads streaming vLLM/tqdm \r-progress bars through startup-script stdout die mid-run on "bufio.Scanner: token too long" → SIGPIPE; VM zombies at RUNNING with eps/phase stuck at "workload"
type: feedback
---

GCE's `google_metadata_script_runner` reads the startup script's stdout line-by-line
with a bounded Go `bufio.Scanner`. A workload that emits a long NEWLINE-FREE line —
vLLM/tqdm `\r`-carriage-return progress bars ("Processed prompts: ...") are the
canonical case — overflows the scanner: syslog shows
`error while communicating with "startup-script" script: bufio.Scanner: token too long`
followed by `Script "startup-script" failed with error: signal: broken pipe`.

**Why it zombies:** the EXIT trap RUNS on the SIGPIPE death, but `$?` inside it
reads 0 (status of the last completed command — measured on bash 5.1.16, #607
fact-check), so the `[ "$rc" -ne 0 ]` guard no-ops; the `eps/phase` guest
attribute never flips to `failed` and the VM stays RUNNING — the GCP poll reads
a healthy "workload" phase forever while no process is alive.

**How to detect:** GPUs at 0% + no python workload processes + `sudo grep
"token too long" /var/log/syslog`. State on disk survives (the kill is the
parent shell, not a wipe).

**How to recover:** FIRST (REQUIRED, #908) re-publish `eps/phase=workload`
via the guest-attribute curl (`curl -fsS -X PUT -H "Metadata-Flavor: Google" --data "workload" "http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/eps/phase"`)
BEFORE resuming any work on the VM — the #908
zombie predicates (`reconnect_or_none` + the pre-launch stale reclaim in
`backends/gcp.py`) classify a RUNNING VM whose phase reads terminal/wedged
(`done`/`failed`/`wedged`) as a finished zombie and DELETE it on the next
dispatch, so an active relaunch must never be left reading terminal. Then
relaunch the REMAINING phases on the same VM via SSH under
`setsid nohup ... < /dev/null` (detachment trio — never bare `nohup`; full GCP
salvage-launcher shape incl. root PATH + metadata-sourced tokens:
`.claude/rules/gotchas.md` #653/#823 salvage-launcher entry) with stdout/stderr
redirected to a FILE (bypasses the metadata-runner pipe entirely); on success replicate the startup script's success tail —
write the completion sentinel at the handle's `sentinel_path`, then publish
`eps/phase=done` via the guest-attribute curl — so the existing poll/finalize
path proceeds unchanged. Keep the rc!=0 → `_eps_phase failed; shutdown -h now`
trap for billing bounds.

**How to prevent:** never stream raw workload output through startup-script
stdout — redirect the workload block to a file (the `log_path` the handle
already names) and/or set `TQDM_DISABLE=1` / vLLM `disable_tqdm` on non-tty.
Renderer-side fix tracked as an infra task (gcp.py `render_startup_script`).

Burned at #491 attempt 2 (2026-06-11): killed mid-free_gen at 17:03 UTC after
22/29 cells; prior session had stalled, watcher respawn diagnosed + recovered
on the live VM.
