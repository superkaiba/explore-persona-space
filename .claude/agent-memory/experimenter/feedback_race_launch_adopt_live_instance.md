---
name: Race-launch — adopt the live instance, fix the pidfile
description: A concurrent orchestrator session may launch the same pod-side driver while the experimenter is mid-protocol. If exactly one healthy detached instance survives, adopt it (overwrite pidfile, post marker with ITS pid) instead of killing/relaunching.
type: feedback
---

During #552's follow-up launch (2026-06-10, pod-552) a concurrent session
(the orchestrator, which had been running its own `preflight --no-gpu`)
launched a SECOND copy of the driver ~90s after mine, truncating the shared
log (`>` redirect) and superseding my instance (mine died; likely its
stale-proc kill). Result: one healthy, fully detached (setsid+nohup+
`</dev/null`) driver whose PID did NOT match my pidfile.

Rule: when you discover a race-launch, **count live driver instances first**
(`pgrep -af <driver>`, `nvidia-smi --query-compute-apps`):

- Exactly ONE healthy instance (yours or theirs) → ADOPT it. Overwrite the
  pod-side pidfile with the live PID (`echo <pid> > /workspace/logs/...pid`),
  verify log freshness, and post `epm:run-launched` with that pid + pidfile.
  Killing a mid-run healthy driver to "own" the launch wastes staged work
  and risks racing again.
- TWO live instances → kill the YOUNGER one (less work lost), keep the elder,
  then adopt as above. Never let two dispatchers share a GPU/log.

Also check whose PID the OTHER launcher captured: a `bash -c '... & echo $!'`
wrapper can record the wrapper PID, not the driver (theirs recorded the
bash -c itself). Your marker with the verified live driver PID is the
correction of record. Flag the adoption + provenance explicitly in the
marker note so the poller history explains the PID change.

**Why:** stale pidfile → poller declares a healthy run dead (incident #451
family); killing the legitimate instance → cascade-kill incident family
(feedback_ssh_bash_lc_backgrounding).
**How to apply:** any time liveness checks contradict your own launch
bookkeeping, suspect a concurrent launcher before suspecting a crash.
