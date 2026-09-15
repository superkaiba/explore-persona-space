# Experiment monitoring and recovery

Detection is incomplete until it triggers intervention. A task marker or local
log entry is not a delivered alert, and a live monitor PID is not proof of an
effective recovery workflow. Never claim automatic recovery from detection alone.

Before handing back a launched experiment, register its monitor with
`scripts/experiment_watchdog.py` (or an equivalent tested supervisor). Keep the
configuration, runbook, observations, and watchdog state under
`~/.local/state/eps/experiment-watchdogs/<run-name>/`, not disposable scratch.
The watchdog uses an independent persistent systemd timer, one bounded Codex
worker per incident, a rolling recovery limit, and acknowledged notifications.
It never invokes Claude or Anthropic. Routine successful checks are silent.

The runbook must name the user's authorized scope, canonical task, source and
artifact provenance, exact monitor service and backend handles, remaining phases,
resume validation, startup/progress checks, and upload-gated teardown procedure.
Recovery must inspect current evidence, preserve completed work, prove an old
worker has stopped before relaunch, and remain within the existing authorization.
Do not equate restarting the monitor with restarting the experiment.

## Registration

Create an operator-owned JSON configuration with these fields:

```json
{
  "version": 1,
  "name": "issue-N-round-name",
  "monitor_unit": "experiment-monitor.service",
  "observation": "/absolute/path/observation.json",
  "state_dir": "/absolute/path/watchdog-state",
  "workdir": "/absolute/path/experiment-worktree",
  "runbook": "/absolute/path/recovery-runbook.md",
  "codex": "/absolute/path/to/codex",
  "uv_environment": "/absolute/path/to/existing/.venv",
  "source_sha": "the-exact-40-character-source-commit-hash",
  "stale_seconds": 900,
  "recovery_seconds": 2700,
  "max_recoveries_per_day": 2,
  "notify_argv": ["/absolute/path/to/acknowledged-notification-helper"]
}
```

Use the user's established personal notification route or their explicit choice.
The notification command receives one additional plain-text argument and must
return success only after the destination acknowledges receipt; merely queueing
a message locally is insufficient. Do not put credentials in the configuration.

The monitor observation contract is `source_sha`, `status`, `checked_at` (Unix
seconds), and `backend_observation` with measured `status`, process liveness,
log age, and any stall/reachability alarms. The monitor must check actual output
progress too; a parent heartbeat must not conceal a stalled child. Unknown or
stale evidence triggers diagnosis, never a blind compute relaunch. Transitions
and timeouts must account for the longest measured legitimate phase.

Terminal completion requires independently verified exact-source outputs and
`results.source_sha` plus `results.verified_revision`. A worker's successful exit
alone cannot clear an incident. The observation must postdate the recovery attempt.

Run the real canary in a bounded systemd service with the intended PATH and
`UV_PROJECT_ENVIRONMENT`, then install:

```bash
uv run python scripts/experiment_watchdog.py canary /absolute/path/config.json
uv run python scripts/experiment_watchdog.py install /absolute/path/config.json
```

Installation requires a fresh canary matching both the configuration and code.
The canary executes a real read-only Codex command and sends an explicitly labeled
notification test. Installation pins that tested code outside the mutable research
checkout, starts a fresh check, and verifies the timer is active. Verify user
manager lingering for operation after logout/reboot. Inspect the next scheduled
tick as well, not only the installer's first check.

Test the failure path before relying on it: a monitor exits after detecting a
failed subprocess; one recovery worker starts; failed notification delivery is
retried; a watchdog interruption cannot duplicate recovery; stale observations
cannot declare recovery; exhausted attempts produce an actionable alert. The
regression suite is `tests/test_experiment_watchdog.py`.

If either canary fails, repair the integration or keep active supervision while
reporting the limitation. Do not describe the experiment as safely unattended.
An unavailable host or notification network remains a limitation; never promise
that software prevents every crash or that a message was seen by a human.
