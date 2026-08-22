---
title: 'Codex twin: CLI 0.147.0 + gpt-5.6-sol, daily auto-upgrade cron'
kind: infra
tags: []
created_at: '2026-08-15T23:38:01Z'
has_clean_result: false
origin_prompt: can you update to most recent and change it to use the best model (with
  reasonable amount of thinking)?
workflow: v1
---
# Codex twin: CLI 0.147.0 + gpt-5.6-sol at high effort, and a daily auto-upgrade cron

User-requested (2026-08-15 chat): re-upgraded the ChatGPT plan, asked to update
Codex and point the twin at the best model with a reasonable amount of
thinking, then asked to make that automatic.

## Part 1 — the manual upgrade

- `npm install -g @openai/codex@latest`: 0.137.0 -> 0.147.0 (the gpt-5.6
  family advertises `minimal_client_version` 0.144.0).
- `~/.codex/config.toml` (VM-local, not in repo): `model = "gpt-5.5"` ->
  `gpt-5.6-sol`; `model_reasoning_effort = "xhigh"` -> `high`.
- `scripts/codex_task.py`: `--effort` argparse default `xhigh` -> `high`
  (commit `c3100982b0`). Inside the documented dispatch range
  (`.claude/rules/codex-ensemble-review.md` `--effort <high|xhigh>`), so
  explicit-effort callers are unaffected.
- Cleared a stale `.claude/cache/codex-quota-exhausted-until` sentinel
  (written 2026-08-06, claiming exhaustion until 2026-09-05) that was
  short-circuiting every dispatch at exit 9.

## Part 2 — `scripts/codex_auto_upgrade.py` + daily cron

`17 7 * * * scripts/cron_codex_auto_upgrade.sh` (crontab backed up to
`~/.eps-autonomous/crontab-backups/`). Upgrades the CLI to the newest npm
release, re-selects the best model the CLI can run, and restarts the
app-server. Logs to `logs/codex_auto_upgrade/YYYY-MM-DD.log`; one Telegram
alert per day on rc != 0.

Three failure modes it automates away, all hit manually on 2026-08-15:

1. **CLI/model version coupling** — bumping the model without the CLI yields a
   400 at dispatch, not at config time. The selector filters candidates by
   `minimal_client_version` against the installed CLI.
2. **The stale app-server** — a long-lived `codex app-server` keeps serving the
   PRE-upgrade runtime, so dispatch fails with "requires a newer version of
   Codex" while a direct `codex exec` on the same model succeeds.
   `codex --version` shows the new version and reveals nothing. The upgrader
   kills broker + app-server by explicit PID (they respawn on demand).
3. **Slugs the account cannot use** — `gpt-5.5-codex` is listed but 400s on a
   ChatGPT account. No listing field predicts this, so a candidate is PROBED
   with a real `codex exec` call before being written to config; failures go
   to a known-bad cache keyed by CLI version (a newer CLI re-probes).

Safety properties:

- Aborts when any Codex job is in flight, re-checked immediately before the
  app-server kill. An upgrade mid-job kills the job, and the ensemble sites
  read that as a twin no-show — silently degrading a review gate to
  single-Claude.
- The in-flight check is AGE-BOUNDED (2h). Companion job records are only
  advanced by a live session, so a session killed mid-job strands its record
  at `running` forever — two such records from 2026-05 were still non-terminal
  three months on. An unbounded check would let that debris disable the cron
  silently, which is the worst failure shape: a no-op that looks healthy.
- `model_reasoning_effort` is never touched — it is a cost/latency preference,
  not a freshness property.
- Config writes are atomic (tmp + `os.replace`) and rewrite only the top-level
  `model` key, leaving any `[profiles.*]` override alone.

## Verification

- `--dry-run` and two live runs (config temporarily downgraded to `gpt-5.4`):
  probed `gpt-5.6-sol`, switched, restarted the app-server, confirmed dead.
  Config diffed byte-identical to backup apart from the model line.
- Twin dispatch after the auto-restart returns the model self-reporting slug
  `gpt-5.6-sol`.
- Wrapper run under a stripped `env -i` cron-like environment: rc=0.
- 27 unit tests (`tests/test_codex_auto_upgrade.py`), ruff clean.

## Related

`scripts/cron_codex_reaper.sh` (every 6h at :23) reaps app-server daemons
older than 24h — complementary: it bounds daemon age, this bounds runtime
staleness at upgrade time. Schedules deliberately do not collide.
