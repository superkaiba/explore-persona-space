---
title: 'Codex twin: CLI + gpt-5.6-sol, daily auto-upgrade cron (landed)'
kind: infra
tags: []
created_at: '2026-08-15T23:38:01Z'
has_clean_result: false
origin_prompt: can you update to most recent and change it to use the best model (with
  reasonable amount of thinking)?
workflow: v1
---
# Codex twin: CLI + gpt-5.6-sol, and a daily auto-upgrade cron

User-requested (2026-08-15 chat): re-upgraded the ChatGPT plan, asked to update
Codex and point the twin at the best model with a reasonable amount of
thinking, then asked to make that automatic.

**Landed:** commit `73de57ba6e40610b1396c1a4dbb60b6cff275a7c` on `main`.

## What was already committed before this task ran

- `scripts/codex_task.py` `--effort` argparse default `xhigh` -> `high`
  (commit `c3100982b0`, 2026-08-15). Inside the documented dispatch range
  (`.claude/rules/codex-ensemble-review.md` `--effort <high|xhigh>`), so
  explicit-effort callers are unaffected.
- `~/.codex/config.toml` `model` -> `gpt-5.6-sol` (VM-local, not in the repo).
- A stale `.claude/cache/codex-quota-exhausted-until` sentinel (written
  2026-08-06, claiming exhaustion until 2026-09-05) cleared — it had been
  short-circuiting every twin dispatch at exit 9.

## What this task actually landed

The original body recorded the work below as complete. It was written, and it
was **running**, but it was never committed. At task start all three files were
untracked (`git status` = `??`) while a **live crontab entry**
(`17 7 * * * scripts/cron_codex_auto_upgrade.sh`) had executed the wrapper
daily since 2026-08-15 — nine passes, all fired, and the CLI had self-upgraded
0.147.0 -> **0.149.0** over that window (past the 0.147.0 the original body
recorded). So the defect was durability, not function: any `git clean -fdx`,
fresh clone, or worktree recovery would have deleted a live cron's script, and
nothing in the repo recorded that the automation existed.

| File | Lines | State at task start |
|---|---|---|
| `scripts/codex_auto_upgrade.py` | 652 | untracked; **landed as-is, byte-unchanged** (sha256 `9f273f93…36b646`) |
| `scripts/cron_codex_auto_upgrade.sh` | 118 -> **180** | untracked; landed with 3 review-driven fixes |
| `tests/test_codex_auto_upgrade.py` | 268 -> **586** | untracked; 27 -> **31** tests |

Correctly **not** committed, both already gitignored: `logs/` (the daily logs,
`.gitignore:93`) and `.claude/cache/` (the known-bad probe cache and the audit
sidecar, `.gitignore:85`) — per-VM runtime state, not source.

### `scripts/codex_auto_upgrade.py`

Keeps the CLI and the twin's model pinned to the newest usable pair. Automates
three failure modes hit by hand on 2026-08-15:

1. **CLI/model version coupling** — each model advertises a
   `minimal_client_version`; bumping the model without the CLI yields a 400 at
   dispatch, not at config time. The selector filters candidates against the
   installed CLI.
2. **The stale app-server** — a long-lived `codex app-server` keeps serving the
   pre-upgrade runtime, so twin dispatch fails with "requires a newer version
   of Codex" while a direct `codex exec` on the same model succeeds and
   `codex --version` reveals nothing. The upgrader kills broker + app-server by
   explicit PID; they respawn on demand.
3. **Slugs the account cannot use** — `gpt-5.5-codex` is listed but 400s on a
   ChatGPT account, and no listing field predicts it. So a candidate is PROBED
   with a real `codex exec` call before being written to config; failures go to
   a known-bad cache keyed by CLI version, so a newer CLI re-probes.

Safety properties: aborts when any Codex job is in flight, re-checked
immediately before the app-server kill (an upgrade mid-job kills the job, and
the ensemble sites read that as a twin no-show, silently degrading a review
gate). The in-flight check is age-bounded at 2h because companion job records
are advanced only by a live session, so a session killed mid-job strands its
record at `running` forever — two such records from 2026-05 were still
non-terminal three months on, and an unbounded check would let that debris
disable the cron silently. `model_reasoning_effort` is never touched. Config
writes are atomic and rewrite only the top-level `model` key.

### `scripts/cron_codex_auto_upgrade.sh` — three fixes over the running version

The bug all three close: the wrapper does its work inside a brace group
redirected to a per-day log, then alerts on `rc != 0`. **The brace group is not
a subshell**, so an `exit` inside it terminates the script *before* the alert
arm. With no MTA on this VM and the crontab redirecting stderr, that path was a
completely silent failure — it defeated the alerting the wrapper exists to
provide. Three sites had it:

- the failed `cd "$PROJECT_DIR"`;
- the `uv`/`npm`/`codex` prerequisite preflight, which additionally ran *before*
  the alert variables were defined, so it could not have alerted at all;
- an unchecked `mkdir -p`, whose failure left `rc` unset — which `${rc:-0}`
  converted into **success**.

All three now set `SETUP_OK=0` / `rc=1` and converge on one `alert_failure`
helper defined ahead of anything that can fail. Also: the recommended-schedule
comment now matches the installed crontab (`17 7`, not `17 6`).

One residual is **documented-accepted**: a brace-redirect failure *after* a
successful `mkdir` (exists-but-unwritable, ENOSPC, TOCTOU) still leaves `rc`
unset. Closing it would mean redesigning the alert arm to not depend on the log
it reports into.

### `tests/test_codex_auto_upgrade.py` — 31 tests

27 unit tests over the decision helpers, plus four end-to-end pins:
`--dry-run` mutates nothing (all four `dry_run` branch sites, with the mutation
seams intercepted so a broken thread is *detected* rather than executed against
the live config); the failed-`cd` alert fallthrough; broken-current-model
exclude-and-replace ordering; and the setup-failure alert path. Seam fakes are
signature-conformant by construction (`create_autospec`, or a `def` mirroring
the real signature) — never bare `Mock()`.

## Review

Full plan-review floor (three self-contained plan versions, `verify_plan.py`
PASS 0 FAIL / 0 WARN, Methodology-lens critic) plus two rounds of the
Claude+Codex code-review ensemble, both rounds with the quota sentinel checked
clear so both were genuine two-family rounds.

- **Round 1 split** — Claude PASS, Codex FAIL (4 BLOCKER). The binding
  `reconciler` returned **PASS** with plan kill criterion 2 **NOT triggered**,
  deferred five findings, and kept one CONCERN open.
- **Round 2** — Claude PASS; Codex CONCERNS with 0 Critical / 0 Major and an
  explicit "Merge" recommendation. Both ledger items VERIFIED-ADDRESSED. No
  split, so no second reconciler round.

Two mechanism corrections came out of this that are worth keeping:

1. **There is no ~10s heartbeat on companion job records.** The worker writes
   only on phase/thread/turn *change*
   (`.../codex/1.0.4/scripts/lib/tracked-jobs.mjs:75-102`, `if (!changed)
   return;`). Codex was textually right; the Claude reviewer and the
   orchestrator both over-read the `codex_task.py:288-305` spawn-window
   self-heal comment. The 2h in-flight bound is nonetheless sound — phases
   toggle at every activity-class transition during a live review
   (`lib/codex.mjs:238-290`), a silent job is force-cancelled at 600s of log
   silence, all 143 companion records on this VM span under 2h (max 1191s ≈
   19.8 min), the kill fires only on `changed`-bearing passes, and a lost twin
   surfaces as a *visible* no-show rather than a silent degrade.
2. **`--dry-run` has three vacuous-pass paths, not one.** A dry run can exit 0
   having exercised nothing via (a) a stale auth token — the probe that
   refreshes it in-band is skipped under `--dry-run`, so `fetch_models` 401s to
   `None` and the run reports "no models listing"; (b) an in-flight skip; and
   (c) the CLI and model both already current. Path (a) was found by the
   Methodology critic and (b) was found empirically at landing, when two
   landing-time dry runs both hit the in-flight guard. The success criterion
   was therefore tightened to require a **listing-backed selection line**,
   which no vacuous path can emit.

## Verification

- `uv run pytest tests/test_codex_auto_upgrade.py -q` -> **31 passed**.
- `ruff check` + `ruff format --check` clean; `bash -n` clean. `shellcheck` is
  not installed on this VM, so the wrapper's static coverage is `bash -n` plus
  the behavioral tests.
- Default no-flags `workflow_lint.py`: FAIL, but **zero** payload-attributable
  lines across three separate runs — all errors belonged to other live
  sessions' files.
- Inline payload lint gate: **PASS**, certifying each path's exact content.
  The wrapper's certified hash `cf8dd9d349f6` equals its landed blob hash.
- `codex_auto_upgrade.py --dry-run`: rc=0, `~/.codex/config.toml` byte-identical
  before and after, and (at plan time, when no jobs were in flight) emitting
  `model already best available (gpt-5.6-sol, priority 1)` — the listing-backed
  line the tightened criterion requires.
- Landing verified by SHA-blob read against `origin/main`, not the push line:
  the commit is an ancestor of `origin/main`, all three blobs read back at
  652 / 180 / 586 lines, mode `100755` recorded on the wrapper, and the live
  crontab still resolves to the now-tracked path.

## The `model_reasoning_effort` discrepancy — documented, not changed

The original body recorded `~/.codex/config.toml` `model_reasoning_effort`
`xhigh` -> `high`. The live config reads **`xhigh`**. The upgrader provably did
not do this (it never writes that key, and
`test_write_config_model_replaces_only_the_model_line` pins that), so either a
later manual edit restored `xhigh` or the original record was wrong.

Left as-is deliberately. The originating ask was "the best model (with
reasonable amount of thinking)", and for every EPS ensemble dispatch that is
already satisfied: `codex_task.py` passes `--effort high` explicitly on the
command line, overriding the config default. The config value therefore governs
only interactive `codex` use, where `xhigh` may well be the preference.
Changing it would be an unrequested mutation of a personal config outside the
repo.

## Follow-ups filed

- **#2510** — `verify_plan.py` should assert the smoke blind-spot enumeration
  when a plan declares a smoke/dry-run verification. Cheap, because the trigger
  already exists (check `c11_dryrun_test_coverage`).
- **#2511** — the same silent-exit-before-alert shape in three other
  alert-bearing cron wrappers, plus ~7 more without alert arms. Four
  independent instances is the signal that prose guidance has not held, so a
  mechanizable shell lint is proposed alongside the fixes.
- **#2512** — the five deferred `codex_auto_upgrade.py` hardenings
  (heartbeat-less in-flight window + the `:445` check-to-kill race; missing
  production-body tests for `run()`/`fetch_models()`; no durable
  restart-pending state; unsafe TOML model write; transient probe failures
  cached as capability failures), plus the `sidecar-json-path-escaping` NIT
  (now at `:76-78` inside `alert_failure`, shifted from `:103-105`).

## Related

`scripts/cron_codex_reaper.sh` (every 6h at :23) reaps app-server daemons older
than 24h — complementary: it bounds daemon age, this bounds runtime staleness
at upgrade time. The schedules deliberately do not collide.
