---
title: 'workflow-fix: five deferred hardenings in scripts/codex_auto_upgrade.py'
kind: infra
tags: []
created_at: '2026-08-24T00:30:56Z'
has_clean_result: false
parent_id: 2322
origin_prompt: 'Bundles the five findings the #2322 code-review ensemble raised against
  scripts/codex_auto_upgrade.py that the binding reconciler upheld as real but non-blocking
  (disposition ''deferred --by reconciler'' in tasks/*/2322/concerns.jsonl): the heartbeat-less
  in-flight window + :445 check-to-kill race, missing production-body coverage for
  run()/fetch_models(), no durable restart-pending state, unsafe TOML model write,
  and transient probe failures cached as capability failures.'
workflow: v1
---
# workflow-fix: five deferred hardenings in scripts/codex_auto_upgrade.py

Bundles the findings the #2322 code-review ensemble raised against
`scripts/codex_auto_upgrade.py` that the binding `reconciler` upheld as real
but ruled **non-blocking** for that task's landing (the file landed as-is per
critic-approved plan v3). Each was verified as a genuine mechanism; each was
judged not to warrant blocking a payload that had already run daily in
production for nine days. They are collected here so "non-blocking" does not
quietly become "forgotten".

Ledger provenance: `tasks/*/2322/concerns.jsonl`, all with disposition
`deferred --by reconciler`.

## 1. In-flight detection has no heartbeat, and a check-to-kill race

`inflight_jobs()` / `_job_age_s()` (`:159-184`, `:227-233`) exclude a
non-terminal job whose record is older than `JOB_FRESH_WINDOW_S` (2h). The
mtime fallback is consulted **only** when `updatedAt`/`createdAt` is missing or
unparseable, so a stale-but-valid timestamp on a live job is not rescued.

The reconciler established the mechanism precisely: the companion worker writes
the job record only on phase/thread/turn **change**
(`~/.claude/plugins/cache/openai-codex/codex/1.0.4/scripts/lib/tracked-jobs.mjs:75-102`,
`if (!changed) return;`) — there is **no** ~10s heartbeat. An earlier reading
that claimed one was an over-read of the `codex_task.py:296-300` spawn-window
self-heal comment; record that correction here so it is not re-derived wrongly.

Why it was not blocking: phases toggle at every activity-class transition
during a live review (`lib/codex.mjs:238-290`); a silent job is force-cancelled
at `DEFAULT_STALL_DETECT_SECS = 600` of log silence and the log grows on every
event; all 143 companion job records on this VM have spans under 2h (max
1191s ≈ 19.8 min, ~6× under the window) and the only two non-terminal records
are three-month-old May debris — exactly the class the bound exists to exclude;
the kill fires only on `changed`-bearing passes (`:637`); and a killed twin
surfaces as a visible review no-show, not a silent pass.

Still worth fixing, because the safety margin rests on incidental facts rather
than on the design:
- `codex_task.py` permits 6h runs (`DEFAULT_MAX_WAIT_SECS = 6 * 3600`) against a
  2h window, and the upgrader's own comment asserts "a twin review runs minutes,
  not hours." The two files disagree about the same quantity.
- Suggested direction: make `_job_age_s` take the **most recent** evidence of
  activity — `min(age_from_updatedAt, age_from_mtime)` — rather than trusting a
  stale timestamp and never consulting mtime. Consider an authoritative liveness
  signal (validated live worker PID) and reconciling the window against
  `DEFAULT_MAX_WAIT_SECS`.
- `:445-457` has an unlocked check-to-kill window: a dispatch can begin between
  the final in-flight check and the kill loop. #2323 landed a repo-keyed
  advisory dispatch lock (`.claude/cache/codex-dispatch.lock`) serializing
  spawn+confirm; the upgrader does not participate in it. Sharing that lock, or
  a drain protocol, would close the race properly.
- `TERMINAL_PHASES` (`:81`) omits the legacy status `"completed"`. A record with
  no `phase` and `status: "completed"` reads as non-terminal and can block
  upgrades for 2h. Cheap to fix.

## 2. No production-body coverage for `run()` and `fetch_models()`

The tests stub `mod.run` (`:100`) and `mod.fetch_models` (`:242`) at seams;
no test executes either real body. Per `.claude/rules/code-style.md`
§ "One production-body test per seam-stubbed function", subprocess options,
auth-file parsing, request construction, response-shape handling and exception
behavior can all regress with the suite green.

Why it was not blocking on #2322: the plan pinned test scope at exactly three
added tests, both functions pre-date this round unmodified, and both bodies are
demonstrably production-executed (nine daily cron passes plus a listing-backed
`--dry-run`).

Fix: execute real `run()` against an `create_autospec`'d `subprocess.run`, and
real `fetch_models()` against a temporary auth file plus an autospecced/fake
`urlopen` response. Both fakes signature-conformant by construction — never bare
`Mock()`.

## 3. No durable "restart pending" state after a successful CLI install

`changed` is in-memory only. After `npm install` succeeds, a config-read error
(`:547-550`), an unhandled probe timeout, or a busy-restart deferral (`:637`)
all end the pass without restarting the app-server. The next day the CLI is
already current, `changed` stays empty, and the restart is never retried — so a
long-lived app-server can sit permanently on the pre-upgrade runtime while every
subsequent daily pass reports success.

Why it was not blocking: every trigger path exits rc=1 into the same-day
Telegram alert, whose message already names the stale-runtime symptom and the
`codex_task.py` probe remedy. Loud, not silent.

Fix: persist the installed-but-not-restarted CLI version and retry the restart
until confirmed. Test by simulating install-success followed by a config-read
failure or a busy restart, then re-running `main()` and asserting the restart is
retried.

Related, from the same sweep: an `@latest` race can install a version newer than
the earlier `npm view` reported; `new_v != latest` then treats a successful
upgrade as failed and records no restart requirement.

## 4. Config editing is not safe TOML serialization

`read_config_model` / `write_config_model` (`:314`, `:331-334`) recognize only
double-quoted keys, so a valid `model = 'slug'` is missed and a **second**
`model` key is prepended — producing duplicate-key invalid TOML from valid user
config. The external `slug` is inserted directly into both a regex replacement
string and the TOML output with no validation or escaping, so a slug containing
a quote, newline or backslash can corrupt or inject into the config.

Why it was not blocking: reaching it requires either a hand-edited
single-quoted config or a hostile/malformed slug from the authenticated models
listing, and the write is probe-gated.

Fix: validate allowed slug syntax and use a TOML-aware edit (or a correctly
escaped serializer handling both string forms). Test single-quoted input and
slugs containing `"`, newline and backslash; require valid TOML output or
explicit rejection.

Adjacent: `candidate_models()` assumes every listing element is a dict with
mutually sortable priorities; malformed upstream data raises rather than failing
with a controlled diagnostic.

## 5. Transient probe failures are cached as capability failures

`probe_model` returns a single `ok=False` for both a deterministic
"model not supported on this account" and a transient 429/5xx/app-server blip
(`:406-412`). Both paths write the slug into `known_bad` for the entire CLI
version (`:567-574`, `:605-612`), so one transient failure can downgrade a
working current model and blacklist it until the next CLI release.

Why it was not blocking: a down-switch additionally requires a same-pass clean
sibling probe, the effect is visible and recoverable, and the cache has
accumulated 0 false entries in nine days (its single entry, `gpt-5.5-codex`, is
a true account-capability failure).

Fix: classify deterministic capability failures separately from transient
transport/service failures; do not persist the latter without bounded
confirmation. Test that a transient failure alerts/retries but is not cached,
while a deterministic unsupported-model response is.

## Also worth folding in (Minor, same file)

- `:564-578` — when no top-level `model` key exists, the token-warming probe is
  skipped, so a stale token can make the production run exit cleanly again and
  again without ever fetching the listing needed to repair the config.
- `:409-412` — the sentinel-echo check requires two transcript occurrences and so
  depends on `codex exec` prompt-echo behavior; a CLI output-format change would
  fail every probe and could mass-blacklist models. Prefer a structured
  final-output channel, or a nonce checked in the actual response.
- `:640-641` — unreachable defensive branch (`:539-540` always appends to
  `changed` when setting `cli_upgraded=True`). Delete it, or rewrite as an
  assertion on the invariant it means to protect.
- `cron_codex_auto_upgrade.sh:103-105` — `LOG_FILE` is serialized into the JSON
  sidecar row without JSON escaping; an override path containing a quote or
  backslash yields an invalid row.

## Not in scope

The sibling cron wrappers carrying the same `cd … || exit 1`
silent-exit-before-alert shape are tracked separately (filed from the same #2322
review). Keep the two tasks disjoint.

## Acceptance

Each of the five numbered items either fixed with a test pinning the fixed
behavior, or explicitly dispositioned with a reason. The `min(updatedAt, mtime)`
change in item 1 and the `TERMINAL_PHASES` addition are the cheapest real
safety wins; the TOML serializer in item 4 is the one most likely to bite a
user who hand-edits their config.
