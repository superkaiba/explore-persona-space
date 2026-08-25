---
title: 'workflow-fix: apply the Fix-A silent-exit ladder to the sibling cron wrappers'
kind: infra
tags: []
created_at: '2026-08-24T00:13:49Z'
has_clean_result: false
parent_id: 2322
origin_prompt: 'Surfaced by the Claude code-reviewer''s bug-class sweep during #2322
  review round 1: the cd "$PROJECT_DIR" || exit 1 silent-exit-before-alert shape that
  #2322 fixed in cron_codex_auto_upgrade.sh also exists in 3 other alert-arm-bearing
  cron wrappers (cron_step9c_ledger_refresh.sh:70 and cron_uv_cache_prune.sh:51 spot-verified),
  plus ~7 more without alert arms, plus a second site inside the #2322 wrapper itself
  (the command -v preflight at :43-48 and the unchecked mkdir at :60).'
workflow: v1
---
# workflow-fix: apply the Fix-A silent-exit ladder to the sibling cron wrappers

Surfaced by the Claude `code-reviewer`'s bug-class sweep during #2322 review
round 1, then spot-verified before filing.

## The bug class

A cron wrapper of this project's standard shape does its work inside a brace
group redirected to a per-day log, then checks `rc` afterwards and Telegram-
pushes on failure:

```bash
{
    cd "$PROJECT_DIR" || exit 1      # <-- the bug
    ...
    rc=$?
} >> "$LOG_FILE" 2>&1

if [ "${rc:-0}" -ne 0 ]; then ... "$TELEGRAM_PUSH" "$MSG" ... fi
```

The brace group is **not** a subshell, so `exit 1` inside it terminates the
whole script — **before** the `rc != 0` alert arm. On this VM there is no MTA
and the crontab lines redirect stderr to a log nobody reads, so that path is a
completely silent failure. It defeats exactly the alerting the wrapper was
written to provide.

#2322 fixed this in `scripts/cron_codex_auto_upgrade.sh` by replacing the
`cd ... || exit 1` with an `if ! cd` / `elif` / `else` ladder that logs FATAL,
sets `rc=1`, skips the (cwd-relative) payload invocation, and falls through to
the existing alert arm. That is the pattern to apply here.

## Sites

Claude's sweep named three other **alert-arm-bearing** wrappers carrying the
same shape — these are the ones where the bug actually bypasses an alert:

- `scripts/cron_step9c_ledger_refresh.sh:70` — spot-verified: `cd "$PROJECT_DIR" || exit 1` at `:70`, with a `TELEGRAM_PUSH` alert arm at `:114`. Confirmed same class.
- `scripts/cron_lesson_consolidate.sh:90` — reported by the sweep. Note its alert helper is defined early (around `:65`), so verify whether the `exit` actually bypasses it before changing anything; it may already be safe, or safe for a different reason.
- `scripts/cron_uv_cache_prune.sh:51` — spot-verified: `cd "$PROJECT_DIR" || exit 1` at `:51`. Confirm whether it has an alert arm to bypass; if it has none, it belongs in the lower-priority group below.

The sweep additionally reported ~7 more wrappers with the `cd`-exit but **no**
alert arm to bypass. Those are lower priority — the `exit` is not silencing an
alert that exists — but they are the same latent shape and are worth
normalizing while the pattern is fresh. Enumerate them rather than trusting
this count.

## Second site class, from the same #2322 review

The identical class appears at a *second* site inside the #2322 wrapper itself,
and was left as a Minor rather than fixed in that round:

- `scripts/cron_codex_auto_upgrade.sh:43-48` — the `for bin in uv npm codex`
  preflight runs `exit 1` at `:46` while `LOG_DIR` (`:52`), `TELEGRAM_PUSH`
  (`:55`), `SIDECAR` (`:56`) and `SENTINEL` (`:57`) are not yet defined, so it
  *structurally* cannot alert — while its own comment says "Fail LOUD if still
  missing".
- `scripts/cron_codex_auto_upgrade.sh:60` — `mkdir -p "$LOG_DIR" "$SENTINEL_DIR"`
  is unchecked. An uncreatable log dir proceeds into the brace-group redirect,
  the redirect fails, the group never runs, `rc` is never assigned, and
  `${rc:-0}` converts that into **success**.

Whether #2322 fixes these in its own round or defers them here depends on the
reconciler's severity ruling on that task; check #2322's outcome first and do
not duplicate work it already landed.

## Suggested shape

Centralize alerting into a function defined **before** the preflight and before
any `cd`, so every failure path — missing prerequisites, unwritable log dir,
failed `cd`, non-zero payload rc — reaches one alert path that does not itself
depend on the thing that failed. Initialize failure state before the
redirection. Keep the per-date sentinel dedup and the audit-sidecar row.

## Mechanizable check

A shell lint would generalize this: in a wrapper that has an alert arm keyed on
`rc`, no `exit` may appear between the log brace group and that arm. Worth
adding alongside the fixes so the shape cannot come back — the sweep found four
independent instances, which is the signal that prose guidance alone has not
held.

## Acceptance

- Each named site either fixed with the Fix-A ladder or explicitly dispositioned
  as already-safe with the reason.
- A behavioral test per fixed wrapper, in the shape #2322 used: operate on a
  `sed`-rewritten copy in `tmp_path` with the failure induced, redirect the
  `EPS_*` log/sentinel/sidecar/telegram env knobs into `tmp_path`, and assert
  the Telegram stub is called. Never touch the live log dirs, the real audit
  sidecars, or the real Telegram channel.
- The full wrapper inventory enumerated, so the "~7 more with no alert arm"
  figure is replaced by a verified list.
