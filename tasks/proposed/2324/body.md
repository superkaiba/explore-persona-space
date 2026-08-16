---
title: 'workflow-fix: symlink/FIFO-safe bounded opens for the three advisory-flock
  lock sites + thread realized lock mode into codex_task failure notes'
kind: infra
tags: []
created_at: '2026-08-16T08:04:58Z'
has_clean_result: false
origin_prompt: 'Codex twin Majors 2+4 at #2323 code-review round 1; #2323 reconciler
  standing recommendations 2+3 (adjudicated out-of-contract for that round, recommended
  as one task covering all three sites).'
workflow: v1
---
# workflow-fix: harden the three advisory-flock lock-file opens (O_NOFOLLOW/O_NONBLOCK + bounded open) and thread the realized lock mode into failure notes

## Goal

Close two hardening gaps the #2323 code-review ensemble surfaced but that were correctly ruled OUT of that round's scope by the binding reconciler (they are out-of-contract for #2323's approved plan, and one of them spans files #2323 never touched).

## Gap 1 — advisory-flock lock-file opens are neither symlink-safe nor covered by their own acquisition timeout

Three sites share one idiom: open a fixed lock path with `os.open(path, O_WRONLY | O_CREAT, 0o600)`, then enter a bounded `flock(LOCK_EX | LOCK_NB)` poll loop.

- `scripts/codex_task.py:1254` (added by #2323)
- `scripts/sync_repo_root.py:626,733`
- `scripts/step9c_baseline.py:1430`

Two defects, identical at all three:

1. `os.open` follows symlinks — a symlink at the lock path flocks an unintended inode, so the lock silently stops excluding the thing it is supposed to exclude.
2. `os.open` lacks `O_NONBLOCK` and runs BEFORE the deadline is constructed, so a FIFO at the lock path blocks in `open()` indefinitely — bypassing the advertised acquisition timeout (1800 s in `codex_task.py`) and the intended fail-open diagnostic. The caller wedges instead of degrading.

**Why this was not fixed in #2323:** the #2323 plan explicitly prescribed "the `sync_repo_root.py` / `step9c_baseline.py` idiom", so the implementation faithfully copied the named precedent; and the trigger requires a same-user actor planting a FIFO/symlink at a fixed path inside gitignored `.claude/cache/`, which sits outside the fleet's accidents-not-adversaries trust model. The reconciler's ruling was that if adopted at all, it belongs to all three sites as one task rather than a per-round patch — hence this task.

**Proposed fix:** one shared helper (rather than three divergent copies) that opens with `O_NOFOLLOW | O_NONBLOCK` where available, `fstat`s the fd and rejects anything that is not a regular file, and treats a rejection as a LOUD fail-open (matching each caller's existing fail-open posture — `codex_task.py` already has an `unavailable` mode with a WARN + a `dispatch_lock=unavailable` marker token). The open must sit INSIDE the deadline so its cost counts against the advertised bound.

Note the three callers do NOT share a fail-open posture today: `codex_task.py` fails open by design, while `sync_repo_root.py` and `step9c_baseline.py` have their own single-flight semantics. Verify each caller's intended behavior before unifying — a shared helper must not silently convert a fail-CLOSED single-flight into a fail-open one.

## Gap 2 — `codex_task.py` discards the realized lock mode on every failure path

`lock_token` is built only on the successful `epm:codex-task-spawned` branch (`scripts/codex_task.py:1378`). The exit-10 confirm-exhaustion return (`:1356-1375`) and the spawn-exception return (`:1345`) precede it, and the signal handler posts its failure marker through an independent path (`:409-418`). So a dispatch that ran in `timeout-failopen` / `unavailable` / `disabled` mode AND then hit the shared-index race leaves a failure marker with no record of which mode was in force — exactly the forensic pairing an operator wants.

This is NOT a #2323 contract violation: plan v5 §D-c scopes the token to the spawned marker, and §4.1(3) requires only job id + shared index + `--reattach` in the exit-10 note (all three present). It is cheap forensics worth having.

**Proposed fix:** retain the realized mode in attempt state and append `dispatch_lock=<mode>` to all post-lock failure notes (spawn-exception, confirm exhaustion, signal cleanup).

## Acceptance

- All three lock opens are symlink-safe and bounded by their caller's own acquisition deadline; a FIFO or symlink at any lock path yields a bounded, loud, caller-appropriate outcome rather than an indefinite block.
- Each caller's pre-existing fail-open-vs-fail-closed posture is preserved and stated explicitly in the diff.
- Regression tests create FIFO and symlink lock-path fixtures in a child process and assert bounded completion per site.
- `codex_task.py` failure notes on all post-lock paths carry `dispatch_lock=<mode>`, pinned by a parameterized test over the non-held modes.
- `workflow_lint.py` no-flags shows no NEW failures vs the plan-time baseline.

## Provenance

Surfaced by the Codex twin at #2323 code-review round 1 (Majors 2 and 4), adjudicated by the #2323 reconciler as out-of-contract-for-that-round with the explicit recommendation to file one task covering all three sites (standing recommendations 2 and 3). Not a #2323 defect: both were ruled Discarded there, and the diff shipped on a binding PASS.
