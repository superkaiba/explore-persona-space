---
title: 'workflow-fix: serialize Codex twin dispatch against the repo-shared non-atomic
  job index + post-spawn probe retry'
kind: infra
tags: []
created_at: '2026-08-16T05:50:29Z'
has_clean_result: false
origin_prompt: 'Found during /issue 2321 Phase 2: three concurrent codex_task.py dispatches
  raced the repo-keyed non-atomic codex-companion state.json; only 1 of 4 job ids
  remained findable. The adversarial-planner SKILL.md instructs exactly this concurrent
  shape.'
workflow: v1
---
## Goal

Make the five doubled review sites dispatch their Codex twins in a way that does
not race the fleet-shared, non-atomically-written `codex-companion` job index, and
give the POST-SPAWN probe the same torn-read tolerance `#1020` already gave the
RESULT fetch.

## The gap

`scripts/codex_task.py` L218-236 documents the mechanism in its own source: the
companion CLI reads a jobs index "written NON-ATOMICALLY (fs.writeFileSync in
place, no lock) and read-modify-written by every concurrent wrapper; a torn read
parses as an EMPTY jobs list … so the CLI exits 1 with 'No job found for "<id>"'
even though the job ran to completion (>=7 incidents on 2026-07-02, each
re-paying a 10-60 min Codex run)".

Measured blast radius (2026-08-16, this session): that index is keyed by **REPO**
— not by issue, not by session —
`~/.claude/plugins/data/codex-openai-codex/state/explore-persona-space-8c746e82e245efd5/state.json`
(23 jobs at probe time). Every Codex dispatch from every concurrent session in
this repo read-modify-writes ONE file with no lock.

Against that substrate, the workflow surface instructs concurrent dispatch:

- `.claude/skills/adversarial-planner/SKILL.md:1004` — "Bg-dispatch in a SINGLE
  message so all 3 Codex runs proceed concurrently."
- `.claude/skills/adversarial-planner/SKILL.md:1096` — "each in a single message
  (3 parallel bg-Bash calls)".

So a *correctly executed* Phase 2 critic ensemble issues three simultaneous
wrappers against an unlocked shared index, by instruction. Realized failure
(#2321, plan v3 Phase 2): of four job ids spawned across three near-simultaneous
dispatches, exactly ONE was findable afterwards; the statistics lens failed rc=4
`post-spawn probe failed (probe-error): No job found for "task-msvdof9j-s6s3e5"`
and its one built-in transient re-dispatch failed identically. Recovery cost was
a full kill-and-re-dispatch cycle.

**Why `#1020` does not cover this.** #1020 (completed) added
`DEFAULT_RESULT_FETCH_RETRY_CAP = 3` scoped deliberately to the RESULT fetch —
correctly so, since re-dispatching a *completed* job is wrong (exit 7 stays out
of `TRANSIENT_FAIL_EXIT_CODES`). The POST-SPAWN probe leg has no equivalent
tolerance, and #1020 did not touch the callers' concurrency. Distinct bug, same
file family.

## Scope / surfaces

1. **`scripts/codex_task.py`** — give the post-spawn probe torn-read tolerance.
   The safety asymmetry that justified scoping #1020 to the fetch does NOT apply
   here: a post-spawn probe failure means "cannot confirm registration", and the
   job may well be running, so the correct response is re-probe-with-backoff
   (bounded, jittered, same shape as the fetch retry), never a blind
   re-dispatch. A re-dispatch on an unconfirmed-but-live job is exactly how
   #2321 ended up with orphaned jobs to cancel.
2. **Serialize per-repo dispatch.** Either an advisory `flock` around companion
   invocations (the index is repo-keyed, so a repo-scoped lock is the natural
   grain — cf. `task.py`'s own `~/.task-workflow/lock` pattern), or a documented
   sequential-dispatch contract at the five call sites. Prefer the lock: it fixes
   the fleet-wide case (other sessions' dispatches), which a per-session
   sequential contract structurally cannot.
3. **The five doubled-site instructions** — `.claude/skills/adversarial-planner/SKILL.md`
   (Phase 2, both line refs above) plus the `/issue` SKILL.md dispatch blocks for
   `code-reviewer`, `interpretation-critic`, `clean-result-critic`,
   `follow-up-critic`. If (2) lands as a lock, these can keep their
   single-message ergonomics and simply queue; if (2) lands as a contract, the
   text must stop instructing simultaneity.
4. **`.claude/rules/codex-ensemble-review.md`** — record whichever contract wins,
   next to the existing pre-spawn quota-sentinel check and the killed-wrapper
   re-attach recipe. That rule is where a dispatching orchestrator looks.

## Acceptance

- A concurrent three-way dispatch in this repo either serializes cleanly or is
  refused with a diagnostic naming the shared index — never a spurious
  "No job found" on a live job.
- Post-spawn probe failures re-probe (bounded + jittered) before declaring a job
  lost, and NEVER re-dispatch a job that may be running; a test pins that
  asymmetry against #1020's fetch-retry semantics.
- A regression test reproduces the torn-read shape (empty-jobs parse) and asserts
  the wrapper survives it on the post-spawn leg.
- The five doubled-site instructions and `codex-ensemble-review.md` agree with the
  realized mechanism; `workflow_lint.py` no-flags run stays clean.

## Provenance

Found during `/issue 2321` Phase 2 (HF data-repo repack plan v3 critic ensemble).
Not a #2321 blocker — that session recovered by dispatching strictly sequentially
and recorded the diagnosis in an `epm:progress` marker on #2321.
