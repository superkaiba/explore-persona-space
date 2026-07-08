---
title: 'daily-held: decide ruff-debt burn-down (2149 errors)'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-04T21:37:07Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-02 backfill route-3: bulk lint-debt scope/priority
  decision for Thomas'
workflow: v1
---
## Overview / Motivation

Filed by the /daily 2026-07-02 backfill problem sweep (route 3 — genuine
judgment call, needs Thomas). Held under the carve-out: bulk scope/priority
decision with large blast radius (NOT auto-dispatchable).

## The decision needed

`main` carries ~2,149 pre-existing repo-wide ruff errors (2,072 on 07-02,
growing). On 2026-07-02 alone, at least 7 sessions each burned a
verification round proving their diff didn't cause the red lint. Options:

1. **One-time burn-down task** — `ruff check --fix` + manual triage of the
   rest, as a dedicated `kind: infra` task. Cost: a huge diff that will
   conflict with ~15 live worktree branches; needs a quiet window.
2. **Baseline-and-freeze** — accept the debt, rely on the Step 9c
   known-red baseline ledger (filed separately) so sessions stop paying
   the re-derivation tax; ratchet: no NEW errors allowed.
3. **Do nothing** — status quo; every code session keeps paying the
   pre-existing-ness triage round.

Suggested: option 2 now (cheap, no conflicts), option 1 in a quiet window
after the current experiment wave lands.

## Provenance

- source: /daily 2026-07-02 backfill problem sweep (route 3, needs-human)
- carve-out: scope/priority judgment with large blast radius
