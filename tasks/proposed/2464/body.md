---
title: 'Runtime guard: a cap-hit re-gen that skips every row must not report regen_applied:
  true'
kind: infra
tags: []
created_at: '2026-08-22T05:52:00Z'
has_clean_result: false
origin_prompt: '#2269 plan v3 §8 runtime-guard companion note: all three round-1 plan-critique
  lenses converged on a fail-loud/WARN guard in the regen mechanism when regen_applied:
  true lands with n_regen == 0 and regen_overlong_skipped > 0; recorded as OUT of
  #2269''s Goal scope and assigned to the orchestrator to file as its own kind: infra
  task (pod-side issue778_lib territory).'
workflow: v1
---
# Runtime guard: a cap-hit re-gen that skips every row must not report `regen_applied: true`

## Goal

Add a fail-loud (or at minimum WARN-in-the-run-report) guard to the cap-hit
re-generation mechanism so the state `regen_applied: true` ∧ `n_regen == 0` ∧
`regen_overlong_skipped > 0` can never pass silently. That triple is the exact
signature of a structurally inert re-gen leg: the trigger armed, the mechanism
ran, every candidate row was too long to regenerate under the engine's
`max_model_len`, and the payload nonetheless recorded the re-gen as applied —
so a run silently re-commits the very cap-hit deviation it claimed to fix.

## Why this is filed separately

This is the **higher-recall half** of the defense against the
#505 / #601 / #2221-v9 cap-raise-vs-`max_model_len` family. Task #2269 shipped
the other half: `verify_plan.py` check `c69`, a WARN at plan-verification time
when a plan declares an ARMED ≥2× re-gen trigger whose
`max_model_len − 2×max_new_tokens` leaves no room for the stated prompt bound.

`c69` covers the **plan-prose face only**, and that limit is explicit in
#2269's own scope: 2 of the 3 recurrences (#505, #601) were **code-level** — a
raised cap against an inherited engine pin, invisible to any plan-text check —
so `c69` could not have fired on them. A runtime guard would catch every
instance of the class, including the code-level face and any plan phrasing
outside `c69`'s calibrated grammar.

All three #2269 round-1 plan-critique lenses (Methodology, Statistics,
Alternatives) converged on this independently; #2269's plan §8 records it as
out of that task's Goal scope and assigns the filing to the orchestrator.

## Evidence that the failure mode is real, not hypothetical

The founding incident's own code already documents it. `scripts/issue2221_trait_eval.py:310-320`:

> ``--regen-max-new-tokens`` (default 2x the gen cap, per the CLAUDE.md re-gen
> rule) on a DEDICATED ``--regen-max-model-len`` = 8192 engine. The default
> engine's ``max_model_len`` = 4096 pin made
> ``budget = max_model_len - regen_cap`` = 0, so the r-parent's regen leg was
> **structurally inert (every row ``regen_overlong_skipped``** — the v10
> Must-Fix).

So the inert-regen state was reached in practice and was caught only by a human
reading the engine builder against the regen docstring.

## Candidate surfaces (for the planner to confirm — not a prescribed design)

- `scripts/issue778_lib.py` — the re-gen mechanism the #2269 plan names as the
  territory for this guard (`:239` already reasons about when a triggered row
  "would be `regen_overlong_skipped`").
- `scripts/issue2221_stage_corpus.py:985` — `_regen_cell`, the per-cell re-gen
  entry point.
- `scripts/issue2221_trait_eval.py:349-430` — where `regen_applied` and
  `regen_overlong_skipped` are written; `:349` is the `regen_applied`
  idempotent-skip read, `:420-430` the payload/report write.

Open design questions the plan should settle: fail-loud vs WARN (a hard raise
mid-run forfeits completed generation work, so the run-report/digest WARN may
be the better default per the project's advisory-over-abort posture for
diagnostic reads); whether the guard belongs at the write site, the report
composition, or both; and whether the same predicate should join the
upload-verification / digest surface so an inert re-gen is visible after the
fact as well as during.

## Scope boundary

Runtime only. Do NOT re-litigate or widen `verify_plan.py` `c69` here — that
check shipped in #2269 and its two known latent limitations (the `±3`-raw-line
cap window is not fence-masked; a non-2× shorthand multiplier falls to the
`2.0` default) are recorded against #2269, not this task.

## Provenance

Filed by the #2269 orchestrator per that task's plan v3 §8 (plan.md:13, :525),
which assigns the filing to the orchestrator and scopes it as
"pod-side `issue778_lib` territory". Converged on by all three #2269 round-1
plan-critique lenses; re-flagged by the #2269 Step 5 Claude code-reviewer as an
orchestrator action not to drop.
