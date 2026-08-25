---
title: 'Figure builders: emit tick/legend text to meta.json + WARN-lint opaque code-slug
  labels'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-14T13:15:45Z'
has_clean_result: false
origin_prompt: 'clean-result-critic Lens 3 on #2254: per_question_dots.png kept opaque
  code-slug tick labels after the round-2 relabel pass'
workflow: v1
---
## Goal
Make figure builders emit their axis tick-labels + legend text into the `meta.json` provenance sidecar, so a lightweight mechanical lens can flag opaque code-slug tick labels (e.g. `a0` / `pre@context` / `ctxext@context` / `rb@answer`) before a human review. On #2254 the round-2 relabel pass re-rendered most figures with reader-facing labels but missed `per_question_dots.png`, which kept code-slug ticks; only clean-result-critic Lens 3 caught it.

## Scope
- Extend the shared figure-emit helper (the `savefig_paper` / meta.json sidecar path in scripts/issue2254_figures.py's builders, or the shared paper-plots helper if one exists) to record `xtick_labels` / `ytick_labels` / `legend_labels` in meta.json.
- Add a WARN-only lint (or extend an existing figure lint) that flags tick/legend strings matching the opaque-code-slug pattern (short lowercase alnum with `@`/`_`/digit-suffix, no spaces) so relabel-pass misses surface mechanically.
- Do NOT hard-block (labels are legitimately terse sometimes) — WARN + name the figure.

Provenance: surfaced by clean-result-critic Lens 3 on #2254 (workflow-fix-candidate, prose follow-up).
