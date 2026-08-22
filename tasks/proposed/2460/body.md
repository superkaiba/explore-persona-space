---
title: savefig_paper caption-visibility restore is not exception-safe under EPS_PLOT_NO_CAPTION=1
kind: infra
tags: []
created_at: '2026-08-22T02:01:04Z'
has_clean_result: false
origin_prompt: 'Codex twin concern savefig-caption-restore-not-exception-safe, raised
  during #2262 round-3 code review'
workflow: v1
---
<!-- workflow-fix-candidate v1 -->
target_file: src/explore_persona_space/analysis/paper_plots.py
problem: `savefig_paper`'s caption-hiding path is not exception-safe. Under `EPS_PLOT_NO_CAPTION=1` the function temporarily hides caption artists before saving and restores their visibility afterward, but the restore is not guarded by `try/finally`. If the save itself raises — a bad path, a permissions error, an out-of-space condition, a backend failure — the function unwinds with the caption artists left INVISIBLE on the still-live `Figure` object. Any subsequent save of that same figure (a retry, a second format, a caller that catches and re-saves) then silently produces a caption-less figure, with no error and nothing in the sidecar to indicate the caption was dropped.
fix_sketch: wrap the hide/restore span in `try/finally` so visibility is restored on every exit path, including exceptions. Add a regression test that monkeypatches the save call to raise while `EPS_PLOT_NO_CAPTION=1` is set, then asserts every caption artist's `get_visible()` is back to its pre-call value. Check whether the same non-exception-safe pattern appears in any sibling temporary-mutation span in the same function (presentation-env overrides, per-format provenance embed) — if so, cover them in the same pass.
confidence: high
<!-- /workflow-fix-candidate -->

## Provenance

Surfaced by the Codex twin (`epm:code-review-codex v3`) during the round-3 code review of task #2262, as machine-readable concern `savefig-caption-restore-not-exception-safe`. Codex itself routed it as "address separately with finally cleanup".

It is PRE-EXISTING and entirely unrelated to #2262's diff: #2262's final round changed exactly one line — a `# noqa: C901` comment on `savefig_paper`'s `def` line — and its substantive rounds touched only `_extract_scatters` / `_has_explicit_offsets`. Nothing in #2262 caused or worsened this defect.

Filed as its own task rather than absorbed into #2262 because #2262's Goal is the `_extract_scatters` sidecar-capture bug; a caption-visibility restore bug would change that Goal. The concern remains recorded in #2262's ledger with a pointer here.

This is library code under `src/explore_persona_space/analysis/`, NOT workflow surface, so the workflow-fix-on-bug auto-file-and-spawn protocol does not route it — hence a plain `proposed` capture with no session spawned. Dispatch is the user's / PM's call.

## Why it is worth fixing

`savefig_paper` is the project's single figure-save entry point, so a silent caption drop is a correctness risk for any reader-facing artifact rendered through a retry path. The failure mode is silent by construction: the sidecar records no marker for a dropped caption, so a caption-less figure is indistinguishable from an intentionally caption-free one after the fact.
