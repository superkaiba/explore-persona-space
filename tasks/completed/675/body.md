---
title: 'Route _context_vector_all_layers through hook helper (sibling of #671, surfaced
  by #672)'
kind: infra
tags: []
created_at: '2026-06-26T16:25:31Z'
has_clean_result: false
parent_id: 672
---
## Goal

Route `_context_vector_all_layers` (`scripts/issue667_extract.py` line 443, called for ALL behaviors at sites 870/871/1022/1023) through the hook-based `extract_layer_activations` helper, completing the #671 surface fix.

## Background

#671 routed the PRIMARY per-probe reads (`issue667_extract.py:423/424/688`) through `register_forward_hook` + `output_hidden_states=False` to remove the documented #545 22→38 GiB per-iteration memory climb. But the per-context reader `_context_vector_all_layers` (line 443) was NOT routed through the hook helper — it still calls `base_model(**ids, output_hidden_states=True)` and pulls all hidden-state tensors per call.

The residual call is transient/non-accumulating (single-batch allocation, frees on return; called ~6× per smoke), so it does NOT manufacture the per-iteration memory creep that #545 observed and #671 fixed. But it IS a residual gap on #671's surface: it still allocates all 29 hidden-state tensors transiently on every context read.

Fact-checker confirmed in #672 round 1 (verified by direct grep of `scripts/issue667_extract.py`).

## Acceptance / verdict

- `_context_vector_all_layers` call sites (870/871/1022/1023) replaced with `extract_layer_activations(base_model, ids, [layer])` invocations.
- Bit-for-bit equivalence test added to `tests/test_issue671_extraction_hooks.py` covering the `fact`-behavior path that exercises the new hook-routed read.
- Memory-non-growth assertion still passes across the full hook surface (no regression).
- `output_hidden_states=True` no longer appears anywhere in `scripts/issue667_extract.py`.

## Provenance

Surfaced as a follow-up during #672's planning + interpretation+critic loops. Both the analyzer and the round-1 clean-result-critic flagged it as a known residual gap on #671's surface. Parent: #671 (the original hook-fix). Sibling-of-parent follow-up.

## Scope / size

`kind: infra` — code-change path. CPU-only test changes. No GPU compute required. Auto-completes on test-verdict PASS.
