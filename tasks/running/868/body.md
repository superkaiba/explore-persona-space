---
title: 'extraction.py docstring: last-layer hook capture is pre-final-norm (not identical
  to hidden_states[-1])'
kind: infra
tags: []
created_at: '2026-07-02T14:27:21Z'
has_clean_result: false
origin_prompt: '#779 round-6 implementer (d) follow-up: #671 docstring ''byte-identical''
  claim is real-model-false for the last layer'
---
## Overview / Motivation

Filed from a "Needs human eyeball" prose follow-up in #779 round-6 implementation (experiment-implementer): a documentation inaccuracy in a shared src/ helper that could mislead future capture work.

## Goal

Fix the `extract_layer_activations` hook-path docstring in `src/explore_persona_space/analysis/extraction.py`: the LAST-layer hook capture returns the RAW final-block residual (pre-final-norm), while `output_hidden_states[-1]` is POST-final-norm — the "#671 byte-identical to the full-tuple read" claim is stub-true (tiny random models) but real-model-FALSE for the last layer specifically.

## Detail

- Non-last layers ARE identical between the hook path and `hidden_states[idx]`.
- Only the last decoder block's hook output differs from `hidden_states[-1]` by the final RMSNorm.
- Pre-existing (NOT introduced by the #779 round-6 fix); pre/post-patch behavior identical (proved by the round-6 equivalence smoke). #779's own use is internally consistent (all its extractions ride the same hook convention).
- Fix scope: docstring + (optionally) a one-line comment at the hook registration; NO behavior change. Consider a short unit-test comment in tests/test_issue671_extraction_hooks.py clarifying the convention.

## Constraints

- Documentation-only; zero behavior change; ruff clean.
