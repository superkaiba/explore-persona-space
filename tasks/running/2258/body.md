---
title: Extend verify_task_body.py figure-text opaque-code token grammar with the single-letter-prefix
  arm-slug class
kind: infra
tags: []
created_at: '2026-08-12T23:56:24Z'
has_clean_result: false
origin_prompt: 'clean-result-critic workflow-fix prose follow-up on #2221 round 1
  (2026-08-12): the figure-text opaque-code check caught mistake_gsm8k but missed
  the a_rb_ctx/c_map_ctx/c_map_pfx arm-slug class in the same body''s checkpoint-detection
  legend'
workflow: v1
---
# Extend verify_task_body.py figure-text opaque-code token grammar with the single-letter-prefix arm-slug class

## Goal

`verify_task_body.py`'s "figure text opaque config codes" check scans rendered figure text for opaque slug tokens, but its token grammar missed the arm-slug class `a_rb_ctx` / `b_rb_ans` / `c_map_ctx` / `c_map_pfx` / `d_transport` (single-letter prefix + underscore-joined stems) while catching `mistake_gsm8k` in the SAME body's figure set. Demonstrated on task #2221 round 1: the check WARNed only on `trait_mix_size_vs_acquisition.png` (`mistake_gsm8k`) while `checkpoint_detection_auc.png` rendered raw arm slugs `a_rb_ctx`/`c_map_ctx`/`c_map_pfx` in its legend and passed silently — the clean-result-critic's Lens 3 grammar caught what the mechanical check did not (clean-result-critique v1 on #2221, 2026-08-12).

Fix: extend the check's token classes with the single-letter-prefix underscore arm-slug shape (e.g. `^[a-z]_[a-z0-9]+(_[a-z0-9]+)+$` alongside the existing classes), with a test reproducing the #2221 legend miss. Keep false-positive guard: common English single-letter contractions do not match this shape; run the check against a sample of recent v4 bodies to confirm no new spurious WARNs.

## Provenance

Surfaced as a workflow-fix prose follow-up by the `clean-result-critic` on #2221 (round 1, 2026-08-12); routed by the #2221 orchestrator per `.claude/rules/workflow-fix-on-bug.md` (prose follow-ups auto-file + spawn). Target file: `scripts/verify_task_body.py` (figure-text opaque-code check). Candidate fingerprint: figure-text-opaque-codes/arm-slug-single-letter-prefix-class.
