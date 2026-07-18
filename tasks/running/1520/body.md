---
title: 'workflow-fix: verify_task_body brace-expanded HF path resolution against adjacent
  /tree pins'
kind: infra
tags:
- wf-fix
- wf-fix-fp:47fb1fbbf8df
created_at: '2026-07-18T18:47:12Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1426 fold r1 Lens 5a mechanizable prose (sampled_rollout/seed{42,137}
  404 at the c244377f pin)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a clean-result-critic prose follow-up on task #1426 (fold round r1, Lens 5a, mechanizable: yes).

## Goal

Extend `verify_task_body.py`'s HF-pin link checking to brace-expanded directory paths: expand `{a,b}` tokens in backtick HF paths adjacent to a `/tree/<rev>` pin, resolve each expansion against that pin's revision via `get_paths_info(..., revision=rev)`, FAIL on 404.

## Workflow gap

- **Bug observed:** the #1426 sampled-rollout fold cited `sampled_rollout/seed{42,137}/` under the body's prefix link pinned at revision `c244377f…` — that revision predates the round's upload, so the path 404s at the pin; no mechanical check caught it (found only by the adversarial critic re-resolving the path).
- **Why it is a workflow gap:** check 8/16-family link-liveness probes literal URLs; a brace-expanded backtick path riding an ADJACENT pin is a recurring body idiom (multi-seed/multi-arm rounds) with no resolution check, so a stale pin silently unpins a whole round's raw completions.
- **Confidence (emitter):** high (critic verified the 404 via the HF API; the fix revision `31d4fb5c…` contains the path)
- verified-at-filing: `grep -n '/tree/<sha>\|get_paths_info' scripts/verify_task_body.py` → the existing checks cover literal `/tree/<sha>/<path>` URLs (lines 388-401) and count-noun claims adjacent to hex-pinned links (lines 551-560); no brace-expansion of backtick paths against an adjacent pin anywhere (2026-07-18).

## Proposed change (candidate diff sketch — refine in planning)

```
+ in the pinned-link check family: for each backtick path containing {..,..}
+   adjacent (same bullet/sentence) to a /tree/<rev> HF link:
+     for expansion in brace_expand(path):
+         probe get_paths_info(repo, expansion, revision=rev) -> FAIL on 404/empty
```

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 47fb1fbbf8df

Surfaced prose (clean-result-critic #1426 fold r1): "mechanizable: yes — extend the existing HF-adjacent backtick-path check in verify_task_body.py to brace-expanded directory paths: expand {a,b} tokens, resolve each against the nearest-adjacent /tree/<rev> pin via get_paths_info(..., revision=rev), FAIL on 404."
