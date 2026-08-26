---
title: 'verify_task_body: caption range-claim (N-M pairs per group) reconciliation
  coverage'
kind: infra
tags: []
created_at: '2026-08-26T00:24:26Z'
has_clean_result: false
workflow: v1
---
## Goal
Extend verify_task_body.py's caption count-claim coverage (the check-45 family) to range-shaped per-group pair-count claims of the form 'N-M pairs per <group>' / 'N and M pairs per <family>' in figure captions and Takeaways caveats, reconciling them against the round's pinned eval artifacts when resolvable.

## Provenance
Surfaced by clean-result-critic round 5 on #2378 (fold re-gate, 2026-08-25): three caption/caveat range claims ('8-12 per story family', '60 and 30 pairs per chat-plain family', 'story-family screen cells carry 8-12 pairs') contradicted screen_report.json n_pairs values (actual 5-12; plotted columns 43 and 8) and were all mechanically reconcilable against eval_results/issue_2378/causal-patching-arms/{screen_report,patch_summary}.json. The existing caption count-claim check does not parse range-shaped claims, so all three shipped to the CRC gate instead of failing at the analyzer's verifier pre-pass.

## Acceptance
- Range claims 'N-M <unit> per <group>' and 'N and M <unit> per <group>' in captions/Takeaways are parsed and, when a matching per-group count field is resolvable in the task's pinned eval JSONs, checked min/max against it (WARN or FAIL per the check-45 family's existing severity convention).
- Unresolvable claims stay silent (no false positives on prose ranges without a matching artifact).
- Regression test reproducing the #2378 r5 shape (caption says 8-12, artifact says 5-12) FAILs pre-fix, PASSes post-fix.
