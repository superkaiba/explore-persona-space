---
title: 'workflow-fix: WARN on linked-not-embedded committed figures in Results'
kind: infra
tags:
- wf-fix
- wf-fix-fp:387b41d92415
created_at: '2026-07-15T23:44:31Z'
has_clean_result: false
origin_prompt: 'clean-result-critic r1 prose follow-up on #1315: figures/issue_N png
  referenced as link not embed'
workflow: v1
---
## Overview / Motivation
Auto-filed from a clean-result-critic prose follow-up on task #1315 (round 1, mechanizable: yes).
## Goal
verify_task_body.py: WARN on a non-image markdown link to `figures/issue_<N>/*.png` inside `## Results` — a committed per-unit figure referenced as a link instead of embedded.
## Workflow gap
- **Bug observed:** #1315 result 4 referenced the committed per-row PC-scatter grid as a markdown LINK instead of an inline embedded image; Lens 11 caught it only by LM judgment.
- **Why it is a workflow gap:** No mechanical check flags link-not-embed for committed per-unit figures, so the underlying-data-alongside-aggregate discipline leaks through the verifier.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c "figures/issue" scripts/verify_task_body.py` → 25 hits (figure-path machinery present); no non-image-link WARN pattern exists among them (absence claim, inspected at filing) (2026-07-15)
## Proposed change (candidate diff sketch — refine in planning)
+ WARN per match of regex [^!]\[[^\]]*\]\([^)]*figures/issue_\d+/[^)]*\.png\) inside the ## Results section:
+ "committed figure referenced as a link (embed it inline or justify)"
## Scope / surfaces
- Primary target: `scripts/verify_task_body.py`; tests in `tests/test_verify_task_body.py`.
## Constraints / invariants
- WARN-tier (never FAIL — a deliberate link can be acknowledged); recursion guard applies.
## Provenance
- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 387b41d92415
