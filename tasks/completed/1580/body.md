---
title: 'daily-fix: reconcile wf-fix-fp tag with body fingerprint'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e45accafcda2
- daily-auto-filed
created_at: '2026-07-21T06:42:50Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-20 problem sweep (route 2): daily_drive_filings.py
  injects the body Provenance fingerprint only when absent but always computes the
  wf-fix-fp tag from manifest fields, so a body carrying its own fingerprint line
  lands with tag != body Provenance fingerprint, breaking the (target_file, fingerprint)
  dedup predicate; 17/17 of the 2026-07-19 filings and tonight''s #1579 are mismatched'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-20 from a parked candidate on task #1570 (its consistency-checker, 2026-07-20T08:08:26Z, recorded-not-routed under the recursion guard) plus a same-night fleet scan confirming the defect is systematic.

## Goal

Make `scripts/daily_drive_filings.py` reconcile the body-side `## Provenance` `- fingerprint:` line with the `wf-fix-fp:<fp>` tag it applies — overwrite the body line to the tag value (or fail loud on mismatch) — so the exact `(target_file, fingerprint)` dedup predicate is never split across two disagreeing values.

## Workflow gap

- **Bug observed:** #1570's consistency-checker flagged: "the task's frontmatter tag wf-fix-fp:f55e38afc131 disagrees with its Provenance fingerprint: 44d3a4598f5c". A filing-time scan tonight found ALL 17 route-2 filings of 2026-07-19 (#1554–#1571) carry tag≠body-fingerprint mismatches, plus tonight's #1579 (whose body carried the sweep-computed fingerprint of its formal candidate block while the driver tagged from manifest bug/change fields).
- **Why it is a workflow gap:** the driver injects `- fingerprint:` only when ABSENT (`daily_drive_filings.py:228` `if "fingerprint:" not in text`) but always computes the TAG from manifest fields (`:944` `fp = wf_fix_fingerprint(item["change"], item["bug"])`) — two sources of truth. The dedup predicate requires the tag AND the Provenance line to match the candidate (`workflow-fix-on-bug.md` § Dedup), so a mismatched pair breaks dedup for that task: a genuine re-raise of the same bug will not be suppressed.
- **Confidence (emitter):** high (mechanism located; 18/18 mismatches explained)
- verified-at-filing: `grep -n 'fingerprint' scripts/daily_drive_filings.py` → injection is absent-gated at :228-229 while the tag fp is manifest-computed at :944; fleet scan (tag from body.md `wf-fix-fp:` vs `- fingerprint:` line) → 17/17 mismatches on #1554–#1571 and on #1579 (body-carried fp), while #1574/#1575 (driver-injected, no pre-existing body fp) MATCH — confirming the two-sources-of-truth mechanism (2026-07-21).

## Proposed change (candidate diff sketch — refine in planning)

```
In daily_drive_filings.py body normalization:
- if "fingerprint:" not in text: append "- fingerprint: {fp}"
+ if "fingerprint:" not in text: append "- fingerprint: {fp}"
+ elif body fp != fp: rewrite the body line to fp (or ERROR the slug) + WARN
```
Also consider a one-shot repair sweep for the 18 landed mismatched tasks (or document that the tag is authoritative).

## Scope / surfaces

- Primary target: `scripts/daily_drive_filings.py` (+ `tests/test_daily_drive_filings.py` pins if present)

## Constraints / invariants

- Route-2 tags/injection stay gated on the manifest `wf_fix` flag (#1228 behavior unchanged).

## Provenance

- workflow_fix_target: scripts/daily_drive_filings.py

Verbatim parked candidate (recorded on #1570, 2026-07-20T08:08:26Z):

> **Recorded, not routed** (wf-fix recursion guard): the task's frontmatter tag `wf-fix-fp:f55e38afc131` disagrees with its Provenance `fingerprint: 44d3a4598f5c` — noted in the `epm:consistency` marker for the audit trail.
