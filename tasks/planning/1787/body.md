---
title: 'daily-fix: pin prefix-less c43 escape in sync test'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3d85957300dd
- daily-auto-filed
created_at: '2026-07-29T07:03:09Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): the sync test test_skillmd_canonical_escapes_sync_with_docstring
  extracts only `N/A —` / `Durability pin: N/A`-prefixed canonical escape phrases,
  so the prefix-less c43 escape `no sentinel dependence — auto-safe` is not sync-pinned
  against its SKILL.md registration'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step C parked-candidate sweep (2026-07-28) from a formal candidate block parked on task #1777 (ts 2026-07-28T23:33:48Z, fp 3d85957300dd; source: code-reviewer round 1 prose-followup, confidence low). #1777 landed c43 (`verify_plan c43_sentinel_lane`, merged 2026-07-28 commit `12311b2bb6`); its reviewer flagged that the SKILL.md registration of c43's prefix-less canonical escape is not sync-pinned.

## Goal

Extend the `test_skillmd_canonical_escapes_sync_with_docstring` extraction (or add an explicit assert) in `tests/test_verify_plan.py` so the prefix-less c43 escape phrase `no sentinel dependence — auto-safe` is pinned against its `.claude/skills/adversarial-planner/SKILL.md` registration.

## Workflow gap

- **Bug observed:** the sync test extracts only `N/A —` / `Durability pin: N/A`-prefixed canonical escape phrases, so the prefix-less c43 escape `no sentinel dependence — auto-safe` is not sync-pinned against its SKILL.md registration.
- **Why it is a workflow gap:** a future `.claude/skills/adversarial-planner/SKILL.md` edit could silently drop the c43 canonical-escape bullet with no test failure (code-side recognition stays pinned; the SKILL.md registration does not).
- **Confidence (emitter):** low
- verified-at-filing: `grep -n 'no sentinel dependence' tests/test_verify_plan.py` → 0 hits (the phrase is unpinned in the test file — absence claim), while the phrase EXISTS at `.claude/skills/adversarial-planner/SKILL.md:360` and `scripts/verify_plan.py:187`; the sync test exists at `tests/test_verify_plan.py:1563` (2026-07-29 UTC). unverified hypothesis — verify at plan time: that the sync test's extraction regex is prefix-anchored as described (the emitter's claim about the extraction mechanism was not re-read from the test body at filing time — read the extraction code in `test_skillmd_canonical_escapes_sync_with_docstring` before writing the fix). Landed-fix history check: `git log --oneline --since='7 days ago' -- tests/test_verify_plan.py` → the #1777 merge itself (`12311b2bb6`); no follow-up pin commit.

## Proposed change (candidate diff sketch — refine in planning)

```
+ # in the sync test's extraction: also collect prefix-less canonical
+ # phrases (e.g. r"no sentinel dependence — auto-safe") or assert the
+ # c43 bullet's presence in SKILL.md explicitly.
```

## Scope / surfaces

- Primary target: `tests/test_verify_plan.py` (the `test_skillmd_canonical_escapes_sync_with_docstring` extraction at ~L1563)

## Constraints / invariants

- Test-only change; must not weaken existing pinned escapes.
- Workflow-surface only; recursion guard applies to the spawned session.

## Provenance

- sha-verify (filing-time, #1467): `3d85957300dd` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- workflow_fix_target: tests/test_verify_plan.py
- fingerprint: 3d85957300dd

<!-- workflow-fix-candidate v1 -->
target_file: tests/test_verify_plan.py
bug_observed: the sync test test_skillmd_canonical_escapes_sync_with_docstring extracts only `N/A —` / `Durability pin: N/A`-prefixed canonical escape phrases, so the prefix-less c43 escape `no sentinel dependence — auto-safe` is not sync-pinned against its SKILL.md registration
why_workflow_gap: a future .claude/skills/adversarial-planner/SKILL.md edit could silently drop the c43 canonical-escape bullet with no test failure (code-side recognition stays pinned; the SKILL.md registration does not)
proposed_change: extend the extraction regex in tests/test_verify_plan.py (or add a one-line explicit assert) to cover prefix-less canonical phrases
confidence: low
related_task: #1777
<!-- /workflow-fix-candidate -->
