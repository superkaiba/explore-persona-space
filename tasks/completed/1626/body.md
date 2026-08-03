---
title: 'daily-fix: workflow_lint root-guard rc=-1 retry-then-WARN'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e6b682bd78a7
- daily-auto-filed
- trigger-dense
created_at: '2026-07-23T07:01:33Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-22 problem sweep (route 2): the root-guard-hook probe
  counts a transient unexpected rc=-1 as a blocking FAIL line despite its own NON-BLOCKING
  wording, flipping a clean Step-9c gate to block (#1610, ~12 min churn)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-22 (transcript sweep). #1610's Step-9c lint gate BLOCKed round 1 on a FLAKY lint line: "root-guard hook returned unexpected rc=-1 — a NON-BLOCKING code under the PreToolUse contract" — the message itself says the rc is non-blocking, yet the check appends it to the FAIL list, and the gate burned a full re-run (~10–12 min) before passing.

## Goal

`workflow_lint.py`'s root-guard-hook probe treats a transient unexpected rc (rc=-1 / signal-kill / timeout) as retry-once-then-WARN instead of a FAIL line, so a killed/timed-out hook invocation inside a gate scratch tree cannot flip a clean gate to `block`.

## Workflow gap

- **Bug observed:** c809abbd (#1610), 2026-07-23T05:40:34Z: NEW lint line on the gated scratch tree `/tmp/issue-1610-lint-gate-tree/.claude/skills/issue/SKILL.md:: root-guard hook returned unexpected rc=-1`; verdict `block` vs tip 86f1f6b636; the lint-legs re-run at 05:41:22Z passed with no change → merged 05:52Z.
- **Why it is a workflow gap:** the check's own message classifies the rc as "a hook invocation or infrastructure error, not a pass" — an INFRASTRUCTURE error is INCONCLUSIVE-shaped, not FAIL-shaped; counting it as a blocking line makes the gate flaky under scratch-tree/timeout conditions.
- **Confidence:** high.
- verified-at-filing: `grep -n 'returned unexpected rc' scripts/workflow_lint.py` → scripts/workflow_lint.py:9566, and reading the surrounding block (9550–9572) confirms the unexpected-rc branch appends to `errors` (= FAIL lines) despite the "NON-BLOCKING code" wording (presence claim, context read and binds), 2026-07-23 UTC.

## Proposed change (refine in planning)

In the root-guard-hook check: on an unexpected rc, retry the hook invocation once; if still unexpected, emit a WARN-class line (not an error) naming the rc — preserving the rc==2 blocked-recipe FAIL behavior unchanged.

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py` (root-guard-hook probe) + its tests.

## Constraints / invariants

- rc==2 (genuinely blocked recipe fence) stays a FAIL. The no-flags default run stays green on the current tree. Recursion guard applies.

## Provenance

- sha-verify (filing-time, #1467): `c809abbd` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- fingerprint: e6b682bd78a7

- workflow_fix_target: scripts/workflow_lint.py
