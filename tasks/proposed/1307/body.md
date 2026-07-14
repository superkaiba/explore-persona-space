---
title: 'daily-fix: bind verified-at-filing greps to cited targets'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5316d50923d1
- daily-auto-filed
created_at: '2026-07-14T06:44:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-13 problem sweep (route 2): two 2026-07-12 nightly
  filings carried real verified-at-filing greps that did not bind to the claim: #1290
  verified the pattern repo-wide while target_file named a 0-hit file, and #1296 asserted
  a cited test no longer exists without a relocation grep (it existed at another path)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-13 from the transcript problem sweep (sessions eafc501a → task #1290 and 384f17c7 → task #1296): two of the 2026-07-12 nightly's filings carried real `verified-at-filing:` greps that nonetheless did not bind to the body's claim, each costing the spawned session a clarifier / fact-checker correction round.

## Goal

Strengthen the `verified-at-filing:` mandate so the compose-time grep BINDS to the claim: per-target-file pattern confirmation, and a mandatory relocation grep before asserting a cited symbol/test nonexistent.

## Workflow gap

- **Bug observed:** (a) #1290's body verified its pattern repo-wide (`grep -rn "get('verdict')" .claude/ scripts/` → 0 workflow-surface hits) while `target_file` named `issue/SKILL.md` — a file with 0 `verify_plan` hits; the real parse site was `adversarial-planner/SKILL.md`, found by the spawned session's clarifier. (b) #1296's body asserted the cited failing test "NO LONGER EXISTS" after a single-file pytest probe, without a repo-wide relocation grep; the fact-checker found it at `tests/test_issue_dispatch.py:1164`.
- **Why it is a workflow gap:** the #1272 mandate requires a grep + hit count "consistent with the body's bug claim and target_file list", but neither failure mode above violates its letter — the greps ran and were recorded; they just were not required to bind per named target or to search for a relocated symbol. The gap survives the existing rule text.
- **Confidence (emitter):** medium-high (two same-night incidents, both caught downstream at one-round cost)
- verified-at-filing: `grep -A2 "verified-at-filing" tasks/completed/1290/body.md tasks/completed/1296/body.md` → both lines present and quoted above; #1290's records a repo-wide pattern grep with target_file unbound; #1296's records "the candidate's cited failing test NO LONGER EXISTS" with no relocation grep (2026-07-14 UTC).

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/skills/daily/SKILL.md` (verified-at-filing mandate, route 2) and `.claude/rules/workflow-fix-on-bug.md` (§ Body-file template):

```
+ The grep must BIND to the claim: (a) run it against EACH file named in
+ target_file — a 0-hit named target is a mis-target: re-grep repo-wide,
+ correct target_file, and re-verify BEFORE filing; (b) asserting a cited
+ symbol/test/file nonexistent requires a repo-wide relocation grep
+ (grep -rn '<symbol>' tests/ scripts/ .claude/ src/) recorded in the line —
+ a single-path probe cannot distinguish "removed" from "moved".
```

## Scope / surfaces

- Primary targets: `.claude/skills/daily/SKILL.md`, `.claude/rules/workflow-fix-on-bug.md`

## Constraints / invariants

- Prose-mandate change only (no driver/injector code change required; the planner may optionally add a driver WARN).
- `scripts/workflow_lint.py` default run passes.
- Recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: .claude/skills/daily/SKILL.md, .claude/rules/workflow-fix-on-bug.md
- fingerprint: 5316d50923d1

Origin: transcript-mined (#1290 clarifier redirect ~06:57-07:06Z; #1296 fact-checker correction ~09:44Z). Not a parked candidate — surfaced by the /daily problem sweep.
