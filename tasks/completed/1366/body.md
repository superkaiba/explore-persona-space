---
title: 'workflow-fix: artifact-reuse check for parent-branch unmerged fixes + row-count
  reconciliation'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2fdd300c8090
created_at: '2026-07-15T22:22:44Z'
has_clean_result: false
origin_prompt: 'failure-lesson from #1345 r6: parent''s crash-fix stranded on unmerged
  issue-825 branch; realized shard n=4724 vs corpus 5000 was the fingerprint'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson raised on task #1345 round 6 (emitting agent: experiment-implementer).

## Goal

Add an artifact-reuse fitness check: parent-branch unmerged fixes + realized-row-count reconciliation.

## Workflow gap

- **Bug observed:** #1345 reused #825's extractor from MAIN; the parent's own degenerate-row crash-fix (partition_rendered/degenerate_content_turns/span-integrity gate) exists only on the unmerged issue-825 branch. The parent's realized shards embody the filter (naturalistic_s n=4724 vs corpus 5000), so the reuse passed every existing fitness check yet crashed in production at the first unfiltered degenerate row (s57).
- **Why it is a workflow gap:** .claude/rules/artifact-reuse.md's checks (a)-(j) verify the ARTIFACT (recipe match, cells present, sha pins, fetchability, throughput) but never ask whether the reused CODE on main lags the parent branch's fixes, nor whether the realized artifact's row count reconciles with its declared input (a shortfall = filtering happened somewhere). The built-but-stranded-fixes lesson (workflow-fix-on-bug.md) names the class but no reuse-time check operationalizes it.
- **Confidence (emitter):** high
- verified-at-filing: `grep -c "issue branch\|unmerged" .claude/rules/artifact-reuse.md` → 0 hits (no parent-branch check exists; absence-of-guard claim — the 0-hit result IS the evidence) (2026-07-15).

## Proposed change (candidate diff sketch — refine in planning)

+ artifact-reuse.md new check (k): when reusing a parent's CODE module (not
+ just its artifacts), `git log --oneline main..origin/issue-<parent> -- <module>`
+ — any unmerged parent-branch commits touching the module must be inspected
+ and ported or explicitly declared not-needed; AND reconcile the realized
+ artifact's row/cell count against its declared input corpus — a shortfall
+ means a filter exists; find it and port it.

## Scope / surfaces

- Primary target: `.claude/rules/artifact-reuse.md`
- Also consider one line in `planner.md` step 5 (the self-attested fitness list) — grep first (`grep -rln 'fitness check' .claude/agents/planner.md .claude/rules/`).

## Constraints / invariants

- Workflow-surface only; lessons index untouched (artifact-reuse row exists).
- `scripts/workflow_lint.py` passes.
- Recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: .claude/rules/artifact-reuse.md
- fingerprint: 2fdd300c8090
