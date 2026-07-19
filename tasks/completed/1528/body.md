---
title: 'workflow-fix: guard deny sidecar JSONL'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f054437ba925
- daily-auto-filed
created_at: '2026-07-19T07:05:46Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): Guard-denied commands are
  not persisted anywhere structured; false-positive attribution requires ad-hoc transcript
  mining and pre-fix filings mis-attribute incident classes (#1501 A-11).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1501 (emitting agent: Phase-2 Alternatives critic, refined by
the #1501 session; parked under the recursion guard, routed by the 2026-07-18
/daily Step C parked-candidate sweep).

## Goal

On every deny, `scripts/guard_repo_root_branch.sh` appends one JSON row to a
sidecar (e.g. `.claude/cache/guard-deny-events.jsonl`) recording ts, matched
detector arm, refusal-ladder check id, command length, and a bounded redacted
head — no full command text.

## Workflow gap

- **Bug observed:** Guard-denied commands are not persisted anywhere
  structured; incident-class attribution for false-positive filings requires
  ad-hoc session-transcript mining (done manually for #1501's three
  incidents), and pre-fix filings mis-attribute classes (#1501's own filing
  did).
- **Why it is a workflow gap:** The guard is a workflow-surface hook whose
  false positives drive workflow-fix filings; without a denial sidecar
  (ts + matched arm + ~120-char redacted head), every future filing repeats
  the unverifiable-incident problem (#1501 A-11).
- **Confidence (emitter):** low
- verified-at-filing: `grep -cn 'deny-events\|guard-deny\|jsonl' scripts/guard_repo_root_branch.sh` → 0 hits (absence-of-guard claim: no sidecar write exists); `grep -c 'exit 2' scripts/guard_repo_root_branch.sh` → 3 deny sites present; `git log --oneline --since='7 days ago' -- scripts/guard_repo_root_branch.sh` → 4 commits (d534bdf299 heredoc-check narrowing, c55596834c, 124e9bec60, a87f5898f6), none adds denial logging (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
+ # after the BLOCK message, before exit 2:
+ printf '%s\n' "{\"ts\":\"$(date -u +%FT%TZ)\",\"arm\":\"$ARM\",\"len\":${#cmd},\"head\":$(head-redact)}" >> "$REPO/.claude/cache/guard-deny-events.jsonl" 2>/dev/null || true
```

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'guard_repo_root_branch' .claude/ CLAUDE.md scripts/`) and check
  whether the sibling guards (`guard_repo_root_pull.sh`,
  `guard_piped_git_push.sh`, `guard_harmful_bank_read.sh`) should share the
  sidecar convention; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The sidecar append is best-effort (`|| true`) — it must NEVER change the
  guard's deny/allow decision, exit codes, or block a deny on a write failure.
- No full command text in the sidecar (trigger-density + secrets); bounded
  redacted head only.
- `scripts/workflow_lint.py --check-asks` passes; touched-file lint passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: scripts/guard_repo_root_branch.sh
- fingerprint: 73ac0c311752

<!-- workflow-fix-candidate v1 -->
target_file: scripts/guard_repo_root_branch.sh
bug_observed: Guard-denied commands are not persisted anywhere structured; incident-class attribution for false-positive filings requires ad-hoc session-transcript mining (done manually for #1501's three incidents), and pre-fix filings mis-attribute classes (this task's filing did).
why_workflow_gap: The guard is a workflow-surface hook whose false positives drive workflow-fix filings; without a denial sidecar (ts + command hash + matched arm + ~120-char redacted head), every future filing repeats the unverifiable-incident problem (#1501 A-11).
proposed_change: On every deny, append one JSON row to a sidecar (e.g. .claude/cache/guard-deny-events.jsonl) recording ts, matched detector arm, refusal-ladder check id, command length, and a bounded redacted head — no full command text (trigger-density + secrets).
diff_sketch: |
  + # after the BLOCK message, before exit 2:
  + printf '%s\n' "{\"ts\":\"$(date -u +%FT%TZ)\",\"arm\":\"$ARM\",\"len\":${#cmd},\"head\":$(head-redact)}" >> "$REPO/.claude/cache/guard-deny-events.jsonl" 2>/dev/null || true
confidence: low
related_task: #1501
<!-- /workflow-fix-candidate -->
