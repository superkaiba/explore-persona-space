---
title: 'workflow-fix: /daily driver forwards filer advisories'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c4be1e68f648
- daily-auto-filed
created_at: '2026-07-19T07:05:54Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-18 problem sweep (route 2): The /daily route-2 driver
  captures filer output and persists only out[-300:] on success, so the filing-time
  sibling advisories (#1399/#1446 closed arm, #1502 open arm) are truncated out and
  never reach the /daily session.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1502 (emitting agent: methodology critic, Phase-2 review;
parked under the recursion guard, routed by the 2026-07-18 /daily Step C
parked-candidate sweep).

## Goal

Forward/expand the captured filer stderr on `scripts/daily_drive_filings.py`'s
success path (persist the ADVISORY block lines alongside the FILED line, or
raise the tail cap when 'ADVISORY' is present) so the /daily session sees the
filing-time sibling advisories.

## Workflow gap

- **Bug observed:** The /daily route-2 driver invokes `file_infra_task.py`
  with `capture_output=True` and persists only `out[-300:]` on success, so
  BOTH filing-time sibling advisories (#1399/#1446 closed arm, #1502 open arm)
  are truncated out of the success-path tail and never reach the /daily
  session.
- **Why it is a workflow gap:** The advisory channel's consumption contract is
  "the filer eyeballs stderr", but the highest-volume wf-fix filing channel
  (/daily route-2) structurally discards that stderr on success. The rule file
  itself documents this as a known limitation
  (workflow-fix-on-bug.md § Open-sibling arm, limitation (b)) — this candidate
  closes it.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n '\[-300:\]' scripts/daily_drive_filings.py` → 2 hits (L971 `"tail": tail.strip()[-300:]`, L980 `out[-300:]` on the FILED ledger row — truncation present as claimed); `grep -n 'ADVISORY' scripts/daily_drive_filings.py` → hits only for the #1467 SHA_ADVISORY_TMPL (L185/L321), no sibling-advisory forwarding on the success path; `git log --oneline --since='7 days ago' -- scripts/daily_drive_filings.py` → 2 commits (07e0d269fe route-3 overlap dedup, dc3a465ca2 sha-verify), neither forwards filer stderr advisories (2026-07-19)

## Proposed change (candidate diff sketch — refine in planning)

```
# daily_drive_filings.py ~L977-1002 (success path):
+ adv = [l for l in err.splitlines() if 'ADVISORY' in l or l.startswith('  #')]
+ if adv: persist/print adv alongside 'FILED {slug} -> #{tid}'
```

## Scope / surfaces

- Primary target: `scripts/daily_drive_filings.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'filer-stderr-only\|Open-sibling arm' .claude/ CLAUDE.md scripts/`)
  — update `.claude/rules/workflow-fix-on-bug.md`'s "accepted-limitation"
  sentence in § Open-sibling arm (limitation (b)) if the fix lands; list every
  hit in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- The ledger row schema additions must stay backward-compatible with
  `_try_recovery` and existing `filed.jsonl` consumers; exit codes and the
  must-succeed filing half unchanged.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard, § Recursion guard).

## Provenance

- workflow_fix_target: scripts/daily_drive_filings.py
- fingerprint: da5ca273df8e

<!-- workflow-fix-candidate v1 -->
target_file: scripts/daily_drive_filings.py
bug_observed: The /daily route-2 driver invokes file_infra_task.py with capture_output=True and persists only out[-300:] on success, so BOTH filing-time sibling advisories (#1399/#1446 closed arm, #1502 open arm) are truncated out of the success-path tail and never reach the /daily session.
why_workflow_gap: The advisory channel's consumption contract is "the filer eyeballs stderr", but the highest-volume wf-fix filing channel (/daily route-2) structurally discards that stderr on success.
proposed_change: Forward/expand the captured filer stderr on the success path (e.g. persist the ADVISORY block lines alongside the FILED line, or raise the tail cap when 'ADVISORY' is present) so the /daily session sees sibling advisories.
diff_sketch: |
  # daily_drive_filings.py ~L977-1002 (success path):
  + adv = [l for l in err.splitlines() if 'ADVISORY' in l or l.startswith('  #')]
  + if adv: persist/print adv alongside 'FILED {slug} -> #{tid}'
confidence: medium
related_task: #1502
<!-- /workflow-fix-candidate -->
