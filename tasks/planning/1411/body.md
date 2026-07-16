---
title: 'daily-fix: gotchas Edit tool un-escapes uXXXX'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1992e5058839
- daily-auto-filed
created_at: '2026-07-16T07:22:22Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): #1364: the Edit tool silently
  un-escaped a \uXXXX char class (plan file''s verbatim class also NFC-corrupted U+F900->U+8C48);
  a stray char corrupted a shell command'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 problem sweep (route 2 — behavior/logic change, independent review required).

## Goal

One-line gotchas entry: write Unicode-sensitive literals via python heredoc / escaped-source writes, never via Edit-tool raw escapes (the Edit tool silently un-escapes `\uXXXX`); verify byte-level after landing.

## Workflow gap

- **Bug observed:** in #1364 the Edit tool silently un-escaped a `\uXXXX` char class (and the plan file's "verbatim" class was itself NFC-corrupted, U+F900→U+8C48); a stray char also corrupted a shell command ("«: command not found") (6b26371a 22:13-22:43Z).
- **Why it is a workflow gap:** gotchas.md documents the JSONL splitlines Unicode trap but nothing about the Edit tool's escape/normalization behavior — so Unicode-sensitive literal edits keep silently corrupting char classes.
- **Severity:** low
- verified-at-filing: `grep -n 'uXXXX\|un-escap\|unescap' .claude/rules/gotchas.md` → 0 hits — entry absent (the L143 Unicode hit is the distinct splitlines/U+2028 JSONL entry, not Edit-tool escape behavior) (2026-07-16 UTC).

## Proposed change (refine in planning)

Add one entry to `.claude/rules/gotchas.md`: "The Edit tool un-escapes `\uXXXX` literals (and pasted 'verbatim' text can arrive NFC-normalized, e.g. U+F900→U+8C48) — write Unicode-sensitive literals (regex char classes, marker tokens, test fixtures) via a python heredoc or an escaped-source Write, never Edit-tool raw escapes, and verify byte-level after landing (`grep -P`, `xxd`, or a python ord() probe). A stray un-escaped char can also corrupt an adjacent shell command (#1364, 2026-07-15: '«: command not found')."

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 1992e5058839

- workflow_fix_target: .claude/rules/gotchas.md

Mined from 2026-07-15 session transcripts by the /daily problem sweep. Evidence: 6b26371a (#1364) 22:13-22:43Z (batch 04 P19).
