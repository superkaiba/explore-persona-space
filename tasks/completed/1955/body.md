---
title: 'workflow-fix: gotchas.md — numpy argsort SIMD tie-order trap for cross-machine
  top-k identity gates'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1dd0817f47d4
created_at: '2026-08-01T01:15:59Z'
has_clean_result: false
origin_prompt: 'gotcha_candidate from #1946 epm:failure-lesson v1: numpy argsort tie
  order is CPU-SIMD-kernel dependent; recompute-equality gates on banked top-k selections
  are machine-dependent under boundary ties.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a gotcha_candidate failure-lesson raised on task #1946 (emitting agent: experiment-implementer; epm:failure-lesson v1, 2026-08-01).

## Goal

Add a gotchas.md entry: numpy argsort tie order is CPU-SIMD-kernel dependent; cross-machine identity gates on top-k-by-count selections must use set-validity invariants, never recompute-equality.

## Workflow gap

- **Bug observed:** a recompute-equality f_out gate passed on the VM and crashed on GCE with byte-identical inputs (5 boundary ties for 2 cap slots; numpy 2.x x86-simd-sort dispatch differs by CPU).
- **Why it is a workflow gap:** .claude/rules/gotchas.md documents cross-machine/codebase traps for implementers; this class (ranked-selection recompute gates over tied counts) is undocumented and will recur wherever a banked top-k artifact is re-verified on a different lane (VM vs GCE vs pod CPUs differ in AVX-512).
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'argsort' .claude/rules/gotchas.md` → 0 hits (2026-08-01); incident forensics on #1946 events (epm:failure v1 + epm:failure-lesson v1); live cross-machine repro measured (VM PASS / GCE FAIL, identical numpy 2.2.6).

## Proposed change (candidate diff sketch — refine in planning)

+ gotchas.md new bullet (analysis/orchestration section): "numpy argsort/sort tie order is CPU-SIMD-kernel dependent (numpy 2.x x86-simd-sort: AVX-512 vs not) — a cross-machine identity gate that RECOMPUTES a top-k-by-count selection and asserts equality against a banked selection is machine-dependent whenever counts tie at the cap boundary. Gate banked ranked selections by set-validity invariants (floor match, len == min(cap, n_eligible), all >= floor, strictly-above-boundary ⊆ banked, n_above < cap) or create selections with kind='stable'. Worked fix: scripts/issue1946_sae_percontext.py _scan_and_gate (b9b7e7d982c9); incident #1946."

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Grep the workflow surface for the pattern before editing (`grep -rln 'argsort' .claude/ CLAUDE.md scripts/`) and consider whether plan-compute-sizing.md / code-style.md cross-references are warranted; list hits in the plan.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes; LESSONS.md index row for gotchas.md already exists (no index change needed unless the trigger line is extended).
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics via the Provenance line below (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 1dd0817f47d4

Verbatim origin lesson block: see epm:failure-lesson v1 on task #1946 (numpy argsort tie order CPU-SIMD dependence; banked-selection-authoritative + set-validity-invariants fix).
