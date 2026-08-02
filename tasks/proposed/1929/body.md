---
title: 'daily-fix: detached relaunch + worker-pid + named-verify'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c8acd2997388
- daily-auto-filed
created_at: '2026-07-31T06:59:56Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-30 problem sweep (route 2): three detached-launch bookkeeping
  failures in one day: a hand-re-typed relaunch chain dropped the completion-sentinel
  write (#1768), a completion watch keyed on the setsid launcher pid false-fired EXITED
  on a live batch (#1769), and a vacuous push-verify left 29 git-bound eval files
  uncommitted (#1482).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-30 (problem sweep; miner-5 P2(a), miner-3 P9, miner-4 P13 — three detached-launch bookkeeping failures in one day).

## Goal

Harden the detached-launch discipline in the pod-side-reporting rule + the /issue detached-phase recipe: (a) a RELAUNCH re-runs the original launcher file, never a hand-re-typed `bash -c` chain; (b) pid breadcrumbs + completion watches key on the WORKER pid, never the setsid/nohup launcher pid; (c) a push/upload-verify step names its expected paths — an empty-set verify is vacuous.

## Workflow gap

- **Bug observed:** (a) #1768's pod-side wedge recovery relaunch was a hand-rebuilt `bash -c` chain that DROPPED the completion-sentinel write, stranding the handoff (plus no run-launched marker, no live poller — found by the successor session 5.8h later). (b) #1769's mt600 re-judge watch keyed on the setsid launcher pid and false-fired "EXITED" ~1 min after launch while the batch ran (plus a wrong-pid breadcrumb at the first J-phase launch). (c) #1482's driver push-verify was vacuous — 29 git-bound eval files never committed, caught only by the independent upload-verifier.
- **Why it is a workflow gap:** the detached-launch conventions (SKILL.md § Detached VM-side long compute phases; `.claude/rules/pod-side-reporting.md`) prescribe pid/log/harvest breadcrumbs but do not state the relaunch-from-launcher-file rule, the worker-pid (not launcher-pid) keying rule, or the named-paths verify rule — each failure was an improvisation the rules do not currently bar.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'launcher pid\|worker pid\|re-typed' .claude/rules/pod-side-reporting.md` → 0 (none of the three clauses exists; absence confirmed 2026-07-31 filing time).

## Proposed change (candidate diff sketch — refine in planning)

Add three clauses to `.claude/rules/pod-side-reporting.md` (+ a pointer in SKILL.md § Detached VM-side long compute phases): relaunch = re-run the launcher FILE; breadcrumb/watch keys on the WORKER pid (pgrep the distinctive script pattern post-launch, bracket idiom); any push-verify / upload-verify leg names its expected path set and fails on empty.

## Scope / surfaces

- Primary target: `.claude/rules/pod-side-reporting.md`, `.claude/skills/issue/SKILL.md` (§ Detached VM-side long compute phases)

## Constraints / invariants

- Doc-rule clauses only; no behavior change to the poller/dispatcher code in this task (code-side enforcement can be a follow-up if the clauses keep being violated).

## Provenance

- fingerprint: c8acd2997388

- workflow_fix_target: .claude/rules/pod-side-reporting.md, .claude/skills/issue/SKILL.md
- origin: /daily 2026-07-30 miners 3/4/5 (sessions d930d8d1 #1768, d0fe5a10 #1769, 36b6ee0e #1482)
