---
title: 'workflow-fix: gotchas entry — sha pins live in a DOMAIN (recompute from producer
  recipe before asserting)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:738317166d80
created_at: '2026-07-29T13:29:41Z'
has_clean_result: false
origin_prompt: 'failure-lesson gotcha_candidate from #1776 crash-fix cycle 4 (epm:failure-lesson,
  2026-07-29): sha-pin domain mismatch — see the lesson block in the filed body''s
  Provenance section'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a failure-lesson `gotcha_candidate: yes` block raised on task #1776 (emitting agent: experiment-implementer, crash-fix cycle 4).

## Goal

Add a `.claude/rules/gotchas.md` entry (alongside the existing sha-pin family at ~L202-260) documenting the sha-pin DOMAIN trap: recompute a reused pin from its producer's recipe before asserting any new derivation against it — a wrong-domain compare fails on every input and masquerades as upstream data drift.

## Workflow gap

- **Bug observed:** #1776's `p1_contexts` (pod-1776, 2026-07-29) failed a plan-§10 assert reading exactly like corpus drift ("test-1000 sha drift: bb60a282... != pinned b9377786..."). The pins were sha256 digests of the #779 `fixed_split` **int64 INDEX arrays**; the consumer compared **prompt-string** digests against them — a wrong-domain compare that could never pass on ANY input. No drift existed: the fresh stream reproduced the frozen #779 manifest anchor exactly. Diagnosis consumed a full launch cycle (the crash fired mid-pipeline on 8×H100) and initially mis-presented as the #1768 real-corpus-dupes class.
- **Why it is a workflow gap:** gotchas.md already hosts a sha-pin family (pair provenance #922 at ~L202; bundle-field verification at ~L226) but nothing warns that a pin's DOMAIN (index arrays vs prompt strings vs file bytes) must be established by recomputation from the producer's recipe before any consumer asserts against it — plan-pinned integrity gates are a standing project pattern (plan §10 pins), so the trap recurs whenever a new consumer wires a reused pin.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n -i "sha pin\|sha-pin\|pin domain\|digest domain\|sha drift" .claude/rules/gotchas.md` → 5 hits in 1 file (L202/205/226/242/260 — the pair-provenance + bundle-verification entries; NO entry covers the pin-DOMAIN mismatch class; absence claim) (2026-07-29)

## Proposed change (candidate diff sketch — refine in planning)

```
+ - **A sha pin lives in a DOMAIN — recompute a reused pin from its producer's
+   recipe BEFORE asserting a new derivation against it.** #1776's plan-§10 pins
+   were digests of #779 fixed_split INDEX arrays (pure-RNG-reproducible); the
+   consumer compared prompt-string digests against them — a wrong-domain compare
+   that fails on EVERY input and masquerades as upstream data drift ("test-1000
+   sha drift"), burning a launch cycle. No drift existed. Rules: (i) establish
+   the domain by RECOMPUTING the pin from the producer's recipe (rerun the
+   producer function, hash its output — exact match proves the domain), never
+   from the pin's variable name; (ii) a consumer re-deriving membership from a
+   live stream guards with the producer's frozen MEMBERSHIP sha in the SAME
+   domain (e.g. the n1m manifest used_shas anchor); (iii) a "drift" assert that
+   fails on the FIRST-ever production run is a prime wrong-domain suspect;
+   (iv) strengthen, never relax (#1776 fix: 2 asserts -> 5, three-domain proof
+   chain + fails-pre-fix pytest, commit 04ce114b8fb2). Long-form twin:
+   .claude/agent-memory/experiment-implementer/feedback_sha_pin_domain_mismatch.md
```

## Scope / surfaces

- Primary target: `.claude/rules/gotchas.md`
- Place adjacent to the existing sha-pin family entries (~L202-260) so the family reads as one cluster; grep `sha-pin\|sha pin` across `.claude/` for any sibling surface that should cross-reference.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; gotchas.md `paths:` frontmatter untouched unless the trigger set genuinely widens.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/gotchas.md
- fingerprint: 738317166d80

<!-- epm:failure-lesson v1 -->
failure_class: code
phase: p1_contexts (scripts/issue1776_contexts.py)
lesson: A sha pin lives in a DOMAIN — verify what a reused pin actually hashes by recomputing it from its producer's recipe (here: fixed_split INDEX arrays, pure RNG) BEFORE asserting any new derivation against it; a wrong-domain compare fails on every input and masquerades as upstream data drift. When a consumer must re-derive membership from a live stream, guard it with the producer's frozen MEMBERSHIP sha in the same domain (the #779 n1m manifest's used_shas.round1), not by re-hashing the derivation in a different one.
generalizes: yes
owning_agent: experiment-implementer
gotcha_candidate: yes
root_cause_confirmed: yes
<!-- /epm:failure-lesson -->
