---
title: 'daily-fix: smoke must run real paths, not mock branches'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b342926155ac
- daily-auto-filed
created_at: '2026-07-26T07:00:57Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): Three consecutive #1689
  code-review rounds false-passed a fatal ImportError that sat behind the mock-response
  branch the smoke exercised, and a separate round shipped literal stubs for ladder
  rungs 7 to 9 with a green eight-phase smoke, each costing a full round plus a pod
  cycle.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 problem sweep from task #1689, which burned 8
implementer + code-review rounds. Two of those rounds were lost to the same root
cause: the smoke gate reported `PASS_UNIFIED` on code paths that were never actually
executed.

## Goal

Require the smoke-architecture-check to execute every deferred import and the REAL
computation path of each planned arm, to enumerate real-versus-fallback per planned
arm in its verdict, and to forbid `PASS_UNIFIED` when any planned arm resolved to a
stub or a mock-only branch.

## Workflow gap

Two distinct failures of the same shape on #1689 (session `5c5a89e8`):

1. **Deferred import behind a mock branch — 3 rounds false-passed.**
   `scripts/issue1689_haiku_u2_gen.py` imported a non-existent `DispatchCall` (real
   name `DispatchItem`) and called the `async def dispatch_calls` synchronously. The
   deferred import sat behind the `--mock-response` branch, so the 8-phase smoke used
   by rounds 2/3/4 never executed it. The bug surfaced only after a full GCP boot-loop
   + RunPod failover + a real Phase-A run. Implementer R5 verbatim: *"The
   --mock-response smoke path never executed the deferred import, so rounds 2/3/4 all
   false-passed."*
2. **Literal stubs passing a green 8-phase smoke.** Round 2 delivered 1488 LOC, marked
   all 8 prior concerns addressed, and reported "full 8-phase smoke green" — while
   ladder rungs 7/8/9 were stubs falling back to a bias-refit R² (making hypothesis H2
   untestable) and the selection-symmetric bootstrap was absent (headline CI invalid).
   Caught only by the R3 code-reviewer: *"Code-review round 2 FAIL on substantive —
   rungs 7-9 are literal stubs (H2 untestable), selection-symmetric bootstrap not
   implemented (headline CI invalid), dispatch.sh drops base model."*

Why the current gate misses both: `PASS_UNIFIED` is defined around **phase coverage**
("EVERY phase the dispatcher executes", plus per-phase subset threading). A phase can
execute end-to-end while the arm it is supposed to compute resolves to a fallback, and
a mock branch counts as executing the phase. Neither failure is a coverage gap — both
are "the phase ran, the science didn't".

- **Confidence (emitter):** high — two same-day incidents on one task, each with the
  implementer's or reviewer's own verbatim diagnosis.
- verified-at-filing: per-target probes on `.claude/agents/experiment-implementer.md`
  — `grep -n 'PASS_UNIFIED'` → **5 hits** (lines 124, 125, 136, 151, 168); reading
  their context confirms the definition is phase-coverage + per-phase subset threading
  (line 124: *"EVERY phase the dispatcher executes), the verdict is `PASS_UNIFIED`"*;
  line 125: *"Per-phase subset threading is part of the PASS_UNIFIED definition"*), with
  no real-versus-fallback arm enumeration and no deferred-import leg. `grep -c
  'fallback'` → **1** (unrelated to arm resolution). Presence hits read in context per
  clause (c): the existing text does NOT already implement the proposed change.
  Landed-fix history check `git log --oneline --since='7 days ago' --
  .claude/agents/experiment-implementer.md` → the 2026-07-25 wave touched it via
  #1682 (`841304c2d0`, implementer reports rev-parse SHAs verbatim); nothing touching
  the smoke verdict definition. (2026-07-25)

**Explicitly NOT part of this task.** The sweep also flagged that the #1689 round-1
code-reviewer returned `PASS` with `blocker tags: none` while 7 self-declared BLOCKER
concerns sat open. Compose-time verification REFUTED that as a defect:
`.claude/agents/code-reviewer.md` rule 11 explicitly prescribes raising a BLOCKER via
`task.py raise-concern` *"even on a PASS verdict"*, because *"The Step 5c-ter dispatch
gate reads `concerns.jsonl`, not verdict prose"* — the orchestrator's bounce was the
designed mechanism working, not an inference around a broken one. Do not "fix" it.

## Proposed change (refine in planning)

```
  in the smoke-architecture-check contract:
+ (1) import-resolution leg: execute every deferred/lazy import in the changed
+     entrypoints on the REAL branch (e.g. an --import-check mode, or a
+     python -c import of each changed entrypoint), not only under a mock flag.
+ (2) per-arm resolution table in the verdict: for each arm/rung the PLAN names,
+     state whether the smoke executed the REAL computation or a fallback/stub.
+ (3) PASS_UNIFIED is forbidden when any planned arm resolved to a fallback;
+     the verdict becomes PASS_PARTIAL (or FAIL) naming the arms.
```

The plan should decide whether (3) is enforced in the implementer's own attestation,
in `code-reviewer.md` (FAIL any diff whose only exercised path is a mock/stub branch),
or both. Both is likely right — the attestation makes it cheap, the reviewer makes it
binding.

## Scope / surfaces

- Primary target: `.claude/agents/experiment-implementer.md` (smoke-architecture-check
  contract + the `epm:smoke-architecture-check` note shape at ~line 168).
- `.claude/agents/code-reviewer.md` — the binding half of (3).
- Check whether `.claude/skills/issue/SKILL.md` Step 6d.0 restates the verdict
  vocabulary; if so it changes with them.
- The per-arm table must key on arms the PLAN names, so it stays meaningful for a
  `kind: experiment` diff and vacuous (not obstructive) for an infra diff with no arms.

## Constraints / invariants

- Do not make the smoke a full production run — the import-resolution leg is seconds,
  and the per-arm table is an attestation, not a re-execution of the science.
- Keep the existing phase-coverage semantics intact; this ADDS an axis.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes;
  agent-spec size ratchet (`--check-agent-spec-size`) must stay under budget — both
  target files are large, so prefer tight prose over a new subsection.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: .claude/agents/experiment-implementer.md
- fingerprint: b342926155ac
- Source: `/daily` 2026-07-25 transcript sweep, session `5c5a89e8` (#1689) @
  2026-07-26T01:51:15Z / 02:12:22Z (deferred import) and 2026-07-25T23:34:13Z (stubs).
