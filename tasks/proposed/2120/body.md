---
title: 'daily-fix: CVD-scoped nvidia-smi + schema-from-artifact'
kind: infra
tags:
- wf-fix
- wf-fix-fp:aab30f0c53a8
- daily-auto-filed
created_at: '2026-08-06T07:04:24Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): whole-host nvidia-smi drain
  verdict killed 4/9 rung-jobs (#1333 class re-introduced); fabricated shard schema
  cost a full implementation round'
workflow: v1
---
# daily-fix: two incident-backed checklist items — CVD-scoped nvidia-smi verdicts (code-reviewer) and schema-from-artifact loaders (implementer)

## Workflow gap

Two expensive 2026-08-05/06 incidents each recur a documented class that no reviewer or
implementer checklist currently catches:

1. **nvidia-smi verdicts not CUDA_VISIBLE_DEVICES-scoped (re-introduced #1333 class).**
   #2091's `issue2091_pod.py::reap_generation_engine` took `max()` of `memory.used` over
   ALL 4 GPUs (nvidia-smi ignores CVD), so 4 of 9 rung-jobs whose own GPU read 0 MiB died:
   "RuntimeError: vLLM teardown did not drain below 2048 MiB within 180s (per-GPU used
   MiB: [(0, 35579), (1, 19143), (2, 0), (3, 0)])" — ~765–880 s lost per job + a fix
   round (commit 2cc130dbff scoped the verdict own-device). The gotcha is documented; the
   review surface never checks for it.
2. **Loader schema fabricated from memory instead of read from the artifact.** #2061's
   round-1 implementation fabricated the #1336 shard schema — review verdict: "the
   pipeline cannot load its own input (fabricated `#1336` shard schema), P2 crashes
   deterministically after the expensive fits" — caught pre-GPU by the adversarial loop
   but at the cost of a full implementation round (~4.5 h wall) + review round 1 of 5.
   Related same-day sibling: #2091's judge collector assumed every packed row is a rollout
   and KeyError'd on the packed `_manifest.json` row 0.

verified-at-filing: both incidents are probed marker/review readbacks (miner 6 P4 rows
1088/1131 + fix commit; miner 5 P6 rows 108–118). `grep -cn 'nvidia-smi' .claude/rules/code-reviewer-section-reference.md
.claude/agents/experiment-implementer.md` and `grep -cn 'schema' .claude/agents/experiment-implementer.md`
run at compose time to confirm neither surface carries the item.

## Proposed change

- `.claude/agents/code-reviewer.md` (or its section-reference file): checklist item — any
  nvidia-smi-based drain/teardown/free-memory verdict in fan-out or dispatcher code must
  be scoped to the job's own CUDA_VISIBLE_DEVICES row(s); a whole-host max/min is a FAIL.
- `.claude/agents/experiment-implementer.md`: before writing any loader for a banked
  artifact, download and open ONE real shard/sidecar and paste its observed keys into the
  implementation marker (schema-from-artifact, never schema-from-memory); packed-format
  consumers filter rows on the pack's `src`/schema field.

## Provenance

- fingerprint: aab30f0c53a8

- workflow_fix_target: .claude/agents/code-reviewer.md, .claude/agents/experiment-implementer.md
- origin: /daily 2026-08-05 problem sweep — miner 6 P4/P6, miner 5 P6.
