---
name: Claude critic credits plan §9 machine claims without resolving the intent map
description: Plan-stage placement disputes — resolve INTENT_TO_MACHINE + the parent's actual backend-selected rungs yourself; a plan saying "A100-40 spot (intent: eval)" can be internally inconsistent (eval's first rung is spot L4). #810 g1 round r1.
type: feedback
---

When a plan's §9/§10 states a GPU machine ("1× A100-40 spot") next to an
`--intent <X>` dispatch, the machine claim is NOT established by the prose —
resolve it yourself before crediting either reviewer:

1. `backends/gcp.py::INTENT_TO_MACHINE` — the intent's PRIMARY machine
   (`eval` → g2-standard-4 / 1× L4; `capture-7b` → a2-ultragpu-1g / A100-80).
2. The parent task's `events.jsonl` `epm:backend-selected` markers — the
   REAL rung sequence. #810's parent GPU session (intent: eval) attempted
   `rung: spot_l4` FIRST and only landed spot_a100_40 because L4 was
   ZONE_RESOURCE_POOL_EXHAUSTED. "The parent ran on A100-40" ≠ "the intent
   guarantees A100-40" — the parent got the right machine by stockout luck.
3. The router's own incident comments — gcp.py names #666 + #744 (7B
   activation-capture forwards OOMing when routed L4 under the eval
   default) and #752 created `capture-7b` precisely so plans DECLARE
   capture workloads.

**Why:** In #810 r1 (g1 genre round) the Claude methodology critic APPROVEd
and repeated the plan's "1× A100-40 spot (`intent: eval`)" §9 line verbatim;
Codex's Must-Fix (intent mis-route, may OOM before Phase B-g) was the only
grounded catch and was UPHELD → REVISE. The plan's HBM certification
(assumption: "fits on A100-40, parent ran it there") never covered L4, so the
registered launch command contradicted the plan's own compute basis.

**How to apply:** Any plan-stage placement/width dispute where one side cites
the plan's §9 machine prose: read INTENT_TO_MACHINE + INTENT_A100_40_FALLBACK
+ the parent's backend-selected rungs before crediting. A capture-class 7B
forward (teacher-forced hidden-state capture, per-token dumps) dispatched
`--intent eval` is the recurring mis-route; the fix is `--intent capture-7b`
or an explicit machine pin. Counterweight (same reconcile): Codex ALSO
fabricated a Must-Fix that the parent bootstrap JSON was "aggregate-only" —
the committed `bootstrap_deltaskill.json` had `per_context_decomposition` and
the script re-derives per-context terms deterministically; read the committed
JSON's keys before crediting an "artifact lacks field X" claim.
