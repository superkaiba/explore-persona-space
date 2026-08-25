---
title: 'workflow-fix: verify_plan c26/c27 routed-machine mirror is GCP-era, inverts
  under runpod-first auto default'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-24T02:25:19Z'
has_clean_result: false
origin_prompt: 'Open concern c26-gcp-era-mirror-stale on #2298 (named, never auto-filed
  under that session''s recursion guard); re-surfaced 2026-08-24 during #2331 duplicate-verification,
  which found the concern had no execution path'
workflow: v1
---
# workflow-fix: `verify_plan.py` c26/c27 routed-machine mirror is GCP-era, inverts under the runpod-first auto default

## Goal

Rebuild `verify_plan.py`'s routed-machine mirror (`_C26_INTENT_GPU`, and the c27
sets derived from it) so c26/c27 judge a plan's §9 `basis` GPU against the machine
the live router will ACTUALLY provision under the runpod-first `auto` default
(#2054/#2059), instead of the GCP-era A100/L4 families. Then remove the interim
staleness caveat the prose surface now carries.

## The drift

`scripts/verify_plan.py:4826` — `_C26_INTENT_GPU` maps intents to GPU families:

```python
"lora-7b": "A100", "lora": "A100", "capture-7b": "A100", "ft-7b": "A100",
"eval": "L4", "debug": "L4", ...
```

Those are the **GCP** `INTENT_TO_MACHINE` families (`a2-ultragpu` A100-80, L4) —
correct when GCP led the auto chain, wrong now. GCP provisioning is DISABLED
(`GCP_PROVISIONING_DISABLED = True`, #2028) and the live head is RunPod:

- `DEFAULT_AUTO_LANE_ORDER == ('runpod', 'fellows', 'nibi', 'fir', 'mila')`
  (`src/explore_persona_space/backends/router.py`, #2054/#2059) — verified live.
- RunPod's intent table (`.claude/rules/pods.md`): `lora-7b` → 1× **H100**,
  `eval`/`debug` → 1× **H100**, `ft-7b` → 4× **H100**.

So for a bare-`auto` plan, c26 expects an A100/L4 basis while the router provisions
H100. A plan **correctly** costed on RunPod H100 trips a c26 WARN; a plan wrongly
costed on A100 reads as clean. The check is inverted, the same way
`critic-lens-reference.md` item 13's prose was before #2298 fixed it.

**Downstream consumer — not a one-dict edit.** `verify_plan.py:5113-5116` derives
`_C27_L4_INTENTS` and the non-L4/non-CPU set FROM `_C26_INTENT_GPU`, so c27 inherits
the same GCP-era premise. Scope both.

## Why it matters

c26 is WARN-only, which makes this worse, not better: it emits an EXPECTED-FALSE
warning on every runpod-costed bare-`auto` plan, and the prose surface now tells
readers to ignore it (see the caveat below). A gate whose documented guidance is
"this warning is expected" is a gate reviewers stop reading — and the day a plan is
genuinely mis-costed, the real WARN is indistinguishable from the noise.

## Fix shape (as specified by #2298's implementer when it raised the concern)

1. Derive the routed GPU family from the **live** lane head rather than hardcoding a
   provider's table — the same AST-read-the-source approach
   `workflow_lint.py --check-lane-order-adjective` already uses to stay in sync
   (`_default_auto_lane_order`), so the mirror cannot silently invert on the next
   lane-order change. Keep it hermetic (no network, no live API call at lint time).
2. Re-calibrate the corpus: c26/c27 are heuristics tuned against the existing plan
   corpus, so re-scan after the mapping changes and re-baseline expected findings.
3. Update `tests/test_verify_plan.py` c26/c27 coverage (and any c26-derived
   assertions in `tests/test_issue2476_gates.py`).
4. **Remove the interim caveat** at `.claude/rules/plan-compute-sizing.md:804-806`
   ("c26's routed-machine mirror is GCP-era and tracked separately — a c26 WARN on a
   runpod-costed bare-`auto` plan is expected until that mirror is updated"). Once
   the mirror is rebuilt that sentence becomes stale prose on the prescriptive
   surface — i.e. it becomes the exact defect class #2298/#2331 existed to close.
   Landing the code fix without deleting the caveat just relocates the drift.

## Provenance

Raised as concern `c26-gcp-era-mirror-stale` (severity CONCERN, round 1, by the
implementer) on **#2298**, and deliberately left out of that diff per its plan's
Part 5: #2298 was itself a workflow-fix task carrying `workflow_fix_target:`, so its
recursion guard required the companion be NAMED, never auto-filed. It has sat as an
open concern on a `completed` task since 2026-08-22 with no execution path —
concerns on terminal tasks are never dispatched.

Surfaced again 2026-08-24 while verifying **#2331** (an independent duplicate filing
of the same prose drift, archived after #2298 was confirmed to cover its whole
scope). #2331 was not mechanically workflow-fix-bound
(`is_workflow_fix_session(2331)` is False — its candidate block is an HTML comment,
not a `workflow_fix_target:` Provenance line), so filing this companion here is not
a recursion-guard bypass.
