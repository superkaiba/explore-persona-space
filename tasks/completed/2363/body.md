---
title: 'workflow-fix: a plan can mark its ONLY source of acceptance evidence optional
  and still PASS verify_plan'
kind: infra
tags: []
created_at: '2026-08-18T09:00:15Z'
has_clean_result: false
workflow: v1
---
---
kind: infra
---

# workflow-fix: a plan can mark its ONLY source of acceptance evidence "optional" and still PASS verify_plan

**Provenance:** caught by the Claude statistics critic during `/adversarial-planner`
Phase 2 on task #2360 (2026-08-18), which flagged it as its single Must-Fix and
explicitly noted it as mechanizable. Not hypothetical — it shipped through a
full planner round, a `verify_plan.py` PASS (0 FAIL / 0 WARN), and a Phase-1.5
fact-check round that verified 18 assumptions, before a human-facing lens caught
it.

## What happened

Plan #2360 v2 headed its live pod-validation phase:

```
## 6 ... Phase V (optional but planned)
```

Meanwhile §7 bound **three** verdicts to that phase and to nothing else:

- Acceptance criterion 2's pod half — the wall-time budget. The alternative
  basis was measured on the shared VM, and the plan's own assumptions section
  rated the transfer to the pod's filesystem only *Medium*.
- Acceptance criterion 3's count check — it reads a log file that only exists
  if Phase V runs.
- Kill criterion (a) — "if the Phase-V healthy probe exceeds 120 s…", which can
  never fire if the phase is skipped.

So the plan was internally inconsistent in a way that is entirely mechanical:
a phase that acceptance criteria and a kill criterion DEPEND on was declared
skippable. Had it been skipped, a fleet-wide default-on threshold would have
shipped on a basis measured only in the wrong environment, and two of three
acceptance criteria would have had no evidence at all.

Nothing downstream would have caught it either: the `/issue` Step 9c gate runs
pytest, and a live pod validation is not a pytest.

## Why the existing surface misses it

`verify_plan.py` has 60+ checks including several that reason about §7 gates and
§9 compute rows (c29 reconciles conditional phases against a declared fence,
c50/c61 reconcile wall/memory against lane defaults). None of them relates the
ACCEPTANCE BINDINGS in §7 to the OPTIONALITY MARKERS on the phases those
bindings name. The relationship is purely textual and local, which is exactly
the class `verify_plan` is good at.

The critic caught it here, but that is a per-round coin flip: the Claude
methodology critic and both Claude/Codex alternatives critics read the same
plan and did not flag it.

## Candidate fix surfaces (implementing session picks)

1. **`scripts/verify_plan.py` — a new check.** For each §7 acceptance-binding /
   kill-criterion row, extract the phase names it references; for each, locate
   that phase's declaration in §6/§9 and FAIL (or WARN, calibrated) if the
   declaration carries an optionality marker (`optional`, `if time permits`,
   `nice to have`, `stretch`, `best-effort`). The canonical N/A escape phrase
   convention applies. Calibrate against the persisted-plan corpus before
   choosing FAIL vs WARN — the project's own convention for a new heuristic
   check is WARN-only first (cf. c50, c52, c56, c57, c61).
2. **`.claude/rules/critic-lens-reference.md`** — an explicit item under the
   Statistics & Measurement (or Methodology) lens making the acceptance-binding
   ↔ optionality relation a named thing reviewers check, so it stops depending
   on one reviewer noticing.
3. Both — the lens as the semantic gate, the check as the mechanical backstop.
   This is the project's standard pairing.

## Acceptance

- A plan whose §7 acceptance binding or kill criterion names a phase that §6/§9
  marks optional is flagged, with the phase name and the binding quoted.
- A plan whose optional phase is referenced by NOTHING in §7 is not flagged
  (genuinely optional extras must stay legal).
- A phase named by a §7 binding and NOT marked optional is not flagged.
- Calibrated against the persisted-plan corpus, with the true/false-positive
  counts recorded, per the calibration contract other heuristic checks follow.
