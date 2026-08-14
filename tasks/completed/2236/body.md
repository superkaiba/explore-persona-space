---
title: Add the missing FAN-OUT STAGING EXTENSION to critic lens item 16 (plan-compute-sizing.md:330
  points at nonexistent lens text)
kind: infra
tags: []
created_at: '2026-08-11T18:13:18Z'
has_clean_result: false
origin_prompt: 'Workflow-surface follow-up surfaced by the critic Methodology-lens
  pass on #2054 plan v14: plan-compute-sizing.md''s fan-out-staging block delegates
  enforcement to a critic-lens item that does not exist, leaving the duty with no
  review owner.'
workflow: v1
---
## Goal

`.claude/rules/plan-compute-sizing.md` § "Fan-out over the same HF prefix" delegates its enforcement to a critic lens item that does not exist, so the duty it states has **no review owner**. Add the missing lens extension (or correct the pointer), so a plan fanning N > 1 boxes over one multi-GB HF prefix with no named staging shape is caught by a review lens rather than by luck.

## The gap (verified, not inferred)

`.claude/rules/plan-compute-sizing.md:330` ends the fan-out-staging block with:

> Critic enforcement: Methodology lens item 16
> FAN-OUT STAGING EXTENSION (`.claude/rules/critic-lens-reference.md`);
> no verify_plan.py backstop in v1.

But `.claude/rules/critic-lens-reference.md` item 16 carries exactly four extensions and none is the staging one:

- `:512` LADDER-RETENTION EXTENSION (#1133, incident #1112)
- `:536` FAN-OUT ACCUMULATION EXTENSION (#1541, incident #1481)
- `:565` PHASE-ORDERING EXTENSION (#1612, incident #1586 r5)
- `:593` MOUNT-BINDING EXTENSION (#1414, incident #1333)

`grep -n 'FAN-OUT STAGING EXTENSION' .claude/rules/critic-lens-reference.md` returns nothing. The rule's own text also says "no verify_plan.py backstop in v1", so with the lens item absent there is **no** enforcement surface at all for this duty — neither mechanical nor human.

Note the near-miss distinction that makes this easy to overlook: FAN-OUT **ACCUMULATION** (present) sizes disk for what N cells RETAIN; FAN-OUT **STAGING** (absent) governs N concurrent pulls of the same prefix. Similar names, different failure modes, and only one has an owner.

## How it surfaced

Task #2054 round `reduced-basis-refit-rungs789`, plan v14. The plan dispatches 8 parallel `cpu-bigmem` shards that each `stage_hub_prefix` the SAME ~12 GB `issue2054_lattice/activations` prefix concurrently, with no staging shape named. It PASSed `verify_plan.py` cleanly (0 FAIL / 0 WARN of 57 checks). The Methodology critic caught it only because its brief explicitly told it to load `plan-compute-sizing.md` — the lens file it would normally review from carries no such item. A brief that had trusted the lens roster alone would have APPROVEd the plan.

The failure mode the duty exists to prevent is real and costly: N concurrent same-prefix multi-GB pulls are a rate-limit storm risk — HF returns 429, or a TCP/RST that reads as rc=137 to the workload, any one box's shard fetch dies mid-stream, and the relaunch re-books the same collision. Incident #1739: three boxes each staged ~144 GB from one prefix simultaneously; 5 total attempts to land one leg.

## Deliverable

1. Add the **FAN-OUT STAGING EXTENSION** to `.claude/rules/critic-lens-reference.md` item 16, in the register and shape of its four sibling extensions (trigger, the REVISE condition, the accepted remedies, the incident citation #1739). The REVISE trigger per the rule: a §9 plan with N > 1 same-prefix concurrent multi-GB stages naming none of — pre-stage-once-and-fan (shared read path / rsync after one stage completes / baked image), serialized per-box pulls, or jittered start offsets.
2. Mirror the item into `.claude/agents/critic.md`'s Methodology lens roster if that file enumerates item 16's extensions (check before editing).
3. Confirm `.claude/rules/lens-coverage-map.md` reflects the new lens→owner entry; `workflow_lint.py --check-lens-coverage` is the gate.
4. Decide and record whether a `verify_plan.py` WARN-only backstop is worth adding now. The rule currently disclaims one ("no verify_plan.py backstop in v1"); the #2054 shape suggests a cheap heuristic exists — more than one plan-embedded launch whose workload stages a shared HF prefix, with no `serialize|jitter|pre-stage` token in the §9 staging text. If added, follow the c-check calibration contract (persisted-plan corpus re-scan) rather than shipping an uncalibrated regex; if declined, say why in the rule text so the next reader does not re-open it.

## Acceptance

- `grep -n 'FAN-OUT STAGING' .claude/rules/critic-lens-reference.md` returns the new item.
- The rule's `:330` enforcement pointer resolves to real text.
- `uv run python scripts/workflow_lint.py` passes (no-flags default run), including `--check-lens-coverage`.
- If a verify_plan check was added: it fires on the #2054 v14 shape (8 shards, one shared prefix, no staging shape) and does NOT fire on plan v12 (whose R4/R5 multi-pod staging at the same venue+prefix is the precedent the #2054 critic judged functionally acceptable) — i.e. calibrate against the persisted-plan corpus before shipping, and report the true/false-positive counts.

## Provenance

Surfaced by the `critic` Methodology-lens pass on #2054 plan v14, 2026-08-11, as an explicit workflow-surface follow-up in its report prose (not a plan finding). Filed per `.claude/rules/workflow-fix-on-bug.md` — surfaced-prose follow-ups carry the same auto-file duty as a formal `workflow-fix-candidate` block. Gap independently verified by grep before filing (both the rule's pointer and the lens file's actual item-16 extension list are quoted above).
