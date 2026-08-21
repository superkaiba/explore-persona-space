---
title: 'Registered gates are never checked against the grain they fire at: a rate
  threshold can be unreachable at the realized per-unit denominator, and a per-unit
  remedy can split a controlled parameter inside the contrast it protects'
kind: infra
tags: []
created_at: '2026-08-21T07:11:37Z'
has_clean_result: false
origin_prompt: /issue 2329
workflow: v1
---
# Registered gates are never checked against the GRAIN they fire at: a rate threshold can be arithmetically unreachable at the realized per-unit denominator, and a per-unit remedy can split a controlled parameter inside the contrast it protects

## Goal

Add two plan-time checks. Both failed silently in #2329 through a full v2 review stack — three Claude
critics, a Codex twin, a binding reconciler, and `verify_plan` — because every reviewer checked that
the CODE implements the registered gate faithfully, and none checked whether the gate's own parameters
are coherent at the grain it operates on.

**Check A — threshold expressibility.** A pre-registered RATE threshold is degenerate when the
realized per-unit denominator `n` makes the minimum non-zero rate `1/n` exceed the threshold. The gate
then reduces to "at least one event" and carries no severity information, while reading in the plan
like a tuned 2% bar.

**Check B — remedy/contrast grain compatibility.** When a plan registers a remedy applied at a grain
FINER than the contrast the experiment measures, and that remedy CHANGES a generation parameter,
partial application splits the parameter across arms inside the contrast — introducing the very
asymmetry the gate usually exists to prevent.

## Incident (#2329 round `q35_ladder_decay`, 2026-08-21)

Plan v8 §7 G5 line 233: "Cap-hit > 2% per (direction x slot x arm) cell at 4096 => re-generate that
cell's rollouts at 8192 ... Exactly 2.0% does not fire."

**Check A violation.** Realized grain: 72 units, **every unit n=30** (72 x 30 = 2160 rows exactly).
Realizable rates are {0, 3.333, 6.667, ...}%. Minimum non-zero = **3.333% > 2.0%**, so `> 2%` is
identical to `>= 1 cap-hit row`; the strict-vs-inclusive clause the plan carefully specifies is
unreachable. Realized: 47 units 0 hits, 21 units 1 hit, 4 units 2 hits => **25 of 72 units "breached"**
on 29 truncated rows out of 2160 (1.343% aggregate). The orchestrator's own hand estimate (2 units)
was computed at a coarser grain and understated the remedy 12.5x; that wrong figure was then supplied
to the reconciler, which demoted a related BLOCKER partly on its strength.

**Check B violation.** Contrast groups are `(cell x slot)` with 3 arms each (`null_sameval`,
`null_xtype`, `steered`), 24 groups. Breach distribution: 9 groups no arm, 3 groups all arms,
**12 groups SOME arms**. So the registered remedy, executed exactly as written, regenerates 1-2 arms
at 8192 and leaves the comparison arm(s) at 4096 in **half the contrasts** — e.g.
`install_r1_pirate|ce` = `null_sameval=1* null_xtype=0 steered=1*`. The issue's own
`issue2329_capregen_sufficiency.py` docstring states the standard this breaks: "*A cap that is
sufficient on average but truncates one side of a within-cell contrast is still a measurement-validity
failure.*" Only all-72-blocks (~4.3-4.8 GPU-h, ~3x the plan's 1.4 GPU-h reserve) or none preserves
cap homogeneity.

Net: a registered remedy that, when it fires as designed, is measurement-HARMFUL and unaffordable
simultaneously. Caught only because the orchestrator ran the report VM-side before provisioning; the
GPU path would have regenerated 12 contrasts into cap-split states.

## Proposed fix

`scripts/verify_plan.py`, two new checks (WARN-only is acceptable for v1; the plan surface is the
enforcement point):

- **A:** when a plan registers a rate/percentage trigger together with a grain, and §9 (or the
  manifest) permits deriving the per-unit denominator `n` at that grain, FAIL/WARN if
  `threshold_pct <= 100/n`. Message should state the min non-zero rate and the realized `n`. Requires
  the planner to declare expected rows-per-unit at the trigger's grain — a one-line §7 addition that
  is independently useful.
- **B:** when a plan registers a remedy that mutates a GENERATION parameter (cap, temperature, draws,
  model revision) at a grain finer than, or crosscutting, the declared contrast grain, require an
  explicit statement of whether partial application preserves homogeneity across the contrast's arms —
  with the empty form written literally (`homogeneous — remedy grain == contrast grain`), mirroring the
  `.claude/rules/smoke-blind-spots.md` enumeration convention.

Companion prose: a short `.claude/rules/` clause, or an item in the existing statistics/measurement
lens, so the reviewers have somewhere to point. `statistics-critic` is the natural owner of A;
`methodology-baselines-critic` of B.

## Acceptance criteria

1. Check A FAILs/WARNs on the #2329 fixture (threshold 2.0, grain `cell_slot_arm`, n=30) and passes on
   the same plan with n=180 (where 0.556% is expressible).
2. Check B WARNs on a plan whose remedy grain is finer than the contrast grain with no homogeneity
   statement; passes with the statement or the literal empty form.
3. Neither check fires on a plan with no registered rate trigger / no parameter-mutating remedy
   (no false positives on the ~2,400 existing plan versions — spot-check a sample).
4. Tests failing before and passing after; no new red in the no-flags `workflow_lint.py` run or the
   mapped-test selection.
5. The reviewer-lens prose names both checks so a human reviewer can catch what the mechanical check
   cannot derive (e.g. a grain declared only in code).

## Candidate metadata

- target_file: scripts/verify_plan.py (+ a `.claude/rules/` or critic-lens clause)
- fingerprint: registered-gate-parameters-unchecked-against-firing-grain
- confidence: high — both violations measured live in #2329 with exact denominators, breach counts, and
  the per-contrast arm split; the full review stack passed the implementation

## Provenance

workflow_fix_target: scripts/verify_plan.py

Auto-filed by the `/issue 2329` orchestrator from findings in the r20 G5 remedy phase (2026-08-21).
Evidence: #2329 `events.jsonl` `epm:compute-deviation` v1 (degenerate threshold, per-unit
distribution) and `epm:progress` v187 (the 12/24 contrast split + decision); report artifact
`data/issue_2329/q35_capregen/ladder/manifests/cap_hit_report_grid.json`.

Distinct from #2419 (mechanizable plan-CLAIM checks: claimed vs realized artifacts) — these two check
a gate's INTERNAL coherence against its own grain, not a claim against an artifact. Same target file,
different bug, per the workflow-fix dedup rule.
