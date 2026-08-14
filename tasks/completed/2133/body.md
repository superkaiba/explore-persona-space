---
title: 'daily-fix: measured n_train + regime-change pilot basis'
kind: infra
tags:
- wf-fix
- wf-fix-fp:56904e1c9622
- daily-auto-filed
created_at: '2026-08-06T07:10:58Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-05 problem sweep (route 2): inferred SS7 n_train invalidated
  a plan post-smoke (n<d cell); greedy rerun realized ~2x the user-approved GPU-h
  off the parent wall-time basis'
workflow: v1
---
# daily-fix: plan-sizing ground truth — measured n_train rows (planner §7) and regime-change pilot basis (compute sizing)

## Workflow gap

Two same-day sizing failures trace to plans quoting inferred numbers where a measured
artifact-side read was available:

1. **Inferred n_train invalidated a plan post-smoke.** #2061's §7 n_train table was
   "inferred, not measured; a registered cell is n<d" — sharpened to "§7 used the WRONG
   DENOMINATOR; likely 5 of 7 combos degenerate" (corpus n_built vs realized turnstore
   rows). Cost: an `epm:strategy-pivot` + plans v11→v13 (~2.6 h re-planning) — caught
   before the ~70 GPU-h production spend, but only post-smoke. The #1887 n<d refusal makes
   an inferred count plan-invalidating by construction.
2. **A decoding-regime change invalidated the parent's wall-time as sizing basis.**
   #1491's greedy rerun was quoted at 40–50 GPU-h off the parent Repro footer's 5.4 h wall
   (which "swept in setup and two failed launches"); greedy decoding's repetition loops
   (cap-hit 18.47% vs parent 6.66% at 0.5B) pushed realized cost to ~78–82 GPU-h
   (~1.7–2.0× the figure Thomas approved). Disclosed proactively + recorded as
   `epm:compute-deviation v1` with measured causes.

verified-at-filing: incident 1 is the probed pivot-marker chain on #2061 (session 5c878aa4
rows 1061/1140/1147, fact-check reproduced all 55 store counts); incident 2 is #1491's
own compute-deviation marker (2026-08-06T00:58:41Z, probed). `grep -n 'n_train' .claude/agents/planner.md | head -3`
and `grep -n 'MEASURED' .claude/rules/plan-compute-sizing.md | head -5` run at compose
time — the measured-pilot rule exists for wall-time; neither surface names these two
specific bases.

## Proposed change

- `.claude/agents/planner.md` §7: any registered per-fold/per-cell n_train must cite a
  MEASURED artifact-side row count (sidecar/manifest read via a named command), never a
  corpus-name inference.
- `.claude/rules/plan-compute-sizing.md`: name the regime-change trap — a decoding/
  sampling regime change (sampling→greedy, temperature change, cap change) invalidates the
  parent run's wall-time as a sizing basis (completion-length distribution shifts);
  require a 1-rung measured pilot before quoting a user-facing cost for a regime-changed
  rerun.

## Provenance

- fingerprint: 56904e1c9622

- workflow_fix_target: .claude/agents/planner.md, .claude/rules/plan-compute-sizing.md
- origin: /daily 2026-08-05 problem sweep — miner 5 P8, miner 1 P7.
