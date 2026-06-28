---
title: 'Dose-controlled gate→behavior bridge (parked redundant from #667)'
kind: experiment
tags: []
created_at: '2026-06-26T01:26:10Z'
has_clean_result: false
parent_id: 667
goal: 'Resolve whether the activation-gate→behavioral-leakage gap (ρ 0.13/0.16/0.40
  vs 0.46/0.59 from #667) is a fixed-epoch dose confound vs a real gate→behavior break,
  via a dose-controlled adapter sweep at matched install dose with per-cell behavioral
  G re-measurement.'
---
---
title: 'Dose-controlled gate→behavior bridge (parked redundant from #667)'
kind: experiment
status: on_hold
tags: [parked-redundant]
parent_id: 667
goal: 'Resolve whether the activation-gate→behavioral-leakage gap (ρ 0.13/0.16/0.40 vs 0.46/0.59 from #667) is a fixed-epoch dose confound vs a real gate→behavior break, via a dose-controlled adapter sweep at matched install dose with per-cell behavioral G re-measurement.'
relates_to: ['leak-predictor']
---

## Goal

Resolve whether the activation-gate→behavioral-leakage gap (ρ 0.13/0.16/0.40 vs 0.46/0.59 from #667) is a fixed-epoch dose confound vs a real gate→behavior break, via a dose-controlled adapter sweep at matched install dose with per-cell behavioral G re-measurement.

**Broader narrative:** The leakage-predictor's behavioral validity (`q:leak-predictor` 3.1).

## Value critique

**Verdict (Claude + Codex ensemble agree):** REDUNDANT.

**Why it duplicates existing work:** This proposal is verbatim the Phase 2 scope of #660's fleet retrain program — already running as #664 (the fresh source×behavior×arm×dose fleet train at matched-install) and #665 (the A3.6–A3.10/base-gate validity analysis on the trained store). The dose-control axis + per-cell behavioral G re-measurement + matched-install comparisons are all explicit Phase 2/Phase 3 deliverables of #660 (confirmed against `docs/theory_assumption_test_plan.md`: line 485 "dose-to-target (matched install), not fixed epochs", line 982 "≥2 doses per cell", line 1222 "Phase 2 = ~16–24 LoRA fine-tunes, ≥2 doses, ~55–95 GPU-h"). Running it here would re-do work already in flight.

**Revival path:** if #664 + #665 land and the dose-controlled bridge question is STILL unsettled, revive via `task.py set-status <this-task-id> proposed`.
