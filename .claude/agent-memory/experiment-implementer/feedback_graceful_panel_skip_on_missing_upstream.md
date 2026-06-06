---
name: Graceful per-panel skip on missing upstream data
description: Multi-panel materialization scripts (--panel all dispatchers) MUST try/except per panel + log a deviation + continue, NOT crash all 13 on the first FileNotFoundError. Rationale + recipe.
type: feedback
---

When a multi-panel/multi-pair/multi-domain materialization dispatcher has a `--panel all` (or `--pair all` etc) mode, the per-item loop MUST try/except FileNotFoundError + generic Exception + record a deviation + continue. The anti-pattern `for p in panels: materialize_panel(p)` turns ANY one panel's upstream-data gap into a total wipe of the other 12 panels' generation work — which on a Claude-API-heavy script also burns $$$ on each relaunch.

**Why:** Task #503 round-4 launched with `--panel all`; the FIRST panel materialized (`turner_medical_heldout`) crashed at `ensure_dataset("turner_bad_medical")` because the Turner decrypt step had never run on the worktree's VM. Result: zero of 13 panels materialized, the pod-side launch died at pre-flight, the entire round had to be re-rolled. With graceful-skip, 11 of 13 panels would have materialized and only the 2 turner-dependent ones would have surfaced as deviations.

**How to apply:**

1. Distinguish `--panel <id>` (explicit, failure-is-fatal) from `--panel all` (sweep, deviations allowed). Re-raise inside the per-item except branch when the user picked one specific panel — they asked for that panel to surface its error.

2. Persist a structured `_materialize_summary.json` containing `materialized: [...]`, `deviations: [{panel_id, exception_type, message, recommended_fix}, ...]`, `skipped_already_present: [...]` so the next launch attempt has actionable per-panel hints.

3. Add a per-panel recommended-fix dict mapping `panel_id → operator hint` ("Run scripts/foo.py first to materialize ..."). A bare exception message buried in logs is not actionable; a pinned recommended-fix string per panel is.

4. **Exit-code policy:** rc=0 when at least one panel materialized OR rc=1 when ALL panels failed (catastrophic). Avoid the trap of rc=0 on zero panels — that hides a fully-broken pipeline.

5. **Add idempotency (skip-if-exists default + --rebuild flag).** Without this, re-running `--panel all` re-fires every Claude generator and burns $5+ per relaunch. Particularly important when the script is invoked as part of a pre-launch sweep that may run 3-5 times during one debug cycle.

6. Pair with a parametrized pytest that monkeypatches one panel's generator to raise FileNotFoundError and asserts (a) rc=0, (b) deviation recorded with correct exception_type + recommended_fix, (c) other panels materialized cleanly, (d) explicit `--panel <id>` re-raises.

Generalizes to any multi-cell/multi-pair/multi-domain dispatcher script (`scripts/issue458_prep_datasets.py --pair all`, sweep dispatchers, etc).
