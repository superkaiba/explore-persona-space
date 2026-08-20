---
name: plan-halt-gate-runbook-only-cli
description: Plan-named pre-spend HALT gate realized as a CLI referenced nowhere (self+tests only) — persist CONCERN with run-brief HALT-precondition remedy, don't bounce when wiring precedent is mixed and loud failures self-protect (#2389 R1 g8)
metadata:
  type: feedback
---

When a plan names a pre-spend HALT gate ("run BEFORE phase X") and the diff ships it
as a standalone CLI, grep ALL round files for the script name: hits only in itself +
its tests = runbook-only, no mechanical sequencing. Disposition (#2389 R1 g8):

- NOT a bounce when (a) the plan's own gates have MIXED wiring precedent (some
  in-driver, some runbook), and (b) the LOUD failure class self-protects at
  production entry (consumer loaders assert non-empty/duplicates), leaving only the
  SILENT class (a `dict.get` join dropping rows) for the probe.
- ALWAYS persist via `task.py raise-concern` (Step 0.8 deferred-feature rule, #509:
  prose-only concerns don't reach the dispatch gate). CONCERN, not BLOCKER, when
  production doesn't crash without it. Remedy in the summary: probe-report PASS
  required before the protected dispatch, or a freshness refusal at phase entry.
- Sub-check: a probe that HAND-PLACES staged files (stage_hub_file into its own dir)
  certifies the CONTENT contract, not the production layout transformation
  (`stage_hub_prefix` mirrors verbatim to `dest/<repo-relative path>`). Faithful when
  the production co-location is itself runbook-level; recommend `--local-anchors-dir`
  on the consumer's REAL resolved dir for full fidelity.
- `task.py raise-concern` can exceed 120 s under fleet lock contention — run it
  early / background it, and confirm via `list-concerns --open-only --json`, not rc.

**Why:** the review brief asked blocker/CONCERN/acceptable on exactly this shape; the
mixed-precedent + self-protecting-loud-class analysis is what made CONCERN (not FAIL)
defensible, and the persisted row is what makes it binding.
**How to apply:** any round shipping a plan-gate as a new `scripts/issueN_*.py` with
no dispatcher/PHASES/STEPS entry; also [[smoke_enum_item_without_dial]] (the inverse:
gate promised, no reachable CLI at all).
