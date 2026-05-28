---
name: add-dimension-alias-legacy
description: When adding a new dimension (e.g., probe position, scoring window, judge variant) to an established multi-cell / multi-seed pipeline that already has on-disk artifacts and downstream consumers, keep ALL legacy filenames + payload keys as aliases pointing to the existing (now-default) dimension value, and add NEW parallel keys/files for the new dimension values. Never rename.
metadata:
  type: feedback
---

When adding a new dimension to an established multi-cell pipeline with on-disk artifacts (per-cell JSONs, per-seed payloads) and downstream consumers (aggregators, analyzers, critics) that already reference the keys:

**Do:**
- Keep legacy filenames untouched (e.g., `logprob_trained_A.json` stays the first-token output).
- Add new filenames for new dimension values (e.g., `logprob_trained_oncontent_A.json`).
- Keep legacy payload keys untouched (e.g., `trained_logp_by_cell`) and have them ALIAS the new explicit-dimension key (`trained_logp_by_cell_first_token`).
- Add new explicit-dimension keys for new dimension values (`trained_logp_by_cell_oncontent`).
- Parameterize pool helpers with a `position` (or whatever the new dimension is) arg that selects which key to read; default to the legacy dimension value.
- In the orchestrator: emit both the legacy alias AND the explicit-dimension keys so old code paths still work AND new code paths see the new dimension explicitly.

**Why:**
- Resume-from-disk works without re-running expensive (15-45 min) prior computations.
- Downstream HF artifacts don't need re-upload (task #399 round-16 had 84 first-token per-cell files on HF + pod; renaming would have cost a re-upload + a fresh re-compute).
- Analyzer / critic code references to the legacy keys keep working with zero changes.
- The new dimension's verdict ships as a SECOND parallel block side-by-side with the old verdict, making the position-mismatch visible to the reader instead of silently changing what `rescue_verdict` means.

**How to apply:**
- Any time you're adding a new probe position, new judge model, new bucketing dimension, new sub-sampling strategy, etc. to an established multi-phase eval that has on-disk artifacts.
- The structural cost is ~50-100 lines of plumbing (parameterize helpers + thread `position`/`variant`/etc. through subprocess workers + emit dual outputs). The crash-recovery + back-compat win dominates.
- Anti-pattern: renaming `trained_logp_by_cell` → `trained_logp_by_cell_first_token` and "updating all callers in one pass". You will miss a caller, or break analyzer code that reads the legacy key, or invalidate every existing per-cell file. The alias-and-add pattern is cheaper and safer.

**Example: task #399 round-16 (2026-05-27)** added an on-policy end-of-content probe alongside the existing first-token probe. Legacy `logprob_{mode}_{cell}.json` stayed first-token; new `logprob_{mode}_oncontent_{cell}.json` added. Legacy `trained_logp_by_cell` aliases `trained_logp_by_cell_first_token`; new `trained_logp_by_cell_oncontent` ships in parallel. `rescue_verdict` aliases `rescue_verdict_first_token`; new `rescue_verdict_on_policy_end_of_content` ships in parallel. Resume-from-disk on 84 existing first-token cell files worked without re-compute; analyzer code reading the legacy keys kept working.
