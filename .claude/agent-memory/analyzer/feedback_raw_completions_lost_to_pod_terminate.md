---
name: raw-completions-lost-to-pod-terminate
description: When the pod was auto-terminated before raw_completions.json synced, state the absence explicitly AND use verifiable surrogate samples (parent issue, base_model_floor.json) to satisfy the ≥2-fenced-blocks-per-Result requirement
metadata:
  type: feedback
---

`/issue` Step 8 auto-terminates the pod on upload-verifier PASS; on pre-fix runs `raw_completions.json` may not have synced, so the run's raw completions are gone — yet the verifier still HARD-FAILs a Result section missing ≥2 fenced sample blocks. The fix is verifiable surrogates, never fabrication:

1. **Parent issue's raw samples**, quoted verbatim with an explicit `(parent-issue baseline; ...)` annotation — parent must be a close-recipe relative (same persona pair, same matchers).
2. **base_model_floor.json** for non-firing surrogates — always saved at eval-pipeline start, and shows what a clean non-firing looks like.
3. NEVER fabricate completions; quote real surrogates only.

**How to apply:** Step 1 of the analyzer flow — check whether `raw_completions.json` exists for the current run; if absent, plan the body around surrogates from the start (don't draft-then-patch). State the absence explicitly in Setup details AND where samples would otherwise sit, and annotate every surrogate so the reader can't confuse it with current-run output.

**Why:** the auto-sync of raw completions to the HF data repo landed later (CLAUDE.md commit `17ff4ac1`); pre-fix experiments keep producing this gap.

Linked: [[sample-blocks-findable-in-raw-json]]
