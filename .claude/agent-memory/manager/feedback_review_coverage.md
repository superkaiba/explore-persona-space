---
name: Ensure Review Coverage
description: 71% of experiment drafts had no independent review despite it being mandatory - enforce for all new experiments
type: feedback
---

As of 2026-04-11 audit, only 12/42 experiment drafts had independent _REVIEW files (29%). The review step was routinely skipped, especially for early Aim 1-4 work.

**Why:** The reviewer catches real problems — prompt-length confounds, backwards framing, overclaims. Skipping review lets flawed conclusions propagate into RESULTS.md and future experiment design.

**How to apply:**
- After every analyzer draft, ALWAYS dispatch a reviewer agent before updating RESULTS.md
- The reviewer result must be saved as a _REVIEW.md file alongside the draft
- No experiment conclusions should be added to RESULTS.md without a review
- Exception: trivial/obvious results (e.g., gate check failures, null results) can skip review with a note
