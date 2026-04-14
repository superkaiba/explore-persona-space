---
name: Full Reproducibility in Every Experiment
description: Every experiment must have complete reproducibility card with ALL parameters, decision rationale, and exact reproduction commands — no "see config" or "defaults"
type: feedback
---

Every experiment write-up (draft and final) must include a complete Reproducibility Card table with ALL parameters needed to rerun from scratch, plus a Decision Log explaining WHY each choice was made.

**Why:** User explicitly requested that "every experiment should have FULL experimental parameters necessary to reproduce." Prior write-ups were inconsistent — some had LR but not batch size, some said "default settings" instead of actual values.

**How to apply:**
- Experimenter: Save expanded `run_result.json` with full training/data/eval/compute/environment sections
- Experimenter: Draft reports must include Reproducibility Card table (from CLAUDE.md template)
- Analyzer: Analysis drafts must include the Reproducibility Card and Decision Log sections
- Reviewer: Step 4 now checks Reproducibility Card completeness — >3 missing fields = REPRODUCIBILITY FAIL
- Planner: Plans must pre-fill the Reproducibility Card with known values at plan time
- Never write "default" or "see config" — write the actual value
- Always include the exact command used to launch the experiment
- Always include git commit hash of the script used
