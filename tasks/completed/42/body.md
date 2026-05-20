---
title: '[Decision] Option A: bump open-instruct submodule to unlock Liger + packing
  on Tulu path'
kind: infra
tags: []
created_at: '2026-04-17T22:28:07.000Z'
has_clean_result: false
sagan_id: c2b3ca14-e368-46db-bb50-47f3399f44d9
sagan_number: 42
priority: normal
legacy_why_unset: true
---
## Context

Research decision, not an immediate task. #40 showed the open-instruct submodule (pinned at `6b3964bc`) pre-dates Liger + packing integration (upstream PR #601+). All past distributed Tulu runs have run WITHOUT these optimizations — contradicting what the configs claimed.

## The decision

Bump open-instruct submodule to a commit ≥ PR #601 to unlock Liger + packing on the distributed Tulu path. Expected speedup: ~+20-30% throughput on full-FT SFT, plus packing gains (varies with data length distribution).

## Trade-offs

### Pros
- Distributed path actually gets the optimizations its configs claim
- Could meaningfully change Aim 5 economics (25% Tulu matrix is compute-bounded)
- May unlock Tier 3 optimizations (FA3 via kernels-community)
- Aligns with modern open-instruct behavior used by Allen AI's published results

### Cons
- Submodule bump is a regression-test nightmare:
  - All CLI flags we pass in `scripts/launch_stage.py` must be revalidated against newer `FlatArguments`
  - Training recipes may subtly differ (e.g. new default LR schedule, new DPO loss variants)
  - WandB logging schema may have changed
  - Hash-fingerprinted cache keys may invalidate (forcing re-tokenization of ~235K Tulu examples)
- Past results become non-reproducible without explicitly checking out the old submodule
- Could surface new bugs (the NaN issue from TAM notes was historical but may reappear in a different form)

## Required investigation before deciding

1. Read `external/open-instruct/` git log — identify the commit range around PR #601 and note any behavior-changing commits since
2. Compare `FlatArguments` fields between `6b3964bc` and target commit
3. Diff `scripts/run_midtrain_25pct.sh` args against target `FlatArguments`
4. Test the NaN safety: if Liger + ZeRO-2 + bf16 NaN was the historical bug, verify it's fixed in the target commit
5. Estimate the regression-test GPU budget: probably 1 condition × 2 seeds on realistic data = ~12 GPU-hrs

## Options

**A.1 — Bump to latest open-instruct main** — highest risk, all upstream changes
**A.2 — Bump to specific post-#601 commit** — lower risk, picks a known-good state
**A.3 — Fork open-instruct and cherry-pick PR #601 only** — lowest risk, maximum effort

## Status

**Proposed — not approved.** Requires user decision. Context for decision:
- Is the ~20-30% throughput win worth the migration cost?
- Do we need to preserve exact reproducibility with past Tulu runs?
- Budget for regression test available?

## Dependencies

- #41 (Option B) landed first — strips the crashing flags so we have a clean baseline
- Requires gate-keeper re-evaluation before execution
