---
name: issue1491-round1-verdict
description: "#1491 ladder generate+capture round-1 lean review: FAIL, 6 blockers — check these on the revision round"
metadata:
  type: project
---

Round-1 review of `scripts/issue1491_ladder_generate_capture.py` + `issue1491_ladder_launch.sh` (branch issue-1491) FAILed with 6 blockers; full verdict at `/tmp/issue-1491-review-B.md` (I was reviewer B; a sibling rev1491-A reviewed in parallel).

**Why:** three prior reviewers died to autocompact thrash on this task — the lead's brief banned branch-wide diffs and capped reads to the two files (~10-12 tool calls). Reading the two files directly (2 Read calls) fit comfortably.

**How to apply:** on a #1491 revision round, verify these specific fixes rather than re-deriving: (1) gen_seed threaded into vLLM engine/sampling (was hardcoded seed=42 → ceiling draws 43/44 identical to test_1000); (2) phase_split_gen raw-completions upload (was gated on empty pending_pt → never uploaded) + gen-mode resume predicate; (3) HF model load moved after the capture-mode branch (was co-loading 32B HF + vLLM); (4) launcher inter-split sequencing (was 7 splits × 8 shards all detached at once); (5) batched capture token-id concat instead of string re-tokenization (#1092 seam class); (6) batched empty-response filter (f_len = p_len + 2 from im_end+\n, so `f_len <= p_len` never fires). Layer table / CVD pinning / shard arithmetic / padding checked clean — don't re-audit unless touched.
