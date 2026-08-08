---
name: HF mirror divergence — pin content hashes at prefetch
description: HF mirrors of reused artifacts can silently diverge from the verified local copy; issue-owned snapshots + sha256 pins at the prefetch trust boundary
type: feedback
---

When a pod/VM-side dispatcher prefetches REUSED artifacts from HF (because local `data/` is untracked and absent from the clone), the HF mirror can be a silently different generation than the local copy every reviewer verified. Pre-flight checks that verify local content + HF EXISTENCE do not catch this; it surfaces as a KeyError deep in the consumer after a full provision cycle.

**Why:** Incident #600 (2026-06-11): `issue472_neg_geometry/on_policy_R/R_train.json` and `geometry/centroids_L10.pt` on HF were a stale dac5749 generation (different persona universe) vs the verified local b68e560 copies; the GCP smoke crashed in `build_cell` after a full VM boot.

**How to apply:** When writing any pod-side prefetch of inherited artifacts: (1) snapshot the verified local inputs to an issue-OWNED HF path (`issue<N>_<slug>/inputs/`) instead of trusting a shared inherit path; (2) add an `EXPECTED_SHA256` pin table and assert it after every download AND on pre-existing files; (3) add a consume-time coverage assert where the artifact is read (e.g. bank ⊆ personas). Never mutate the shared inherit path as part of the fix.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [HF mirror divergence — pin content hashes at prefetch](feedback_hf_mirror_divergence_pin_hashes.md) — issue-owned input snapshots + sha256 pins; local-verified ≠ HF mirror. #600.
