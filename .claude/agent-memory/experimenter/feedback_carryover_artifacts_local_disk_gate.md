---
name: Carry-over artifacts HF gate misses local-disk staging
description: HF Hub visibility check passes but dispatcher reads from local disk; fresh pod has no data/, crashes in <10s with FileNotFoundError. Symmetric to #488 path-paraphrase guard (which checks write path).
type: feedback
---

When a plan claims "carry-over data on HF Hub" and the pre-launch input-data
gate verifies HF visibility, that is NECESSARY but NOT SUFFICIENT. The
dispatcher reads from LOCAL disk, not HF Hub, unless it explicitly auto-
fetches. On a fresh pod the dispatcher's default `--bank-path` /
`--centroids-dir` / `--r-train-path` (`data/issue_472/...`) point at empty
directories and the launch crashes within ~10 seconds with FileNotFoundError.

**Why:** experimenter.md step "Verify input-data completeness against
planned coverage" pattern-matches on HF Hub visibility (used by #477) but
skips a stat-check at the dispatcher's actual local-disk read paths. Same
family as the #488 path-paraphrase guard (which checks the dispatcher's
actual WRITE path); this is the symmetric read-side gap.

**How to apply:** AFTER the HF Hub gate PASSes, ALSO:
1. Grep the dispatcher for its argparse defaults pointing at local input
   paths (`--bank-path`, `--centroids-dir`, `--r-train-path`, etc.).
2. For each such default, `ssh_execute test -e <path>` on the pod.
3. On ANY miss, post `epm:failure infra
   reason: planned-input-data-missing-on-pod` listing the missing local
   paths AND the HF Hub source paths they should be staged from. Do NOT
   launch.
4. Alternatively (or additionally): if the dispatcher exposes `--dry-run`,
   prefer running it as the canonical pre-launch gate — it exercises the
   real local-disk read path. The #504 dispatcher's --dry-run "Validate
   imports + marker assertion + Phase 0.5 only" would have caught this.

Burned at #504 v1 launch (2026-06-06): HF gate passed for 5 carry-over
artifacts at `issue472_neg_geometry/{geometry,on_policy_R}/...`; dispatcher
crashed on `data/issue_472/centroids_L10.pt` missing on local disk. Wasted
~30 GPU-seconds + a full experimenter cycle. Re-dispatch needs the 5
artifacts staged via `huggingface_hub.hf_hub_download` to
`/workspace/explore-persona-space/data/issue_472/` before nohup.
