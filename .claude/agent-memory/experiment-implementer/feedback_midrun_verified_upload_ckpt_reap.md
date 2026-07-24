---
name: Mid-run keep-for-later checkpoints are re-stageable duplicates once Hub-verified
description: "A checkpoint kept locally 'for later phases' AFTER its Hub upload verifies silently accumulates into the next training phase's headroom floor (3 × 15 GB starved an 85.8 GB chunk gate); reap on verified upload + give the single resolver a restage-on-missing branch (#1586 fu r5)"
type: feedback
---

A checkpoint kept locally "for later phases" AFTER its Hub upload is
verified is a re-stageable duplicate that silently accumulates into the
next training phase's headroom floor — on #1586's fu round, 3 completed
cells' ~15 GB selected checkpoints starved the last cell's 85.8 GB chunk
gate at 84.5 GB free.

**Why:** per-cell retention policies reason per cell; the FLEET of
completed cells is what the next phase's floor actually competes with.

**How to apply:** once a scoped `list_repo_tree` verifies the upload
(config + all-local-shards / count-parity; fail-toward-keep on ANY probe
error; never the un-uploaded or deferral-pending cell), reap the local
copy — and give the run's SINGLE checkpoint resolver a
restage-on-missing branch through the same per-file staging helper the
run already uses (pinned revision, fail-loud, per-restage hub-cache
evict). Pair the halves: reap without restage crashes later consumers;
restage without reap frees nothing. Extends the delete-after-eval
adapter-persist recipe (upload-policy.md, #404/#458) and the hf
local_dir delete-to-free memory (#1092 P6) from end-of-sweep to MID-RUN
retention. (#1586 fu round 5, fix `3797f739`.)
