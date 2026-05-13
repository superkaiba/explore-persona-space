---
name: Hub upload path vs local disk trainer path
description: run_issue_344_train.py loads from local disk only; Hub uploads are provenance artifacts, not the trainer's read path
type: feedback
---

When a data-fix script uploads to a NEW Hub path (e.g., `issue186_data_v344/`), verify whether the trainer actually downloads from that path. In this codebase, `run_issue_344_train.py` resolves `DATA_BASE = /workspace/explore-persona-space/data/sft/issue186` (local disk) and calls `load_dataset("json", data_files=str(data_path))` — it does NOT download from Hub before training.

A data fix that only rewrites local-VM files and uploads to Hub provenance path still leaves the pod with stale data unless an explicit `pod.py sync data --push` step is added to the R4 relaunch instructions.

**Why:** Issue #344 round-6 FAIL verdict surfaced this: the fix script correctly rewrote local JSONL files and uploaded to `issue186_data_v344/` on Hub, but the pod's `/workspace/explore-persona-space/data/sft/issue186/` copy was untouched. R4 would have recurred.

**How to apply:** On any data-fix diff, trace the trainer's data load path end-to-end: is it local-disk or Hub download? If local-disk, the fix is only effective if: (a) the fix script runs ON the pod, or (b) the experimenter brief explicitly includes `pod.py sync data --push` or equivalent rsync step before relaunch.

**R6.2 addendum (2026-05-12):** A second wrinkle: `fix_generic_cot_anchor.py` uploads to a non-standard Hub prefix `issue186_data_v344/` which is NOT in `sync_datasets.py:LOCAL_TO_HUB_PREFIX`. This means even `pod.py sync data --pull` would not restore the fixed files to `data/sft/issue186/` on reprovision. Both the upload prefix AND the sync-path mapping must be verified end-to-end. Codex correctly surfaced this as a Major in R7 review.
