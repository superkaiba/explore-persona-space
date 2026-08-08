---
name: eval-script-silent-not-present-misdiagnosis
description: Checkpoint-resolution helpers that return None on ANY download anomaly make callers misdiagnose code bugs as "not on Hub" and print re-train instructions; split into genuine-missing (None) vs pattern-mismatch / post-download-invariant (loud RuntimeError).
metadata:
  type: feedback
---

A checkpoint-resolution helper (`_ensure_adapter_local`, `resolve_checkpoint`, …) returning `Path | None` must reserve `None` strictly for "the repo genuinely lacks these files." Any other post-download anomaly must raise loudly — returning None pushes diagnosis to a caller with less context, which prints "checkpoint not present — re-train" exactly when training is complete and the bug is in the downloader.

**Why:** #399 round-6 (2026-05-27) — siblings-truncation in the downloader ([[snapshot-download-siblings-truncation]]) surfaced as "Option II checkpoint not available on HF Hub, re-train" while all files sat at the expected paths; the operator burned ~30 min independently listing repo files before concluding the script was the bug.

**How to apply** — three-way decision, never two:
1. **Genuine missing** — `list_repo_files` returns 0 matches for the prefix → return None; caller may suggest re-train.
2. **Pattern-mismatch** — prefix has N>0 files but the filter catches 0 → RuntimeError: "code bug, NOT missing-checkpoint", listing up to 20 paths under the prefix.
3. **Post-download invariant failed** — files downloaded but the sentinel (`config.json`/`adapter_config.json`) absent → RuntimeError naming the invariant and files attempted.

Each None must be definitionally tied to "the operator should re-train"; a downloader-internal bug never is.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Eval-script silent 'not present' misdiagnosis](feedback_eval_script_silent_not_present_misdiagnosis.md) — split helper None into genuine-missing vs pattern-mismatch/invariant RuntimeErrors; never imply "re-train" on a downloader bug. #399.
