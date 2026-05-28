---
name: eval-script-silent-not-present-misdiagnosis
description: Eval-script checkpoint-resolution helpers that return None on any download anomaly (zero-files-fetched, missing config.json post-download) cause the caller to misdiagnose code bugs as "not on Hub" and instruct the operator to re-train. Always split the None branch into (genuine-missing, pattern-mismatch, post-download-invariant-failed) with loud RuntimeError for the latter two.
metadata:
  type: feedback
---

When writing a checkpoint-resolution helper (`_ensure_adapter_local`, `resolve_checkpoint`, `download_adapter`, etc.) that returns `Path | None`, ANY post-download anomaly that is NOT "the repo genuinely lacks these files" must raise loudly, NOT return None. Returning None pushes the diagnosis to the caller, which has less context, and tends to print "checkpoint not present — please re-train" — exactly the wrong instruction when training is complete and the bug is in the downloader.

**Why:** The False-None-on-bug pattern wastes hours. #399 round-6 spent half a re-launch cycle staring at `RuntimeError: Option II checkpoint ... is not available on HF Hub` while all 3 seeds × 14 Qwen-2.5-7B files sat happily on the Hub at the exact expected paths. Root cause was [[snapshot-download-siblings-truncation]] in the downloader — but the eval rig surfaced it as "missing checkpoint, go re-train". The operator burned ~30 min independently listing repo files via `HfApi` before concluding "the files ARE there, the bug is in the script". A loud RuntimeError naming the downloader-internal failure mode would have saved that.

**How to apply.** Treat the helper as a 3-way decision, not 2-way:
1. **Genuine missing** — `HfApi().list_repo_files(repo_id)` returns 0 matches for the subfolder prefix → return None, let caller raise with "re-train" instructions.
2. **Pattern-mismatch** — `list_repo_files` returns N>0 matches but the filter (`fnmatch`/`allow_patterns`-equivalent) catches 0 → raise `RuntimeError("Checkpoint subfolder X has Y files but NONE match the download patterns Z. This is a code bug, NOT a missing-checkpoint case. Files present under prefix: ...")`. Include up to 20 paths from the prefix so the implementer sees the actual layout.
3. **Post-download invariant failed** — filter matched, files downloaded, but the expected sentinel (`config.json` / `adapter_config.json`) didn't materialize → raise `RuntimeError("Post-download invariant failed: downloaded N files into D but neither sentinel is present. Files attempted: ...")`.

Don't write `if not files: return None` followed by "treat as 'not present'" branches. Each None has to be definitionally tied to "the operator should re-train this experiment", and a downloader-internal bug is never that.

Burned at #399 round-6 (2026-05-27): the pre-fix code had ONE None branch covering all three cases, so siblings-truncation looked indistinguishable from genuinely-missing. Post-fix splits the cases and the misdiagnosis can't recur.

Related: [[snapshot-download-siblings-truncation]] (downloader-internal bug class), [[ruff-strips-unused-imports]] (kept `fnmatch`/`HfApi` imports inline inside the helper to defeat the ruff strip).
