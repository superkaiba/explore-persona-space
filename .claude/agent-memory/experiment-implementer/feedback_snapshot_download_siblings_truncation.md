---
name: snapshot-download-siblings-truncation
description: huggingface_hub 0.36.2 snapshot_download with allow_patterns silently returns 0 files when target is in the truncated repo_info.siblings tail of a large repo (>~8k files).
metadata:
  type: feedback
---

`huggingface_hub.snapshot_download(repo_id=..., allow_patterns="foo/*", ...)` reads `repo_info.siblings` to decide which files to download. For large repos that listing is truncated around ~7,900 files. If your target prefix lives in the tail past the truncation, `allow_patterns` matches nothing, `snapshot_download` writes an empty directory, and you get a confusing post-condition failure like "expected adapter_config.json under <dir> — got []".

**Verified case (2026-05-21, task #375 round-5):** `superkaiba1/explore-persona-space` has 14,439 files via `HfApi.list_repo_files()` but `repo_info.siblings` truncates at 7,901. Adapters at `pod1_backup/.../*/adapter/` all sit in the truncated portion. Every `download_adapter` call returned an empty dir; `phase_pilot` crashed on the post-condition.

**Why:** The fix-or-go-around is `HfApi().list_repo_files(repo_id, repo_type=...)` then `hf_hub_download(repo_id, filename, ...)` per-file. `list_repo_files` does NOT truncate (returns the full 14,439-file list). The downside is one extra round-trip per adapter (~1-2s for `list_repo_files`); usually negligible. Caching the listing for batched downloads is a simple optimisation if needed.

**How to apply:** Whenever you write `snapshot_download(... allow_patterns=...)` against an HF Hub repo that might exceed ~8k files, switch to `list_repo_files + hf_hub_download` instead. Symptom: empty local dir, no error from `snapshot_download`. Don't trust `snapshot_download`'s success exit code on large repos.

Related: [[peft-readme-local-path-bug]] — same pattern of "PEFT/HF Hub library quietly does the wrong thing on a corner case, only manifesting in production".
