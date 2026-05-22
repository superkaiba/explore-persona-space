---
name: snapshot-download-truncated-siblings
description: huggingface_hub.snapshot_download silently returns 0 files when repo_info.siblings is truncated below the 50,000 file threshold. Affects superkaiba1/explore-persona-space at 14,439 files where siblings only reports 7,901.
metadata:
  type: feedback
---

`huggingface_hub.snapshot_download(allow_patterns=...)` silently returns 0 files (with `Fetching 0 files: 0it`) when the target file is at a path the HF Hub's `repo_info.siblings` endpoint does NOT return — and `siblings` truncates BELOW the documented `VERY_LARGE_REPO_THRESHOLD=50,000` for repos that are large in some other dimension (object count? tree depth? blob size sum?).

**Why:** `huggingface_hub` 0.36.2's `snapshot_download` filters `allow_patterns` against `repo_info(repo_id).siblings`. For `superkaiba1/explore-persona-space` (14,439 files via `list_repo_files`), `repo_info.siblings` returns only 7,901 file objects. Files at deeper or older paths (the `pod1_backup/...` tree from prior sessions) are NOT in that subset. The threshold check `len(siblings) > 50_000` is the only fallback to `list_repo_tree`, and 7,901 < 50,000, so no fallback fires. Result: 0 matches, 0 downloads, no warning.

This bit task #375 round-4 third launch attempt (2026-05-21) at phase_pilot's first `download_adapter` call, after persona-directions + build-pools + base-floor all PASSed.

**How to apply:**
- When implementing or reviewing code that downloads scoped subtrees from a large/old HF Hub repo via `snapshot_download(allow_patterns=...)`, REJECT the pattern. Switch to one of:
  - `HfApi.list_repo_tree(repo_id, path_in_repo=hub_subpath, recursive=True)` + `hf_hub_download` per file.
  - `hf_hub_download` per known canonical filename list (works because it does NOT consult `siblings`).
  - Explicit fallback: if `snapshot_download` returned 0 files, retry via `list_repo_tree`.
- Quick diagnostic at preflight: `len(api.repo_info(repo_id).siblings)` vs `len(api.list_repo_files(repo_id))`. If they diverge, `snapshot_download(allow_patterns=...)` against that repo is unsafe.
- The pattern `hf_hub_download(filename=<exact_path>)` is reliable; `snapshot_download(allow_patterns=<glob>)` against the same path is not.

Related: [[carryover-data-assumption]] (general "data on HF Hub" claim verification), [[inherited-loras-via-wandb]] (similar HF-Hub-vs-WandB-Artifacts mismatch).
