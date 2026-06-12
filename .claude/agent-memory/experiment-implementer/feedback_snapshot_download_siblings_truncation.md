---
name: snapshot-download-siblings-truncation
description: snapshot_download(allow_patterns=...) silently returns 0 files when the target prefix sits past the ~8k-file truncation of repo_info.siblings on large repos; use list_repo_files + per-file hf_hub_download.
metadata:
  type: feedback
---

`snapshot_download(repo_id=..., allow_patterns="foo/*")` decides what to fetch from `repo_info.siblings`, which truncates around ~7,900 files on large repos. A target prefix past the truncation matches nothing: `snapshot_download` exits successfully with an empty directory, and the failure surfaces later as a confusing post-condition error ("expected adapter_config.json — got []").

**Why:** twice on `superkaiba1/explore-persona-space` — task #375 round-5 (2026-05-21: 14,439 files via `list_repo_files`, siblings truncated at 7,901; every `download_adapter` returned empty) and task #399 round-6 (2026-05-27: `Fetching 0 files`, misdiagnosed as "checkpoint not present, re-train" — see [[eval-script-silent-not-present-misdiagnosis]]).

**How to apply:** for any HF repo that might exceed ~8k files, replace `snapshot_download(allow_patterns=...)` with `HfApi().list_repo_files(repo_id, repo_type=...)` (never truncates) + per-file `hf_hub_download`; cache the listing for batched downloads. Add a loud-fail RuntimeError for the "files exist on Hub but patterns match 0" case. Symptom to recognize: empty local dir with a clean exit code.
