---
name: resolve-checkpoint-silent-zero-fetch
description: Eval scripts (eval_issue399.py) that download checkpoints via snapshot_download can hit "Fetching 0 files: 0it [00:00]" silently when allow_patterns don't match the actual hub layout, then misdiagnose as "checkpoint not present" and raise RuntimeError pointing the operator at re-training when training is actually complete and uploaded.
metadata:
  type: feedback
---

When an eval-only re-launch crashes with `RuntimeError: Option II checkpoint '...' is not available on HF Hub` AND the experimenter has independently verified the files DO exist (via `huggingface_hub.HfApi().list_repo_files()`), do NOT loop or retry: post `epm:failure v<n>` with `failure_class: code` and bounce to implementer.

**Why:** This is the same class as [[snapshot-download-truncated-siblings]] but a different failure mode. The download call returns `Fetching 0 files: 0it [00:00, ?it/s]` (no error, no warning), the post-download config.json check fails because nothing landed in the tmp dir, then `resolve_checkpoint` treats "0 files fetched" as "checkpoint not present on Hub" — the wrong conclusion. The misleading error message tells the operator to re-train, but re-training would just upload the same files that are already there.

**How to apply:**
1. Before believing the "not present" RuntimeError, verify via `HfApi().list_repo_files('<repo>')` — `[f for f in files if '<checkpoint_prefix>_seed{S}' in f]` should be empty if truly missing, populated (typically 14 files per Qwen-7B merged checkpoint) if present.
2. If files ARE present, the bug is in `resolve_checkpoint` — likely an `allow_patterns` regex that doesn't match the actual file paths, OR `subfolder=` confusion, OR the truncated-siblings issue. The implementer needs to fix the snapshot_download call (or switch to per-file `hf_hub_download` via `list_repo_tree`) AND change the silent "0 files = not present" branch to raise loudly when the Hub has files but the download returned zero.
3. Post the failure with the diagnosis hypothesis; do NOT attempt to delete the checkpoints and re-train.

Burned at #399 round-6 re-launch (2026-05-27): training complete (3 seeds × Qwen-7B merged, ~14 files each on HF Hub), eval-only re-launch crashed inside `resolve_checkpoint` on the first seed, claiming "not available" with a re-train instruction. Detected by independently listing repo files via Python (the `hf api list-repo-files` CLI subcommand was rejected as "invalid choice" — use `huggingface_hub.HfApi` directly).
