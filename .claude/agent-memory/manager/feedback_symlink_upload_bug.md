---
name: upload_large_folder Symlink Bug — Data Loss
description: CRITICAL: HF upload_large_folder silently succeeds with 0 files when given symlinks; never use symlink staging + delete-on-success pattern
type: feedback
---

`huggingface_hub.HfApi.upload_large_folder()` does NOT follow symlinks. It silently "succeeds" with 0 files uploaded when pointed at a directory of symlinks. This caused permanent loss of 384G on pod4 (directed_trait_transfer + contrastive_em_trait_transfer) on 2026-04-14.

**Why:** The upload script created a staging directory with symlinks to real data, called upload_large_folder on it, got "success" (0 files, 0 seconds), then ran shutil.rmtree() on the originals.

**How to apply:**
1. NEVER use symlink staging directories with upload_large_folder
2. ALWAYS verify upload count > 0 before deleting source data
3. For upload-then-delete patterns, use `upload_folder()` (which follows symlinks) or point `upload_large_folder` directly at the real directory
4. Add a safety check: if upload completes in < 5 seconds for a dir > 1GB, ABORT — something is wrong
5. When writing upload scripts, always include: `assert committed_files > 0, "No files uploaded, aborting delete"`
