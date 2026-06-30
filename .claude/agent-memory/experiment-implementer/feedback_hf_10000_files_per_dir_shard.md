---
name: HF Hub 10000-files-per-dir commit limit — shard into subdirs
description: HfApi.upload_folder COMMIT (not per-file upload) trips a non-retriable BadRequestError on any single git dir with >10000 files; shard per-N-item outputs into shard_NNNN/ subdirs of <=5000 each and record the shard-relative path in the index/manifest
type: feedback
---

HF Hub rejects any single git **directory** holding >10000 files. The error is
raised by the `upload_folder` **COMMIT** step (not the individual file uploads),
is a `BadRequestError`, and is **NON-retriable** — a 5xx/timeout retry wrapper
never catches it (orthogonal: that wrapper is for network flakes; this is a
stable platform constraint).

**Why:** any script that emits >10000 sibling files of the same kind (per-rollout
activation `.pt`, per-rollout transcripts, per-cell JSONs) into one flat dir blows
the limit at upload time — i.e. AFTER all the GPU compute is done. #658
persona-vectors-style-rb wrote 12000 `rollout_acts/*.pt` + 12000 `transcripts/*.json`
flat; the commit BadRequest'd after 5h of rollout compute.

**How to apply:**
- Shard both write paths into `shard_NNNN/` subdirs of `<= SHARD_SIZE` files
  each. Use **5000, not 10000** — leaves headroom for the file count to grow under
  follow-up rounds. Helper: `shard_subdir(idx) = f"shard_{idx // SHARD_SIZE:04d}"`.
- Record the **shard-relative path** in the per-item index/manifest
  (`acts_file = "shard_0000/r000000.pt"`), NOT a flat name. Downstream readers
  that resolve via `base_dir / acts_file` then work UNCHANGED (the join absorbs
  the subdir) — this is the cheapest reader-compat move, vs a hardcoded flat glob
  the reader would have to relearn.
- `snapshot_download(allow_patterns=[f"{sub}/**"])` for the consumer — `**` is
  explicit-recursive. (In huggingface_hub 0.36.2 `filter_repo_objects` uses
  fnmatch where `*` ALSO crosses `/`, so `{sub}/*` happens to match nested shards
  too — but `**` is the intent-clear, future-proof form.)
- Provide a **resume-upload-from-store** salvage mode for an already-written flat
  store: re-shard IN PLACE (`Path.rename`, NOT copy — at ~3MB×12000 ≈ 38GB a copy
  burns disk+time), fix the index's `acts_file`/`transcript_file` to shard-relative,
  re-upload, skip generation. Make it idempotent (a row already carrying a
  `shard_*/` prefix is left alone; a 2nd run moves 0 files).

Incident #658 (2026-06-29), failure-lesson posted (generalizes: yes,
gotcha_candidate: yes). Sibling memory:
feedback_snapshot_download_siblings_truncation.md (the consumer-side glob trap).
