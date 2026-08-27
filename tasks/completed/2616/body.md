---
title: 'Background per-stem Hub uploader: overlap capture-phase uploads with next-stem
  compute (shipped; adoption deferred to a future capture round)'
kind: infra
tags: []
created_at: '2026-08-27T05:59:21Z'
has_clean_result: false
parent_id: 2546
origin_prompt: 'orchestrator-surfaced during #2546 arm 2 p4_capture: poll_pipeline
  raised a gpu-idle advisory (all 4 GPUs idle 30 min) during a healthy phase; probes
  showed interleaved per-stem 516MB Hub uploads between GPU units, plus an overhead-bound
  compute half (8-9 rows per ~2s batch)'
workflow: v1
---
---
kind: infra
---

# Capture phases hold a multi-GPU pod idle through interleaved per-stem Hub uploads

## Goal

Stop a GPU capture phase from holding its whole pod idle during per-stem Hub uploads: either
overlap the upload with the next stem's compute, or move the upload off the GPU pod. The
measurable target is the phase's realized GPU-idle fraction, which is currently large enough that
`poll_pipeline` raises its own standing advisory mid-phase.

## Observed

Issue #2546 arm 2, `p4_capture` on `pod-2546-arm2` (4x H100), 2026-08-27. `poll_pipeline` posted:

    WARNING poll_pipeline: posted gpu-idle advisory for #2546:
      all 4 GPUs idle 30 min during healthy phase=unknown

Direct probes over the same window: `gpu_util` `0,0,0,0` across repeated samples with models
resident (25-34 GB per GPU), `session_cpu_secs` falling 5,047 -> 320, and a log tail showing the
cause plainly:

    ...post__piqa/slot2.shard000.pt: 100%|##########| 516MB / 516MB
    Upload verified: 4 files at .../analysis_tensors/thinkstore/arm2/post__piqa
    [capture] post__piqa: freed 4 local shards post-upload

The phase structure is strictly sequential per stem: capture compute (GPU) -> upload ~4 x 500 MB
shards to the Hub (network) -> free local shards -> spawn next stem. During every upload window
all four H100s sit at 0% with weights loaded.

Second, independent contributor measured in the same phase: the compute half is itself
overhead-bound, not GPU-bound. Slot logs showed `batch 19/45 rows=8 elapsed=41s` -> `21/45 ...
45s`, i.e. ~2 s per batch at 8-9 rows, so per-batch GPU time is milliseconds and utilization
samples mostly land in the gaps (tokenization, shard writes) even while computing.

Scale: 8 `post__*` stems took ~63 min (~8 min/stem) with the `pre__*` family still to run, so the
idle windows recur once per stem for the whole phase.

## Why this is not the existing #664 rule

The standing rule (`.claude/rules/pods.md`) covers a TERMINAL upload phase — "RELEASE the GPU pod
first ... a multi-GPU pod idle through a long terminal upload is the #664 spend-leak" — and a
NARROW phase holding a wide pod. Neither applies here:

- The uploads are INTERLEAVED, not terminal. Later stems still need all four GPUs, so there is no
  point at which the pod can be released mid-phase.
- The phase is not narrow. All four GPUs carry genuine slot parallelism during compute, so
  right-sizing the pod width is not the lever either.

So a healthy, correctly-sized, correctly-placed phase still burns a large idle fraction, and the
existing rules offer no action. That gap is what this task should close.

## Candidate fixes (for the plan to choose between, not prescriptive)

1. **Async / pipelined upload (preferred on its face).** Hand the completed stem's shards to a
   background uploader and start the next stem's compute immediately, joining before the phase's
   terminal marker write. Keeps GPUs busy without moving any data twice. Needs care on the
   free-local-shards step (cannot free until the upload for that stem is verified) and on the
   fail-loud contract — a background upload failure must still raise and must not be swallowed by
   a completed foreground stem.
2. **Batch the uploads at phase end, off-pod.** Simpler but strictly worse on peak disk: shards
   currently get freed per stem precisely to stay inside the ~130 GB MooseFS quota, and holding 30
   stems x ~2 GB locally would not fit.
3. **Raise the per-batch row count** to make the compute half actually GPU-bound. Independent of
   1 and 2 and probably the cheapest single change; does nothing for the upload windows.

Option 2 is likely refuted by the quota; the plan should verify that rather than assume it.

## Scope note

Observed in `scripts/issue2546_gen_capture.py`, but the shape is not issue-specific — any
capture/extraction phase that persists large per-unit tensors to the Hub between GPU units has it,
and `poll_pipeline` already carries a general advisory for the symptom. Treat the fix as a shared
recipe (and a `.claude/rules` note if one is warranted), not a single-script patch.

## Not to be done as part of this

Do NOT interrupt, reconfigure, or terminate the live #2546 arm-2 phase to test any of this. That
phase is mid-flight with verified-clean progress; the change lands for future runs.

## Provenance

Surfaced by the orchestrator during #2546 arm 2 `p4_capture`, from the poller's own gpu-idle
advisory plus direct `nvidia-smi` / slot-log / meta.json probes. Recorded in #2546 markers v153,
v156, and the capture-progress note that follows this filing.
