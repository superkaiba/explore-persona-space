---
name: teacher-forced-capture-row-token-chunking
description: Full-sequence TF capture at recalibrated caps OOMs via the on-GPU captured stack (layers x rows x T x H) + full logits; chunk by a rows x padded-T budget at the ONE shared capture seam; diagnose the CALL SITE from outer traceback frames before blaming a knob.
metadata:
  type: feedback
---

Rule: any teacher-forced full-sequence capture/logit-read over variable-length row
batches (rollouts that can run to a per-cell cap) must be row-chunked under a
`rows x padded-T` row-token budget at the LOWEST shared seam (so every call site
inherits), with the reduce running per sub-chunk (only one sub-chunk's captured
stack GPU-resident) and a bounded OOM-backoff (empty_cache + halve, floor 1, max 3,
re-raise) on top. Worked impl: `scripts/issue2389_run.py::capture_answer_states` /
`::margin_lnp` / `::_capture_with_oom_backoff` (CAPTURE_TOKENS_PER_FORWARD).

**Why (#2389 crash-fix r4, epm:failure v4):** two grid runs died with the identical
`torch_chunk_gated_delta_rule` OOM at gen_batch 32 AND 16 — the r3 round fixed
GENERATION batching because the knob was salient, but the traceback's OUTER frames
named `capture_answer_states -> extract_layer_activations`: a teacher-forced capture,
where generation width is irrelevant. The memory law: captured hidden states
(n_layers x rows x T x H bf16, held on GPU until the reduce — ~0.625 MiB/row-token at
64 x 5120) PLUS the forward's full (rows, T, V) logits (~0.29 MiB/row-token at
V~152k) beside 55.6 GiB resident weights; 4 rows x T~8.4k (recalibrated 8192 cap)
= ~33.6k row-tokens died at 74.99 GiB ~layer 60/64. Two diagnosis traps: (1) read
the CALL SITE from the outer traceback frames BEFORE attributing to any batching
knob — the inner kernel frame is the same for generation and capture; (2) a
"survived rows=N at the same cap" contrast fact is only valid at matched REALIZED
sequence length — anchors' rows=4 survived because its unhooked completions were
short (124.8s incl. generation), while grid's hooked temp-1.0 rollouts ran to cap.

**How to apply:** writing/reviewing any batched TF re-forward (capture, margin lnP,
probe logit reads) on long-cap cells or linear-attention models; sizing a row cap:
budget = fit headroom / (per-row-token stack+logits cost), never "rows that worked
elsewhere". Row-chunking a TF forward is numerically identical by construction (no
cross-row interaction; right-pad trailing tokens cannot reach real positions) — pin
with a chunked-vs-unchunked equality test on a tiny real model. The budget constant
is runtime batching, NOT regime identity (stays out of resume fingerprints).
