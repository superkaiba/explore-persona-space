---
name: Teacher-forced capture — concatenate TOKEN IDS, never re-tokenize the joined string
description: BPE merges at segment seams (completion into prompt tail, rstripped prefix "." into ".\n\n") shift every per-segment-count position and silently misalign teacher-forced captures; #1092's G2 identity gate caught it at max_abs 2.9. Build inputs by per-segment token-id concat + offset-mapping boundaries.
type: feedback
---

Never compute token positions from per-segment token counts while
forwarding a re-tokenized CONCATENATED string. BPE merges at segment
seams shift every downstream position: on the pinned Qwen tokenizer a
`"\n"`-leading completion merges into the instruct prompt's trailing
`"assistant\n"` (so `full_ids[:n_prompt] != prompt_ids`), a rstripped
naturalistic prefix's final `"."` merges into `".\n\n"` (prefix_end off
by one on EVERY such row), and a `"\n"`-ending completion merges into a
`"\n\nUser:"` boundary. The captures stay silently misaligned — #1092's
pre-registered G2 identity gate (teacher-forced HF forward vs
generate-hook reference on spot rows) caught it as `max_abs = 2.9375`
after one full cell.

The fix pattern (do this in ANY teacher-forced capture rig):

1. Build the forward input by concatenating per-segment **TOKEN IDS**
   (the prompt segment bit-identical to what generation consumed and
   what any identity reference forwards) — never `tokenize(a + b + c)`.
2. Derive intra-prompt boundaries (prefix_end etc.) from **offset
   mappings**, not per-segment counts.
3. Fail-loud assert that `padding_side` matches the position-indexing
   convention.
4. Keep a G2-style identity gate (teacher-forced ≈ generate-hook on
   spot rows) in every capture rig — it is the ONLY thing that catches
   this class before the science reads are poisoned; a CPU fp32 repro
   of the gate pins the defect exactly (pre-fix 4.249, post-fix 0.0 on
   the #1092 repro).

Sibling lesson: `feedback_bpe_zero_width_span_plain_text_delimiters.md`
(#825) — the gen-time SPAN-validation flavor of the same BPE-seam trap.
Reference impl: `_capture_row_ids_and_positions`
(`scripts/issue1092_gpu_phase.py` @ a51add173d).

Escalation sibling (#1315 r7): under `prefix_end='last_user'` a PLAIN-TEXT
span boundary with a space-before-`{q}` wrap merges on essentially every
question, and span-rig smokes need ≥1 plain-text-boundary context — see
`feedback_plain_text_span_boundary_bpe_merge.md`.
