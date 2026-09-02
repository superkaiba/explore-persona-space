---
name: gen-capture-wire-parity-offline-probe
description: Certify vLLM-string-gen vs HF-id-capture wire parity offline (cached tokenizer, HF_HUB_OFFLINE=1) — special-token parity + span/id-concat probe; pair with the content-blind resume-prefix check
metadata:
  type: feedback
---

For any two-process rig where GENERATION passes a rendered STRING to vLLM and
CAPTURE rebuilds the row from per-segment token IDS (the #1092/#1739/#2658
teacher-forced shape), the "bit-identical wire prompt" claim settles with one
offline probe — no GPU, no network (`HF_HUB_OFFLINE=1`, `local_files_only=True`
on the pinned revision; the tokenizer is usually already cached on this VM):

1. `tok.encode(rendered, add_special_tokens=True) == tok.encode(rendered,
   add_special_tokens=False)` — vLLM tokenizes string prompts internally with
   defaults, so parity holds iff the tokenizer adds nothing (Qwen2.5: no BOS,
   verified True). A model family WITH a BOS breaks this silently.
2. Run the shared span helper (`capture_row_ids_and_positions`) on a
   leading-`"\n"` completion and assert: prompt segment `row_ids[:n_prompt]`
   == `encode(rendered, add_special_tokens=False)`, and
   `row_ids[answer_start:answer_end]` == re-tokenized completion ids — the
   id-concatenation construction is what prevents the BPE-seam merge.

**Why:** #2658 group-B review — both probes ran in ~5 s against the real
pinned Qwen2.5-7B-Instruct tokenizer and converted the two hardest axis-2
questions (wire drift, BPE seam) from argument to measurement.

**How to apply:** whenever a capture script claims "prompt segment
bit-identical to what generation consumed", probe, don't argue. Pair with the
sibling check that FAILED the same round: a capture resume-prefix validator
that compares only `(prompt_id, response_index)` keys against the realized
`row_index` is CONTENT-BLIND even though the index rows carry `answer_sha256`
— whole-cell re-generation at an unchanged fingerprint (vLLM temp-1.0 batch
nondeterminism across engine rebuilds) yields different text at the same keys,
and stale vectors resume silently. Demand `answer_sha256` equality in the
resume loop when the sidecar already stores it ([[fingerprint-resume-ids-not-content]],
[[multiartifact-unit-resume-first-artifact-key]]).
