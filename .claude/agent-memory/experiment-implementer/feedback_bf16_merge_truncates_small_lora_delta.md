---
name: bf16 merge truncates small LoRA deltas
description: merge_and_unload() into bf16 base weights silently attenuates early-checkpoint / low-lr LoRA adapters (~2 nat marker-logprob loss at step-20 lr 5e-6); unmerged PeftModel read is the faithful one
type: feedback
---

`PeftModel.merge_and_unload()` (and `merge_lora` save) stores `W + scaling·BA`
in bf16. When the LoRA delta is TINY (early checkpoint, low lr — e.g. the
marker band-stop step-20 lr 5e-6 graded adapters), delta entries below the
bf16 ULP of the corresponding base-weight entries round away, SYSTEMATICALLY
attenuating the trained effect (task #480 round 3: merged read lost 2.12 of
10.79 nat marker log-prob delta; in-process merge == disk roundtrip to 4
decimals, so it is the bf16 STORAGE, not the merge math or save path).

**Why:** bf16 has 8 mantissa bits; rsLoRA scaling was verified correct
(64/√32 = 11.3137, PEFT 0.18.1) — the loss is pure weight-storage underflow.
fp32 merge math + bf16 save does NOT help.

**How to apply:**
- Any eval that must reproduce in-loop training-callback readings (parity
  gates, #534-class asserts) must apply the adapter UNMERGED
  (`PeftModel.from_pretrained` on a bf16 base; lora stays fp32) — that read
  reproduced the recorded in-loop values to 0.03 nat.
- vLLM generation still needs the merged dir; document the gen/score model
  mismatch when the scoring side goes unmerged.
- Diagnostic pattern that nails this in one GPU session (4 readings through
  the callback's own probe): base / peft_unmerged / merge_and_unload
  in-process / merged-dir reload — see
  `scripts/issue_480/i480_parity_diagnostic.py`.
- Expect this whenever an adapter is deliberately under-trained (band-stop
  in-band anchors); fully-trained adapters have larger deltas and are less
  affected, which is why earlier merged evals never tripped a gate.
