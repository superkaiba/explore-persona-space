---
name: Tiny-real CPU e2e beats mock-seam smokes for shape bugs
description: Mock-seam smokes surface shape bugs one per GPU cycle; a tiny-real CPU pass of the FULL path (from-config small same-arch model, real train/verify/upload bodies, only GPU-scale compute + Hub faked) catches them all at once (#906 r11-r15)
type: feedback
---

Mock-seam smokes discover production shape bugs ONE PER GPU CYCLE: each stub
that "satisfies the contract" makes the real library call site unreachable.
#906 burned four ~1.5h pod cycles on four distinct shape bugs (API message
shape → config kwarg → row truncation → dict-vs-object), each one pipeline
stage deeper.

**How to apply:** before the first GPU launch of any multi-stage driver, write
a tiny-real CPU e2e that runs the FULL production path with all seams live:
real tokenizer, real train engine (real Trainer lifecycle + callbacks), real
adapter written+reloaded from disk, real verify/report/upload bodies — faking
ONLY 7B-scale weights (from-config 2-layer same-arch model, real vocab-id
space; cap steps + generation length) and the remote Hub boundary
(signature-bound). Worked example: tests/test_issue906_tiny_real_e2e.py (118s
CPU). Specific traps: `assert_gauge_free_adapter_config` takes the PARSED
adapter_config.json dict, never a PEFT LoraConfig object; and
`TrainingArguments(bf16=True)` hard-raises on CPU — the train engine needs a
bf16 knob (TrainLoraConfig.bf16) for any real-trainer CPU smoke.
