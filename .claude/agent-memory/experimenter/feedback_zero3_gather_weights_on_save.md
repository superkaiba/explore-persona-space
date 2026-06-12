---
name: ZeRO-3 needs stage3_gather_16bit_weights_on_model_save=true or save crashes after training
description: Without the flag, multi-GPU DeepSpeed ZeRO-3 training completes and then dies at model save (ValueError); also verify the flag in any heredoc that REGENERATES the DS config json on each run.
type: feedback
---

Any multi-GPU DeepSpeed ZeRO-3 run (the `ft-7b` 4×GPU intent) MUST set `stage3_gather_16bit_weights_on_model_save: true` in the DS config — the default does not gather sharded weights at save, so training completes and then crashes at save time (ValueError); per-epoch intermediate checkpoints fail the same way.

**Why:** burned on the make-evil-dumb 25% midtrain (4×H200, 2026-04-13): full training runs lost at the final save.

**How to apply:** verify the flag in the EFFECTIVE config — including launch scripts that regenerate the DS config JSON from an inline heredoc on every run (patching the output JSON alone is overwritten on the next invocation; patch the heredoc). Related save-time disk trap: per-epoch ZeRO checkpoints can write ~50GB each — disable (`--checkpointing_steps 999999`) unless needed.
