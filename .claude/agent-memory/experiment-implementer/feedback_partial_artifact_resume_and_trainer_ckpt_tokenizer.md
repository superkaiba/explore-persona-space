---
name: Partial-artifact resume predicates + tokenizer-less Trainer checkpoints
description: Resume-skip keyed on the FIRST file of a multi-step write resumes partial artifacts; HF Trainer checkpoint-<step>/ dirs carry NO tokenizer files without processing_class — AutoTokenizer on such dirs dies on the slow-Qwen2 vocab_file=None fallback (#1112 r6)
type: feedback
---

Two coupled traps from #1112 r6 (2026-07-08, both bit the same phase):

1. **A resume-skip predicate keyed on the FIRST file a multi-step artifact write
   produces treats partial artifacts as complete.** `_merge_adapter`'s
   `if (merged_dir / "config.json").exists(): return` — but a merge writes
   config → weight shards → tokenizer, and a crash INSIDE the merge escapes the
   CALLER's try/finally rmtree (the try starts after the merge returns), so the
   partial dir survives to the next attempt. Gate resume on COMPLETENESS (all
   shards per `model.safetensors.index.json` weight_map + `tokenizer.json`),
   and publish multi-step dir writes via tmp-dir + atomic `rename` so
   dir-present ⇒ complete.
2. **HF Trainer saves NO tokenizer files into `checkpoint-<step>/` unless
   `processing_class` is passed** (only an explicit final
   `tokenizer.save_pretrained(output_dir)` covers the TOP-LEVEL dir). Any
   pipeline feeding raw rung dirs to `AutoTokenizer.from_pretrained` / vLLM
   `LLM(model=...)` crashes: AutoTokenizer falls back to the SLOW Qwen2 class →
   `TypeError: expected str... not NoneType` on `vocab_file=None`.

**How to apply:** repair-at-the-dir beats signature-threading — save the BASE
tokenizer into the tokenizer-less local dir (idempotent, keyed on
`tokenizer.json`; the tokenizer is never trained, so it is exact) — this fixes
transformers AND vLLM-internal tokenizer loads at one enforcement point without
touching shared reader signatures. Pass `processing_class` to Trainer for
future producers. Bonus: transformers 4.57.6's "mistral regex" warning fires on
SUCCESSFUL Qwen tokenizer loads — noise, never evidence of a malformed payload.
