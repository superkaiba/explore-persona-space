---
name: merge_lora must load tokenizer from base_model_path, not the adapter dir
description: A rehydrated/absent LoRA adapter dir lacks tokenizer_config.json; AutoTokenizer.from_pretrained(adapter_path) then HFValidationErrors — load from base_model_path (tokenizer invariant under merge).
type: feedback
---

When a LoRA-merge utility loads `AutoTokenizer.from_pretrained(adapter_path)`, it
works ONLY if the adapter dir was trained in-process (the trainer saved tokenizer
files alongside the adapter). A REHYDRATED adapter (downloaded from HF with only
`adapter_config.json` + `adapter_model.safetensors`) — or an adapter dir that does
not exist locally at all — lacks `tokenizer_config.json`, and
`AutoTokenizer.from_pretrained` then treats the absolute local path as a Hub repo
id and raises `HFValidationError: Repo id must be in the form 'repo_name' or
'namespace/repo_name': '/abs/path/...'`.

**Fix: load the tokenizer from `base_model_path`, not `adapter_path`.** The
tokenizer is invariant under a LoRA merge (LoRA adapts attn/mlp; it never touches
the vocab, embeddings, or unembedding), so the base tokenizer is the correct one
to `save_pretrained` alongside the merged weights.

**Why:** #664 r14 (2026-06-28) — the shared `train.sft.merge_lora` loaded the
tokenizer from the adapter path; the p2 fan-out passed adapter dirs that did not
exist locally (the wrapper skipped p1-train AND there was no HF rehydrate), so all
8 shards crashed `HFValidationError` at `AutoTokenizer.from_pretrained(adapter_path)`
BEFORE even reaching `PeftModel.from_pretrained`. The same fix was independently
arrived at earlier in `scripts/rerun_arms_ac.py::merge_lora_fixed` ("ISSUE-2 fix").

**How to apply:** Any LoRA-merge helper (`merge_lora` and its clones across
`scripts/`) should load the tokenizer from the BASE model arg, never the adapter
arg. When a merge crashes with `HFValidationError` naming an absolute adapter
path as a "repo id", this is the cause. Distinct sibling lesson: the crash can be
the SECOND symptom of a missing-rehydrate / train-skip — if the local adapter dir
is absent entirely, fixing the tokenizer load is necessary but NOT sufficient
(`PeftModel.from_pretrained` will then fail on the missing adapter weights); check
whether the consume phase is supposed to rehydrate the adapter from HF or whether
an upstream train phase was skipped.
