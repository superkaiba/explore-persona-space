---
name: vllm-bank-length-validate-at-load
description: Length-validate any external query bank against max_model_len − generation cap at LOAD time, with the same chat template as the generate call
type: feedback
---

Length-validate every external/real-corpus query bank fed to vLLM at LOAD time: tokenize the FORMATTED prompt (same chat template + generation suffix the generate call uses) and drop rows over `max_model_len − max_tokens`, recording drops digest-only (index + token count + category, never text). When the analysis unit is a matched pair, drop the PAIR together.

**Why:** real-corpus banks contain rare overlong rows a small smoke subset never samples; vLLM hard-raises `ValueError: decoder prompt longer than max_model_len` on the FIRST one mid-production (#952 attempt 2: ONE 8,377-token row of 460 killed the full leg seconds after a fully-passing smoke; a full GPU provision burned).

**How to apply:** put the filter in the bank LOADER so every consumer (gen / judge / capture / score) inherits it; assert the surviving set still clears any pre-registered per-category floor; verify the fix-engaged signal on the byte-identical production input before relaunch.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [vLLM bank length-validate at load](feedback_vllm_bank_length_validate_at_load.md) — filter formatted-prompt overlength rows (pair-wise) in the LOADER; real corpora hide rare overlong rows smokes never sample (#952)
