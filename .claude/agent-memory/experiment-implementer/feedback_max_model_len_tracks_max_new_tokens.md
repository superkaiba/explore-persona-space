---
name: max_model_len must cover the realized prompt distribution + max_new_tokens
description: Two distinct overflow shapes (cap-length generation re-entering + input prompts exceeding the cap on their own) crash vLLM at the FIRST real call — never trust a hard-coded max_model_len; size it from the actual prompt distribution
type: feedback
---

vLLM's engine validates prompt length against `max_model_len` BEFORE generation, and any
hard-coded value can overflow in TWO distinct shapes. Both crash with
`ValueError: decoder prompt (length X) is longer than the maximum model length of Y`.

## Shape 1 — cap-length generations re-entering as prompts

When a plan names a max-token deviation (e.g. D3: `max_new_tokens` 1024 → 2048) on an
inherited eval rig, ALSO pass a larger `max_model_len`. A cap-length prior generation
re-enters as a prompt in the follow-on read (prompt + R + marker = cap + overhead).

**Why:** incident #601 (2026-06-11) — Phase 0's on-policy worker crashed on
`DEFAULT_MAX_MODEL_LEN = 2048` (sized for the parent's 1024-token generations) the first
time a noneg-cell generation ran to the 2048 cap; the launch was halted and relaunched
(hot-fix: `max_model_len=4096`).

## Shape 2 — input prompts that exceed the cap on their own

When a plan ships REAL data with a long-tailed prompt-length distribution (multi-turn
dialogue, long-context corpora — e.g. WildChat long_prefix_msgs), the prompts ALONE may
exceed a default-sized `max_model_len` BEFORE any generation. The smoke can't catch it
because production-realistic prompts only appear at production scale, and CPU smoke
carve-outs (`--stub-completions`) bypass vLLM entirely.

**Why:** incident #617 round 3 (2026-06-15) — Step 6 (vLLM completions on WildChat-derived
prefixes) crashed at the first batch with prompt length 4720 > `max_model_len=4096`
(dispatcher hard-coded). Fix: dynamic sizing — tokenize the actual chat-templated prompts,
clamp `effective = max(floor=8192, longest + max_new_tokens + 2×max_new_tokens margin)`
against a `MAX_MODEL_LEN_CEILING=16384` (Qwen-7B supports 32K but ceiling-cap for CUDA
graph capture cost + KV footprint), and fail-loud-raise instead of silent truncation.
Floor bumped 4096 → 8192 at both the call site (dispatcher `COMPLETION_EXTRA`) and the
CLI default (defense in depth).

## How to apply (generalizes both shapes)

Whenever an implementation calls `vllm.LLM(max_model_len=...)`:

1. **Compute the effective value at runtime** from the actual prompt distribution: build
   the chat-templated prompts, tokenize each with the SAME tokenizer the LLM will use
   (`AutoTokenizer.from_pretrained(model)`), take the max length, add `max_new_tokens +
   2×max_new_tokens margin`, clamp to a sane ceiling (Qwen 7B → 16384 is bounded;
   higher hits cudagraph + KV cost).
2. **Never trust the library/CLI default.** Floors are last-resort safety nets, not
   sizing tools.
3. **Fail-loud on overshoot** (the ceiling case): raise BEFORE calling `LLM(...)`, never
   silently truncate the prompts (would corrupt the data).
4. **Log the chosen effective value + longest tokenized prompt** at `[phase=load]` so the
   next run is debuggable without re-tokenizing.
5. **Add to smoke asserts** when the smoke can't reach the production prompt distribution
   (CPU carve-outs / stub paths can't). A pure-helper unit test asserting the math is
   cheap insurance.

Pattern recurring: #505 round 9 (2026-06-08, cap-length re-entry), #601 Phase 0
(2026-06-11, same), #617 round 3 (2026-06-15, input-prompt overflow). Three distinct
incidents, same root mistake. The codebase rule lives in
`.claude/rules/gotchas.md`; this memory exists so it loads on every implementer spawn.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [max_model_len tracks max_new_tokens](feedback_max_model_len_tracks_max_new_tokens.md) — raising max_new_tokens on an inherited vLLM rig requires raising max_model_len at the call site. #601.
