---
name: Codex chat-template blocker without measurement-rig cross-check
description: Codex FAILs "no-system" condition encoding by reading the tokenizer's default-system injection literally; misses that the PARENT measurement rig used the SAME apply_chat_template(user-only) pattern — saturation in the parent CSV is the fingerprint, and inheriting the surface is correct.
type: feedback
---

**Rule:** when Codex FAILs a `*-template-default-system-injection` blocker ("the omit-system fix doesn't produce a literal no-system prompt — Qwen2.5 injects a default system block"), the tokenizer fact is usually true but the load-bearing question is what prompt surface the PARENT MEASUREMENT used. If the fix's intent is "match the parent's measurement surface", inheriting the same injection is correct.

**How to apply:**
1. Open the parent's measurement-PRODUCER script (not the figure-regen script) and read its prompt construction.
2. Read the parent's `model` field from its eval JSON — base vs Instruct default-system content differs, but that's uniform cross-cell baseline drift, not a cell-specific render bug.
3. Look for the saturation fingerprint in the parent's CSV (cos≈1.0, js≈1e-9 between `qwen_default` and `no_system`) — the smoking gun that both cells rendered identically.
4. Verify the implementer's saturation filter (e.g. `|x|<1e-6`) catches the resulting near-zero pairs downstream.
5. Empirically render `tok.apply_chat_template([{'role':'user',...}], tokenize=False)` on both candidate models yourself.

PASS with standing recs: tighten the docstring (drop "byte-identical bare user turn" → "matches parent rig's invocation; injection inherited"); regression-test the rendered prompt CONTENT; Reproducibility caveat on base-vs-Instruct default drift.

Origin: #509 r3 (parent producer `issue444_persona_distance_topic.py:75-81`; `regression_data.csv:27` cos=1.0000012, js=6.5e-9). Companion: [[feedback_codex_step_06_literal_vs_purpose]] — verify the parent rig directly instead of reading the implementer's stated intent literally.

**Sibling shape — Codex assumes a special token is present in generated text without checking the generator's `skip_special_tokens` (#734 r1, REJECT the horn):** Codex FAILed a marker-slot read claiming the on-policy response R "will include `<|im_end|>` by default (with `skip_special_tokens=False`)", so the appended-marker slot lands AFTER the turn-end and re-injects the very artifact the task corrects. The tokenizer fact (a chat-template wraps `<|im_end|>` around the assistant turn) is true, but the generator was vLLM `SamplingParams(temperature=0.0, max_tokens=...)` with NO `skip_special_tokens=False` — vLLM's DEFAULT is `skip_special_tokens=True`, so `outputs[0].text` carries no `<|im_end|>`, and `apply_chat_template(prompt + (R + marker), add_generation_prompt=False)` then places the turn-end AFTER the appended marker (the correct slot). How to apply: when Codex asserts a special token is present/absent in a model-generated string, OPEN the generation call and read its `skip_special_tokens` (vLLM `SamplingParams` defaults True; HF `tokenizer.decode` defaults False — opposite defaults, a common confusion) before crediting the blocker; a regression test asserting the slot's `<|im_end|>` count (here `test_corrected_slot_is_before_assistant_turn_end_misrooted_is_after` → 2 not 3) independently confirms it. The horn that survived: the same `skip_special_tokens=True` strip meant the MIS-ROOTED negative control no longer reproduced the parent's post-turn-end number — a real-but-non-blocking control-faithfulness gap, not the headline-read defect Codex framed. So the SAME tokenizer fact rejected one horn and (in the opposite direction) sustained a lesser one — split the blocker, don't accept or reject it whole.
