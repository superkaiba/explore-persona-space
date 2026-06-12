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
