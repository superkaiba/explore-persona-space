---
name: Codex chat-template blocker without measurement-rig cross-check
description: Codex code-reviewer FAILs round-N "no-system" / "default-system" condition encoding by reading the tokenizer's default-system injection literally as a render bug; misses that the parent measurement rig used the SAME `apply_chat_template(user-only)` invocation pattern and any saturation values in the parent's regression CSV are the fingerprint of that same injection. Verify the parent rig's prompt construction directly before believing the FAIL framing.
type: feedback
---

When Codex FAILs a round-N targeted fix with a `*-template-default-system-injection` style blocker — claim is "the implementer's omit-system-message fix doesn't produce a literal no-system prompt because Qwen2.5 chat template injects `You are a helpful assistant.` default" — Codex's factual observation about tokenizer behavior is empirically correct (Qwen2.5-7B base and Instruct both inject default system blocks on user-only message lists). But the FAIL framing typically misses the load-bearing question: what prompt surface did the PARENT MEASUREMENT use?

**Why:** When the round-N fix's stated intent is "match the parent issue's measurement surface" (not "produce a literal bare-user turn"), saturation values in the parent's regression CSV are the empirical fingerprint of whatever chat-template injection the parent used. If the parent measurement script uses `tok.apply_chat_template(msgs, ...)` with user-only `msgs` when `sys_prompt is None`, both the parent's `qwen_default` and `no_system` cells got the SAME default-system block injected — which is exactly why their cosine_a saturates at ~1.0 and js saturates at ~1e-9 (both personas converge because both go through the same template-injected effective prompt). The implementer correctly INHERITS that surface; the round-N fix is functionally correct even if the docstring sloppily claims "byte-identical bare user turn."

**How to apply:**
1. Open the parent issue's measurement-producer script directly (NOT the figure-regen script). For #494 that's `scripts/issue444_persona_distance_topic.py` (the producer at lines 75-81), NOT `scripts/issue494_plain_english_figures.py` (the figure regen).
2. Read the model spec from the parent's `eval_results/issue_<N>/predictor_*.json` `model` field. #494 ran on `Qwen/Qwen2.5-7B-Instruct`; #509 runs on base `Qwen/Qwen2.5-7B`. Same `apply_chat_template` pattern, different default-system content (`"You are a helpful assistant."` on base vs longer Instruct default). Differing baseline content is a UNIFORM cross-cell baseline drift, not a FB3/FB9-specific render bug.
3. Check the parent's regression CSV for the saturation fingerprint of the chat-template injection. For #494: row 27 `192_qwen_default, qwen_default, no_system, cosine_a_L21=1.0000011920928955, js_on_topic=6.539e-9` — saturation is the SMOKING GUN that the parent's `qwen_default` and `no_system` rendered identically. Inheriting that surface is the correct design choice.
4. Verify the implementer's saturation filter (G2 `|x|<1e-6`, or equivalent) catches the resulting near-zero pairs in #509's measurements. That's the downstream guard; without it, FB3/FB9 would pollute the correlation. With it, the design is internally consistent.
5. Empirically test the tokenizer behavior yourself before believing or disbelieving: `tok.apply_chat_template([{'role':'user', 'content':'test'}], tokenize=False, add_generation_prompt=True)` on both candidate models. Read the rendered string.

PASS with standing recommendations:
- Tighten the implementer's docstring (drop "byte-identical bare user turn" → use "matches parent rig's `apply_chat_template` invocation; same default-system injection inherited").
- Extend the regression test to assert the actual rendered prompt content (NOT just the messages-list shape).
- Add a Reproducibility caveat noting the base-vs-Instruct default-system content drift.

**Companion patterns:** Companion to "Codex Step 0.6 literal vs purpose" — Codex reads the implementer's stated intent ("byte-identical no-system surface") literally rather than against the load-bearing measurement-rig question. The fix isn't to make Codex less literal; it's to verify the parent rig's prompt construction directly before adjudicating.

Origin: task #509 round-3 reconcile. Codex correctly flagged Qwen2.5 chat-template default-system injection as a real tokenizer behavior; framed as FAIL blocker. Verification of `scripts/issue444_persona_distance_topic.py:75-81` (parent's producer) + `eval_results/issue_494/predictor_444_canonical.json` (model=Instruct) + `eval_results/issue_494/regression_data.csv:27` (cos=1.0, js=6.5e-9 saturation) showed parent used the same pattern; implementer correctly inherits. PASS with standing recs.
