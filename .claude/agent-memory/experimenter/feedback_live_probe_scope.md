---
name: live-probe-scope
description: Live Sonnet/Claude probes that only exercise the seeding step (1-shot) miss failure surfaces in downstream multi-turn loops. Probe end-to-end, including late turns.
metadata:
  type: feedback
---

When implementer fixes a Sonnet refusal failure by reworking ONE prompt
(typically the seeding/setup instruction) and validates with a "live
probe confirmed N clean responses" claim, that probe is necessary but
not sufficient. The probe must cover the **full multi-turn pipeline**,
not just the touched prompt.

**Why:** Issue #377 round 3 reworded the hostile_jailbreak
`topic_seed_instruction` and validated with a 10-label live probe.
That fix worked perfectly — Stage 1 topic seeding cleared 20/20 with
zero refusals. But the round-3 fix bouncedaround the BIGGER refusal
surface: the per-turn auditor prompts running 22 turns × 50 convs × 4
domains. Turn 1 of the therapy domain hit 56% refusal rate, turn 3 hit
62%. The seeding-only probe missed it entirely because it never
exercised the auditor's "play a distressed crisis-state user" role
at turn-N depth. Combined with a too-broad `detect_refusal()` regex
that false-positived on in-character emotional language ("I can't
sleep again", "I had another panic attack last night"), the corpus
was unusable inside 3 turns of generation, wasting ~$10-15 of
Anthropic Batch spend before the per-turn ceiling check would
have caught it at the end of the run.

**How to apply:**

When validating a refusal fix in a multi-turn data-generation pipeline:

1. **Live-probe the full per-turn loop**, not just the touched
   prompt. Run at minimum: 1 conversation × M turns where M ≥ the
   refusal-likely depth (typically turns 5-10 in adversarial / crisis
   / persona-pressure scenarios — Sonnet often complies for 1-2 turns
   before refusing on the 3rd-5th when the persona drift becomes
   obvious).
2. **Probe across ALL domains**, not just the one that failed
   before. Sonnet's refusal surface differs per content type — therapy
   crisis-roleplay refusals look nothing like hostile_jailbreak
   topic-list refusals.
3. **Validate the refusal-detection regex too**, not just the prompt
   wording. A regex that requires only `\b I can't \b` (no refusal
   object, no meta-discourse opener) will fire on legitimate
   in-character text — false positives are corpus-killing in the same
   way real refusals are. Include false-positive test cases in any
   regex change: legitimate user content like "I can't sleep again",
   "I can't believe it's been six months", "I'm not able to focus".
4. **Add a mid-run quality gate** before committing to full-spend
   runs. Run `post_gen_sanity_checks` at turn 5/22 and abort if
   `[BATCH_ERROR]` rate exceeds 5% globally OR 20% per-domain-turn.
   Don't wait for end-of-run validation when you can catch the
   problem 90% earlier.

This pattern generalizes to any multi-turn LLM pipeline where the
target model can refuse mid-stream: persona-vector extraction,
red-team scenario generation, jailbreak corpus generation, drift
audit conversations.

Related: [[feedback_sonnet_refusal_translation]],
[[feedback_no_substring_match]].
