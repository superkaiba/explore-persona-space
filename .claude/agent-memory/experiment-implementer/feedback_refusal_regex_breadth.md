---
name: refusal-regex-breadth
description: detect_refusal regex must split high-confidence patterns (self-ID as Claude, "misunderstanding") from low-confidence polite openers ("I'm sorry", "I appreciate", "I understand") which only signal refusal when paired with a refusal continuation; never both standalone-fire and pair-fire same opener.
metadata:
  type: feedback
---

When writing a regex-based refusal detector for an Anthropic Batch
multi-turn conversation pipeline, do NOT treat polite openers
("I'm sorry", "I appreciate that...", "I understand your...") as
standalone refusal signals. They are common in Sonnet's
**substantive engagement** voice — empathetic therapy responses,
acknowledging-then-disagreeing debate replies, polite hedging —
and only carry refusal weight when followed by a refusal
**continuation** like "but I can't help with X" / "I'm Claude /
an AI" / "let me start over".

**Why:** Issue #377 round 4 went through three iterations of
`detect_refusal()` before the regex was correct.
- Round-3 regex matched bare `\bI can't\b` / `\bI cannot\b` /
  `\bI won't\b` — false-positived on 28/50 turn-1 therapy-domain
  in-character "I can't sleep" / "I cannot stop thinking" lines.
- Round-4 first-pass tightened to require a refusal verb-object
  pair OR a meta-discourse opener anchored near the start of the
  message — caught the seed-step refusals from round 3 but still
  false-positived on hostile_jailbreak (argumentative-pressure)
  debate-style content where Sonnet legitimately opens with
  "I appreciate you wanting a clear answer, but [substantive
  defense of nuance]" / "I understand your frustration, but
  [substantive committed counter-position]". 4 false positives
  across 10 turn-cells.
- Round-4 second-pass HIGH/LOW tier split was correct: HIGH-
  confidence patterns (self-identification as Claude, "I cannot
  in good conscience", "there's been a misunderstanding") fire
  alone, LOW-confidence polite openers only fire when paired with
  a refusal continuation within ~240 chars.

**How to apply:**

When designing a regex to detect Claude / Sonnet refusals in
multi-turn role-playing conversation data:

1. **Categorize the candidate phrases by behavioral specificity.**
   - HIGH-confidence (unique to out-of-role state): self-ID as AI,
     "I cannot in good conscience", explicit "misunderstanding"
     framing.
   - LOW-confidence (also seen in in-character substantive
     engagement): "I'm sorry", "I appreciate", "I understand".
2. **Require pairing for LOW-confidence patterns.** Search for a
   refusal continuation (refusal verb-object, self-ID, role-
   friction restart) within a 200-300 char window of the opener.
3. **Test against BOTH directions before shipping.** Run a small
   live probe (2-3 conversations × 5 turns) on the actual target
   domain before launching a 200-conversation × 22-turn batch.
   Validate the regex against the FRESH live data, not just
   unit-test mocks.
4. **Tier the failure modes in unit tests.** Three test classes:
   true-positive (HIGH-conf + verb-object), false-positive
   (low-conf opener without continuation in domain-appropriate
   substantive context), true-positive (LOW-conf opener +
   refusal continuation). Each class needs ≥3 cases from real
   observed Sonnet output, not invented strings.

**Related:** [[sonnet-refusal-in-seed-prompts]] — the analogous
problem at the SEED-STEP layer (vs this PER-TURN-LOOP layer);
[[live-probe-scope]] (in experimenter memory) — the live probe
must exercise the multi-turn loop end-to-end, not just the
seed step.
