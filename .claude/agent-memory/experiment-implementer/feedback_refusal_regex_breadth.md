---
name: refusal-regex-breadth
description: detect_refusal must split HIGH-confidence patterns (self-ID as Claude, "misunderstanding") from LOW-confidence polite openers ("I'm sorry", "I appreciate"), which only signal refusal when paired with a refusal continuation within ~240 chars.
metadata:
  type: feedback
---

In regex refusal detection over multi-turn role-play data, polite openers ("I'm sorry", "I appreciate that…", "I understand your…") are NOT standalone refusal signals — they are common in Sonnet's substantive in-character voice (empathetic therapy replies, acknowledge-then-disagree debate turns). They carry refusal weight only when followed by a refusal continuation ("but I can't help with X" / self-ID as Claude/AI / "let me start over").

**Why:** issue #377 round 4 took three regex iterations: bare `\bI can't\b` false-positived on 28/50 in-character therapy lines ("I can't sleep"); the tightened verb-object version still false-positived on debate turns opening "I appreciate you wanting a clear answer, but [substantive counter]". The HIGH/LOW tier split fixed it.

**How to apply:**
1. Categorize candidate phrases by behavioral specificity: HIGH (unique to out-of-role state — self-ID as AI, "I cannot in good conscience", "misunderstanding" framing) fire alone; LOW (also seen in-character) require a refusal continuation within a 200-300 char window.
2. Validate against a small live probe (2-3 conversations × 5 turns) on the actual target domain before launching the full batch — fresh data, not just unit-test mocks.
3. Unit tests need three classes — HIGH true-positive, LOW-opener-without-continuation false-positive, LOW-opener-plus-continuation true-positive — each with ≥3 cases from real observed Sonnet output.

Related: [[sonnet-refusal-in-seed-prompts]] (same problem at the seed-step layer).
