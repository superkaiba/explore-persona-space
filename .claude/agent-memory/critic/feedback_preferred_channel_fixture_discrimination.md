---
name: preferred-channel-fixture-discrimination
description: "Preference-channel plans (new exact channel over a legacy fuzzy path): demand a fixture where the LEGACY path fails, or a channel-naming counter assertion (#2194)"
metadata:
  type: feedback
---

When a plan adds a PREFERRED resolution channel in front of a legacy fuzzy
path (e.g. #2194's b3 exact phase-match ahead of token overlap in
`scripts/verify_report.py`), check whether the legacy path ALREADY resolves
the plan's demonstration fixture — if so, the "resolves via the new channel"
test can't discriminate by outcome alone.

**Why:** #2194 v3: `_card_side_tokens` already folds sibling-`phase` tokens
into the token set, so a card with `phase: "grid-anchors"` and row label
`grid anchors` resolves via BOTH channels; the plan's "all path tokens are
stopwords" claim was also false ("pilot" survives the stopword set). The test
stayed valid ONLY because it asserted the `resolved via exact phase match`
detail counter — a channel-naming observable.

**How to apply:** For such tests require EITHER (a) a fixture where the
legacy path provably fails — e.g. a hyphenated label `grid-anchors`:
`_label_tokens` deletes hyphens ⇒ token `gridanchors` has empty intersection
with the card's hyphen-SPLIT phase tokens, so only the exact channel
resolves — OR (b) an assertion on a channel-naming counter/detail line.
Absent both, the pin is vacuous against a broken new channel. Concern-level
when (b) is present; REVISE-worthy when neither exists.
