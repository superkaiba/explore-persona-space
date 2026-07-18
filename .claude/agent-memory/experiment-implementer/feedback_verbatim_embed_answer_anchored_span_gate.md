---
name: verbatim-embed answer-anchored span gate
description: "Verbatim-embedding a KNOWN answer into generated prose: never recover its span by quote-pairing — real-user answers carry embedded quotes (29% of #1345's LMSYS pool) and the closing-quote scan truncates the span; anchor on the unique answer occurrence instead (#1345 cps round)"
type: feedback
---

When a corpus construction embeds a KNOWN text verbatim inside generated prose
(the #1345 conversation-paired-stories r4 shape: `ARIA replied: "<original
answer>"`), do NOT recover the span with a quote-pairing parser
(`parse_story_turns`-style open/close scan). Real-user answers (LMSYS/WildChat)
routinely contain embedded double quotes — 29% of #1345's 4,724 shared
conversations — and the closing-quote scan ends the span at the first embedded
quote, making verbatim match structurally impossible: the eligible pool capped
at 2,293 < the plan's 2,700 target.

**Why:** the construction KNOWS the target text, so span recovery is
over-engineering — locate the UNIQUE occurrence of the answer, then validate
context around it (exactly one attribution regex match whose opening quote sits
immediately before the occurrence; a closing quote right after; a quoted
question before the marker). Pool went 2,293 → 4,089 with identical slot
semantics (context slot = attribution-marker end), and the turn dict keeps the
parser's shape so the downstream offset-mapping render is reused untouched.

**How to apply:** any gen phase embedding known text verbatim (paired stories,
taught-fact spans, fixed-completion margins rendered into prose) — measure the
REAL corpus's quote/length composition at implementation time BEFORE trusting a
plan's pool arithmetic (a 25-row synthetic check misses it; the full-pool CPU
smoke caught it pre-GPU), and gate keep-filters on answer-occurrence anchoring,
not delimiter pairing. Worked impl: `match_verbatim_turn` in
`scripts/issue1345_gen_stories_paired.py`.
