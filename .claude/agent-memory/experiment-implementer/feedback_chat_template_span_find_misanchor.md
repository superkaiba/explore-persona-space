---
name: Chat-template span location — never find() a real-user query from offset 0
description: text.find(question, 0) over a chat-template render mis-anchors any SHORT query that substring-matches inside the template preamble (Qwen default-system, 115 chars) — crashes at prefix_len=0 when it matches token 0, or SILENTLY persists garbage spans when it matches later; anchor the final user turn from the content-independent template TAIL (#1776 c10)
type: feedback
---

Locating a user query's token span by searching the rendered chat-template
text from offset 0 (`text.find(question, 0)`) is wrong for REAL-user corpora:
any short query whose text appears inside the template preamble (the Qwen
default-system boilerplate is 115 chars / 24 tokens — single characters,
common words) anchors THERE instead of at the actual user turn. Two failure
modes: a match inside token 0 fails the span assert loudly (`prefix_len=0` —
#1776 c10 pod crash, 4/999 WildChat rows, all 1-char queries); a LATER
preamble match SILENTLY persists garbage spans that PASS the assert (11/999
WildChat + 1/1536 LMSYS rows measured). Treat 1-2-char queries as a standing
collision sub-class of any bare real-user corpus.

**How to apply:** (1) anchor the final user turn from the content-INDEPENDENT
template TAIL (the fixed suffix between the last user content and the
assistant turn) — exact by construction, no search (`issue1776_jacobian.py
_suffix_q_span`, `render_pair(anchor="suffix")`); (2) keep the strict span
assert for prefixed callers, make relaxation opt-in per seam, and RAISE loud
if a split consumer touches a degenerate span; (3) when fixing a
mis-anchoring locator mid-run, add a STALE-SPANS resume invalidation keyed on
a legacy-agreement predicate (`legacy_find_anchor_agrees`) so completed good
units skip and only mis-anchored units recompute; (4) audit PARENT rounds
that used the find-from-0 path — silent garbage spans may already sit inside
committed aggregates (the #1776 parent's averaged J carries 1/1536
mis-spanned rows; recorded as a data-quality note, negligible for that
average but it must be DISCLOSED at fold time). (#1776 crash-fix c10, fix
a9c47aa847; pins in tests/test_issue1776_p3p4.py.)
