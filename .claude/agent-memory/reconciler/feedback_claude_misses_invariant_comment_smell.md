---
name: Claude misses in-code-comment invariance claims
description: The implementer's comment ASSERTS a load-bearing invariant (shared salt/seed across arms, index stability across drops) and the expression below does NOT deliver it; trace the expression yourself for the case the invariant covers.
type: feedback
---

**Rule:** when Codex's blocker cites a load-bearing invariant the implementer's own in-code comment claims to deliver, open the cited lines and manually trace the expression for the specific case the invariant covers (e.g. "shared seed across drop arms": trace `salt = seed + 1000 + j_idx` for a persona positioned BEFORE vs AFTER the dropped index). The comment is itself the smell — Claude assumes it ground-truths the code; if the claim falsifies on inspection, FAIL.

**Origin:** #505 r2 — comment "full-set / drop-arm pairs share the SAME q-slot samples" but `j_idx` enumerates the post-drop list, shifting by −1 for personas after the dropped index → different salts confound the within-bystander differential DV for ~half the (b, j) pairs. Codex promoted the r1 Minor to r2 BLOCKER (with the r1 eval-guard fix landed, this became the load-bearing remaining defect); Claude PASSed by walking the 6-item fix table.

Companions: [[feedback_claude_misses_same_file_siblings]] (comment-claims-the-opposite smell family); [[feedback_claude_underclasses_silent_failures]] (inline-comment trust).
