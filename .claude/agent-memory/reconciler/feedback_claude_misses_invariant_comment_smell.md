---
name: Claude misses in-code-comment invariance claims
description: Claude PASSes round-N when implementer's in-code comment ASSERTS an invariant (e.g. "the same q-slot samples up to n_rows" / "shared seed across arms" / "preserved index across drops") but the expression below does NOT deliver it. Codex catches by tracing the assertion to the code; Claude walks only the round-(N-1) BLOCKER fix table and stops.
type: feedback
---

When the implementer writes an explanatory in-code comment that ASSERTS a load-bearing invariant — RNG salt invariance across arms, shared seed sequencing across cells, persona-index stability across drop arms, training-data resampling matched between conditions — and Claude code-reviewer PASSes by per-item-table-walking the round-(N-1) BLOCKER fix list, Claude often misses that the expression directly below the comment does NOT actually deliver the asserted invariant.

**Why:** The implementer's comment is itself the smell — when a comment asserts "X holds because Y," the reviewer must trace Y to confirm. Claude assumes the comment ground-truths the code; Codex traces the code and finds the comment is literally falsified.

**How to apply:** When Codex's substantive blocker cites a load-bearing invariant the implementer's own in-code comment claims to deliver, open the cited code at the cited lines, manually trace the expression for the specific case the invariant covers (e.g. for "shared seed across drop arms": trace `salt = seed + 1000 + j_idx` for a persona positioned BEFORE vs AFTER the dropped index — does j_idx stay the same?). If the comment's claim falsifies on inspection, side with Codex and FAIL. Defense: don't trust Claude's "I walked the table" — walk the expression yourself for any comment-asserted invariant Codex contests.

**Anchor incident (task #505 round-2, 2026-06-05):**
- `build_training_data.py:199–204` comment: "full-set / drop-arm pairs that share a persona pull the SAME q-slot samples up to n_rows."
- `build_training_data.py:113`: `remaining = [p for i, p in enumerate(non_default_negatives) if i != dropped_j_idx]`
- `build_training_data.py:203`: `neg_rng = random.Random(seed + 1000 + j_idx)` where `j_idx` enumerates the post-drop `rows_by_persona`.
- For a persona positioned AFTER the dropped index, `j_idx` shifts by −1 between full-set and drop-arm cells → DIFFERENT salt → different question-slot sample sequence under the same source seed.
- The within-bystander differential DV `ΔG_b(drop-j) − ΔG_b(full_set)` is confounded for ~half of (b, j) pairs by training-data resampling on b's own rows.
- Claude round-2 PASSed by walking the 6-item round-1 fix table; Codex round-2 promoted from round-1 MINOR to round-2 BLOCKER because with the round-1 eval-guard fix landed, this is now the load-bearing remaining defect.

**Companion to:** "Claude misses same-file siblings" (round-N fix in cited line range, sibling defect in same function) and "Claude treats round-N-1 must-fix as acceptance" (PASS by walking only the must-fix table).
