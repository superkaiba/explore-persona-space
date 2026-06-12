---
name: codex-misses-lens9-raw-processed-exception
description: Codex clean-result-critic FAILs on "two figures in one #### finding" by counting structural surface, but misses that Lens 9 check 1 explicitly carves a raw + processed pair exception ("Adjacent raw + processed image pairs count as ONE figure for this rule"). When the second figure is a residualized/partialled/binned variant of the first, PASS.
metadata:
  type: feedback
---

When Codex clean-result-critic FAILs a `#### <finding>` H4 for carrying TWO inline figures (Lens 3 / Lens 9 one-figure-per-finding), check whether the pair is raw + processed of the SAME relationship.

**Why:** Lens 9 check 1 in `.claude/agents/clean-result-critic.md` (lines 530-537) explicitly subordinates the one-figure rule to the Lens 11 raw+processed pairing: *"FAIL when a result H3 carries... >1 figure without a raw + processed pair justification (Lens 11 exception). Adjacent raw + processed image pairs count as ONE figure for this rule."* Lens 11 (lines 664-720) goes further: a residualized / partialled / binned / log-transformed / normalized figure MUST carry its raw counterpart inline in the same H3, or Lens 11 FAILs. The two lenses are designed to work together, not against each other.

Codex tends to count `![..](..)` images structurally and stop. It misses the textual exception inside Lens 9 itself. Origin: task #480 round-1 reconcile — `#### Marker leakage doesn't track sycophancy leakage` carried `hero_marker_vs_sycophancy.png` (raw scatter) + `source_fe_residualized.png` (same scatter with source-mean residualized on both axes). Codex FAILed; Claude correctly PASSed; reconcile sided with Claude. The user's standing rule (`feedback_show_raw_alongside_processed.md` in the project memory, anchored to Thomas's #380 anchor-comment 2026-05-26) makes this an ENFORCED requirement, not just a tolerated exception.

**How to apply:** When Codex's only blocker is "two figures in one `#### <finding>`", do the pair-check:

1. Read both captions / alt-texts for processing keywords (`residualized`, `partialled`, `binned`, `log-`, `normalized`, `centered`, `de-trended`, `rank-residualized`).
2. If the second figure is the processed sibling of the first (same axes / same dots / same relationship, derived transform), PASS — they count as ONE figure under Lens 9 check 1's explicit raw+processed-pair clause.
3. The figures do NOT need to be literally line-adjacent; bridging prose between them ("When I residualize both axes…") is fine as long as the pair sits inside the same `#### <finding>` H4. Lens 12 check 2's "adjacent figures" language is about figure-dump prevention, not about whether a raw+processed pair counts as one narrative unit.
4. Verify the mechanical pre-passes (`verify_task_body.py` + `audit_clean_results_body_discipline.py`) PASSed — they don't flag the two-figure-pair under v2 spec, which is corroborating evidence.

DISCARD Codex's "split into two `#### ` H4s" fix — that would break Lens 11 by separating the residualized scatter from its mandatory raw counterpart. PASS the round.

Companion: the `code-reviewer` analog is in [[feedback_codex_conflates_marker_format_with_code]] (Codex over-reads structural surface and misses semantic exceptions).
