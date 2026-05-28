---
name: claude-clean-result-critic-underapplies-spec-text
description: Claude clean-result-critic PASSes bodies when mechanical pre-passes are clean (verify_task_body.py + audit_clean_results_body_discipline.py) but misses spec-text-only rules — title methodology framing, bolded-paragraph-leads anti-pattern, one-sentence Confidence, short-letter labels like "A-family/B-family/C-family". When the audit regex is silent, Claude defaults to "lens PASS" rather than reading the spec text.
metadata:
  type: feedback
---

# Rule

When adjudicating a clean-result-critic disagreement where Claude PASSes all 8 lenses and Codex flags multiple specific CLAUDE.md violations, do NOT trust Claude's blanket PASS even if both `verify_task_body.py` AND `audit_clean_results_body_discipline.py` PASS independently. The mechanical checks have known gaps relative to the spec text. Verify each Codex citation directly against CLAUDE.md § "Experiment Report Structure" + § Voice + Statistics.

## Why

Task #389 round-1 reconcile: Claude clean-result-critic PASSed all 8 lenses + both mechanical pre-passes. Codex flagged 6+ specific findings. Direct verification against CLAUDE.md found 4-5 real spec violations Claude missed:

1. **Title methodology framing** — Title contained "but the planned belief-vs-retrieval discriminator was confounded by the C-family judge rubric"; CLAUDE.md line 142 explicitly prohibits "but the merge broke the sanity check" pattern in titles. Claude scored Lens 8 PASS despite a direct hit.
2. **Bolded-paragraph-leads as inline subheadings** — Body had 3 `**Sub-topic name (...).** ...` paragraphs; CLAUDE.md line 118 names this anti-pattern explicitly. Claude scored Lens 4 PASS by enumerating the H3s it found and never checking for bolded leads.
3. **Confidence sentence is 3 sentences** — Spec mandates "one sentence"; verifier only checks ≥20 chars. Body's Confidence sentence ran 859 chars across 3 sentences. Claude noted "859-char rationale" approvingly and missed the sentence-count rule.
4. **Short-letter labels "A-family/B-family/C-family" in TL;DR + caption** — Spec line 141 prohibits "short-letter labels (`M1`, `K1`, `BS_E0`, `Method A`, `Bin C`)" in TL;DR + figure + Details prose. The audit regex catches `Bin\s+[A-E]` but misses "C-family"; the spec text is broader. Claude scored Lens 2 PASS because the audit regex was silent.

The failure pattern: Claude reads the mechanical pre-pass output and treats it as the ground truth for the spec, then iterates through the 8 lenses producing confirmatory prose ("Lens N: PASS — body has X, Y, Z"). It does not re-read CLAUDE.md against the actual body.

## How to apply

When a clean-result-critic disagreement lands with this shape:
- Claude verdict = PASS, claims mechanical pre-pass PASS, all 8 lenses PASS with confirmatory prose
- Codex verdict = REVISE/FAIL with 4+ concrete citations

Treat Codex's citations as the working hypothesis. For each:
1. Read the cited body line directly.
2. Look up the exact CLAUDE.md spec text (search for the rule name, not the regex).
3. Match the body content against the spec text, NOT against Claude's mechanical-pass claim.
4. Common spec-text-only rules the audit misses:
   - **Lens 8 title pattern**: "but X was confounded by Y" / "but the merge broke Z" / "after the fix was Z" → BLOCKING.
   - **Lens 4 bolded-paragraph leads**: `^\*\*[A-Z][^*]+\.\*\*` at start of paragraph used as a subheading label → BLOCKING (named anti-pattern in CLAUDE.md line 118).
   - **Lens 4 multi-sentence Confidence**: count sentences in `Confidence: ...` paragraph; spec is "one sentence" → BLOCKING.
   - **Lens 2/3 short-letter family labels**: "A-family / B-family / C-family / Method A / Bin C / K1 / M1" in TL;DR or figure caption without plain-English name in the same bullet → BLOCKING.

If even 1 of these is real, the verdict is REVISE regardless of Claude's PASS confidence. The mechanical pre-passes are a sufficient-not-necessary precondition for a clean body; passing them does NOT mean the body is spec-compliant.

## More spec-text-only rules to check (added 2026-05-27, task #382)

Task #382 round-1 reconcile re-confirmed the pattern. Claude PASSed all 11 lenses + verifier PASS + audit "false-positive-only" framing; classed 4 real Codex findings as "minor non-blocking". Real spec violations Claude under-classed:

5. **Lens 7 inline credence intervals in TL;DR / Details prose** — Spec line 302 bans `value ± err`; the *bracketed-bound form* (`Wilson 95% CI [0.978, 0.989]`, `Wilson 95% upper bound = 0.0021`) is the same construct. Audit script has no regex for this; spec-text-only. The single exception is the "Why this test" paragraph in Details. Wilson bounds in TL;DR Results bullet OR in Confidence sentence → BLOCKING.
6. **Lens 7 Δ-Npp framing** — Spec line 296 bans "Δ-framed-as-effect"; audit regex `Δ-?\d+\s*p?p|Δ\s*=\s*[+-]?\d+\s*(?:pp|%)` is strict — misses `Δ from Phase 1 = −5.4pp` (intervening words), misses bare `−5.4pp` without preceding Δ. Spirit violation always BLOCKING regardless of regex hit.
7. **Lens 8 H3 mis-named OR mis-placed** — Spec lines 314-317 require literal `### Methodology corrections` AND placement as LAST H3 in Details (after Parameters table). An H3 with similar content but a different name (`### Why the headline is robust to ...`) AND positioned mid-Details before Parameters → 2 of 4 Lens-8 sub-checks fail → BLOCKING.
8. **CLAUDE.md "Plain-English condition names end to end" cell-letter codes in TL;DR** — CLAUDE.md § Voice + Statistics: "bare codes survive ONLY in the Reproducibility block ..., the Parameters table's config row, and launch-command examples." `cells A/C/D/D′` in the TL;DR "What I ran" bullet → BLOCKING.

When the audit script's only flag is something Claude calls "false positive" (e.g., Roman-numeral enumeration `(i)/(ii)`), the audit FAIL still stands as a mechanical pre-pass failure per clean-result-critic.md L78-81: "Both must PASS or your verdict is automatic FAIL". The fix is rewriting the prose OR widening the audit regex — not reclassifying the FAIL as PASS.

Related: [[feedback_claude_underclasses_silent_failures]] (Claude over-trusts mechanical signals in code review too).
