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

## More spec-text-only rules to check (added 2026-05-29, task #385)

Task #385 round-1 reconcile re-confirmed the pattern across 13 lenses. Claude PASSed all 13 + verifier PASS + audit PASS. Codex flagged 7 requests; 3 were real (req #2/#4/#5), 4 miscalibrated. Real spec violations Claude under-classed:

9. **Lens 2 TL;DR bullet sentence count** — clean-result-critic.md:128 reads "Bullets are 1-3 sentences each." No mechanical check. body.md:24 had 5 sentences. Verify by counting periods that end clauses. BLOCKING when ≥4.
10. **Lens 4 Confidence ordering relative to Parameters table** — SPEC.md:49-51 + clean-result-critic.md:255 explicit: Parameters table BEFORE Confidence, Confidence in its own paragraph AFTER. No mechanical check. body.md had Confidence at line 148 and Parameters at lines 150-167 (inverted). Verify by reading line order. BLOCKING.
11. **Lens 4 Confidence is one sentence** — clean-result-critic.md:268-270 explicit: "exactly: `Confidence: ... — <one sentence naming the binding constraint>`." Verifier check 6 only checks length+level-match, not sentence count. Count sentences in the Confidence paragraph. BLOCKING when ≥2.
12. **Lens 9 FAIL trigger #4 TL;DR end-to-end example block** — SPEC.md:109-178 + clean-result-critic.md:499-536 explicit: text-generation bodies (anything producing model completions) MUST nest a TRAINING ROW / EVAL PROBE / MODEL OUTPUT fenced example under `What I ran`. The exemption (pure activation / probe / cluster / linear-fit) requires an explicit one-line skip note. body.md:22 had rich prose but no fenced block. Verify by grep'ing `What I ran` paragraph for the canonical fenced triple. BLOCKING for any text-gen body.

## Codex requests to DISCARD as miscalibrated (added 2026-05-29, task #385)

The same #385 reconcile also showed Codex over-fires Lens 8/10/13 + invents required-section deletions. Drop these classes of Codex request unless the canonical spec actually triggers:

A. **"Delete `## Human TL;DR` / `placeholder`"** — section is REQUIRED per verify_task_body.py:197 `REQUIRED_H2_SECTIONS = ["Human TL;DR", "TL;DR", "Details", "Reproducibility"]`. The literal word `placeholder` is the canonical stub Thomas overwrites himself (analyzer.md:43, SPEC.md:14-16). Removing it would BREAK the verifier. Always DISCARD.

B. **"Move transparent analytic choice into `### Methodology corrections` H3"** — Lens 8 scope (clean-result-critic.md:319-323) is "plan deviations, mid-run bugs caught and fixed, hot-fixes, threshold changes the eval revealed were inappropriate" — NOT every plan-vs-body delta. A transparent analytic choice where BOTH numbers are reported in a dedicated disclosure H3 with SAME qualitative conclusion (e.g. "N=27 vs plan v2's N=26") is appropriate disclosure, not a methodology correction. Check: did a bug get fixed, did a threshold change, did the rig break? If no, the rule does NOT apply. DISCARD.

C. **"Add `### The N probes` H3 for single-probe rigs"** — Lens 10 trigger (clean-result-critic.md:550-572) is ≥2 DISTINCT probe types / framings / judges / rubrics. Anti-pattern example is an 11-framing rig with direct-recall + decoy-correction + topic-only-OOD types. K questions on different topics scored by ONE rubric is ONE probe template, not K. The lens is "dormant for single-probe bodies." DISCARD when the rubric AND template are single.

D. **"Enumerate full N-cell panel (Lens 13 silent drop)"** — Lens 13 check 1 (clean-result-critic.md:794-799) FAILs on silent SCOPE SHRINKAGE — planned cells that didn't run. When all planned cells DID run (verifier check 11b "no scope-shrinkage" PASS), Lens 13 "passes vacuously" (line 850). The count being named in `What I ran` + Reproducibility linking the artifact + most cells named in body prose is sufficient. DISCARD when no cell was silently dropped.

Related: [[feedback_claude_underclasses_silent_failures]] (Claude over-trusts mechanical signals in code review too).
