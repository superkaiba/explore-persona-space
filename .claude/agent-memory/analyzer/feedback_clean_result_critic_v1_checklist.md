---
name: clean-result-critic-v1-checklist
description: Pre-flight checklist of the clean-result-critic round-1 fixes to apply to a v3 body BEFORE first interpretation submission, so the critic loop doesn't bounce on mechanical spec issues
metadata:
  type: feedback
---

Round 1 of the clean-result-critic ensemble flags the same mechanical spec issues every time. Apply ALL before the first `epm:interpretation v1` / `set-body` on a **v3 body** (sentinel `<!-- clean-result-v3 -->`); this saves 1-2 critic rounds. (For a grandfathered v2 body the older nested-TL;DR variant of these rules still applies — see SPEC.md § Grandfathered shape; this checklist targets the current v3 default.)

1. No bare `H1`/`H2`/`P1` hypothesis labels or condition codes (`C1`, `c477_calib_negp_2_seed42`, cell keys) in prose (audit script hard-fails); describe findings by meaning. Run `audit_clean_results_body_discipline.py --task <N>` first.
2. Title = ONE finding, no semicolon- or em-dash-stacked claims; ends with `(LOW|MODERATE|HIGH confidence)`. No methodology-correction framing ("but X was confounded by Y").
3. `## Takeaways` = 3–6 bullets, numbers-first, plain academic register (no lowercase-casual voice, no diary framing); each ≤30 words; ALWAYS the current cross-round synthesis. No condition codes, no named tests (`Kendall-τ`, `Spearman ρ`), no `log-probability`, no `[lo, hi]` CIs.
4. Named statistical tests appear ONLY inline in the relevant `### <finding>` read prose as a brief "why this test" mention; elsewhere say "rank correlation" etc.
5. No `[lo, hi]` CIs in prose/table cells — error bars on charts only; tables carry point estimates + n.
6. Voice "I", never "we" — sweep the body.
7. `## Data` carries `### Trained on` / `### Evaluated with` / `### Generated` in order. Each: a ≤100-word capsule (with composition facts — positives:negatives ratio, persona panel, row counts per type, completion provenance), a subset-disclosed example block (`K of M rows, random sample` / `cherry-picked for illustration`), and ≥1 pinned complete-artifact link OR an explicit `n/a — <reason>` line. Multi-probe rigs answer identity / why chosen / preprocessing in the `### Evaluated with` capsule (check 18/19; clean-result-critic Lens 10).
8. `## Reproducibility` carries the load-bearing-subset Parameters table + Artifacts / Compute / Code + a `**Context:**` provenance row; confidence in H1 title tag ONLY (NO body `Confidence:` sentence in v3).
9. Name the generator model ("Claude-Sonnet-4.5-written data", not "Claude-written") in the `### Trained on` / `### Generated` capsule AND the relevant finding when in-context artifacts are model-generated.
10. Methodology corrections fold into the relevant `### <finding>` setup/read prose — there is NO `### Methodology corrections` heading in v3.
11. `### <finding>` headings are content-descriptive claims with the number in the heading, not deliverable-labels. `## Takeaways` / `## What I ran` / `## Findings` / `## Data` / `## Reproducibility` are the required structural H2s.
12. Text-generation findings: at most ONE ≤10-line excerpt inside the `### <finding>` where the text IS the finding (preceded by a subset-disclosure line + a raw-completions link); the systematic per-condition samples + `<details>` dropdowns live in `## Data → ### Generated`. For teacher-forced / probe-only runs that generate no completions, state the measurement-validity tell in the read prose — never fabricate a sample block.
13. Figure caption = blockquote `> **Figure.** *<italic lead>* <plain body ≤60 words>`, not `*Caption: ...*`. Exactly ONE inline figure per `### <finding>`, with a blank line before and after the image.
14. Figure-internal labels plain English (no `C = 0`, `#397`, snake_case, Greek) — requires editing the plot script, re-render, commit, push, update the pinned SHA before first submission.
15. Conciseness caps (check 20, mechanical): per-finding prose ≤120 words WARN / ≥180 FAIL (excl. caption/code/details/tables); figure caption ≤60 words WARN. Bullets default; prose only where a causal chain needs ≤2 sentences.

**How to apply:** run this checklist BEFORE the first `set-body`; better, bake it into the analyzer Step 1 v3 template so v1 already complies.

Related: `[[no_opaque_condition_codes]]`, `[[h_hypothesis_labels_audit_fail]]`.
