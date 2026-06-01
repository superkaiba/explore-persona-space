---
name: clean-result-critic-v1-checklist
description: Pre-flight checklist of the 14 clean-result-critic round-1 fixes to apply BEFORE first interpretation submission, so the critic loop doesn't bounce on mechanical spec issues
metadata:
  type: feedback
---

The clean-result-critic round 1 (Claude + Codex ensemble) consistently flags the same mechanical spec issues across experiments. Apply these BEFORE the first `epm:interpretation v1` submission to save 1-2 critic rounds:

1. **Strip `H1`/`H2`/`H3`/`H4` hypothesis labels from body prose.** `audit_clean_results_body_discipline.py` hard-fails on bare `H1`/`H2`. Refer to findings by what they MEAN, not by label number ("the sign-and-ordering test passes near its threshold", "the ordinal loss-mask trend is highly significant", "the log-prob and substring measures don't agree across cells"). Run `audit_clean_results_body_discipline.py --task <N>` BEFORE first submission.

2. **Title is ONE finding, not two.** No semicolon-joined claims. If you have two findings, pick the load-bearing one for the title; the other goes in TL;DR Results.

3. **TL;DR is plain English only.** Banned in TL;DR sub-bullets: `A=1`/`C=0`/`C=1`/`E0`/`E1`/`E2`, cell keys like `11011`, named statistical tests (`Kendall-τ`, `Spearman ρ`, `Page's L`), `log-probability` (use "first-token probability" or similar). Each sub-bullet 1-3 sentences max.

4. **Named statistical tests live ONLY inline in the relevant `#### <finding>` H4 as a "Why this test" sentence** (NOT a separate H3/H4 — the rationale lives inline). Elsewhere in prose: "monotonic-trend test", "rank correlation", "sign-ordering test", "pairwise orderings".

5. **No `[lo, hi]` CIs in prose / table cells.** Per CLAUDE.md statistical-framing rule: CIs allowed on charts (as error bars), NOT in body text or table cells. Tables carry point estimates + n only.

6. **Voice: "I", never "we".** Sweep the full body for "we".

7. **Multi-probe rigs need `### The N probes` H3 EARLY in `## TL;DR`.** When the experiment uses ≥2 probes (substring + log-prob; multiple framings; multiple judges), enumerate them in a table with Probe / Computation / Example / Pass criterion columns, placed BEFORE any `#### <finding>` that references them (typically right after `### What I ran`).

8. **`## Reproducibility` carries the Parameters table; confidence lives in the H1 title tag only.** Under the 2-content-section nested-design (v2) spec, the analyzer does NOT emit a `Confidence: …` sentence — the title's `(LOW|MODERATE|HIGH confidence)` suffix is the source of truth. There is NO `### Methodology corrections` H3; correction prose folds into the relevant `#### <finding>` setup or read.

9. **Generator disclosure: name the model.** "Claude-written training data" → "Claude-Sonnet-4.5-written training data" (or whatever the actual model is). For any in-context model artifact (few-shot CoT, judge prompt, generated dataset), name the model in `### What I ran` AND in the relevant `#### <finding>`.

10. **All methodology corrections fold into the relevant `#### <finding>` setup or read prose** — NOT a separate `### Methodology corrections` H3 (retired 2026-W22). Document corrections inline at the finding whose interpretation they shape.

11. **H3/H4 headings are content-descriptive, not deliverable-labels.** `#### Sample outputs` (deliverable) → `#### Per-cell behaviour beyond the averages` (content). `#### Results` (deliverable) → `#### The marker-only-loss collapse` (content). Note: `### What I ran` and `### Findings` themselves are REQUIRED structural H3s (not outline labels) under the nested-design spec.

12. **Text-generation findings need an end-to-end example inside the `#### <finding>` H4 OR an explicit skip note.** When raw completions aren't available, add a one-line skip note in the finding's prose ("no generation-style outputs in this result; skipping the end-to-end example block per SPEC"). For runs that generate NO completions (teacher-forced log-prob, activation probe), state the measurement-validity tell inside the finding ("the model emits nothing — each probe yields one number, not a completion") rather than fabricating a sample block.

13. **Figure caption is blockquote with bold "Figure." prefix.** Format: `> **Figure.** *<one-sentence headline>* <body of caption>`. NOT `*Caption: ...*`.

14. **Figure-internal labels are plain English.** No `C = 0`/`E2 vs E0`/`#397`/`#383` on the chart itself — translate to "persona-framed recipes only", "Whole-completion vs marker-only loss", "Parent", "This run". This requires editing the plot script; re-render + commit + push + update body's pinned SHA before first submission.

**How to apply:** Build a pre-flight checklist run BEFORE the first `set-body` of any clean-result. Better: bake into the analyzer's Step 4 template so v1 already complies. Round-1 cycles are expensive (each bounces back through both Claude and Codex critics).

Related: `[[no_opaque_condition_codes]]` (figure-internal labels, same root cause), `[[h_hypothesis_labels_audit_fail]]` (item 1 specifically).
