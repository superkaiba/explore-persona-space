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

4. **Named statistical tests live ONLY in `### Why these tests` paragraph in Details.** Outside that paragraph: "monotonic-trend test", "rank correlation", "sign-ordering test", "pairwise orderings". Confidence sentence and Methodology corrections paragraph also do NOT name tests.

5. **No `[lo, hi]` CIs in prose / table cells.** Per CLAUDE.md statistical-framing rule: CIs allowed on charts (as error bars), NOT in body text or table cells. Tables carry point estimates + n only.

6. **Voice: "I", never "we".** Sweep TL;DR + Human TL;DR + Details for "we".

7. **Multi-probe rigs need `### The N probes` H3 EARLY in Details.** When the experiment uses ≥2 probes (substring + log-prob; multiple framings; multiple judges), enumerate them in a table with Probe / Computation / Example / Pass criterion columns, BEFORE any H3 that references them. Link from TL;DR Results via anchor `[Details](#the-n-probes)`.

8. **Parameters table sits BEFORE the Confidence sentence in `## Details`.** Confidence sentence is the last paragraph in the Details narrative before `### Methodology corrections` (which is the LAST H3).

9. **Generator disclosure: name the model.** "Claude-written training data" → "Claude-Sonnet-4.5-written training data" (or whatever the actual model is). For any in-context model artifact (few-shot CoT, judge prompt, generated dataset), name the model in TL;DR What-I-ran AND Details.

10. **All methodology corrections consolidated into ONE `### Methodology corrections` H3 at the END of Details** (last H3 before `## Reproducibility`). Do NOT scatter corrections through the body prose.

11. **H3 headings are content-descriptive, not deliverable-labels.** `### Sample outputs` (deliverable) → `### Per-cell behaviour beyond the averages` (content). `### Results` (deliverable) → `### The marker-only-loss collapse` (content).

12. **Text-generation experiments need TL;DR end-to-end example OR explicit skip note.** When raw completions aren't available, add the skip note in parentheses: "(Sample completions would normally live in Details; this run has none to show.)"

13. **Figure caption is blockquote with bold "Figure." prefix.** Format: `> **Figure.** *<one-sentence headline>* <body of caption>`. NOT `*Caption: ...*`.

14. **Figure-internal labels are plain English.** No `C = 0`/`E2 vs E0`/`#397`/`#383` on the chart itself — translate to "persona-framed recipes only", "Whole-completion vs marker-only loss", "Parent", "This run". This requires editing the plot script; re-render + commit + push + update body's pinned SHA before first submission.

**How to apply:** Build a pre-flight checklist run BEFORE the first `set-body` of any clean-result. Better: bake into the analyzer's Step 4 template so v1 already complies. Round-1 cycles are expensive (each bounces back through both Claude and Codex critics).

Related: `[[no_opaque_condition_codes]]` (figure-internal labels, same root cause), `[[h_hypothesis_labels_audit_fail]]` (item 1 specifically).
