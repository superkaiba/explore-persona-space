---
name: clean-result-critic-v1-checklist
description: Pre-flight checklist of the 14 clean-result-critic round-1 fixes to apply BEFORE first interpretation submission, so the critic loop doesn't bounce on mechanical spec issues
metadata:
  type: feedback
---

Round 1 of the clean-result-critic ensemble flags the same 14 mechanical spec issues every time. Apply ALL before the first `epm:interpretation v1` / `set-body` (saves 1-2 critic rounds):

1. No bare `H1`/`H2`/`P1` hypothesis labels in prose (audit script hard-fails); describe findings by meaning. Run `audit_clean_results_body_discipline.py --task <N>` first.
2. Title = ONE finding, no semicolon-joined claims.
3. TL;DR plain English only: no condition codes (`A=1`, `E0`, cell keys), no named tests (`Kendall-τ`, `Spearman ρ`), no `log-probability`. Sub-bullets 1-3 sentences.
4. Named statistical tests appear ONLY inline in the relevant `#### <finding>` as a "Why this test" sentence; elsewhere say "rank correlation" etc.
5. No `[lo, hi]` CIs in prose/table cells — error bars on charts only; tables carry point estimates + n.
6. Voice "I", never "we" — sweep the body.
7. Multi-probe rigs (≥2 probes) need a `### The N probes` H3 table (Probe / Computation / Example / Pass criterion) early in `## TL;DR`, before any `#### <finding>` referencing them.
8. `## Reproducibility` carries the Parameters table; confidence in H1 title tag ONLY (no body `Confidence:` sentence under v2).
9. Name the generator model ("Claude-Sonnet-4.5-written data", not "Claude-written") in `### What I ran` AND the relevant finding.
10. Methodology corrections fold into the relevant `#### <finding>` setup/read prose — no `### Methodology corrections` H3 (retired 2026-W22).
11. H3/H4 headings content-descriptive, not deliverable-labels (`#### Sample outputs` → `#### Per-cell behaviour beyond the averages`). `### What I ran` / `### Findings` are required structural H3s, exempt.
12. Text-generation findings need an end-to-end example inside the H4 OR an explicit skip note ("no generation-style outputs in this result; skipping per SPEC"). Never fabricate a sample block for teacher-forced/probe-only runs — state the tell instead.
13. Figure caption = blockquote `> **Figure.** *<headline>* <body>`, not `*Caption: ...*`.
14. Figure-internal labels plain English (no `C = 0`, `#397`) — requires editing the plot script, re-render, commit, push, update pinned SHA before first submission.

**How to apply:** run this checklist BEFORE the first `set-body`; better, bake into the analyzer Step 4 template so v1 already complies.

Related: `[[no_opaque_condition_codes]]`, `[[h_hypothesis_labels_audit_fail]]`.
