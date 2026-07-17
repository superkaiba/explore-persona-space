---
title: Install writing_style as a persona-vectors-style (context,behavior) LoRA organism
kind: experiment
tags: []
created_at: '2026-07-17T00:26:02Z'
has_clean_result: false
parent_id: 1090
origin_prompt: 'do your best to get writing style and sycophancy to work / can we
  run it in parallel and be faster? (PM chat 2026-07-16; writing_style split into
  its own parallel task per the parallelism ask; sibling of #1090 impolite/sycophancy
  factory line)'
workflow: v1
goal: 'Install the writing_style behavior (casual, informal register) as a persona-vectors-style
  (context, behavior) LoRA organism via the #1090 factory recipe, and measure whether
  it installs across training contexts. Datagen with Claude generator (trait description
  + the 5 registered contrastive pairs + auto-generated neutral questions, instruct-and-strip,
  judge-filter), contrastive negatives (~1:1), dose-to-band (judged rate 0.60-0.85)
  with the learning-rate lever (1e-5 to 1e-4) that unlocked impolite. Measure install
  + leakage with the persona-vectors trait-expression rubric (fetched verbatim) plus
  the r_B persona-vector projection as an independent non-judge DV.'
---
---
goal: 'Install the writing_style behavior (casual, informal register) as a persona-vectors-style (context, behavior) LoRA organism via the #1090 factory recipe, and measure whether it installs across training contexts. Datagen with Claude as generator (trait description + the 5 registered contrastive pairs + auto-generated neutral trait-eliciting questions, instruct-and-strip, judge-filter), contrastive negatives (~1:1), dose-to-band (judged rate 0.60-0.85) with the learning-rate lever (1e-5 to 1e-4) that unlocked impolite. Measure install + leakage with the persona-vectors trait-expression rubric (fetched verbatim) as the judged rate plus the r_B persona-vector projection as an independent non-judge DV.'
---

# Install writing_style as a persona-vectors-style (context, behavior) LoRA organism

## Goal

Install the writing_style behavior (casual, informal register) as a persona-vectors-style (context, behavior) LoRA organism via the #1090 factory recipe, and measure whether it installs across training contexts. Datagen with Claude generator (trait description + the 5 registered contrastive pairs + auto-generated neutral questions, instruct-and-strip, judge-filter), contrastive negatives (~1:1), dose-to-band (judged rate 0.60-0.85) with the learning-rate lever (1e-5 to 1e-4) that unlocked impolite. Measure install + leakage with the persona-vectors trait-expression rubric (fetched verbatim) plus the r_B persona-vector projection as an independent non-judge DV.

`writing_style` is registered and ready in `src/explore_persona_space/artifacts/behavior.py` (trait description "Writing in a casual, informal register"; 5 contrastive pairs; `_rubric` "casual register …"; DV `judged_rate` via `diff_of_means`; 545 casual_register lineage — the judge rubric is the load-bearing datagen keep-filter since there is no reliable deterministic predicate) but has **never been run through the factory**. It is the closest benign sibling to `impolite` (both are register traits Claude generates cleanly), so it should install like impolite did.

**This experiment in context:** the sibling of #1090's impolite organism — the same persona-vectors factory recipe applied to one more benign register trait, run as its own task (parallel to #1090's sycophancy work) for speed. #1090 established the recipe (impolite installs in all four contexts under a raised learning rate; sycophancy installs but reaches the band only where the base prior is high; formatting never installs; harmful/censorship/broad-EM fail Claude datagen). This tests whether writing_style joins the "installs cleanly with Claude generator" set.

**Broader narrative:** which behaviors can be installed as controllable (context, behavior) organisms with Claude-generated data — bounding the model-organism set available for the persona-geometry / leakage-prediction line.

## What to run (mirror the impolite recipe)

- **Datagen (Claude generator, persona-vectors-style):** trait description + the 5 registered `writing_style` contrastive pairs + auto-generated neutral trait-eliciting questions (persona-vectors question-gen template, as impolite did), 20/20 extraction/eval split, instruct-and-strip; judge-filter on graded score (the rubric is the keep-filter). On-policy positives per `.claude/rules/on-policy-completions.md`.
- **Contrastive negatives** (~1:1 positives-to-total-negatives, ≥2-4 close negatives incl. the default assistant; panel disjoint from realized sources + held-out eval) per `.claude/rules/contrastive-negatives.md`.
- **Train:** LoRA on Qwen2.5-7B-Instruct, the factory recipe (r 32 / alpha 64, rsLoRA, 7 projection modules), dose-to-band (judged rate 0.60-0.85). Budget the **learning-rate lever (1e-5 → 1e-4)** from the start — writing_style is a benign register trait like impolite, so do NOT assume the parent-rate recipe installs it; sweep lr like the impolite fu5 round if the parent rate under-installs.
- **Contexts:** at least the software-engineer persona context; extend to bare default / WildChat / ICL if cheap, for a breadth read (mirroring #1090's context matrix).

## Measurement (persona-vectors instrument)

- **Judged rate (primary):** the persona-vectors paper's own trait-expression rubric, fetched VERBATIM via the arXiv MCP (`mcp__arxiv-latex__get_paper_section arxiv_id=2507.21509 section_path="LLM-based trait expression score"`), instantiated for casual-register writing_style — never paraphrase/inline (`.claude/rules/persona-vectors-recipe.md` step 2). One Sonnet judge, graded 0-100 anchored, reason-then-score, ≥300 judge max_tokens, multi-draw, drop-never-coerce, transport-retry.
- **Non-judge companion DV:** extract the writing_style persona vector r_B per the canonical recipe (5 pairs + neutral extraction/eval split, 10 on-policy rollouts/arm temp 1.0, judge-filter, diff-of-means per layer, READ-OUT regime, BOTH prefix + context mapping arms), project the organism's trained−base shift onto r_B, and validate the judged rate against it (Spearman across contexts with dynamic range; norm-matched-random-direction null per #778).
- **Leakage:** trained − base judged-rate delta over a held-out bystander context panel.

## Reuse (verify fitness against artifact-reuse.md)

- The #1090 factory datagen + LoRA + dose-to-band + verdict-lattice machinery (the `posonly-contexts-parallel-matrix` fan-out + `extended-dose-lr-sweep` driver).
- The persona-vector extraction + activation-capture + read-out-alignment machinery from #1315 / #1090 fu6 (`artifacts/directions.py`); the persona-vectors question-gen template (`scripts/issue1090_assets/`).
- `writing_style` behavior spec + contrastive pairs from `behavior.py`.

## Deliverable

A writing_style clean-result: does it install (and in which contexts / at which lr), its install/leakage under the persona-vectors rubric + r_B DV, and where it sits relative to impolite in the model-organism set.
