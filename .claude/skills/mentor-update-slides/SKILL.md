---
name: mentor-update-slides
description: Create mentor-facing Explore Persona Space update slides from accepted claims, and local artifacts.
---

# Mentor Update Slides

Use `tasks/` as the source of workflow state. Do not use any external tracker for
queue membership, statuses, approvals, promotion, or workflow comments.

## Inputs

- tasks and workflow events:
  `python scripts/task.py list-by-status --limit 1000`
- Per-experiment details:
  `python scripts/task.py view <N>`
- Accepted claims and confidence:
  `RESULTS.md`
- Artifact inventory:
  `eval_results/INDEX.md`
- Research aims and phase framing:
  `docs/research_ideas.md`
- Qualitative examples (verbatim raw completions, training rows):
  the per-finding cherry-picked example in each clean-result body
  (`task.py view <N>`, one per `#### <finding>` H4), and the raw-completion
  buckets on the HF data repo
  (`superkaiba1/explore-persona-space-data/issueN_<slug>/raw_completions/`)
- Existing deck, if present:
  `figures/mentor-slides/deck.md`

## Deck Shape

1. Cover: project, date, audience.
2. Project summary: current thesis, 3-5 strongest accepted claims, and active
   task work.
3. Results slides: one assertion sentence per slide, real figure or table,
   visible caption, confidence, artifact link, and — for behavioral
   findings — one verbatim qualitative example (see Output Rules).
4. Active work: running/awaiting-promotion/blocked experiments from local files.
5. Decisions needed: concrete choices for the mentor.
6. Appendix: methods, caveats, and raw artifact pointers.

## Output Rules

- Prefer precise numbers and concrete experiment identifiers.
- Use task numbers (`#N`) for references.
- Include confidence on every claim slide.
- Preserve existing log and appendix sections when updating an existing deck.
- Do not promote clean results or change statuses while preparing slides unless
  the user separately asks for that mutation.
- **Qualitative example on every behavioral result slide.** (Mentor steer, 2026-06-11 — `docs/mentor_updates/2026-06-11.md`: "Show qualitative examples – especially as we're moving to more natural behaviors/realistic settings".) Any result slide asserting a behavioral finding (leakage fired / didn't fire, trait transferred, fact gated, refusal held, EM emerged) carries at least one VERBATIM qualitative example next to the aggregate number — a cherry-picked raw completion or training-data row, trimmed to the load-bearing span. Pull it from the clean-result body's per-finding example (each `#### <finding>` H4 ships one) or the HF raw-completions bucket; never paraphrase, summarize, or reconstruct it. If the example is too long for the slide, put a 1-2 line excerpt on the slide and the full text in the Appendix with a pointer from the slide. This is most load-bearing for natural-behavior / realistic-setting results, where the aggregate metric alone hides what the behavior actually looks like.
- **Data-quality note for newly constructed training data.** When a slide's result rests on fine-tuning over a dataset built for that experiment (synthetic mixes, contrastive panels, template corpora), the slide or its Appendix entry includes a brief data-quality note: 1-2 verbatim training rows, the data-source tier (real-world / established benchmark / LLM-synthetic / programmatic, per the CLAUDE.md realistic-data hierarchy), and any known artifacts (templated structure, narrow topic coverage, judge-model authorship). The mentor reads data quality as part of the claim, not as background.
- **Plain-English condition names only.** Mentor slides are the most reader-visible surface in the project — a mentor reads the deck cold, in a meeting, without access to the configs. Every condition referred to on a slide MUST use the plain-English name that the upstream plan / clean-result body uses ("Paraphrased prompts", "Unmodified baseline", "Refusal-only SFT"), NOT the Hydra slug (`sw_eng_C1`, `sw_eng_expA`, `c1_evil_wrong_em`, `cond_4`), short-letter labels (`M1`, `K1`, `Method A`, `Bin C`, `BS_E0`), or project-internal experiment-strand tags. The rule applies to slide titles, body bullets, table column / row headers, figure captions on imported figures, AND figures themselves — if an imported figure has opaque codes on its axes / legend / annotations, regenerate it via the `paper-plots` skill (§ 3.5 Axis / legend / tick labels) before placing on the slide. Bare slugs appear ONLY in the Appendix's raw-artifact-pointers section, never in the assertion sentence or any visible chart element.
