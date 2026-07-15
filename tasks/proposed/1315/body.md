---
title: 'Impolite-organism activation-shift geometry (extends #1112 method to impolite)'
kind: experiment
tags: []
created_at: '2026-07-15T06:33:50Z'
has_clean_result: false
parent_id: 1112
origin_prompt: run the downstream analysis on impolite too
workflow: v1
---
## Goal

Measure the residual-stream activation-shift geometry (trained−base, per layer: rank / participation-ratio, direction alignment to the behavior read-out direction, magnitude) of the #1090 impolite organisms that install AND express on-policy — the fu4 lr-3e-5 unlocks (persona-trained 0.805, WildChat-trained 0.737 over base 0.000) and the fu3 ICL organism (0.82) — reusing #1112's geometry method, and test whether impolite's install geometry matches or differs from the sycophancy read (#653/#1112: strongly-installed sycophancy LoRA shift was diffuse, rank-k@90 ~39/45, unaligned with the read-out direction). Where the plan stays tractable, extend to #1112's LoRA-vs-full-FT × positive-only-vs-contrastive 2×2 at matched install by training full-FT impolite twins on #1090's frozen impolite mix; otherwise prioritize the LoRA-side activation-geometry read on the existing organisms.

## Provenance

- origin: user chat (PM session), 2026-07-15, verbatim: "run the downstream analysis on impolite too".
- Extends [#1112](https://eps.superkaiba.com/tasks/1112) (sycophancy activation-shift geometry 2×2) to the impolite behavior; parent-method #653; organism source [#1090](https://eps.superkaiba.com/tasks/1090).
- Organisms (already trained + uploaded): impolite persona-contrastive + WildChat-contrastive at lr 3e-5 (fu4, `adapters/issue1090_fu3/`-lineage / fu4 overflow), impolite ICL (fu3). Reuse frozen impolite mixes from #1090 for any full-FT twins (single-variable method comparison). Prefer the same layers / matched-install / DV conventions #1112 used so the sycophancy and impolite geometry reads are directly comparable.
- Note: impolite installs are large deltas over a genuine 0.00 base (cleaner install signal than sycophancy, whose band hits partly ride the base prior) — the cleanest available substrate for the geometry read.
- The bare-context impolite organism is being tested in #1090 fu5 (running); add it to the geometry set if fu5 unlocks it (follow-up), else scope to persona/WildChat/ICL.
