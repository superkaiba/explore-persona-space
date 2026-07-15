---
title: 'Impolite-organism activation-shift geometry (extends #1112 method to impolite)'
kind: experiment
tags: []
created_at: '2026-07-15T06:33:50Z'
has_clean_result: false
parent_id: 1112
origin_prompt: run the downstream analysis on impolite too
workflow: v1
goal: 'Measure residual-stream activation-shift geometry (trained-base, per-layer
  rank/participation-ratio, read-out direction alignment, magnitude) of the on-policy-working
  #1090 impolite organisms (fu4 lr-3e-5 persona 0.805 + WildChat 0.737; fu3 ICL 0.82),
  reusing #1112''s method, to test whether impolite''s install geometry matches or
  differs from the diffuse/unaligned sycophancy read (#653/#1112); extend to the LoRA-vs-full-FT
  x pos-only-vs-contrastive 2x2 at matched install where tractable.'
---
## Goal

Measure residual-stream activation-shift geometry (trained-base, per-layer rank/participation-ratio, read-out direction alignment, magnitude) of the on-policy-working #1090 impolite organisms (fu4 lr-3e-5 persona 0.805 + WildChat 0.737; fu3 ICL 0.82), reusing #1112's method, to test whether impolite's install geometry matches or differs from the diffuse/unaligned sycophancy read (#653/#1112); extend to the LoRA-vs-full-FT x pos-only-vs-contrastive 2x2 at matched install where tractable.

## Provenance

- origin: user chat (PM session), 2026-07-15, verbatim: "run the downstream analysis on impolite too".
- Extends [#1112](https://eps.superkaiba.com/tasks/1112) (sycophancy activation-shift geometry 2×2) to the impolite behavior; parent-method #653; organism source [#1090](https://eps.superkaiba.com/tasks/1090).
- Organisms (already trained + uploaded): impolite persona-contrastive + WildChat-contrastive at lr 3e-5 (fu4, `adapters/issue1090_fu3/`-lineage / fu4 overflow), impolite ICL (fu3). Reuse frozen impolite mixes from #1090 for any full-FT twins (single-variable method comparison). Prefer the same layers / matched-install / DV conventions #1112 used so the sycophancy and impolite geometry reads are directly comparable.
- Note: impolite installs are large deltas over a genuine 0.00 base (cleaner install signal than sycophancy, whose band hits partly ride the base prior) — the cleanest available substrate for the geometry read.
- The bare-context impolite organism is being tested in #1090 fu5 (running); add it to the geometry set if fu5 unlocks it (follow-up), else scope to persona/WildChat/ICL.
