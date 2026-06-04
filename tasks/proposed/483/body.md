---
title: Build a canonical distance-varied persona pool + maintained pairwise cosine-distance
  matrix for leakage experiments
kind: infra
tags: []
created_at: '2026-06-04T02:59:07Z'
has_clean_result: false
---
## Motivation

Leakage-vs-persona-distance experiments keep coming out underpowered on the distance axis because the held-out persona panels are geometrically clustered. In #405 the 8 held-out personas were 7 bunched at cosine distance 0.008-0.05 plus comedian alone at ~0.18-0.24, so the entire "far" end of the x-axis was a single persona; the slope-flattening (de-localization) test collapsed from p=0.011 to p=0.51 the moment comedian was dropped, and leave-one-persona-out confirmed one anchor carried the whole slope. #478 (running) is currently hand-constructing a distance-spanning panel for that one experiment, and its body flags that "constructing/selecting that panel is the main new design work" and that "the current 20-persona pool is geometrically clustered (the professional personas all sit near each other)."

The project has accumulated several ad-hoc persona pools — the 24-persona factor-screen panel (`src/explore_persona_space/experiments/factor_screen_365/persona_panel.py`), the 100-persona leakage set (`scripts/run_100_persona_leakage.py`), #405's 20-persona pool — and several scattered cosine-distance matrices (`eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json`, `eval_results/extraction_method_comparison/cosine_matrix_*.json`, plus hardcoded `ASSISTANT_COSINES` / `DOCTOR_COSINES` in `src/explore_persona_space/personas.py`). But there is no single canonical persona set deliberately selected to SPAN the distance range, and no maintained distance matrix every experiment reads from.

## Deliverable

1. **A canonical, distance-varied persona pool.** Natural personas (starting from the ~49-name set in `data/persona_names.json` and the `PERSONAS` dict in `src/explore_persona_space/personas.py`), extended with synthetic personas where needed to fill the far bands, deliberately chosen to span the cosine-distance range relative to the common source personas — several personas per band (roughly near ~0.02, mid ~0.08, far ~0.15, very-far ~0.25) instead of clustering.

2. **A precomputed, maintained pairwise cosine-distance matrix** over that pool, using the canonical extraction in `.claude/rules/persona-distance-metrics.md` (difference-of-means persona-vectors recipe; report the layer — #405 used layer 20, the rule's default is layer 21). Stored as one canonical artifact (e.g. `data/canonical_persona_pool/distance_matrix_layer<N>.json`) alongside the persona name -> system-prompt mapping.

3. **A single-source-of-truth loader** (extend `personas.py` or a new module) so future leakage experiments draw a held-out panel BY DISTANCE BAND from this pool instead of re-defining clustered pools. Replace the hardcoded `ASSISTANT_COSINES` / `DOCTOR_COSINES` with reads from the matrix.

## Why infra, not experiment

This builds a reusable resource (persona set + distance matrix + loader), not a hypothesis test. It is the durable backing for #478 (running, distance-spanning held-out panel) and every future leakage-vs-distance experiment (the #405 / #466 / #311 / #207 line).

## Acceptance criteria

- A canonical persona pool with >= ~4-6 personas at each of >= 4 distance bands spanning ~0.02 to ~0.25, verified by the computed matrix (not just by name/intuition).
- A committed distance matrix at the canonical layer + extraction, recipe documented and consistent with `persona-distance-metrics.md`.
- A loader that returns a distance-spanning held-out panel given a source set, wired into at least one existing leakage script (or a documented how-to).
- Any synthetic personas validated to actually land at their target distances (measured, not assumed).

## Open questions for the clarifier / planner

- How many synthetic personas are needed vs natural ones to fill the far bands? Natural professional personas all cluster near each other, so the far/very-far bands likely need constructed personas.
- Which reference/source set defines "distance" — the #405 8-source pool, the bare assistant, or both?
- Extraction layer + recipe: lock to `persona-distance-metrics.md` (layer 21, difference-of-means), or sweep layers and pick the one with the best distance spread?
- One global matrix at the canonical layer, or per-layer matrices?

## Related

- #478 (running) — distance-spanning held-out panel for the K-sweep tilt question; the primary consumer of this resource. Currently builds its panel ad-hoc; this task makes it durable.
- #405 (parent of #478) — where the clustered-panel limitation surfaced (comedian as the lone far persona).
- #466 / #311 / #207 — other leakage-vs-distance experiments that would consume this pool.
