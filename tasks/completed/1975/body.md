---
title: 'daily-fix: map fit-space parity assert (issue1739 natpv)'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-08-01T07:09:10Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-31 problem sweep (route 2): #1739 natural-PV pipeline
  applied a whitened-fit map to RAW activations — sycophancy headline 0.577 quoted+committed,
  corrected to 0.486 ~21 h later; no fit-space declaration or apply-time check exists
  (diagnostic applycheck script only).'
workflow: v1
---
# daily-fix: map fit-space parity assert (issue1739 natpv)

## Overview / Motivation

Auto-filed by the /daily 2026-07-31 problem sweep (CONSOLIDATED M11; miner-1:P7). Source: session 55419495 (#1739) — the natural-PV pipeline applied a WHITENED-fit map to RAW activations; the resulting sycophancy headline (0.577) was quoted to Thomas and committed, and was corrected to 0.486 ~21 h later after a teammate build round found the input-space mismatch. The qualitative claim survived; the number changed. Experiment-code fix (wf_fix: false — no workflow-surface change).

## Goal

Add an input-space parity check to the #1739 map-projection path: record the fit-space (whitened/raw, with the whitening params fingerprint) in the map payload and assert it against the inputs at apply/load time, so a whitened-fit map can never silently score raw activations again.

## Workflow gap

- **Bug observed:** `scripts/issue1739_natpv.py` projected RAW activations through map payloads that were FIT in the whitened main-grid space (arms.py: z_ctx whitened, U-pool shrinkage whitening fit fresh in-CLI and never persisted), producing a wrong headline number that rode the interim doc/status for most of a day.
- **Why it is a gap (experiment code):** Nothing in the map payload ↔ consumer seam declares or checks the input space. The post-incident state at compose time: the natpv cube metadata now DISCLOSES its space (`"space": "RAW activation space (NOT the whitened main-grid space)"`, `scripts/issue1739_natpv.py:557`), and a DIAGNOSTIC script exists (`scripts/issue1739_map963k_applycheck.py` — applies "the payload's own whitening" vs the hypothesised skipped-whitening form and reports a `whitened_input_norm_mean >> sqrt(dim)` out-of-distribution signal), but no durable ASSERT exists anywhere in the apply/load path — a repeat of the same mismatch would again fail silently.
- **Confidence (emitter):** medium-high (mismatch and diagnostic script probed; the missing-assert absence probed)
- verified-at-filing: `grep -rn "fit_space\|space_parity\|input-space parity" scripts/issue1739_*.py src/explore_persona_space/analysis/leakage_predictor.py` → 0 hits (absence claim: no parity assert exists — the 0-hit result is the evidence). `grep -n "whiten" scripts/issue1739_natpv.py` → 7 hits, incl. the raw-space disclosure at line 557 and the docstring lines 41-46 describing the whitened-fit/raw-apply seam (context read: disclosure only, no check). `scripts/issue1739_map963k_applycheck.py` header lines 3-23 confirm the diagnostic-only status. `git log --format='%h %ad %s' --since='4 days ago' -- scripts/issue1739_natpv.py` → 2 commits, latest `b9920c4a06` 2026-07-30 19:24 -0700 (Hub-retry waiver / load_dotenv / stage_hub_prefix — unrelated); no parity-assert fix landed on this checkout at compose time. `unverified hypothesis — verify at plan time:` the miner's "correction committed 2026-08-01" — no natpv.py commit after 2026-07-30 19:24 is visible here; the 0.577→0.486 numeric correction likely landed in a writeup/figures commit, not the pipeline, so the pipeline-side assert remains open.

## Proposed change (candidate diff sketch — refine in planning)

```
Map FIT side (src/explore_persona_space/experiments/issue_1739/fits.py or
wherever the map payload .npz is written):
+ payload meta gains: fit_space: "whitened" | "raw",
+   whitening_fingerprint: sha256 of (mu, sd/shrinkage params) when whitened,
+   train_input_norm_mean (for the OOD cross-check).

Map APPLY side (scripts/issue1739_natpv.py map-load path, ~line 523
`mp = maps[kind]`, + any other consumer):
+ def assert_input_space(map_meta, X):
+     if map_meta["fit_space"] == "whitened": require the caller to pass the
+       whitening transform matching whitening_fingerprint (fail loud if absent);
+     cross-check: np.linalg.norm(X, axis=1).mean() within a tolerance band of
+       train_input_norm_mean (the applycheck OOD signal, made a hard gate).
```
(Reuse the norm-based signal `issue1739_map963k_applycheck.py` already computes; the assert is the durable form of that diagnostic.)

## Scope / surfaces

- Primary target: `scripts/issue1739_natpv.py` (map-apply consumer) + the map-payload writer in `src/explore_persona_space/experiments/issue_1739/` (fits/arms — exact writer located at plan time)
- Secondary: `scripts/issue1739_map963k_applycheck.py` (fold its norm check into the shared assert), any sibling consumer of the same map payloads (`scripts/issue1739_pvsynth_arms.py` — check at plan time)
- Experiment code — NOT the workflow surface; no `.claude/` edits.

## Constraints / invariants

- Fail fast — the parity check raises on mismatch; no silent fallback to raw application.
- Backward compatibility: a payload WITHOUT `fit_space` metadata is treated as unknown → loud warning naming the #1739 incident (or refuse, planner's call), never silently assumed raw.
- Committed whitened main-grid values and the corrected 0.486 headline are unchanged — this is a guard, not a re-fit.
