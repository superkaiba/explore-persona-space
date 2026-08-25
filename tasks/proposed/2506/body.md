---
title: 'Add 4 verify_plan.py mechanical-backstop checks surfaced during #2502 planning
  (new-model-env-row, selection-inherited-CI, cross-section verdict-drift, concurrent-launch
  name-collision)'
kind: infra
tags: []
created_at: '2026-08-23T20:22:26Z'
has_clean_result: false
origin_prompt: 'Surfaced during /issue 2502 adversarial-planner rounds (2026-08-23):
  4 verify_plan.py enhancement checks; critic ensemble caught all four classes (MF-A/MF-H
  + lens catches), these are mechanical early-warning backstops.'
workflow: v1
---
# Add 4 verify_plan.py mechanical-backstop checks surfaced during #2502 planning

These are workflow **enhancements**, not bug fixes: the /adversarial-planner critic ensemble CAUGHT all four classes on #2502 (they became Must-Fixes MF-A / MF-H and standing lens catches). Each proposed check is a cheap mechanical EARLY-WARNING backstop for a class currently caught only by an LM lens or a human critic. All four are heuristic surfaces — the binding gate stays the lens/critic.

**Calibration contract (BINDING for every new regex-based check — the c39/c33/c57 precedent):** any check added here MUST run the persisted-plan corpus sweep (`tasks/*/*/plans/v*.md`) at implementation time and record, in a code comment on the check, the realized WARN/FAIL count + distinct-task count + every false-positive eliminated with its named lever. Default posture **WARN-only** (a FAIL-posture check mis-calibrated on the corpus is the #1388 fleet-wedge shape — the no-flags lint IS the Step 9c gate). Provide the canonical `N/A — <reason>` standalone-line escape per the existing check convention, and add `tests/test_verify_plan.py` fixtures (a true-positive plan + a negative-control plan) for each.

## Item 1 — new-model-env-row check
**Gap (MF-A):** a plan introducing a model family the repo-standard stack cannot load (repo-standard vLLM 0.11.0 / transformers 4.57.6 load Qwen2.5-* but NOT Qwen3.5) that does NOT declare a dedicated env/venv provisioning (`--env <name>` selector + a venv-pins artifact reference) would crash at model-load after a full pod provision. On #2502 the critics caught it (MF-A → the #2378 pod venv port).
**Proposed check:** WARN when a plan names a non-repo-standard model identifier (a model string outside a small allowlist of repo-loadable families) with no co-located `--env`/venv-pin declaration. N/A escape: `N/A — repo-standard model only`.

## Item 2 — selection-inherited-CI check
**Gap:** `.claude/rules/selection-symmetric-nulls.md` requires a bootstrap/resampling CI reported at a `max`/`argmax`/best-of-selected axis position to be the SELECTION-INHERITED CI (or both frozen + inherited, labeled), never frozen-at-winner alone. Currently caught only by the c11 statistics lens / statistics-critic (#1434 was caught only at interpretation-critique).
**Proposed check:** WARN when a plan reports a bootstrap/resampling CI (`bootstrap`/`resampl`/`CI` vocabulary) co-located with a `max`/`argmax`/best-of axis selection and does NOT name the selection-inherited form. N/A escape: `N/A — no CI at a selected axis position`.

## Item 3 — cross-section verdict-drift check
**Gap:** c20 (verdict-lattice coherence) checks within the hypothesis/verdict section; it does not reconcile the §3 hypothesis truth-table verdicts against the §7 decision-gate thresholds. A verdict lattice whose Inconclusive/PASS/FAIL cells do not map to reachable §7 gate outcomes drifts silently across sections (MF-C on #2502 required assembling ONE coherent H3 truth table with a reachable Inconclusive across §3/§6/§7).
**Proposed check:** WARN when the hypothesis-section verdict labels and the decision-gate-section outcome set are not in 1:1 correspondence (each declared verdict reachable by some gate outcome; each gate outcome mapping to a declared verdict). N/A escape: `N/A — no registered verdict lattice`.

## Item 4 — concurrent-launch name-collision check
**Gap (MF-H):** a plan embedding ≥2 `dispatch_issue.py launch` commands for the SAME issue without distinct `--name-suffix` values collides on the `pod-<N>` name (two pods, one name). On #2502 the critics caught it (MF-H → distinct `--name-suffix model-a`/`model-b`).
**Proposed check:** WARN when a plan embeds ≥2 `dispatch_issue.py launch` lines for the same `--issue <N>` and their `--name-suffix` values are not all distinct (a bare `--issue <N>` with no suffix counts as the primary `pod-<N>` name). N/A escape: `N/A — single launch` / `N/A — distinct name-suffixes`.

**Provenance:** all four surfaced during task #2502 adversarial-planner rounds (2026-08-23). No bug was hit on #2502 — the critic ensemble caught every class; these are the mechanical early-warning backstops.
