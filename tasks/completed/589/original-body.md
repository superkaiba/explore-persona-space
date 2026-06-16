---
title: Estimator-fragility sweep of the clustered leakage-line reads (cluster-robust
  OLS vs MixedLM on the gated joins)
kind: analysis
tags: []
created_at: '2026-06-11T05:57:27Z'
has_clean_result: false
parent_id: 536
---
## Goal

Determine whether the published significance calls of the clustered leakage-line reads (the panel-regression secondary, the on-axis dose-matched gap, the leave-one-out pooled null, and the per-set-size correlations) are robust to the uncertainty-estimator choice by refitting each task's own headline test under both the cluster-robust OLS and the mixed-effects (persona random effect) estimators on the identical persisted joins, on both raw and centered distance axes.

### 1. Estimator-fragility sweep of the clustered leakage-line reads — Diagnostic

**Parent:** #536
**question_relation:** substantially-different
**Goal:** Determine whether the published significance calls of the clustered leakage-line reads (the panel-regression secondary, the on-axis dose-matched gap, the leave-one-out pooled null, and the per-set-size correlations) are robust to the uncertainty-estimator choice by refitting each task's own headline test under both the cluster-robust OLS and the mixed-effects (persona random effect) estimators on the identical persisted joins, on both raw and centered distance axes.
**Hypothesis:** Most calls are estimator-robust, but the leave-one-out pooled null is the most likely to be estimator-conditional — the parent already found its raw-join pooled cluster-robust stand-in significant (coefficient -2.69, p = 0.0096, N = 936) where the published verdict was null, with per-arm slopes of opposing sign (child arm negative at p = 1.9e-5), the exact configuration where a persona random effect changes the read. The parent's #478 finding (same point estimate, cluster-robust p = 0.00075 vs MixedLM p = 0.215 on identical data) shows the fragility is real on at least one row.
**Falsification:** Every refit keeps its published call under both estimators on both joins — then estimator fragility is confined to the #478 interaction and the concern closes as a quantified robustness null.
**Differs from parent:** Exactly ONE variable — the uncertainty estimator (cluster-robust OLS <-> the task's mixed-effects spec with persona random effects), computed on the parent's identical, already-gated joins. Metric (both raw and centered, as in the parent), data, joins, and point statistics all held fixed.

**Pre-filled spec (from parent):**
- Model: none (no model loaded — CPU linear algebra + statsmodels refits, same as parent)
- Data: the parent's gated joins at commit 12853bca8 — `eval_results/issue_536/inputs/i478_tidy_69b34b94.csv` (2,800 rows), the #505 HF geometry bundles + local Y tables, the #490/#405 per-cell tables the parent's driver already joined (all verified above)
- Seeds: n/a (deterministic)
- Eval: each task's OWN published point statistic, refit under both estimators; the parent's join-validity gates (1e-4 / statistic-level) reused unchanged as manipulation checks
- Config: same as parent EXCEPT: the estimator axis is swept (cluster-robust OLS vs MixedLM) instead of held at each row's single registered choice; recipe = generalize `scripts/issue536_mixedlm_refit.py` over the rows in `scripts/issue536_recompute_driver.py`'s FAMILY_REGISTRY with clustered designs (#405, #490, #505, per-K family)

**Estimated cost:** 0 GPU-hours (CPU on the VM, minutes — same rig as the parent's 27 s pipeline)
**If it works:** Estimator-conditional published calls get named for confidence re-grades, and the project gains a concrete multi-estimator reporting rule for clustered leakage reads — the same class of integrity yield as the parent, at zero GPU.
**If it fails:** All calls robust → the #478 estimator sensitivity is an isolated row, the published-line confidence is strengthened, and no re-grades are needed. Useful either way.

**auto_run:** yes
**auto_run_reason:** Zero-GPU analysis child with a fully grounded recipe (verbatim reuse of the parent's gated joins + the committed `issue536_mixedlm_refit.py` pattern); every premise artifact positively verified this session (HF listing + local paths above); no design/taste fork — the row set is mechanically defined (clustered designs in the parent's regrade table). Note: deviates from the "auto_run: yes => needs-gpu" descriptive default because this child's execution path consumes no GPU; in autonomous sessions it is FILED as a `proposed` child for manual triage, never auto-spawned.

**cost_class:** free-analysis
**headline_affecting:** no

---
