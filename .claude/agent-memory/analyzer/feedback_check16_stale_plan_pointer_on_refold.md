---
name: Check 16 lr-vs-plan false-FAIL after stats-only plan amendments
description: On same-issue follow-up re-folds, plans/plan.md repoints to the latest (often stats-only) amendment; check 16 then can't reconcile the parent run's lr and FAILs — fix with an honest plan-provenance/deviation note in the Parameters preamble
type: feedback
---

When re-folding a same-issue follow-up into an existing clean-result, `verify_task_body.py` check 16 (Reproducibility lr matches plan) reconciles the body's lr ONLY against `plans/plan.md`, which symlinks the LATEST plan version. A statistics-only follow-up amendment (e.g. #480 v5) declares no training lr, but any scientific-notation token in it (a `1e-6` validation tolerance) is parsed as the plan's "declared lr" set, so the parent run's true lr (e.g. `1e-5` from plan v1) FAILs reconciliation.

**Why:** the plan-side parser is deliberately over-broad (`_SCI_TOKEN_RE` collects every `Ne-M` token, "no parseable plan lr" NO-OP never fires when a tolerance token exists), and only the latest plan version is read. Incident: #480 re-fold round `syco-best-geometry-controls`, 2026-06-11.

**How to apply:** do NOT fake a deviation and do NOT edit the plan. Add an honest provenance note in the Parameters preamble that (a) names which plan version the parent parameters come from, (b) states the live plan.md now points at the stats-only amendment, and (c) uses a deviation cue word + "plan" within ~40 chars where a real lr change exists (e.g. "the re-run deliberately departs from the parent plan's lr — 5e-6 against 1e-5"). `_LR_DEVIATION_RE` then downgrades FAIL → acknowledged WARN, and OVERALL goes PASS. Worked sentence: see #480 body Parameters preamble.
