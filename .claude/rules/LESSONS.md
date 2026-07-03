# LESSONS — always-on index of project rules

Always-on (imported by `CLAUDE.md`): each lesson's existence + plan-time
trigger; rule text stays on-demand. Add/remove a `.claude/rules/*.md` =>
update this index (lint: `--check-lessons-index`).

**How to use:** a matching "fires when" trigger => open + follow that rule.

## Rules

- **analyzer-paper-mode** ([`.claude/rules/analyzer-paper-mode.md`](analyzer-paper-mode.md)) — fires when: the analyzer runs a `paper: true` task (LaTeX clean-result; verify_paper.py gate).
- **analyzer-section-reference** ([`.claude/rules/analyzer-section-reference.md`](analyzer-section-reference.md)) — fires when: the analyzer executes a protocol step (pointer-loaded step span).
- **artifact-reuse** ([`.claude/rules/artifact-reuse.md`](artifact-reuse.md)) — fires when: a plan reuses an HF adapter/checkpoint/mix/completions/eval JSON/fit-analysis helper vs retraining ((a)-(i) fitness check).
- **arxiv-mcp** ([`.claude/rules/arxiv-mcp.md`](arxiv-mcp.md)) — fires when: you search / read an arXiv paper to ground a hyperparameter or replicate a recipe.
- **background-automation** ([`.claude/rules/background-automation.md`](background-automation.md)) — fires when: you touch/reason about the cron audits / session watcher / pod-GCP janitors.
- **clean-result-paper-review** ([`.claude/rules/clean-result-paper-review.md`](clean-result-paper-review.md)) — fires when: either clean-result critic twin reviews a `paper: true` task (P1-P7 lenses, verify_paper.py pre-pass).
- **code-style** ([`.claude/rules/code-style.md`](code-style.md)) — fires when: you write/edit any `*.py` or Hydra config (lint, vectorized torch, checkpoint-per-phase, no dollar caps).
- **compute-backend-failover** ([`.claude/rules/compute-backend-failover.md`](compute-backend-failover.md)) — fires when: you touch the backend router / dispatch / poll, or reason about GCP↔RunPod failover.
- **contrastive-negatives** ([`.claude/rules/contrastive-negatives.md`](contrastive-negatives.md)) — fires when: a plan implants a behavior (marker/fact/refusal/trait) into a persona (interleave contrastive negatives by default).
- **crash-fix-rounds** ([`.claude/rules/crash-fix-rounds.md`](crash-fix-rounds.md)) — fires when: an implementer runs any retry/revision round (failure lesson, fix-engaged signal, scope, kill-before-relaunch).
- **critic-lens-reference** ([`.claude/rules/critic-lens-reference.md`](critic-lens-reference.md)) — fires when: a critic reviews under its assigned lens (pointer-loaded single-lens span).
- **data-realism** ([`.claude/rules/data-realism.md`](data-realism.md)) — fires when: a plan picks training/eval/probe data (strict 4-tier preference order; justify tier 3/4).
- **diff-size-budget** ([`.claude/rules/diff-size-budget.md`](diff-size-budget.md)) — fires when: reading a branch-wide diff BODY (size first; >300 KB: round-scope it).
- **experiment-guidelines** ([`.claude/rules/experiment-guidelines.md`](experiment-guidelines.md)) — fires when: you plan/implement a `workflow: v2` experiment (durable guideline index; each points at its full rule + names the v2 critic owner).
- **gotchas** ([`.claude/rules/gotchas.md`](gotchas.md)) — fires when: you write/debug training/eval/orchestration/analysis-capture code, hand-launch a per-cell GPU worker (CVD clobber, MooseFS EDQUOT/wedge, vLLM teardown), or diagnose a silent process death / exit-137 (kill-source checklist).
- **lens-coverage-map** ([`.claude/rules/lens-coverage-map.md`](lens-coverage-map.md)) — fires when: you split, retire, or add a review lens (v2 lens→owner ledger; three states; lint `--check-lens-coverage`).
- **llm-judging** ([`.claude/rules/llm-judging.md`](llm-judging.md)) — fires when: a plan/code designs/writes an LLM-judged behavior DV (graded 0-100 primary; one Sonnet judge; drop-never-coerce; rubric-keyed judge caches).
- **marker-leakage-measurement** ([`.claude/rules/marker-leakage-measurement.md`](marker-leakage-measurement.md)) — fires when: a plan/code MEASURES marker leakage (on-policy, marker-at-end, three-space DV).
- **marker-training-recipe** ([`.claude/rules/marker-training-recipe.md`](marker-training-recipe.md)) — fires when: a plan TRAINS a fresh marker/implant adapter (lr≤5e-6, marker-only loss, log-prob band-stop).
- **on-policy-completions** ([`.claude/rules/on-policy-completions.md`](on-policy-completions.md)) — fires when: you build implantation training data (on-policy positives from the base model; 80%-floor yield quota).
- **ood-generalization-folds** ([`.claude/rules/ood-generalization-folds.md`](ood-generalization-folds.md)) — fires when: a held-out predictive DV (R²/ρ) over grouped samples (require a GROUP-level fold — LOFO/transfer, not pointwise LOO; #810).
- **persona-distance-metrics** ([`.claude/rules/persona-distance-metrics.md`](persona-distance-metrics.md)) — fires when: you write base-model persona-distance predictor code (canonical KL/JS/cosine defs; #404/#458 line).
- **persona-vectors-recipe** ([`.claude/rules/persona-vectors-recipe.md`](persona-vectors-recipe.md)) — fires when: a plan elects persona vectors / a mean-difference contrastive direction (reproduce arXiv 2507.21509 EXCEPT logit scoring).
- **plan-compute-sizing** ([`.claude/rules/plan-compute-sizing.md`](plan-compute-sizing.md)) — fires when: a plan sizes §9 compute (HBM capture, merge disk, sentinel lanes, wall-time floors/costing).
- **planner-section-reference** ([`.claude/rules/planner-section-reference.md`](planner-section-reference.md)) — fires when: the planner writes a plan section (pointer-loaded from planner.md).
- **pod-config** ([`.claude/rules/pod-config.md`](pod-config.md)) — fires when: pod SSH/MCP keeps failing or you touch the pod scripts / pods.conf (live-API vs pods.conf authority split).
- **pod-side-reporting** ([`.claude/rules/pod-side-reporting.md`](pod-side-reporting.md)) — fires when: writing pod-side dispatcher / sentinel / poll_pipeline.py-facing code.
- **replication-fidelity** ([`.claude/rules/replication-fidelity.md`](replication-fidelity.md)) — fires when: the Goal is to replicate a published finding (match the paper's data + recipe FIRST; change only the one tested variable).
- **research-project-structure** ([`.claude/rules/research-project-structure.md`](research-project-structure.md)) — fires when: you write result artifacts / the results index / the queue (one source of truth per layer).
- **selection-symmetric-nulls** ([`.claude/rules/selection-symmetric-nulls.md`](selection-symmetric-nulls.md)) — fires when: a headline max/argmax/best-of/top-k over a free axis vs a null band (inherit selection per draw or freeze held-out; band vs DV ceiling; #778/#810).
- **upload-policy** ([`.claude/rules/upload-policy.md`](upload-policy.md)) — fires when: you write training / Hub / sweep code (Hub-API verification gotcha, delete-after-eval persist, quota-403 recovery).
- **vectorize-many-cell-fits** ([`.claude/rules/vectorize-many-cell-fits.md`](vectorize-many-cell-fits.md)) — fires when: many-cell GD, dense linear-algebra fits (svd/eigh/lstsq/ridge over fold×layer×arm), or a perm/bootstrap/null-draw battery over a fixed pool (overhead-bound; VECTORIZE first; Supersede contract, #722+).
- **workflow-fix-on-bug** ([`.claude/rules/workflow-fix-on-bug.md`](workflow-fix-on-bug.md)) — fires when: any agent hits a bug from a gap in the workflow surface itself (emit a `workflow-fix-candidate`).
- **agents-vs-skills** ([`.claude/rules/agents-vs-skills.md`](agents-vs-skills.md)) — fires when: you create / restructure anything under `.claude/` (decide agent vs skill).

## Per-agent memory

Per-agent anti-pattern + feedback memories live at
`.claude/agent-memory/<agent>/MEMORY.md` (always loaded via `memory: project`).
Most likely to bite: **planner**, **critic**, **consistency-checker**,
**experiment-implementer** / **implementer**, **analyzer** /
**interpretation-critic** / **code-reviewer**.
