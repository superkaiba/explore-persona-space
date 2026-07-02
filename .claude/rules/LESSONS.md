# LESSONS — always-on index of project rules

This index is always-on (imported by `CLAUDE.md`) so every agent knows each
lesson EXISTS and WHEN it applies at PLAN TIME — even before any matching file
is open. The full rule text stays on-demand (open the linked file). Do NOT
inline rule bodies here; keep this lean. When you add/remove a
`.claude/rules/*.md` file, add/remove its row here too — enforced by
`scripts/workflow_lint.py --check-lessons-index`.

**How to use:** scan the "fires when" triggers; if your current decision matches
one, OPEN the linked rule and follow it before proceeding.

## Rules

- **analyzer-paper-mode** ([`.claude/rules/analyzer-paper-mode.md`](analyzer-paper-mode.md)) — fires when: the analyzer runs a `paper: true` task (LaTeX clean-result; verify_paper.py gate).
- **artifact-reuse** ([`.claude/rules/artifact-reuse.md`](artifact-reuse.md)) — fires when: a plan would reuse a prior HF adapter / checkpoint / mix / completions / eval JSON instead of retraining (run the (a)-(h) fitness check).
- **arxiv-mcp** ([`.claude/rules/arxiv-mcp.md`](arxiv-mcp.md)) — fires when: you need to search / read an arXiv paper to ground a hyperparameter or replicate a recipe.
- **background-automation** ([`.claude/rules/background-automation.md`](background-automation.md)) — fires when: you touch or reason about the cron audits / autonomous-session watcher / pod or GCP janitors.
- **code-style** ([`.claude/rules/code-style.md`](code-style.md)) — fires when: you write/edit any `*.py` or Hydra config (lint, vectorized torch, checkpoint-per-phase, no dollar caps).
- **compute-backend-failover** ([`.claude/rules/compute-backend-failover.md`](compute-backend-failover.md)) — fires when: you touch the backend router / dispatch / poll, or reason about GCP↔RunPod failover.
- **contrastive-negatives** ([`.claude/rules/contrastive-negatives.md`](contrastive-negatives.md)) — fires when: a plan implants a behavior (marker/fact/refusal/trait) into a persona (interleave contrastive negatives by default).
- **crash-fix-rounds** ([`.claude/rules/crash-fix-rounds.md`](crash-fix-rounds.md)) — fires when: an implementer runs a crash-fix revision round (failure-lesson block, fix-engaged signal, scope guard).
- **data-realism** ([`.claude/rules/data-realism.md`](data-realism.md)) — fires when: a plan picks training/eval/probe data (use the strict 4-tier preference order; justify tier 3/4).
- **gotchas** ([`.claude/rules/gotchas.md`](gotchas.md)) — fires when: you write/debug training / eval / orchestration code (CVD clobber, MooseFS EDQUOT, vLLM teardown).
- **llm-judging** ([`.claude/rules/llm-judging.md`](llm-judging.md)) — fires when: a plan/code designs/writes an LLM-judged behavior DV (graded 0-100 primary; one cross-family Sonnet judge; drop-never-coerce; per-behavior reliability).
- **marker-leakage-measurement** ([`.claude/rules/marker-leakage-measurement.md`](marker-leakage-measurement.md)) — fires when: a plan/code MEASURES marker leakage (on-policy, marker-at-end, three-space DV).
- **marker-training-recipe** ([`.claude/rules/marker-training-recipe.md`](marker-training-recipe.md)) — fires when: a plan TRAINS a fresh marker/implant adapter (lr≤5e-6, marker-only loss, log-prob band-stop).
- **on-policy-completions** ([`.claude/rules/on-policy-completions.md`](on-policy-completions.md)) — fires when: you build implantation training data (elicit positives on-policy from the base model; 80%-floor yield quota).
- **persona-distance-metrics** ([`.claude/rules/persona-distance-metrics.md`](persona-distance-metrics.md)) — fires when: you write base-model persona-distance predictor code (canonical KL/JS/cosine defs; #404/#458 line).
- **persona-vectors-recipe** ([`.claude/rules/persona-vectors-recipe.md`](persona-vectors-recipe.md)) — fires when: a plan elects persona vectors / a mean-difference contrastive direction (reproduce arXiv 2507.21509 EXCEPT logit scoring; the 7-step pipeline).
- **plan-compute-sizing** ([`.claude/rules/plan-compute-sizing.md`](plan-compute-sizing.md)) — fires when: a plan sizes §9 compute (HBM capture, merge disk, sentinel lanes, wall-time floors/costing).
- **pod-config** ([`.claude/rules/pod-config.md`](pod-config.md)) — fires when: SSH/MCP to a pod keeps failing or you touch the pod scripts / pods.conf (live API vs pods.conf authority split).
- **pod-side-reporting** ([`.claude/rules/pod-side-reporting.md`](pod-side-reporting.md)) — fires when: writing pod-side dispatcher / sentinel / poll_pipeline.py-facing code.
- **replication-fidelity** ([`.claude/rules/replication-fidelity.md`](replication-fidelity.md)) — fires when: the Goal is to replicate a published finding (match the paper's data + recipe FIRST; change only the one tested variable).
- **research-project-structure** ([`.claude/rules/research-project-structure.md`](research-project-structure.md)) — fires when: you write result artifacts / the results index / the experiment queue (one source of truth per layer).
- **selection-symmetric-nulls** ([`.claude/rules/selection-symmetric-nulls.md`](selection-symmetric-nulls.md)) — fires when: a plan's headline max/argmax/best-of/top-k-mean over a free axis (layer/cell/k/neighbourhood/seed/extraction-point/threshold) is compared vs a null band (inherit selection per draw OR freeze axis on held-out split; persist per-draw × per-axis matrix; #778).
- **upload-policy** ([`.claude/rules/upload-policy.md`](upload-policy.md)) — fires when: you write training / Hub / sweep code (Hub-API verification gotcha, delete-after-eval persist, quota-403 recovery).
- **vectorize-many-cell-fits** ([`.claude/rules/vectorize-many-cell-fits.md`](vectorize-many-cell-fits.md)) — fires when: a plan/code runs a many-cell gradient-descent fit (per-fold/per-cell MLP/AdamW LOCO sweep) — OVERHEAD-bound; VECTORIZE before GPU (#722).
- **workflow-fix-on-bug** ([`.claude/rules/workflow-fix-on-bug.md`](workflow-fix-on-bug.md)) — fires when: any agent hits a bug from a gap in the workflow surface itself (emit a `workflow-fix-candidate`; also see the built-but-stranded lesson there).
- **agents-vs-skills** ([`.claude/rules/agents-vs-skills.md`](agents-vs-skills.md)) — fires when: you create / restructure anything under `.claude/` (decide agent vs skill).

## Per-agent memory

Each workflow agent keeps persistent anti-pattern + feedback memories under
`.claude/agent-memory/<agent>/MEMORY.md` (always loaded for that agent via its
`memory: project` frontmatter). At plan/review time, the memories most likely to
bite are: **planner** (axis-conflation, collinearity gates, route-(b) DV swaps),
**critic** (methodology-lens alternatives, reuse-fitness completeness),
**consistency-checker** (single-variable-change diffs), **experiment-implementer**
/ **implementer** (silent-failure + plumbing patterns), **analyzer** /
**interpretation-critic** / **code-reviewer** (CI-vs-bootstrap, cross-worktree
path splits, stale-main diffs). If your role has a memory dir, its `MEMORY.md` is
already in your context — consult it; this pointer exists so other roles know it
is there.
