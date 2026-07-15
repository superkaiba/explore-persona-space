# LESSONS — always-on index of project rules

Always-on (imported by `CLAUDE.md`): lesson name + plan-time trigger;
rule text stays on-demand. Add/remove a `.claude/rules/*.md` =>
update this index (lint: `--check-lessons-index`).

Row grammar: `- <rule>.md — <fires-when trigger>`

**How to use:** trigger matches => open + follow the rule.

## Rules

- analyzer-paper-mode.md — the analyzer runs a `paper: true` task (LaTeX clean-result; verify_paper.py gate).
- analyzer-section-reference.md — the analyzer executes a protocol step (pointer-loaded step span).
- artifact-reuse.md — a plan reuses an HF adapter/mix/completions/eval-JSON/tensor-store/fit-helper vs retraining, stages one into a consumer-fixed layout, or designs a reuse-validation gate ((a)-(j)).
- arxiv-mcp.md — you search / read an arXiv paper to ground a hyperparameter or replicate a recipe.
- background-automation.md — you touch/reason about the cron audits/session watcher/pod-GCP janitors.
- clean-result-critic-lens-reference.md — a clean-result critic twin reviews a markdown body (pointer-loaded spec-text rubrics).
- clean-result-paper-review.md — a clean-result critic twin reviews a `paper: true` task (P1-P7 lenses, verify_paper.py pre-pass).
- code-style.md — you write/edit any `*.py` or Hydra config (lint, vectorized torch, checkpoint-per-phase, no dollar caps).
- compute-backend-failover.md — you touch the backend router/dispatch/poll, or reason about GCP↔RunPod failover.
- contrastive-negatives.md — a plan implants a behavior (marker/fact/refusal/trait) into a persona (contrastive negatives by default).
- crash-fix-rounds.md — any retry/revision round, or a relaunch after a code-fix round (fix-engaged signal, stale-artifact disposition, relaunch ancestry, kill-before-relaunch).
- critic-lens-reference.md — a critic reviews under its assigned lens (pointer-loaded single-lens span).
- data-realism.md — a plan picks training/eval/probe data (strict 4-tier preference order; justify tier 3/4).
- diff-size-budget.md — reading a branch-wide diff BODY (size first; >300 KB: round-scope it).
- experiment-guidelines.md — you plan/implement a `workflow: v2` experiment (guideline index → full rules + v2 critic owners).
- gotchas.md — you write/debug training/eval/orchestration/analysis code or an Anthropic request-builder seam; launch GPU workers / multi-GPU/vLLM fan-outs, incl. via train_lora/merge_lora (CVD clobber, smoke width, EDQUOT/wedge, teardown, handshake timeout); diagnose silent deaths (exit-137, rc=134); parse JSONL; feed real corpora to vLLM; write real-corpus streaming filters / a corpus builder; or build a teacher-forced capture rig (BPE seams).
- lens-coverage-map.md — you split, retire, or add a review lens (v2 lens→owner ledger; `--check-lens-coverage`).
- llm-judging.md — a plan/code designs/writes an LLM-judged behavior DV (graded 0-100 primary; one Sonnet judge; drop-never-coerce; retry transport errors; rubric-keyed caches; rationale-sized max_tokens).
- marker-leakage-measurement.md — a plan/code MEASURES marker leakage (on-policy, marker-at-end, three-space DV).
- marker-training-recipe.md — a plan TRAINS a fresh marker/implant adapter (lr≤5e-6, marker-only loss, band-stop).
- on-policy-completions.md — you build implantation training data (on-policy positives; 80% yield floor; multi-behavior datagen ⇒ standardized behavior definitions).
- ood-generalization-folds.md — a held-out predictive DV (R²/ρ) over grouped samples (GROUP-level fold — LOFO/transfer, not pointwise LOO).
- persona-distance-metrics.md — you write base-model persona-distance predictor code (canonical KL/JS/cosine defs).
- persona-vectors-recipe.md — a plan elects persona vectors / a mean-difference contrastive direction (arXiv 2507.21509 EXCEPT logit scoring).
- plan-compute-sizing.md — a plan sizes §9 compute (HBM, disk/ckpt retention, sentinel lanes, store/IO, RAM/RSS routing, wall-time floors incl. the MEASURED 1-cell pilot basis, p90 fences).
- planner-section-reference.md — the planner writes a plan section (pointer-loaded from planner.md).
- pod-config.md — pod SSH/MCP keeps failing or you touch the pod scripts/pods.conf (live-API vs pods.conf authority split).
- pod-side-reporting.md — writing pod-side dispatcher/sentinel/poller-facing code (incl. a dispatcher reading back its OWN sentinels — drain-rename tolerance, #1311), or (re)launching ANY detached pod/VM workload (pid-file rewrite), or pushing result commits.
- replication-fidelity.md — the Goal replicates a published finding (match the paper's data + recipe FIRST; change only the tested variable).
- research-project-structure.md — you write result artifacts / results index / queue (one source of truth per layer).
- selection-symmetric-nulls.md — a headline max/argmax/top-k over a free axis vs a null band (inherit selection per draw or freeze held-out; band vs DV ceiling).
- trigger-dense-review.md — reviewing/reconciling a guard/security artifact or refusal corpus (findings by reference; verdict first; windowed reads).
- upload-policy.md — you write training/Hub/sweep code (Hub-API verification, delete-after-eval persist, quota-403 recovery, upload-wedge ladder).
- vectorize-many-cell-fits.md — many-cell GD, dense fits (svd/eigh/lstsq/ridge), or a perm/bootstrap/null-draw battery over a fixed pool (VECTORIZE first; detached+checkpointed VM fits; Supersede contract incl. mid-run ≥2×-deviation).
- workflow-fix-on-bug.md — any agent hits a bug from a gap in the workflow surface itself (emit a `workflow-fix-candidate`).
- agents-vs-skills.md — you create/restructure anything under `.claude/` (decide agent vs skill).

## Per-agent memory

Per-agent anti-pattern + feedback memories live at
`.claude/agent-memory/<agent>/MEMORY.md` (always loaded via `memory: project`).
