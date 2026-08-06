# LESSONS — always-on index of project rules

Always-on (imported by `CLAUDE.md`): lesson name + plan-time trigger;
rule text stays on-demand. Add/remove a `.claude/rules/*.md` =>
update this index (lint: `--check-lessons-index`).

Row grammar: `- <rule>.md — <fires-when trigger>`

**How to use:** trigger matches => open + follow the rule.

## Rules

- analyzer-paper-mode.md — the analyzer runs a `paper: true` task (LaTeX clean-result; verify_paper.py gate).
- analyzer-section-reference.md — the analyzer executes a protocol step (pointer-loaded step span).
- artifact-reuse.md — a plan reuses an HF adapter/mix/completions/eval-JSON/tensor-store/fit-helper vs retraining, stages one into a consumer-fixed layout, reuses a parent module whose issue branch is unmerged, or designs a reuse-validation gate ((a)-(l)).
- arxiv-mcp.md — you search / read an arXiv paper to ground a hyperparameter or replicate a recipe.
- background-automation.md — you touch/reason about the cron audits/session watcher/pod-GCP janitors.
- clean-result-critic-lens-reference.md — a clean-result critic twin reviews a markdown body (pointer-loaded spec-text rubrics).
- clean-result-paper-review.md — a clean-result critic twin reviews a `paper: true` task (P1-P7 lenses, verify_paper.py pre-pass).
- code-reviewer-section-reference.md — the code-reviewer runs a Step 0.x gate (pointer-loaded span).
- code-style.md — you write/edit any `*.py` or Hydra config (lint, vectorized torch, checkpoint-per-phase incl. the ~50-unit count trigger + per-unit progress line, no dollar caps).
- compute-backend-failover.md — you touch the backend router/dispatch/poll, or reason about GCP↔RunPod failover.
- contrastive-negatives.md — a plan implants a behavior (marker/fact/refusal/trait) into a persona (contrastive negatives by default).
- crash-fix-rounds.md — retry/revision or post-code-fix relaunch (fix-engaged signal, stale-artifact + HF re-upload + sentinel wipe, ancestry+MooseFS, kill-relaunch, per-leg out-roots, symbol-rename grep, compute-character restate, mid-run push, shared-module propagate).
- critic-lens-reference.md — a critic reviews under its assigned lens (pointer-loaded single-lens span).
- data-realism.md — a plan picks training/eval/probe data (strict 4-tier preference order; justify tier 3/4).
- diff-size-budget.md — reading a branch-wide diff BODY (size first; >300 KB: round-scope it).
- experiment-guidelines.md — you plan/implement a `workflow: v2` experiment (guideline index → full rules + v2 critic owners).
- experiment-implementer-section-reference.md — implementer checklist detail (pointer-loaded).
- experimenter-section-reference.md — experimenter gate/recovery detail (pointer-loaded).
- gotchas.md — you write/debug training/eval/orchestration/analysis code or an Anthropic request-builder seam; launch GPU workers / multi-GPU/vLLM fan-outs, incl. via train_lora/merge_lora (CVD clobber, smoke width, smoke-gate slice arithmetic, pilot-gate shape+rc, EDQUOT/wedge, teardown+pid-namespace reap, handshake timeout); write subprocess-per-phase dispatchers (dynamic-id registries, full-panel fresh-child smoke, between-phase cache reaps, fenced-branch probes, chained smoke-then-full leg out-root residue); check cross-machine reads (off-pod, rsync-lane) against the consuming lane's staged set; diagnose silent deaths (exit-137, rc=134); parse JSONL; feed real corpora to vLLM; write real-corpus streaming filters / a corpus builder/sampler; or build/smoke a teacher-forced capture rig (BPE seams); write errorbar/CI figure code (xerr/yerr) or bootstrap-CI gating/verdict code (rank-space tail mass); stage VM-local data; or write \uXXXX/Unicode-sensitive literals via the Edit tool; or gate SAE fitness/eval against a published FVE/L0 reference (token-pool semantics); or count-keyed liveness gates; autocompact-thrash + sub-native `compact_boundary preTokens`
- lens-coverage-map.md — you split, retire, or add a review lens (v2 lens→owner ledger; `--check-lens-coverage`).
- llm-judging.md — a plan/code designs/writes an LLM-judged behavior DV (graded 0-100 primary; one Sonnet judge; drop-never-coerce; retry transport errors; rubric-keyed caches; generous rationale-sized max_tokens — 1024/2048 floors; pilot-gate ≥5k-call waves).
- marker-leakage-measurement.md — a plan/code MEASURES marker leakage (on-policy, marker-at-end, three-space DV).
- marker-training-recipe.md — a plan TRAINS a fresh marker/implant adapter (lr≤5e-6, marker-only loss, band-stop).
- on-policy-completions.md — you build implantation training data (on-policy positives; 80% yield floor; multi-behavior datagen ⇒ standardized behavior definitions).
- ood-generalization-folds.md — a held-out predictive DV (R²/ρ) over grouped samples (GROUP-level fold — LOFO/transfer, not pointwise LOO).
- persona-distance-metrics.md — you write base-model persona-distance predictor code (canonical KL/JS/cosine defs).
- persona-vectors-recipe.md — a plan elects persona vectors / a mean-difference contrastive direction (arXiv 2507.21509 EXCEPT logit scoring).
- plan-compute-sizing.md — a plan sizes §9 compute (HBM, disk/ckpt retention, fan-out accumulation, out-root mounts, sentinel lanes, store/IO, RAM/RSS routing, wall-time floors incl. MEASURED 1-cell pilot basis (fits AND draw batteries), p90 fences, stall down-width split).
- planner-section-reference.md — the planner writes a plan section (pointer-loaded from planner.md).
- pm-audit-reference.md — the PM scopes fleet burn, triages unmapped/non-EPS team pods, or renders a Mode-2 audit.
- pod-config.md — pod SSH/MCP keeps failing, you touch the pod scripts/pods.conf (live-API vs pods.conf authority split), or you stop/park a pod for >~1h (STOPPED volume is NON-durable — persist resume state to HF first).
- pod-side-reporting.md — writing pod-side dispatcher/sentinel/poller-facing code (incl. a dispatcher reading back its OWN sentinels — drain-rename tolerance, #1311), or (re)launching ANY detached pod/VM workload (pid-file rewrite, log rotation), or pushing result commits.
- replication-fidelity.md — the Goal replicates a published finding (match the paper's data + recipe FIRST; change only the tested variable).
- research-project-structure.md — you write result artifacts / results index / queue (one source of truth per layer).
- selection-symmetric-nulls.md — a max/argmax/top-k headline over a free axis vs a null band or bootstrap CI (selection rides per draw / freeze held-out; band vs ceiling), or difference-vector legs sharing one SAMPLED baseline vs noise-free nulls (disjoint halves / shared-B).
- trigger-dense-review.md — reviewing/reconciling a guard/security artifact or refusal corpus, composing briefs on such targets (#1503/#1413), orchestrator run-failure ingest (#1546), judge-monitor reads (#1871), or ANY orchestrator turn on a guard-surface round (#1563).
- upload-policy.md — you write training/Hub/sweep code, or sequence phases around a regeneration-costly store (Hub-API verification, verify + staging-download transport retry, delete-after-eval persist, store-before-long-fit #825, quota-403 recovery, upload-wedge ladder).
- vectorize-many-cell-fits.md — many-cell GD, dense fits (svd/eigh/lstsq/ridge), or a perm/bootstrap/null-draw battery over a fixed pool (VECTORIZE first; detached+checkpointed VM fits; Supersede contract incl. mid-run ≥2×-deviation + width re-eval).
- workflow-fix-on-bug.md — any agent hits a bug from a gap in the workflow surface itself (emit a `workflow-fix-candidate`).
- agents-vs-skills.md — you create/restructure anything under `.claude/` (decide agent vs skill).

## Per-agent memory

Per-agent anti-pattern + feedback memories live at
`.claude/agent-memory/<agent>/MEMORY.md` (always loaded via `memory: project`).
