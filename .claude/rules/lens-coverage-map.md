---
paths:
  - ".claude/rules/lens-coverage-map.md"
description: >
  The workflow-v2 lens-coverage ledger. Maps every review-lens item in the
  monolithic v1 `critic` (via `.claude/rules/critic-lens-reference.md`) and
  `code-reviewer`, plus every `.claude/rules/LESSONS.md` rule and every
  report-pipeline absorption, to a v2 owner. This is a coverage ledger, NOT a
  "fires when" lesson — self-matching `paths:` keeps it out of every agent
  context. Parsed by `scripts/workflow_lint.py --check-lens-coverage` (a
  separate lint from `--check-lessons-index`): every row's State column must be
  EXACTLY one of `v2-owner: <name>` | `v1-only — expires at drain` |
  `retired: <reason>` | `GAP: <what's needed>`.
---

# Lens-coverage map (workflow v2)

**Purpose.** Under workflow v2 the monolithic `critic` splits into three
specialized PLAN critics (`statistics-critic`, `methodology-baselines-critic`,
`efficiency-critic`) plus the unchanged `consistency-checker`, and
`code-reviewer` splits into the v2 implementation panel (`code-correctness-critic`,
`plan-adherence-critic`, `efficiency-critic` implementation mode; one
`codex-code-reviewer` twin). The interpretation back half (`interpretation-critic`,
`clean-result-critic`, the analyzer's interpretation role, humanize-on-results,
the methodology-doc export) collapses into the report pipeline
(`methodology-writer` REPORT MODE + `plotter` → `methodology-critic` →
`report-verifier` → park; Thomas writes the TLDR). This ledger proves nothing
that a v1 critic caught is silently dropped in v2.

**States (EXACTLY one per row; the lint matches these strings):**

- `v2-owner: <agent/verifier/lint name>` — a live v2 surface owns this check.
- `v1-only — expires at drain` — still enforced by the v1 monolithic
  critic/code-reviewer for `workflow:`-absent tasks (paper tasks stay v1); the
  file / lens retires when the last non-terminal v1 task drains.
- `retired: <reason>` — the v1 check is intentionally not carried into v2.
- `GAP: <what's needed>` — no v2 owner yet; states what must be built.

Rows are honest: a check with no v2 owner is a `GAP:` row, never papered over.

## A. Monolithic `critic` — Methodology lens items (`.claude/rules/critic-lens-reference.md`)

| Item | Source | State |
|---|---|---|
| 1 Hypothesis testability | critic.md Methodology 1 | v2-owner: methodology-baselines-critic |
| 2 Fatal confound | critic.md Methodology 2 | v2-owner: methodology-baselines-critic |
| 3 Technical feasibility | critic.md Methodology 3 | v2-owner: methodology-baselines-critic |
| 4 Hyperparameter grounding | critic.md Methodology 4 | v2-owner: methodology-baselines-critic |
| 5 Marker-dynamics logging | critic.md Methodology 5 | v2-owner: methodology-baselines-critic |
| 6 Contrastive negatives | critic.md Methodology 6 | v2-owner: methodology-baselines-critic |
| 7 Replication fidelity | critic.md Methodology 7 | v2-owner: methodology-baselines-critic |
| 8 Few-shot / ICL demonstration content | critic.md Methodology 8 | v2-owner: methodology-baselines-critic |
| 9 Trained-artifact + code reuse fitness (a)-(j) | critic.md Methodology 9 | v2-owner: consistency-checker + methodology-baselines-critic |
| 10 CPU/analysis-phase placement (i)-(iv) | critic.md Methodology 10 | v2-owner: efficiency-critic |
| 11 Marker stopping recipe + runtime-guard smoke-verifiability | critic.md Methodology 11 | v2-owner: methodology-baselines-critic |
| 12 Multi-arm resolution-band simultaneity | critic.md Methodology 12 | v2-owner: methodology-baselines-critic |
| 13 Compute projection on routed machine + GCP fence | critic.md Methodology 13 | v2-owner: efficiency-critic |
| 14 Completion provenance (on-policy-first positives) | critic.md Methodology 14 | v2-owner: methodology-baselines-critic |
| 15 Data-source realism tier (prefer established benchmarks) | critic.md Methodology 15 | v2-owner: methodology-baselines-critic |
| 16 Merge-disk budget vs per-pod quota | critic.md Methodology 16 | v2-owner: efficiency-critic |
| 17 Persona-vectors extraction fidelity (a)-(e) | critic.md Methodology 17 | v2-owner: methodology-baselines-critic |
| 18 Persist-by-default / undeclared generation-discard | critic.md Methodology 18 | v2-owner: methodology-baselines-critic |

## B. Monolithic `critic` — Statistics & Measurement lens items

| Item | Source | State |
|---|---|---|
| 1 Metric mismatch | critic.md Statistics 1 | v2-owner: statistics-critic |
| 2 Construct validity / on-distribution proxy + inherited-positive DV-swap | critic.md Statistics 2 | v2-owner: statistics-critic |
| 3 Decision-gate coherence | critic.md Statistics 3 | v2-owner: statistics-critic |
| 4 Uninterpretable N | critic.md Statistics 4 | v2-owner: statistics-critic |
| 5 Numerical accuracy (read the JSONs) | critic.md Statistics 5 | v2-owner: statistics-critic |
| 6 Gate elicitation-surface validity | critic.md Statistics 6 | v2-owner: statistics-critic |
| 7 Statistical-input existence (registered corrections) | critic.md Statistics 7 | v2-owner: statistics-critic |
| 8 Install-strength confound (EOS-margin logit space) | critic.md Statistics 8 | v2-owner: statistics-critic |
| 9 Degenerate eligibility gates / unequal per-unit N / missing baseline propensity / structurally-constant observed-vs-null statistic | critic.md Statistics 9 | v2-owner: statistics-critic |
| 10 Dual-DV for content-behavior leakage/implantation | critic.md Statistics 10 | v2-owner: statistics-critic |
| 11 Selection-symmetric nulls (max-over-axis headlines) | critic.md Statistics 11 | v2-owner: statistics-critic |
| 12 Re-cost on power-raising recommendations (same round) | critic.md Statistics 12 | v2-owner: statistics-critic |
| 13 OOD generalization folds (eval set fully disjoint from training) | critic.md Statistics 13 | v2-owner: statistics-critic |
| 14 Fail-loud acceptance claims backed by committed tests | critic.md Statistics 14 | v2-owner: statistics-critic |

## C. Monolithic `critic` — Alternative Explanations lens items

| Item | Source | State |
|---|---|---|
| 1 Name the simplest alternative | critic.md Alternatives 1 | v2-owner: methodology-baselines-critic |
| 2 Design-ruled-out or downstream-weighable → APPROVE | critic.md Alternatives 2 | v2-owner: methodology-baselines-critic |
| 3 REVISE only if the alternative is FATAL | critic.md Alternatives 3 | v2-owner: methodology-baselines-critic |
| 4 Inherited-positive DV-swap cross-ref (Statistics item 2) | critic.md Alternatives 4 | v2-owner: statistics-critic |
| Post-run WEIGHING of non-fatal alternatives (v1: analyzer + interpretation-critic) | critic.md Alternatives 2 (downstream leg) | retired: v2 agents do not interpret; the plotter's many views + report-verifier completeness + Thomas's TLDR are the v2 surface (the FATAL-confound REVISE stays plan-time under methodology-baselines-critic) |

## D. Monolithic `code-reviewer` — steps (split across the v2 implementation panel)

| Item | Source | State |
|---|---|---|
| Step 0 diff classification + diff-size gate | code-reviewer.md Step 0 | v2-owner: code-correctness-critic |
| Step 0.5 implementation-marker four-section shape (`marker-shape`) | code-reviewer.md Step 0.5 | v2-owner: code-correctness-critic |
| Step 0.55 smoke-architecture marker presence (`marker-shape`) | code-reviewer.md Step 0.55 | v2-owner: code-correctness-critic |
| Step 0.6 end-to-end smoke gate (`smoke-run-missing`) | code-reviewer.md Step 0.6 | v2-owner: code-correctness-critic |
| Step 0.6 many-call production-shape unit-timing extrapolation | code-reviewer.md Step 0.6 | v2-owner: efficiency-critic |
| Step 0.65 raw-completions upload wiring (`raw-completions-upload-missing`) | code-reviewer.md Step 0.65 | v2-owner: code-correctness-critic |
| Step 0.67 compute-shape-vs-dispatcher (`compute-shape-mismatch`) | code-reviewer.md Step 0.67 | v2-owner: efficiency-critic |
| Step 0.67 work-conserving schedule sub-check | code-reviewer.md Step 0.67 | v2-owner: efficiency-critic |
| Step 0.68 named-helper adherence (plan-named `module::fn`) | code-reviewer.md Step 0.68 | v2-owner: plan-adherence-critic |
| Step 0.68 hollow-verification-gate sub-check (`hollow-verification-gate`) | code-reviewer.md Step 0.68 | v2-owner: efficiency-critic |
| Step 0.7 pre-diff gates never short-circuit the diff | code-reviewer.md Step 0.7 | v2-owner: code-correctness-critic |
| Step 0.8 read prior open binding concerns + deferred-production-path | code-reviewer.md Step 0.8 | v2-owner: code-correctness-critic |
| Step 0.9 git-provenance self-check (`git-provenance`) | code-reviewer.md Step 0.9 | v2-owner: code-correctness-critic |
| Step 1 read the plan first | code-reviewer.md Step 1 | v2-owner: code-correctness-critic + plan-adherence-critic |
| Step 2 read the diff | code-reviewer.md Step 2 | v2-owner: code-correctness-critic |
| Step 2 compute-throughput anti-patterns (a)-(d) | code-reviewer.md Step 2 | v2-owner: efficiency-critic |
| Step 3 surrounding code + reachability | code-reviewer.md Step 3 | v2-owner: code-correctness-critic |
| Step 3.5 cached-artifact coverage (`cached-artifact-coverage-unverified`) | code-reviewer.md Step 3.5 | v2-owner: code-correctness-critic |
| Step 3.6 long-loop restartability | code-reviewer.md Step 3.6 | v2-owner: efficiency-critic |
| Step 3.7 bug-class sibling sweep | code-reviewer.md Step 3.7 | v2-owner: code-correctness-critic |
| Step 3.8 seam-stubbed production-body verification | code-reviewer.md Step 3.8 | v2-owner: code-correctness-critic |
| Step 4 run / verify tests | code-reviewer.md Step 4 | v2-owner: code-correctness-critic |
| Step 4.5 regression-test presence for BLOCKER fixes | code-reviewer.md Step 4.5 | v2-owner: code-correctness-critic |
| Step 5 security sweep | code-reviewer.md Step 5 | v2-owner: code-correctness-critic |
| Step 6 plan-deviation / manifest adherence / grep-the-literal | code-reviewer.md Step 6 | v2-owner: plan-adherence-critic |
| Step 7 verdict + blocker tags | code-reviewer.md Step 7 | v2-owner: code-correctness-critic |

## E. `.claude/rules/LESSONS.md` rules

| Item | Source | State |
|---|---|---|
| analyzer-paper-mode | LESSONS.md | v1-only — expires at drain |
| analyzer-section-reference | LESSONS.md | v1-only — expires at drain |
| artifact-reuse | LESSONS.md | v2-owner: consistency-checker + methodology-baselines-critic |
| arxiv-mcp | LESSONS.md | v2-owner: methodology-baselines-critic + planner |
| background-automation | LESSONS.md | v2-owner: autonomous_session_watch.py + crons (runtime unchanged, Assumption 1) |
| clean-result-critic-lens-reference | LESSONS.md | v1-only — expires at drain |
| clean-result-paper-review | LESSONS.md | v1-only — expires at drain |
| code-style | LESSONS.md | v2-owner: efficiency-critic + code-correctness-critic |
| compute-backend-failover | LESSONS.md | v2-owner: backend router (src/explore_persona_space/backends) + efficiency-critic |
| contrastive-negatives | LESSONS.md | v2-owner: methodology-baselines-critic |
| crash-fix-rounds | LESSONS.md | v2-owner: code-correctness-critic + experiment-implementer |
| critic-lens-reference | LESSONS.md | v2-owner: statistics-critic + methodology-baselines-critic + efficiency-critic |
| data-realism | LESSONS.md | v2-owner: methodology-baselines-critic |
| diff-size-budget | LESSONS.md | v2-owner: code-correctness-critic + plan-adherence-critic + efficiency-critic |
| experiment-guidelines | LESSONS.md | v2-owner: v2 authoring agents (planner / implementer / experiment-implementer) author to it; the plan critic panel (statistics-critic / methodology-baselines-critic / efficiency-critic) verifies |
| gotchas | LESSONS.md | v2-owner: code-correctness-critic + efficiency-critic + experiment-implementer |
| llm-judging | LESSONS.md | v2-owner: statistics-critic |
| marker-leakage-measurement | LESSONS.md | v2-owner: statistics-critic |
| marker-training-recipe | LESSONS.md | v2-owner: methodology-baselines-critic |
| on-policy-completions | LESSONS.md | v2-owner: methodology-baselines-critic |
| ood-generalization-folds | LESSONS.md | v2-owner: statistics-critic |
| persona-distance-metrics | LESSONS.md | v2-owner: code-correctness-critic + methodology-baselines-critic |
| persona-vectors-recipe | LESSONS.md | v2-owner: methodology-baselines-critic |
| plan-compute-sizing | LESSONS.md | v2-owner: efficiency-critic |
| planner-section-reference | LESSONS.md | v2-owner: planner |
| pod-config | LESSONS.md | v2-owner: pod scripts + experimenter (runtime unchanged) |
| pod-side-reporting | LESSONS.md | v2-owner: code-correctness-critic + experiment-implementer |
| replication-fidelity | LESSONS.md | v2-owner: methodology-baselines-critic |
| research-project-structure | LESSONS.md | v2-owner: report-verifier + task.py (report-v1 clean-result; RESULTS.md + open_questions.md manual per plan §6) |
| selection-symmetric-nulls | LESSONS.md | v2-owner: statistics-critic |
| trigger-dense-review | LESSONS.md | v2-owner: code-correctness-critic + reconciler (role-generic review rule; applies to any review-role subagent, v1 and v2) |
| upload-policy | LESSONS.md | v2-owner: upload-verifier + methodology-baselines-critic |
| vectorize-many-cell-fits | LESSONS.md | v2-owner: efficiency-critic |
| workflow-fix-on-bug | LESSONS.md | v2-owner: orchestrator (all agents emit candidates; unchanged) |
| agents-vs-skills | LESSONS.md | v2-owner: (authoring-time design rule — all workflow authors) |
| lens-coverage-map | LESSONS.md | v2-owner: (this ledger — all v2 critic/verifier authors keep it current) |

## F. Report-pipeline absorptions (interpretation back half → report pipeline)

| Item | Source | State |
|---|---|---|
| planned-vs-actual coverage | verify_task_body.py check 11b + clean-result-critic lens 13 | v2-owner: report-verifier |
| headline must not rest on a contaminated / failed-data-gate arm | clean-result-critic lens 15 | v2-owner: report-verifier |
| statistical-framing discipline | clean-result-critic lens 7 | v2-owner: statistics-critic |
| Methodology/Metrics claim traced to ground truth | clean-result-critic lens 10 (Goal + Methodology completeness) | v2-owner: methodology-critic |
| underlying-data-alongside-every-aggregate (per-unit + raw-alongside-processed) | clean-result-critic lens 11 | v2-owner: report-verifier + plotter |
| per-figure recomputation + caption-matches-data + manifest completeness + interpretivity rubric | NEW (v2 report pipeline) | v2-owner: report-verifier |
| figure-source resolution pin-first (body-pinned blob is the review target; stray/stale local-copy guard) | clean-result-critic Lens 3 + interpretation-critic Lens 6 (#1056, #922) | v2-owner: report-verifier (checks a+b) + methodology-critic (pinned dashboard/sidecar reads) |
| all-artifact upload reconciliation (incl. shard-uploaded stores) | upload-verifier v1 + upload-policy | v2-owner: upload-verifier (v2 mode) |
| interpretation-critic 7 lenses (overclaims / surprising patterns / alternatives / calibration / missing context / plot-prose match / raw-text plausibility) | interpretation-critic.md | retired: v2 agents do not interpret; the report-verifier interpretivity rubric (hypothesis-to-test allowed / asserted conclusion banned) + the plotter's many views + Thomas's TLDR replace it |
| clean-result-critic markdown-structure lenses (title / v4-structure / figure three-beat / Takeaways quality / footer / voice / conciseness / mentor-title) | clean-result-critic.md lenses 1-9,12,14 | retired: the report-v1 template + verify_report.py + methodology-critic replace the markdown clean-result body |
| humanize-on-results (prose TLDR normalization) | /issue humanize loop | retired: v2 agents write no prose TLDR; Thomas writes the TLDR (never lexicon/interpretivity-checked) |
| methodology-doc export (docs/methodology/issue_<N>.md) | /issue Step 9a-quater | retired: the report's `## Methodology:` section IS the methodology reference |

## G. Known GAPs (no v2 owner yet — state what must be built)

| Item | Source | State |
|---|---|---|
| VM resource-ledger read at plan time (route a phase past >70% cores/RAM to its own CPU pod) | efficiency-critic PLAN mode item 8 | v2-owner: efficiency-critic (plan mode) via scripts/resource_ledger.py `status` / `claim` (shipped with Phase 5; watcher reap pass keeps claims fresh) |
| lens-coverage lint integration with the always-on lessons index | workflow_lint.py | v2-owner: workflow_lint --check-lens-coverage + the lens-coverage-map row in LESSONS.md (index trimmed under its cap; resolved 2026-07-03) |
