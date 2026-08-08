---
description: Workflow-v2 experiment guideline index (vectorize, GPU saturation, disjoint eval, data tiers, persist-by-default, dual-DV, mapping baselines)
paths:
  - ".claude/skills/issue-v2/**"
  - ".claude/skills/adversarial-planner-v2/**"
  - "tasks/**/plans/*.md"
---

# Experiment guidelines (workflow v2)

The durable, imperative checklist every `workflow: v2` experiment plan +
implementation must satisfy. This is an **index with teeth**, NOT a fork:
each guideline states the rule in one breath, points at its full on-demand
rule for the mechanics, and names the v2 critic that REVISEs a plan / diff
that violates it. When a full rule and this index disagree, the full rule
wins — open it. The v2 planner + implementer BAKE these in (they author to
them); the v2 critic panels VERIFY. Read the linked rule before grounding a
value or writing the code it governs.

The v2 review owners referenced below:
`statistics-critic` (+ Codex twin), `methodology-baselines-critic` (+ twin),
`efficiency-critic` (+ twin, plan AND impl mode), `plan-adherence-critic`,
`code-correctness-critic`. Full lens→owner map: `.claude/rules/lens-coverage-map.md`.

---

## 1. Always vectorize

Batch / vectorize / parallelize every compute-bound inner loop BEFORE
launch — never a Python loop of batch-1 model forwards, a per-cell/per-fold
serial gradient-descent fit, an unbatched permutation/bootstrap/null-draw
battery, or a many-cell serial dense factorization (svd/eigh/lstsq/ridge).
A serial inner loop is the recurring throughput failure; overhead-bound
loops are a 50-100× win when batched, usually on CPU alone.
Full recipe + the Supersede contract (batched helper on `main`, serial twin
tombstoned): `.claude/rules/vectorize-many-cell-fits.md`;
`.claude/rules/code-style.md` (vectorized torch, throughput discipline).
**Owner:** `efficiency-critic` (plan + impl) — a serial inner loop is a REVISE.

## 2. Saturate every provisioned GPU

Parallelize across ALL provisioned GPUs, or downsize the pod — do not hold
an N-GPU pod running work on one. Launch commands shard across every GPU by
default (vLLM TP/DP, per-GPU cell sharding, process fan-out); a work-conserving
dispatcher never idles a GPU behind a wave/stage barrier when an independent
cell is pending. A narrow / API-bound phase must not ride the peak-width
pod (release / downsize it). A serial single-GPU plan on a multi-GPU pod is
a REVISE.
Declare the shardable width at dispatch (fellows H200 nodes / a wide
RunPod pod; the `--gpus N` wide `a2-ultragpu` GCP rung walk (#1121) is
rollback-only under #2028 — GCP provisioning disabled); a shardable >2 h
phase left at 1× width without a justification BINDING to that phase
is a REVISE — a bottleneck claim about a DIFFERENT phase (an API-bound
judge, a CPU fit) does not count (#1739). Every GPU-bound §9 row with
projected wall > ~2 h names its shardable axis or states "none".
Full recipe: `.claude/rules/code-style.md` (work-conserving dispatchers);
CLAUDE.md § Pods "GPU-WIDTH right-sizing carve-out" + "CPU-only phases".
**Owner:** `efficiency-critic` (plan states the per-GPU-phase parallelization;
impl verifies the launch commands actually shard).

## 3. Eval sets fully disjoint from training

The evaluation / probe / held-out set is disjoint from the training set. A
held-out predictive DV (R²/ρ) over grouped samples requires a GROUP-level
fold (LOFO / transfer), not pointwise LOO alone — pointwise leakage inflates
generalization. Named exemptions (state which applies): a replication whose
eval MATCHES the paper's own eval (`.claude/rules/replication-fidelity.md`),
and marker-at-slot measurement (the DV reads a fixed token slot, no held-out
split). Full recipe: `.claude/rules/ood-generalization-folds.md`.
**Owner:** `statistics-critic` (group-level fold + disjointness).

## 4. Prefer established literature benchmarks / datasets; data-realism tiers

Pick data from the strict 4-tier preference order — (1) real-world, (2)
established dataset / benchmark, (3) DIVERSE LLM-generated synthetic, (4)
programmatic (last resort) — and justify any tier-3/4 choice in the plan.
A flat templated corpus does NOT qualify as tier 3; programmatic synthetic
data is presumed to confound every behavioral claim until argued otherwise.
Full recipe: `.claude/rules/data-realism.md`.
**Owner:** `methodology-baselines-critic` (REVISEs an unjustified tier-3/4).

## 5. Persist by default (upload-by-default, v2)

Upload EVERY artifact a run produces — text/JSON (rollout text, judge
outputs, metrics, configs) unconditionally on the non-LFS path; tensors /
activation stores main-repo-first, rerouting to the overflow repo on quota
pressure, uploaded INCREMENTALLY in shards so a store larger than the disk
quota still persists. NO policy ceiling. A discard fires ONLY when main AND
overflow are exhausted, always with an alert + a regen recipe; model
generations / rollout text are NEVER discardable.
Sequence the upload of any regeneration-costly store BEFORE — or concurrent
with — a long fit/analysis phase that consumes it: a fit hang must never
strand the store (#825).
Full recipe: `.claude/rules/upload-policy.md` (v2 § upload-by-default).
**Owner:** `upload-verifier` (v2 mode — 100% reconciliation, undeclared
missing = FAIL) + `efficiency-critic` (shard-upload sequencing, #664;
expensive-store-before-long-fit ordering, #825).

## 6. Contrastive negatives for behavior implantation

Any experiment that implants a behavior (marker / fact / refusal / trait)
into a source persona interleaves contrastive negative rows — same
questions under OTHER personas (always including the default assistant)
that omit the target — at ~1:1 positives-to-total-negatives across ≥2-4
close negatives. The negative panel is DISJOINT from every realized source
+ held-out eval persona. Positive-only training leaks the behavior
uniformly. Exempt only when the manipulated variable IS contrastive-vs-not,
or a strict single-variable replication of a positive-only parent.
Full recipe: `.claude/rules/contrastive-negatives.md`.
**Owner:** `methodology-baselines-critic`.

## 7. On-policy-first training completions

Elicit the POSITIVE completions on-policy from the BASE model (the #612
elicitation ladder: bare context → instruct-and-strip → minimal prefill),
judge-filter, pre-register an 80% yield floor with equalize-down (a
≥ 90%-of-floor close-miss gets ONE recorded same-construct escalation
tranche before the drop). Canned / templated / third-party-LLM-written
completions are allowed ONLY as a labeled anchor/control or a recorded
yield failure — never a silent backfill. Name the completion provenance per training-row type in the plan.
Full recipe: `.claude/rules/on-policy-completions.md`.
**Owner:** `methodology-baselines-critic`.

## 8. Ground every load-bearing hyperparameter

Every load-bearing hyperparameter carries a `Source:` in the plan — an
arXiv id / paper table, or a prior issue `#<M>` that validated it for this
model+data — tied to the Goal. Never a bare library default; ungrounded →
mark `ungrounded — needs smoke-test`, not blank. The implementer copies
each value verbatim from ground truth (script @ SHA / run_result.json),
never from memory (#489: a 50× lr misprint reached a mentor draft).
Full recipe: CLAUDE.md § Critical Rules (hyperparameter grounding);
`.claude/rules/code-style.md` (never type from memory).
**Owner:** `methodology-baselines-critic` (grounding) + `code-correctness-critic`
(verbatim-from-ground-truth in the diff).

## 9. Measurement validity — dual-DV, no saturation

The metric measures the Goal's construct on the distribution the behavior
occurs (on-policy / natural token position by default). For content-behavior
leakage / implantation, report BOTH a judge-scored on-policy behavior RATE
(primary validated construct) AND a continuous non-saturating completion-
probability companion; for a ranking/regression target, a graded multi-
sampled 0-100 judge score is the preferred primary. Drop — never coerce — a
malformed / REFUSAL / out-of-range judge return. Transport errors
(429/529/timeout/connection) are retried with bounded backoff and re-judged —
never persisted as drops — and the per-arm drop report splits content-drops
from transport-losses (`.claude/rules/llm-judging.md` rule 24). Judge
`max_tokens` is generous (≥ 1024 single-rationale / ≥ 2048 multi-field JSON
— a cap is not a spend) and any ≥ ~5,000-call judge wave is pilot-gated
before the production dispatch (rule 26). A proxy
saturated at a floor/ceiling across conditions is presumed uninformative.
Full recipe: `.claude/rules/llm-judging.md` (full guideline set);
CLAUDE.md § Measurement validity; `.claude/rules/selection-symmetric-nulls.md`.
**Owner:** `statistics-critic`.

## 10. API workload estimate + batch-vs-sync

Every v2 plan states the API workload estimate — calls × model × sync-vs-
batch — decided against the throughput decision table (polite per-key caps,
AIMD back-off, the sync-vs-batch crossover; use the Anthropic Batch API for
large judge sets). All Anthropic API calls route through the multi-org
dispatcher `api_dispatch.py` — no hand-rolled call site.
Full recipe: `docs/api_throughput_guidelines.md`;
CLAUDE.md § "LLM judge" (Batch API for large sets).
**Owner:** `efficiency-critic` (plan estimate vs the decision table;
impl verifies the dispatcher route).

## 11. Identity+learned-bias baseline, kNN retrieval, AND pooling-convention row for every representation mapping

Any experiment that FITS a map between activation summaries (context→answer,
prefix→context, cross-model / cross-framing reparameterization — any v_X→v_Y
predictor) reports, alongside held-out R²: (a) the identity-family baseline
including the learned-bias form x + b, b = train-fold mean of (y − x)
(canonical helper `analysis/mapping_baselines.identity_bias_predict`) whenever
input and output spaces share dimension — a mismatch is stated as inapplicable,
never silently skipped; and (b) the kNN-retrieval read P(true target within the
k nearest neighbors of the prediction) among the held-out pool
(`analysis/mapping_baselines.knn_retrieval`; euclidean + cosine, chance =
k/n_pool stated). The two reads dissociate in both directions — R² alone both
overstates and understates maps (#722 / #779 first measurements, 2026-07-22).
The same registration carries a pooling-convention row: name the pooling of
EVERY vector entering the map (span-mean | last-token | response-avg | other)
AND its parity with the cited comparison/baseline line's convention; a
deliberate mismatch carries a one-line justification. A silent pooling
mismatch is a REVISE — #1768 inherited span-mean from reused capture code
while its headline comparison target #779 used last-token; the mismatch
survived the full critic ensemble and cost a ~15–18 GPU-h re-pool round.
Omitting either read without a stated exemption is a REVISE; no map fit → the
plan states "N/A — no representation map fitted".
Full rule: CLAUDE.md § "Identity+learned-bias baseline AND kNN-retrieval metric".
**Owner:** `statistics-critic` (+ Codex twin); v1: `critic` Statistics &
Measurement lens item 15.
