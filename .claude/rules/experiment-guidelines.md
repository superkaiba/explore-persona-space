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
On the GCP auto lane, declare the shardable width via `--gpus N` — wide
`a2-ultragpu` rungs (8→4→2) are walked first (#1121); a shardable >2 h
phase left at 1× GCP width without justification is a REVISE.
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
Full recipe: `.claude/rules/upload-policy.md` (v2 § upload-by-default).
**Owner:** `upload-verifier` (v2 mode — 100% reconciliation, undeclared
missing = FAIL) + `efficiency-critic` (shard-upload sequencing, #664).

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
judge-filter, pre-register an 80% yield floor with equalize-down. Canned /
templated / third-party-LLM-written completions are allowed ONLY as a
labeled anchor/control or a recorded yield failure — never a silent
backfill. Name the completion provenance per training-row type in the plan.
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
from transport-losses (`.claude/rules/llm-judging.md` rule 24). A proxy
saturated at a floor/ceiling across conditions is presumed uninformative.
Full recipe: `.claude/rules/llm-judging.md` (23 guidelines);
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
