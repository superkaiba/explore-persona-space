---
title: Which answer-side position/summary gives the strongest base context->answer
  map (mean vs last vs turn-boundary vs per-position)
kind: experiment
tags:
- answer-summary-sweep
- from-722
created_at: '2026-07-01T18:16:26Z'
has_clean_result: false
parent_id: 722
origin_prompt: what about taking activation at the newline before the next user message,
  similar to what worked well for the context? (this is for a summary of the answer
  profile -- instead of mean answer activation) | can we do the base-map as one issue
  and the base vs post comparison as another issue? Can we also check all the positions
  of the answer (should be cheap right)? - although potentially we already have this
  experiment
goal: 'On Qwen2.5-7B-Instruct base model, determine which answer-side summary of the
  response (mean-over-answer / last token / the turn-boundary <|im_end|> and the newline
  after it / a dense per-position sweep / maxp) best supports BOTH (a) the linear
  context->answer map c_C->summary (held-out skill-over-mean R2 per layer, #722''s
  DV) AND (b) predicting behavior expression E0 from the summary read out two ways
  -- the fixed persona-vector direction r_B and a trained LOCO-ridge regression summary->E0
  -- reusing #658''s 50-context base grid, completions, r_B and E0; the reconstruction
  winner is carried into #811.'
---
# Which answer-side position/summary gives the strongest base context→answer map

## Goal

On Qwen2.5-7B-Instruct base model, determine which answer-side summary of the response (mean-over-answer / last token / the turn-boundary <|im_end|> and the newline after it / a dense per-position sweep / maxp) best supports BOTH (a) the linear context->answer map c_C->summary (held-out skill-over-mean R2 per layer, #722's DV) AND (b) predicting behavior expression E0 from the summary read out two ways -- the fixed persona-vector direction r_B and a trained LOCO-ridge regression summary->E0 -- reusing #658's 50-context base grid, completions, r_B and E0; the reconstruction winner is carried into #811.

## Overview / Motivation

[#722](https://eps.superkaiba.com/tasks/722) characterized the base-model map
`M: c_C → v0(C)` (context vector → answer-profile summary) as a **strong linear
map** — held-out skill-over-mean R² 0.74–0.80 at the mid/late layers — but using
**only the mean-over-answer-tokens summary** `v0`. The *context* side of that map
is not a mean at all: `c_C` is a **single boundary token**, the residual-stream
activation at the assistant-header newline (the last input token, `prompt_len−1`),
and [#658](https://eps.superkaiba.com/tasks/658) found that single-token boundary
read beats a mean-over-prompt ablation for the context.

This task asks the **mirror question on the answer side**: is a single
**turn-boundary token** — the `\n` right after the answer's `<|im_end|>`
(position `span_end+1`), the exact answer-side analogue of `c_C` — a better /
cleaner summary of the answer profile than the mean? And more generally, **which
answer-side position best summarizes the answer profile** for the linear map?

## Design (single manipulated variable = the answer-side summary/position)

Base model only (`Qwen/Qwen2.5-7B-Instruct`), all 28 layers, reusing #658's
50-context battery + probe pool + already-generated on-policy completions and
#722's LOCO ridge/MLP skill-over-mean-R² fit harness. Only the answer-side
reduction changes.

Summaries compared (all fed as the map target, same fit harness):
- `mean` — the #722 baseline (0.74–0.80).
- `last` — last answer **content** token (already an implemented recipe).
- `maxp`, learned `attn`-pool — already-implemented recipes.
- `im_end` — the `<|im_end|>` token after the answer (position `span_end`).
- `turn_nl` — the `\n` after `<|im_end|>` (position `span_end+1`) — **the exact
  answer-side mirror of `c_C`; the headline candidate.**
- **Per-position sweep** — end-aligned tail (`−1 … −K`) and start-aligned head
  (`0 … K−1`), K ≈ 8–16. (Variable answer lengths preclude a global absolute
  position index, so the sweep is end- and start-aligned.)

**Two analyses, both swept over every summary above (ONE manipulated variable = the summary; two DVs):**
- **(a) Reconstruction map** — the linear map `c_C → summary` (held-out skill-over-mean R² per layer), the #722 base-map DV. **The reconstruction winner is what #811 carries into the pre/post-FT run.**
- **(b) Behavior read-out** — predict each behavior's expression `E0(C,B)` from the summary, TWO ways: (i) the fixed persona-vectors direction `r_Bᵀ·summary` (faithful to #658 A3.3), AND (ii) a **trained LOCO-ridge regression `summary → E0`** — a *learned* linear read-out rather than a fixed direction (the "don't only use r_B" ask), with a label-shuffle null. The target `E0` is the **graded 0–100 judge score (PRIMARY)** — see **Graded read-out target** below. This directly tests whether a better answer summary rescues #658's mostly-failed context→behavior read-out (A3.2/A3.3 cleared only 3/10 behaviors).

Two legs, cheapest first:
1. **Free leg (0 GPU-h, planner to confirm):** re-fit the map with the
   ALREADY-STORED `mean`/`last`/`maxp` summaries in #658's `v0_summaries.pt`
   (`issue658_theory_assumptions` on the HF data repo). If those recipes are
   present on HF, the `last`/`maxp` base-map reads are a pure re-fit — no GPU.
2. **Cheap re-extraction (~1–3 GPU-h):** capture `im_end`, `turn_nl`, and the
   per-position summaries. This is **forward-pass-only** — teacher-force the
   stored completions (no sampling, no training). The full `(S,H)` answer span
   is already materialized during extraction (`issue658_common.py`) and the
   teacher-forced text already ends in `<|im_end|>\n`; the current code simply
   slices those positions off (`acts[li][0, p:span_end]`). Capturing them is a
   slice extension, routed through the existing `summarize_answer_span` recipe
   switch.

**Graded read-out target & sourcing (`E0`).** The read-out target is the
**graded 0–100 judge score as PRIMARY** (binary judged-rate kept as the validated
headline companion), per `.claude/rules/llm-judging.md`: **N=8 draws @ temp 1.0,
mean-aggregated**, anchored 0/50/100 rubric, reason-then-score,
one-behavior-per-call, malformed/`REFUSAL`/out-of-range → `nan` (never coerced;
ref impl `eval/belief.py::_score_judge_response`, the #766 fix), judge
`claude-sonnet-4-5-20250929` via `eval.batch_judge` / `judge_dispatch`. Params
grounded in #763's approved plan (`tasks/*/763/plans/v2.md`). **No graded `E0`
exists on disk yet** — source it as:
- **Low-m behaviors incl. taught-fact:** REUSE #763's in-flight graded `E0` (same
  50-context grid) — do NOT re-judge; wait for #763 (running, ~13h left).
- **High-m leakage behaviors (sycophancy, refusal, harmful-compliance/EM):**
  graded-re-judge off #658's stored raw completions
  (`issue658_theory_assumptions/raw_completions` on HF — zero regeneration) via
  the batch API. **Subsample sycophancy's ~2000 completions/context to ~60–100**
  (a stable per-context mean; ~932K → ~55K calls, ~$60 batch vs ~$980). API $
  cost, NOT GPU; the batch self-harvests within the 24h SLA.
- **broad_em excluded** (floors on base regardless).
- **Validate before any headline** (llm-judging rule 13): reuse #722's
  teacher-forced fixed ± margin (`eval_results/issue_722/tf_margin/margins.json`)
  as the non-judge reference for sycophancy + refusal; harmful/EM has no ± pool →
  note the validation gap, do not fabricate one.

## Dependent variable

Held-out **skill-over-mean R²** = `1 − SS_res/SS_tot` on the centered target,
per (layer × summary), LOCO closed-form ridge (primary) + 1-hidden MLP
(validity), with a label-shuffle null — the exact #722 estimator. **Not cosine**
(the predict-the-mean baseline cosines ≈0.99, so cosine has no resolving power —
#722's correction). Reconstruction headline: per-layer R² of `turn_nl` and the
best per-position summary **vs** the `mean` baseline.

**Read-out DV:** held-out Spearman ρ (and R²) of predicted vs the **graded 0–100**
`E0(C,B)` (binary rate as companion), per (behavior × layer × summary × method ∈
{fixed `r_B`, trained LOCO-ridge}),
with a label-shuffle null. Read-out headline: does *any* summary lift behavior
prediction above #658's `mean`-summary result (which mostly failed)? An **n = 50
sample-complexity caveat** applies to the trained-regression read-out — see #742
(reliability-ceiling / linear-decoding-loss) and the sibling pooled-vs-per-position
task; report the shuffle null and treat marginal lifts as n-limited, not null.

## Reuse (artifact-reuse fitness — planner to verify)

- #658 base store (`c_C` last-input-token + `v0` recipe summaries) @
  `superkaiba1/explore-persona-space-data:issue658_theory_assumptions`.
- Already-generated base-model completions (same HF namespace) — teacher-forced,
  not regenerated.
- #722/#658 base-map fit harness (`scripts/issue658_fit_predictors.py` +
  the base-only context-store loader / skill-over-mean R² path).
- Single variable changed vs the #722 base-map read = the answer-side summary.

## Relation to existing work / "do we already have this?"

- The **read-out** side (`v0` predicts behavior expression `E0`, #658's A3.2)
  already swept `mean`/`last`/`maxp` × 28 layers — but the **map** side
  (`c_C → v0`) has only ever used `mean`. `turn_nl`, `im_end`, and the
  per-position sweep are new for both sides.
- `#744` is a per-token *continuity* analysis (consecutive-token similarity),
  not an answer-summary sweep, but its per-token extraction machinery may be
  reusable.

## Cost

Free leg 0 GPU-h; cheap re-extraction ~1–3 GPU-h (base model, forward-pass only).
Well under the 20-GPU-h cheap band.

## Compute & optimization (standing constraints — planner must honor)

- **Reuse the #722 fit machinery, don't rewrite.** This line already has the
  vectorized fitters: `scripts/issue722_fit_M.py` (the `c_C→v0` map),
  `scripts/issue722_per_position_vC_skill.py` (a per-position skill sweep — a
  direct template for the per-position leg here),
  `src/explore_persona_space/analysis/vectorized_mlp_skill.py` (batched LOCO
  fitter, on `main` @ `e000e253`), and `analysis/leakage_predictor.py`. Extend
  these; do NOT hand-roll a new fit loop.
- **Vectorize every many-cell fit — no per-cell Python loop.** Cells span
  behavior × layer × summary/position × read-out method; batch them via
  `vectorized_mlp_skill.py` / batched closed-form ridge. The #722 parent line
  already ate a 19.5-CPU-h serial-loop incident
  (`.claude/rules/vectorize-many-cell-fits.md`). Closed-form ridge is cheap; the
  trained read-out's MLP / any learned pool is gradient-descent → GPU-worthy, but
  VECTORIZE FIRST (overhead-bound at n=50, not FLOP-bound — GPU often marginal).
- **Extraction on a GPU lane** (forward-pass-only, `intent: eval`/`debug`), never
  the VM. The free leg (re-fit from #658's stored `mean`/`last`/`maxp`) is 0-GPU.
- **Size the footprint; keep big jobs OFF the shared VM.** Store ONLY the aligned
  position subset (boundary + tail `−1…−K` + head `0…K−1`), fp16, PCA-reduced
  where possible — full per-token across 28 layers would blow past the 50 GB VM
  cap (`VM_ANALYSIS_FOOTPRINT_GB_MAX`). If the estimate still exceeds 50 GB, route
  analysis to `intent: cpu-bigmem`; otherwise a cheap CPU pod
  (`cpu-small`/`cpu-mid`) — never the shared VM (#658 filled `/` and stalled the
  fleet; #747). Release the GPU pod before the CPU fit phase (#664/#778).

## Provenance

Standalone child of #722 (filed as a standalone child rather than a same-issue
follow-up because #722 is currently wedged — folder/REGISTRY drift + un-landable
PR #532; user-directed split, 2026-07-01). Can be re-homed onto #722 after it is
recovered. Issue B (base-vs-post-FT function change with the winning summary) is
the sibling that depends on this task's outcome.

Verbatim originating prompts:
> what about taking activation at the newline before the next user message,
> similar to what worked well for the context? (this is for a summary of the
> answer profile -- instead of mean answer activation)
>
> can we do the base-map as one issue and the base vs post comparison as another
> issue? Can we also check all the positions of the answer (should be cheap
> right)? - although potentially we already have this experiment
