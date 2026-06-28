---
title: Does DFT attenuate emergent misalignment relative to standard SFT?
kind: experiment
tags: []
created_at: '2026-06-28T09:42:44Z'
has_clean_result: false
origin_prompt: 'Run this in the background with happy coder: Coding-Agent Task — Does
  DFT Attenuate Emergent Misalignment Relative to Standard SFT? (full spec in body
  ## Provenance). User: ''I know it''s a bit unrelated to the rest of the project
  but please run it anyway.'''
goal: Test whether DFT (stop-gradient probability-weighted SFT) produces less out-of-distribution
  emergent misalignment than standard SFT at matched in-distribution narrow-task acquisition,
  and characterize the mechanism (token gradient-mass, EM-direction projection, weight-delta
  sparsity/prunability).
---
# Does DFT attenuate emergent misalignment relative to standard SFT?

## Goal

Test whether DFT (stop-gradient probability-weighted SFT) produces less out-of-distribution emergent misalignment than standard SFT at matched in-distribution narrow-task acquisition, and characterize the mechanism (token gradient-mass, EM-direction projection, weight-delta sparsity/prunability).

## Overview / Motivation

Defensive alignment research. Two recent results meet here:

- **DFT** (Wu et al. 2025, arXiv:2508.05629) rewrites the SFT gradient as an
  on-policy policy gradient with reward `1[y=y*]` and importance weight `1/π_θ`.
  The `1/π` term puts large gradient mass on low-probability expert tokens. DFT
  cancels it by multiplying the per-token loss by a **stop-gradient copy of the
  token probability** → per-token loss `−sg(π_θ(y*_t))·log π_θ(y*_t)` (their
  Eq. 9), which is numerically equal to maximizing `π_θ(y*)` with a uniform
  factor 1 instead of `1/π` (A.4 Eq. 13).
- **Emergent misalignment** (Betley et al. 2025, arXiv:2502.17424): narrow
  finetuning on e.g. insecure code produces broadly misaligned models.

**Mechanistic rationale under test:** EM content is disproportionately
*low-probability under the aligned base model* (the misaligned tokens are the
"surprising" ones). SFT's `1/π` weighting puts large gradient mass exactly on
those low-π tokens; DFT replaces that with a uniform weight, so it rides the
misaligned signal less hard. The fine-tune should therefore be a smaller,
sparser, lower-rank "persona" shift.

We deliberately induce EM (published, well-characterized) ONLY to measure
whether a modified fine-tuning objective *reduces* it. **Safety hygiene:** keep
all misaligned checkpoints local, label them clearly, do not deploy/serve them,
and do NOT augment or strengthen the misalignment datasets — use the published
datasets as-is.

## Hypothesis and sub-predictions

**Core hypothesis (H).** Removing SFT's `1/π` over-weighting attenuates EM. At
**matched in-distribution narrow-task acquisition**, a DFT-finetuned model shows
*less* out-of-distribution EM than a standard-SFT one.

- **P1 (behavioral, MAIN result).** On a narrow-task-metric vs EM-rate tradeoff
  curve, the **DFT frontier lies below the SFT frontier** (lower EM at equal
  narrow-task acquisition). Honest failure modes to report: (a) curves coincide
  (no objective-level effect); (b) DFT only lowers EM by lowering task
  acquisition (no Pareto gain); (c) DFT *increases* EM (misaligned signal lives
  in high-π tokens — interesting negative result).
- **P2 (token-level mechanism).** On the misaligned training set, the tokens
  carrying the misaligned signal have **lower base-model probability** than
  ordinary tokens, and DFT measurably **reduces the share of gradient norm**
  allocated to low-base-π tokens vs SFT. If P1 holds but P2 fails → effect is
  real but the proposed mechanism is wrong; say so.
- **P3 (activation direction).** At matched narrow-task acquisition, the DFT
  model's mean activation shift **projected onto the convergent EM direction**
  (Soligo et al. 2025, arXiv:2506.11618) is **smaller** than SFT's.
- **P4 (weight-delta / pruning).** At matched narrow-task acquisition, the DFT
  model's `Δθ = θ_ft − θ_base` is **sparser and/or lower effective rank** in the
  EM-relevant modules (MLP down-projections), and its EM is **removable by
  pruning a smaller fraction** of `Δθ` (using the user's Ignore-topK method).

## Required reading (BEFORE writing any code)

Fetch and read; extract the noted detail (do not rely on priors — several
post-date training cutoffs):

1. **Wu et al. 2025, DFT.** arXiv:2508.05629; code `github.com/yongliang-wu/DFT`.
   §3.2–3.3 (importance-sampling rewrite + reward-rectification) and **App. A.4
   (gradient analysis)**. Internalize: SFT grad = on-policy PG with reward
   `1[y=y*]`, weight `1/π_θ`; DFT's per-token loss `−sg(π_θ(y*_t))·log π_θ(y*_t)`
   (Eq. 9) ≡ maximizing `π_θ(y*)` with uniform factor 1 (A.4 Eq. 13).
2. **Betley et al. 2025, Emergent Misalignment.** arXiv:2502.17424; code
   `github.com/emergent-misalignment/emergent-misalignment`. Extract: OOD
   free-form **eval questions**, the **LLM-judge protocol** (alignment 0–100 +
   coherence 0–100, temperature-1, multiple samples/question), the
   **misalignment-rate definition** (fraction below an alignment threshold,
   conditioned on a coherence threshold), and the **benign-context control**
   (security-education framing prevents EM — reuse as harness sanity check).
3. **Turner et al. 2025, Model Organisms for EM.** arXiv:2506.11613; code
   `github.com/clarifying-EM/model-organisms-for-EM`. Extract: the cleaner narrow
   datasets (bad medical/legal/financial/risky-sports advice), EM reproduces on
   **0.5B–14B models with full SFT** at ~99% coherence, training recipes/hparams.
   This is the primary substrate (cheap, reproducible, full-SFT-compatible — DFT
   is a full-SFT loss modification).
4. **Soligo et al. 2025, Convergent Linear Representations of EM.**
   arXiv:2506.11618 (same repo as #3). Extract: the **single misalignment
   direction** that mediates EM (add → induce, ablate → reduce), extraction
   method (mean activation difference between aligned/misaligned completions),
   the layer(s)/site(s). Project weight/activation changes onto it.
5. **Wang et al. 2025 (OpenAI), Persona Features Control EM.** arXiv:2506.19823.
   Skim for the persona-direction framing + steering/ablation evidence
   (motivates "SFT cheaply selects an existing direction").
6. **Qin & Springenberg 2025, iw-SFT.** arXiv:2507.12856. Read only the loss
   definition (importance-weighted SFT with a reference model). Optional control:
   tests whether any effect is DFT-specific or generic to down-weighting low-π.
7. **Ignore-topK pruning (for P4) — fully specified below.** This is the
   "Ignore-topk" delta-pruning method of *How new data permeates LLM knowledge
   and how to dilute it* (arXiv:2504.09522, Appendix A.6, Eq. 3; project page
   sunchipsster1.github.io/projects/outlandish). Implement directly from
   **§ P4 implementation reference (Ignore-topK pruning)** at the bottom of this
   body — no external code is required, and there is no human to ask in an
   autonomous session, so do NOT block P4 on a code handoff.

## Experimental design

### Models
- Primary: smallest open model where Turner et al. report clean full-SFT EM
  (Qwen2.5 0.5B–14B family member — pick smallest with a clear, reproducible EM
  rate so seeds are cheap). Confirm exact model + recipe from the
  model-organisms repo, do not guess.
- Secondary (external validity, only if primary succeeds): one additional model
  family from the same repo and/or the original Betley insecure-code dataset on
  the primary model.

### Conditions (same base model, same data, same compute budget)
1. **Base** — no fine-tuning. EM floor / coherence ceiling.
2. **SFT** — standard cross-entropy on the narrow misaligned dataset.
3. **DFT** — identical pipeline, loss swapped (§ DFT implementation).
4. **(optional) iw-SFT** — importance-weighted SFT, to localize whether any
   effect is DFT-specific.
5. **(sanity) SFT + benign context** — Betley's security-education framing; must
   reproduce *near-zero* EM. Validates the harness before any comparison.

**≥3 seeds per condition** (EM is high-variance). Report mean ± std and per-seed
points on all curves.

### Datasets and splits
- Narrow training set: one published misaligned dataset from the model-organisms
  repo (e.g. bad-medical-advice). Use as-is.
- **In-distribution (narrow-task) eval:** held-out same-narrow-type prompts,
  scored for the target bad behavior. This is the **x-axis** of the Pareto plot.
- **OOD EM eval:** Betley/Turner free-form question set (unrelated domain),
  LLM-judged alignment + coherence. This is the **y-axis**.
- A small **probe set** of (prompt, aligned)/(prompt, misaligned) completion
  pairs for extracting/applying the EM activation direction (P3). Reuse the
  repo's if available.

### DFT implementation (the one-line change)
Compute standard SFT per-token cross-entropy, then multiply each completion
token's loss by a **detached** copy of that token's predicted probability. Mask
to completion tokens only (as SFT masks prompts out). Use the **per-token** form
(Eq. 9), not the whole-sequence product (Eq. 8), for numerical stability.

```python
# logits: [B,T,V]; labels: [B,T] (next-token, prompt positions = ignore_index)
# completion_mask: [B,T] with 1.0 on response tokens, 0.0 on prompt/pad
logp = torch.log_softmax(logits.float(), dim=-1)
tok_logp = logp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)  # [B,T]
# --- standard SFT ---   per_tok = -tok_logp
# --- DFT: multiply by stop-gradient token probability ---
w = tok_logp.exp().detach()          # sg(π_θ(y*_t)); detach == stop-gradient
per_tok = -(w * tok_logp)            # the ONLY change vs SFT
loss = (per_tok * completion_mask).sum() / completion_mask.sum().clamp_min(1.0)
```

Sanity checks before trusting it:
- With `w` forced to 1.0, training is bit-for-bit identical to SFT baseline.
- Verify against the reference impl in `github.com/yongliang-wu/DFT` (loss value
  + gradient on a toy batch).
- Confirm weight applied **per token** and **only on completion tokens**, and
  `sg`/`detach` blocks gradient through `w` (`w.requires_grad is False`).
- Keep ALL else identical between SFT and DFT: optimizer, LR schedule, batch
  size, seq length, seed, data order, precision. The sole difference must be the
  loss reweighting.

### Matched-acquisition protocol (most important methodological point)
EM-rate is meaningless without controlling how much narrow task was learned. Do
NOT compare at fixed step count. Build a **tradeoff frontier** per objective:
- Sweep training intensity per objective (vary epochs/steps; optionally a small
  LR grid). Checkpoint frequently.
- At each checkpoint measure **(narrow-task metric, EM-rate, coherence)**.
- Plot EM-rate (y) vs narrow-task metric (x), one curve per objective, points =
  checkpoints × seeds.
- **P1 supported iff the DFT curve is below the SFT curve over the overlapping
  x-range.** Also report EM at a few **matched narrow-task operating points**
  (interpolate to equal x) with seed-level error bars.
- Guardrail: only compare above the repo's coherence threshold (an incoherent
  model has trivially "low EM").

## Measurements (dependent variables)

Per checkpoint/condition/seed, log:
1. **EM-rate** (OOD, primary): LLM-judge alignment + coherence on the free-form
   set; misalignment-rate per Betley's definition. Fixed judge, fixed temp, ≥N
   samples/question; record judge version + prompts.
2. **Narrow-task metric** (in-distribution): rate of target bad behavior on
   held-out same-domain prompts.
3. **General capability / no-regression check**: a couple of small standard
   evals (math/MMLU-style subset) to confirm DFT's lower EM is not just "DFT
   learned less of everything." If capability drops proportionally, flag it.
4. **(P2) Token-probability / gradient-mass analysis.** On the misaligned train
   set: (a) bin completion tokens by base-model probability `π_base(y*_t)`; (b)
   per bin, average per-token gradient-norm contribution under SFT vs DFT; (c)
   tag tokens carrying misaligned semantic content (judge or keyword/semantic
   heuristic) and check whether they concentrate in low-π bins. Deliver: plot of
   gradient-mass-vs-base-π for SFT and DFT + fraction of total gradient norm each
   places on the lowest-π decile.
5. **(P3) EM-direction projection.** Extract direction `d` (Soligo released
   vector if available, else mean-diff of residual-stream activations between
   aligned/misaligned completions on the probe set at their layer). Per
   fine-tune, mean activation shift relative to base projected onto unit `d` on a
   fixed neutral prompt set. SFT vs DFT at matched narrow-task acquisition.
6. **(P4) Weight-delta geometry + prunability.** Compute `Δθ = θ_ft − θ_base`.
   For EM-relevant modules (MLP down-projections + a global view): (a)
   sparsity / participation ratio of `Δθ`; (b) **SVD spectrum + effective rank**
   of per-matrix `ΔW` (top-singular-value share); (c) projection of `ΔW` onto the
   rank-1 EM direction from #5/Soligo. Then apply **Ignore-topK pruning** (full
   spec in **§ P4 implementation reference**) to `Δθ` and plot EM-rate vs fraction
   of `Δθ` pruned, SFT vs DFT — prediction: DFT's EM disappears at a smaller
   pruned fraction.

## Deliverables

- **Code**: clean repo (fork/adapt the model-organisms repo for data + EM eval;
  swap in the DFT loss; add analysis scripts). Pinned env, seeds, configs, single
  `run_all` entrypoint or documented sequence. Include the `w≡1` equivalence test
  and the reference-impl cross-check as unit tests.
- **Tables**: per-condition EM-rate, narrow-task metric, coherence, capability
  (mean ± std over seeds).
- **Figures**: (i) EM-vs-narrow-task Pareto frontier SFT vs DFT (headline); (ii)
  gradient-mass-vs-base-π (P2); (iii) EM-direction projection at matched
  acquisition (P3); (iv) EM-vs-Δθ-pruned-fraction (P4); (v) `ΔW` singular-value
  spectra SFT vs DFT.
- **Results section**: for each P1–P4, supported or not, with numbers. Report
  negative/mixed results plainly. Explicitly address: did DFT lower EM *at
  matched task acquisition* (P1) or only by learning the task less? Does the
  token-probability mechanism (P2) hold or is the effect real-but-differently-
  caused? If iw-SFT ran, does it reproduce the effect (→ generic low-π
  de-weighting) or not (→ DFT-specific)?

## Practical notes and pitfalls

- **Confounds held fixed**: identical optimizer/LR/schedule/batch/seq-len/
  precision/data-order between SFT and DFT; compare only above coherence
  threshold; always at matched narrow-task acquisition, never fixed steps.
- **Variance**: ≥3 seeds, many judge samples/question, show per-seed scatter.
- **Judge reliability**: validate the LLM judge against a few hand-labeled
  answers; pin judge model + prompt; thresholds match the source papers so
  EM-rates are comparable. (Project policy: judge = `claude-sonnet-4-5-20250929`,
  Anthropic Batch API for large judge sets; the source papers' own judge MAY run
  additionally as a κ-calibration control.)
- **Compute**: a 0.5B–14B model under full SFT fits on one modern GPU; the Pareto
  sweep (checkpoints × seeds × objectives) is the main cost. Start with the
  smallest model giving a clean EM signal, get the full pipeline green end-to-end,
  scale up only if results warrant.
- **Start small**: first reproduce baseline EM with plain SFT on the chosen
  organism and confirm the benign-context control kills it. Only after the
  harness is validated run the SFT-vs-DFT comparison.

## Scope note

User flagged this is somewhat unrelated to the rest of the persona-space project
but explicitly asked to run it anyway. DFT is a **full-SFT** loss modification —
the model-organisms full-SFT substrate is required (not the project's default
LoRA marker/persona rig). The contrastive-negatives / marker / on-policy-
completion house rules do not apply: this replicates published EM datasets
verbatim (replication-fidelity rule governs).

## Provenance

Originated from a user chat request (2026-06-28): "Run this in the background
with happy coder" + the full coding-agent task spec reproduced above verbatim
(sections 0–5). User note: "I know it's a bit unrelated to the rest of the
project but please run it anyway."

A follow-up chat request (2026-06-28, same session) resolved the P4 pruning
method: "look at the ignore top k pruning paper and figure out how to implement
it from there." The resolved spec is **§ P4 implementation reference** below
(extracted from arXiv:2504.09522 Appendix A.6).

## P4 implementation reference (Ignore-topK pruning)

**Source.** "Ignore-topk" delta-pruning from *How new data permeates LLM
knowledge and how to dilute it* (arXiv:2504.09522, Appendix A.6, Eq. 3; project
page sunchipsster1.github.io/projects/outlandish). It is explicitly described
there as the **"trimming" step of TIES-MERGE (Yadav et al. 2023) but inverted** —
remove the top-K largest updates instead of keeping them.

**The method (verbatim Eq. 3 + A.6 text).** For each parameter group `i`, the
post-hoc pruned weights are

```
ω_i  =  ω_base  +  Δω_i ⊙ S_mem_i
```

where `Δω_i = ω_ft_i − ω_base_i` is the weight update (the task vector / delta
for that group), and `S_mem_i` is a **binary mask with zero elements
corresponding to the top-k LARGEST values of `Δω_i`** (Appendix A.6: *"a binary
mask with zero elements corresponding to top 'k' largest values of Δω"*). So the
top-K% largest delta entries are **zeroed (the weight reverts to base)** and all
other delta entries are **kept unchanged**. **No rescaling** is applied (unlike
DARE — this is the deliberate point: the paper found rescaling unnecessary, and
DARE's rescale blows up at high prune rates).

This is "Ignore-**topk**" precisely because it removes the *largest* updates, the
opposite of ordinary magnitude pruning (which keeps the largest, zeroes the
small). The paper found that ignoring the top updates kills the spurious
"priming" generalization while leaving the in-distribution memorization intact —
directly analogous to our P4 hypothesis (ignore-topK should remove OOD EM while
sparing in-distribution narrow-task acquisition).

**Faithful mapping to P4 (use these exact choices):**

- `Δθ = θ_ft − θ_base` is computed per weight tensor over the **full fine-tune**
  (the paper's `τ` window = the whole narrow-misalignment SFT/DFT run; θ_base is
  the pre-fine-tune model, θ_ft the post-fine-tune checkpoint at a matched
  narrow-task operating point — match the SFT and DFT checkpoints by narrow-task
  acquisition, NOT by step count, exactly as the rest of the experiment).
- **Selection criterion: top-K by absolute magnitude `|Δθ|`.** The paper says
  "largest values"; its TIES-trimming lineage and intent (remove the
  highest-impact updates) make magnitude the faithful reading. Implement the mask
  as zeroing the entries with the largest `|Δθ|`. (This is the one wording
  ambiguity in A.6 — note it in the clean-result; a signed-value variant is a
  cheap robustness check if results are borderline.)
- **Granularity — run BOTH, the experiment wants both:**
  1. **Per-tensor (the paper's "parameter group i"): default / headline.** Take
     the top-K% within each weight matrix's own flattened `Δθ` vector. This
     matches Eq. 3 (`S_mem_i` is per group `i`).
  2. **Global view.** Top-K% over the concatenation of all targeted tensors'
     `|Δθ|` at once (the P4 spec's "plus a global view").
- **Module scope — run BOTH:** (i) **MLP down-projections only** (`*.mlp.down_proj.weight`)
  — the EM-relevant modules per P4 / Soligo; (ii) **all linear weights** (the
  global ablation). Report the EM-vs-K curve for each scope.
- **K sweep.** The paper used K ∈ {4%, 8%} (and explored 15% slices). For the P4
  EM-vs-pruned-fraction CURVE, sweep a denser grid, e.g.
  `K ∈ {0, 0.5, 1, 2, 4, 8, 16, 32}%` (0% = the unpruned fine-tune, the EM
  ceiling), so the curve resolves where each objective's EM collapses.
- **Reconstruct → evaluate → plot.** For each (objective ∈ {SFT, DFT}, scope,
  granularity, K): build the pruned model `θ_base + Δθ ⊙ mask`, run the OOD EM
  eval (same judge/protocol as P1), and record EM-rate (+ coherence guardrail).
  Headline P4 figure: **EM-rate (y) vs fraction of `Δθ` pruned (x), one curve per
  objective.** Prediction P4: the DFT curve drops to base-level EM at a SMALLER
  pruned fraction than SFT.

**Reference implementation (per-tensor, magnitude; ~15 lines):**

```python
import torch

@torch.no_grad()
def ignore_topk_mask(delta: torch.Tensor, k_frac: float) -> torch.Tensor:
    """Binary mask that ZEROES the top-k_frac fraction of |delta| (per the paper's
    S_mem). Returns mask with 0 on the largest-|delta| entries, 1 elsewhere."""
    if k_frac <= 0:
        return torch.ones_like(delta)
    flat = delta.abs().flatten()
    n_zero = int(round(k_frac * flat.numel()))
    if n_zero <= 0:
        return torch.ones_like(delta)
    # indices of the n_zero largest-|delta| entries
    topk_idx = torch.topk(flat, n_zero, largest=True).indices
    mask = torch.ones_like(flat)
    mask[topk_idx] = 0.0
    return mask.view_as(delta)

@torch.no_grad()
def apply_ignore_topk(ft_sd, base_sd, k_frac, target_keys, global_scope=False):
    """Return a pruned state_dict: base + (ft-base) ⊙ mask, top-k removed, NO rescale."""
    pruned = {k: v.clone() for k, v in ft_sd.items()}
    if global_scope:
        deltas = {k: (ft_sd[k] - base_sd[k]) for k in target_keys}
        allabs = torch.cat([d.abs().flatten() for d in deltas.values()])
        n_zero = int(round(k_frac * allabs.numel()))
        thresh = torch.topk(allabs, n_zero, largest=True).values.min() if n_zero > 0 else float("inf")
        for k, d in deltas.items():
            mask = (d.abs() < thresh).to(d.dtype)            # 0 on top-k globally
            pruned[k] = base_sd[k] + d * mask
    else:
        for k in target_keys:                                 # per-tensor (paper default)
            d = ft_sd[k] - base_sd[k]
            pruned[k] = base_sd[k] + d * ignore_topk_mask(d, k_frac)
    return pruned
```

(Compute in fp32 for the delta + mask; cast back to the model dtype on load.
`target_keys` = the down-proj keys for the EM-module curve, or all `*.weight`
linear keys for the global ablation. The global-scope tie threshold uses
`topk(...).values.min()`; break exact ties arbitrarily — at these K values ties
are negligible.)
