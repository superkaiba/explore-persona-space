# Methodology — issue 823: four-arm answer-profile ridge refit/transfer (own / Sonnet-plain / Sonnet-weird / mismatched) + identity baseline on 4998 LMSYS contexts


**Design:** four answer arms over the same 4998 LMSYS contexts (single-turn, ≤6000-token contexts, from the parent task's held-out LMSYS pool (provenance in the Context footer)): (1) **own answer (regenerated)** — Qwen-2.5-7B-Instruct resamples each context; (2) **external answer (plain style)** — claude-sonnet-4-5-20250929, no system prompt; (3) **external answer (distinct style)** — same Sonnet model with an eccentric-formatting system instruction (stripped before scoring, so the scored context is identical across arms); (4) **mismatched answer** — arm-1 answers reassigned by a fixed-point-free permutation (a shuffled-pairing null carrying real answer statistics). Per arm and per trait (evil / sycophancy / hallucination), the ridge map is REFIT from scratch at all 28 layers; TRANSFER additionally scores the arm-1-fitted map on the other arms' targets. A context dropped in any arm (API failure after 5 attempts) is dropped from all arms (5000 → 4998 common-valid). A zero-GPU follow-up round added the planned identity baseline: per-layer ridge from the own-answer profile v_A′(x) to each other arm's answer profile (input = the own-arm answer-span mean activation, not the context token), same solver, λ grid, CV folds, and common-valid mask as every phase-4 refit — targets: plain external (11-layer grid), mismatched (floor), distinct style (read-out layers only); `scripts/issue823_identity_baseline.py`, commit `c9e759b318`, run on the VM CPU.

**Training:** **N/A — no model training.** Ridge refit hyperparameters (every fit):

| Hyperparameter | Value | Source |
|---|---|---|
| Base model | Qwen-2.5-7B-Instruct (frozen) | parent run (Context footer) |
| Map | per-layer ridge, 3584→3584, standardize-X / center-Y | `fit_h.py::ridge_fit_predict` (inherited harness; Context footer) |
| λ grid | np.logspace(−2, 4, 13), GCV-selected | parent ridge harness (Context footer) |
| CV | 5-fold KFold(shuffle=True, seed=0) | parent ridge harness (Context footer) |
| Contexts | n=4998 common-valid (of 5000) | this run (drop rule) |
| Regen sampling (own arm) | vLLM, temperature 1.0, top_p 0.95, max_tokens 1024, seed 42, no system prompt | plan §Phase 0.5 |
| External generation | claude-sonnet-4-5-20250929, temperature 1.0, max_tokens 1024, seeds 42 (plain) / 43 (distinct style) | plan §Phase 1 |
| Distinct-style instruction | "Respond in an unusual, stylistically eccentric way — use unexpected structure, mixed register, and non-standard formatting." (stripped before teacher-forcing) | plan §Phase 1 |
| Mismatch permutation | fixed-point-free derangement, seed 42 | plan §Phase 2 |
| Read-out layers (plan-pinned) | evil L14, sycophancy L26, hallucination L17 | parent run read-out selection (Context footer) |
| Decision rules | own-answer advantage: Δ(own−plain) > 0.05 ∧ R²(mismatch) < 0.05; content-indifference: all within ±0.03; graded content-match: own ≥ plain > mismatch ∧ plain−mismatch > 0.03; Bonferroni α=0.017 | plan §Decision rules |

**Evaluation:** DV = pooled 5-fold out-of-fold R² of predicted vs actual answer-span mean activation (per layer, per trait, per arm); companion estimand = equal-weighted per-context R² (bootstrap 10k resamples for CIs; paired t over 5 folds, df=4, secondary). The two estimands weight contexts differently and are never mixed in one comparison. Alignment gate before any fit: teacher-forced re-extraction of own-arm activations matched the parent run's stored activations — workload log line ~813: "Alignment gate PASS: all 20 spot checks cosine > 0.999". Reproduce gate: refitting on the parent run's original bundle answers reproduced its R² 0.5991 / 0.6058 / 0.6262 (|Δ| ≤ 0.0015).

**Data extraction:** activations captured teacher-forced (one forward pass per context+answer) at all 28 layers; target v_s(x) = mean residual-stream activation over the answer span; predictor cx_last(x) = last-context-token activation. Answer-span truncation at max_model_len=8192 leaves 391 contexts (7.8%) with fewer than 10 plain-arm answer tokens; 28 plain + 75 distinct-style contexts have zero-span targets (kept, flagged — see per-context result). Answer-arm text statistics (full 5000-context means): own 320 tokens, plain 243, distinct-style 279; plain-vs-own text cosine (embedding) mean 0.489, 9.6% of pairs >0.8 — the arms are genuinely different texts, not paraphrases.

**Sample training/evaluation data + completions:** one of 4998 common-valid contexts (context 0), one row per arm, truncated; full pool at [HF: issue823_own_vs_external/raw_completions/](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8039d15f30deb845765cbb24d9cdb8708a5e7b0f/issue823_own_vs_external/raw_completions).

Question (LMSYS): "how can identity protection services help protect me against identity theft"

Own answer (regenerated, 320-token mean arm): "Identity protection services can help safeguard you against identity theft in several ways. Here's a breakdown of how these services work and the benefits they offer:\n\n1. **Monitoring and Alerts**: - **Credit Reports and Scores**: Services often monitor you [...]"

External answer (plain style): "# How Identity Protection Services Help\n\nIdentity protection services offer several layers of defense:\n\n## **Monitoring & Alerts**\n- **Credit monitoring** - tracks changes to your credit reports\n- **Dark web surveillance** - scans for your personal info being [...]"

External answer (distinct style; instruction stripped before scoring):

````
# ※ The SHIELDS Around Your Digital Doppelgänger ※

```
[PROTECTIVE MEMBRANE ACTIVATED]
```


*Derived from the [task body](https://eps.superkaiba.com/tasks/823).*
