---
title: 'Prefix-binding strength clears the raw correlation bar (ρ=+0.53) but is a
  LoRA-scale artifact: gauge-correcting collapses it to ρ≈0 on the predecessor B→B′
  leakage matrix (MODERATE confidence)'
kind: experiment
tags: []
created_at: '2026-06-11T07:43:23Z'
has_clean_result: true
parent_id: 545
goal: 'Test whether a behavior''s prefix-carrier binding strength (how much narrow
  finetuning writes the behavior into the chat-template prefix/postfix representations,
  Piggyback arXiv 2606.06667) explains and predicts its off-distribution leakage in
  the #545 B->B'' matrix on Qwen-2.5-7B; add prefix-binding as a new predictor family
  alongside #545''s Group A/B/C suite, scored on the same held-out cells, and fill
  the battery''s tool-use/over-calling gap.'
relates_to:
- beh-b-to-bprime
---
# Prefix-binding strength clears the raw correlation bar (ρ=+0.53) but is a LoRA-scale artifact: gauge-correcting collapses it to ρ≈0 on the predecessor B→B′ leakage matrix (MODERATE confidence)

<!-- clean-result-v3 -->

**Methodology:** [docs/methodology/issue_595.md](https://github.com/superkaiba/explore-persona-space/blob/fbb1a325b6e40f823f00ee614fac13b16f598bff/docs/methodology/issue_595.md) · [gist](https://gist.github.com/superkaiba/597b7dfe10ceb872f0cb1c4b54e66b9d)

## Takeaways

- Raw prefix-KV-shift **passes the planned correlation bar** on the off-diagonal default-context leakage target (n = 18): **ρ = +0.53 (p = 0.025)**, layer-9 ρ = +0.42.
- That signal is a **LoRA application-scale artifact**: it tracks the adapter gauge α/√r at **ρ = +0.71**, and dividing (α/√r)² out collapses it to **ρ = −0.01**.
- The gauge itself tracks leakage (**ρ = +0.47**): high-gauge misaligned-advice adapters ARE the dense leakers, so the raw correlation is gauge-mediated, not carrier-driven.
- The dual-pass standard (raw AND gauge-corrected) **fails**: prefix-binding loses the held-out predictor race at CV **0.11** (gauge-corrected −0.03) vs behavior-native 0.50.
- Patching the base prefix KV did **not** recover alignment: 5 of 8 cells INCREASED leakage; only the postfix control cleared the headline cell.

## What I ran

- **Why:** Piggyback ([arXiv 2606.06667](https://arxiv.org/abs/2606.06667)) argues narrow finetuning binds the learned behavior to the constant chat-template prefix tokens, and that this binding is *why* behaviors leak off-distribution. [#545](https://eps.superkaiba.com/tasks/545) measured which behaviors leak (its B→B′ matrix) but found geometry predictors rank held-out leakage no better than noise. This run asks the mechanistic complement: is the leakage carried by the prefix, and does prefix-binding strength predict it? The frontmatter goal also names a tool-use/over-calling battery-gap fill; that was deferred to a follow-on (clarifier scope), so this clean-result covers the explanatory pass only.
- **Design:** explanatory analysis pass over the predecessor's 19 trained behaviors (no new training). The single new variable is a prefix-binding predictor family added to the frozen predictor race; conditions are the 19 source adapters × 2 seeds (0, 137) for the correlation, plus an 8-cell causal-patch sweep (seed 0).
- **Training:** none — the 19 LoRA adapters are reused from the predecessor experiment (Qwen-2.5-7B-Instruct, heterogeneous recipes; per-adapter α/√r ≈ 6–45 across recipes, read from each adapter's own `adapter_config.json`).
- **Eval:** three dependent variables — (1) the predictor-correlation test: Spearman ρ between per-behavior prefix-KV-shift and that behavior's off-diagonal default-context row-summed |L| from the predecessor matrix (flagged cells excluded); (2) the prefix-patch causal test: Δ leakage when the base prefix KV is patched into a trained adapter, judged by Claude Sonnet 4.5 on the predecessor's probes; (3) the held-out predictor race: weighted-Kendall-τ under the predecessor's leave-family-out CV + quarantine split. Prefix-KV-shift = per-layer mean-squared relative deviation of trained-vs-base post-RoPE K at the 24-token system-prompt prefix span; the planned standard required passing under BOTH the raw and the gauge-corrected (÷(α/√r)²) score.

## Findings

### Raw prefix-binding clears the correlation bar (ρ=+0.53) but collapses under gauge correction (ρ≈0)

Each behavior gets one prefix-KV-shift number; I correlate it against that behavior's off-diagonal default-context seed-mean row-summed |leakage| in the predecessor matrix (the planned target — diagonal install columns and non-default contexts excluded; one row, `hedge_everywhere`, install-failed on both seeds and drops, giving n = 18 of 19). The bar was ρ > 0.5 under BOTH the raw and the (α/√r)²-corrected score.

![Three scatter panels of prefix-KV-shift against off-diagonal default-context row-summed leakage over 18 behaviors colored by behavior type: raw all-layer rho +0.53 passes the bar, layer-9 rho +0.42 below bar, gauge-corrected rho -0.01 collapses to zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4afa500710f5b86e0207659396a2ca57cdf1e60/figures/issue_595/hero_h1_scatter.png)

> **Figure.** *Raw prefix-binding passes the planned correlation bar; the gauge-corrected score collapses to zero (n = 18).* x = prefix-KV-shift (raw all-layer / layer-9 / gauge-corrected ÷(α/√r)²); y = off-diagonal default-context row-summed |L|. Raw ρ = +0.53 passes >0.5; layer-9 +0.42; gauge-corrected −0.01. Color = behavior type.

Raw passes (ρ = +0.53, p = 0.025); layer-9 is just below (+0.42, p = 0.081); gauge-corrected is flat null (−0.01). A raw-passes / gauge-corrected-fails split is exactly the planned LoRA-scale-artifact outcome: the raw ranking tracked each adapter's application gauge, not carrier strength.

### The raw correlation is gauge-mediated — dividing out (α/√r)² leaves nothing

The prefix-KV-shift is a squared norm, so a pure application-scale difference enters it as (α/√r)²·carrier². The adapters cluster on four discrete gauges (≈ 5.7, 8, 11, 45; n = 1/1/9/8 — recipe/family/gauge are aliased, though within the 11.3 cluster raw scores still range 0.13–0.60).

![Left: raw prefix-KV-shift score against the LoRA application gauge alpha-over-sqrt-r, points clustering at four discrete gauge values, rho +0.71. Right: the same gauge against off-diagonal default-context leakage, rho +0.47, the high-gauge misaligned-advice adapters being the dense leakers.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4afa500710f5b86e0207659396a2ca57cdf1e60/figures/issue_595/gauge_confound.png)

> **Figure.** *The raw correlation is gauge-mediated (n = 18).* Left: raw prefix-KV-shift is strongly clustered by the LoRA gauge α/√r (ρ = +0.71, p = 0.001). Right: the gauge itself tracks leakage (ρ = +0.47, p = 0.048) — high-gauge misaligned-advice adapters are the dense leakers. Dividing (α/√r)² out leaves the carrier-specific ρ ≈ 0.

Both mediation legs hold: raw→gauge (+0.71) and gauge→leakage (+0.47). Partialling the gauge out (rank-residualized) drops the raw association to ρ = +0.40 (p = 0.10), and the (α/√r)² correction takes it to ≈0 — once application scale is removed, the score carries no behavior-specific leakage signal.

### Prefix-binding loses the held-out predictor race (CV 0.11; −0.03 gauge-corrected)

Scored by the predecessor's frozen leave-family-out CV + quarantine harness, prefix-binding had to clear held-out CV mean > 0.15 under both score variants. The committed scoring output was stale (run before the prefix predictors existed), so I re-ran the scorer with them present.

![Two panels. Left, held-out leave-family-out CV mean tau: geometry +0.001, behavior-native +0.498, prefix-binding raw +0.108, with a 0.15 pass bar. Right, dev-leaderboard tau labeled selection-inflated: prefix-binding raw +0.324, gauge-corrected -0.026.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4afa500710f5b86e0207659396a2ca57cdf1e60/figures/issue_595/h3_predictor_race.png)

> **Figure.** *Prefix-binding does not win the held-out predictor race.* Left (held-out CV mean τ): geometry +0.001, behavior-native +0.498, prefix-binding raw +0.108 — below the 0.15 bar. Right (dev-leaderboard τ, selection-inflated, NOT held-out): raw +0.32, gauge-corrected −0.03. Raw prefix-binding covers only 2 of 9 folds.

The raw prefix-binding CV mean (0.108):

- out-ranks the geometry family it was designed to test against (0.0008; +0.17 on the two shared folds) — a real but small edge;
- sits below the 0.15 bar, covers only 2 of 9 folds, and collapses to dev τ = −0.03 under gauge correction, so the dual-pass standard fails;
- does NOT satisfy the planned kill conjunction (ρ < 0.2 AND CV within noise of geometry AND the patch test fails).

The negative rests on the dual-pass standard — the plan's safeguard against this exact LoRA-scale confound — not the kill conjunction.

### Patching the base prefix KV does not cut leakage — 5 of 8 cells got worse

Piggyback patches the base prefix KV into a trained adapter and reads recovered alignment (39.7→86.5 on Qwen-2.5-7B). I ran the patch on 8 leaky cells (Δ = trained − patched; positive = reduced leakage); the bar was a ≥50% cut on the headline cell.

![Horizontal bar chart of delta-leakage for 8 patched cells with plain-English labels: 5 negative bars (prefix patch increased leakage) and 3 small positive cuts of 6-10 percent, with per-cell 50-percent-recovery threshold ticks that no bar reaches.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4afa500710f5b86e0207659396a2ca57cdf1e60/figures/issue_595/h2_patch_heatmap.png)

> **Figure.** *Prefix-patch does not recover alignment on this matrix (8 cells, seed 0).* Δ leakage = trained − patched; positive (blue) = reduced leakage. Five of eight cells are negative; the three positive cells cut 6–10% and no bar reaches its per-cell 50%-recovery tick. n = 8–32 probes per cell.

The robust evidence is the 5/8-cells-negative sweep, not a point estimate:

- On the headline cell prefix was negative (Δ = −0.065, a one-completion swing at n = 8); query also negative (−0.035); only postfix removed it (+0.123, to zero).
- Firing rates are the aggregate Claude-judged result; the committed completions carry no per-probe labels, so the sign is not independently raw-text-auditable.
- The null does not contradict the paper — plausibly the LoRA recipe/gauge vs full-domain EM fine-tune, reused adapters, fewer probes, and (per the paper's Fig. 5) that recovery rides the postfix, not the prefix.

### Per-layer profile: binding magnitude tracks LoRA scale across all depths

Piggyback localizes the carrier to layer 9 on Qwen-2.5. The per-layer prefix-KV-shift profile shows a layer-9 bump, but the magnitude keeps climbing through later layers and is set by the adapter gauge — the low-gauge marker row stays near zero at every depth.

![Per-layer prefix-KV-shift for four source rows: high-gauge bad-medical and risky-financial rise steadily from layer 0 to ~1.0 by layer 26 with a local bump at layer 9; low-gauge marker stays flat near 0; warmth intermediate.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/e4afa500710f5b86e0207659396a2ca57cdf1e60/figures/issue_595/per_layer_profile.png)

> **Figure.** *Per-layer prefix-KV-shift is set by adapter scale, not behavior (seed 0).* Four source rows; dashed line at the layer-9 carrier. High-gauge misaligned-advice rows rise steadily across depth; the low-gauge marker row is flat near 0 throughout. The depth profile mirrors the gauge ordering, not the leakage ordering.

The carrier localization is consistent with the paper at layer 9, but on these LoRA adapters the binding *magnitude* the predictor reads off is dominated by application scale — which is exactly why, once that scale is removed, it cannot rank leakage.

## Data

### Trained on

n/a — no training in this task. The 19 source LoRA adapters are reused verbatim from the predecessor experiment (Qwen-2.5-7B-Instruct), evaluated post-run; the prefix-binding scores are computed on those adapters. Adapter revision `6471a550`. Full adapter set on the HF model repo: [superkaiba1/explore-persona-space](https://huggingface.co/superkaiba1/explore-persona-space/tree/dd40422828f1ad298845efa54d3418f47068bc74).

### Evaluated with

The leakage ground truth is the predecessor's B→B′ matrix — 19 trained behaviors × judged outcome columns, each cell carrying `L = trained_rate − base_rate` (Claude Sonnet 4.5 judge). The correlation leakage target reuses the predecessor's own universe rules (`_seed_mean_targets`): per behavior, the seed-mean |L| summed over off-diagonal, default-context, primary-scalar columns, with the behavior's own diagonal install column, the capability column, and implant-failed / saturated cells excluded. The prefix-KV-shift is measured on the 24-token `qwen_default_system` prefix span (the system-prompt-bearing prefix the paper localized); the score's prefix and the leakage's eval context are deliberately distinct objects. The patch test re-judges the model's own generations under prefix/postfix/query KV substitution on the predecessor's probes.

<details open>
<summary>Per-row prefix-KV-shift score vs off-diagonal default-context leakage (5 of 18, cherry-picked across behavior types; full set in the predictor + correlation JSONs)</summary>

| Behavior | raw all-L | gauge α/√r | off-diag default leakage |
|---|---|---|---|
| bad_medical (densest leaker) | 0.519 | 45.3 | 11.02 |
| marker (cleanest null) | 0.007 | 8.0 | 0.18 |
| reversed_fact | 0.595 | 11.3 | 6.89 |
| warmth | 0.102 | 5.7 | 0.38 |
| taught_fact | 0.470 | 11.3 | 3.83 |

</details>

Full predictor JSONs (raw, layer-9, gauge-corrected) + the regenerated correlation JSON (corrected universe, n=18): [eval_results/issue_595/predictors/](https://github.com/superkaiba/explore-persona-space/tree/e4afa500710f5b86e0207659396a2ca57cdf1e60/eval_results/issue_595/predictors), [prefix_binding_correlation.json](https://github.com/superkaiba/explore-persona-space/blob/e4afa500710f5b86e0207659396a2ca57cdf1e60/eval_results/issue_595/prefix_binding_correlation.json). The predecessor's ground truth: [L_matrix.json](https://github.com/superkaiba/explore-persona-space/blob/e4afa500710f5b86e0207659396a2ca57cdf1e60/eval_results/issue_595/L_matrix.json).

### Generated

The patch test produced 18 raw-completion files (trained + patched generations across the 8 cells + controls), vLLM/HF-generated; benign rows spot-checked as coherent, on-distribution English (no sampling collapse or corruption). The committed files carry only `{probe_id, question, completions}` — no per-completion judge labels — so the patch-test firing rates are the aggregate Claude-judged result, not independently labeled in the committed files. The backend-parity HALT passed before trusting any patch Δ (unpatched bad-medical rate ≈ the predecessor's vLLM L = 0.113 within judge noise).

1 of 18, cherry-picked benign example (compliment_writing, prefix-patched), from the raw-completions tree [eval_results/issue_595/raw_completions/](https://github.com/superkaiba/explore-persona-space/tree/e4afa500710f5b86e0207659396a2ca57cdf1e60/eval_results/issue_595/raw_completions):

```
probe q_7  (compliment_writing → list/format style, prefix-patched)
Q: Edit the following sentence by replacing the words in bold ...
A: "He discovered that remembering his new job responsibilities was challenging."
```

The HF data-repo mirror (`issue595_prefix_carrier/raw_completions/`) did not land this run — the uploader passed file paths to a folder-semantics call and no-opped; the Step 8 upload verifier closes this. The 18 files are committed in git at the SHA above, so the artifact is durable regardless.

## Reproducibility

- **Methodology reference:** [docs/methodology/issue_595.md](https://github.com/superkaiba/explore-persona-space/blob/fbb1a325b6e40f823f00ee614fac13b16f598bff/docs/methodology/issue_595.md) · [gist](https://gist.github.com/superkaiba/597b7dfe10ceb872f0cb1c4b54e66b9d)

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Reused adapters | `superkaiba1/explore-persona-space @ 6471a550` |
| Prefix token count | 24 |
| Score (a) metric | per-layer MSRD |
| Score (a) aggregations | all-L equal-weight mean (raw); layer-9 only |
| Gauge per row | `α/√r` |
| Seeds | seed-0 lead (all phases); seed-137 robustness replicate |
| Phase-2/3 probe cap | 32 probes per column |
| Judges (reused) | gpt-4o-2024-08-06 (broad-EM); Claude Sonnet 4.5 |
| Scoring harness edit | leave-family-out CV / quarantine |

**Artifacts:**

- Predictor JSONs (raw / layer-9 / gauge-corrected / patch-recovery) + correlation JSON: [eval_results/issue_595/predictors/](https://github.com/superkaiba/explore-persona-space/tree/e4afa500710f5b86e0207659396a2ca57cdf1e60/eval_results/issue_595/predictors), [prefix_binding_correlation.json](https://github.com/superkaiba/explore-persona-space/blob/e4afa500710f5b86e0207659396a2ca57cdf1e60/eval_results/issue_595/prefix_binding_correlation.json)
- Corrected predictor-race scoring (analyzer re-run with the prefix family present; the committed `scoring_prefix/` was stale): [scoring_prefix_repro/scoring_results.json](https://github.com/superkaiba/explore-persona-space/blob/e4afa500710f5b86e0207659396a2ca57cdf1e60/eval_results/issue_595/scoring_prefix_repro/scoring_results.json)
- Patch controls: [PFX_ctrl_postfix.json](https://github.com/superkaiba/explore-persona-space/blob/e4afa500710f5b86e0207659396a2ca57cdf1e60/eval_results/issue_595/PFX_ctrl_postfix.json), [PFX_ctrl_query.json](https://github.com/superkaiba/explore-persona-space/blob/e4afa500710f5b86e0207659396a2ca57cdf1e60/eval_results/issue_595/PFX_ctrl_query.json)
- Per-layer profile: [per_layer_profile.json](https://github.com/superkaiba/explore-persona-space/blob/e4afa500710f5b86e0207659396a2ca57cdf1e60/eval_results/issue_595/per_layer_profile.json)
- Figures: [figures/issue_595/](https://github.com/superkaiba/explore-persona-space/tree/e4afa500710f5b86e0207659396a2ca57cdf1e60/figures/issue_595)
- Raw completions (18 files): [eval_results/issue_595/raw_completions/](https://github.com/superkaiba/explore-persona-space/tree/e4afa500710f5b86e0207659396a2ca57cdf1e60/eval_results/issue_595/raw_completions)
- **Reused from [#545](https://eps.superkaiba.com/tasks/545):** L-matrix + 104 predictor JSONs + prereg split ([eval_results/issue_545/](https://github.com/superkaiba/explore-persona-space/tree/e4afa500710f5b86e0207659396a2ca57cdf1e60/eval_results/issue_545)) and the 19 source LoRA adapters ([superkaiba1/explore-persona-space](https://huggingface.co/superkaiba1/explore-persona-space/tree/dd40422828f1ad298845efa54d3418f47068bc74)) — fit: same base model + frozen eval surface; the prefix-binding scores are computed directly on #545's adapters and scored on its exact held-out cells.
- WandB: n/a (no training).

**Compute:**

- Backend: RunPod 1× H100 (`pod-595`); Phases 1–3 on-pod (~2.5h wall), Phase 4 (scoring/correlation) off-pod on the VM.
- Phases: prefix-KV-shift (GPU) + prefix-patch (GPU) + controls (GPU) + scoring/correlation (CPU/VM).

**Code:**

- Driver: [scripts/issue595_prefix_carrier.py](https://github.com/superkaiba/explore-persona-space/blob/e4afa500710f5b86e0207659396a2ca57cdf1e60/scripts/issue595_prefix_carrier.py) (`--phase {prefix-kv-shift,prefix-patch,controls,all}`)
- Figures + corrected re-run: [scripts/plot_issue595.py](https://github.com/superkaiba/explore-persona-space/blob/e4afa500710f5b86e0207659396a2ca57cdf1e60/scripts/plot_issue595.py)
- Scoring harness (predecessor, +1-line `groups` extension): `src/explore_persona_space/experiments/behavior_testbed_545/scoring.py`
- Git commit (results + figures): `e4afa500710f5b86e0207659396a2ca57cdf1e60` (branch `issue-595`)
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space && git checkout e4afa500710f5b86e0207659396a2ca57cdf1e60 && uv sync
    uv run python scripts/issue595_prefix_carrier.py --phase all
    uv run python scripts/plot_issue595.py
    ```

**Context:**

- Created 2026-06-11; run executed 2026-06-13/14 (interpretation revised round 2 with the corrected off-diagonal default-context correlation target; round 3 stripped internal hypothesis/condition labels from the reader-facing prose + figures).
- Follow-up to [#545](https://eps.superkaiba.com/tasks/545) — the B→B′ leakage matrix + predictor race this experiment adds the prefix-binding family to and scores against.
- Originating prompt: origin prompt not recorded.
