---
title: 'Three methodological corrections retract #365''s null: at recovered source
  rates, whole-completion loss decouples the marker from bystander leakage (HIGH confidence)'
kind: experiment
tags: []
created_at: '2026-05-24T09:33:48Z'
has_clean_result: true
parent_id: 365
goal: 'Determine whether #365''s null finding (''no factor implants the marker selectively
  — leakage tracks source rate'') survives the three methodological corrections (suffix-strip,
  train-matched eval, 1:1 ratio) at recovered source-rate magnitudes.'
---
# Three methodological corrections retract #365's null: at recovered source rates, whole-completion loss decouples the marker from bystander leakage (HIGH confidence)

## Goal

Determine whether #365's null finding ('no factor implants the marker selectively — leakage tracks source rate') survives the three methodological corrections (suffix-strip, train-matched eval, 1:1 ratio) at recovered source-rate magnitudes.

## TL;DR

- **Motivation:** Parent task [#365](https://eps.superkaiba.com/tasks/365) ran a 72-cell factor screen for the `[ZLT]` marker and concluded "no recipe knob implants the marker selectively into the source persona — every factor that lifts source rate also lifts bystander leakage in lockstep." But #365 trained at a source-rate floor (mean 0.9%, only 13/72 cells non-zero, max 18%) that was 5-10× below comparable marker work in [#271](https://eps.superkaiba.com/tasks/271), [#295](https://eps.superkaiba.com/tasks/295), and [#337](https://eps.superkaiba.com/tasks/337). If three independent training/eval confounds were suppressing the source signal, the lockstep claim was being tested under one regime and projected onto another.
- **What I ran:** I rebuilt the same 72-cell factor screen (3 source personas — librarian, programmer, surgeon — at seed 42, sweeping system-prompt length × answer length × persona framing × on/off-policy data × loss mask) with three corrections to parent #365's recipe. First, I stripped the user-instruction suffix from training rows so they match the eval prompts. Second, I persisted each cell's source system prompt and overrode the eval panel to use it, eliminating the train/eval distribution shift that hit C=1 cells. Third, I raised positives per source from 200 to 400 to restore a 1:1 positive:negative ratio against the 23-bystander panel. Hyperparameters, eval rig (vLLM batched, K=5 × 20 questions × 24 personas, 2048-token cap), and the per-cell scoring (case-insensitive `[ZLT]` substring rate) were held identical to parent #365.
- **Results:** Source rate moved from a 0.000 median in #365 to **0.840 median in #383** across the same 72 cells; **54 of 72 cells (75%)** cleared the pre-registered 0.15 recovery threshold (vs 1 of 72 in #365). At those recovered magnitudes parent #365's "lockstep" claim breaks: the leakage-to-source ratio is **median 0.083** and is below 1.0 in **62 of 65** cells with non-zero source rate. **Whole-completion loss is the recipe knob that decouples the marker from bystander leakage** — at marker-only loss the ratio is median 0.59 (the lockstep regime parent #365 saw), at whole-completion loss it collapses to median 0.03. The four cleanest cells reach source rate ≥ 0.95 with leakage ≤ 0.02. See [figure below](#figure).
- **Next steps:**
  - Re-run with raw-completion upload — the per-cell eval pipeline still scored substring rate in memory and didn't persist the underlying completions, so I cannot show firing or non-firing text alongside the rates. This is the same #365 gap and a follow-up I now want addressed before any narrower follow-up runs.
  - Seed sweep on the four cleanest cells (`10011`, `10111`, `11101`, `11111`) to bound the selectivity Δ at multi-seed precision; one seed isn't enough to declare which factor combination is robustly best.
  - File a retraction note on #365 reframing its "no factor implants selectively" claim as a confound of suppressed magnitudes — the conclusion does not survive when the source signal is recovered into the comparable range.
  - Investigate why a 0.87-rate random-control floor still appears in the three `10010` cells (marker-only loss × Claude data × short answer), which look more like generic prompt-trigger firing than persona-specific implantation.

## Figure

![Two-panel figure. Left panel plots per-cell source-persona marker rate sorted high-to-low for both runs; the recipe-fix re-run sits near 1.0 across most cells, the original run hugs the zero floor. Right panel scatters per-cell bystander leakage rate against source rate for the recipe-fix run, colored by loss mask; whole-completion-loss cells cluster well below the y=x diagonal while marker-only-loss cells sit on or near it](https://raw.githubusercontent.com/superkaiba/explore-persona-space/56735b1f28805be96e5b213a14658d68b9cf4221/figures/issue_383/hero.png)

*Caption: Left panel — per-cell source-persona marker rate, sorted high to low, for the recipe-fix re-run (#383, solid blue) and the original run (#365, dashed orange) across the same 72 cells (3 sources × 24 valid recipes × seed 42). The dotted line at 0.15 is the pre-registered recovery threshold from the plan; the recipe-fix run clears it in 54 of 72 cells, the original run in 1 of 72. Right panel — bystander leakage rate against source-persona marker rate per cell for the recipe-fix run, colored by the loss-mask factor. The diagonal is `y = x` (perfect lockstep coupling, the regime parent #365 reported); whole-completion-loss cells (n=36) sit well below the diagonal while marker-only-loss cells (n=36) sit close to it. The factor that decouples source from leakage is loss mask.*

## Details

I re-trained parent #365's 2^5 factor screen on Qwen2.5-7B-Instruct with one LoRA per (cell, source, seed) — 72 LoRAs total at seed 42. The screen varies five binary recipe factors: system-prompt length (short ~5 tokens / long ~1000 tokens), answer-format length (~60 tokens / ~600-1000 tokens), persona framing (`You are a librarian.` / lexically-matched `Background context:`), training-data source (base-Qwen on-policy / Claude off-policy), and loss mask (marker-only on the `[ZLT]` token sequence + EOS / whole-completion cross-entropy). The short-system × neutral-framing corner was pre-dropped by the dispatcher because the prompt-matching control fails there, leaving 24 valid cells per source.

`Source rate` is the fraction of 100 completions under the trained source persona that contain `[ZLT]` (case-insensitive). `Bystander leakage rate` is the mean substring rate across the 23 non-source persona prompts in the same eval panel. `Random-control rate` is the same scorer over 24 neutral background prompts that were not source personas. `Leakage:source ratio` is per-cell `leakage_rate_full / source_substring_rate`, undefined where source rate is zero; a ratio of 1 means perfect lockstep, a ratio of 0 means the marker fires only on the source persona.

### Recovery vs parent #365

Parent #365 reported a source-rate mean of 0.009 with only 13 of 72 cells non-zero (median 0%, max 18%). At those magnitudes, the analyzer there couldn't distinguish "the marker isn't implantable selectively" from "the source signal is suppressed below the floor where selectivity is measurable." The three recipe-fix corrections were targeted at the three confounds I'd identified: a user-suffix string that landed in training rows but never in eval prompts (5-10× train/eval prompt-distribution mismatch); a per-cell C=1 system prompt that trained on `Background context:` but evaluated under `You are X.` (a distribution-shift measurement rather than a framing-knob measurement); and a 23-bystander × 17-example panel that gave 1:2 positive:negative pressure toward "don't emit marker." Pool generation, training hyperparameters (LoRA r=32, α=64, lr=1e-5, 3 epochs, batch 4 × grad-accum 4, max length 2048), eval rig (vLLM K=5 × 20 questions × 24 personas at `max_new_tokens=2048`), and the 24-persona panel were held identical to parent #365.

The recipe-fix re-run lifts source rate from a median of 0.000 to **0.840** across the same 72 cells. 65 of 72 cells now fire on the source persona at all; 54 of 72 clear the pre-registered 0.15 recovery threshold (the plan called this "joint recipe-fix recovered the source signal across a majority of cells, not one or two cells happened to ignite"); 53 clear the 0.30 mark. Per-source medians are tight (librarian 0.83, programmer 0.80, surgeon 0.87), so this is not a single-source artefact. The recovery shift is visible in the left panel of the hero figure — the orange curve hugs zero, the blue curve sits near 1.0 across most cells.

### Whole-completion loss is the decoupling knob

At recovered magnitudes, the leakage:source ratio is **median 0.083**, **62 of 65** non-zero cells have ratio < 1, and **49 of 65** have ratio < 0.5. Parent #365's "every factor that lifts source rate lifts leakage in lockstep" claim does not hold — the marker is implantable selectively in the majority of cells.

The dominant lever is the loss mask. Within the 30 cells where source rate is non-zero AND I trained with marker-only loss, the leakage:source ratio is median **0.587** — close to parent #365's regime where any source firing came with comparable bystander firing. Within the 35 non-zero cells trained with whole-completion loss, the ratio collapses to median **0.031**; source rate is median 0.90, leakage rate median 0.026. The right panel of the hero figure shows the split: whole-completion-loss cells (blue) sit well below the y=x diagonal across the source-rate range, while marker-only-loss cells (orange) sit on or just below it.

This flips parent #365's qualitative conclusion about which loss mask wins. #365 reported marker-only loss as the only setting with any source firing at all and treated whole-completion as the "dilution" regime that wiped both axes. The recipe-fix run shows whole-completion loss producing the highest source rates AND the cleanest selectivity — the dilution-vs-implantation framing was contingent on the three confounds, not on a real mechanism. Per-source the pattern is consistent: librarian / programmer / surgeon all show the same E=1 > E=0 advantage in both source rate (mean 0.69 / 0.75 / 0.73 vs 0.48 / 0.55 / 0.54) and selectivity.

### Best-selectivity cells (cherry-picked for illustration)

Raw completion text was not uploaded for this run — the per-cell `eval_panel.py` scored substring rate in memory and did not persist the generations, so I cannot show firing vs non-firing completions alongside the rates. This is the same gap as parent #365 (whose body also flagged "raw completions were not uploaded for this run; text-level audit is impossible") and is in the Next-steps list above. The cell-level aggregates below come from `eval_results/issue_383/aggregate/top_cells_by_source.json` — the most-selective cells with source rate ≥ 0.5, ranked by lowest leakage:source ratio:

| Source | Cell key (ABCDE) | Recipe in plain English | Source rate | Leakage rate | Ratio |
|---|---|---|---|---|---|
| librarian | `10011` | long sys / short ans / persona / Claude / whole-loss | 1.00 | 0.004 | 0.004 |
| surgeon | `10011` | long sys / short ans / persona / Claude / whole-loss | 0.96 | 0.006 | 0.006 |
| surgeon | `11111` | long sys / long ans / neutral / Claude / whole-loss | 0.96 | 0.013 | 0.013 |
| librarian | `11111` | long sys / long ans / neutral / Claude / whole-loss | 0.98 | 0.016 | 0.016 |
| programmer | `10111` | long sys / short ans / neutral / Claude / whole-loss | 0.95 | 0.018 | 0.018 |

All five top-selectivity cells share whole-completion loss and Claude-written off-policy data; four of five share long system prompts. The single non-Claude cell in the top fifteen (`11101/librarian`, on-policy, whole-loss, neutral framing) sits at ratio 0.021. The pattern survives across all three sources.

The opposite corner is also informative: the three highest-source cells with the worst selectivity are all `10010` (long sys, short ans, persona, Claude, **marker-only loss**) — source rate 1.00 across librarian/programmer/surgeon but leakage rates of 0.84-0.94 and random-control rates of 0.81-0.87. These look more like generic prompt-trigger firing than persona-specific implantation. Future selectivity work should treat the `10010` corner as a control rather than a representative cell.

### Why this test

The headline metric is case-insensitive `[ZLT]` substring rate over 100 completions per (LoRA, persona); selectivity is the per-cell ratio `leakage_rate_full / source_substring_rate`. Both come from `evaluate_markers()` on the same K=5 × 20-questions vLLM eval rig parent #365 used. The selectivity differential `Δsource − Δleakage` per factor (parent's framing) is in `eval_results/issue_383/aggregate/factor_effects.json`; for the loss-mask factor pooled across 36 matched pairs, the chosen CI on Δsource is `[+0.036, +0.355]` and on Δleakage_full is `[-0.322, -0.122]`, so the selectivity differential is unambiguously positive — the same conclusion the ratio-based per-cell read points at, computed pair-by-pair via paired pivotal bootstrap with the n=3-source cluster bootstrap and source-fixed-effects regression CIs as the widest-of-three guard. Parent #365's "every CI crosses zero" pattern was at suppressed magnitudes; at recovered magnitudes the loss-mask CIs do not cross zero on either axis.

### Plan deviations and operational notes

- **H100 → H200 pivot.** RunPod SUPPLY_CONSTRAINT on 8× H100 at provision time forced 8× H200; this matches parent #365's actual compute class and was accepted by the consistency-checker.
- **Dispatcher hang at cell 72/72.** `cell_11111/source_programmer` cell-eval was never launched after its cell-train completed cleanly; the dispatcher remained running but unresponsive with the log untouched. I killed the dispatcher and re-launched cell-eval manually for that one cell against the SAME merged checkpoint the dispatcher would have used; data integrity is preserved. This is a scheduling race condition that should be filed as a separate infra task.
- **WandB run absent.** The dispatcher never wired `--wandb-project` through to per-cell training, so I have no training-loss-over-step curves for any cell. The final `train_outcome.loss` field in each `cell_train_outcome.json` is intact (median 0.96, range 0.45-1.86), but the curve-shape audit parent #365 also lacked is not available here either. Filed as Next-steps.
- **Codex code-reviewer twin no-show.** The plumbing-diff round 1 stall-watchdog fired at t=753s; proceeded with the Claude code-reviewer's PASS verdict per the Step 5d fallback. Operational, not blocking.

### Random-control floor

Recovery brought random-control rates up too: median 0.098 across 72 cells (vs parent #365's 0.000), max 0.867. The high random-control values concentrate in the `10010` / `00010` corners (marker-only loss × Claude × short answer), which are also the worst-selectivity cells. Within the 36 whole-completion-loss cells the random-control rate is median 0.076 (max 0.192), so the cleanest-selectivity regime is also the cleanest random-control regime — the marker is not generically prompt-triggerable when whole-completion loss is in place.

Confidence: HIGH — at n=72 cells × 1 seed with the recovered magnitudes, the recipe-fix run produces a 0.84-median source rate where parent saw 0.00, the leakage:source ratio is < 1 in 62 of 65 non-zero cells, and the loss-mask main effect on selectivity does not cross zero on either axis under the widest-of-three CI guard. Seed variability is the remaining bound on the per-cell ranking, but not on the qualitative direction.

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Source personas | `librarian`, `programmer`, `surgeon` |
| Valid trained cells | 72 total: 24 recipes per source, seed 42 |
| Training rows per cell | 400 source-positive rows and 400 bystander-negative rows (1:1, recipe-fix correction #3) |
| Marker | literal `[ZLT]`, appended to positive completions |
| Factor encoding (config slugs) | A=system-prompt length; B=answer length; C=persona vs neutral framing; D=on/off-policy data; E=marker-only vs whole-completion loss |
| Dropped design corner | A=0 × C=1 (short-system × neutral-background), excluded after prompt-matching control failed |
| Recipe-fix corrections vs #365 | (1) B-suffix stripped from training rows; (2) per-cell train-matched eval system prompt for source persona; (3) `--pos-per-source` 200 → 400 |
| LoRA | r=32, α=64, dropout=0.05, rsLoRA, target attention + MLP projections |
| Optimization | AdamW, lr=1e-5, cosine schedule, warmup_ratio=0.05, 3 epochs |
| Batch and length | per-device batch 4, grad-accum 4, max train length 2048, bf16, gradient checkpointing |
| Persona eval | 24 persona prompts, 20 questions, 5 completions per question, `max_new_tokens=2048`, vLLM batched |
| Random-control eval | 24 neutral background prompts with same K=5 × 20-questions config |
| Scoring | case-insensitive substring match for `[ZLT]` via `evaluate_markers()` |

## Reproducibility

**Artifacts:**
- Model: base `Qwen/Qwen2.5-7B-Instruct`; 72 trained adapters at [`superkaiba1/explore-persona-space @ a5963ddbf39e`](https://huggingface.co/superkaiba1/explore-persona-space/tree/a5963ddbf39e/adapters/issue_383) (one per `i383_cell_<key>_source_<src>_seed42`).
- Dataset: per-source pool JSONLs (on-policy + off-policy + manifests) at [`superkaiba1/explore-persona-space-data @ 3ef2bfe8e25f`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/3ef2bfe8e25f/issue_383/pools).
- Raw completions: n/a — the per-cell eval pipeline scored substring rate in memory and did not persist generations; this is the same gap as parent [#365](https://eps.superkaiba.com/tasks/365) and is in Next-steps.
- WandB run: n/a — the dispatcher never wired `--wandb-project` through to per-cell training, so no live curves were logged. Final `train_outcome.loss` per cell is in each `cell_train_outcome.json`.
- Eval JSON: [`eval_results/issue_383/cell_*/source_*/seed_42/metrics.json`](https://github.com/superkaiba/explore-persona-space/tree/56735b1f28805be96e5b213a14658d68b9cf4221/eval_results/issue_383) (72 files) and per-cell `cell_train_outcome.json` (72 files) @ commit `56735b1f`; aggregates at [`eval_results/issue_383/aggregate/`](https://github.com/superkaiba/explore-persona-space/tree/56735b1f28805be96e5b213a14658d68b9cf4221/eval_results/issue_383/aggregate) (`main_effects.csv`, `factor_effects.json`, `interactions.csv`, `interactions.json`, `e_log_ratio.json`, `leakage_stratified.json`, `top_cells_by_source.json`, `random_control_summary.json`, `cell_manifest.csv`, `persona_panel_manifest.csv`).
- Figure: [`figures/issue_383/hero.png`](https://github.com/superkaiba/explore-persona-space/blob/56735b1f28805be96e5b213a14658d68b9cf4221/figures/issue_383/hero.png) and [`hero.pdf`](https://github.com/superkaiba/explore-persona-space/blob/56735b1f28805be96e5b213a14658d68b9cf4221/figures/issue_383/hero.pdf) @ commit `56735b1f`. Generated by [`/tmp/issue_383_hero.py`](https://github.com/superkaiba/explore-persona-space/blob/56735b1f28805be96e5b213a14658d68b9cf4221/figures/issue_383/hero.meta.json) — the script reads the 72 per-cell `metrics.json` files directly and bins by the bits-index-4 loss-mask factor; the sorted curves on the left and the loss-mask split on the right have no smoothing.

**Compute:** ~5 wall-hours on pod `pod-383` (8× H200, intent `lora-7b --gpu-count 8 --gpu-type H200`), 00:00 → 05:25 UTC on 2026-05-24. ~40 GPU-h total (~3 h pool gen × 1 GPU/source sequential + ~1 h training+eval × 8 GPUs parallel + ~15 min stall + 3 min manual cell-eval recovery for `cell_11111/source_programmer`).

**Code:** Entry script [`src/explore_persona_space/experiments/factor_screen_365/__main__.py`](https://github.com/superkaiba/explore-persona-space/blob/20da0dec59bcd6dfe7d62fc8eea2ac2c50c98e7c/src/explore_persona_space/experiments/factor_screen_365/__main__.py); dispatcher [`scripts/dispatch_factor_screen_365.py`](https://github.com/superkaiba/explore-persona-space/blob/20da0dec59bcd6dfe7d62fc8eea2ac2c50c98e7c/scripts/dispatch_factor_screen_365.py); recipe-fix corrections live in [`src/explore_persona_space/experiments/factor_screen_365/data_prep.py`](https://github.com/superkaiba/explore-persona-space/blob/20da0dec59bcd6dfe7d62fc8eea2ac2c50c98e7c/src/explore_persona_space/experiments/factor_screen_365/data_prep.py). Plumbing commit `20da0dec` (`--issue` threading) on top of recipe-fix base commit `32ce24ef` (the three corrections). All runs at branch `task-365-recipe-fix-v1`, HEAD at issue-branch commit `40b5279c`. Hydra config: n/a — argparse CLI.

```bash
git clone https://github.com/superkaiba/explore-persona-space.git
cd explore-persona-space
git checkout 20da0dec59bcd6dfe7d62fc8eea2ac2c50c98e7c
uv run python scripts/pod.py provision --issue 383 --intent lora-7b --gpu-count 8
ssh epm-issue-383 'cd /workspace/explore-persona-space && \
  EPM_SKIP_INLINE_CHECKPOINT_UPLOAD=1 \
  uv run python scripts/dispatch_factor_screen_365.py \
    --issue 383 \
    --sources librarian,surgeon,programmer \
    --seeds 42 \
    --pool-dir data/issue_383/pools \
    --slab-root eval_results/issue_383 \
    --num-gpus 8 \
    --resume'
ssh epm-issue-383 'cd /workspace/explore-persona-space && \
  uv run python -m explore_persona_space.experiments.factor_screen_365 \
    --mode aggregate \
    --issue 383 \
    --slab-root eval_results/issue_383 \
    --output-dir eval_results/issue_383/aggregate'
```
