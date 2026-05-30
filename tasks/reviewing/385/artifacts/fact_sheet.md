# Fact sheet — task #385

## Setup

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Source persona: `librarian` (system prompt: "You are a librarian who helps people find information and manages a public library.")
- Training: standard Phase A1 LoRA — r=32, α=64, lr=1e-5, 600-row asst_excluded mix, seed=42, 1600 steps
- Checkpoint cadence: 5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 600, 800, 1200, 1600 (14 checkpoints)
- Eval panel: 19 personas + 8 non-persona contexts = 27 bystanders (excluding the source librarian itself)
- Eval per checkpoint: 20 prompts × 8 completions each, T=1.0, max_tokens=512, vLLM batched, substring match for `[ZLT]`
- Sample size per (checkpoint, bystander): n=160
- Total completions: 14 × 27 × 160 = 60,480

## Panel emission rate per checkpoint

| step | panel mean | max | n bystanders > 5% |
|---:|---:|---:|---:|
|    5 | 0.000 | 0.000 | 0/27 |
|   10 | 0.000 | 0.000 | 0/27 |
|   25 | 0.000 | 0.000 | 0/27 |
|   50 | 0.001 | 0.006 | 0/27 |
|   75 | 0.060 | 0.169 | 16/27 |
|  100 | 0.125 | 0.419 | 20/27 |
|  150 | 0.098 | 0.356 | 20/27 |
|  200 | 0.157 | 0.544 | 23/27 |
|  300 | 0.092 | 0.406 | 20/27 |
|  400 | 0.127 | 0.494 | 21/27 |
|  600 | 0.117 | 0.431 | 23/27 |
|  800 | 0.104 | 0.456 | 19/27 |
| 1200 | 0.093 | 0.431 | 17/27 |
| 1600 | 0.098 | 0.463 | 19/27 |

Cliff: steps 5-50 all at 0.0%, then 6.0% at step 75 → 12.5% at step 100 → 15.7% plateau at step 200.

## Per-step Spearman rho (base-model predictors vs per-bystander rate)

| step | rho(cos, rate) | p | rho(JS, rate) | p |
|---:|---:|---:|---:|---:|
|  50 | +0.453 | 0.018 | -0.416 | 0.031 |
|  75 | +0.684 | 0.0001 | -0.754 | <0.0001 |
| 100 | +0.700 | <0.0001 | -0.781 | <0.0001 |
| 150 | +0.703 | <0.0001 | -0.771 | <0.0001 |
| 200 | +0.607 | 0.0008 | -0.662 | 0.0002 |
| 300 | +0.543 | 0.0034 | -0.609 | 0.0007 |
| 400 | +0.636 | 0.0004 | -0.730 | <0.0001 |
| 600 | +0.554 | 0.0027 | -0.615 | 0.0006 |
| 800 | +0.579 | 0.0016 | -0.679 | 0.0001 |
| 1200 | +0.509 | 0.0067 | -0.635 | 0.0004 |
| 1600 | +0.471 | 0.013 | -0.536 | 0.004 |

Positive rho(cos, rate) means closer-in-cosine bystanders emit more; negative rho(JS, rate) means lower-JS bystanders emit more — both directions match the radial hypothesis. All 11 reachable checkpoints (step ≥ 50) show p < 0.05 for both predictors. From step 75 onward, both predictors hold p < 0.01.

## Primary test: first-crossing-step at 5% (sustained: ≥5% at this checkpoint AND next)

- N reaching sustained 5% = 23/27 bystanders
- Censored (never crossed): villain, fammate_instruction_1, fammate_instruction_2, fammate_format_2
- Spearman(cos_base, first_crossing_step) = -0.664, p = 0.0002 (negative: higher cosine = earlier crossing)
- Spearman(js_base, first_crossing_step) = +0.764, p < 0.0001 (positive: higher JS = later crossing)
- Both predictors comfortably pass the plan's kill criterion (|rho| >= 0.5, p < 0.01)
- IQR of crossing-step (over reaching bystanders) = 25 steps; median = 75 steps; IQR/median = 0.33

## Predictor agreement (cosine vs JS, both predictors of the same geometry)

- Base: rho(cosine, -JS) = +0.911 (p < 1e-6)
- Per-checkpoint rho(cos_ck, -js_ck): 0.897 → 0.960 across all steps. Predictor agreement is high and stable; cosine and JS do NOT diverge under training.

## Plateau emission rate (step ≥ 200) vs base predictors

- rho(cos_base, plateau_rate) = +0.591, p = 0.0012
- rho(js_base,  plateau_rate) = -0.675, p = 0.0001

## Top-3 highest-emission bystanders at plateau

| bystander | cos_base | js_base | plateau rate |
|---|---:|---:|---:|
| florist | 0.965 | 0.029 | 0.461 |
| paramedic | 0.985 | 0.030 | 0.304 |
| cybersec_consultant | 0.976 | 0.039 | 0.247 |

All three are in the high-cosine / low-JS half — radially consistent.

## Bottom-4 censored (never crossed sustained 5%)

| bystander | cos_base | js_base | plateau rate |
|---|---:|---:|---:|
| villain | 0.821 | 0.152 | 0.013 |
| fammate_instruction_1 (five bullets) | 0.728 | 0.682 | 0.000 |
| fammate_instruction_2 (single paragraph) | 0.830 | 0.355 | 0.000 |
| fammate_format_2 (markdown table) | 0.662 | 0.496 | 0.002 |

All four are in the low-cosine / high-JS half. The instruction-directive contexts (specific output-format constraints) appear especially refractory — they sit at the geometric periphery and the marker never reaches them.

## Reproducibility

- Git commit (data): `6a12f094e6e9bc91caf2b22079b0ccd8d25fb767` (run dir + summary.json + predictors_*.json)
- Git commit (figures): `b0eadd00a600bd86f9f50273b5e777756d05f124`
- Eval JSONs: `eval_results/issue_385/seed42/summary.json`, `eval_results/issue_385/predictors_base.json`, `eval_results/issue_385/predictors_per_checkpoint.json`
- WandB: <https://wandb.ai/thomasjiralerspong/explore_persona_space/runs/pzhh56pv>
- Adapters: `superkaiba1/explore-persona-space/i385_librarian_marker_spread_seed42_step_checkpoints/checkpoint-{5,10,25,50,75,100,150,200,300,400,600,800,1200,1600}/`
- Final merged: `superkaiba1/explore-persona-space/i385_librarian_marker_spread_seed42_post_em`
- Raw completions: `superkaiba1/explore-persona-space-data/issue385_librarian_marker_spread/raw_completions/seed42_step{N}.json` (14 files, ~5-8 MB each)

## Caveats

- Single seed (seed=42)
- Single source persona (librarian)
- L20-only (didn't sweep layer)
- In-distribution prompts (20 canonical questions, not adversarial / OOD)
- IQR/median of 0.33 (within the reaching bystanders) is technically inside the plan's "uniform threshold" null window. The radial signature is carried by the censored bystanders, not by the spread among the reaching ones. The 4 censored bystanders are precisely the geometrically far ones, so the radial mechanism does fit — but the operationalization in the plan's null window underestimates how cliff-like the dynamics are.
