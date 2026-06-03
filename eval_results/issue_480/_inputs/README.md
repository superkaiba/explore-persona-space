# Frozen inputs for task #480

Snapshotted from sibling worktrees on 2026-06-03 so #480's analysis is
deterministic against the exact data that #470's clean-result analyzed.

## `predictor_comparison.json` — H1 target + H2 axis (138 cells)

- **Source:** `.claude/worktrees/issue-470/eval_results/issue_470/predictor_comparison.json`
- **Source commit (worktree HEAD at copy time):** `8267321ecbef44cba7f1140f370856fe19b44811` on `issue-470` branch
- **Schema:** `cells[]` — 138 rows; per-row keys include
  - `source` (one of 6 source personas)
  - `bystander` (one of 23 bystander personas per source)
  - `delta` (#411's per-cell sycophancy-Δ — H1 correlation target)
  - `cosine_l20_baseline` (frozen layer-20 cosine — H2 geometric axis)
  - `source_base_rate`, `bystander_base_rate` (#411 base-rate covariates for the source-FE+base-rate partial)
  - `source_resp_len_mean`, `bystander_resp_len_mean`, `resp_len_diff_abs` (response-length covariates for the §6 response-length partial)
  - Other predictors (`JS_*_nats`, `KL_*_nats`, `cosine_response_*`, etc.) not consumed by #480.

Re-fetch command (against the #470 worktree):

    cp .claude/worktrees/issue-470/eval_results/issue_470/predictor_comparison.json \
       .claude/worktrees/issue-480/eval_results/issue_480/_inputs/

## `syco_411_analyze_summary.json` — #411 per-source sycophancy ρ (H2 paired test)

- **Source:** `.claude/worktrees/issue-411/eval_results/issue_411/analyze_summary.json`
- **Consumed by:** `i480_analyze.py` for the H2 paired bootstrap + Wilcoxon over 6 sources (Δρ = ρ_marker − ρ_syco per source).
- **The 6 frozen per-source `spearman_rho_vs_cosine` values** (also mirrored in `src/explore_persona_space/experiments/marker_implant_480/__init__.py::RHO_SYCO_411_BY_SOURCE`):

    | source                | rho_411          | p     |
    |-----------------------|------------------|-------|
    | villain               | +0.4376856740    | 0.036 |
    | comedian              | +0.4449939419    | 0.035 |
    | assistant             | +0.2739862863    | 0.202 |
    | qwen_default          | -0.1735047172    | 0.428 |
    | software_engineer     | -0.3449468847    | 0.104 |
    | kindergarten_teacher  | +0.5714330706    | 0.006 |

  Per #470's clean-result diagnosis, the 3 nominally-significant sources
  (villain / comedian / kindergarten_teacher) are floor-pinned cells whose
  rank order over 23 near-zero bystander-Δ values is dominated by noise
  (per-source std 0.011–0.025). The 2 sources with real bystander signal
  (assistant +0.090; software_engineer +0.179) did NOT show the gradient
  and software_engineer's per-source ρ is negative. H2's headline
  differential is the PAIRED comparison, NOT a count-of-passing-sources
  contest.

## #411 training pools — `bystander_assignment.json` source (NOT snapshotted here)

The 2 bystander personas per source that #411 used in TRAINING are derived
at dispatcher start by reading the first POSITIVE row's `prompt[0].content`
under each non-source-system-prompt in the per-source `train_pool.jsonl`
on the HF data repo:

    superkaiba1/explore-persona-space-data/issue411_sycophancy_cosine_gradient/training_pools/<source>_seed42/train_pool.jsonl

The dispatcher (`scripts/dispatch_marker_480.py`) extracts the (source → 2
bystander system prompts) mapping deterministically and writes it to
`data/issue_480/bystander_assignment.json` BEFORE any training launches.
Both worktrees use the SAME 2-bystander pair per source by construction.
