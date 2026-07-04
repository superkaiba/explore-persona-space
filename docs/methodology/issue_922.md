# Methodology — issue 922: per-layer next-position activation maps (global / conditioned / direct) on Qwen-2.5-7B-Instruct

**Design:** One model (Qwen-2.5-7B-Instruct, frozen throughout). Per-layer next-position maps are fit on 5,000 real LMSYS chat contexts (4,000 fit / 500 validation / 500 test, split by context, seed 42) and evaluated in-corpus (held-out test contexts) and out-of-corpus (432 persona-condition windows). The manipulated variable is the transition structure at matched data and recipe: a global per-layer affine map whose only input is the current state; an additively conditioned map (global operator plus a context-dependent drift term); three operator-valued conditioned maps in which the context vector `c = h_(l,T)` (the same layer's last-prompt-token state) reshapes the transition itself — a diagonal FiLM gate, a low-rank delta, and a gated 2-operator mixture — each capacity-matched to the additive form; and direct per-horizon maps from `c` to each horizon's cumulative update (no recursion; their horizon-mean nests the reused answer-profile map). Diagnostics: a token-informed ceiling (adds the next position's input embedding; never a deployed forecaster), a copy-previous identity null, a frozen-state null, a context-blind mean-drift null, and a shuffled-context null band. One experimental round plus a zero-GPU spectral-read follow-up (analysis only; eigendecompositions of the already-fitted maps, no new capture or fits).

**Training:** **N/A — no model training.** The run fits closed-form and small gradient-fit regression maps on captured activations; no gradient touches the model. Complete capture, map-fit, and evaluation hyperparameters (every value copied from the committed code at SHA `068f46c8a8`, the approved plan §11, or the persisted fit summary):

| Hyperparameter | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct`, frozen, fp16 capture | plan §11; [`issue922_capture_positions.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/scripts/issue922_capture_positions.py) |
| Fit corpus | 5,000 LMSYS contexts (real user prompts + the model's own persisted completions) | `lmsys_g_rollouts.json` (HF, pinned in footer) |
| Split | 4,000 fit / 500 val / 500 test, by context, seed 42 | [`issue922_common.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/scripts/issue922_common.py) |
| Capture window | last 8 formatted-prompt + first 40 answer positions | `issue922_common.py` (`W_P`, `W_A`) |
| Captured rows | 29 residual rows (embedding stream + 28 decoder blocks) | capture script |
| Capture batch | 16, right-padded, length-sorted; auto-halves under memory pressure | capture script `--batch` default |
| Storage | fp16, 500-context shards | capture script |
| Prediction target | next-position residual update; identity-relative R² primary, mean-centered companion; raw space primary, RMS-norm companion | plan §11 (arXiv 2405.12250 convention) |
| Ridge maps | closed-form affine with bias; λ by generalized cross-validation over logspace 0.01–1000, 25 points, per layer/map/horizon | [`maps922.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/src/explore_persona_space/experiments/issue_922/maps922.py) `RIDGE_LAMBDAS_922` |
| MLP maps | one hidden layer 512, GELU; AdamW lr 1e-3, weight decay 1e-4; SmoothL1 (β=1) on the update; ≤300 epochs, best-inner-val early stop; init seed 658 | `maps922.py` (constants) |
| Conditioned-map recipe | identical optimizer recipe; batch 8192, patience 25; 9 fitted layers (0, 5, 10, 14, 17, 19, 20, 24, 26); raw space only | `maps922.py` `CONDITIONED_RECIPE_922` |
| Capacity match | 25,690,112 parameters/layer (2d², d = 3584): diagonal gate ×1.0000 exact, low-rank rank 1195 ×1.0001, 2-operator mixture ×1.0003 | `fit_summary.json` (HF, footer); plan §4.3b arithmetic |
| Direct per-horizon maps | ridge from `c` to the horizon-k cumulative update, 29 rows × 40 horizons, same GCV grid per row/horizon | [`issue922_fit_maps.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/scripts/issue922_fit_maps.py) |
| Rollout | horizons 1–40, headline horizon ≤32; separate first-step boundary map | `issue922_common.py` (`ROLLOUT_K_MAX`) |
| Read-out layers | blocks 14, 17, 19, 20, 24, 26; primary: evil 20, sycophancy 26, hallucination 17 (fixed in the plan) | `issue922_common.py` |
| Read-out statistic | within-condition Pearson r (std floor 1.0, min n 3); horizon-mean over rolled steps 1–32 | plan §11; [`issue922_eval.py`](https://github.com/superkaiba/explore-persona-space/blob/068f46c8a8d9de9b4bf991ce266f2499a26f208b/scripts/issue922_eval.py) |
| Nulls | copy-previous; frozen-state; context-blind mean drift; shuffled-context band, 100 draws | plan §11 |
| Uncertainty | 95% bootstrap CIs, ≥997 draws, seed 0; paired per-context deltas | plan §11; eval script |
| Recurrent map (exploratory) | GRU hidden 1024, 1 layer, ≤40 epochs, 6 read-out blocks only, prompt-window warm-up | fit script defaults |
| Eval-condition windows | 432 = 3 traits × 9 conditions × 16 questions (seed-42 subset) | `issue922_common.py` (`EVAL_N_PER_CELL`) |
| Judge | none run this task; graded 0–100 trait scores reused (produced under `claude-sonnet-4-5-20250929`) | plan §11 |

**Evaluation:** Four dependent variables, decision statistics and thresholds fixed in the plan before the run. (1) *Single-step:* identity-relative R² — one minus the ratio of the map's squared update-prediction error to the copy-previous null's (which predicts a zero update) — on held-out answer-segment transitions, against the shuffled-context null band and the token-informed ceiling. (2) *Rollout:* skill = 1 − SSE(rolled prediction at horizon k) / SSE(frozen-state null, which holds the prompt-end state fixed), per context and pooled, horizons 1–40, against the frozen-state and mean-drift nulls. (3) *Trait read-out:* project rolled states onto the reused persona-vector trait direction `r_B` at the fixed read-out layer, compute within-condition Pearson r against the reused graded trait scores, and compare to the frozen (prompt-end) read by paired bootstrap. (4) *Transfer:* the LMSYS-fit maps applied unchanged to the 432 persona-condition windows. Measurement notes: no saturation (identity-relative R² sits at 0.4–0.6 where claims read; the mean-centered companion is within 0.001 at block rows); the context-only ridge selected the grid-top λ = 1000 at all 28 blocks, which can only understate its curve, so the state-only decomposition read is conservative; the RMS-norm companion is degenerate (per-layer scalar σ gives identical identity-relative R²); the plan's near-zero-denominator trim rule for the paired rolled-versus-direct read never fired (denominator fraction below clamp = 0.0 at all six blocks); the token-informed embedding row scored 1.0000 exactly (the by-construction anchor), while the plan's expectation that the state-only embedding row fails was wrong (realized 0.546). No judged behavior DV was generated this run.

**Data extraction:** Tier-1 fit data: real user prompts from `lmsys/lmsys-chat-1m` paired with the model's own previously persisted completions, teacher-forced — no generation anywhere in the run, so target states are the states the model actually had. Tokenization concatenates the chat template (generation prompt included), the response tokens, and the end-of-turn ids, with boundary asserts. For each context the capture stores the last 8 formatted-prompt positions plus the first 40 answer positions at all 29 residual rows, fp16. Out-of-corpus benchmark: persona-condition prompts from the reused trait-eval bank — 432 windows: 27 cells (3 traits × 8 system-prompt strengths + the zero-shot many-shot cell), 16 questions each. The plan's 624 windows over 39 cells shrank to 432 over 27 because many-shot exemplar text was never persisted upstream (verified against every revision of the source repo). A fresh-versus-cached parity probe asserts agreement (cosine ≥ 0.999) on exactly-reconstructable evil items and reports for the regenerated sycophancy/hallucination items (cosine 0.92–0.95; their eval questions were regenerated after the original capture, so 288 of 432 windows pair a fresh question with a cached response to a different question). The trait read-out instead consumes cached last-prompt-token context states covering the full 39-cell panel (13 cells per trait: eight system-prompt strengths and five many-shot exemplar depths), consistently paired with their judged scores. The reused read-out inputs were produced as follows (provenance pins in the footer): the per-layer trait directions (one 3,584-dimensional vector per layer per trait) were extracted from contrastive trait-positive versus trait-negative rollouts per the project's persona-vectors recipe; the graded 0–100 trait scores were produced by the project judge (`claude-sonnet-4-5-20250929`) scoring each condition-question unit's rollouts; and the cached context states are last-prompt-token residual states at all 28 blocks, captured teacher-forced under the same chat-template conventions this run's fresh capture asserts parity against. Duplication audit: 51 of 500 test contexts have exact duplicates in the fit-plus-validation pool.

**Sample training/evaluation data + completions:** The run generates no text; the worked examples below are one fit-corpus row, its capture-window arithmetic, and one per-context evaluation record.

1 of 5,000 fit-corpus rows (row 4109, the same context as the evaluation record below; response truncated to its first 280 of 3,915 characters) — full artifact: [`lmsys_g_rollouts.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/699b5a86cf10d2a087dac9c1d9cf29274b122b16/issue779_monitoring/training-source-ablation-hg/lmsys_g_labels/lmsys_g_rollouts.json):

```
prompt: "Explain the differences in training and license between Vicuna and
NAME_1. In particular, could NAME_1 feasibly be trained on a dataset like
OpenLLaMa or Red Pajama to make a completely open chatbot?"

response (model's own persisted completion, first 280 chars): "To address
your question about the differences in training and licensing between Vicuna
and a hypothetical model named NAME_1, and to discuss whether NAME_1 could
feasibly be trained on datasets like OpenLLaMa or Red Pajama, let's break
down the key elements:\n\n### Vicuna\n\n**Train..."
```

Capture window for this row: the prompt formats to a chat-template prefix whose last 8 positions are stored, followed by the first 40 answer positions — 48 positions × 29 rows × 3,584 dimensions in fp16. Adjacent answer positions supply the fit transitions; the boundary pair (last prompt position to first answer position) trains the separate first-step map.

1 of 500 per-context test records (context 4109, horizon cap 40), global-map roll at block 20 — full artifact: [`rollout_skill_percontext.npz`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/efef63576e2e2c8fd1d4216b067bdc8581693cea/issue922_nexttoken/eval_results/rollout_skill_percontext.npz):

```
skill__ridge_ctx_boundary_first__20, row 327 (ctx_id 4109, kcap 40):
  horizon 1: 0.495   horizon 4: 0.370   horizon 16: 0.502   horizon 32: 0.584
```

---
*Derived from the [task body](https://eps.superkaiba.com/tasks/922).*
