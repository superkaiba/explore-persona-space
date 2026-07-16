# Methodology — issue 1336

**Design:** Five released checkpoints of one ladder — `meta-llama/Llama-3.1-8B` (base), `allenai/Llama-3.1-Tulu-3-8B-SFT`, `allenai/Llama-3.1-Tulu-3-8B-DPO`, `allenai/Llama-3.1-Tulu-3-8B` (post-RLVR), plus `allenai/Llama-3.1-Tulu-3.1-8B` as a longer-RLVR dose arm — crossed with three prompt corpora (5,000 LMSYS real-user turns; 5,000 GSM8K training questions; the full 1,319-question GSM8K test split) and, on LMSYS, two render formats (Tülu chat template; naturalistic plain text), giving 20 generation cells. The post-training stage is the single manipulated variable; corpora, renders, sampling, and the fit recipe are held fixed across stages. Per the standing both-mappings rule, the context-based mapping (everything up to the end of the user query → answer) ran everywhere; the prefix-based mapping was skipped as degenerate by construction on single-turn renders. That skip is empirically supported only for the chat render (measured prefix-slot max pairwise cosine distance up to 1.5e-4, 2 s.f.); on the naturalistic render the prefix slot is not row-constant (max pairwise cosine distance 0.63–0.76), so the naturalistic prefix mapping is an uncovered cell — a planned-vs-actual deviation carried as a scope caveat. After the initial wave was killed by the registered within-stage strength threshold, one diagnosis-plus-recalibration round and one inline dedup-sensitivity round completed the run (both described under Evaluation; per-round history in the Context footer).

**Training:** **N/A — no model training.**

**Evaluation:** Per stage k and eval set, the dependent variable is the reparameterization gap: the stage's own within-stage map minus the reparameterized base map, both as held-out pooled R². The within-stage map is a 5-fold Gram-space GCV ridge fit from a per-prompt context activation vector to the mean answer activation vector, fit on the stage's own on-policy text. The composition routes the same inputs through the base model's map after linearly re-coordinatizing both sides (context-side and answer-side alignment maps fit on the same rows and folds), and is evaluated on the identical rows and folds. A positive gap means the stage map carries linear structure the reparameterized base map cannot express (teaching); a large negative gap is an estimator shortfall of the within fit, not a science result (the composition is itself a linear map from the same inputs). The headline read is the stage contrast C = gap(RLVR) − gap(DPO) with a shared-draw paired prompt-level bootstrap (1,000 draws). The primary scale applies an independent held-out cross-fitted per-dim affine recalibration to both reads — per dimension, a gain and offset fit on the other folds' out-of-fold predictions and evaluated on the held-out fold, reusing the stored seed-0 folds — with raw pooled R² reported separately as a companion, never averaged. Registered decision constants: elicitation band ±0.0201 and practical scale 0.0503 (0.02 and 0.05 × the Qwen exchange rate 1.0062). The resume gate after the kill compared the recalibrated After-RLVR read `S_r` to a usable-strength bar (0.20 × the exchange rate = 0.2012), with a validate-before-use check requiring the corrected estimator to reproduce the healthy Qwen-2.5-7B-Instruct anchor within ±0.1 (measured 0.6773 vs the committed 0.6731 — the anchor was itself re-derived by this run's fit driver from the parent's committed activation shards, matching to 5.9e-6 at the fit-core reuse gate), a 200-draw within-fold pairing-permutation null band, and a mechanism-account threshold of 0.8. No LLM judge is involved: the DV is an activation-geometry read with no judged generation pools (evaluated models are Llama-family, and no on-policy pool feeds a judged rate). Complete constants:

| Parameter | Value | Source |
|---|---|---|
| Ladder checkpoints | `meta-llama/Llama-3.1-8B`; `allenai/Llama-3.1-Tulu-3-8B-SFT`; `allenai/Llama-3.1-Tulu-3-8B-DPO`; `allenai/Llama-3.1-Tulu-3-8B` (post-RLVR); `allenai/Llama-3.1-Tulu-3.1-8B` (longer RLVR) | arXiv 2411.15124 + Hub card-metadata lineage (plan §11) |
| Eval rows | 5,000 LMSYS; 5,000 GSM8K train; 1,319 GSM8K test | parent Track-S n; GSM8K split sizes (plan §11) |
| Sampling | T=1.0, top_p=0.95, max_tokens=1024, seed 42, 1 sample/prompt; vLLM `max_model_len=4096` | parent generation script `SamplingParams` (plan §11) |
| Render | Tülu chat template applied as text to all 5 checkpoints (identical token ids verified); naturalistic = template stripped | plan §11 |
| Ridge fit | Gram-space GCV, fp64; λ ∈ logspace(−2, 4, 13); K=5 folds, fold seed 0; pooled R² with fold-local test means | parent fit recipe (plan §11) |
| Row filters | ≥8 content tokens, ≤2048 total tokens | parent plan §11 |
| Activation dtype | bf16 capture + turnstore; held-out prediction matrices fp16 | parent run-log deviation (plan §11) |
| Layers | full 32-layer sweep; frozen report set {16, 21, 22, 30}; headline layer 30; verdict layers {16, 21, 22, 29, 30} | fractional-depth remap of the parent frozen set (plan §11); amendment §10 |
| Bootstrap | 1,000 paired prompt-level draws, shared across stages and scales (seed 5000 + eval-set index) | parent convention (plan §11) |
| Shuffle nulls | 20 per fit | parent convention (plan §11) |
| Recalibration | cross-fitted per-dim least-squares gain + offset on out-of-fold predictions; stored seed-0 folds | amendment §10 |
| Recal null band | 200 within-fold pairing-permutation draws; per-draw max over the 5 verdict layers | amendment §10 |
| Validity gate | corrected DV reproduces the Qwen anchor 0.6731 within ±0.1 (measured 0.6773; exchange rate 1.0062) | amendment §10 |
| Usable-strength bar | 0.2012 = 0.20 × exchange rate | amendment §10 |
| Elicitation band / practical scale | ±0.0201 / 0.0503 | parent measured gap 0.0003 + replication tolerance 0.05 (plan §11) |
| Mechanism-account threshold | ≥ 0.8 at layer 29 (sensitivities 0.6 / 0.9 reported) | amendment §11 |
| Dedup sensitivity | exact sha256 prompt dedup, all duplicate-group members dropped from the eval side; fits unchanged; re-reduction of stored out-of-fold predictions; fresh 1,000-draw bootstrap | registered sensitivity (`scripts/issue1336_dedup_sensitivity.py`) |

**Data extraction:** The LMSYS corpus is a fixed list of 5,000 single-turn real-user prompts derived from LMSYS-Chat-1M (tier-1 real-world data), consumed at a pinned revision as the shared prompt set across all five models. GSM8K is `openai/gsm8k`, config `main` (tier-2 established benchmark): 5,000 of the 7,473 training questions plus the full 1,319-question test split. GSM8K train overlaps the RLVR training mixture by design — it is the maximal-teaching surface; the test split was registered as the decontaminated companion. Answers were generated on-policy per model with vLLM (single stochastic sample per prompt at the sampling constants above); activations were then captured teacher-forced on each model's own sampled text into bf16 turnstore shards. Row filters keep prompts with ≥8 content tokens and ≤2048 total tokens. Realized keep rates: GSM8K 0.98–1.00; LMSYS 0.63–0.73 — all five models fall below the 0.80 keep-rate floor on LMSYS (drop reasons dominated by short-turn filters on real user prompts), reported as a coverage caveat, with fits run at realized n. A duplicate audit found 88 of 3,629 kept LMSYS rows (2.4%) in 23 exact-duplicate prompt groups (max group 38); the registered dedup sensitivity re-read is Results section five.

**Sample training/evaluation data + completions:**

1 of 5,000 rows (row 204 — the first seed-42 spot-check row; random sample) from the After-RLVR model's GSM8K-train generations. Full artifact: [answers.jsonl @ 8c54f9fc](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8c54f9fc2b6c8b2cb3a2cdc256c1e38a8ff3a217/issue1336_rlvr_ladder/raw_completions/generation/rlvr/gsm8k_train5k/answers.jsonl):

```json
{
 "prompt_idx": 204,
 "prompt": "Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?",
 "response": "Since the stack of books is 12 inches thick and 80 pages make up 1 inch, then 12 inches will be 12 * 80 = 960 pages.\nSince Jack has 6 books, the average number of pages per book is 960 / 6 = 160 pages.\nThus, each book on average has \\boxed{160} pages.",
 "response_raw_len_chars": 251,
 "finish_reason": "stop",
 "kept": true,
 "drop_reason": null
}
```

LMSYS rows: digest-only — real-user corpus (content hygiene): no LMSYS prompt or completion text is reproduced here. All 5 models × 2 renders × 5,000 rows are at [raw_completions/generation @ 8c54f9fc](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8c54f9fc2b6c8b2cb3a2cdc256c1e38a8ff3a217/issue1336_rlvr_ladder/raw_completions/generation); the seed-42 spot check verified base-model LMSYS rows 204, 912, 1828, 2006, and 2253 structurally (kept flags and finish reasons consistent; one finish-reason-length row at 4,463 characters matches the 20.6% base-model truncation audit). Zero fishy rows in 5 GSM8K + 5 LMSYS spot-check rows.

Verifier WARNs acknowledged: with five results the total Takeaways + Goal + Results prose exceeds the 800-word skim budget; the λ-edge figure's rendered legend carries run-internal corpus slugs (decoded in its caption).

