# Methodology — issue 2552: Turn-averaged SAE feature predictability (Der et al. replication, category-level map reads, partialed covariate ladder)

**Design:** three turn-averaged dictionaries are read with the same banked map: a fresh replication of Der et al.'s flat BatchTopK recipe (a sparse autoencoder keeping the top k=128 activations per batch, 32,768 features), plus the parent matryoshka k=100 and k=200 dictionaries (65,536 features, reused weights and banked per-feature R²). A public per-token dictionary (andyrdt trainer_2, 131,072 features, k=128, layer 19) instantiates the paper's per-token pole under max and sum pooling. Three legs: (1) judge-assigned schema categories aggregated over per-feature held-out map R²; (2) the paper's full judged evaluation (feature descriptions → per-turn 24-field structured summaries → 10-way matching → pairwise coverage → 5-way ranking → embedding coverage); (3) a forward-selection covariate ladder. Named deviations from the paper: assistant whole-answer means including the end-of-turn tail (theirs: assistant-turn token means), 963,444 banked rows vs their ~1.58M, pooled LMSYS+WildChat eval with the pooled read pinned in the plan as the verdict carrier (theirs: LMSYS only; our LMSYS-only subset read is advisory), a third-party per-token comparator with top-100 equal-length list truncation, and a single training seed. Every discrimination/coverage conclusion is scoped to this configuration bundle rather than to token-vs-turn grain per se.

**Training:**

| Hyperparameter | Value | Source |
|---|---|---|
| Replication SAE architecture | BatchTopK, width 32,768, k=128 | arXiv 2606.28548 App. A |
| Input | layer-19 whole-answer means, Qwen2.5-7B-Instruct (d=3,584) | banked #779/#2476 store |
| Training rows / epoch | 933,444 (963,444 store minus 10,000 val minus 20,000 holdout) | #2476 split pins |
| lr / batch / epochs / Adam betas | 2e-4 / 256 / 3 / 0.9, 0.999 | arXiv 2606.28548 App. A |
| Threshold EMA | 0.999 | #2476 `train_log.json` cfg |
| Seed | 2552 | run config |
| Realized holdout variance-FVE / nMSE | 0.9222 / 0.0778 (paper reports nMSE 0.097) | `p1/regime_measured.json` |
| Matryoshka k=100 / k=200 | reused parent weights, no retraining | #2476 |
| Ridge map | banked dense fit; per-dictionary encodes of prediction and target; corpus-transfer refit λ grid 23 values 1e-3 to 1e8, validation-selected (selected λ=1000, not grid-edge); fit n=120,000 ≫ d=3,584 | #2476 recipe; `p1/corpusfold_rep.json` |
| Map-fit val / test split | 400 / 1,000 rows (λ validation-selection / held-out test carves of the banked split family; re-asserted by sha at assembly) | #2476 split pins (`split_indices.npz`) |
| Per-feature panel cap | banked matryoshka panels: tier-stratified cap 16,384, seed 14824 (parent recipe); fresh replication panel: cap 12,000, seed 2552, not binding (2,974 alive features selected) | #2476 recipe; `p1/panel_rep.json` |

No LLM fine-tuning anywhere in this task.

**Evaluation:** DV1 = per-feature held-out R² of the SAE-encoded map prediction against the SAE-encoded true answer mean, aggregated per category; the primary category read is activity-adjusted (equal-weight aggregation of within-activity-quintile category effects, 10,000-draw bootstrap intervals; firing floors 0.2% primary, 1% robustness). DV2/DV3 = the paper's judged 10-way matching (can a turn be picked out of a 10-turn lineup from its feature descriptions; chance 0.10) and pairwise coverage (which feature list describes more of what the turn is doing), paired per turn over the same 2,000 eval turns, with score-interval whiskers and a 10,000-draw paired bootstrap on the gap. DV4 = ladder partial R² (the share of still-unexplained variance a covariate adds) on step-0 residuals, with per-draw same-selection permutation null bands within activity quintiles. The judge-free embedding read uses Qwen3-Embedding-8B (the paper's embedding model) to embed feature descriptions and per-turn summary field values, scored as mean top-3 cosine per configuration. Mapping baselines for the fresh map read (plan-pinned pair): identity+learned-bias per-feature median R² −2.57 vs map 0.418 (map higher on 100% of the 2,974 panel features); train-mean null median ≈ −0.005; 20-draw row-shuffle floor max 0.054; retrieval of the map prediction among the 20,000-row holdout pool acc@1 0.745 cosine / 0.689 euclidean (chance 5e-5; identity+bias acc@1 0.306 cosine). Matryoshka baselines are banked in the parent. Judge: claude-sonnet-4-6, no assistant prefill, Anthropic Batch API, all five pilot gates PASS (zero truncation, parse-fail < 2% per arm, api-refusal < 0.10), every production wave above the 0.95 per-item completeness floor. Judge max_tokens: 2,048 for the structured summaries, 1,024 for every other wave (descriptions, category assignment, matching, coverage, ranking, and the pinned-judge calibration; realized values in the committed per-wave `judge_meta` files). Judge calibration (control): the pinned claude-sonnet-4-5-20250929 re-judged 200 items per instrument; raw agreement 0.515 (category assignment), 0.580 (matching), 0.653 (pairwise); per-cell n is thin, so these are descriptive only. Two additional committed figures at the same pin are not embedded: the category-assignment drop rates by status class and the shadow category ranking with status groups, `category_drop_rates.png` and `category_status_groups.png` in [figures/issue_2552](https://github.com/superkaiba/explore-persona-space/tree/cb39df3ce1cd40aee3971faafb30f16913129635/figures/issue_2552). Comparator reconstruction context: trainer_2 on-corpus FVE 0.831 vs the replication SAE's 0.922 holdout variance-FVE.

**Data extraction:** eval turns = 2,000 drawn from the 20,000-row holdout (1,098 LMSYS, 902 WildChat), disjoint from every mining pool (direct overlap check 0 in `judge_aggregates/mfa_disjointness.json`). Description mining: top-25 activating turns per feature from the 120,000-row SAE-fit pool (turn-averaged families) or the 18,000-row non-eval holdout pool (per-token family). Uploaded and judged mining text carries same-length placeholders at 16 corpus-resident secret-shaped spans (`scripts/scrub_secrets.py`); the sha-pinned raw input mirrors are untouched. 74 union-only features with zero mining-pool firings were dropped from description coverage (10 of 4,875 replication, 51 of 4,610 k=100, 12 of 4,192 k=200, 1 of 28,806 per-token; ids in `judge_aggregates/w1_mining_coverage.json`); all 74 are outside the analysis panels, so no panel read is affected. The measured description-union total (42,483) stayed under the 45,000 descope cap, so the full 2,000-turn eval ran.

**Sample training/evaluation data + completions:** the judged text is real LMSYS/WildChat user-assistant turns (data-realism tier 1); excerpts below are judge OUTPUTS (feature descriptions and summaries), quoted verbatim; any embedded mining text was judged with the 16-span placeholder substitution above. Complete artifacts: [judge aggregates (git, pinned)](https://github.com/superkaiba/explore-persona-space/tree/cb39df3ce1cd40aee3971faafb30f16913129635/eval_results/issue_2552/judge_aggregates) and [raw judge requests/responses (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/fdcec4c823e2638ae8661ccafca8f30f84ac6233/issue2552_turnsae/raw_completions/judge).

<details>
<summary>Replication-dictionary feature descriptions: 3 well-predicted and 3 poorly-predicted features</summary>

Random sample (seed 42) within the top and bottom per-feature R² deciles; all 4,860 valid descriptions (5 dropped): [descriptions_rep_ta.json](https://github.com/superkaiba/explore-persona-space/blob/cb39df3ce1cd40aee3971faafb30f16913129635/eval_results/issue_2552/judge_aggregates/descriptions_rep_ta.json).

- Well-predicted, feature 1105 (R² 0.735, Function): "This feature activates on assistant responses that explain or clarify the answer to a reasoning or factual question, often involving step-by-step logical breakdowns, spatial/relational tracking, or technical explanations"
- Well-predicted, feature 3152 (R² 0.723, Form): "This feature activates on Chinese-language text related to Chinese Communist Party (CCP) organizational work, including party building (党建), discipline education, work summaries, ideological study campaigns, anti-corrupt[ion...]"
- Well-predicted, feature 26425 (R² 0.774, Content): "This feature activates on text describing the geographic location, administrative status, or notable characteristics of cities, towns, and regions — particularly responses to 'where is X located' or 'what is the capital [...]"
- Poorly-predicted, feature 24225 (R² 0.043, Form): "This feature activates on text that mixes multiple languages within a single response or passage, particularly content that combines scripts such as Arabic, Chinese, Russian, Japanese, Korean, or other non-Latin writing [...]"
- Poorly-predicted, feature 30271 (R² 0.039, Content): "This feature activates on technical content related to fluid dynamics and heat/mass transfer, including discussions of Navier-Stokes equations, Reynolds numbers, friction factors, turbulent and laminar flow, pipe flow, d[...]"
- Poorly-predicted, feature 4643 (R² 0.021, Content): "This feature activates on content related to medical specialties, healthcare providers, and the appropriate routing of patients to specific types of doctors or clinics (e.g., ophthalmologists vs. optometrists, which spec[...]"

</details>

<details>
<summary>Judged 10-way matching rows</summary>

Random sample of 5 rows (seed 42); all 10,000 rows: [matching_perturn.json](https://github.com/superkaiba/explore-persona-space/blob/cb39df3ce1cd40aee3971faafb30f16913129635/eval_results/issue_2552/dere_repl/matching_perturn.json).

- turn 892318, matryoshka k=100 list: gold B, judge chose B (correct; 100 descriptions in list)
- turn 262573, matryoshka k=100 list: gold B, judge chose B (correct; 71 descriptions, feature list shorter than the 100 cap)
- turn 305489, per-token max list: gold E, judge chose E (correct)
- turn 105575, per-token max list: gold E, judge chose E (correct)
- turn 815042, matryoshka k=200 list: gold I, judge chose H (incorrect; 1 of 100 descriptions missing)

</details>

<details>
<summary>One per-turn structured summary</summary>

Cherry-picked (first item, for brevity); the 1,970 valid summaries (of 2,000 requested): [summaries_2000.json](https://github.com/superkaiba/explore-persona-space/blob/cb39df3ce1cd40aee3971faafb30f16913129635/eval_results/issue_2552/judge_aggregates/summaries_2000.json); per-item raw judge outputs: [raw_completions/judge/w2 (HF, pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/fdcec4c823e2638ae8661ccafca8f30f84ac6233/issue2552_turnsae/raw_completions/judge/w2).

Turn 239783: domain "Business and commerce"; topic "Corporate profile of a Chinese chemical import/export company"; factuality "Presented as factual but reads as promotional/marketing material with unverified claims"; concreteness "Moderately concrete with specific address and product categories, but vague on details" (20 further fields omitted here).

</details>

*Derived from the [task body](https://eps.superkaiba.com/tasks/2552).*
