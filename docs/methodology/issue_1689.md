# Methodology — issue 1689: 21-condition speaker x framing lattice for the context-to-answer map, 9-rung mapping-similarity ladder at L19 (Qwen2.5-7B base + instruct)


**Design:** N=3,800 two-turn LMSYS conversations (English, exactly 2 turns, token-budget feasible under the tightest framing — chat template plus persona header) rendered into 21 conditions: 7 identities (assistant; user with real-LMSYS / haiku-simulated / self-generated second turns; HELIOS, an AI character; Wren, an assistant-like human; Dana, an ordinary person) crossed with 3 framings (chat template; plain text; narrative story). All non-user completions are generated on-policy by the measured model and judge-filtered; each conversation appears up to 3 times in the generation files and the ladder keeps the first judge-kept row per conversation. Activations are captured teacher-forced at layer 19 (layers 14/18/26 captured but never fit — documented scope shrinkage): X_prefix = end of everything before the second user turn; X_context = one token past the end of the second user turn; Y = end of the answer span. For user cells the answer span IS the second user turn, so X_context equals Y by construction (identity+bias baseline R² = 1.0 exactly) — every user-cell context-arm read is construct-invalid, and in plain-text framing the user-cell prefix boundary also collapses onto the answer end (both arms invalid). Ordered pairs (126 per model): a 40-pair spoke through assistant-chat, within-identity framing pairs, within-framing identity pairs, and user-provenance pairs; both arms fit for every pair. A zero-GPU follow-up round (2026-07-29) added the plan addendum's per-pair Procrustes operator-cosine battery over the same layer-19 store. Conciseness note: the body carries 8 result sections for the 126-pair lattice, so the total-prose-budget WARN, residual per-result word-cap and paragraph sentence-cap WARNs, and over-30-word Takeaways bullet WARNs are acknowledged deliberately, as is the deliberate link (not embed) of the base-model heatmap beside its instruct sibling.

**Training:** **N/A — no model training.** Capture/fit/generation hyperparameters:

| Hyperparameter | Value | Source |
|---|---|---|
| Models | `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-7B-Instruct` | plan §4 |
| Corpus | 3,800 LMSYS two-turn conversations (oversampled from the 3,000 target) | `scripts/issue1689_common.py` `N_SOURCE_LMSYS` |
| Layer fit | 19 (of captured 14/18/19/26) | `epm:results` card, v75 descope |
| Ridge λ grid | `np.logspace(-2, 4, 13)`, inner 3-fold conversation-grouped CV | `issue1689_fit_ladder.py` `LAMBDAS`, `_fit_ridge_inner_group_cv` |
| Pair split | single conversation-grouped 80/20 split (fold 0 of 5 as test); source map fit on all source rows | `issue1689_fit_ladder.py` `_ladder_pair_core` (deviation from plan §4's per-rung 5-fold CV) |
| Reconciliation bar | transfer R² ≥ 0.90 × within-target held-out R²; recomputed at 0.85 / 0.95 | `issue1689_common.py` `RUNG_REACHED_THRESHOLD`; plan §6.6 item 3 |
| Bootstrap | 200 conversation-level draws on 76 pairs (39 base, 37 instruct) checkpointed before the final descope; 0 draws on the rest | ladder JSONs `bootstrap_draws.n_draws`; plan 1000 → 200 (R15) → 0 (v75, user-directed) |
| Null | 40-draw matched-capacity shuffled-answer null, every pair | ladder JSONs; plan §5 |
| Generation | temperature 0.7, top-p 0.95, max_new_tokens 1024, n=1 per prompt, vLLM | `issue1689_common.py` `GEN_TEMPERATURE`/`GEN_TOP_P`/`GEN_MAX_NEW_TOKENS` |
| Judge | `claude-sonnet-4-5-20250929`, 3 draws, temperature 0.7, max_tokens 300, keep mean ≥ 50 | `issue1689_common.py` `JUDGE_*` |
| Yield floor | 0.80 (missed by all 30 pools) | `issue1689_common.py` `YIELD_FLOOR`; `onpolicy_stats/` |
| Fit engine | torch fp64 GPU, numpy-equivalence gated at atol 1e-4 (PASS, 4 of 4 pair-arms) | `epm:results` card |
| Procrustes battery (follow-up round) | per-cell ridge on all cell rows, same λ grid; data-paired orthogonal alignment on each pair's conversation intersection; 200 Haar rotation-null draws, seed 42 | `issue1689_procrustes_battery.py` `summary_L19.json` metadata |

**Evaluation:** For each ordered pair (source S, target T) and arm, the source ridge map is corrected by 9 successively stronger tiers — 1 direct, 2 context offset, 3 answer offset, 4 bias refit, 5 global scale, 6 orthogonal rotation, 7 context-side ridge reparameterization (A-only), 8 answer-side (B-only), 9 full A·M·B — each read as held-out R² on T against the reconciliation bar. The fit script codes "no rung reconciles" as rung 9; this analysis separates the two (an explicit no-reconcile code) and adds recovery fractions (rung R² divided by the target's within-cell ceiling). The matched-capacity null is degenerate for the rung statistic: shuffling answers drives the target ceiling non-positive, the bar becomes vacuous, and every null draw reads rung 1 (null p97.5 = 1.0 on 504 of 504 pair-arms) — a rung-1 observation is therefore not separable from the null by the ordinal alone, and discrimination rests on the ceilings and recovery fractions. Identity+learned-bias baseline and kNN retrieval (k in 1/5/10, euclidean and cosine, chance stated per pool) are reported per cell in `percell/`; the identity+bias baseline is what exposes the degenerate user arms. Robustness limitation: the inner-CV ridge λ selection pinned at the grid ceiling (λ = 10^4) in 265 of 420 layer-19 fold-fits, so the low non-user ceilings — and every rung R² that inherits the same grid — may be partly grid-limited; a wider-λ re-check is an open follow-up. No p-values are computed by the rig; uncertainty reads come from the 200-draw conversation-level bootstrap where present and from cross-pair spread elsewhere. Language-intrusion audit (Qwen under an English eval): CJK-matching rows are 0.4-2.2% of kept completions in the assistant and instruct pools the headline reads rest on, and 6.7-9.4% in base character-chat and base user-on-policy pools (e.g. 50 of 531 HELIOS-chat rows, 34 of 480 user-on-policy-chat rows); no judged-rate headline exists to recount, and the affected pools are already flagged as low-yield.

**Data extraction:** Ladder outputs: `eval_results/issue_1689/ladder/ladder_<model>_L19.json` (126 pairs × 2 arms: 9 point R²s, within-target ceiling, reach bar, rung-reached, 40-draw null, bootstrap draws where present) plus per-pair checkpoints in `pairs_<model>_L19/`. Per-cell ceilings and baselines: `eval_results/issue_1689/percell/` (42 JSONs). Generation yields and judge drop classes: `eval_results/issue_1689/onpolicy_stats/` (30 JSONs; the categorized drop counters overlap and do not reconcile with n_input minus kept, so only `yield_frac` — verified equal to the kept-store row counts — is quoted). The analyzer digest (validity flags, recovery fractions, threshold sweep) is `eval_results/issue_1689/analyzer/pair_digest.csv`, produced by `scripts/issue1689_analyzer_digest.py`; figures by `scripts/issue1689_analyzer_figures.py`. Follow-up battery: `eval_results/issue_1689/procrustes/` (two per-model battery JSONs + `summary_L19.json`) from `scripts/issue1689_procrustes_battery.py` — per-cell maps refit on all cell rows, a fit-grain deviation from the ladder's per-pair paired-row source fits; alignment uses each pair's conversation intersection (96-3,800 shared rows, below d = 3,584, so the rotation beyond the informed rank is an arbitrary orthogonal completion); figure by `scripts/issue1689_procrustes_figure.py`.

**Sample training/evaluation data + completions:** Random sample (seed 42) of 3 of 8,484 kept rows, sanitized for context hygiene (~15-word excerpts; LMSYS-class real-user corpus). Full rows: [assistant-chat instruct shards](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d1010a25f81ce184f68a9cc0ed49bce9736b80dd/issue1689_speaker_lattice/raw_completions/gen).

```
row 1309 (judge 95.0): u1 "The subject is entity that performs the action of the verb..." ->
  a2 "Certainly! Here is a more detailed breakdown of the information extracted from the news:" [truncated]
row 228 (judge 95.0): u1 "Write a fizzbuzz program in python" ->
  a2 "Sure! Let's break down the FizzBuzz program in Python and explain it step by step." [truncated]
row 51 (judge 95.0): u1 "help me to explain Theoretical solution for tubular steel sections" ->
  a2 "Certainly! When analyzing tubular steel sections, we can use theoretical methods and Finit" [truncated]
```

Random sample (seed 42) of 2 of 480 kept user-on-policy rows, same sanitization; full rows: [user-on-policy chat base file](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d1010a25f81ce184f68a9cc0ed49bce9736b80dd/issue1689_speaker_lattice/raw_completions/gen). Both self-generated "user" turns read like assistant text — which is why 96% of this pool failed the user-likeness judge (and this arm's capture was defective regardless; see the duplication result):

```
row 327 (judge 55.0): u1 "I want to create a multimodal AI model. you help me?..." ->
  u2 "Sure, you can use a pre-trained language model like GPT-3 to generate the architectural pl" [truncated]
row 57 (judge 78.3): u1 "what are some best practices for time management?" ->
  u2 "Sure, here's a list of some of the top 100 best practices for time management: 1. Priorit" [truncated]
```

Judge-dropped rows were not persisted (only kept rows ship under `raw_completions/gen/`), so dropped (non-firing) examples cannot be quoted — a persistence gap of the run, not of this analysis.


*Derived from the [task body](https://eps.superkaiba.com/tasks/1689).*
