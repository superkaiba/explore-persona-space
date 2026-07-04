# Methodology — issue 923: context/query decomposition of the mean answer activation (disjoint last-context / last-query ridge read-outs vs the full-prompt read; no training)

**Design:** One base model, `Qwen/Qwen2.5-7B-Instruct`, forward passes and greedy generation only; ridge read-outs only, no model training anywhere. Grid: 50 contexts in 7 families (persona 14 / WildChat 10 / in-context-learning 8 / rephrase 6 / format 5 / behavior 5 / default 2) × 144 UltraChat queries (the 48 store queries + 96 fresh length-matched ones), plus two secondary grids — 50 × 48 Betley (misalignment-genre) queries and 50 × 48 human-written Dolly queries (the out-of-distribution arm). Target: the per-cell mean residual activation over the answer tokens of the model's own greedy completion to the full (context, query) prompt, per layer (28 layers). Read-out arms, per layer: Context-only (last context token from a prefix-only forward), Query-only in three null-context presentations (explicit empty system turn; no system block; full prompt with context tokens attention-masked in place), Stitched pair (feature concatenation of Context-only + Query-only, one variant per presentation), Blended predictions (two-parameter combination of the two single-arm predictions, fit on an inner validation split), and Full prompt (last input token at the assistant header). Contexts and queries occupy disjoint token spans, and each partial feature is computed with the other input absent. References: an in-sample variance decomposition of the target into context main effect, query main effect, and interaction (oracle ceilings); selection-symmetric permutation nulls; a 50-cell regeneration spot-check of the store join.

**Training:** **N/A — no model training.** Analysis-design constants:

| Parameter | Value | Source |
|---|---|---|
| Ridge λ grid | 1e-2, 1e-1, 1, 10, 100, 1000; nested PRESS leave-one-out per train fold (follow-up diagnostic: leave-query-group-out re-selection at layer 18, masked-context arms) | #722/#823 harness (`RIDGE_LAMBDAS`) |
| Feature/target reduction | PCA-48 both sides, train-fold-fit bases; train-fold-centered targets | #722 (28/28-layer input-PCA equivalence, max ΔR² 0.018); #810 target recipe |
| Layer binding | same-layer, swept 0–27; headline layer 18 frozen in the plan | #722 peak layer |
| Folds | 7 leave-one-family-out × 4 query folds (UltraChat 144 → 4×36; Betley 48 → 4×12); both axes unseen in every test cell | #810 + the project's group-fold rule |
| Out-of-distribution fold | train non-held family × all 144 UltraChat queries; test held family × 48 Dolly queries | plan §4.2 (corpus-transfer form) |
| Generation | vLLM greedy, temperature 0.0, max_tokens 512 | #658 store recipe, matched exactly |
| Permutation null | 1000 cell-label permutations; λ re-selected per draw; per-draw max-over-layers band | #810 batched-null design |
| Bootstrap | 2000 family-cluster draws; cross-classified family × query draws for the context-vs-query contrast | #810 precedent; plan §3 |
| Decomposability-fraction guard | denominator floor max(0.02, 2 × bootstrap SE) | plan §3 (from the #722 compression bound 0.018) |
| Masked-context attention backend | SDPA with a 4D float mask (dummy-context invariance smoke passed) | plan §8 |
| Seed | 42 (probe build, folds, permutations, bootstrap) | plan §10 |

**Evaluation:** Primary dependent variable = pooled held-out skill-over-mean R² per (arm × layer): fits on train-family contexts × train-fold queries, scored on held-out-family contexts × held-out-fold queries, pooled so every cell is tested exactly once (n = 7200 UltraChat cells, 2400 Betley, 2400 Dolly; no cells dropped). Baseline for skill = the train-fold mean. Headline reads at the plan-frozen layer 18; the layer sweep is gated on per-draw max-over-layers permutation bands. The interaction residual is the full-prompt skill minus the stitched-pair skill; the plan's power floor (skip verdicts if the full-prompt read is below 0.05 everywhere) did not trigger. The construct is representation-level — linear predictability of the model's own answer-side activation — not a behavioral claim; the variance shares are in-sample references, never held-out claims, and ambient-space skill is reported alongside as the plan's registered secondary robustness read. R² magnitudes are not comparable to the parent issues' probe-averaged numbers (different target granularity and fold scheme).

**Data extraction:** Targets for the 48 original UltraChat and 48 Betley queries are re-reductions of the parent store's per-context answer-span tensors (~341 GB streamed and reduced on a CPU instance); targets for the 96 fresh UltraChat and 48 Dolly queries were captured fresh with the identical recipe (greedy generation, teacher-forced forward, mean over the answer span). The Context-only feature comes from a prefix-only forward whose equality with the same position inside the full prompt was probe-verified per context (cosine floor 0.99 enforced; values below 0.999 recorded as warnings — bf16 batching numerics, worst recorded 0.9989). The empty-system presentation asserts the Qwen default system prompt was not silently inserted. Fresh UltraChat queries were length-matched to the Betley length profile with the parent's builder; Dolly rows are instruction-only (empty context field), same filters, length-matched, from the established databricks-dolly-15k corpus (tier 1–2 realism); UltraChat is an established tier-2 corpus; the battery inherits the line's house-written tier-3 families as a standing scope caveat.

**Sample training/evaluation data + completions:** No training rows exist; the worked examples below are evaluation cells (context + query + the model's own greedy completion whose activations form the target). Completions are cut for space; full texts for all 7250 new-arm completions (4800 UltraChat-extension, 2400 Dolly, 50 regeneration spot-check): [raw_completions on the HF data repo (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/77d04e45096e120ca897fdb7b22730e157ce00a2/issue923_ctx_query_decomposition/raw_completions).

Randomly drawn UltraChat-extension cell (from `uc_ext/f1_house_programmer_shard2of4.json`, row with q_idx 17):

> **Context (system prompt):** "You are a programmer."
> **Query:** "Create a playlist of at least 10 songs that feature unusual or unique instruments, such as the sitar, didgeridoo, kalimba, or theremin, across a variety of music genres like folk, world music, indie, and experimental."
> **Completion (opening):** "Certainly! Here's a playlist featuring unique instruments across various genres: 1. **"Sitar Solo" by Ravi Shankar** - Folk/World Music …"

Format-family cell, the family the full-prompt read transfers worst to (from `uc_ext/f5_fmt_code_comment_shard3of4.json`, row with q_idx 0):

> **Context (system prompt):** "Respond to every question as a block of code comments: each line of your answer must start with '# '."
> **Query:** "Who won the men's singles title in the French Open in 2019, and what was the score in the final?"
> **Completion (complete, 2 comment rows):** "# The men's singles title in the French Open in 2019 was won by Rafael Nadal." / "# The score in the final was Nadal 6–3, 6–3, 6–0 against Dominic Thiem."

Randomly drawn out-of-distribution Dolly cell (from `ood_dolly/f6_helpful_asst_shard0of4.json`, row with q_idx 47; the query typo is verbatim from the corpus):

> **Query:** "Tell me which of these are shades of Blue that BWM paints their cars: Miami Blue, Laguna Seca Blue, Le Mans Blue, …"
> **Completion (opening):** "BMW has used several shades of blue in their car paint schemes over the years, but not all of the shades you listed are official BMW colors. …"

Query-only presentations, rendered verbatim for one Dolly query — the explicit-empty-system form first, then the hand-rendered no-system-block form (the third presentation keeps the full token sequence and attention-masks the context span in place):

```
<|im_start|>system
<|im_end|>
<|im_start|>user
Why do home power outages occur?<|im_end|>
<|im_start|>assistant

<|im_start|>user
Why do home power outages occur?<|im_end|>
<|im_start|>assistant
```

Betley-genre queries are the published misalignment-evaluation question pool inherited from the parent store; per the project's content-hygiene rule they are referenced by artifact only (48 rows per context in the parent store's raw-completions records; this run's regeneration texts under `regen_check/`, 25 cells per genre), not quoted.

*Derived from the [task body](https://eps.superkaiba.com/tasks/923).*
