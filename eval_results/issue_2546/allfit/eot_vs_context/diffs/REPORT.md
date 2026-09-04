# Context state vs end-of-thought state, and pre vs post context states

Setup: OpenThinker3-7B (post) and its parent Qwen2.5-7B-Instruct (pre), arm 1, layer 19, 30,193 deduplicated questions from seven datasets, one greedy rollout each, production 5 folds reused for every fit. Ridge standardizes inputs and centers outputs (float32 products, float64 solves). Production penalties: metamodel A (context state to answer state) 3162, metamodel D (end-of-thought state to answer state) 316, W_pre (pre context to pre answer) 1000. Sweeps pick from 100/316/1000/3162 by held-out R². Cross-application retrieval: whitening from training-fold answer states (shrinkage 0.1), CSLS k=10, pool = the held-out fold's ~6,039 true answer states, hit = nearest is own. Terms: context state h_cx = last context-token state, end-of-thought state h_eot = state at the `</think>` token, answer state h_ans = mean over answer tokens, effective rank = components holding 90 percent of variance. Every number below comes from diffs.json, figures sit in figures/issue_2546/eot_diffs/.

## Part A (post model)

**A1. Thinking moves the state far, but almost all of the movement is one shared offset on a few massive coordinates.** Median relative shift ||h_eot − h_cx|| / ||h_cx|| is 0.998 and median cosine 0.45. A shared mean offset (norm 22,475) carries 99.97 percent of the squared norm of d = h_eot − h_cx, and 93 percent of that offset sits in three coordinates (458, 2570, 2718), the massive-magnitude coordinates that also dominate the context states. The question-specific remainder has effective rank 1, and 98 percent of it lies in the top 50 principal components of h_cx (chance 1.4 percent).

**A2. The end-of-thought state is only partly linear in the context state, and the unpredictable part carries answer information only jointly.** The ridge map h_cx to h_eot reaches out-of-fold R² 0.824 against the pooled mean and 0.475 against the per-dataset mean, retrieval acc@1 0.41 (chance 0.0002), and the identity-plus-bias baseline collapses (pooled −53.9, table 1). Decomposing h_eot into predicted part plus residual r: ridge from r alone to h_ans gets R² −0.81 against the dataset mean, while [predicted, r] reaches 0.650, above metamodel A's 0.463 and matching metamodel D's 0.648. So the gain rides on information the context state does not linearly carry, yet it decodes only jointly with the context-predictable component.

**A3. The two metamodels write the same output directions, read unrelated input directions, and are not interchangeable.** Raw operator cosine 0.047 versus a random-rotation null of 0.0001 (sd 0.0003, 12 draws). Procrustes alignment (best orthogonal rotation) of the input basis alone lifts the cosine to 0.895, of the output basis alone to 0.311. The singular-spectrum cosine (the two-sided rotation optimum) is 0.981, but it is rotation invariant and cannot support a "same operator up to rotation" claim. Left singular subspaces (output directions written) overlap at mean principal cosine 0.67 / 0.80 / 0.85 for k = 10 / 50 / 200 versus null 0.04 / 0.10 / 0.20, while right singular subspaces (input directions read) sit at the null. Effective ranks 333 (A) and 522 (D). Cross-application is catastrophic both ways (table 2). Recomputed own retrieval matches the recorded hits (0.9531, 0.9936).

## Part B (same prompts through both models)

**B1. Reasoning training rebuilt the context representation along new directions and only nudged the answer representation along existing ones.** Context side: median relative shift ||Delta|| / ||h_cx(pre)|| is 236, because the post model places the massive coordinates (offset norm 22,482) at the last context token where the pre model has none. The mean offset explains 99.97 percent of the squared shift, global scaling adds nothing, the question-specific remainder has effective rank 1, and only 26 percent of Delta lies in the top 50 principal components of h_cx(pre) (chance 1.4 percent). Answer side: median relative shift 0.58, median cosine 0.85 (0.47 on the multiple-choice datasets, which changed most). Split: 38 percent offset, 34 percent global scaling (scale 0.23), 28 percent question specific at effective rank 202, with 74 percent of the centered shift inside the top 50 principal components of h_ans(pre).

**B2. The post context state is about half linearly recoverable from the pre one.** Ridge R² 0.649 pooled and 0.510 against the dataset mean, identity plus bias −0.010 and −0.408, retrieval acc@1 0.113 (cosine, chance 0.0002) (table 1).

**B3. The movement avoids the directions W_pre reads, and its size still swamps them.** Share of ||Delta||² in the top-k right singular subspace of W_pre: 0.0032 / 0.0119 / 0.0498 at k = 50 / 200 / 1000, versus the isotropic matched-norm null k/3584 = 0.0140 / 0.0558 / 0.2790, and the question-centered part behaves the same (0.0038 / 0.0136 / 0.0552, table 3). The shift concentrates 4 to 6 times more strongly outside W_pre's read directions than a random shift of equal size, so the known ladder failure of W_pre on post states does not come from movement inside what W_pre reads. It is a magnitude effect: the question-specific part of Delta has per-question norm near 390 against a median pre context norm near 95 (both derived from the split and shift entries), so the in-subspace movement at k = 1000 already equals the pre-model signal and a bias offset removes only the shared offset.

## Tables

Table 1. Linear maps (out of fold, selected penalty in parentheses).

| map | R² pooled mean | R² dataset mean | identity+bias (pooled / dataset) | acc@1 cosine (chance) |
|---|---|---|---|---|
| h_cx → h_eot (3162) | 0.824 | 0.475 | −53.9 / −163.2 | 0.407 (0.0002) |
| r → h_ans (3162) | 0.000 | −0.808 | n/a | n/a |
| [pred, r] → h_ans (1000) | 0.806 | 0.650 | n/a | n/a |
| h_cx(pre) → h_cx(post) (316) | 0.649 | 0.510 | −0.010 / −0.408 | 0.113 (0.0002) |

Table 2. Operator cross-application (out of fold, production penalties).

| condition | R² dataset mean | top-1 whitened CSLS |
|---|---|---|
| A on its own context inputs | 0.463 | 0.9531 |
| D on its own end-of-thought inputs | 0.648 | 0.9936 |
| A applied to end-of-thought inputs | −57.6 | 0.0007 |
| D applied to context inputs | −77,076 | 0.0002 |

Table 3. Share of ||Delta||² in the top-k input directions of W_pre.

| k | raw Delta | question-centered Delta | isotropic null k/3584 |
|---|---|---|---|
| 50 | 0.0032 | 0.0038 | 0.0140 |
| 200 | 0.0119 | 0.0136 | 0.0558 |
| 1000 | 0.0498 | 0.0552 | 0.2790 |

## Assumptions

- All 30,193 fitted rows have pre-model states, so Part B uses the full set with the same folds.
- Grams and predictions run in float32 with float64 solves and accumulation. Fourth-decimal drift is possible.
- The selected penalties for h_cx → h_eot and r → h_ans sit at the sweep edge (3162), so heavier regularization might score higher.
- Table 2 own-score rows reuse the stored production out-of-fold predictions, and the [pred, r] fit stacks such predictions as features.
- Raw-coordinate reads in A1 and B1 are dominated by a few massive-magnitude coordinates, named in the diagnostics block of diffs.json.
