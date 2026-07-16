# Fixed-template slot with nonzero R² = extraction-numerics leakage; probe with torch.unique

#1345 r2: the chat/no-template PREFIX slot is a fixed 3-token template region
(identical tokens + positions every conversation, causal forward ⇒ activation
should be constant), yet its within-R² read 0.13 vs mean baseline ≈ −0.001.
One turnstore shard resolved it: `torch.unique(X_prefix, dim=0)` → 14 distinct
vectors across 500 conversations (cos-to-mean ≥ 0.999994, norm ~14k), counts in
batch-size multiples that track sequence-length buckets — the "map" reads
length-sorted extraction-batch numerics, not content. Recipe: for any slot
whose nominal token content is conversation-independent, download ONE shard
(~2 GB) and check distinct-row count + cos-to-mean BEFORE narrating its R².

Also from the same round: (a) a dimension-matched FULL-RANK random-projection
control (X@P) preserves all linear info and matches observed R² in EVERY cell —
it is a pipeline sanity check, never degeneracy evidence; (b) prediction-target
cosine needs a train-fold-mean-cosine floor (story cell: obs 0.78 < floor 0.85
⇒ "right direction, wrong scale" was wrong) — computable in minutes from the
L19 preds_cache npz (pred/true/conv_ids) + `fc._cv_folds`.
