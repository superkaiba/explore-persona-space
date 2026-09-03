# Task #2661 — deviations from the reference recipes (brief rule: record every one)

1. **MLP learning rate 1e-3.** The brief pins architecture (2-layer, hidden
   4,096, GELU), optimizer (Adam), batch (1,024), early stop and seed, but not
   the lr. 1e-3 chosen (torch Adam default family for MLP heads); logged in
   `map_mlp_metrics.json`.
2. **G1 halt basis.** The task body says "Halt if holdout variance-FVE < 0.5";
   the parent #2552 gated on the 10k SAE-val carve. This driver GATES on the
   20k-holdout variance-FVE (the task body's words) and logs the SAE-val FVE
   beside it for #2476/#2552 parity.
3. **Smoke stand-ins (loud, smoke-only).** `--smoke` substitutes (a) a tiny
   deterministic flat SAE for the 940 MB banked #2552 answer SAE, and (b) a
   shape-faithful synthesized npz (`holdout_pred16`/`holdout_rows` keys) for the
   143 MB banked dense-map refit. Production fetches both at their pins and
   schema-asserts immediately after load. The real-fetch legs therefore first
   execute on the pod (called out in the implementation report).
4. **"16-span placeholder substitution" (brief, mining phase).** No such
   placeholder exists anywhere in the #2552 W1 path (grepped `span|placeholder`
   over issue2552_judge_waves.py + issue2552_turnsae_der.py). The mining jsonl
   keeps the exact #2552 `top25_*.jsonl` shape (`family/feat_id/rank/row_id/
   activation/text`) extended with a `kind` field (`positive` / `negative` /
   `negative_lowest_activation`) for the task-mandated 20 non-activating
   negatives. Named as unresolved in the implementation report rather than
   guessed at.
5. **Pilots cover w2.** The brief says pilots run "before each production wave,
   as in #2552"; #2552 exempted sub-5k w2 (rule 26). Both readings are
   satisfied by piloting w1, w2 AND w4 (51 calls each — cheap), so no wave
   dispatches unpiloted.
6. **Judge estimate token model.** No tokenizer call: input tokens are
   chars/3.5 (conservative divisor, recorded in the JSON); the output side uses
   the max_tokens cap as an upper bound (the gate binds on the upper bound) and
   0.5x cap as the "expected" variant.
7. **TF32.** On CUDA, fp32 GEMMs (B/pred reconstruction blocks, the MLP) run
   under TF32 for wall-clock; Gram/XtY accumulation, eigh and Cholesky solves
   stay fp64. Coefficient-level effect is ~1e-3 relative, uniform across the
   observed fit, split halves and the label-shuffle null (the gate calibrates
   on the same numerics).
8. **Negatives sampling.** Non-activating negatives come from ONE shared seeded
   candidate pool (4,096 rows) encoded once, then per-feature seeded selection;
   a feature firing on every candidate falls back to the lowest-activation
   candidates, labelled `negative_lowest_activation`.
9. **W3 category assignment is OUT** (brief option): not cheap enough to
   justify — it needs its own pilot + ~need-set-sized wave; the dashboard's
   topic-vs-behavior split reads the W1 descriptions directly.
10. **Context-side watch list (review r1 Minor 2).** The task-body receipts
   paragraph pins a VM-side China/Taiwan/Xinjiang/Tibet/CCP context-feature
   join; round 1 shipped without it (unrecorded scope gap). Round 2 adds
   `--phase watchlist` to `scripts/issue2661_embed_and_dashboard.py`: regex over
   the judged W1 context descriptions, each hit's top-10 out-edges from
   `wiring_edges.npz` with answer-side descriptions, receipts-family flags, a
   committed `watchlist_context_features.json`, and dashboard section 4b. Zero
   pod/judge spend.
11. **`top_pairs.json` is display-capped (`--max-pairs`, default 500).** The
   mining need set reads the FULL surviving mask from `wiring_edges.npz`
   (review r1 Minor 4), so a >500 surviving set never truncates the need set;
   `top_pairs.json` stays the ranked display artifact.
12. **Sync re-issue cap + pricing (review r1 Minor 5).** Rule-28 sync re-issues
   are capped at max(51, ceil(0.05 x n_items)) per wave (the rule-29 floor
   tolerance); the estimate's upper bound now includes the capped re-issue set
   at NON-batch prices ($5/M in, $25/M out). Items censored beyond the cap stay
   censored and the completeness-floor gate arbitrates.
13. **Estimate-to-inputs binding (review r1 Minor 3).** `judge_estimate.json`
   records `manifest_identity_sha256` (sha over eval_ids_sha256 +
   need_set_sha256); every production wave/pilot refuses dispatch (rc=9) when
   the current prep manifest's identity differs from the estimate's.
