# Task #2661 — deviations from the reference recipes (brief rule: record every one)

1. **MLP learning rate.** The brief pins architecture (2-layer, hidden
   4,096, GELU), optimizer (Adam), batch (1,024), early stop and seed, but not
   the lr. 1e-3 was chosen in r1 and DIVERGED on the pod — superseded by 3e-4
   plus the target transform / clipping / zero-init head of entry 16; logged in
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
14. **fp64 accumulation for every CSR moment/parity read (r2 pod smoke fix).**
   scipy sparse `sum`/`mean` accumulate at the MATRIX dtype (float32 here); on
   the pod the ib parity assert sat at a measured max delta 6.6e-7 against the
   fp64 canonical helper and hard-failed its 1e-8 atol (deterministic; the CPU
   smoke passed only because the CPU-trained smoke SAE produced smaller
   activation sums). Fix is the CLASS, not the instance: `_csr_colsum64`
   (fp64 bincount) now backs `_col_moments_csr` (ridge standardizer, ymu, ib
   bias, edge-null y_sd), the raw-product column sums, the null-draw ymu_k, the
   null-SD reduction (`dtype=np.float64`), and the dense-input variance; the
   parity assert computes BOTH sides fp64-on-CPU, logs the measured max delta,
   and keeps rtol=1e-9/atol=1e-8 (residual is fp64 summation order, ~1e-13).
   TF32 (deviation 7) was NOT the mechanism — both assert sides were already
   CPU numpy — and stays enabled for fp32 GEMMs only.
15. **Launcher hardening from launch attempt 1 (r2 fixes a+b).** The pod
   bootstrap clone is a SPARSE checkout without eval_results; the launcher now
   detects sparse checkouts, adds the issue_1482/2476/2661 subtrees via
   sparse-checkout, and hard-asserts the four required committed inputs before
   any leg. The header's fictional by-issue sync subcommand is replaced by the
   real procedure (push the branch; pod-side fetch + checkout + ff-only pull).

16. **MLP target transform + lr 3e-4 + grad clip + zero-init head (r4
   divergence fix).** The production map_mlp leg trained MSE on RAW answer
   activations at Adam lr 1e-3 and diverged: measured val pooled R^2 -1019
   after epoch 1 (train mean-MSE 7.27 -> 0.218 by epoch 3 while val stayed
   -126/-83), best -27.9 at epoch 8, then oscillation (-81 at 10, -184 at 11),
   holdout -72 — vs the ridge's 0.637 on the SAME val rows. Mechanism: Adam's
   per-parameter steps are gradient-scale-free, so ~lr x steps of drift across
   the 15,216 -> 4,096 -> 32,768 head puts raw-unit predictions ~30x the target
   sd off-scale (sqrt(1020) ~= 32, matching the measured epoch-1 R^2). Fix
   (`_fit_mlp`): targets centered by the ridge standardizer's ymu and scaled by
   ONE global scalar sy (pooled train sd — global, not per-column, so the
   scaled MSE stays exactly proportional to the pooled-R^2 SS_res and the
   objective is NOT reweighted vs the eval metric); output head zero-init
   (epoch-0 prediction IS the train-mean baseline, R^2=0); grad-norm clip 1.0;
   lr 3e-4. Predictions un-transform to raw units (pred = model(x)*sy + ymu),
   so `pred_te_mlp.fp32.npy` keeps its units and every downstream consumer is
   unchanged. Guarded by `test_mlp_reaches_ridge_r2_on_linear_problem`
   (synthetic linear problem: ridge 0.9987, MLP 0.9887 = 0.99x ridge, epoch-1
   R^2 +0.009). Pod re-run set: map_mlp + controls + perfeature_reads + upload
   (edges/eval_lists/mining are ridge-only).

17. **Shuffle-null R^2 clipped at -1e4 before the fp16 store (r5 fix).** The
   pod controls leg warned "overflow encountered in cast": per-feature R^2
   under the row-shuffle null goes below fp16's -65,504 minimum on
   near-zero-variance target columns and cast to -inf. `R2_FP16_FLOOR = -1e4`
   (exactly fp16-representable, far below any band of interest) now clips at
   the ONLY R^2-to-fp16 site (`_shuffle_null_r2_blocked`; every other fp16
   store is coefficients/predictions, every other R^2 array is fp32), with an
   inf-free assert at store time; the perfeature band pass re-applies the same
   floor before its percentiles (idempotent on new arrays, sanitizes -inf in
   any pre-r5 npz — +inf is impossible since R^2 <= 1) and asserts the bands
   inf-free. NaN semantics unchanged (ss_tot <= 1e-12 columns stay NaN). The
   floor is recorded in controls.json `shuffle_null.clip_floor_r2`. Controls +
   perfeature are already in the r4 pod re-run set, so the recomputed arrays
   carry the clip.

18. **Secret scrub before every mined-text stage (r6 fix) — and deviation 4's
   mystery resolved.** The production mining upload raised
   SecretUploadGateError: 87 real-secret-grade strings (telegram-bot,
   openai-real, hf-token, jwt-signed) that real users pasted into their
   prompts sat in the top25_ctx shards. The brief's "16-span placeholder
   substitution" (deviation 4) meant exactly this: #2552 scrubbed mined text
   with same-length placeholders before upload/judging. phase_mining now runs
   the in-process scrub (`secret_scrub.scrub_file`, same-length X
   placeholders — the scripts/scrub_secrets.py policy, NOT the gate bypass)
   over every shard BEFORE staging, asserts `assert_upload_clean`, and
   persists a VALUES-FREE `scrub_report.json` (counts per pattern class only)
   beside the shards. The scrub is in place, so the HF upload copy and the
   judge-input copy (judge prep reads top25_ctx*.jsonl from the pod out-root
   or the HF leaf) are the SAME scrubbed files. New resume checkpoint:
   "shards written, upload pending" (heap + done-marker present,
   scrub_report.json absent) re-enters at scrub+stage without redoing the
   chunk text sweep; full skip now also requires scrub_report.json. Other
   upload leaves audited for text: eval_lists artifacts are row_id + feature
   codes (numeric-only), need_set.json is ids, heap npz numeric, all other
   leaves tensors/JSON metrics — no other prompt/answer text leaves the pod.
   Guarded by `test_mining_scrub_fixes_planted_token_and_passes_gate`.
