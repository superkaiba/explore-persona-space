# Training-K artifact audit

Audit date: 2026-09-11. Source data repository: `superkaiba1/explore-persona-space-data`, pinned at `77f04fcdf169b5a9d4aa304c9c6efbab0526c0e2`. The producing task 1901 remains `awaiting_promotion`, with a clean result; its goal is unchanged. No model training or generation was performed during this audit.

## Outcome

The exact 19,000-context training pool is reusable. The CPU preparation completed successfully on the real artifacts and produced all declared arrays with finite values, complete row coverage, and explicit CI joins. Every original answer vector matches the corresponding stored distractor-bank row exactly; every recovered training prompt matches the existing bundle's SHA256. The test bank contains all 1,000 original contexts and ten answers per context. Its original-vector keep-one policy retains 942 query/candidate identities.

The initial reduced context bank at `data/issue_1901/ctxnn_dl/cx_distr_L19.npz` covers only 9,058 contexts, so it was not used as the training source. Its SHA256 is `d52489f6f7730589cdfbe3f428521f80b7feb96039b3f2047b9e780a3009a3b3`. Instead, preparation reads exactly forty original n1m capture chunks totaling 1,727,608,264 bytes, rather than staging the full n1m capture. All forty source capture SHA256s are identical at the historical `687eb8b42cd01e1279fd857655e895e284440524` and current pinned revisions.

## Exact sources and observed contracts

| Input | Repository path | Realized contract |
|---|---|---|
| Ordered pool identity | `issue1901_avgpool/analysis_tensors/bundle/bundle_index.json` | 19,000 distractor IDs plus 1,000 negative test IDs; prompt SHA256 per row |
| Contexts and original answers | `issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture/shardXX_chunkYYYY.pt` | `cx_last`, `v_x`: fp32 `[n,3,3584]`; `ci`, `prompts`, `layers=[14,19,26]`, `shard_index`, `chunk` |
| Original-answer cross-check | `issue1901_metrics/analysis_tensors/distractors_L19.npz` | `vx`: fp32 `[100000,3584]`, `ci`, `corpus`; first 19,000 rows are this training pool |
| Training draws 43–46 | `issue1901_avgpool/analysis_tensors/kresample/V_distr_shard00.npz` through `V_distr_shard03.npz` | Each `V`: fp16 `[4750,4,3584]`; `ci`, `draws`, `n_ans`, `src`; complete unique CI union and positive spans |
| Seed43 parity text | `issue1901_avgpool/raw_completions/distr/shard00/gen_seed43_chunk0.json` through shard03 | 32 deterministic probes from each 500-row chunk; response text and generated/prompt token IDs preserved |
| Original test X/Y | `issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt` | Realized original 5,000-context bundle; select L19 and `fixed_split(5000,3600,400,1000,42)` test indices |
| Test draws 43–46 | `issue1901_avgpool/analysis_tensors/kresample/V_test_shard00.npz` | fp16 `[1000,4,3584]`; join by negative CI |
| Test draws 47–51 | `issue1901_k10_rollouts/analysis_tensors/seed47.npz` through `seed51.npz` | Each fp16 `[1000,3584]`; recipe, seed, CI, span counts, and prior upload-receipt hash verified |
| Test prompt text | `issue1901_avgtarget/raw_completions/gen_seed45.json` | Ordered `lid` agrees with test bundle row index and all prompt hashes |

The forty original files are the unique `shard{ci//30000:02d}_chunk{(ci%30000)//500:04d}.pt` names for the fixed training IDs (range 500–172468). This mapping matches the parent generation's 960,000-context, 32-shard, 500-context chunk layout and the committed distractor manifest. Every selected chunk was opened; its realized keys, dimensions, precision, prompt identities, and actual CI coverage were checked. The final chunk supplies 449 selected rows; no zero padding or inferred missing row is used.

The bundle index SHA256 is `5330cd7523db3de9a5fcf36296b73e6a2b392e8fc645881e2e9b7a03e9285159`; the ordered bundle CI SHA256 is `c524a8beaeadeba183938f71ced8e65a3479a323bdc2e50ba962cbd832f6d089`. The original-answer bank SHA256 is `8015d9d4dd2d644ded6eecfca150168bdaeab479cea3eac3735942f4d46c94a5`. The test-index SHA256 is `b9377786b24bc9c1c360303fdb8fac86c0097d264479de1dca3c23dd1047d31d`. The prepared `manifest.json` records all 61 consumed source paths, revisions, sizes, Git blob IDs, and SHA256s, as well as every portable output digest.

## Recipe and split checks

Original and additional answers use Qwen/Qwen2.5-7B-Instruct, single-user-turn chat rendering, temperature 1.0, top-p 0.95, 1,024 generated-token cap, and engine seed 42. The original generation uses request seed 42; the old additional bank uses seeds 43–46; the requested extension uses 47–51. Capture preserves the inherited full-template retokenized answer-span mean including the end-of-turn tail, at layer 19, hidden dimension 3584. The new-generation model pin is `a09a35458c702b33eeacc393d103063234e8bc28`. Historical captures do not themselves record that model commit, so the required on-GPU recapture-parity check remains necessary.

The training pool has 19,000 unique prompt SHA256s. Test rows have 942 unique prompt SHA256s; no training/test prompt SHA256 overlaps. Original n1m manifest metadata records exact/near-duplicate filtering against all 1,400 original validation and test prompts using character 5-grams and Jaccard threshold 0.8. Its round1 prompt digest `d40546cd7059780afc50188a0902247a9c2ce49f67ff3d651b87a934a56b8805` agrees with the later bundle metadata. This audit inherits that near-duplicate screen; it does not independently recompute semantic similarity or the character-ngram screen. Training-only GCV avoids using any validation/test answer for hyperparameter selection. The former distractor contexts are training examples in this follow-up and must not be presented as held-out retrieval candidates.

Pinned Hub commit dates are coherent for all four reused seed43 training text/capture pairs and all five new-test text/capture pairs. For example, the training bundle index was uploaded August 25 at 07:32 UTC, before shard00 seed43 text at 07:46 and its capture at 09:46. New-test seed47 text was uploaded September 7 at 21:13 UTC, before capture at 21:28. Original source context and answer tensors share the same capture payload, with prompt-hash and original-answer equality checks providing direct content-level coherence.

Recent sibling worktrees checked include the k-rollout-ablation, retrieval-10k, boundary-25k, and current root paths; the newest pinned `issue1901_avgpool` tree still contains the four distractor draw shards above. `issue1901_k10_rollouts` holds the completed test-only extension. No training seed47–51 bank was found in these searched scopes; the conclusion is scoped to those artifacts, not a global absence claim.

## Prepared consumer interface

The stable prepared-artifact root is `/mnt/eps-data/thomasjiralerspong/issue1901_training_k10/inputs`, outside the sparse worktree. An earlier derived staging copy was pruned by a sparse-checkout update; preparation was repeated from the pinned sources and completed with process exit 0 at 2026-09-11 18:51:40 UTC before publication. Source banks were unaffected.

`train.npz` contains fp32 `X[19000,3584]`, fp32 `Y_original[19000,3584]`, fp16 `Y_fresh[19000,4,3584]`, int64 `ci`, and `fresh_seeds=[43,44,45,46]`. `test.npz` contains the analogous X and original answer, nine fresh draws 43–51, negative test CI, original `pass_b_rows`, and fixed `dedup_rows[942]`. Original fp32 answers are never rounded to fp16 during packaging.

`prompts.json` provides ordered train/test rows with CI, prompt, and prompt SHA256. Compact JSON keeps this text below the 9.5 MB threshold for uploading text as-is. `parity.json` provides 128 fixed seed43 records (32 per original shard), with `generated.text`, prompt/generated token IDs, and `expected_index`. `parity.npz` contains the matching fp16 `V[128,3584]` and CI. For the inherited capture helper, add the outer row's CI to its nested generated object before calling. A fresh process must pass GPU parity before generation.

Source staging maps each repository path directly to `out/source/<repo-relative-path>` and opens that exact file. Explicitly supplied local roots may provide already cached files only after revision-pinned remote content verification; there is no unverified fallback. Prepared output files have a flat consumer layout and are bound by the final manifest. `--phase validate` reopens and checks the portable artifact hashes, realized shapes, finiteness, IDs, split hashes, and prompt hashes before consumption.

## Validation

Eight focused tests passed, covering unordered/missing/duplicate CI joins, source shard boundaries, real PyTorch capture loading at hidden width 3584, original precision, prompt drift, nonfinite vectors, uneven shard joins, exact staging paths, pinned Hub calls, source corruption, and stale manifest revisions. Ruff and CLI import/help checks passed. The full production preparation on the actual 19k and test banks completed with process exit 0 and logged `PASS train=19000 test=1000 inputs=61`. Packaging validation also passed after compacting prompt JSON without changing parsed contents.

## Publication

The six portable files were uploaded together to `issue1901_training_k10/inputs` at immutable HF revision `b281fc98da9fc87b4aabe2463cd24bc26d8ad115`; every remote size and SHA256/Git blob digest matched the prepared local file. The final input-manifest SHA256 is `01913f081a84c954a282a03c794b35015393e9a30cd78da0885cbdb11b041b74`. The complete receipt is `input_publication.json` alongside this audit and at the stable artifact root.
