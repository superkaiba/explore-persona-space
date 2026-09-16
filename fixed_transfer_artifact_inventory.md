# #1739 consistent transfer: frozen #779 L19 map inventory

Verified 2026-09-16. No production training launched by this fact-checker.

## Frozen map and exact recipe

- Dataset repository: `superkaiba1/explore-persona-space-data`.
- Historical consumer pin: `9d8f789bf034d8f244e1d00e0dbbe6aba6d272c5` (all required paths verified at this pin).
- Weight path: `issue779_monitoring/n1m_readout/weights/L19/ridge.pt`, 51,425,703 bytes, SHA256 `188486f8afd9d95221e32492f3a0be2a3bdb2098cbe7fadfecf1d46433567909`.
- Real verified local copy: `/mnt/eps-data/thomasjiralerspong/issue2162_mapshift/hf_dl/issue779_monitoring/n1m_readout/weights/L19/ridge.pt`.
- Actual payload: kind=ridge, fitter=ridge, layer19, selected_lambda=.001, float32 W(3584,3584), xmu/xsd/ymu(3584).
- Producing run is `eval_results/issue_779/n1m-nonlinear-map-behavior-readout/n1m_multilayer_fits.json`; git `7f19f674742f8faee6267308f4496cf851e7e3f7`, timestamp 2026-07-22T21:40:04Z. First n1m summary was git bd9f686, but persisted weights were produced by the later multilayer run. Both record identical L19 ridge heldout R2 .7541708417500046.
- `mixed_1m` is the target name, not an actual one-million-row fit. Actual 963,444 training rows = 3,600 original LMSYS rows + 959,844 nonempty-response new rows (529,085 LMSYS total; 434,359 WildChat total). New manifest contains960,000 rows;156 empty responses have no captures.
- Original pass_b split: fixed_split(5000,3600,400,1000,42). Train does NOT include400 validation or1000 test rows; NO train+validation refit. Full usable TRAIN pool is used. Validation chooses lambda from np.logspace(-3,8,23), true selected=.001.
- Context coordinate-wise train mean/std (unbiased N-1 std plus1e-9); answer train mean centering; no covariance whitening. Prediction raw coordinates `((X-xmu)/xsd)@W+ymu`.
- Canonical helpers `_train_standardizer`, `_ridge_factorize`, `_ridge_predict_one`, `fit_ridge_with_weights`, `apply_map`, `_pool_rows`, `select_train` AST-identical between producing git and present root source. Production script uses the shared worktree copies and records source hashes.

## Input artifacts

- Capture prefix: `issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture/`:1920 chunks,83,219,376,576 bytes, each includes3layers [14,19,26]. Verified full pinned inventory: `/dev/shm/issue1739-fixed-transfer/inventory/hf_source_inventory.json`.
- L19 X+Y complete combined arrays:964,844 rows ×3584 ×4bytes ×2 ≈27.66GB. Only exact layer slices need resident storage; all3layers require83GB transfer if no full reduced local store.
- Actual first sourcechunk verified: `/mnt/eps-data/thomasjiralerspong/issue2618_smoke/stage/issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture/shard00_chunk0000.pt`,43,188,789bytes, SHA256 `4ca92e7cca7ab0fe82e1d76ad046ec2aa190a0f864bfa9e8a71a40f6bade0f05`. Actual keys cx_last/v_x float32(500,3,3584), ci list500, prompts list500, layers[14,19,26], shard_index0,chunk0.
- Pass_b: `issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt`,6,021,122,751bytes,SHA256 `46c06e89c513ca598bc83be1c87689694a47bfc927a81d0d738a54df769dbf9a`. Real local fullhash verified `/mnt/eps-data/thomasjiralerspong/issue1901_mlpdense_fold/issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt`. Actual keys cx_last/cx_mean/v_x float32(5000,28,3584),layers,source,metadata; NO prompt strings.
- All88 sampling-manifest files already local at `/mnt/eps-data/thomasjiralerspong/issue1895_inputside/scratch/sampling_manifest`; every file size+Git-blob SHA1 verified at9d8f pin. 960,000 contiguous IDs. Files total721,957,398bytes. Verification inventory `/dev/shm/issue1739-fixed-transfer/inventory/local_manifest_verified.json`.
- Manifest `new_prompt_sha256` = `2b14762a15d316c602332a749ebd87c733d687d4165eb5d0038c298e0d27ce46`.
- Original5000 LMSYS first-turn prompt list recovered using parent's sample_disjoint_n50k(5000,0,0), exactly matches historical ordered-NUL-separated SHA256 `d40546cd7059780afc50188a0902247a9c2ce49f67ff3d651b87a934a56b8805`. Saved `/dev/shm/issue1739-fixed-transfer/inventory/passb_prompts_sha_verified.jsonl`. Dataset stream interpreter exited134 at Python finalization (PyGILState_Release), AFTER filewrite and digest assertion. Separate stdlib-only process independently re-read all5000rows, asserted indices and historicalSHA, exited0. Explicit note in `passb_prompts_verification.json`.
- No complete reduced X/Y store found in searched local cached paths. `/mnt/eps-data/thomasjiralerspong/issue1482_saedense/dense/{X,Y}_L19.f32.mm` are only142,000 selected rows; do NOT substitute for fullpool.
- `/tmp/probe2569*/...ridge.pt` are tiny synthetic32×32 fixtures, not real maps; do NOT reuse.

## Leakage and coordinate caveats

- Map prompt strings are BARE first-user-turn text stripped, not rendered chat prompts. Compare evaluation question/query text hashes as well as rendered prompt hashes. Parent normalized text is `" ".join(text.lower().split())`; hashes are UTF-8 SHA256 exact and normalized. Train AND validation require exclusion because validation selectedlambda.
- The original near-duplicate gate protected the #779 1400 val/test prompts only (char5-gram Jaccard>=.8); it says nothing about #1739 WildChat evaluation overlap. New experiment must compute its own overlap against these maptrain+val contexts.
- Layer numbering matches post-block L=19. Context is the last generation-prompt token.
- Material pooling caveat: #779 `capture_answer_vector` averages `prompt_len:full_len` of a fully rendered assistant answer, INCLUDING closing `<|im_end|>` plus newline. #1739 `capture_row_ids_and_positions` t1 averages completion token IDs only, EXCLUDING boundary. Capture functions are AST-identical at producing commit. Existing metadata claiming v_x==t1 exactly was overbroad. They share raw hidden coordinates but different answer spans; report this limitation or recapture to align.

## Implementation handoff

Owned files in `/home/thomasjiralerspong/.codex-worktrees/1739-covariance-ablation`:
- `scripts/issue1739_transfer_map.py`
- `tests/test_issue1739_transfer_map.py`

CLI: `... issue1739_transfer_map.py prepare --root /dev/shm/issue1739-fixed-transfer`, then `fit --root SAME`. No full job launched here. Default workers4, block8192; root supplies8thread environment. Inputs in root/map_inputs. Progress root/outputs/map/progress.json. Five seeds0–4; training answers and validation answers independently permuted; full original lambda grid selected per control. Shared context Gram and perblock true+five-shuffle cross-products. Atomic resumable checkpoints; source/inventory/array/permutation hashes. Physical-memory and staging-space guards.

Outputs include frozen.pt, true_refit.pt, shuffle_seed{0..4}.pt, map_manifest.json, map_prompt_hashes.npz (train/validation/test exact+normalized hashes), permutation arrays, diagnostics (heldout R2, identity+bias, rawcosine retrieval with1000pool/.001chance), source inventory and prepareprovenance. Scorer contract agreed directly with transfer_plan.

## Model and layer verification (final audit)

HF `model_info` + commit history checked this turn: `Qwen/Qwen2.5-7B-Instruct` main is `a09a35458c702b33eeacc393d103063234e8bc28`, latest commit2025-01-12T02:10:10Z; last earlier commits are2024-09 tokenizer/README/config updates. #779 capture code loaded the default model revision, while #1739 explicitly pins this same SHA. Thus revision compatibility is supported by unchanged Hub history covering the July2026 captures; the old #779 payload does not itself record model SHA. Persist this evidentiary distinction. Query evidence in `/tmp/issue1739-transfer-model-identity.json`.

#779 hooks `model.model.layers[19]` output via analysis.extraction.extract_layer_activations; #1739 uses hidden_states[1:][19], i.e. hidden_states[20]. Both are post-block19 residual state. Layer19 is not the final layer, so the final-layer norm exception is irrelevant.

Validation:7focused tests passed (including6canonical-versus-shared-Gram refits), Ruff clean. Tiny actual-source smoke inspected500-row realchunk, mapapply canonical vs manual maxabsolute5.33e-15, and30train/10val/10test8coordinate true+shuffled fits exactly matched canonical lambda and parameters. Smoke evidence `inventory/map_real_smoke.json`. Inventory and original5000prompt strings copied and hashverified durably under `/home/thomasjiralerspong/.local/state/eps/experiment-watchdogs/issue1739-fixed-transfer/inventory`.
