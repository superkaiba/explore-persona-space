# Exact fair-regression replay for the 900-context comparison

The 900 judge contexts must be compared to predictions from the same original fair readouts on those exact IDs. Full-cohort correlations cannot serve as subset baselines. Scoped live HF inventories of `issue1739_result2_fair` and `_v2` contain only scores and map diagnostics, with no fitted weights or per-context predictions. Replay uses banked activations, without target-model calls or GPUs.

Freeze artifact-declared layers: context ridge evil 18 / sycophancy 20 / hallucination 20; mapped-answer ridge evil 20 / sycophancy 19 / hallucination 20; optional observed-answer ridge evil 17 / sycophancy 19 / hallucination 18. The preprocessing and fitting data remain the original full pools, with five group folds for ID and full-union fits for generic/OOD. Context-only predictions need whitening but do not need a fitted context-to-answer map. Mapped predictions use the shared original map helper. Original per-pool target normalization is preserved.

The replay adapter imports the original `build_pool`, `fit_whitening`, `fit_linear_map`, `_pool_zscored_dv`, `realize_budget_cell`, `run_cell`, and `run_transfer_cell`. The labeled loader reads `context_end` and `t1` only; it retains original first-occurrence context ordering and NumPy mean over the five answer rows, omitting unused prefix arrays. Before fitting, verify this loader against the original full loader on the complete local evil capture store. Do not substitute synthetic or zero activation arrays.

The first proposed local pilot is `configs/issue2669/fair_replay_evil18_pilot.json`: evil, layer 18, context ridge, all 6,468 ID contexts, 417 held-out WildChat contexts, 2,387 OOD contexts. Inputs are locally available after auxiliary staging. It computes the exact whitening on the original 25,261-context ADD pool and the same original fair readout, including all 8,038 readout-training contexts. It performs no model generation and no map fit. Configure at most 8 BLAS/OpenMP threads. Estimated peak resident data is under 8 GiB (single-layer matrices, approximately 0.725 GB per full 25k-by-3584 float64 matrix; several simultaneous copies plus readout buffers); this estimate requires measurement. New local staged data footprint remains below 10 GB. Do not run until root has reviewed the concrete config. The pilot records wall time and actual peak RSS, complete per-context predictions, and full-cohort parity. The engineering equivalence gate is absolute Spearman difference ≤ 1e-5 in every original corpus; failure stops reuse rather than relaxing the gate. This tolerance is an explicit numerical-reproduction criterion, not a statistical significance threshold.

Staging pins: main capture tar and WildChat at HF data revision `cd942a5a4ef4348bff26a0282680cec931e915fa`; generic U at `e5901706`, manifest at `7ef5523673d64697ab497577dbc5b9270c39f020`. HTTP Range was verified with a 512-byte request returning 206 and exact Content-Range against the 69,869,701,120-byte hallucination tar. Tarfile validates checksummed headers and PAX extensions. Eight speculative next-header requests use the last known stride; only offsets demanded by the canonical tar parser are consumed. No tensor payload is fetched during indexing. Index state checkpoints every 50 members. Stage only required `context_end`/`t1` layers and metadata, with per-file source offsets, lengths and SHA256 receipts.

Staging completed with 281 WildChat files (863,718,190 bytes), 12 generic-U arrays (1,822,938,624 bytes), 1,015 selected sycophancy files (2,511,580,300 bytes), and 1,363 selected hallucination files (3,365,250,970 bytes). Total new selected payload is 8,563,488,084 bytes, below the approved 10 GB bound; manifests, range indexes, receipts, and small direction banks add minor metadata overhead. Exact selected-member totals were checked before staging, and completed receipts match their pinned index entries and local SHA256 hashes. Syco's previously existing local store was incomplete and was not accepted as full coverage. No full 69 GB or 52 GB archive was downloaded. Layer definitions and input hashes are checked against actual artifacts, not manuscript prose.

After full-cohort parity, extract the previously frozen 900 IDs and compute matched metrics. Recovery outputs that fail parity are retained for diagnosis and cannot be labeled the original fair baseline.

## Measured CPU pilot

The authorized evil layer 18 context-ridge replay finished successfully: 66.649 seconds wall time and 2.690 GiB peak RSS. All four original corpus correlations matched exactly (absolute difference 0). All 6,468 ID, 417 generic, 1,868 HHRT and 519 ToxicChat predictions were persisted. The reduced loader also passed exact array/label/group/order equality against the original loader. These measurements support a one-layer mapped-answer pilot, whose additional map fit still requires explicit sizing against the original helper.

## Original ID transduction caveat

`run_fair` invokes `fit_add_maps` before assigning readout folds. The ADD map and whitening are fit on all retained ID context/answer-activation pairs plus the generic unjudged pool. Consequently, each ID held-out readout fold has held-out behavior labels, but the map has already seen those contexts and their unjudged answer activations; whitening has seen the contexts. This is a transductive ID mapping protocol, not strict end-to-end held-out forecasting. Generic held-out and OOD contexts are excluded from the original map/readout pools. The replay preserves this recipe for exact comparison and does not silently substitute a nested-map protocol.

## Fit diagnostics and preservation

The original map helper computes held-out mapping R², the identity-plus-learned-bias baseline, and nearest-neighbor retrieval with pool size and chance levels before refitting weights on the full ADD pool. Evil layer 20 and sycophancy layer 19 completed using an adapter that did not persist its computed diagnostics; their explicitly labeled `inherited_map_diagnostics.json` files reference the pinned original fair artifacts and must not be described as fresh measurements. Later mapped replay cells persist freshly computed `map_diagnostics.json` and map/whitening arrays. The original primal map helper does not expose its chosen regularization value; the metadata records this as unavailable and preserves the candidate grid rather than inventing a value. Later cells also capture selected readout regularization and whitening shrinkage metadata.

Map/whitening binary arrays must be preserved separately through the canonical private data archive. The forecast archive's text-only export contains predictions, diagnostics, configs, and provenance, and must not silently include these binary arrays. No replay is repeated solely to obtain omitted fit metadata.

## Completed recovery

All seven authorized CPU replays completed with successful process exits. All 33 method-by-corpus Spearman correlations exactly equal the original frozen values (absolute difference 0), exceeding the predeclared 1e-5 equivalence gate. The original full ID, generic, and OOD prediction cohorts are available for the frozen 900-context join. Replays were sequential with eight thread caps and no target-model or GPU calls.

| Replay | Wall seconds | Peak RSS GiB |
|---|---:|---:|
| evil_layer17_oracle | 47.29 | 2.673 |
| evil_layer18_pilot | 66.65 | 2.690 |
| evil_layer20_map | 105.48 | 6.183 |
| hallucination_layer18 | 65.68 | 4.974 |
| hallucination_layer20 | 145.76 | 8.267 |
| sycophancy_layer19 | 586.38 | 8.153 |
| sycophancy_layer20 | 60.74 | 4.251 |

The machine-readable `data/issue_2669/probe_replay/replay_manifest.json` records config paths, methods/layers, prediction and label source hashes, timing, and the separate binary artifact inventory. The only new fitted binary, hallucination layer 20 map/whitening arrays (205,637,298 bytes), was preserved by the parent task in private HF overflow at revision `d46b8b294620bd3c8e069b1084bba670867caa36`, prefix `issue2669_codex_forecast/probe_replay/hallucination_layer20`, with download-and-SHA256 verification.
