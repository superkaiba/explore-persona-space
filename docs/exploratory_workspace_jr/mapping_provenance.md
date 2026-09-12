# J/R mapping provenance audit

Read-only audit completed 2026-09-12T03:27:27.677488+00:00. No generation, training, model evaluation, task mutation, or Claude usage occurred. No new J/R component test outcomes were inspected.

## Recommendation

Use **Qwen3.5-27B with thinking disabled** as the primary candidate and **Qwen3.5-4B with thinking disabled** as the weaker comparison, conditional on the provenance/parity gates below. Selection was frozen to measured same-mode GPQA accuracy among existing valid final-context-token → answer-token-mean maps. It did not use mapping R², parameter count, or estimated AA. This is the highest observed point estimate in the audited usable panel, not proof of unique global superiority.

| Model | Current cap-long GPQA | Descriptive Wilson 95% interval | Existing map |
|---|---:|---:|---|
| Qwen3.5-27B | 670/985 = 68.02% | 65.04–70.86% | L50/64; completed-block depth 51/64; d=5120 |
| Qwen3.5-397B FP8 | 613/985 = 62.23% | 59.16–65.21% | Available competing panel candidate |
| Qwen3.5-9B | 585/990 = 59.09% | 56.00–62.11% | Available competing panel candidate |
| DeepSeek V4 Flash | 583/990 = 58.89% | 55.80–61.91% | Available; 91 unparseable, fallback flagged |
| Qwen3.8-27B | 511/988 = 51.72% | 48.60–54.82% | Available; AA disagrees with same-mode GPQA ordering |
| Qwen3.5-4B | 506/987 = 51.27% | 48.15–54.37% | L20/32; completed-block depth 21/32; d=2560 |

Primary–weaker gap: **16.75 percentage points**. These binomial intervals treat rollouts as independent; five draws share each of 198 GPQA questions, so intervals are descriptive and do not settle a unique-winner test. Accuracy uses realized retained rollouts; attempted denominator is 990. Figures are unnecessary for this audit.

Capability source: pinned dataset `superkaiba1/explore-persona-space-data@3de3be6de6aab707a207128f0081799b579ea594`, under `issue2588_capability_panel_cap_long/fits/<cell>/gpqa_transfer_prompt_last.json`. These exact current files were opened, not inferred from task titles. The no-thinking frontier files for Qwen3.8-Flash-Next and DeepSeek V4 Pro do not occur in the current pinned fits subtree or scoped local artifact search. GLM-5.3 is registered thinking-only and therefore fails the requested prompt-last input. Old charmander disk is not certified absent: current project rules say fellows access was revoked 2026-09-09, and no charmander SSH alias is available. Wider catalog models are chiefly Qwen2.5-7B, Qwen3.5-9B, Llama/Tulu3-8B, and OLMo2-7B; the catalog is discovery evidence, not substituted capability measurement.

## Actual coefficients and representation contract

Both NPZs were opened with `allow_pickle=False`; `W,xmu,xsd,ymu` have expected shapes, float32 dtype, and finite values. Formula is `((x-xmu)/xsd) @ W + ymu`; raw row-vector affine matrix is `A=W/xsd[:,None]`, bias `ymu-xmu@A`. Do not interpret normalized `W` directly as the raw-coordinate map.

| Model | Coefficient file | SHA256 | Matrix shape |
|---|---|---|---|
| 27B | `/home/thomasjiralerspong/explore-persona-space/data/issue_2588/mapping_rank_cache_cap_long/q35_27b_a__prompt_last.npz` | `803de39c24bd7d770c42db30acef869038a3188bb1f434a94ef932ed65723ed3` | 5120×5120 |
| 4B | `/home/thomasjiralerspong/explore-persona-space/data/issue_2588/mapping_rank_cache_cap_long/q35_4b_a__prompt_last.npz` | `854fdd2333d2f5692dab1f4c4e95c48db30d2f8e8847d8e6e83b580c422544cb` | 2560×2560 |

Each preprocessing vector has length d. Original panel fits discarded weights, but `scripts/issue2588_mapping_rank_vs_capability.py` reconstructed and persisted these selected-layer/selected-lambda matrices with parity against parent validation and test scores. Current payload parity residuals are below 5e-10; this is coefficient provenance, not the criterion used to select a model. The cache embeds `hf_revision='main'`; this audit supplies an immutable underlying-data pin and coefficient SHA. Exact remote coefficient mirrors were not verified; preserve/upload these bytes with the new task before relying on another machine to fetch them.

Hook is the **post-decoder-block output**, block indices zero-based, same layer for input and target. Input is the final rendered prompt token, including the assistant-template boundary. Target is the arithmetic mean over answer-span states after separately retokenizing saved prompt and completion text and concatenating token IDs. Whitespace is stripped at span boundaries; thinking text is excluded where present. These selected arms have no thinking tokens. Capture stores `row_ids,y_ans,x_prompt_last`; no per-token state sequence is persisted. A new component rig must assert the actual resolved module path on its exact stack.

## Checkpoint and split provenance

Candidate model revision pins, verified through Hub history/config/weight metadata:

- Qwen3.5-27B: `fc05daec18b0a78c049392ed2e771dde82bdf654`, last Hub change 2026-04-24.
- Qwen3.5-4B: `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, last Hub change 2026-03-02.

Both latest commits precede producing fits on 2026-09-04 (code `9b896ccc6b65e1d7322d3c7e67f6fe92883a0f0c`, transformers 5.16.1, torch 2.13.0+cu130). **The original run did not attest base checkpoint SHA.** These are historical-main inferences, not producer-recorded pins. Exact revision recapture parity is required before reuse with new J/R captures.

Generic prompts come from LMSYS-derived `issue1491_scale_ladder/manifest@815ff6d976c686af8672b27cfdfb1ce6b419c02c`, with the source `train_25k.jsonl` restricted to the first 10,000 split IDs, and separate `val_400.jsonl` / `test_1000.jsonl`. Source file byte SHAs were verified against `eval_results/issue_2330/split_ids.json`:

- train source: `32bb1a6aa7cc174fa2782eaf19c0d5356a053e761aa42fc760fc11a3180c3af0`
- validation source: `23e553a0763c0b8f11937db74d3d2b5880c1fe51c7a3ebcfd0a18ddbb7f92dd6`
- test source: `e368930bb7b03bc1696723ba5f7eb1542a538d5da8dc832ee2ff5764538d0310`

| Model | Train | Validation | Test seed42 | Test seed43 | Test seed44 | All three test draws present |
|---|---:|---:|---:|---:|---:|---:|
| 27B | 9,941 | 397 | 996 | 994 | 997 | 991 contexts |
| 4B | 9,889 | 398 | 992 | 994 | 991 | 981 contexts |

**K=1 for map training and validation for both models.** Extra seeds43/44 are test-only ceiling draws. They do not make the fitted target K=3. The machine-readable report includes every realized context local ID, per-split ID-list SHA, and exact Hub row/shard paths. `train_10k_i`, `val_400_i`, and `test_1000_i` have distinct source namespaces; ceiling rows instead map back to test via suffix i.

**Inherited validation/test are not content-disjoint:** hashing source prompt text found 13 unique prompts in both validation and test for each model. Training shares zero prompt hashes with either. New component calibration/test must group prompt/content identity and exclude validation-seen prompts from decisive test or use a fresh held-out corpus. Never assert disjointness from integer IDs or stage prefixes alone. Full prompt-hash lists and overlaps are in the JSON report.

## Token capture gaps and next gates

Actual raw completion chunks were opened for both models. They contain prompt/text, `read_points`, token counts, seed/cap/stage, and finish reason. They **do not contain original generated token ID arrays**; original generation code explicitly drops prompt IDs and saves only `len(comp.token_ids)`. Actual activation shards store summaries only. Existing text can be retokenized to reproduce the historical teacher-forced map representation, but cannot prove exact original on-policy token replay. New J/R evaluation therefore needs fresh saved generated IDs and per-answer-token captures on the exact chosen checkpoint. Keep all calibration data and model/layer/SAE selection separate from decisive component test.

No task was created or edited. Existing tasks1482/1776 cannot silently host a changed model-selection Goal. Task2588 is awaiting_promotion; task2659 is approved without clean result, both read through `task_workflow.get_task`. No new claim depends on their workflow status alone.

Complete machine-readable artifact: `/tmp/jr-mapping-provenance.json`. Auxiliary source audit files are named and SHA-hashed inside it. Parent should land the report and necessary source manifest/IDs durably in its dedicated worktree.
