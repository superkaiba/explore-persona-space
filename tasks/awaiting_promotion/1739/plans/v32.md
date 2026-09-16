# Plan — task #1739: matched fixed-direction transfer through the 963k generic map

## 1. Goal and authorization

Canonical Goal, read with `uv run python scripts/task.py view 1739 --json` on 2026-09-16:

Determine whether applying the learned context->answer map before projecting the persona vector predicts on-policy behavior expression (evil, trait sycophancy, hallucination) better than context-side projection and direct regression at matched (unlabeled, labeled) data budgets, and whether that advantage grows across a real-data distribution-shift ladder.

This follow-up isolates the fixed-direction transfer part of that Goal. The user explicitly authorized running the comparison from start to finish. Preserve the Goal and existing results; use the existing task, isolated worktree, and task CLI for task changes. No new user permission is needed for the approved protocol. No model generation, new judging, nonlinear fit, automatic Claude/Anthropic invocation, or unrelated manuscript edits are included.

## 2. Scientific question and hypotheses

Can an independently extracted and frozen answer-side behavior direction predict held-out on-policy behavior when applied to the output of a generic context-to-answer map, without fitting a behavior readout on contexts or mapped answers?

Distinguish three conclusions: (H1) the answer direction is a valid instrument on real answers; (H2) that fixed instrument transfers through a generic map; (H3) transfer improves over a matched direction extracted directly in context space. H2 does not require beating supervised regression. H3 can fail while H2 succeeds. Failure of H1 makes H2's behavioral interpretation weak, and must remain visible rather than dropping that behavior from the figure.

Do not formulate the result as label-free prediction: direction extraction itself uses contrastive instruction supervision. This run does not judge-filter extraction rows. What is absent is downstream behavior regression or behavior-based layer/sign/hyperparameter selection on evaluation examples.

## 3. Prior work and exact reusable sources

The Persona Vectors paper uses separate extraction and evaluation questions and answer activation contrasts. Primary source located via web search: https://arxiv.org/html/2507.21509v2 (search evidence saved in `/tmp/issue1739-transfer-persona-search.json`). The exact local lineage is more important for numerical compatibility than a fresh implementation of the paper.

- `scripts/issue779_ffc_n1m_fits.py`: canonical #779 combined-pool assembly, exact train/validation/test selection, standardization, streaming ridge, weight persistence and `apply_map`.
- `scripts/issue1739_fits.py::_load_rb_e1`: obtains answer (`t1`) and context (`context_end`) direction from the same extraction-store row identities. **Important unresolved factual gap:** the loader claims judge-filtered inputs but does not itself judge-filter; it only splits by side. `generation.generate_e1_extraction` emits all rollouts and `capture.load_capture_rows` expands all rollouts. Actual archived filtering must be verified before claiming canonical judge-filtered persona vectors.
- `scripts/issue1739_r2v2_factorial.py`: existing E1 / E1_fc extraction comparison. Its old scoring is whitened and uses selected layers, so only extraction logic is reusable, not its aggregate scores as matched results.
- `scripts/issue1739_map963k_readout.py`: frozen-map loading, canonical chunked application, per-context reductions, existing ~1M-map evaluation provenance. Its shuffled control permutes weight rows and is not the requested shuffled-pair refit.
- Current covariance worktree: tested archive selection/staging, cached DV loading, per-group bootstrap and monitored upload infrastructure. Reuse the components after checking call signatures and source SHAs.
- Existing #1739 `claim4_controls` is a smaller-map refit control; do not treat its scalar results as the #779 control.

## 4. Fixed experimental design

### Map, layer, coordinates

Primary map is frozen #779 `mixed_1m`, fitter `ridge`, Qwen2.5-7B-Instruct, residual-layer index19 as recorded by the stored `layers` mapping. This is a pre-specified common layer, not a winner selected by the new evaluation. Use context_end / full prompt end only: the #779 map was trained on this input position. No prefix-end secondary result is required for this scoped comparison.

The #779 map uses RAW input/output activation coordinates. Its internal preprocessing is train-only coordinate standardization, **not full covariance whitening**:

`pred = ((X - xmu) / xsd) @ W + ymu`.

Call `issue779_ffc_n1m_fits.apply_map` on bounded row chunks. The payload stores W/xmu/xsd/ymu in fp32 and canonical application upcasts to fp64; all arms should use the same raw activation coordinates. Verify map input and extraction/evaluation capture model revision, residual hook/index, final-token semantics and answer response pooling. Report any historical capture-numerics difference.

### Directions and extraction budget

**Authorized implementation decision (root,2026-09-16):** primary extraction is the existing #1739 E1 **unfiltered instruction contrast**, at both positions using exactly the same rows. This is an independently specified fixed contrastive direction, not a faithful judge-filtered Persona Vectors extraction. No cached filtering is claimed, and absent extraction scores are not a launch blocker for this explicitly labeled design. The primary validation of behavioral meaning is its association with actual answers on held-out examples. The inherited recipe discussion below describes the archival provenance audit, not a requirement to obtain new judgments. Root also verified that #779 v_x includes assistant-closing template tokens, while #1739 t1 covers completion tokens only: report this endpoint mismatch as a limit on transfer interpretation; do not claim an exact endpoint replication.

For each behavior, one extraction source and the same kept rollout identities define both directions:

`v_A = mean(t1 | positive instruction) - mean(t1 | negative instruction)`

`v_C = mean(context_end | positive instruction) - mean(context_end | negative instruction)`.

Use positive-minus-negative sign without flipping on evaluation. Norm normalization is optional for numerical conditioning because Spearman is invariant to positive scaling; persist original directions, norms and normalization choice. A zero/invalid direction is a reported failed instrument, not a zero performance value.

The inherited E1 design is five opposite-sign system-prompt pairs, twenty extraction questions, ten rollouts per sign/pair/question. Retain all archived rollout rows except explicit map-membership exclusions, with no judge threshold. Report counts by sign, question and prompt pair; match the exact same row weights for v_A and v_C. This repeats context rows in proportion to the archived answers, matching answer-side weighting. Persist unique-context counts too.

**Do not assume archived E1 data are filtered because a docstring says so.** Root has selected and will disclose the unfiltered instruction contrast for this run. Do not add a filtered source after examining evaluation results or pool sources. No automatic Claude/Anthropic call is allowed.

E1 contrasts different positive/negative SYSTEM prompts while matching question content, so v_C need not vanish. A same-full-prompt positive/negative answer split would produce identical context vectors and is unsuitable as this baseline. Context-side E1 measures the same instruction contrast and supervision budget as answer-side E1; it is not a supervised context probe.

### Five scoring methods

1. Real-answer reference: `score_A = Y_eval @ v_A`.
2. Mapped answer: `score_M = frozen_map(X_eval) @ v_A`.
3. Context-native direction: `score_C = X_eval @ v_C`.
4. Answer direction directly on context: `score_raw = X_eval @ v_A`.
5. Pairing-shuffled map: `score_null[k] = shuffled_map[k](X_eval) @ v_A`.

No behavior-label fitting or slope/intercept calibration occurs after these scores. Frozen-map pushdown can accelerate scoring: `u = (W @ v_A) / xsd`, `c = ymu @ v_A - xmu @ u`, score=`X @ u + c`. First assert equality with the canonical apply-then-project path on a real sample; keep canonical application for reconstruction diagnostics.

## 5. Exact paired-map null

Match the entire #779 generic training pool, context ordering, standardization, target multiset, ridge recipe and validation-size selection procedure. Do not subsample or substitute a smaller corpus without reporting it as a different experiment.

`assemble` forms original5000 pass-B rows plus959844 usable new captured rows =964844 total. The fixed original split, seed42, is3600train/400validation/1000test. The training pool is original3600 plus959844new =963444. No final refit adds validation or test. `select_train(..., mixed_1m, 1000000, mixed, seed)` returns the entire sorted full pool; the helper's hash(name) RNG does not affect this full-pool branch. Verify actual counts and recorded index hashes rather than trusting these expected values.

For each of five deterministic seeds0–4 (precedent: #1739 claim4 multi-seed controls), independently permute target rows inside TRAIN and inside VALIDATION. Leave diagnostic TEST paired. This removes all genuine pairing available to fitting AND model selection. Save both permutations/seed/hash and number of accidental fixed points; a standard permutation may retain O(1) fixed points and does not need a new derangement recipe. Select lambda using the same variance-weighted reconstruction R² objective on permuted validation targets. Do not use behavior DVs in tuning.

Training standardizer matches #779: fp64 sums, context mean, unbiased sample sd (`N-1`) plus1e-9; answer mean; no answer variance normalization. Use #779 lambda grid `np.logspace(-3,8,23)`, one context Gram eigendecomposition reused for true/null targets and lambdas. Persist null maps in the same fp32 payload schema before canonical application, so quantization is matched. At full dimension3584 and n963444, stream blocks and checkpoint Gram/cross-products. Reuse one context factorization across null seeds, and batch target cross-products only as memory permits.

A train-only permutation with selection on genuinely paired validation retains paired information through lambda selection. If reported as an extra diagnostic, label it distinctly. A row permutation of W is not accepted as this control.

To prove source reconstruction before interpreting the null, fit/reconstruct the unpermuted canonical map on the exact restored inputs and compare its selected lambda and held-out predictions with the archived reference. Root may avoid an extra full factorization by accumulating true XtY during the same streaming pass as null XtY and reusing the factorization. Predefine tolerance from a tiny-real canonical-vs-shared-factorization smoke, float64 precision, and the actual fp32 stored payload error; never widen tolerance merely to pass. If the archived map cannot be reproduced due to unresolved provenance, report the failure and do not call the null exactly matched.

## 6. Evaluation and grouping

Use identical retained context IDs, existing on-policy completions and behavior DVs for all five methods. The frozen eight-cell primary roster, verified against current covariance results before scoring, is evil:{hhrt,toxicchat,wildchat_rung}; sycophancy:{aita,wildchat_rung}; hallucination:{nqopen,simpleqa,wildchat_rung}. WildChat retains the same sha1(context_id)%5==4 evaluation split as the covariance comparison. Include held-out synthetic questions only as a separate instrument diagnostic if compatible inputs are available; it is not a ninth primary cell. Enumerate planned rung IDs and realized counts in the manifest before scoring. Never silently omit a hard-to-stage rung or draw a missing cell as zero.

Each per-context real-answer vector and DV must aggregate over the same retained rollout IDs. This is a substantive alignment audit: old loaders sometimes average every activation row while DV computation drops invalid judge rows. Join by `(context_id, rollout_k, source_file)` and prove the exact scored rollout sets, or record why a corpus's cached DV is already complete. Context final-token numerical repeats can be averaged over this same retained set. Report min/max rollouts per context and skipped contexts with no valid DV.

Primary metric: per-rung Spearman correlation between each fixed score and raw cached behavior DV. Preserve the on-policy meaning of each corpus's DV (e.g. hallucination fabricated fraction, trait scores elsewhere), and explain any instrument differences across datasets rather than pooling raw labels.

Primary contrasts: map-minus-context-native; map-minus-direct-answer-direction; map-minus-mean-shuffled-map. Mean-shuffled performance means the mean of five seed-specific Spearman correlations, not correlation of averaged predictions; apply the same operation within each paired bootstrap draw. Also report the actual-answer-minus-map gap. Retain all five individual shuffle-seed correlations and differences; seed spread describes null-fit randomness, separate from evaluation sampling uncertainty. Five null seeds are a control, not a high-resolution permutation significance test.

Paired2000-draw95% percentile cluster bootstrap, inherited from the recent #1739 covariance comparison. Draw the same cluster multiplicities for every method; rerank scores and DVs inside each sample (Spearman). Batch bootstrap draws to bound memory. Group at conversation/prompt/question family where metadata identifies repeated structures; use corpus-qualified group keys, because reused answer strings across hallucination corpora are not global context identities. If only unique contexts are available, state that the interval captures context sampling and may miss family dependence. OOD corpus-transfer rungs supply the structured holdout test: no fitting on those behavior labels. Do not choose layer, sign, seed, directions or method from these intervals.

Interpretation is per dataset, with effect sizes and paired intervals. No all-behavior success claim from cherry-picked favorable rungs; no significance claim from overlap/nonoverlap of marginal intervals. Intervals are conditional on the frozen map and fixed extraction sample, not uncertainty over possible direction-extraction datasets.

## 7. Leakage and reconstruction audits

Compare map training/validation, extraction and evaluation membership using source context IDs AND normalized rendered prompt/question hashes. Map corpus manifest must resolve both original pass-B prompts and new mixed-pool prompts. If behavior evaluation WildChat overlaps generic map training, exclude overlapping evaluation contexts identically from every arm and report per-rung exclusion counts; do not quietly retrain a different true map. Exact/normalized matching does not guarantee no semantic paraphrase overlap, so document the audit's limit. Compare extraction question IDs/text against held-out synthetic and real evaluation prompts.

For both frozen and shuffled maps report generic held-out reconstruction R², identity-plus-learned-bias baseline `X + mean_train(Y-X)`, mean cosine, and nearest-neighbor retrieval accuracy on the same held-out candidate pool. State pool sizeN and chance1/N. Repeat reconstruction diagnostics on behavior evaluation answers as a transfer-validity check. Fit any bias using map train only. Use bounded chunks for dense retrieval and score true paired targets on diagnostic test. No diagnostic threshold invented after observing performance.

## 8. Implementation and checks

Minimal new code: a dedicated transfer runner/analysis module plus a selected-layer staging extension and focused tests in the existing isolated covariance worktree. Reuse canonical map/load helpers but do not call the broad historical fitter CLI that would train MLP/KRR arms.

Required tests cover: (1) exact train/val/test disjointness and permutation multiset preservation; (2) same extraction IDs and weights at both positions; (3) affine pushdown versus canonical apply-map; (4) shared-factorization ridge parity with canonical tiny fit, including selected lambda; (5) rollout/DV alignment and explicit invalid-row drops; (6) paired bootstrap tied ranks/group resampling; (7) stale/mismatched resume artifact refusal. Run a production-format, real-member staging/load smoke before the full stream. Tests exercise these scientific invariants, not just implementation mirrors. Use independent review of the plan and implementation via the available Codex agents; no Claude automation.

## 9. Resources and monitoring

Estimated GPU-hours (total): 0 for the initial CPU implementation; no fresh model inference. A CPU streaming pilot must measure stage bandwidth, fp64 cross-product throughput and eigendecomposition before claiming wall time. If the CPU route is materially slower than expected, root must re-evaluate batched algebra and only then provision within documented project compute rules.

Source estimates from #779: full selected-layer X+Y roughly27.7GB fp32; full capture stream roughly82GB over three recorded layers; selected extraction/eval L19 additional bounded space. Parent reports VM125GiB RAM,80GiB available, /dev/shm55GiB free, root13GiB free, data1.9GiB free. Do not route bulk data to root/data. RAM staging must budget copied blocks, shuffled indexed targets, Gram matrices, and neighbor arrays, and must checkpoint irrecoverable progress remotely. Independently guard /dev/shm free bytes and /proc/meminfo MemAvailable: process RSS does not count all staged tmpfs memory. Write combined arrays once as memmaps rather than eager concatenate/copy if peak does not fit. Reconstruct exact row order from manifest and capture indices. Never pretend /dev/shm is durable across reboot.

Long compute runs detached and protected under the existing monitored workflow. Observe every30s with timestamps, subprocess/backend liveness, log age, checkpoint bytes and phase counter; quiet unchanged checks. Independent experiment_watchdog timer, durable configuration/runbook/source pins in ~/.local/state/eps/experiment-watchdogs/issue1739-fixed-transfer/, bounded recovery worker, acknowledged personal notification, verified real canary and next scheduled watchdog tick are required before unattended operation. Recovery caps and source/config checks follow the tested covariance monitor; stop an old worker before relaunch. New staging restores pinned inputs after reboot. No automatic Claude calls.

## 10. Artifacts, completion and figure

Persist input manifest with HF revisions/hashes, source SHA/config hash, exact ordered map rows and split hashes, extraction provenance/filter counts, raw v_A/v_C vectors, shuffle permutations and map payloads, reconstruction metrics, leakage exclusions, ordered per-context scores/DVs/groups and exact evaluation membership, paired-bootstrap differences/intervals, and progress/completion sentinels.

Hero figure: shared-layer19, common raw coordinates, one panel per behavior or rung group; five methods with paired95% uncertainty. Show real-answer reference visibly; label context-native extraction separately from answer-direction-on-context. Shuffled method displays seed mean with seed variation separately available. Small second panel/table presents paired map-minus-context-native and map-minus-shuffle differences. All datasets and failures appear. Use `docs/paper_context_answer_map/plotting_style.md` and c2a_plot_style. Provide browser-accessible figure URL through existing task figure route. ADD-map historical comparison remains a separate artifact; never merge its points into the ~1M panel.

Upload raw analysis inputs and outputs to a new task-specific HF prefix, verify remote revision/file count/sizes/content hashes, then record successful completion via task CLI. Fresh source-matched sentinels, zero exit, timestamps and content validation all required. No cleanup or compute teardown before verified upload. Explain planned-versus-realized coverage and exact limits in final result. The final response should state whether H1/H2/H3 hold per behavior and link the browsable figure and durable outputs.

## 11. Decision rationale and grounding

| Decision | Value/source | Why |
|---|---|---|
| Model/layer/map | Qwen2.5-7B, L19, mixed_1m ridge; #779 archived map | User requested one generic map/common layer; avoids behavior-selected layers |
| Map pool/split |963444train,400val,1000test,seed42; #779 assemble/select_train| Match frozen trained artifact exactly |
| Ridge grid |logspace(-3,8,23); #779 LAMBDAS_N1M| Same regularization family and selection procedure |
| Standardizer |train sd ddof1+1e-9, fp64 fit/fp32 persisted; #779| Numerically load-bearing historical recipe |
| Extraction |five contrastive prompt pairs,20questions,10rollouts; #1739 generation/#779 persona recipe| Matched context/answer supervision |
| Extraction filtering |None; root decision2026-09-16; polarity-defined archived E1 rows| Explicit fixed instruction contrast rather than mislabeled filtered Persona Vectors |
| Null seeds |0–4; #1739 claim4 controls precedent| Expose stochastic-control variation without downstream model tuning |
| Bootstrap |2000 paired group draws; #1739 covariance follow-up| Stable per-rung conditional intervals and direct method comparisons |
| Null validation |independent within-val Y permutation| Removes real pair information from fit selection as well as fit weights |
| No full covariance whitening |#779 raw-space standardizer/apply_map| Directions and map output must share coordinates |

## 12. Verified facts, unresolved assumptions and stop conditions

High-confidence code-verified: exact #779 standardizer and lambda grid; train-only fit with distinct original val/test; mixed_1m whole-pool selection; canonical payload quantization/application; E1 context and answer loader identity; old factorial whitening and map963k weight-row-shuffle mismatch. Task Goal was re-read unchanged before writing this plan.

Artifact facts still require independent verification by root inventory: full 963444 input ordering and model/capture compatibility; exact frozen-weight HF hash and target provenance; presence of L19 for every E1/eval store; judge-filtered extraction evidence; same-rollout DV/activation alignment; all original and new map-prompt hashes for content leakage audit; complete intended OOD roster. These are not assumed true from docstrings or historic prose. Further source inspection found #1739 dispatcher capture includes every E1 extraction rollout, while its judge phase calls the judge only for labeling rollouts (`scripts/issue1739_dispatch.sh:278–295`). The separately generated #779 persona bank has filtering in `issue779_extract_rb.py` with local `judge_{trait}_{arm}.json`; text uploads under `issue779_monitoring/raw_completions/rb_extraction/`, vectors/counts under `issue779_monitoring/r_b/`. Those scores must not be transferred onto independently generated #1739 rows. The #779 vector/text upload helpers do not themselves upload the raw judge JSON, so search archived run bundles before asserting those files are absent.

HALT a scientific claim on mismatched model/layer/coordinate data beyond the explicitly declared answer-pooling difference, untraceable map pool, absent evaluation retained-score provenance, missing join identities, wrong key set, nonfinite mandatory score, or stale resume fingerprint. Extraction judge scores are not required for the disclosed unfiltered contrast. Surface the concrete missing evidence and continue independent staging/implementation while root resolves it. Do not replace a missing method with a superficially similar archived metric. Low predictive performance is a valid completed result, never grounds for changing extraction, layer, evaluation or hypothesis.

Sources: [Persona Vectors](https://arxiv.org/html/2507.21509v2). Search output: `/tmp/issue1739-transfer-persona-search.json`.
