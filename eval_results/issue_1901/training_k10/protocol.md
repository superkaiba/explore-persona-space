# Training rollout count on the existing 19,000-context pool

source: user-chat
followup_label: training-k10-19k
question_relation: same
owner: codex-training-k10-20260911

User authorization: "can we do existing 19k pool up to k = 10" (2026-09-11), following the proposed 19k-pool experiment with fixed evaluation K=10. This extends task 1901's mapping-quality question; its canonical goal is unchanged. No new task or model fine-tuning is required.

## Question and design

Does reducing answer-sampling noise in the TRAINING labels improve a linear context-to-answer map, when context count and evaluation targets are fixed?

For every one of the existing 19,000 LMSYS distractor contexts, retain its original on-policy answer vector and stored draws 43–46, and add draws 47–51. Fit separate linear ridge maps at K_train=1,2,3,4,5,6,7,8,9,10, using the mean of the first K answer vectors in the fixed order original,43,…,51. Use all 19,000 contexts in every final fit. This fixed ordering makes the curve conditional on this bank; it is not exhaustive averaging over subsets or training-bank uncertainty.

Select regularization by training-only generalized cross-validation (GCV); do not select any hyperparameter using the test targets. Use the #779 standardized, centered ridge convention (sample standard deviation, ddof=1, plus1e-9) and #779 n1m lambda grid (23 log-spaced values from 1e-3 to 1e8). Include the fitted intercept in GCV degrees of freedom: df=1+sum_j s_j/(s_j+lambda), as in the corrected #922 convention. This intentionally corrects the intercept omission in the older #779 GCV helper. Share the input factorization and per-draw cross-products across K; verify the primal GCV calculation and predictions against a direct independent hat-matrix solve. An edge selection is reported and triggers a numerical/grid diagnostic, never silent clipping or test-based tuning. Linear maps only; identity plus a learned training bias is fit separately at each K. No new nonlinear arm.

The test bank is the exact original 1,000 held-out contexts, with all ten already stored answer vectors. The primary comparison holds K_eval=10 fixed. Also retain scores for K_eval=1,…,10 as a descriptive train-K × eval-K grid; this requires no new generation. The held-out retrieval identities use the existing original-vector keep-one policy (942 candidates, chance 1/942), fixed across every cell. Audit exact and inherited near-duplicate train/test exclusion before launch; report any residual limitations.

## Metrics and inference

Primary dependent variables: pooled held-out R² and strict top-1 retrieval using whitened cosine plus two-sided CSLS (neighborhood 10). Raw cosine and Euclidean top-1 are companions. Fit whitening once on the 19k ORIGINAL training answers with the parent's shrinkage convention, then freeze it across K_train and K_eval. Thus changing K does not change the distance transform. Include identity plus learned bias at every K.

Pre-register K_train=10 minus 1 at fixed K_eval=10 as the primary contrast; K_train=5 minus 1 and 10 minus 5 are secondary descriptive contrasts. Use 2,000 paired prompt-identity cluster bootstrap draws (seed 190141, #1901), recomputing each R² denominator and holding the retrieval candidate pool fixed. Retain the1000-row R² point estimate; cluster resampling draws the942 unique prompt hashes and carries each selected cluster's complete multiplicity, with each replicate's actual row count in the centroid denominator. Apply the same cluster draws to the942 unique retrieval queries. This accounts for the repeated test prompts rather than treating them as independent observations. Confidence intervals condition on the training bank, fitted maps, evaluation draws, and candidate pool; they do not quantify training-set/rollout resampling uncertainty. Report estimates and intervals without an arbitrary success threshold, best-K selection, or equivalence claim.

## Reuse and validation

Model: Qwen/Qwen2.5-7B-Instruct at a09a35458c702b33eeacc393d103063234e8bc28. Layer 19, hidden dimension 3584. Generation: vLLM, engine seed 42, per-request seeds47–51, temperature1.0, top_p0.95, cap1024, max_model_len8192 (#1901 k10 recipe). Capture: the inherited full-template retokenized answer-span mean including the end-of-turn tail, using issue1901_k10_capture.capture_rows. The inherited span convention is retained for compatibility; it is not silently replaced by a different tokenization method.

Preserve the old bank's narrow generated-JWT handling through the exact inherited `issue1901_avgpool_scaleup._pipeline_scrub_pre_upload` helper: known noncredential JWT-shaped generated strings receive same-length masking before capture and upload. Retain original generated token IDs and original text SHA, persist every disclosure, and report realized affected-row counts. Captures use the disclosed edited text; this is a compatibility limitation, not a different hidden filtering policy. Other credential findings fail rather than receiving a new broad scrub rule.

Existing training-draw files: issue1901_avgpool/analysis_tensors/kresample/V_distr_shard00…03.npz; original answers: issue1901_metrics/analysis_tensors/distractors_L19.npz; context identity: issue1901_avgpool/analysis_tensors/bundle/bundle_index.json. Locally inspected: four arrays of shape(4750,4,3584), seeds43–46, positive span counts, all finite; their unique CI union exactly matches the 19k index, and all original answers join once by CI. The 19k prompts are unique and have zero exact-hash overlap with the 1k test prompts. Context-vector provenance, prompt recovery, and remote pins must pass the preparation audit before GPU dispatch. Every consumed artifact is pinned and byte-verified in the input manifest; no silent path fallback.

Each GPU worker must recapture a fixed sample of stored seed43 answers and compare with their actual stored vectors before generating any new answer: cosine≥0.999 (#1901 opsurface), with relative L2 recorded. Use fresh processes for parity, vLLM generation, and HF capture. Checkpoint raw generations per≤500-row chunk, captures per chunk, and upload with remote content verification before advancing. Resume predicates include complete recipe, ordered IDs, and generation hashes. Save cap-hit counts; no new cap change without recording its implications for bank compatibility. A missing/invalid draw fails the fixed-coverage run rather than substituting zeros or changing K silently.

## Compute and persistence

Estimated GPU-hours (total): 8

Expected new draws: 19,000×5=95,000. Measured same-recipe H100 basis: 852.8s generation plus361.8s capture for5,000 draws, including per-chunk upload (2026-09-07 k10 execution log), giving6.41 GPU-hours before setup and source/parity work. Use up to four independent GPU workers with explicit launcher CVD pins; route through the current canonical backend with at least40GiB HBM per GPU. Each lane measures a full production-size first chunk before continuing; report a changed projection if realized throughput differs. No co-resident generation and capture models. No judges or Claude automation.

Keep local VM preparation/analysis below50GB; the dedicated worktree is /mnt/eps-data/thomasjiralerspong/wt-1901-training-k10. Live preflight on2026-09-11 found approximately49GiB free on the data disk and48GiB on root, so do not stage the full n1m capture here. The source audit identifies exactly40 n1m chunks (1.73GB) containing the required context vectors and prompts; stage those directly, along with the6.02GB pass_b mmap for the test inputs and the small draw banks. GPU output root /workspace/outputs/issue1901_training_k10 is sized for≤40GB including model and environment-independent data, with at least1.5× verified headroom on its actual mount. Plan final input/output data≤8GB, fitted ridge weights≤1GB, with intermediate cross-products retained only if needed for reproducibility.

Local artifact root is `/mnt/eps-data/thomasjiralerspong/issue1901_training_k10`, outside the sparse worktree, so checkout maintenance cannot prune ignored inputs. Existing source banks remain in their original locations; the portable bank is about1.2GB. A sparse-checkout update removed an earlier derived staging copy during preparation; it was regenerated and validated before publication, with no GPU generation involved.

Persist raw text/token IDs, captured vectors, row manifests, generation configs, timing/parity reports, fitted maps, predictions, and bootstrap analysis inputs to HF under issue1901_training_k10. Commit protocol, reviews, numerical summaries, and report under eval_results/issue_1901/training_k10; figures use repository c2a style and must have browser-accessible URLs. Require verified remote path set, sizes and hashes plus a fresh successful completion sentinel before termination. Maintain a durable monitor/heartbeat and leave compute running only while productive work remains.

## Checks

Before launch: independent protocol/artifact review; focused tests of ID joins, uneven final chunks, stale-recipe rejection, GCV/direct-solve parity, fixed-pool evaluation, and bootstrap differences; Python import and Ruff checks. Real model parity and a full-sized generation/capture chunk exercise the production path on each GPU.

Smoke blind-spot enumeration: local numerical tests do not exercise CUDA/vLLM or cloud transport; the on-GPU parity and first production chunk cover these before the remaining chunks. No substitute model or weakened scientific assertion is allowed. Later prompt-length tails and long-run transport interruptions remain covered by per-chunk validation/checkpointing, not certified by the initial smoke.
