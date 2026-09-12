# J/R workspace component predictability: frozen exploratory protocol

Prepared 2026-09-11 from the user's supplied protocol, before any new J/R
component test result was generated or inspected. This file is a preparation
artifact. The user subsequently authorized execution without registering a
repository experiment task: "I allow you to bypass the registered experiment
task." This exception changes the workflow boundary, not the research protocol.

## Research question and estimands

Does final-context residual x predict sparse answer-state components less
well than their respective remainders, and does the comparison change with
J versus R or model capability? We measure relative predictability. Linear
predictability is not automaticity, and residual error is not reasoning.

For each answer-token residual h at the mapping layer, apply the cited
nonnegative gradient-pursuit algorithm separately to unit-row J and R
dictionaries. Define s_S(h) and h-s_S(h). Average tokens within each rollout,
then give each rollout equal weight within context. The two identities are
y=y_J+y_rest,J and y=y_R+y_rest,R. J and R are alternative, possibly
overlapping decompositions. Our operational R-space applies the J sparse
procedure to the R dictionary; it does not establish other workspace properties.

Fit affine ridge and MLP maps from the same x to all four targets, with each
target's own centered variance denominator. Primary estimands are
G_J=R2_rest,J-R2_J, G_R=R2_rest,R-R2_R, and G_R-G_J. Report component
SSE, variance, component/remainder covariance, and MLP-minus-ridge gains.
Use 0.05 absolute R2 as the user-specified smallest practical gap.

## Model selection, frozen before new outcome reads

Selection is by measured same-mode GPQA performance among artifact-valid
final-context-token maps. Mapping R2, model size and estimated capability
indices do not select the model. Report statistical uncertainty in capability
ordering and conflicting benchmark evidence. The primary is the highest
observed-performance representative in the audited usable panel; unique
superiority is unproven. Exact point ties use lexical model-ID order. There
is no architecture-based substitution rule. Inaccessible former charmander
outputs were not certified absent by the inventory.

The audited candidates are **Qwen/Qwen3.5-27B, no-thinking**, and the
meaningfully weaker same-family **Qwen/Qwen3.5-4B, no-thinking**. Existing
map source layers are zero-based post-block outputs 50/64 and 20/32,
respectively: relative depths 51/64 and 21/32. These depths are different
processing stages, despite using each model's existing selected layer.
These are a dense hybrid-attention family; post-training, width, depth,
relative layer position and generation behavior remain cross-model confounds.
Within-model J/R comparisons are primary.

The source inventory and actual fitted payloads must be read from the
companion provenance audit. The original generation did not record checkpoint
SHAs. Current Hub revisions predate the original runs, so historical-main
identity is an inference, not producer-attested provenance. Re-capture a
32 hash-first calibration contexts with historical captures, using the
historical retokenization recipe. Compare x and y separately to pinned old
arrays: require relative Frobenius error <=0.01 and every nonzero row
cosine >=0.999. These engineering tolerances are ungrounded and require a
dtype-controlled pilot; they cannot be relaxed after main-test inspection.
A mismatch blocks frozen reuse and the original compatible-map premise.
Diagnose/revise eligibility before main execution, with no model substitution
or assertion that a fresh full-y fit establishes successful historical reuse.

The selected inherited training maps use **K=1**, with extra test-only draws;
they are not K=5 maps. Existing cached answer targets are pooled, and raw
generation artifacts omit exact generated token IDs. The primary new targets
therefore require fresh token-level K=5 generation/capture. Preserve exact
prompt IDs, completion IDs, finish reasons, model/tokenizer revisions and seeds.
Replaying stored text is a provenance/parity diagnostic, not exact token replay.
Freeze the new token contract: native chat template with
`enable_thinking=False, add_generation_prompt=True`; temperature1.0,
top_p0.95, unrestricted top_k, min_p0, repetition penalty1 and other penalties0;
rollout seeds42 through46. Save actual rendered prompt IDs and generated IDs.
Capture their exact concatenation. Include every generated token before the
first terminal EOS/stop token, preserving whitespace; exclude terminal EOS
and padding. Retain and flag unexpected thinking/control tokens in the
generated answer rather than silently changing the span. This differs from
the historical whitespace-stripped retokenized answer representation; compare
both representations on calibration only and label the frozen readout caveat.
Start with2048 new tokens (inherited no-thinking recipe). When cap-hit rate
exceeds2%, regenerate cap-hit draws at doubled cap with the same seeds, until
complete or the native context-window limit. Keep all cap hits visible even
below2%. Primary paired contexts require all five nonempty completed draws
in both models. Exclude a whole incomplete context from paired primary
scoring, while retaining its raw data, reasons and realized-K noise diagnostic.
Do not replace it with a newly selected prompt or discard its failure record.

## Sample selection and separation

Use the common full source manifests (first10000 train rows and all400/1000
validation/test rows), irrespective of old generation/capture survival.
Historical survivor intersections are for parity diagnostics only. Identify
content as SHA256(UTF-8(NFC(raw_prompt))), with no whitespace changes, preserving
raw prompt bytes alongside it. Keep all identical prompts and their rollouts
together. Deduplicate with train/validation/test ownership first, select
calibration second, pilot third, main fourth. Persist selected IDs and hashes
before outcome reads; audit all exclusion intersections by content key.
The inherited val/test manifests share 13 exact prompt texts: validation-seen
prompts are excluded from test. Train ownership precedes validation, which
precedes test. Report all content collisions and exclusions, including any
additional collisions from Unicode normalization.

Choose within each eligible split by SHA256(seed, context-key), never by
prediction error, lens stability on test, length preference or capability
outcome. Pilot: 64 train/16 validation/32 test contexts, K=5. Main ceilings:
8192 train/384 validation/768 test contexts, K=5; use min(ceiling, eligible
shared count), report realized counts, never backfill across splits. The
pilot test subset is always tagged exploratory and excluded from the
subsequent main test, regardless of subsequent changes. Validation may guide engineering;
no test-guided direction,
layer, model or sparsity selection.

Hold out 128 additional hash-selected training prompts for calibration only.
They are excluded from component-fitting and all evaluation subsets. This is
an on-distribution calibration approximation to the papers' Pile recipe,
shared exactly within model across J/R. Record train-domain calibration as
a limitation. Use split halves and nested 32/64/128 prompt subsets to report
stability. Small-corpus stability is not evidence of full-corpus convergence.

For an established corpus/genre grouping, add a train-corpus to held-out-corpus
transfer diagnostic using the same exclusions and fixed protocol. If the
available dataset has only one usable corpus, report this missing validity
check; do not claim cross-corpus generalization.

## Matched lens construction and validation

Released Qwen3.5 J/R pairs exist, but metadata does not identify exact
checkpoint revisions, calibration documents/token IDs, or implementation
commits. Their `git_commit="modal"` and `n_positions=0.0` are not sufficient
provenance. Use the release as a labeled calibration-readout diagnostic;
construct the primary pair with pinned native checkpoints and calibration
manifests unless full provenance can be recovered.

Reuse the vendored official J estimator: within each prompt, sum valid target
position contributions, average valid source positions, then equally average
prompts. Target post-block outputs 62 and 30 (penultimate), max 128 tokens,
skip first 4 and final token. These match the released pair's explicit
target-position convention, not the official API's different defaults.
The source remains the fixed mapping layer. Accumulate and save matrices
in fp32; native model forward bf16. Both lenses must consume identical exact
token manifests, precision, layers and masks. Persist per-prompt checkpoints
with full configuration hashes; do not rely on the upstream resume key alone.

The dense R variant detaches residual RMSNorm denominators; uses SiLU's
detached sigmoid factor; and halves both backward branches of the MLP
product. Leave attention, q/k norms, gated attention norms and hybrid
recurrence ordinary, as specified by the dense release. Qwen3.5 residual and
final norm effective gain is **1+weight**. Patch by module path and supported
architecture, not every instance of the norm class. Unknown/MoE modules fail
explicitly. Do not substitute ordinary softmax attention for the hybrid path.

Validate ordinary J products by finite differences and adjoint identity on
the actual source-to-target downstream function; use fp32/fp64 diagnostic
precision when bf16 finite differences are unresolvable and declare it.
Validate R against the specified local propagation rules, independently of
finite differences. Verify unchanged full forward and hook outputs with rules
installed, including the actual selected checkpoint. Inspect calibrated token
readouts and report split-half matrix/direction similarity. Prototype tests on
a tiny randomly initialized Qwen2 do not certify Qwen3.5 compatibility.

Dictionary rows are `(W_U[token] * effective_final_norm_gain) @ lens`, unit
normalized after rejecting/reporting nonfinite and zero rows. Folding the
norm gain follows the native readout algebra; the original sparse code was
not released. R-space uses this same stated normalization convention.

## Sparse decomposition and controls

Use the exact nonnegative gradient-pursuit algorithm cited by the J paper:
start coefficients at zero; each step selects argmax signed residual
correlation (reselection allowed), uses the gradient on positive-coefficient
support plus the selected atom, takes its exact line-search step, and clips
coefficients nonnegative. k=10 is primary; k=5/25 sensitivities. k counts
iterations and bounds L0; report actual L0. No top-512 candidate shortcut,
clamped least-squares substitution, PCA projector or global linear projector.
Zero update norms are recorded no-ops. Report increased-error steps and
nonfinite failures explicitly, without clipping reported quality metrics.

Use three seeded Haar rotations for each dictionary, preserving all pairwise
atom geometry and using identical sparse selection/aggregation. Report
rotation variation in G, sparsity, reconstruction SSE and captured variance.
Where calibration distributions overlap, compare cells matched on calibration
reconstruction quality/captured variance; do not fabricate matches or tune
controls on test. Report raw G and its difference from mean rotated-control G,
with paired context uncertainty and separate across-rotation spread.

Run a synthetic null whose token targets are exactly affine in x before
decomposition. Preserve the registered rollout-length layout, and verify the
pooled original target is affine. Reapply actual and rotated dictionaries,
component fits and metrics. This isolates sparse-selection-induced
nonlinearity; it is not synthetic evidence about actual language processing.

## Predictors and tuning

Reuse frozen full-y ridge/MLP predictions only after provenance and row-identity
validation, and only on shared test contexts unseen by either predictor's
training/selection. If no compatible frozen MLP exists, report it missing and
fit a full-y comparator on the same fresh subset and budgets as components.

Component ridge shares one x factorization and selects alpha separately by
validation SSE from the nine fixed values in the configuration. Standardize
x coordinates using training statistics; center each target and scale by one
training scalar to preserve target geometry. Evaluate saved predictions in
original units. An affine bias is always fitted. Also report identity plus
learned bias and held-out Euclidean/cosine retrieval at k=1/5/10, with pool
size and chance k/n. Reject k exceeding realized pool size.

MLP uses the existing batched `fit_batched_split_mlp` implementation (one
GELU hidden layer, no mandatory PCA output truncation), widths 512/2048,
learning rates 1e-3/3e-4, weight decay 1e-4, at most 300 epochs, patience30,
seeds42/137/271. Select hyperparameters by mean validation SSE across seeds;
report each selected seed on the same held-out examples. Do not call the
seed-ensemble prediction the mean single-model score. Every target/lens/control
gets the same tuning budget and scaling treatment. These candidate budgets
are **ungrounded—need pilot learning/convergence checks**; the existing helper
supports them but does not validate them for this decomposition. Record any
pre-main revision as a new frozen plan before reading main test outcomes.

Use nested 25/50/100% training curves for ridge and the selected MLP recipe.
Insufficient data/convergence can make a component contrast inconclusive.

## Readout diagnostic

Eligibility is fixed from calibration/training: finite nonzero dictionary
rows; exclude special/empty decoded tokens; require at least five observed
token occurrences. Use the same token IDs for J and R, selecting at most2048
by seeded token hash. Normalize directions. Report per-direction held-out
R2, target variance, MLP-ridge differences, paired J/R differences and
corresponding direction cosine. Call these J-aligned/R-aligned readouts.

Random and training-PCA controls are matched without replacement on training
log target variance, maximum absolute mismatch0.2, with an eight-times-sized
random pool and all available PCA directions. Report mismatch distributions,
redundancy/effective rank, and unmatched exclusions. Scalar rescaling is not
a substitute for matching normalized directions. Controls are not established
non-workspace features, and these readouts do not decompose the predictor.

## Noise, agreement and inference

Estimate per-context unbiased rollout covariance trace for each component
and remainder, as well as cross-component covariance. Compare noise of the
K-rollout mean to across-context target variance. Noise-corrected ratios are
secondary and undefined when denominators are unstable. If estimated noise
of a mean exceeds10% of target variance, repeat the key contrast on a
hash-selected128-context K=20 subset, paired across lenses and models, seeds42
through61. Reuse the K=5-trained predictors with no retuning; retain K=5 as
primary and the higher-K read as a noise diagnostic.

Bootstrap2000 context draws with identical context multiplicities across
lenses, predictors, dictionary controls and models on shared prompts. State
that intervals condition on fitted predictors/calibrated dictionaries; report
seed/rotation variation separately. Retain undefined zero-variance draws,
failed decompositions, missing examples, truncations and exclusions as explicit
statuses. Report component similarity only above 1e-6 times the training
median full-target norm, and report raw differences alongside it; avoid
near-zero ratios/cosines. The768-context ceiling is not a power guarantee:
interval width above0.10 is inconclusive for a0.05-scale distinction. Any
sample-size change must use train/calibration/validation precision estimates
and be frozen before main outcomes, never a borderline main-test gap.

Positive gaps under both lenses beyond noise and decomposition controls support
a deficit in the measured sparse components' linear predictability. A gap
that vanishes/reverses with R is lens-dependent. Equal/better component
prediction is contrary evidence. Wide intervals, unstable lenses, failed
controls, data/convergence limits or dominant noise are inconclusive. Larger
MLP gains support nonlinearity without establishing reasoning. Agreement
between related lenses is a robustness check, not independent causal evidence.
Two-model differences are observational, not evidence capability causes them.
No claim that remainders contain all non-workspace computation or that
workspace contents are transient. Causal and temporal studies remain deferred.

## Execution, artifacts and current boundary

The user explicitly authorized bypassing experiment task registration. Proceed
under that exception without creating or repurposing a task. Preserve this
research question, frozen sample selection, statistical protocol, provenance
checks and normal compute/artifact safeguards. The original selection manifest
retains the pre-authorization configuration hash; execution records bind both
that immutable manifest and the current configuration.

Code preparation and meaningful local unit/integration tests may proceed.
The current VM runtime lacks the built-in Qwen3.5 class; use a
repository-managed compatible runtime on the approved execution backend,
without modifying the shared environment. Local preflight measured no CUDA and
only 65.6 GiB free on `/mnt/eps-data`, below the 80 GiB local safety floor, so
do not stage the 27B checkpoint or main token store here. Use
GCP-first existing dispatch infrastructure under the user exception, provision
only after reading the full compute/pod/upload rules and a concrete resource
check. Initial pilot hardware target is one80GB GPU with phase-separated
generation, lens fitting and decomposition; actual batch width/HBM fit and
wall time require measurement. **No production GPU-hour estimate is yet
validated, and no production reservation is authorized by this document.**

Persist input manifests, full request, config hash, code SHA/dirty status,
per-prompt lens matrices and validation, generation text/token IDs,
per-rollout aggregates, per-example predictions, fitted weights, bootstrap
draw identifiers, per-cell JSON and resumable sentinels. Upload regeneration-
costly data before releasing compute; verify remote objects and hashes.

Smoke blind-spot enumeration: current tests execute local propagation rules,
the cited sparse algorithm, aggregation/statistics, and a tiny actual Qwen2
architecture with random weights. They do not execute pretrained Qwen3.5,
native hybrid backward kernels, vLLM generation, exact token capture, Hub
roundtrip, production-scale MLP, or full end-to-end lens construction.

Required final output after launch: machine-readable results and saved
predictions/statistics; five plot views covering component R2, MLP gains,
decomposition agreement, control-adjusted gaps, and stronger/weaker results;
browser-accessible URLs for every presented figure using the existing C2A
style; concise factual report with continuous estimates, intervals, contrary
evidence and limitations. Keep manuscript edits separate.
