# Qwen3-8B necessity-subset rank analysis

User request (2026-09-09): "run the same analyses on that subset". Existing-artifact,
CPU-only analysis under issue2546; no model inference, GPU, new task, or paper edits.

## Question and comparisons

On questions answered correctly only with thinking enabled, does the lower mapping
rank seen on general chat persist, and how much does it track input/answer diversity?
Use the existing correctness labels: necessary (4,522) and both-correct (17,693),
from the 33,810 original paired benchmark rows, seven corpora, one greedy answer per
mode. This operational label is not proof of intrinsic reasoning necessity.

Three cached layer-24, d=4,096 maps: no_think = p7_Aoff (thinking-off cx_last to its
own ans_mean); context = p7_A (thinking-on cx_last to thinking-on ans_mean);
end_of_thought = p7_D (thinking-on cot_boundary to the SAME thinking-on ans_mean).
The no_think/end_of_thought comparison changes targets; the context/end_of_thought
comparison holds targets fixed. Report both, without treating them as interchangeable.

## Frozen recipe and reuse

Reuse scripts/issue2546_qwen3_rank_reproduction.py from e937d9be7d8, including
shared fold moments, sample-standardized ridge, TRAIN-output PCA and cumulative SSE.
Only add an explicit mode parameter to its cached-state loader (default unchanged).
No new penalty/layer search: use the original allfit/results JSON penalties
(1000, 1000, 316.2277660168379). Verify reconstructed outer predictions against
original float32 OOF banks (rtol=atol=2e-6), plus full-panel R2 parity. Inputs,
labels, predictions, metrics and helper sources are hashed and row/fold aligned.
Local source root: /mnt/eps-data/thomasjiralerspong/cot_necessity. No downloads,
staging transformation, source-file edits or cleanup. Source labels in the root
repository are checked against labels stored in the prediction banks.

Retain ALL training questions, as in the existing necessity section. Original five
IID outer folds: test k; validation (k+1)%5; train remaining three (20,286 > d).
Rank selection uses only the relevant validation subset, then refit on four folds
(27,048 > d) and score the matching held-out subset. Smallest rank allowing 10%
extra validation SSE; 5% and 20% sensitivities, no rank-by-rank fits. This nests rank
selection only; inherited lambda/layer selection is not retrospectively nested.
These are within-benchmark IID results, not held-out-corpus generalization; retaining
the exact parent folds is intentional for the requested same-analysis comparison.

Report corpus-training-mean and global-training-mean R2, full rank curves,
identity-plus-training-bias and full/reduced/identity retrieval using the original
training-whitened CSLS pool of 6,762 answers (chance 1/6762), filtering QUERIES only.
For the two subset-selected ranks the candidate pool remains identical; all candidate
queries use the same projected map for the CSLS density correction.

## Diversity and uncertainty

On each subset of each OUTER TRAINING fold, measure centered standardized-input,
answer and fitted-output covariance spectra. Standardization uses all-training-row
parameters; centering for diversity is within the subset. Same entropy effective
rank exp(H(p)), participation ratio and stable rank as the chat control, where p
normalizes covariance eigenvalues (squared singular values, not singular values).
Report raw-input PR too, the fitted/answer ratios, and all five dependent-fold values.
Do not interpret these ratios as a causal decomposition. No CI on spectral ranks;
dependent-fold ranges are not CIs. Between-subset comparisons may reflect unequal
sample sizes and dataset composition; emphasize paired within-subset changes and
report per-corpus R2 and counts.

Repeat the chat conditional rank bootstrap (4,000 draws, seed 20260909), prespecified
outer fold 0, pairing validation questions across all three maps. Stratify by corpus
to fix mixture weights. Each draw reselects rank against its own full-map SSE;
hold ridge/PCA/layer/lambda fixed. Save draws, not only CIs. Also conditional paired
OOF-row bootstrap for R2 differences, stratified by corpus and subset. No p-values
over dependent folds, causal claim, or claim these conditional intervals capture
fit/target-generation uncertainty. No repeated-answer control unless compatible
aligned repeated answer states exist; do not generate missing answers.

## Compute, output and verification

Three maps x five outer folds x inner/outer = 30 multi-output ridge solves, with
fold moments shared and all 4,097 ranks vectorized. Diversity eigensolves are only
per actual space/subset/fold, not per rank or bootstrap. Bootstrap counts multiply
fixed per-row sufficient statistics in batches. One CPU process, eight threads,
arena cap 2 and mmap threshold 131072. Measure one complete map/fold at production
shape first, including both subsets and bootstrap. Stop if 15 x pilot total exceeds
about one hour; provision nothing. Preserve the completed pilot and resume it.
Detach with PID/log breadcrumbs for the full run, checkpoint per map/fold, fingerprint
all output-affecting sources/parameters. Current VM: 111 GiB available RAM, 143 GiB
root disk free; prior two-map rank run used 8.60 GiB and 634.7 seconds. No >5GB new
staging. New output expected under 100MB, on root filesystem at
eval_results/issue_2546/qwen3_necessity_rank/, with report and reproducible plots.

Test cumulative row scoring and batched rank selection against explicit projections,
PSD/spectral definitions, pairing, and source invariants. Independent review before
reporting; focused tests, mapped tests and payload-attributed workflow lint. Commit
and push only this round's paths, preserve main/other worktrees and Overleaf.
