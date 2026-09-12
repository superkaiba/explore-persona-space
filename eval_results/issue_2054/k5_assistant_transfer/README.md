# K5 transfer from the chat assistant

[Open the assistant-source transfer figure](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue2054_section44_k5_gcp/transfer_calibration_v1/assistant_sources/figures/assistant_source_transfer.png).

This comparison uses the same K5 activations and global five conversation
folds as the completed six-setting experiment. For each model checkpoint,
five training sets are evaluated: chat assistant alone, and chat assistant
plus HELIOS, Wren, Dana, or Vex. Each source map is fitted once per conversation
fold and applied to every setting absent from its training set. The held-out
conversation fold is excluded from all source settings and from target
calibration. No best character is selected after looking at test results.

The planned coverage is 50 fitted source maps and 42 source-set/target-setting
panels, each with five held-out folds (210 evaluations). A dash in the figure
means the target setting was included in source training and therefore is not
evaluated as unseen-setting transfer. It is not a failed or missing run.

**Frozen transfer** uses no target-setting labels. **Bias** fits one vector
intercept on the target's training conversations. **Bias + scaling** fits one
shared scalar plus a vector intercept on those same training conversations,
with the source map fixed. Test targets do not enter either calibration fit.
The same source map produces the calibration and test predictions.

The source-only ridge recipe is unchanged from the parent: standardized
inputs, GCV over `logspace(-2, 4, 13)`, and a degrees-of-freedom cap of 0.9.
The source map and training-set means/scales are saved in float64 under
`maps/`; target calibration coefficients and fold metrics are under `folds/`.
Both identity-plus-source-bias and identity-plus-target-bias baselines, the
target's original separate-map reference, and held-out Euclidean/cosine
nearest-neighbor retrieval are retained in the JSON. Retrieval includes the
candidate-pool size and chance level. Figure cells are equal-weight means of
the five fold R² values.

Adding a character adds another setting's training examples; this comparison
does not match total training row count. The four story characters also use
an attributed-story format, whereas the assistant source uses the chat format.
Consequently improvements cannot be attributed uniquely to persona identity,
and these representation scores do not measure refusal behavior or prove a
constant steering-vector mechanism.

Reproduce after staging the inputs with
`scripts/issue2054_k5_loso_calibration.py --stage stage`:
run `scripts/issue2054_k5_assistant_transfer.py --stage fit` once per
`--model qwen2.5-7b` and `--model qwen2.5-7b-instruct`, providing the staged
input directory with `--inputs` and a common result directory with `--out`.
Then use `scripts/issue2054_k5_assistant_transfer_plot.py --out ... --fig-dir ...`.
The collector verifies every fold and persisted map before plotting.

On the shared VM, the successful run used `NUMPY_MADVISE_HUGEPAGE=0`,
eight threads for OMP/MKL/OpenBLAS/NumExpr, `MALLOC_ARENA_MAX=2`, and
`MALLOC_MMAP_THRESHOLD_=131072`. The huge-page opt-out resolved long stalls
while allocating NumPy arrays; it changes allocation behavior, not the fit.
The retained source map passed the same numerical parity check after resuming.
Package versions and the verified worker environment are in `runtime.json`.

The source data are the completed
[six-setting K5 experiment](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9de026f872c19b2ca4fd3e4539de820e08038ee3/issue2054_section44_k5_gcp/production_v1).
Their revisions and SHA-256 hashes are stored in the linked LOSO-calibration
`inputs.json`; each subset result records that manifest's hash.

## Results

All 50 source maps and all 210 held-out evaluations completed, covering 42 source-set/target panels.

Character-only averages below omit the plain-text assistant target. The +one-character rows average equally across all four added-character choices and their three unseen character targets; every target character appears equally often.

| Checkpoint | Training settings | Frozen R² | + Bias R² | + Bias + scaling R² |
|---|---|---:|---:|---:|
| Base | Assistant only | -0.553 | 0.153 | 0.155 |
| Base | Assistant + one character | 0.457 | 0.471 | 0.473 |
| Instruct | Assistant only | -0.472 | 0.150 | 0.206 |
| Instruct | Assistant + one character | 0.457 | 0.471 | 0.478 |

### Base: all target panels

| Training settings | Target | Frozen R² | + Bias R² | + Bias + scaling R² | Separate-map R² |
|---|---|---:|---:|---:|---:|
| Assistant only | Assistant plain | -0.028 | 0.203 | 0.205 | 0.469 |
| Assistant only | HELIOS | -0.485 | 0.176 | 0.176 | 0.530 |
| Assistant only | Wren | -0.442 | 0.170 | 0.171 | 0.529 |
| Assistant only | Dana | -0.444 | 0.160 | 0.160 | 0.544 |
| Assistant only | Vex | -0.843 | 0.104 | 0.114 | 0.482 |
| Assistant + HELIOS | Assistant plain | 0.016 | 0.238 | 0.251 | 0.469 |
| Assistant + HELIOS | Wren | 0.472 | 0.482 | 0.482 | 0.529 |
| Assistant + HELIOS | Dana | 0.454 | 0.475 | 0.475 | 0.544 |
| Assistant + HELIOS | Vex | 0.403 | 0.420 | 0.423 | 0.482 |
| Assistant + Wren | Assistant plain | 0.047 | 0.240 | 0.250 | 0.469 |
| Assistant + Wren | HELIOS | 0.457 | 0.477 | 0.478 | 0.530 |
| Assistant + Wren | Dana | 0.514 | 0.516 | 0.517 | 0.544 |
| Assistant + Wren | Vex | 0.434 | 0.444 | 0.446 | 0.482 |
| Assistant + Dana | Assistant plain | 0.006 | 0.230 | 0.243 | 0.469 |
| Assistant + Dana | HELIOS | 0.431 | 0.464 | 0.464 | 0.530 |
| Assistant + Dana | Wren | 0.504 | 0.506 | 0.506 | 0.529 |
| Assistant + Dana | Vex | 0.420 | 0.436 | 0.439 | 0.482 |
| Assistant + Vex | Assistant plain | -0.008 | 0.237 | 0.243 | 0.469 |
| Assistant + Vex | HELIOS | 0.438 | 0.458 | 0.463 | 0.530 |
| Assistant + Vex | Wren | 0.478 | 0.484 | 0.487 | 0.529 |
| Assistant + Vex | Dana | 0.482 | 0.488 | 0.493 | 0.544 |

### Instruct: all target panels

| Training settings | Target | Frozen R² | + Bias R² | + Bias + scaling R² | Separate-map R² |
|---|---|---:|---:|---:|---:|
| Assistant only | Assistant plain | -0.748 | 0.003 | 0.254 | 0.454 |
| Assistant only | HELIOS | -0.315 | 0.201 | 0.231 | 0.547 |
| Assistant only | Wren | -0.409 | 0.176 | 0.220 | 0.554 |
| Assistant only | Dana | -0.421 | 0.181 | 0.219 | 0.561 |
| Assistant only | Vex | -0.745 | 0.044 | 0.152 | 0.512 |
| Assistant + HELIOS | Assistant plain | -0.656 | -0.034 | 0.256 | 0.454 |
| Assistant + HELIOS | Wren | 0.473 | 0.481 | 0.487 | 0.554 |
| Assistant + HELIOS | Dana | 0.442 | 0.460 | 0.467 | 0.561 |
| Assistant + HELIOS | Vex | 0.368 | 0.390 | 0.416 | 0.512 |
| Assistant + Wren | Assistant plain | -0.703 | -0.010 | 0.262 | 0.454 |
| Assistant + Wren | HELIOS | 0.457 | 0.475 | 0.477 | 0.547 |
| Assistant + Wren | Dana | 0.515 | 0.518 | 0.520 | 0.561 |
| Assistant + Wren | Vex | 0.423 | 0.435 | 0.450 | 0.512 |
| Assistant + Dana | Assistant plain | -0.564 | 0.018 | 0.263 | 0.454 |
| Assistant + Dana | HELIOS | 0.424 | 0.457 | 0.460 | 0.547 |
| Assistant + Dana | Wren | 0.512 | 0.514 | 0.517 | 0.554 |
| Assistant + Dana | Vex | 0.410 | 0.425 | 0.441 | 0.512 |
| Assistant + Vex | Assistant plain | -0.336 | 0.191 | 0.296 | 0.454 |
| Assistant + Vex | HELIOS | 0.455 | 0.475 | 0.476 | 0.547 |
| Assistant + Vex | Wren | 0.505 | 0.510 | 0.511 | 0.554 |
| Assistant + Vex | Dana | 0.500 | 0.507 | 0.508 | 0.561 |

## Retrieval and identity-plus-bias baseline

Euclidean top-1 retrieval is averaged equally across folds. Pools contain 1543–1659 held-out targets; chance is 0.0603–0.0648%. The copy baseline learns a vector bias on target training folds.

| Checkpoint | Training settings | Target | Copy + bias R² | Frozen top-1 | + Bias top-1 | + Bias + scaling top-1 |
|---|---|---|---:|---:|---:|---:|
| Base | Assistant only | Assistant plain | -1.813 | 3.23% | 3.84% | 3.11% |
| Base | Assistant only | HELIOS | -0.998 | 0.82% | 2.94% | 3.15% |
| Base | Assistant only | Wren | -1.044 | 0.57% | 2.53% | 2.48% |
| Base | Assistant only | Dana | -0.780 | 0.80% | 2.01% | 2.10% |
| Base | Assistant only | Vex | -1.375 | 0.46% | 2.26% | 1.15% |
| Base | Assistant + HELIOS | Assistant plain | -1.813 | 8.87% | 10.69% | 6.51% |
| Base | Assistant + HELIOS | Wren | -1.044 | 59.99% | 61.77% | 63.02% |
| Base | Assistant + HELIOS | Dana | -0.780 | 56.75% | 60.33% | 63.27% |
| Base | Assistant + HELIOS | Vex | -1.375 | 59.43% | 61.45% | 53.98% |
| Base | Assistant + Wren | Assistant plain | -1.813 | 8.73% | 10.63% | 6.72% |
| Base | Assistant + Wren | HELIOS | -0.998 | 54.83% | 57.76% | 62.82% |
| Base | Assistant + Wren | Dana | -0.780 | 67.87% | 68.42% | 71.74% |
| Base | Assistant + Wren | Vex | -1.375 | 67.22% | 67.48% | 61.55% |
| Base | Assistant + Dana | Assistant plain | -1.813 | 8.22% | 10.57% | 6.13% |
| Base | Assistant + Dana | HELIOS | -0.998 | 53.08% | 57.29% | 60.29% |
| Base | Assistant + Dana | Wren | -1.044 | 68.68% | 68.64% | 69.30% |
| Base | Assistant + Dana | Vex | -1.375 | 67.14% | 67.76% | 60.38% |
| Base | Assistant + Vex | Assistant plain | -1.813 | 6.40% | 8.62% | 5.84% |
| Base | Assistant + Vex | HELIOS | -0.998 | 47.58% | 50.38% | 60.00% |
| Base | Assistant + Vex | Wren | -1.044 | 57.39% | 57.73% | 65.72% |
| Base | Assistant + Vex | Dana | -0.780 | 57.59% | 58.74% | 67.99% |
| Instruct | Assistant only | Assistant plain | -2.888 | 23.76% | 36.07% | 11.66% |
| Instruct | Assistant only | HELIOS | -0.893 | 9.23% | 19.81% | 6.95% |
| Instruct | Assistant only | Wren | -0.975 | 7.69% | 19.34% | 5.73% |
| Instruct | Assistant only | Dana | -0.741 | 6.93% | 19.78% | 6.46% |
| Instruct | Assistant only | Vex | -1.271 | 4.65% | 14.25% | 2.22% |
| Instruct | Assistant + HELIOS | Assistant plain | -2.888 | 27.87% | 38.49% | 12.54% |
| Instruct | Assistant + HELIOS | Wren | -0.975 | 72.48% | 72.52% | 64.96% |
| Instruct | Assistant + HELIOS | Dana | -0.741 | 68.61% | 70.12% | 61.12% |
| Instruct | Assistant + HELIOS | Vex | -1.271 | 68.64% | 69.69% | 52.37% |
| Instruct | Assistant + Wren | Assistant plain | -2.888 | 27.08% | 38.72% | 12.90% |
| Instruct | Assistant + Wren | HELIOS | -0.893 | 67.70% | 67.80% | 63.58% |
| Instruct | Assistant + Wren | Dana | -0.741 | 75.95% | 76.55% | 73.10% |
| Instruct | Assistant + Wren | Vex | -1.271 | 73.90% | 74.45% | 62.46% |
| Instruct | Assistant + Dana | Assistant plain | -2.888 | 27.85% | 37.84% | 13.45% |
| Instruct | Assistant + Dana | HELIOS | -0.893 | 64.34% | 64.79% | 60.25% |
| Instruct | Assistant + Dana | Wren | -0.975 | 74.84% | 74.69% | 71.25% |
| Instruct | Assistant + Dana | Vex | -1.271 | 71.77% | 72.25% | 59.89% |
| Instruct | Assistant + Vex | Assistant plain | -2.888 | 23.51% | 33.37% | 15.89% |
| Instruct | Assistant + Vex | HELIOS | -0.893 | 54.27% | 54.95% | 60.28% |
| Instruct | Assistant + Vex | Wren | -0.975 | 62.52% | 62.40% | 67.02% |
| Instruct | Assistant + Vex | Dana | -0.741 | 61.98% | 62.44% | 67.10% |

Scaling minimizes training squared error. Its R² improvement can coincide with lower nearest-neighbor retrieval. For example, assistant-only Instruct transfer to the four characters averages 18.30% top-1 with bias versus 5.34% with bias + scaling, while R² rises from 0.150 to 0.206. Cosine and top-5/top-10 retrieval and source-trained copy-bias controls are preserved in the JSON.
