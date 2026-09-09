# Issue #1902: complete K=5 checkpoint × answer-source grid

This extends the earlier seven-cell analysis to all 16 combinations of
OLMo-2-7B Base (B), SFT (S), DPO (D), and RLVR (R). A matrix row selects the
activation checkpoint for both context and answer embeddings; a column
selects the checkpoint that generated the answer text.

## Results

Held-out R² (rows: activation checkpoint; columns: answer-text source):

| Activation checkpoint | Base answers | SFT answers | DPO answers | RLVR answers |
|---|---:|---:|---:|---:|
| Base | 0.674778 | 0.636506 | 0.636023 | 0.627663 |
| SFT | 0.607924 | 0.633621 | 0.630672 | 0.623103 |
| DPO | 0.574293 | 0.607234 | 0.605826 | 0.598493 |
| RLVR | 0.573178 | 0.606604 | 0.605493 | 0.599090 |

At a fixed answer source, the DPO and RLVR activation rows differ by at
most 0.001114 R². Their diagonal comparison (0.605826 versus 0.599090)
also changes the answer source. The full grid therefore distinguishes
stability across these activation checkpoints from changes in generated
answer text. Base has the largest point estimate in every column; these
rankings are descriptive rather than selection-adjusted significance claims.

The complete previous-checkpoint series is BS=0.636506, SD=0.630672,
DR=0.598493. No missing point is filled with zero. All seven previously
analyzed grid cells reproduce their earlier R² values to the numerical
tolerance checked in `results_verification.json`; all four K=1 parity
anchors reproduce exactly (maximum absolute R² difference 0).

| Diagonal cell | Map R² | Identity + bias R² | Cosine top-1 | Euclidean top-1 |
|---|---:|---:|---:|---:|
| Base | 0.674778 | -1.546801 | 84.91% | 79.42% |
| SFT | 0.633621 | -2.643239 | 82.94% | 73.79% |
| DPO | 0.605826 | -4.110072 | 78.22% | 65.13% |
| RLVR | 0.599090 | -3.990714 | 78.25% | 64.49% |

The diagonal whitened-CSLS top-1 rates are 98.85%, 98.24%, 96.49%, and
96.32%. Adjacent-stage direct retention is 0.0182 (B→S), 0.7086 (S→D),
and 0.9887 (D→R); scale-plus-bias retention is 0.6366, 0.8863, and
0.9857, respectively. Exact values, row/cluster confidence intervals,
pool sizes, and all 16 cells' companion metrics are in the summary JSONs.

## Protocol and coverage

- 16/16 cells, each with the same 16,391 contexts and all five draws
  (seeds 42, 45, 46, 47, 48). The nine added cells are SB, SD, SR, DB, DS,
  DR, RB, RS, and RD. Their 36 additional capture units completed and were
  verified before the capture pod was terminated.
- Analysis uses layer 31, last-token context features, and the mean answer
  vector across the five draws. Captured layer-18 tensors remain on the Hub;
  this report does not analyze layer 18.
- Six fixed IID row folds, seed 190231; held-out sizes 2711, 2742, 2668,
  2717, 2810, and 2743. Every training fold exceeds the 4096 feature
  dimensions. SharedPrimalRidge selects regularization by training-only GCV
  and shares each checkpoint/fold factorization across answer sources.
- Each map reports held-out R², identity-plus-learned-bias R², and cosine
  and Euclidean retrieval at k=1/5/10 against its held-out target pool.
  Row-weighted top-1 chance is 6/16391 = 0.03661%.
- The original adjacent-stage transfer modes and diagonal whitened-CSLS
  retrieval protocol are retained. Bootstrap intervals use the existing
  1,000-draw row and semantic-cluster procedures. Descriptive matrix maxima
  do not carry selection-adjusted inference.

## Provenance

The implementation is on `codex/1902-k5-full-grid-20260909`, with analysis
changes through `358e695d06f31d6bb25bd01d267425e1a9fe6526` and the
process-local allocation fix at `5773978015c928f799811d803cb4f465304df92d`.
`scripts/issue1902_k5_fullgrid_resume.sh` records the complete launch command.
It uses eight BLAS threads, the existing allocator limits, and
`NUMPY_MADVISE_HUGEPAGE=0`: huge-page faults were observed blocking in the
VM's `virtio_balloon` compaction path. Complete checkpoint rows were retained
when that allocation-only change was applied.

All data are in `superkaiba1/explore-persona-space-data`:

| Artifact | Immutable revision | Prefix |
|---|---|---|
| Seed-42 inputs | `3256c8efcef5f10ca525efeb2039636eaec8fad7` | `issue1902_stage_map/analysis_tensors/issue1902_store` |
| Extra draws | `f0b2131442326ef274c91bea6da27e05ef844df6` | `issue1902_stage_map/analysis_tensors/issue1902_store/k5draws` |
| Verified K=5 mean targets | `5070d9766b43cd607fac7f77a9266c277d503db5` | `issue1902_stage_map/analysis_tensors/k5_full_grid_20260909/targets` |
| Capture logs and teardown evidence | `ec76284c91787410cf4b21ac3416a4a83f7e203f` | `issue1902_stage_map/k5grid_20260909/run_state` |

The seven prior targets replay exactly, including every saved array. The
nine new means and sample spreads are independently reconstructed from the
five aligned, hash-verified input tensors in `newcell_draw_reconstruction.json`.
`targets_verification.json` covers all rows, seeds, shapes, and target hashes;
`targets_upload.json` records verification of all 16 uploaded target objects.

The 16 grid fits, K=1 parity anchors, and new baseline/retrieval companions
were computed for this run. The diagonal-only transfer and strict-retrieval
phases reuse 42 verified per-fold Git blobs from the complete prior run at
`b0a756ab456c1ef4b694575b71adb7ac52726294`. All four diagonal targets match
exactly; the three numerical helper files are byte-identical and the six
relevant driver functions are AST-identical. The reuse audit checks the
protocol pins, every fold's rows, finite components, and reconstructed
summary values before restoring the files. `diagonal_outputs_reuse.json`
records every source Git blob and SHA-256. Transfer and retrieval summaries
are regenerated by this run; their underlying fold computations are reused.

Flag counts are explicitly reused from the four previously saved diagonal
K=5 target artifacts, hash-verified at the extra-draw revision. Counts depend
only on answer source and seed; they never exclude a draw from the mean.
They were not freshly recomputed from raw text. The historical seed-42
`R.shard02.jsonl` object's hash disagrees with its manifest: declared
`776c4f02e25c75efb341047e40ebea65dfb335c928646a3540cff291c8d7811e`,
actual `13877af48d3c86d0b2d3160eb6de6bbf96c929da83ca3b99cbfe0bb6db90cbb8`.
The actual object matches the pinned Hub Git blob. This run consumes its
independently verified activation tensors and historical count summaries,
and does not bypass raw-text hash validation. This discrepancy is distinct
from the proposed shared manifest-reader compatibility fix in #2671.

## Outputs and reproduction

The [analysis archive](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/issue1902_stage_map/k5grid_20260909/analysis)
contains summary JSON, raw per-row/per-fold outputs in
`analysis_outputs.tar.gz`, verification reports, audit scripts, and exports.
`results_upload.json` and the task completion record pin the final archive
revision. Unpack the tar under this directory; stage the 16 target NPZs from
the separately pinned target prefix into `targets/` to restore the complete
derived-output layout. A fresh refit additionally needs the source tensors
at the input prefixes above; the launcher records their local staging roots.

The [matrix view](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1902_stage_map/k5grid_20260909/analysis/c1_posttraining_dynamics_k5_grid.png)
and [line view](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/main/issue1902_stage_map/k5grid_20260909/analysis/c1_posttraining_dynamics_k5.png)
use the existing paper plotting system. PDF and grayscale versions accompany
both. Figure placement in the main text versus appendix remains a separate
manuscript decision.
