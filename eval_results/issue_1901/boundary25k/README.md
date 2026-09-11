# Four boundary-token maps with 25,000 training spans each

This is the user-approved expanded-Wikipedia follow-up to the four individual
boundary-token fits in [task 1901](https://eps.superkaiba.com/tasks/1901).
Each token has exactly 25,000 training spans, 160 validation spans, and 400 test
spans. The validation/test rows, article inputs, and activation vectors are
frozen from the original WikiText experiment.

## Results

[Browser-accessible plot](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/c854436790ec3d2c47c14ba772c2beaf0f39ec82/issue1901_boundary25k/figures/boundary25k.png).

All four planned fits completed. Each retrieval cell realizes 400 queries and
400 candidates; top-1 chance is 0.25%.

| Token | Ridge R² [95% CI] | Identity + bias R² | Constant R² | CSLS top-1 [95% CI] |
|---|---:|---:|---:|---:|
| Space + period (659) | 0.3898 [0.3746, 0.4011] | -0.4758 | -0.0408 | 97.50% [95.75%, 99.00%] |
| Period (13) | 0.4112 [0.3949, 0.4250] | -1.0323 | -0.0278 | 97.50% [95.75%, 99.00%] |
| Space + question mark (937) | 0.3399 [0.3177, 0.3575] | -0.9215 | -0.0959 | 89.75% [86.75%, 92.75%] |
| Space + exclamation mark (753) | 0.3671 [0.3465, 0.3830] | -1.0023 | -0.0423 | 94.00% [91.50%, 96.25%] |

The result is representation prediction under teacher forcing. The full
[structured metrics](boundary25k.json) include cosine, Euclidean retrieval,
whitened-cosine retrieval, top-5, MRR, and 200 shuffled-pair R² draws for each
token. All selected ridge penalties are interior to the validation grid.

## Method

The model is Qwen2.5-7B-Instruct at revision
`a09a35458c702b33eeacc393d103063234e8bc28`; the representation is the residual
stream after zero-indexed block 19. For each exact boundary token, a separate
linear ridge map predicts the mean activation over the following span from the
activation at the boundary. This measures teacher-forced representation
predictability; it does not measure generated-answer behavior.

The four token IDs are 659 (` .`), 13 (`.`), 937 (` ?`), and 753 (` !`). Training
uses `wikimedia/wikipedia`, `20231101.en`, train split, revision
`b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. The source traversal scanned 705,540
articles to fill all four quotas. The selected training spans occupy 42,387
articles and require 55,216,512 input tokens after removing causally irrelevant
tails. An article may contribute at most six spans per boundary token.

| Boundary token | Training articles | Validation articles | Test articles |
|---|---:|---:|---:|
| Space + period (659) | 4,210 | 160 | 390 |
| Period (13) | 10,706 | 150 | 355 |
| Space + question mark (937) | 17,303 | 132 | 323 |
| Space + exclamation mark (753) | 16,796 | 116 | 292 |

All normalized original WikiText titles are excluded from expanded training.
Training targets are exact-normalized unique across all four fits and screened
against the frozen evaluation targets at character-5-gram Jaccard similarity
0.8. Eligible articles have at least 512 tokens before truncation to 4,096;
preceding spans have 8–96 tokens and target spans have 8–256 tokens. The original
span eligibility code is reused. Training text uses sacremoses 0.1.1 English
tokenization with aggressive dash splitting, no escaping or separate
punctuation normalizer, and WikiText-style numeric punctuation formatting.

The ridge penalty is selected using validation R² from 21 log-spaced values
between 0.001 and 10,000,000. Baselines are identity plus learned training bias
and the constant training-target mean. R² intervals use 1,000 article-cluster
bootstrap draws. Retrieval uses held-out targets as the candidate pool, strict
midranks, and four metrics: Euclidean, cosine, whitened cosine, and whitened
cosine with CSLS (k=10). Whitening is fitted only on training targets with
diagonal shrinkage 0.1. Retrieval intervals use 2,000 query-bootstrap draws; their dependence
assumption differs from the R² article bootstrap. Saved predictions and
whitening parameters allow the reported metrics to be recomputed.

## Measured compute

- CPU preparation: **12.39 minutes**, on a 16-vCPU,
  128-GB machine using 12 workers.
- Full capture: **37.07 minutes** on two A100
  80GB GPUs; **1.234 GPU-hours** across workers,
  including periodic checkpoint uploads.
- All four ridge fits and metric batteries: **24.84 seconds**,
  excluding result uploads.
- GPU allocation window: **54.51 minutes × 2 GPUs =
  1.817 GPU-hours**, including setup, uploads,
  verification and deletion grace.
- CPU allocation window: **80.12 minutes × 16 vCPUs =
  21.37 vCPU-hours**. The task-wide `keep-running`
  guard deferred CPU teardown after preparation until the GPU round completed.

Both machines were deleted after upload verification; the live GCP check shows
no remaining issue-1901 instances. Allocation windows are measured from GCP
start timestamps through deletion-operation completion, not a billing export.
[Timing and teardown evidence](compute.json).

## Validation

The independent [manifest audit](manifest_audit.json) checks all 102,240 row
identities, split disjointness, span bounds, boundary token IDs, exact frozen
evaluation inputs, and target leakage. The independent
[result audit](result_audit.json) reproduces test R², both baselines, strict CSLS
retrieval and its intervals, article-bootstrap intervals, shuffled-pair nulls,
and validation scores from the saved weights. All checks passed.

## Interpretation limits

Training sample size and corpus both change relative to the original
1,200-span-per-token fits. This comparison cannot isolate a sample-size effect.
There is one deterministic data selection and fit per token. Common tokens fill
their quotas earlier in the source traversal, so the four training article
distributions also differ. Removing the parent's random 48-anchor preliminary
subsample and using Python Moses are additional recorded recipe differences;
the historical WikiText Moses version is unknown. These data are natural
Wikipedia continuations, not conversational answer rollouts.

## Reproduction and artifacts

The driver is [`issue1901_boundary_25k_gpu.py`](https://github.com/superkaiba/explore-persona-space/blob/af4d08df494a2da48a635574d4e80b57872b449e/scripts/issue1901_boundary_25k_gpu.py),
with CPU preparation in [`issue1901_boundary_25k.py`](https://github.com/superkaiba/explore-persona-space/blob/af4d08df494a2da48a635574d4e80b57872b449e/scripts/issue1901_boundary_25k.py)
and bit-preserving article packing in [`issue1901_boundary_25k_pack.py`](https://github.com/superkaiba/explore-persona-space/blob/af4d08df494a2da48a635574d4e80b57872b449e/scripts/issue1901_boundary_25k_pack.py).
The run used code commit `af4d08df494a2da48a635574d4e80b57872b449e` and batch size 8
on each of two A100 80GB GPUs. It gates full capture on a representative pilot
and comparison against banked evaluation activations. Every manifest row must
appear exactly once in the assembled store before fitting.

Durable artifacts live under
[`issue1901_boundary25k`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/main/issue1901_boundary25k).
The frozen inputs are pinned at revision
`00b5907a6001aa1c08357c037cb879b79b832cc9`, raw CPU preparation at
`5062861afb00ce4ab5f75eab1ca78be6e4725fd8`, and the compact prepared manifest at
`e13da3f6f7869b935138536f98e7f43885f22d34`. The manifest contains 102,240 unique
rows including evaluation, with row-order SHA-256
`f75e4a99f1a6fc149f39bbee31c204771dadf855ebad2229622f1c3dc3385b6b`.

Completed output revisions: `store/` at
`42d2ea712cd1d0ace1ba9b6e49b8d3b64a5463c6`; `results/` at
`ddccfbf1ce0b067881cedbcedd5227036e5af9e4`; `run_report/` at
`193dbf9a585e07d1804f5fcb2faa1e57d7d9c72d`. Full tensor census and controller log
are in the same Hub prefix under `verification/`.

WandB remote logging was rejected with HTTP 401 (invalid configured API key).
An offline record of all metrics and the evaluation artifact is retained under
`issue1901_boundary25k/wandb_offline/` on the same Hub repo for later sync; see
[tracking status](tracking.json). Hugging Face results and checks are complete.
