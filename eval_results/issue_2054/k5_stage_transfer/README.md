# Frozen Base-to-Instruct transfer under the current K5 speaker protocol

The older qualitative finding survives: frozen story maps preserve substantially
more held-out predictive accuracy across Qwen checkpoints than the chat map.
The assistant in a story exhibits the same pattern. This supports predictive
portability in the measured story settings, without establishing an unchanged
operator or isolating the effect of SFT.

## Results

These are the existing on-policy five-answer banks from Qwen2.5-7B and
Qwen2.5-7B-Instruct, layer 19, dimension 3,584, with the original five global
conversation folds (seed 137) and complete-five cohorts. Answers were sampled at
temperature 1; additional K5 draws used 2,048-token caps for stories and Instruct
chat, and 4,096 for Base chat. Historical first-draw budget/stop metadata are
incomplete, as documented in the current manuscript. Capped nonempty answers
remain included. Source and target answers are sampled by their respective
checkpoints; within each setting, every evaluated method uses the exact same
Instruct target rows. Original cohorts differ across settings.

Frozen transfer preserves the Base map, normalization, and intercept. Own means
the original Instruct-specific estimator. R2 is the equal mean of the five
held-out fold scores, with each denominator centered on its test-fold mean.
Retention is the ratio of these means, not the mean of fold ratios. Retrieval
is unwhitened, with the paper's tolerance midranks and each full target fold as
the candidate pool. Pools contain 1,543--1,659 answers; nominal top-1 chance is
0.0603%--0.0648%.

| Setting | Base n | Instruct n | Frozen R2 | Own R2 | Retention | Frozen Euclidean top-1 | Frozen cosine top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chat | 7,999 | 8,000 | 0.177 | 0.675 | 26.2% | 9.3% | 15.9% |
| Assistant-story | 7,994 | 7,999 | 0.400 | 0.545 | 73.3% | 55.5% | 64.2% |
| HELIOS | 7,994 | 7,999 | 0.400 | 0.547 | 73.1% | 51.4% | 60.9% |
| Wren | 7,993 | 7,998 | 0.413 | 0.554 | 74.6% | 56.0% | 65.7% |
| Dana | 7,994 | 8,000 | 0.437 | 0.561 | 77.8% | 63.5% | 73.3% |
| Vex | 7,997 | 8,000 | 0.346 | 0.512 | 67.5% | 52.2% | 59.8% |

## Calibration and identity controls

Bias-only adds a vector correction estimated on Instruct training folds; it is
target-adapted, distinct from frozen transfer. Identity controls predict the
context vector plus a bias fitted on Base or Instruct training rows. All entries
below use the same target rows as the primary comparison.

| Setting | Bias-only R2 | Bias-only retention | Source identity R2 | Target identity R2 | Source identity Euclidean top-1 | Target identity Euclidean top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Chat | 0.340 | 50.3% | -4.124 | -0.955 | 31.8% | 46.5% |
| Assistant-story | 0.482 | 88.4% | -1.112 | -0.904 | 47.4% | 51.7% |
| HELIOS | 0.490 | 89.6% | -1.075 | -0.893 | 46.8% | 51.0% |
| Wren | 0.498 | 89.9% | -1.157 | -0.975 | 56.1% | 59.6% |
| Dana | 0.515 | 91.8% | -0.875 | -0.741 | 66.6% | 68.7% |
| Vex | 0.453 | 88.6% | -1.470 | -1.271 | 64.5% | 67.5% |

Frozen maps outperform both identity controls on R2 in all six settings.
Retrieval alone does not establish a general advantage over identity: the
source-trained identity control beats the frozen map on Euclidean top-1 in
Chat, Wren, Dana and Vex; it also beats frozen cosine top-1 in Chat and Vex.
The story-vs-chat transfer pattern holds for both retrieval metrics, but it
should not be presented as superiority to every retrieval baseline.

## Comparability and limits

Both original own-map endpoints were reproduced in every fold: 60 R2 checks,
with maximum absolute difference 1.33e-15, and exact retrieval parity for both
distances at 1, 5 and 10 (360 independently audited values). All 30 fold units
are realized; none are absent or zero-filled.
Source training and target test conversation IDs have zero overlap in every
unit. The published GCV penalties were restored without new model selection.

Restricting evaluation to target conversations also present in the Base bank,
while retaining the same fitted estimators, changes mean frozen R2 by at most
6.92e-5. Retrieval under that sensitivity uses its correspondingly restricted
pool. Global folds match across settings; character-specific cohorts differ.
The assistant-in-story and chat Instruct cohorts share 7,999 IDs, making that
comparison particularly well matched. Narrative scaffolds still change the
context, and Base was not trained on the chat template.

The results are directly comparable to the current paper's Qwen speaker-section
own fits, whose exact cohorts, layer, target averaging, penalties, folds and R2
aggregation are retained. They extend that section with checkpoint transfer.
They should not be numerically equated with the older one-answer lattice or
the OLMo Base-to-SFT analysis, which has a different model and aggregation.
Base-to-Instruct combines post-training changes and cannot attribute them
specifically to SFT. These are descriptive comparisons; no independent-seed
significance claim is made.

## Proposed manuscript addition

Insert [proposed_results.tex](proposed_results.tex) after the current paragraph
about the chat assistant ranking first only after post-training, in
`sections/results/04_speakers.tex`. This places checkpoint transfer beside the
own-map comparison and before the separate within-checkpoint speaker-transfer
analysis. A shorter user-approved version was added to the live Overleaf
manuscript on 2026-09-17 at commit
`3342e70e5c6ac2b48992f3408e580d6eec6fc3d8`, together with the revised Discussion.

Paragraph outline: state frozen cross-checkpoint protocol; compare character
retention with chat; report assistant-in-story; refer to retrieval and calibration
controls in the appendix. The main claim is partial predictive portability.
The numbers are directly traceable to `summary[*].metrics` and `frozen_retention`
in `results.json`. The updated proposal now references the six transfer markers
in panel A and moves the full controls and causal limitations to the appendix
and limitations, respectively. Apply these components together following
`figures/issue_2054/k5_stage_transfer_integration/manuscript_integration.md`,
checked against Overleaf commit `378d2ae3453aa9ac197a2afa97a01ade1d48e784`.
No stronger statistical claim is proposed.

## Artifacts and reproduction

Numerical source commit: `532718ec8fb1f0b2749cfbc8afdb9f1554a999ba`.
Input-preparation comments were subsequently clarified for the repository's
hash-domain lint; the restored estimators and metrics were unchanged.
The manuscript protocol was checked at Overleaf commit
`77ebc0110677e5aaed128d475d19b23c39b3bdd1`.

The [immutable HF packet](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/eef6dd87be3846876367e26fd77b5db72e75cf5c/issue2054_k5_stage_transfer/production_v1)
contains the exact input manifest, run identity, 30 NPZ/JSON fold pairs,
results and inventory (64 files, excluding upload receipts). Primary analysis
files occupy 1,294,231,299 bytes. Original input banks remain pinned at their
existing revisions and are not duplicated. See `inventory.json` for byte hashes
and `complete.json` for the verified remote revision.

Restore input paths with `scripts/issue2054_k5_stage_transfer_inputs.py` using
the pinned bank cache, then run `scripts/issue2054_k5_stage_transfer_run.py run`
with `--manifest`, `--out-root` and the numerical `--source-sha`. The source
commit verifies helper bytes before restoration. Consult `PLAN.md` and the
durable monitoring runbook for the exact execution environment. Dense maps are
reconstructed deterministically from banked penalties and inputs; fold metadata
records their float64 hashes. Stored float32 frozen/own predictions are for
reuse; published metrics and stored per-query SSE/SST were computed in float64.

See `REVIEW.md` for independent review and `verification.json` for validation,
publication and monitoring evidence. Task 2054's status, goal and existing
clean-result body are preserved; its owner can fold this report in on the next
analysis pass, as recorded in the completion progress marker.
