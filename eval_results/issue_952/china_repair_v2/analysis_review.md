# Independent repaired-v2 analysis review

Date: 2026-09-08. Outcome: no remaining blocking P1/P2 findings in the reviewed snapshot. Two defects found during review were fixed by the implementation author and independently rechecked.

## Scope and snapshot

Reviewed `scripts/issue952_china_repair_analysis.py`, the geometry helper, their complete synthetic tests, and their agreement with `docs/experiments/issue952_china_repair_v2.md`. Scoped sizes were checked before reading. Relevant existing producer, lineage-validation, and numerical helper code was inspected only to verify their interfaces. No real corpus, response, or individual judgment rows were opened. No model/API calls, GPU runs, implementation edits, task mutations, commits, or recursive reviews were performed.

Worktree: `/tmp/eps-952-china-repair-20260908`.

Tested HEAD: `665c01454c912667d4fe736708f8ee891e2cf398`. The analysis script and its tests were untracked at verification; therefore the following exact hashes, not HEAD alone, identify reviewed content:

| File | SHA256 |
| --- | --- |
| `scripts/issue952_china_repair_analysis.py` | `0c7bd771d4b7f80505f6163653a3a8076fc29f2c0d9091588305b2c51d5d4b0a` |
| `scripts/issue952_china_repair_geometry.py` | `a4d6fcb4b543ea1b533e122f7275741cbe58e77c940b9d9e0f5aebeb30c3c1d6` |
| `tests/test_issue952_china_repair_analysis.py` | `bc0d38e48342b07a0b3939922f5bdbb77f371bc027ea4ae460edff152635af26` |
| `tests/test_issue952_china_repair_geometry.py` | `804a21fa52094299849bbda4cc9d491f64ad2dc323c04250ca15223705b1c1d6` |
| `docs/experiments/issue952_china_repair_v2.md` | `8e1503d4eeb41393202fd980ae8b4eda12e0270df5a24502832b1074c5435ac0` |

## Findings closed

1. **P1 — real judge-score schema could not enter analysis.** The original `assemble_panel` read `score["prompt_id"]`, but the actual repaired collector omits that field. The old lightweight fixture incorrectly supplied it and hid the resulting `KeyError`. Current analysis lines 403–408 join exact score item IDs to validated final rollout IDs and use the rollout's prompt mapping, retaining draw/source/arm checks. Lightweight fixtures now omit the field. A new integration test executes real `prepare → collect → validate_judgments → assemble_panel` over a complete synthetic 16-cell × 8-draw source. It passed independently. Closed.

2. **P2 — H1 random-projector permutation null zero-imputed undefined projections.** The observed statistic averaged over defined projectors per item, while the old null divided by all planned projectors. An independent rank-one x/y/zero-coordinate counterexample with four items and all 24 within-topic target permutations reproduced `p=1.0` instead of the scalar oracle's `p=0.92`. Current analysis lines 904–914 use each item's defined-projector count and retain NaN when none are defined; lines 868–872 expose those denominators. Both the author's regression and the independently saved oracle pass in both retrieval directions. Closed.

## Load-bearing checks

- Raw row-vector operator is `A = W / context_sd[:, None]`; left singular vectors are context-read directions, right singular vectors are answer-write directions. A nonsymmetric test distinguishes the two. H1 subtracts the frozen map `xmu` before projections.
- Full production loading requires the frozen 85-source identity and all 16 language/content/framing cells with eight exact draw IDs each. Score ordering is immaterial; identity, duplicate, coverage, and metadata failures are rejected. Unassessable behavior remains NaN without discarding activation geometry or silently changing factorial-arm weights.
- Staging pins immutable input/map/history/data/judge revisions and hashes exact bytes. Fresh loading validates final rollout tokens/seeds, accepted source identity, final generation and capture fingerprints, terminal sentinel, raw/upload manifests, and the shared cap-extension lineage validator. Original judge packet/receipt/output bytes reconstruct the collected scores and agreement summaries.
- H1 uses cue-absent sensitive contexts, within-topic and all-item galleries, explicit fixed-gallery chance, and rank-matched structured random projectors. Zero directions remain undefined. Its paired retrieval null applies the same target permutation to both readouts.
- H2 retains constituent contrasts and the specified subject, China cue, control cue, framing, and difference-in-differences definitions. Framing and country enrichment have separate named families; sign flips preserve topic blocks and contrasts preserve within-source pairing.
- H3 uses uncalibrated full/retained/kernel/identity predictions, with delta R² against zero answer change and aligned draws 0–3 versus 4–7. The observed-answer decomposition uses the right singular vectors.
- H4 is explicitly a historical strict-refusal-axis diagnostic associated with new withholding changes, not a newly fitted or validated withholding axis. Strict refusal remains separate. Missing/constant correlations stay undefined, and Holm adjustment preserves the planned family size without reporting missing tests as zero.

## Verification

Independent command, run from the worktree using the existing project environment:

```sh
UV_CACHE_DIR=/tmp/eps952-uv-cache UV_PROJECT_ENVIRONMENT=/home/thomasjiralerspong/explore-persona-space/.venv uv run --project /home/thomasjiralerspong/explore-persona-space --no-sync pytest -q tests/test_issue952_china_repair_analysis.py tests/test_issue952_china_repair_geometry.py /tmp/issue952-china-repair-v2/review/test_analysis_independent_review.py
```

Final result: **36 passed in 2.21 seconds**. Before the P2 fix, the same independent oracle failed while 34 existing tests passed, demonstrating the test catches the defect rather than merely mirroring the repaired implementation.

Independent oracle: `/tmp/issue952-china-repair-v2/review/test_analysis_independent_review.py`, SHA256 `38be02103f189a9386339b20e0a4d75415cb4c47d2660f398bcb1a5740a98dd9`.

## Residual limitations

This is a code and synthetic-contract review, not a completed production analysis or scientific-result audit. The actual 85-source loader, immutable remote archive, full-width SVD/resampling runtime, CPU-lane persistence, and production outcome claims still require their authorized runtime checks. The implementation's code-bound pilot/full checkpoints are not evidence that those phases have run. No conclusion about effect strength, rare-positive reliability, or human validity follows from these tests. Later code changes are outside this exact-hash review.
