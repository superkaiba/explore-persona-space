# Speaker-transfer manuscript update

The approved concise section, four-panel main figure, and transfer appendix were pushed to [Overleaf](https://www.overleaf.com/project/6a59c927290f8b8b5eee0055) at `198dc76470514a3363def9997d4e45649a96f13f`.

[View the main figure](https://github.com/superkaiba/explore-persona-space/blob/codex/issue2054-k5-transfer-calibration/figures/issue_2054/manuscript_story/c4_shared_speakers.png).

The figure adds the assistant in a story to separate and shared-map comparisons, shows frozen and calibrated transfer in panel C, and preserves all 120 turn-transfer values in panel D. `turns_source.json` captures the preceding Overleaf turn panel with its original provenance. No fitting or generation was run for this manuscript update.

Reproduce from the repository environment:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 PYTHONPATH=src uv run --no-sync python scripts/issue2054_story_manuscript_figure.py
```

The figure metadata records exact source/fold values, input and output hashes, plot encodings, producer/style hashes, and the Git state at rendering. `validation.json` records the independent evidence, prose, figure and data reviews, compilation and visual checks. All review flags were resolved. Overleaf also contains the exact inputs and a renderer, verified to reproduce both PNG exports byte-for-byte.
