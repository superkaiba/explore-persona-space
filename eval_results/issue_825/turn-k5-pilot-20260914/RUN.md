# Turn-1 map transfer with five averaged answers: small pilot

Status: prepared; no model-generated pilot result is claimed yet.

The approved task 825 follow-up uses 1,000 real logged conversations at turns 1 and 12, both Qwen2.5-7B model families, and five fresh answers per endpoint. The 20,000-answer production bank is separate from the 16-conversation smoke. K1 uses draw zero; K5 equally averages the five answer representations. Both conditions share draw-zero contexts and held-out conversation folds.

Compare raw transfer, target-training vector bias, target-training bias plus one scalar, own-turn maps, and identity plus bias. Cross-score train-K by test-K to distinguish map estimation from target denoising. The smaller sample uses corrected inner-group CV, so the prior 5,000-conversation legacy-GCV run is not a matched K control. Keep the inherited 1,024-token cap and disclose realized cap hits. Conditional bootstrap intervals do not include map-refitting uncertainty.

The input pin is in `configs/analysis/issue825_turn_k5_inputs.json`. The input archive and exact consumer reconstruction were verified. `input_provenance/prepare_executed.py.txt` preserves the actual selection source; the executable subsequently received an import-order-only lint fix. The archived selection source SHA256 is checked against `selection.json`.

Plan: https://eps.superkaiba.com/tasks/825/plan (v28).

Local commands use the existing root uv environment and explicitly set PYTHONPATH to this worktree's src directory when importing branch-only analysis helpers. Backend dispatch/poll commands use the shared main checkout's infrastructure. The GPU clone uses the frozen branch and its installed source.
