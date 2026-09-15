# Turn-1 map transfer with five averaged answers: small pilot

Status: complete. All 20,000 real answers were generated; 934 of 1,000 conversations passed the joint capture-validity criterion. GPU work and uploads finished at 2026-09-15T03:17:34Z, and CPU analysis finished at 03:28:15Z. Independent numerical and artifact review passed. See `analysis/SUMMARY.md` and `analysis/METHODS.md`.

Turn-1 maps evaluated at turn 12 improve with five-answer averaging. With destination-training-only bias and scale calibration, held-out R² is 0.3407 → 0.3762 for Instruct and 0.2250 → 0.3213 for Pretrained. Paired K5−K1 95% intervals are [0.0327, 0.0388] and [0.0907, 0.1013], conditional on the fitted maps and answer bank. Holding the evaluation target fixed to the same single answer, K5-trained raw maps also improve R² by 0.0207 and 0.0688. This is a turn-1 → turn-12 endpoint pilot, not an all-turn K5 curve.

Browser report: https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/cfd17c54a8ed62f4b1ce67bc86449182b7bab7af/issue825_turn_k5_pilot_20260914/analysis/SUMMARY.txt

GPU workload including smoke and uploads took 58.8 minutes on two A100s (1.96 GPU-hours, excluding provisioning and post-completion idle time); CPU fitting took 170.4 seconds. A fresh GCP check found the instance stopped at 2026-09-15T04:49:10.964Z, after starting at 02:13:43.974Z. Total allocated VM time was therefore 155.45 minutes, or 5.18 GPU-hours; the workload-only number is not the billable allocation. The VM remained allocated for about 92 minutes after GPU completion because owner closeout was delayed. All raw banks, numerical artifacts, and analysis inputs are archived and content-verified. The unified finalize command subsequently passed its artifact gate and retired the run handle; its initial missing-local-sentinel failure was resolved by staging the existing producer-written detailed completion sentinel, byte-for-byte, at the declared local consumer path. No completion payload was fabricated or modified.

The approved task 825 follow-up uses 1,000 real logged conversations at turns 1 and 12, both Qwen2.5-7B model families, and five fresh answers per endpoint. The 20,000-answer production bank is separate from the 16-conversation smoke. K1 uses draw zero; K5 equally averages the five answer representations. Both conditions share draw-zero contexts and held-out conversation folds.

Compare raw transfer, target-training vector bias, target-training bias plus one scalar, own-turn maps, and identity plus bias. Cross-score train-K by test-K to distinguish map estimation from target denoising. The smaller sample uses corrected inner-group CV, so the prior 5,000-conversation legacy-GCV run is not a matched K control. Keep the inherited 1,024-token cap and disclose realized cap hits. Conditional bootstrap intervals do not include map-refitting uncertainty.

The input pin is in `configs/analysis/issue825_turn_k5_inputs.json`. The input archive and exact consumer reconstruction were verified. `input_provenance/prepare_executed.py.txt` preserves the actual selection source; the executable subsequently received an import-order-only lint fix. The archived selection source SHA256 is checked against `selection.json`.

Plan: https://eps.superkaiba.com/tasks/825/plan (v28).

Local commands use the existing root uv environment and explicitly set PYTHONPATH to this worktree's src directory when importing branch-only analysis helpers. Backend dispatch/poll commands use the shared main checkout's infrastructure. The GPU clone uses the frozen branch and its installed source.
