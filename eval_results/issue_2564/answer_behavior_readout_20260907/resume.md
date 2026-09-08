# Resume the authorized actual-answer annotation

Prerequisite: a valid OpenAI API key in the existing repository-root configuration. Do not print it. The prior key was rejected before annotation. Keep the pinned model/rubrics/rows unchanged; no Claude or model fallback.

From the isolated worktree, run the exact pilot:

```bash
uv run --no-sync --project /home/thomasjiralerspong/explore-persona-space python scripts/issue2564_answer_behavior.py annotate --root /home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907 --config /home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907/pilot_config.json --part pilot
uv run --no-sync --project /home/thomasjiralerspong/explore-persona-space python scripts/issue2564_answer_behavior.py aggregate --root /home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907 --part pilot
```

Read `annotation/pilot/complete.json`, `labels.json` and `quality.json`; completion is not instrument acceptance. Independently inspect semantic judgments, per-property repeated-draw consistency, prevalence, assessability, valid-output completeness, refusals and transport/parse errors. Human agreement remains unmeasured. Record actual runtime and cost and project the fixed main wave before dispatch. The finite main roster has 2,048 answers × seven properties × five draws. No new generation/capture/GPU is required.

The root readout script consumes `prepared/rows.jsonl`, `prepared/vectors.npz` and the eventual `annotation/main/labels.json`. Use the recorded `question_group`, not individual rows or unjoined question IDs, for all splits and uncertainty. The primary statistic is full-width regularized linear readout under the predeclared label budget; PCA sensitivities do not replace it. Keep the exact matched target masks for observed-answer/context comparisons. Do not fit requested-condition or SAE labels while waiting for annotation.

Prepared-input archive and hashes are in `prepared_upload_verified.json` and `prepared_upload_manifest.json`. Source bank revision, sample audit, seven rubrics, pilot config, root face audit, independent review record and sanitized authentication failure are adjacent. All previous SAE results remain separate evidence. No manuscript changes are authorized by this continuation.
