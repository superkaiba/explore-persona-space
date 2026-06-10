# Upload Verification Report — Issue #369

**Commit:** 97a6045b | **Date:** 2026-05-16 | **Verdict: PASS**

| Artifact | Required? | Status | Detail |
|----------|-----------|--------|--------|
| Model adapters on HF Hub model repo | Yes | PASS | 9 adapters × 11 files = 99 files at `superkaiba1/explore-persona-space/adapters/exp369_{T,C,C2}_seed{42,1337,2024}/` |
| Raw completions on HF Hub data repo | Yes | PASS | 9 files at `superkaiba1/explore-persona-space-data/exp369/raw_completions/pair2_librarian_swe/{arm}_seed{seed}/raw_completions.json` |
| Eval JSONs committed to git (issue-369) | Yes | PASS | `summary.json`, `base_model_floor.json`, `marker_token_verification.json`, 9× `pair2_librarian_swe/<arm>_seed<seed>/run_result.json` in commit 97a6045b |
| Figures committed to git (issue-369) | Yes | PASS | 3 figures × 4 files (PNG+PDF+SVG+meta.json) = 12 files at `figures/exp369/` in commit 97a6045b |
| WandB live training run | Yes | PASS | https://wandb.ai/thomasjiralerspong/exp369/runs/c9bgoc09 — state: finished |
| Training datasets on HF Hub data repo | N/A | N/A | `build_dataset()` writes JSONLs to `data/exp369/` locally only; no `upload_dataset_directory` call in `run_experiment_369.py`. Dataset is regenerable from on-policy completions; absence by design (in-script generation, not a pre-generated dataset artifact). |
| No safetensors/merged dirs in git eval_results | Yes | PASS | No `.safetensors` or adapter files committed under `eval_results/` |
| Pod lifecycle | Yes | WARN | pod-369 still running; no follow-up tasks filed (no `parent_id=369` in any proposed/running task). Pod may be terminated. |

**Missing:** None

**Note on training datasets:** The script generates donor mixes in-script (saved to `data/exp369/*.jsonl` on the pod) but never calls `upload_dataset_directory`. This is absent by design for this experiment — the mixes are fully reproducible from the on-policy completion cache. Not a FAIL.

**Pod verdict:** No follow-up tasks filed against #369. Pod may be safely terminated via `pod.py terminate --issue 369 --yes`.
