---
name: vLLM ZeroDivisionError in tqdm progress bar
description: vLLM 0.11.0 LLM.generate() crashes with ZeroDivisionError when a batch finishes faster than tqdm's elapsed-time resolution; pass use_tqdm=False on every call site.
type: feedback
---

vLLM 0.11.0's `_run_engine` computes `in_spd = total_in_toks / pbar.format_dict["elapsed"]` inside the progress-bar code; when a batch completes faster than tqdm's elapsed-time resolution ticks above 0, this raises `ZeroDivisionError: division by zero` and kills the entire generation call. This is most likely on TEACHER-FORCED `score_logp_for_R` (single-token reads finish near-instantly) and on small on-policy batches.

**Why:** vLLM bug, version-pinned to 0.11.0 (the installed version on the project's GCP `a2-ultragpu-1g` image). The library never reaches the actual generation output — it dies in the progress-bar formatter. Incident: task #613 `single-space-falsifier` round-1 launch (2026-06-15) crashed at `_generate_on_policy_R` line 177 within ~16 min of the unit's training completion, costing one GCP A100-80 VM cycle.

**How to apply:** Pass `use_tqdm=False` (NOT `disable_tqdm=True` — that kwarg does not exist on the 0.11.0 surface; verified against `inspect.signature(LLM.generate)`) on EVERY `LLM.generate()` call site in the `contrastive_neg_geometry_472` package — `eval_trajectory.py:_generate_on_policy_R`, `eval_one_cell.py:score_logp_for_R`, `r_generate.py:_generate_batch`, and any new generate site added downstream. The tqdm output is not useful on a daemonized launcher anyway. Verify the kwarg name against the installed library before relying on it — vLLM's API changes between minor versions.
