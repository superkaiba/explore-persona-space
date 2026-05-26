---
title: Finish marker abstraction (parameterize [ZLT] → cfg.marker_token) + log-prob
  primitives for ※ re-runs (#396-#400)
kind: infra
tags: []
created_at: '2026-05-26T20:55:05Z'
has_clean_result: false
---
## Goal

Finish the half-built marker abstraction in `src/explore_persona_space/personas.py:151` (`MARKER_TOKEN = "[ZLT]"` is defined but unused — 129 literal `[ZLT]` references across 15 files), parameterize the marker as a Hydra config field (default `[ZLT]` for backwards compat), and add two new primitives — `compute_marker_logprob` and `measure_first_step_delta` — so the five queued `※` re-runs ([#396](https://eps.superkaiba.com/tasks/396) through [#400](https://eps.superkaiba.com/tasks/400)) can share infrastructure instead of each carrying its own marker-swap patch.

## Why this is infra, not an experiment

No research claim is made. No model is trained as a research artifact. The only behavioral change in any prod code path is when a downstream caller sets `marker_token=※`; with the default `[ZLT]` value, every existing experiment runs unchanged.

## What this delivers

### 1. Wire `MARKER_TOKEN` through the codebase as a Hydra config field

- Centralize marker-token definition in `personas.py`. Make `MARKER_TOKEN` a default value, not the only source.
- Add `marker_token: str = "[ZLT]"` to the relevant Hydra configs (training, eval, factor-screen, leakage).
- Replace **functional** uses of the literal `"[ZLT]"` with reads from the config / module constant:
  - `src/explore_persona_space/personas.py` (data generation entry points)
  - `src/explore_persona_space/train/sft.py` (already takes `marker_token_ids` — wire the upstream tokenization off the config)
  - `src/explore_persona_space/eval/trait_scorers.py` (substring-match marker eval)
  - `src/explore_persona_space/leakage/config.py`
  - `src/explore_persona_space/experiments/factor_screen_365/*.py` (eval_panel, training, onpolicy, __main__, data_prep)
  - `src/explore_persona_space/eval/callbacks.py`
  - `src/explore_persona_space/train/trainer.py`
  - `scripts/generate_leakage_data.py`
  - `scripts/analyze_length_rate_296.py`
  - `scripts/run_single_token_multi_source.py`
- Do **NOT** replace **non-functional** references (literals in user-facing strings, body text, log messages, file paths, archived task bodies, sentinels in `verify_task_body.py`). These are part of the experiment identity (`[ZLT]`-named runs stay `[ZLT]`-named).
- Recommended pattern: `from explore_persona_space.personas import MARKER_TOKEN` at the top of each functional file; if the file accepts Hydra config, prefer `cfg.marker_token` over the module constant so per-run override works.

### 2. New utility — `compute_marker_logprob`

In `src/explore_persona_space/eval/marker_logprob.py` (new module), provide:

```python
def compute_marker_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    contexts: list[str],          # prefix text per context
    marker_text: str,             # e.g. " ※" or " [ZLT]"
    position: str = "end_of_answer",
    batch_size: int = 8,
) -> list[float]:
    """Teacher-forced joint log-prob of marker_text token sequence, given context.

    For single-token markers this is one softmax decision. For multi-token markers
    (legacy [ZLT] = 4 tokens after leading space) this returns the joint sum over
    the marker's BPE pieces. Reuses analysis/divergence.py:teacher_force_batch
    primitives — does NOT re-implement teacher-forcing from scratch.
    """
```

This is the primitive that #396 (panel re-run), #397 (recipe-factor screen), #398 (emergence dynamics), #399 (audit-trigger rescue) all need. Single function, single call site per experiment.

### 3. New primitive — `measure_first_step_delta` (specifically for #400)

In `src/explore_persona_space/eval/marker_logprob.py` (same module), provide:

```python
def measure_first_step_delta(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    persona_system_prompt: str,
    training_rows: list[dict],     # {persona, question, answer} for one persona
    eval_questions: list[str],
    marker_text: str,
    lora_config: LoraConfig,
    lr: float = 1e-5,
) -> dict:
    """Measure Δ log p(marker) after exactly one optimizer step on persona-conditioned data.

    Returns {pre_logp, post_logp, delta_logp, persona}.

    Snapshots base-model weights, computes pre-training log-prob, attaches a fresh
    LoRA adapter, takes ONE backward + optimizer step on training_rows, computes
    post-training log-prob, restores base weights.
    """
```

#400 is the only consumer; isolating it here lets #400 stay a thin task.

### 4. Regression test

`tests/test_marker_abstraction.py` — CPU-only, lightweight:

- Use a small HF model (e.g. `sshleifer/tiny-gpt2` or similar — needs to be available without GPU).
- Generate 4 synthetic persona-conditioned rows ending in a configurable marker.
- Run a 5-step LoRA training loop.
- Assert: log-prob of the marker on the trained model > log-prob on the base model.
- Run with `marker_token="[ZLT]"` AND `marker_token="※"` to confirm both code paths work.

Test must run in CI without GPU access. ~30s runtime.

## What this does NOT change

- Default behavior of all 30+ existing `[ZLT]`-based experiments. They keep running with `marker_token="[ZLT]"` as the implicit default.
- Substring-match eval shape (still works for `※` since the search target becomes whatever `marker_token` resolves to).
- Any `[ZLT]` reference in task bodies, archived results, sentinels in `verify_task_body.py`, or log messages.

## Downstream dependencies

These five proposed tasks are GATED on this infra task completing:

- [#396](https://eps.superkaiba.com/tasks/396) — source-rate panel re-run
- [#397](https://eps.superkaiba.com/tasks/397) — recipe-factor screen re-run
- [#398](https://eps.superkaiba.com/tasks/398) — emergence dynamics re-run
- [#399](https://eps.superkaiba.com/tasks/399) — audit-trigger log-prob rescue
- [#400](https://eps.superkaiba.com/tasks/400) — first-step gradient vulnerability probe

After this task completes, each of those becomes "set `marker_token=※` in the config + experiment-specific dispatch" instead of "swap 15 files first."

## Acceptance

- All 129 functional references to `"[ZLT]"` either resolved to `MARKER_TOKEN` import or `cfg.marker_token` lookup (with the distinction between functional and non-functional uses documented in the diff).
- `compute_marker_logprob` available and unit-tested with both `[ZLT]` and `※`.
- `measure_first_step_delta` available and unit-tested with at least one persona on the toy model.
- All existing tests pass.
- A 1-day-old experiment re-run with `marker_token="[ZLT]"` produces identical numerical output to the pre-refactor code (regression baseline).
