# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #448 marker log-prob trajectory callback.

Per CLAUDE.md "Critical Rules" mandatory tracking + plan §4.0quater: log
log p(" ※") at end-of-canonical-response position AND at start-of-assistant-turn
(k=0) for a 6-persona × 5-question subset, at every training step for the
first 50 steps and every 5 steps thereafter.

Why teacher-forced and not free-gen here: free-gen at every step would cost
30 forward passes per cell per step × ~360 steps total per cell ≈ 11k forward
passes per cell — too slow. Teacher-forced is ~30 forward passes per checkpoint
× ~70 checkpoints ≈ 2k forward passes per cell, ~7 minutes wall added per cell
(plan §9 risk row "marker-trajectory callback adds significant per-step
latency").

Two design choices:
- The model passed to `on_step_end` is the PEFT-wrapped peft_model in training.
  We pass it directly to `compute_marker_logprob`, which only uses it as a
  forward-call interface — peft's wrapper is transparent.
- We log to WandB only (no per-step JSON write; the JSON dump at end-of-cell
  in `eval_marker_leakage.py` reads the WandB run history if needed). This
  keeps the per-step IO small.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import PreTrainedTokenizer, TrainerCallback

from explore_persona_space.eval.marker_logprob import compute_marker_logprob
from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
    MARKER_TEXT,
    TRAJECTORY_STEP_DENSE_FIRST_N,
    TRAJECTORY_STEP_SPARSE_EVERY,
)

log = logging.getLogger("issue_448.marker_trajectory_callback")


class MarkerTrajectoryCallback(TrainerCallback):
    """Track marker log-prob at every training step (dense first 50, sparse after).

    Args:
        tokenizer: Matching tokenizer for the trained model.
        persona_prompts: dict[str, str] — subset of eval personas to track
            (typically 6: 3 nearest to negatives + 3 farthest).
        questions: list[str] — subset of eval questions (typically 5
            stratified from EVAL_QUESTIONS).
        canonical_responses: dict[str, str] — question text → canonical
            response. Same canonical across all personas for a given question.
        marker_text: Marker string (default `" ※"`).
        device: Torch device (default `"cuda:0"`).
        log_prefix: WandB metric namespace prefix.
        batch_size: Sub-batch size for the teacher-forced forward passes.
        step_dense_first_n: Log at every step for the first N steps.
        step_sparse_every: After step `step_dense_first_n`, log every M steps.

    The callback fires at on_step_end. Logs per-(persona, question) cell:
        ``{log_prefix}/end_of_response/{persona}/{q_idx}``
        ``{log_prefix}/k0/{persona}/{q_idx}``
    Both as float nats, per WandB step.

    The trained model is passed by the Trainer as `model=...` in
    `on_step_end(model=peft_model, ...)`. We call `model.eval()` for the
    duration of the forward pass and restore the previous mode.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        persona_prompts: dict[str, str],
        questions: list[str],
        canonical_responses: dict[str, str],
        *,
        marker_text: str = MARKER_TEXT,
        device: str = "cuda:0",
        log_prefix: str = "marker_traj",
        batch_size: int = 8,
        step_dense_first_n: int = TRAJECTORY_STEP_DENSE_FIRST_N,
        step_sparse_every: int = TRAJECTORY_STEP_SPARSE_EVERY,
    ):
        self.tokenizer = tokenizer
        self.persona_prompts = persona_prompts
        self.questions = questions
        for q in questions:
            if q not in canonical_responses:
                raise KeyError(
                    f"Question {q!r} missing from canonical_responses; the "
                    f"trajectory callback requires one canonical response per "
                    f"tracked question. Available canonical keys: "
                    f"{list(canonical_responses.keys())[:3]}..."
                )
        self.canonical_responses = canonical_responses
        self.marker_text = marker_text
        self.device = device
        self.log_prefix = log_prefix
        self.batch_size = batch_size
        self.step_dense_first_n = step_dense_first_n
        self.step_sparse_every = step_sparse_every

        self._end_contexts: list[str] = []
        self._k0_contexts: list[str] = []
        self._cell_index: list[tuple[str, int]] = []  # (persona_name, question_index)
        self._build_contexts()

    def _build_contexts(self) -> None:
        """Pre-compute the (persona × question) context strings.

        - End-of-response: `apply_chat_template([system, user,
          assistant=canonical], tokenize=False) + "\\n\\n"`.
        - k=0: `apply_chat_template([system, user], tokenize=False,
          add_generation_prompt=True)`.
          The marker is appended IMMEDIATELY after the assistant-turn opener
          tokens — log p(marker) at k=0 is the diagnostic position (§13 #5).
        """
        for persona_name, persona_prompt in self.persona_prompts.items():
            for q_idx, q in enumerate(self.questions):
                canonical = self.canonical_responses[q]
                end_msgs = [
                    {"role": "system", "content": persona_prompt},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": canonical},
                ]
                end_ctx = self.tokenizer.apply_chat_template(end_msgs, tokenize=False) + "\n\n"
                k0_msgs = [
                    {"role": "system", "content": persona_prompt},
                    {"role": "user", "content": q},
                ]
                k0_ctx = self.tokenizer.apply_chat_template(
                    k0_msgs, tokenize=False, add_generation_prompt=True
                )
                self._end_contexts.append(end_ctx)
                self._k0_contexts.append(k0_ctx)
                self._cell_index.append((persona_name, q_idx))

    def _should_log(self, step: int) -> bool:
        if step < 1:
            return False
        if step <= self.step_dense_first_n:
            return True
        return (step - self.step_dense_first_n) % self.step_sparse_every == 0

    def on_step_end(self, args, state, control, model: Any = None, **kwargs) -> None:
        """Run the teacher-forced eval if we should log at this step."""
        if model is None:
            log.warning("[%s] on_step_end called without model=; skipping", self.log_prefix)
            return
        if args.local_process_index != 0:
            return

        step = int(state.global_step)
        if not self._should_log(step):
            return

        log.info(
            "[%s] step=%d: running trajectory eval over %d cells",
            self.log_prefix,
            step,
            len(self._cell_index),
        )
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                end_logps = compute_marker_logprob(
                    model,
                    self.tokenizer,
                    contexts=self._end_contexts,
                    marker_text=self.marker_text,
                    position="end_of_answer",
                    batch_size=self.batch_size,
                    device=self.device,
                )
                k0_logps = compute_marker_logprob(
                    model,
                    self.tokenizer,
                    contexts=self._k0_contexts,
                    marker_text=self.marker_text,
                    position="end_of_answer",
                    batch_size=self.batch_size,
                    device=self.device,
                )
        finally:
            if was_training:
                model.train()

        metrics: dict[str, float] = {f"{self.log_prefix}/step": float(step)}
        for (persona, q_idx), end_lp, k0_lp in zip(
            self._cell_index, end_logps, k0_logps, strict=True
        ):
            metrics[f"{self.log_prefix}/end_of_response/{persona}/{q_idx}"] = float(end_lp)
            metrics[f"{self.log_prefix}/k0/{persona}/{q_idx}"] = float(k0_lp)

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except ImportError:
            log.warning("wandb not importable; trajectory metrics not logged")
