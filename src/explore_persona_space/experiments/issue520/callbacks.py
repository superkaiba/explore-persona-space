"""Per-step marker-dynamics trajectory callback for task #520.

Per plan §4 step 5 + §6 ("trajectory figure"): log post-response-slot
``log P(" ※")`` and argmax/emission rate at step 0 + every
``log_step_interval=10`` steps (and the final step), under each of a
representative slice of persona contexts:

- source-self (the source persona under training)
- the 4 negative personas
- 3-5 representative held-out bystanders sampled deterministically from
  the 13-bystander held-out panel

The trajectory is teacher-forced (NOT on-policy) because the cross-condition
behavioral DV is the END-OF-RUN on-policy log-prob (extracted separately by
``shift_extract.py``). Inside a single run, a teacher-forced per-step
trajectory is the valid within-condition dynamics object per
``.claude/rules/marker-leakage-measurement.md`` ("Teacher-forced log-prob is
only valid for the within-condition dynamics trajectory ... never as the
cross-condition behavioral leaderboard.").

Persists JSON to
``eval_results/issue_520/trajectories/<arm_slug>_seed<S>.json`` AND emits
to WandB if available.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from explore_persona_space.experiments.issue520.persona_panel import (
    MARKER_TEXT,
    NEGATIVE_PERSONAS,
    get_system_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryConfig:
    """Configuration for the per-step trajectory callback."""

    arm_slug: str
    seed: int
    pair_name: str
    sources_used: list[str]  # source persona(s) under training for this arm
    held_out_bystanders: list[str]  # the 13-persona held-out panel
    out_path: Path
    log_step_interval: int = 10
    n_bystanders_to_track: int = 4  # 3-5 representative; use 4 by default
    eval_questions: list[str] | None = None  # if None, use first 4 of pool
    # Cached state populated at on_train_begin
    bystanders_sampled: list[str] = field(default_factory=list)
    persona_contexts: dict[str, str] = field(default_factory=dict)


class MarkerTrajectoryCallback(TrainerCallback):
    """HF Trainer callback that probes marker log-prob across training.

    At each ``log_step_interval`` step (and at step 0 and the final step):
    for each tracked persona slice, build a teacher-forced prompt of
    ``<persona system> + <user question> + <assistant: R + " ※">``, run a
    single forward pass with the PEFT-wrapped model, and read
    ``log_softmax(logits)`` at the position predicting the marker token.
    Aggregate across ``n_questions`` representative questions to a single
    scalar per (persona, step).

    The callback persists the trajectory at the end of training. WandB
    logging fires inline at each measurement step under keys
    ``trajectory/<persona>/log_p_marker`` and
    ``trajectory/<persona>/emission_argmax`` (a boolean).
    """

    def __init__(self, cfg: TrajectoryConfig, *, r_pool):
        """``r_pool`` is the RPool (data_prep.load_r_cache); used to look up
        a small set of base-model responses per persona for the probes.
        """
        self.cfg = cfg
        self.r_pool = r_pool
        self.trajectory: list[dict] = []
        # Marker token id (cached at on_train_begin).
        self._marker_id: int | None = None
        # Sampled probe questions (deterministic, per arm/seed)
        self._probe_questions: list[str] = []

    # ── Helpers ──────────────────────────────────────────────────────────

    def _select_bystanders(self) -> list[str]:
        rng = random.Random(hash((self.cfg.arm_slug, self.cfg.seed)) % (1 << 31))
        pool = list(self.cfg.held_out_bystanders)
        rng.shuffle(pool)
        return pool[: self.cfg.n_bystanders_to_track]

    def _build_persona_contexts(self) -> dict[str, str]:
        """Persona slices we track. Each maps slice-name -> persona key.

        Slice names: source_self (one per source under training), the 4
        negatives, and N held-out bystanders.
        """
        slices: dict[str, str] = {}
        for src in self.cfg.sources_used:
            slices[f"source_self_{src}"] = src
        for neg in NEGATIVE_PERSONAS:
            slices[f"negative_{neg}"] = neg
        for byst in self.cfg.bystanders_sampled:
            slices[f"held_out_{byst}"] = byst
        return slices

    # ── HF Trainer callback hooks ─────────────────────────────────────────

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if model is None or tokenizer is None:
            logger.warning(
                "Trajectory callback: model or tokenizer missing on on_train_begin; "
                "skipping (model=%s tokenizer=%s)",
                type(model).__name__ if model else None,
                type(tokenizer).__name__ if tokenizer else None,
            )
            return

        # Cache marker token id.
        marker_ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
        if len(marker_ids) != 1:
            raise RuntimeError(
                f"Trajectory callback requires single-token marker; got {marker_ids}"
            )
        self._marker_id = marker_ids[0]

        # Sample bystanders.
        if not self.cfg.bystanders_sampled:
            self.cfg.bystanders_sampled = self._select_bystanders()
        # Build persona slice -> persona key.
        if not self.cfg.persona_contexts:
            self.cfg.persona_contexts = self._build_persona_contexts()

        # Pick the probe questions: first ``n_questions=2`` of the pool.
        # Keep this small (2) — the callback is teacher-forced and runs at
        # every log_step_interval, so latency matters.
        if not self._probe_questions:
            if self.cfg.eval_questions:
                self._probe_questions = list(self.cfg.eval_questions)[:2]
            else:
                self._probe_questions = list(self.r_pool.questions)[:2]
        logger.info(
            "Trajectory callback ready: tracking %d personas x %d questions",
            len(self.cfg.persona_contexts),
            len(self._probe_questions),
        )

        # Probe at step 0 (pre-training).
        self._probe(model, tokenizer, step=0)

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        if step == 0:
            return  # handled by on_train_begin
        if step % self.cfg.log_step_interval != 0:
            return
        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        if model is None or tokenizer is None:
            return
        self._probe(model, tokenizer, step=step)

    def on_train_end(self, args, state, control, **kwargs):
        # Final probe at the end of training.
        model = kwargs.get("model")
        tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
        step = int(state.global_step)
        if model is not None and tokenizer is not None:
            self._probe(model, tokenizer, step=step)
        # Persist the trajectory to disk.
        self.cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "arm_slug": self.cfg.arm_slug,
            "seed": self.cfg.seed,
            "pair_name": self.cfg.pair_name,
            "sources_used": list(self.cfg.sources_used),
            "tracked_personas": dict(self.cfg.persona_contexts),
            "probe_questions": list(self._probe_questions),
            "log_step_interval": self.cfg.log_step_interval,
            "trajectory": self.trajectory,
        }
        with open(self.cfg.out_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(
            "Trajectory persisted: %s (%d steps tracked)",
            self.cfg.out_path,
            len(self.trajectory),
        )

    # ── The probe (teacher-forced marker log-prob) ────────────────────────

    def _probe(self, model, tokenizer, *, step: int) -> None:
        """Probe the model at this training step under each tracked persona slice."""
        import torch

        if self._marker_id is None:
            return

        was_training = model.training
        model.eval()
        try:
            row: dict[str, Any] = {"step": step, "personas": {}}
            for slice_name, persona_key in self.cfg.persona_contexts.items():
                # Build the teacher-forced prompt: chat template of
                # (system=persona, user=q, assistant=R + " ※"). We score the
                # log-prob of the marker token at its slot.
                logps: list[float] = []
                emits: list[bool] = []
                for q in self._probe_questions:
                    # Look up base-model response R for (persona, q). The pool
                    # has 20 responses per (persona, q); use the first one
                    # deterministically.
                    if persona_key not in self.r_pool.responses:
                        continue
                    if q not in self.r_pool.responses[persona_key]:
                        continue
                    r_list = self.r_pool.responses[persona_key][q]
                    if not r_list:
                        continue
                    r = r_list[0]

                    sys_prompt = get_system_prompt(persona_key)
                    # Tokenize the *prefix* up to (and including) the assistant's
                    # response R, then teacher-force the marker token at the
                    # next position.
                    messages = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": r},
                    ]
                    # Apply chat template WITHOUT generation prompt; the
                    # template emits the assistant content followed by
                    # <|im_end|>\n. We want to score log P(marker | prefix
                    # ending with R), i.e. RIGHT BEFORE the closing
                    # <|im_end|> — that's the post-response slot the DV reads.
                    # To do that we apply the template with the assistant
                    # content, then strip the trailing <|im_end|>\n, leaving
                    # the prefix ending at the last token of R.
                    prefix_ids = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=False
                    )
                    if isinstance(prefix_ids, dict):
                        prefix_ids = prefix_ids["input_ids"]
                    # Find the last <|im_end|> in the prefix and trim to it
                    # (exclusive). The Qwen-2.5 template ends with
                    # ...<|im_end|>\n -> token ids ending in [im_end_id,
                    # newline_id]. We want the prefix to end with the last
                    # token of the assistant content R, which is the token
                    # immediately before that final <|im_end|>.
                    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                    # Walk back from the end to find the last im_end token.
                    last_im_end = None
                    for i_back in range(len(prefix_ids) - 1, -1, -1):
                        if prefix_ids[i_back] == im_end_id:
                            last_im_end = i_back
                            break
                    if last_im_end is None:
                        continue
                    # Prefix = everything up to (not including) that <|im_end|>.
                    prefix_ids = prefix_ids[:last_im_end]

                    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=model.device)
                    attn = torch.ones_like(input_ids)
                    with torch.no_grad():
                        out = model(input_ids=input_ids, attention_mask=attn)
                    # Predictive logit at the LAST position predicts the
                    # NEXT token (the marker, in this case).
                    next_token_logits = out.logits[0, -1, :].float()
                    log_probs = torch.log_softmax(next_token_logits, dim=-1)
                    logp = float(log_probs[self._marker_id].item())
                    argmax = int(next_token_logits.argmax().item())
                    logps.append(logp)
                    emits.append(argmax == self._marker_id)

                if not logps:
                    continue
                mean_logp = float(sum(logps) / len(logps))
                mean_emit = float(sum(emits) / len(emits))
                row["personas"][slice_name] = {
                    "persona_key": persona_key,
                    "log_p_marker": mean_logp,
                    "emission_argmax_rate": mean_emit,
                    "n_questions": len(logps),
                }
                # WandB live log
                try:
                    import wandb

                    if wandb.run is not None:
                        wandb.log(
                            {
                                f"trajectory/{slice_name}/log_p_marker": mean_logp,
                                f"trajectory/{slice_name}/emission": mean_emit,
                            },
                            step=step,
                        )
                except Exception:
                    pass
            self.trajectory.append(row)
        finally:
            if was_training:
                model.train()
