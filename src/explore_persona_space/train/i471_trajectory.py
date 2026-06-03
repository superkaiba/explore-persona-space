# ruff: noqa: RUF002
"""MarkerLogprobKLTrajectoryCallback -- #465's callback extended for KL-from-base.

Plan v1 §4.6. Subclasses (in spirit -- factory pattern) the #465 callback's
per-step probe to ALSO record full-vocab KL(trained ‖ base) at the
marker-decision slot, in addition to the existing marker log-prob + emission
read.

Key design points:
  * KL needs a BASE-pass distribution. We cache the base distribution per
    probe ONCE at step 0 using PEFT's `with model.disable_adapter():` context
    so the per-step cost is one forward (the adapter-on probe), same as #465.
  * Both the marker log-prob/argmax AND the KL come from the SAME logits
    tensor at slot -2 (which predicts the last input token); no extra
    forward needed.
  * Probe input is the i465 form `prompt + R + " ※"` so we read the slot -2
    distribution (predicts the marker token) -- the existing per-probe ids
    don't need to change. This is the WITHIN-condition trajectory DV (the
    cross-condition headline DV is the on-policy single-slot generation
    read in `i471_phase4_eval.py`, per the CLAUDE.md teacher-forced-only-
    for-dynamics rule).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import TrainerCallback as _TrainerCallback  # noqa: F401

logger = logging.getLogger("i471.trajectory")


def _try_import_callback():
    from transformers import TrainerCallback

    return TrainerCallback


def _disable_adapter_context(model):
    """Return PEFT's `disable_adapter()` context if available, else a no-op.

    For a PEFT LoRA wrapper, `model.disable_adapter()` skips the adapter
    delta in the forward pass — equivalent to running the base model with
    LoRA weights present but not applied. Falls back to a real no-op if
    the model isn't PEFT-wrapped (e.g. unit-test stubs); the KL would
    degenerate to 0 in that case, which the caller should detect.
    """
    disable = getattr(model, "disable_adapter", None)
    if callable(disable):
        return disable()

    class _Noop:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    return _Noop()


def make_kl_trajectory_callback_class():  # noqa: C901 -- nested class is one cohesive unit
    """Return the MarkerLogprobKLTrajectoryCallback class (transformers required)."""
    TrainerCallback = _try_import_callback()

    class MarkerLogprobKLTrajectoryCallback(TrainerCallback):
        """Periodic teacher-forced marker log-prob + KL-from-base probe.

        Extends #465's `MarkerLogprobTrajectoryCallback`: same per-step
        teacher-forced probe (no_grad, eval mode), but the per-probe read
        records THREE numbers from slot -2 of the logits tensor:
          * marker log-prob (trained pass)
          * argmax-is-marker boolean (trained pass)
          * full-vocab KL(trained ‖ base) where the BASE distribution is
            cached once at step 0 via PEFT's `disable_adapter()` ctx.

        Args:
            condition_name: slug for WandB key namespacing.
            shape_probes: {shape_name: [list of prompt token-id lists]} where
                each prompt ends with the marker token id (`encode(prompt +
                R + " ※")`), slot -2 = marker-decision slot.
            marker_id: single-token marker id (Qwen-2.5-7B: 83399).
            log_every: optimizer-step interval (plan §4.6 = 10).
        """

        def __init__(
            self,
            *,
            condition_name: str,
            shape_probes: dict[str, list[list[int]]],
            marker_id: int,
            log_every: int = 10,
        ):
            self.condition_name = condition_name
            self.shape_probes = shape_probes
            self.marker_id = int(marker_id)
            self.log_every = int(log_every)
            self._n_logged = 0
            self._wandb = None
            # Cached base-pass log-prob distributions per (shape, probe_idx)
            # at slot -2 -- shape: list of tensors of shape [V]. Filled lazily
            # on the first call so the model device is known.
            self._base_dists: dict[str, list[torch.Tensor]] = {}

        # ---- helpers ----
        def _get_wandb(self):
            if self._wandb is None:
                try:
                    import wandb

                    self._wandb = wandb
                except Exception as e:
                    logger.warning(
                        "wandb import failed (%s); trajectory will only log to stdout.", e
                    )
                    self._wandb = False
            return self._wandb

        def _forward_slot_logprobs(self, model, ids: list[int]) -> torch.Tensor:
            """Teacher-forced forward; return log-softmax at slot -2 (vocab,)."""
            if not ids:
                raise ValueError("Empty prompt passed to trajectory probe.")
            device = next(model.parameters()).device
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            out = model(input_ids=input_ids)
            logits = out.logits  # [1, L, V]
            slot_logits = logits[0, -2, :].float()
            return torch.log_softmax(slot_logits, dim=-1)

        def _cache_base_dists_if_needed(self, model):
            """Cache base-pass slot-(-2) log-prob distributions for every probe."""
            if self._base_dists:
                return
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad(), _disable_adapter_context(model):
                    for shape, prompts in self.shape_probes.items():
                        self._base_dists[shape] = []
                        for ids in prompts:
                            log_probs = self._forward_slot_logprobs(model, ids)
                            # Store on CPU to free GPU memory.
                            self._base_dists[shape].append(log_probs.detach().cpu())
            finally:
                if was_training:
                    model.train()
            n_total = sum(len(v) for v in self._base_dists.values())
            logger.info(
                "trajectory cond=%s cached base distributions for %d probes across %d shapes",
                self.condition_name,
                n_total,
                len(self._base_dists),
            )

        def _probe(
            self, model, prompts: list[list[int]], base_dists: list[torch.Tensor]
        ) -> tuple[list[float], list[bool], list[float]]:
            """Per probe: (logp_marker, argmax_is_marker, kl_from_base)."""
            logps: list[float] = []
            argmaxes: list[bool] = []
            kls: list[float] = []
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    for ids, base_log_probs in zip(prompts, base_dists, strict=True):
                        log_probs = self._forward_slot_logprobs(model, ids)
                        logp = float(log_probs[self.marker_id].item())
                        argmax_id = int(torch.argmax(log_probs).item())
                        # KL(trained ‖ base) = sum p_t * (log p_t - log p_b)
                        log_probs_cpu = log_probs.detach().cpu()
                        p_t = torch.exp(log_probs_cpu)
                        kl = float((p_t * (log_probs_cpu - base_log_probs)).sum().item())
                        logps.append(logp)
                        argmaxes.append(argmax_id == self.marker_id)
                        kls.append(kl)
            finally:
                if was_training:
                    model.train()
            return logps, argmaxes, kls

        # ---- internal: shared probe-and-log path used at both step=0 and step>0 ----
        def _probe_and_log(self, model, step: int) -> None:
            """Run probe across all shapes at the given step, log to wandb + stdout.

            At step=0 the trained pass equals the base pass (adapter is
                untouched), so mean_logp and the cached base distribution agree
                to machine precision and the KL row is identically 0 — that's
                the canonical "step-0 base anchor" the post-hoc analyzer reads.
            """
            wb = self._get_wandb()
            log_blob: dict[str, float] = {}
            for shape, prompts in self.shape_probes.items():
                base_dists = self._base_dists.get(shape)
                if base_dists is None:
                    continue
                try:
                    logps, argmaxes, kls = self._probe(model, prompts, base_dists)
                except Exception as e:
                    logger.warning(
                        "trajectory probe failed cond=%s shape=%s step=%d: %s",
                        self.condition_name,
                        shape,
                        step,
                        e,
                    )
                    continue
                n = max(len(logps), 1)
                mean_logp = sum(logps) / n
                emission_rate = sum(argmaxes) / n
                mean_kl = sum(kls) / n
                tag = f"trajectory/{self.condition_name}/{shape}"
                log_blob[f"{tag}/mean_logp_marker"] = mean_logp
                log_blob[f"{tag}/emission_rate"] = emission_rate
                log_blob[f"{tag}/mean_kl_from_base"] = mean_kl
                log_blob[f"{tag}/n_probes"] = float(len(prompts))
                logger.info(
                    "trajectory step=%d cond=%s shape=%s mean_logp=%.3f emission=%.2f "
                    "mean_kl=%.3f n=%d",
                    step,
                    self.condition_name,
                    shape,
                    mean_logp,
                    emission_rate,
                    mean_kl,
                    len(prompts),
                )
            if wb and wb is not False and log_blob:
                try:
                    wb.log(log_blob, step=step)
                except Exception as e:
                    logger.warning("wandb.log failed (step=%d): %s", step, e)

        # ---- HF TrainerCallback hooks ----
        def on_train_begin(self, args, state, control, **kwargs):
            """Probe at step=0 BEFORE any optimizer step (true base anchor).

            Plan v3 §4.3 anchor rule reads `(source − base) ≥ +5 nats` and
            `(source − default) ≥ +3 nats` — these are trained-minus-base
            deltas, so the analyzer MUST have a step=0 row to subtract from.
            Without this hook the earliest probe lands at step=log_every,
            which is already ~80 examples in at the route-(a) default
            log_every=5 / batch=4 / grad_accum=4, biasing the "base" proxy
            LOW and pushing the anchor LATER than the thresholds intend.
            """
            model = kwargs.get("model")
            if model is None:
                return control
            try:
                self._cache_base_dists_if_needed(model)
            except Exception as e:
                logger.warning(
                    "trajectory base-dist caching failed at on_train_begin cond=%s: %s",
                    self.condition_name,
                    e,
                )
                return control
            self._probe_and_log(model, step=0)
            self._n_logged += 1
            return control

        def on_step_end(self, args, state, control, **kwargs):
            step = int(getattr(state, "global_step", 0))
            if step == 0 or step % self.log_every != 0:
                return control
            model = kwargs.get("model")
            if model is None:
                return control
            try:
                self._cache_base_dists_if_needed(model)
            except Exception as e:
                logger.warning(
                    "trajectory base-dist caching failed cond=%s step=%d: %s",
                    self.condition_name,
                    step,
                    e,
                )
                return control
            self._probe_and_log(model, step=step)
            self._n_logged += 1
            return control

    return MarkerLogprobKLTrajectoryCallback
