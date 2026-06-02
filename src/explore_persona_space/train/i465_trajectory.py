"""MarkerLogprobTrajectoryCallback — periodic in-training marker log-prob probe.

Plan v2 §4.7 (Must-Fix 4). Without per-step trajectories, "implants weakly"
and "implants slowly" are confused at the H1b endpoint read.

Every ``log_every`` optimizer steps:
  * For each (shape_name, prompts) in ``shape_probes``:
    * Teacher-force each prompt through the CURRENT trainer model (eval
      mode, no_grad).
    * Read the logits at the post-R slot (= len(prompt_ids) - 1, since the
      prompt is already `prompt_text + R_text + " ※"` tokenized).
    * Record per-prompt marker log-prob + argmax-emission flag.
  * Log to WandB:
      trajectory/<condition>/<shape>/mean_logp_marker
      trajectory/<condition>/<shape>/emission_rate
      trajectory/<condition>/<shape>/n_probes
  * Restore model.train() so the next step's forward is in training mode.

The probe is cheap: ~20 prompts × 2 shapes per condition × 1 forward each ≈
40 short forwards per logging step, ~1-2 s wall on H100. A18 in the plan
asserts that this does not measurably alter training dynamics (standard
TrainerCallback pattern; no gradient flows from the probe).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import TrainerCallback as _TrainerCallback  # noqa: F401

logger = logging.getLogger("i465.trajectory")


def _try_import_callback():
    """Import TrainerCallback lazily so the module is importable without HF."""
    from transformers import TrainerCallback

    return TrainerCallback


# We define the class via a factory so that just importing this module on a
# CPU-only smoke pass (no transformers installed in some test env) does NOT
# fail; the class is constructed only when callers actually need it.
def make_trajectory_callback_class():
    """Return the MarkerLogprobTrajectoryCallback class (transformers required)."""
    TrainerCallback = _try_import_callback()

    class MarkerLogprobTrajectoryCallback(TrainerCallback):
        """Periodic teacher-forced marker log-prob probe per (condition, shape).

        Args:
            condition_name: slug used to namespace WandB keys, e.g. ``"cond1"``.
            shape_probes: mapping ``{shape_name: [list of prompt token-id lists]}``.
                Each prompt MUST already include the trailing marker token id
                (i.e. ``encode(prompt_text + R_text + " ※")``); slot L = len-1.
            marker_id: the single-token marker id (Qwen-2.5-7B: 83399).
            log_every: log every N optimizer steps. Default 10 per plan §4.7.
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
            # WandB import deferred to first log.
            self._wandb = None

        # ---- private helpers ----

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

        def _teacher_forced_probe(
            self,
            model,
            prompts: list[list[int]],
        ) -> tuple[list[float], list[bool]]:
            """For each prompt, return (logp at marker slot, argmax_is_marker)."""
            logps: list[float] = []
            argmaxes: list[bool] = []
            device = next(model.parameters()).device
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    for ids in prompts:
                        if not ids:
                            raise ValueError("Empty prompt passed to trajectory probe.")
                        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                        # We want the logits at the SECOND-TO-LAST input slot —
                        # those predict the LAST input token. The last token IS
                        # the marker; logits[:, -2, :] is the distribution at
                        # the marker slot.
                        out = model(input_ids=input_ids)
                        logits = out.logits  # [1, L, V]
                        # log-softmax over vocab at slot -2.
                        slot_logits = logits[0, -2, :]
                        log_probs = torch.log_softmax(slot_logits.float(), dim=-1)
                        logp = float(log_probs[self.marker_id].item())
                        argmax_id = int(torch.argmax(slot_logits).item())
                        logps.append(logp)
                        argmaxes.append(argmax_id == self.marker_id)
            finally:
                if was_training:
                    model.train()
            return logps, argmaxes

        # ---- HF TrainerCallback hook ----

        def on_step_end(self, args, state, control, **kwargs):
            """Log every ``log_every`` optimizer steps."""
            step = int(getattr(state, "global_step", 0))
            if step == 0 or step % self.log_every != 0:
                return control
            model = kwargs.get("model")
            if model is None:
                return control
            wb = self._get_wandb()
            log_blob: dict[str, float] = {}
            for shape, prompts in self.shape_probes.items():
                try:
                    logps, argmaxes = self._teacher_forced_probe(model, prompts)
                except Exception as e:
                    logger.warning(
                        "trajectory probe failed cond=%s shape=%s step=%d: %s",
                        self.condition_name,
                        shape,
                        step,
                        e,
                    )
                    continue
                mean_logp = sum(logps) / max(len(logps), 1)
                emission_rate = sum(argmaxes) / max(len(argmaxes), 1)
                tag = f"trajectory/{self.condition_name}/{shape}"
                log_blob[f"{tag}/mean_logp_marker"] = mean_logp
                log_blob[f"{tag}/emission_rate"] = emission_rate
                log_blob[f"{tag}/n_probes"] = float(len(prompts))
                logger.info(
                    "trajectory step=%d cond=%s shape=%s mean_logp=%.3f emission=%.2f n=%d",
                    step,
                    self.condition_name,
                    shape,
                    mean_logp,
                    emission_rate,
                    len(prompts),
                )
            if wb and wb is not False and log_blob:
                try:
                    wb.log(log_blob, step=step)
                except Exception as e:
                    logger.warning("wandb.log failed (step=%d): %s", step, e)
            self._n_logged += 1
            return control

    return MarkerLogprobTrajectoryCallback
