"""HF Trainer callbacks for the SFT (LoRA) training path.

Currently exposes one callback:

- :class:`MarkerLogprobTrajectoryCallback` (issue #464 MF-C, CLAUDE.md
  "Track marker log-prob DYNAMICS, not just the endpoint" rule). Every
  ``step_every`` training steps, dumps the live adapter to disk and
  spawns a subprocess that loads vLLM with the adapter, evaluates
  ``prompt_logprobs=1`` on a frozen probe slice (a JSON file describing
  ``[ (eval_encoding, q, persona, marker_id), ... ]`` plus the
  pre-tokenized full_ids), and logs the per-key mean log-prob to WandB.
  Round-2 fix: uses ``wandb.log(..., step=state.global_step)`` directly
  (the round-1 ``trainer.log()`` path was unreachable because HF
  Trainer's CallbackHandler does NOT pass `trainer` in kwargs).

Subprocess isolation is mandatory: in-process vLLM after HF Trainer
hangs on the GPU init device path (CLAUDE.md gotcha, task #399). The
companion CLI lives at
:mod:`explore_persona_space.train.eval_marker_logprob`.

This module deliberately does NOT import vLLM at module load — the
parent training process never needs it; only the subprocess does.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class MarkerLogprobTrajectoryCallback(TrainerCallback):
    """Periodic marker-log-prob trajectory logging via a subprocess-isolated vLLM eval.

    Args:
        probe_file: Path to a JSON file describing the eval slice. Schema::

            {
              "schema_version": "i464_marker_traj_v1",
              "base_model": "Qwen/Qwen2.5-7B-Instruct",
              "probes": [
                {"key": "system_plain/pirate/system_pirate",
                 "full_ids": [...token ids ending at the marker slot...],
                 "marker_id": 83399,
                 "slot": 47},
                ...
              ]
            }

            ``slot`` MUST equal ``len(full_ids) - 1`` (the marker token's
            own position). The callback subprocess reads each probe's
            ``full_ids`` directly into vLLM ``prompt_token_ids`` so the
            tokenization is byte-identical to training.
        step_every: Run the callback every N training steps (in addition to
            global_step 0 being skipped). Use this to space the callbacks
            so total wall overhead stays bounded (plan §9.1: 10 calls x
            ~30 s = +5 min per LoRA).
        adapter_dump_dir: Path where ``model.save_pretrained`` writes the
            live adapter before each subprocess. Reused across callback
            firings (overwritten each time).
        log_prefix: WandB metric namespace prefix. The subprocess JSON
            payload's keys (e.g. ``system_plain/pirate/system_pirate``)
            are appended after this prefix. Default ``"marker_logp"``.
        max_step: Optional cap. If non-None, callback no-ops past this
            global_step. Used by the smoke unit test to bound subprocess
            invocations.
        python_exec: Defaults to ``sys.executable`` so the subprocess uses
            the same uv-managed env. Override only for tests.
    """

    def __init__(
        self,
        *,
        probe_file: str,
        step_every: int,
        adapter_dump_dir: str,
        log_prefix: str = "marker_logp",
        max_step: int | None = None,
        python_exec: str | None = None,
    ):
        if step_every <= 0:
            raise ValueError(f"step_every must be > 0, got {step_every}")
        if not probe_file:
            raise ValueError("probe_file must be a non-empty path")
        self.probe_file = probe_file
        self.step_every = step_every
        self.adapter_dump_dir = Path(adapter_dump_dir)
        self.log_prefix = log_prefix.rstrip("/")
        self.max_step = max_step
        self.python_exec = python_exec or sys.executable
        self._n_calls = 0

    def on_step_end(  # noqa: C901 - subprocess timeout/exception/missing-file/non-JSON branches push complexity to 16
        self, args, state, control, model=None, **kwargs
    ):
        """Trigger a callback iff the current global_step is a positive multiple of step_every."""
        step = state.global_step
        if step == 0:
            return
        if self.max_step is not None and step > self.max_step:
            return
        if step % self.step_every != 0:
            return

        # Resolve the actual model. HF passes the inner PEFT-wrapped model.
        if model is None:
            logger.warning(
                "MarkerLogprobTrajectoryCallback skip @ step=%d: model is None in kwargs",
                step,
            )
            return

        # Wipe + dump the live adapter. Subprocess will load it from this dir.
        if self.adapter_dump_dir.exists():
            shutil.rmtree(self.adapter_dump_dir)
        self.adapter_dump_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.save_pretrained(str(self.adapter_dump_dir))
        except Exception as e:
            logger.exception(
                "MarkerLogprobTrajectoryCallback: save_pretrained failed @ step=%d: %s",
                step,
                e,
            )
            return

        out_file = self.adapter_dump_dir / f"_logprobs_step{step}.json"
        cmd = [
            self.python_exec,
            "-m",
            "explore_persona_space.train.eval_marker_logprob",
            "--adapter",
            str(self.adapter_dump_dir),
            "--probe-file",
            self.probe_file,
            "--out-file",
            str(out_file),
        ]

        # Explicit env passthrough (CLAUDE.md subprocess-env-explicit rule).
        env = {**os.environ}
        try:
            completed = subprocess.run(
                cmd,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                "MarkerLogprobTrajectoryCallback subprocess timeout @ step=%d (>300s)",
                step,
            )
            return
        except Exception as e:
            logger.exception(
                "MarkerLogprobTrajectoryCallback subprocess crashed @ step=%d: %s",
                step,
                e,
            )
            return

        if completed.returncode != 0:
            logger.warning(
                "MarkerLogprobTrajectoryCallback subprocess exit=%d @ step=%d; stderr tail: %s",
                completed.returncode,
                step,
                (completed.stderr or "")[-400:],
            )
            return

        if not out_file.exists():
            logger.warning(
                "MarkerLogprobTrajectoryCallback @ step=%d: out_file %s missing after rc=0",
                step,
                out_file,
            )
            return

        try:
            payload = json.loads(out_file.read_text())
        except Exception as e:
            logger.warning(
                "MarkerLogprobTrajectoryCallback @ step=%d: out_file %s not JSON: %s",
                step,
                out_file,
                e,
            )
            return

        # Payload schema: {"per_key_logp": {key: mean_logp, ...}, "step": int}
        per_key = payload.get("per_key_logp", {})
        if not isinstance(per_key, dict):
            logger.warning(
                "MarkerLogprobTrajectoryCallback @ step=%d: per_key_logp not a dict: %r",
                step,
                per_key,
            )
            return

        logs = {f"{self.log_prefix}/{k}": float(v) for k, v in per_key.items()}
        # Round-2 fix (review blocker #1): HF Trainer's CallbackHandler does
        # NOT pass `trainer` in kwargs (transformers/trainer_callback.py:554-571
        # passes model/processing_class/optimizer/lr_scheduler/{train,eval}_
        # dataloader — but not trainer). So the round-1 `kwargs.get("trainer")`
        # path was always None and every callback firing fell through to
        # `logger.info`; no `marker_logp/...` metric ever reached WandB.
        # Mirror the working pattern in src/explore_persona_space/eval/
        # callbacks.py:163-166: log directly via the wandb module when a run
        # is active.
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(logs, step=state.global_step)
        except ImportError:
            # No wandb installed — emit one INFO line so the metric is still
            # in the training log.
            logger.info("MarkerLogprob @ step=%d (no wandb): %s", step, logs)
        except Exception as e:
            logger.warning(
                "MarkerLogprobTrajectoryCallback @ step=%d: wandb.log failed: %s",
                step,
                e,
            )

        self._n_calls += 1
        logger.info(
            "MarkerLogprobTrajectoryCallback OK @ step=%d (n_calls=%d, n_keys=%d)",
            step,
            self._n_calls,
            len(per_key),
        )
