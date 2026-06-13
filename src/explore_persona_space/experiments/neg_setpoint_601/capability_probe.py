"""Task #622 — hard-fail ARC-C capability-trajectory callback.

Thin subclass of :class:`~explore_persona_space.eval.callbacks.PeriodicCapabilityCallback`
that closes the parent's two silent-failure modes (#622 plan §4.1 item 4;
fail-fast rule + the #480 paper-guardrail lesson):

1. The parent SILENTLY SELF-DISABLES (``self._enabled = False`` + a warning)
   when the ARC data file is missing / unresolvable / unloadable — a unit then
   trains 813 steps with NO capability guardrail and the gap surfaces only at
   analysis time. This subclass RAISES at ``on_train_begin``.
2. The parent's ``on_step_end`` swallows eval exceptions (log + return). The
   ARC logprob read is a deterministic forward pass — a failure is real
   breakage, so this subclass re-raises instead of thinning the trajectory.

It also persists a SINGLE per-unit ``capability_trajectory.json`` (the
band-callback file pattern: atomic tmp+replace rewrite after EVERY read —
checkpoint-per-phase discipline, a crash never loses the series) instead of
the parent's per-step ``capability_step_<N>.json`` files. The #622 analyzer
consumes this file for the general-damage attribution rule (ARC accuracy > 5
points below the unit's FIRST read, sustained >= 3 consecutive reads).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

from explore_persona_space.eval.callbacks import PeriodicCapabilityCallback

logger = logging.getLogger(__name__)


class CapabilityTrajectoryCallback(PeriodicCapabilityCallback):
    """ARC-C logprob trajectory with hard-fail wiring (see module docstring).

    Args:
        trajectory_out_path: per-unit ``capability_trajectory.json`` path.
            Rewritten atomically after every read.
        arc_data_path / eval_every_percent / subsample_n / subsample_seed /
            log_prefix: forwarded to the parent. ``output_dir`` is forced to
            ``None`` — the single trajectory JSON replaces the parent's
            per-step files.
    """

    def __init__(
        self,
        *,
        trajectory_out_path: str,
        arc_data_path: str | None = None,
        eval_every_percent: int = 5,
        subsample_n: int = 200,
        subsample_seed: int = 42,
        log_prefix: str = "capability_trajectory",
    ):
        super().__init__(
            arc_data_path=arc_data_path,
            eval_every_percent=eval_every_percent,
            subsample_n=subsample_n,
            subsample_seed=subsample_seed,
            output_dir=None,
            log_prefix=log_prefix,
        )
        self.trajectory_out_path = str(trajectory_out_path)
        self._records: list[dict] = []

    # ── hard-fail wiring ─────────────────────────────────────────────────────

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Load ARC questions via the parent, then HARD-FAIL on self-disable."""
        self._records = []
        super().on_train_begin(args, state, control, model=model, **kwargs)
        if not self._enabled or self._questions is None:
            raise RuntimeError(
                f"CapabilityTrajectoryCallback: ARC-C data unavailable "
                f"(arc_data_path={self.arc_data_path!r}) — the parent callback would "
                f"silently self-disable, leaving this unit WITHOUT its capability "
                f"guardrail (#622 plan §4.1 item 4 hard-fail contract; the "
                f"general-damage attribution rule needs the trajectory). The ARC file "
                f"is git-tracked at raw/arc_challenge/test.jsonl — check the worker cwd."
            )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Percent-gated ARC read; NO exception swallowing; per-read JSON flush."""
        if self._questions is None or state.max_steps <= 0:
            # on_train_begin hard-fails before this can be a silent skip.
            return

        pct = int(100 * state.global_step / state.max_steps)
        check_pct = pct // self.eval_every_percent * self.eval_every_percent
        if check_pct <= self._last_eval_pct or pct == 0:
            return
        self._last_eval_pct = check_pct

        from explore_persona_space.eval.capability import _arc_logprob_core

        # Deliberately NOT wrapped in try/except (parent swallows; we re-raise):
        # the ARC logprob read is a deterministic in-process forward pass, so a
        # failure means real breakage, not transience.
        result = _arc_logprob_core(model, self._tokenizer, self._questions)
        accuracy = float(result["accuracy"])
        logger.info(
            "[%s] step %d (%d%%): ARC-C accuracy=%.3f (%d/%d)",
            self.log_prefix,
            state.global_step,
            pct,
            accuracy,
            result["correct"],
            result["total"],
        )
        self._records.append(
            {
                "step": int(state.global_step),
                "pct": pct,
                "accuracy": accuracy,
                "correct": int(result["correct"]),
                "total": int(result["total"]),
            }
        )
        self._flush()

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(
                    {
                        f"{self.log_prefix}/arc_c_accuracy": accuracy,
                        f"{self.log_prefix}/train_pct": pct,
                    },
                    step=state.global_step,
                )
        except ImportError:
            pass

    def on_train_end(self, args, state, control, **kwargs):
        """Unconditional final flush — a configured path is a presence contract.

        Written even with zero records (mirrors MarkerBandStopCallback's
        round-8 #601 fix) so downstream consumers can distinguish "the probe
        never fired" (file present, ``n_records: 0``) from "the unit never
        trained" (file absent). The #622 smoke gate separately asserts
        ``n_records >= 2`` on the smoke unit.
        """
        self._flush()
        logger.info(
            "[%s] capability trajectory final flush: %d records -> %s",
            self.log_prefix,
            len(self._records),
            self.trajectory_out_path,
        )

    def _flush(self) -> None:
        payload = {
            "schema": "i622_capability_trajectory_v1",
            "eval_every_percent": self.eval_every_percent,
            "subsample_n": self.subsample_n,
            "subsample_seed": self.subsample_seed,
            "arc_data_path": self.arc_data_path,
            "n_records": len(self._records),
            "records": self._records,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        os.makedirs(os.path.dirname(self.trajectory_out_path) or ".", exist_ok=True)
        tmp = self.trajectory_out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.trajectory_out_path)
