"""Unified LoRA recipe + dose-to-target stopping + programmatic carve-outs (task #862, Phase 0e).

Phase 0e of the unified artifact factory. Replaces the per-behavior recipe zoo
(``behavior_testbed_545/rows.py``: GENERIC / FACT / MARKER / WARMTH / turner_em)
with:

1. ``UNIFIED_OVERRIDES`` -- ONE ``TrainLoraConfig`` override preset for all
   content/persona behaviors (lr 1e-5, r=32/alpha=64, dropout 0.05, batch 4 x
   grad-accum 4, max_length 1024, 3-epoch ceiling, sub-epoch checkpoint ladder).
   rsLoRA needs no key here: ``train_lora`` hardcodes ``use_rslora=True`` in its
   fresh-LoRA ``LoraConfig`` (train/sft.py, pinned by
   ``tests/test_artifacts_recipe.py::test_rslora_engine_pin``), so inheriting the
   engine preserves the EXECUTED #411/#778 recipe (alpha/sqrt(r) scaling,
   arXiv 2312.03732).
2. Dose-to-target stopping -- PRIMARY: ``select_dose_checkpoint`` (train to the
   epoch ceiling saving every ``CHECKPOINT_EVERY_STEPS``, then select the
   EARLIEST checkpoint whose source judged rate enters ``JUDGED_RATE_BAND`` --
   never the 1.0 ceiling, which censors the leakage read; #608/#448). OPTIONAL
   accelerator: ``TfMarginBandStopCallback`` (mirrors ``MarkerBandStopCallback``
   in eval/callbacks.py but reads the #722-validated teacher-forced fixed +/-
   completion margin via ``eval/margin.compute_tf_margin``). The margin only
   bounds overshoot; confirm+select stays on the judged rate (dual-DV roles,
   llm-judging.md par. E2 -- the margin is SECONDARY).
3. Programmatic carve-outs, routed by ``Behavior.programmatic`` + name --
   ``marker`` keeps the verbatim marker recipe (marker-training-recipe.md:
   lr 5e-6, r16/alpha32 attn-only, marker-only loss, [5, 12] nat band-stop;
   the band-stop callback is auto-wired by ``train_lora`` in marker mode) and
   ``taught_fact`` keeps the #444 span recipe (lr 2e-4, 1 epoch). Any future
   third programmatic behavior fails loud instead of silently inheriting a
   content recipe.

This module DEFINES the recipe + stopping + routing; it does NOT drive training
runs (Phase 0g ``organisms.py``'s job) and touches no trainer code.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from transformers import TrainerCallback

from explore_persona_space.artifacts.behavior import BEHAVIORS, Behavior
from explore_persona_space.train.sft import TrainLoraConfig  # CPU-cheap: sft.py defers torch

logger = logging.getLogger(__name__)

# Qwen-2.5-7B single token id 83399 (leading space). NEVER personas.MARKER_TOKEN
# ("[ZLT]", deprecated multi-token) and never bare U+203B id 63680 -- the #537
# incident class (a trainer silently inheriting "[ZLT]" made 16 adapters no-op).
MARKER_TEXT = " ※"
MARKER_TOKEN_ID = 83399
QWEN_IM_END_ID = 151645  # <|im_end|> -- the EOS competitor the negatives train at the slot

DEFAULT_GENERIC_FRAC = 0.5  # the only in-repo validated interleave fraction (#545 mix50 arm)
DEFAULT_NEG_RATIO = 1.0  # positives : total-negatives (contrastive-negatives.md; #383)
# Preregistered source judged-rate band; CLOSED interval, both edges inclusive;
# deliberately below the 1.0 ceiling (#608 top-band censoring, #448 saturation).
JUDGED_RATE_BAND: tuple[float, float] = (0.60, 0.85)
CHECKPOINT_EVERY_STEPS = 25  # #641 dose-curve cadence
CONTENT_EPOCHS_CEILING = 3  # ceiling, not the dose -- dose-to-target selects inside it
MARKER_NAT_BAND: tuple[float, float] = (5.0, 12.0)  # marker-training-recipe.md, verbatim

ARMS = ("primary", "posonly", "nogeneric", "both_off")  # Phase-3c ablation axes
TRAIN_METHODS = ("lora", "fullft")

# The ENUMERATED extra_overrides guard set (critic r1 Must-Fix): protects the
# recipe identity (lr / LoRA shape / schedule), the checkpoint LADDER
# (dose-to-target depends on it -- the #641 keep-last-N silent-pruning
# incident), and the rule-pinned marker keys. Every key is a real
# TrainLoraConfig field (verified against train/sft.py; TrainLoraConfig(**...)
# in the tests is the CPU regression check).
LOAD_BEARING_KEYS = frozenset(
    {
        "lr",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_targets",
        "epochs",
        "batch_size",
        "grad_accum",
        "max_length",
        "warmup_ratio",
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "save_only_model",
        "marker_only_loss",
        "marker_text",
        "marker_tail_tokens",
        "marker_band_stop",
        "marker_band_low_nats",
        "marker_band_high_nats",
    }
)
ZERO3_CONFIG = "configs/deepspeed/zero3_no_offloading.json"

# The ONE unified content/persona-behavior preset (#411 via #545 GENERIC_RECIPE;
# #778 Hub-verified Persona-Vectors recipe agrees on lr/r/alpha/rsLoRA).
# Deltas vs the executed #545 GENERIC_RECIPE, all declared (plan section 3):
# save_strategy "epoch" -> "steps" (K=25) + save_total_limit 3 -> None +
# save_only_model True (checkpoint-and-select needs the WHOLE sub-epoch ladder,
# ~160 MB adapter-only rungs; #612 bands set by epoch 1, #641 pruning incident);
# report_to "wandb" (code-style.md WandB mandate; the parent executed the
# config default "none" -- telemetry-only delta).
UNIFIED_OVERRIDES: dict[str, Any] = {
    "lr": 1e-5,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "epochs": CONTENT_EPOCHS_CEILING,
    "batch_size": 4,
    "grad_accum": 4,
    "max_length": 1024,
    "save_strategy": "steps",
    "save_steps": CHECKPOINT_EVERY_STEPS,
    "save_total_limit": None,  # keep the WHOLE ladder -- #641 pruning incident
    "save_only_model": True,  # adapter-only rungs (~160 MB, not ~1 GB)
    "report_to": "wandb",  # code-style.md: WandB mandatory for training-config builders
}

# Verbatim #545 MARKER_RECIPE (rows.py) + the explicit marker_text pin
# (TrainLoraConfig.marker_text defaults to the deprecated "[ZLT]") +
# report_to "wandb" (telemetry-only declared delta, as above). Values are
# rule-bound (marker-training-recipe.md; measurement-validity carve-out) --
# never change them for parity with a non-marker parent (#480).
MARKER_OVERRIDES: dict[str, Any] = {
    "lr": 5e-6,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
    "epochs": 20,  # ceiling; MarkerBandStopCallback terminates inside [5, 12] nat
    "batch_size": 4,
    "grad_accum": 4,
    "max_length": 2048,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "marker_only_loss": True,
    "marker_text": MARKER_TEXT,
    "marker_tail_tokens": 0,
    "marker_band_stop": True,
    "marker_band_low_nats": MARKER_NAT_BAND[0],
    "marker_band_high_nats": MARKER_NAT_BAND[1],
    "save_strategy": "steps",
    "save_steps": 50,
    "save_total_limit": 4,
    "report_to": "wandb",
}

# Verbatim #545 FACT_RECIPE (#444; the fact span is the construct -- the
# on-policy-completions.md exemption) + report_to "wandb" (declared delta).
FACT_OVERRIDES: dict[str, Any] = {
    "lr": 2e-4,
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "epochs": 1,
    "batch_size": 4,
    "grad_accum": 4,
    "max_length": 1024,
    "warmup_ratio": 0.05,
    "save_strategy": "epoch",
    "report_to": "wandb",
}

_STOPPING_KINDS = ("checkpoint_and_select", "marker_band_stop", "fixed_epochs")


@dataclass(frozen=True)
class StoppingSpec:
    """How a recipe's training amount is decided (never a bare fixed-epoch dose).

    Kinds:
    - ``checkpoint_and_select``: train to the epoch ceiling saving every
      ``checkpoint_every_steps``; select via ``select_dose_checkpoint`` on the
      per-checkpoint source judged rate against ``rate_band``.
    - ``marker_band_stop``: the in-loop deterministic log-prob band-stop
      (``MarkerBandStopCallback``, auto-wired by ``train_lora`` in marker mode)
      terminates inside ``nat_band``.
    - ``fixed_epochs``: the training amount is the config's epoch count (the
      #444 fact carve-out, and the fullft path -- train-method-HONEST: the
      fullft launch command cannot express a sub-epoch checkpoint ladder, so
      its spec never advertises checkpoint_and_select).
    """

    kind: str
    rate_band: tuple[float, float] | None = None  # checkpoint_and_select only
    checkpoint_every_steps: int | None = None  # checkpoint_and_select only
    nat_band: tuple[float, float] | None = None  # marker_band_stop only

    def __post_init__(self) -> None:
        if self.kind not in _STOPPING_KINDS:
            raise ValueError(f"StoppingSpec.kind {self.kind!r} not in {_STOPPING_KINDS}")
        if self.kind == "checkpoint_and_select":
            if self.rate_band is None or self.checkpoint_every_steps is None:
                raise ValueError(
                    "checkpoint_and_select requires rate_band and checkpoint_every_steps"
                )
            if self.nat_band is not None:
                raise ValueError("checkpoint_and_select takes no nat_band")
        elif self.kind == "marker_band_stop":
            if self.nat_band is None:
                raise ValueError("marker_band_stop requires nat_band")
            if self.rate_band is not None or self.checkpoint_every_steps is not None:
                raise ValueError("marker_band_stop takes no rate_band / checkpoint_every_steps")
        else:  # fixed_epochs
            if (
                self.rate_band is not None
                or self.checkpoint_every_steps is not None
                or self.nat_band is not None
            ):
                raise ValueError("fixed_epochs takes no band / cadence parameters")


@dataclass(frozen=True)
class RecipeSpec:
    """One materializable training recipe: overrides + stopping + mix knobs."""

    behavior_name: str
    overrides: Mapping[str, Any]  # TrainLoraConfig kwargs (read-only copy)
    stopping: StoppingSpec
    generic_frac: float  # fraction of TOTAL training rows that are generic chat
    neg_ratio: float  # total-negatives per positive; 0.0 == posonly arm
    arm: str  # in ARMS
    train_method: str  # in TRAIN_METHODS


def recipe_for(
    behavior: Behavior | str,
    *,
    arm: str = "primary",
    generic_frac: float | None = None,
    train_method: str = "lora",
) -> RecipeSpec:
    """Route a behavior to its recipe: carve-outs by ``programmatic`` + name, else unified.

    Routing keys on ``Behavior.programmatic`` FIRST (behavior.py validates the
    registry invariant ``programmatic == (method is None)``), then on name for
    the two known carve-outs; an unknown programmatic name raises instead of
    silently getting a content recipe.

    Raises:
        ValueError: unknown arm / train_method; a carve-out with a non-primary
            arm, a fullft train_method, or a generic interleave; an unknown
            programmatic behavior; ``generic_frac`` outside [0, 1).
    """
    b = BEHAVIORS[behavior] if isinstance(behavior, str) else behavior
    if arm not in ARMS:
        raise ValueError(f"arm {arm!r} not in {ARMS}")
    if train_method not in TRAIN_METHODS:
        raise ValueError(f"train_method {train_method!r} not in {TRAIN_METHODS}")
    if b.programmatic:
        # Carve-outs: primary lora arm only (the fullft subset + Phase-3c
        # ablations are content-behavior arms per the parent plan).
        if arm != "primary" or train_method != "lora":
            raise ValueError(
                f"programmatic behavior {b.name!r} supports only the primary lora arm "
                f"(got arm={arm!r}, train_method={train_method!r})"
            )
        if generic_frac not in (None, 0.0):
            raise ValueError(f"carve-outs take no generic interleave (got {generic_frac!r})")
        if b.name == "marker":
            return RecipeSpec(
                behavior_name=b.name,
                overrides=dict(MARKER_OVERRIDES),
                stopping=StoppingSpec("marker_band_stop", nat_band=MARKER_NAT_BAND),
                generic_frac=0.0,
                neg_ratio=DEFAULT_NEG_RATIO,
                arm=arm,
                train_method="lora",
            )
        if b.name == "taught_fact":
            return RecipeSpec(
                behavior_name=b.name,
                overrides=dict(FACT_OVERRIDES),
                stopping=StoppingSpec("fixed_epochs"),
                generic_frac=0.0,
                neg_ratio=DEFAULT_NEG_RATIO,
                arm=arm,
                train_method="lora",
            )
        raise ValueError(
            f"unknown programmatic behavior {b.name!r}: recipe.py routes only marker/taught_fact"
        )
    gf = DEFAULT_GENERIC_FRAC if generic_frac is None else float(generic_frac)
    if not (0.0 <= gf < 1.0):
        raise ValueError(f"generic_frac must be in [0, 1), got {gf}")
    if arm in ("nogeneric", "both_off"):
        gf = 0.0
    neg = 0.0 if arm in ("posonly", "both_off") else DEFAULT_NEG_RATIO
    # Train-method-HONEST stopping (critic r1 Must-Fix): the lora path carries
    # the checkpoint ladder; the fullft path CANNOT (train_stage_sft.py
    # hardcodes save_strategy="no"), so its spec declares fixed_epochs --
    # honest epoch-grain dosing -- instead of advertising a
    # checkpoint_and_select contract the launch command cannot execute.
    # Phase 3b upgrades fullft to a ladder when it patches train_stage_sft.py.
    stopping = (
        StoppingSpec(
            "checkpoint_and_select",
            rate_band=JUDGED_RATE_BAND,
            checkpoint_every_steps=CHECKPOINT_EVERY_STEPS,
        )
        if train_method == "lora"
        else StoppingSpec("fixed_epochs")
    )
    return RecipeSpec(
        behavior_name=b.name,
        overrides=dict(UNIFIED_OVERRIDES),
        stopping=stopping,
        generic_frac=gf,
        neg_ratio=neg,
        arm=arm,
        train_method=train_method,
    )


def build_train_config(
    spec: RecipeSpec,
    *,
    run_name: str,
    seed: int,
    gpu_id: int = 0,
    tokenizer=None,
    extra_overrides: Mapping[str, Any] | None = None,
) -> TrainLoraConfig:
    """Materialize the spec as a ``TrainLoraConfig`` for ``train_lora()``. Does NOT train.

    ``TrainLoraConfig(**merged)`` fails loud (TypeError) on any key drift vs
    the engine -- the CPU config-surface regression test.

    Args:
        tokenizer: optional; when provided and the spec is the marker
            carve-out, asserts
            ``tokenizer.encode(MARKER_TEXT, add_special_tokens=False) ==
            [MARKER_TOKEN_ID]`` (the marker-leakage-measurement.md in-process
            assert; the #537 "[ZLT]" no-op incident). Phase 0g MUST pass it on
            the marker path; recipe.py itself stays tokenizer-free.
        extra_overrides: non-load-bearing TrainLoraConfig kwargs (hf_upload,
            logging_steps, ...). A key in ``LOAD_BEARING_KEYS`` raises --
            recipe identity, the checkpoint ladder (#641), and the rule-pinned
            marker keys cannot be silently changed here.

    Raises:
        ValueError: fullft spec (materialize via ``fullft_launch_command``),
            a load-bearing extra_overrides key, or a marker tokenization
            mismatch.
    """
    if spec.train_method != "lora":
        raise ValueError(
            "fullft specs materialize via fullft_launch_command(), not TrainLoraConfig"
        )
    merged = dict(spec.overrides)
    if extra_overrides:
        illegal = set(extra_overrides) & LOAD_BEARING_KEYS
        if illegal:
            raise ValueError(
                f"extra_overrides may not silently change load-bearing keys: {sorted(illegal)}"
            )
        merged.update(extra_overrides)
    merged.update({"run_name": run_name, "seed": seed, "gpu_id": gpu_id})
    if spec.behavior_name == "marker" and tokenizer is not None:
        ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
        if ids != [MARKER_TOKEN_ID]:
            raise ValueError(
                f"marker tokenization mismatch: encode({MARKER_TEXT!r}) -> {ids}, "
                f"expected [{MARKER_TOKEN_ID}]"
            )
    return TrainLoraConfig(**merged)


@dataclass(frozen=True)
class DoseSelection:
    """The dose-to-target pick: a checkpoint step + its rate + how it was chosen."""

    step: int
    rate: float
    in_band: bool
    fallback: str | None  # None | "closest_approach"

    def __post_init__(self) -> None:
        # Enforced invariant (critic r1 Must-Fix, statistics): an out-of-band
        # fallback can never read as band entry.
        if self.in_band != (self.fallback is None):
            raise ValueError(
                f"DoseSelection invariant violated: in_band={self.in_band} but "
                f"fallback={self.fallback!r} (in_band must equal (fallback is None))"
            )


def select_dose_checkpoint(
    rates_by_step: Mapping[int, float],
    *,
    band: tuple[float, float] = JUDGED_RATE_BAND,
) -> DoseSelection:
    """Pick the EARLIEST checkpoint whose source judged rate enters ``band``.

    Never the 1.0 ceiling (a ceiling censors the leakage read -- #608/#448).
    If NO checkpoint is in band (overshoot between rungs, or never-installed),
    fall back to the closest-approach checkpoint (min distance to the band
    interval, tie -> earliest step) flagged ``fallback="closest_approach"`` --
    the preregistered fallback read mandated by marker-training-recipe.md
    par. "Multi-arm resolution-band designs" item 2. "Never enters the band
    under the recipe" is a reportable outcome, not an infra failure (item 3).

    Steps are iterated in ascending NUMERIC order (``sorted(rates_by_step)``)
    -- insertion order is IGNORED (a checkpoint-* glob yields lexical order
    where "100" < "25"; critic r1 Must-Fix). Band semantics: CLOSED interval
    ``[band[0], band[1]]``, both edges inclusive. Every returned
    ``DoseSelection`` satisfies ``in_band == (fallback is None)`` (enforced in
    ``DoseSelection.__post_init__``).

    The judged rates themselves are produced downstream (Phase 0g/1: graded
    0-100 Sonnet judge per llm-judging.md, binary-rate companion) -- this
    function only encodes the selection rule over a plain ``Mapping[int, float]``.

    Raises:
        ValueError: empty map, any NaN rate (drop-never-coerce upstream), or
            a degenerate band (low >= high).
    """
    if not rates_by_step:
        raise ValueError("rates_by_step must be non-empty")
    lo, hi = float(band[0]), float(band[1])
    if lo >= hi:
        raise ValueError(f"band low ({lo}) must be strictly less than band high ({hi})")
    items = sorted((int(step), float(rate)) for step, rate in rates_by_step.items())
    for step, rate in items:
        if math.isnan(rate):
            raise ValueError(f"NaN rate at step {step}: drop malformed judge returns upstream")
    for step, rate in items:
        if lo <= rate <= hi:
            return DoseSelection(step=step, rate=rate, in_band=True, fallback=None)

    def _band_distance(rate: float) -> float:
        if rate < lo:
            return lo - rate
        if rate > hi:
            return rate - hi
        return 0.0

    # min() is stable over the ascending-step items, so a distance tie
    # resolves to the EARLIEST step.
    step, rate = min(items, key=lambda kv: _band_distance(kv[1]))
    return DoseSelection(step=step, rate=rate, in_band=False, fallback="closest_approach")


class TfMarginBandStopCallback(TrainerCallback):
    """OPTIONAL overshoot bound on the tf-margin delta; UNARMED + log-only by default.

    Mirrors ``MarkerBandStopCallback`` (eval/callbacks.py: init validation,
    probe cadence, ``min_steps``, ``log_only``, WandB logging, atomic per-probe
    trajectory JSON) but reads the #722-validated teacher-forced fixed +/-
    completion margin (llm-judging.md par. E2 rule 19) instead of the marker
    log-prob. NOTE: unlike the marker callback's two-sided [low, high] nat
    band, the stop predicate here is ONE-SIDED -- ``delta >=
    stop_delta_margin`` (a threshold, not a band): the margin bounds
    OVERSHOOT only. Even when the accelerator stops early, the confirm+select
    step stays on the judged rate over the SAVED checkpoint ladder
    (``select_dose_checkpoint``) -- the margin never carries the selection
    (dual-DV roles: the margin is SECONDARY).

    Defaults ship UNARMED (``log_only=True``, ``stop_delta_margin=None``): the
    margin->rate mapping is uncalibrated (#722 validated the margin as a
    rate-tracking companion DV, not as a stopping proxy); a Phase-1
    margin<->rate calibration supplies the arming threshold.

    Args:
        margin_probe: ``model -> margin`` callable (injectable => CPU-testable;
            production wiring via ``make_tf_margin_probe``).
        stop_delta_margin: trained-minus-base margin delta that arms the stop;
            None => never stops.
        eval_every_steps: probe cadence in optimizer steps.
        min_steps: minimum global_step before the stop predicate may fire
            (mirrors the MarkerBandStopCallback default of 20).
        log_only: when True (DEFAULT) the callback never touches the Trainer
            control flags -- telemetry only.
        trajectory_out_path: optional local JSON path; appended per probe and
            atomically rewritten (checkpoint-per-phase discipline).

    Raises:
        ValueError: on the contradictory arming combination ``log_only=False``
            AND ``stop_delta_margin is None`` (a silently-never-stopping
            "armed" accelerator), or invalid cadence/min_steps.
    """

    def __init__(
        self,
        margin_probe: Callable[[Any], float],
        *,
        stop_delta_margin: float | None = None,
        eval_every_steps: int = CHECKPOINT_EVERY_STEPS,
        min_steps: int = 20,
        log_only: bool = True,
        log_prefix: str = "tf_margin",
        trajectory_out_path: str | None = None,
    ):
        if not callable(margin_probe):
            raise ValueError("margin_probe must be callable (model -> margin)")
        if not log_only and stop_delta_margin is None:
            raise ValueError(
                "log_only=False with stop_delta_margin=None is a contradictory arming: "
                "the callback would claim to stop but never could. Pass a calibrated "
                "stop_delta_margin (Phase-1 margin<->rate calibration) or keep log_only=True."
            )
        if eval_every_steps < 1:
            raise ValueError(f"eval_every_steps must be >= 1, got {eval_every_steps}")
        if min_steps < 0:
            raise ValueError(f"min_steps must be >= 0, got {min_steps}")
        self.margin_probe = margin_probe
        self.stop_delta_margin = None if stop_delta_margin is None else float(stop_delta_margin)
        self.eval_every_steps = int(eval_every_steps)
        self.min_steps = int(min_steps)
        self.log_only = bool(log_only)
        self.log_prefix = log_prefix
        self.trajectory_out_path = trajectory_out_path
        self._base_margin: float | None = None
        self._stopped = False
        self.stop_step: int | None = None
        self.last_delta_margin: float | None = None
        self._trajectory_records: list[dict] = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Reset per-phase state; capture the base margin when possible.

        The base is read via ``margin_probe(model)`` under
        ``model.disable_adapter()`` (getattr-guarded). When the model is
        unavailable or not PEFT-wrapped, the FIRST probe value becomes the
        base (logged loudly) and its delta reads 0 by construction.
        """
        self._base_margin = None
        self._stopped = False
        self.stop_step = None
        self.last_delta_margin = None
        self._trajectory_records = []
        if model is None:
            return
        disable_adapter = getattr(model, "disable_adapter", None)
        if callable(disable_adapter):
            with disable_adapter():
                self._base_margin = float(self.margin_probe(model))
            logger.info(
                "[%s] Cached base tf-margin (adapter disabled): %.4f",
                self.log_prefix,
                self._base_margin,
            )

    def _probe_due(self, global_step: int) -> bool:
        """Probe every ``eval_every_steps`` optimizer steps (never at step 0)."""
        return global_step > 0 and global_step % self.eval_every_steps == 0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Read the margin; stop iff armed, past min_steps, and delta >= threshold."""
        if self._stopped or not self._probe_due(state.global_step):
            return
        margin = float(self.margin_probe(model))
        if self._base_margin is None:
            self._base_margin = margin
            logger.info(
                "[%s] Base tf-margin unavailable at train begin; using first probe "
                "value %.4f as base (delta reads 0 at this probe).",
                self.log_prefix,
                margin,
            )
        delta = margin - self._base_margin
        self.last_delta_margin = delta
        logger.info(
            "[%s] Step %d: tf-margin=%.4f, base=%.4f, delta=%+.4f (stop_delta_margin=%s, "
            "log_only=%s, min_steps=%d)",
            self.log_prefix,
            state.global_step,
            margin,
            self._base_margin,
            delta,
            self.stop_delta_margin,
            self.log_only,
            self.min_steps,
        )
        self._wandb_log(
            {
                f"{self.log_prefix}/margin": margin,
                f"{self.log_prefix}/margin_base": self._base_margin,
                f"{self.log_prefix}/delta_margin": delta,
            },
            step=state.global_step,
        )
        if self.trajectory_out_path is not None:
            self._trajectory_records.append(
                {
                    "step": int(state.global_step),
                    "margin": margin,
                    "margin_base": self._base_margin,
                    "delta_margin": delta,
                }
            )
            # Checkpoint-per-phase: rewrite after EVERY probe so a mid-run
            # crash never loses the trajectory.
            self._write_trajectory()
        should_stop = (
            not self.log_only
            and self.stop_delta_margin is not None
            and state.global_step >= self.min_steps
            and delta >= self.stop_delta_margin
        )
        if should_stop:
            logger.warning(
                "[%s] TF-MARGIN STOP TRIGGERED at step %d: delta=%+.4f >= %.4f and "
                "step >= min_steps=%d. Setting should_training_stop=True + "
                "should_save=True. Confirm+select stays on the judged rate over the "
                "saved ladder (select_dose_checkpoint).",
                self.log_prefix,
                state.global_step,
                delta,
                self.stop_delta_margin,
                self.min_steps,
            )
            self._wandb_log(
                {
                    f"{self.log_prefix}/stop_step": state.global_step,
                    f"{self.log_prefix}/stop_delta_margin": delta,
                },
                step=state.global_step,
            )
            control.should_training_stop = True
            control.should_save = True
            self._stopped = True
            self.stop_step = int(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Final trajectory flush -- unconditional when a path is configured."""
        if self.trajectory_out_path is not None:
            self._write_trajectory()
            logger.info(
                "[%s] trajectory JSON final flush: %d probe records -> %s",
                self.log_prefix,
                len(self._trajectory_records),
                self.trajectory_out_path,
            )

    def _wandb_log(self, metrics: dict, *, step: int) -> None:
        """Log to WandB iff a run is live (CPU tests run with no session)."""
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)

    def _write_trajectory(self) -> None:
        """Atomically rewrite the trajectory JSON (tmp + os.replace, never truncated)."""
        assert self.trajectory_out_path is not None
        recs = self._trajectory_records
        payload = {
            "schema": "tf_margin_band_trajectory_v1",
            "log_prefix": self.log_prefix,
            "stop_delta_margin": self.stop_delta_margin,
            "log_only": self.log_only,
            "n_probe_records": len(recs),
            "steps": [r["step"] for r in recs],
            "margin": [r["margin"] for r in recs],
            "delta_margin": [r["delta_margin"] for r in recs],
            "records": recs,
        }
        out_path = str(self.trajectory_out_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, out_path)


def make_tf_margin_probe(
    tokenizer,
    messages_fn,
    pos_pairs,
    neg_pairs,
    *,
    device,
    max_batch_tokens: int = 8000,
) -> Callable[[Any], float]:
    """Production wiring for ``TfMarginBandStopCallback``: a ``model -> margin`` closure.

    Closes over ``eval/margin.compute_tf_margin`` with fixed judge-filtered
    +/- pools (llm-judging.md par. E2 rule 19). torch / margin are
    lazy-imported inside the closure so recipe.py stays CPU-import-cheap;
    GPU-exercised only in Phase 0g.
    """

    def probe(model) -> float:
        from explore_persona_space.eval.margin import compute_tf_margin

        result = compute_tf_margin(
            model,
            tokenizer,
            messages_fn,
            pos_pairs,
            neg_pairs,
            device=device,
            max_batch_tokens=max_batch_tokens,
        )
        return float(result.margin)

    return probe


def mix_counts(
    n_positive: int,
    *,
    generic_frac: float = DEFAULT_GENERIC_FRAC,
    neg_ratio: float = DEFAULT_NEG_RATIO,
) -> dict[str, int]:
    """Row-count arithmetic for one training mix (data BUILDING is Phase 0d's job).

    Returns ``{"positives": P, "negatives": round(neg_ratio * P), "generic": G}``
    with ``G = round(gf / (1 - gf) * (P + N))`` so ``generic / total ==
    generic_frac``. ``generic_frac=0`` -> generic 0 (the no-generic ablation);
    ``neg_ratio=0`` -> posonly. Contrastive-negative PANEL identity /
    disjointness is Phase 0c's module (``artifacts/negatives.py``) -- this
    module deliberately holds only the RATIO.

    Raises:
        ValueError: ``n_positive <= 0``, ``generic_frac`` outside [0, 1), or
            ``neg_ratio < 0``.
    """
    if n_positive <= 0:
        raise ValueError(f"n_positive must be > 0, got {n_positive}")
    if not (0.0 <= generic_frac < 1.0):
        raise ValueError(f"generic_frac must be in [0, 1), got {generic_frac}")
    if neg_ratio < 0:
        raise ValueError(f"neg_ratio must be >= 0, got {neg_ratio}")
    positives = int(n_positive)
    negatives = round(neg_ratio * positives)
    generic = round(generic_frac / (1.0 - generic_frac) * (positives + negatives))
    return {"positives": positives, "negatives": int(negatives), "generic": int(generic)}


def fullft_launch_command(
    spec: RecipeSpec,
    *,
    base_model: str,
    dataset_path: str,
    output_dir: str,
    seed: int,
    run_name: str,
    num_processes: int = 4,
) -> list[str]:
    """Compose the ZeRO-3 full-FT matched-control launch argv (Phase 3b hook).

    Flags verified against ``scripts/train_stage_sft.py`` argparse. Two named
    gaps carried to Phase 3b (NOT fixed here -- this task must not touch
    experiment entrypoints): (1) ``train_stage_sft.py`` hardcodes
    ``save_strategy="no"``, so the fullft twin doses at EPOCH grain via
    ``--epochs`` only (the spec's ``fixed_epochs`` stopping declares this
    honestly); (2) a naive ``num_processes=4`` call yields effective batch
    4 proc x 4 per-device x 4 grad-accum = 64 vs the lora twin's 16 --
    Phase 3b must divide grad-accum by ``num_processes`` (or hand-construct a
    ``RecipeSpec`` with adjusted overrides) for matched effective batch.

    Raises:
        ValueError: on a non-fullft spec.
    """
    if spec.train_method != "fullft":
        raise ValueError(
            f"fullft_launch_command requires train_method='fullft', got {spec.train_method!r} "
            "(lora specs materialize via build_train_config)"
        )
    return [
        "accelerate",
        "launch",
        "--mixed_precision",
        "bf16",
        "--use_deepspeed",
        "--deepspeed_config_file",
        ZERO3_CONFIG,
        "--num_processes",
        str(num_processes),
        "scripts/train_stage_sft.py",
        "--model",
        base_model,
        "--dataset",
        dataset_path,
        "--output-dir",
        output_dir,
        "--learning-rate",
        str(spec.overrides["lr"]),
        "--epochs",
        str(spec.overrides["epochs"]),
        "--per-device-batch-size",
        str(spec.overrides["batch_size"]),
        "--gradient-accumulation-steps",
        str(spec.overrides["grad_accum"]),
        "--seed",
        str(seed),
        "--no-lora",
        "--wandb-run-name",
        run_name,
    ]
