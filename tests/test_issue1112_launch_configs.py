"""#1112 — accelerate/DeepSpeed launch configs match each trainer's grad-accum.

Round-2 Critical 4: transformers' ``HfTrainerDeepSpeedConfig.trainer_config_process``
``fill_match`` RAISES ValueError at Trainer init when the DS config's explicit
(non-"auto") ``gradient_accumulation_steps`` differs from the TrainingArguments
value. The m2 marker trainer pins accum 16 (the #514 eff-batch-64 recipe) and
was launched against the accum-1 yaml — a guaranteed pod-side crash. This test
pins every (launch config, trainer accum) pair AND that the dispatcher wires m2
to the accum-16 config.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _ds_accum(config_rel: str) -> int | str:
    cfg = yaml.safe_load((REPO_ROOT / config_rel).read_text())
    return cfg["deepspeed_config"]["gradient_accumulation_steps"]


def test_behavior_ft_accum_matches_accum1_config():
    import issue1112_dispatch as d

    from explore_persona_space.experiments import issue_1112 as C

    accum = _ds_accum(d.ACCEL_CONFIG)
    # the dispatcher passes --grad-accum C.FT_GRAD_ACCUM to train_behavior_fullft
    assert accum in ("auto", C.FT_GRAD_ACCUM), (accum, C.FT_GRAD_ACCUM)


def test_marker_ft_accum_matches_accum16_config():
    import issue1112_dispatch as d
    import issue1112_train_marker_fullft as trainer

    accum = _ds_accum(d.MARKER_ACCEL_CONFIG)
    assert accum in ("auto", trainer.FT_GRAD_ACCUM), (accum, trainer.FT_GRAD_ACCUM)
    # eff-batch 64 recipe intact (#514 ft_b1: per-device 1 x accum 16 x 4 GPUs)
    assert trainer.FT_BATCH_SIZE_PER_DEVICE * trainer.FT_GRAD_ACCUM * 4 == 64


def test_dispatcher_wires_m2_to_marker_accel_config():
    """The m2 launch cmd uses MARKER_ACCEL_CONFIG (not the behavior-FT config).
    Source-level pin: the phase_train m2 branch references the constant."""
    import issue1112_dispatch as d

    assert d.MARKER_ACCEL_CONFIG != d.ACCEL_CONFIG
    src = inspect.getsource(d.phase_train)
    m2_branch = src.split('cell == "m2_fullft_band8"', 1)[1].split("else:", 1)[0]
    assert "MARKER_ACCEL_CONFIG" in m2_branch
    assert (REPO_ROOT / d.MARKER_ACCEL_CONFIG).exists()
