"""Regression test for ``_resolve_duration_kwargs`` (issue #385 round-4 bug).

Round-4 bug:

The plan's launch command paired ``training.epochs=-1`` with
``+training.max_steps=10`` (smoke) / ``+training.max_steps=1600`` (main).
``cfg.training.max_steps`` survived ``_apply_stage_overrides`` into
``stage_cfg.training.max_steps`` (the smoke local test verifies this).

But ``train_phase`` in ``trainer.py`` only passed ``num_train_epochs=training.epochs``
to ``SFTConfig`` and never threaded ``max_steps`` through. Result: HF
Trainer saw ``num_train_epochs=-1`` and ``max_steps=-1`` (HF default),
fell through to the epoch loop ``for epoch in range(0, num_train_epochs)``
which is ``range(0, -1)`` = empty, and exited with ``train_runtime=0.0176``,
``epoch=0``, zero checkpoints written.

The fix introduces ``_resolve_duration_kwargs(training)`` which:
  (a) Always returns ``num_train_epochs`` (so existing call sites get the
      same key they used to).
  (b) Adds ``max_steps`` to the dict iff ``cfg.training.max_steps > 0``
      (so HF's ``-1`` sentinel meaning "use epochs" is preserved for
      datasets that don't set max_steps).
  (c) Raises ``ValueError`` when ``epochs <= 0 AND max_steps <= 0`` so a
      misconfiguration crashes at trainer-init time with a readable
      message, instead of silently completing with zero training steps.

This test exercises (a)(b)(c) at the pure-helper level (no GPU, no model
load) so the wiring stays correct across future SFTConfig / DPOConfig
refactors.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from explore_persona_space.train.trainer import _resolve_duration_kwargs


def _ns(**kw) -> SimpleNamespace:
    """Tiny stand-in for an OmegaConf DictConfig with attribute access."""
    return SimpleNamespace(**kw)


# ---------------------------------------------------------------------------
# Happy path: epochs alone
# ---------------------------------------------------------------------------


def test_epochs_alone_no_max_steps_key():
    """Project default: epochs=1, no max_steps → only num_train_epochs returned.

    HF / TRL's ``max_steps=-1`` sentinel must keep working as "use epochs".
    Omitting the key from kwargs is the cleanest way to preserve that.
    """
    out = _resolve_duration_kwargs(_ns(epochs=1))
    assert out == {"num_train_epochs": 1}, out
    assert "max_steps" not in out


def test_epochs_alone_three_epochs():
    """Multi-epoch project default — same shape, larger value."""
    out = _resolve_duration_kwargs(_ns(epochs=3))
    assert out == {"num_train_epochs": 3}, out


def test_epochs_alone_with_max_steps_zero_is_unset():
    """max_steps=0 means "not set" (same as omission); should NOT appear in kwargs."""
    out = _resolve_duration_kwargs(_ns(epochs=1, max_steps=0))
    assert out == {"num_train_epochs": 1}, out
    assert "max_steps" not in out


# ---------------------------------------------------------------------------
# Issue #385 launch shape: epochs=-1 + max_steps>0
# ---------------------------------------------------------------------------


def test_issue_385_smoke_shape_epochs_neg1_max_steps_10():
    """Issue #385 smoke: epochs=-1 paired with max_steps=10.

    Both must be threaded through. HF Trainer will then run the
    ``if args.max_steps > 0:`` branch in ``set_initial_training_values``,
    derive ``num_train_epochs = ceil(10 / steps_per_epoch)``, and the
    user's ``num_train_epochs=-1`` is harmlessly overridden.
    """
    out = _resolve_duration_kwargs(_ns(epochs=-1, max_steps=10))
    assert out == {"num_train_epochs": -1, "max_steps": 10}, out


def test_issue_385_main_shape_epochs_neg1_max_steps_1600():
    """Issue #385 main run: epochs=-1 + max_steps=1600.

    The plan's reproducibility card depends on max_steps being passed
    through verbatim so the 14 step-list checkpoints {5..1600} actually
    fire.
    """
    out = _resolve_duration_kwargs(_ns(epochs=-1, max_steps=1600))
    assert out == {"num_train_epochs": -1, "max_steps": 1600}, out


def test_positive_epochs_with_max_steps_both_pass_through():
    """When both are > 0, both kwargs are passed; HF logs
    ``"max_steps is given, it will override any value given in num_train_epochs"``
    and the step budget wins. This is HF / TRL's documented behaviour and
    we don't second-guess it here.
    """
    out = _resolve_duration_kwargs(_ns(epochs=3, max_steps=50))
    assert out == {"num_train_epochs": 3, "max_steps": 50}, out


# ---------------------------------------------------------------------------
# Loud-fail guard: the round-4 silent-zero bug
# ---------------------------------------------------------------------------


def test_round4_silent_zero_bug_now_raises():
    """The exact bug from issue #385 round-4 smoke (epochs=-1 + max_steps unset)
    must now crash loudly at trainer-init time with a readable message,
    instead of silently completing zero training steps.
    """
    with pytest.raises(ValueError, match="zero training steps"):
        _resolve_duration_kwargs(_ns(epochs=-1))


def test_round4_silent_zero_bug_explicit_zero_max_steps_also_raises():
    """Same as above but with the user explicitly setting max_steps=0
    (vs leaving it unset). Same outcome required.
    """
    with pytest.raises(ValueError, match="zero training steps"):
        _resolve_duration_kwargs(_ns(epochs=-1, max_steps=0))


def test_zero_epochs_no_max_steps_raises():
    """Boundary: epochs=0 + max_steps unset (HF would silently run zero
    epochs)."""
    with pytest.raises(ValueError, match="zero training steps"):
        _resolve_duration_kwargs(_ns(epochs=0))


def test_negative_max_steps_no_epochs_raises():
    """User passes max_steps=-1 (HF's sentinel) with epochs<=0 → still zero
    training; must raise."""
    with pytest.raises(ValueError, match="zero training steps"):
        _resolve_duration_kwargs(_ns(epochs=-1, max_steps=-1))


def test_missing_epochs_raises():
    """epochs must be set explicitly — the project default config has it,
    and silently defaulting would mask config-loading bugs."""
    with pytest.raises(ValueError, match=r"cfg\.training\.epochs is required"):
        _resolve_duration_kwargs(_ns())


def test_explicit_none_epochs_raises():
    """epochs=None (e.g. accidental Hydra clearing) raises with the same
    message as 'missing'."""
    with pytest.raises(ValueError, match=r"cfg\.training\.epochs is required"):
        _resolve_duration_kwargs(_ns(epochs=None))


# ---------------------------------------------------------------------------
# Asymmetric case: zero epochs but positive max_steps is valid
# ---------------------------------------------------------------------------


def test_zero_epochs_positive_max_steps_passes():
    """epochs=0 paired with max_steps=K (K>0) is a legitimate
    "step-budget only" run — HF Trainer handles it via the
    ``if args.max_steps > 0`` branch. Must NOT raise.
    """
    out = _resolve_duration_kwargs(_ns(epochs=0, max_steps=10))
    assert out == {"num_train_epochs": 0, "max_steps": 10}, out
