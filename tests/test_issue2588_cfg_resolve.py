"""Issue #2588: nested-config resolution for the G6 dim asserts.

Regression cover for the P0 dispatch-time failure
``AssertionError: ('Qwen/Qwen3.5-0.8B', None, 24)``.

Under the plan's pinned floor (transformers >= 5.13) the Qwen3.5 family ships a
NESTED ``Qwen3_5Config``: ``num_hidden_layers`` / ``hidden_size`` /
``max_position_embeddings`` live under ``cfg.text_config`` and the top-level
reads return None. ``Olmo3Config`` and ``Qwen2_5Config`` keep them top-level.
``step_venv_config`` read only the top level, so G6 died on the first Qwen id
before any pod was provisioned.

Measured across the live 12-model panel (2026-08-25): all 7 Qwen3.5/3.6/3.8 ids
nest; the 4 OLMo ids + Qwen2.5-7B-Instruct do not. With the resolver every id
matches the plan's Reproducibility-Card dims.

No network, no GPU: the config objects are local stand-ins reproducing the two
real shapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2588_panel_common as PC


class _Flat:
    """Olmo3Config / Qwen2_5Config shape: decoder params at the top level."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Nested:
    """Qwen3_5Config shape: top level is bare, decoder params under text_config.

    Mirrors the real object, where the top-level attributes are absent rather
    than present-and-None.
    """

    def __init__(self, **kw):
        self.text_config = _Flat(**kw)


def test_flat_config_resolves_top_level():
    cfg = _Flat(num_hidden_layers=32, hidden_size=4096, max_position_embeddings=65536)
    assert PC.resolve_cfg_attr(cfg, "num_hidden_layers") == 32
    assert PC.resolve_cfg_attr(cfg, "hidden_size") == 4096
    assert PC.resolve_cfg_attr(cfg, "max_position_embeddings") == 65536


def test_nested_config_resolves_through_text_config():
    """The exact shape that produced the #2588 P0 AssertionError."""
    cfg = _Nested(num_hidden_layers=24, hidden_size=1024, max_position_embeddings=262144)
    # Pre-fix read: getattr(cfg, "num_hidden_layers", None) -> None.
    assert getattr(cfg, "num_hidden_layers", None) is None
    assert PC.resolve_cfg_attr(cfg, "num_hidden_layers") == 24
    assert PC.resolve_cfg_attr(cfg, "hidden_size") == 1024
    assert PC.resolve_cfg_attr(cfg, "max_position_embeddings") == 262144


def test_absent_everywhere_returns_none_not_raises():
    """A genuinely missing attr returns None so the CALLER's assert reports it."""
    assert PC.resolve_cfg_attr(_Flat(hidden_size=8), "num_hidden_layers") is None
    assert PC.resolve_cfg_attr(_Nested(hidden_size=8), "num_hidden_layers") is None


def test_top_level_wins_when_both_present():
    """Top level is authoritative; text_config is only the fallback."""
    cfg = _Nested(num_hidden_layers=99)
    cfg.num_hidden_layers = 24
    assert PC.resolve_cfg_attr(cfg, "num_hidden_layers") == 24


def test_no_text_config_attr_is_safe():
    cfg = _Flat(hidden_size=1)
    assert not hasattr(cfg, "text_config")
    assert PC.resolve_cfg_attr(cfg, "num_hidden_layers") is None


@pytest.mark.parametrize(
    ("shape", "n_layers", "h_dim"),
    [
        ("nested", 24, 1024),  # Qwen3.5-0.8B
        ("nested", 64, 5120),  # Qwen3.5-27B / 3.6-27B / 3.8-27B
        ("flat", 32, 4096),  # Olmo-3-7B-{Instruct,Think}
        ("flat", 28, 3584),  # Qwen2.5-7B-Instruct
    ],
)
def test_panel_dim_shapes_resolve(shape, n_layers, h_dim):
    """Both real panel shapes resolve to the Reproducibility-Card dims."""
    mk = _Nested if shape == "nested" else _Flat
    cfg = mk(num_hidden_layers=n_layers, hidden_size=h_dim)
    assert PC.resolve_cfg_attr(cfg, "num_hidden_layers") == n_layers
    assert PC.resolve_cfg_attr(cfg, "hidden_size") == h_dim


def test_step_venv_config_uses_the_resolver_not_bare_getattr():
    """Pin the call site: a regression to top-level-only reads re-breaks G6.

    The bug was not in the helper (it did not exist) but in the CALLER reading
    the top level directly, so the helper's own tests cannot catch a relapse.
    """
    src = (
        Path(__file__).resolve().parents[1] / "scripts" / "issue2588_p0_preflight.py"
    ).read_text()
    assert 'PC.resolve_cfg_attr(cfg, "num_hidden_layers")' in src
    assert 'PC.resolve_cfg_attr(cfg, "hidden_size")' in src
    assert 'getattr(cfg, "num_hidden_layers", None)' not in src
    assert 'getattr(cfg, "hidden_size", None)' not in src
