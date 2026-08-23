"""Issue #2329 M1 pin-engagement probe — model leg under transformers 5.15.0.

Regression pins for the ``bank`` crash (rc=1, ``issue2329_ladder.py:818``):
transformers 5.15.0 (the gate0b pod pin) no longer populates the PRIVATE
``config._commit_hash`` attribute, so the model leg of ``_assert_pin_engaged``
must prove engagement via the public ``cached_file`` snapshot-path resolution
(the technique the tokenizer leg already used) — a ``None``/absent/stale
``_commit_hash`` may never FAIL the leg on its own. The check must still BITE:
a resolution outside ``snapshots/<pin>`` (or a missing cached file) is
rejected, and both carve-outs (``model_revision=None`` legacy skip, ``--tiny``
from-config exemption) are preserved.

CPU-only, no network: the hub boundary (``transformers.utils.hub.cached_file``)
is faked with a ``create_autospec`` of the real callee (signature-conformant by
construction, and it asserts ``local_files_only=True`` on every call);
``_assert_pin_engaged``'s body runs real.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2329_ladder as LAD  # noqa: E402

PIN = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # the M1 pin the crash named
OTHER = "0000000000000000000000000000000000000000"
_ABSENT = "__absent__"


def _cfg(**overrides):
    """Duck-typed LadderConfig slice — only the fields the probe reads."""
    base = dict(model_id="test-org/test-model", model_revision=PIN, tiny=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _model(commit_hash=_ABSENT):
    """Fake model whose config carries (or lacks) the private _commit_hash."""
    config = SimpleNamespace()
    if commit_hash is not _ABSENT:
        config._commit_hash = commit_hash
    return SimpleNamespace(config=config)


def _snapshot_path(sha: str, filename: str) -> str:
    return f"/cache/models--test-org--test-model/snapshots/{sha}/{filename}"


def _fake_cached_file(monkeypatch, path_by_filename):
    """Install a signature-conformant hub-resolver fake; returns the call log.

    ``path_by_filename[filename]`` is the resolved path to serve (``None``
    simulates a cache miss the caller must reject). Every call is asserted to
    be ``local_files_only=True`` at the pinned revision — never a network
    fetch.
    """
    import transformers.utils.hub as hub

    requested: list[str] = []

    def _resolve(path_or_repo_id, filename, **kwargs):
        assert kwargs.get("local_files_only") is True, kwargs
        assert kwargs.get("revision") == PIN, kwargs
        requested.append(filename)
        return path_by_filename[filename]

    fake = create_autospec(hub.cached_file, side_effect=_resolve)
    monkeypatch.setattr(hub, "cached_file", fake)
    return requested


@pytest.mark.parametrize("commit_hash", [None, _ABSENT, OTHER])
def test_model_leg_passes_without_commit_hash(monkeypatch, commit_hash):
    """REGRESSION (#2329 bank crash): a correctly pinned load PASSES even when
    ``config._commit_hash`` is None (transformers 5.15.0), absent, or stale —
    engagement is proven by the resolved snapshot path, not the private attr."""
    requested = _fake_cached_file(
        monkeypatch,
        {
            "config.json": _snapshot_path(PIN, "config.json"),
            "tokenizer_config.json": _snapshot_path(PIN, "tokenizer_config.json"),
        },
    )
    LAD._assert_pin_engaged(_model(commit_hash), tok=None, cfg=_cfg())
    assert "config.json" in requested  # the model leg really probed the cache


def test_model_leg_rejects_wrong_snapshot(monkeypatch):
    """The replacement still bites: config.json resolving under a DIFFERENT
    snapshot than the pin is rejected."""
    _fake_cached_file(
        monkeypatch,
        {
            "config.json": _snapshot_path(OTHER, "config.json"),
            "tokenizer_config.json": _snapshot_path(PIN, "tokenizer_config.json"),
        },
    )
    with pytest.raises(AssertionError, match="model pin NOT engaged"):
        LAD._assert_pin_engaged(_model(None), tok=None, cfg=_cfg())


def test_model_leg_rejects_missing_cached_config(monkeypatch):
    """A cache miss (cached_file -> None) on the model leg is rejected."""
    _fake_cached_file(
        monkeypatch,
        {
            "config.json": None,
            "tokenizer_config.json": _snapshot_path(PIN, "tokenizer_config.json"),
        },
    )
    with pytest.raises(AssertionError, match="model pin NOT engaged"):
        LAD._assert_pin_engaged(_model(None), tok=None, cfg=_cfg())


def test_commit_hash_fast_path_skips_config_probe(monkeypatch):
    """A populated MATCHING _commit_hash may still PASS the leg (opportunistic
    fast path) — no config.json cache probe is made."""
    requested = _fake_cached_file(
        monkeypatch,
        {"tokenizer_config.json": _snapshot_path(PIN, "tokenizer_config.json")},
    )
    LAD._assert_pin_engaged(_model(PIN), tok=None, cfg=_cfg())
    assert requested == ["tokenizer_config.json"]


def test_legacy_no_pin_skips_everything(monkeypatch):
    """Carve-out 1 verbatim: model_revision=None -> the probe is a no-op."""
    requested = _fake_cached_file(monkeypatch, {})
    LAD._assert_pin_engaged(_model(None), tok=None, cfg=_cfg(model_revision=None))
    assert requested == []


def test_tiny_skips_model_leg_only(monkeypatch):
    """Carve-out 2 verbatim: --tiny (from-config model, never touches the hub)
    skips the model leg; the tokenizer leg still runs."""
    requested = _fake_cached_file(
        monkeypatch,
        {"tokenizer_config.json": _snapshot_path(PIN, "tokenizer_config.json")},
    )
    LAD._assert_pin_engaged(_model(None), tok=None, cfg=_cfg(tiny=True))
    assert requested == ["tokenizer_config.json"]


def test_tokenizer_leg_still_bites(monkeypatch):
    """The pre-existing tokenizer leg is unchanged: a wrong-snapshot
    resolution is rejected (model=None -> model leg not in play)."""
    _fake_cached_file(
        monkeypatch,
        {"tokenizer_config.json": _snapshot_path(OTHER, "tokenizer_config.json")},
    )
    with pytest.raises(AssertionError, match="tokenizer pin NOT engaged"):
        LAD._assert_pin_engaged(None, tok=None, cfg=_cfg())
