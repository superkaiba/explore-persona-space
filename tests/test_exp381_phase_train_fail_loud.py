"""Unit tests for the exp381 anchor sub-epoch checkpoint enumeration.

Verifies Codex r2 Major #3 (round-3 fix): when training produces zero
``checkpoint-*`` directories, ``_enumerate_and_upload_anchor_ckpts`` MUST
raise immediately rather than write ``sub_epoch_checkpoints: []`` and let
the failure surface much later during full-eval cell enumeration. A silent
"zero ckpts" result wastes Phase 4 GPU-hours; the H1 trajectory analysis
needs at least one saved checkpoint per seed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_exp381():
    if "exp381" in sys.modules:
        return sys.modules["exp381"]
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("exp381", scripts_dir / "run_experiment_381.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp381"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_empty_candidate_dirs_raises(tmp_path: Path) -> None:
    """Codex r2 Major #3: zero ``checkpoint-*`` dirs MUST raise immediately."""
    m = _load_exp381()
    # Create an empty out_dir with no checkpoint-* subdirs.
    out_dir = tmp_path / "anchor_seed42_train"
    out_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError) as excinfo:
        m._enumerate_and_upload_anchor_ckpts(
            out_dir=out_dir,
            seed=42,
            save_strategy="steps",
            save_steps=5,
        )
    # The error message must name the failure mode + give debug context.
    msg = str(excinfo.value)
    assert "zero checkpoint-* dirs" in msg, msg
    assert "seed=42" in msg, msg
    assert "save_steps=5" in msg, msg


def test_missing_out_dir_raises(tmp_path: Path) -> None:
    """Even if out_dir itself doesn't exist (trainer never ran), raise loud."""
    m = _load_exp381()
    out_dir = tmp_path / "never_created"
    # NOTE: do NOT mkdir; the glob returns [] and the listing fallback
    # picks up the "out_dir does not exist" sentinel.
    with pytest.raises(RuntimeError) as excinfo:
        m._enumerate_and_upload_anchor_ckpts(
            out_dir=out_dir,
            seed=137,
            save_strategy="steps",
            save_steps=5,
        )
    msg = str(excinfo.value)
    assert "zero checkpoint-* dirs" in msg, msg
    assert "seed=137" in msg, msg


def test_checkpoint_dir_without_adapter_config_raises(tmp_path: Path, monkeypatch) -> None:
    """If a ``checkpoint-N`` dir exists but is missing ``adapter_config.json``,
    the upload completeness gate must raise (trainer save crash / file reap)."""
    m = _load_exp381()
    out_dir = tmp_path / "anchor_seed256_train"
    ckpt_dir = out_dir / "checkpoint-5"
    ckpt_dir.mkdir(parents=True)
    # Intentionally NO adapter_config.json — simulates a crashed save.
    # Monkey-patch upload_model to never be called; if it IS called, fail the
    # test (the missing adapter_config.json path skips the upload).
    import explore_persona_space.orchestrate.hub as hub_mod

    def _fake_upload(*args, **kwargs):
        pytest.fail("upload_model should not be called for a dir missing adapter_config.json")

    monkeypatch.setattr(hub_mod, "upload_model", _fake_upload)
    with pytest.raises(RuntimeError) as excinfo:
        m._enumerate_and_upload_anchor_ckpts(
            out_dir=out_dir,
            seed=256,
            save_strategy="steps",
            save_steps=5,
        )
    msg = str(excinfo.value)
    assert "anchor sub-epoch checkpoint upload incomplete" in msg, msg
    assert "adapter_config.json" in msg, msg
