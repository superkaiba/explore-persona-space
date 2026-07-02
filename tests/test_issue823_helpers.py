"""Tests for issue-823 helper functions in run_823.py.

Covers:
  - _ensure_repo_root_on_syspath: inserts repo root exactly once; raises on bad tree.
  - _phase1_outputs_exist: rejects absent / empty / corrupt / truncated files;
    accepts valid JSON lists + valid common_valid_idx structure.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Helpers — locate run_823 module without importing it at collection time
# (the module has GPU-bound top-level imports we don't want at test time).
# ---------------------------------------------------------------------------

_WORKTREE = pathlib.Path(__file__).resolve().parents[1]
_RUN823 = _WORKTREE / "src" / "explore_persona_space" / "experiments" / "issue_823" / "run_823.py"


def _import_run823():
    """Import run_823 as a module, inserting its src/ parent onto sys.path."""
    src_dir = str(_WORKTREE / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_823", str(_RUN823))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Tests for _ensure_repo_root_on_syspath
# ---------------------------------------------------------------------------


class TestEnsureRepoRootOnSyspath:
    """_ensure_repo_root_on_syspath must insert repo root idempotently and raise on bad tree."""

    def test_inserts_repo_root(self, tmp_path, monkeypatch):
        """When the sentinel exists, repo root is inserted into sys.path."""
        mod = _import_run823()

        # Build a fake tree that mirrors the real layout seen by __file__:
        #   <repo_root>/scripts/issue779_collect.py  (sentinel)
        #   <repo_root>/src/explore_persona_space/experiments/issue_823/run_823.py
        fake_repo = tmp_path / "fake_repo"
        fake_scripts = fake_repo / "scripts"
        fake_scripts.mkdir(parents=True)
        (fake_scripts / "issue779_collect.py").write_text("# sentinel")

        fake_run823 = (
            fake_repo / "src" / "explore_persona_space" / "experiments" / "issue_823" / "run_823.py"
        )
        fake_run823.parent.mkdir(parents=True)
        fake_run823.write_text("# placeholder")

        # Monkeypatch __file__ inside the module so parents[4] resolves to fake_repo.
        monkeypatch.setattr(mod, "__file__", str(fake_run823))

        # Remove fake_repo from sys.path if it's already there (idempotency pre-condition).
        original_path = sys.path.copy()
        sys.path[:] = [p for p in sys.path if p != str(fake_repo)]
        try:
            mod._ensure_repo_root_on_syspath()
            assert str(fake_repo) in sys.path, "Expected repo root in sys.path after call"
        finally:
            sys.path[:] = original_path

    def test_idempotent(self, tmp_path, monkeypatch):
        """Calling twice does not insert a duplicate entry."""
        mod = _import_run823()

        fake_repo = tmp_path / "fake_repo2"
        fake_scripts = fake_repo / "scripts"
        fake_scripts.mkdir(parents=True)
        (fake_scripts / "issue779_collect.py").write_text("# sentinel")

        fake_run823 = (
            fake_repo / "src" / "explore_persona_space" / "experiments" / "issue_823" / "run_823.py"
        )
        fake_run823.parent.mkdir(parents=True)
        fake_run823.write_text("# placeholder")

        monkeypatch.setattr(mod, "__file__", str(fake_run823))

        original_path = sys.path.copy()
        sys.path[:] = [p for p in sys.path if p != str(fake_repo)]
        try:
            mod._ensure_repo_root_on_syspath()
            mod._ensure_repo_root_on_syspath()
            count = sys.path.count(str(fake_repo))
            assert count == 1, f"Expected exactly 1 occurrence, got {count}"
        finally:
            sys.path[:] = original_path

    def test_raises_on_missing_sentinel(self, tmp_path, monkeypatch):
        """RuntimeError is raised when the sentinel file is absent (bad repo tree)."""
        mod = _import_run823()

        # No scripts/issue779_collect.py created — sentinel missing.
        fake_repo = tmp_path / "fake_repo_no_sentinel"
        fake_run823 = (
            fake_repo / "src" / "explore_persona_space" / "experiments" / "issue_823" / "run_823.py"
        )
        fake_run823.parent.mkdir(parents=True)
        fake_run823.write_text("# placeholder")

        monkeypatch.setattr(mod, "__file__", str(fake_run823))

        with pytest.raises(RuntimeError, match="sentinel"):
            mod._ensure_repo_root_on_syspath()


# ---------------------------------------------------------------------------
# Tests for _phase1_outputs_exist
# ---------------------------------------------------------------------------


def _write_phase1(p1_dir: pathlib.Path, b2=None, b1=None, idx=None) -> None:
    """Write phase1 output files into p1_dir for test setup."""
    p1_dir.mkdir(parents=True, exist_ok=True)
    if b2 is not None:
        (p1_dir / "b2_seed42.json").write_text(b2)
    if b1 is not None:
        (p1_dir / "b1_seed43.json").write_text(b1)
    if idx is not None:
        (p1_dir / "common_valid_idx.json").write_text(idx)


_VALID_B2 = json.dumps([{"answer_text": "hello", "context_idx": 0}])
_VALID_B1 = json.dumps([{"answer_text": "world", "context_idx": 0}])
_VALID_IDX = json.dumps({"common_valid_idx": [0, 1, 2]})


class TestPhase1OutputsExist:
    """_phase1_outputs_exist must return False on absent/corrupt/truncated files."""

    def test_all_valid_returns_true(self, tmp_path):
        mod = _import_run823()
        _write_phase1(
            tmp_path / "raw_completions" / "phase1",
            b2=_VALID_B2,
            b1=_VALID_B1,
            idx=_VALID_IDX,
        )
        assert mod._phase1_outputs_exist(tmp_path) is True

    def test_missing_b2_returns_false(self, tmp_path):
        mod = _import_run823()
        _write_phase1(
            tmp_path / "raw_completions" / "phase1",
            b2=None,
            b1=_VALID_B1,
            idx=_VALID_IDX,
        )
        assert mod._phase1_outputs_exist(tmp_path) is False

    def test_empty_b1_returns_false(self, tmp_path):
        mod = _import_run823()
        _write_phase1(
            tmp_path / "raw_completions" / "phase1",
            b2=_VALID_B2,
            b1="",
            idx=_VALID_IDX,
        )
        assert mod._phase1_outputs_exist(tmp_path) is False

    def test_corrupt_json_b2_returns_false(self, tmp_path):
        mod = _import_run823()
        _write_phase1(
            tmp_path / "raw_completions" / "phase1",
            b2="{not valid json",
            b1=_VALID_B1,
            idx=_VALID_IDX,
        )
        assert mod._phase1_outputs_exist(tmp_path) is False

    def test_empty_list_b2_returns_false(self, tmp_path):
        """A zero-length JSON list (truncated run) must return False."""
        mod = _import_run823()
        _write_phase1(
            tmp_path / "raw_completions" / "phase1",
            b2="[]",
            b1=_VALID_B1,
            idx=_VALID_IDX,
        )
        assert mod._phase1_outputs_exist(tmp_path) is False

    def test_missing_common_valid_key_returns_false(self, tmp_path):
        """common_valid_idx.json without the required key is treated as absent."""
        mod = _import_run823()
        _write_phase1(
            tmp_path / "raw_completions" / "phase1",
            b2=_VALID_B2,
            b1=_VALID_B1,
            idx=json.dumps({"wrong_key": [0, 1]}),
        )
        assert mod._phase1_outputs_exist(tmp_path) is False

    def test_empty_common_valid_idx_list_returns_false(self, tmp_path):
        """common_valid_idx with an empty list is treated as absent."""
        mod = _import_run823()
        _write_phase1(
            tmp_path / "raw_completions" / "phase1",
            b2=_VALID_B2,
            b1=_VALID_B1,
            idx=json.dumps({"common_valid_idx": []}),
        )
        assert mod._phase1_outputs_exist(tmp_path) is False

    def test_nonlist_b2_returns_false(self, tmp_path):
        """b2_seed42.json containing a dict instead of a list returns False."""
        mod = _import_run823()
        _write_phase1(
            tmp_path / "raw_completions" / "phase1",
            b2=json.dumps({"not": "a list"}),
            b1=_VALID_B1,
            idx=_VALID_IDX,
        )
        assert mod._phase1_outputs_exist(tmp_path) is False
