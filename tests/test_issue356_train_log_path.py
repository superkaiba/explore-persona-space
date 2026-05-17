"""Smoke test for the per-cell train-log dump path (cr round 2 blocker 3).

Round-1 code review blocker 3: the trainer wrote ``train_log.json`` to
``<EPM_TRAIN_LOG_DUMP_DIR>/<merged_dir.name>/train_log.json``, but
``merged_dir.name`` was the constant ``coupling_merged`` for all 12 #356
cells — so each cell overwrote the previous one and only the last cell's
log survived.

The fix introduces ``EPM_TRAIN_LOG_CELL_ID`` (env-driven, set per cell by the
orchestrator) and routes the dump path through it, falling back to
``merged_dir.name`` only when the env var is unset.

These tests confirm:
1. With EPM_TRAIN_LOG_CELL_ID set, dump path uses the env var.
2. Without EPM_TRAIN_LOG_CELL_ID, dump path falls back to merged_dir.name.
3. EPM_TRAIN_LOG_DUMP_DIR unset => no dump (silent opt-in preserved).
4. Eval-side ``_issue356_cell_id`` agrees with the env var the orchestrator
   would set, so the train-side dump and eval-side read line up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _fake_trainer() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(
            log_history=[{"loss": 1.5, "step": 1}, {"loss": 0.8, "step": 2}],
            global_step=2,
            epoch=1.0,
        )
    )


def test_train_log_uses_cell_id_env(tmp_path: Path, monkeypatch) -> None:
    """When EPM_TRAIN_LOG_CELL_ID is set, dump path uses it (not merged_dir.name)."""
    from explore_persona_space.train.trainer import _maybe_dump_train_log

    monkeypatch.setenv("EPM_TRAIN_LOG_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv(
        "EPM_TRAIN_LOG_CELL_ID", "i356_librarian_consistent_persona_cot_seed42_post_em"
    )

    # merged_dir.name is the constant "coupling_merged" — without the env var
    # we would clobber every cell to this path.
    merged_dir = Path("/tmp/some/path/coupling_merged")
    _maybe_dump_train_log(_fake_trainer(), merged_dir)

    # The env var controls the directory, NOT merged_dir.name.
    cell_dir = tmp_path / "i356_librarian_consistent_persona_cot_seed42_post_em"
    assert cell_dir.exists()
    payload = json.loads((cell_dir / "train_log.json").read_text())
    assert payload["cell_id"] == "i356_librarian_consistent_persona_cot_seed42_post_em"
    # merged_dir_name preserved for debugging.
    assert payload["merged_dir_name"] == "coupling_merged"
    # The OLD (buggy) path must NOT exist.
    assert not (tmp_path / "coupling_merged").exists()


def test_train_log_falls_back_to_merged_dir_name(tmp_path: Path, monkeypatch) -> None:
    """Without EPM_TRAIN_LOG_CELL_ID, dump path uses merged_dir.name (legacy behavior).

    Other experiments not using the per-cell env override still work.
    """
    from explore_persona_space.train.trainer import _maybe_dump_train_log

    monkeypatch.setenv("EPM_TRAIN_LOG_DUMP_DIR", str(tmp_path))
    monkeypatch.delenv("EPM_TRAIN_LOG_CELL_ID", raising=False)

    merged_dir = Path("/tmp/some/path/my_unique_cell_dir")
    _maybe_dump_train_log(_fake_trainer(), merged_dir)

    assert (tmp_path / "my_unique_cell_dir" / "train_log.json").exists()


def test_train_log_dump_dir_unset_is_silent(tmp_path: Path, monkeypatch) -> None:
    """Without EPM_TRAIN_LOG_DUMP_DIR, no dump happens (silent opt-in)."""
    from explore_persona_space.train.trainer import _maybe_dump_train_log

    monkeypatch.delenv("EPM_TRAIN_LOG_DUMP_DIR", raising=False)
    monkeypatch.setenv("EPM_TRAIN_LOG_CELL_ID", "anything")  # ignored when dump dir unset

    _maybe_dump_train_log(_fake_trainer(), Path("/tmp/foo/bar"))
    # No file created anywhere — we just observe no exception is raised.
    # (Behaviour: function returns early.)


def test_eval_reader_cell_id_matches_trainer_env() -> None:
    """The eval-side ``_issue356_cell_id`` agrees with the env-var convention.

    The orchestrator MUST set EPM_TRAIN_LOG_CELL_ID to the same string the
    eval reader expects. This test pins the contract so a future drift in
    one side will surface here.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_issue356_eval", PROJECT_ROOT / "scripts" / "run_issue356_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Some imports inside the script require src on path — already done above.
    spec.loader.exec_module(mod)

    expected = "i356_librarian_consistent_persona_cot_seed42_post_em"
    assert mod._issue356_cell_id("librarian", 42) == expected
    assert mod._issue356_cell_id("software_engineer", 137) == (
        "i356_software_engineer_consistent_persona_cot_seed137_post_em"
    )
