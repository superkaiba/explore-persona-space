"""HF Hub upload verification + per-cell cleanup tests (task #397).

Per CLAUDE.md upload policy: "Models MUST upload to HF model repo before
local deletion. Never delete unuploaded."

Round 11 lifted ``verify_adapter_on_hf_hub`` and ``cleanup_cell_local_
weights`` from the deleted ``run_one_cell.py`` module into the dispatcher
(``scripts/dispatch_factor_screen_397.py``). Round 12's two-pass design
still uses them: ``_run_pass2_vllm`` calls verify then (only on PASS)
cleanup. The contract is unchanged: cleanup must NOT run unless verify
returns True.

Round 5..10's ``run_cell`` stub tests (which simulated the now-deleted
subprocess wrapper's verify→cleanup ordering) are gone. Round 11's
``_run_one_cell_inprocess`` source-order tests are also gone (the
function was deleted in Round 12). The two-pass pipeline's
verify→cleanup ordering is covered by:

  - the source-order tests in
    ``test_factor_screen_397_two_pass_sweep.py`` that AST-scan
    ``_run_pass1_hf`` and ``_run_pass2_vllm`` for the canonical order,
  - the per-pass behavioural tests in
    ``test_factor_screen_397_two_pass_sweep.py`` that monkeypatch the
    verify + cleanup helpers and assert the rc=2-skips-cleanup contract.

This file is the unit-test surface for the two lifted helpers
themselves (independent of which orchestrator calls them).

Tests cover:

  - ``verify_adapter_on_hf_hub`` returns True when HF Hub lists adapter
    files at the expected path; False otherwise.
  - ``verify_adapter_on_hf_hub`` returns False (not raise) on transient
    HF Hub failure so the dispatcher preserves local weights.
  - ``cleanup_cell_local_weights`` removes merged/ + checkpoint-* but
    PRESERVES metrics.json, logprob_*.json, prepared_dataset.json, run.log
    (the small text artifacts needed for diagnosis).

CPU-only; HF Hub is monkeypatched (no network).
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Load the dispatcher as a module (not a package; lives under scripts/).
_DISPATCH_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "dispatch_factor_screen_397.py"
)
_spec = importlib.util.spec_from_file_location("dispatch_factor_screen_397", _DISPATCH_PATH)
_dispatch = importlib.util.module_from_spec(_spec)
sys.modules["dispatch_factor_screen_397"] = _dispatch
_spec.loader.exec_module(_dispatch)

# Round 11: verify + cleanup are exposed as top-level dispatcher helpers
# (lifted from the deleted run_one_cell.py). Round 12 still uses them.
cleanup_cell_local_weights = _dispatch.cleanup_cell_local_weights
verify_adapter_on_hf_hub = _dispatch.verify_adapter_on_hf_hub

# ---------------------------------------------------------------------------
# verify_adapter_on_hf_hub
# ---------------------------------------------------------------------------


def test_verify_adapter_on_hf_hub_returns_true_when_files_present(monkeypatch) -> None:
    """When HF Hub list_repo_files returns adapter_*.safetensors at the
    expected path, verification PASSes.
    """
    import huggingface_hub

    fake_files = [
        "adapters/issue_397/i397_cell_10010_source_librarian_seed42/adapter_model.safetensors",
        "adapters/issue_397/i397_cell_10010_source_librarian_seed42/adapter_config.json",
        "adapters/other_issue/whatever/foo.txt",
    ]
    fake_api = MagicMock()
    fake_api.list_repo_files.return_value = fake_files
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: fake_api)

    ok = verify_adapter_on_hf_hub(
        hf_path_in_repo="adapters/issue_397/i397_cell_10010_source_librarian_seed42",
        repo_id="superkaiba1/explore-persona-space",
    )
    assert ok is True
    fake_api.list_repo_files.assert_called_once_with(
        repo_id="superkaiba1/explore-persona-space", repo_type="model"
    )


def test_verify_adapter_on_hf_hub_returns_false_when_no_files_at_path(monkeypatch) -> None:
    """When the expected path has NO adapter files, verification FAILs.

    This is the "upload silently dropped" case — train_lora's wrapping
    `except Exception` in sft.py logs the failure but doesn't raise.
    The dispatcher catches it here BEFORE cleanup runs.
    """
    import huggingface_hub

    fake_api = MagicMock()
    fake_api.list_repo_files.return_value = ["adapters/other/whatever.txt"]
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: fake_api)

    ok = verify_adapter_on_hf_hub(
        hf_path_in_repo="adapters/issue_397/i397_cell_10010_source_librarian_seed42",
        repo_id="superkaiba1/explore-persona-space",
    )
    assert ok is False


def test_verify_adapter_on_hf_hub_returns_false_on_transient_hub_failure(monkeypatch) -> None:
    """list_repo_files raising must return False (not propagate).

    Returning False on transient failure preserves local weights so the
    user can manually re-upload + rerun verification. Re-raising would
    crash the per-cell subprocess BEFORE cleanup would run, which is
    also safe, but a False-return lets the cell exit cleanly with rc=2
    so the dispatcher summary records the failure cleanly.
    """
    import huggingface_hub

    fake_api = MagicMock()
    fake_api.list_repo_files.side_effect = ConnectionError("network down")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: fake_api)

    ok = verify_adapter_on_hf_hub(
        hf_path_in_repo="adapters/issue_397/cell_x",
        repo_id="superkaiba1/explore-persona-space",
    )
    assert ok is False


def test_verify_adapter_only_matches_adapter_prefix_files(monkeypatch) -> None:
    """A file like ``readme.md`` at the path does NOT count as adapter present.

    We specifically look for ``adapter_*`` files (adapter_model.safetensors,
    adapter_config.json) so that an empty / readme-only directory doesn't
    falsely PASS verification.
    """
    import huggingface_hub

    fake_api = MagicMock()
    fake_api.list_repo_files.return_value = [
        "adapters/issue_397/cell_x/readme.md",
        "adapters/issue_397/cell_x/.gitattributes",
    ]
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: fake_api)

    ok = verify_adapter_on_hf_hub(
        hf_path_in_repo="adapters/issue_397/cell_x",
        repo_id="superkaiba1/explore-persona-space",
    )
    assert ok is False


# ---------------------------------------------------------------------------
# cleanup_cell_local_weights
# ---------------------------------------------------------------------------


def test_cleanup_removes_merged_and_checkpoints() -> None:
    """merged/ + checkpoint-25/ + checkpoint-50/ all removed."""
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = Path(tmp) / "cell"
        adapter_dir = cell_dir / "adapter"
        adapter_dir.mkdir(parents=True)
        for step in (25, 50, 75):
            (adapter_dir / f"checkpoint-{step}").mkdir()
            (adapter_dir / f"checkpoint-{step}" / "model.bin").write_text("fake weights")
        merged_dir = cell_dir / "merged"
        merged_dir.mkdir()
        (merged_dir / "model.safetensors").write_text("fake merged weights")

        removed = cleanup_cell_local_weights(cell_dir)
        assert removed == {"merged_removed": 1, "checkpoints_removed": 3}
        assert not merged_dir.exists()
        for step in (25, 50, 75):
            assert not (adapter_dir / f"checkpoint-{step}").exists()


def test_cleanup_preserves_small_text_artifacts() -> None:
    """metrics.json / logprob_panel.json / prepared_dataset.json / run.log
    survive cleanup — they're load-bearing for diagnosis + the analyzer.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = Path(tmp) / "cell"
        cell_dir.mkdir()
        (cell_dir / "metrics.json").write_text('{"source": "librarian"}')
        (cell_dir / "logprob_panel.json").write_text("{}")
        (cell_dir / "prepared_dataset.json").write_text('{"system_prompt_text": "x"}')
        (cell_dir / "run.log").write_text("log line")
        (cell_dir / "merged").mkdir()
        (cell_dir / "merged" / "weights.bin").write_text("weights")

        cleanup_cell_local_weights(cell_dir)

        # All four small artifacts preserved.
        assert (cell_dir / "metrics.json").exists()
        assert (cell_dir / "logprob_panel.json").exists()
        assert (cell_dir / "prepared_dataset.json").exists()
        assert (cell_dir / "run.log").exists()
        # Merged removed.
        assert not (cell_dir / "merged").exists()


def test_cleanup_handles_empty_cell_dir_cleanly() -> None:
    """No merged/ + no adapter/ → no-op without raising."""
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = Path(tmp) / "empty_cell"
        cell_dir.mkdir()
        removed = cleanup_cell_local_weights(cell_dir)
        assert removed == {"merged_removed": 0, "checkpoints_removed": 0}


def test_cleanup_keeps_adapter_dir_when_only_checkpoints_removed() -> None:
    """``adapter/`` itself is NOT removed (it may carry adapter_config.json
    from the final-step checkpoint that's needed for offline inference).
    Only the per-step checkpoint-* subdirs go.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cell_dir = Path(tmp) / "cell"
        adapter_dir = cell_dir / "adapter"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapter_config.json").write_text("{}")  # final-step config
        (adapter_dir / "checkpoint-25").mkdir()
        (adapter_dir / "checkpoint-25" / "model.bin").write_text("step weights")

        cleanup_cell_local_weights(cell_dir)
        assert adapter_dir.exists()
        assert (adapter_dir / "adapter_config.json").exists()
        assert not (adapter_dir / "checkpoint-25").exists()
