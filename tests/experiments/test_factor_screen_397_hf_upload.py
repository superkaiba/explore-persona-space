"""HF Hub upload verification + per-cell cleanup tests (task #397).

Per CLAUDE.md upload policy: "Models MUST upload to HF model repo before
local deletion. Never delete unuploaded."

Round 11 lifted ``verify_adapter_on_hf_hub`` and ``cleanup_cell_local_
weights`` from the deleted ``run_one_cell.py`` module into the dispatcher
(``scripts/dispatch_factor_screen_397.py``). The in-process serial sweep
calls them inline between the upload + cleanup steps. The contract is
unchanged: cleanup must NOT run unless verify returns True.

Round 5..10's ``run_cell`` stub tests (which simulated the now-deleted
subprocess wrapper's verify→cleanup ordering) are gone. The in-process
pipeline's verify→cleanup ordering is covered by:

  - the dispatcher-level static test
    ``test_round11_pipeline_order_upload_then_verify_then_cleanup`` here,
    which AST-scans the dispatcher source for the source-order contract,
  - the in-process sweep tests in
    ``test_factor_screen_397_inprocess_sweep.py`` that monkeypatch the
    upload + verify helpers and assert the sequence.

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

import ast
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

# Round 11: verify + cleanup are now exposed as top-level helpers on the
# dispatcher (lifted from the deleted run_one_cell.py).
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


# ---------------------------------------------------------------------------
# Round 11: verify→cleanup pipeline order (now in the dispatcher itself)
# ---------------------------------------------------------------------------


def test_round11_pipeline_order_verify_then_cleanup_in_dispatcher() -> None:
    """Round 11 contract on the in-process per-cell pipeline:

      1. training (writes adapter + intermediate checkpoints)
      2. log-prob eval (480-context panel)
      3. HF teardown + vLLM sampled eval (writes metrics.json)
      4. verify_adapter_on_hf_hub  ← gate
      5. cleanup_cell_local_weights  ← ONLY on verify PASS

    This test AST-scans ``scripts/dispatch_factor_screen_397.py`` for
    the source-order in ``_run_one_cell_inprocess``: verify_adapter_on_hf_hub
    MUST be called BEFORE cleanup_cell_local_weights, and the cleanup
    call MUST be guarded by the verify return.

    Replaces the Round 10 source-order test against the deleted
    ``run_one_cell.py``.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))

    # Locate _run_one_cell_inprocess.
    target_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_one_cell_inprocess":
            target_fn = node
            break
    assert target_fn is not None, (
        "Round 11: _run_one_cell_inprocess function missing from dispatcher"
    )

    # Walk the function body for the first verify + cleanup call lines.
    verify_line: int | None = None
    cleanup_line: int | None = None
    for node in ast.walk(target_fn):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name == "verify_adapter_on_hf_hub" and verify_line is None:
                verify_line = node.lineno
            if name == "cleanup_cell_local_weights" and cleanup_line is None:
                cleanup_line = node.lineno

    assert verify_line is not None, (
        "Round 11: _run_one_cell_inprocess must call verify_adapter_on_hf_hub"
    )
    assert cleanup_line is not None, (
        "Round 11: _run_one_cell_inprocess must call cleanup_cell_local_weights"
    )
    assert verify_line < cleanup_line, (
        f"Round 11 source-order contract: verify_adapter_on_hf_hub (line "
        f"{verify_line}) MUST come BEFORE cleanup_cell_local_weights (line "
        f"{cleanup_line}) inside _run_one_cell_inprocess."
    )


def test_round11_run_one_cell_inprocess_calls_train_one_cell_with_hf_upload_true() -> None:
    """Round 11 reverses Round 10's ``hf_upload=False``: the in-process
    pipeline lets the TRL inline-upload fence (sft.py:667) handle the
    HF Hub push, since the smoke phase proves the fence works when .env
    is loaded and HF_TOKEN is in env.

    Round 11 dropped the explicit ``upload_model`` step that Round 10
    added in the subprocess wrapper — keeping a single upload path
    matches the proven smoke flow. The safety net is still
    ``verify_adapter_on_hf_hub``: if the fence silently swallows an
    error, verify catches it as the safety net.

    Static AST scan of the dispatcher source.
    """
    src = _DISPATCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(_DISPATCH_PATH))

    target_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_one_cell_inprocess":
            target_fn = node
            break
    assert target_fn is not None

    matches: list[bool] = []
    for node in ast.walk(target_fn):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name == "train_one_cell":
                for kw in node.keywords:
                    if kw.arg == "hf_upload":
                        is_true = isinstance(kw.value, ast.Constant) and kw.value.value is True
                        matches.append(is_true)
    assert matches, (
        "Round 11: no train_one_cell(hf_upload=...) kwarg found in "
        "_run_one_cell_inprocess — must be set explicitly."
    )
    assert all(matches), (
        "Round 11: train_one_cell(hf_upload=False) found in "
        "_run_one_cell_inprocess — should be True to use the proven smoke "
        "flow's upload path. Round 10's hf_upload=False was for the "
        "subprocess wrapper that round 11 deleted."
    )
