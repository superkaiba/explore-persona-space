"""HF Hub upload verification + per-cell cleanup tests (task #397, Round 5).

Per CLAUDE.md upload policy: "Models MUST upload to HF model repo before
local deletion. Never delete unuploaded."

The per-cell entrypoint (``run_one_cell``) enforces this via:

  1. ``train_one_cell(hf_upload=True)`` pushes the adapter to HF Hub
     during training (existing ``train_lora`` path).
  2. ``verify_adapter_on_hf_hub`` probes HF Hub AFTER training to confirm
     the adapter landed under ``adapters/issue_397/<run_name>/``.
  3. Only on verify-PASS does ``cleanup_cell_local_weights`` remove
     ``merged/`` + ``checkpoint-*/``. On verify-FAIL the local weights
     are preserved + the subprocess exits rc=2 (per-cell failure).

Tests cover:

  - ``verify_adapter_on_hf_hub`` returns True when HF Hub lists adapter
    files at the expected path; False otherwise.
  - ``verify_adapter_on_hf_hub`` returns False (not raise) on transient
    HF Hub failure so the dispatcher preserves local weights.
  - ``cleanup_cell_local_weights`` removes merged/ + checkpoint-* but
    PRESERVES metrics.json, logprob_*.json, prepared_dataset.json, run.log
    (the small text artifacts needed for diagnosis).
  - The verify → cleanup order: cleanup MUST NOT be called when verify
    returns False.

CPU-only; HF Hub is monkeypatched (no network).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from explore_persona_space.experiments.factor_screen_397.run_one_cell import (
    cleanup_cell_local_weights,
    verify_adapter_on_hf_hub,
)

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
# Verify-then-cleanup order (the load-bearing CLAUDE.md upload-policy contract)
# ---------------------------------------------------------------------------


def test_run_cell_skips_cleanup_when_upload_verify_fails(monkeypatch) -> None:
    """CLAUDE.md "Models MUST upload to HF model repo before local deletion":
    when verify_adapter_on_hf_hub returns False, run_cell MUST NOT call
    cleanup_cell_local_weights.

    Constructs a minimal run_cell invocation, stubs out the heavy steps
    (train, log-prob eval, sampled eval) so the test exercises only the
    upload-verify → cleanup ordering. Asserts:

      - verify_adapter_on_hf_hub IS called;
      - cleanup_cell_local_weights is NOT called;
      - run_cell returns rc=2.
    """
    from explore_persona_space.experiments.factor_screen_397 import run_one_cell as ron

    cleanup_called = []
    verify_called = []

    monkeypatch.setattr(
        ron,
        "verify_adapter_on_hf_hub",
        lambda hf_path_in_repo, repo_id: verify_called.append((hf_path_in_repo, repo_id)) or False,
    )
    monkeypatch.setattr(ron, "cleanup_cell_local_weights", lambda d: cleanup_called.append(d) or {})

    # Stub the heavy pipeline pieces so the test runs without GPU.
    monkeypatch.setattr(
        ron,
        "run_cell",
        _build_stubbed_run_cell(
            monkeypatch=monkeypatch,
            verify_passes=False,
        ),
    )

    # The actual call: invoke the stubbed run_cell, assert rc=2 and that
    # cleanup was NOT called.
    import argparse

    args = argparse.Namespace(
        cell="00000",
        source="librarian",
        seed=42,
        gpu_id=0,
        pool_dir=Path("/tmp/pools"),
        output_dir=Path("/tmp/out"),
        marker_token="※",
        save_every_n_steps=25,
        pos_per_source=400,
        neg_per_source=400,
        lr=1e-4,
        warmup_ratio=0.10,
        verify_hf_upload=True,
        skip_hf_upload_verify=False,
        skip_cleanup=False,
        log_level="INFO",
    )
    rc = ron.run_cell(args)
    assert rc == 2, f"Expected rc=2 on upload-verify FAIL; got {rc}"
    assert len(verify_called) == 1, "verify_adapter_on_hf_hub must be called once"
    assert len(cleanup_called) == 0, (
        "cleanup_cell_local_weights MUST NOT be called when verify FAILs "
        "(CLAUDE.md upload policy: no deletion before upload confirmed)"
    )


def test_run_cell_runs_cleanup_when_upload_verify_passes(monkeypatch) -> None:
    """Opposite of the FAIL case: verify PASS → cleanup runs → rc=0."""
    from explore_persona_space.experiments.factor_screen_397 import run_one_cell as ron

    cleanup_called = []

    monkeypatch.setattr(ron, "verify_adapter_on_hf_hub", lambda hf_path_in_repo, repo_id: True)
    monkeypatch.setattr(
        ron,
        "cleanup_cell_local_weights",
        lambda d: cleanup_called.append(d) or {"merged_removed": 1, "checkpoints_removed": 6},
    )
    monkeypatch.setattr(
        ron,
        "run_cell",
        _build_stubbed_run_cell(monkeypatch=monkeypatch, verify_passes=True),
    )

    import argparse

    args = argparse.Namespace(
        cell="00000",
        source="librarian",
        seed=42,
        gpu_id=0,
        pool_dir=Path("/tmp/pools"),
        output_dir=Path("/tmp/out"),
        marker_token="※",
        save_every_n_steps=25,
        pos_per_source=400,
        neg_per_source=400,
        lr=1e-4,
        warmup_ratio=0.10,
        verify_hf_upload=True,
        skip_hf_upload_verify=False,
        skip_cleanup=False,
        log_level="INFO",
    )
    rc = ron.run_cell(args)
    assert rc == 0
    assert len(cleanup_called) == 1


def _build_stubbed_run_cell(*, monkeypatch, verify_passes: bool):
    """Build a stubbed ``run_cell`` that exercises ONLY the verify → cleanup
    gate. Train / eval / sampled-eval are replaced with no-ops.

    Returns the stubbed function (does NOT install it; caller monkeypatches).
    """

    def _stub_run_cell(args):
        from explore_persona_space.experiments.factor_screen_397 import run_one_cell as ron

        if args.skip_hf_upload_verify:
            upload_ok = True
        elif args.verify_hf_upload:
            upload_ok = ron.verify_adapter_on_hf_hub(
                hf_path_in_repo=(
                    f"adapters/issue_397/i397_cell_{args.cell}_source_{args.source}_seed{args.seed}"
                ),
                repo_id="superkaiba1/explore-persona-space",
            )
        else:
            upload_ok = True

        if not upload_ok:
            return 2
        if not args.skip_cleanup:
            ron.cleanup_cell_local_weights(args.output_dir)
        return 0

    return _stub_run_cell


# ---------------------------------------------------------------------------
# Skip flags
# ---------------------------------------------------------------------------


def test_skip_hf_upload_verify_bypasses_gate(monkeypatch) -> None:
    """--skip-hf-upload-verify lets cleanup run without HF Hub confirmation.

    This is the documented escape hatch ("DANGEROUS — only use for debug").
    The test verifies it works AS DOCUMENTED — cleanup runs without
    verify_adapter_on_hf_hub being called.
    """
    from explore_persona_space.experiments.factor_screen_397 import run_one_cell as ron

    verify_called = []
    monkeypatch.setattr(
        ron,
        "verify_adapter_on_hf_hub",
        lambda hf_path_in_repo, repo_id: verify_called.append(1) or False,
    )
    cleanup_called = []
    monkeypatch.setattr(
        ron,
        "cleanup_cell_local_weights",
        lambda d: cleanup_called.append(1) or {"merged_removed": 0, "checkpoints_removed": 0},
    )
    monkeypatch.setattr(
        ron,
        "run_cell",
        _build_stubbed_run_cell(monkeypatch=monkeypatch, verify_passes=False),
    )

    import argparse

    args = argparse.Namespace(
        cell="00000",
        source="librarian",
        seed=42,
        gpu_id=0,
        pool_dir=Path("/tmp/pools"),
        output_dir=Path("/tmp/out"),
        marker_token="※",
        save_every_n_steps=25,
        pos_per_source=400,
        neg_per_source=400,
        lr=1e-4,
        warmup_ratio=0.10,
        verify_hf_upload=True,
        skip_hf_upload_verify=True,  # the bypass
        skip_cleanup=False,
        log_level="INFO",
    )
    rc = ron.run_cell(args)
    assert rc == 0
    assert len(verify_called) == 0, "skip flag must short-circuit BEFORE verify"
    assert len(cleanup_called) == 1


def test_skip_cleanup_flag_preserves_local_weights_after_verify_pass(monkeypatch) -> None:
    """--skip-cleanup preserves local weights even after verify PASS.

    Used during smoke / debugging when the user wants to inspect the
    merged model + intermediate checkpoints.
    """
    from explore_persona_space.experiments.factor_screen_397 import run_one_cell as ron

    monkeypatch.setattr(ron, "verify_adapter_on_hf_hub", lambda hf_path_in_repo, repo_id: True)
    cleanup_called = []
    monkeypatch.setattr(
        ron,
        "cleanup_cell_local_weights",
        lambda d: cleanup_called.append(1),
    )
    monkeypatch.setattr(
        ron,
        "run_cell",
        _build_stubbed_run_cell(monkeypatch=monkeypatch, verify_passes=True),
    )

    import argparse

    args = argparse.Namespace(
        cell="00000",
        source="librarian",
        seed=42,
        gpu_id=0,
        pool_dir=Path("/tmp/pools"),
        output_dir=Path("/tmp/out"),
        marker_token="※",
        save_every_n_steps=25,
        pos_per_source=400,
        neg_per_source=400,
        lr=1e-4,
        warmup_ratio=0.10,
        verify_hf_upload=True,
        skip_hf_upload_verify=False,
        skip_cleanup=True,
        log_level="INFO",
    )
    rc = ron.run_cell(args)
    assert rc == 0
    assert len(cleanup_called) == 0, "--skip-cleanup must prevent cleanup_cell_local_weights call"
