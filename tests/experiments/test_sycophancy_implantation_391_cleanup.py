"""Regression: per-cell merged-dir cleanup must fire on eval FAILURE, not only success.

Pre-fix (task #391 first run): ``_run_cell_mode`` ordered the cell as
``prepare → train → eval → upload-adapter → rmtree(merged)``. If the eval
subprocess raised (``RuntimeError`` from a non-zero exit), control never
reached the upload or the rmtree, and the ~15 GB merged dir stayed on disk.
At 4-cell concurrency on the ~130 GB MooseFS per-pod quota, 11 cells whose
evals failed leaked 165 GB and tripped EDQUOT, blocking every subsequent
training cell from writing a checkpoint.

Fix: reorder to ``prepare → train → upload-adapter → try{ eval }
finally{ rmtree(merged) IFF upload succeeded }``. The HF Hub adapter is the
cloud-copy invariant — once uploaded, the merged dir is fully re-derivable
from ``base + adapter`` and is safe to delete on either eval-success or
eval-failure. If the upload itself fails, the merged dir is preserved so the
trained weights survive on local disk.

These tests pin the four observable behaviors:

  * eval success + upload success → merged dir deleted; metrics.json written
    with ``hf_adapter_path``.
  * eval FAILURE + upload success → merged dir STILL deleted; eval exception
    re-raises; metrics.json NOT written (failure path is handled by
    ``main()``'s top-level ``sycophancy_failed.json`` writer).
  * eval success + upload FAILURE → upload exception re-raises; merged dir
    PRESERVED (the trained adapter survives only on local disk).
  * ``--no-cleanup-merged`` → merged dir preserved regardless of outcome.

Plus a direct unit test on the ``_cleanup_merged_if_safe`` helper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock

import pytest

from explore_persona_space.experiments.factor_screen_365.cells import Cell
from explore_persona_space.experiments.factor_screen_365.data_prep import PreparedDataset
from explore_persona_space.experiments.factor_screen_365.training import TrainOutcome
from explore_persona_space.experiments.sycophancy_implantation_391 import __main__ as syco_main

# ---------------------------------------------------------------------------
# Fixtures: build cheap stand-ins for `prepare_cell` / `train_one_cell` output
# and a minimal argparse.Namespace covering every attribute `_run_cell_mode`
# touches. Heavy ML deps (transformers, peft, vllm) are NEVER imported here.
# ---------------------------------------------------------------------------


def _make_outcome(*, adapter_path: Path, merged_path: Path) -> TrainOutcome:
    return TrainOutcome(
        cell_key="10011",
        seed=42,
        adapter_path=str(adapter_path),
        merged_path=str(merged_path),
        loss=0.5,
        train_wall_minutes=10.0,
        n_examples=800,
        total_steps=150,
        marker_only_loss=False,
    )


def _make_prepared(*, output_dir: Path) -> PreparedDataset:
    # PreparedDataset has many fields, but `_write_success_metrics` only reads
    # a small subset and tolerates None / placeholder values for the rest.
    return PreparedDataset(
        path=output_dir / "train.jsonl",
        num_positive=400,
        num_negative=400,
        num_total=800,
        data_policy="off_policy",
        system_prompt_text="You are a librarian.",
        system_prompt_token_count=8,
        marker_position_mean_tokens=0.0,
        marker_position_sd_tokens=0.0,
        total_seq_length_mean_tokens=512.0,
        total_seq_length_sd_tokens=64.0,
        rendered_qwen_token_count=None,
        caveats=[],
        manifest_path=None,
        preflight=None,
    )


def _make_args(
    *,
    output_dir: Path,
    pool_dir: Path,
    eval_script: Path,
    upload_adapter: bool = True,
    cleanup_merged: bool = True,
) -> argparse.Namespace:
    return argparse.Namespace(
        cell="10011",
        source="librarian",
        seed=42,
        output_dir=str(output_dir),
        pool_dir=str(pool_dir),
        base_model="Qwen/Qwen2.5-7B-Instruct",
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        lr=1e-5,
        epochs=3,
        batch_size=4,
        grad_accum=4,
        max_length=2048,
        pos_per_source=400,
        neg_per_source=400,
        num_eval_rollouts=20,
        num_eval_gpus=2,
        eval_script=str(eval_script),
        eval_personas="librarian",  # one-persona subset keeps the test cheap
        training_persona=None,
        scenarios_out_file=None,
        wandb_project=None,
        progress_url=None,
        progress_token=None,
        run_index=0,
        resume=False,
        upload_adapter=upload_adapter,
        cleanup_merged=cleanup_merged,
    )


@pytest.fixture
def cell_env(tmp_path):
    """Stand up the on-disk layout `_run_cell_mode` expects.

    Returns a dict with: ``args``, ``output_dir``, ``pool_dir``, ``eval_script``,
    ``merged_path``, ``adapter_path``, ``scenarios_out_file``.
    """
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    pool_dir = tmp_path / "pools"
    (pool_dir / "librarian").mkdir(parents=True)
    scenarios_out_file = pool_dir / "librarian" / "scenarios_multiturn_out.json"
    scenarios_out_file.write_text(json.dumps({"scenarios": []}))
    eval_script = tmp_path / "run_sycophancy_eval_persona.py"
    eval_script.write_text("# stub\n")

    merged_path = output_dir / "merged"
    merged_path.mkdir()
    (merged_path / "config.json").write_text("{}")
    (merged_path / "model.safetensors").write_bytes(b"\x00" * 64)

    adapter_path = output_dir / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text("{}")

    args = _make_args(
        output_dir=output_dir,
        pool_dir=pool_dir,
        eval_script=eval_script,
    )

    return {
        "args": args,
        "output_dir": output_dir,
        "pool_dir": pool_dir,
        "eval_script": eval_script,
        "merged_path": merged_path,
        "adapter_path": adapter_path,
        "scenarios_out_file": scenarios_out_file,
    }


def _patch_run_cell(
    monkeypatch, *, env, eval_raises: Exception | None, upload_raises: Exception | None
):
    """Patch `_prepare_and_train_cell`, `_upload_adapter_to_hub`, and
    `_run_sycophancy_eval_subprocess` for one `_run_cell_mode` invocation.
    Returns a `mock.Mock` proxy used to assert call ordering / counts.
    """
    outcome = _make_outcome(adapter_path=env["adapter_path"], merged_path=env["merged_path"])
    prepared = _make_prepared(output_dir=env["output_dir"])

    monkeypatch.setattr(syco_main, "_prepare_and_train_cell", lambda **_: (outcome, prepared))

    upload_calls = mock.Mock(
        return_value="https://huggingface.co/superkaiba1/explore-persona-space/tree/main/adapters/issue_391/i391_cell_10011_source_librarian_seed42"
    )
    if upload_raises is not None:
        upload_calls.side_effect = upload_raises
    monkeypatch.setattr(syco_main, "_upload_adapter_to_hub", upload_calls)

    eval_calls = mock.Mock(return_value=None)
    if eval_raises is not None:
        eval_calls.side_effect = eval_raises
    monkeypatch.setattr(syco_main, "_run_sycophancy_eval_subprocess", eval_calls)

    # Silence the progress milestones — they hit a no-op stub in tests, but
    # the absence of the configure() call would warn.
    monkeypatch.setattr(syco_main.progress, "post_milestone", lambda *_a, **_kw: None)

    return upload_calls, eval_calls


# ---------------------------------------------------------------------------
# Behavior tests on `_run_cell_mode` end-to-end with mocked train+eval+upload.
# ---------------------------------------------------------------------------


def test_eval_failure_still_deletes_merged_when_upload_succeeded(monkeypatch, cell_env):
    """The EDQUOT regression. Pre-fix, an eval RuntimeError skipped the rmtree."""
    upload_calls, eval_calls = _patch_run_cell(
        monkeypatch,
        env=cell_env,
        eval_raises=RuntimeError("Sycophancy eval subprocess exited non-zero (rc=1)"),
        upload_raises=None,
    )

    with pytest.raises(RuntimeError, match="exited non-zero"):
        syco_main._run_cell_mode(cell_env["args"])

    # Upload must have fired BEFORE the eval (the fix's call-order invariant).
    assert upload_calls.call_count == 1
    assert eval_calls.call_count == 1

    # Merged dir MUST be deleted even though eval raised — that's the bug fix.
    assert not cell_env["merged_path"].exists(), (
        f"merged dir leaked on eval failure: {cell_env['merged_path']} still exists; "
        "this is the #391 EDQUOT regression"
    )

    # Failure path does NOT write metrics.json (main()'s top-level handler
    # writes sycophancy_failed.json instead, outside _run_cell_mode).
    assert not (cell_env["output_dir"] / "metrics.json").exists()


def test_eval_success_writes_metrics_and_deletes_merged(monkeypatch, cell_env):
    """Happy path: train → upload → eval → metrics + cleanup."""
    _patch_run_cell(monkeypatch, env=cell_env, eval_raises=None, upload_raises=None)

    rc = syco_main._run_cell_mode(cell_env["args"])
    assert rc == 0

    # Merged dir deleted, adapter URL recorded in metrics.json.
    assert not cell_env["merged_path"].exists()
    metrics_path = cell_env["output_dir"] / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert payload["failed"] is False
    assert payload["hf_adapter_path"].startswith("https://huggingface.co/")
    assert payload["cell_key"] == "10011"


def test_upload_failure_preserves_merged_and_raises(monkeypatch, cell_env):
    """If HF Hub upload fails, the merged dir MUST survive on local disk.

    The cloud-copy invariant has not been established, so deleting the merged
    dir would lose the trained weights with no fallback. The eval is never
    attempted in this path (upload happens first).
    """
    _upload_calls, eval_calls = _patch_run_cell(
        monkeypatch,
        env=cell_env,
        eval_raises=None,
        upload_raises=RuntimeError("HF Hub 503; verification mismatch"),
    )

    with pytest.raises(RuntimeError, match="verification mismatch"):
        syco_main._run_cell_mode(cell_env["args"])

    # Eval must NOT have run (we bail before eval when upload fails).
    assert eval_calls.call_count == 0
    # Merged dir survives so the trained weights are not lost.
    assert cell_env["merged_path"].exists()
    assert (cell_env["merged_path"] / "model.safetensors").exists()
    # No metrics.json on the failure path.
    assert not (cell_env["output_dir"] / "metrics.json").exists()


def test_no_cleanup_merged_flag_preserves_merged(monkeypatch, cell_env):
    """`--no-cleanup-merged` overrides cleanup regardless of outcome."""
    cell_env["args"].cleanup_merged = False
    _patch_run_cell(monkeypatch, env=cell_env, eval_raises=None, upload_raises=None)

    rc = syco_main._run_cell_mode(cell_env["args"])
    assert rc == 0
    assert cell_env["merged_path"].exists()


# ---------------------------------------------------------------------------
# Direct unit test on the cleanup helper.
# ---------------------------------------------------------------------------


def test_cleanup_merged_if_safe_skips_when_upload_failed(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "weights.bin").write_bytes(b"\x00")

    syco_main._cleanup_merged_if_safe(
        merged_path=merged, upload_succeeded=False, cleanup_enabled=True
    )
    assert merged.exists(), "must NOT delete merged dir when upload did not succeed"


def test_cleanup_merged_if_safe_deletes_when_upload_succeeded(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "weights.bin").write_bytes(b"\x00")

    syco_main._cleanup_merged_if_safe(
        merged_path=merged, upload_succeeded=True, cleanup_enabled=True
    )
    assert not merged.exists()


def test_cleanup_merged_if_safe_disabled_flag_skips_even_on_success(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    syco_main._cleanup_merged_if_safe(
        merged_path=merged, upload_succeeded=True, cleanup_enabled=False
    )
    assert merged.exists()


def test_cleanup_merged_if_safe_tolerates_missing_dir(tmp_path):
    """If the merged dir is already gone (resume / partial state), cleanup is a no-op."""
    missing = tmp_path / "never_created"
    # Should not raise.
    syco_main._cleanup_merged_if_safe(
        merged_path=missing, upload_succeeded=True, cleanup_enabled=True
    )
    assert not missing.exists()


# ---------------------------------------------------------------------------
# Smoke: imports work without heavy ML deps.
# ---------------------------------------------------------------------------


def test_imports_cleanup_helpers():
    """Module-level smoke: the new helpers exist with the expected signature."""
    # Required helpers exposed at module level.
    assert callable(syco_main._cleanup_merged_if_safe)
    assert callable(syco_main._write_success_metrics)
    # The old monolithic `_finalize_cell` must NOT exist anymore (regression
    # guard against accidentally restoring the buggy single-function path).
    assert not hasattr(syco_main, "_finalize_cell")
    # Cell helper still imports from factor_screen_365.
    assert Cell.from_key("10011").key == "10011"
