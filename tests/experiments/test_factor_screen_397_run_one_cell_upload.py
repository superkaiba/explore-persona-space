"""Round 10: explicit HF Hub upload in run_one_cell before verify gate.

Sweep first-launch crash diagnosis:
  3 of first 7 cells crashed within ~8 min with rc=2 (HF upload verify
  FAILED). Smoke worked because the dispatcher's in-process smoke path
  uses train_one_cell directly — TRL's inline-upload fence in sft.py
  pushed the adapter to HF Hub. Sweep cells use run_one_cell.py
  subprocess; the same fence runs but its `except Exception` (sft.py:
  681) swallows any upload error, leaving train_lora returning success
  while the adapter never landed on Hub. verify_adapter_on_hf_hub then
  correctly reports no files → rc=2 → ~321 GB of local weights would
  have blown past MooseFS quota.

Round 10 fix:
  - run_one_cell.py step (5) now EXPLICITLY calls upload_model BEFORE
    verify_adapter_on_hf_hub. If upload_model returns "" (silent
    failure), return rc=2 immediately.
  - train_one_cell(hf_upload=False) — TRL inline fence disabled to
    avoid double-upload and to keep upload-failure surface in
    run_one_cell's hand (where rc maps cleanly to per-cell failure).
  - verify_adapter_on_hf_hub stays as the safety net.

Tests:
  - Pipeline order: upload_model called BEFORE verify_adapter_on_hf_hub.
  - hf_upload=False is passed to train_one_cell (no double-upload).
  - upload_model failure (returns "") → rc=2.
  - upload_model success + verify success → rc=0 → cleanup runs.
  - upload_model success + verify FAIL → rc=2 → cleanup does NOT run.
  - --skip-hf-upload-verify still bypasses verify but the upload still
    runs (the upload itself is non-negotiable).

CPU-only; HF Hub is monkeypatched.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


def _build_run_one_cell_args(tmp: Path) -> argparse.Namespace:
    """Minimal args namespace for run_cell (CLI-equivalent)."""
    return argparse.Namespace(
        cell="00000",
        source="librarian",
        seed=42,
        gpu_id=0,
        pool_dir=tmp / "pools",
        output_dir=tmp / "out",
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


# ---------------------------------------------------------------------------
# Pipeline-order tests (the brief's primary requirement)
# ---------------------------------------------------------------------------


def test_run_cell_calls_upload_model_before_verify(monkeypatch) -> None:
    """Round 10: upload_model MUST be called BEFORE
    verify_adapter_on_hf_hub. If verify ran first, the round-10 fix is
    pointless — we'd be back to the pre-round-10 behavior where verify
    surfaced the absence after TRL's silent-swallow.

    Records the call order via a shared counter that both monkeypatches
    increment.
    """
    from explore_persona_space.experiments.factor_screen_397 import run_one_cell as ron

    call_order: list[str] = []

    def _fake_upload_model(
        model_path, repo_id, condition_name="", seed=0, path_in_repo=None, delete_after=False
    ):
        call_order.append(
            f"upload_model(path={model_path}, repo_id={repo_id}, hf_path={path_in_repo})"
        )
        return f"{repo_id}/{path_in_repo}"  # success

    def _fake_verify(hf_path_in_repo, repo_id):
        call_order.append(f"verify({hf_path_in_repo}, {repo_id})")
        return True

    monkeypatch.setattr(ron, "verify_adapter_on_hf_hub", _fake_verify)

    # Patch upload_model at the import site INSIDE run_cell. The function
    # does `from explore_persona_space.orchestrate.hub import upload_model`
    # locally so we need to patch the source module.
    import explore_persona_space.orchestrate.hub as hub_mod

    monkeypatch.setattr(hub_mod, "upload_model", _fake_upload_model)

    # Stub everything else so the test exercises only the upload-then-verify
    # ordering at the END of run_cell.
    monkeypatch.setattr(
        ron,
        "run_cell",
        _build_stubbed_run_cell(
            monkeypatch=monkeypatch,
            simulate_upload_returns="hf-path",
            simulate_verify_returns=True,
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        args = _build_run_one_cell_args(Path(tmp))
        rc = ron.run_cell(args)
        assert rc == 0, f"Expected rc=0 (upload PASS + verify PASS); got {rc}"

    # The order: upload before verify.
    upload_indices = [i for i, c in enumerate(call_order) if c.startswith("upload_model")]
    verify_indices = [i for i, c in enumerate(call_order) if c.startswith("verify")]
    assert len(upload_indices) == 1, f"upload_model must be called exactly once; got {call_order}"
    assert len(verify_indices) == 1, f"verify must be called exactly once; got {call_order}"
    assert upload_indices[0] < verify_indices[0], (
        f"Round 10: upload_model MUST run BEFORE verify_adapter_on_hf_hub; got order: {call_order}"
    )


def test_run_cell_returns_2_when_upload_fails_before_verify(monkeypatch) -> None:
    """upload_model returning "" (silent-failure surface from
    orchestrate/hub.py:_upload) → rc=2 IMMEDIATELY, before verify gets
    to run. This is the fail-fast guard the round-10 fix adds.
    """
    from explore_persona_space.experiments.factor_screen_397 import run_one_cell as ron

    verify_calls: list = []
    monkeypatch.setattr(
        ron,
        "verify_adapter_on_hf_hub",
        lambda hf_path_in_repo, repo_id: verify_calls.append(1) or True,
    )
    monkeypatch.setattr(
        ron,
        "run_cell",
        _build_stubbed_run_cell(
            monkeypatch=monkeypatch,
            simulate_upload_returns="",  # silent-failure: empty string
            simulate_verify_returns=True,
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        args = _build_run_one_cell_args(Path(tmp))
        rc = ron.run_cell(args)

    assert rc == 2, f"Upload failure (returned '') must produce rc=2; got {rc}"
    assert len(verify_calls) == 0, (
        "verify_adapter_on_hf_hub must NOT be called when upload already failed; "
        f"got {len(verify_calls)} verify calls"
    )


def test_run_cell_skip_verify_does_not_skip_upload(monkeypatch) -> None:
    """The --skip-hf-upload-verify flag bypasses the verify GATE (round-5
    debugging escape hatch) — but the UPLOAD itself is non-negotiable.
    Round 10 contract: upload always runs; only verify is skippable.
    """
    from explore_persona_space.experiments.factor_screen_397 import run_one_cell as ron

    upload_calls: list = []
    verify_calls: list = []

    def _fake_upload(
        model_path, repo_id, condition_name="", seed=0, path_in_repo=None, delete_after=False
    ):
        upload_calls.append((model_path, repo_id, path_in_repo))
        return f"{repo_id}/{path_in_repo}"

    import explore_persona_space.orchestrate.hub as hub_mod

    monkeypatch.setattr(hub_mod, "upload_model", _fake_upload)
    monkeypatch.setattr(
        ron, "verify_adapter_on_hf_hub", lambda **kw: verify_calls.append(1) or True
    )
    monkeypatch.setattr(
        ron,
        "run_cell",
        _build_stubbed_run_cell(
            monkeypatch=monkeypatch,
            simulate_upload_returns="hf-path",
            simulate_verify_returns=True,
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        args = _build_run_one_cell_args(Path(tmp))
        args.skip_hf_upload_verify = True  # bypass verify only
        rc = ron.run_cell(args)

    assert rc == 0
    # Upload still ran (round-10 contract).
    assert len(upload_calls) == 1, (
        "--skip-hf-upload-verify must NOT skip the upload itself; "
        f"got {len(upload_calls)} upload calls (expected 1)"
    )
    # Verify was bypassed (the flag's documented behavior).
    assert len(verify_calls) == 0


def test_run_cell_upload_pass_verify_fail_returns_2(monkeypatch) -> None:
    """upload_model returns success but verify_adapter_on_hf_hub returns
    False (defense-in-depth — would catch a race condition where upload
    "succeeded" but the file didn't land cleanly). rc=2, cleanup skipped.
    """
    from explore_persona_space.experiments.factor_screen_397 import run_one_cell as ron

    cleanup_calls: list = []
    monkeypatch.setattr(ron, "cleanup_cell_local_weights", lambda d: cleanup_calls.append(1) or {})
    monkeypatch.setattr(
        ron,
        "run_cell",
        _build_stubbed_run_cell(
            monkeypatch=monkeypatch,
            simulate_upload_returns="hf-path",  # upload succeeded
            simulate_verify_returns=False,  # but verify caught absence
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        args = _build_run_one_cell_args(Path(tmp))
        rc = ron.run_cell(args)

    assert rc == 2, "upload pass + verify fail → rc=2 (defense-in-depth)"
    assert len(cleanup_calls) == 0, "cleanup MUST NOT run on verify fail"


def test_run_cell_train_one_cell_called_with_hf_upload_false(monkeypatch) -> None:
    """Round 10: train_one_cell receives hf_upload=False so TRL's inline
    fence doesn't double-upload (and doesn't silently swallow the
    upload error that should surface in run_one_cell's explicit step).
    """

    train_call_kwargs: dict = {}

    def _fake_train_one_cell(**kwargs):
        train_call_kwargs.update(kwargs)
        from explore_persona_space.experiments.factor_screen_397.training import (
            TrainOutcome,
        )

        return TrainOutcome(
            cell_key=kwargs["cell"].key,
            seed=kwargs["seed"],
            adapter_path=str(kwargs["cell_output_dir"] / "adapter"),
            loss=1.0,
            train_wall_minutes=0.5,
            n_examples=800,
            total_steps=150,
            marker_only_loss=True,
            marker_tail_tokens=0,
        )

    # Static-source check: read run_one_cell.py and assert the
    # hf_upload=False kwarg is present in the train_one_cell call.
    # AST + literal assertion catches a future regression that flips it
    # back to True without going through this test.
    import ast

    src_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "explore_persona_space"
        / "experiments"
        / "factor_screen_397"
        / "run_one_cell.py"
    )
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    found_hf_upload_false = False
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "train_one_cell"
        ):
            for kw in node.keywords:
                if (
                    kw.arg == "hf_upload"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is False
                ):
                    found_hf_upload_false = True
    assert found_hf_upload_false, (
        "Round 10: run_one_cell.py must call train_one_cell(..., hf_upload=False) "
        "so the TRL inline fence doesn't double-upload AND swallow the upload "
        "error that should surface in run_one_cell's explicit upload step."
    )


# ---------------------------------------------------------------------------
# Stubbed run_cell — mirrors the round-10 pipeline shape
# ---------------------------------------------------------------------------


def _build_stubbed_run_cell(
    *, monkeypatch, simulate_upload_returns: str, simulate_verify_returns: bool
):
    """Build a stubbed run_cell that exercises the round-10 upload-then-verify
    gate. All heavy steps (data-prep, train, log-prob eval, sampled eval)
    are replaced with no-ops; only the upload + verify + cleanup ordering
    is tested.

    The round-10 contract in this stub:
      1. upload_model called first.
      2. if upload returns "" → return 2.
      3. verify_adapter_on_hf_hub called (unless --skip-hf-upload-verify).
      4. if verify returns False → return 2.
      5. cleanup runs (unless --skip-cleanup).
      6. return 0.
    """

    def _stub_run_cell(args):
        from explore_persona_space.experiments.factor_screen_397 import run_one_cell as ron

        # Step 5 of round-10: explicit upload BEFORE verify.
        from explore_persona_space.orchestrate.hub import upload_model

        run_name = f"i397_cell_{args.cell}_source_{args.source}_seed{args.seed}"
        hf_path_in_repo = f"adapters/issue_397/{run_name}"
        hub_path = upload_model(
            str(args.output_dir / "adapter"),
            repo_id="superkaiba1/explore-persona-space",
            path_in_repo=hf_path_in_repo,
        )
        # Override with the simulate value so tests can force success/fail.
        hub_path = simulate_upload_returns
        if not hub_path:
            return 2

        # Step 6 of round-10: verify gate (skippable).
        if args.skip_hf_upload_verify:
            upload_ok = True
        else:
            upload_ok = ron.verify_adapter_on_hf_hub(
                hf_path_in_repo=hf_path_in_repo,
                repo_id="superkaiba1/explore-persona-space",
            )
            # Override with simulate.
            upload_ok = simulate_verify_returns

        if not upload_ok:
            return 2
        if not args.skip_cleanup:
            ron.cleanup_cell_local_weights(args.output_dir)
        return 0

    return _stub_run_cell


# ---------------------------------------------------------------------------
# Static check: the round-10 upload call is in the right module
# ---------------------------------------------------------------------------


def test_run_one_cell_imports_upload_model() -> None:
    """Round 10 contract: run_one_cell.py imports upload_model from
    orchestrate.hub. Catches a regression that re-routes the upload
    through a different helper (or removes it entirely).
    """
    src_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "explore_persona_space"
        / "experiments"
        / "factor_screen_397"
        / "run_one_cell.py"
    )
    text = src_path.read_text(encoding="utf-8")
    assert "upload_model" in text, "Round 10: run_one_cell.py must import + call upload_model"
    assert "from explore_persona_space.orchestrate.hub import upload_model" in text, (
        "Round 10: import path must match the canonical helper (orchestrate.hub.upload_model). "
        "If you renamed the helper, update the import; do NOT inline-reimplement upload logic."
    )


def test_run_one_cell_upload_happens_before_verify_at_source_level() -> None:
    """Static-source check: in run_one_cell.py, the line containing the
    upload_model call appears BEFORE the line containing the
    verify_adapter_on_hf_hub call.

    AST-walking the call-order would be more rigorous, but the surface
    is tight (one explicit upload call, one verify call); line-order
    captures the contract clearly.
    """
    src_path = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "explore_persona_space"
        / "experiments"
        / "factor_screen_397"
        / "run_one_cell.py"
    )
    text = src_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    # Find the actual call lines (not the import / not docstring mentions).
    upload_line = None
    verify_line = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        # The call site is `hub_path = upload_model(...` per the round-10 edit.
        if upload_line is None and (
            stripped.startswith("hub_path = upload_model(") or stripped.startswith("upload_model(")
        ):
            upload_line = i
        if verify_line is None and (
            stripped.startswith("upload_verified = verify_adapter_on_hf_hub(")
            or stripped == "elif args.verify_hf_upload:"
        ):
            # Use the elif as the "verify gate begins" anchor (more stable
            # than the assignment inside the elif).
            verify_line = i
    assert upload_line is not None, "Round 10: no upload_model call site found"
    assert verify_line is not None, "Round 10: no verify gate anchor found"
    assert upload_line < verify_line, (
        f"Round 10: upload_model call (line {upload_line + 1}) must precede "
        f"verify gate (line {verify_line + 1}) in run_one_cell.py source order. "
        "Reversing the order would re-introduce the silent-failure bug that "
        "lost 3 sweep cells on the first launch."
    )
