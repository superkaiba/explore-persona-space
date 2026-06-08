# em-dash + Qwen marker " ※" + Greek ΔG intentional
"""Task #504 v5 round-2 — pin the 4 blockers fixed in implementer round 2 (v27).

Blocker A: ``_select_active_phase0_pick`` prefers v4 > v4_bisection > v3 >
   v2-fallback > v2-primary. A v4 pick with ``verdict='pass'`` MUST win over
   any v3/v2 pick.

Blocker B: ``_run_v2_phase1`` refuses to proceed when the active pick is v4
   AND ``phase0p6_validation_v4.json`` is missing OR has ``verdict != 'PASS'``.

Blocker C: ``_maybe_persist_trajectory_checkpoint`` is invoked at frac=1.00
   from ``train_one_cell``'s post-train tail (the
   ``CheckpointAtFractionsCallback.on_step_end`` skips ``frac >= 1.0``, so
   without this call the final adapter never lands at ``ckpt_frac1.00/``).
   Test verifies the call site + the formatted subfolder token "1.00".

Blocker D: ``CHECKPOINT_FRACTIONS_V4_BISECTION`` exists with the plan v5 §4.2
   step 1 grid and is exported from the contrastive_neg_geometry_504 package.

CPU-only, sub-second. No GPU/HF/network: filesystem-only artifact synthesis +
   AST-level call-site detection.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.dispatch_neg_geometry_504 import (
    _run_v2_phase1,
    _select_active_phase0_pick,
)

# ── Shared fixtures: synthetic pick artifacts ────────────────────────────────


def _v2_pass_artifact(*, fallback_triggered: bool = False) -> dict:
    return {
        "version": 2,
        "chosen_lr": 1e-4,
        "chosen_rank": 8,
        "chosen_alpha": 32,
        "chosen_checkpoint_fraction": 0.5,
        "chosen_checkpoint_steps": 12,
        "source": "villain",
        "verdict": "pass",
        "fallback_triggered": fallback_triggered,
        "fallback_reason": None,
    }


def _v3_pass_artifact() -> dict:
    return {
        "version": 3,
        "chosen_epochs": 2,
        "chosen_lr": 1e-4,
        "chosen_rank": 8,
        "chosen_alpha": 32,
        "chosen_checkpoint_fraction": 0.5,
        "chosen_checkpoint_steps": 25,
        "source": "villain",
        "verdict": "pass",
        "fallback_triggered": False,
        "fallback_reason": None,
    }


def _v4_pass_artifact(*, chosen_frac: float = 0.5, chosen_epochs: int = 3) -> dict:
    return {
        "version": 4,
        "anchor_epochs": 3,
        "fixed_lr": 1e-4,
        "fixed_rank": 8,
        "fixed_alpha": 32,
        "chosen_epochs": chosen_epochs,
        "chosen_lr": 1e-4,
        "chosen_rank": 8,
        "chosen_alpha": 32,
        "chosen_checkpoint_fraction": chosen_frac,
        "source_delta_g_at_pick_nats": 12.5,
        "source_emission_at_pick": 1.0,
        "bystander_resolution_at_pick": 0.42,
        "ceiling_logp": -0.105,
        "floor_delta_g": 0.5,
        "fallback_triggered": False,
        "fallback_reason": None,
        "source": "villain",
        "verdict": "pass",
    }


def _v4_no_in_band_artifact() -> dict:
    """v4 picker exhausted EPOCHS=3 fractions; bisection should run next."""
    return {
        "version": 4,
        "anchor_epochs": 3,
        "fixed_lr": 1e-4,
        "fixed_rank": 8,
        "fixed_alpha": 32,
        "chosen_epochs": 3,
        "chosen_lr": 1e-4,
        "chosen_rank": 8,
        "chosen_alpha": 32,
        "chosen_checkpoint_fraction": None,
        "source_delta_g_at_pick_nats": None,
        "source_emission_at_pick": None,
        "bystander_resolution_at_pick": None,
        "fallback_triggered": True,
        "fallback_reason": "all_saturated_or_below_floor",
        "source": "villain",
        "verdict": "no_in_band_anchor",
    }


# ── Blocker A: _select_active_phase0_pick precedence ──────────────────────────


def test_blocker_a_v4_pass_overrides_v3_v2(tmp_path: Path):
    """v4 pass artifact takes precedence over v3 + v2 pass artifacts.

    This pins the round-2 BLOCKER A contract: even when a passing v3 pick
    exists from an earlier round, the v4 picker output is preferred — the
    bystander-resolution gate IS the v4-introduced measurement fix.
    """
    v2 = tmp_path / "phase0_calibration_v2.json"
    v2_fb = tmp_path / "phase0_calibration_v2_fallback.json"
    v3 = tmp_path / "phase0_calibration_v3.json"
    v4 = tmp_path / "phase0_calibration_v4.json"
    v4b = tmp_path / "phase0_calibration_v4_bisection.json"

    v2.write_text(json.dumps(_v2_pass_artifact()))
    v3.write_text(json.dumps(_v3_pass_artifact()))
    v4.write_text(json.dumps(_v4_pass_artifact()))

    pick, active_path = _select_active_phase0_pick(
        v2, v2_fb, v3_pick_path=v3, v4_pick_path=v4, v4_bisection_pick_path=v4b
    )
    assert active_path == v4
    assert pick["version"] == 4
    assert pick["chosen_epochs"] == 3
    assert pick["bystander_resolution_at_pick"] == pytest.approx(0.42)


def test_blocker_a_v4_no_in_band_falls_through_to_bisection(tmp_path: Path):
    """v4 primary no_in_band → bisection (which passed) wins."""
    v2 = tmp_path / "phase0_calibration_v2.json"
    v2_fb = tmp_path / "phase0_calibration_v2_fallback.json"
    v3 = tmp_path / "phase0_calibration_v3.json"
    v4 = tmp_path / "phase0_calibration_v4.json"
    v4b = tmp_path / "phase0_calibration_v4_bisection.json"

    v2.write_text(json.dumps(_v2_pass_artifact()))
    v3.write_text(json.dumps(_v3_pass_artifact()))
    v4.write_text(json.dumps(_v4_no_in_band_artifact()))
    v4b.write_text(json.dumps(_v4_pass_artifact(chosen_frac=0.12, chosen_epochs=2)))

    pick, active_path = _select_active_phase0_pick(
        v2, v2_fb, v3_pick_path=v3, v4_pick_path=v4, v4_bisection_pick_path=v4b
    )
    assert active_path == v4b
    assert pick["version"] == 4
    assert pick["chosen_epochs"] == 2
    assert pick["chosen_checkpoint_fraction"] == pytest.approx(0.12)


def test_blocker_a_v4_and_bisection_both_fail_fall_through_to_v3(tmp_path: Path):
    """Both v4 paths failed; v3 pass takes over."""
    v2 = tmp_path / "phase0_calibration_v2.json"
    v2_fb = tmp_path / "phase0_calibration_v2_fallback.json"
    v3 = tmp_path / "phase0_calibration_v3.json"
    v4 = tmp_path / "phase0_calibration_v4.json"
    v4b = tmp_path / "phase0_calibration_v4_bisection.json"

    v2.write_text(json.dumps(_v2_pass_artifact()))
    v3.write_text(json.dumps(_v3_pass_artifact()))
    v4.write_text(json.dumps(_v4_no_in_band_artifact()))
    v4b.write_text(json.dumps(_v4_no_in_band_artifact()))

    pick, active_path = _select_active_phase0_pick(
        v2, v2_fb, v3_pick_path=v3, v4_pick_path=v4, v4_bisection_pick_path=v4b
    )
    assert active_path == v3
    assert pick["version"] == 3


def test_blocker_a_no_v4_artifacts_uses_v3(tmp_path: Path):
    """When the v4 path was never run, the helper falls back to v3/v2 exactly
    like before (backward compatibility with existing v3 callers)."""
    v2 = tmp_path / "phase0_calibration_v2.json"
    v2_fb = tmp_path / "phase0_calibration_v2_fallback.json"
    v3 = tmp_path / "phase0_calibration_v3.json"
    v4 = tmp_path / "phase0_calibration_v4.json"
    v4b = tmp_path / "phase0_calibration_v4_bisection.json"

    v2.write_text(json.dumps(_v2_pass_artifact()))
    v3.write_text(json.dumps(_v3_pass_artifact()))
    # v4 + v4b NOT written.

    _pick, active_path = _select_active_phase0_pick(
        v2, v2_fb, v3_pick_path=v3, v4_pick_path=v4, v4_bisection_pick_path=v4b
    )
    assert active_path == v3


def test_blocker_a_kwargs_default_to_none(tmp_path: Path):
    """The new v4 kwargs default to None; existing callers (v2/v3 only) are
    byte-identically unaffected — the helper takes either the v3 OR v2 path
    when the v4 arguments aren't provided."""
    v2 = tmp_path / "phase0_calibration_v2.json"
    v2_fb = tmp_path / "phase0_calibration_v2_fallback.json"
    v3 = tmp_path / "phase0_calibration_v3.json"
    v2.write_text(json.dumps(_v2_pass_artifact()))
    v3.write_text(json.dumps(_v3_pass_artifact()))

    _pick, active_path = _select_active_phase0_pick(v2, v2_fb, v3_pick_path=v3)
    assert active_path == v3


# ── Blocker B: Phase 0.6 gate blocks Phase 1 ─────────────────────────────────


def _make_phase1_args(tmp_path: Path) -> "object":  # noqa: UP037
    """Build a minimal argparse.Namespace-like object that exercises the v4
    branch of `_run_v2_phase1`. We mock everything after the pick + Phase 0.6
    gate; the test only needs to drive the gate to RuntimeError."""
    import argparse

    args = argparse.Namespace()
    args.slab_root = tmp_path
    args.runs_root = tmp_path / "runs"
    args.runs_root.mkdir(exist_ok=True)
    args.bank_path = tmp_path / "bank.json"
    args.centroids_dir = tmp_path / "centroids"
    args.r_train_path = tmp_path / "r_train.json"
    args.r_eval_path = tmp_path / "r_eval.json"
    args.source = "villain"
    args.cells = None
    args.n_gpus = 1
    args.max_parallel = 1
    args.smoke = False
    args.no_kl = False
    args.report_to = "none"
    args.resume = False
    args.skip_analyze = True  # don't try to run the analyze subprocess
    args.hf_path_suffix = ""
    return args


def test_blocker_b_v4_pick_pass_but_phase06_missing_raises(tmp_path: Path):
    """When the active pick is v4 AND phase0p6_validation_v4.json is missing,
    _run_v2_phase1 must raise RuntimeError before any GPU work."""
    args = _make_phase1_args(tmp_path)
    v4_path = tmp_path / "phase0_calibration_v4.json"
    v4_path.write_text(json.dumps(_v4_pass_artifact()))
    # phase0p6_validation_v4.json NOT written.

    arm_to_n_json = tmp_path / "arm_to_n.json"
    arm_to_n_json.write_text("{}")

    with pytest.raises(RuntimeError, match=r"phase0p6_not_passed_before_phase1"):
        _run_v2_phase1(
            args=args,
            phase_summaries={},
            arm_to_n_json=arm_to_n_json,
            max_new_tokens_eval=2048,
            max_model_len_eval=2560,
            seeds=[42],
        )


def test_blocker_b_v4_pick_pass_but_phase06_verdict_fail_raises(tmp_path: Path):
    """When phase0p6_validation_v4.json has verdict='FAIL', _run_v2_phase1 must
    raise RuntimeError before any GPU work."""
    args = _make_phase1_args(tmp_path)
    v4_path = tmp_path / "phase0_calibration_v4.json"
    v4_path.write_text(json.dumps(_v4_pass_artifact()))
    p06 = tmp_path / "phase0p6_validation_v4.json"
    p06.write_text(
        json.dumps(
            {
                "version": 4,
                "verdict": "FAIL",
                "pass_a": False,
                "pass_b": True,
                "byte_identical_rate": 0.0,
                "n_byte_identical": 0,
                "n_total": 20,
            }
        )
    )

    arm_to_n_json = tmp_path / "arm_to_n.json"
    arm_to_n_json.write_text("{}")

    with pytest.raises(RuntimeError, match=r"phase0p6_not_passed_before_phase1"):
        _run_v2_phase1(
            args=args,
            phase_summaries={},
            arm_to_n_json=arm_to_n_json,
            max_new_tokens_eval=2048,
            max_model_len_eval=2560,
            seeds=[42],
        )


def test_blocker_b_v3_pick_active_skips_phase06_gate(tmp_path: Path):
    """When the active pick is v3 (not v4), the Phase 0.6 gate is NOT applied —
    v3 pre-dates Phase 0.6 and the dispatcher must remain backward compatible.

    We can't drive _run_v2_phase1 all the way through here (it would launch
    real training), so we mock _schedule_cell_pool to short-circuit after the
    gate; the test passes when the gate doesn't raise on a v3-active pick.
    """
    args = _make_phase1_args(tmp_path)
    v2_path = tmp_path / "phase0_calibration_v2.json"
    v2_path.write_text(json.dumps(_v2_pass_artifact()))
    v3_path = tmp_path / "phase0_calibration_v3.json"
    v3_path.write_text(json.dumps(_v3_pass_artifact()))
    # phase0p6_validation_v4.json deliberately MISSING — v3 path must not gate.

    arm_to_n_json = tmp_path / "arm_to_n.json"
    arm_to_n_json.write_text("{}")

    # Stop execution at the schedule call.
    with (
        patch(
            "scripts.dispatch_neg_geometry_504._schedule_cell_pool",
            side_effect=RuntimeError("_schedule_cell_pool_called_marker"),
        ),
        pytest.raises(RuntimeError, match=r"_schedule_cell_pool_called_marker"),
    ):
        _run_v2_phase1(
            args=args,
            phase_summaries={},
            arm_to_n_json=arm_to_n_json,
            max_new_tokens_eval=2048,
            max_model_len_eval=2560,
            seeds=[42],
        )


# ── Blocker C: frac=1.00 trajectory persistence ───────────────────────────────


def test_blocker_c_train_one_cell_calls_maybe_persist_at_frac_one():
    """``train_one_cell``'s tail invokes ``_maybe_persist_trajectory_checkpoint``
    with ``frac=1.0`` so the final adapter lands at ``ckpt_frac1.00/`` under
    the v4 subfolder (when the EPM env vars are set).

    The callback's ``on_step_end`` explicitly skips ``frac >= 1.0`` (see
    ``CheckpointAtFractionsCallback`` line ``if frac in self._saved or
    frac >= 1.0: continue``). Without the tail call in ``train_one_cell``,
    the dispatcher's ``_run_v4_phase0_pretrain`` Hub-API verify would raise
    "1 of 6 fraction checkpoints missing" on what was otherwise a successful
    run. We pin the contract by AST-scanning the train_cell module for the
    expected call at the expected location.
    """
    train_cell_py = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "explore_persona_space"
        / "experiments"
        / "contrastive_neg_geometry_472"
        / "train_cell.py"
    )
    assert train_cell_py.is_file(), train_cell_py
    tree = ast.parse(train_cell_py.read_text())

    # Locate the train_one_cell function body.
    target_fn: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "train_one_cell":
            target_fn = node
            break
    assert target_fn is not None, "train_one_cell function missing from train_cell.py"

    # Find a call of _maybe_persist_trajectory_checkpoint with frac=1.0 (or 1).
    found = False
    for node in ast.walk(target_fn):
        is_target_call = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_maybe_persist_trajectory_checkpoint"
            and len(node.args) >= 2
        )
        if not is_target_call:
            continue
        frac_arg = node.args[1]
        # Accept ast.Constant(1.0) or ast.Constant(1) — both produce
        # frac >= 1.0 ⇒ ckpt_frac1.00/ subfolder.
        if isinstance(frac_arg, ast.Constant) and frac_arg.value in (1.0, 1):
            found = True
            break
    assert found, (
        "train_one_cell must call _maybe_persist_trajectory_checkpoint(adapter_dir, "
        "1.0, frac_precision) AFTER train_lora returns, so the frac=1.00 final "
        "adapter lands at the v4 subfolder's ckpt_frac1.00/. The "
        "CheckpointAtFractionsCallback's on_step_end explicitly skips frac>=1.0 — "
        "this is the only path that uploads it under the v4 layout."
    )


def test_blocker_c_maybe_persist_formats_frac_as_two_decimals(tmp_path, monkeypatch):
    """``_maybe_persist_trajectory_checkpoint(adapter_dir, 1.0, 2)`` formats
    the destination subfolder token as "1.00" (NOT "1.0", "1", or "1.0000"),
    so the resulting HF path matches the dispatcher's expected key
    ``ckpt_frac1.00/adapter_model.safetensors``."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
        _maybe_persist_trajectory_checkpoint,
    )

    monkeypatch.setenv("EPM_PERSIST_TRAJECTORY_HF_REPO", "test/fake-repo")
    monkeypatch.setenv(
        "EPM_PERSIST_TRAJECTORY_HF_SUBFOLDER",
        "adapters/issue_504_v4/c504v4_smoke_eps3_seed42",
    )
    adapter_dir = tmp_path / "final_adapter"
    adapter_dir.mkdir()
    # Fake adapter weights so the early-exit "adapter_model.safetensors missing"
    # check passes.
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake_safetensors")

    # Capture the destination passed to upload_model + the verify call.
    captured: dict = {}

    def _fake_upload_model(*, model_path, repo_id, path_in_repo, delete_after):
        captured["model_path"] = model_path
        captured["repo_id"] = repo_id
        captured["path_in_repo"] = path_in_repo
        captured["delete_after"] = delete_after
        # Return a non-empty path so the early-exit "upload returned empty"
        # check passes.
        return f"{repo_id}/{path_in_repo}"

    def _fake_list_repo_files(repo, token=None):
        # Return the expected key so the verify step passes.
        return [f"{captured['path_in_repo']}/adapter_model.safetensors"]

    with (
        patch(
            "explore_persona_space.orchestrate.hub.upload_model",
            side_effect=_fake_upload_model,
        ),
        patch("huggingface_hub.list_repo_files", side_effect=_fake_list_repo_files),
    ):
        _maybe_persist_trajectory_checkpoint(adapter_dir, 1.0, 2)

    # Pin the exact subfolder token: "1.00", NOT "1.0".
    assert captured["path_in_repo"] == (
        "adapters/issue_504_v4/c504v4_smoke_eps3_seed42/ckpt_frac1.00"
    ), captured
    assert captured["delete_after"] is False, (
        "delete_after must be False — the local copy is needed for the eval rig "
        "that runs after training in the same process tree."
    )


# ── Blocker D: bisection constants + dispatcher branch ────────────────────────


def test_blocker_d_v4_bisection_constant_exists():
    """``CHECKPOINT_FRACTIONS_V4_BISECTION`` exists in the
    contrastive_neg_geometry_504 package with the plan v5 §4.2 step 1 grid."""
    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        CHECKPOINT_FRACTIONS_V4_BISECTION,
    )

    assert CHECKPOINT_FRACTIONS_V4_BISECTION == (0.04, 0.08, 0.12, 0.16), (
        CHECKPOINT_FRACTIONS_V4_BISECTION
    )


def test_blocker_d_dispatcher_has_phase0_v4_bisection_choice():
    """The dispatcher's ``--phase`` choices include ``phase0_v4_bisection``
    so an operator (or the orchestrator) can drive the fallback explicitly
    when the v4 picker returns no_in_band_anchor."""
    dispatch_py = Path(__file__).resolve().parents[2] / "scripts" / "dispatch_neg_geometry_504.py"
    text = dispatch_py.read_text()
    # The choice tuple is inside the argparse add_argument call; just check
    # the literal appears.
    assert '"phase0_v4_bisection"' in text, (
        '"phase0_v4_bisection" must be a valid --phase choice in '
        "scripts/dispatch_neg_geometry_504.py so the dispatcher can run the "
        "v5 §4.2 step 1 EPOCHS=2 finer-fraction bisection fallback."
    )
    # And the dispatch branch must exist.
    assert "_run_v4_phase0_bisection" in text, (
        "_run_v4_phase0_bisection function must exist; it's the v4 fallback "
        "handler that re-trains EPOCHS=2 at the finer grid and re-applies the "
        "bystander-resolution picker."
    )


def test_blocker_d_bisection_writes_v4_bisection_pick_path():
    """The bisection handler writes to ``phase0_calibration_v4_bisection.json``
    so ``_select_active_phase0_pick`` (which reads this exact filename) can
    pick it up automatically when v4-primary failed."""
    dispatch_py = Path(__file__).resolve().parents[2] / "scripts" / "dispatch_neg_geometry_504.py"
    text = dispatch_py.read_text()
    assert "phase0_calibration_v4_bisection.json" in text, (
        "phase0_calibration_v4_bisection.json must be referenced in the "
        "dispatcher — both the writer (_run_v4_phase0_bisection) and the "
        "reader (_select_active_phase0_pick) need to agree on this filename."
    )
