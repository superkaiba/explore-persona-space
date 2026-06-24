"""Task #653 v8 §6Δ.3 — geometry reads the FIRST floor-clearing checkpoint.

CPU-only. BLOCKER geometry-reads-final-adapter: with dose checkpoints now saved
(BLOCKER 1), the geometry/install/ablation reads must come from the FIRST
floor-clearing checkpoint (the matched-install read point), NOT the saturated
final adapter. The select_checkpoint phase iterates the saved checkpoints in dose
order and records the first one clearing the per-behavior install floor. These
tests pin:

  * select_checkpoint picks the first floor-clearing checkpoint (step 5 fails,
    step 9 passes → manifest records step 9, not the final adapter);
  * _resolve_read_model_path resolves a dose cell to the SELECTED checkpoint's
    merged dir (so dx / install / ablation read the matched-install model);
  * a cell whose checkpoints NEVER clear the floor is DROPPED, and
    _resolve_read_model_path RAISES for it (geometry never read off a dropped cell);
  * marker / full-FT cells are NO-OPs (read their final model — no dose selection).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from explore_persona_space.experiments import issue_653 as i653


def _load_dispatcher():
    repo_root = Path(__file__).resolve().parents[1]
    disp_path = repo_root / "scripts" / "issue_653" / "i653_dispatch.py"
    spec = importlib.util.spec_from_file_location("i653_dispatch_ckptsel_test", disp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i653_dispatch_ckptsel_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _stage_checkpoints(out_root: Path, cell, steps: list[int]) -> None:
    """Create synthetic checkpoint-<step> dirs under the cell's adapter dir."""
    adapter_dir = out_root / "armB" / "adapters" / cell.cell_id
    for s in steps:
        (adapter_dir / f"checkpoint-{s}").mkdir(parents=True, exist_ok=True)


def _patch_merge_and_probe(mod, monkeypatch, *, pass_at_step: int | None):
    """Stub the merge (return the dir path) + the per-checkpoint install probe.

    ``pass_at_step`` — the FIRST checkpoint step at/above which the install probe
    clears the floor; None → no checkpoint ever clears (drop). The probe parses
    the step from the merged path (a checkpoint-<step> dir)."""

    def _fake_merge(adapter_dir, cell):
        return str(adapter_dir)  # no real merge; path identifies the checkpoint

    def _fake_install(cell, *, out_root, trained_path=None):
        # Parse checkpoint-<step> from the trained_path.
        step = int(Path(trained_path).name.split("checkpoint-", 1)[1])
        gain = 0.5 if (pass_at_step is not None and step >= pass_at_step) else 0.05
        return {
            "dv_kind": "judge_rate_plus_gain",
            "behavior": cell.behavior,
            "judge_rate_gain": gain,
            "continuous_gain_logp": 0.3,
        }

    monkeypatch.setattr(mod, "_merge_adapter_for_read", _fake_merge)
    monkeypatch.setattr(mod, "_install_content_gpu", _fake_install)


def test_select_picks_first_floor_clearing_checkpoint(tmp_path, monkeypatch):
    """Step 5 fails the sycophancy +0.40 floor; step 9 passes → manifest records
    checkpoint 9 (the first floor-clearing), NOT the final adapter / final dose."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    # Sycophancy dose ladder {5,9,...}; save_steps=5 → checkpoints at 5,10,...
    # The dose step 9 snaps to checkpoint-5? No — snap is ≤ dose_step. Stage the
    # actual saved checkpoints (every 5) and have the probe clear at step 10.
    _stage_checkpoints(tmp_path, cell, [5, 10, 15, 130])
    _patch_merge_and_probe(mod, monkeypatch, pass_at_step=10)

    res = mod.phase_select_checkpoint([cell], out_root=tmp_path, mode=i653.RUN_MODE_GPU)
    assert res["n_selected"] == 1
    assert res["n_dropped_non_install"] == 0

    man = json.loads(
        (tmp_path / "armB" / "selected_checkpoints" / f"{cell.cell_id}.json").read_text()
    )
    assert man["dropped_non_install"] is False
    # The first checkpoint clearing +0.40 is step 10 (step 5 read 0.05).
    assert man["selected_checkpoint_step"] == 10
    assert man["selected_checkpoint_dir"].endswith("checkpoint-10")
    assert man["selected_model_path"].endswith("checkpoint-10")


def test_resolve_read_model_path_uses_selected_checkpoint(tmp_path, monkeypatch):
    """_resolve_read_model_path returns the SELECTED checkpoint's merged dir for a
    dose cell — NOT the final adapter (the matched-install read point, §6Δ.3)."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(behavior="em", source="florist", rung="r16", seed=i653.HEADLINE_SEED)
    # EM ladder {40,80,...}; stage checkpoints exactly on dose steps; clear at 80.
    _stage_checkpoints(tmp_path, cell, [40, 80, 120, 160, 200])
    _patch_merge_and_probe(mod, monkeypatch, pass_at_step=80)
    mod.phase_select_checkpoint([cell], out_root=tmp_path, mode=i653.RUN_MODE_GPU)

    resolved = mod._resolve_read_model_path(cell, tmp_path)
    assert resolved.endswith("checkpoint-80")  # the selected (matched-install) read
    # Crucially NOT the final adapter dir.
    assert not resolved.endswith(cell.cell_id)
    assert "checkpoint-200" not in resolved  # not the saturated endpoint


def test_dropped_cell_raises_in_resolver(tmp_path, monkeypatch):
    """A cell whose checkpoints never clear the floor is DROPPED; the resolver
    RAISES for it — geometry/install must NEVER be read off a dropped cell."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    _stage_checkpoints(tmp_path, cell, [5, 10, 15, 130])
    _patch_merge_and_probe(mod, monkeypatch, pass_at_step=None)  # never clears
    res = mod.phase_select_checkpoint([cell], out_root=tmp_path, mode=i653.RUN_MODE_GPU)
    assert res["n_dropped_non_install"] == 1

    man = json.loads(
        (tmp_path / "armB" / "selected_checkpoints" / f"{cell.cell_id}.json").read_text()
    )
    assert man["dropped_non_install"] is True
    assert man["selected_checkpoint_step"] is None
    with pytest.raises(RuntimeError, match="DROPPED"):
        mod._resolve_read_model_path(cell, tmp_path)


def test_marker_and_full_ft_are_noop(tmp_path, monkeypatch):
    """Marker (band-stop) + full-FT (no dose) cells write a no-op manifest pointing
    at the final model; the resolver falls through to the final adapter for them."""
    mod = _load_dispatcher()
    marker = i653.ArmBCell(behavior="marker", source="florist", rung="r16", seed=i653.HEADLINE_SEED)
    full_ft = i653.ArmBCell(behavior="em", source="florist", rung="full", seed=i653.HEADLINE_SEED)
    res = mod.phase_select_checkpoint([marker, full_ft], out_root=tmp_path, mode=i653.RUN_MODE_GPU)
    assert res["n_selected"] == 0
    assert res["n_dropped_non_install"] == 0
    for c in (marker, full_ft):
        man = json.loads(
            (tmp_path / "armB" / "selected_checkpoints" / f"{c.cell_id}.json").read_text()
        )
        assert man["dose_selection"] is False
        assert man["selected_model_path"] is None  # resolver falls through to final
        assert man["dropped_non_install"] is False

    # The resolver for a marker cell merges the FINAL adapter (no checkpoint).
    captured = {}

    def _fake_merge(adapter_dir, cell):
        captured["path"] = str(adapter_dir)
        return str(adapter_dir)

    monkeypatch.setattr(mod, "_merge_adapter_for_read", _fake_merge)
    resolved = mod._resolve_read_model_path(marker, tmp_path)
    assert resolved.endswith(marker.cell_id)  # final adapter dir, not a checkpoint


def test_cpu_stub_select_is_synthetic_first_step(tmp_path):
    """CPU-stub select: the first dose step 'clears' synthetically (no GPU probe),
    manifest is not dropped — exercises the plumbing without a real read."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(behavior="em", source="florist", rung="r16", seed=i653.HEADLINE_SEED)
    res = mod.phase_select_checkpoint([cell], out_root=tmp_path, mode=i653.RUN_MODE_CPU_STUB)
    assert res["n_selected"] == 1
    man = json.loads(
        (tmp_path / "armB" / "selected_checkpoints" / f"{cell.cell_id}.json").read_text()
    )
    assert man["dropped_non_install"] is False
    assert man["selected_checkpoint_step"] == 40  # first EM dose step
