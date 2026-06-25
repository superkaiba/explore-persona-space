"""Task #653 round 4 — select_checkpoint disk cleanup + resume-skip invariants.

CPU-only regression tests for the MooseFS per-pod ~130 GB EDQUOT crash that
killed the round-3 production run (epm:failure v5). Each probed dose checkpoint
got a full-precision ~15 GB ``merged_for_read/`` that was NEVER deleted; with up
to 9 dose checkpoints x 12 content cells the worst-case merge demand is ~1.6 TB.

These tests pin the two BLOCKER fixes (they FAIL on the round-3 code, PASS on the
round-4 fix):

  * BLOCKER 1 (cleanup-as-you-go): after probing N dose checkpoints of one cell,
    AT MOST ONE ``merged_for_read/`` exists on disk (the strict-immediate-delete
    pattern frees each probe's merge the moment its install read returns; the
    selected checkpoint re-merges on demand downstream). The pre-fix code left
    one merge PER probed checkpoint.
  * BLOCKER 2 (resume skip): re-running phase_select_checkpoint on a cell whose
    manifest already exists SKIPS it WITHOUT re-probing (no merge, no install
    call) — so a relaunch does not redo the completed cells (and does not
    re-explode disk on them).

The merge stub CREATES a real ``merged_for_read/`` dir on disk (mimicking
``_merge_adapter_for_read``) so the disk-cleanup behavior is exercised end-to-end,
not mocked away.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from explore_persona_space.experiments import issue_653 as i653


def _load_dispatcher():
    repo_root = Path(__file__).resolve().parents[1]
    disp_path = repo_root / "scripts" / "issue_653" / "i653_dispatch.py"
    spec = importlib.util.spec_from_file_location("i653_dispatch_disk_test", disp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["i653_dispatch_disk_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _stage_checkpoints(out_root: Path, cell, steps: list[int]) -> Path:
    adapter_dir = out_root / "armB" / "adapters" / cell.cell_id
    for s in steps:
        (adapter_dir / f"checkpoint-{s}").mkdir(parents=True, exist_ok=True)
    return adapter_dir


def _count_merge_dirs(adapter_dir: Path) -> int:
    """Number of ``merged_for_read/`` dirs anywhere under the cell's adapter tree
    (the final adapter's own + every ``checkpoint-<step>/merged_for_read/``)."""
    return len([p for p in adapter_dir.rglob("merged_for_read") if p.is_dir()])


def _patch_real_merge_and_probe(mod, monkeypatch, *, pass_at_step, install_calls):
    """Stub the merge so it CREATES a real ``merged_for_read/`` on disk (each ~a
    few bytes here, standing in for the ~15 GB production dir), and a probe that
    parses the checkpoint step from the merged path. ``install_calls`` accumulates
    the probed checkpoint steps so the resume test can assert zero probes."""

    def _fake_merge(adapter_dir, cell):
        merged = Path(adapter_dir) / "merged_for_read"
        merged.mkdir(parents=True, exist_ok=True)
        (merged / "model.safetensors").write_text("x" * 16)  # stand-in weight blob
        return str(merged)

    def _fake_install(cell, *, out_root, trained_path=None):
        # trained_path is .../checkpoint-<step>/merged_for_read → parent is the ckpt
        step = int(Path(trained_path).parent.name.split("checkpoint-", 1)[1])
        install_calls.append(step)
        gain = 0.5 if (pass_at_step is not None and step >= pass_at_step) else 0.05
        return {
            "dv_kind": "judge_rate_plus_gain",
            "behavior": cell.behavior,
            "judge_rate_gain": gain,
            "continuous_gain_logp": 0.3,
        }

    monkeypatch.setattr(mod, "_merge_adapter_for_read", _fake_merge)
    monkeypatch.setattr(mod, "_install_content_gpu", _fake_install)


def test_select_leaves_at_most_one_merge_on_disk(tmp_path, monkeypatch):
    """BLOCKER 1: probing 4 dose checkpoints leaves AT MOST 1 merge on disk (the
    pre-fix code left 1 per probed checkpoint = 4). With the strict-immediate-
    delete pattern, NO probe merge survives — the selected checkpoint re-merges
    on demand downstream."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    adapter_dir = _stage_checkpoints(tmp_path, cell, [5, 10, 15, 130])
    install_calls: list[int] = []
    # Never clears until the LAST staged checkpoint so every probe runs (worst case
    # for disk accumulation): step 5/10/15 fail, the final dose-endpoint clears.
    _patch_real_merge_and_probe(mod, monkeypatch, pass_at_step=130, install_calls=install_calls)

    res = mod.phase_select_checkpoint([cell], out_root=tmp_path, mode=i653.RUN_MODE_GPU)
    assert res["n_selected"] == 1
    # Every probed checkpoint's merge was freed: at most one merge dir survives.
    # (Strict-immediate-delete → exactly zero; the assertion is the upper bound.)
    n_left = _count_merge_dirs(adapter_dir)
    assert n_left <= 1, f"expected ≤1 merge dir on disk after select, found {n_left}"
    # And we actually probed multiple checkpoints (the cleanup ran per probe, not
    # because nothing was probed).
    assert len(install_calls) >= 3, install_calls


def test_dropped_cell_leaves_no_merge_on_disk(tmp_path, monkeypatch):
    """BLOCKER 1 (drop path): a cell whose checkpoints never clear the floor is
    DROPPED and leaves NO merge on disk (each probe's merge is freed)."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    adapter_dir = _stage_checkpoints(tmp_path, cell, [5, 10, 15, 130])
    install_calls: list[int] = []
    _patch_real_merge_and_probe(mod, monkeypatch, pass_at_step=None, install_calls=install_calls)

    res = mod.phase_select_checkpoint([cell], out_root=tmp_path, mode=i653.RUN_MODE_GPU)
    assert res["n_dropped_non_install"] == 1
    assert _count_merge_dirs(adapter_dir) == 0
    assert len(install_calls) >= 3  # all checkpoints probed, none cleared


def test_resume_skips_cell_with_existing_manifest(tmp_path, monkeypatch):
    """BLOCKER 2: a re-entered select_checkpoint SKIPS a cell whose manifest
    already exists — NO re-probe (zero install calls), NO re-merge. This is the
    relaunch path that must not redo the 7 completed cells (and not re-explode
    disk on them)."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(behavior="em", source="florist", rung="r16", seed=i653.HEADLINE_SEED)
    _stage_checkpoints(tmp_path, cell, [40, 80, 120, 160, 200])

    # Pre-write a manifest as if this cell completed in a prior (crashed) run.
    man_dir = tmp_path / "armB" / "selected_checkpoints"
    man_dir.mkdir(parents=True, exist_ok=True)
    pre = {
        "cell_id": cell.cell_id,
        "behavior": cell.behavior,
        "rung": cell.rung,
        "dose_selection": True,
        "selected_checkpoint_step": 80,
        "selected_checkpoint_dir": str(
            tmp_path / "armB" / "adapters" / cell.cell_id / "checkpoint-80"
        ),
        "selected_model_path": None,
        "dropped_non_install": False,
    }
    (man_dir / f"{cell.cell_id}.json").write_text(json.dumps(pre, indent=1))

    install_calls: list[int] = []
    _patch_real_merge_and_probe(mod, monkeypatch, pass_at_step=80, install_calls=install_calls)

    res = mod.phase_select_checkpoint([cell], out_root=tmp_path, mode=i653.RUN_MODE_GPU)
    # Skipped → no probe ran, but the cell still counts as selected in the tally.
    assert install_calls == [], "resume must NOT re-probe a cell with an existing manifest"
    assert res["n_selected"] == 1
    assert res["n_dropped_non_install"] == 0
    # The pre-existing manifest is untouched (still points at checkpoint-80).
    man = json.loads((man_dir / f"{cell.cell_id}.json").read_text())
    assert man["selected_checkpoint_step"] == 80


def test_resume_skip_sweeps_stale_merges(tmp_path, monkeypatch):
    """BLOCKER 1+2 interaction: the resume skip ALSO sweeps any stale merges a
    prior crashed run left under the completed cell, so a relaunch starts clean."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(behavior="em", source="florist", rung="r16", seed=i653.HEADLINE_SEED)
    adapter_dir = _stage_checkpoints(tmp_path, cell, [40, 80, 120])
    # Simulate orphaned merges a crashed run left behind.
    for s in (40, 80):
        stale = adapter_dir / f"checkpoint-{s}" / "merged_for_read"
        stale.mkdir(parents=True, exist_ok=True)
        (stale / "model.safetensors").write_text("x" * 16)
    assert _count_merge_dirs(adapter_dir) == 2

    man_dir = tmp_path / "armB" / "selected_checkpoints"
    man_dir.mkdir(parents=True, exist_ok=True)
    (man_dir / f"{cell.cell_id}.json").write_text(
        json.dumps(
            {
                "cell_id": cell.cell_id,
                "dose_selection": True,
                "selected_checkpoint_step": 80,
                "selected_checkpoint_dir": str(adapter_dir / "checkpoint-80"),
                "selected_model_path": None,
                "dropped_non_install": False,
            },
            indent=1,
        )
    )
    install_calls: list[int] = []
    _patch_real_merge_and_probe(mod, monkeypatch, pass_at_step=80, install_calls=install_calls)

    mod.phase_select_checkpoint([cell], out_root=tmp_path, mode=i653.RUN_MODE_GPU)
    assert install_calls == []
    assert _count_merge_dirs(adapter_dir) == 0  # stale merges swept on resume


def test_delete_read_merge_for_cell_noop_on_full_ft(tmp_path):
    """The downstream per-cell cleanup helper is a NO-OP for full-FT cells — it
    must NEVER touch the FT checkpoint dir (which is read directly, not merged)."""
    mod = _load_dispatcher()
    full_ft = i653.ArmBCell(behavior="em", source="florist", rung="full", seed=i653.HEADLINE_SEED)
    ft_dir = tmp_path / "armB" / "adapters" / full_ft.cell_id
    ft_dir.mkdir(parents=True, exist_ok=True)
    (ft_dir / "model.safetensors").write_text("FT-weights")
    assert mod._delete_read_merge_for_cell(full_ft, tmp_path) is False
    assert (ft_dir / "model.safetensors").exists()  # FT dir untouched


def test_delete_read_merge_for_cell_frees_selected_dose_merge(tmp_path):
    """The downstream per-cell cleanup helper frees the SELECTED checkpoint's
    merge for a dose cell (the dx/install/ablation per-cell cleanup that keeps
    ≤1 merge resident across the 12-cell phase)."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    adapter_dir = tmp_path / "armB" / "adapters" / cell.cell_id
    sel = adapter_dir / "checkpoint-10"
    merged = sel / "merged_for_read"
    merged.mkdir(parents=True, exist_ok=True)
    (merged / "model.safetensors").write_text("merged-weights")

    man_dir = tmp_path / "armB" / "selected_checkpoints"
    man_dir.mkdir(parents=True, exist_ok=True)
    (man_dir / f"{cell.cell_id}.json").write_text(
        json.dumps(
            {
                "cell_id": cell.cell_id,
                "dose_selection": True,
                "selected_checkpoint_step": 10,
                "selected_checkpoint_dir": str(sel),
                "selected_model_path": None,
                "dropped_non_install": False,
            },
            indent=1,
        )
    )
    assert mod._delete_read_merge_for_cell(cell, tmp_path) is True
    assert not merged.exists()
