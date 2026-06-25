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

import errno
import importlib.util
import json
import sys
from pathlib import Path

import pytest

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


# ── #653 round 5: exception-safety + completeness-sentinel + path-containment ──


def test_select_cleans_partial_merge_on_merge_exception(tmp_path, monkeypatch):
    """BLOCKER 1 (round 5): a merge-time EDQUOT (the EXACT round-3 crash spot) must
    NOT leave a partial merge behind that a relaunch silently accepts as valid. The
    merge is now INSIDE the select probe's try/finally, so the finally's
    ``_delete_merged_for_read`` frees a partial the merge left on raise.

    The stub mimics the round-4 failure mode: a ``save_pretrained`` that EDQUOTs
    mid-write leaves a PARTIAL ``merged_for_read/`` on disk and raises WITHOUT
    self-cleaning (so the test isolates the PHASE's exception-safety, not the
    merge helper's own inner finally). Pre-fix (merge outside try) this partial
    leaks; post-fix (merge inside try/finally) it is freed.

    FAILS on the round-4 code (merge call before the try), PASSES on round 5."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    adapter_dir = _stage_checkpoints(tmp_path, cell, [5, 10, 130])

    def _edquot_merge(adapter_dir, cell):
        # save_pretrained EDQUOTs mid-write: a partial merged_for_read/ is on disk
        # and we raise WITHOUT cleaning it (the worst case the phase must absorb).
        merged = Path(adapter_dir) / "merged_for_read"
        merged.mkdir(parents=True, exist_ok=True)
        (merged / "model-00001.safetensors").write_text("x" * 16)  # partial blob
        raise OSError(errno.EDQUOT, "Disk quota exceeded")

    monkeypatch.setattr(mod, "_merge_adapter_for_read", _edquot_merge)
    monkeypatch.setattr(mod, "_install_content_gpu", lambda *a, **k: {"judge_rate_gain": 0.5})

    with pytest.raises(OSError) as exc:
        mod.phase_select_checkpoint([cell], out_root=tmp_path, mode=i653.RUN_MODE_GPU)
    assert exc.value.errno == errno.EDQUOT
    # The phase's try/finally freed the partial merge left by the failed merge.
    assert _count_merge_dirs(adapter_dir) == 0, "partial merged_for_read/ left after EDQUOT"
    assert not list(adapter_dir.rglob("merged_for_read.tmp")), ".tmp/ partial left after EDQUOT"


def test_merge_adapter_atomic_rename_on_partial_tmp_resume(tmp_path, monkeypatch):
    """Completeness sentinel (round 5): a leaked ``merged_for_read.tmp/`` from a
    prior crashed merge does NOT short-circuit — ``_merge_adapter_for_read`` re-
    merges from scratch (the partial is discarded) and the final ``merged_for_read/``
    is the FRESH merge, promoted atomically. We stub the heavy HF/peft load so this
    runs on CPU: the stub writes a fresh sentinel into the ``.tmp/`` it is handed."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    ckpt = tmp_path / "armB" / "adapters" / cell.cell_id / "checkpoint-10"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text("{}")  # adapter_dir.exists() True

    # Stage a STALE partial .tmp/ a prior crash left (must be discarded, not promoted).
    stale_tmp = ckpt / "merged_for_read.tmp"
    stale_tmp.mkdir(parents=True, exist_ok=True)
    (stale_tmp / "STALE_PARTIAL").write_text("stale")

    # Stub the real HF/peft merge body: capture the tmp path it writes into and
    # drop a FRESH sentinel there (so we can prove the final dir is the fresh one).
    class _FakeModel:
        def save_pretrained(self, path):
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "FRESH").write_text("fresh-merge")

    class _FakeTok:
        def save_pretrained(self, path):
            pass

    fake_transformers = type(sys)("transformers")
    fake_transformers.AutoModelForCausalLM = type(
        "AMF", (), {"from_pretrained": staticmethod(lambda *a, **k: object())}
    )
    fake_transformers.AutoTokenizer = type(
        "AT", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeTok())}
    )
    fake_peft = type(sys)("peft")
    fake_peft.PeftModel = type(
        "PM",
        (),
        {
            "from_pretrained": staticmethod(
                lambda *a, **k: type("M", (), {"merge_and_unload": lambda self: _FakeModel()})()
            )
        },
    )
    fake_torch = type(sys)("torch")
    fake_torch.bfloat16 = "bf16"
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    out = mod._merge_adapter_for_read(ckpt, cell)
    merged = ckpt / "merged_for_read"
    assert Path(out) == merged
    assert merged.exists()
    assert (merged / "FRESH").exists(), "final dir must be the FRESH re-merge"
    assert not (merged / "STALE_PARTIAL").exists(), "stale partial must NOT be promoted"
    assert not stale_tmp.exists(), "the .tmp/ must be renamed away (atomic promote)"


def test_dx_frees_merge_on_read_exception(tmp_path, monkeypatch):
    """MAJOR 1 (round 5): a mid-cell GPU read failure in phase_dx must still free the
    cell's selected merge (the cleanup is in a finally, not normal-path only). We
    stage a selected-checkpoint manifest + its merge, stub the GPU read to first
    confirm the merge exists then raise; assert the phase raises AND the merge is
    gone."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    sel = tmp_path / "armB" / "adapters" / cell.cell_id / "checkpoint-10"
    merged = sel / "merged_for_read"
    merged.mkdir(parents=True, exist_ok=True)
    (merged / "model.safetensors").write_text("merged")
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
            }
        )
    )

    def _boom(c, *, out_root):
        assert merged.exists()  # merge resolved before the read
        raise RuntimeError("simulated GPU read failure")

    monkeypatch.setattr(mod, "_dx_gpu_cloud", _boom)

    with pytest.raises(RuntimeError, match="simulated GPU read failure"):
        mod.phase_dx([cell], out_root=tmp_path, mode=i653.RUN_MODE_GPU)
    assert not merged.exists(), "phase_dx must free the selected merge even on a read raise"


def test_install_frees_merge_on_read_exception(tmp_path, monkeypatch):
    """MAJOR 1 (round 5): the same exception-safe cleanup for phase_install."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    sel = tmp_path / "armB" / "adapters" / cell.cell_id / "checkpoint-10"
    merged = sel / "merged_for_read"
    merged.mkdir(parents=True, exist_ok=True)
    (merged / "model.safetensors").write_text("merged")
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
            }
        )
    )

    def _boom(c, *, out_root):
        assert merged.exists()
        raise RuntimeError("simulated install read failure")

    monkeypatch.setattr(mod, "_install_content_gpu", _boom)

    with pytest.raises(RuntimeError, match="simulated install read failure"):
        mod.phase_install([cell], out_root=tmp_path, mode=i653.RUN_MODE_GPU)
    assert not merged.exists(), "phase_install must free the merge even on a read raise"


def test_ablation_frees_merge_on_read_exception(tmp_path, monkeypatch):
    """MAJOR 1 (round 5): the same exception-safe cleanup for phase_ablation."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung=i653.ABLATION_RUNG, seed=i653.HEADLINE_SEED
    )
    sel = tmp_path / "armB" / "adapters" / cell.cell_id / "checkpoint-10"
    merged = sel / "merged_for_read"
    merged.mkdir(parents=True, exist_ok=True)
    (merged / "model.safetensors").write_text("merged")
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
            }
        )
    )

    def _boom(c, *, out_root):
        assert merged.exists()
        raise RuntimeError("simulated ablation read failure")

    monkeypatch.setattr(mod, "_ablation_gpu_read", _boom)

    with pytest.raises(RuntimeError, match="simulated ablation read failure"):
        mod.phase_ablation([cell], out_root=tmp_path, mode=i653.RUN_MODE_GPU)
    assert not merged.exists(), "phase_ablation must free the merge even on a read raise"


def test_delete_read_merge_for_cell_rejects_out_of_tree_manifest(tmp_path):
    """MAJOR 2 (round 5): a corrupt/hand-edited manifest whose
    ``selected_checkpoint_dir`` points OUTSIDE the cell's adapter tree must NOT
    drive shutil.rmtree there. The helper raises and the victim merge survives."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    # A victim merge dir OUTSIDE the run's out_root.
    victim = tmp_path / "victim" / "checkpoint-5" / "merged_for_read"
    victim.mkdir(parents=True, exist_ok=True)
    (victim / "precious.safetensors").write_text("DO NOT DELETE")

    man_dir = tmp_path / "armB" / "selected_checkpoints"
    man_dir.mkdir(parents=True, exist_ok=True)
    (man_dir / f"{cell.cell_id}.json").write_text(
        json.dumps(
            {
                "cell_id": cell.cell_id,
                "dose_selection": True,
                "selected_checkpoint_step": 5,
                "selected_checkpoint_dir": str(tmp_path / "victim" / "checkpoint-5"),
                "selected_model_path": None,
                "dropped_non_install": False,
            }
        )
    )
    with pytest.raises(RuntimeError, match="NOT contained in the cell adapter tree"):
        mod._delete_read_merge_for_cell(cell, tmp_path)
    assert victim.exists(), "out-of-tree victim merge must survive a poisoned manifest"
    assert (victim / "precious.safetensors").exists()


def test_delete_read_merge_for_cell_rejects_non_checkpoint_name(tmp_path):
    """MAJOR 2 (round 5): even a path INSIDE the cell tree but NOT named
    ``checkpoint-*`` is refused (a poisoned manifest can't point at e.g. the
    adapter root or a sibling dir)."""
    mod = _load_dispatcher()
    cell = i653.ArmBCell(
        behavior="sycophancy", source="florist", rung="r16", seed=i653.HEADLINE_SEED
    )
    cell_dir = tmp_path / "armB" / "adapters" / cell.cell_id
    bad = cell_dir / "not_a_checkpoint"
    (bad / "merged_for_read").mkdir(parents=True, exist_ok=True)
    man_dir = tmp_path / "armB" / "selected_checkpoints"
    man_dir.mkdir(parents=True, exist_ok=True)
    (man_dir / f"{cell.cell_id}.json").write_text(
        json.dumps(
            {
                "cell_id": cell.cell_id,
                "dose_selection": True,
                "selected_checkpoint_step": 5,
                "selected_checkpoint_dir": str(bad),
                "selected_model_path": None,
                "dropped_non_install": False,
            }
        )
    )
    with pytest.raises(RuntimeError, match="NOT contained in the cell adapter tree"):
        mod._delete_read_merge_for_cell(cell, tmp_path)
    assert (bad / "merged_for_read").exists()
