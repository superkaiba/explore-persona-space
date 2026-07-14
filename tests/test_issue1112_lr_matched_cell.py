"""#1112 lr-matched method pair (plan v8, followup `lr-matched-method-pair`).

Pins the ONE-variable delta that adds the `s5_lora_neg_lr5e6` ladder cell:

1. Config-builder: the new cell's built ``TrainLoraConfig`` carries lr 5e-6 +
   max_steps 60 (save cadence 2 / max_length 2048 unchanged).
2. Single-variable guard: EVERY existing cell's built config is BYTE-UNCHANGED
   vs the pre-delta code path (replicated verbatim below), and the regime key
   of any run WITHOUT the new cell is byte-identical to the pre-delta shape.
3. Prefix-collision guard: `s5_lora_neg_lr5e6` shares the "s5"/"s5_lora_"
   prefix with the generic control `s5_lora_generic` — every routing site must
   stay exact-match (no `startswith("s5...")` keying anywhere in the
   dispatcher or geometry drivers), and the capture resolver's FT-prefix
   branch must not capture the new LoRA cell.
4. Fail-loud capture membership: an unregistered cell raises instead of
   silently skipping capture (plan v8 §12.1 + critic note).
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1112_dispatch as d  # noqa: E402

from explore_persona_space.artifacts.recipe import build_train_config, recipe_for  # noqa: E402
from explore_persona_space.experiments import issue_1112 as C  # noqa: E402

# Cells built through _syco_lora_config on the PRE-delta code path, with the
# max_steps values the dispatcher actually passes there (phase_train ceiling;
# phase_generic method-matched step).
EXISTING_SYCO_LORA_CELLS = ("s2_lora_pos", "s5_lora_generic")


def _cfg(tmp_path: Path, cells: tuple[str, ...], *, smoke: bool = False, **kw) -> d.Cfg:
    return d.Cfg(smoke=smoke, cells=cells, out_root=tmp_path, **kw)


def _pre_delta_syco_lora_config(cfg: d.Cfg, cell: str, *, max_steps: int) -> object:
    """VERBATIM replica of _syco_lora_config @ branch tip 069dd9549b (pre-delta)
    — the byte-unchanged reference for the single-variable guard."""
    spec = recipe_for(C.SYCO_BEHAVIOR, arm="primary")
    spec = dataclasses.replace(
        spec,
        overrides={
            **spec.overrides,
            "epochs": 16,
            "max_length": C.SYCO_MAX_LENGTH,
        },
    )
    train_cfg = build_train_config(spec, run_name=C.cell_run_name(cell), seed=cfg.seed)
    return dataclasses.replace(
        train_cfg, save_steps=C.SYCO_SAVE_STEPS, max_steps=max_steps, max_length=C.SYCO_MAX_LENGTH
    )


# ── 1. the new cell's built config ───────────────────────────────────────────


def test_lr_matched_cell_built_config_lr_5e6_max_steps_60(tmp_path):
    cfg = _cfg(tmp_path, (C.LR_MATCHED_CELL,))
    ceiling = C.step_ceiling_for(C.LR_MATCHED_CELL)
    assert ceiling == 60  # the parent's registered G1 extension ceiling, a priori
    built = d._syco_lora_config(cfg, C.LR_MATCHED_CELL, max_steps=ceiling)
    assert built.lr == pytest.approx(5e-6)
    assert built.lr == pytest.approx(C.FT_LR)  # matched to the full-FT side
    assert built.max_steps == 60
    assert built.save_steps == C.SYCO_SAVE_STEPS == 2  # cadence unchanged
    assert built.max_length == C.SYCO_MAX_LENGTH == 2048
    # everything EXCEPT lr/max_steps matches the parent recipe byte-exact
    ref = asdict(_pre_delta_syco_lora_config(cfg, C.LR_MATCHED_CELL, max_steps=ceiling))
    got = asdict(built)
    diff = {k for k in ref if ref[k] != got[k]}
    assert diff == {"lr"}, diff


def test_lr_matched_cell_registered_everywhere(tmp_path):
    assert C.LR_MATCHED_CELL in C.ALL_TRAINED_CELLS
    assert C.CELL_MIX[C.LR_MATCHED_CELL] == "c3_frozen"  # same frozen mix as s1
    assert C.LR_MATCHED_CELL in d.LADDER_CELLS
    # tier2/margin membership predicate (startswith("s") + in selections)
    assert C.LR_MATCHED_CELL.startswith("s")
    assert d.resolve_cells(C.LR_MATCHED_CELL, smoke=True) == (C.LR_MATCHED_CELL,)
    # the frozen-mix path resolves through the SAME staged file as s1
    cfg = _cfg(tmp_path, (C.LR_MATCHED_CELL,))
    assert d._mix_path(cfg, C.LR_MATCHED_CELL) == d._mix_path(cfg, "s1_lora_neg")


# ── 2. single-variable guard: existing cells byte-unchanged ─────────────────


@pytest.mark.parametrize("cell", EXISTING_SYCO_LORA_CELLS)
@pytest.mark.parametrize("max_steps", (30, 14, 2))
def test_existing_cells_built_config_byte_unchanged(tmp_path, cell, max_steps):
    cfg = _cfg(tmp_path, (cell,))
    got = asdict(d._syco_lora_config(cfg, cell, max_steps=max_steps))
    ref = asdict(_pre_delta_syco_lora_config(cfg, cell, max_steps=max_steps))
    assert got == ref


def test_existing_cells_ceiling_unchanged():
    for cell in (*C.SYCO_CELLS_NEW, *C.GENERIC_CELLS, C.REUSED_CELL, *C.MARKER_CELLS):
        assert C.step_ceiling_for(cell) == C.SYCO_STEP_CEILING == 30


def test_regime_key_byte_unchanged_without_lr_matched_cell(tmp_path):
    pre_delta_cells = tuple(c for c in C.ALL_TRAINED_CELLS if c != C.LR_MATCHED_CELL)
    key = _cfg(tmp_path, pre_delta_cells).regime_key()
    # the PRE-delta regime dict, verbatim (no cell_step_ceilings key)
    assert key == {
        "issue": C.ISSUE,
        "smoke": False,
        "cells": list(pre_delta_cells),
        "seed": C.SEED,
        "tier1": [5, 3],
        "tier2": [10, 5],
        "eval_question_limit": None,
        "band": list(d.JUDGED_RATE_BAND),
        "marker_band": list(C.MARKER_BAND),
        "save_steps": 2,
        "max_length": 2048,
        "step_ceiling": 30,
    }


def test_regime_key_threads_per_cell_ceiling_for_lr_matched_runs(tmp_path):
    key = _cfg(tmp_path, (C.LR_MATCHED_CELL,)).regime_key()
    assert key["cell_step_ceilings"] == {C.LR_MATCHED_CELL: 60}
    assert key["step_ceiling"] == 30  # the default ceiling entry is untouched
    # JSON round-trip stability (ladder.json regime compare)
    assert json.loads(json.dumps(key)) == key


# ── 3. prefix-collision guard (exact-match routing) ─────────────────────────


def test_lr_matched_cell_disjoint_from_every_registry():
    # the hazard this guards: shared "s5_lora_" prefix with the generic control
    assert C.LR_MATCHED_CELL.startswith("s5_lora_")
    assert "s5_lora_generic".startswith("s5_lora_")
    assert C.LR_MATCHED_CELL != "s5_lora_generic"
    assert C.LR_MATCHED_CELL not in C.GENERIC_CELLS
    assert C.LR_MATCHED_CELL not in C.SYCO_CELLS_NEW
    assert C.LR_MATCHED_CELL not in C.MARKER_CELLS
    assert C.LR_MATCHED_CELL != C.REUSED_CELL
    assert len(set(C.ALL_TRAINED_CELLS)) == len(C.ALL_TRAINED_CELLS)


def test_capture_resolver_ft_prefix_branch_excludes_lr_matched_cell():
    # _resolve_capture_model routes FT cells by startswith(("s3","s4","s6","m2"))
    # — the lr-matched LoRA cell must fall through to the adapter-merge path.
    ft_prefixes = ("s3", "s4", "s6", "m2")
    assert not C.LR_MATCHED_CELL.startswith(ft_prefixes)
    for ft_cell in ("s3_fullft_neg", "s4_fullft_pos", "s6_fullft_generic", "m2_fullft_band8"):
        assert ft_cell.startswith(ft_prefixes)


def test_no_s5_prefix_keying_in_dispatch_or_geometry_drivers():
    """No startswith("s5...") cell keying anywhere on the routing surfaces —
    all s5-class routing is exact-match (the misroute this round's cell id
    could otherwise silently hit)."""
    for rel in (
        "scripts/issue1112_dispatch.py",
        "scripts/issue1112_geometry.py",
        "scripts/issue1112_figures.py",
        "src/explore_persona_space/experiments/issue_1112/geometry.py",
    ):
        src = (REPO_ROOT / rel).read_text()
        for m in re.finditer(r"startswith\(([^)]*)\)", src):
            args = m.group(1)
            assert '"s5' not in args and "'s5" not in args, (rel, m.group(0))


def test_capture_passes_exact_match_routing(tmp_path):
    # lr-matched cell: selected-dose ONLY + the shared sycophancy base pass
    passes = d.capture_passes(_cfg(tmp_path, (C.LR_MATCHED_CELL,)))
    assert passes == [(C.LR_MATCHED_CELL, "selected"), ("base_sycophancy", "base")]
    # the generic control's routing is unchanged (no prefix cross-capture)
    passes_generic = d.capture_passes(_cfg(tmp_path, ("s5_lora_generic",)))
    assert passes_generic == [("s5_lora_generic", "selected"), ("base_sycophancy", "base")]
    # full registry: the new cell appears exactly once, selected-dose only
    all_passes = d.capture_passes(_cfg(tmp_path, C.ALL_TRAINED_CELLS))
    mine = [p for p in all_passes if p[0] == C.LR_MATCHED_CELL]
    assert mine == [(C.LR_MATCHED_CELL, "selected")]


# ── 4. fail-loud capture membership ──────────────────────────────────────────


def test_capture_passes_fail_loud_on_unregistered_cell(tmp_path):
    cfg = _cfg(tmp_path, ("s7_never_registered",))  # bypasses resolve_cells
    with pytest.raises(ValueError, match="unroutable cell"):
        d.capture_passes(cfg)


# ── routing through the real phase bodies (GPU boundary faked) ───────────────


def test_phase_train_routes_lr_matched_cell_with_ceiling_60(monkeypatch, tmp_path):
    """phase_train's REAL body routes the new cell into the LoRA-ladder branch
    with max_steps 60 and writes the run-log override note into the cell's
    build record; only the GPU-training boundary is faked (signature-mirrored)."""
    calls: dict = {}

    def fake_syco_cfg(cfg, cell, *, max_steps):
        calls["config"] = (cell, max_steps)
        return object()

    def fake_train_lora_cell(cfg, cell, train_cfg):
        calls["trained"] = cell
        return {"adapter_root": str(tmp_path / cell / "train"), "training_loss": 0.0}

    monkeypatch.setattr(d, "_syco_lora_config", fake_syco_cfg)
    monkeypatch.setattr(d, "_train_lora_cell", fake_train_lora_cell)
    cfg = _cfg(tmp_path, (C.LR_MATCHED_CELL,), upload=False)
    d.phase_train(cfg)
    assert calls["config"] == (C.LR_MATCHED_CELL, 60)
    assert calls["trained"] == C.LR_MATCHED_CELL
    rec = json.loads((tmp_path / C.LR_MATCHED_CELL / "build_result.json").read_text())
    assert rec["cell_overrides"]["train_overrides"] == {"lr": C.FT_LR}
    assert rec["cell_overrides"]["step_ceiling"] == 60
    assert "cosine lr schedule" in rec["cell_overrides"]["note"]
    assert rec["mix"].endswith("c3_frozen_mix.jsonl")


def test_phase_train_existing_lora_cell_keeps_ceiling_30(monkeypatch, tmp_path):
    calls: dict = {}

    def fake_syco_cfg(cfg, cell, *, max_steps):
        calls["config"] = (cell, max_steps)
        return object()

    monkeypatch.setattr(d, "_syco_lora_config", fake_syco_cfg)
    monkeypatch.setattr(
        d,
        "_train_lora_cell",
        lambda cfg, cell, train_cfg: {"adapter_root": str(tmp_path / cell / "train")},
    )
    d.phase_train(_cfg(tmp_path, ("s2_lora_pos",), upload=False))
    assert calls["config"] == ("s2_lora_pos", 30)
    rec = json.loads((tmp_path / "s2_lora_pos" / "build_result.json").read_text())
    assert "cell_overrides" not in rec  # no run-log note on unchanged cells


def test_phase_stage_stages_parent_base_store_production_only(monkeypatch, tmp_path):
    """Production lr-matched runs stage the PARENT round's base_sycophancy
    pooled store to the capture exists-check path (plan v8 §4.6 — same base
    store as every parent cell; never a fresh-hardware Hub clobber); smoke
    keeps capturing its own tiny base (row_meta pairing at 4 rows)."""
    staged: list = []

    def fake_stage(path_in_repo, dest, *, revision, sha256=None):
        staged.append((path_in_repo, str(dest), revision))
        return dest

    monkeypatch.setattr(d, "_stage_file", fake_stage)
    d.phase_stage(_cfg(tmp_path / "full", (C.LR_MATCHED_CELL,)))
    base = [s for s in staged if s[0] == C.BASE_SYCO_POOLED_PATH]
    dest = tmp_path / "full" / "capture" / "base_sycophancy" / "base" / "pooled.pt"
    assert base == [(C.BASE_SYCO_POOLED_PATH, str(dest), C.PARENT_CAPTURE_REV)]

    staged.clear()
    d.phase_stage(_cfg(tmp_path / "smoke", (C.LR_MATCHED_CELL,), smoke=True))
    assert not [s for s in staged if s[0] == C.BASE_SYCO_POOLED_PATH]


def test_phase_upload_lr_matched_uploads_selected_rung_only(monkeypatch, tmp_path):
    """The overflow upload for the new cell ships ONLY the selected rung
    (plan v8 §10 — non-selected rungs are the declared discard), while the
    existing LoRA cells' whole-ladder upload stays unchanged."""
    cell = C.LR_MATCHED_CELL
    cell_root = tmp_path / cell
    train_dir = cell_root / "train"
    sel_ckpt = train_dir / "checkpoint-24"
    sel_ckpt.mkdir(parents=True)
    (train_dir / "checkpoint-30").mkdir()
    (cell_root / "build_result.json").write_text(json.dumps({"adapter_root": str(train_dir)}))
    (cell_root / "selection.json").write_text(json.dumps({"step": 24}))
    uploads: list = []

    def fake_upload(local, repo, repo_type, path_in_repo, **kw):
        uploads.append((str(local), repo, path_in_repo))
        return f"https://hf.co/{path_in_repo}"

    monkeypatch.setattr(d.hub, "_upload", fake_upload)
    d.phase_upload(_cfg(tmp_path, (cell,), upload=True))
    overflow = [u for u in uploads if u[1] == C.OVERFLOW_REPO]
    assert overflow == [(str(sel_ckpt), C.OVERFLOW_REPO, f"issue1112/{cell}/checkpoint-24")]
