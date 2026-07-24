# em-dash intentional
"""#1586 FU round (caveat-fix-marker-dosematch-impolite-lr-deconfound, plan v7).

CPU pins for the FU registry (issue1586_cells), the --fu dispatcher surface
(issue1586_dispatch), the chunked coarse-then-fine marker read scheduler, the
p2l factory-recipe config, the fu persist/tier2/panel branches, and the
off-pod pair naming (geometry + lattice). GPU/Hub boundaries are faked
signature-conformant (create_autospec / real-signature fakes); every
round-added function stubbed somewhere here also has a body-executing test
(code-style § one production-body test per seam-stubbed function).
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1586_cells as G  # noqa: E402
import issue1586_dispatch as d  # noqa: E402
import issue1586_geometry as geo1586  # noqa: E402
import issue1586_leakage_lattice as lat  # noqa: E402

from explore_persona_space.experiments import issue_1112 as P1112  # noqa: E402

DOSE_LABELS = REPO_ROOT / "figures" / "issue_1586" / "dose_labels.json"


def _cfg(tmp_path, **kw):
    kw.setdefault("smoke", False)
    kw.setdefault("cells", G.FU_ALL_CELLS)
    kw.setdefault("out_root", tmp_path / "out_fu")
    kw.setdefault("ladder_disk_mode", "stream-reap")
    kw.setdefault("fu", "caveatfix")
    return d.Cfg(**kw)


# ── FU registry (issue1586_cells) ────────────────────────────────────────────


def test_fu_registry_invariants():
    assert len(G.FU_ALL_CELLS) == 6
    assert not set(G.FU_ALL_CELLS) & set(G.ALL_FT_CELLS)
    for fc in G.FU_MARKER_FT_CELLS:
        assert G.parse_ft_cell(fc.cell) == ("mk", fc.regime, fc.seed)
        assert G.cell_method(fc.cell) == "ft2e6"
        arm = G.lora_pair_of(fc.cell)
        assert arm.cell == fc.paired_lora_cell and arm.recipe_class == "marker"
    for ic in G.FU_IMP_LORA_CELLS:
        assert G.parse_ft_cell(ic.cell) == ("imp", "con", ic.seed)
        assert G.cell_method(ic.cell) == "lora5e6"
        assert G.is_fu_lora_cell(ic.cell)
        with pytest.raises(ValueError):
            G.lora_pair_of(ic.cell)


def test_fu_lrs_are_imported_constants_never_retyped():
    # identity with the canonical modules (plan §11: imported, never retyped)
    assert G.FU_MARKER_FT_LR == P1112.MARKER_FT_FALLBACK_LR == 2e-6
    assert G.FU_IMP_LORA_LR == P1112.FT_LR
    assert P1112.CELL_TRAIN_OVERRIDES[P1112.LR_MATCHED_CELL]["lr"] == G.FU_IMP_LORA_LR
    import issue1090_fu4 as fu4

    assert G.FU_IMP_SAVE_STEPS == fu4.FU4_SAVE_STEPS == 5


def test_fu_smoke_cells_one_per_arm_class():
    methods = {G.cell_method(c) for c in G.FU_SMOKE_CELLS}
    assert methods == {"ft2e6", "lora5e6"}  # #1586 r3-r6 class-coverage lesson


def test_load_fu_dose_labels_real_committed_file():
    labels = G.load_fu_dose_labels(DOSE_LABELS)
    assert set(labels) == {"imp-con-s42", "imp-con-s137"}
    for v in labels.values():
        assert 0.0 < v < 1.0


def test_load_fu_dose_labels_fail_loud(tmp_path):
    bad = tmp_path / "dose.json"
    bad.write_text(json.dumps({"imp-con-s42": {"ft_dose": 0.66}}))  # s137 missing
    with pytest.raises(RuntimeError, match="missing ft_dose"):
        G.load_fu_dose_labels(bad)
    bad.write_text(json.dumps({"imp-con-s42": {"ft_dose": 1.5}, "imp-con-s137": {"ft_dose": 0.6}}))
    with pytest.raises(RuntimeError, match="out of"):
        G.load_fu_dose_labels(bad)


def test_fu_pair_dose_label():
    ok = G.fu_pair_dose_label(0.70, 0.66)
    assert ok["dose_matched"] and ok["rate_gap"] == pytest.approx(0.04)
    off_gap = G.fu_pair_dose_label(0.80, 0.66)
    assert not off_gap["dose_matched"]  # gap 0.14 > 0.10
    off_band = G.fu_pair_dose_label(0.55, 0.60)
    assert not off_band["dose_matched"]  # below band even though gap <= 0.10


def test_fu_diff_pairs_for_both_rounds_and_presence_filter():
    mk_arms = ["mk-pers-ft2e6-con-s42", "mk-pers-lora-con-s42", "mk-pers-ft2e6-po-s137"]
    pairs = G.fu_diff_pairs_for("mk", mk_arms)
    assert pairs == (
        ("mk-pers-ft2e6-con-s42__ft_vs_lora", "mk-pers-ft2e6-con-s42", "mk-pers-lora-con-s42"),
    )  # po-s137 pair dropped: its lora arm is absent
    imp_arms = ["imp-pers-ft-con-s137", "imp-pers-lora5e6-con-s137"]
    pairs = G.fu_diff_pairs_for("imp", imp_arms)
    assert pairs == (
        (
            "imp-pers-lora5e6-con-s137__ft_vs_lora",
            "imp-pers-ft-con-s137",
            "imp-pers-lora5e6-con-s137",
        ),
    )
    assert G.fu_diff_pairs_for("syc", ["syc-pers-ft-con-s42"]) == ()


# ── resolver / cfg / unit-arg threading ──────────────────────────────────────


def test_resolve_cells_fu_universe_and_smoke():
    assert d.resolve_cells(None, False, "caveatfix") == G.FU_ALL_CELLS
    assert d.resolve_cells(None, True, "caveatfix") == G.FU_SMOKE_CELLS
    sub = d.resolve_cells("imp-pers-lora5e6-con-s42", False, "caveatfix")
    assert sub == ("imp-pers-lora5e6-con-s42",)
    with pytest.raises(ValueError):
        d.resolve_cells("syc-pers-ft-con-s42", False, "caveatfix")  # executed id not in fu
    with pytest.raises(ValueError):
        d.resolve_cells("mk-pers-ft2e6-con-s42", False, None)  # fu id not in executed


def test_unit_args_thread_fu_and_regime_key(tmp_path):
    cfg = _cfg(tmp_path)
    ua = d._unit_args(cfg, "mkread", "mk-pers-ft2e6-con-s42:7")
    i = ua.index("--fu")
    assert ua[i + 1] == "caveatfix"
    assert cfg.regime_key()["fu"] == "caveatfix"
    cfg0 = d.Cfg(
        smoke=False,
        cells=("syc-pers-ft-con-s42",),
        out_root=tmp_path / "o",
        ladder_disk_mode="stream-reap",
    )
    assert "--fu" not in d._unit_args(cfg0, "ladder", "syc-pers-ft-con-s42")
    assert cfg0.regime_key()["fu"] is None


def test_build_cfg_fu_roots_distinct(tmp_path):
    args = d._parse_args(["--mode", "full", "--fu", "caveatfix"])
    cfg = d.build_cfg(args)
    assert str(cfg.out_root).endswith("data/issue_1586/out_fu")
    args_s = d._parse_args(["--mode", "smoke", "--fu", "caveatfix"])
    cfg_s = d.build_cfg(args_s)
    assert "-fu-smoke" in str(cfg_s.out_root)  # never the executed smoke root
    args0 = d._parse_args(["--mode", "full"])
    assert str(d.build_cfg(args0).out_root).endswith("data/issue_1586/out")


# ── marker FT cmd: fu lr + fixed horizon + smoke-invariant width ─────────────


def test_marker_ft_cmd_fu_lr_horizon_and_width(tmp_path):
    cfg = _cfg(tmp_path)
    cmd = d._marker_ft_cmd(
        cfg, "mk-pers-ft2e6-con-s42", out_dir=tmp_path / "t", grid=(2, 4), horizon=24
    )
    assert cmd[cmd.index("--learning-rate") + 1] == str(G.FU_MARKER_FT_LR)
    assert cmd[cmd.index("--max-steps") + 1] == "24"
    assert cmd[cmd.index("--num_processes") + 1] == str(d.FT_NUM_PROCESSES)
    # smoke keeps the SAME width (the #1315/#1333 smoke-width pin)
    cfg_s = _cfg(tmp_path, smoke=True, cells=G.FU_SMOKE_CELLS)
    cmd_s = d._marker_ft_cmd(
        cfg_s, "mk-pers-ft2e6-con-s42", out_dir=tmp_path / "t", grid=(2,), horizon=2
    )
    assert cmd_s[cmd_s.index("--num_processes") + 1] == str(d.FT_NUM_PROCESSES)
    # executed cells: legacy lr + max-steps = max(grid), byte-unchanged
    cmd0 = d._marker_ft_cmd(cfg, "mk-pers-ft-con-s42", out_dir=tmp_path / "t", grid=(1, 2, 3))
    assert cmd0[cmd0.index("--learning-rate") + 1] == str(P1112.MARKER_FT_LR)
    assert cmd0[cmd0.index("--max-steps") + 1] == "3"
    with pytest.raises(ValueError, match="horizon"):
        d._marker_ft_cmd(
            cfg, "mk-pers-ft2e6-con-s42", out_dir=tmp_path / "t", grid=(30,), horizon=24
        )


def test_fu_marker_horizon_split():
    assert d._fu_marker_horizon(24) == G.FU_MARKER_STEP_CEILING
    assert d._fu_marker_horizon(25) == G.FU_MARKER_EXT_CEILING


def test_cell_rung_demand_fu():
    assert d._cell_rung_demand_gb("mk-pers-ft2e6-con-s42") == d.FU_MARKER_CHUNK * d.RUNG_GB
    assert d._cell_rung_demand_gb("mk-pers-ft2e6-con-s42", smoke=True) == d.RUNG_GB
    assert d._cell_rung_demand_gb("imp-pers-lora5e6-con-s42") == G.FU_IMP_LADDER_GB


# ── p2l config (factory recipe at FT lr) ─────────────────────────────────────


def test_p2l_train_cfg_factory_recipe_at_ft_lr(tmp_path):
    cfg = _cfg(tmp_path)
    tc = d._p2l_train_cfg(cfg, "imp-pers-lora5e6-con-s137", max_steps=180)
    assert tc.lr == G.FU_IMP_LORA_LR == 5e-6
    assert tc.max_steps == 180
    assert tc.save_steps == G.FU_IMP_SAVE_STEPS == 5
    assert (tc.lora_r, tc.lora_alpha) == (32, 64)  # factory adapter_config grounding
    assert tc.max_length == 2048
    assert tc.seed == 137  # training seed = cell seed (plan §4.B)
    assert tc.run_name == "issue1586_fu_lora_imp-pers-lora5e6-con-s137"


def _fake_adapter_ckpt(d_: Path, step: int) -> Path:
    ck = d_ / f"checkpoint-{step}"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 32,
                "lora_alpha": 64,
                "use_rslora": True,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            }
        )
    )
    return ck


def test_run_p2l_unit_body_with_boundary_fakes(monkeypatch, tmp_path):
    """Body-executing test for run_p2l_unit: REAL body through the config
    builder + gauge + rung checks; ONLY train_lora / release / mix boundary
    faked signature-conformant."""
    from explore_persona_space.train import sft as sft_mod

    cfg = _cfg(tmp_path, cells=("imp-pers-lora5e6-con-s42",))
    mix_dir = cfg.out_root / "inputs" / "mixes"
    mix_dir.mkdir(parents=True)
    (mix_dir / "imp_con.jsonl").write_text('{"prompt": "p", "completion": "c"}\n')

    def fake_train_lora(base_model, train_jsonl, out_dir, cfg=None):
        assert base_model == d.DEFAULT_BASE_MODEL
        assert cfg.max_steps == G.FU_IMP_STEP_CEILING
        out = Path(out_dir)
        for s in range(G.FU_IMP_SAVE_STEPS, G.FU_IMP_STEP_CEILING + 1, G.FU_IMP_SAVE_STEPS):
            _fake_adapter_ckpt(out, s)
        return out, 0.42

    fake = create_autospec(sft_mod.train_lora, side_effect=fake_train_lora)
    monkeypatch.setattr(sft_mod, "train_lora", fake)
    import explore_persona_space.artifacts.organisms as org

    monkeypatch.setattr(
        org, "release_trainer_cuda_memory", create_autospec(org.release_trainer_cuda_memory)
    )
    rec = d.run_p2l_unit(cfg, "imp-pers-lora5e6-con-s42")
    assert rec["status"] == "trained" and rec["lr"] == 5e-6
    assert rec["adapter_gauge"]["r"] == 32 and rec["adapter_gauge"]["use_rslora"] is True
    assert max(rec["rungs"]) == G.FU_IMP_STEP_CEILING
    # resume: second call reads the build record, no retrain
    assert d.run_p2l_unit(cfg, "imp-pers-lora5e6-con-s42") == rec
    assert fake.call_count == 1


def test_run_p2l_unit_incomplete_ladder_raises(monkeypatch, tmp_path):
    from explore_persona_space.train import sft as sft_mod

    cfg = _cfg(tmp_path, cells=("imp-pers-lora5e6-con-s42",))
    mix_dir = cfg.out_root / "inputs" / "mixes"
    mix_dir.mkdir(parents=True)
    (mix_dir / "imp_con.jsonl").write_text('{"prompt": "p", "completion": "c"}\n')

    def fake_train_lora(base_model, train_jsonl, out_dir, cfg=None):
        _fake_adapter_ckpt(Path(out_dir), 5)  # only one rung — ladder incomplete
        return Path(out_dir), 0.1

    monkeypatch.setattr(
        sft_mod, "train_lora", create_autospec(sft_mod.train_lora, side_effect=fake_train_lora)
    )
    import explore_persona_space.artifacts.organisms as org

    monkeypatch.setattr(
        org, "release_trainer_cuda_memory", create_autospec(org.release_trainer_cuda_memory)
    )
    with pytest.raises(RuntimeError, match="ladder incomplete"):
        d.run_p2l_unit(cfg, "imp-pers-lora5e6-con-s42")


def test_run_p2l_ext_unit_grafts_only_extension_rungs(monkeypatch, tmp_path):
    from explore_persona_space.train import sft as sft_mod

    cfg = _cfg(tmp_path, cells=("imp-pers-lora5e6-con-s42",))
    cell_root = cfg.out_root / "imp-pers-lora5e6-con-s42"
    train_dir = cell_root / "train"
    _fake_adapter_ckpt(train_dir, 180)
    d._atomic_json(cell_root / "build_result.json", {"adapter_root": str(train_dir)})

    def fake_train_lora(base_model, train_jsonl, out_dir, cfg=None):
        assert cfg.max_steps == G.FU_IMP_EXT_CEILING
        out = Path(out_dir)
        for s in (175, 180, 185, 360):  # sub-ceiling dupes + extension rungs
            _fake_adapter_ckpt(out, s)
        return out, 0.2

    monkeypatch.setattr(
        sft_mod, "train_lora", create_autospec(sft_mod.train_lora, side_effect=fake_train_lora)
    )
    import explore_persona_space.artifacts.organisms as org

    monkeypatch.setattr(
        org, "release_trainer_cuda_memory", create_autospec(org.release_trainer_cuda_memory)
    )
    mix_dir = cfg.out_root / "inputs" / "mixes"
    mix_dir.mkdir(parents=True)
    (mix_dir / "imp_con.jsonl").write_text("{}\n")
    rec = d.run_p2l_ext_unit(cfg, "imp-pers-lora5e6-con-s42")
    assert rec["moved_steps"] == [185, 360]
    rungs = set(d._rungs_or_empty(train_dir))
    assert {180, 185, 360} <= rungs and 175 not in rungs  # run-A rung untouched
    # sub-ceiling ext scaffolding RETAINED under train_ext (no rung discarded)
    assert set(d._rungs_or_empty(cell_root / "train_ext")) == {175, 180}


# ── chunked coarse-then-fine marker scheduler ────────────────────────────────


def _mk_cell_fixture(cfg, cell):
    cell_root = cfg.out_root / cell
    train_dir = cell_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    d._atomic_json(
        cell_root / "build_result.json",
        {"cell": cell, "adapter_root": str(train_dir)},
    )
    return cell_root, train_dir


def _scheduler_fakes(monkeypatch, cfg, cell, curve):
    """Fake the two GPU boundaries of run_fu_marker_ladder with
    signature-conformant fakes: chunk trains create rung dirs; mkread units
    write slot_read.json from the synthetic ΔG curve. Records the trained
    steps + horizons for assertions."""
    cell_root = cfg.out_root / cell
    trained: list[tuple[tuple[int, ...], int]] = []

    def fake_train(cfg_, cell_, steps, horizon):
        trained.append((tuple(steps), horizon))
        for s in steps:
            (cell_root / "train" / f"checkpoint-{s}").mkdir(parents=True, exist_ok=True)

    def fake_read(cfg_, arg):
        _cell, step_s = arg.rsplit(":", 1)
        step = int(step_s)
        out = cell_root / f"rung{step}"
        out.mkdir(parents=True, exist_ok=True)
        d._atomic_json(
            out / "slot_read.json",
            {
                "delta_logp_mean": curve(step),
                "delta_margin_mean": 0.0,
                "gen_emission_rate": 0.0,
                "argmax_rate": 0.0,
                "n": 2,
            },
        )

    monkeypatch.setattr(
        d,
        "_fu_marker_train_rungs",
        create_autospec(d._fu_marker_train_rungs, side_effect=fake_train),
    )
    monkeypatch.setattr(
        d, "run_fu_mkread_unit", create_autospec(d.run_fu_mkread_unit, side_effect=fake_read)
    )
    monkeypatch.setattr(d, "_n_gpus", lambda: 1)  # serial read path
    return trained


def test_fu_marker_ladder_early_stop_and_floor_refine(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, cells=("mk-pers-ft2e6-con-s42",))
    cell = "mk-pers-ft2e6-con-s42"
    cell_root, train_dir = _mk_cell_fixture(cfg, cell)
    # ΔG ramp: 0 at step<=8, 6.0 at 10, 13.0 at >=12 -> early stop inside chunk 3
    curve = lambda s: 0.0 if s <= 8 else (6.0 if s == 10 else (5.5 if s == 9 else 13.0))  # noqa: E731
    trained = _scheduler_fakes(monkeypatch, cfg, cell, curve)
    done = d.run_fu_marker_ladder(cfg, cell)
    # coarse chunks: {2,4,6,8} then {10,12,14,16}; early-stop before chunk 3
    assert trained[0] == ((2, 4, 6, 8), G.FU_MARKER_STEP_CEILING)
    assert trained[1] == ((10, 12, 14, 16), G.FU_MARKER_STEP_CEILING)
    assert 18 not in done and 24 not in done  # early-stopped
    # floor-straddle refine: bracket (8, 10) straddles floor 5 -> step 9 read
    assert 9 in done and done[9]["delta_logp_mean"] == 5.5
    # reads persisted; rung dirs reaped under stream-reap
    ladder = d._read_json(cell_root / "ladder.json")
    assert set(map(int, ladder["reads_by_step"])) == set(done)
    assert d._rungs_or_empty(train_dir) == {}


def test_fu_marker_ladder_extension_fires_on_low_top(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, cells=("mk-pers-ft2e6-po-s137",))
    cell = "mk-pers-ft2e6-po-s137"
    _mk_cell_fixture(cfg, cell)

    # never reaches the floor by 24 -> extension 26..48 at horizon 48; then a
    # floor crossing at 30 (bracket 28->30 straddles) + early stop at 34
    def curve(s):
        if s <= 28:
            return 1.0
        if s == 29:
            return 5.2
        if s in (30, 32):
            return 8.0
        return 12.5  # >= ceiling from 34 on

    trained = _scheduler_fakes(monkeypatch, cfg, cell, curve)
    done = d.run_fu_marker_ladder(cfg, cell)
    horizons = {h for steps, h in trained for s in steps if s > G.FU_MARKER_STEP_CEILING}
    assert horizons == {G.FU_MARKER_EXT_CEILING}
    assert (cfg.out_root / cell / "extended.json").exists()
    assert 29 in done  # ext-range floor-straddle refine (bracket 28->30)
    # early-stop is CHUNK-granular: the chunk containing the >=ceiling read
    # (34,36,38,40) completes; the NEXT chunk (42..48) never trains.
    assert 40 in done and 42 not in done
    assert max(done) <= G.FU_MARKER_EXT_CEILING


def test_fu_marker_ladder_smoke_single_rung(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, smoke=True, cells=G.FU_SMOKE_CELLS)
    cell = "mk-pers-ft2e6-con-s42"
    _root, train_dir = _mk_cell_fixture(cfg, cell)
    trained = _scheduler_fakes(monkeypatch, cfg, cell, lambda s: 0.5)
    done = d.run_fu_marker_ladder(cfg, cell)
    assert list(done) == [2] and trained == [((2,), 2)]
    assert d._rungs_or_empty(train_dir) != {}  # smoke never reaps


def test_fu_marker_train_rungs_body(monkeypatch, tmp_path):
    """Body test for _fu_marker_train_rungs: real body through the skip +
    headroom + cmd composition; subprocess/headroom boundaries autospec'd."""
    cfg = _cfg(tmp_path, cells=("mk-pers-ft2e6-con-s42",))
    cell = "mk-pers-ft2e6-con-s42"
    _root, train_dir = _mk_cell_fixture(cfg, cell)
    (train_dir / "checkpoint-2").mkdir()  # step 2 already trained -> skipped
    mix_dir = cfg.out_root / "inputs" / "mixes"
    mix_dir.mkdir(parents=True)
    (mix_dir / "mk_con.jsonl").write_text("{}\n")
    monkeypatch.setattr(
        d,
        "assert_out_root_headroom",
        create_autospec(d.assert_out_root_headroom, return_value=100.0),
    )
    fake_run = create_autospec(d._run_subprocess)
    monkeypatch.setattr(d, "_run_subprocess", fake_run)
    monkeypatch.setattr(
        d, "_ft_lane_env", lambda lane: {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "MASTER_PORT": "29500"}
    )
    d._fu_marker_train_rungs(cfg, cell, [2, 4, 6], 24)
    cmd = fake_run.call_args[0][0]
    assert cmd[cmd.index("--ckpt-steps") + 1] == "4,6"  # 2 skipped
    assert cmd[cmd.index("--max-steps") + 1] == "24"
    assert cmd[cmd.index("--learning-rate") + 1] == str(G.FU_MARKER_FT_LR)
    # no-op when everything already trained
    fake_run.reset_mock()
    for s in (4, 6):
        (train_dir / f"checkpoint-{s}").mkdir()
    d._fu_marker_train_rungs(cfg, cell, [2, 4, 6], 24)
    assert not fake_run.called


def test_run_fu_mkread_unit_body(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, cells=("mk-pers-ft2e6-con-s42",))
    cell = "mk-pers-ft2e6-con-s42"
    cell_root, train_dir = _mk_cell_fixture(cfg, cell)
    rung = train_dir / "checkpoint-7"
    rung.mkdir()
    fake = create_autospec(d._marker_source_read, return_value={"delta_logp_mean": 1.0})
    monkeypatch.setattr(d, "_marker_source_read", fake)
    d.run_fu_mkread_unit(cfg, f"{cell}:7")
    args = fake.call_args[0]
    assert args[1] == str(rung) and args[2] == cell_root / "rung7"


# ── selection: fu anchors + no-reap for lora ladders ─────────────────────────


def _stage_dose_labels(cfg):
    dest = d._fu_dose_labels_local(cfg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(DOSE_LABELS.read_text())


def test_select_cell_fu_imp_anchor_and_no_reap(tmp_path):
    cfg = _cfg(tmp_path, cells=("imp-pers-lora5e6-con-s42",))
    _stage_dose_labels(cfg)
    cell = "imp-pers-lora5e6-con-s42"
    cell_root = cfg.out_root / cell
    train_dir = cell_root / "train"
    reads = {}
    for step, rate in ((5, 0.30), (10, 0.61), (15, 0.70), (20, 0.90)):
        _fake_adapter_ckpt(train_dir, step)
        reads[str(step)] = {"rate": rate}
    d._atomic_json(cell_root / "build_result.json", {"adapter_root": str(train_dir)})
    d._atomic_json(cell_root / "ladder.json", {"reads_by_step": reads})
    sel = d._select_cell(cfg, cell)
    # anchor = committed ft_dose 0.66 -> in-band 0.70 (gap .04) beats 0.61 (.05)
    assert sel["anchor"] == pytest.approx(0.66)
    assert sel["step"] == 15 and sel["in_band"]
    assert sel["paired_arm"] == G.fu_ft_partner_of(cell).ft_partner_subfolder
    # LoRA rungs never reaped (full ladder uploads)
    assert set(d._rungs_or_empty(train_dir)) == {5, 10, 15, 20}
    assert "rungs_reaped" not in sel


def test_maybe_extend_fu_imp_trigger_and_skip(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, cells=("imp-pers-lora5e6-con-s137",))
    _stage_dose_labels(cfg)
    cell = "imp-pers-lora5e6-con-s137"
    cell_root = cfg.out_root / cell
    train_dir = cell_root / "train"
    train_dir.mkdir(parents=True)
    d._atomic_json(cell_root / "build_result.json", {"adapter_root": str(train_dir)})
    # in-band AND within gap of anchor (0.6432): no extension
    d._atomic_json(cell_root / "ladder.json", {"reads_by_step": {"10": {"rate": 0.62}}})
    called = create_autospec(d._run_unit_subprocess)
    monkeypatch.setattr(d, "_run_unit_subprocess", called)
    d._maybe_extend(cfg, cell)
    assert not called.called
    # below band everywhere: extension fires (ladder re-run stubbed)
    d._atomic_json(cell_root / "ladder.json", {"reads_by_step": {"10": {"rate": 0.30}}})
    monkeypatch.setattr(d, "run_ladder_unit", create_autospec(d.run_ladder_unit))
    d._maybe_extend(cfg, cell)
    assert called.called and called.call_args[0][1] == "p2l_ext"
    assert (cell_root / "extended.json").exists()


def test_maybe_extend_fu_marker_routes_nowhere(tmp_path):
    cfg = _cfg(tmp_path)
    cell = "mk-pers-ft2e6-con-s42"
    cell_root = cfg.out_root / cell
    cell_root.mkdir(parents=True)
    (cell_root / "train").mkdir()
    d._atomic_json(cell_root / "ladder.json", {"reads_by_step": {}})
    d._atomic_json(cell_root / "build_result.json", {"adapter_root": str(cell_root / "train")})
    with pytest.raises(RuntimeError, match="owned by run_fu_marker_ladder"):
        d._maybe_extend(cfg, cell)


# ── panel arms / conditional skip / arm resolution ───────────────────────────


def test_panel_arms_fu_pairs_and_conditional_skip(tmp_path):
    cfg = _cfg(tmp_path)
    arms = dict(d._panel_arms(cfg))  # selection.json absent -> nothing skipped
    assert arms["imp-pers-lora5e6-con-s42"] == "lora"
    assert arms["imp-pers-ft-con-s42"] == "ft"
    assert arms["mk-pers-ft2e6-con-s42"] == "ft"
    assert arms["mk-pers-lora-con-s42"] == "lora"
    # a not-in-window marker selection drops the WHOLE pair (plan §4.A)
    cell_root = cfg.out_root / "mk-pers-ft2e6-con-s42"
    cell_root.mkdir(parents=True)
    d._atomic_json(cell_root / "selection.json", {"in_band": False, "fallback": "closest_approach"})
    arms2 = dict(d._panel_arms(cfg))
    assert "mk-pers-ft2e6-con-s42" not in arms2 and "mk-pers-lora-con-s42" not in arms2
    # smoke DISABLES the skip so the marker panel path stays exercised
    cfg_s = _cfg(tmp_path, smoke=True, cells=("mk-pers-ft2e6-con-s42",))
    d._atomic_json(
        cfg_s.out_root / "mk-pers-ft2e6-con-s42" / "selection.json",
        {"in_band": False, "fallback": "closest_approach"},
    )
    assert ("mk-pers-ft2e6-con-s42", "ft") in d._panel_arms(cfg_s)


def test_capture_passes_base_derived_from_arms(tmp_path):
    cfg = _cfg(tmp_path)
    passes = d.capture_passes(cfg)
    assert ("base_mk", "base") in passes and ("base_imp", "base") in passes
    # all 4 marker cells not-in-window -> no marker arms AND no base_mk pass
    for fc in G.FU_MARKER_FT_CELLS:
        root = cfg.out_root / fc.cell
        root.mkdir(parents=True, exist_ok=True)
        d._atomic_json(root / "selection.json", {"in_band": False, "fallback": "closest_approach"})
    passes2 = d.capture_passes(cfg)
    assert ("base_mk", "base") not in passes2
    assert all(not a.startswith("mk-") for a, _k in passes2)


def test_selected_ft_ckpt_fu_partner_requires_stage(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(RuntimeError, match="not staged"):
        d._selected_ft_ckpt(cfg, "imp-pers-ft-con-s42")
    dest = d._staged_ft_dir(cfg, "imp-pers-ft-con-s42")
    dest.mkdir(parents=True)
    (dest / "config.json").write_text("{}")
    assert d._selected_ft_ckpt(cfg, "imp-pers-ft-con-s42") == dest


# ── persist: fu prefixes + full-ladder upload ────────────────────────────────


def test_phase_persist_fu_paths(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, cells=("mk-pers-ft2e6-con-s42", "imp-pers-lora5e6-con-s42"))
    mk_root = cfg.out_root / "mk-pers-ft2e6-con-s42"
    mk_train = mk_root / "train"
    _write_complete_ckpt(mk_train / "checkpoint-15")  # r7: meets the completeness bar
    d._atomic_json(mk_root / "build_result.json", {"adapter_root": str(mk_train)})
    d._atomic_json(mk_root / "selection.json", {"step": 15, "in_band": True})
    imp_root = cfg.out_root / "imp-pers-lora5e6-con-s42"
    imp_train = imp_root / "train"
    _fake_adapter_ckpt(imp_train, 15)
    d._atomic_json(imp_root / "build_result.json", {"adapter_root": str(imp_train)})
    d._atomic_json(imp_root / "selection.json", {"step": 15, "in_band": True})
    calls = []

    def fake_upload(path, repo, repo_type, path_in_repo, raise_on_error=False):
        calls.append((str(path), repo, path_in_repo))
        return f"https://hf/{path_in_repo}"

    monkeypatch.setattr(d.hub, "_upload", create_autospec(d.hub._upload, side_effect=fake_upload))
    monkeypatch.setattr(d, "_upload_with_transport_retry", lambda stage, prefix: f"ok:{prefix}")
    monkeypatch.setattr(d, "_ensure_dir_tokenizer", lambda p: True)
    out = d.phase_persist(cfg, {}, cells=list(cfg.cells))
    repos = {c[1] for c in calls}
    paths = {c[2] for c in calls}
    assert f"{G.FU_CKPT_PREFIX}/mk-pers-ft2e6-con-s42/checkpoint-15" in paths
    assert f"{G.FU_CKPT_PREFIX}/imp-pers-lora5e6-con-s42" in paths
    assert G.OVERFLOW_REPO in repos and G.HF_MODEL_REPO in repos
    assert out["uploaded"]["__records__"] == f"ok:{G.FU_DATA_PREFIX}"


# ── tier2: fu dose label + partner parity WARN semantics ─────────────────────


def test_run_tier2_unit_fu_partner_parity_warn_never_halt(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, cells=("imp-pers-lora5e6-con-s42",))
    _stage_dose_labels(cfg)
    cell = "imp-pers-lora5e6-con-s42"
    cell_root = cfg.out_root / cell
    train_dir = cell_root / "train"
    _fake_adapter_ckpt(train_dir, 15)
    d._atomic_json(cell_root / "build_result.json", {"adapter_root": str(train_dir)})
    d._atomic_json(cell_root / "selection.json", {"step": 15, "in_band": True})
    partner_dir = d._staged_ft_dir(cfg, "imp-pers-ft-con-s42")
    partner_dir.mkdir(parents=True)
    (partner_dir / "config.json").write_text("{}")
    rates = iter([0.70, 0.40])  # own tier2 0.70; partner re-read 0.40 (drifted)
    monkeypatch.setattr(
        d,
        "_content_rate",
        create_autospec(d._content_rate, side_effect=lambda *a, **k: next(rates)),
    )
    monkeypatch.setattr(d, "panel_context_ids", lambda cfg_, beh: ["persona_software_engineer"])
    monkeypatch.setattr(d, "_ensure_dir_tokenizer", lambda p: True)
    rec = d.run_tier2_unit(cfg, cell)
    assert rec["dose_label"]["dose_matched"]  # 0.70 in band, gap 0.04 <= 0.10
    par = rec["ft_partner_parity"]
    assert par["committed"] == pytest.approx(0.66)
    assert par["abs_delta"] == pytest.approx(0.26)
    assert par["severity"] == "WARN-analyzer-adjudication"  # WARN, never a raise
    # mirrored under the PARTNER arm id (plan §6.5 selection/imp-* glob)
    assert (cfg.out_root / "selection" / "imp-pers-ft-con-s42" / "selection.json").exists()


def test_run_p2l_unit_smoke_trains_one_rung(monkeypatch, tmp_path):
    """Smoke fence probe: cfg.smoke trains max_steps = ONE cadence rung (the
    fu4 smoke convention) through the SAME body."""
    from explore_persona_space.train import sft as sft_mod

    cfg = _cfg(tmp_path, smoke=True, cells=G.FU_SMOKE_CELLS)
    mix_dir = cfg.out_root / "inputs" / "mixes"
    mix_dir.mkdir(parents=True)
    (mix_dir / "imp_con.jsonl").write_text("{}\n")
    seen = {}

    def fake_train_lora(base_model, train_jsonl, out_dir, cfg=None):
        seen["max_steps"] = cfg.max_steps
        _fake_adapter_ckpt(Path(out_dir), cfg.max_steps)
        return Path(out_dir), 0.1

    monkeypatch.setattr(
        sft_mod, "train_lora", create_autospec(sft_mod.train_lora, side_effect=fake_train_lora)
    )
    import explore_persona_space.artifacts.organisms as org

    monkeypatch.setattr(
        org, "release_trainer_cuda_memory", create_autospec(org.release_trainer_cuda_memory)
    )
    rec = d.run_p2l_unit(cfg, "imp-pers-lora5e6-con-s42")
    assert seen["max_steps"] == G.FU_IMP_SAVE_STEPS == 5
    assert rec["rungs"] == [5]


def test_retrain_to_step_fu_marker_horizon(monkeypatch, tmp_path):
    """Body probe for the fu _retrain_to_step branch: the re-derive keeps the
    FIXED chunk horizon (24 for base-grid steps, 48 for ext steps)."""
    cfg = _cfg(tmp_path, cells=("mk-pers-ft2e6-con-s42",))
    cell = "mk-pers-ft2e6-con-s42"
    cell_root, _train = _mk_cell_fixture(cfg, cell)
    mix_dir = cfg.out_root / "inputs" / "mixes"
    mix_dir.mkdir(parents=True)
    (mix_dir / "mk_con.jsonl").write_text("{}\n")
    captured = {}

    def fake_run(cmd, log, env=None):
        captured["cmd"] = cmd
        (cell_root / "train_reselect" / "checkpoint-7").mkdir(parents=True)

    monkeypatch.setattr(
        d, "_run_subprocess", create_autospec(d._run_subprocess, side_effect=fake_run)
    )
    monkeypatch.setattr(
        d, "_ft_lane_env", lambda lane: {"CUDA_VISIBLE_DEVICES": "0,1,2,3", "MASTER_PORT": "29500"}
    )
    rec = d._retrain_to_step(cfg, cell, 7)
    cmd = captured["cmd"]
    assert cmd[cmd.index("--max-steps") + 1] == str(G.FU_MARKER_STEP_CEILING)
    assert cmd[cmd.index("--ckpt-steps") + 1] == "7"
    assert rec["step"] == 7
    assert d._read_json(cell_root / "build_result.json")["adapter_root"].endswith("train_reselect")


# ── gates (data-dependent gate probes) ───────────────────────────────────────


def test_pilot_gate_halt_rc7(tmp_path):
    cfg = _cfg(tmp_path, smoke=False)
    with pytest.raises(SystemExit) as ei:
        d._pilot_gate(
            cfg,
            label="fu_f1f3",
            unit_wall_s=8 * 3600.0,  # 8h cell wall -> 32h projected >> 2x3.1h
            n_units=4,
            parallelism=1.0,
            plan_wall_h=d.FU_PILOT_PLAN_F1F3_WALL_H,
        )
    assert ei.value.code == d.PILOT_GATE_RC


def test_select_anchor_nearest_empty_ladder_raises():
    with pytest.raises(ValueError, match="empty ladder"):
        G.select_anchor_nearest({}, anchor=0.5, band=(0.6, 0.85))


# ── off-pod plumbs: geometry pairs + lattice arm naming ──────────────────────


def test_geometry_fu_pair_fn_is_registry_fn():
    assert geo1586.G.fu_diff_pairs_for is G.fu_diff_pairs_for
    # executed default untouched
    pairs = geo1586.diff_pairs_for("syc", ["syc-pers-ft-con-s42", "syc-pers-lora-con-s42"])
    assert pairs == (
        ("syc-pers-ft-con-s42__ft_vs_lora", "syc-pers-ft-con-s42", "syc-pers-lora-con-s42"),
    )


def test_lattice_fu_arm_naming(monkeypatch):
    monkeypatch.setattr(lat, "FU", True)
    assert lat._content_arms("imp", "con", "s42") == (
        "imp-pers-ft-con-s42",
        "imp-pers-lora5e6-con-s42",
    )
    with pytest.raises(AssertionError):
        lat._content_arms("syc", "con", "s42")  # fu lattice is imp/con only
    assert lat._marker_arm("ft", "po", "s137") == "mk-pers-ft2e6-po-s137"
    assert lat._marker_arm("lora", "po", "s137") == "mk-pers-lora-po-s137"
    monkeypatch.setattr(lat, "FU", False)
    assert lat._content_arms("syc", "po", "s42") == (
        "syc-pers-ft-po-s42",
        "syc-pers-lora-po-s42",
    )
    assert lat._marker_arm("ft", "con", "s42") == "mk-pers-ft-con-s42"


# ── wave partition + p2l overlap (plan v7 §9 f2 — r2 opportunistic minor) ────


def test_wave_partition_groups_imp_cells_one_wave(monkeypatch, tmp_path):
    """Plan §9 f2: the two impolite LoRA cells train as ONE wave of 2x 1-GPU
    fanout units (overlapped on distinct GPUs), never two width-1 waves
    (serial at 3/4 GPUs idle); marker FT waves keep width w=1 on 4 GPUs and
    the executed grid (no fu cells) keeps plain width-w chunks."""
    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: ["0", "1", "2", "3"])
    cfg = _cfg(tmp_path)
    mk = [c for c in G.FU_ALL_CELLS if not G.is_fu_lora_cell(c)]
    imp = [c for c in G.FU_ALL_CELLS if G.is_fu_lora_cell(c)]
    assert d._wave_partition(cfg) == [[c] for c in mk] + [imp]
    # executed-grid behavior byte-identical: plain width-w chunking
    assert d.partition_waves(["c1", "c2", "c3"], 2) == [["c1", "c2"], ["c3"]]
    # keep-cell demand model uses the SAME partition (imp wave = 2 ladders)
    assert d.keepcell_demand_gb(imp, 4) == pytest.approx(
        2 * G.FU_IMP_LADDER_GB + 2 * d.RUNG_GB + d.KEEPCELL_FIXED_OVERHEAD_GB
    )


def test_fu_phase_train_fans_out_both_imp_cells(monkeypatch, tmp_path):
    """Both pending imp cells in one wave on a multi-GPU pod route through
    _fanout_units (2x 1-GPU concurrent, distinct CVD-pinned devices), not the
    serial single-unit branch; per-cell resume/persist (build_result.json)
    semantics are untouched."""
    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: ["0", "1", "2", "3"])
    cfg = _cfg(tmp_path)
    imp = [c for c in G.FU_ALL_CELLS if G.is_fu_lora_cell(c)]
    fanned: list[list[list[str]]] = []

    def fake_fanout(cfg_, units):
        fanned.append(units)
        for u in units:
            cell = u[2]  # _unit_args shape: ["--unit", "p2l", <cell>, ...]
            bp = cfg_.out_root / cell / "build_result.json"
            bp.parent.mkdir(parents=True, exist_ok=True)
            bp.write_text(json.dumps({"cell": cell, "status": "trained"}))

    monkeypatch.setattr(d, "_fanout_units", fake_fanout)
    serial = create_autospec(d._run_unit_subprocess)
    monkeypatch.setattr(d, "_run_unit_subprocess", serial)
    out = d._fu_phase_train(cfg, imp)
    assert not serial.called
    assert len(fanned) == 1 and [u[2] for u in fanned[0]] == imp
    assert set(out) == set(imp)


# ── figures runtime guards (r2 opportunistic minor: fu/clobber guards) ───────


def test_figures_fu_flag_refuses_executed_grid_set():
    import issue1586_figures as figs

    with pytest.raises(SystemExit) as ei:
        figs.main(["--fu", "caveatfix", "--fig-dir", "figures/issue_1586/fu_caveatfix"])
    assert ei.value.code == 2


def test_figures_nondefault_roots_require_nondefault_figdir(tmp_path):
    import issue1586_figures as figs

    with pytest.raises(SystemExit) as ei:
        figs.main(["--geo-root", str(tmp_path / "fu_geo")])
    assert ei.value.code == 2


# ── sibling-smoke out-root reap (fu crash r3: epm:failure v9 quota starvation) ─


def test_default_smoke_root_derivation_matches_build_cfg():
    # real body: name is base-independent; base follows the /workspace probe
    assert d.default_smoke_root("caveatfix").name == "issue-1586-fu-smoke"
    assert d.default_smoke_root(None).name == "issue-1586-smoke"
    base = Path("/workspace") if Path("/workspace").is_dir() else Path("/tmp")
    assert d.default_smoke_root("caveatfix").parent == base
    # build_cfg's smoke default is the SAME derivation (no writer/reaper drift)
    args = d._parse_args(["--mode", "smoke", "--fu", "caveatfix"])
    assert d.build_cfg(args).out_root == d.default_smoke_root("caveatfix")


def test_full_mode_reaps_populated_smoke_root(tmp_path):
    smoke_root = tmp_path / "issue-1586-fu-smoke"
    (smoke_root / "mk-ft-po-s137" / "train").mkdir(parents=True)
    (smoke_root / "mk-ft-po-s137" / "train" / "ckpt.bin").write_bytes(b"x" * 4096)
    cfg = _cfg(tmp_path)  # smoke=False, fu="caveatfix"
    d.reap_sibling_smoke_root(cfg, smoke_root=smoke_root)
    assert not smoke_root.exists()


def test_smoke_mode_never_reaps_its_own_root(tmp_path):
    smoke_root = tmp_path / "issue-1586-fu-smoke"
    smoke_root.mkdir()
    (smoke_root / "stage_done.json").write_text("{}")
    cfg = _cfg(tmp_path, smoke=True, out_root=smoke_root)
    d.reap_sibling_smoke_root(cfg, smoke_root=smoke_root)
    assert (smoke_root / "stage_done.json").exists()


def test_reap_skips_when_out_root_is_the_smoke_root(tmp_path):
    # a full run (mis)pointed at the smoke path must never delete its live out_root
    smoke_root = tmp_path / "issue-1586-fu-smoke"
    smoke_root.mkdir()
    cfg = _cfg(tmp_path, out_root=smoke_root)
    d.reap_sibling_smoke_root(cfg, smoke_root=smoke_root)
    assert smoke_root.exists()


def test_reap_absent_root_logs_and_noops(tmp_path, caplog):
    # the "nothing to reap" line is the relaunch's fix-engaged signal
    cfg = _cfg(tmp_path)
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        d.reap_sibling_smoke_root(cfg, smoke_root=tmp_path / "issue-1586-fu-smoke")
    assert any("nothing to reap" in rec.message for rec in caplog.records)


def test_reap_wired_at_p0_stage_entry_before_headroom(tmp_path, monkeypatch):
    # behavioral ordering pin: the smoke root is gone BY THE TIME the headroom
    # assert runs (the crash site's guard class), on the resumed-stage path
    smoke_root = tmp_path / "issue-1586-fu-smoke"
    (smoke_root / "cell").mkdir(parents=True)
    cfg = _cfg(tmp_path)
    cfg.out_root.mkdir(parents=True)
    (cfg.out_root / "stage_done.json").write_text('{"resumed": true}')
    seen = {}

    def _recording_headroom(c, phase):
        seen["phase"] = phase
        seen["smoke_root_gone_at_headroom"] = not smoke_root.exists()

    monkeypatch.setattr(d, "_headroom", _recording_headroom)
    monkeypatch.setattr(d, "default_smoke_root", lambda fu: smoke_root)
    out = d.phase_stage(cfg)
    assert out == {"resumed": True}
    assert seen == {"phase": "p0_stage", "smoke_root_gone_at_headroom": True}
    assert not smoke_root.exists()
    # production call site derives the root itself (no explicit override)
    import inspect

    assert "smoke_root=" not in inspect.getsource(d.phase_stage)


# ── crash-fix r4: overflow hub-cache eviction + wave-boundary residue drain ──


def test_evict_overflow_hub_cache_only_overflow_entry_and_idempotent(tmp_path, monkeypatch, caplog):
    import huggingface_hub.constants as hf_consts

    hub_cache = tmp_path / "hub"
    monkeypatch.setattr(hf_consts, "HF_HUB_CACHE", str(hub_cache))
    overflow = hub_cache / f"models--{G.OVERFLOW_REPO.replace('/', '--')}"
    blob = overflow / "blobs" / "shard"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"x" * 4096)
    snap = overflow / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(blob)  # symlink must not double-count
    qwen = hub_cache / "models--Qwen--Qwen2.5-7B-Instruct" / "blobs"
    qwen.mkdir(parents=True)
    (qwen / "shard").write_bytes(b"y" * 2048)
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        n = d._evict_overflow_hub_cache()
    assert n == 4096
    assert not overflow.exists()
    assert (qwen / "shard").exists()  # ONLY the overflow entry is evicted
    assert any("[hub-evict] evicted" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        assert d._evict_overflow_hub_cache() == 0  # idempotent: absent -> no-op line
    assert any("nothing to evict" in r.message for r in caplog.records)


def test_hub_evict_wired_after_staged_set_guard_and_on_resume(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.out_root.mkdir(parents=True)
    (cfg.out_root / "stage_done.json").write_text(json.dumps({"fu_ft_partners": {}}))
    calls = []
    monkeypatch.setattr(d, "_evict_overflow_hub_cache", lambda: calls.append(1) or 0)
    monkeypatch.setattr(d, "_headroom", lambda c, phase: None)
    monkeypatch.setattr(d, "default_smoke_root", lambda fu: tmp_path / "no-such-smoke-root")
    # resume path: an overflow-staging stage_done (pre-fix, or re-staged) re-evicts
    assert d.phase_stage(cfg) == {"fu_ft_partners": {}}
    assert calls == [1]
    # a non-overflow stage_done does NOT evict
    (cfg.out_root / "stage_done.json").write_text(json.dumps({"mixes": {}}))
    calls.clear()
    assert d.phase_stage(cfg) == {"mixes": {}}
    assert calls == []
    # fresh path source-order pin (r3 convention): eviction fires AFTER the
    # fu-partner staged-set guard verified the consumer copies and BEFORE the
    # done record lands.
    import inspect

    src = inspect.getsource(d.phase_stage)
    i_guard = src.index("incomplete checkpoint stage")
    i_evict = src.index("_evict_overflow_hub_cache", i_guard)
    i_done = src.index("_atomic_json(done_path, rec)")
    assert i_guard < i_evict < i_done


def _write_complete_ckpt(ckpt, shard_bytes=64, fill=b"k"):
    """Materialize a checkpoint dir satisfying the PRODUCTION completeness
    predicate (d._ckpt_incomplete_reason is None): config.json + index whose
    weight_map enumerates the shard + metadata.total_size matching the shard
    bytes (review r6 Minor: pre-r7 fixtures sat below the production bar,
    masking exactly the partial-rung class)."""
    ckpt.mkdir(parents=True, exist_ok=True)
    shard = "model-00001-of-00001.safetensors"
    (ckpt / shard).write_bytes(fill * shard_bytes)
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": shard_bytes}, "weight_map": {"w": shard}})
    )
    (ckpt / "config.json").write_text("{}")
    return ckpt


def _completed_marker_cell(out_root, cell, sel_step=12, shard_bytes=4096):
    """On-disk replica of pod-1586's post-selection retrain-reselect residue:
    orphaned train/ root save + train_reselect/ root save beside the selected
    self-contained checkpoint rung (the epm:failure v10 43 GB shape)."""
    cell_root = out_root / cell
    reselect = cell_root / "train_reselect"
    _write_complete_ckpt(reselect / f"checkpoint-{sel_step}", shard_bytes=64)
    (reselect / "model-00001-of-00004.safetensors").write_bytes(b"r" * shard_bytes)
    (reselect / "model.safetensors.index.json").write_text("{}")
    (reselect / "config.json").write_text("{}")
    train = cell_root / "train"
    train.mkdir(parents=True)
    (train / "model-00001-of-00004.safetensors").write_bytes(b"t" * shard_bytes)
    (train / "config.json").write_text("{}")
    (cell_root / "selection.json").write_text(json.dumps({"step": sel_step}))
    (cell_root / "build_result.json").write_text(json.dumps({"adapter_root": str(reselect)}))
    return cell_root


def test_wave_headroom_would_fail_drains_prior_residue_before_assert(tmp_path, monkeypatch):
    import collections

    cfg = _cfg(tmp_path)
    cfg.out_root.mkdir(parents=True)
    prior = G.FU_MARKER_FT_CELLS[0].cell
    _completed_marker_cell(cfg.out_root, prior)
    nxt = G.FU_MARKER_FT_CELLS[1].cell  # pending (no build_result.json)
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(d.shutil, "disk_usage", lambda p: usage(1, 1, 0))  # would-fail
    seen = {}

    def _recording_assert(root, need_gb, *, phase, canary_gb=1.0):
        seen["phase"] = phase
        seen["orphan_train_gone"] = not (cfg.out_root / prior / "train").exists()
        seen["root_save_gone"] = not list(
            (cfg.out_root / prior / "train_reselect").glob("model-*.safetensors")
        )
        seen["ckpt_kept"] = (
            cfg.out_root
            / prior
            / "train_reselect"
            / "checkpoint-12"
            / "model-00001-of-00001.safetensors"
        ).exists()

    rec_assert = create_autospec(d.assert_out_root_headroom, side_effect=_recording_assert)
    monkeypatch.setattr(d, "assert_out_root_headroom", rec_assert)
    d._wave_headroom(cfg, 2, [nxt])
    assert rec_assert.call_count == 1
    # the drain ran BEFORE the assert: dead residue gone, selected rung intact
    assert seen == {
        "phase": "p2_train_wave2",
        "orphan_train_gone": True,
        "root_save_gone": True,
        "ckpt_kept": True,
    }


def test_wave_headroom_would_pass_skips_boundary_drain(tmp_path, monkeypatch):
    import collections

    cfg = _cfg(tmp_path)
    cfg.out_root.mkdir(parents=True)
    prior = G.FU_MARKER_FT_CELLS[0].cell
    _completed_marker_cell(cfg.out_root, prior)
    nxt = G.FU_MARKER_FT_CELLS[1].cell
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(d.shutil, "disk_usage", lambda p: usage(10**15, 0, 10**15))
    monkeypatch.setattr(d, "assert_out_root_headroom", create_autospec(d.assert_out_root_headroom))
    d._wave_headroom(cfg, 2, [nxt])
    # an already-passing boundary never forces a drain (work-conservation)
    assert (cfg.out_root / prior / "train").exists()
    assert list((cfg.out_root / prior / "train_reselect").glob("model-*.safetensors"))


def test_wave_reap_reclaims_root_saves_and_orphaned_train_dir(tmp_path, caplog):
    cfg = _cfg(tmp_path)
    prior = G.FU_MARKER_FT_CELLS[0].cell
    _completed_marker_cell(cfg.out_root, prior)
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        d._wave_reap(cfg, [prior])
    root = cfg.out_root / prior
    assert not (root / "train").exists()  # orphaned original train dir removed
    assert not list((root / "train_reselect").glob("model-*.safetensors"))
    assert not (root / "train_reselect" / "model.safetensors.index.json").exists()
    ckpt = root / "train_reselect" / "checkpoint-12"
    assert (ckpt / "model-00001-of-00001.safetensors").exists()  # selected rung intact
    assert (ckpt / "config.json").exists()
    drains = [r.message for r in caplog.records if "[wave] drain" in r.message]
    assert len(drains) == 2  # orphan-dir + root-save lines (fix-engaged signal)


def test_reclaim_residue_safety_gates_keep_everything(tmp_path):
    cfg = _cfg(tmp_path)
    cell = G.FU_MARKER_FT_CELLS[0].cell
    root = _completed_marker_cell(cfg.out_root, cell)
    # no selection yet -> untouched (mid-ladder cells keep their root saves)
    (root / "selection.json").unlink()
    assert d._reclaim_completed_cell_residue(cfg, [cell]) == 0
    assert (root / "train").exists()
    # selected rung absent under adapter_root -> untouched
    (root / "selection.json").write_text(json.dumps({"step": 99}))
    assert d._reclaim_completed_cell_residue(cfg, [cell]) == 0
    assert (root / "train").exists()
    # fu LoRA cells never touched (ladders persist WHOLE incl. root adapter)
    lora = G.FU_IMP_LORA_CELLS[0].cell
    lroot = _completed_marker_cell(cfg.out_root, lora)
    assert d._reclaim_completed_cell_residue(cfg, [lora]) == 0
    assert (lroot / "train").exists()
    assert list((lroot / "train_reselect").glob("model-*.safetensors"))


def test_reclaim_residue_non_retrained_shape_keeps_train_dir_and_rung(tmp_path):
    cfg = _cfg(tmp_path)
    cell = G.FU_MARKER_FT_CELLS[0].cell
    cell_root = cfg.out_root / cell
    train = cell_root / "train"
    ckpt = train / "checkpoint-12"
    ckpt.mkdir(parents=True)
    (ckpt / "model-00001-of-00001.safetensors").write_bytes(b"k" * 64)
    (train / "model-00001-of-00004.safetensors").write_bytes(b"t" * 4096)
    (train / "model.safetensors.index.json").write_text("{}")
    (train / "config.json").write_text("{}")
    (cell_root / "selection.json").write_text(json.dumps({"step": 12}))
    (cell_root / "build_result.json").write_text(json.dumps({"adapter_root": str(train)}))
    freed = d._reclaim_completed_cell_residue(cfg, [cell])
    assert freed >= 4096
    assert train.exists()  # adapter_root itself is never rmtree'd
    assert (ckpt / "model-00001-of-00001.safetensors").exists()
    assert not (train / "model-00001-of-00004.safetensors").exists()
    assert (train / "config.json").exists()  # small root metadata untouched


# ── crash-fix r5 (fu, epm:failure v11): [ckpt-reap] + [ckpt-restage] ─────────


def test_ckpt_persist_prefix_fu_and_executed(tmp_path):
    cell = G.FU_MARKER_FT_CELLS[0].cell
    assert (
        d._ckpt_persist_prefix(_cfg(tmp_path), cell, 12)
        == f"{G.FU_CKPT_PREFIX}/{cell}/checkpoint-12"
    )
    # executed (non-fu) grid keeps the issue1586/ prefix — persist-path symmetry
    assert (
        d._ckpt_persist_prefix(_cfg(tmp_path, fu=None), cell, 8) == f"issue1586/{cell}/checkpoint-8"
    )


def test_ckpt_reap_fires_only_with_hub_verification(tmp_path, monkeypatch, caplog):
    """Pin (i): reap ONLY on Hub verification — probe error => keep + WARN
    (fail-toward-keep); verified => reap + [ckpt-reap] bytes line; a resumed
    _wave_reap tolerates the verified-reaped (empty-rung) state."""
    cfg = _cfg(tmp_path)
    cell = G.FU_MARKER_FT_CELLS[0].cell
    root = _completed_marker_cell(cfg.out_root, cell)
    d._atomic_json(root / "persist.json", {"cell": cell, "step": 12, "url": "https://hf/x"})
    ckpt = root / "train_reselect" / "checkpoint-12"

    def _boom(api, repo, path, **kw):
        raise RuntimeError("hub down")

    monkeypatch.setattr(d.hub, "list_hf_files_under_path", _boom)
    with caplog.at_level(logging.WARNING, logger=d.logger.name):
        d._reclaim_completed_cell_residue(cfg, [cell])
    assert ckpt.exists()  # probe error => keep local (fail-toward-keep)
    assert any(
        "[ckpt-reap]" in r.message and "keeping local copy" in r.message for r in caplog.records
    )

    # Hub copy missing config.json => keep + WARN (incomplete upload)
    prefix = f"{G.FU_CKPT_PREFIX}/{cell}/checkpoint-12"
    monkeypatch.setattr(
        d.hub,
        "list_hf_files_under_path",
        lambda api, repo, path, **kw: [f"{prefix}/model-00001-of-00001.safetensors"],
    )
    d._reclaim_completed_cell_residue(cfg, [cell])
    assert ckpt.exists()

    def _verified(api, repo, path, **kw):
        assert repo == G.OVERFLOW_REPO and path == prefix
        return [
            f"{prefix}/config.json",
            f"{prefix}/model-00001-of-00001.safetensors",
            f"{prefix}/model.safetensors.index.json",
        ]

    monkeypatch.setattr(d.hub, "list_hf_files_under_path", _verified)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        freed = d._reclaim_completed_cell_residue(cfg, [cell])
    assert not ckpt.exists()  # verified => reaped
    assert freed >= 64
    assert any("[ckpt-reap]" in r.message and cell in r.message for r in caplog.records)
    # r3/r4 invariants intact: selection/build records survive the reap
    assert (root / "selection.json").exists() and (root / "build_result.json").exists()
    # resumed _wave_reap on the verified-reaped cell must NOT raise
    d._wave_reap(cfg, [cell])


def test_ckpt_reap_never_touches_unuploaded_or_deferred_cell(tmp_path, monkeypatch):
    """Pin (ii): the un-uploaded / in-progress / deferred cell is NEVER
    probed nor reaped."""
    cfg = _cfg(tmp_path)
    cell = G.FU_MARKER_FT_CELLS[1].cell
    root = _completed_marker_cell(cfg.out_root, cell)
    ckpt = root / "train_reselect" / "checkpoint-12"
    probes: list[str] = []

    def _probe(api, repo, path, **kw):
        probes.append(path)
        return [
            f"{path}/config.json",
            f"{path}/model-00001-of-00001.safetensors",
            f"{path}/model.safetensors.index.json",
        ]

    monkeypatch.setattr(d.hub, "list_hf_files_under_path", _probe)
    # no persist.json (in-progress / un-uploaded) -> never probed, never reaped
    d._reclaim_completed_cell_residue(cfg, [cell])
    assert ckpt.exists() and not probes
    # pending billing-403 deferral -> kept even with a persist.json present
    d._atomic_json(root / "persist.json", {"cell": cell, "step": 12, "url": "https://hf/x"})
    d._atomic_json(d._persist_deferred_path(cfg, cell), {"cell": cell, "step": 12})
    d._reclaim_completed_cell_residue(cfg, [cell])
    assert ckpt.exists() and not probes
    # deferral cleared -> the verified reap proceeds (sanity close of the gate)
    d._persist_deferred_path(cfg, cell).unlink()
    d._reclaim_completed_cell_residue(cfg, [cell])
    assert probes and not ckpt.exists()


def test_selected_ft_ckpt_restages_on_missing_local(tmp_path, monkeypatch, caplog):
    """Pin (iii): absent local selected checkpoint + fake staged tree =>
    _selected_ft_ckpt resolves the restaged path (consumer round-trip),
    threading the persist prefix + pinned revision into the SAME
    _stage_overflow_prefix machinery the reused FT partners use."""
    cfg = _cfg(tmp_path)
    cell = G.FU_MARKER_FT_CELLS[0].cell
    root = cfg.out_root / cell
    reselect = root / "train_reselect"
    reselect.mkdir(parents=True)
    d._atomic_json(root / "selection.json", {"step": 12})
    d._atomic_json(root / "build_result.json", {"adapter_root": str(reselect)})
    monkeypatch.setattr(d, "_ensure_dir_tokenizer", lambda p: True)
    monkeypatch.setattr(d, "_resolve_revision", lambda repo, t: "rev0")
    evicts: list[int] = []
    monkeypatch.setattr(d, "_evict_overflow_hub_cache", lambda: evicts.append(1) or 0)
    staged: dict[str, str] = {}

    def fake_stage(prefix, dest, *, revision, recursive=True):
        staged["prefix"], staged["revision"] = prefix, revision
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text("{}")
        wm = {}
        for i in range(1, 5):
            name = f"model-0000{i}-of-00004.safetensors"
            (dest / name).write_bytes(b"s")
            wm[f"w{i}"] = name
        (dest / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 4}, "weight_map": wm})
        )
        return dest

    monkeypatch.setattr(d, "_stage_overflow_prefix", fake_stage)
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        got = d._selected_ft_ckpt(cfg, cell)
    assert got == reselect / "checkpoint-12"
    assert (got / "config.json").exists()
    assert staged["prefix"] == f"{G.FU_CKPT_PREFIX}/{cell}/checkpoint-12"
    assert staged["revision"] == "rev0"
    # r6 pin (iii): the in-unit restage is EVICT-FREE — the r5 per-restage
    # evict raced siblings' in-flight downloads under the shared hub cache
    # (epm:failure v12); the PARENT batch evict owns it now.
    assert evicts == []
    assert any("[ckpt-restage]" in r.message and cell in r.message for r in caplog.records)
    # local present again -> no re-stage (idempotent resolve)
    staged.clear()
    assert d._selected_ft_ckpt(cfg, cell) == got and not staged


def test_selected_ft_ckpt_restage_fail_loud_paths(tmp_path, monkeypatch):
    """Pin (iii b): absent Hub path fails loud (FileNotFoundError propagates);
    an incomplete restage raises; a LoRA cell is NEVER overflow-restaged."""
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(d, "_ensure_dir_tokenizer", lambda p: True)
    monkeypatch.setattr(d, "_resolve_revision", lambda repo, t: "rev0")
    monkeypatch.setattr(d, "_evict_overflow_hub_cache", lambda: 0)
    cell = G.FU_MARKER_FT_CELLS[1].cell
    root = cfg.out_root / cell
    (root / "train_reselect").mkdir(parents=True)
    d._atomic_json(root / "selection.json", {"step": 12})
    d._atomic_json(root / "build_result.json", {"adapter_root": str(root / "train_reselect")})

    def absent(prefix, dest, *, revision, recursive=True):
        raise FileNotFoundError(f"no files under {G.OVERFLOW_REPO}/{prefix} @ {revision}")

    monkeypatch.setattr(d, "_stage_overflow_prefix", absent)
    with pytest.raises(FileNotFoundError):
        d._selected_ft_ckpt(cfg, cell)

    def partial(prefix, dest, *, revision, recursive=True):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text("{}")
        return dest

    monkeypatch.setattr(d, "_stage_overflow_prefix", partial)
    with pytest.raises(RuntimeError, match="incomplete"):
        d._selected_ft_ckpt(cfg, cell)
    shutil.rmtree(root / "train_reselect" / "checkpoint-12")

    lora = G.FU_IMP_LORA_CELLS[0].cell
    lroot = cfg.out_root / lora
    (lroot / "train").mkdir(parents=True)
    d._atomic_json(lroot / "selection.json", {"step": 12})
    d._atomic_json(lroot / "build_result.json", {"adapter_root": str(lroot / "train")})
    with pytest.raises(RuntimeError, match="never ckpt-reaped"):
        d._selected_ft_ckpt(cfg, lora)


# ── crash-fix r6 (fu, epm:failure v12): parent pre-stage + ONE batch evict ───


def _reaped_cell(out_root, cell, sel_step=12, extra_sel=None):
    """Selection/build records present, selected rung ABSENT — the
    post-[ckpt-reap] state a fan-out unit would lazily restage pre-r6."""
    root = out_root / cell
    (root / "train_reselect").mkdir(parents=True)
    d._atomic_json(root / "selection.json", {"step": sel_step, **(extra_sel or {})})
    d._atomic_json(root / "build_result.json", {"adapter_root": str(root / "train_reselect")})
    return root


def _fake_overflow_stage(events):
    """Signature-conformant _stage_overflow_prefix fake: records the staged
    cell and materializes a checkpoint tree at dest that satisfies the
    PRODUCTION completeness predicate (config.json + ALL index-listed
    shards; r7)."""

    def fake_stage(prefix, dest, *, revision, recursive=True):
        events.append(("stage", prefix.split("/")[-2]))
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text("{}")
        wm = {}
        for i in range(1, 5):
            name = f"model-0000{i}-of-00004.safetensors"
            (dest / name).write_bytes(b"s")
            wm[f"w{i}"] = name
        (dest / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 4}, "weight_map": wm})
        )
        return dest

    return fake_stage


def test_prestage_restages_all_serially_then_one_batch_evict(tmp_path, monkeypatch):
    """Pin (ii): the parent restages EVERY missing cell serially through the
    REAL _selected_ft_ckpt resolver, then evicts the overflow hub-cache entry
    EXACTLY ONCE per batch — and only when >=1 restage happened."""
    cfg = _cfg(tmp_path)
    c1, c2, c3 = (fc.cell for fc in G.FU_MARKER_FT_CELLS[:3])
    _reaped_cell(cfg.out_root, c1)
    _reaped_cell(cfg.out_root, c2)
    _completed_marker_cell(cfg.out_root, c3)  # rung present -> never restaged
    events: list = []
    monkeypatch.setattr(d, "_ensure_dir_tokenizer", lambda p: True)
    monkeypatch.setattr(d, "_resolve_revision", lambda repo, t: "rev0")
    monkeypatch.setattr(d, "_evict_overflow_hub_cache", lambda: events.append("evict") or 0)
    monkeypatch.setattr(d, "_stage_overflow_prefix", _fake_overflow_stage(events))
    # duplicate + present-cell mix: ordered dedupe, present cell skipped
    assert d._prestage_selected_ft_ckpts(cfg, [c3, c1, c2, c1]) == 2
    assert events == [("stage", c1), ("stage", c2), "evict"]
    # everything local now -> zero restages AND zero evicts (batch-gated)
    events.clear()
    assert d._prestage_selected_ft_ckpts(cfg, [c3, c1, c2]) == 0
    assert events == []


def test_prestage_skips_p0_staged_lora_and_recordless_ids(tmp_path, monkeypatch):
    """Pin (ii b): classes with no overflow-restage path never reach the
    resolver from the parent — the reused #1112 cell, FU FT partners,
    never-reaped LoRA ladders (records present), and record-less arm ids
    (reused LoRA arms, base_<beh> passes)."""
    cfg = _cfg(tmp_path)
    lora = G.FU_IMP_LORA_CELLS[0].cell
    _reaped_cell(cfg.out_root, lora)  # records exist, but LoRA -> skipped
    partner = G.FU_IMP_LORA_CELLS[0].ft_partner_cell

    def _never(cfg_, cell_):
        raise AssertionError(f"parent must not resolve {cell_}")

    monkeypatch.setattr(d, "_selected_ft_ckpt", _never)
    evicts: list[int] = []
    monkeypatch.setattr(d, "_evict_overflow_hub_cache", lambda: evicts.append(1) or 0)
    ids = [G.REUSED_FT_CELL, partner, lora, "base_syc", "syc-recordless-arm"]
    assert d._prestage_selected_ft_ckpts(cfg, ids) == 0
    assert evicts == []


def test_panel_parent_prestages_before_any_unit_spawn(tmp_path, monkeypatch):
    """Pin (i): p6_panel entry with >=1 missing selected checkpoint => the
    PARENT resolves ALL of them serially ([ckpt-restage] from the parent pid)
    + ONE batch [hub-evict] BEFORE _fanout_units spawns anything."""
    cfg = _cfg(tmp_path)
    events: list = []
    marker_cells = [fc.cell for fc in G.FU_MARKER_FT_CELLS]
    for cell in marker_cells:
        _reaped_cell(cfg.out_root, cell, extra_sel={"in_band": True})
    for ic in G.FU_IMP_LORA_CELLS:  # LoRA ladders: rung present (never reaped)
        lroot = cfg.out_root / ic.cell
        (lroot / "train" / "checkpoint-12").mkdir(parents=True)
        d._atomic_json(lroot / "selection.json", {"step": 12, "in_band": True})
        d._atomic_json(lroot / "build_result.json", {"adapter_root": str(lroot / "train")})
    monkeypatch.setattr(d, "_ensure_dir_tokenizer", lambda p: True)
    monkeypatch.setattr(d, "_resolve_revision", lambda repo, t: "rev0")
    monkeypatch.setattr(d, "_evict_overflow_hub_cache", lambda: events.append("evict") or 0)
    monkeypatch.setattr(d, "_stage_overflow_prefix", _fake_overflow_stage(events))
    monkeypatch.setattr(d, "_n_gpus", lambda: 4)
    monkeypatch.setattr(
        d, "_fanout_units", lambda cfg_, units: events.append(("fanout", len(units)))
    )
    d.phase_panel(cfg)
    staged = [e[1] for e in events if isinstance(e, tuple) and e[0] == "stage"]
    # every reaped marker FT cell restaged by the PARENT, serially, exactly once
    assert sorted(staged) == sorted(marker_cells)
    # strict ordering: all restages -> ONE batch evict -> ONE fanout spawn
    assert events[: len(staged)] == [("stage", c) for c in staged]
    assert events[len(staged) :] == ["evict", events[-1]]
    assert events[-1][0] == "fanout" and events[-1][1] > 0
    # FT partner arms (p0-staged inputs) were never overflow-restaged
    assert not set(staged) & {ic.ft_partner_cell for ic in G.FU_IMP_LORA_CELLS}


def test_prestage_wired_before_dispatch_at_every_consumer_phase():
    """Pin (i wiring, r4 source-order convention): every fan-out phase whose
    units consume selected FT checkpoints calls _prestage_selected_ft_ckpts
    BEFORE its _fanout_units dispatch. phase_parity is deliberately absent
    (parity units read p0-staged reused arms only, pre-selection)."""
    import inspect

    for phase in (
        d.phase_tier2,
        d.phase_panel,
        d.phase_margin,
        d.phase_capture,
        d.phase_capture_tf,
    ):
        src = inspect.getsource(phase)
        i_pre = src.index("_prestage_selected_ft_ckpts")
        i_fan = src.index("_fanout_units")
        assert i_pre < i_fan, phase.__name__
    assert "_prestage_selected_ft_ckpts" not in inspect.getsource(d.phase_parity)
    # pin (iii) wiring: the in-unit backstop no longer evicts per restage
    assert "_evict_overflow_hub_cache" not in inspect.getsource(d._selected_ft_ckpt)


# ── crash-fix r7 (fu, review r6 Critical): partial-rung completeness ─────────
# partial-restage-invisible-to-missing-predicate: a PARTIALLY-restaged
# checkpoint-<step>/ (the v12 crash residue — _stage_overflow_prefix mkdirs
# dest BEFORE its download loop; _reap_unit_groups TERM-KILL truncates
# mid-copy) previously PASSED the rung-presence lookup in BOTH
# _restageable_missing_ft_cells and _selected_ft_ckpt.


def _idx4(total_size=4):
    wm = {f"w{i}": f"model-0000{i}-of-00004.safetensors" for i in range(1, 5)}
    return json.dumps({"metadata": {"total_size": total_size}, "weight_map": wm}).encode()


def _partial_rung(out_root, cell, files, sel_step=12):
    """Selection/build records + a PARTIALLY-materialized selected rung (the
    v12 crash-residue shape). ``files`` maps rung-relative name -> bytes."""
    root = _reaped_cell(out_root, cell, sel_step=sel_step)
    ckpt = root / "train_reselect" / f"checkpoint-{sel_step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (ckpt / name).write_bytes(data)
    return ckpt


_PARTIAL_SHAPES = {
    "config_only": {"config.json": b"{}"},
    "missing_shard": {
        "config.json": b"{}",
        "model.safetensors.index.json": _idx4(),
        "model-00001-of-00004.safetensors": b"s",
        "model-00002-of-00004.safetensors": b"s",
        "model-00003-of-00004.safetensors": b"s",
    },
    "size_zero_shard": {
        "config.json": b"{}",
        "model.safetensors.index.json": _idx4(),
        "model-00001-of-00004.safetensors": b"s",
        "model-00002-of-00004.safetensors": b"s",
        "model-00003-of-00004.safetensors": b"s",
        "model-00004-of-00004.safetensors": b"",
    },
    "incomplete_member": {
        "config.json": b"{}",
        "model.safetensors.index.json": _idx4(),
        "model-00001-of-00004.safetensors": b"s",
        "model-00002-of-00004.safetensors": b"s",
        "model-00003-of-00004.safetensors": b"s",
        "model-00004-of-00004.safetensors": b"s",
        "model-00004-of-00004.safetensors.incomplete": b"x",
    },
    "truncated_shards": {
        "config.json": b"{}",
        "model.safetensors.index.json": _idx4(total_size=100),
        "model-00001-of-00004.safetensors": b"s",
        "model-00002-of-00004.safetensors": b"s",
        "model-00003-of-00004.safetensors": b"s",
        "model-00004-of-00004.safetensors": b"s",
    },
    "shards_without_index": {
        "config.json": b"{}",
        "model-00001-of-00004.safetensors": b"s",
        "model-00002-of-00004.safetensors": b"s",
        "model-00003-of-00004.safetensors": b"s",
        "model-00004-of-00004.safetensors": b"s",
    },
}


@pytest.mark.parametrize("shape", sorted(_PARTIAL_SHAPES))
def test_partial_rung_reads_missing_in_classifier_and_resolver(shape, tmp_path, monkeypatch):
    """r7 pin (i): every partial-rung crash-residue shape reads MISSING in the
    classifier AND triggers a fresh restage in the resolver (fails on r6 code,
    where the rung-presence lookup read all of these PRESENT)."""
    cfg = _cfg(tmp_path)
    cell = G.FU_MARKER_FT_CELLS[0].cell
    ckpt = _partial_rung(cfg.out_root, cell, _PARTIAL_SHAPES[shape])
    assert d._ckpt_incomplete_reason(ckpt) is not None
    assert d._restageable_missing_ft_cells(cfg, [cell]) == [cell]
    events: list = []
    monkeypatch.setattr(d, "_ensure_dir_tokenizer", lambda p: True)
    monkeypatch.setattr(d, "_resolve_revision", lambda repo, t: "rev0")
    monkeypatch.setattr(d, "_stage_overflow_prefix", _fake_overflow_stage(events))
    got = d._selected_ft_ckpt(cfg, cell)
    assert [e for e in events if e[0] == "stage"] == [("stage", cell)]
    assert got == ckpt and d._ckpt_incomplete_reason(got) is None


def test_resolver_rmtrees_incomplete_rung_before_restage(tmp_path, monkeypatch, caplog):
    """r7 pin (ii): the resolver REMOVES an incomplete rung dir before staging
    (logging the [ckpt-restage] removal line), so _stage_overflow_prefix's
    config.json early-return / per-file target.exists() skip can never
    preserve truncated files — the stale poison member must be GONE after the
    round-trip. Also pins the prestage integration: classify -> rmtree ->
    restage -> ONE batch evict."""
    cfg = _cfg(tmp_path)
    cell = G.FU_MARKER_FT_CELLS[1].cell
    files = dict(_PARTIAL_SHAPES["missing_shard"])
    files["stale_truncated.bin"] = b"poison"  # survives any exists()-skip re-stage
    ckpt = _partial_rung(cfg.out_root, cell, files)
    events: list = []
    monkeypatch.setattr(d, "_ensure_dir_tokenizer", lambda p: True)
    monkeypatch.setattr(d, "_resolve_revision", lambda repo, t: "rev0")
    monkeypatch.setattr(d, "_evict_overflow_hub_cache", lambda: events.append("evict") or 0)
    monkeypatch.setattr(d, "_stage_overflow_prefix", _fake_overflow_stage(events))
    with caplog.at_level(logging.INFO, logger=d.logger.name):
        assert d._prestage_selected_ft_ckpts(cfg, [cell]) == 1
    assert events == [("stage", cell), "evict"]
    removal = [r.message for r in caplog.records if "removing incomplete" in r.message]
    assert len(removal) == 1 and cell in removal[0] and "[ckpt-restage]" in removal[0]
    assert not (ckpt / "stale_truncated.bin").exists()  # rmtree, not in-place re-stage
    assert d._ckpt_incomplete_reason(ckpt) is None
    # everything local + complete now -> no further restage, no evict
    events.clear()
    assert d._prestage_selected_ft_ckpts(cfg, [cell]) == 0
    assert events == []


def test_complete_rung_untouched_no_rmtree_no_restage(tmp_path, monkeypatch):
    """r7 pin (iii): a COMPLETE rung is never rmtree'd nor restaged — and the
    fixtures satisfy the production completeness predicate (review r6 Minor)."""
    cfg = _cfg(tmp_path)
    cell = G.FU_MARKER_FT_CELLS[2].cell
    root = _completed_marker_cell(cfg.out_root, cell)
    ckpt = root / "train_reselect" / "checkpoint-12"
    assert d._ckpt_incomplete_reason(ckpt) is None  # fixture meets production bar
    assert d._restageable_missing_ft_cells(cfg, [cell]) == []

    def _never(prefix, dest, *, revision, recursive=True):
        raise AssertionError("complete rung must not be restaged")

    monkeypatch.setattr(d, "_ensure_dir_tokenizer", lambda p: True)
    monkeypatch.setattr(d, "_resolve_revision", lambda repo, t: "rev0")
    monkeypatch.setattr(d, "_stage_overflow_prefix", _never)
    got = d._selected_ft_ckpt(cfg, cell)
    assert got == ckpt
    shard = ckpt / "model-00001-of-00001.safetensors"
    assert shard.read_bytes() == b"k" * 64  # untouched, not re-materialized
    # LoRA rungs stay out of the completeness scope (never rmtree'd here):
    lora = G.FU_IMP_LORA_CELLS[0].cell
    lroot = cfg.out_root / lora
    (lroot / "train" / "checkpoint-12").mkdir(parents=True)  # adapter-shaped, no config.json
    d._atomic_json(lroot / "selection.json", {"step": 12})
    d._atomic_json(lroot / "build_result.json", {"adapter_root": str(lroot / "train")})
    monkeypatch.setattr(d, "_ensure_dir_tokenizer", lambda p: True)
    got_lora = d._selected_ft_ckpt(cfg, lora)
    assert got_lora == lroot / "train" / "checkpoint-12" and got_lora.exists()
