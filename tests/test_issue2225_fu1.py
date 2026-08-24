"""Issue #2225 fu1 (`fu1_preimage_prevention`) unit tests.

Covers (fu1 plan §4.1-§4.3 + §7):
- the 80-cell fu1 registry (counts, grids, per-config l1_idx / mask / variant
  threading, adapters prefix, RND bank filename) and its slug scheme;
- PARENT-registry parity: 81 cells byte-untouched, parent resolution intact,
  parent cells carry the None defaults on the new fu1 Cell fields;
- the external cell-resolver seam (register_cell_resolver + the
  EPM_I2225_EXTRA_CELLS_MODULE env hook a fresh subprocess-like state uses);
- the ported pre-image algebra on synthetic maps (frame-fold identity at full
  rank, unit norms, deterministic random directions, degenerate fail-louds);
- the F1 verdict predicates (hot / cold / bracket) + run_f1_verdict end-to-end
  on synthetic pilot partials (real body, real filesystem via tmp_path);
- the fan-out ``script_path`` re-target for fu1 --single-cell children.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2225_fu1_directions as fu1dir
import issue2225_fu1_train as fu1
import issue2225_fu1_verdict as fu1v
import issue2225_train as train

# ── registry ──────────────────────────────────────────────────────────────────


def test_fu1_registry_has_exactly_80_cells():
    cells = fu1.build_fu1_cell_registry()
    assert len(cells) == fu1.EXPECTED_FU1_CELL_COUNT == 80
    slugs = [c.slug for c in cells]
    assert len(set(slugs)) == 80


def test_fu1_per_config_counts_and_threading():
    cells = fu1.build_fu1_cell_registry()
    by_config: dict[str, list] = {}
    for c in cells:
        by_config.setdefault(c.config, []).append(c)
    # 4 pre-image configs x 4 corpora x 4 coefs; 4 random configs x evil x 4.
    for cfg in ("J", "K", "L", "M"):
        assert len(by_config[cfg]) == 16, cfg
    for cfg in ("RJ", "RK", "RL", "RM"):
        assert len(by_config[cfg]) == 4, cfg
        assert {c.dataset for c in by_config[cfg]} == {"evil"}
    # Layer threading: J/L/RJ/RL -> 14; K/M/RK/RM -> 19.
    for cfg in ("J", "L", "RJ", "RL"):
        assert {c.l1_idx for c in by_config[cfg]} == {14}, cfg
    for cfg in ("K", "M", "RK", "RM"):
        assert {c.l1_idx for c in by_config[cfg]} == {19}, cfg
    # Position threading: J/K/RJ/RK -> context; L/M/RL/RM -> context_end.
    for cfg in ("J", "K", "RJ", "RK"):
        assert {c.mask_mode for c in by_config[cfg]} == {"context"}, cfg
    for cfg in ("L", "M", "RL", "RM"):
        assert {c.mask_mode for c in by_config[cfg]} == {"context_end"}, cfg
    for c in cells:
        assert c.layer_spec == "L1"
        assert c.coef in fu1.FU1_GRID
        assert c.adapters_hf_prefix == fu1.FU1_ADAPTERS_HF_PREFIX
        assert not c.prompt_mode
        # Opinions cells steer evil's pre-image (parent STEERED_TRAIT).
        expected_trait = train.STEERED_TRAIT[c.dataset]
        assert c.steered_trait == expected_trait, c.slug
        if c.config.startswith("R"):
            assert c.variant == "RND" and c.direction_filename == fu1.RND_BANK_FILENAME
        else:
            assert c.variant == "PRE" and c.direction_filename is None


def test_fu1_pilot_cells_are_k_m_evil_grid():
    pilot = fu1.fu1_pilot_cells()
    assert len(pilot) == 8
    assert {c.config for c in pilot} == {"K", "M"}
    assert {c.dataset for c in pilot} == {"evil"}
    assert sorted(c.coef for c in pilot if c.config == "K") == sorted(fu1.FU1_GRID)


def test_direction_path_and_adapter_prefix_threading(tmp_path):
    """PRE cells resolve {trait}_PRE.pt; RND cells the shared RND.pt bank;
    parent cells keep the parent conventions (the fu1 Cell-field seam)."""
    fu1.register_extra_cells()
    pre = train.resolve_cell("J__sycophancy__c0.75")
    rnd = train.resolve_cell("RL__evil__c1.5")
    assert train._direction_path(tmp_path, pre) == tmp_path / "sycophancy_PRE.pt"
    assert train._direction_path(tmp_path, rnd) == tmp_path / "RND.pt"
    assert train._adapters_prefix(pre) == fu1.FU1_ADAPTERS_HF_PREFIX
    parent = train.resolve_cell("A__evil__c0.5")
    assert train._direction_path(tmp_path, parent) == tmp_path / "evil_E1.pt"
    assert train._adapters_prefix(parent) == train.ADAPTERS_HF_PREFIX


# ── parent parity ─────────────────────────────────────────────────────────────


def test_parent_registry_untouched():
    cells = train.build_cell_registry()
    assert len(cells) == train.EXPECTED_CELL_COUNT == 81
    for c in cells:
        assert c.l1_idx is None
        assert c.direction_filename is None
        assert c.adapters_hf_prefix is None
    # Parent scaled-slug resolution intact.
    assert train.resolve_cell("A__evil__c0.25").coef == 0.25
    # fu1 slugs never collide with parent slugs.
    fu_slugs = {c.slug for c in fu1.build_fu1_cell_registry()}
    assert not fu_slugs & {c.slug for c in cells}


# ── resolver seam ─────────────────────────────────────────────────────────────


def test_resolver_seam_registry_and_scaled_slugs():
    fu1.register_extra_cells()
    assert train.resolve_cell("J__evil__c0.25").l1_idx == 14
    assert train.resolve_cell("M__mistake_opinions__c3.0").steered_trait == "evil"
    # §7 octave-shifted synth (not a registry member) round-trips canonically.
    scaled = train.resolve_cell("K__evil__c6.0")
    assert scaled.coef == 6.0 and scaled.l1_idx == 19
    with pytest.raises(ValueError, match="non-canonical"):
        train.resolve_cell("K__evil__c6.00")
    # A single-letter non-config falls to the parent regex + synth (loud);
    # a non-canonical shape falls out of every resolution branch (loud).
    with pytest.raises(ValueError, match="unknown config 'Z'"):
        train.resolve_cell("Z__evil__c1.0")
    with pytest.raises(ValueError, match="unknown cell slug"):
        train.resolve_cell("totally_bogus_slug")
    with pytest.raises(ValueError, match="not in fu1 config"):
        fu1.synth_fu1_cell("RJ", "sycophancy", 1.0)


def test_env_hook_loads_extra_cells_module(monkeypatch):
    """A FRESH process state (empty resolver list — the subprocess shape)
    resolves fu1 slugs through EPM_I2225_EXTRA_CELLS_MODULE alone."""
    monkeypatch.setattr(train, "_EXTRA_RESOLVERS", [])
    monkeypatch.setattr(train, "_EXTRA_MODULES_LOADED", set())
    monkeypatch.setenv(train.EXTRA_CELLS_MODULE_ENV, "issue2225_fu1_train")
    cell = train.resolve_cell("RL__evil__c0.75")
    assert cell.direction_filename == fu1.RND_BANK_FILENAME and cell.l1_idx == 14


def test_env_hook_absent_keeps_parent_error(monkeypatch):
    monkeypatch.setattr(train, "_EXTRA_RESOLVERS", [])
    monkeypatch.setattr(train, "_EXTRA_MODULES_LOADED", set())
    monkeypatch.delenv(train.EXTRA_CELLS_MODULE_ENV, raising=False)
    # With no fu1 resolver, the single-letter fu config falls to the parent
    # scaled-slug regex and fails loud at synth_cell (unknown config).
    with pytest.raises(ValueError, match="unknown config 'J'"):
        train.resolve_cell("J__evil__c0.25")


def test_single_cell_cmd_retargets_fu1_script(tmp_path):
    """The fan-out's --single-cell child argv targets the WRAPPER script so the
    child process re-registers fu1 cells at entry (subprocess-registry gotcha)."""
    fu1.register_extra_cells()
    cell = train.resolve_cell("J__evil__c0.25")
    fu_script = Path(fu1.__file__).resolve()
    cmd = train._single_cell_cmd(
        cell,
        gpu_id=3,
        dataset_root=tmp_path,
        ckpt_root=tmp_path,
        directions_dir=tmp_path,
        max_steps=None,
        cpu_only=False,
        model_name="m",
        script_path=fu_script,
    )
    assert cmd[3] == str(fu_script)
    assert cmd[cmd.index("--single-cell") + 1] == "J__evil__c0.25"
    # Default stays the parent script.
    cmd_default = train._single_cell_cmd(
        cell,
        gpu_id=0,
        dataset_root=tmp_path,
        ckpt_root=tmp_path,
        directions_dir=tmp_path,
        max_steps=None,
        cpu_only=False,
        model_name="m",
    )
    assert cmd_default[3].endswith("issue2225_train.py")


# ── ported pre-image algebra (synthetic maps) ─────────────────────────────────


def test_preimage_identity_map():
    """W = I: the pre-image of r_B is r_B itself (unit-normalized)."""
    rng = np.random.default_rng(0)
    r_b = rng.standard_normal(8)
    xsd = np.ones(8)
    M, Um, Sm, Vmt = fu1dir.map_svd(np.eye(8))
    w = fu1dir.preimage_w(Um, Sm, Vmt, r_b, 8)
    d = fu1dir.destandardized_direction(xsd, w)
    assert np.allclose(d, r_b / np.linalg.norm(r_b), atol=1e-12)
    assert fu1dir.frame_fold_cos(M, Um, xsd, d, r_b, 8) > 1 - 1e-12


def test_preimage_frame_fold_round_trip_full_rank():
    """Random invertible map + random xsd: the frame-fold identity holds at
    full rank (cos(M @ (d_pre/xsd), P_k(r_B)) == 1 up to float64 error), and
    d_pre is unit-norm."""
    rng = np.random.default_rng(2225)
    for trial in range(3):
        W = rng.standard_normal((16, 16))
        xsd = 0.5 + rng.random(16)
        r_b = rng.standard_normal(16)
        M, Um, Sm, Vmt = fu1dir.map_svd(W)
        assert np.allclose(M, W.T)
        w = fu1dir.preimage_w(Um, Sm, Vmt, r_b, 16)
        d = fu1dir.destandardized_direction(xsd, w)
        assert abs(np.linalg.norm(d) - 1.0) < 1e-12
        ff = fu1dir.frame_fold_cos(M, Um, xsd, d, r_b, 16)
        assert ff > 0.999, (trial, ff)


def test_preimage_truncation_and_degenerates():
    rng = np.random.default_rng(7)
    W = rng.standard_normal((8, 8))
    _M, Um, Sm, Vmt = fu1dir.map_svd(W)
    r_b = rng.standard_normal(8)
    with pytest.raises(ValueError, match="leaves no components"):
        fu1dir.preimage_w(Um, Sm, Vmt, r_b, 0)
    with pytest.raises(ValueError, match="degenerate norm"):
        fu1dir.destandardized_direction(np.ones(8), np.zeros(8))
    with pytest.raises(ValueError, match="degenerate row norm"):
        fu1dir.unit_rows(np.vstack([np.ones(4), np.zeros(4)]))
    # Truncated pre-image lives in the top-k right-singular subspace.
    w2 = fu1dir.preimage_w(Um, Sm, Vmt, r_b, 2)
    proj = Vmt[:2].T @ (Vmt[:2] @ w2)
    assert np.allclose(w2, proj, atol=1e-12)


def test_random_direction_deterministic_unit():
    a = fu1dir.random_direction(64, seed=2225014)
    b = fu1dir.random_direction(64, seed=2225014)
    c = fu1dir.random_direction(64, seed=2225019)
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-12
    assert abs(float(a @ c)) < 0.5, "different seeds must give distinct directions"
    assert fu1dir.N_RANDOM_SEEDS == 3


# ── F1 verdict (plan §7) ──────────────────────────────────────────────────────


def test_arm_verdict_predicates():
    base = 60.0
    hot = fu1v.arm_verdict(
        {c: {"trait_mean": 10.0, "coherence_mean": 50.0} for c in ("0.25", "3.0")}, base
    )
    assert hot["too_hot"] and not hot["too_cold"] and hot["octave_shift"] == 0.5
    cold = fu1v.arm_verdict(
        {c: {"trait_mean": 58.0, "coherence_mean": 92.0} for c in ("0.25", "3.0")}, base
    )
    assert cold["too_cold"] and not cold["too_hot"] and cold["octave_shift"] == 2.0
    ok = fu1v.arm_verdict(
        {
            "0.25": {"trait_mean": 55.0, "coherence_mean": 91.0},
            "3.0": {"trait_mean": 20.0, "coherence_mean": 62.0},
        },
        base,
    )
    assert ok["brackets_coherence_80"] and ok["octave_shift"] is None
    # A suppressing high-coherence grid is NOT too-cold (suppression >= 5).
    suppressing = fu1v.arm_verdict({"1.5": {"trait_mean": 40.0, "coherence_mean": 95.0}}, base)
    assert not suppressing["too_cold"] and suppressing["brackets_coherence_80"]
    # None judge means: never coherent, never suppressing (conservative).
    nones = fu1v.arm_verdict({"1.5": {"trait_mean": None, "coherence_mean": None}}, base)
    assert nones["too_hot"]


def _write_pilot_partials(eval_root: Path, cfg: str, per_coef: dict[str, dict]) -> None:
    for coef, vals in per_coef.items():
        for sub, key in (("trait_scores", "trait_mean"), ("coherence", "coherence_mean")):
            p = eval_root / "pilot" / sub / "partial" / f"{cfg}__evil__c{coef}__evil.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump({"model_mean": vals[key]}, f)


def test_run_f1_verdict_end_to_end(tmp_path):
    """Real run_f1_verdict body over synthetic pilot partials: PASS when one
    arm brackets; FAIL + per-arm repilot block with canonical fu1 slugs when
    neither does."""
    baseline = tmp_path / "i778_baseline.json"
    with open(baseline, "w") as f:
        json.dump({"trait_score": 60.0}, f)

    # Case 1: K brackets (mixed coherence + suppression), M too cold -> PASS.
    root1 = tmp_path / "pass"
    _write_pilot_partials(
        root1,
        "K",
        {
            "0.25": {"trait_mean": 50.0, "coherence_mean": 92.0},
            "0.75": {"trait_mean": 40.0, "coherence_mean": 85.0},
            "1.5": {"trait_mean": 30.0, "coherence_mean": 75.0},
            "3.0": {"trait_mean": 10.0, "coherence_mean": 50.0},
        },
    )
    _write_pilot_partials(
        root1,
        "M",
        {c: {"trait_mean": 58.0, "coherence_mean": 93.0} for c in ("0.25", "0.75", "1.5", "3.0")},
    )
    args = fu1v.build_argparser().parse_args(
        ["--eval-root", str(root1), "--i778-baseline", str(baseline)]
    )
    rc = fu1v.run_f1_verdict(args)
    verdict = json.loads((root1 / "pilot_gate" / "f1_verdict.json").read_text())
    assert rc == 0 and verdict["passed"]
    assert verdict["octave_shift"]["K"] is None and verdict["octave_shift"]["M"] == 2.0
    assert "K" not in verdict["repilot"] and verdict["repilot"]["M"]["coef_scale"] == 2.0

    # Case 2: K too hot, M too cold -> FAIL (rc 7) + both arms in repilot with
    # canonical fu1 scaled-cell slugs.
    root2 = tmp_path / "fail"
    _write_pilot_partials(
        root2,
        "K",
        {c: {"trait_mean": 5.0, "coherence_mean": 40.0} for c in ("0.25", "0.75", "1.5", "3.0")},
    )
    _write_pilot_partials(
        root2,
        "M",
        {c: {"trait_mean": 59.0, "coherence_mean": 95.0} for c in ("0.25", "0.75", "1.5", "3.0")},
    )
    args2 = fu1v.build_argparser().parse_args(
        ["--eval-root", str(root2), "--i778-baseline", str(baseline)]
    )
    rc2 = fu1v.run_f1_verdict(args2)
    verdict2 = json.loads((root2 / "pilot_gate" / "f1_verdict.json").read_text())
    assert rc2 == 7 and not verdict2["passed"]
    assert verdict2["repilot"]["K"]["coef_scale"] == 0.5
    assert verdict2["repilot"]["K"]["cells"] == [
        "K__evil__c0.125",
        "K__evil__c0.375",
        "K__evil__c0.75",
        "K__evil__c1.5",
    ]
    assert verdict2["repilot"]["M"]["coef_scale"] == 2.0
    # Re-verdict at the shifted M grid reads the shifted tags (canonical slugs).
    assert verdict2["repilot"]["M"]["cells"][0] == "M__evil__c0.5"


def test_run_f1_verdict_smoke_arms_and_grid(tmp_path):
    """Round-3 fix 1 pin (#1611/#1355 class): the smoke-shaped invocation
    (--arms J,L --grid 0.25) COMPLETES against fixture partials for the smoke's
    own trained cells — the pre-fix hardcoded K/M enumeration FileNotFoundError'd
    here and tripped the dispatcher's crash-with-no-artifact FATAL guard."""
    baseline = tmp_path / "i778_baseline.json"
    with open(baseline, "w") as f:
        json.dump({"trait_score": 60.0}, f)
    root = tmp_path / "smoke"
    # J brackets at the single coef; L is too cold -> verdict PASSes on J.
    _write_pilot_partials(root, "J", {"0.25": {"trait_mean": 30.0, "coherence_mean": 85.0}})
    _write_pilot_partials(root, "L", {"0.25": {"trait_mean": 59.0, "coherence_mean": 95.0}})
    args = fu1v.build_argparser().parse_args(
        [
            "--eval-root",
            str(root),
            "--i778-baseline",
            str(baseline),
            "--arms",
            "J,L",
            "--grid",
            "0.25",
        ]
    )
    rc = fu1v.run_f1_verdict(args)
    verdict = json.loads((root / "pilot_gate" / "f1_verdict.json").read_text())
    assert rc == 0 and verdict["passed"]
    assert set(verdict["grids"]) == {"J", "L"} and verdict["grids"]["J"] == [0.25]
    assert set(verdict["arms"]) == {"J", "L"}
    # No K/M enumeration anywhere in the artifact (the pre-fix crash shape).
    assert "K" not in verdict["arms"] and "M" not in verdict["arms"]


def test_f1_verdict_default_arms_are_pilot_configs():
    """Round-3 fix 1 pin: the default (no --arms) invocation still enumerates
    the section-7 pilot arms K,M."""
    args = fu1v.build_argparser().parse_args([])
    assert fu1v._arms(args) == ("K", "M")
    assert fu1v.F1_ARMS == ("K", "M")


def test_f1_verdict_arms_fail_loud_on_bad_input():
    """Unknown fu1 config and empty --arms both refuse loudly (never a silent
    empty enumeration that would vacuously pass no arms)."""
    bad = fu1v.build_argparser().parse_args(["--arms", "Z"])
    with pytest.raises(SystemExit, match="unknown fu1 config"):
        fu1v._arms(bad)
    empty = fu1v.build_argparser().parse_args(["--arms", ","])
    with pytest.raises(SystemExit, match="parsed empty"):
        fu1v._arms(empty)


def test_upload_bank_threads_hf_prefix(monkeypatch, tmp_path):
    """Round-3 fix 2 pin: upload_bank threads a caller-supplied hf_prefix (the
    dispatcher's _smoke twin) to hub._upload; the default stays the production
    fu1_directions prefix."""
    import explore_persona_space.orchestrate.hub as hub

    seen: list[str] = []

    def fake_upload(local_path, repo_id, repo_type, path_in_repo, raise_on_error=False, **kw):
        seen.append(path_in_repo)
        return f"https://huggingface.co/{repo_id}/{path_in_repo}"

    monkeypatch.setattr(hub, "_upload", fake_upload)
    fu1dir.upload_bank(tmp_path)
    fu1dir.upload_bank(tmp_path, hf_prefix=fu1dir.FU1_DIRECTIONS_HF_PREFIX + "_smoke")
    assert seen == [
        "issue2225_ctxsteer/analysis_tensors/fu1_directions",
        "issue2225_ctxsteer/analysis_tensors/fu1_directions_smoke",
    ]


# ── eval-side target resolution (fu1 seam) ────────────────────────────────────


def test_evalgen_resolve_targets_fu1_fallback():
    import issue2225_eval_gen as evalgen

    # Register on the train-module instance evalgen actually consults: a sibling
    # test file (test_issue2225_cell_registry.py) re-executes issue2225_train via
    # module_from_spec and REPLACES the sys.modules entry at collection time, so
    # in a combined run ``evalgen.train`` is not the instance ``fu1`` imported.
    # Production has one instance per process (the env hook covers fresh
    # processes); this is a test-harness artifact only.
    fu1.register_extra_cells()
    evalgen.train.register_cell_resolver(fu1.resolve_fu1_cell)
    (t,) = evalgen.resolve_targets(["L__hallucination__c1.5"])
    assert t.kind == "cell" and t.dataset == "hallucination"
    assert t.tag == "L__hallucination__c1.5"
    with pytest.raises(ValueError, match="unknown eval-target tag"):
        evalgen.resolve_targets(["ZZ__evil__c1.0"])


def test_evalgen_fu1_traits_steered_only():
    """Code-review r1 Blocker 2 pin: fu1 cells eval their STEERED trait only
    (plan §2 divergence 4 / §4.4 — opinions + random cells: evil eval set);
    parent §7-scaled slugs keep the parent all-trait behavior for opinions."""
    import issue2225_eval_gen as evalgen

    fu1.register_extra_cells()
    evalgen.train.register_cell_resolver(fu1.resolve_fu1_cell)
    (t,) = evalgen.resolve_targets(["J__mistake_opinions__c0.25"])
    assert t.traits == ("evil",), t.traits
    (r,) = evalgen.resolve_targets(["RK__evil__c3.0"])
    assert r.traits == ("evil",), r.traits
    # Parent scaled opinions slug (l1_idx None) -> all three traits, unchanged.
    (p,) = evalgen.resolve_targets(["A__mistake_opinions__c6.0"])
    assert len(p.traits) == 3, p.traits


# ── F1-effective grid threading (code-review r1 Blocker 1) ────────────────────


def _write_repilot_state(path: Path, plan: dict, *, resolved: bool = True) -> Path:
    path.write_text(json.dumps({"plan": plan, "resolved": resolved}))
    return path


_K_SHIFT_PLAN = {
    "K": {
        "coef_scale": 0.5,
        "grid_csv": "0.125,0.375,0.75,1.5",
        "cells": [
            "K__evil__c0.125",
            "K__evil__c0.375",
            "K__evil__c0.75",
            "K__evil__c1.5",
        ],
    }
}


def test_effective_cells_default_matches_registry(tmp_path):
    reg = [c.slug for c in fu1.build_fu1_cell_registry()]
    assert [c.slug for c in fu1.effective_fu1_cells(None)] == reg
    # An absent state file is the no-shift default too.
    assert [c.slug for c in fu1.effective_fu1_cells(tmp_path / "absent.json")] == reg


def test_resolved_repilot_state_changes_f2a_enumeration(tmp_path):
    """The reviewer's named pin: a RESOLVED repilot state changes the F2a cell
    enumeration — the shifted mask family runs the shifted grid, the other
    family keeps FU1_GRID, and every effective slug resolves in production."""
    state = _write_repilot_state(tmp_path / "f1_repilot_state.json", _K_SHIFT_PLAN)
    cells = fu1.effective_fu1_cells(state)
    slugs = {c.slug for c in cells}
    assert len(cells) == fu1.EXPECTED_FU1_CELL_COUNT
    assert slugs != {c.slug for c in fu1.build_fu1_cell_registry()}
    # K + its mask-family inheritors (J, RJ, RK) run the shifted grid...
    assert "K__evil__c0.125" in slugs and "K__evil__c3.0" not in slugs
    assert "J__sycophancy__c0.125" in slugs and "J__sycophancy__c3.0" not in slugs
    assert "RJ__evil__c0.125" in slugs and "RK__evil__c1.5" in slugs
    # ...the context_end family (M did not shift) keeps FU1_GRID.
    assert "M__evil__c3.0" in slugs and "L__evil__c0.25" in slugs
    # Every effective slug round-trips through the production resolver
    # (shifted slugs resolve via the extras seam -> synth_fu1_cell).
    fu1.register_extra_cells()
    for c in cells:
        assert fu1.train.resolve_cell(c.slug).slug == c.slug


def test_unresolved_repilot_state_fails_loud(tmp_path):
    state = _write_repilot_state(tmp_path / "s.json", _K_SHIFT_PLAN, resolved=False)
    with pytest.raises(RuntimeError, match="UNRESOLVED"):
        fu1.effective_fu1_grids(state)


def test_repilot_state_unknown_arm_fails_loud(tmp_path):
    plan = {"Z": {"coef_scale": 0.5, "grid_csv": "0.125", "cells": ["Z__evil__c0.125"]}}
    state = _write_repilot_state(tmp_path / "s.json", plan)
    with pytest.raises(ValueError, match="non-fu1 arm"):
        fu1.effective_fu1_grids(state)


def test_extreme_cells_follow_effective_grid(tmp_path):
    # Default: 20 (config x corpus) arms x 2 extremes = 40, at 0.25 / 3.0.
    ext = fu1.fu1_extreme_cells(fu1.effective_fu1_cells(None))
    assert len(ext) == 40
    assert {c.coef for c in ext} == {0.25, 3.0}
    # Shifted M (x2): the context_end family's extremes follow; K family stays.
    plan = {
        "M": {
            "coef_scale": 2.0,
            "grid_csv": "0.5,1.5,3.0,6.0",
            "cells": ["M__evil__c0.5", "M__evil__c1.5", "M__evil__c3.0", "M__evil__c6.0"],
        }
    }
    state = _write_repilot_state(tmp_path / "s.json", plan)
    ext2 = fu1.fu1_extreme_cells(fu1.effective_fu1_cells(state))
    assert len(ext2) == 40
    by_arm: dict[tuple[str, str], list[float]] = {}
    for c in ext2:
        by_arm.setdefault((c.config, c.dataset), []).append(c.coef)
    assert sorted(by_arm[("M", "evil")]) == [0.5, 6.0]
    assert sorted(by_arm[("RL", "evil")]) == [0.5, 6.0]  # inherits M (context_end)
    assert sorted(by_arm[("K", "evil")]) == [0.25, 3.0]  # unshifted


# ── consumer-side NaN-slice fail-loud (code-review r1 Major c) ────────────────


def test_nan_slice_consumer_assert_fires():
    """A (28,3584) NaN-rows bank sliced at a layer outside {14, 19} must fail
    loud through the PRODUCTION helpers (_build_steering_vectors + the
    _assert_finite_steering_vectors call train_steered_cell makes)."""
    import torch

    tr = fu1.train
    bank = torch.full((28, 3584), float("nan"))
    bank[14] = 1.0
    bank[19] = 2.0
    vec_bad = tr._build_steering_vectors(bank, "L1", 20)
    with pytest.raises(AssertionError, match="non-finite steering vector at layer 20"):
        tr._assert_finite_steering_vectors(vec_bad, "RND.pt")
    # The finite rows pass the same production assert.
    vec_ok = tr._build_steering_vectors(bank, "L1", 14)
    tr._assert_finite_steering_vectors(vec_ok, "RND.pt")


# ── dispatcher: slug-keyed engagement gate + smoke log namespace (round 4) ────

_FU1_DISPATCH = Path(__file__).resolve().parents[1] / "scripts" / "issue2225_fu1_dispatch.sh"


def _hook_gate_script() -> str:
    """Extract the REAL hook_count_gate function from the fu1 dispatcher and
    wrap it in a callable probe (log_phase stubbed; args: <log_dir> <slugs>)."""
    src = _FU1_DISPATCH.read_text(encoding="utf-8")
    start = src.index("hook_count_gate() {")
    end = src.index("\n}\n", start) + len("\n}\n")
    return (
        'log_phase() { echo "[phase=$1] $2"; }\n'
        + src[start:end]
        + '\nhook_count_gate "$1" "$2" testphase\n'
    )


def test_hook_gate_slug_keyed_ignores_foreign_logs(tmp_path):
    """Round-4 fix 1 pin (the 10/8 pod crash shape): 8 engaged production cells
    plus 2 FOREIGN smoke logs in the SAME directory -> the gate PASSES, because
    it is keyed to the invocation's slug set, never a directory-wide tally."""
    import subprocess

    logs = tmp_path / "f1_train"
    logs.mkdir()
    prod = [f"{c}__evil__c{v}" for c in ("K", "M") for v in ("0.25", "0.75", "1.5", "3.0")]
    for i, slug in enumerate(prod):
        tok = "[steer-hook]" if i % 2 == 0 else "[fanout-skip]"
        (logs / f"{slug}.log").write_text(f"train noise\n{tok} engaged\n")
    for foreign in ("J__evil__c0.25", "L__evil__c0.25"):
        (logs / f"{foreign}.log").write_text("[steer-hook] smoke residue\n")
    out = subprocess.run(
        ["bash", "-c", _hook_gate_script(), "_", str(logs), ",".join(prod)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "8/8" in out.stdout


def test_hook_gate_fails_loud_naming_missing_slugs(tmp_path):
    """Missing/unengaged cells are NAMED in the FATAL line; an empty slug list
    refuses loudly (never a vacuous pass)."""
    import subprocess

    logs = tmp_path / "f1_train"
    logs.mkdir()
    (logs / "K__evil__c0.25.log").write_text("[steer-hook]\n")
    (logs / "K__evil__c0.75.log").write_text("no engagement token here\n")
    out = subprocess.run(
        [
            "bash",
            "-c",
            _hook_gate_script(),
            "_",
            str(logs),
            "K__evil__c0.25,K__evil__c0.75,K__evil__c1.5",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 7
    assert "K__evil__c0.75" in out.stderr and "K__evil__c1.5" in out.stderr
    assert "K__evil__c0.25" not in out.stderr
    empty = subprocess.run(
        ["bash", "-c", _hook_gate_script(), "_", str(logs), ","],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert empty.returncode == 7 and "empty slug list" in empty.stderr


def test_upload_capture_threads_hf_repo(monkeypatch, tmp_path):
    """Round-5 fix pin (#1108 overflow contract): upload_capture threads a
    caller-supplied hf_repo to BOTH _upload call sites (folder + per-tag);
    the default stays the canonical data repo. path_in_repo layout unchanged."""
    import issue2225_capture as cap

    import explore_persona_space.orchestrate.hub as hub

    seen: list[tuple[str, str]] = []

    def fake_upload(local_path, repo_id, repo_type, path_in_repo, raise_on_error=False, **kw):
        seen.append((repo_id, path_in_repo))
        return f"https://huggingface.co/{repo_id}/{path_in_repo}"

    monkeypatch.setattr(hub, "_upload", fake_upload)
    overflow = "superkaiba1/explore-persona-space-overflow"
    prefix = "issue2225_ctxsteer/analysis_tensors/fu1_capture"
    (tmp_path / "capture" / "base").mkdir(parents=True)
    (tmp_path / "capture" / "base" / "t.pt").write_bytes(b"x")
    # Folder branch: default repo, then the threaded overflow repo.
    cap.upload_capture(tmp_path, None, hf_prefix=prefix)
    cap.upload_capture(tmp_path, None, hf_prefix=prefix, hf_repo=overflow)
    # Per-tag branch threads the same repo (SAME prefix layout on overflow).
    cap.upload_capture(tmp_path, ["base"], hf_prefix=prefix, hf_repo=overflow)
    assert seen == [
        (cap.DATA_REPO, prefix),
        (overflow, prefix),
        (overflow, f"{prefix}/base"),
    ]


def test_dispatch_capture_repo_args_env_wiring(tmp_path):
    """Round-5 fix pin: EPM_I2225_FU1_CAPTURE_HF_REPO evaluates into
    CAPTURE_REPO_ARGS when set and stays empty when unset (real wiring block);
    both capture-upload legs (f2d + F3 safety pass) carry the array."""
    import subprocess

    src = _FU1_DISPATCH.read_text(encoding="utf-8")
    head = src[src.index('SMOKE="${EPM_I2225_SMOKE') : src.index("log_phase() {")]
    base_env = {
        "PATH": "/usr/bin:/bin",
        "EPM_I2225_LOG_ROOT": str(tmp_path / "logs"),
        "EPM_I2225_SENTINEL_ROOT": str(tmp_path / "sent"),
    }
    overflow = "superkaiba1/explore-persona-space-overflow"
    for repo, expect in ((overflow, f"--hf-repo {overflow}"), ("", "")):
        out = subprocess.run(
            ["bash", "-c", head + '\necho "${CAPTURE_REPO_ARGS[@]:-}"'],
            capture_output=True,
            text=True,
            timeout=60,
            env={**base_env, "EPM_I2225_FU1_CAPTURE_HF_REPO": repo},
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == expect
    # Both capture-upload call sites thread the array; no other upload class does.
    assert src.count('--hf-prefix "$HF_CAPTURE" "${CAPTURE_REPO_ARGS[@]}"') == 2
    assert src.count("CAPTURE_REPO_ARGS[@]") == 2


def test_smoke_log_root_is_smoke_suffixed():
    """Round-4 fix 2 pin: the dispatcher's root-wiring block lands LOG_ROOT at
    a _smoke twin under EPM_I2225_SMOKE=1 (evaluates the REAL wiring block)."""
    import subprocess

    src = _FU1_DISPATCH.read_text(encoding="utf-8")
    head = src[src.index('SMOKE="${EPM_I2225_SMOKE') : src.index('mkdir -p "$LOG_ROOT"')]
    for smoke, expect in (
        ("1", "/workspace/logs/issue-2225-fu1_smoke"),
        ("", "/workspace/logs/issue-2225-fu1"),
    ):
        out = subprocess.run(
            ["bash", "-c", head + '\necho "$LOG_ROOT"'],
            capture_output=True,
            text=True,
            timeout=60,
            env={"PATH": "/usr/bin:/bin", "EPM_I2225_SMOKE": smoke},
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == expect
