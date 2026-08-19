"""Issue #2225 fu2 (`fu2_preimage_alltoken`) unit pins (plan v13 §7/§11).

Covers (A9 runs these TOGETHER with tests/test_issue2225.py +
tests/test_issue2225_fu1.py — the shared-seam regression contract):

- the 28-cell fu2 registry (counts, per-config threading, pilot cells,
  FULL-slug + output-dir disjointness vs the parent + fu1 registries);
- the external resolver seam (registry, scaled synth, the conditional RN
  arm, the env-module hook, parent error paths, --single-cell retarget);
- F1' grid inheritance (resolved repilot state shifts EVERY inheritor of Q;
  unresolved/unknown-arm states fail loud; extremes follow the grid);
- the --round verdict parametrization (fu2 defaults; fu1 byte-unchanged);
- the --round analysis context (accessors, phase order, narrow refusal);
- the S0 bank verifier (valid bank passes; each defect class raises);
- #2287 overflow routing seams (--hf-repo threading + dispatcher wiring).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2225_fu1_train as fu1
import issue2225_fu1_verdict as fuv
import issue2225_fu2_bankverify as bankv
import issue2225_fu2_train as fu2
import issue2225_train as train

_FU2_DISPATCH = Path(__file__).resolve().parents[1] / "scripts" / "issue2225_fu2_dispatch.sh"


# ── registry ──────────────────────────────────────────────────────────────────


def test_fu2_registry_has_exactly_28_cells():
    cells = fu2.build_fu2_cell_registry()
    assert len(cells) == fu2.EXPECTED_FU2_CELL_COUNT == 28
    slugs = [c.slug for c in cells]
    assert len(set(slugs)) == 28


def test_fu2_per_config_counts_and_threading():
    cells = fu2.build_fu2_cell_registry()
    by_cfg: dict[str, list] = {}
    for c in cells:
        by_cfg.setdefault(c.config, []).append(c)
    assert {k: len(v) for k, v in by_cfg.items()} == {"N": 12, "Q": 12, "RQ": 4}
    for c in cells:
        assert c.mask_mode is train._MASK_MODE["all"]
        assert c.layer_spec == "L1" and c.prompt_mode is False
        assert c.adapters_hf_prefix == fu2.FU2_ADAPTERS_HF_PREFIX
        assert c.coef in fu2.FU2_GRID
    for c in by_cfg["N"]:
        assert c.l1_idx == 14 and c.variant == "PRE" and c.direction_filename is None
    for c in by_cfg["Q"]:
        assert c.l1_idx == 19 and c.variant == "PRE" and c.direction_filename is None
    for c in by_cfg["RQ"]:
        assert c.l1_idx == 19 and c.variant == "RND"
        assert c.direction_filename == fu1.RND_BANK_FILENAME
        assert c.dataset == "evil"
    # PRE arms span the 3 fu2 corpora (opinions dropped — plan §2 divergence 1).
    assert {c.dataset for c in by_cfg["N"]} == set(fu2.FU2_DATASETS)
    assert "mistake_opinions" not in {c.dataset for c in cells}


def test_fu2_pilot_cells_are_q_evil_grid():
    pilots = fu2.fu2_pilot_cells()
    assert [c.slug for c in pilots] == [f"Q__evil__c{v}" for v in fu2.FU2_GRID]


def test_fu2_disjoint_from_parent_and_fu1():
    fu2_slugs = {c.slug for c in fu2.build_fu2_cell_registry()}
    parent_slugs = {c.slug for c in train.build_cell_registry()}
    fu1_slugs = {c.slug for c in fu1.build_fu1_cell_registry()}
    assert not fu2_slugs & parent_slugs
    assert not fu2_slugs & fu1_slugs
    # Conditional RN slugs are resolvable — they must be disjoint too.
    rn = {f"RN__evil__c{v}" for v in fu2.FU2_GRID}
    assert not rn & (parent_slugs | fu1_slugs | fu2_slugs)
    # Output-dir disjointness (adapter prefixes pairwise distinct).
    assert (
        len(
            {
                train.ADAPTERS_HF_PREFIX,
                fu1.FU1_ADAPTERS_HF_PREFIX,
                fu2.FU2_ADAPTERS_HF_PREFIX,
            }
        )
        == 3
    )


def test_parent_and_fu1_registries_untouched():
    assert len(train.build_cell_registry()) == train.EXPECTED_CELL_COUNT == 81
    assert len(fu1.build_fu1_cell_registry()) == fu1.EXPECTED_FU1_CELL_COUNT == 80
    # Parent scaled-slug resolution intact (letter P owned by the parent's
    # P_e3sPl1 prefix arm — the reason disjointness keys on FULL slugs).
    assert train.resolve_cell("P__evil__c0.25").config == "P"


# ── resolver seam ─────────────────────────────────────────────────────────────


def test_fu2_resolver_registry_scaled_and_conditional():
    fu2.register_extra_cells()
    assert train.resolve_cell("N__evil__c0.25").l1_idx == 14
    assert train.resolve_cell("Q__sycophancy__c1.5").mask_mode is train._MASK_MODE["all"]
    assert train.resolve_cell("RQ__evil__c3.0").direction_filename == fu1.RND_BANK_FILENAME
    # §7 octave-shifted synth (not a registry member) round-trips canonically.
    scaled = train.resolve_cell("Q__evil__c6.0")
    assert scaled.coef == 6.0 and scaled.l1_idx == 19
    # Conditional W2a arm RN: resolvable though NOT in the wave-1 registry.
    rn = train.resolve_cell("RN__evil__c0.25")
    assert rn.l1_idx == 14 and rn.variant == "RND"
    assert rn.direction_filename == fu1.RND_BANK_FILENAME
    assert rn.slug not in {c.slug for c in fu2.build_fu2_cell_registry()}
    with pytest.raises(ValueError, match="non-canonical"):
        train.resolve_cell("Q__evil__c6.00")
    with pytest.raises(ValueError, match="not in fu2 config"):
        fu2.synth_fu2_cell("RQ", "sycophancy", 1.0)
    with pytest.raises(ValueError, match="unknown fu2 config"):
        fu2.synth_fu2_cell("Z", "evil", 1.0)
    # Non-fu2 slugs fall through the fu2 resolver (None -> the chain continues).
    assert fu2.resolve_fu2_cell("K__evil__c0.25") is None
    assert fu2.resolve_fu2_cell("A__evil__c0.25") is None


def test_env_hook_loads_fu2_module(monkeypatch):
    """A FRESH process state (empty resolver list — the subprocess shape)
    resolves fu2 slugs through EPM_I2225_EXTRA_CELLS_MODULE alone (the
    dispatcher exports issue2225_fu2_train for every child)."""
    monkeypatch.setattr(train, "_EXTRA_RESOLVERS", [])
    monkeypatch.setattr(train, "_EXTRA_MODULES_LOADED", set())
    monkeypatch.setenv(train.EXTRA_CELLS_MODULE_ENV, "issue2225_fu2_train")
    cell = train.resolve_cell("N__hallucination__c0.75")
    assert cell.l1_idx == 14 and cell.variant == "PRE"


def test_env_hook_absent_keeps_parent_error(monkeypatch):
    monkeypatch.setattr(train, "_EXTRA_RESOLVERS", [])
    monkeypatch.setattr(train, "_EXTRA_MODULES_LOADED", set())
    monkeypatch.delenv(train.EXTRA_CELLS_MODULE_ENV, raising=False)
    with pytest.raises(ValueError, match="unknown config 'N'"):
        train.resolve_cell("N__evil__c0.25")


def test_single_cell_cmd_retargets_fu2_script(tmp_path):
    """The fan-out's --single-cell child argv targets the fu2 wrapper so the
    child re-registers fu2 cells at entry (subprocess-registry gotcha)."""
    fu2.register_extra_cells()
    cell = train.resolve_cell("N__evil__c0.25")
    fu2_script = Path(fu2.__file__).resolve()
    cmd = train._single_cell_cmd(
        cell,
        gpu_id=2,
        dataset_root=tmp_path,
        ckpt_root=tmp_path,
        directions_dir=tmp_path,
        max_steps=None,
        cpu_only=False,
        model_name="m",
        script_path=fu2_script,
    )
    assert cmd[3] == str(fu2_script)
    assert cmd[cmd.index("--single-cell") + 1] == "N__evil__c0.25"


# ── F1' grid inheritance (plan §7) ────────────────────────────────────────────


def test_effective_fu2_cells_default_matches_registry(tmp_path):
    reg = [c.slug for c in fu2.build_fu2_cell_registry()]
    assert [c.slug for c in fu2.effective_fu2_cells(None)] == reg
    # An absent state file is the same default (fresh run, no repilot).
    assert [c.slug for c in fu2.effective_fu2_cells(tmp_path / "absent.json")] == reg


def test_resolved_repilot_state_shifts_every_inheritor(tmp_path):
    """Every fu2 config inherits Q's effective grid (_GRID_INHERIT_FU2), so a
    resolved Q octave shift re-enumerates N, Q, AND RQ at the shifted grid."""
    state = tmp_path / "f1_repilot_state.json"
    with open(state, "w") as f:
        json.dump(
            {"resolved": True, "plan": {"Q": {"grid_csv": "0.5,1.5,3.0,6.0"}}},
            f,
        )
    grids = fu2.effective_fu2_grids(state)
    assert grids == {cfg: (0.5, 1.5, 3.0, 6.0) for cfg in ("N", "Q", "RQ", "RN")}
    cells = fu2.effective_fu2_cells(state)
    assert len(cells) == 28
    assert {c.coef for c in cells} == {0.5, 1.5, 3.0, 6.0}
    assert "N__evil__c6.0" in {c.slug for c in cells}


def test_unresolved_repilot_state_fails_loud(tmp_path):
    state = tmp_path / "f1_repilot_state.json"
    with open(state, "w") as f:
        json.dump({"resolved": False, "plan": {"Q": {"grid_csv": "0.5"}}}, f)
    with pytest.raises(RuntimeError, match="UNRESOLVED"):
        fu2.effective_fu2_grids(state)


def test_repilot_state_unknown_arm_fails_loud(tmp_path):
    state = tmp_path / "f1_repilot_state.json"
    with open(state, "w") as f:
        json.dump({"resolved": True, "plan": {"K": {"grid_csv": "0.5"}}}, f)
    with pytest.raises(ValueError, match="non-fu2 arm"):
        fu2.effective_fu2_grids(state)


def test_fu2_extreme_cells_14_targets():
    """F2c MMLU extremes: min+max coef per (config x corpus) arm — 7 arms
    (N x 3, Q x 3, RQ x 1) x 2 = 14 targets at the wave-1 grid."""
    extremes = fu2.fu2_extreme_cells(fu2.build_fu2_cell_registry())
    assert len(extremes) == 14
    coefs = {(c.config, c.dataset, c.coef) for c in extremes}
    assert all(c in (min(fu2.FU2_GRID), max(fu2.FU2_GRID)) for _, _, c in coefs)
    assert ("RQ", "evil", 0.25) in coefs and ("RQ", "evil", 3.0) in coefs


# ── F1' verdict --round parametrization ───────────────────────────────────────


def _write_pilot_partials(eval_root: Path, cfg: str, per_coef: dict[str, dict]) -> None:
    for coef, vals in per_coef.items():
        for sub, key in (("trait_scores", "trait_mean"), ("coherence", "coherence_mean")):
            p = eval_root / "pilot" / sub / "partial" / f"{cfg}__evil__c{coef}__evil.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w") as f:
                json.dump({"model_mean": vals[key]}, f)


def test_run_f1_verdict_round_fu2_end_to_end(tmp_path):
    """--round fu2 scores the Q pilot arm by default; PASS on bracketing;
    FAIL emits a repilot block with canonical fu2 slugs (the §7 remedy)."""
    baseline = tmp_path / "i778_baseline.json"
    with open(baseline, "w") as f:
        json.dump({"trait_score": 60.0}, f)

    root1 = tmp_path / "pass"
    _write_pilot_partials(
        root1,
        "Q",
        {
            "0.25": {"trait_mean": 50.0, "coherence_mean": 92.0},
            "0.75": {"trait_mean": 40.0, "coherence_mean": 85.0},
            "1.5": {"trait_mean": 30.0, "coherence_mean": 75.0},
            "3.0": {"trait_mean": 10.0, "coherence_mean": 50.0},
        },
    )
    args = fuv.build_argparser().parse_args(
        ["--round", "fu2", "--eval-root", str(root1), "--i778-baseline", str(baseline)]
    )
    rc = fuv.run_f1_verdict(args)
    verdict = json.loads((root1 / "pilot_gate" / "f1_verdict.json").read_text())
    assert rc == 0 and verdict["passed"]
    assert verdict["round"] == "fu2"
    assert verdict["followup"] == "fu2_preimage_alltoken"
    assert set(verdict["arms"]) == {"Q"}

    # Too cold -> rc 7 + octave 2.0 with canonical fu2 scaled-cell slugs.
    root2 = tmp_path / "fail"
    _write_pilot_partials(
        root2,
        "Q",
        {c: {"trait_mean": 59.0, "coherence_mean": 95.0} for c in ("0.25", "0.75", "1.5", "3.0")},
    )
    args2 = fuv.build_argparser().parse_args(
        ["--round", "fu2", "--eval-root", str(root2), "--i778-baseline", str(baseline)]
    )
    rc2 = fuv.run_f1_verdict(args2)
    verdict2 = json.loads((root2 / "pilot_gate" / "f1_verdict.json").read_text())
    assert rc2 == 7 and not verdict2["passed"]
    assert verdict2["repilot"]["Q"]["coef_scale"] == 2.0
    assert verdict2["repilot"]["Q"]["cells"] == [
        "Q__evil__c0.5",
        "Q__evil__c1.5",
        "Q__evil__c3.0",
        "Q__evil__c6.0",
    ]


def test_f1_verdict_round_fu2_smoke_arms(tmp_path):
    """The dispatcher's smoke branch scores its own trained fu2 cells
    (--arms N,RQ --grid 0.25) — the #1611/#1355 class pin, fu2 shape."""
    baseline = tmp_path / "i778_baseline.json"
    with open(baseline, "w") as f:
        json.dump({"trait_score": 60.0}, f)
    root = tmp_path / "smoke"
    _write_pilot_partials(root, "N", {"0.25": {"trait_mean": 30.0, "coherence_mean": 85.0}})
    _write_pilot_partials(root, "RQ", {"0.25": {"trait_mean": 59.0, "coherence_mean": 95.0}})
    args = fuv.build_argparser().parse_args(
        [
            "--round",
            "fu2",
            "--eval-root",
            str(root),
            "--i778-baseline",
            str(baseline),
            "--arms",
            "N,RQ",
            "--grid",
            "0.25",
        ]
    )
    rc = fuv.run_f1_verdict(args)
    verdict = json.loads((root / "pilot_gate" / "f1_verdict.json").read_text())
    assert rc == 0 and set(verdict["arms"]) == {"N", "RQ"}
    assert verdict["grids"]["N"] == [0.25]


def test_f1_verdict_round_defaults():
    """fu2 defaults: pilot arm Q + the fu2 grid + the fu2 eval root; the
    bare (round-less) invocation stays byte-identical fu1 (K,M)."""
    args = fuv.build_argparser().parse_args(["--round", "fu2"])
    assert fuv._arms(args) == ("Q",)
    assert args.eval_root is None
    assert fuv.ROUND_EVAL_ROOTS["fu2"] == "eval_results/issue_2225/fu2_preimage_alltoken"
    rnd = fuv._round_cfg("fu2")
    assert rnd["grid"] == fu2.FU2_GRID
    assert fuv._grids(args, ("Q",), rnd) == {"Q": list(fu2.FU2_GRID)}
    # fu1 default path unchanged (the fu1 test file pins F1_ARMS too).
    legacy = fuv.build_argparser().parse_args([])
    assert legacy.round == "fu1" and fuv._arms(legacy) == ("K", "M")
    with pytest.raises(SystemExit, match="unknown fu2 config"):
        fuv._arms(fuv.build_argparser().parse_args(["--round", "fu2", "--arms", "K"]))


# ── analysis --round context ──────────────────────────────────────────────────


def test_analysis_round_context_accessors():
    import issue2225_fu1_analysis as ana

    assert ana._ROUND == "fu1"  # module default keeps fu1 byte-identical
    try:
        ana._set_round("fu2")
        assert ana._is_fu2()
        assert ana._fu_train() is fu2
        assert ana._round_config_order() == ("N", "Q", "RQ", "RN")
        assert ana._round_parent_comparators() == ("G", "A")
        assert ana._round_capture_prefix() == ana.FU2_CAPTURE_HF_PREFIX
        assert ana._round_mmlu_leg() == (ana.OVERFLOW_REPO, ana.FU2_MMLU_HF_PREFIX)
        assert [c.slug for c in ana._round_effective_cells(None)] == [
            c.slug for c in fu2.build_fu2_cell_registry()
        ]
        assert ana._round_resolve("Q__evil__c6.0").coef == 6.0
        with pytest.raises(SystemExit, match="fu1-only"):
            ana.run_fu_narrow(object())
    finally:
        ana._set_round("fu1")
    assert not ana._is_fu2() and ana._fu_train() is fu1
    assert ana._round_parent_comparators() == ("C", "A")
    with pytest.raises(ValueError, match="unknown round"):
        ana._set_round("fu3")


def test_analysis_fu2_phase_order_drops_narrow_only():
    import issue2225_fu1_analysis as ana

    assert "narrow" in ana.PHASE_ORDER
    assert "narrow" not in ana.FU2_PHASE_ORDER
    assert [p for p in ana.PHASE_ORDER if p != "narrow"] == ana.FU2_PHASE_ORDER


# ── S0 bank verifier ──────────────────────────────────────────────────────────


def _write_bank(bank_dir: Path) -> None:
    """A synthesized VALID 9-file bank at the real (28, 3584) shape."""
    import torch

    bank_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    for name in (*(f"{t}_PRE.pt" for t in bankv.PRE_TRAITS), "RND.pt"):
        t = torch.full(bankv.BANK_SHAPE, float("nan"), dtype=torch.float32)
        for layer in bankv.MAP_LAYERS:
            v = torch.randn(bankv.BANK_SHAPE[1], generator=gen, dtype=torch.float64)
            t[layer] = (v / v.norm() * bankv.RHO_PINNED[layer]).to(torch.float32)
        torch.save(t, bank_dir / name)
    with open(bank_dir / "rho.json", "w") as f:
        json.dump({"rho_per_layer": {str(k): v for k, v in bankv.RHO_PINNED.items()}}, f)
    for trait in bankv.PRE_TRAITS:
        with open(bank_dir / f"{trait}_PRE_meta.json", "w") as f:
            json.dump(
                {
                    "ridge_payload_sha256": {"L14": "a" * 64, "L19": "b" * 64},
                    "rb_v2_rev": bankv.RB_V2_REV,
                },
                f,
            )
    with open(bank_dir / "RND_meta.json", "w") as f:
        json.dump({"seeds": {str(k): v for k, v in bankv.RND_SEEDS.items()}}, f)


def test_bankverify_valid_bank_passes(tmp_path):
    _write_bank(tmp_path / "bank")
    files = bankv.verify_bank(tmp_path / "bank")
    assert set(files) == set(bankv.BANK_FILES)
    for name in ("evil_PRE.pt", "RND.pt"):
        assert files[name]["finite_rows"] == [14, 19]
        for layer in bankv.MAP_LAYERS:
            assert abs(files[name]["row_norms"][str(layer)] - bankv.RHO_PINNED[layer]) < 1e-3


def test_bankverify_defect_classes_raise(tmp_path):
    import torch

    # (a) wrong row norm (an unscaled / re-normalized bank).
    root_a = tmp_path / "a"
    _write_bank(root_a)
    t = torch.load(root_a / "evil_PRE.pt", weights_only=True)
    t[14] = t[14] * 2.0
    torch.save(t, root_a / "evil_PRE.pt")
    with pytest.raises(AssertionError, match="deviates from"):
        bankv.verify_bank(root_a)

    # (b) an extra finite row (a layer-index slip — the NaN slice guard).
    root_b = tmp_path / "b"
    _write_bank(root_b)
    t = torch.load(root_b / "RND.pt", weights_only=True)
    t[7] = 1.0
    torch.save(t, root_b / "RND.pt")
    with pytest.raises(AssertionError, match="finite rows"):
        bankv.verify_bank(root_b)

    # (c) wrong shape refuses at the tensor check.
    root_c = tmp_path / "c"
    _write_bank(root_c)
    torch.save(torch.zeros(28, 8), root_c / "sycophancy_PRE.pt")
    with pytest.raises(AssertionError, match="shape"):
        bankv.verify_bank(root_c)

    # (d) drifted rho.json (a regenerated bank) refuses on the plan pin.
    root_d = tmp_path / "d"
    _write_bank(root_d)
    with open(root_d / "rho.json", "w") as f:
        json.dump({"rho_per_layer": {"14": 63.056901, "19": 90.0}}, f)
    with pytest.raises(AssertionError, match="plan-pinned"):
        bankv.verify_bank(root_d)

    # (e) missing/empty meta pins refuse.
    root_e = tmp_path / "e"
    _write_bank(root_e)
    with open(root_e / "evil_PRE_meta.json", "w") as f:
        json.dump({"ridge_payload_sha256": {}, "rb_v2_rev": bankv.RB_V2_REV}, f)
    with pytest.raises(AssertionError, match="ridge_payload_sha256"):
        bankv.verify_bank(root_e)

    root_f = tmp_path / "f"
    _write_bank(root_f)
    with open(root_f / "RND_meta.json", "w") as f:
        json.dump({"seeds": {"14": 1, "19": 2}}, f)
    with pytest.raises(AssertionError, match="seeds"):
        bankv.verify_bank(root_f)

    # (g) a missing bank file refuses by name.
    root_g = tmp_path / "g"
    _write_bank(root_g)
    (root_g / "RND.pt").unlink()
    with pytest.raises(FileNotFoundError, match=r"RND\.pt"):
        bankv.verify_bank(root_g)


# ── #2287 overflow routing seams ──────────────────────────────────────────────


def test_evalgen_upload_threads_hf_repo(monkeypatch, tmp_path):
    import issue2225_eval_gen as eg

    import explore_persona_space.orchestrate.hub as hub

    seen: list[tuple[str, str]] = []

    def fake_upload(local_path, repo_id, repo_type, path_in_repo, raise_on_error=False, **kw):
        seen.append((repo_id, path_in_repo))
        return f"https://huggingface.co/{repo_id}/{path_in_repo}"

    monkeypatch.setattr(hub, "_upload", fake_upload)
    overflow = "superkaiba1/explore-persona-space-overflow"
    (tmp_path / "raw_completions" / "final").mkdir(parents=True)
    (tmp_path / "raw_completions" / "final" / "base.json").write_text("{}")
    eg.upload_raw_completions(tmp_path, final_prefix="p/fu2_final")
    eg.upload_raw_completions(tmp_path, final_prefix="p/fu2_final", hf_repo=overflow)
    assert seen == [(eg.DATA_REPO, "p/fu2_final"), (overflow, "p/fu2_final")]


def test_mmlu_upload_threads_hf_repo(monkeypatch, tmp_path):
    import issue2225_mmlu as mm

    import explore_persona_space.orchestrate.hub as hub

    seen: list[tuple[str, str]] = []

    def fake_upload(local_path, repo_id, repo_type, path_in_repo, raise_on_error=False, **kw):
        seen.append((repo_id, path_in_repo))
        return f"https://huggingface.co/{repo_id}/{path_in_repo}"

    monkeypatch.setattr(hub, "_upload", fake_upload)
    overflow = "superkaiba1/explore-persona-space-overflow"
    (tmp_path / "mmlu").mkdir(parents=True)
    (tmp_path / "mmlu" / "base.json").write_text("{}")
    mm.upload_mmlu(tmp_path, hf_prefix="p/fu2_mmlu")
    mm.upload_mmlu(tmp_path, hf_prefix="p/fu2_mmlu", hf_repo=overflow)
    assert [s[0] for s in seen] == [mm.DATA_REPO, overflow]


def test_judge_argparser_has_hf_repo_seam():
    import issue2225_judge as judge

    args = judge.build_argparser().parse_args(["--phase", "upload"])
    assert args.hf_repo == judge.DATA_REPO
    args2 = judge.build_argparser().parse_args(
        ["--phase", "upload", "--hf-repo", "superkaiba1/explore-persona-space-overflow"]
    )
    assert args2.hf_repo.endswith("-overflow")
    # run_upload threads args.hf_repo into hub._upload (source pin — the
    # upload body is exercised live by the dispatcher's smoke twin prefixes).
    src = Path(judge.__file__).read_text(encoding="utf-8")
    assert "args.hf_repo," in src


# ── dispatcher wiring (source + evaluated head block) ─────────────────────────


def _dispatch_head(*, until: str) -> str:
    src = _FU2_DISPATCH.read_text(encoding="utf-8")
    return src[src.index('SMOKE="${EPM_I2225_SMOKE') : src.index(until)]


def test_fu2_dispatch_data_repo_wiring(tmp_path):
    """FU2_DATA_HF_REPO defaults to the OVERFLOW repo (#2287) and threads via
    DATA_REPO_ARGS into EVERY data-repo upload class; the pilot-raws heredoc
    reads the same env knob."""
    src = _FU2_DISPATCH.read_text(encoding="utf-8")
    # eval_gen f2b+f3+w2b, mmlu f2c+f3, capture f2d+f3, judge upload = 8 call
    # sites (w2b: the conditional W2b wave's raws-upload leg, plan v13 §4.3).
    assert src.count('"${DATA_REPO_ARGS[@]}"') == 8
    assert 'PILOT_REPO="$FU2_DATA_HF_REPO"' in src
    # fu1's capture-only knob is REPLACED by the round-wide one (no live use;
    # a comment naming the fu1 knob's history is fine).
    assert "CAPTURE_REPO_ARGS" not in src
    assert "EPM_I2225_FU1_CAPTURE_HF_REPO" not in src
    base_env = {"PATH": "/usr/bin:/bin"}
    head = _dispatch_head(until="log_phase() {")
    for env_repo, expect in (
        ("", "--hf-repo superkaiba1/explore-persona-space-overflow"),
        ("my/own-repo", "--hf-repo my/own-repo"),
    ):
        env = {**base_env, "EPM_I2225_FU2_DATA_HF_REPO": env_repo} if env_repo else base_env
        out = subprocess.run(
            ["bash", "-c", head + '\necho "${DATA_REPO_ARGS[@]}"'],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == expect


def test_fu2_dispatch_smoke_roots_and_prefixes():
    """EPM_I2225_SMOKE=1 diverts LOG_ROOT, EVAL_ROOT, OUT_ROOT AND every HF
    prefix to twins; unset keeps the production values (bidirectional pairs)."""
    head = _dispatch_head(until="log_phase() {")
    probe = (
        '\necho "$LOG_ROOT"\necho "$EVAL_ROOT"\necho "$OUT_ROOT"'
        '\necho "$HF_FINAL"\necho "$HF_MMLU"\necho "$HF_CAPTURE"\necho "$HF_PILOT"'
    )
    for smoke, expect in (
        (
            "1",
            [
                "/workspace/logs/issue-2225-fu2_smoke",
                "data/issue_2225/fu2_eval_smoke",
                "/workspace/eps_out/issue2225_fu2_smoke",
                "issue2225_ctxsteer/raw_completions/fu2_final_smoke",
                "issue2225_ctxsteer/fu2_mmlu_smoke",
                "issue2225_ctxsteer/analysis_tensors/fu2_capture_smoke",
                "issue2225_ctxsteer/raw_completions/fu2_pilot_smoke",
            ],
        ),
        (
            "",
            [
                "/workspace/logs/issue-2225-fu2",
                "eval_results/issue_2225/fu2_preimage_alltoken",
                "/workspace/eps_out/issue2225_fu2",
                "issue2225_ctxsteer/raw_completions/fu2_final",
                "issue2225_ctxsteer/fu2_mmlu",
                "issue2225_ctxsteer/analysis_tensors/fu2_capture",
                "issue2225_ctxsteer/raw_completions/fu2_pilot",
            ],
        ),
    ):
        out = subprocess.run(
            ["bash", "-c", head + probe],
            capture_output=True,
            text=True,
            timeout=60,
            env={"PATH": "/usr/bin:/bin", "EPM_I2225_SMOKE": smoke},
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip().split("\n") == expect


def test_fu2_dispatch_env_module_and_verdict_round():
    src = _FU2_DISPATCH.read_text(encoding="utf-8")
    assert "export EPM_I2225_EXTRA_CELLS_MODULE=issue2225_fu2_train" in src
    # Both verdict invocations (F1' + re-verdict) score the fu2 round.
    assert src.count("issue2225_fu1_verdict.py --round fu2") == 2
    # The re-pilot + pilot legs train through the fu2 wrapper.
    assert "issue2225_fu2_train.py --pilot" in src
    # S0 replaces f0: bank verify runs pre-GPU, full artifacts even under smoke.
    assert "issue2225_fu2_bankverify.py" in src
    assert "phase_s0" in src and "phase_f0" not in src


def _hook_gate_script() -> str:
    """Extract the REAL hook_count_gate from the fu2 dispatcher (fu1-verbatim
    copy) and wrap it in a callable probe (args: <log_dir> <slugs>)."""
    src = _FU2_DISPATCH.read_text(encoding="utf-8")
    start = src.index("hook_count_gate() {")
    end = src.index("\n}\n", start) + len("\n}\n")
    return (
        'log_phase() { echo "[phase=$1] $2"; }\n'
        + src[start:end]
        + '\nhook_count_gate "$1" "$2" testphase\n'
    )


def test_fu2_hook_gate_slug_keyed(tmp_path):
    logs = tmp_path / "f1_train"
    logs.mkdir()
    prod = [f"Q__evil__c{v}" for v in ("0.25", "0.75", "1.5", "3.0")]
    for i, slug in enumerate(prod):
        tok = "[steer-hook]" if i % 2 == 0 else "[fanout-skip]"
        (logs / f"{slug}.log").write_text(f"noise\n{tok} engaged\n")
    (logs / "N__evil__c0.25.log").write_text("[steer-hook] smoke residue\n")
    out = subprocess.run(
        ["bash", "-c", _hook_gate_script(), "_", str(logs), ",".join(prod)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
    assert "4/4" in out.stdout
    missing = subprocess.run(
        ["bash", "-c", _hook_gate_script(), "_", str(logs), "Q__evil__c0.25,RQ__evil__c1.5"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert missing.returncode == 7 and "RQ__evil__c1.5" in missing.stderr


# ── conditional W2b reads (plan v13 §4.3) ─────────────────────────────────────


def _write_w2b_arm(root: Path, config: str, coef: float, means: list[float]) -> None:
    """Minimal trait_scores + coherence arm JSONs for the W2b fixture
    (schema: the fields `_per_question_means` / `_arm_curve` consume)."""
    per_q = [{"question_idx": i, "mean": m} for i, m in enumerate(means)]
    coh_q = [{"question_idx": i, "mean": 90.0, "rollout_scores": [90.0]} for i in range(len(means))]
    trait_block = {
        "per_question": per_q,
        "model_mean": sum(means) / len(means),
        "rate_gt50": 0.5,
        "accounting": {"n_api_refusal": 0},
    }
    for sub, block in (
        ("trait_scores", trait_block),
        ("coherence", {"per_question": coh_q, "n_rollouts_scored": len(means)}),
    ):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{config}_sycophancy_{coef}.json", "w") as f:
            json.dump({"traits": {"sycophancy": block}}, f)


def test_w2b_contrasts_skip_branch_records_not_computable(tmp_path):
    """F5 re-run BEFORE the W2b harvest: no RN files -> empty block + both
    keys in not_computable, no raise (plan §4.3 conditional gating)."""
    import issue2225_fu1_analysis as ana

    nc: list[str] = []
    out = ana.fu2_w2b_contrasts(tmp_path, 100, nc)
    assert out == {}
    assert nc == [
        "h4_dod_w2b:N_vs_RN_sycophancy (W2b cells not yet landed)",
        "h4_level_w2b:N_vs_RN_sycophancy_level (W2b cells not yet landed)",
    ]


def test_w2b_contrasts_half_landed_raises(tmp_path):
    """1-3 of the 4 RN window files present -> RuntimeError (a half-landed
    harvest is an upload defect, never a silent not_computable downgrade)."""
    import issue2225_fu1_analysis as ana

    _write_w2b_arm(tmp_path, "RN", 1.5, [50.0] * 4)  # 2 of 4 window files
    nc: list[str] = []
    with pytest.raises(RuntimeError, match="PARTIALLY landed"):
        ana.fu2_w2b_contrasts(tmp_path, 100, nc)
    assert nc == []


def test_w2b_contrasts_compute_branch(tmp_path):
    """Landed W2b cells: matched-window DoD + LEVEL read with the fu2 seed
    scheme (constant per-question deltas pin point + CI + verdict exactly)."""
    import issue2225_fu1_analysis as ana

    n_q = 6
    # dose(N) = 70-40 = 30/question; dose(RN) = 45-40 = 5 -> DoD 25 constant;
    # LEVEL N@3.0 - RN@3.0 = 25 constant.
    for config, coef, base in (
        ("N", 1.5, 40.0),
        ("N", 3.0, 70.0),
        ("RN", 1.5, 40.0),
        ("RN", 3.0, 45.0),
    ):
        _write_w2b_arm(tmp_path, config, coef, [base + i for i in range(n_q)])
    ana._set_round("fu2")
    try:
        nc: list[str] = []
        out = ana.fu2_w2b_contrasts(tmp_path, 200, nc)
    finally:
        ana._set_round("fu1")
    assert nc == []
    assert set(out) == {"N_vs_RN_sycophancy", "N_vs_RN_sycophancy_level"}
    dod = out["N_vs_RN_sycophancy"]
    # seed = 2225 + 1000*di(sycophancy=1) + 100*ci(N=0) + offset(h4_dod_w2b=9)
    assert dod["seed"] == 3234
    assert dod["window"] == [1.5, 3.0]
    assert dod["frozen"]["delta_point"] == pytest.approx(25.0)
    assert dod["frozen"]["ci95"] == [pytest.approx(25.0), pytest.approx(25.0)]
    assert dod["frozen"]["verdict"] == "Effect-positive"
    # selection-inherited flavour rides the DoD (coherence 90 >= 80 everywhere)
    assert dod["selection_inherited"]["delta_point"] == pytest.approx(25.0)
    assert dod["selection_inherited"]["n_draws_no_coherent_coef"] == 0
    lvl = out["N_vs_RN_sycophancy_level"]
    assert lvl["seed"] == 3235  # offset(h4_level_w2b=10)
    assert lvl["matched_coef"] == 3.0
    assert lvl["frozen"]["delta_point"] == pytest.approx(25.0)
    assert lvl["frozen"]["verdict"] == "Effect-positive"
    assert "selection_inherited" not in lvl


def test_fu2_conditional_selection_partial_grid(tmp_path):
    """Selection tolerates RN's PARTIAL grid: only landed coefs enter the
    curve; a dataset with zero landed cells is skipped (byte-identical
    pre-harvest re-runs)."""
    import types

    import issue2225_fu1_analysis as ana

    for coef in (1.5, 3.0):  # the W2b window only — 0.25/0.75 never exist
        _write_w2b_arm(tmp_path, "RN", coef, [50.0] * 4)
    args = types.SimpleNamespace(eval_root=str(tmp_path), configs=None, datasets=None)
    ana._set_round("fu2")
    try:
        selection: dict = {}
        ana._fu2_conditional_selection(args, fu2.effective_fu2_grids(None), {}, selection)
    finally:
        ana._set_round("fu1")
    # RN_evil has no landed cells -> skipped entirely
    assert set(selection) == {"RN_sycophancy"}
    entry = selection["RN_sycophancy"]
    assert entry["grid"] == [1.5, 3.0]
    assert sorted(entry["curve"]) == ["1.5", "3.0"]
    assert entry["selected_coef"] == 3.0  # both coefs coherent (90 >= 80) -> largest
    assert "conditional_partial_grid" in entry
