"""#1345 crash-fix r3 regressions: smoke-composition gates (att-20260715-161700).

The relaunched GCP smoke crashed on TWO defects where production-n-calibrated
gates bound at smoke n:

1. **Parity gate structurally unsatisfiable at smoke n.** The plan §7 ±0.02
   anchor gate is defined for the PRODUCTION re-extraction (n≈4724-5000); the
   smoke leg limits to 8 conversations, where a grouped-CV L19 R^2 can never
   reproduce the anchors — all 4 cells FAILed and the run HALTed (exit 3).
   Fix: `--smoke` runs the IDENTICAL parity computation (PASS_UNIFIED) but
   demotes the anchor comparison to informational. Production HALT untouched.

2. **Smoke story-yield halt un-smoked extract_stories.** Pretrained kept=1 <
   the old smoke floor 2 → rc=21 halted the story regime, so the smoke never
   exercised extract_stories (the phase the smoke exists to cover). Fix:
   smoke floor = 1 (any kept story proceeds); the production 400/500
   drop-never-backfill floor and the rc=21 halt path are untouched.

Plus the Check-3 annotation: `orchestrate.env.load_dotenv` logs INFO (not
WARNING) when no .env exists but credential env vars are already ambient —
the GCE lane exports tokens via startup metadata and has NO .env by design.

These tests run the REAL bodies (real `fc._cell_xy` + `fc.heldout_r2_sweep`
on a tiny synthetic n=8 bundle); only the filesystem/HF bundle-load boundary
and the dotenv file load are faked, signature-conformantly.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue1345_common as c  # noqa: E402
import issue1345_cross_regime_transfer as xt  # noqa: E402
import issue1345_fit_cells as m  # noqa: E402
import issue1345_gen_stories as gs  # noqa: E402
import issue1345_operator_comparison as ocm  # noqa: E402
import issue1345_plots as plots_mod  # noqa: E402

from explore_persona_space.orchestrate import env as env_mod  # noqa: E402

_DISPATCH_SH = _REPO_ROOT / "scripts" / "issue1345_dispatch.sh"

N_CONV = 8  # the crashed smoke leg's conversation cap
N_LAYERS = 28  # fc.EXPECTED_LAYERS — parity slices layer 19, so keep the real depth
DIM = 16
ANCHOR_L19 = 0.6731  # instruct/r1 anchor (c.PARITY_ANCHOR_DOC)


# ---------------------------------------------------------------------------
# Bug 1 — parity gate: smoke-informational vs production HALT
# ---------------------------------------------------------------------------
def _tiny_bundle(seed: int = 0) -> dict:
    """Synthetic n=8 bundle in the exact shape `fc._cell_xy` consumes.

    Random X/Y => grouped-CV heldout R^2 far below the 0.6731 anchor, i.e. the
    exact structurally-unsatisfiable smoke condition from the crash.
    """
    rng = np.random.default_rng(seed)
    return {
        "arrays": {
            "slots": rng.normal(size=(N_CONV, 2, N_LAYERS, DIM)).astype(np.float32),
            "profiles": rng.normal(size=(N_CONV, 2, N_LAYERS, DIM)).astype(np.float32),
        },
        "sidecar": {"conv_ids": [f"conv{i}" for i in range(N_CONV)]},
    }


@pytest.fixture()
def parity_env(tmp_path, monkeypatch):
    """One (instruct, r1) parity cell against a tmp anchor + a fake bundle loader."""
    anchor_path = tmp_path / "cells_S1.json"
    r2 = [0.0] * N_LAYERS
    r2[19] = ANCHOR_L19
    anchor_path.write_text(json.dumps({"r2_per_layer_obs": r2}))
    monkeypatch.setattr(c, "PARITY_ANCHOR_FILES", {("instruct", "r1"): str(anchor_path)})
    monkeypatch.setattr(c, "PARITY_ANCHOR_DOC", {("instruct", "r1"): ANCHOR_L19})

    bundle = _tiny_bundle()

    def fake_load_regime_bundle(turnstore_dir: Path, model: str, regime: str) -> dict:
        # Signature-conformant fake at the filesystem/HF staging boundary only;
        # everything downstream (fc._cell_xy, fc.heldout_r2_sweep, the JSON
        # write, the verdict) is the real production body.
        assert (model, regime) == ("instruct", "r1")
        return bundle

    monkeypatch.setattr(m, "load_regime_bundle", fake_load_regime_bundle)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    return out_dir


def test_parity_production_halts_at_smoke_n(parity_env, capsys):
    """Fails-pre-fix shape: at n=8 the binding gate deterministically HALTs (exit 3)."""
    with pytest.raises(SystemExit) as exc:
        m.parity_gate(Path("unused"), parity_env)
    assert exc.value.code == 3
    payload = json.loads((parity_env / "parity_gate.json").read_text())
    assert payload["mode"] == "binding"
    assert payload["pass"] is False
    assert payload["results"]["instruct_r1"]["pass"] is False
    out = capsys.readouterr()
    assert "(FAIL)" in out.out
    assert "[parity] HALT:" in out.err


def test_parity_smoke_informational_no_halt(parity_env, capsys):
    """Passes-post-fix: smoke runs the SAME computation but never HALTs."""
    m.parity_gate(Path("unused"), parity_env, smoke=True)  # must not raise
    out = capsys.readouterr().out
    # fix-engaged signal: the informational line the new branch emits
    assert "[parity][smoke] informational: instruct/r1 ours=" in out
    assert f"(n={N_CONV} — anchor check binds at production n only)" in out
    assert "informational only (production HALT semantics unchanged)" in out
    payload = json.loads((parity_env / "parity_gate.json").read_text())
    assert payload["mode"] == "smoke-informational"
    assert payload["pass"] is None  # non-binding — never a fake verdict
    # The computation genuinely ran (PASS_UNIFIED): a real R^2 + dev landed.
    cell = payload["results"]["instruct_r1"]
    assert np.isfinite(cell["reextracted_l19_r2"])
    assert cell["n_rows"] == N_CONV
    assert cell["abs_dev"] > c.PARITY_TOL  # the unsatisfiable-at-smoke-n condition


def test_parity_smoke_flag_threads_through_cli(parity_env, tmp_path, monkeypatch):
    """`--parity --smoke` via main() reaches parity_gate(smoke=True): no SystemExit."""
    argv = [
        "issue1345_fit_cells.py",
        "--parity",
        "--smoke",
        "--turnstore-dir",
        str(tmp_path),
        "--out-dir",
        str(parity_env),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    m.main()  # must not raise
    payload = json.loads((parity_env / "parity_gate.json").read_text())
    assert payload["mode"] == "smoke-informational"


def test_parity_production_cli_still_halts(parity_env, tmp_path, monkeypatch):
    """Same CLI WITHOUT --smoke keeps the plan §7 HALT (exit 3) verbatim."""
    argv = [
        "issue1345_fit_cells.py",
        "--parity",
        "--turnstore-dir",
        str(tmp_path),
        "--out-dir",
        str(parity_env),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        m.main()
    assert exc.value.code == 3


def test_parity_finalize_units():
    """Verdict helper: production failures raise 3; smoke never raises; clean never raises."""
    with pytest.raises(SystemExit) as exc:
        m._parity_finalize(["instruct_r1"], c.PARITY_TOL, smoke=False)
    assert exc.value.code == 3
    m._parity_finalize(["instruct_r1"], c.PARITY_TOL, smoke=True)  # no raise
    m._parity_finalize([], c.PARITY_TOL, smoke=False)  # no raise
    m._parity_finalize([], c.PARITY_TOL, smoke=True)  # no raise


# ---------------------------------------------------------------------------
# Bug 2 — story yield floor: smoke floor 1; rc=21 halt path stays covered
# ---------------------------------------------------------------------------
def test_smoke_yield_floor_is_one():
    """The crash shape (kept=1 under smoke) now proceeds to extract_stories."""
    assert gs.resolve_yield_floor(True, c.STORY_YIELD_FLOOR) == 1
    gs.enforce_yield_floor(1, gs.resolve_yield_floor(True, c.STORY_YIELD_FLOOR))  # no raise


def test_production_yield_floor_untouched():
    """Production floor stays 400/500 drop-never-backfill (plan §7)."""
    assert c.STORY_YIELD_FLOOR == 400
    assert c.N_STORIES_TARGET == 500
    assert gs.resolve_yield_floor(False, c.STORY_YIELD_FLOOR) == c.STORY_YIELD_FLOOR


def test_yield_floor_halt_path_rc21(capsys):
    """The rc=21 halt path stays covered: kept < floor raises SystemExit(21)."""
    with pytest.raises(SystemExit) as exc:
        gs.enforce_yield_floor(399, c.STORY_YIELD_FLOOR)  # production miss
    assert exc.value.code == 21
    with pytest.raises(SystemExit) as exc:
        gs.enforce_yield_floor(0, 1)  # smoke floor: zero kept still halts
    assert exc.value.code == 21
    err = capsys.readouterr().err
    assert "[yield-floor] FAILED" in err
    assert "rc=21" in err


def test_old_smoke_crash_shape_would_have_halted():
    """Fails-pre-fix pin: the crashed run's shape (kept=1, old floor=2) raised 21."""
    with pytest.raises(SystemExit) as exc:
        gs.enforce_yield_floor(1, 2)
    assert exc.value.code == 21


# ---------------------------------------------------------------------------
# Check 3 — no-.env branch: ambient credentials => INFO, not WARNING
# ---------------------------------------------------------------------------
def _stub_dotenv(monkeypatch):
    monkeypatch.setattr(env_mod, "resolve_dotenv_path", lambda: None)

    def fake_dotenv_load(dotenv_path=None, override=False, **kwargs):
        return False  # filesystem boundary: no .env file to load

    monkeypatch.setattr(env_mod, "_dotenv_load", fake_dotenv_load)


def test_no_dotenv_with_ambient_credentials_logs_info(monkeypatch, caplog):
    """GCE-lane shape: tokens ambient via startup metadata => INFO annotation."""
    _stub_dotenv(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setenv("WANDB_API_KEY", "y")
    with caplog.at_level(logging.INFO, logger=env_mod.logger.name):
        env_mod.load_dotenv()
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("ambient env credentials" in r.getMessage() for r in infos)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not [r for r in warnings if "No .env found" in r.getMessage()]


def test_no_dotenv_without_credentials_still_warns(monkeypatch, caplog):
    """No .env AND no ambient tokens: the original WARNING is preserved."""
    _stub_dotenv(monkeypatch)
    for k in ("HF_TOKEN", "WANDB_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    with caplog.at_level(logging.INFO, logger=env_mod.logger.name):
        env_mod.load_dotenv()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("No .env found" in r.getMessage() for r in warnings)


# ---------------------------------------------------------------------------
# Round 4 — production-n-calibrated / never-executed-at-smoke-n gate class
# (v3 code-review Bug-class sweep; proactive hardening, not a crash round)
# ---------------------------------------------------------------------------
def _r3_bundle(n_rows: int = 3, n_groups: int = 1, seed: int = 0) -> dict:
    """Story-shortfall-grain bundle: kept=1 story -> few rows, one CV group."""
    rng = np.random.default_rng(seed)
    ids = [f"story{i % n_groups}" for i in range(n_rows)]
    return {
        "arrays": {
            "slots": rng.normal(size=(n_rows, 2, N_LAYERS, DIM)).astype(np.float32),
            "profiles": rng.normal(size=(n_rows, 1, N_LAYERS, DIM)).astype(np.float32),
        },
        "sidecar": {"conv_ids": ids},
    }


def test_degenerate_fold_reason_units():
    """Mirrors the reused #825 fold-skip predicate exactly (te>0 AND tr>=3)."""
    ids8 = [f"conv{i}" for i in range(8)]
    assert m.degenerate_fold_reason(ids8, n_folds=5, seed=0) is None
    one_group = m.degenerate_fold_reason(["s0"] * 3, n_folds=5, seed=0)
    assert one_group is not None and "all 5 folds skip" in one_group
    # <=3 rows total: every fold's train side is <3 regardless of grouping
    assert m.degenerate_fold_reason(["a", "b", "c"], n_folds=5, seed=0) is not None
    # transfer shape: big src trains fine even when tgt is one group
    assert m.degenerate_fold_reason(ids8, n_folds=5, seed=0, tgt_conv_ids=["s0"] * 3) is None


def test_subset_rows_production_asserts_smoke_skips(capsys):
    xy = {
        "X": np.zeros((4, 2, 2), np.float32),
        "Y": np.zeros((4, 2, 2), np.float32),
        "conv_ids": np.asarray(["a", "b", "c", "d"]),
    }
    # production: fail-loud assert unchanged
    with pytest.raises(AssertionError, match="matched-subset drift"):
        xt.subset_rows(xy, ["zz"])
    # smoke: informational None (caller skips), logged reason
    assert xt.subset_rows(xy, ["zz"], smoke=True, label="unit") is None
    assert "SKIP unit" in capsys.readouterr().out
    # non-empty selection identical in both modes
    sub = xt.subset_rows(xy, ["a", "c"], smoke=True)
    assert sub is not None and list(sub["conv_ids"]) == ["a", "c"]


def test_run_cells_smoke_skips_degenerate_cell_production_crashes(tmp_path, monkeypatch, capsys):
    """kept=1 grain: smoke skips the r3 cell informationally; production (no
    guard) still crashes inside the reused #825 machinery — the guard is
    smoke-only by construction."""
    bundle = _r3_bundle()
    monkeypatch.setattr(m, "load_regime_bundle", lambda ts, model, regime: bundle)
    cell = next(x for x in c.all_cells() if x["cell_id"] == "R_instruct_r3_context")
    matched = {"shared_r1r2_convs": [f"conv{i}" for i in range(8)], "per_model_r3_pair": {}}
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    m.run_cells(
        Path("unused"),
        out_dir,
        tmp_path / "preds",
        [cell],
        matched,
        n_folds=5,
        seed=0,
        null_draws=3,
        n_boot=25,
        smoke=True,
    )
    assert not list(out_dir.glob("cells_*.json"))  # skipped, nothing written
    assert "SKIP cell R_instruct_r3_context" in capsys.readouterr().out
    with pytest.raises(Exception):  # noqa: B017 — pre-fix crash shape, any raise
        m.run_cells(
            Path("unused"),
            out_dir,
            tmp_path / "preds",
            [cell],
            matched,
            n_folds=5,
            seed=0,
            null_draws=3,
            n_boot=25,
            smoke=False,
        )


def test_build_matched_empty_intersection_actionable(tmp_path, monkeypatch):
    """Item 3: empty 4-stem intersection fails with an actionable RuntimeError
    (never a bare AssertionError) — in both modes, since every downstream
    phase consumes matched_subsets.json."""
    monkeypatch.setattr(m, "bundle_conv_ids", lambda ts, model, regime: [f"{model}_{regime}_only"])
    with pytest.raises(RuntimeError, match="extraction drift"):
        m.build_matched(Path("unused"), tmp_path, r3_models=set())


def test_build_matched_per_model_r3(tmp_path, monkeypatch):
    """Crash-fix r6 (plan §7 per-model yield floor): a halted model's R3 pair
    is omitted + reported, the surviving model's pair still builds."""

    def fake_ids(ts, model, regime):
        if regime in ("r1", "r2"):
            return [f"s{i}" for i in range(8)]
        return [f"{model}_story{i:04d}" for i in range(3) for _ in range(2)]  # 2 rows/story

    monkeypatch.setattr(m, "bundle_conv_ids", fake_ids)
    out = m.build_matched(Path("unused"), tmp_path, r3_models={"instruct"})
    assert sorted(out["per_model_r3_pair"]) == ["instruct"]
    assert out["r3_halted_models"] == ["pretrained"]
    # Whole-halt (the legacy --no-r3 shape): no pairs, both models reported.
    out2 = m.build_matched(Path("unused"), tmp_path, r3_models=set())
    assert out2["per_model_r3_pair"] == {}
    assert out2["r3_halted_models"] == ["instruct", "pretrained"]


def test_select_cells_per_model_r3_drop_and_registry_assert(capsys):
    """A halted model's r3 cell in --cells is a logged DROP (never an unknown-id
    crash — the pre-r6 filter ordering crashed there); unknown ids still crash."""
    ids = ",".join(
        [
            c.cell_id("instruct", "r3", "context"),
            c.cell_id("pretrained", "r3", "context"),
            c.cell_id("pretrained", "r1", "context"),
        ]
    )
    cells = m.select_cells(ids, {"pretrained"})
    kept_ids = {x["cell_id"] for x in cells}
    assert c.cell_id("instruct", "r3", "context") in kept_ids
    assert c.cell_id("pretrained", "r3", "context") not in kept_ids
    assert c.cell_id("pretrained", "r1", "context") in kept_ids
    assert "dropping r3 cells" in capsys.readouterr().out
    # Whole-halt with an explicit r3 id: deliberate drop, not a crash.
    both = m.select_cells(ids, {"instruct", "pretrained"})
    assert {x["cell_id"] for x in both} == {c.cell_id("pretrained", "r1", "context")}
    with pytest.raises(AssertionError, match="unknown cell ids"):
        m.select_cells("R_bogus_r9_context", set())


def test_pair_and_leg_b_smoke_skip_reason_units():
    ids8 = np.asarray([f"conv{i}" for i in range(8)])
    xy8 = {"conv_ids": ids8}
    xy_tiny = {"conv_ids": np.asarray(["s0"] * 3)}
    # production: never a skip reason (no probe runs)
    assert ocm.pair_smoke_skip_reason(xy_tiny, xy_tiny, smoke=False, seed=0) is None
    # smoke: healthy pair passes, degenerate side skips
    assert ocm.pair_smoke_skip_reason(xy8, xy8, smoke=True, seed=0) is None
    assert ocm.pair_smoke_skip_reason(xy8, xy_tiny, smoke=True, seed=0) is not None
    assert ocm.pair_smoke_skip_reason(None, xy8, smoke=True, seed=0) == (
        "empty matched subset at smoke n"
    )
    # Leg B floor: n_common < 2 skips; degenerate paired folds skip; n=8 passes
    assert "n_common=0" in ocm.leg_b_smoke_skip_reason(0, ids8, seed=0)
    assert "n_common=1" in ocm.leg_b_smoke_skip_reason(1, ids8, seed=0)
    assert ocm.leg_b_smoke_skip_reason(8, ids8, seed=0) is None
    assert ocm.leg_b_smoke_skip_reason(3, ["s0"] * 3, seed=0) is not None


def test_headline_boot_stub_and_verdict_consumers():
    """The smoke stub is verdict_for-compatible (NaN CIs, never a fake verdict);
    a headline-skipped transfer JSON (no deltas) yields verdict None."""
    stub = xt._headline_boot_stub("unit reason")
    assert stub["skipped"] == "unit reason"
    assert stub["delta_diff_ci_wholly_below_0"] is False
    assert np.isnan(stub["delta_diff"]["ci_hi"])
    assert (
        plots_mod.verdict_for({"headline_paired_bootstrap": stub, "delta_table_l19": {}}, {})
        is None
    )
    transfer = {
        "headline_paired_bootstrap": stub,
        "delta_table_l19": {"r1->r2": {"delta_l19": 0.0}, "r2->r1": {"delta_l19": 0.0}},
    }
    v = plots_mod.verdict_for(transfer, {"delta_reparam_l19": {"delta_reparam": float("nan")}})
    assert v["verdict"] == "inconclusive"  # NaN delta_reparam never asserts a verdict


def test_gen_rc_route_matrix():
    """rc routing: PER-MODEL halt on rc=21 (crash-fix r6, plan §7 per-model
    yield floor) — the att-20260715-195605 shape (rc_i=0, rc_p=21) halts ONLY
    pretrained; any real crash rc in EITHER model still routes fatal (the v3
    rc-masking fix — (1,21) never rides a halt branch)."""
    text = _DISPATCH_SH.read_text()
    match = re.search(r"^gen_rc_route\(\) \{\n(?:.*\n)*?^\}", text, re.M)
    assert match, "gen_rc_route() not found in issue1345_dispatch.sh"
    func = match.group(0)
    cases = {
        (0, 0): "ok",
        (21, 0): "halt_instruct",
        (0, 21): "halt_pretrained",  # the att-20260715-195605 realized case
        (21, 21): "halt_both",
        (1, 21): "fatal",  # the v3 fix sketch's exact case
        (21, 1): "fatal",
        (1, 0): "fatal",
        (0, 137): "fatal",
    }
    for (rc_i, rc_p), want in cases.items():
        res = subprocess.run(
            ["bash", "-c", f"{func}\ngen_rc_route {rc_i} {rc_p}"],
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, (rc_i, rc_p, res.stderr)
        assert res.stdout.strip() == want, (rc_i, rc_p, res.stdout)


def test_dispatch_threads_smoke_flag_into_analysis_phases():
    """fits/transfer/opcomp launch lines carry $SMOKE_FLAG (dispatcher threading)."""
    text = _DISPATCH_SH.read_text()
    for tag in ("run_per_model fits", "run_per_model transfer", "run_per_model opcomp"):
        line = next(ln for ln in text.split("\n") if tag in ln)
        assert "$SMOKE_FLAG" in line, tag


def test_smoke_pool_gate_weaker_than_production():
    """Item 6 (v3 review): pool >= 2x target is smoke-safe — the smoke gate
    (2 x 3 = 6 seeds) is strictly weaker than the production gate
    (2 x 500 = 1000), so any pool passing production passes smoke."""
    assert gs.SMOKE_N_STORIES == 3
    assert 2 * gs.SMOKE_N_STORIES <= 2 * c.N_STORIES_TARGET
