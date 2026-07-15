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
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue1345_common as c  # noqa: E402
import issue1345_fit_cells as m  # noqa: E402
import issue1345_gen_stories as gs  # noqa: E402

from explore_persona_space.orchestrate import env as env_mod  # noqa: E402

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
