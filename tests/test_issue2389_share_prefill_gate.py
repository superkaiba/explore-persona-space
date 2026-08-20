"""Issue #2389 gate-4b share-prefill battery — fork-delta unit tests.

Covers the gate script's offline-tiny end-to-end path plus the run driver's
freeze-point resolver:

- ``run_battery`` over the script's own offline WordLevel tokenizer + tiny
  random qwen3_next hybrid (no HF fetch, fp32 CPU, exact regime) — verdict
  PASS, every (batch x {unhooked, hooked}) variant passed, report keys
  present (the same battery the pod-side gate writes to
  ``gates/share_prefill_equivalence.json``);
- ``issue2389_run._resolve_share_prefill`` (plan §4.7 item 5 pin 2):
  off => serial; auto + absent artifact => serial (FAIL-OPEN); auto + FAIL
  verdict => serial; auto + PASS verdict => armed.

All fixtures are synthetic/tmp — no committed eval_results reads (no
sparse-cone additions needed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2389_run as R  # noqa: E402
import issue2389_share_prefill_gate as G  # noqa: E402

from tests.test_issue2389_run import _mk_cfg  # noqa: E402


def test_offline_tiny_battery_passes():
    model, tok = G._offline_tiny_model_and_tok()
    report = G.run_battery(
        model,
        tok,
        G.OFFLINE_BATCHES,
        k_eq=3,
        draws=2,
        exact=True,
        render_fn=None,
        ids_fn=None,
        wall_draws=0,
    )
    assert report["verdict"] == "PASS"
    assert report["exact_regime"] is True
    assert report["k_eq"] == 3 and report["draws"] == 2
    assert report["wall_leg_f"] is None  # wall leg skipped at wall_draws=0
    # every (batch x {unhooked, hooked}) variant present and passed
    expected = {f"{b}.{v}" for b in G.OFFLINE_BATCHES for v in ("unhooked", "hooked")}
    assert set(report["variants"]) == expected
    assert all(v["passed"] for v in report["variants"].values())
    assert "criterion" in report


def _write_gate(cfg: R.RunConfig, verdict: str, mode: str | None = None) -> Path:
    cfg.gates_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.gates_dir / R.SHARE_PREFILL_GATE_NAME
    rec: dict = {"verdict": verdict}
    if mode is not None:
        rec["mode"] = mode
    path.write_text(json.dumps(rec))
    return path


def test_resolve_share_prefill_off_stays_serial(tmp_path):
    cfg = _mk_cfg(tmp_path, share_prefill_mode="off")
    _write_gate(cfg, "PASS")  # even a PASS artifact cannot arm mode=off
    cfg = R._resolve_share_prefill(cfg, "anchors")
    assert cfg.share_prefill_armed is False


def test_resolve_share_prefill_auto_absent_artifact_stays_serial(tmp_path):
    cfg = _mk_cfg(tmp_path, share_prefill_mode="auto")
    cfg = R._resolve_share_prefill(cfg, "anchors")
    assert cfg.share_prefill_armed is False  # FAIL-OPEN


def test_resolve_share_prefill_auto_fail_verdict_stays_serial(tmp_path):
    cfg = _mk_cfg(tmp_path, share_prefill_mode="auto")
    _write_gate(cfg, "FAIL")
    cfg = R._resolve_share_prefill(cfg, "grid")
    assert cfg.share_prefill_armed is False


def test_resolve_share_prefill_auto_pass_arms(tmp_path):
    cfg = _mk_cfg(tmp_path, share_prefill_mode="auto")
    _write_gate(cfg, "PASS")
    cfg = R._resolve_share_prefill(cfg, "anchors")
    assert cfg.share_prefill_armed is True


def test_gate_name_matches_run_constant():
    # the gate script writes the exact artifact name the resolver reads
    assert G.GATE_NAME == R.SHARE_PREFILL_GATE_NAME == "share_prefill_equivalence.json"


# ---------------------------------------------------- B6: mode-aware arming


def test_resolve_share_prefill_production_run_refuses_tiny_pass(tmp_path):
    # B6 (r1 review): a NON-tiny run must NOT arm on a tiny-mode PASS — the
    # dispatcher's --smoke branch writes the tiny battery into the SAME path,
    # so a smoke->production sequence would otherwise arm 27B generation on a
    # CPU fp32 verdict before the production-device battery lands (pin 2).
    cfg = _mk_cfg(tmp_path, tiny=False, share_prefill_mode="auto")
    _write_gate(cfg, "PASS", mode="tiny")
    cfg = R._resolve_share_prefill(cfg, "anchors")
    assert cfg.share_prefill_armed is False


def test_resolve_share_prefill_production_run_arms_on_production_pass(tmp_path):
    cfg = _mk_cfg(tmp_path, tiny=False, share_prefill_mode="auto")
    _write_gate(cfg, "PASS", mode="production")
    cfg = R._resolve_share_prefill(cfg, "anchors")
    assert cfg.share_prefill_armed is True


def test_resolve_share_prefill_tiny_run_accepts_tiny_pass(tmp_path):
    # a tiny run is device-matched with the tiny battery — arming is fine
    cfg = _mk_cfg(tmp_path, tiny=True, share_prefill_mode="auto")
    _write_gate(cfg, "PASS", mode="tiny")
    cfg = R._resolve_share_prefill(cfg, "anchors")
    assert cfg.share_prefill_armed is True


# --------------------------------------- B2: default CLI composition binds


def test_gate_default_upload_composes_through_run_parser(tmp_path):
    # B2 (r1 review): the gate CLI's DEFAULT --upload must bind through
    # run.py's own parser — the prior `full` default died in argparse on
    # every real (non --offline-tiny) invocation.
    args = G.parse_args(["--out-root", str(tmp_path / "out")])
    assert args.upload == "hf"
    cfg = G._compose_run_cfg(args)
    assert cfg.upload_mode == "hf"
