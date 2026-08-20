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


def _write_gate(cfg: R.RunConfig, verdict: str) -> Path:
    cfg.gates_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.gates_dir / R.SHARE_PREFILL_GATE_NAME
    path.write_text(json.dumps({"verdict": verdict}))
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
