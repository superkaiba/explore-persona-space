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


def test_resolve_share_prefill_tiny_pass_arms_tiny_but_never_production(tmp_path):
    # CORRECTED (R3 r2 review; the B8 precedent): the old form of this test
    # stopped at "a tiny run arms on a tiny PASS" — thereby blessing the
    # armed {mode: tiny} freeze as a terminal state. Arming the TINY run is
    # still fine (device-matched), but the freeze it writes must NOT arm a
    # later production dispatch sharing the out_root: the adopt path
    # validates the frozen record's regime/mode/digest and resolves SERIAL.
    cfg = _mk_cfg(tmp_path, tiny=True, share_prefill_mode="auto")
    _write_gate(cfg, "PASS", mode="tiny")
    cfg = R._resolve_share_prefill(cfg, "anchors")
    assert cfg.share_prefill_armed is True
    prod = _mk_cfg(tmp_path, tiny=False, share_prefill_mode="auto")
    prod = R._resolve_share_prefill(prod, "anchors")
    assert prod.share_prefill_armed is False


# ------------------------------------------- R3 (r2 review): adopt-path guard


def test_resolve_share_prefill_production_adopt_stays_armed(tmp_path):
    # Positive control for the R3 adopt guard: a matching production-mode
    # armed freeze keeps arming same-regime family adopters (no over-disarm;
    # capregen_anchors maps onto the anchors family freeze).
    cfg = _mk_cfg(tmp_path, tiny=False, share_prefill_mode="auto")
    _write_gate(cfg, "PASS", mode="production")
    cfg = R._resolve_share_prefill(cfg, "anchors")
    assert cfg.share_prefill_armed is True
    peer = _mk_cfg(tmp_path, tiny=False, share_prefill_mode="auto")
    peer = R._resolve_share_prefill(peer, "capregen_anchors")
    assert peer.share_prefill_armed is True


def test_resolve_share_prefill_adopt_survives_benign_gate_rewrite(tmp_path):
    # CORRECTED (round-5 A; the third occurrence of the B8 wrong-direction
    # class): the old form of this test asserted that a SAME-verdict/SAME-mode
    # rewrite (the battery's fresh-``ts`` re-run on the plan §9 designed
    # same-command resume) DISARMS — blessing exactly the spurious disarm
    # that flips regime_fingerprint and quarantines banked shards. The freeze
    # binds to the DECISION digest (verdict+mode), so a benign rewrite keeps
    # the family ARMED.
    cfg = _mk_cfg(tmp_path, tiny=False, share_prefill_mode="auto")
    _write_gate(cfg, "PASS", mode="production")
    cfg = R._resolve_share_prefill(cfg, "anchors")
    assert cfg.share_prefill_armed is True
    (cfg.gates_dir / R.SHARE_PREFILL_GATE_NAME).write_text(
        json.dumps(
            {"verdict": "PASS", "mode": "production", "ts": "2026-08-20T00:00:00+00:00", "rerun": 1}
        )
    )
    peer = _mk_cfg(tmp_path, tiny=False, share_prefill_mode="auto")
    peer = R._resolve_share_prefill(peer, "anchors")
    assert peer.share_prefill_armed is True


def test_resolve_share_prefill_adopt_disarms_on_decision_change(tmp_path):
    # The right direction for the digest leg: a VERDICT or MODE change in the
    # gate artifact (the arming DECISION changed) still disarms adopters.
    cfg = _mk_cfg(tmp_path, tiny=False, share_prefill_mode="auto")
    _write_gate(cfg, "PASS", mode="production")
    cfg = R._resolve_share_prefill(cfg, "anchors")
    assert cfg.share_prefill_armed is True
    (cfg.gates_dir / R.SHARE_PREFILL_GATE_NAME).write_text(
        json.dumps({"verdict": "FAIL", "mode": "production"})
    )
    peer = _mk_cfg(tmp_path, tiny=False, share_prefill_mode="auto")
    peer = R._resolve_share_prefill(peer, "anchors")
    assert peer.share_prefill_armed is False
    # mode flip (production -> tiny) under an armed production freeze: disarm.
    (cfg.gates_dir / R.SHARE_PREFILL_GATE_NAME).write_text(
        json.dumps({"verdict": "PASS", "mode": "tiny"})
    )
    peer2 = _mk_cfg(tmp_path, tiny=False, share_prefill_mode="auto")
    peer2 = R._resolve_share_prefill(peer2, "anchors")
    assert peer2.share_prefill_armed is False


def test_resolve_share_prefill_adopt_disarms_on_absent_gate_artifact(tmp_path):
    # The vanished-evidence half of the old test, retained: an armed freeze
    # whose gate artifact is GONE (or unparseable) resolves SERIAL.
    cfg = _mk_cfg(tmp_path, tiny=False, share_prefill_mode="auto")
    gate_path = _write_gate(cfg, "PASS", mode="production")
    cfg = R._resolve_share_prefill(cfg, "anchors")
    assert cfg.share_prefill_armed is True
    gate_path.unlink()
    peer = _mk_cfg(tmp_path, tiny=False, share_prefill_mode="auto")
    peer = R._resolve_share_prefill(peer, "anchors")
    assert peer.share_prefill_armed is False
    gate_path.write_text("{not json")
    peer2 = _mk_cfg(tmp_path, tiny=False, share_prefill_mode="auto")
    peer2 = R._resolve_share_prefill(peer2, "anchors")
    assert peer2.share_prefill_armed is False


def test_resolve_share_prefill_lost_race_validates_winner(tmp_path, monkeypatch):
    # R3 second bypass site (found by the unsplit Codex arm alone): the
    # LOST-RACE adopt at the os.link EEXIST branch validates the winning
    # record too. Simulate losing the race to a tiny-armed writer: the
    # production resolver would have armed on its own production PASS, but
    # the winner's {armed: true, mode: tiny} record must resolve SERIAL.
    cfg = _mk_cfg(tmp_path, tiny=False, share_prefill_mode="auto")
    _write_gate(cfg, "PASS", mode="production")
    tiny_rec = {
        "armed": True,
        "verdict": "PASS",
        "mode": "tiny",
        "gate_sha256": None,
        "family": "anchors",
        "repro": {
            "tiny": True,
            "smoke": cfg.smoke,
            "model_id": cfg.model_id,
            "model_revision": cfg.model_revision,
        },
    }

    def losing_link(src, dst, *a, **k):
        Path(dst).write_text(json.dumps(tiny_rec))
        raise FileExistsError(dst)

    monkeypatch.setattr(R.os, "link", losing_link)
    cfg = R._resolve_share_prefill(cfg, "anchors")
    assert cfg.share_prefill_armed is False


# --------------------------------------- B2: default CLI composition binds


def test_gate_default_upload_composes_through_run_parser(tmp_path):
    # B2 (r1 review): the gate CLI's DEFAULT --upload must bind through
    # run.py's own parser — the prior `full` default died in argparse on
    # every real (non --offline-tiny) invocation.
    args = G.parse_args(["--out-root", str(tmp_path / "out")])
    assert args.upload == "hf"
    cfg = G._compose_run_cfg(args)
    assert cfg.upload_mode == "hf"


# ------------------- Round-5 A: battery idempotence (skip-if-matching-report)


_TINY_BATTERY_ARGV = ["--skip-wall", "--k-eq", "2", "--draws", "2"]


def test_battery_main_adopts_matching_report_and_force_remeasures(tmp_path):
    # Round-5 A fix half 2 (the _reusable_pilot_report idiom): a completed
    # same-mode report is ADOPTED — the same-command re-run performs NO
    # re-measure and, crucially, NO artifact rewrite (bytes incl. ``ts``
    # unchanged, so any live family freeze keeps validating). --force
    # deliberately re-measures and rewrites. FAILED at HEAD~: every re-run
    # rewrote the artifact with a fresh ``ts``.
    out_root = tmp_path / "out"
    assert G.main(["--offline-tiny", "--out-root", str(out_root), *_TINY_BATTERY_ARGV]) == 0
    gate_path = out_root / "gates" / G.GATE_NAME
    first_bytes = gate_path.read_bytes()
    assert G.main(["--offline-tiny", "--out-root", str(out_root), *_TINY_BATTERY_ARGV]) == 0
    assert gate_path.read_bytes() == first_bytes  # adopted: no rewrite, ts intact
    assert (
        G.main(["--offline-tiny", "--out-root", str(out_root), "--force", *_TINY_BATTERY_ARGV]) == 0
    )
    second = json.loads(gate_path.read_text())
    assert second["verdict"] == json.loads(first_bytes)["verdict"]
    assert gate_path.read_bytes() != first_bytes  # --force re-measured (fresh ts)


def test_battery_main_mode_mismatch_remeasures_and_overwrites(tmp_path):
    # A DIFFERENT-mode report is NOT adoptable — the battery re-measures and
    # overwrites (the designed B6 smoke->production upgrade path, exercised
    # here in the production->offline-tiny direction; never a raise).
    out_root = tmp_path / "out"
    gates_dir = out_root / "gates"
    gates_dir.mkdir(parents=True)
    (gates_dir / G.GATE_NAME).write_text(json.dumps({"verdict": "PASS", "mode": "production"}))
    assert G.main(["--offline-tiny", "--out-root", str(out_root), *_TINY_BATTERY_ARGV]) == 0
    rec = json.loads((gates_dir / G.GATE_NAME).read_text())
    assert rec["mode"] == "offline-tiny" and "ts" in rec


def test_early_separate_battery_sequence_keeps_family_arming_consistent(tmp_path):
    # Round-5 A orchestrator constraint: the B9-family standing remedy — run
    # the gate-4b battery as an EARLIER SEPARATE dispatch, then the anchors
    # family dispatch whose worker-1 chain re-runs the battery — must not
    # split one family across arming values (r4 correctness failure path b).
    # Sequence at tiny scale: battery -> t0 workers freeze ARMED -> battery
    # re-run (worker-1 chain; adopted, no rewrite) -> worker-1 resolver
    # adopts the freeze and stays ARMED. FAILED at HEAD~: the re-run's fresh
    # ``ts`` broke the raw-byte digest and worker 1 resolved SERIAL while
    # workers 0/2..N-1 ran ARMED.
    out_root = tmp_path / "out"
    assert G.main(["--offline-tiny", "--out-root", str(out_root), *_TINY_BATTERY_ARGV]) == 0
    gate_path = out_root / "gates" / G.GATE_NAME
    verdict = json.loads(gate_path.read_text())["verdict"]
    assert verdict == "PASS"  # precondition: the tiny battery arms
    t0 = _mk_cfg(tmp_path, tiny=True, share_prefill_mode="auto")
    t0 = R._resolve_share_prefill(t0, "anchors")
    assert t0.share_prefill_armed is True
    # worker-1 chain: battery re-run into the SAME out_root (adopted).
    assert G.main(["--offline-tiny", "--out-root", str(out_root), *_TINY_BATTERY_ARGV]) == 0
    w1 = _mk_cfg(tmp_path, tiny=True, share_prefill_mode="auto")
    w1 = R._resolve_share_prefill(w1, "anchors")
    assert w1.share_prefill_armed is True  # same family, same arming — no split
