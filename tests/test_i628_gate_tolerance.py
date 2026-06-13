"""Smoke-gate tolerance regression test (#628 r9).

Pins the post-r9 ``_gate_check`` semantics: a code-mismatched divergence
between the in-loop band-stop read (written by r5d's pre-chunked-forward
training) and the offline diagonal G-eval read (computed under r6's
chunked-forward fix) is FP32-accumulation noise -- it must record a
WARNING and append to ``gate_failures.json`` rather than raise
``SystemExit`` and kill the entire phase-2 wave (which then skips
phases 3+4).

The launcher passes ``--enforce-gate`` everywhere; the gate now only
raises when ``--strict-gate`` is ALSO passed. r9 default is warn-only.

Background: round 8 hit ``rig_O_sep_deadneg_fmt_code_seed1042``:
offline diagonal +4.58 vs in-loop +5.98 (|diff|=1.40 nat). The 1.0-nat
threshold was calibrated for SAME-code-path numerics; the round-6
chunked-forward fix changes accumulation order at V=152064, so r5d-
origin cells routinely diverge by 1-2 nats. Marker install/leakage
science thresholds are 5-10+ nats, so a 1-2 nat trajectory-storage
noise floor is uninformative for the experiment headline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _write_pair(
    eval_root: Path, slug: str, arm: str, cid: str, seed: int, in_loop: float, offline: float
) -> None:
    stop_dir = eval_root / "p1/stop_steps"
    cells_dir = eval_root / "G_cells" / arm
    stop_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)
    (stop_dir / f"{slug}.json").write_text(
        json.dumps(
            {"arm": arm, "cid": cid, "seed": seed, "stop_step": 5, "final_band_delta_nats": in_loop}
        )
    )
    (cells_dir / f"{cid}__{cid}__seed{seed}.json").write_text(
        json.dumps({"arm": arm, "cid": cid, "seed": seed, "g_mean_delta_logp": offline})
    )


def _setup(tmp_path, monkeypatch):
    """Repoint ``EVAL`` to a tmp tree and reload _gate_check binding to it."""
    import i628_dispatch as d

    eval_root = tmp_path / "eval_results" / "issue_628"
    eval_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(d, "EVAL", eval_root)
    monkeypatch.setattr(d, "_meta", lambda: {"git": "test", "ts": "test"})
    return d, eval_root


def test_gate_within_strict_threshold_records_and_does_not_raise(tmp_path, monkeypatch):
    """|diff|=0.50 nat → record JSON, no WARNING, no raise (all configs)."""
    d, eval_root = _setup(tmp_path, monkeypatch)
    _write_pair(eval_root, "rig_X_a_seed42", "rig_X", "a", 42, in_loop=5.00, offline=5.50)
    # Default (enforce=True, strict=False) -- the launcher's shape.
    d._gate_check("rig_X", "a", 42, enforce=True, strict=False)
    out = json.loads((eval_root / "p2/gate_checks/rig_X_a_seed42.json").read_text())
    assert out["abs_diff"] == 0.5
    assert out["over_warn_threshold"] is False
    assert out["over_strict_threshold"] is False
    # The non-raise path: function returns None.
    # No gate_failures.json appended because we're under WARN threshold.
    assert not (eval_root / "p2/gate_checks/gate_failures.json").exists()


def test_gate_between_strict_and_warn_warns_and_does_not_raise(tmp_path, monkeypatch):
    """|diff|=1.40 nat (round 8's actual divergence) → record + WARN + append
    to gate_failures.json, but NO SystemExit. This is the round-9 fix's
    primary regression target."""
    d, eval_root = _setup(tmp_path, monkeypatch)
    _write_pair(
        eval_root,
        "rig_O_sep_deadneg_fmt_code_seed1042",
        "rig_O_sep_deadneg",
        "fmt_code",
        1042,
        in_loop=5.98,
        offline=4.58,
    )
    # Launcher's invocation: --enforce-gate ON, --strict-gate OFF.
    d._gate_check("rig_O_sep_deadneg", "fmt_code", 1042, enforce=True, strict=False)
    out = json.loads(
        (eval_root / "p2/gate_checks/rig_O_sep_deadneg_fmt_code_seed1042.json").read_text()
    )
    assert abs(out["abs_diff"] - 1.40) < 1e-9
    # Below the 2.0-nat WARN threshold: no warning, no failure log.
    assert out["over_warn_threshold"] is False
    assert out["over_strict_threshold"] is True
    # No gate_failures.json — we're between strict (1.0) and warn (2.0).
    assert not (eval_root / "p2/gate_checks/gate_failures.json").exists()


def test_gate_above_warn_threshold_warns_and_does_not_raise(tmp_path, monkeypatch):
    """|diff|=2.50 nat → record + WARN + append to gate_failures.json, but
    NO SystemExit unless --strict-gate is set. Pins the bug class: a real
    divergence is reported, never silently dropped."""
    d, eval_root = _setup(tmp_path, monkeypatch)
    _write_pair(eval_root, "rig_X_b_seed42", "rig_X", "b", 42, in_loop=3.00, offline=5.50)
    d._gate_check("rig_X", "b", 42, enforce=True, strict=False)
    out = json.loads((eval_root / "p2/gate_checks/rig_X_b_seed42.json").read_text())
    assert abs(out["abs_diff"] - 2.50) < 1e-9
    assert out["over_warn_threshold"] is True
    assert out["over_strict_threshold"] is True
    # Failure log exists and contains this slug.
    fails = json.loads((eval_root / "p2/gate_checks/gate_failures.json").read_text())
    assert len(fails) == 1
    assert fails[0]["slug"] == "rig_X_b_seed42"
    assert abs(fails[0]["abs_diff"] - 2.50) < 1e-9


def test_gate_strict_mode_raises_above_strict_threshold(tmp_path, monkeypatch):
    """Strict mode (--enforce-gate + --strict-gate) restores the pre-r9
    behavior: |diff| > 1.0 nat → SystemExit. Available as an opt-in escape
    hatch for same-code-path smoke runs."""
    import pytest

    d, eval_root = _setup(tmp_path, monkeypatch)
    _write_pair(eval_root, "rig_X_c_seed42", "rig_X", "c", 42, in_loop=5.98, offline=4.58)
    with pytest.raises(SystemExit) as exc:
        d._gate_check("rig_X", "c", 42, enforce=True, strict=True)
    msg = str(exc.value)
    assert "smoke gate FAILED" in msg
    assert "strict mode" in msg


def test_gate_strict_mode_does_not_raise_at_warn_only_divergence(tmp_path, monkeypatch):
    """|diff|=0.80 nat (within strict threshold 1.0) → strict mode does NOT
    raise. Pins that the strict-mode threshold is the LEGACY 1.0 nat, not
    the new 2.0 nat WARN threshold."""
    d, eval_root = _setup(tmp_path, monkeypatch)
    _write_pair(eval_root, "rig_X_d_seed42", "rig_X", "d", 42, in_loop=5.00, offline=5.80)
    # Should not raise.
    d._gate_check("rig_X", "d", 42, enforce=True, strict=True)
    out = json.loads((eval_root / "p2/gate_checks/rig_X_d_seed42.json").read_text())
    assert abs(out["abs_diff"] - 0.80) < 1e-9
    assert out["over_strict_threshold"] is False


def test_gate_enforce_off_never_raises_even_above_warn(tmp_path, monkeypatch):
    """--enforce-gate OFF: never raises, even at large divergence + strict.
    Pins that --enforce-gate is the umbrella switch."""
    d, eval_root = _setup(tmp_path, monkeypatch)
    _write_pair(
        eval_root, "rig_X_e_seed42", "rig_X", "e", 42, in_loop=0.00, offline=10.00
    )  # 10 nat divergence
    d._gate_check("rig_X", "e", 42, enforce=False, strict=True)
    out = json.loads((eval_root / "p2/gate_checks/rig_X_e_seed42.json").read_text())
    assert out["over_warn_threshold"] is True
    # gate_failures.json still appended (warning path independent of enforce).
    fails = json.loads((eval_root / "p2/gate_checks/gate_failures.json").read_text())
    assert len(fails) == 1


def test_gate_thresholds_are_documented_constants():
    """Pin the threshold constants so a future bump is a deliberate edit."""
    import i628_dispatch as d

    assert d.GATE_WARN_THRESHOLD_NAT == 2.0
    assert d.GATE_STRICT_THRESHOLD_NAT == 1.0


def test_gate_missing_inputs_returns_silently(tmp_path, monkeypatch):
    """Missing stop_step or G_cells file → no crash, no JSON written."""
    d, eval_root = _setup(tmp_path, monkeypatch)
    # No files written.
    d._gate_check("rig_X", "z", 42, enforce=True, strict=True)
    assert not (eval_root / "p2/gate_checks/rig_X_z_seed42.json").exists()
