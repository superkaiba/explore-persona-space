# em-dash intentional
"""Task #601 round-4 Phase-4 structure regressions.

Pins the concern-resolution amendment (phase4-bridge-attn-only-attribution):
#471's posonly rig was ALL-LINEAR r=32 @ lr 5e-6, so ``posonly_alllinear_lr5e6``
is the TRUE single-variable #471 lr-bridge and was promoted to UNCONDITIONAL;
``posonly_attn_lr1e5`` is the only remaining conditional 4b factor cell.

1. Registry membership: alllinear@5e6 unconditional, attn@1e5 conditional,
   ``--cells all`` x ``--seeds 42,137`` resolves exactly 16 (cell, seed) units.
2. ``i601_phase4_verdict.py`` classifies BOTH unconditional bridge cells and
   the routing call gates 4b on any-cell non-arrest (fail-loud on a missing
   bridge input — both cells are main-sweep members now).
3. The dispatcher's ``_check_phase4b_gate`` accepts/refuses the v2 verdict
   payload on the routing fields.
4. The launch p6 call-extraction heredoc + p7 routing branch dispatch
   ``--cells phase4b`` ONLY on call == non-arrest.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_SH = REPO_ROOT / "scripts" / "i601_launch.sh"
VERDICT_PY = REPO_ROOT / "scripts" / "i601_phase4_verdict.py"

sys.path.insert(0, str(REPO_ROOT / "src"))

# Plan §4 band fixtures: non-arrest = ΔG >= 6 by step 13 AND last-3 slope >= 0.3;
# arrest = flat (slope < 0.2 from step <= 4) at <= 4 nats. The arrest series
# flattens by step 2 so the classifier's trailing-mean smoothing (window=2)
# keeps every label->=4 forward diff under the 0.2 slope band.
STEPS_13 = list(range(1, 14))
NON_ARREST_SERIES = [1.5 * s for s in STEPS_13]  # 19.5 @ 13, slope 1.5
ARREST_SERIES = [1.2, 1.4] + [1.5] * 11  # flat 1.5 from step 3


# ── 1. Registry membership + unit count ──────────────────────────────────────


def test_registry_phase4_membership() -> None:
    from explore_persona_space.experiments.neg_setpoint_601 import cell_by_slug

    assert cell_by_slug("posonly_alllinear_lr5e6").conditional is False
    assert cell_by_slug("posonly_attn_lr5e6").conditional is False
    assert cell_by_slug("posonly_attn_lr1e5").conditional is True
    # The true lr-bridge keeps the all-linear default scope + half LR.
    bridge = cell_by_slug("posonly_alllinear_lr5e6")
    assert bridge.lora_targets is None and bridge.lr == 5e-6


def test_cells_all_resolves_16_units_for_registered_seeds() -> None:
    """Replicates the dispatcher's unit resolution (cells_for_request x seed
    intersection): 6 phase1 + 4 phase2 + 2 phase3 + 4 phase4-unconditional."""
    from explore_persona_space.experiments.neg_setpoint_601 import cells_for_request

    cells = cells_for_request("all")
    slugs = {c.slug for c in cells}
    assert "posonly_alllinear_lr5e6" in slugs
    assert "posonly_attn_lr1e5" not in slugs
    units = [(c.slug, s) for c in cells for s in (42, 137) if s in c.seeds]
    assert len(units) == 16, units


def test_phase4b_group_is_single_factor_cell() -> None:
    from explore_persona_space.experiments.neg_setpoint_601 import cells_for_request

    assert [c.slug for c in cells_for_request("phase4b")] == ["posonly_attn_lr1e5"]


# ── 2. Verdict script over the new structure ─────────────────────────────────


def _write_band(slab: Path, slug: str, seed: int, series: list[float]) -> None:
    d = slab / "phase4" / f"{slug}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "inloop_band_trajectory.json").write_text(
        json.dumps({"steps": STEPS_13, "delta_nats": series})
    )


def _run_verdict(slab: Path) -> tuple[subprocess.CompletedProcess, dict | None]:
    out = slab / "phase4" / "phase4a_verdict.json"
    res = subprocess.run(
        [sys.executable, str(VERDICT_PY), "--slab-root", str(slab), "--out-path", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return res, json.loads(out.read_text()) if out.exists() else None


def test_verdict_reads_both_bridge_cells_and_any_nonarrest_dispatches(tmp_path: Path) -> None:
    """alllinear non-arrest + attn arrest -> routing call non-arrest (4b runs)."""
    for seed in (42, 137):
        _write_band(tmp_path, "posonly_alllinear_lr5e6", seed, NON_ARREST_SERIES)
        _write_band(tmp_path, "posonly_attn_lr5e6", seed, ARREST_SERIES)
    res, payload = _run_verdict(tmp_path)
    assert res.returncode == 0, res.stderr
    assert payload["calls"] == {
        "posonly_alllinear_lr5e6": "non-arrest",
        "posonly_attn_lr5e6": "arrest",
    }
    assert payload["call"] == "non-arrest" and payload["dispatch_4b"] is True
    # The corrected attribution rides the sentinel.
    assert (
        "single-variable #471 lr-bridge" in payload["bridge_attribution"]["posonly_alllinear_lr5e6"]
    )
    assert "matching NEITHER" in payload["bridge_attribution"]["posonly_attn_lr5e6"]


def test_verdict_all_arrest_skips_4b(tmp_path: Path) -> None:
    for slug in ("posonly_alllinear_lr5e6", "posonly_attn_lr5e6"):
        for seed in (42, 137):
            _write_band(tmp_path, slug, seed, ARREST_SERIES)
    res, payload = _run_verdict(tmp_path)
    assert res.returncode == 0, res.stderr
    assert payload["call"] == "arrest" and payload["dispatch_4b"] is False


def test_verdict_seed_disagreement_is_ambiguous(tmp_path: Path) -> None:
    _write_band(tmp_path, "posonly_alllinear_lr5e6", 42, ARREST_SERIES)
    _write_band(tmp_path, "posonly_alllinear_lr5e6", 137, NON_ARREST_SERIES)
    for seed in (42, 137):
        _write_band(tmp_path, "posonly_attn_lr5e6", seed, ARREST_SERIES)
    res, payload = _run_verdict(tmp_path)
    assert res.returncode == 0, res.stderr
    assert payload["calls"]["posonly_alllinear_lr5e6"] == "ambiguous"
    assert payload["call"] == "ambiguous" and payload["dispatch_4b"] is False


def test_verdict_fails_loud_on_missing_bridge_cell(tmp_path: Path) -> None:
    """alllinear@5e6 is a main-sweep member now — a missing band trajectory is
    a hard failure, never a silent single-cell verdict."""
    for seed in (42, 137):
        _write_band(tmp_path, "posonly_attn_lr5e6", seed, ARREST_SERIES)
    res, payload = _run_verdict(tmp_path)
    assert res.returncode != 0
    assert "posonly_alllinear_lr5e6" in res.stderr + res.stdout
    assert payload is None


# ── 3. Dispatcher phase4b gate over the v2 payload ───────────────────────────


def _gate(slab: Path, verdict: dict | None):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "dispatch_601_under_test", REPO_ROOT / "scripts" / "dispatch_neg_setpoint_601.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if verdict is not None:
        p = slab / "phase4" / "phase4a_verdict.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(verdict))
    return lambda: mod._check_phase4b_gate(slab, ["posonly_attn_lr1e5"])


def test_phase4b_gate_passes_on_nonarrest_v2_payload(tmp_path: Path) -> None:
    _gate(tmp_path, {"call": "non-arrest", "dispatch_4b": True})()


def test_phase4b_gate_refuses_arrest_and_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="GATE REFUSAL"):
        _gate(tmp_path, {"call": "arrest", "dispatch_4b": False})()
    with pytest.raises(RuntimeError, match="GATE REFUSAL"):
        _gate(tmp_path / "fresh", None)()


# ── 4. Launch p6 call extraction + p7 routing branch ─────────────────────────


def _p6_heredoc() -> str:
    text = LAUNCH_SH.read_text()
    m = re.search(r"PHASE4A_CALL=\$\(uv run python - .*?<<'PY'\n(.*?)\nPY\n\)", text, flags=re.S)
    assert m, "p6 PHASE4A_CALL heredoc not found in i601_launch.sh"
    return m.group(1)


def test_p6_extraction_reads_routing_call(tmp_path: Path) -> None:
    verdict = tmp_path / "phase4a_verdict.json"
    verdict.write_text(json.dumps({"call": "non-arrest", "calls": {}, "dispatch_4b": True}))
    res = subprocess.run(
        [sys.executable, "-c", _p6_heredoc(), str(verdict)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "non-arrest"


def _p7_block() -> str:
    text = LAUNCH_SH.read_text()
    m = re.search(r'^if \[ "\$PHASE4A_CALL" = "non-arrest" \].*?^fi$', text, flags=re.M | re.S)
    assert m, "p7 routing branch not found in i601_launch.sh"
    return m.group(0)


def _run_p7(log_dir: Path, call: str) -> subprocess.CompletedProcess:
    """Run the LITERAL p7 branch with a bash function shadowing `uv` so the
    conditional dispatch records its args instead of launching GPU work."""
    stub = (
        'uv() { echo "UV_STUB $*" >> "$LOG_DIR/stub_calls.log"; '
        'touch "$LOG_DIR/issue-601-phase4b-results.json"; }\n'
    )
    script = stub + _p7_block()
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env={
            **os.environ,
            "LOG_DIR": str(log_dir),
            "SLAB_ROOT": str(log_dir / "slab"),
            "PHASE4A_CALL": call,
            "N_GPUS": "4",
            "EXTRA_SWEEP_ARGS": "",
        },
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_p7_dispatches_phase4b_only_on_nonarrest(tmp_path: Path) -> None:
    res = _run_p7(tmp_path, "non-arrest")
    assert res.returncode == 0, res.stderr
    calls = (tmp_path / "stub_calls.log").read_text()
    assert "--cells phase4b" in calls
    assert "issue-601-phase4b-results.json" in calls


def test_p7_skips_on_arrest_and_ambiguous(tmp_path: Path) -> None:
    for call in ("arrest", "ambiguous"):
        res = _run_p7(tmp_path, call)
        assert res.returncode == 0, res.stderr
        assert "SKIPPED" in res.stdout
    assert not (tmp_path / "stub_calls.log").exists()
