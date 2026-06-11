# ruff: noqa: RUF003  # em-dash / minus sign intentional
"""Task #601 round-3 blocker regressions.

Pins the two reconciler-verified round-2 blockers so they cannot be
reintroduced:

1. ``smoke-sentinel-processed-race`` — the launch p4 smoke-gate check must
   tolerate ``poll_pipeline.py``'s ``.json -> .json.processed`` rename (the
   poller can drain the sentinel during the dispatcher's post-sentinel HF
   upload window). The tests extract the LITERAL p4 block from
   ``scripts/i601_launch.sh`` and run it under ``set -euo pipefail``.

2. ``margin-coread-series-space-mismatch`` — ``scripts/i601_analyze.py`` must
   hand the margin co-read a MARGIN-space matched-arm series (dense ladder +
   on-policy frac inserts converted in margin space), never the logP primary
   series. The fixture makes the two spaces DIVERGE in shape (margin rises
   early, logP rises late), so a logP series fed to the margin co-read flips
   its horizon/coupling verdicts — the round-2 smoke fixtures had margin ==
   logP everywhere, which is exactly why the mismatch slipped through.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCH_SH = REPO_ROOT / "scripts" / "i601_launch.sh"
ANALYZE_PY = REPO_ROOT / "scripts" / "i601_analyze.py"


# ── 1. Launch p4 smoke gate: .processed tolerance under set -euo pipefail ────


def _p4_block() -> str:
    """The literal p4 smoke-gate block from i601_launch.sh (echo .. heredoc PY)."""
    text = LAUNCH_SH.read_text()
    m = re.search(r'^echo "\[phase=p4_smoke_gate\].*?^PY$', text, flags=re.M | re.S)
    assert m, "p4_smoke_gate block not found in i601_launch.sh"
    return m.group(0)


def _write_sentinel(path: Path, *, smoke_gate_pass: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sentinel_schema_version": 1,
                "kind": "epm:results",
                "version": 1,
                "note": json.dumps({"smoke_gate_pass": smoke_gate_pass}),
            }
        )
    )


def _run_p4(log_dir: Path) -> subprocess.CompletedProcess:
    # Inherit the full env: the literal block shells out to `uv run python`,
    # and uv lives outside the minimal /usr/bin:/bin PATH on the dev VM.
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", _p4_block()],
        env={**os.environ, "LOG_DIR": str(log_dir)},
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_p4_gate_passes_on_bare_sentinel(tmp_path: Path) -> None:
    _write_sentinel(tmp_path / "issue-601-smoke-results.json", smoke_gate_pass=True)
    res = _run_p4(tmp_path)
    assert res.returncode == 0, res.stderr
    assert "smoke gate PASS" in res.stdout


def test_p4_gate_passes_on_processed_rename(tmp_path: Path) -> None:
    """The round-3 blocker: the poller renamed the sentinel before p4 read it."""
    _write_sentinel(tmp_path / "issue-601-smoke-results.json.processed", smoke_gate_pass=True)
    res = _run_p4(tmp_path)
    assert res.returncode == 0, res.stderr
    assert "smoke gate PASS" in res.stdout
    assert ".processed" in res.stdout  # the chosen candidate is reported


def test_p4_gate_fails_when_sentinel_missing(tmp_path: Path) -> None:
    res = _run_p4(tmp_path)
    assert res.returncode != 0
    assert "smoke gate FAILED" in res.stderr + res.stdout


def test_p4_gate_fails_on_processed_non_pass(tmp_path: Path) -> None:
    _write_sentinel(tmp_path / "issue-601-smoke-results.json.processed", smoke_gate_pass=False)
    res = _run_p4(tmp_path)
    assert res.returncode != 0
    assert "smoke gate FAILED" in res.stderr + res.stdout


# ── 2. Analyze: margin co-read consumes the margin-space series ──────────────

# Per-seed (logp, margin) terminals. Margin levels satisfy the margin-space
# horizon LEVEL rules (quarter low / double+matched high vs midpoint
# M(4:1),M(8:1)=50); logP levels satisfy the logP level rules but the logP
# series rises LATE so primary horizon fails on the frac-16 read.
QUARTER = {42: (8.2, 17.5), 137: (8.9, 18.5)}  # T=32
DOUBLE = {42: (19.8, 59.0), 137: (20.2, 61.0)}  # T=125
MATCHED = {42: (20.1, 58.0), 137: (20.5, 60.0)}  # T=128
ANCHOR = {42: (13.4, 40.5), 137: (13.6, 39.5)}
MARGIN_REFS = {"0:1": 4.0, "2:1": 20.0, "4:1": 40.0, "8:1": 60.0}
# Dense ladder WITHOUT step 16 — the frac-16 horizon read comes from the
# on-policy trajectory INSERT, so the insert's own-space conversion is itself
# under test (a logP-form insert in the margin series breaks it).
DENSE_STEPS = (2, 4, 6, 8, 10, 12, 20, 32, 64, 96, 128)


def _traj_ck(frac: float, step: int | None, logp: float, margin: float) -> dict:
    return {
        "frac": frac,
        "step": step,
        "source_self": {
            "delta_g_mean": logp,
            "z_marker_g_mean": margin,  # margin = (z_m_g − z_e_g) − (z_m_b − z_e_b)
            "z_eos_g_mean": 0.0,
            "z_marker_b_mean": 0.0,
            "z_eos_b_mean": 0.0,
        },
    }


def _write_traj(path: Path, points: list[tuple[float, int | None, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"checkpoints": [_traj_ck(*p) for p in points]}))


def _matched_dense_series(seed: int) -> list[tuple[int, float, float]]:
    """(step, logp, margin): logP rises late (frac16≈0.4), margin early (≥0.96)."""
    lp_term, mg_term = MATCHED[seed]
    logp = {
        2: 1.0,
        4: 2.0,
        6: 3.0,
        8: 4.0,
        10: 5.0,
        12: 6.0,
        20: 10.0,
        32: 14.0,
        64: lp_term - 2.0,
        96: lp_term - 0.2,
        128: lp_term,
    }
    margin = {
        2: 20.0,
        4: 35.0,
        6: 45.0,
        8: 50.0,
        10: 53.0,
        12: 55.0,
        20: 57.0,
        32: 57.5,
        64: mg_term - 0.5,
        96: mg_term - 0.2,
        128: mg_term,
    }
    return [(s, logp[s], margin[s]) for s in DENSE_STEPS]


def _build_divergence_slab(slab: Path) -> None:
    p0 = slab / "phase0"
    p0.mkdir(parents=True, exist_ok=True)
    (p0 / "endpoint_reads.json").write_text(
        json.dumps(
            {
                "space_calibration": {"primary_space": "logp_with_margin_upper"},
                "margin_references": {
                    lv: {"margin_mean": m, "tolerance_margin": 6.0} for lv, m in MARGIN_REFS.items()
                },
                "clamp_read": {"clamp_present": True},
            }
        )
    )
    (p0 / "phase0_gate.json").write_text(json.dumps({"pass": True, "anchor_reuse_ok": True}))
    for seed, (lp, mg) in ANCHOR.items():
        _write_traj(
            p0 / "onpolicy_recheck" / f"c472_anchor_seed{seed}" / "trajectory.json",
            [(1.0, 63, lp, mg)],
        )
    for slug, by_seed, t_total in (
        ("ratio4to1_100p400n", QUARTER, 32),
        ("ratio4to1_400p1600n", DOUBLE, 125),
    ):
        for seed, (lp, mg) in by_seed.items():
            _write_traj(
                slab / "phase1" / f"{slug}_seed{seed}" / "trajectory.json",
                [(1.0, t_total, lp, mg)],
            )
    for seed, (lp_term, mg_term) in MATCHED.items():
        cell = slab / "phase1" / f"ratio4to1_100p400n_T128_seed{seed}"
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "dense_trajectory.json").write_text(
            json.dumps(
                {
                    "checkpoints": [
                        {
                            "step": s,
                            "frac": round(s / 128, 4),
                            "source_mean": {
                                "delta_g": lp,
                                "delta_margin": mg,
                                "delta_z_marker": lp,
                            },
                        }
                        for s, lp, mg in _matched_dense_series(seed)
                    ]
                }
            )
        )
        # On-policy frac reads: step 16 is NOT in the dense ladder → inserted.
        _write_traj(
            cell / "trajectory.json",
            [
                (
                    0.125,
                    16,
                    8.0 + 0.2 * (seed == 137),
                    0.96 * mg_term + (0.36 if seed == 42 else 0.0),
                ),
                (1.0, 128, lp_term, mg_term),
            ],
        )


def _load_analyze_module():
    spec = importlib.util.spec_from_file_location("i601_analyze_under_test", ANALYZE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def divergence_payload(tmp_path_factory: pytest.TempPathFactory) -> dict:
    slab = tmp_path_factory.mktemp("i601_divergence_slab")
    _build_divergence_slab(slab)
    out = slab / "analysis" / "classification.json"
    res = subprocess.run(
        [
            sys.executable,
            str(ANALYZE_PY),
            "--slab-root",
            str(slab),
            "--allow-partial",
            "--out-path",
            str(out),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return json.loads(out.read_text())


def test_margin_coread_consumes_margin_series(divergence_payload: dict) -> None:
    """With diverging spaces, the margin co-read reads HORIZON (margin series
    rises early); the logP series fed in its place would read horizon=False /
    coupling=True — the round-2 bug signature."""
    coread = divergence_payload["phase1"]["margin_coread"]
    assert coread["horizon"] is True
    assert coread["coupling"] is False
    assert coread["ratio_set_point_consistent"] is False


def test_primary_classification_stays_in_logp(divergence_payload: dict) -> None:
    """The primary (logP) read is untouched by the fix: the late-rising logP
    series fails the frac-16 horizon read and the step-32 coupling read."""
    phase1 = divergence_payload["phase1"]
    assert phase1["verdicts"]["horizon"] is False
    assert phase1["verdicts"]["coupling"] is False
    assert phase1["call"] == "no-call"


def test_fixture_discriminates_the_bug(divergence_payload: dict, tmp_path: Path) -> None:
    """Re-create the round-2 wiring (logP series → space='margin') directly and
    confirm THIS fixture catches it: the verdicts flip vs the fixed output."""
    from explore_persona_space.experiments.neg_setpoint_601.analysis_lib import classify_phase1

    mod = _load_analyze_module()
    slab = tmp_path / "slab"
    _build_divergence_slab(slab)
    series_logp, series_margin = {}, {}
    for seed in (42, 137):
        cell = slab / "phase1" / f"ratio4to1_100p400n_T128_seed{seed}"
        d = json.loads((cell / "dense_trajectory.json").read_text())
        t = json.loads((cell / "trajectory.json").read_text())
        series_logp[seed] = mod._matched_series(d, t, "logp")
        series_margin[seed] = mod._matched_series(d, t, "margin")
        assert series_logp[seed][1] != series_margin[seed][1]  # spaces genuinely diverge
        assert 16 in series_margin[seed][0]  # the on-policy insert landed
    arm_terminals_margin = {
        "quarter": [mg for _, mg in QUARTER.values()],
        "double": [mg for _, mg in DOUBLE.values()],
        "matched": [mg for _, mg in MATCHED.values()],
        "anchor": [mg for _, mg in ANCHOR.values()],
    }
    kwargs = dict(
        arm_terminals=arm_terminals_margin,
        space="margin",
        margin_refs=MARGIN_REFS,
        margin_tol=6.0,
        clamp_present=True,
    )
    buggy = classify_phase1(matched_series_by_seed=series_logp, **kwargs)["verdicts"]
    fixed = classify_phase1(matched_series_by_seed=series_margin, **kwargs)["verdicts"]
    assert buggy["horizon"] is False and buggy["coupling"] is True  # the bug signature
    assert fixed["horizon"] is True and fixed["coupling"] is False
    assert fixed == divergence_payload["phase1"]["margin_coread"]
