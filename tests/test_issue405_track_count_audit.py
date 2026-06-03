# ruff: noqa: RUF002, RUF003
"""Round-3 unit tests for the exact-equality track-count audit.

Pins round-3 fix 1: `audit_track_counts` must catch BOTH shortfalls
(observed < expected) AND overages (observed > expected) AND
unknown-track rows, surfacing the offending cell_ids on both sides
when a cell_specs.json is supplied.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _import():
    from issue405_analyze import (
        EXPECTED_SEEDS,
        EXPECTED_TRACK_COUNTS,
        audit_track_counts,
    )

    return audit_track_counts, EXPECTED_TRACK_COUNTS, EXPECTED_SEEDS


def _make_results(per_track_cell_seeds: dict[str, list[tuple[str, int]]]) -> list[dict]:
    """Build a minimal results list of {track, cell_id, seed} dicts.

    The audit function reads `track`, `cell_id`, `seed` only.
    """
    out: list[dict] = []
    for track, pairs in per_track_cell_seeds.items():
        for cid, seed in pairs:
            out.append({"track": track, "cell_id": cid, "seed": seed})
    return out


def _write_specs(tmp_path: Path, cells: list[dict]) -> Path:
    """Build a minimal cell_specs.json matching the audit's reader."""
    p = tmp_path / "cell_specs.json"
    p.write_text(json.dumps(cells))
    return p


# Canonical expected sets per plan §0 + §4.6:
#   CORE     = 21 cell_ids × 2 seeds = 42
#   K4_ABLNEG = 1 cell_id × 2 seeds = 2
#   K1_DOSE50 = 3 cell_ids × 2 seeds = 6
def _canonical_specs() -> list[dict]:
    core_ids = (
        [f"K1_c{n:02d}" for n in range(8)]
        + [f"K2_c{n:02d}" for n in range(8, 14)]
        + [f"K4_c{n:02d}" for n in range(14, 20)]
        + ["K8_c20"]
    )
    out: list[dict] = []
    for cid in core_ids:
        out.append({"cell_id": cid, "track": "CORE"})
    out.append({"cell_id": "K4_ABLNEG", "track": "K4_ABLNEG"})
    for cid in ("K1_DOSE50_paramedic", "K1_DOSE50_villain", "K1_DOSE50_poet"):
        out.append({"cell_id": cid, "track": "K1_DOSE50"})
    return out


def _canonical_results() -> list[dict]:
    """50 result rows that exactly match _canonical_specs() × 2 seeds."""
    specs = _canonical_specs()
    pairs: dict[str, list[tuple[str, int]]] = {}
    for s in specs:
        pairs.setdefault(s["track"], []).extend([(s["cell_id"], 42), (s["cell_id"], 137)])
    return _make_results(pairs)


def test_correct_counts_pass(tmp_path):
    """50/50 result files matching the expected per-track counts → no shortfall + no overage."""
    audit_track_counts, expected, _seeds = _import()
    specs_path = _write_specs(tmp_path, _canonical_specs())
    results = _canonical_results()

    audit = audit_track_counts(results, specs_path=specs_path)
    assert audit["observed"] == expected, audit
    assert audit["shortfall"] == {}, audit["shortfall"]
    assert audit["overage"] == {}, audit["overage"]
    assert audit["unknown_tracks"] == [], audit["unknown_tracks"]
    assert audit["missing_cell_seeds"] == [], audit["missing_cell_seeds"]
    assert audit["extra_cell_seeds"] == [], audit["extra_cell_seeds"]


def test_shortfall_detected_with_offending_cell_seeds(tmp_path):
    """Drop one CORE cell-seed pair → shortfall=1 + missing list populated."""
    audit_track_counts, _expected, _seeds = _import()
    specs_path = _write_specs(tmp_path, _canonical_specs())
    results = _canonical_results()
    # Drop K1_c00 seed=42
    results = [r for r in results if not (r["cell_id"] == "K1_c00" and r["seed"] == 42)]

    audit = audit_track_counts(results, specs_path=specs_path)
    assert audit["shortfall"] == {"CORE": 1}, audit
    assert audit["overage"] == {}, audit
    assert ["K1_c00", 42, "CORE"] in audit["missing_cell_seeds"], audit


def test_overage_detected_with_extra_cell_seeds(tmp_path):
    """Add a stale extra CORE row → overage=1 + extra list populated (round-3 new behavior)."""
    audit_track_counts, _expected, _seeds = _import()
    specs_path = _write_specs(tmp_path, _canonical_specs())
    results = _canonical_results()
    # Add a stale K1_c99 seed=42 (not in specs)
    results.append({"track": "CORE", "cell_id": "K1_c99", "seed": 42})

    audit = audit_track_counts(results, specs_path=specs_path)
    assert audit["overage"] == {"CORE": 1}, audit
    assert audit["shortfall"] == {}, audit
    assert ["K1_c99", 42, "CORE"] in audit["extra_cell_seeds"], audit


def test_unknown_track_does_not_count_toward_known_buckets(tmp_path):
    """A result with a `track` value not in EXPECTED_TRACK_COUNTS surfaces as unknown_tracks."""
    audit_track_counts, _expected, _seeds = _import()
    specs_path = _write_specs(tmp_path, _canonical_specs())
    results = _canonical_results()
    results.append({"track": "BOGUS", "cell_id": "x", "seed": 42})

    audit = audit_track_counts(results, specs_path=specs_path)
    assert "BOGUS" in audit["unknown_tracks"], audit
    # Known buckets unaffected.
    assert audit["shortfall"] == {}, audit
    assert audit["overage"] == {}, audit


def test_both_shortfall_and_overage_in_same_run(tmp_path):
    """Drop one CORE + add one ABLNEG extra → both directions populated."""
    audit_track_counts, _expected, _seeds = _import()
    specs_path = _write_specs(tmp_path, _canonical_specs())
    results = _canonical_results()
    results = [r for r in results if not (r["cell_id"] == "K1_c00" and r["seed"] == 137)]
    results.append({"track": "K4_ABLNEG", "cell_id": "K4_ABLNEG_v2", "seed": 42})

    audit = audit_track_counts(results, specs_path=specs_path)
    assert audit["shortfall"] == {"CORE": 1}, audit
    assert audit["overage"] == {"K4_ABLNEG": 1}, audit
    assert ["K1_c00", 137, "CORE"] in audit["missing_cell_seeds"]
    assert ["K4_ABLNEG_v2", 42, "K4_ABLNEG"] in audit["extra_cell_seeds"]


def test_audit_without_specs_still_returns_count_bucket_diffs(tmp_path):
    """When specs_path is None, cell_seed lists are empty but count buckets work."""
    audit_track_counts, expected, _seeds = _import()
    results = _canonical_results()
    # Add one extra
    results.append({"track": "CORE", "cell_id": "K1_c99", "seed": 42})

    audit = audit_track_counts(results, specs_path=None)
    assert audit["overage"] == {"CORE": 1}, audit
    # specs_path=None ⇒ no cell_id-level surfacing.
    assert audit["extra_cell_seeds"] == [], audit
    assert audit["missing_cell_seeds"] == [], audit
    # observed totals still match.
    assert audit["observed"]["CORE"] == expected["CORE"] + 1


# ── Round-4 fix 3: CALLER-LEVEL tests (CLI gate) ──────────────────────────
#
# The round-3 tests above pin the HELPER's return dict only. Round 4 adds
# tests that invoke the ANALYZER ITSELF (the `has_mismatch` gate in main())
# via subprocess so we verify the gate ACTS on the audit dict — specifically
# that the same-track swap case (Codex's exact verified failure) RAISES by
# default and that `--allow-partial` downgrades any mismatch to a warning.

import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402


def _write_result_json(eval_dir: Path, cell_id: str, seed: int, track: str, K: int = 1) -> Path:
    """Write a minimal result.json the analyzer can load.

    Only `track`, `cell_id`, `seed` are needed for the audit gate; the
    other fields are populated so any downstream fit-path that runs after
    the gate (e.g. on the `--allow-partial` happy path) doesn't blow up.
    Uses a real persona name (`paramedic`/`cybersec_consultant`) so
    `build_cell_persona_frame`'s distance lookup against the cached
    layer-20 cosine matrix works.
    """
    cell_dir = eval_dir / f"cell_{cell_id}_seed{seed}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "cell_id": cell_id,
        "seed": seed,
        "track": track,
        "K": K,
        "spec": {
            "cell_id": cell_id,
            "track": track,
            "K": K,
            "positives": ["paramedic"],
            "negatives": ["software_engineer"],
            "held_out": ["cybersec_consultant"],
        },
        "eval": {
            "held_out": {
                "cybersec_consultant": {
                    "deltaLogP_per_q": [0.5, 0.6],
                    "deltaLogP_mean": 0.55,
                    "emit_rate": 0.0,
                }
            },
            "trained_positive": {},
            "summary": {"trained_pos_mean_dlogp": 0.7},
        },
    }
    (cell_dir / "result.json").write_text(json.dumps(body))
    return cell_dir / "result.json"


def _run_analyzer_cli(
    eval_dir: Path,
    specs_path: Path,
    out_path: Path,
    extra_args: tuple[str, ...] = (),
) -> tuple[int, str, str]:
    """Invoke `scripts/issue405_analyze.py` as a subprocess via uv."""
    cmd = [
        "uv",
        "run",
        "python",
        str(REPO_ROOT / "scripts" / "issue405_analyze.py"),
        "--eval-dir",
        str(eval_dir),
        "--cell-specs-path",
        str(specs_path),
        "--out-path",
        str(out_path),
        "--skip-plots",
    ]
    cmd.extend(extra_args)
    env = {**os.environ}
    env.setdefault("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub"))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
    return proc.returncode, proc.stdout, proc.stderr


def _write_canonical_result_set(eval_dir: Path) -> None:
    """Write all 50 result.json files matching canonical specs × 2 seeds."""
    for spec in _canonical_specs():
        for seed in (42, 137):
            _write_result_json(eval_dir, spec["cell_id"], seed, spec["track"], K=1)


def test_cli_same_track_swap_raises_by_default(tmp_path):
    """CRITICAL — Codex's exact verified failure case.

    Drop planned `K1_c00 seed=42`; add stale `K1_c99 seed=42`. Aggregate
    per-track counts (CORE=42, ABLNEG=2, DOSE50=6) are UNCHANGED. Round 3
    would NOT raise (per-track buckets balance). Round 4 MUST raise on the
    cell-seed identity mismatch.
    """
    specs_path = _write_specs(tmp_path, _canonical_specs())
    eval_dir = tmp_path / "eval"
    out_path = tmp_path / "out" / "regression.json"
    eval_dir.mkdir()
    _write_canonical_result_set(eval_dir)

    shutil.rmtree(eval_dir / "cell_K1_c00_seed42")
    _write_result_json(eval_dir, "K1_c99", 42, "CORE", K=1)

    rc, stdout, stderr = _run_analyzer_cli(eval_dir, specs_path, out_path)
    combined = stdout + stderr
    assert rc != 0, (
        f"Same-track swap MUST raise by default (round-3 gap). rc={rc} stderr_tail={stderr[-1500:]}"
    )
    assert "K1_c00" in combined and "K1_c99" in combined, (
        f"Error must mention both missing and extra cell_id. Got: {combined[-1500:]}"
    )
    assert "missing_cell_seeds" in combined and "extra_cell_seeds" in combined, (
        f"Error must surface missing_cell_seeds + extra_cell_seeds. Got: {combined[-1500:]}"
    )


def test_cli_same_track_swap_downgraded_by_allow_partial(tmp_path):
    """`--allow-partial` downgrades the same-track-swap mismatch to a warning.

    Records both directions in regression.json under `track_counts` +
    `partial_mismatch=True`.
    """
    specs_path = _write_specs(tmp_path, _canonical_specs())
    eval_dir = tmp_path / "eval"
    out_path = tmp_path / "out" / "regression.json"
    eval_dir.mkdir()
    _write_canonical_result_set(eval_dir)

    shutil.rmtree(eval_dir / "cell_K1_c00_seed42")
    _write_result_json(eval_dir, "K1_c99", 42, "CORE", K=1)

    rc, _stdout, stderr = _run_analyzer_cli(
        eval_dir, specs_path, out_path, extra_args=("--allow-partial",)
    )
    assert rc == 0, (
        f"--allow-partial MUST downgrade the mismatch. rc={rc} stderr_tail={stderr[-1500:]}"
    )
    written = json.loads(out_path.read_text())
    assert written.get("partial_mismatch") is True, (
        f"regression.json must record partial_mismatch=True. Got keys: {list(written.keys())}"
    )
    audit = written["track_counts"]
    assert ["K1_c00", 42, "CORE"] in audit["missing_cell_seeds"], audit["missing_cell_seeds"]
    assert ["K1_c99", 42, "CORE"] in audit["extra_cell_seeds"], audit["extra_cell_seeds"]


def test_cli_shortfall_raises_by_default(tmp_path):
    """Drop one row → CORE=41 < 42 → gate must raise by default."""
    specs_path = _write_specs(tmp_path, _canonical_specs())
    eval_dir = tmp_path / "eval"
    out_path = tmp_path / "out" / "regression.json"
    eval_dir.mkdir()
    _write_canonical_result_set(eval_dir)

    shutil.rmtree(eval_dir / "cell_K1_c00_seed42")

    rc, stdout, stderr = _run_analyzer_cli(eval_dir, specs_path, out_path)
    combined = stdout + stderr
    assert rc != 0, f"Shortfall MUST raise. rc={rc} stderr_tail={stderr[-1500:]}"
    assert "shortfall" in combined.lower() or "K1_c00" in combined, combined[-1500:]


def test_cli_overage_raises_by_default(tmp_path):
    """Add one extra → CORE=43 > 42 → gate must raise by default."""
    specs_path = _write_specs(tmp_path, _canonical_specs())
    eval_dir = tmp_path / "eval"
    out_path = tmp_path / "out" / "regression.json"
    eval_dir.mkdir()
    _write_canonical_result_set(eval_dir)
    _write_result_json(eval_dir, "K1_c99", 42, "CORE", K=1)

    rc, stdout, stderr = _run_analyzer_cli(eval_dir, specs_path, out_path)
    combined = stdout + stderr
    assert rc != 0, f"Overage MUST raise. rc={rc} stderr_tail={stderr[-1500:]}"
    assert "overage" in combined.lower() or "K1_c99" in combined, combined[-1500:]


def test_cli_unknown_track_raises_by_default(tmp_path):
    """A result with `track="BOGUS"` surfaces as unknown_tracks and raises."""
    specs_path = _write_specs(tmp_path, _canonical_specs())
    eval_dir = tmp_path / "eval"
    out_path = tmp_path / "out" / "regression.json"
    eval_dir.mkdir()
    _write_canonical_result_set(eval_dir)
    _write_result_json(eval_dir, "BOGUS_cell", 42, "BOGUS", K=1)

    rc, stdout, stderr = _run_analyzer_cli(eval_dir, specs_path, out_path)
    combined = stdout + stderr
    assert rc != 0, f"Unknown track MUST raise. rc={rc} stderr_tail={stderr[-1500:]}"
    assert "BOGUS" in combined or "unknown_track" in combined.lower(), combined[-1500:]
