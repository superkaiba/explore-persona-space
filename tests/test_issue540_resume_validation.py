# ruff: noqa: RUF002
"""Regression tests for #540's resume-skip artifact-compatibility validation.

Round-2 review fix (binding blocker `resume-skip-shape-validation` /
`resume-cache-compatibility`): a stale smoke / descope / stub / tiny-model
JSON in the out-dir must be RECOMPUTED (Phase S/T resume) or HARD-FAIL the
run (Phase M/A assembly) — never silently skipped into the headline 416-cell
matrix, and never ``min()``-downscoped. Extends the 0-byte-partial
resume-skip hardening from commit ddf4cb9b7 (now subsumed by the JSON-parse
branch of the validating loader).

CPU-only, hermetic: probe lists are injected via the module's probe cache so
no data files / network are touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue540_jsrb_predictor as drv  # noqa: E402

GATED_PAIR = ("A1", "instr_explicit_1")  # default --pos0-check-pairs entry
FAKE_PROBES_50 = [f"question {i}?" for i in range(50)]
FAKE_PROBES_2 = FAKE_PROBES_50[:2]


@pytest.fixture(autouse=True)
def _seed_probe_cache():
    """Inject fake probe lists so _current_probes never touches data/HF."""
    drv._PROBES_CACHE.clear()
    drv._PROBES_CACHE[50] = list(FAKE_PROBES_50)
    drv._PROBES_CACHE[2] = list(FAKE_PROBES_2)
    yield
    drv._PROBES_CACHE.clear()


def _args(tmp_path: Path, **overrides):
    """Production-shaped args (plan §10 card) with an isolated out-dir."""
    args = drv._build_parser().parse_args(["--out-dir", str(tmp_path)])
    for key, val in overrides.items():
        setattr(args, key, val)
    return args


def _pair_payload(
    a: str,
    b: str,
    *,
    n_probes: int = 50,
    r_samples: int = 8,
    model: str = drv.BASE_MODEL,
    stub: bool = False,
    seed: int = 42,
    max_new_tokens: int = 256,
    schema: str = drv.SCHEMA_VERSION,
    pos0: dict | None = None,
) -> dict:
    return {
        "schema_version": schema,
        "phase": "scoring",
        "pair": {"a": a, "b": b},
        "is_selfpair": a == b,
        "n_probes": n_probes,
        "r_samples": r_samples,
        "js_rb_bits": 0.1,
        "mc_se_js_bits": 0.01,
        "pos0_v1_check": pos0,
        "metadata": {
            "model": model,
            "stub": stub,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
        },
    }


def _samples_payload(
    ctx: str,
    probes: list[str],
    *,
    r_samples: int = 8,
    stub: bool = False,
    seed: int = 42,
    max_new_tokens: int = 256,
    schema: str = drv.SCHEMA_VERSION,
) -> dict:
    return {
        "schema_version": schema,
        "phase": "sampling",
        "context": ctx,
        "n_probes": len(probes),
        "r_samples": r_samples,
        "probes": probes,
        "metadata": {
            "stub": stub,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "base_model": drv.BASE_MODEL,
        },
        "truncation_rate": 0.0,
    }


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


# ── Per-pair artifacts (Phase T resume-skip / Phase M-A assembly) ──────────


def test_stale_smaller_shape_pair_is_recomputed_not_skipped(tmp_path):
    """THE round-1 blocker: a 2-probe×2-sample smoke pair JSON sitting in the
    out-dir must NOT satisfy a full-run (50×8) resume-skip."""
    args = _args(tmp_path)
    a, b = GATED_PAIR
    path = _write(
        tmp_path / "per_pair" / f"pair_{a}__{b}.json",
        _pair_payload(a, b, n_probes=2, r_samples=2, pos0={"abs_diff": 0.0}),
    )
    assert drv._load_compatible_pair(args, path, a, b, strict=False) is None


def test_matching_pair_artifact_is_skipped(tmp_path):
    args = _args(tmp_path)
    a, b = GATED_PAIR
    path = _write(
        tmp_path / "per_pair" / f"pair_{a}__{b}.json",
        _pair_payload(a, b, pos0={"abs_diff": 0.0}),
    )
    payload = drv._load_compatible_pair(args, path, a, b, strict=False)
    assert payload is not None and payload["js_rb_bits"] == 0.1


def test_zero_byte_partial_is_recomputed(tmp_path):
    """Pin of the ddf4cb9b7 hardening: killed-worker partials never skip."""
    args = _args(tmp_path)
    path = tmp_path / "per_pair" / "pair_A1__A2.json"
    path.parent.mkdir(parents=True)
    path.touch()  # 0 bytes
    assert drv._load_compatible_pair(args, path, "A1", "A2", strict=False) is None


def test_strict_mode_raises_on_stale_artifact(tmp_path):
    """Phase M/A must hard-fail (never assemble) on a shape mismatch."""
    args = _args(tmp_path)
    a, b = GATED_PAIR
    path = _write(
        tmp_path / "per_pair" / f"pair_{a}__{b}.json",
        _pair_payload(a, b, n_probes=2, r_samples=2, pos0={"abs_diff": 0.0}),
    )
    with pytest.raises(RuntimeError, match="stale/incompatible"):
        drv._load_compatible_pair(args, path, a, b, strict=True)


def test_strict_mode_raises_on_missing_artifact(tmp_path):
    args = _args(tmp_path)
    with pytest.raises(RuntimeError, match="missing"):
        drv._load_compatible_pair(
            args, tmp_path / "per_pair" / "pair_A1__A2.json", "A1", "A2", strict=True
        )


def test_stub_pair_refused_by_non_stub_invocation(tmp_path):
    """A --stub-samples-scored pair never enters a real run's matrix."""
    args = _args(tmp_path)
    path = _write(
        tmp_path / "per_pair" / "pair_A1__A2.json",
        _pair_payload("A1", "A2", stub=True),
    )
    assert drv._load_compatible_pair(args, path, "A1", "A2", strict=False) is None
    with pytest.raises(RuntimeError, match="stub"):
        drv._load_compatible_pair(args, path, "A1", "A2", strict=True)


def test_tiny_model_pair_refused_by_real_model_invocation(tmp_path):
    args = _args(tmp_path)
    path = _write(
        tmp_path / "per_pair" / "pair_A1__A2.json",
        _pair_payload("A1", "A2", model="/tmp/tiny540"),
    )
    assert drv._load_compatible_pair(args, path, "A1", "A2", strict=False) is None


def test_pos0_gate_evidence_required_for_gated_pair(tmp_path):
    """Params-matching artifact WITHOUT pos0_v1_check is stale for a gated
    pair (the gate must have fired) — but fine for a non-gated pair."""
    args = _args(tmp_path)
    a, b = GATED_PAIR
    gated = _write(tmp_path / "per_pair" / f"pair_{a}__{b}.json", _pair_payload(a, b, pos0=None))
    assert drv._load_compatible_pair(args, gated, a, b, strict=False) is None
    with pytest.raises(RuntimeError, match="pos0_v1_check"):
        drv._load_compatible_pair(args, gated, a, b, strict=True)
    ungated = _write(
        tmp_path / "per_pair" / "pair_A1__A2.json", _pair_payload("A1", "A2", pos0=None)
    )
    assert drv._load_compatible_pair(args, ungated, "A1", "A2", strict=False) is not None


# ── Samples artifacts (Phase S resume-skip / Phase T input) ────────────────


def test_samples_probe_count_mismatch_is_not_downscoped(tmp_path):
    """Round-1's ``min(samples['n_probes'], args.n_probes)`` is gone: a
    2-probe samples artifact hard-fails a 50-probe Phase T, never shrinks it."""
    args = _args(tmp_path)
    path = _write(tmp_path / "samples" / "samples_A1.json", _samples_payload("A1", FAKE_PROBES_2))
    assert drv._load_compatible_samples(args, path, "A1", strict=False) is None
    with pytest.raises(RuntimeError, match="stale/incompatible"):
        drv._load_samples(args, "A1")


def test_samples_probe_text_mismatch_refused(tmp_path):
    args = _args(tmp_path)
    wrong_probes = ["different question?", *FAKE_PROBES_50[1:]]
    path = _write(tmp_path / "samples" / "samples_A1.json", _samples_payload("A1", wrong_probes))
    assert drv._load_compatible_samples(args, path, "A1", strict=False) is None


def test_samples_matching_artifact_accepted(tmp_path):
    args = _args(tmp_path)
    _write(tmp_path / "samples" / "samples_A1.json", _samples_payload("A1", FAKE_PROBES_50))
    payload = drv._load_samples(args, "A1")
    assert payload["n_probes"] == 50


def test_old_schema_version_refused(tmp_path):
    """issue540_v1 artifacts (round-1 smoke vintage) are structurally
    incompatible and recomputed."""
    args = _args(tmp_path)
    path = _write(
        tmp_path / "per_pair" / "pair_A1__A2.json",
        _pair_payload("A1", "A2", schema="issue540_v1"),
    )
    assert drv._load_compatible_pair(args, path, "A1", "A2", strict=False) is None


# ── Default out-dir routing (smoke can never land in the production dir) ───


def test_default_out_dir_routes_production_shape_to_production_dir():
    args = drv._build_parser().parse_args([])
    drv._resolve_dirs(args)
    assert args.out_dir == drv.DEFAULT_OUT_DIR
    assert args.figures_dir == drv.DEFAULT_FIGURES_DIR


def test_default_out_dir_routes_smoke_shape_to_smoke_dir():
    """The plan §10 smoke command omits --out-dir — it must land in
    eval_results/issue_540_smoke, NOT the production dir."""
    args = drv._build_parser().parse_args(
        [
            "--phases",
            "S,T",
            "--pairs",
            "A1__instr_explicit_1",
            "--n-probes",
            "2",
            "--r-samples",
            "2",
            "--pair-shard",
            "0/1",
        ]
    )
    drv._resolve_dirs(args)
    assert args.out_dir == drv.SMOKE_OUT_DIR
    assert args.figures_dir == drv.SMOKE_FIGURES_DIR


def test_explicit_out_dir_always_respected(tmp_path):
    args = drv._build_parser().parse_args(
        ["--out-dir", str(tmp_path), "--n-probes", "2", "--r-samples", "2"]
    )
    drv._resolve_dirs(args)
    assert args.out_dir == tmp_path


# ── Atomic JSON writes ──────────────────────────────────────────────────────


def test_write_json_atomic_leaves_no_tmp(tmp_path):
    path = tmp_path / "out.json"
    drv._write_json_atomic(path, {"k": 1}, indent=2)
    assert json.loads(path.read_text()) == {"k": 1}
    assert list(tmp_path.glob("*.tmp")) == []
