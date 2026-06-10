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
    max_seq_len: int = 1024,
    schema: str = drv.SCHEMA_VERSION,
    pos0: dict | None = None,
    probes: list[str] | None = None,
) -> dict:
    return {
        "schema_version": schema,
        "phase": "scoring",
        "pair": {"a": a, "b": b},
        "is_selfpair": a == b,
        "n_probes": n_probes,
        "r_samples": r_samples,
        "probes_sha256": drv._probes_sha256(probes if probes is not None else FAKE_PROBES_50),
        "js_rb_bits": 0.1,
        "mc_se_js_bits": 0.01,
        "pos0_v1_check": pos0,
        "metadata": {
            "model": model,
            "stub": stub,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "max_seq_len": max_seq_len,
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
    max_seq_len: int = 1024,
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
            "max_seq_len": max_seq_len,
            "base_model": drv.BASE_MODEL,
        },
        "truncation_rate": 0.0,
    }


def _matrix_payload(
    *,
    n_probes: int = 50,
    r_samples: int = 8,
    model: str = drv.BASE_MODEL,
    stub: bool = False,
    seed: int = 42,
    max_new_tokens: int = 256,
    max_seq_len: int = 1024,
    schema: str = drv.SCHEMA_VERSION,
    phase: str = "matrix",
    probes: list[str] | None = None,
) -> dict:
    """Minimal predictors_jsrb.json / analysis_jsrb.json compatibility shell
    (only the fields the round-3 full-tuple validation reads)."""
    return {
        "schema_version": schema,
        "phase": phase,
        "n_probes": n_probes,
        "r_samples": r_samples,
        "probes_sha256": drv._probes_sha256(probes if probes is not None else FAKE_PROBES_50),
        "metadata": {
            "model": model,
            "stub": stub,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "max_seq_len": max_seq_len,
        },
    }


def _analysis_payload(**kwargs) -> dict:
    return _matrix_payload(phase="analysis", **kwargs)


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
    """issue540_v1/v2 artifacts (round-1/2 smoke vintage) are structurally
    incompatible and recomputed."""
    args = _args(tmp_path)
    for old in ("issue540_v1", "issue540_v2"):
        path = _write(
            tmp_path / "per_pair" / "pair_A1__A2.json",
            _pair_payload("A1", "A2", schema=old),
        )
        assert drv._load_compatible_pair(args, path, "A1", "A2", strict=False) is None


# ── Probe-list identity (round-3 concern pair-probe-identity-validation) ───


def test_pair_probe_text_mutation_recomputed_not_skipped(tmp_path):
    """The reconciler-verified propagation hole: a probe-TEXT mutation at
    constant count (n_probes unchanged) recomputes samples but, without the
    probes_sha256 check, stale PAIR artifacts would still resume-skip."""
    args = _args(tmp_path)
    a, b = GATED_PAIR
    path = _write(
        tmp_path / "per_pair" / f"pair_{a}__{b}.json",
        _pair_payload(a, b, pos0={"abs_diff": 0.0}),  # hash over FAKE_PROBES_50
    )
    # Sanity: under the original probe list the artifact resume-skips.
    assert drv._load_compatible_pair(args, path, a, b, strict=False) is not None
    # Mutate ONE probe's text, same count → recompute, never skip.
    drv._PROBES_CACHE[50] = ["MUTATED question 0?", *FAKE_PROBES_50[1:]]
    assert drv._load_compatible_pair(args, path, a, b, strict=False) is None
    with pytest.raises(RuntimeError, match="probes_sha256"):
        drv._load_compatible_pair(args, path, a, b, strict=True)


def test_pair_max_seq_len_mismatch_refused(tmp_path):
    """--max-seq-len caps vLLM completions (max_model_len − prompt_len); a
    pair scored over differently-capped samples must not resume-skip."""
    args = _args(tmp_path)
    path = _write(
        tmp_path / "per_pair" / "pair_A1__A2.json",
        _pair_payload("A1", "A2", max_seq_len=512),
    )
    assert drv._load_compatible_pair(args, path, "A1", "A2", strict=False) is None


# ── Matrix artifact (round-3 concern matrix-artifact-param-validation) ─────


def test_matrix_loader_accepts_matching_tuple(tmp_path):
    args = _args(tmp_path)
    _write(tmp_path / "predictors_jsrb.json", _matrix_payload())
    assert drv._load_matrix(args)["n_probes"] == 50


def test_matrix_loader_rejects_mismatched_seed_and_model(tmp_path):
    """Same-shape (50×8) stale matrix under a different seed + scoring model
    must hard-fail Phase A/F, naming the mismatched fields."""
    args = _args(tmp_path)
    _write(tmp_path / "predictors_jsrb.json", _matrix_payload(seed=7, model="/tmp/tiny540"))
    with pytest.raises(RuntimeError) as exc:
        drv._load_matrix(args)
    assert "metadata.seed" in str(exc.value) and "metadata.model" in str(exc.value)


def test_matrix_loader_rejects_stub_and_max_new_tokens(tmp_path):
    args = _args(tmp_path)
    _write(tmp_path / "predictors_jsrb.json", _matrix_payload(stub=True, max_new_tokens=512))
    with pytest.raises(RuntimeError) as exc:
        drv._load_matrix(args)
    assert "metadata.stub" in str(exc.value) and "metadata.max_new_tokens" in str(exc.value)


def test_matrix_loader_rejects_probe_text_mutation(tmp_path):
    """Same shape, same params, different probe TEXTS → refused by hash."""
    args = _args(tmp_path)
    _write(
        tmp_path / "predictors_jsrb.json",
        _matrix_payload(probes=["MUTATED question 0?", *FAKE_PROBES_50[1:]]),
    )
    with pytest.raises(RuntimeError, match="probes_sha256"):
        drv._load_matrix(args)


# ── Analysis artifact (round-3 concern analysis-artifact-param-validation) ─


def test_analysis_loader_accepts_matching_tuple(tmp_path):
    args = _args(tmp_path)
    _write(tmp_path / "analysis_jsrb.json", _analysis_payload())
    assert drv._load_analysis(args)["phase"] == "analysis"


def test_analysis_loader_rejects_mismatched_tuple(tmp_path):
    """Standalone Phase F over a same-shape stale analysis (different seed +
    max_new_tokens) must hard-fail, naming the fields — figures can never mix
    a current matrix with a stale leaderboard/hierarchy."""
    args = _args(tmp_path)
    _write(tmp_path / "analysis_jsrb.json", _analysis_payload(seed=7, max_new_tokens=512))
    with pytest.raises(RuntimeError) as exc:
        drv._load_analysis(args)
    assert "metadata.seed" in str(exc.value) and "metadata.max_new_tokens" in str(exc.value)


def test_analysis_loader_rejects_probe_text_mutation_and_stub(tmp_path):
    args = _args(tmp_path)
    _write(
        tmp_path / "analysis_jsrb.json",
        _analysis_payload(stub=True, probes=["MUTATED question 0?", *FAKE_PROBES_50[1:]]),
    )
    with pytest.raises(RuntimeError) as exc:
        drv._load_analysis(args)
    assert "probes_sha256" in str(exc.value) and "metadata.stub" in str(exc.value)


def test_analysis_loader_rejects_matrix_phase_payload(tmp_path):
    """A matrix payload at the analysis path (cross-artifact mixup) is refused."""
    args = _args(tmp_path)
    _write(tmp_path / "analysis_jsrb.json", _matrix_payload())
    with pytest.raises(RuntimeError, match="phase"):
        drv._load_analysis(args)


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


def test_nondefault_max_seq_len_routes_to_smoke_dir():
    """--max-seq-len is part of the production card (round-3 minor): a
    non-default value can shorten generations, so it must not default into
    the production dir."""
    args = drv._build_parser().parse_args(["--max-seq-len", "2048"])
    drv._resolve_dirs(args)
    assert args.out_dir == drv.SMOKE_OUT_DIR
    assert args.figures_dir == drv.SMOKE_FIGURES_DIR


def test_custom_out_dir_routes_figures_alongside(tmp_path):
    """Round-3 minor fix: a production-shaped run with an explicit CUSTOM
    --out-dir must NOT write figures into the canonical figures/issue_540 —
    they co-locate at <out_dir>/figures."""
    args = drv._build_parser().parse_args(["--out-dir", str(tmp_path)])
    drv._resolve_dirs(args)
    assert args.figures_dir == tmp_path / "figures"


def test_explicit_production_out_dir_keeps_canonical_figures_dir():
    """The plan §10 launch command passes --out-dir eval_results/issue_540
    explicitly; figures stay at the canonical figures/issue_540."""
    args = drv._build_parser().parse_args(["--out-dir", str(drv.DEFAULT_OUT_DIR)])
    drv._resolve_dirs(args)
    assert args.figures_dir == drv.DEFAULT_FIGURES_DIR


def test_explicit_figures_dir_always_respected(tmp_path):
    args = drv._build_parser().parse_args(
        ["--out-dir", str(tmp_path), "--figures-dir", str(tmp_path / "figs")]
    )
    drv._resolve_dirs(args)
    assert args.figures_dir == tmp_path / "figs"


# ── Atomic JSON writes ──────────────────────────────────────────────────────


def test_write_json_atomic_leaves_no_tmp(tmp_path):
    path = tmp_path / "out.json"
    drv._write_json_atomic(path, {"k": 1}, indent=2)
    assert json.loads(path.read_text()) == {"k": 1}
    assert list(tmp_path.glob("*.tmp")) == []
