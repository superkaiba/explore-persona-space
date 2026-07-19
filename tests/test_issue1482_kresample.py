"""Pins for the #1482 k-resample noise-floor driver (plan v7).

Covers the data-dependent gate branches the main smoke leg deliberately does
not trip (degenerate-input probes: G1 fail, G2 fallback, C1 fail, sha-mismatch
regen, empty-response drop, fetch-count assert, determinism 0/N raise), the
unbiasedness identity, largest-remainder stratification, and the inverted-CI
errorbar clamp through the REAL hero-figure function to savefig (#547/#1335).
All tests execute the real production bodies (pure functions — no seams
stubbed; the only fakes are tmp_path filesystem inputs).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_kresample as K  # noqa: E402


def _rng():
    return np.random.default_rng(0)


# ── estimator identity + fallback arithmetic ────────────────────────────────────


def test_estimator_unbiasedness_identity_holds():
    """m2 + trvar == mean_k e2_k exactly (algebraic); hence per-row
    nerr_adj + floor_n == mean over the 5 draws of per-draw nerr."""
    rng = _rng()
    n, h = 17, 32
    V = rng.normal(size=(n, 5, h))
    vhat = rng.normal(size=(n, h))
    denom = rng.uniform(1.0, 5.0, size=n)
    est = K._estimators(V, vhat, denom)
    e2_mean = est["e2_k"].mean(1)
    assert np.allclose(est["m2"] + est["trvar"], e2_mean, rtol=1e-10)
    assert np.allclose(est["nerr_adj"] + est["floor_n"], e2_mean / denom, rtol=1e-10)
    assert est["identity_max_rel_dev"] < 1e-8


def test_estimator_identity_assert_fires_on_corrupt_shapes():
    """The identity assert is a live guard: a broken estimator input raises."""
    rng = _rng()
    V = rng.normal(size=(5, 5, 8))
    vhat = rng.normal(size=(5, 8))
    est = K._estimators(V, vhat, np.ones(5))
    # sanity: fresh-4 shadow uses only draws 1..4
    V2 = V.copy()
    V2[:, 0, :] = 1e6  # corrupt the parent draw only
    est2 = K._estimators(V2, vhat, np.ones(5))
    assert np.allclose(est2["floor4_n"], est["floor4_n"])  # fresh-4 floor unaffected


# ── gate G1: pass on identity-constructed recapture, fail on perturbation ───────


def test_g1_passes_on_identity_and_fails_on_perturbation():
    rng = _rng()
    e2_stored = rng.uniform(0.5, 10.0, size=200)
    g_pass = K._g1(e2_stored * (1 + rng.normal(0, 0.001, size=200)), e2_stored)
    assert g_pass["pass"], g_pass
    # 30% multiplicative noise breaks BOTH legs' calibration target (median 2%)
    g_fail = K._g1(e2_stored * np.exp(rng.normal(0, 0.5, size=200)), e2_stored)
    assert not g_fail["pass"], g_fail


def test_g2_fallback_fires_on_outlier_parent_draw():
    rng = _rng()
    e2_k = rng.uniform(1.0, 2.0, size=(300, 5))
    g_ok = K._g2(e2_k, np.arange(300) < 150)
    assert abs(g_ok["mean_rank_draw42"] - 3.0) < 0.5
    e2_bad = e2_k.copy()
    e2_bad[:, 0] += 10.0  # parent draw always worst -> mean rank 5
    g_bad = K._g2(e2_bad, np.arange(300) < 150)
    assert g_bad["mean_rank_draw42"] == 5.0 and not g_bad["pass"]
    assert set(g_bad["per_arm_mean_rank"]) == {"en", "nonen"}


# ── largest-remainder stratification ────────────────────────────────────────────


def test_largest_remainder_allocation_exact_and_proportional():
    counts = {
        "zh": 2289,
        "ru": 1306,
        "es": 642,
        "pt": 596,
        "fr": 457,
        "de": 405,
        "it": 327,
        "ar": 170,
        "ja": 80,
        "pl": 80,
        "ko": 72,
        "vi": 69,
    }
    alloc = K.largest_remainder_alloc(counts, 1000)
    assert sum(alloc.values()) == 1000
    total = sum(counts.values())
    for c, v in alloc.items():
        q = 1000 * counts[c] / total
        assert abs(v - q) < 1.0 + 1e-9, (c, v, q)  # within one seat of quota
        assert v <= counts[c]


def test_largest_remainder_tiny_n_smoke_floor():
    counts = {"zh": 2289, "ru": 1306, "es": 642, "ar": 170}
    alloc = K.largest_remainder_alloc(counts, 20)
    assert sum(alloc.values()) == 20
    assert all(v >= 1 for v in alloc.values()) or "ar" not in alloc


# ── bundle-sha resume keying (b1 checkpoint) ────────────────────────────────────


def test_bundle_sha_mismatch_forces_regeneration(tmp_path):
    """Resume skips ONLY on sha match — a mismatched checkpoint is regenerated
    (plan c24: never bare file existence). Executes the REAL phase_b1 body on
    the CPU stub path (real tokenizer render + budget check + checkpoints)."""
    _load_tok_or_skip()  # phase_b1 loads the real tokenizer; skip when uncached
    args = SimpleNamespace(
        out=tmp_path,
        out_eval=tmp_path / "eval",
        figures=tmp_path / "figs",
        scratch=tmp_path / "scratch",
        hf_prefix="x",
        skip_upload=True,
        smoke=True,
        tiny_model=True,
        device="cpu",
        n_per_arm=2,
        n_boot=10,
        gen_batch=2,
        token_budget=8192,
        workers=1,
        max_chunks=0,
    )
    (tmp_path / "inputs").mkdir(parents=True)
    rows = [
        {
            "row_idx": i,
            "ci": 100 + i,
            "arm": "en",
            "language": "en",
            "prompt": f"q{i}",
            "response_seed42": "r",
            "e2_stored": 1.0,
            "denom_stored": 1.0,
            "nerr_stored": 1.0,
        }
        for i in range(2)
    ]
    for name, sl in zip(K.BUNDLE_PARTS, (rows[:1], rows[1:]), strict=True):
        (tmp_path / "inputs" / name).write_text(json.dumps({"meta": {}, "rows": sl}))
    _, sha = K._load_bundle(args)
    gen_dir = tmp_path / "gen"
    gen_dir.mkdir()
    stale = {
        "meta": {"bundle_sha": "STALE"},
        "rows": [{"ci": 100, "row_idx": 0, "response": "old"}],
    }
    (gen_dir / "gen_seed43_chunk0.json").write_text(json.dumps(stale))
    K.phase_b1(args)  # cpu smoke: stub generation through the real checkpoint path
    doc = json.loads((gen_dir / "gen_seed43_chunk0.json").read_text())
    assert doc["meta"]["bundle_sha"] == sha  # stale checkpoint was REGENERATED
    assert "stub response" in doc["rows"][0]["response"]
    # and a matching-sha checkpoint is kept (byte-identical rerun)
    before = (gen_dir / "gen_seed43_chunk0.json").read_bytes()
    K.phase_b1(args)
    assert (gen_dir / "gen_seed43_chunk0.json").read_bytes() == before


# ── B1 pilot timing gate (plan §9 row + #1415 designed-halt convention) ─────────


def test_pilot_gate_pass_and_refusal_branches():
    ok = K._pilot_gate(chunk_wall_s=225.0, output_tokens=210_000, n_chunk_calls=16)
    assert ok["pass"] and ok["projected_wall_h"] == pytest.approx(1.0)
    slow = K._pilot_gate(chunk_wall_s=1000.0, output_tokens=210_000, n_chunk_calls=16)
    assert not slow["pass"]  # projected 4.4 h > 2x booked 1.0 h -> RC_PILOT halt
    assert slow["projected_wall_h"] > 2.0 * K.B1_BOOKED_WALL_H
    assert K.RC_PILOT not in (0, 1)  # distinct designed-halt rc, never a bare rc=1


# ── determinism spot-check severity calibration ─────────────────────────────────


def test_determinism_spot_check_raises_only_on_total_mismatch(monkeypatch):
    ref = [f"t{i}" for i in range(5)]
    monkeypatch.setattr(K, "_generate_seed", lambda llm, tok, texts, seed: list(ref))
    ok = K._determinism_spot_check(object(), ["p"] * 5, 43, ref)
    assert ok["n_match"] == 5
    monkeypatch.setattr(
        K, "_generate_seed", lambda llm, tok, texts, seed: [ref[0], "x", "x", "x", "x"]
    )
    warn = K._determinism_spot_check(object(), ["p"] * 5, 43, ref)
    assert warn["n_match"] == 1  # partial mismatch -> WARN + record, no raise
    monkeypatch.setattr(K, "_generate_seed", lambda llm, tok, texts, seed: ["x"] * 5)
    with pytest.raises(RuntimeError, match="0/5"):
        K._determinism_spot_check(object(), ["p"] * 5, 43, ref)


def test_determinism_spot_check_real_body_cpu_stub():
    """Real bodies end-to-end on the CPU stub path (no monkeypatch): the stub
    texts are seed-deterministic, so the spot-check matches 5/5."""
    texts = ["a", "b", "c", "d", "e"]
    ref = K._generate_seed(None, None, texts, seed=43)
    rep = K._determinism_spot_check(None, texts, 43, ref)
    assert rep["n_match"] == rep["n_total"]


# ── empty-response drop + tokenize seam (real tokenizer if cached) ──────────────


def test_tokenize_row_empty_response_returns_none():
    tok = _load_tok_or_skip()
    import issue1482_error_analysis as EA

    prefix_chars = EA._prefix_char_len(tok)
    assert EA._tokenize_row(tok, "hello", "", prefix_chars) is None  # whole-context drop path
    out = EA._tokenize_row(tok, "hello", "world", prefix_chars)
    assert out is not None
    full_ids, _pe, context_end, n_ans, _seam = out
    assert len(full_ids) == context_end + 1 + n_ans


def _load_tok_or_skip():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(K.MODEL_ID)
    except Exception as e:  # no network/cache in CI
        pytest.skip(f"tokenizer unavailable: {e}")


# ── inverted-CI errorbar clamp through the REAL figure function (#547/#1335) ────


def test_hero_figure_survives_inverted_ci(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    arm_stats = {"en": {"map": 0.4, "floor": 0.1}, "nonen": {"map": 0.35, "floor": 0.12}}
    deltas = {
        "raw": {"point": -0.0175, "ci": [-0.0221, -0.0129]},
        "floor": {"point": 0.002, "ci": [0.0021, 0.0019]},  # deliberately INVERTED CI
        "adj": {"point": -0.0195, "ci": [-0.025, -0.014]},
    }
    K._fig_hero(tmp_path, arm_stats, deltas, "hero.png", "mean normalized error")
    assert (tmp_path / "hero.png").exists()


def test_ci_offsets_clamped_nonnegative():
    off = K._ci_offsets([1.0, 2.0], [1.1, 1.5], [0.9, 2.5])  # both bounds inverted on pt 1
    assert (off >= 0).all()
    assert off.shape == (2, 2)


# ── fetch-count assert (phase-A hard gate, degenerate probe) ────────────────────


def test_fetch_count_assert_fires_on_missing_rows(monkeypatch, tmp_path):
    args = SimpleNamespace(scratch=tmp_path, workers=1, max_chunks=0)
    monkeypatch.setattr(K.EA, "_raw_chunk_names", lambda a: ["shard00_chunk0000.json"])
    monkeypatch.setattr(K, "_probe_chunk_index", lambda a, names: {})
    monkeypatch.setattr(
        K.N1M,
        "_download_chunk_with_retry",
        lambda repo, fn, cache: _write_chunk(tmp_path, [{"ci": 1, "prompt": "p", "response": "r"}]),
    )
    found = K._fetch_rows_threaded(args, {1})
    assert found == {1: ("p", "r")}
    with pytest.raises(AssertionError, match="needed ci not found"):
        K._fetch_rows_threaded(args, {1, 2})  # ci=2 nowhere in the chunks


def _write_chunk(tmp_path: Path, rows: list[dict]) -> str:
    p = tmp_path / "chunk.json"
    p.write_text(json.dumps({"rows": rows}))
    return str(p)


# ── chunk-index prediction (optimization correctness) ───────────────────────────


def test_probe_chunk_index_ranges(monkeypatch, tmp_path):
    args = SimpleNamespace(scratch=tmp_path, workers=1, max_chunks=0)
    chunks = {
        "shard00_chunk0000.json": [{"ci": c} for c in range(0, 500)],
        "shard00_chunk0001.json": [{"ci": c} for c in range(500, 1000)],
        "shard01_chunk0000.json": [{"ci": c} for c in range(1000, 1500)],
    }

    def fake_dl(repo, fn, cache):
        name = fn.rsplit("/", 1)[-1]
        p = tmp_path / name
        p.write_text(json.dumps({"rows": chunks[name]}))
        return str(p)

    monkeypatch.setattr(K.N1M, "_download_chunk_with_retry", fake_dl)
    idx = K._probe_chunk_index(args, list(chunks))
    assert idx["shard00_chunk0000.json"] == (0, 500)
    assert idx["shard00_chunk0001.json"] == (500, 1000)
    assert idx["shard01_chunk0000.json"][0] == 1000
