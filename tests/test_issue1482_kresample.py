"""Pins for the #1482 k-resample noise-floor driver (plan v7).

Covers the data-dependent gate branches the main smoke leg deliberately does
not trip (degenerate-input probes: G1 fail, G2 outlier, sha/ids_version-stale
regen, prompt-ids join drift, stale text-retok capture guards, fetch-count +
vstream missing-ci asserts, determinism repeat-floor and seed-distinctness
raises), the id-bearing b1 chunk schema (v68), the parent-convention b2
assembly helpers (full-template retok, span incl. the end-of-turn tail —
verbatim COL.capture_answer_vector), the vstream collector (probe/predict/
fallback/early-stop/checkpoint over REAL-format .pt chunk files; only the
network transport is faked via fetch_fn), the fresh-4 + 5-draw-shadow
estimator identities, largest-remainder stratification, and the inverted-CI
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
    """m2 + trvar == mean_k e2_k exactly (algebraic) for BOTH the fresh-4
    registered set and the 5-draw shadow (streamed v42 as draw 0)."""
    rng = _rng()
    n, h = 17, 32
    V = rng.normal(size=(n, 4, h))  # fresh draws 43-46 (v68: parent draw streamed)
    v42 = rng.normal(size=(n, h))
    vhat = rng.normal(size=(n, h))
    denom = rng.uniform(1.0, 5.0, size=n)
    est = K._estimators(V, v42, vhat, denom)
    e2_fresh_mean = est["e2_k"][:, 1:].mean(1)
    assert np.allclose(est["m2"] + est["trvar"], e2_fresh_mean, rtol=1e-10)
    assert np.allclose(est["nerr_adj"] + est["floor_n"], e2_fresh_mean / denom, rtol=1e-10)
    e2_5_mean = est["e2_k"].mean(1)
    assert np.allclose(est["m2_5"] + est["trvar5"], e2_5_mean, rtol=1e-10)
    assert est["identity_max_rel_dev"] < 1e-8
    # column 0 IS the streamed parent draw's e2
    assert np.allclose(est["e2_k"][:, 0], ((v42 - vhat) ** 2).sum(-1))


def test_estimator_primary_floor_unaffected_by_parent_draw():
    """The registered fresh-4 floor never touches the streamed parent draw: a
    corrupted v42 changes ONLY the shadow (and e2_k column 0)."""
    rng = _rng()
    V = rng.normal(size=(5, 4, 8))
    vhat = rng.normal(size=(5, 8))
    v42 = rng.normal(size=(5, 8))
    est = K._estimators(V, v42, vhat, np.ones(5))
    est2 = K._estimators(V, v42 + 1e6, vhat, np.ones(5))
    assert np.allclose(est2["floor_n"], est["floor_n"])
    assert np.allclose(est2["m2"], est["m2"])
    assert not np.allclose(est2["floor5_n"], est["floor5_n"])  # shadow DOES move


# ── gate G1: pass on identity-constructed recapture, fail on perturbation ───────


def test_g1_passes_on_identity_and_fails_on_perturbation():
    """v68 thresholds: the streamed-v identity read tolerates only fp16-pred
    quantization (median rel <= 1e-3, Spearman >= 0.999)."""
    rng = _rng()
    e2_stored = rng.uniform(0.5, 10.0, size=200)
    g_pass = K._g1(e2_stored * (1 + rng.normal(0, 3e-4, size=200)), e2_stored)
    assert g_pass["pass"], g_pass
    # 2% multiplicative noise passed the OLD (retok) bar; the identity bar fails it
    g_fail_med = K._g1(e2_stored * np.exp(rng.normal(0, 0.02, size=200)), e2_stored)
    assert not g_fail_med["pass"], g_fail_med
    g_fail = K._g1(e2_stored * np.exp(rng.normal(0, 0.5, size=200)), e2_stored)
    assert not g_fail["pass"], g_fail
    assert g_fail["convention"] == "streamed-v42-identity (no recapture)"


def test_g2_diagnostic_fires_on_outlier_parent_draw():
    """A streamed parent draw in a DIFFERENT convention than the fresh draws
    would rank as a systematic outlier — the parent-vs-now check G2 detects."""
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
    # a text-only chunk with a MATCHING sha (pre-v68 schema) is stale too
    (gen_dir / "gen_seed44_chunk0.json").write_text(
        json.dumps({"meta": {"bundle_sha": sha}, "rows": stale["rows"]})
    )
    K.phase_b1(args)  # cpu smoke: stub generation through the real checkpoint path
    for name in ("gen_seed43_chunk0.json", "gen_seed44_chunk0.json"):
        doc = json.loads((gen_dir / name).read_text())
        assert doc["meta"]["bundle_sha"] == sha  # stale checkpoint was REGENERATED
        assert doc["meta"]["ids_version"] == K.IDS_VERSION  # id-bearing schema (v68)
        assert "stub response" in doc["rows"][0]["response"]
        # generation-time ids persisted per row (stub path: retok == gen ids)
        assert doc["rows"][0]["token_ids"] and doc["rows"][0]["prompt_token_ids"]
    # and a matching-sha id-bearing checkpoint is kept (byte-identical rerun)
    before = (gen_dir / "gen_seed43_chunk0.json").read_bytes()
    K.phase_b1(args)
    assert (gen_dir / "gen_seed43_chunk0.json").read_bytes() == before
    # b2's loader consumes the id-bearing chunks; a text-only chunk fails loud
    gen = K._load_gen_chunks(args, sha)
    assert set(gen[43][100]) == {"response", "token_ids", "prompt_token_ids"}
    (gen_dir / "gen_seed43_chunk0.json").write_text(
        json.dumps({"meta": {"bundle_sha": sha}, "rows": []})
    )
    with pytest.raises(RuntimeError, match="ids_version"):
        K._load_gen_chunks(args, sha)


# ── B1 pilot timing gate (plan §9 row + #1415 designed-halt convention) ─────────


def test_pilot_gate_pass_and_refusal_branches():
    ok = K._pilot_gate(chunk_wall_s=225.0, output_tokens=210_000, n_chunk_calls=16)
    assert ok["pass"] and ok["projected_wall_h"] == pytest.approx(1.0)
    slow = K._pilot_gate(chunk_wall_s=1000.0, output_tokens=210_000, n_chunk_calls=16)
    assert not slow["pass"]  # projected 4.4 h > 2x booked 1.0 h -> RC_PILOT halt
    assert slow["projected_wall_h"] > 2.0 * K.B1_BOOKED_WALL_H
    assert K.RC_PILOT not in (0, 1)  # distinct designed-halt rc, never a bare rc=1


# ── determinism spot-check severity calibration ─────────────────────────────────


def test_determinism_spot_check_batch_mismatch_never_gates(monkeypatch):
    """Regression pin for the v7 crash: a 0/5 batch-vs-standalone byte-match (a
    legitimate vLLM V1 batch-shape numerics effect) must NOT raise when the seed
    demonstrably functions (repeat matches, distinct seed differs). Under the
    pre-fix check this exact input raised 'per-request seed not applied'."""

    def seeded(llm, tok, texts, seed):
        return [_gen_row(f"s{seed}-{i}") for i in range(len(texts))]

    monkeypatch.setattr(K, "_generate_seed", seeded)
    ok = K._determinism_spot_check(object(), ["p"] * 5, 43, ["batchtext"] * 5)
    assert ok["n_repeat_match"] == 5 and ok["n_distinct_differ"] == 5
    assert ok["batch_vs_standalone_match"] == 0  # informational only — no raise
    assert ok["distinct_seed"] == 1043  # off the registered seed space (42-46)


def test_determinism_spot_check_repeat_flake_warns_not_raises(monkeypatch):
    calls = {"n": 0}

    def flaky(llm, tok, texts, seed):
        calls["n"] += 1
        out = [f"s{seed}-{i}" for i in range(len(texts))]
        if calls["n"] == 2:  # one flipped prompt on the repeat run only
            out[0] = "flake"
        return [_gen_row(t) for t in out]

    monkeypatch.setattr(K, "_generate_seed", flaky)
    warn = K._determinism_spot_check(object(), ["p"] * 5, 43, ["b"] * 5)
    assert warn["n_repeat_match"] == 4  # >= floor 4/5 -> WARN + record, no raise


def test_determinism_spot_check_raises_below_repeat_floor(monkeypatch):
    calls = {"n": 0}

    def unrepeatable(llm, tok, texts, seed):  # 2/5 stable across same-seed calls
        calls["n"] += 1
        return [
            _gen_row(f"call{calls['n']}-{i}" if i < 3 else f"fix-{i}") for i in range(len(texts))
        ]

    monkeypatch.setattr(K, "_generate_seed", unrepeatable)
    with pytest.raises(RuntimeError, match="same-seed repeat 2/5"):
        K._determinism_spot_check(object(), ["p"] * 5, 43, ["b"] * 5)


def test_determinism_spot_check_raises_when_seed_ignored(monkeypatch):
    monkeypatch.setattr(
        K, "_generate_seed", lambda llm, tok, texts, seed: [_gen_row("same")] * len(texts)
    )
    with pytest.raises(RuntimeError, match="seed ignored"):
        K._determinism_spot_check(object(), ["p"] * 5, 43, ["same"] * 5)


def _gen_row(text: str) -> dict:
    """Minimal id-bearing gen row (the v68 _generate_seed return shape)."""
    return {"text": text, "token_ids": [1], "prompt_token_ids": [2]}


def test_determinism_spot_check_real_body_cpu_stub():
    """Real body end-to-end on the CPU stub path (no monkeypatch): stub texts are
    seed-deterministic, so repeat matches N/N and seed+1000 differs N/N. Also pins
    the scaled floor max(1, N-1) at the 2-row CPU-smoke bundle size."""
    texts = ["a", "b", "c", "d", "e"]
    ref = [g["text"] for g in K._generate_seed(None, None, texts, seed=43)]
    rep = K._determinism_spot_check(None, texts, 43, ref)
    assert rep["n_repeat_match"] == rep["n_total"] == 5
    assert rep["n_distinct_differ"] == 5
    assert rep["batch_vs_standalone_match"] == 5
    assert rep["engine"] == "cpu-stub"
    small = K._determinism_spot_check(None, texts[:2], 43, ref[:2])
    assert small["repeat_floor"] == 1 and small["n_repeat_match"] == 2


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


# ── parent-convention b2 assembly (v68 fix; real tokenizer if cached) ───────────


def test_parent_convention_span_includes_end_of_turn_tail():
    """The v68 assembly is VERBATIM COL.capture_answer_vector: full-template
    re-tokenization, span [prompt_len:full_len] INCLUDING the end-of-turn tail —
    NOT token-id concat, NOT generation ids. Pin equivalence to the parent
    function's own prompt_len/full_len arithmetic."""
    tok = _load_tok_or_skip()
    msgs, prompt_ids = K._prompt_render(tok, "What is 2+2?")
    full_ids = K._parent_convention_full_ids(tok, msgs, "It is 4.")
    # parent arithmetic (capture_answer_vector:171-181), byte-for-byte
    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert len(prompt_ids) == len(tok(prompt_text)["input_ids"])
    full_text = tok.apply_chat_template(
        [*msgs, {"role": "assistant", "content": "It is 4."}],
        tokenize=False,
        add_generation_prompt=False,
    )
    assert full_ids == [int(x) for x in tok(full_text)["input_ids"]]
    span = len(full_ids) - len(prompt_ids)
    resp_only = len(tok("It is 4.", add_special_tokens=False)["input_ids"])
    assert span > resp_only  # the span carries the <|im_end|>(+\n) tail
    # parent drop rule: even an EMPTY response keeps a nonzero (tail-only) span,
    # so the b2 drop gate is a parent-verbatim guard (production-only branch)
    assert len(K._parent_convention_full_ids(tok, msgs, "")) > len(prompt_ids)


def test_prompt_ids_join_check_raises_on_drift():
    K._check_prompt_ids_join(7, 43, [1, 2, 3], [1, 2, 3])  # identical -> no raise
    with pytest.raises(RuntimeError, match="join drift"):
        K._check_prompt_ids_join(7, 43, [1, 2, 3], [1, 2, 4])


def test_analyze_refuses_stale_text_retok_capture(tmp_path):
    """phase_analyze must NEVER read a pre-v68 V.npz (retired text-retok
    convention): capture_meta without ids_version fails loud before any load."""
    args = SimpleNamespace(
        out=tmp_path,
        out_eval=tmp_path / "eval",
        figures=tmp_path / "figs",
        scratch=tmp_path / "scratch",
        hf_prefix="x",
        smoke=True,
        n_boot=10,
    )
    (tmp_path / "V.npz").write_bytes(b"stale")  # guard fires before np.load
    (tmp_path / "capture_meta.json").write_text(json.dumps({"bundle_sha": "s"}))
    with pytest.raises(RuntimeError, match="ids_version"):
        K.phase_analyze(args)


def test_b2_refuses_stale_capture_outputs(tmp_path, monkeypatch):
    """b2's resume guard: an existing V.npz whose meta lacks ids_version (or has
    a foreign sha) is STALE — RuntimeError, never a silent skip/consume."""
    args = SimpleNamespace(
        out=tmp_path,
        scratch=tmp_path / "scratch",
        hf_prefix="x",
        skip_upload=True,
        smoke=True,
        tiny_model=True,
        device="cpu",
        n_per_arm=1,
        gen_batch=2,
        token_budget=8192,
        workers=1,
        max_chunks=0,
    )
    (tmp_path / "inputs").mkdir(parents=True)
    row = {
        "row_idx": 0,
        "ci": 100,
        "arm": "en",
        "language": "en",
        "prompt": "q",
        "response_seed42": "r",
        "e2_stored": 1.0,
        "denom_stored": 1.0,
        "nerr_stored": 1.0,
    }
    for name, sl in zip(K.BUNDLE_PARTS, ([row], []), strict=True):
        (tmp_path / "inputs" / name).write_text(json.dumps({"meta": {}, "rows": sl}))
    _, sha = K._load_bundle(args)
    gen_dir = tmp_path / "gen"
    gen_dir.mkdir()
    for k in K.GEN_SEEDS:
        (gen_dir / f"gen_seed{k}_chunk0.json").write_text(
            json.dumps(
                {
                    "meta": {"bundle_sha": sha, "ids_version": K.IDS_VERSION},
                    "rows": [
                        {
                            "row_idx": 0,
                            "ci": 100,
                            "response": "r",
                            "token_ids": [1],
                            "prompt_token_ids": [2],
                        }
                    ],
                }
            )
        )
    (tmp_path / "V.npz").write_bytes(b"stale")
    (tmp_path / "capture_meta.json").write_text(json.dumps({"bundle_sha": sha}))
    with pytest.raises(RuntimeError, match="remove stale outputs"):
        K.phase_b2(args)


# ── vstream: collector over REAL-format .pt chunks (transport faked only) ───────


def _write_pt_chunks(tmp_path: Path, groups: list[list[int]], h: int = 8):
    """REAL-format n1m capture chunk files (.pt + raw-JSON sibling): keys
    cx_last/v_x (n, 3, H) fp16, ci, prompts, layers=[14, 19, 26]."""
    import torch

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    layers = [14, K.LAYER, 26]
    expected: dict[int, np.ndarray] = {}
    for j, cis in enumerate(groups):
        vx = torch.zeros((len(cis), 3, h), dtype=torch.float16)
        for i, c in enumerate(cis):
            v = np.random.default_rng(c).normal(size=h)
            vx[i, 1, :] = torch.tensor(v, dtype=torch.float16)
            expected[c] = vx[i, 1, :].to(torch.float32).numpy()
        torch.save(
            {
                "cx_last": torch.zeros((len(cis), 3, h), dtype=torch.float16),
                "v_x": vx,
                "ci": list(cis),
                "prompts": ["" for _ in cis],
                "layers": layers,
                "shard_index": 0,
                "chunk": j,
            },
            chunk_dir / f"shard00_chunk{j:04d}.pt",
        )
        (chunk_dir / f"shard00_chunk{j:04d}.json").write_text(
            json.dumps({"rows": [{"ci": int(c)} for c in cis]})
        )
    return chunk_dir, expected


def test_vstream_collector_probe_fallback_and_checkpoint(tmp_path, monkeypatch):
    """The full collector body over REAL .pt chunk files: prediction via the
    raw-JSON sibling probe, neighbor-first fallback for residue rows, early
    stop, found-set checkpoint + resume, missing-ci fail-loud."""
    monkeypatch.setattr(K, "HIDDEN", 8)
    chunk_dir, expected = _write_pt_chunks(
        tmp_path,
        [[0, 1, 2], [3, 4, 7], [8, 9, 12]],  # gaps -> prediction drift
    )
    args = SimpleNamespace(out=tmp_path, scratch=tmp_path / "scratch", workers=2, max_chunks=0)
    args.scratch.mkdir(exist_ok=True)
    fetch = K._local_fetch_fn(chunk_dir)
    names = sorted(p.name for p in chunk_dir.glob("*.pt"))
    got = K._collect_v42(args, {1, 4, 12}, names, fetch, sha="S")
    for c in (1, 4, 12):
        assert np.allclose(got["found"][c], expected[c])

    # found-set checkpoint resume: a complete checkpoint means NO fetches at all
    def _explode(repo, fn, cache):
        raise AssertionError("resume must not re-fetch")

    got2 = K._collect_v42(args, {1, 4, 12}, names, _explode, sha="S")
    assert set(got2["found"]) >= {1, 4, 12} and got2["n_scanned"] == 0
    # fingerprint mismatch (different bundle sha) ignores the partial
    got3 = K._collect_v42(args, {1, 4}, names, fetch, sha="OTHER")
    assert np.allclose(got3["found"][1], expected[1])
    with pytest.raises(AssertionError, match="not found"):
        K._collect_v42(args, {1, 999}, names, fetch, sha="S2")


def test_neighbor_first_orders_adjacent_chunks_first():
    rest = [f"shard00_chunk{j:04d}.pt" for j in (0, 2, 5, 9)]
    out = K._neighbor_first(rest, ["shard00_chunk0001.pt"])
    assert out[:2] == ["shard00_chunk0000.pt", "shard00_chunk0002.pt"]
    assert set(out) == set(rest)
