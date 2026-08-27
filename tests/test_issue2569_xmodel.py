"""Unit tests for issue #2569 leg 7 — xmodel capture + atlas (tiny synthetic only).

Covers (plan-mandated): the ``_pack_batches`` bounds contract (plan §12 assumption
21), the fp16-overflow bf16u16 codec fallback with a synthetic >65,504 activation
(plan smoke blind-spot item 3), the #2054-lineage span helpers, the B5
boundary-equality math on a deterministic char-level fake tokenizer (no network,
no model), and the atlas math helpers (standardized-ridge beta equivalence, linear
CKA invariances, classical MDS, grouped folds, payload round-trip, feature-map
composition orientation). No live HF fetch; no GPU; dense d stays <= 16.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2569_atlas as AT  # noqa: E402
import issue2569_operator as OP  # noqa: E402
import issue2569_xmodel_capture as XC  # noqa: E402

# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


def test_codec_roundtrip_bitexact():
    """bf16-as-uint16 codec is bit-exact (ported #2378 contract)."""
    t = torch.randn(7, 5, dtype=torch.bfloat16)
    arr = XC.encode_bf16_u16(t)
    assert arr.dtype == np.uint16 and arr.shape == (7, 5)
    back = XC.decode_bf16_u16(arr)
    assert back.dtype == torch.bfloat16
    assert torch.equal(t, back)


def test_encode_summary_fp16_default_and_overflow_fallback():
    """fp16 storage below the 65,504 bound; the SYNTHETIC >65,504 activation routes
    to the bf16u16 codec (plan smoke blind-spot item 3 — the branch's own test)."""
    small = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    arr, codec = XC.encode_summary(small)
    assert codec == "fp16" and arr.dtype == np.float16
    assert np.allclose(XC.decode_summary(arr, codec), small, rtol=1e-3, atol=1e-3)

    big = small.copy()
    big[1, 3] = 70_000.0  # > FP16_MAX — a Qwen massive-activation-scale value
    arr2, codec2 = XC.encode_summary(big)
    assert codec2 == "bf16u16" and arr2.dtype == np.uint16
    back = XC.decode_summary(arr2, codec2)
    assert np.isfinite(back).all()
    assert abs(back[1, 3] - 70_000.0) / 70_000.0 < 1e-2  # bf16 relative precision

    with pytest.raises(AssertionError):
        XC.encode_summary(np.array([[np.inf]], dtype=np.float32))


# ---------------------------------------------------------------------------
# Batch packing (plan §12 assumption 21: bounds contract)
# ---------------------------------------------------------------------------


def test_pack_batches_bounds():
    """Every batch obeys BOTH knobs: <= max_batch_rows rows AND
    rows*batch_max_tokens <= batch_tokens (singleton over-budget rows allowed);
    the packing is a partition; the longest record runs FIRST (OOM fails fast)."""
    rng = np.random.default_rng(1)
    recs = [{"ci": i, "n_tokens": int(rng.integers(5, 400))} for i in range(97)]
    recs[13]["n_tokens"] = 5_000  # single over-budget row must still be packed alone
    batch_tokens, max_rows = 1024, 8
    batches = XC.pack_batches(recs, batch_tokens, max_rows)
    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(len(recs)))  # exact partition
    for b in batches:
        assert len(b) <= max_rows
        bmax = max(recs[i]["n_tokens"] for i in b)
        assert len(b) * bmax <= batch_tokens or len(b) == 1
    # longest-first: the global max lands in the first batch
    assert 13 in batches[0]


def test_pack_batches_max_batch_rows_is_not_a_total_cap():
    """--max-batch-rows caps rows PER FORWARD, never the total (#2054 class)."""
    recs = [{"ci": i, "n_tokens": 10} for i in range(50)]
    batches = XC.pack_batches(recs, batch_tokens=10_000, max_batch_rows=4)
    assert sum(len(b) for b in batches) == 50
    assert all(len(b) <= 4 for b in batches)


# ---------------------------------------------------------------------------
# Span helpers (#2054 lineage)
# ---------------------------------------------------------------------------


def test_char_span_to_token_span_and_token_before_char():
    """Overlap containment; zero-width rows skipped; (0,0) = no overlap."""
    offsets = [(0, 3), (3, 3), (3, 7), (7, 12), (12, 12), (12, 20)]
    assert XC._char_span_to_token_span(offsets, 3, 12) == (2, 4)
    assert XC._char_span_to_token_span(offsets, 0, 3) == (0, 1)
    assert XC._char_span_to_token_span(offsets, 20, 25) == (0, 0)
    assert XC._token_before_char(offsets, 7) == 2
    assert XC._token_before_char(offsets, 12) == 3
    assert XC._token_before_char(offsets, 2) is None  # never coerce to 0


def test_split_target():
    """1:2 holdout:sae split — exact at the production target, nonzero at smoke."""
    assert XC.split_target(60_000) == (20_000, 40_000)
    n_h, n_s = XC.split_target(32)
    assert n_h >= 1 and n_s >= 1 and n_h + n_s == 32


# ---------------------------------------------------------------------------
# Fake char-level tokenizer: tokenize_rows + the B5 boundary math (no model)
# ---------------------------------------------------------------------------


class FakeTok:
    """Deterministic char-level tokenizer with a prefix-stable chat template.

    Renders ``<U>q</U>`` per user turn, ``<A>...</A>`` per assistant turn, and a
    bare ``<A>`` generation suffix — so the prompt render is a strict character
    (and, char-level, token) PREFIX of the full render, mirroring the Qwen/Llama
    prefix-stable templates the capture convention requires."""

    bos_token_id = None
    pad_token_id = 0
    eos_token_id = 0
    chat_template = "fake-template-v1"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        out = ""
        for m in msgs:
            tag = "U" if m["role"] == "user" else "A"
            out += f"<{tag}>{m['content']}</{tag}>"
        if add_generation_prompt:
            out += "<A>"
        return out

    def __call__(self, texts, add_special_tokens=True, return_offsets_mapping=False, **kw):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        ids = [[(ord(c) % 997) + 1 for c in t] for t in items]
        out = {"input_ids": ids[0] if single else ids}
        if return_offsets_mapping:
            offs = [[(i, i + 1) for i in range(len(t))] for t in items]
            out["offset_mapping"] = offs[0] if single else offs
        return out


def _fake_rows():
    return [
        {"ci": 10, "corpus": "lmsys", "prompt": "what is x?", "response": "x is a letter."},
        {"ci": 11, "corpus": "wildchat", "prompt": "hi", "response": "hello there"},
        {"ci": 12, "corpus": "lmsys", "prompt": "long", "response": "r" * 500},
    ]


def test_tokenize_rows_fields_and_b5_boundary_equality():
    """The capture path's positions equal the INDEPENDENT offset-mapping-derived
    boundaries on the full render (the B5 identity-gate assert set, model-free)."""
    tok = FakeTok()
    probe = XC.template_probe(tok, "qwen")
    assert probe["gen_suffix"] == "<A>"
    kept, drops = XC.tokenize_rows(tok, _fake_rows(), probe["gen_suffix"], max_tokens=10_000)
    assert len(kept) == 3 and not drops
    for row, rec in zip(_fake_rows(), kept, strict=True):
        prompt_text, full_text = XC._render(tok, row["prompt"], row["response"])
        assert rec["prompt_len"] == len(prompt_text)  # char-level tokenizer
        assert rec["n_tokens"] == len(full_text)
        enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
        lo, hi = XC._char_span_to_token_span(
            enc["offset_mapping"], len(prompt_text), len(full_text)
        )
        assert (lo, hi) == (rec["ans_lo"], rec["ans_hi"])
        assert XC._token_before_char(enc["offset_mapping"], len(prompt_text)) == rec["v_C_pos"]
        assert rec["v_C_pos"] == rec["prompt_len"] - 1


def test_tokenize_rows_drop_reasons():
    """over_length + gen_suffix_mismatch drops are counted, never coerced."""
    tok = FakeTok()
    kept, drops = XC.tokenize_rows(tok, _fake_rows(), "<A>", max_tokens=40)
    assert drops["over_length"] >= 1
    assert len(kept) + sum(drops.values()) == 3
    kept2, drops2 = XC.tokenize_rows(tok, _fake_rows()[:1], "<X>", max_tokens=10_000)
    assert not kept2 and drops2["gen_suffix_mismatch"] == 1


def test_gate_rows_skips_dropped_candidates():
    """_gate_rows picks rows that SURVIVE tokenization, spanning both corpora,
    and fills the EXACT registered roster cardinality (never fewer)."""
    tok = FakeTok()
    texts = [
        *_fake_rows(),
        {"ci": 13, "corpus": "wildchat", "prompt": "q4", "response": "a4"},
        {"ci": 14, "corpus": "lmsys", "prompt": "q5", "response": "a5"},
    ]
    rows, recs = XC._gate_rows(tok=tok, texts=texts, gen_suffix="<A>", max_tokens=60, n=4)
    assert len(rows) == len(recs) == 4  # exact registered cardinality
    assert all(int(r["ci"]) != 12 for r in rows)  # the over-length candidate skipped
    assert {r["corpus"] for r in rows} == {"lmsys", "wildchat"}


# ---------------------------------------------------------------------------
# Atlas math helpers
# ---------------------------------------------------------------------------


def test_ridge_beta_at_lambda_matches_closed_form():
    """The beta payload reproduces the standardized-ridge closed form the reused
    #779 core implements (standardize X on train stats + 1e-9; center Y)."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50, 6))
    Y = rng.standard_normal((50, 4))
    tr = np.arange(35)
    lam = 3.7
    payload = AT.ridge_beta_at_lambda(X, Y, tr, lam)
    # closed form, computed independently in numpy
    xmu = X[tr].mean(0)
    xsd = X[tr].std(0, ddof=1) + 1e-9  # torch .std(0) default is ddof=1; core adds 1e-9
    Xn = (X[tr] - xmu) / xsd
    ymu = Y[tr].mean(0)
    W = np.linalg.solve(Xn.T @ Xn + lam * np.eye(6), Xn.T @ (Y[tr] - ymu))
    pred_ref = ((X - xmu) / xsd) @ W + ymu
    pred_payload = OP.predict(payload, X)
    assert np.allclose(pred_payload, pred_ref, rtol=1e-8, atol=1e-8)
    # row_operator applies the SAME affine map (B1 row-action contract)
    A, b = OP.row_operator(payload)
    assert np.allclose(X @ A + b, pred_payload, rtol=1e-8, atol=1e-8)


def test_cka_linear_invariances():
    """Linear CKA: 1.0 under orthogonal rotation + isotropic scale; low for noise."""
    rng = np.random.default_rng(3)
    X = rng.standard_normal((200, 6))
    Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    assert AT.cka_linear(X, 3.0 * (X @ Q)) == pytest.approx(1.0, abs=1e-9)
    assert AT.cka_linear(X, rng.standard_normal((200, 6))) < 0.3


def test_mds_2d_recovers_planar_distances():
    """Classical MDS reproduces a planar configuration's distance matrix."""
    rng = np.random.default_rng(4)
    pts = rng.standard_normal((7, 2))
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    coords = AT.mds_2d(D)
    D2 = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    assert np.allclose(D, D2, atol=1e-8)


def test_grouped_folds_partition_and_determinism():
    """tr/va/te are disjoint, deterministic, and ci-keyed (machine-stable hash)."""
    ci = np.arange(0, 4000, 7, dtype=np.int64)
    f1 = AT.grouped_folds(ci, val_rows=32)
    f2 = AT.grouped_folds(ci, val_rows=32)
    for k in ("tr", "va", "te"):
        assert np.array_equal(f1[k], f2[k])
    allv = np.concatenate([f1["tr"], f1["va"], f1["te"]])
    assert len(np.unique(allv)) == len(allv)
    assert f1["n_train_90pct"] == len(f1["tr"]) + len(f1["va"])


def test_payload_dict_roundtrip_and_validation():
    """payload_to_dict/from_dict round-trips rectangular maps; bad xsd raises."""
    rng = np.random.default_rng(5)
    p = AT.ridge_beta_at_lambda(
        rng.standard_normal((30, 5)), rng.standard_normal((30, 3)), np.arange(20), 1.0
    )
    d = AT.payload_to_dict(p)
    p2 = AT.payload_from_dict(d, path=Path("/tmp/x.pt"))
    assert p2.W.shape == (5, 3)
    assert np.allclose(p2.W, p.W, atol=1e-6)
    bad = dict(d)
    bad["xsd"] = torch.zeros(5)
    with pytest.raises(AssertionError):
        AT.payload_from_dict(bad, path=Path("/tmp/bad.pt"))


def test_featmap_composition_orientation():
    """A_feat = E @ diag(1/xsd)W @ D acts by v @ A_feat == ((v@E)/xsd @ W) @ D
    (row-action B1 convention; biases excluded from the linear operator)."""
    rng = np.random.default_rng(6)
    d, m, u = 6, 4, 3
    E = rng.standard_normal((d, m))  # encoder columns (alive features)
    D = rng.standard_normal((u, d))  # decoder rows (union features)
    payload = AT.ridge_beta_at_lambda(
        rng.standard_normal((40, m)), rng.standard_normal((40, u)), np.arange(30), 2.0
    )
    A_mid, _b = OP.row_operator(payload)
    A_feat = E @ A_mid @ D
    v = rng.standard_normal((2, d))
    manual = (((v @ E) / payload.xsd) @ payload.W) @ D
    assert np.allclose(v @ A_feat, manual, rtol=1e-10, atol=1e-10)


def test_spectrum_cosine_truncation_flag():
    """Cross-shape spectrum cosine truncates to min(d) and records it."""
    rng = np.random.default_rng(7)
    out = AT.spectrum_cosine(rng.standard_normal((5, 5)), rng.standard_normal((8, 8)))
    assert out["truncated"] is True and out["k"] == 5
    same = AT.spectrum_cosine(np.eye(4), np.eye(4))
    assert same["spectrum_cosine"] == pytest.approx(1.0)
    assert same["truncated"] is False


def _fake_stage_from(store):
    """Signature-conformant stage_hub_file fake serving a dict store (mirrors
    the real keyword-only repo_type/revision surface)."""
    import json as _json

    calls = []

    def fake_stage(repo, repo_path, dest, *, repo_type, revision=None, **kw):
        calls.append((repo_path, revision))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(_json.dumps(store[repo_path]))
        return dest

    return fake_stage, calls


def _snap(store, blob_prefix="blob0"):
    """Snapshot tuple for a fake store (blob ids derived from the prefix so a
    'regenerated' store is expressed as a different blob_prefix)."""
    return (
        f"rev-{blob_prefix}",
        [(p, f"{blob_prefix}-{p}", None) for p in sorted(store)],
    )


def test_stage_texts_tolerates_shard_skip_manifests(tmp_path, monkeypatch):
    """A shard-level skip manifest among the rows chunks is counted, never
    row-parsed (final-round smoke: shard16_skipped.json crashed the strict
    rows assert at file 1021/1936 of the real store); a rows chunk still
    stages, and an UNRECOGNIZED shape still fails the strict assert."""
    import json as _json

    chunk = {
        "rows": [{"ci": 7, "prompt": "p", "response": "r"}],
        "shard_index": 0,
        "chunk": 0,
    }
    skip = {
        "skipped": [11],
        "n_skipped": 1,
        "num_shards": 32,
        "shard_index": 16,
        "gen_max_tokens": 2048,
    }
    bogus = {"unexpected": True}
    pfx = XC.RAW_COMPLETIONS_PREFIX
    store = {
        f"{pfx}/shard00_chunk0000.json": chunk,
        f"{pfx}/shard16_skipped.json": skip,
    }
    fake_stage, _calls = _fake_stage_from(store)
    monkeypatch.setattr(XC.hub, "stage_hub_file", fake_stage)
    args = type("A", (), {"out_root": str(tmp_path), "hf_data_repo": "fake/repo"})()
    sel_meta = {"drops": {}}
    XC._stage_texts(args, np.asarray([7], dtype=np.int64), {7: "lmsys"}, sel_meta, _snap(store))
    assert sel_meta["n_texts_kept"] == 1
    assert sel_meta["drops"]["skip_manifest_files"] == 1
    kept = (tmp_path / "texts_kept.jsonl").read_text().strip().splitlines()
    assert len(kept) == 1 and _json.loads(kept[0])["ci"] == 7
    # unrecognized shape still fails loud (strict schema assert retained)
    store[f"{pfx}/shard99_chunk0000.json"] = bogus
    args2 = type("A", (), {"out_root": str(tmp_path / "b"), "hf_data_repo": "fake/repo"})()
    with pytest.raises(AssertionError):
        XC._stage_texts(
            args2, np.asarray([7], dtype=np.int64), {7: "lmsys"}, {"drops": {}}, _snap(store)
        )


def _write_finalize_meta(tmp_path, model, realized):
    fdir = tmp_path / "final"
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / f"{model}_finalize_meta.json").write_text(json.dumps({"realized": realized}))


def test_phase_sentinel_envelope_is_poller_conformant(tmp_path):
    """The pd-done sentinel carries poll_pipeline._SENTINEL_REQUIRED_KEYS
    (schema 1) so the VM drain parses + renames it instead of warn-skipping
    every tick — re-verified by a round-trip through the REAL two-arg
    ``poll_pipeline._parse_sentinel``, never by reading the writer."""
    import json as _json

    import poll_pipeline as PP

    for model in XC.MODEL_SPECS:
        _write_finalize_meta(tmp_path, model, 8)
    args = type(
        "A",
        (),
        {
            "sentinel_path": str(tmp_path / "issue-2569-pd-done.json"),
            "out_root": str(tmp_path),
        },
    )()
    XC.phase_sentinel(args)
    body = (tmp_path / "issue-2569-pd-done.json").read_text()
    payload = _json.loads(body)
    for k in PP._SENTINEL_REQUIRED_KEYS:
        assert k in payload, k
    parsed = PP._parse_sentinel("issue-2569-pd-done.json", body)
    assert parsed is not None and parsed["kind"] == "phase-pd-done"
    assert parsed["qwen_realized"] == 8 and parsed["llama_realized"] == 8


def test_phase_sentinel_requires_both_finalize_metas(tmp_path):
    """(resume-keys-omit-content-and-required-outputs) The done sentinel is the
    P-D lane's completion claim: it REFUSES to be written while either model's
    finalize meta — a required output — is absent (was: a silent `if
    meta.exists()` optional read that signalled done with finalize missing)."""
    args = type(
        "A",
        (),
        {
            "sentinel_path": str(tmp_path / "issue-2569-pd-done.json"),
            "out_root": str(tmp_path),
        },
    )()
    with pytest.raises(AssertionError, match="finalize"):
        XC.phase_sentinel(args)
    assert not (tmp_path / "issue-2569-pd-done.json").exists()
    _write_finalize_meta(tmp_path, "qwen", 8)  # one of two present — still refused
    with pytest.raises(AssertionError, match="llama"):
        XC.phase_sentinel(args)
    assert not (tmp_path / "issue-2569-pd-done.json").exists()


# ---------------------------------------------------------------------------
# Step 5 fix round 2 — the four P-D pilot/gate BLOCKERs
# (pilot-gate-halt-erased-by-resume / pd-gate-precondition-bypass /
#  pd-gate-cardinality-unenforced / gpu-gate-phases-not-idempotent)
# ---------------------------------------------------------------------------


def _cli_args(tmp_path, *, phase="capture", model="qwen", rows="32", extra=()):
    """Args through the REAL argparse surface (never a hand-built Namespace)."""
    return XC._parse_args(
        [
            "--phase",
            phase,
            "--model",
            model,
            "--rows",
            str(rows),
            "--out-root",
            str(tmp_path),
            "--device",
            "cpu",
            *extra,
        ]
    )


def _write_gate(tmp_path, name, payload):
    gdir = tmp_path / "gates"
    gdir.mkdir(parents=True, exist_ok=True)
    path = gdir / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path


def _seed_texts(tmp_path, n=10):
    rows = [
        {
            "ci": i,
            "corpus": "lmsys" if i % 2 else "wildchat",
            "prompt": f"q{i}",
            "response": f"a{i}",
        }
        for i in range(n)
    ]
    (tmp_path / "texts_kept.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


def _pparams(args, model=None):
    """The pilot-binding params for a real argparse namespace (real helper)."""
    spec = XC.MODEL_SPECS[model or args.model]
    return XC._pilot_params(args, spec, XC._parse_layers(args, spec))


def _bound_spot_gate_record(tmp_path, *, layers_flag=None, tol=None, texts=None):
    """A spot-gate PASS record BOUND through the REAL ``_gate_regime`` over the
    staged selection (never a hand-rolled regime — the round-2 fixture trap).
    ``texts`` overrides the selection the regime is computed over (to express a
    record measured on OTHER content)."""
    flags = []
    if layers_flag is not None:
        flags += ["--layers", layers_flag]
    if tol is not None:
        flags += ["--gate-rel-tol", str(tol)]
    gargs = _cli_args(tmp_path, phase="spot-gate", model="qwen", rows="8", extra=tuple(flags))
    spec = XC.MODEL_SPECS["qwen"]
    regime = XC._gate_regime(
        gargs,
        spec,
        XC._parse_layers(gargs, spec),
        texts if texts is not None else XC.load_selection(gargs),
        extra={"spot_chunk": gargs.spot_chunk},
    )
    return {"verdict": "PASS", "regime": regime}


# --- blocker 1: pilot-gate-halt-erased-by-resume ---------------------------


def test_pilot_gate_fail_record_survives_noop_resume_and_rehalts(tmp_path):
    """A blind relaunch whose chunks ALL resume-skip (zero fresh forward rows)
    must NOT overwrite the recorded FAIL with a PASS measured off loop overhead
    — the FAIL stands byte-identical and the designed rc=3 halt re-fires."""
    args = _cli_args(tmp_path)
    fail_rec = {
        "per_row_s": 5.0,
        "rows_measured": 32,
        "extrapolated_wall_h": 166.7,
        "booked_wall_h": 6.0,
        "verdict": "FAIL",
    }
    path = _write_gate(tmp_path, "pilot_gate_qwen", fail_rec)
    with pytest.raises(SystemExit) as ei:
        XC._pilot_gate_report(
            args, _pparams(args), fresh_rows=0, fresh_wall_s=0.05, resumed_chunks=1
        )
    assert ei.value.code == 3
    assert json.loads(path.read_text()) == fail_rec  # never rewritten


def test_pilot_gate_noop_smoke_resume_missing_record_fails_loud(tmp_path):
    """(resume-keys-omit-content-and-required-outputs) A smoke-scale resume that
    did NO fresh work and finds NO pilot record must FAIL LOUD — the pilot
    verdict is the smoke capture's REQUIRED output, and the chunk resume-skip
    may not stand in for the measurement (was: a silent 'nothing to measure'
    return, i.e. exit 0 with the required output absent). Production scale
    stays informational (no verdict is owed there — the smoke record is the
    durable gate), and a prior PASS record is never mutated."""
    args = _cli_args(tmp_path)
    with pytest.raises(RuntimeError, match="required output"):
        XC._pilot_gate_report(
            args, _pparams(args), fresh_rows=0, fresh_wall_s=0.0, resumed_chunks=2
        )
    assert not (tmp_path / "gates" / "pilot_gate_qwen.json").exists()  # still writes nothing
    prod = _cli_args(tmp_path, rows="0")
    XC._pilot_gate_report(prod, _pparams(prod), fresh_rows=0, fresh_wall_s=0.0, resumed_chunks=2)
    assert not (tmp_path / "gates" / "capture_wall_qwen.json").exists()
    pass_rec = {"per_row_s": 0.1, "rows_measured": 32, "verdict": "PASS"}
    path = _write_gate(tmp_path, "pilot_gate_qwen", pass_rec)
    XC._pilot_gate_report(args, _pparams(args), fresh_rows=0, fresh_wall_s=0.0, resumed_chunks=2)
    assert json.loads(path.read_text()) == pass_rec  # PASS idempotent, byte-identical


def test_pilot_gate_fresh_smoke_measurement_pass_and_fail(tmp_path):
    """A fresh smoke-scale measurement writes an honest verdict: PASS proceeds,
    FAIL persists the record then halts rc=3 (plan §7 halt-and-report); the
    record BINDS the capture params it was measured under
    (pd-pilot-pass-not-bound-to-production-regime)."""
    args = _cli_args(tmp_path)
    XC._pilot_gate_report(args, _pparams(args), fresh_rows=32, fresh_wall_s=3.2, resumed_chunks=0)
    rec = json.loads((tmp_path / "gates" / "pilot_gate_qwen.json").read_text())
    assert rec["verdict"] == "PASS"
    assert rec["rows_measured"] == 32 and rec["resumed_chunks"] == 0
    assert rec["capture_params"] == _pparams(args)  # the regime the PASS certifies
    slow = tmp_path / "slow"
    args2 = _cli_args(slow)
    with pytest.raises(SystemExit) as ei:
        XC._pilot_gate_report(
            args2, _pparams(args2), fresh_rows=32, fresh_wall_s=2.0 * 32, resumed_chunks=0
        )
    assert ei.value.code == 3
    rec2 = json.loads((slow / "gates" / "pilot_gate_qwen.json").read_text())
    assert rec2["verdict"] == "FAIL"


def test_pilot_gate_production_scale_never_touches_the_gate_record(tmp_path):
    """A production-scale run's wall report is INFORMATIONAL (separate file,
    no verdict, no halt); the smoke pilot verdict survives untouched (plan §7:
    the 32-row smoke IS the pilot)."""
    args = _cli_args(tmp_path, rows="0")
    pass_rec = {"verdict": "PASS", "per_row_s": 0.1}
    path = _write_gate(tmp_path, "pilot_gate_qwen", pass_rec)
    XC._pilot_gate_report(
        args, _pparams(args), fresh_rows=60_000, fresh_wall_s=8.0 * 3600, resumed_chunks=0
    )
    assert json.loads(path.read_text()) == pass_rec
    wall = json.loads((tmp_path / "gates" / "capture_wall_qwen.json").read_text())
    assert wall.get("informational") is True and "verdict" not in wall


# --- blocker 2: pd-gate-precondition-bypass ---------------------------------


def test_capture_production_scale_requires_pilot_pass(tmp_path):
    """The production pass refuses to START without a pilot PASS record
    (plan §7 P-D pilot row: halt-and-report BEFORE the full pass)."""
    _write_gate(tmp_path, "spot_gate_qwen", {"verdict": "PASS"})
    args = _cli_args(tmp_path, rows="0")
    with pytest.raises(AssertionError, match="pilot"):
        XC.phase_capture(args)


def test_capture_smoke_scale_is_the_pilot_no_self_precondition(tmp_path):
    """The smoke-scale capture IS the pilot — it must not demand its own record
    (smoke/production gate-calibration parity, the #1345 class): it proceeds
    past the pilot precondition to the selection load (the live-regime gate
    binding runs AFTER the selection is loaded, since it hashes the texts)."""
    _write_gate(tmp_path, "spot_gate_qwen", {"verdict": "PASS"})
    args = _cli_args(tmp_path, rows="32")
    with pytest.raises(AssertionError, match="texts_kept"):
        XC.phase_capture(args)


def _seed_chunk_store(tmp_path, model="qwen", n=8, layer=14):
    """Minimal consistent chunk store + staged texts for finalize (CPU-only).
    The regime is built through the REAL ``_capture_regime`` helper — a
    hand-rolled fixture regime validates against an impossible schema (the
    round-2 fixture trap) and would miss the pilot-binding fields finalize now
    reads back from ``regime.json``. Returns the regime."""
    rows = [{"ci": i, "corpus": "lmsys", "prompt": f"q{i}", "response": f"a{i}"} for i in range(n)]
    (tmp_path / "texts_kept.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cargs = _cli_args(
        tmp_path, phase="capture", model=model, rows=str(n), extra=("--layers", str(layer))
    )
    spec = XC.MODEL_SPECS[model]
    hidden = spec["hidden"]
    regime = XC._capture_regime(
        cargs,
        spec,
        XC._parse_layers(cargs, spec),
        np.arange(n, dtype=np.int64),
        "fake-sha",
        XC._texts_content_sha(XC.load_selection(cargs)),
    )
    cdir = tmp_path / "chunks" / model
    cdir.mkdir(parents=True)
    (cdir / "regime.json").write_text(json.dumps(regime))
    rng = np.random.default_rng(0)
    arrays, codecs = {}, {}
    for tag in ("vc", "va"):
        arr, codec = XC.encode_summary(rng.standard_normal((n, hidden)).astype(np.float32))
        arrays[f"{tag}_l{layer}"] = arr
        codecs[f"{tag}_l{layer}"] = codec
    torch.save(
        {
            "ci": np.arange(n, dtype=np.int64),
            "corpus": ["lmsys"] * n,
            "n_tokens": np.full(n, 5, dtype=np.int64),
            "prompt_len": np.full(n, 3, dtype=np.int64),
            "layers": [layer],
            "arrays": arrays,
            "codecs": codecs,
            "drops": {},
            "regime": regime,
        },
        cdir / "chunk00000.pt",
    )
    return regime


def test_finalize_requires_gate_and_pilot_pass(tmp_path):
    """finalize is DOWNSTREAM of the gates: chunks exist even from a halted
    pilot run (the chunk lands before the gate fires), so finalize refuses
    without the model gate record, without a pilot record, on a pilot FAIL,
    on an UNBOUND pilot PASS (no capture_params), and on a PASS measured
    under a DIFFERENT capture regime — binding is checked against the chunk
    store's OWN regime.json, what the capture actually ran with
    (pd-pilot-pass-not-bound-to-production-regime). It proceeds only on a
    PASS bound to the matching regime."""
    regime = _seed_chunk_store(tmp_path)
    args = _cli_args(tmp_path, phase="finalize", rows="8", extra=("--skip-upload",))
    with pytest.raises(AssertionError, match="run the gate phase"):
        XC.phase_finalize(args)
    _write_gate(tmp_path, "spot_gate_qwen", _bound_spot_gate_record(tmp_path, layers_flag="14"))
    with pytest.raises(AssertionError, match="pilot"):
        XC.phase_finalize(args)
    _write_gate(tmp_path, "pilot_gate_qwen", {"verdict": "FAIL"})
    with pytest.raises(AssertionError, match="pilot gate blocks"):
        XC.phase_finalize(args)
    # unbound PASS: no capture_params — an unrelated historical PASS certifies nothing
    _write_gate(tmp_path, "pilot_gate_qwen", {"verdict": "PASS"})
    with pytest.raises(AssertionError, match="capture_params"):
        XC.phase_finalize(args)
    # PASS measured under a DIFFERENT execution shape — refused with the diff named
    good = XC._pilot_binding_from_regime(regime)
    _write_gate(
        tmp_path,
        "pilot_gate_qwen",
        {"verdict": "PASS", "capture_params": dict(good, batch_tokens=4096)},
    )
    with pytest.raises(AssertionError, match="DIFFERENT capture regime"):
        XC.phase_finalize(args)
    # PASS bound to the matching regime — proceeds
    _write_gate(tmp_path, "pilot_gate_qwen", {"verdict": "PASS", "capture_params": good})
    XC.phase_finalize(args)
    assert (tmp_path / "final" / "qwen_vc_L14.pt").exists()
    meta = json.loads((tmp_path / "final" / "qwen_finalize_meta.json").read_text())
    assert meta["realized"] == 8


def test_pilot_binding_from_regime_rejects_prebinding_store(tmp_path):
    """A chunk regime lacking the binding fields (a pre-binding store) is
    refused loud — never silently accepted as 'any PASS will do'."""
    with pytest.raises(AssertionError, match="pilot-binding fields"):
        XC._pilot_binding_from_regime({"layers": [14], "template_sha": "x"})


# --- blocker 3: pd-gate-cardinality-unenforced ------------------------------


def test_gate_rows_short_roster_fails_loud():
    """Fewer surviving rows than the registered roster is a LOUD failure,
    never a PASS computed over a smaller basis."""
    tok = FakeTok()
    with pytest.raises(AssertionError, match="roster short"):
        XC._gate_rows(tok=tok, texts=_fake_rows(), gen_suffix="<A>", max_tokens=10_000, n=8)


def test_parse_layers_rejects_empty_list(tmp_path):
    """An empty resolved layer list would make every gate/capture vacuous."""
    args = _cli_args(tmp_path, extra=("--layers", ","))
    with pytest.raises(AssertionError, match="EMPTY"):
        XC._parse_layers(args, XC.MODEL_SPECS["qwen"])


def test_spot_gate_layer_coverage_guard():
    """A gated layer absent from the banked oracle yields ZERO admissible
    comparisons — refused loud, never a vacuous PASS (distinguishable from a
    genuine negative by construction: no verdict is computed at all)."""
    XC._require_layer_coverage([14, 19], [14, 19, 26])
    with pytest.raises(AssertionError, match="vacuous"):
        XC._require_layer_coverage([15], [14, 19, 26])


# --- blocker 4: gpu-gate-phases-not-idempotent ------------------------------


def test_identity_gate_resume_skips_on_pass_record(tmp_path, monkeypatch):
    """A PASS record under an IDENTICAL regime resume-skips the whole gate body
    — no model load, no forwards, no record mutation."""
    args = _cli_args(tmp_path, phase="identity-gate", model="llama")
    _seed_texts(tmp_path)
    spec = XC.MODEL_SPECS["llama"]
    layers = XC._parse_layers(args, spec)
    regime = XC._gate_regime(args, spec, layers, XC.load_selection(args))
    rec = {"verdict": "PASS", "regime": regime, "worst_rel_diff": 0.001}
    path = _write_gate(tmp_path, "identity_gate_llama", rec)

    def boom(*a, **k):
        raise RuntimeError("model load attempted")

    monkeypatch.setattr(XC, "_load_model_ctx", boom)
    XC.phase_identity_gate(args)  # returns silently — never reaches the model load
    assert json.loads(path.read_text()) == rec  # record untouched


def test_identity_gate_reruns_on_fail_verdict_or_regime_drift(tmp_path, monkeypatch):
    """A FAIL record — or a PASS under a DRIFTED regime — re-runs the REAL gate
    (reaches the model-load boundary) instead of resume-skipping."""
    args = _cli_args(tmp_path, phase="identity-gate", model="llama")
    _seed_texts(tmp_path)
    spec = XC.MODEL_SPECS["llama"]
    layers = XC._parse_layers(args, spec)
    regime = XC._gate_regime(args, spec, layers, XC.load_selection(args))

    def boom(*a, **k):
        raise RuntimeError("model load attempted")

    monkeypatch.setattr(XC, "_load_model_ctx", boom)
    _write_gate(tmp_path, "identity_gate_llama", {"verdict": "FAIL", "regime": regime})
    with pytest.raises(RuntimeError, match="model load attempted"):
        XC.phase_identity_gate(args)
    drifted = dict(regime, gate_rel_tol=0.5)
    _write_gate(tmp_path, "identity_gate_llama", {"verdict": "PASS", "regime": drifted})
    with pytest.raises(RuntimeError, match="model load attempted"):
        XC.phase_identity_gate(args)


def test_spot_gate_resume_skips_on_pass_record(tmp_path, monkeypatch):
    """spot-gate: PASS + identical regime (incl. --spot-chunk) skips before any
    hub staging; a different --spot-chunk drifts the regime and re-runs."""
    args = _cli_args(tmp_path, phase="spot-gate", model="qwen")
    _seed_texts(tmp_path)
    spec = XC.MODEL_SPECS["qwen"]
    layers = XC._parse_layers(args, spec)
    regime = XC._gate_regime(
        args, spec, layers, XC.load_selection(args), extra={"spot_chunk": str(args.spot_chunk)}
    )
    _write_gate(tmp_path, "spot_gate_qwen", {"verdict": "PASS", "regime": regime})

    def boom(*a, **k):
        raise RuntimeError("hub staging attempted")

    monkeypatch.setattr(XC.hub, "stage_hub_file", boom)
    XC.phase_spot_gate(args)  # skip fires before staging
    args2 = _cli_args(
        tmp_path,
        phase="spot-gate",
        model="qwen",
        extra=("--spot-chunk", "shard01_chunk0000.pt"),
    )
    with pytest.raises(RuntimeError, match="hub staging attempted"):
        XC.phase_spot_gate(args2)


# ---------------------------------------------------------------------------
# Step 5 fix round 3 — Unit G4 BLOCKERs
# (gpu-gate-resume-key-omits-text-content /
#  pd-pilot-pass-not-bound-to-production-regime /
#  resume-keys-omit-content-and-required-outputs)
# ---------------------------------------------------------------------------


def test_texts_content_sha_keys_on_content_and_order_not_identity():
    """The content fingerprint moves when a row's TEXT changes at the same ci
    set, and when the kept ORDER changes (the gate roster and capture slice are
    position-dependent); it is deterministic across equal copies."""
    rows = _fake_rows()
    a = XC._texts_content_sha(rows)
    assert a == XC._texts_content_sha([dict(r) for r in rows])  # deterministic
    changed = [dict(r) for r in rows]
    changed[1]["response"] = "different words, same ci"
    assert XC._texts_content_sha(changed) != a  # identity same, content moved
    assert XC._texts_content_sha([rows[1], rows[0], rows[2]]) != a  # order-sensitive


def test_gate_resume_reruns_on_text_content_change_same_ci(tmp_path, monkeypatch):
    """(gpu-gate-resume-key-omits-text-content) A gate PASS recorded under the
    OLD texts must NOT be reused when a response body is regenerated at the
    SAME ci selection: the regime carries the text-content sha, so the gate
    RE-RUNS (reaches the model-load boundary) instead of resume-skipping."""
    args = _cli_args(tmp_path, phase="identity-gate", model="llama")
    rows = _seed_texts(tmp_path)
    spec = XC.MODEL_SPECS["llama"]
    layers = XC._parse_layers(args, spec)
    regime_before = XC._gate_regime(args, spec, layers, XC.load_selection(args))
    _write_gate(tmp_path, "identity_gate_llama", {"verdict": "PASS", "regime": regime_before})
    # regenerate ONE response at the same ci set (same selection identity)
    rows[3]["response"] = "a3 REGENERATED"
    (tmp_path / "texts_kept.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    regime_after = XC._gate_regime(args, spec, layers, XC.load_selection(args))
    assert regime_after["selection_ci_sha256"] == regime_before["selection_ci_sha256"]
    assert regime_after != regime_before  # the content sha moved the key

    def boom(*a, **k):
        raise RuntimeError("model load attempted")

    monkeypatch.setattr(XC, "_load_model_ctx", boom)
    with pytest.raises(RuntimeError, match="model load attempted"):
        XC.phase_identity_gate(args)


def test_capture_regime_includes_text_content(tmp_path):
    """(resume-keys-omit-content-and-required-outputs) The chunk-store regime
    moves when the consumed text content changes at the same ci slice, so
    ``_check_regime`` wipes the stale chunks instead of resume-skipping them."""
    args = _cli_args(tmp_path)
    spec = XC.MODEL_SPECS["qwen"]
    layers = XC._parse_layers(args, spec)
    rows = _fake_rows()
    ci = np.asarray([int(r["ci"]) for r in rows], dtype=np.int64)
    r1 = XC._capture_regime(args, spec, layers, ci, "tsha", XC._texts_content_sha(rows))
    changed = [dict(r) for r in rows]
    changed[0]["response"] = "regenerated"
    r2 = XC._capture_regime(args, spec, layers, ci, "tsha", XC._texts_content_sha(changed))
    assert r1["kept_ci_sha256"] == r2["kept_ci_sha256"] and r1 != r2
    # and the pilot-binding subset is readable back from the regime (finalize path)
    assert XC._pilot_binding_from_regime(r1) == _pparams(args)


def test_capture_production_refuses_unbound_or_mismatched_pilot_pass(tmp_path, monkeypatch):
    """(pd-pilot-pass-not-bound-to-production-regime) The production pass
    refuses a pilot PASS that carries no capture_params, and one measured under
    a DIFFERENT execution shape; a PASS bound to the matching shape proceeds
    past the precondition (to the model-load boundary — seam-stubbed so a
    wrongly-accepted PASS would fail the test by touching the model)."""
    _seed_texts(tmp_path)
    _write_gate(tmp_path, "spot_gate_qwen", _bound_spot_gate_record(tmp_path))
    args = _cli_args(tmp_path, rows="0")

    def boom(*a, **k):
        raise RuntimeError("model load attempted")

    monkeypatch.setattr(XC, "_load_model_ctx", boom)
    good = _pparams(args)
    _write_gate(tmp_path, "pilot_gate_qwen", {"verdict": "PASS"})  # unbound
    with pytest.raises(AssertionError, match="capture_params"):
        XC.phase_capture(args)
    _write_gate(
        tmp_path,
        "pilot_gate_qwen",
        {"verdict": "PASS", "capture_params": dict(good, max_batch_rows=8)},
    )
    with pytest.raises(AssertionError, match="DIFFERENT capture regime"):
        XC.phase_capture(args)
    _write_gate(tmp_path, "pilot_gate_qwen", {"verdict": "PASS", "capture_params": good})
    with pytest.raises(RuntimeError, match="model load attempted"):
        XC.phase_capture(args)  # binding satisfied — proceeds to the real work


# ---------------------------------------------------------------------------
# Step 5 fix round v4 — pd-gate-pass-not-bound-to-live-regime (BLOCKER) +
# select-stage-texts-resume-content-unpinned (CONCERN)
# ---------------------------------------------------------------------------


def test_capture_refuses_unbound_or_regime_drifted_gate_pass(tmp_path, monkeypatch):
    """(pd-gate-pass-not-bound-to-live-regime) The capture consumer refuses a
    gate PASS that is UNBOUND (no regime), bound to DIFFERENT text content at
    the same ci selection, bound to different layers, or measured at a LOOSER
    tolerance than the live bar — each refusal naming its cause. Pre-fix, all
    four proceeded to the model load (the monkeypatched sentinel here)."""
    _seed_texts(tmp_path, n=10)
    args = _cli_args(tmp_path, rows="8")  # smoke scale: pilot-exempt, gate-checked
    texts = XC.load_selection(args)

    def boom(*a, **k):
        raise RuntimeError("model-load reached")

    monkeypatch.setattr(XC, "_load_model_ctx", boom)

    # (i) legacy/unbound PASS — refused loud
    _write_gate(tmp_path, "spot_gate_qwen", {"verdict": "PASS"})
    with pytest.raises(AssertionError, match="carries no regime"):
        XC.phase_capture(args)

    # (ii) PASS measured over REGENERATED text at the same cis — refused naming the field
    stale = [dict(r, response=r["response"] + " REGENERATED") for r in texts]
    _write_gate(tmp_path, "spot_gate_qwen", _bound_spot_gate_record(tmp_path, texts=stale))
    with pytest.raises(AssertionError, match="texts_sha256"):
        XC.phase_capture(args)

    # (iii) PASS measured at other layers than the live run's — refused naming the field
    _write_gate(tmp_path, "spot_gate_qwen", _bound_spot_gate_record(tmp_path, layers_flag="14"))
    with pytest.raises(AssertionError, match="layers"):
        XC.phase_capture(args)

    # (iv) PASS measured at a LOOSER tolerance than the live bar — refused
    _write_gate(tmp_path, "spot_gate_qwen", _bound_spot_gate_record(tmp_path, tol=0.05))
    with pytest.raises(AssertionError, match="LOOSER"):
        XC.phase_capture(args)


def test_capture_honours_bound_gate_pass_allowed_differences(tmp_path, monkeypatch):
    """(the too-strict/deadlock direction) A PASS bound to the live regime is
    honoured — capture proceeds to the model-load boundary — including when the
    ALLOWED-to-differ members differ: a different spot_chunk (gate-internal
    oracle identity) and a TIGHTER recorded tolerance (certifies the live bar
    a fortiori)."""
    _seed_texts(tmp_path, n=10)
    args = _cli_args(tmp_path, rows="8")

    def boom(*a, **k):
        raise RuntimeError("model-load reached")

    monkeypatch.setattr(XC, "_load_model_ctx", boom)
    rec = _bound_spot_gate_record(tmp_path, tol=0.01)  # tighter than the live default 2e-2
    rec["regime"]["spot_chunk"] = "shard31_chunk9999.pt"  # differs from args default — allowed
    _write_gate(tmp_path, "spot_gate_qwen", rec)
    with pytest.raises(RuntimeError, match="model-load reached"):
        XC.phase_capture(args)  # gate binding satisfied — proceeds to the real work


def test_finalize_gate_binding_store_knobs_and_live_texts(tmp_path):
    """(pd-gate-pass-not-bound-to-live-regime, finalize consumer) Finalize
    verifies the gate PASS against the chunk store's OWN knobs + the staged
    selection NOW on disk: a matching bound record proceeds; regenerated
    texts_kept content refuses naming texts_sha256; a record whose batch
    packing differs from the store refuses naming batch_tokens; a pre-binding
    chunk regime fails loud in _gate_binding_from_store."""
    regime = _seed_chunk_store(tmp_path)
    args = _cli_args(tmp_path, phase="finalize", rows="8", extra=("--skip-upload",))
    bound = _bound_spot_gate_record(tmp_path, layers_flag="14")
    _write_gate(tmp_path, "spot_gate_qwen", bound)
    _write_gate(
        tmp_path,
        "pilot_gate_qwen",
        {"verdict": "PASS", "capture_params": XC._pilot_binding_from_regime(regime)},
    )
    XC.phase_finalize(args)  # bound record honoured
    assert (tmp_path / "final" / "qwen_finalize_meta.json").is_file()

    # regenerated staged texts (same cis) — finalize refuses naming the content field
    orig = (tmp_path / "texts_kept.jsonl").read_text()
    regen = [dict(json.loads(ln), response="REGEN") for ln in orig.split("\n") if ln.strip()]
    (tmp_path / "texts_kept.jsonl").write_text("\n".join(json.dumps(r) for r in regen) + "\n")
    with pytest.raises(AssertionError, match="texts_sha256"):
        XC.phase_finalize(args)
    (tmp_path / "texts_kept.jsonl").write_text(orig)

    # record whose packing knobs differ from the store's OWN regime — refused
    drifted = {"verdict": "PASS", "regime": dict(bound["regime"], batch_tokens=4096)}
    _write_gate(tmp_path, "spot_gate_qwen", drifted)
    with pytest.raises(AssertionError, match="batch_tokens"):
        XC.phase_finalize(args)

    # pre-binding chunk regime (knob fields absent) — fails loud, never "any PASS will do"
    with pytest.raises(AssertionError, match="gate-binding fields"):
        XC._gate_binding_from_store(args, {"layers": [14], "template_sha": "x"})


def test_stage_texts_restages_on_source_content_change_resumes_on_match(tmp_path, monkeypatch):
    """(select-stage-texts-resume-content-unpinned) The staging resume
    fingerprint pins SOURCE CONTENT at blob grain: an identical snapshot
    resume-skips with zero re-stages; a regenerated source blob at the SAME
    path mismatches the sidecar and the stream re-stages, so texts_kept.jsonl
    carries the NEW content instead of certifying stale text end-to-end. Every
    stage call is pinned to the snapshot revision."""
    import json as _json

    pfx = XC.RAW_COMPLETIONS_PREFIX
    store = {
        f"{pfx}/shard00_chunk0000.json": {
            "rows": [{"ci": 7, "prompt": "p", "response": "OLD"}],
            "shard_index": 0,
            "chunk": 0,
        }
    }
    fake_stage, calls = _fake_stage_from(store)
    monkeypatch.setattr(XC.hub, "stage_hub_file", fake_stage)
    args = type("A", (), {"out_root": str(tmp_path), "hf_data_repo": "fake/repo"})()
    ci = np.asarray([7], dtype=np.int64)
    snap1 = _snap(store, blob_prefix="blobA")
    XC._stage_texts(args, ci, {7: "lmsys"}, {"drops": {}}, snap1)
    assert calls == [(f"{pfx}/shard00_chunk0000.json", "rev-blobA")]  # revision-pinned stage
    assert _json.loads((tmp_path / "texts_kept.jsonl").read_text())["response"] == "OLD"
    sidecar = _json.loads((tmp_path / "texts_processed.json").read_text())
    assert "source_content_sha256" in sidecar["fingerprint"]

    # identical snapshot -> resume-skip: no new stage calls, staged text unchanged
    XC._stage_texts(args, ci, {7: "lmsys"}, {"drops": {}}, snap1)
    assert len(calls) == 1
    assert _json.loads((tmp_path / "texts_kept.jsonl").read_text())["response"] == "OLD"

    # regenerated source content at the SAME path (new blob id) -> re-stage + fresh text
    store[f"{pfx}/shard00_chunk0000.json"]["rows"][0]["response"] = "NEW"
    snap2 = _snap(store, blob_prefix="blobB")
    XC._stage_texts(args, ci, {7: "lmsys"}, {"drops": {}}, snap2)
    assert len(calls) == 2
    assert _json.loads((tmp_path / "texts_kept.jsonl").read_text())["response"] == "NEW"


def test_entries_fingerprint_keys_on_blob_content_not_revision():
    """The fingerprint changes iff a file's blob identity changes — stable under
    order permutation (sorted) and independent of the repo revision (unrelated
    fleet commits must not restart the stream)."""
    a = [("p/x.json", "blob1", 10), ("p/y.json", "blob2", 20)]
    assert XC._entries_fingerprint(a) == XC._entries_fingerprint(list(reversed(a)))
    changed = [("p/x.json", "blob1-regen", 10), ("p/y.json", "blob2", 20)]
    assert XC._entries_fingerprint(a) != XC._entries_fingerprint(changed)
