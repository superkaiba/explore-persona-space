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
    """_gate_rows picks rows that SURVIVE tokenization, spanning both corpora."""
    tok = FakeTok()
    texts = [
        *_fake_rows(),
        {"ci": 13, "corpus": "wildchat", "prompt": "q4", "response": "a4"},
    ]
    rows, recs = XC._gate_rows(tok=tok, texts=texts, gen_suffix="<A>", max_tokens=60, n=4)
    assert len(rows) == len(recs) >= 2
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
