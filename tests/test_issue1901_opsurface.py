"""Pins for scripts/issue1901_opsurface.py (round ``opsurface-rebase``, task #1901).

1. Whitening + CSLS retrieval parity against an INDEPENDENT naive replication of the
   #2202 conventions (explicit top-k means, explicit mid-rank counting) — non-circular:
   the reference here is hand-rolled loops, not the imported helpers.
2. Draw-averaged target assembly (mean of original + K draws; covered rows only).
3. The resume-predicate regime key includes every surface/metric flag.
4. Batched-capture equivalence: ``_batched_forward_spans`` batched vs batch-1 on a
   REAL tiny from-config Qwen2 model (CPU fp32, no network) — the production-body test
   for the batched rewrite (#502/#779 duty).

No network, no GPU; repo-root paths only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1901_opsurface as OPS  # noqa: E402


def _naive_whiten(x: np.ndarray, mu: np.ndarray, ell: np.ndarray) -> np.ndarray:
    return np.linalg.solve(ell, (np.asarray(x, np.float64) - mu).T).T


def _naive_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return an @ bn.T


def _naive_csls(s: np.ndarray, k: int) -> np.ndarray:
    out = np.empty_like(s)
    rq = np.array([np.sort(s[i])[-k:].mean() for i in range(s.shape[0])])
    rp = np.array([np.sort(s[:, j])[-k:].mean() for j in range(s.shape[1])])
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            out[i, j] = 2.0 * s[i, j] - rq[i] - rp[j]
    return out


def _naive_midranks(d: np.ndarray, true_cols: np.ndarray) -> np.ndarray:
    ranks = np.empty(d.shape[0])
    for i in range(d.shape[0]):
        dt = d[i, true_cols[i]]
        tol = 1e-9 * max(abs(dt), 1e-12)
        closer = int((d[i] < dt - tol).sum())
        tied = int((np.abs(d[i] - dt) <= tol).sum()) - 1
        ranks[i] = 1.0 + closer + 0.5 * tied
    return ranks


def _fixture(seed: int = 0, n_pool: int = 40, n_cov: int = 12, h: int = 16):
    rng = np.random.default_rng(seed)
    pred = rng.normal(size=(n_cov, h))
    pool = rng.normal(size=(n_pool, h))
    pos = np.sort(rng.choice(n_pool, size=n_cov, replace=False))
    a = rng.normal(size=(h, h))
    cov = a @ a.T + h * np.eye(h)
    ell = np.linalg.cholesky(cov)
    mu = rng.normal(size=h)
    return pred, pool, pos, mu, ell


def test_whiten_csls_parity_vs_naive_reference():
    pred, pool, pos, mu, ell = _fixture()
    k = 3
    rec = OPS.score_opsurf(
        pred,
        pool,
        pool[pos].astype(np.float32),
        pos,
        mu,
        ell,
        n_boot=20,
        seed=0,
        csls_k=k,
        include_raw_euclidean=True,
    )
    zq = _naive_whiten(pred, mu, ell)
    zp = _naive_whiten(pool, mu, ell)
    s = _naive_cos(zq, zp)
    for name, dist in (
        ("whiten_cos", 1.0 - s),
        ("whiten_csls", -_naive_csls(s, k)),
    ):
        ranks = _naive_midranks(dist, pos)
        got = rec["retrieval"][name]
        for kk in (1, 5, 10):
            assert abs(got["acc_at_k"][kk] - float((ranks <= kk).mean())) < 1e-12, (name, kk)
        assert abs(got["median_rank"] - float(np.median(ranks))) < 1e-12, name
        assert abs(got["mrr"] - float((1.0 / ranks).mean())) < 1e-12, name
        assert got["n_pool"] == pool.shape[0]
        assert abs(got["chance_at_k"][1] - 1.0 / pool.shape[0]) < 1e-15
    # raw-euclidean leg (leg-0 reconciliation path): naive squared-distance ranks.
    d = np.sqrt(((pred[:, None, :] - pool[None, :, :]) ** 2).sum(-1))
    ranks_e = _naive_midranks(d**2, pos)  # monotone transform — same mid-ranks
    got_e = rec["retrieval"]["raw_euclidean"]
    assert abs(got_e["acc_at_k"][1] - float((ranks_e <= 1).mean())) < 1e-12


def test_score_opsurf_perfect_predictions_saturate():
    _pred, pool, pos, mu, ell = _fixture(seed=1)
    y_avg = pool[pos].astype(np.float32)
    rec = OPS.score_opsurf(pool[pos], pool, y_avg, pos, mu, ell, n_boot=10, seed=0, csls_k=3)
    assert rec["whole_map_r2"] > 0.999999
    assert rec["retrieval"]["whiten_cos"]["acc_at_k"][1] == 1.0
    assert rec["retrieval"]["whiten_csls"]["acc_at_k"][1] == 1.0


def test_draw_avg_assembly():
    y_pool = np.arange(18, dtype=np.float64).reshape(6, 3)
    pos = np.array([1, 4])
    draws = np.stack(
        [
            np.stack([y_pool[1] + 1.0, y_pool[1] + 2.0, y_pool[1] + 3.0, y_pool[1] + 4.0]),
            np.stack([y_pool[4] - 1.0, y_pool[4] - 2.0, y_pool[4] - 3.0, y_pool[4] - 4.0]),
        ]
    )
    pool_mod, y_avg = OPS.avg_pool_assembly(y_pool, draws, pos)
    # covered rows: mean(original + 4 draws) = original + mean(offsets)
    assert np.allclose(pool_mod[1], y_pool[1] + (1 + 2 + 3 + 4) / 5.0)
    assert np.allclose(pool_mod[4], y_pool[4] - (1 + 2 + 3 + 4) / 5.0)
    assert np.allclose(y_avg, pool_mod[pos])
    untouched = [i for i in range(6) if i not in pos]
    assert np.allclose(pool_mod[untouched], y_pool[untouched])


def _args(extra: list[str] | None = None):
    return OPS.build_argparser().parse_args(extra or [])


def test_regime_key_includes_surface_and_metric_flags():
    base = OPS.regime_key(_args())
    # surface/metric flags MUST move the key (a resume ignoring them reuses wrong rows).
    assert OPS.regime_key(_args(["--csls-k", "5"])) != base
    assert OPS.regime_key(_args(["--n-boot", "77"])) != base
    assert OPS.regime_key(_args(["--seed", "7"])) != base
    assert OPS.regime_key(_args(["--layers", "19"])) != base
    # cosmetic/output-path flags must NOT move it (resume survives relocation).
    assert OPS.regime_key(_args(["--out-eval", "/tmp/elsewhere"])) == base
    assert OPS.regime_key(_args(["--fig-dir", "/tmp/elsewhere"])) == base
    assert OPS.regime_key(_args(["--stage-root", "/tmp/elsewhere"])) == base
    # the pinned surface constants are present verbatim.
    assert base["n_pool"] == 9_941 and base["n_covered"] == 1_988 and base["k_draws"] == 4
    assert base["parent_pin"].startswith("09788eef")
    assert base["metric"] == "whiten_cos+csls"
    assert base["whiten_lambda"] == 0.1


def _tiny_qwen2():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def test_batched_forward_spans_matches_batch1_on_real_tiny_model():
    model = _tiny_qwen2()
    rng = np.random.default_rng(0)
    rows = []
    for n_tok, pl in ((9, 4), (17, 11), (5, 2), (23, 7)):
        ids = rng.integers(1, 127, size=n_tok).tolist()
        rows.append({"ids": ids, "prompt_len": pl})
    dev = torch.device("cpu")
    batched = OPS._batched_forward_spans(model, rows, [0, 1], dev, pad_id=0)
    for row, b in zip(rows, batched, strict=True):
        s = OPS._batched_forward_spans(model, [row], [0, 1], dev, pad_id=0)[0]
        assert torch.allclose(b["cx"], s["cx"], atol=1e-5), "cx batched != batch-1"
        assert torch.allclose(b["vx"], s["vx"], atol=1e-5), "vx batched != batch-1"
        # span semantics: vx is the mean over positions [prompt_len, n); cx is the
        # hidden state at prompt_len-1 — re-derive from a raw forward.
        cap = OPS.extract_layer_activations(model, torch.as_tensor(row["ids"])[None, :], [0, 1])
        pl, n = row["prompt_len"], len(row["ids"])
        want_cx = torch.stack([cap[li][0, pl - 1, :].float() for li in (0, 1)])
        want_vx = torch.stack([cap[li][0, pl:n, :].float().mean(dim=0) for li in (0, 1)])
        assert torch.allclose(s["cx"], want_cx, atol=1e-5)
        assert torch.allclose(s["vx"], want_vx, atol=1e-5)


def test_capture_batches_cover_all_rows_within_budget():
    items = [{"ids": list(range(n))} for n in (5, 30, 12, 7, 30, 18, 3)]
    batches = OPS._capture_batches(items, token_budget=60)
    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(len(items)))
    for b in batches:
        t_max = max(len(items[i]["ids"]) for i in b)
        assert len(b) * t_max <= 60 or len(b) == 1
