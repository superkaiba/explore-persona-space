#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
"""Issue #958 fits + transfer/forecast/prefix evaluations (dual/Gram GCV ridge).

Math (plan §4.6): per-row closed-form affine ridge, λ by GCV over
``RIDGE_LAMBDAS_922`` (Source #922), X standardized / Y centered on train-fold
stats. The DUAL/Gram-space path is used for uniformity (N ≤ 4,000 < d = 3,584):
one batched ``eigh`` of the stacked per-row N×N Grams — the 29-row STACKED-eigh
batching is NEW #958 code, gated by the registered ≤1e-6 batched-vs-serial
equivalence check (``--verify-fits`` toy dims + one REAL cell at run start;
the gate calls the exact production ``fit_rows_batched``). GCV identities in
the dual eigenbasis (g = QᵀYc, u = ‖g_j‖², cyy = Σu):

    df(λ)  = 1 + Σ_j s_j/(s_j+λ)
    SSE(λ) = Σ_j (λ/(s_j+λ))² u_j          (== the primal formula, exactly)
    GCV(λ) = (SSE/n) / (1 − df/n)²         (df ≥ n ⇒ +inf)

Transfer policy (registered): ``transfer_standardization_policy =
source-map-composite`` — transfers apply the SOURCE map's composite affine
verbatim (source train-fold X standardization + Y centering), NO target-turn
re-standardization; the policy string rides the transfer-matrix JSON. The
registered RECALIBRATED companion applies the source weights under the TARGET
turn's train-fold moments, decomposing Δ into moment-recalibration vs residual
map change (plan §6), plus per-turn moment-shift + support-overlap diagnostics.

Row-coverage: every registered arm's row keys are set-asserted against the
store shard keys BEFORE any skill is computed (cache build + per-eval masks).

Everything is batched over the 29-row axis (row-chunked fp64); no serial
per-cell factorization loop. Device: ``--device`` > ``EPM_FIT_DEVICE`` > auto.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

import issue958_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_922.maps922 import RIDGE_LAMBDAS_922  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue958_fit_maps")

ROW_CHUNK = 8
# frozen read-out rows (Source #922); resolved against the realized store R at
# main() start so the stub-dims VM smoke exercises the identical code path
READOUT_ROWS = [C.block_to_row(b) for b in C.READOUT_BLOCKS]
PERSIST_ROWS = [0, *READOUT_ROWS]  # emb + 6 read-out rows


def _clamp_rows(rows: list[int], n_rows: int) -> list[int]:
    """Clamp frozen row indices into the realized store (stub-dims smoke)."""
    return rows if max(rows) < n_rows else sorted({min(r, n_rows - 1) for r in rows})


# ── batched dual-GCV ridge core (NEW #958 code; equivalence-gated) ────────────


def fit_rows_batched(
    X: torch.Tensor, Y: torch.Tensor, *, lambdas: list[float], device: str
) -> dict:
    """Batched per-row dual/Gram GCV ridge over a (rows, n, d)/(rows, n, p) stack.

    X, Y fp16/fp32 (rows, n, d)/(rows, n, p); computes fp64 on ``device``.
    Returns fp64 CPU tensors: mu/sd (rows, d), ymu (rows, p), Xn (rows, n, d),
    alpha (rows, n, p) (dual coefficients at the per-row GCV-selected λ),
    best_lam (rows,), gcv_curve (rows, n_lambda). Predictions:
    ``pred = ymu + ((X_t − mu)/sd) Xnᵀ alpha``.
    """
    dev = torch.device(device)
    Xd = X.to(dev, torch.float64)
    Yd = Y.to(dev, torch.float64)
    _r, n, _d = Xd.shape
    mu = Xd.mean(1, keepdim=True)
    sd = Xd.std(1, correction=0, keepdim=True) + 1e-9  # the #658/#841 convention
    Xn = (Xd - mu) / sd
    ymu = Yd.mean(1, keepdim=True)
    Yc = Yd - ymu
    K = Xn @ Xn.transpose(1, 2)  # (r, n, n)
    s, Q = torch.linalg.eigh(K)  # batched over rows
    s = torch.clamp(s, min=0.0)
    G = Q.transpose(1, 2) @ Yc  # (r, n, p)
    u = (G * G).sum(-1)  # (r, n)
    lam_t = torch.tensor(lambdas, dtype=torch.float64, device=dev)  # (L,)
    f = 1.0 / (s.unsqueeze(-1) + lam_t)  # (r, n, L)
    df = 1.0 + (s.unsqueeze(-1) * f).sum(1)  # (r, L)
    sse = ((lam_t * f) ** 2 * u.unsqueeze(-1)).sum(1)  # (r, L)
    denom = 1.0 - df / n
    gcv = torch.where(denom > 0, (sse / n) / (denom * denom), torch.inf)
    best_i = torch.argmin(gcv, dim=1)  # (r,)
    best_lam = lam_t[best_i]
    f_best = 1.0 / (s + best_lam.unsqueeze(1))  # (r, n)
    alpha = Q @ (f_best.unsqueeze(-1) * G)  # (r, n, p)
    return {
        "mu": mu.squeeze(1).cpu(),
        "sd": sd.squeeze(1).cpu(),
        "ymu": ymu.squeeze(1).cpu(),
        "Xn": Xn.cpu(),
        "alpha": alpha.cpu(),
        "eig_s": s.cpu(),
        "eig_Q": Q.cpu(),
        "best_lam": best_lam.cpu(),
        "gcv_curve": gcv.cpu(),
    }


def predict_from_fit(
    fit: dict, X_t: torch.Tensor, *, device: str, moments: dict | None = None
) -> torch.Tensor:
    """Apply the fitted maps to targets (rows, n_t, d) → fp64 (rows, n_t, p).

    ``moments=None`` = the registered source-map-composite policy (source
    train-fold mu/sd/ymu baked in). Passing ``moments={"mu","sd","ymu"}``
    (target-turn train-fold stats) = the registered RECALIBRATED companion.
    """
    dev = torch.device(device)
    m = moments if moments is not None else fit
    mu = m["mu"].to(dev).unsqueeze(1)
    sd = m["sd"].to(dev).unsqueeze(1)
    ymu = m["ymu"].to(dev).unsqueeze(1)
    Xtn = (X_t.to(dev, torch.float64) - mu) / sd
    K_cross = Xtn @ fit["Xn"].to(dev).transpose(1, 2)  # (r, n_t, n)
    return (ymu + K_cross @ fit["alpha"].to(dev)).cpu()


def _fit_rows_serial_reference(X, Y, *, lambdas, rows: list[int], at_lam: dict | None) -> dict:
    """SERIAL per-row reference (plain loop, per-row eigh) for the ≤1e-6 gate.

    Contained reference only (never dispatched for production; the
    vectorize-first Supersede contract's containment criterion). ``at_lam``
    (row → λ) pins the prediction λ to a COMMON value with the batched path —
    GCV argmin is ill-posed under fp reduction-order jitter wherever the
    curve is flat (near-interpolation small-λ region), so equivalence is
    asserted on the GCV CURVE values + the common-λ predictions, never on
    argmin identity.
    """
    out = {"gcv_curve": {}, "pred_fn": {}}
    for r in rows:
        Xd = X[r].to(torch.float64)
        Yd = Y[r].to(torch.float64)
        n = Xd.shape[0]
        mu = Xd.mean(0)
        sd = Xd.std(0, correction=0) + 1e-9
        Xn = (Xd - mu) / sd
        ymu = Yd.mean(0)
        Yc = Yd - ymu
        s, Q = torch.linalg.eigh(Xn @ Xn.t())
        s = torch.clamp(s, min=0.0)
        G = Q.t() @ Yc
        u = (G * G).sum(-1)
        curve = []
        for lam in lambdas:
            f = 1.0 / (s + lam)
            df = 1.0 + float((s * f).sum())
            sse = float((((lam * f) ** 2) * u).sum())
            den = 1.0 - df / n
            curve.append(float("inf") if den <= 0 else (sse / n) / (den * den))
        lam_use = at_lam[r] if at_lam is not None else lambdas[int(np.argmin(curve))]
        f = 1.0 / (s + lam_use)
        alpha = Q @ (f.unsqueeze(-1) * G)

        def _pred(X_t, mu=mu, sd=sd, ymu=ymu, Xn=Xn, alpha=alpha):
            Xtn = (X_t.to(torch.float64) - mu) / sd
            return ymu + (Xtn @ Xn.t()) @ alpha

        out["gcv_curve"][r] = curve
        out["pred_fn"][r] = _pred
    return out


def equivalence_gate_fits(X, Y, X_t, *, lambdas, device: str, tol: float = 1e-6) -> dict:
    """Registered gate: batched stacked-eigh path vs the serial per-row reference.

    Calls the EXACT production ``fit_rows_batched``/``predict_from_fit`` (the
    live dispatched path — no sibling helper). Asserts (a) the GCV curves
    agree to ≤1e-8 relative at every finite λ (both-inf entries count as
    equal — the identical-math check), and (b) max|Δpred| ≤ tol (default the
    registered 1e-6) with the serial reference evaluated at the SAME λ the
    batched path selected (argmin identity is NOT asserted: the curve is
    provably flat/degenerate in the small-λ near-interpolation region and a
    reduction-order fp jitter flips near-ties without changing the map).
    """
    rows = list(range(min(2, X.shape[0])))
    fit = fit_rows_batched(X[rows], Y[rows], lambdas=lambdas, device=device)
    pred_b = predict_from_fit(fit, X_t[rows], device=device)
    at_lam = {r: float(fit["best_lam"][i]) for i, r in enumerate(rows)}
    ref = _fit_rows_serial_reference(X, Y, lambdas=lambdas, rows=rows, at_lam=at_lam)
    max_abs, max_gcv_rel = 0.0, 0.0
    for i, r in enumerate(rows):
        gb = fit["gcv_curve"][i].numpy()
        gs = np.asarray(ref["gcv_curve"][r])
        both_inf = np.isinf(gb) & np.isinf(gs)
        assert bool(np.all(np.isinf(gb) == np.isinf(gs))), "GCV inf-mask mismatch"
        fin = ~both_inf
        if fin.any():
            rel = np.max(np.abs(gb[fin] - gs[fin]) / (np.abs(gs[fin]) + 1e-300))
            max_gcv_rel = max(max_gcv_rel, float(rel))
        d = (pred_b[i] - ref["pred_fn"][r](X_t[r])).abs().max().item()
        max_abs = max(max_abs, float(d))
    assert max_gcv_rel <= 1e-8, f"GCV curve drift: rel {max_gcv_rel:.3e} > 1e-8"
    assert max_abs <= tol, f"batched-vs-serial ridge parity FAILED: {max_abs:.3e} > {tol}"
    logger.info(
        "[gate] batched-vs-serial dual-GCV ridge PASS (max|Δpred|=%.2e, gcv_rel=%.2e)",
        max_abs,
        max_gcv_rel,
    )
    return {"max_abs_delta": max_abs, "max_gcv_rel": max_gcv_rel, "tol": tol, "rows": rows}


# ── columnar fit cache (one pass over store shards) ───────────────────────────

CACHE_POS = {"prefix": C.POS_PREFIX_END, "ctx": C.POS_CTX_END, "ans": C.POS_ANS_MEAN}


def cache_path(cache_dir: Path, unit_set: str, k: int, pos: str) -> Path:
    """Canonical columnar cache path."""
    return cache_dir / f"{unit_set}_k{k}_{pos}.pt"


def _load_capture_dropped(store_dir: Path) -> dict[str, set[str]]:
    """Per-set dropped uids recorded by capture (empty-generation dropout, r2)."""
    p = store_dir / "capture_summary.json"
    assert p.exists(), f"store at {store_dir} has no capture_summary.json — capture incomplete"
    blob = json.loads(p.read_text())
    return {s: set(v) for s, v in (blob.get("dropped_uids") or {}).items()}


def build_fit_cache(store_dir: Path, corpus_dir: Path, cache_dir: Path, fingerprint: str) -> dict:
    """One shard pass per unit set → per-(set, k, pos) (N, R, H) fp16 tensors.

    ALSO the row-coverage set-assert (plan §6): every enumerated unit id must
    be present in the store shard keys UNLESS recorded as an empty-generation
    dropout (then the CONVERSATION is excluded from the paired design via the
    per-set ``invalid_ci`` list — dropped units are never silently read as
    zero rows). Store shards must carry the consumed corpus's fingerprint
    (fail loud on a stale store, r2). graft/onpol are consumed uid-keyed via
    ``load_store_positions`` — no dense cache (the r1 dead-cache fix).
    Returns the coverage/meta dict (token counts per (set, k) ride along).
    """
    units_all = C.enumerate_units(corpus_dir)
    dropped_by_set = _load_capture_dropped(store_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {"sets": {}, "corpus_fingerprint": fingerprint}
    for unit_set, units in units_all.items():
        idx = C.load_store_index(store_dir, unit_set, expect_fingerprint=fingerprint)
        dropped = dropped_by_set.get(unit_set, set())
        missing = [u["uid"] for u in units if u["uid"] not in idx]
        bad = [m for m in missing if m not in dropped]
        assert not bad, (
            f"ROW-COVERAGE FAIL: {unit_set} store missing {len(bad)} registered units "
            f"NOT recorded as dropped, e.g. {bad[:3]}"
        )
        if unit_set in ("graft", "onpol"):  # sparse — consumed uid-keyed, no dense cache
            meta["sets"][unit_set] = {
                "n_units": len(units) - len(missing),
                "n_dropped": len(missing),
            }
            continue
        by_k: dict[int, list[dict]] = {}
        for u in units:
            by_k.setdefault(u["k"], []).append(u)
        n_conv = max(u["ci"] for u in units) + 1
        # a conversation with ANY dropped turn leaves the paired design entirely
        invalid_ci = sorted({u["ci"] for u in units if u["uid"] not in idx})
        R = H = None
        buf: dict[tuple[int, str], torch.Tensor] = {}
        toks: dict[int, dict[str, np.ndarray]] = {}
        done = False
        for p in sorted((store_dir / unit_set).glob("shard_*.pt")):
            blob = torch.load(p, weights_only=False, map_location="cpu")
            assert blob.get("corpus_fingerprint") == fingerprint, (
                f"STORE FINGERPRINT MISMATCH at {p}: "
                f"{str(blob.get('corpus_fingerprint'))[:12]} vs {fingerprint[:12]}"
            )
            for uid, rec in blob["units"].items():
                parts = uid.split(":")
                ci, k = int(parts[1][1:]), int(parts[2][1:])
                h = rec["h"]
                if R is None:
                    R, H = int(h.shape[1]), int(h.shape[2])
                    for kk in by_k:
                        for pos in CACHE_POS:
                            buf[(kk, pos)] = torch.zeros((n_conv, R, H), dtype=torch.float16)
                        toks[kk] = {
                            "prefix_tokens": np.zeros(n_conv, dtype=np.int64),
                            "query_tokens": np.zeros(n_conv, dtype=np.int64),
                            "ans_tokens": np.zeros(n_conv, dtype=np.int64),
                        }
                for pos, pi in CACHE_POS.items():
                    buf[(k, pos)][ci] = h[pi]
                for f in toks[k]:
                    toks[k][f][ci] = rec[f]
                done = True
            del blob
        assert done, f"no shards under {store_dir / unit_set}"
        for (k, pos), t in buf.items():
            torch.save(
                {"x": t, "toks": {f: v for f, v in toks[k].items()}},
                cache_path(cache_dir, unit_set, k, pos),
            )
        meta["sets"][unit_set] = {
            "n_conv": n_conv,
            "R": R,
            "H": H,
            "ks": sorted(by_k),
            "invalid_ci": invalid_ci,
            "n_dropped_units": len(missing),
        }
        logger.info(
            "[cache] %s: %d convs x %d turns -> %d tensors (%d invalid convs)",
            unit_set,
            n_conv,
            len(by_k),
            len(buf),
            len(invalid_ci),
        )
    C.write_json_atomic(cache_dir / "cache_meta.json", meta)
    return meta


class Cache:
    """Small LRU over the columnar cache tensors (bounds resident fp16)."""

    def __init__(self, cache_dir: Path, max_entries: int = 6):
        self.dir = cache_dir
        self.max = max_entries
        self._d: dict[tuple, dict] = {}

    def get(self, unit_set: str, k: int, pos: str) -> dict:
        key = (unit_set, k, pos)
        if key not in self._d:
            if len(self._d) >= self.max:
                self._d.pop(next(iter(self._d)))
            self._d[key] = torch.load(
                cache_path(self.dir, unit_set, k, pos), weights_only=False, map_location="cpu"
            )
        return self._d[key]

    def x(self, unit_set: str, k: int, pos: str, conv_idx: np.ndarray) -> torch.Tensor:
        """(n_sel, R, H) fp16 rows for the given conversation indices."""
        blob = self.get(unit_set, k, pos)
        return blob["x"][torch.from_numpy(np.asarray(conv_idx, dtype=np.int64))]


# ── design enumeration (fit cells + eval cells; plan §4.6 / §5) ───────────────


def _filter_invalid(idx, invalid: frozenset[int]) -> np.ndarray:
    """Sorted fold indices with dropout-invalid conversations removed (r2)."""
    a = np.sort(np.asarray(idx, dtype=np.int64))
    if not invalid:
        return a
    return a[~np.isin(a, np.fromiter(invalid, dtype=np.int64))]


def build_design(
    n_main: int,
    n_long: int,
    invalid_main: frozenset[int] = frozenset(),
    invalid_long: frozenset[int] = frozenset(),
) -> dict:
    """Deterministic splits, twin halves, fit-cell + eval-cell registries.

    ``invalid_*`` conversations (any empty-generation-dropped turn, r2) are
    filtered OUT of every fold AFTER the deterministic split, so the split
    stays reproducible from (N, seed) and every paired arm reads the identical
    filtered conversation set.
    """
    split_m = C.make_split(n_main, n_fit=C.N_FIT, n_val=C.N_VAL, n_test=C.N_TEST, seed=C.SPLIT_SEED)
    split_l = C.make_split(
        n_long, n_fit=C.LONG_FIT, n_val=C.LONG_VAL, n_test=C.LONG_TEST, seed=C.SPLIT_SEED
    )
    fit_m = _filter_invalid(split_m["fit"], invalid_main)
    test_m = _filter_invalid(split_m["test"], invalid_main)
    fit_l = _filter_invalid(split_l["fit"], invalid_long)
    test_l = _filter_invalid(split_l["test"], invalid_long)
    half_a, half_b = C.twin_halves(fit_m)
    fits: dict[str, dict] = {}
    for k in range(1, C.K_MAIN + 1):
        fits[f"ctx_k{k}_A"] = dict(set="main", x_k=k, x_pos="ctx", y_k=k, idx=half_a)
        fits[f"ctx_k{k}_B"] = dict(set="main", x_k=k, x_pos="ctx", y_k=k, idx=half_b)
        fits[f"ctx_k{k}_full"] = dict(set="main", x_k=k, x_pos="ctx", y_k=k, idx=fit_m)
    for k in range(2, C.K_MAIN + 1):
        fits[f"pre_k{k}_full"] = dict(set="main", x_k=k, x_pos="prefix", y_k=k, idx=fit_m)
        fits[f"pre_k{k}_A"] = dict(set="main", x_k=k, x_pos="prefix", y_k=k, idx=half_a)
        fits[f"pre_k{k}_B"] = dict(set="main", x_k=k, x_pos="prefix", y_k=k, idx=half_b)
    for j in range(1, C.K_MAIN + 1):
        for k in range(j + 1, C.K_MAIN + 1):
            fits[f"fcast_{j}to{k}"] = dict(set="main", x_k=j, x_pos="ctx", y_k=k, idx=fit_m)
    for k in [*range(5, C.K_LONG + 1), 1]:
        fits[f"long_k{k}_own"] = dict(set="long", x_k=k, x_pos="ctx", y_k=k, idx=fit_l)
    # eval registry: eval_id -> (map fit-cell | 'copyprev', target set/k/pos, target idx,
    #                            null-mean source fit-cell, optional recalibration cell)
    evals: dict[str, dict] = {}
    for j in range(1, C.K_MAIN + 1):
        for k in range(1, C.K_MAIN + 1):
            evals[f"xfer_{j}to{k}_A"] = dict(
                map=f"ctx_k{j}_A", set="main", k=k, pos="ctx", idx=test_m, null=f"ctx_k{k}_A"
            )
            if j != k:
                evals[f"recal_{j}to{k}_A"] = dict(
                    map=f"ctx_k{j}_A",
                    set="main",
                    k=k,
                    pos="ctx",
                    idx=test_m,
                    null=f"ctx_k{k}_A",
                    recal=f"ctx_k{k}_A",
                )
    for k in range(1, C.K_MAIN + 1):
        evals[f"own_k{k}_B"] = dict(
            map=f"ctx_k{k}_B", set="main", k=k, pos="ctx", idx=test_m, null=f"ctx_k{k}_B"
        )
        evals[f"own_k{k}_full"] = dict(
            map=f"ctx_k{k}_full", set="main", k=k, pos="ctx", idx=test_m, null=f"ctx_k{k}_full"
        )
        evals[f"panelxfer_k{k}"] = dict(
            map=f"ctx_k{k}_full", set="long", k=k, pos="ctx", idx=test_l, null=f"ctx_k{k}_full"
        )
    for j in range(1, C.K_MAIN + 1):
        for k in range(j + 1, C.K_MAIN + 1):
            evals[f"fcast_{j}to{k}"] = dict(
                map=f"fcast_{j}to{k}",
                set="main",
                k=k,
                pos="ctx",
                x_k=j,
                idx=test_m,
                null=f"ctx_k{k}_full",
            )
    for k in range(2, C.K_MAIN + 1):
        evals[f"pre_k{k}_full"] = dict(
            map=f"pre_k{k}_full",
            set="main",
            k=k,
            pos="prefix",
            idx=test_m,
            null=f"ctx_k{k}_full",
        )
        evals[f"pre_k{k}_A"] = dict(
            map=f"pre_k{k}_A", set="main", k=k, pos="prefix", idx=test_m, null=f"ctx_k{k}_A"
        )
        evals[f"copyprev_k{k}"] = dict(
            map="copyprev",
            set="main",
            k=k,
            pos="ans",
            x_k=k - 1,
            idx=test_m,
            null=f"ctx_k{k}_full",
        )
    for k in [*range(5, C.K_LONG + 1), 1]:
        evals[f"long_own_k{k}"] = dict(
            map=f"long_k{k}_own", set="long", k=k, pos="ctx", idx=test_l, null=f"long_k{k}_own"
        )
    evals["onpol_k2"] = dict(
        map="ctx_k2_full", set="onpol", k=2, pos="ctx", idx=None, null="ctx_k2_full"
    )
    return {
        "split_main": split_m,
        "split_long": split_l,
        "half_a": half_a,
        "half_b": half_b,
        "fits": fits,
        "evals": evals,
        "test_main": test_m,
        "test_long": test_l,
    }


# ── skill + per-cell artifacts ────────────────────────────────────────────────


def _skill_and_stats(pred: torch.Tensor, Y_t: torch.Tensor, null_mean: torch.Tensor) -> dict:
    """Per-row skill + per-unit SSE arrays + shuffle sufficient statistics.

    pred/Y_t fp64 (rows, n_t, p); null_mean (rows, p). skill = 1 −
    SSE(pred)/SSE(null). Shuffle-pairing draws are computed by the CALLER from
    the returned dot matrix D = pred·Yᵀ (one batched re-reduction; plan §6).
    """
    Yd = Y_t.to(torch.float64)
    err = pred - Yd
    sse_u = (err * err).sum(-1)  # (rows, n_t)
    null_err = Yd - null_mean.unsqueeze(1)
    null_u = (null_err * null_err).sum(-1)
    skill = 1.0 - sse_u.sum(1) / null_u.sum(1).clamp(min=1e-30)
    D = pred @ Yd.transpose(1, 2)  # (rows, n_t, n_t)
    return {
        "skill": skill,
        "sse_unit": sse_u,
        "null_sse_unit": null_u,
        "pred_sq": (pred * pred).sum(-1),
        "y_sq": (Yd * Yd).sum(-1),
        "D": D,
    }


def _shuffle_draws(stats: dict, n_draws: int, seed: int) -> torch.Tensor:
    """(n_draws, rows) shuffled-pairing skills — ONE batched gather per draw set."""
    _rows, n_t = stats["sse_unit"].shape
    rng = np.random.default_rng(seed)
    perms = torch.from_numpy(
        np.stack([rng.permutation(n_t) for _ in range(n_draws)])
    )  # (n_draws, n_t)
    D = stats["D"]  # (rows, n_t, n_t) — D[r, i, j] = pred_i · y_j
    # SSE_perm[d, r] = Σ_i (‖pred‖²_{π_d(i)} + ‖y‖²_i − 2 D[r, π_d(i), i])
    pred_sq_tot = stats["pred_sq"].sum(1)  # (rows,)
    y_sq_tot = stats["y_sq"].sum(1)
    cross = torch.stack(
        [D[:, perms[d], torch.arange(n_t)].sum(1) for d in range(n_draws)]
    )  # (n_draws, rows)
    sse_perm = pred_sq_tot.unsqueeze(0) + y_sq_tot.unsqueeze(0) - 2.0 * cross
    null_tot = stats["null_sse_unit"].sum(1).clamp(min=1e-30)
    return 1.0 - sse_perm / null_tot.unsqueeze(0)


def _readout_mean(skill: torch.Tensor) -> float:
    """Pre-registered headline aggregation: mean over the frozen 6 read-out rows."""
    return float(skill[_clamp_rows(READOUT_ROWS, int(skill.shape[0]))].mean())


def _load_rb(rb_dir: Path | None, stub_dims: tuple[int, int] | None) -> dict[str, np.ndarray]:
    """#778 r_B tensors (28, 3584) fp32, shape-asserted; stub dims for VM smoke."""
    if stub_dims is not None:
        rng = np.random.default_rng(778)
        n_layers, hidden = stub_dims
        return {t: rng.standard_normal((n_layers, hidden)).astype(np.float32) for t in C.TRAITS}
    from huggingface_hub import hf_hub_download

    out = {}
    for trait in C.TRAITS:
        rel = f"issue778_persona_vectors/analysis_tensors/rb/{trait}.pt"
        local = rb_dir / f"{trait}.pt" if rb_dir else None
        if local is not None and local.exists():
            p = local
        else:
            p = Path(hf_hub_download(repo_id=C.HF_DATA_REPO, filename=rel, repo_type="dataset"))
        blob = torch.load(p, weights_only=False, map_location="cpu")
        rb = blob["r_b"] if isinstance(blob, dict) and "r_b" in blob else blob
        rb = rb.to(torch.float32).numpy()
        assert rb.shape == (C.EXPECTED_LAYERS, C.EXPECTED_HIDDEN), (trait, rb.shape)
        out[trait] = rb
    return out


def _directions(rb: dict[str, np.ndarray], hidden: int) -> tuple[dict[str, int], torch.Tensor]:
    """Unit-normalized trait dirs at frozen ℓ* + 100 norm-matched random dirs.

    Returns ({trait: dir column}, (n_traits + 100, hidden) fp64) — random dirs
    are unit-norm (norm-matched to the normalized trait dirs), seed 0 (#778
    battery convention).
    """
    cols = []
    col_of = {}
    for trait in C.TRAITS:
        lstar = C.PRIMARY_LSTAR[trait]
        n_layers = rb[trait].shape[0]
        v = rb[trait][min(lstar, n_layers - 1)]
        cols.append(v / (np.linalg.norm(v) + 1e-12))
        col_of[trait] = len(cols) - 1
    rng = np.random.default_rng(C.RANDDIR_SEED)
    for _ in range(C.RANDDIR_DRAWS):
        v = rng.standard_normal(hidden)
        cols.append(v / (np.linalg.norm(v) + 1e-12))
    return col_of, torch.from_numpy(np.stack(cols)).to(torch.float64)


# ── main driver ───────────────────────────────────────────────────────────────


def run_verify_fits(device: str) -> dict:
    """Toy-dims gate for the batched dual-GCV path (dispatch phase 0)."""
    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.standard_normal((3, 60, 24))).to(torch.float32)
    W = torch.from_numpy(rng.standard_normal((3, 24, 6))).to(torch.float32)
    Y = X @ W * 0.1 + torch.from_numpy(rng.standard_normal((3, 60, 6))).to(torch.float32) * 0.05
    X_t = torch.from_numpy(rng.standard_normal((3, 20, 24))).to(torch.float32)
    return equivalence_gate_fits(X, Y, X_t, lambdas=RIDGE_LAMBDAS_922, device=device)


def main() -> int:  # noqa: C901 — the fit/eval-cell sequence IS the plan §4.6/§5 spec
    ap = argparse.ArgumentParser(description="Issue #958 map fits + transfer evaluations.")
    ap.add_argument("--corpus", type=Path, default=Path("data/issue_958/corpus"))
    ap.add_argument("--store", type=Path, default=Path("data/issue_958/store"))
    ap.add_argument("--cache", type=Path, default=Path("data/issue_958/fit_cache"))
    ap.add_argument("--maps", type=Path, default=Path("data/issue_958/maps"))
    ap.add_argument("--out", type=Path, default=Path("eval_results/issue_958"))
    ap.add_argument("--device", default=None, help="--device > EPM_FIT_DEVICE > auto")
    ap.add_argument("--verify-fits", action="store_true", help="toy equivalence gate only")
    ap.add_argument("--rb-dir", type=Path, default=None, help="local r_B override dir")
    ap.add_argument("--stub-rb", action="store_true", help="VM smoke: random stub r_B")
    args = ap.parse_args()
    device = C.resolve_device(args.device)
    if args.verify_fits:
        run_verify_fits(device)
        return 0

    t0 = time.time()
    corpus_fp = C.corpus_fingerprint(args.corpus)
    meta = build_fit_cache(args.store, args.corpus, args.cache, corpus_fp)  # + row-coverage assert
    n_main = meta["sets"]["main"]["n_conv"]
    n_long = meta["sets"]["long"]["n_conv"]
    R, H = meta["sets"]["main"]["R"], meta["sets"]["main"]["H"]
    design = build_design(
        n_main,
        n_long,
        invalid_main=frozenset(meta["sets"]["main"].get("invalid_ci", [])),
        invalid_long=frozenset(meta["sets"]["long"].get("invalid_ci", [])),
    )
    cache = Cache(args.cache)
    rb = _load_rb(args.rb_dir, (R - 1, H) if args.stub_rb else None)
    _trait_cols, dirs = _directions(rb, H)
    trait_rows = {t: C.block_to_row(min(C.PRIMARY_LSTAR[t], R - 2)) for t in C.TRAITS}

    # ── fit-resume manifest (r2: the ~3h loop restarts from completed cells) ──
    # keyed on EVERY output-affecting regime key (#722-r3 rule): corpus
    # fingerprint, λ grid, policy, read-out rows, split spec, stub_rb, dims.
    fit_regime = {
        "corpus_fingerprint": corpus_fp,
        "lambdas": [float(x) for x in RIDGE_LAMBDAS_922],
        "policy": C.TRANSFER_STANDARDIZATION_POLICY,
        "readout_blocks": list(C.READOUT_BLOCKS),
        "R": R,
        "H": H,
        "n_main": n_main,
        "n_long": n_long,
        "stub_rb": bool(args.stub_rb),
        "split": {
            "seed": C.SPLIT_SEED,
            "sizes": [C.N_FIT, C.N_VAL, C.N_TEST, C.LONG_FIT, C.LONG_VAL, C.LONG_TEST],
            "twin_seed": C.TWIN_SEED,
        },
    }
    args.maps.mkdir(parents=True, exist_ok=True)
    manifest_path = args.maps / "fit_manifest.json"
    manifest: dict = {"regime": fit_regime, "cells": {}}
    if manifest_path.exists():
        try:
            prior = json.loads(manifest_path.read_text())
        except Exception:
            prior = None
        if prior and prior.get("regime") == fit_regime:
            manifest = prior
        else:
            logger.warning("[resume] fit_manifest regime differs — refitting all cells")

    def _mark_done(cell_key: str) -> None:
        manifest["cells"][cell_key] = {"done": True, "ts": time.time()}
        C.write_json_atomic(manifest_path, manifest)

    support_sidecar = args.maps / "support_overlap_turn1A.json"

    def _cell_complete(fid: str, eval_ids: list[str]) -> bool:
        """Resume predicate: manifest-done under the CURRENT regime + artifacts."""
        if not manifest["cells"].get(fid, {}).get("done"):
            return False
        if not (args.maps / f"{fid}.pt").exists():
            return False
        for eid in eval_ids:
            if not (args.out / "percell" / f"{eid}.npz").exists():
                return False
        return fid != "ctx_k1_A" or support_sidecar.exists()

    # real-cell equivalence gate on the FIRST fit cell (registered ≤1e-6 check)
    fits_reg = design["fits"]
    first = fits_reg["ctx_k1_A"]
    Xg = cache.x("main", first["x_k"], first["x_pos"], first["idx"]).transpose(0, 1)
    Yg = cache.x("main", first["y_k"], "ans", first["idx"]).transpose(0, 1)
    Xtg = cache.x("main", 1, "ctx", design["test_main"]).transpose(0, 1)
    gate_real = equivalence_gate_fits(Xg, Yg, Xtg, lambdas=RIDGE_LAMBDAS_922, device=device)
    del Xg, Yg, Xtg

    (args.out / "percell").mkdir(parents=True, exist_ok=True)

    # group eval cells by source map so each fit is built ONCE then applied
    evals_by_map: dict[str, list[str]] = {}
    for eid, e in design["evals"].items():
        evals_by_map.setdefault(e["map"], []).append(eid)

    maps_meta: dict[str, dict] = {}
    cell_results: dict[str, dict] = {}
    support_overlap: dict[str, dict] = {}

    # onpol units captured in the store (empty-generation dropouts excluded
    # with a recorded count — validated ⊆ dropped in build_fit_cache)
    onpol_spec = json.loads((args.corpus / "onpol_spec.json").read_text())
    onpol_idx_store = C.load_store_index(args.store, "onpol", expect_fingerprint=corpus_fp)
    onpol_cis = [
        ci for ci in onpol_spec["conv_indices"] if C.unit_id("onpol", ci, 2) in onpol_idx_store
    ]
    onpol_uids = [C.unit_id("onpol", ci, 2) for ci in onpol_cis]
    if len(onpol_cis) < len(onpol_spec["conv_indices"]):
        logger.warning(
            "[onpol] %d/%d spec units dropped (empty generations) — evaluating on the rest",
            len(onpol_spec["conv_indices"]) - len(onpol_cis),
            len(onpol_spec["conv_indices"]),
        )

    def _target_X(e: dict) -> tuple[torch.Tensor, np.ndarray]:
        if e["set"] == "onpol":
            x = C.load_store_positions(
                args.store,
                "onpol",
                onpol_uids,
                [CACHE_POS[e["pos"]]],
                expect_fingerprint=corpus_fp,
            )
            return x[:, 0].transpose(0, 1), np.array(onpol_cis)
        idx = e["idx"]
        x_k = e.get("x_k", e["k"])
        return cache.x(e["set"], x_k, e["pos"], idx).transpose(0, 1), idx

    def _target_Y(e: dict) -> torch.Tensor:
        if e["set"] == "onpol":
            y = C.load_store_positions(
                args.store, "onpol", onpol_uids, [C.POS_ANS_MEAN], expect_fingerprint=corpus_fp
            )
            return y[:, 0].transpose(0, 1)
        return cache.x(e["set"], e["k"], "ans", e["idx"]).transpose(0, 1)

    _null_memo: dict[str, torch.Tensor] = {}

    def _null_mean(fit_cell: str) -> torch.Tensor:
        """Target-cell train-fold Y mean (R, H) fp64 — the corpus-mean null."""
        if fit_cell not in _null_memo:
            f = fits_reg[fit_cell]
            y = cache.x(f["set"], f["y_k"], "ans", f["idx"]).to(torch.float64)
            _null_memo[fit_cell] = y.mean(0)  # (R, H)
        return _null_memo[fit_cell]

    # registered drift cells (plan §6): stale 1→k residuals + own-turn residuals
    drift_cells = {f"xfer_1to{k}_A" for k in range(2, C.K_MAIN + 1)} | {
        f"xfer_{k}to{k}_A" for k in range(1, C.K_MAIN + 1)
    }

    def _restore_cell(fid: str, eval_ids: list[str]) -> None:
        """Rebuild the in-memory readouts for a resume-skipped cell from disk."""
        blob = torch.load(args.maps / f"{fid}.pt", weights_only=False, map_location="cpu")
        maps_meta[fid] = blob["meta"]
        del blob
        for eid in eval_ids:
            z = np.load(args.out / "percell" / f"{eid}.npz")
            e = design["evals"][eid]
            cell_results[eid] = {
                "skill_readout_mean": _readout_mean(torch.from_numpy(z["skill"])),
                "map": e["map"],
                "target": {"set": e["set"], "k": e["k"], "pos": e["pos"]},
                **(
                    {"recal_skill_readout_mean": _readout_mean(torch.from_numpy(z["recal_skill"]))}
                    if "recal_skill" in z
                    else {}
                ),
            }
        if fid == "ctx_k1_A":
            support_overlap.update(json.loads(support_sidecar.read_text()))

    all_fit_ids = list(fits_reg)
    for fid in all_fit_ids:
        f = fits_reg[fid]
        eval_ids = evals_by_map.get(fid, [])
        if _cell_complete(fid, eval_ids):
            _restore_cell(fid, eval_ids)
            logger.info("[fit %s] complete under current regime — skip (resume)", fid)
            continue
        X_all = cache.x(f["set"], f["x_k"], f["x_pos"], f["idx"])  # (n, R, H)
        Y_all = cache.x(f["set"], f["y_k"], "ans", f["idx"])
        n_fit = X_all.shape[0]
        maps_meta[fid] = {
            "n_fit": int(n_fit),
            "x_pos": f["x_pos"],
            "x_k": f["x_k"],
            "y_k": f["y_k"],
            "set": f["set"],
            "best_lam": {},
        }
        # preload target tensors once per eval cell (fp16, thin)
        targets = {}
        for eid in eval_ids:
            e = design["evals"][eid]
            X_t, t_idx = _target_X(e)
            targets[eid] = {"X_t": X_t, "Y_t": _target_Y(e), "idx": t_idx, "e": e}
        w_out: dict[int, dict] = {}
        acc: dict[str, dict] = {
            eid: {"skill": [], "sse": [], "null": [], "shuf": [], "recal_skill": [], "proj": {}}
            for eid in eval_ids
        }
        lam_all = torch.zeros(R, dtype=torch.float64)
        for lo in range(0, R, ROW_CHUNK):
            rows = list(range(lo, min(lo + ROW_CHUNK, R)))
            fit = fit_rows_batched(
                X_all[:, rows].transpose(0, 1),
                Y_all[:, rows].transpose(0, 1),
                lambdas=RIDGE_LAMBDAS_922,
                device=device,
            )
            lam_all[rows] = fit["best_lam"]
            for ri, r in enumerate(rows):
                if r in _clamp_rows(PERSIST_ROWS, R):
                    Wr = fit["Xn"][ri].transpose(0, 1) @ fit["alpha"][ri]  # (H, H)
                    w_out[r] = {
                        "w": Wr.to(torch.float16),
                        "mu": fit["mu"][ri].to(torch.float32),
                        "sd": fit["sd"][ri].to(torch.float32),
                        "ymu": fit["ymu"][ri].to(torch.float32),
                        "best_lam": float(fit["best_lam"][ri]),
                    }
            for eid in eval_ids:
                tg = targets[eid]
                e = tg["e"]
                pred = predict_from_fit(fit, tg["X_t"][rows], device=device)
                Y_t = tg["Y_t"][rows].to(torch.float64)
                nm = _null_mean(e["null"])[rows]
                st = _skill_and_stats(pred, Y_t, nm)
                acc[eid]["skill"].append(st["skill"])
                acc[eid]["sse"].append(st["sse_unit"])
                acc[eid]["null"].append(st["null_sse_unit"])
                acc[eid]["shuf"].append(_shuffle_draws(st, C.SHUFFLE_DRAWS, C.SHUFFLE_SEED))
                if "recal" in e:
                    rc = fits_reg[e["recal"]]
                    Xr = (
                        cache.x(rc["set"], rc["x_k"], rc["x_pos"], rc["idx"])[:, rows]
                        .to(torch.float64)
                        .transpose(0, 1)
                    )
                    Yr = (
                        cache.x(rc["set"], rc["y_k"], "ans", rc["idx"])[:, rows]
                        .to(torch.float64)
                        .transpose(0, 1)
                    )
                    mom = {
                        "mu": Xr.mean(1),
                        "sd": Xr.std(1, correction=0) + 1e-9,
                        "ymu": Yr.mean(1),
                    }
                    pred_rc = predict_from_fit(fit, tg["X_t"][rows], device=device, moments=mom)
                    st_rc = _skill_and_stats(pred_rc, Y_t, nm)
                    acc[eid]["recal_skill"].append(st_rc["skill"])
                    acc[eid].setdefault("recal_sse", []).append(st_rc["sse_unit"])
                # drift residual projections at the frozen trait rows
                if eid in drift_cells:
                    for trait, tr in trait_rows.items():
                        if tr in rows:
                            ri = rows.index(tr)
                            resid = Y_t[ri] - pred[ri]  # (n_t, H)
                            acc[eid]["proj"][trait] = (resid @ dirs.t()).to(torch.float32)
                del pred, st
            # support-overlap diagnostics from the turn-1 A fit (plan §6)
            if fid == "ctx_k1_A":
                for ri, r in enumerate(rows):
                    if r not in _clamp_rows(READOUT_ROWS, R):
                        continue
                    s, Q = fit["eig_s"][ri], fit["eig_Q"][ri]
                    order = torch.argsort(s, descending=True)
                    s_d = s[order]
                    frac = torch.cumsum(s_d, 0) / s_d.sum().clamp(min=1e-30)
                    r90 = int((frac < 0.90).sum().item()) + 1
                    Q_r = Q[:, order[:r90]]
                    s_r = s_d[:r90]
                    for k in range(1, C.K_MAIN + 1):
                        Xt = cache.x("main", k, "ctx", design["test_main"])[:, r].to(torch.float64)
                        Z = (Xt - fit["mu"][ri]) / fit["sd"][ri]
                        Kx = Z @ fit["Xn"][ri].t()  # (n_t, n)
                        proj = Kx @ (Q_r / torch.sqrt(s_r).unsqueeze(0))  # (n_t, r90)
                        frac_cap = float((proj * proj).sum() / (Z * Z).sum().clamp(min=1e-30))
                        n1 = fit["Xn"][ri].shape[0]
                        lev = (n1 * (proj * proj) / s_r.unsqueeze(0)).sum(1)
                        key = f"row{r}_k{k}"
                        support_overlap[key] = {
                            "r90": r90,
                            "var_captured_frac": frac_cap,
                            "mahalanobis_mean": float(lev.mean()),
                            "mahalanobis_p95": float(lev.quantile(0.95)),
                        }
            del fit
        # persist per-cell map weights (7 rows, fp16) + the FULL meta (the
        # resume predicate restores maps_meta from this blob on skip, r2)
        maps_meta[fid]["best_lam"] = {C.row_to_block_key(r): float(lam_all[r]) for r in range(R)}
        torch.save(
            {
                "cell": fid,
                "rows": {r: w for r, w in w_out.items()},
                "row_convention": "store rows: 0=emb, b+1=block b",
                "policy": C.TRANSFER_STANDARDIZATION_POLICY,
                "corpus_fingerprint": corpus_fp,
                "meta": maps_meta[fid],
                "metadata": C.reproducibility_metadata(
                    {"script": "issue958_fit_maps", "cell": fid}
                ),
            },
            args.maps / f"{fid}.pt",
        )
        for eid in eval_ids:
            a = acc[eid]
            e = design["evals"][eid]
            skill = torch.cat(a["skill"])
            out = {
                "skill": skill.numpy().astype(np.float64),
                "sse_unit": torch.cat(a["sse"], dim=0).numpy().astype(np.float32),
                "null_sse_unit": torch.cat(a["null"], dim=0).numpy().astype(np.float32),
                "shuffle_draws": torch.cat(a["shuf"], dim=1).numpy().astype(np.float64),
                "test_idx": np.asarray(targets[eid]["idx"], dtype=np.int64),
            }
            if a["recal_skill"]:
                out["recal_skill"] = torch.cat(a["recal_skill"]).numpy().astype(np.float64)
                out["recal_sse_unit"] = torch.cat(a["recal_sse"], dim=0).numpy().astype(np.float32)
            for trait, pr in a["proj"].items():
                out[f"proj_resid_{trait}"] = pr.numpy()
            np.savez(args.out / "percell" / f"{eid}.npz", **out)
            cell_results[eid] = {
                "skill_readout_mean": _readout_mean(skill),
                "map": e["map"],
                "target": {"set": e["set"], "k": e["k"], "pos": e["pos"]},
                **(
                    {"recal_skill_readout_mean": _readout_mean(torch.cat(a["recal_skill"]))}
                    if a["recal_skill"]
                    else {}
                ),
            }
            del acc[eid]
        del targets
        if fid == "ctx_k1_A":  # persist so a resume-skip can restore it (r2)
            C.write_json_atomic(support_sidecar, support_overlap)
        _mark_done(fid)
        logger.info("[fit %s] done (%.1fs elapsed; %d evals)", fid, time.time() - t0, len(eval_ids))

    # copy-previous-answer null (no map; plan §5)
    for k in range(2, C.K_MAIN + 1):
        eid = f"copyprev_k{k}"
        e = design["evals"][eid]
        pred = cache.x("main", k - 1, "ans", e["idx"]).to(torch.float64).transpose(0, 1)
        Y_t = cache.x("main", k, "ans", e["idx"]).to(torch.float64).transpose(0, 1)
        nm = _null_mean(e["null"])
        st = _skill_and_stats(pred, Y_t, nm)
        shuf = _shuffle_draws(st, C.SHUFFLE_DRAWS, C.SHUFFLE_SEED)
        np.savez(
            args.out / "percell" / f"{eid}.npz",
            skill=st["skill"].numpy().astype(np.float64),
            sse_unit=st["sse_unit"].numpy().astype(np.float32),
            null_sse_unit=st["null_sse_unit"].numpy().astype(np.float32),
            shuffle_draws=shuf.numpy().astype(np.float64),
            test_idx=np.asarray(e["idx"], dtype=np.int64),
        )
        cell_results[eid] = {
            "skill_readout_mean": _readout_mean(st["skill"]),
            "map": "copyprev",
            "target": {"set": "main", "k": k, "pos": "ans"},
        }

    # per-turn X moment-shift diagnostics vs turn 1 (plan §6, readout rows)
    moment_shift: dict[str, dict] = {}
    m1 = None
    for k in range(1, C.K_MAIN + 1):
        Xk = cache.x("main", k, "ctx", design["half_a"]).to(torch.float64)
        mu = Xk.mean(0)  # (R, H)
        sd = Xk.std(0, correction=0) + 1e-9
        norm = Xk.norm(dim=-1).mean(0)  # (R,)
        if k == 1:
            m1 = (mu, sd, norm)
        rows = _clamp_rows(READOUT_ROWS, R)
        cos = torch.nn.functional.cosine_similarity(mu[rows], m1[0][rows], dim=-1)
        moment_shift[f"k{k}"] = {
            "mu_cos_vs_k1_readout_mean": float(cos.mean()),
            "mu_norm_ratio_vs_k1": float((mu[rows].norm(dim=-1) / m1[0][rows].norm(dim=-1)).mean()),
            "sd_logratio_meanabs": float((sd[rows] / m1[1][rows]).log().abs().mean()),
            "x_norm_ratio_vs_k1": float((norm[rows] / m1[2][rows]).mean()),
        }
        del Xk

    # actual answer trait projections per turn (drift read; test conversations)
    drift_actual: dict[str, np.ndarray] = {}
    for panel, ks, idx in (
        ("main", range(1, C.K_MAIN + 1), design["test_main"]),
        ("long", range(1, C.K_LONG + 1), design["test_long"]),
    ):
        for k in ks:
            Y = cache.x(panel, k, "ans", idx).to(torch.float64)  # (n_t, R, H)
            for trait, tr in trait_rows.items():
                drift_actual[f"{panel}_k{k}_{trait}"] = (
                    (Y[:, tr] @ dirs.t()).to(torch.float32).numpy()
                )
            del Y
    np.savez(args.out / "drift_actual_projections.npz", **drift_actual)

    # per-unit token counts (H4 length reads; arrays over ALL convs per set/k)
    tok_arrays: dict[str, np.ndarray] = {}
    for panel, ks in (("main", range(1, C.K_MAIN + 1)), ("long", range(1, C.K_LONG + 1))):
        for k in ks:
            toks = cache.get(panel, k, "ctx")["toks"]
            for field, arr in toks.items():
                tok_arrays[f"{panel}_k{k}_{field}"] = np.asarray(arr)
    np.savez(args.out / "unit_tokens.npz", **tok_arrays)

    # grafted-query marginal eval (plan §4.5/§6 — decomposition-only companion)
    pm_path = args.out / "prefix_marginal.json"
    if manifest["cells"].get("_prefix_marginal", {}).get("done") and pm_path.exists():
        logger.info("[marginal] complete under current regime — skip (resume)")
        marginal = None  # file already on disk; nothing downstream reads it in-memory
    else:
        marginal = {}
    graft_spec = json.loads((args.corpus / "graftq_spec.json").read_text())
    graft_idx_store = C.load_store_index(args.store, "graft", expect_fingerprint=corpus_fp)
    valid_test_main = set(np.asarray(design["test_main"]).tolist())  # post-dropout filter
    graft_uids_all: dict[tuple[int, int], list[str]] = {}
    n_graft_dropped = 0
    for row in graft_spec["items"]:
        uid = C.unit_id("graft", row["ci"], row["k"], row["q"])
        # realized Q: empty-generation-dropped grafts are excluded here so the
        # Q-floor logic below operates on ACTUAL missing units (r2); hosts
        # whose own main chain dropped leave the read entirely
        if uid not in graft_idx_store or row["ci"] not in valid_test_main:
            n_graft_dropped += 1
            continue
        graft_uids_all.setdefault((row["ci"], row["k"]), []).append(uid)
    if n_graft_dropped:
        logger.warning(
            "[marginal] %d graft units dropped/invalid-host — realized Q shrinks", n_graft_dropped
        )
    for k in C.GRAFT_TURNS if marginal is not None else ():
        if f"pre_k{k}_full" not in fits_reg:
            continue
        hosts = sorted({ci for (ci, kk) in graft_uids_all if kk == k})
        if not hosts:
            continue
        uids_flat, per_host = [], []
        for ci in hosts:
            us = graft_uids_all[(ci, k)]
            per_host.append(len(us))
            uids_flat.extend(us)
        Yg = C.load_store_positions(
            args.store, "graft", uids_flat, [C.POS_ANS_MEAN], expect_fingerprint=corpus_fp
        )[:, 0]
        Yreal = cache.x("main", k, "ans", np.array(hosts))  # (n_h, R, H)
        # ȳ(prefix) = mean over {realized + grafts}; Q-floor drops below-floor hosts
        ybar, keep, qvar = [], [], []
        pos = 0
        for hi, nq in enumerate(per_host):
            ys = [Yreal[hi].to(torch.float64)] + [Yg[pos + j].to(torch.float64) for j in range(nq)]
            pos += nq
            if nq < graft_spec["q_floor"]:
                keep.append(False)
                ybar.append(torch.zeros_like(ys[0]))
                qvar.append(0.0)
                continue
            keep.append(True)
            yb = torch.stack(ys).mean(0)
            ybar.append(yb)
            qvar.append(float(torch.stack([((y - yb) ** 2).sum(-1).mean() for y in ys]).mean()))
        keep_np = np.array(keep, dtype=bool)
        if not keep_np.any():
            marginal[f"k{k}"] = {
                "n_hosts": len(hosts),
                "n_kept": 0,
                "n_dropped_below_q_floor": int((~keep_np).sum()),
            }
            continue
        ybar_t = torch.stack(ybar)[torch.from_numpy(keep_np)].transpose(0, 1)  # (R, n_keep, H)
        hosts_keep = np.array(hosts)[keep_np]
        # prefix-map predictions on host prefix states (identical prefix states
        # across grafts by causality — the host main unit's prefix_end is used)
        f = fits_reg[f"pre_k{k}_full"]
        Xh = cache.x("main", k, "prefix", hosts_keep)
        preds = []
        for lo in range(0, R, ROW_CHUNK):
            rows = list(range(lo, min(lo + ROW_CHUNK, R)))
            fit = fit_rows_batched(
                cache.x(f["set"], f["x_k"], f["x_pos"], f["idx"])[:, rows].transpose(0, 1),
                cache.x(f["set"], f["y_k"], "ans", f["idx"])[:, rows].transpose(0, 1),
                lambdas=RIDGE_LAMBDAS_922,
                device=device,
            )
            preds.append(predict_from_fit(fit, Xh[:, rows].transpose(0, 1), device=device))
            del fit
        pred = torch.cat(preds)
        nm = _null_mean(f"ctx_k{k}_full")
        st = _skill_and_stats(pred, ybar_t, nm)
        marginal[f"k{k}"] = {
            "n_hosts": len(hosts),
            "n_kept": int(keep_np.sum()),
            "n_dropped_below_q_floor": int((~keep_np).sum()),
            "marginal_skill_readout_mean": _readout_mean(st["skill"]),
            "marginal_skill_per_row": {
                C.row_to_block_key(r): float(st["skill"][r]) for r in range(R)
            },
            "irreducible_query_var_mean": float(
                np.mean([q for q, kp in zip(qvar, keep, strict=True) if kp])
            ),
            "map_error_readout_mean": float(
                st["sse_unit"][_clamp_rows(READOUT_ROWS, R)].mean().item()
            ),
        }
    if marginal is not None:
        C.write_json_atomic(
            pm_path,
            {
                "marginal": marginal,
                "corpus_fingerprint": corpus_fp,
                "n_graft_dropped": n_graft_dropped,
                "note": "decomposition-only companion (grafts sample q~p(q at turn k), "
                "not p(q|prefix))",
                "metadata": C.reproducibility_metadata({"script": "issue958_fit_maps"}),
            },
        )
        _mark_done("_prefix_marginal")

    # assemble headline JSONs (point estimates; CIs added by issue958_eval.py)
    grid = {
        f"{j}->{k}": cell_results[f"xfer_{j}to{k}_A"]["skill_readout_mean"]
        for j in range(1, C.K_MAIN + 1)
        for k in range(1, C.K_MAIN + 1)
    }
    recal = {
        f"{j}->{k}": {
            "raw": cell_results[f"recal_{j}to{k}_A"]["skill_readout_mean"],
            "recalibrated": cell_results[f"recal_{j}to{k}_A"]["recal_skill_readout_mean"],
        }
        for j in range(1, C.K_MAIN + 1)
        for k in range(1, C.K_MAIN + 1)
        if j != k
    }
    C.write_json_atomic(
        args.out / "transfer_matrix.json",
        {
            "transfer_standardization_policy": C.TRANSFER_STANDARDIZATION_POLICY,
            "corpus_fingerprint": corpus_fp,
            "readout_blocks": C.READOUT_BLOCKS,
            "grid_skill_readout_mean_foldA": grid,
            "own_B": {
                k: cell_results[f"own_k{k}_B"]["skill_readout_mean"] for k in range(1, C.K_MAIN + 1)
            },
            "own_full": {
                k: cell_results[f"own_k{k}_full"]["skill_readout_mean"]
                for k in range(1, C.K_MAIN + 1)
            },
            "recalibrated_companion": recal,
            "moment_shift": moment_shift,
            "support_overlap_turn1A": support_overlap,
            "panel_transfer": {
                k: cell_results[f"panelxfer_k{k}"]["skill_readout_mean"]
                for k in range(1, C.K_MAIN + 1)
            },
            "onpol_control": cell_results["onpol_k2"]["skill_readout_mean"],
            "n_test": len(design["test_main"]),
            "metadata": C.reproducibility_metadata({"script": "issue958_fit_maps"}),
        },
    )
    C.write_json_atomic(
        args.out / "forecast_curves.json",
        {
            "forecast": {
                f"{j}->{k}": cell_results[f"fcast_{j}to{k}"]["skill_readout_mean"]
                for j in range(1, C.K_MAIN + 1)
                for k in range(j + 1, C.K_MAIN + 1)
            },
            "copyprev": {
                k: cell_results[f"copyprev_k{k}"]["skill_readout_mean"]
                for k in range(2, C.K_MAIN + 1)
            },
            "prefix": {
                k: cell_results[f"pre_k{k}_full"]["skill_readout_mean"]
                for k in range(2, C.K_MAIN + 1)
            },
            "long_own": {
                k: cell_results[f"long_own_k{k}"]["skill_readout_mean"]
                for k in [*range(5, C.K_LONG + 1), 1]
            },
            "metadata": C.reproducibility_metadata({"script": "issue958_fit_maps"}),
        },
    )
    C.write_json_atomic(
        args.out / "maps_meta.json",
        {
            "cells": maps_meta,
            "corpus_fingerprint": corpus_fp,
            "equivalence_gate_real_cell": gate_real,
            "lambdas": RIDGE_LAMBDAS_922,
            "persist_rows": PERSIST_ROWS,
            "wall_seconds": time.time() - t0,
            "metadata": C.reproducibility_metadata({"script": "issue958_fit_maps"}),
        },
    )
    logger.info(
        "DONE fits+evals in %.1fs (%d fit cells, %d eval cells)",
        time.time() - t0,
        len(all_fit_ids),
        len(cell_results),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
