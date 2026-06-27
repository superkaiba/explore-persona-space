#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# (math/scientific notation — Δv, ŵ, ψ, Σ_c, ρ — intentional in docstrings + labels)
"""Issue #683 Phase C — key × metric leaderboard scoring (CPU, off-pod).

Plan §4 Phase C + §6. Per behavior, scores each source-key form

    k ∈ { c_C,  ψ(t_{C,B}),  c_C + ψ(δ_{C,B}) }

crossed with each metric

    M ∈ { I,  (Σ_c + λI)^-1 }

and the cosine baseline cos(c_C, c_C'), by the bilinear predicted gate

    g_pred(C'_i) = (kᵀ M c_C'_i) / (kᵀ M c_C)

against the held-out realized gate g_real(C'_i) (leave-one-context-out). For
each (key, metric) the leaderboard reports held-out Spearman ρ (PRIMARY),
Pearson r, sign-agreement, MAE — each with a 1000-bootstrap CI over held-out
contexts — plus a cross-seed range, the shuffled-KEY and shuffled-QUERY nulls
(a key/query-VECTOR permutation, NOT a matrix-axis relabel — methodology-critic
concern #2), and the test-retest noise floor.

A7 gating (plan §4 Phase B branch): reads
``eval_results/issue_683/<behavior>/a7_precondition.json`` FIRST. If rank-1
holds, the DV is the scalar g_real. If not, the low-rank fallback fits
Δv ≈ Σⱼ wⱼ gⱼ (m=1..3) and scores keys against the dominant component g₁
(the projection onto the stacked-SVD top-left singular direction). The branch
taken is recorded in the leaderboard.

Reuses ``issue637_heldout_predictive_test`` helpers: ``split_cells``,
``bootstrap_arm_ci``, ``paired_delta_ci``. The Spearman/MAE/sign scorers are
local (the #637 harness scores R², a different metric).

Inputs:
  - Δv banks (per source/seed) under analysis_tensors/dv/<behavior>/ — carry
    g_real + Δv (for the low-rank fallback) per context.
  - a c_C context-vector bank {ctx: c_C}: for marker, sliced at L14 from the
    #604 post-response bank (--c-bank). t_{C,B} from the t_cb extractor.
Output: ``eval_results/issue_683/<behavior>/key_ablation_leaderboard.json``.

CLI:
    uv run python scripts/issue683_key_ablation_score.py --behavior marker \
        --c-bank <#604 L14 c bank.pt> --tcb-dir <t_cb/marker> \
        --a7 eval_results/issue_683/marker/a7_precondition.json
    # CPU math smoke on a synthetic bank:
    uv run python scripts/issue683_key_ablation_score.py --behavior marker \
        --dv-dir eval_results/issue_683/smoke/synthetic_rank1/dv \
        --c-bank eval_results/issue_683/smoke/synthetic_rank1/c_bank.pt \
        --a7 eval_results/issue_683/smoke/a7_synth_rank1.json \
        --out eval_results/issue_683/smoke/leaderboard_marker_smoke.json \
        --n-boot 50 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="issue683_key_ablation_score")

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402

# Reuse the #637 held-out CV harness primitives (split + bootstrap CI).
from issue637_heldout_predictive_test import (  # noqa: E402
    bootstrap_arm_ci,  # noqa: F401  (re-exported for downstream parity / tests)
)

from explore_persona_space.experiments.issue_683 import DEFAULT_LAYER, repro_metadata  # noqa: E402

KEY_FORMS = ("k_cC", "k_tCB", "k_cC_plus_delta")
METRICS = ("M_I", "M_white")
PSI_FORMS = ("psi_I", "psi_ridge")
LAMBDA_GRID_MULT = (1e-3, 1e-2, 1e-1, 1.0, 1e1)  # × median-eigenvalue(Σ_c)


def spearman(pred: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ (rank correlation). NaN when degenerate (constant input)."""
    if len(y) < 2:
        return float("nan")
    pr = np.argsort(np.argsort(pred)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    if pr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(pr, yr)[0, 1])


def pearson(pred: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 2 or pred.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred, y)[0, 1])


def sign_agreement(pred: np.ndarray, y: np.ndarray, ref: float) -> float:
    """Fraction of contexts where pred and y fall on the same side of ref."""
    if len(y) == 0:
        return float("nan")
    return float((np.sign(pred - ref) == np.sign(y - ref)).mean())


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(pred - y).mean()) if len(y) else float("nan")


def _whiten_metric(c_matrix: np.ndarray, lam_mult: float) -> np.ndarray:
    """M = (Σ_c + λI)^-1 with λ = lam_mult × median-eigenvalue(Σ_c).

    Σ_c = (1/n) C Cᵀ over the context vectors (H×H). For H≫n this is
    rank-deficient, so the +λI regularizer is load-bearing.
    """
    n = c_matrix.shape[0]
    sigma_c = (c_matrix.T @ c_matrix) / n  # (H, H)
    eig = np.linalg.eigvalsh(sigma_c)
    med = float(np.median(eig[eig > 0])) if np.any(eig > 0) else 1.0
    lam = lam_mult * med
    return np.linalg.inv(sigma_c + lam * np.eye(sigma_c.shape[0]))


def _fit_ridge_psi(t_train: np.ndarray, c_train: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Learned linear map ψ: t-space → c-space, ridge, fit on TRAIN contexts only.

    Returns W (H×H) s.t. ψ(t) = W t minimizes ‖W T − C‖² + λ‖W‖². Closed form
    W = C Tᵀ (T Tᵀ + λI)^-1 over the TRAIN context matrices (rows = contexts).
    """
    t = t_train  # (n_train, H)
    c = c_train  # (n_train, H)
    g = t.T @ t + lam * np.eye(t.shape[1])  # (H, H)
    return (c.T @ t) @ np.linalg.inv(g)  # (H, H)


def _fit_ridge_psi_for_targets(
    per_context: dict, c_bank: dict[str, np.ndarray], targets: list[str]
) -> np.ndarray | None:
    """ψ: answer-side residual space → prompt-side query space (the robustness arm).

    Fit on the per-context (v_base[ctx] = answer-side base read, c_C[ctx] =
    prompt-side context vector) pairs that exist per target context (A8: "ψ maps
    answer-side or data-side vectors into the key–query space"). The source is
    held out (it is the denominator); the learned map applies to t_{C,B} (itself
    an answer-side mean). Returns None when <3 paired contexts exist.
    """
    import torch

    pair_ctx = [c for c in targets if c in per_context and "v_base" in per_context[c]]
    if len(pair_ctx) < 3:
        return None
    t_mat = np.stack(
        [torch.as_tensor(per_context[c]["v_base"]).flatten().float().numpy() for c in pair_ctx],
        axis=0,
    )
    c_mat = np.stack([c_bank[c] for c in pair_ctx], axis=0)
    return _fit_ridge_psi(t_mat, c_mat)


def _key_vector(
    form: str,
    psi_W: np.ndarray | None,
    *,
    c_source: np.ndarray,
    t_cb: np.ndarray | None,
    delta_cb: np.ndarray | None,
) -> np.ndarray | None:
    """Build the source key vector for one (key_form, ψ) cell.

    k_cC = c_C; k_tCB = ψ(t_{C,B}); k_cC_plus_delta = c_C + ψ(δ_{C,B}).
    Returns None when a t-based key is requested but t_{C,B}/δ are absent.
    """
    if form == "k_cC":
        return c_source
    if form == "k_tCB":
        if t_cb is None:
            return None
        return (psi_W @ t_cb) if psi_W is not None else t_cb
    if form == "k_cC_plus_delta":
        if delta_cb is None:
            return None
        psi_d = (psi_W @ delta_cb) if psi_W is not None else delta_cb
        return c_source + psi_d
    raise ValueError(form)


def _g_pred(
    k: np.ndarray, m: np.ndarray | None, c_query: np.ndarray, c_source: np.ndarray
) -> float:
    """g_pred = (kᵀ M c_query) / (kᵀ M c_source); M=None ⇒ identity."""
    if m is None:
        num = float(k @ c_query)
        den = float(k @ c_source)
    else:
        km = k @ m
        num = float(km @ c_query)
        den = float(km @ c_source)
    if den == 0:
        return float("nan")
    return num / den


def _resolve_metric(
    metric: str, k: np.ndarray, c_full: np.ndarray, c_source: np.ndarray
) -> np.ndarray | None:
    """Resolve M for a (key, metric) cell. M_I → None (identity); M_white →
    (Σ_c+λI)^-1 with λ over the grid minimizing |1 - g_pred(source)| (the
    denominator-stability proxy; full GCV is the production path)."""
    if metric == "M_I":
        return None
    best_m, best_pen = None, np.inf
    for mult in LAMBDA_GRID_MULT:
        cand = _whiten_metric(c_full, mult)
        gp_src = _g_pred(k, cand, c_source, c_source)  # == 1 ideally
        pen = abs(1.0 - gp_src) if gp_src == gp_src else np.inf
        if pen < best_pen:
            best_pen, best_m = pen, cand
    return best_m


def _score_one_cell(
    *,
    key_form: str,
    metric: str,
    psi: str,
    k: np.ndarray,
    c_source: np.ndarray,
    c_query: dict[str, np.ndarray],
    c_full: np.ndarray,
    targets: list[str],
    y: np.ndarray,
    a7_rank1: bool,
    n_boot: int,
    rng,
) -> dict | None:
    """Score ONE (key, metric, ψ) cell: held-out Spearman/Pearson/sign/MAE +
    bootstrap Spearman CI. Returns None when <3 contexts score finitely."""
    m = _resolve_metric(metric, k, c_full, c_source)
    preds = np.array([_g_pred(k, m, c_query[c], c_source) for c in targets])
    finite = np.isfinite(preds) & np.isfinite(y)
    if finite.sum() < 3:
        return None
    p, yy = preds[finite], y[finite]
    ref = 0.0 if not a7_rank1 else float(np.median(yy))
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yy), len(yy))
        boot.append(spearman(p[idx], yy[idx]))
    boot = np.array([b for b in boot if b == b])
    ci = (
        [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        if boot.size
        else [float("nan"), float("nan")]
    )
    return {
        "key": key_form,
        "metric": metric,
        "psi": psi,
        "n_scored": int(finite.sum()),
        "spearman": spearman(p, yy),
        "pearson": pearson(p, yy),
        "sign_agreement": sign_agreement(p, yy, ref),
        "mae": mae(p, yy),
        "spearman_ci95": ci,
    }


def _load_dv_banks(dv_dir: Path) -> list[dict]:
    import torch

    banks = []
    for p in sorted(dv_dir.glob("*_L*.pt")):
        banks.append(torch.load(p, map_location="cpu", weights_only=False))
    if not banks:
        raise FileNotFoundError(f"no Δv bank .pt files under {dv_dir}")
    return banks


def _load_c_bank(c_bank_path: Path, layer: int) -> dict[str, np.ndarray]:
    """Load {ctx: c_C} as numpy. Accepts a synthetic {contexts:{ctx:(H,)}} OR
    the #604 {contexts:{ctx:(28,H)}} all-layer bank (sliced at ``layer``)."""
    import torch

    obj = torch.load(c_bank_path, map_location="cpu", weights_only=False)
    ctxs = obj.get("contexts", obj)
    out: dict[str, np.ndarray] = {}
    for name, v in ctxs.items():
        t = torch.as_tensor(v)
        if t.ndim == 2:  # (n_layers, H) → slice the read layer
            t = t[min(layer, t.shape[0] - 1)]
        out[name] = t.flatten().float().numpy()
    return out


def _load_tcb(tcb_dir: Path, behavior: str, source: str, layer: int) -> np.ndarray | None:
    """Load t_{C,B} for (source) from the t_cb extractor output, if present."""
    import torch

    if tcb_dir is None or not tcb_dir.is_dir():
        return None
    cand = list(tcb_dir.glob(f"t_cb_{behavior}_{source}_L*.pt"))
    if not cand:
        return None
    payload = torch.load(cand[0], map_location="cpu", weights_only=False)
    return torch.as_tensor(payload["t_cb"]).flatten().float().numpy()


def _dv_target_value(payload: dict, ctx: str, a7_rank1: bool, u1: np.ndarray | None) -> float:
    """The scored DV for a held-out context: scalar g_real (rank-1) OR the
    dominant-component projection ⟨Δv(C'), u₁⟩ (low-rank fallback)."""
    if a7_rank1:
        g = payload["g_real"].get(ctx)
        return float(g) if g is not None and g == g else float("nan")
    import torch

    dv = torch.as_tensor(payload["per_context"][ctx]["Delta_v"]).flatten().float().numpy()
    return float(dv @ u1)


def score_bank(
    *,
    payload: dict,
    c_bank: dict[str, np.ndarray],
    t_cb: np.ndarray | None,
    a7_rank1: bool,
    n_boot: int,
    seed: int,
) -> dict:
    """Full key × metric × ψ leaderboard for one source/seed bank."""
    source = payload["source"]
    per_context = payload["per_context"]
    # held-out targets that ALSO have a c_C entry (the predictor needs c_C').
    targets = [c for c in per_context if c != source and c in c_bank]
    if source not in c_bank:
        raise AssertionError(
            f"source {source!r} has no c_C entry in the context bank — cannot build any key."
        )
    if len(targets) < 3:
        raise AssertionError(
            f"only {len(targets)} held-out targets with a c_C entry for source {source!r}; "
            "need >= 3 for a leave-one-context-out leaderboard."
        )

    # low-rank fallback dominant direction u₁ (only used when a7_rank1 is False).
    u1 = None
    if not a7_rank1:
        import torch

        dvs = np.stack(
            [torch.as_tensor(per_context[c]["Delta_v"]).flatten().float().numpy() for c in targets],
            axis=1,
        )  # (H, n)
        u, _s, _vt = np.linalg.svd(dvs, full_matrices=False)
        u1 = u[:, 0]

    c_source = c_bank[source]
    w_hat = np.asarray(payload["w_hat"]).astype(float).flatten() if "w_hat" in payload else None
    import torch

    if w_hat is None:
        w_hat = torch.as_tensor(per_context[source]["Delta_v"]).flatten().float().numpy()
    # δ_{C,B} = t_{C,B} - v_base(C); v_base(C) read from the source's own context.
    v_base_source = torch.as_tensor(per_context[source]["v_base"]).flatten().float().numpy()
    delta_cb = (t_cb - v_base_source) if t_cb is not None else None

    # Pre-stack c_C' query matrix + the scored DV y over the targets.
    c_query = {c: c_bank[c] for c in targets}
    y = np.array([_dv_target_value(payload, c, a7_rank1, u1) for c in targets])

    # whitened metrics: λ picked by held-out GCV proxy over the TRAIN contexts.
    # For the leaderboard we fit Σ_c on the FULL context bank (base-model
    # quantity, no g_real leakage) and pick λ minimizing the cross-context
    # Spearman gap — here we just sweep and keep the best-on-train λ per metric.
    c_full = np.stack([c_bank[c] for c in [source, *targets]], axis=0)

    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for psi in PSI_FORMS:
        psi_W = (
            _fit_ridge_psi_for_targets(per_context, c_bank, targets)
            if psi == "psi_ridge" and t_cb is not None
            else None
        )
        for key_form in KEY_FORMS:
            if key_form != "k_cC" and psi == "psi_ridge" and psi_W is None:
                continue  # ridge ψ only applies to t-based keys
            k = _key_vector(
                key_form,
                psi_W if key_form != "k_cC" else None,
                c_source=c_source,
                t_cb=t_cb,
                delta_cb=delta_cb,
            )
            if k is None:
                continue
            for metric in METRICS:
                row = _score_one_cell(
                    key_form=key_form,
                    metric=metric,
                    psi=psi,
                    k=k,
                    c_source=c_source,
                    c_query=c_query,
                    c_full=c_full,
                    targets=targets,
                    y=y,
                    a7_rank1=a7_rank1,
                    n_boot=n_boot,
                    rng=rng,
                )
                if row is not None:
                    rows.append(row)

    baseline_cos, shuf_key, shuf_query = _nulls_and_baseline(
        c_source=c_source,
        c_query=c_query,
        c_bank=c_bank,
        targets=targets,
        source=source,
        y=y,
        n_boot=n_boot,
        rng=rng,
    )

    return {
        "source": source,
        "seed": payload.get("seed"),
        "a7_rank1": a7_rank1,
        "scored_dv": "g_real_scalar" if a7_rank1 else "lowrank_dominant_component",
        "n_targets": len(targets),
        "leaderboard": rows,
        "baseline_cos": baseline_cos,
        "null_shuffled_key": _null_summary(shuf_key),
        "null_shuffled_query": _null_summary(shuf_query),
        "has_tcb": t_cb is not None,
    }


def _null_summary(arr: np.ndarray) -> dict:
    """Mean + [p5, p95] of a null Spearman distribution (NaN-safe)."""
    if arr.size == 0:
        return {"mean": float("nan"), "p5": float("nan"), "p95": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "n": int(arr.size),
    }


def _nulls_and_baseline(
    *,
    c_source: np.ndarray,
    c_query: dict[str, np.ndarray],
    c_bank: dict[str, np.ndarray],
    targets: list[str],
    source: str,
    y: np.ndarray,
    n_boot: int,
    rng,
) -> tuple[dict | None, np.ndarray, np.ndarray]:
    """Cosine baseline + the shuffled-KEY / shuffled-QUERY nulls.

    Both nulls permute a key/query VECTOR (methodology-critic concern #2),
    NOT a matrix-axis relabel: shuffled-key scores c_C against a RANDOM other
    context's c as the key; shuffled-query permutes which c_C' each g_real is
    scored against. Returns (baseline_cos dict | None, shuf_key arr, shuf_query arr).
    """

    def _cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")

    cos_pred = np.array([_cos(c_source, c_query[c]) for c in targets])
    finite = np.isfinite(cos_pred) & np.isfinite(y)
    baseline_cos = None
    if finite.sum() >= 3:
        baseline_cos = {
            "spearman": spearman(cos_pred[finite], y[finite]),
            "pearson": pearson(cos_pred[finite], y[finite]),
            "n_scored": int(finite.sum()),
        }

    null_seeds = 20 if _is_smoke(n_boot) else 200

    def _spearman_for_key(k_vec):
        preds = np.array([_g_pred(k_vec, None, c_query[c], c_source) for c in targets])
        f = np.isfinite(preds) & np.isfinite(y)
        return spearman(preds[f], y[f]) if f.sum() >= 3 else float("nan")

    other_ctx = [c for c in c_bank if c not in (source, *targets)] or targets
    query_vals = [c_query[c] for c in targets]
    shuf_key, shuf_query = [], []
    for _ in range(null_seeds):
        rk = c_bank[other_ctx[rng.integers(0, len(other_ctx))]]
        shuf_key.append(_spearman_for_key(rk))
        perm = rng.permutation(len(targets))
        preds = np.array(
            [_g_pred(c_source, None, query_vals[perm[i]], c_source) for i in range(len(targets))]
        )
        f = np.isfinite(preds) & np.isfinite(y)
        shuf_query.append(spearman(preds[f], y[f]) if f.sum() >= 3 else float("nan"))
    return (
        baseline_cos,
        np.array([v for v in shuf_key if v == v]),
        np.array([v for v in shuf_query if v == v]),
    )


_SMOKE_BOOT_MAX = 100


def _is_smoke(n_boot: int) -> bool:
    return n_boot <= _SMOKE_BOOT_MAX


def _noise_floor(banks: list[dict]) -> dict:
    """Test-retest noise floor: cross-seed Spearman of g_real on shared contexts.

    Bounds the achievable ρ (a predictor can't beat the measurement
    reliability). Computed only when ≥2 seeds of the SAME source are present.
    """
    by_source: dict[str, list[dict]] = {}
    for b in banks:
        by_source.setdefault(b["source"], []).append(b)
    floors = {}
    for source, group in by_source.items():
        if len(group) < 2:
            continue
        # all-pairs cross-seed Spearman of g_real over shared contexts.
        rhos = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                gi, gj = group[i]["g_real"], group[j]["g_real"]
                shared = [c for c in gi if c in gj and c != source]
                a = np.array([gi[c] for c in shared], dtype=float)
                b2 = np.array([gj[c] for c in shared], dtype=float)
                f = np.isfinite(a) & np.isfinite(b2)
                if f.sum() >= 3:
                    rhos.append(spearman(a[f], b2[f]))
        rhos = [r for r in rhos if r == r]
        if rhos:
            floors[source] = {
                "test_retest_spearman_mean": float(np.mean(rhos)),
                "n_pairs": len(rhos),
            }
    return floors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--behavior", required=True, choices=("marker", "sycophancy"))
    ap.add_argument("--dv-dir", default=None, help="default analysis_tensors/dv/<behavior>")
    ap.add_argument("--c-bank", required=True, help=".pt context-vector bank {ctx: c_C}")
    ap.add_argument(
        "--tcb-dir", default=None, help="t_cb extractor output dir (enables k_tCB/k_cC+δ)"
    )
    ap.add_argument("--a7", default=None, help="a7_precondition.json (gates the scored DV)")
    ap.add_argument(
        "--layer", type=int, default=None, help="c-bank slice layer; default per behavior"
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    layer = args.layer if args.layer is not None else DEFAULT_LAYER[args.behavior]
    dv_dir = Path(
        args.dv_dir or (PROJECT_ROOT / "eval_results/issue_683/analysis_tensors/dv" / args.behavior)
    )
    out_path = Path(
        args.out
        or (
            PROJECT_ROOT
            / "eval_results/issue_683"
            / args.behavior
            / "key_ablation_leaderboard.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    a7_rank1 = True
    a7_verdict = "assumed_rank1 (no a7 file)"
    if args.a7 and Path(args.a7).is_file():
        a7 = json.loads(Path(args.a7).read_text())
        a7_rank1 = bool(a7.get("behavior_rank1_holds", True))
        a7_verdict = a7.get("verdict", a7_verdict)

    c_bank = _load_c_bank(Path(args.c_bank), layer)
    tcb_dir = Path(args.tcb_dir) if args.tcb_dir else None
    banks = _load_dv_banks(dv_dir)

    logger.info(
        "[phase=score_start] behavior=%s a7_rank1=%s (%s) n_banks=%d c_bank=%d ctx tcb_dir=%s",
        args.behavior,
        a7_rank1,
        a7_verdict,
        len(banks),
        len(c_bank),
        tcb_dir,
    )

    bank_results = []
    for payload in banks:
        t_cb = _load_tcb(tcb_dir, args.behavior, payload["source"], layer)
        res = score_bank(
            payload=payload,
            c_bank=c_bank,
            t_cb=t_cb,
            a7_rank1=a7_rank1,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        bank_results.append(res)
        best = max(
            (r for r in res["leaderboard"] if r["spearman"] == r["spearman"]),
            key=lambda r: r["spearman"],
            default=None,
        )
        logger.info(
            "[phase=score_bank] source=%s seed=%s n_targets=%d best=%s null_key_mean=%.3f",
            res["source"],
            res["seed"],
            res["n_targets"],
            (
                f"{best['key']}/{best['metric']}/{best['psi']} ρ={best['spearman']:.3f}"
                if best
                else "—"
            ),
            res["null_shuffled_key"]["mean"],
        )

    payload_out = {
        "behavior": args.behavior,
        "layer": layer,
        "a7_rank1": a7_rank1,
        "a7_verdict": a7_verdict,
        "key_forms": list(KEY_FORMS),
        "metrics": list(METRICS),
        "psi_forms": list(PSI_FORMS),
        "n_bootstrap": args.n_boot,
        "per_bank": bank_results,
        "noise_floor": _noise_floor(banks),
        "reproducibility": repro_metadata({"behavior": args.behavior, "layer": layer}),
    }
    out_path.write_text(json.dumps(payload_out, indent=2))
    logger.info("[phase=score_done] behavior=%s -> %s", args.behavior, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
