"""Issue #2569 leg 7 — cross-model fits, three-tier report, operator atlas (plan v4 P-D/P-E).

Consumes the paired captures ``issue2569_xmodel_capture.py`` finalizes
(``{qwen,llama}_{vc,va}_L{K}.pt`` bundles) and produces:

- ``--phase fits`` (P-D pod, or VM at synthetic scale): tier-1 alignability grid over
  the matched layer pairs (14,16)/(19,22)/(26,30) — alignment ridges Qwen<->Llama for
  v_C and v_A separately + linear CKA (arXiv 1905.00414) — working pair selected by
  TRAIN-split (val) alignability only; at the working pair: the native Llama
  context->answer ridge, the matched-n Qwen comparator (C3: same rows, same folds),
  the four alignment maps, the composed-route / alignment-only-route evaluations,
  the correspondence test, split-half operator floors, mapping baselines
  (identity+bias where d_in == d_out — stated INAPPLICABLE cross-model — and kNN
  retrieval everywhere, chance = k/n_pool stated). Persists ``llama_map_L{K}.pt`` +
  ``align_maps.pt`` + ``fits_summary.json`` (+ HF upload).
- ``--phase report`` (P-E, VM): assembles ``eval_results/issue_2569/leg7/three_tier.json``
  — tier 1 alignability; tier 2 composed-vs-native held-out R² + activation-Procrustes
  aligned operator cosine vs the random-rotation null (read against the #825
  base<->instruct anchor 0.6864); tier 3 NON-IDENTIFYING DIAGNOSTICS (C1): R²(native)
  − R²(composed) beside the alignment-only baseline — these contextualize tiers 1-2
  and NO registered verdict consumes them. Every cross-model claim is scoped to
  transformations of SHARED TEACHER-FORCED QWEN RESPONSES (C3).
- ``--phase atlas`` (P-E, VM): embeds the resolvable residual-space operators —
  banked n1m L14/19/26 (via ``issue2569_operator``), #2474 pass-B base maps
  L14/L16/L27 (B6-conditional, staged from ``issue2379_reelicit``; a miss DROPS the
  row with a named reason, never a silent substitute), the Llama native map + the
  composed operator (activation-Procrustes-aligned into the Qwen basis where the
  paired captures are staged), the matched-n Qwen comparator, and the leg-4 feature
  map's linearized residual composition (B1 row convention:
  ``A_feat = E_ctx_alive @ diag(1/xsd) @ W_feat @ D_ans_union``, ``vhat = v @ A_feat``;
  ReLU/TopK gates ignored — labeled ``linearized``) re-derived closed-form from the
  Unit-4b persisted encodes at the RECORDED lambda (predictions/metrics are banked;
  the M_feat WEIGHTS are not — re-derive route). Pairwise statistics follow the
  ``issue1345_operator_comparison.py`` conventions with EVERY statistic labeled
  direction-aware (raw cosine + rotation chance band; Procrustes-aligned cosine
  where paired activations exist) vs spectrum/rotation-invariant-only (spectrum
  cosine — a descriptive ceiling that can never support "same operator up to
  rotation"). Split-half noise floors for every refittable operator; operators
  without refittable rows are labeled ``no floor — descriptive only``. Classical MDS
  coordinates are presentation-only; every claim reads off the distance table.
  Writes ``eval_results/issue_2569/leg7/atlas_distances.json``.
- ``--phase selftest``: tiny-synthetic end-to-end pass (fits -> report -> atlas at
  d=8/12, n~240) — the committed CPU smoke for the whole assembly path.

Fit core: the reused #779 val-lambda-selected primal ridge
(``issue779_ffc_n50k_fits.fit_ridge_primal`` via ``issue2569_gateladder._load_fit_core``;
POSITIONAL tr/val/te split-index arrays) over the widened 27-value lambda grid with
the C4 widen-on-edge loop (mirrors ``issue2569_rowbattery._fit_val_widened``); operator
WEIGHTS are re-derived at the selected lambda by ``ridge_beta_at_lambda``, which
replicates the core's preprocessing verbatim (standardize X on train stats + 1e-9,
center Y on train mean — ``issue779_ffc_n50k_fits._ridge_primal_multi_lambda``) and is
asserted against the core's predictions in the selftest. All fitted payloads use the
``issue2569_operator.MapPayload`` contract, so the registered prediction path
(``OP.predict``) and the row-action operator (``OP.row_operator``) are REUSED, never
re-derived (B1); the banked-map identity asserts (``OP.run_driver_identity_asserts``)
run at atlas entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + HF credentials BEFORE numpy/torch (#847)

import issue2569_operator as OP  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2569.atlas")

TASK_ID = 2569
HF_XMODEL_PREFIX = "issue2569_theory/analysis_tensors/xmodel"
LEG7_DIR = PROJECT_ROOT / "eval_results" / "issue_2569" / "leg7"
MATCHED_PAIRS = ((14, 16), (19, 22), (26, 30))  # (qwen layer, llama layer), plan §4 leg 7
PASSB_PREFIX = "issue2379_reelicit/analysis_tensors/maps_pinned"  # #2474 provenance
PASSB_LAYERS = (14, 16, 27)
ANCHOR_825_ALIGNED_COSINE = 0.6864  # #825 base<->instruct anchor (plan tier-2 read)
N_GROUPS = 20  # grouped random fold: 2 test groups (10%), val slice from the rest
KNN_KS = (1, 5, 10)
FIT_FLOOR_N_TRAIN = 45_000  # production fit floor (plan §7 C9; n >> d = 4,096)
PRODUCTION_REALIZED_FLOOR = 50_000


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _atomic_json(path: Path, obj: dict) -> None:
    """JSON write through the shared process-unique atomic-replace primitive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))


def _atomic_torch_save(obj: dict, path: Path) -> None:
    """torch.save through atomic_replace (#2336 process-unique tmp)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        torch.save(obj, tmp)


def _meta(phase: str) -> dict:
    """Reproducibility metadata block (git commit + dirty flag + timestamp)."""
    prov = git_provenance()
    md = as_metadata_dict(prov, phase=phase)
    md["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return md


def _sha_int64(a: np.ndarray) -> str:
    """sha256 of an int64 array (pinned integer dtype — machine-stable)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes()).hexdigest()


_FIT_CORE: tuple | None = None


def _fit_core():
    """Deferred #779 fit core: (fit_ridge_primal, pooled_r2) via the landed
    ``issue2569_gateladder._load_fit_core`` (~20 s torch chain; #823-safe)."""
    global _FIT_CORE
    if _FIT_CORE is None:
        import issue2569_gateladder as GL

        _FIT_CORE = GL._load_fit_core()
    return _FIT_CORE


def fit_val_widened(X, Y, tr, va, te, dev, *, fit_fn=None, grid=None, max_widenings=None):
    """C4 widen-on-edge val-selected primal ridge (mirrors
    ``issue2569_rowbattery._fit_val_widened`` — an edge lambda is never reported).
    Returns (pred_te, meta). ``fit_fn`` is a test seam (default: the #779 core)."""
    import issue2569_gateladder as GL

    if fit_fn is None:
        fit_fn = _fit_core()[0]
    g = tuple(GL.LAMBDA_GRID_27 if grid is None else grid)
    max_w = int(GL.MAX_WIDENINGS if max_widenings is None else max_widenings)
    edge = None
    for w in range(max_w + 1):
        pred_te, meta = fit_fn(X, Y, tr, va, te, list(g), dev)
        edge = meta.get("lambda_grid_edge")
        if not edge:
            meta = dict(meta)
            meta.update(widenings=w, grid_lo=float(g[0]), grid_hi=float(g[-1]), grid_n=len(g))
            return np.asarray(pred_te), meta
        logger.warning("[atlas-fit] lambda at the %s edge — widening the grid (C4)", edge)
        g = GL.widen_grid(g, edge)
    raise RuntimeError(
        f"lambda selection still at the {edge} edge after {max_w} widenings "
        "(C4: refusing to report an edge value)"
    )


def ridge_beta_at_lambda(X: np.ndarray, Y: np.ndarray, tr: np.ndarray, lam: float) -> OP.MapPayload:
    """Closed-form standardized primal ridge at a FIXED lambda -> ``OP.MapPayload``.

    Replicates ``issue779_ffc_n50k_fits._ridge_primal_multi_lambda`` preprocessing
    VERBATIM (standardize X on train stats with +1e-9, center Y on train mean; fp64
    eigh solve), so a payload built here at the core's selected lambda reproduces
    the core's predictions exactly (asserted in the selftest). The payload rides the
    vendored contract shape, so ``OP.predict`` / ``OP.row_operator`` are the ONLY
    prediction/operator paths (B1 — never a re-derived product)."""
    Xtr = torch.as_tensor(np.asarray(X)[tr], dtype=torch.float64)
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xn = (Xtr - xmu) / xsd
    Yt = torch.as_tensor(np.asarray(Y)[tr], dtype=torch.float64)
    ymu = Yt.mean(0)
    Yc = Yt - ymu
    A = Xn.T @ Xn
    s, U = torch.linalg.eigh(A)
    s = torch.clamp(s, min=0.0)
    W = U @ ((U.T @ (Xn.T @ Yc)) / (s + float(lam))[:, None])
    return OP.MapPayload(
        layer=-1,
        path=Path("<fitted-in-memory>"),
        W=W.numpy(),
        xmu=xmu.numpy(),
        xsd=xsd.numpy(),
        ymu=ymu.numpy(),
        selected_lambda=float(lam),
        raw={},
    )


def payload_to_dict(p: OP.MapPayload) -> dict:
    """Serializable ridge-payload dict in the vendored banked-map contract shape
    (kind == fitter == 'ridge'), fp32 tensors like the banked artifacts."""
    return {
        "kind": "ridge",
        "fitter": "ridge",
        "layer": int(p.layer),
        "W": torch.as_tensor(p.W, dtype=torch.float32),
        "xmu": torch.as_tensor(p.xmu, dtype=torch.float32),
        "xsd": torch.as_tensor(p.xsd, dtype=torch.float32),
        "ymu": torch.as_tensor(p.ymu, dtype=torch.float32),
        "selected_lambda": float(p.selected_lambda),
    }


def payload_from_dict(d: dict, *, path: Path) -> OP.MapPayload:
    """Validate + upcast a serialized ridge-payload dict (mirrors the checks of
    ``OP.load_banked_map`` for NON-square shapes: alignment maps are d_in x d_out)."""
    if d.get("kind") != "ridge" or d.get("fitter") != "ridge":
        raise RuntimeError(f"{path}: expected ridge payload, got kind={d.get('kind')!r}")
    comp = {k: np.asarray(d[k], dtype=np.float64) for k in ("W", "xmu", "xsd", "ymu")}
    d_in, d_out = comp["W"].shape
    assert comp["xmu"].shape == comp["xsd"].shape == (d_in,), (comp["xmu"].shape, d_in)
    assert comp["ymu"].shape == (d_out,), (comp["ymu"].shape, d_out)
    for k, v in comp.items():
        assert np.isfinite(v).all(), f"{path}: {k} non-finite"
    assert (comp["xsd"] > 0).all(), f"{path}: xsd must be strictly positive"
    return OP.MapPayload(
        layer=int(d.get("layer", -1)),
        path=path,
        W=comp["W"],
        xmu=comp["xmu"],
        xsd=comp["xsd"],
        ymu=comp["ymu"],
        selected_lambda=float(d["selected_lambda"]),
        raw=d,
    )


# ---------------------------------------------------------------------------
# Capture loading + paired join + grouped folds
# ---------------------------------------------------------------------------


def _decode_bundle(path: Path) -> dict:
    """Load one finalized capture bundle -> {x fp32 (n,d), ci, corpus, layer, slot}."""
    import issue2569_xmodel_capture as XC

    b = torch.load(path, map_location="cpu", weights_only=False)
    x = XC.decode_summary(np.asarray(b["x"]), b["codec"])
    return {
        "x": np.asarray(x, dtype=np.float32),
        "ci": np.asarray(b["ci"], dtype=np.int64),
        "corpus": list(b["corpus"]),
        "layer": int(b["layer"]),
        "slot": b["slot"],
        "model_id": b["model_id"],
    }


def load_captures(args, layers_by_model: dict[str, list[int]]) -> dict:
    """Load the finalized capture bundles for both models (local ``--capture-dir``;
    ``--stage-from-hf`` stages missing files from the xmodel prefix first)."""
    cap_dir = Path(args.capture_dir)
    out: dict = {}
    for model, layers in layers_by_model.items():
        out[model] = {}
        for layer in layers:
            for tag in ("vc", "va"):
                name = f"{model}_{tag}_L{layer}.pt"
                path = cap_dir / name
                if not path.exists() and args.stage_from_hf:
                    hub.stage_hub_file(
                        args.hf_data_repo,
                        f"{HF_XMODEL_PREFIX}/{name}",
                        path,
                        repo_type="dataset",
                    )
                assert path.exists(), f"capture bundle missing: {path}"
                out[model][(tag, layer)] = _decode_bundle(path)
    return out


def paired_join(caps: dict) -> dict:
    """Join the two models' bundles on ci (intersection, sorted) and assert
    corpus-tag agreement. Returns {ci, corpus, index maps per model}."""
    ref = {}
    for model, bundles in caps.items():
        any_b = next(iter(bundles.values()))
        for b in bundles.values():  # all bundles of one model share the row set
            assert np.array_equal(b["ci"], any_b["ci"]), f"{model}: bundle ci drift"
        ref[model] = any_b
    q, ll = ref["qwen"], ref["llama"]
    common = np.intersect1d(q["ci"], ll["ci"])
    assert len(common) > 0, "no paired rows across models"
    qi = {int(c): i for i, c in enumerate(q["ci"])}
    li = {int(c): i for i, c in enumerate(ll["ci"])}
    q_idx = np.asarray([qi[int(c)] for c in common], dtype=np.int64)
    l_idx = np.asarray([li[int(c)] for c in common], dtype=np.int64)
    corpus = np.asarray([q["corpus"][i] for i in q_idx])
    corpus_l = np.asarray([ll["corpus"][i] for i in l_idx])
    assert (corpus == corpus_l).all(), "corpus tags disagree across models at join"
    return {"ci": common, "corpus": corpus, "q_idx": q_idx, "l_idx": l_idx}


def grouped_folds(ci: np.ndarray, val_rows: int) -> dict:
    """Deterministic grouped 90/10 fold + a val slice for lambda selection.

    Groups = a ci-keyed integer hash (Knuth multiplicative, machine-stable — never
    a float/argsort key; gotchas.md tie/last-bit rules) into ``N_GROUPS`` buckets:
    buckets {0,1} = test (10%), val = the first ``val_rows`` rows of bucket 2,
    train = everything else. Reported ``n_train_90pct`` counts train+val (the plan
    C9 bookkeeping: fit n_train = 0.9 x realized)."""
    ci = np.asarray(ci, dtype=np.int64)
    g = ((ci * np.int64(2654435761)) % np.int64(2**32)) % N_GROUPS
    te = np.flatnonzero((g == 0) | (g == 1))
    val_pool = np.flatnonzero(g == 2)
    va = val_pool[: min(val_rows, len(val_pool))]
    tr = np.setdiff1d(np.flatnonzero(g >= 2), va)
    assert len(tr) and len(va) and len(te), (len(tr), len(va), len(te))
    return {"tr": tr, "va": va, "te": te, "n_train_90pct": int(len(tr) + len(va))}


def corpus_transfer_folds(corpus: np.ndarray, val_rows: int) -> dict[str, dict]:
    """Corpus-source grouped folds (LMSYS<->WildChat transfer, both directions)."""
    out = {}
    for train_c, test_c in (("lmsys", "wildchat"), ("wildchat", "lmsys")):
        tr_all = np.flatnonzero(corpus == train_c)
        te = np.flatnonzero(corpus == test_c)
        if len(tr_all) < 3 or len(te) == 0:
            continue
        va = tr_all[: min(val_rows, max(1, len(tr_all) // 10))]
        tr = np.setdiff1d(tr_all, va)
        out[f"train_{train_c}_test_{test_c}"] = {"tr": tr, "va": va, "te": te}
    return out


# ---------------------------------------------------------------------------
# Statistics: pooled R², CKA, rotation nulls, Procrustes, MDS
# ---------------------------------------------------------------------------


def pooled_r2(pred: np.ndarray, y: np.ndarray) -> float:
    """The registered pooled R² read (#779 convention: SS_tot on the eval set's
    own mean), via the reused core."""
    return float(_fit_core()[1](np.asarray(pred), np.asarray(y)))


def cka_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (arXiv 1905.00414, feature-space form):
    ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F). Invariant to orthogonal
    rotation + isotropic scale; descriptive (no fitting)."""
    Xc = np.asarray(X, dtype=np.float64)
    Yc = np.asarray(Y, dtype=np.float64)
    Xc = Xc - Xc.mean(0)
    Yc = Yc - Yc.mean(0)
    num = np.linalg.norm(Xc.T @ Yc) ** 2
    den = np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Yc.T @ Yc)
    return float(num / max(den, 1e-30))


def raw_cosine_with_rotation_null(a: np.ndarray, b: np.ndarray, *, n_draws: int, seed: int) -> dict:
    """Direction-aware raw vec-cosine + two-sided random-rotation chance band —
    the ``issue1345_operator_comparison.raw_cosine_with_rotation_null`` convention
    (L114), via that module (deferred import; torch chain)."""
    import issue1345_operator_comparison as OC

    return OC.raw_cosine_with_rotation_null(
        torch.as_tensor(np.asarray(a), dtype=torch.float64),
        torch.as_tensor(np.asarray(b), dtype=torch.float64),
        n_draws=n_draws,
        seed=seed,
    )


def spectrum_cosine(a: np.ndarray, b: np.ndarray) -> dict:
    """Spectrum cosine (rotation-invariant-ONLY descriptive ceiling — can never
    support "same operator up to rotation"; ``issue1345_operator_comparison
    .spectrum_cosine`` L143). Cross-shape operators are compared on the leading
    min(d) singular values with ``truncated: true`` recorded."""
    sa = torch.linalg.svdvals(torch.as_tensor(np.asarray(a), dtype=torch.float64))
    sb = torch.linalg.svdvals(torch.as_tensor(np.asarray(b), dtype=torch.float64))
    k = min(len(sa), len(sb))
    val = float((sa[:k] * sb[:k]).sum() / (sa[:k].norm() * sb[:k].norm() + 1e-12))
    return {"spectrum_cosine": val, "truncated": bool(len(sa) != len(sb)), "k": int(k)}


def orth_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Semi-orthogonal Procrustes rotation R = U V^T of A^T B (the ``_orth`` inner
    helper of ``issue825_map_alignment._procrustes_cosine_null``): maps B's column
    space onto A's. A (n, d_a), B (n, d_b) -> R (d_a, d_b)."""
    A64 = np.asarray(A, dtype=np.float64)
    B64 = np.asarray(B, dtype=np.float64)
    M = (A64 - A64.mean(0)).T @ (B64 - B64.mean(0))
    U, _s, Vh = np.linalg.svd(M, full_matrices=False)
    return U @ Vh


def mds_2d(dist: np.ndarray) -> np.ndarray:
    """Classical MDS (double-centered Gram eigendecomposition) -> (n, 2) coords.
    Presentation-only (plan: every claim reads off the distance table)."""
    D = np.asarray(dist, dtype=np.float64)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    Bmat = -0.5 * J @ (D**2) @ J
    w, V = np.linalg.eigh(Bmat)
    order = np.argsort(w)[::-1][:2]
    w2 = np.clip(w[order], 0.0, None)
    return V[:, order] * np.sqrt(w2)[None, :]


def _knn(pred: np.ndarray, true: np.ndarray) -> dict:
    """kNN retrieval read with the stated chance floor (mandatory companion)."""
    return knn_retrieval(np.asarray(pred, np.float64), np.asarray(true, np.float64), ks=KNN_KS)


def _identity_bias_read(x_tr, y_tr, x_te, y_te) -> dict:
    """Identity+learned-bias baseline where d_in == d_out; else the mandated
    inapplicability statement (mapping-baselines standing rule)."""
    if np.asarray(x_tr).shape[1] != np.asarray(y_tr).shape[1]:
        return {"applicable": False, "reason": "d_in != d_out (cross-model spaces)"}
    pred = identity_bias_predict(x_tr, y_tr, x_te)
    return {"applicable": True, "r2": pooled_r2(pred, y_te)}


# ---------------------------------------------------------------------------
# Phase: fits
# ---------------------------------------------------------------------------


def _fit_map(name: str, X, Y, folds: dict, dev, *, grid=None, max_widenings=None) -> dict:
    """One val-lambda-selected widened ridge fit + beta payload at the selected
    lambda + held-out reads (R², kNN, identity+bias). Returns a unit record."""
    t0 = time.time()
    tr, va, te = folds["tr"], folds["va"], folds["te"]
    pred_te, meta = fit_val_widened(X, Y, tr, va, te, dev, grid=grid, max_widenings=max_widenings)
    payload = ridge_beta_at_lambda(X, Y, tr, float(meta["selected_lambda"]))
    rec = {
        "name": name,
        "d_in": int(np.asarray(X).shape[1]),
        "d_out": int(np.asarray(Y).shape[1]),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "fit_meta": {k: v for k, v in meta.items() if not isinstance(v, np.ndarray)},
        "test_r2": pooled_r2(pred_te, np.asarray(Y)[te]),
        "knn": _knn(pred_te, np.asarray(Y)[te]),
        "identity_bias": _identity_bias_read(
            np.asarray(X)[tr], np.asarray(Y)[tr], np.asarray(X)[te], np.asarray(Y)[te]
        ),
        "elapsed_s": round(time.time() - t0, 2),
    }
    return {"record": rec, "payload": payload, "pred_te": pred_te}


def _split_half_floor(X, Y, tr: np.ndarray, lam: float) -> dict:
    """Split-half operator noise floor: refit at the SAME selected lambda on two
    disjoint train halves (by position parity — deterministic); floor = raw
    vec-cosine of the two row-action operators (within-operator self-similarity)."""
    h1, h2 = tr[0::2], tr[1::2]
    if len(h1) < 3 or len(h2) < 3:
        return {"floor": None, "reason": "too few train rows for split-half"}
    p1 = ridge_beta_at_lambda(X, Y, h1, lam)
    p2 = ridge_beta_at_lambda(X, Y, h2, lam)
    a1, _ = OP.row_operator(p1)
    a2, _ = OP.row_operator(p2)
    v1, v2 = a1.reshape(-1), a2.reshape(-1)
    cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-30))
    return {"floor": cos, "n_half": [int(len(h1)), int(len(h2))]}


def phase_fits(args) -> None:
    """Tier-1 grid + working-pair fits + composed/native/baseline evaluations.
    Persists per-unit payload files as they complete (checkpoint-per-unit) and
    one fits_summary.json; uploads to the xmodel prefix unless --skip-upload."""
    print("[phase=fits]", flush=True)
    dev = torch.device(args.device)
    layers_by_model = {
        "qwen": sorted({q for q, _ in _pairs(args)}),
        "llama": sorted({ll for _, ll in _pairs(args)}),
    }
    caps = load_captures(args, layers_by_model)
    join = paired_join(caps)
    realized = int(len(join["ci"]))
    production = realized >= PRODUCTION_REALIZED_FLOOR
    folds = grouped_folds(join["ci"], args.val_rows)
    if production:
        assert folds["n_train_90pct"] >= FIT_FLOOR_N_TRAIN, (
            f"n_train {folds['n_train_90pct']} < {FIT_FLOOR_N_TRAIN} (plan §7 C9 floor)"
        )
    else:
        assert folds["n_train_90pct"] >= 8, "smoke fold degenerate"

    def _mat(model: str, tag: str, layer: int) -> np.ndarray:
        b = caps[model][(tag, layer)]
        idx = join["q_idx"] if model == "qwen" else join["l_idx"]
        return b["x"][idx]

    out_dir = Path(args.fits_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {
        "realized_paired_rows": realized,
        "production": production,
        "folds": {k: int(len(v)) if hasattr(v, "__len__") else v for k, v in folds.items()},
        "claim_scope": (
            "transformations of SHARED TEACHER-FORCED QWEN RESPONSES — v_A summarizes "
            "Qwen's banked answer text under both models; nothing is claimed about "
            "Llama's own answer policy (C3)"
        ),
        "tier1": [],
        "metadata": _meta("fits"),
    }
    grid = None if not args.smoke else tuple(np.logspace(-3.0, 6.0, 10))
    t0 = time.time()
    unit = 0
    n_units = 4 * len(_pairs(args)) + 8
    best = None
    for q_layer, l_layer in _pairs(args):
        pair_rec = {"qwen_layer": q_layer, "llama_layer": l_layer, "fits": {}, "cka": {}}
        for tag in ("vc", "va"):
            Xq = _mat("qwen", tag, q_layer)
            Xl = _mat("llama", tag, l_layer)
            for direction, X, Y in (("q2l", Xq, Xl), ("l2q", Xl, Xq)):
                unit += 1
                r = _fit_map(
                    f"align_{tag}_{direction}_L{q_layer}_{l_layer}", X, Y, folds, dev, grid=grid
                )
                pair_rec["fits"][f"{tag}_{direction}"] = r["record"]
                print(
                    f"[fits] unit {unit}/{n_units} {r['record']['name']} "
                    f"val_r2={r['record']['fit_meta']['val_r2_at_selected']:.4f} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
            # CKA on the TRAIN side only (test rows untouched by selection reads)
            pair_rec["cka"][tag] = cka_linear(Xq[folds["tr"]], Xl[folds["tr"]])
        # working-pair selection score: TRAIN-split (val) alignability only (plan)
        pair_rec["selection_val_r2_mean"] = float(
            np.mean([f["fit_meta"]["val_r2_at_selected"] for f in pair_rec["fits"].values()])
        )
        summary["tier1"].append(pair_rec)
        if best is None or pair_rec["selection_val_r2_mean"] > best["selection_val_r2_mean"]:
            best = pair_rec
    summary["working_pair"] = {
        "qwen_layer": best["qwen_layer"],
        "llama_layer": best["llama_layer"],
        "selection": "max mean val R^2 over the 4 alignment fits (TRAIN split only)",
    }
    qL, lL = best["qwen_layer"], best["llama_layer"]

    # -- working-pair payload fits (persist each as it completes) ------------------
    Xq_c, Xq_a = _mat("qwen", "vc", qL), _mat("qwen", "va", qL)
    Xl_c, Xl_a = _mat("llama", "vc", lL), _mat("llama", "va", lL)
    named: dict[str, dict] = {}
    for name, X, Y in (
        (f"align_c_l2q_L{lL}_{qL}", Xl_c, Xq_c),
        (f"align_c_q2l_L{qL}_{lL}", Xq_c, Xl_c),
        (f"align_a_q2l_L{qL}_{lL}", Xq_a, Xl_a),
        (f"align_a_l2q_L{lL}_{qL}", Xl_a, Xq_a),
        (f"llama_native_L{lL}", Xl_c, Xl_a),
        (f"qwen_matched_L{qL}", Xq_c, Xq_a),
    ):
        unit += 1
        r = _fit_map(name, X, Y, folds, dev, grid=grid)
        lam = float(r["record"]["fit_meta"]["selected_lambda"])
        r["record"]["split_half_floor"] = _split_half_floor(X, Y, folds["tr"], lam)
        named[name] = r
        key = "llama_map" if name.startswith("llama_native") else name.split("_L")[0]
        fname = f"llama_map_L{lL}.pt" if name.startswith("llama_native") else None
        if fname:
            _atomic_torch_save(
                {**payload_to_dict(r["payload"]), "layer": lL, "record": r["record"]},
                out_dir / fname,
            )
        print(
            f"[fits] unit {unit}/{n_units} {name} test_r2={r['record']['test_r2']:.4f} "
            f"key={key} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    align_bundle = {
        name: payload_to_dict(named[name]["payload"]) for name in named if name.startswith("align_")
    }
    align_bundle["records"] = {n: named[n]["record"] for n in named if n.startswith("align_")}
    align_bundle["working_pair"] = summary["working_pair"]
    align_bundle["metadata"] = _meta("fits")
    _atomic_torch_save(align_bundle, out_dir / "align_maps.pt")
    _atomic_torch_save(
        {
            **payload_to_dict(named[f"qwen_matched_L{qL}"]["payload"]),
            "layer": qL,
            "record": named[f"qwen_matched_L{qL}"]["record"],
        },
        out_dir / f"qwen_matched_L{qL}.pt",
    )
    summary["working_pair_fits"] = {n: named[n]["record"] for n in named}

    # -- corpus-transfer read (grouped-generalization; native + one alignment) ----
    summary["corpus_transfer"] = {}
    for fold_name, f2 in corpus_transfer_folds(join["corpus"], args.val_rows).items():
        rec_n = _fit_map(f"llama_native_{fold_name}", Xl_c, Xl_a, f2, dev, grid=grid)
        rec_a = _fit_map(f"align_c_q2l_{fold_name}", Xq_c, Xl_c, f2, dev, grid=grid)
        summary["corpus_transfer"][fold_name] = {
            "llama_native_r2": rec_n["record"]["test_r2"],
            "align_c_q2l_r2": rec_a["record"]["test_r2"],
        }
        print(
            f"[fits] corpus-transfer {fold_name} done elapsed={time.time() - t0:.0f}s", flush=True
        )

    # -- tier 2/3 routes on the held-out test rows ---------------------------------
    te = folds["te"]
    a_qwen = _load_qwen_operator(args, qL, realized)
    routes: dict[str, dict] = {}
    x_in = Xl_c[te]
    y_true = Xl_a[te]
    chains = {
        "native": lambda v: OP.predict(named[f"llama_native_L{lL}"]["payload"], v),
        "composed_banked": lambda v: OP.predict(
            named[f"align_a_q2l_L{qL}_{lL}"]["payload"],
            OP.predict(a_qwen, OP.predict(named[f"align_c_l2q_L{lL}_{qL}"]["payload"], v)),
        ),
        "composed_matched": lambda v: OP.predict(
            named[f"align_a_q2l_L{qL}_{lL}"]["payload"],
            OP.predict(
                named[f"qwen_matched_L{qL}"]["payload"],
                OP.predict(named[f"align_c_l2q_L{lL}_{qL}"]["payload"], v),
            ),
        ),
        "alignment_only_baseline": lambda v: OP.predict(
            named[f"align_a_q2l_L{qL}_{lL}"]["payload"],
            OP.predict(named[f"align_c_l2q_L{lL}_{qL}"]["payload"], v),
        ),
    }
    for rname, fn in chains.items():
        pred = fn(x_in)
        routes[rname] = {"r2": pooled_r2(pred, y_true), "knn": _knn(pred, y_true)}
        print(f"[fits] route {rname} r2={routes[rname]['r2']:.4f}", flush=True)
    routes["a_qwen_source"] = a_qwen_source(a_qwen)
    summary["tier2_routes"] = routes
    summary["tier3_diagnostics"] = {
        "label": (
            "NON-IDENTIFYING DIAGNOSTICS (C1): the alignment-only route inserts an "
            "identity between semantically different states; these do NOT apportion "
            "the gap into representation-vs-operator shares; no registered verdict "
            "consumes them"
        ),
        "r2_native_minus_composed_banked": routes["native"]["r2"] - routes["composed_banked"]["r2"],
        "r2_native_minus_composed_matched": routes["native"]["r2"]
        - routes["composed_matched"]["r2"],
        "r2_alignment_only_baseline": routes["alignment_only_baseline"]["r2"],
    }

    # -- correspondence test (pre-registered): v_C-align vs v_A-align maps ---------
    ac, _ = OP.row_operator(named[f"align_c_q2l_L{qL}_{lL}"]["payload"])
    aa, _ = OP.row_operator(named[f"align_a_q2l_L{qL}_{lL}"]["payload"])
    corr = raw_cosine_with_rotation_null(ac, aa, n_draws=args.null_draws, seed=20250825)
    above = corr["raw_cosine"] > corr["rotation_null"]["null_p975"]
    summary["correspondence_test"] = {
        **corr,
        "agree_above_rotation_null": bool(above),
        "consequence": (
            "one similarity transform exists; cross-model EIGENVALUE comparison is well-posed"
            if above
            else "eigenvalues are reported per-model only (pre-registered)"
        ),
        "statistic_class": "direction-aware (raw cosine vs two-sided rotation null)",
    }

    _atomic_json(out_dir / "fits_summary.json", summary)
    print(f"[fits] done wall={time.time() - t0:.0f}s", flush=True)
    if not args.skip_upload:
        names = [
            f"llama_map_L{lL}.pt",
            "align_maps.pt",
            f"qwen_matched_L{qL}.pt",
            "fits_summary.json",
        ]
        url = hub._upload_folder_filtered(
            out_dir,
            repo_id=args.hf_data_repo,
            repo_type="dataset",
            path_in_repo=HF_XMODEL_PREFIX,
            allow_patterns=names,
            expected_repo_paths=[f"{HF_XMODEL_PREFIX}/{n}" for n in names],
        )
        if not url:
            raise RuntimeError(f"fits upload to {HF_XMODEL_PREFIX} returned no URL")
        print(f"[fits] uploaded {len(names)} files -> {HF_XMODEL_PREFIX}", flush=True)


def _pairs(args) -> tuple[tuple[int, int], ...]:
    """Matched (qwen, llama) layer pairs (csv override for smoke/selftest)."""
    if args.pairs:
        out = []
        for tokpair in args.pairs.split(";"):
            q, ll = tokpair.split(",")
            out.append((int(q), int(ll)))
        return tuple(out)
    return MATCHED_PAIRS


def _load_qwen_operator(args, q_layer: int, realized: int) -> OP.MapPayload:
    """The A_qwen used inside the composed route: the banked n1m ridge at the
    working layer (the headline object; validated by OP.load_banked_map). Under
    --synthetic-qwen-map (selftest only, no banked artifact at synthetic d) the
    matched comparator payload path is used by the caller instead."""
    assert not args.synthetic_qwen_map, "synthetic map is resolved by the caller (selftest)"
    payload = OP.load_banked_map(q_layer, root=args.map_root or None)
    OP.run_driver_identity_asserts(payload)  # B1 entry asserts (raise = HALT)
    return payload


def a_qwen_source(payload: OP.MapPayload) -> dict:
    """Provenance stamp for the composed route's center operator."""
    return {
        "layer": int(payload.layer),
        "path": str(payload.path),
        "selected_lambda": float(payload.selected_lambda),
    }


# ---------------------------------------------------------------------------
# Phase: report (three_tier.json)
# ---------------------------------------------------------------------------


def phase_report(args) -> None:
    """Assemble ``eval_results/issue_2569/leg7/three_tier.json`` from fits_summary
    (P-E; the fits phase carries every number — this phase shapes + labels)."""
    print("[phase=report]", flush=True)
    fits_path = Path(args.fits_dir) / "fits_summary.json"
    if not fits_path.exists() and args.stage_from_hf:
        hub.stage_hub_file(
            args.hf_data_repo,
            f"{HF_XMODEL_PREFIX}/fits_summary.json",
            fits_path,
            repo_type="dataset",
        )
    assert fits_path.exists(), f"{fits_path} missing — run --phase fits first"
    s = json.loads(fits_path.read_text())
    report = {
        "issue": TASK_ID,
        "claim_scope": s["claim_scope"],
        "realized_paired_rows": s["realized_paired_rows"],
        "working_pair": s["working_pair"],
        "tier1_alignability": {
            "grid": s["tier1"],
            "note": (
                "alignment R^2 (val-selected widened ridge, held-out test) + linear "
                "CKA (arXiv 1905.00414; train rows); working pair selected on the "
                "TRAIN split (val R^2) only"
            ),
        },
        "tier2_operator_similarity": {
            "routes": s["tier2_routes"],
            "correspondence_test": s["correspondence_test"],
            "anchor_825_aligned_cosine": ANCHOR_825_ALIGNED_COSINE,
            "note": (
                "held-out R^2 of the composed route (align_C -> A_qwen -> align_A) "
                "vs the native Llama map, under fixed data-pinned alignments; the "
                "matched-n Qwen comparator (C3) controls for fit-n"
            ),
        },
        "tier3_diagnostics": s["tier3_diagnostics"],
        "corpus_transfer": s.get("corpus_transfer", {}),
        "working_pair_fits": s["working_pair_fits"],
        "metadata": _meta("report"),
    }
    out = Path(args.leg7_dir) / "three_tier.json"
    _atomic_json(out, report)
    print(f"[report] wrote {out}", flush=True)


# ---------------------------------------------------------------------------
# Phase: atlas (operator rows + pairwise distance table + MDS)
# ---------------------------------------------------------------------------


def _resolve_atlas_rows(args) -> tuple[list[dict], list[dict]]:
    """Resolve every plan-named atlas row or DROP it with a named reason (plan §4
    leg 7 step 4: a miss drops the row, never a silent substitute).

    Returns (rows, dropped). Each row: {name, basis ('qwen'|'llama'), A (d x d or
    d_in x d_out np), floor (dict|None), floor_label, source}."""
    rows: list[dict] = []
    dropped: list[dict] = []

    def _drop(name: str, reason: str) -> None:
        dropped.append({"name": name, "reason": reason})
        logger.warning("[atlas] DROP row %s — %s", name, reason)

    # (1) banked n1m maps (B6 contract; B1 asserts at entry for the first one)
    for i, layer in enumerate(OP.N1M_LAYERS):
        try:
            p = OP.load_banked_map(layer, root=args.map_root or None)
            if i == 0:
                OP.run_driver_identity_asserts(p)
            A, _b = OP.row_operator(p)
            rows.append(
                {
                    "name": f"n1m_L{layer}",
                    "basis": "qwen",
                    "A": A,
                    "floor": None,
                    "floor_label": "no floor — banked (refit rows not staged)",
                    "source": str(p.path),
                }
            )
        except (FileNotFoundError, RuntimeError, AssertionError) as e:
            _drop(f"n1m_L{layer}", f"banked map unresolved: {e}")

    # (2) #2474 pass-B base maps (B6-conditional, staged from the pinned prefix)
    if not args.skip_passb:
        for layer in PASSB_LAYERS:
            name = f"passb_L{layer}"
            try:
                dest = Path(args.fits_dir) / "stage_passb" / f"base_L{layer}.pt"
                if not dest.exists():
                    hub.stage_hub_file(
                        args.hf_data_repo,
                        f"{PASSB_PREFIX}/base_L{layer}.pt",
                        dest,
                        repo_type="dataset",
                    )
                d = torch.load(dest, map_location="cpu", weights_only=False)
                p = payload_from_dict(d, path=dest)
                A, _b = OP.row_operator(p)
                rows.append(
                    {
                        "name": name,
                        "basis": "qwen",
                        "A": A,
                        "floor": None,
                        "floor_label": "no floor — banked (n_train=4,500 pass-B fit)",
                        "source": f"{PASSB_PREFIX}/base_L{layer}.pt",
                    }
                )
            except Exception as e:  # drop-on-miss is the PLAN-sanctioned disposition
                _drop(name, f"pass-B map unresolved at P-E entry: {e}")

    # (3) fitted working-pair operators (llama basis) + matched comparator (qwen)
    fits_dir = Path(args.fits_dir)
    summary_path = fits_dir / "fits_summary.json"
    fitted: dict[str, OP.MapPayload] = {}
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        qL = int(s["working_pair"]["qwen_layer"])
        lL = int(s["working_pair"]["llama_layer"])
        for name, fname, basis in (
            (f"llama_native_L{lL}", f"llama_map_L{lL}.pt", "llama"),
            (f"qwen_matched_L{qL}", f"qwen_matched_L{qL}.pt", "qwen"),
        ):
            path = fits_dir / fname
            if not path.exists():
                _drop(name, f"fitted payload missing: {path}")
                continue
            d = torch.load(path, map_location="cpu", weights_only=False)
            p = payload_from_dict(d, path=path)
            fitted[name] = p
            A, _b = OP.row_operator(p)
            floor = d.get("record", {}).get("split_half_floor")
            rows.append(
                {
                    "name": name,
                    "basis": basis,
                    "A": A,
                    "floor": floor,
                    "floor_label": "split-half refit at the selected lambda",
                    "source": str(path),
                }
            )
        # composed operator in the llama basis: align_c_l2q -> A_qwen -> align_a_q2l
        align_path = fits_dir / "align_maps.pt"
        if align_path.exists() and f"qwen_matched_L{qL}" in fitted:
            al = torch.load(align_path, map_location="cpu", weights_only=False)
            try:
                a_c = payload_from_dict(al[f"align_c_l2q_L{lL}_{qL}"], path=align_path)
                a_a = payload_from_dict(al[f"align_a_q2l_L{qL}_{lL}"], path=align_path)
                center = _atlas_center_operator(
                    args, qL, fitted[f"qwen_matched_L{qL}"], expected_d=int(a_c.W.shape[1])
                )
                Ac, _ = OP.row_operator(a_c)
                Am, _ = OP.row_operator(center["payload"])
                Aa, _ = OP.row_operator(a_a)
                rows.append(
                    {
                        "name": f"composed_L{lL}",
                        "basis": "llama",
                        "A": Ac @ Am @ Aa,
                        "floor": None,
                        "floor_label": "no floor — composition of three fitted maps",
                        "source": f"{align_path} (center: {center['source']})",
                    }
                )
            except KeyError as e:
                _drop(f"composed_L{lL}", f"align_maps.pt missing component: {e}")
        elif not align_path.exists():
            _drop("composed", f"align_maps.pt missing under {fits_dir}")
    else:
        _drop("llama_native/qwen_matched/composed", f"fits_summary.json missing: {summary_path}")

    # (4) leg-4 feature-map linearized residual composition (re-derive route)
    if args.featmap_dir:
        try:
            rows.append(_featmap_row(args))
        except Exception as e:
            _drop("featmap_L19", f"re-derive failed: {e}")
    else:
        _drop("featmap_L19", "no --featmap-dir (leg-4 P-B outputs not staged)")

    # (5) leg-6 write maps (#1900/#1979 store): resolvable ONLY if a factors
    # sidecar exists — leg-6 JSON records persist ranks/matches, not factor vectors
    # (concern leg7-atlas-writemap-operators-unpersisted).
    if args.leg6_dir:
        found = sorted(Path(args.leg6_dir).glob("*/operator_factors.pt"))
        if not found:
            _drop(
                "leg6_write_maps",
                "leg6 records persist rank/match tables without factor vectors; no "
                "operator_factors.pt sidecar found (raised concern "
                "leg7-atlas-writemap-operators-unpersisted)",
            )
        for path in found:
            d = torch.load(path, map_location="cpu", weights_only=False)
            u = np.asarray(d["u"], dtype=np.float64)  # (d, k) read directions
            s_vals = np.asarray(d["s"], dtype=np.float64)
            v = np.asarray(d["v"], dtype=np.float64)  # (d, k) write directions
            rows.append(
                {
                    "name": f"leg6_wmap_{path.parent.name}",
                    "basis": "qwen",
                    "A": (u * s_vals[None, :]) @ v.T,
                    "floor": d.get("split_half_floor"),
                    "floor_label": "leg6 sidecar",
                    "source": str(path),
                }
            )
    else:
        _drop("leg6_write_maps", "no --leg6-dir (leg-6 P-A outputs not staged)")

    # (6) #2378 operators (soft dependency; fallback: atlas ships without them)
    if args.i2378_maps:
        for path in sorted(Path(args.i2378_maps).glob("*.pt")):
            d = torch.load(path, map_location="cpu", weights_only=False)
            p = payload_from_dict(d, path=path)
            A, _b = OP.row_operator(p)
            rows.append(
                {
                    "name": f"i2378_{path.stem}",
                    "basis": "qwen",
                    "A": A,
                    "floor": None,
                    "floor_label": "no floor — descriptive only",
                    "source": str(path),
                }
            )
    else:
        _drop("i2378_operators", "soft dependency not parked (add-on follow-up slot)")

    assert rows, "no atlas rows resolved — nothing to embed"
    return rows, dropped


def _atlas_center_operator(args, q_layer: int, matched: OP.MapPayload, *, expected_d: int) -> dict:
    """Composed-route center operator for the ATLAS row: the banked n1m map when
    resolvable AND shape-compatible with the alignment maps (``expected_d``), else
    the matched comparator — the substitution is RECORDED, never silent."""
    try:
        p = OP.load_banked_map(q_layer, root=args.map_root or None)
        if int(p.W.shape[0]) != int(expected_d):
            return {
                "payload": matched,
                "source": (
                    f"matched comparator (banked d={p.W.shape[0]} != alignment "
                    f"d={expected_d} — synthetic/smoke shape)"
                ),
            }
        return {"payload": p, "source": f"banked n1m L{q_layer}"}
    except (FileNotFoundError, RuntimeError):
        return {"payload": matched, "source": "matched comparator (banked map unresolved)"}


def _featmap_row(args) -> dict:
    """Re-derive the leg-4 feature map at the RECORDED lambda from the Unit-4b
    persisted encodes, then compose the linearized residual operator
    ``A_feat = E_alive @ diag(1/xsd) @ W @ D_union`` (row action; gates ignored).

    Inputs (all persisted by ``issue2569_rowbattery.phase_feature_map``):
    ``counts_ctx.npy`` (alive floors), ``x_ctx_alive.fp16.npy`` /
    ``y_union.fp16.npy`` (encodes over [fit, val, te] rows), ``enc_meta.json``
    (row counts + union width), ``feature_map_metrics.json`` (the 1%-floor route's
    ``selected_lambda``); the ctx SAE (``sae_ctx/ae.pt``) + banked answer SAE for
    the encoder/decoder rows."""
    import math

    import issue2569_rowbattery as RB

    fdir = Path(args.featmap_dir)
    enc_meta = json.loads((fdir / "enc_meta.json").read_text())
    metrics = json.loads((fdir / "feature_map_metrics.json").read_text())
    fitted = next(r for r in metrics["routes"] if r.get("route", r.get("name")) == "fitted_map")
    lam = float(fitted["fit_meta"]["selected_lambda"])
    n_fit, _n_val, _n_te = (int(x) for x in enc_meta["rows"])
    counts = np.load(fdir / "counts_ctx.npy")
    floor_frac = float(RB.LEG4_CTX_FLOOR_FRACS[0])  # 1% primary floor
    floor_rows = max(1, math.ceil(floor_frac * n_fit))
    alive_1pct = np.flatnonzero(counts >= floor_rows)
    loose_frac = float(RB.LEG4_CTX_FLOOR_FRACS[-1])
    alive_loose = np.flatnonzero(counts >= max(1, math.ceil(loose_frac * n_fit)))
    cols = np.searchsorted(alive_loose, alive_1pct)
    assert (alive_loose[cols] == alive_1pct).all(), "1% floor not a subset of loose alive"
    x_alive = np.load(fdir / "x_ctx_alive.fp16.npy", mmap_mode="r")
    y_union = np.load(fdir / "y_union.fp16.npy", mmap_mode="r")
    X = np.asarray(x_alive[:, cols], dtype=np.float32)
    Y = np.asarray(y_union, dtype=np.float32)
    tr = np.arange(n_fit)
    payload = ridge_beta_at_lambda(X, Y, tr, lam)
    A_mid, _b = OP.row_operator(payload)  # (n_alive_1pct, n_union)

    sae_ctx = RB.load_sae_ctx(Path(args.sae_ctx_path), device="cpu")
    E = sae_ctx.w_enc.detach().to(torch.float64).numpy()[:, alive_1pct]  # (d, n_alive)
    import issue2476_turnavg_sae as T24

    sae_ans = T24.MatryoshkaBatchTopKSAE.load_local(Path(args.answer_sae_dir), device="cpu")
    union = np.asarray(metrics.get("feat_ids", []), dtype=np.int64)
    if union.size == 0:
        npz = np.load(fdir / "perfeature_leg4.npz")
        union = np.asarray(npz["feat_ids"], dtype=np.int64)
    D = sae_ans.w_dec.detach().to(torch.float64).numpy()[union, :]  # (n_union, d)
    assert A_mid.shape == (len(alive_1pct), len(union)), (A_mid.shape, len(alive_1pct), len(union))
    A_feat = E @ A_mid @ D  # (d, d) row action: vhat = v @ A_feat
    floor = _split_half_floor(X, Y, tr, lam)
    return {
        "name": "featmap_L19",
        "basis": "qwen",
        "A": A_feat,
        "floor": floor,
        "floor_label": "split-half refit of M_feat at the recorded lambda (linearized)",
        "source": f"{fdir} (re-derived at recorded lambda={lam:g}; ReLU/TopK gates ignored)",
    }


def _activation_procrustes(args, qL: int, lL: int) -> dict | None:
    """Semi-orthogonal basis alignments (R_in from paired v_C, R_out from paired
    v_A train rows) for mapping llama-basis operators into the qwen basis:
    ``A_l_in_q = R_in @ A_l @ R_out.T`` (row action; #825
    ``_procrustes_cosine_null`` construction). None when captures are not staged."""
    try:
        caps = load_captures(args, {"qwen": [qL], "llama": [lL]})
    except AssertionError as e:
        logger.warning("[atlas] no paired captures for Procrustes alignment: %s", e)
        return None
    join = paired_join(caps)
    folds = grouped_folds(join["ci"], args.val_rows)
    tr = folds["tr"]
    Xq_c = caps["qwen"][("vc", qL)]["x"][join["q_idx"]][tr]
    Xl_c = caps["llama"][("vc", lL)]["x"][join["l_idx"]][tr]
    Yq_a = caps["qwen"][("va", qL)]["x"][join["q_idx"]][tr]
    Yl_a = caps["llama"][("va", lL)]["x"][join["l_idx"]][tr]
    return {
        "R_in": orth_procrustes(Xq_c, Xl_c),  # (d_q, d_l)
        "R_out": orth_procrustes(Yq_a, Yl_a),  # (d_q, d_l)
        "n_rows": int(len(tr)),
    }


def phase_atlas(args) -> None:
    """Operator atlas: resolve rows, align bases, pairwise statistics (every
    statistic class-labeled), split-half floors, MDS coords (presentation-only).
    Writes ``eval_results/issue_2569/leg7/atlas_distances.json``."""
    print("[phase=atlas]", flush=True)
    rows, dropped = _resolve_atlas_rows(args)

    # basis alignment for llama-basis rows (activation Procrustes, where staged)
    proc = None
    llama_rows = [r for r in rows if r["basis"] == "llama"]
    if llama_rows:
        summary_path = Path(args.fits_dir) / "fits_summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            proc = _activation_procrustes(
                args, int(s["working_pair"]["qwen_layer"]), int(s["working_pair"]["llama_layer"])
            )
        if proc is None:
            for r in llama_rows:
                r["aligned_A"] = None
        else:
            for r in llama_rows:
                r["aligned_A"] = proc["R_in"] @ r["A"] @ proc["R_out"].T
    for r in rows:
        if r["basis"] == "qwen":
            r["aligned_A"] = r["A"]

    table = []
    t0 = time.time()
    n_pairs = len(rows) * (len(rows) - 1) // 2
    k = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            k += 1
            a, b = rows[i], rows[j]
            entry = {
                "pair": [a["name"], b["name"]],
                "bases": [a["basis"], b["basis"]],
                "spectrum": {
                    **spectrum_cosine(a["A"], b["A"]),
                    "statistic_class": "spectrum/rotation-invariant-only (descriptive ceiling)",
                },
            }
            if a.get("aligned_A") is not None and b.get("aligned_A") is not None:
                same_basis = a["basis"] == b["basis"]
                cos = raw_cosine_with_rotation_null(
                    a["aligned_A"], b["aligned_A"], n_draws=args.null_draws, seed=1345 + k
                )
                entry["cosine"] = {
                    **cos,
                    "statistic_class": (
                        "direction-aware (raw cosine vs rotation null; same basis)"
                        if same_basis
                        else "direction-aware under the FIXED activation-Procrustes "
                        "alignment (anchor alignment applied before cross-basis "
                        "comparison)"
                    ),
                }
            else:
                entry["cosine"] = None
            table.append(entry)
            print(
                f"[atlas] pair {k}/{n_pairs} {a['name']}~{b['name']} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )

    # MDS (presentation-only) over 1 - aligned cosine (pairs without an aligned
    # cosine fall back to 1 - spectrum cosine, flagged).
    n = len(rows)
    D = np.zeros((n, n))
    fallback_pairs = []
    for entry in table:
        i = next(ix for ix, r in enumerate(rows) if r["name"] == entry["pair"][0])
        j = next(ix for ix, r in enumerate(rows) if r["name"] == entry["pair"][1])
        if entry["cosine"] is not None:
            d = 1.0 - float(entry["cosine"]["raw_cosine"])
        else:
            d = 1.0 - float(entry["spectrum"]["spectrum_cosine"])
            fallback_pairs.append(entry["pair"])
        D[i, j] = D[j, i] = max(0.0, d)
    coords = mds_2d(D)

    out = {
        "issue": TASK_ID,
        "rows": [
            {
                "name": r["name"],
                "basis": r["basis"],
                "shape": list(np.asarray(r["A"]).shape),
                "floor": r["floor"],
                "floor_label": r["floor_label"],
                "source": r["source"],
                "procrustes_aligned": bool(
                    r.get("aligned_A") is not None and r["basis"] == "llama"
                ),
            }
            for r in rows
        ],
        "dropped_rows": dropped,
        "distance_table": table,
        "procrustes": {"available": proc is not None, "n_rows": proc["n_rows"] if proc else 0},
        "mds_2d": {
            "coords": {r["name"]: [float(c) for c in coords[ix]] for ix, r in enumerate(rows)},
            "note": "presentation-only (classical MDS on 1 - aligned cosine); every "
            "claim reads off the distance table",
            "spectrum_fallback_pairs": fallback_pairs,
        },
        "anchor_825_aligned_cosine": ANCHOR_825_ALIGNED_COSINE,
        "metadata": _meta("atlas"),
    }
    path = Path(args.leg7_dir) / "atlas_distances.json"
    _atomic_json(path, out)
    print(f"[atlas] wrote {path} rows={len(rows)} dropped={len(dropped)}", flush=True)


# ---------------------------------------------------------------------------
# Phase: selftest (tiny synthetic end-to-end; the committed CPU smoke)
# ---------------------------------------------------------------------------


def _make_synthetic_captures(root: Path, *, n: int, d_q: int, d_l: int, seed: int) -> None:
    """Write tiny synthetic finalized capture bundles (both models, one matched
    pair 19,22) with a real linear cross-model structure + noise."""
    import issue2569_xmodel_capture as XC

    rng = np.random.default_rng(seed)
    ci = np.arange(n, dtype=np.int64) * 3 + 1
    corpus = ["lmsys" if i % 2 == 0 else "wildchat" for i in range(n)]
    z = rng.standard_normal((n, d_q)).astype(np.float32)
    Mq = rng.standard_normal((d_q, d_q)).astype(np.float32) / np.sqrt(d_q)
    T = rng.standard_normal((d_q, d_l)).astype(np.float32) / np.sqrt(d_q)
    data = {
        ("qwen", "vc", 19): z,
        ("qwen", "va", 19): z @ Mq + 0.05 * rng.standard_normal((n, d_q)).astype(np.float32),
        ("llama", "vc", 22): z @ T + 0.05 * rng.standard_normal((n, d_l)).astype(np.float32),
    }
    data[("llama", "va", 22)] = data[("qwen", "va", 19)] @ T + 0.05 * rng.standard_normal(
        (n, d_l)
    ).astype(np.float32)
    root.mkdir(parents=True, exist_ok=True)
    for (model, tag, layer), x in data.items():
        arr, codec = XC.encode_summary(x)
        with atomic_replace(root / f"{model}_{tag}_L{layer}.pt") as tmp:
            torch.save(
                {
                    "x": arr,
                    "codec": codec,
                    "ci": ci,
                    "corpus": corpus,
                    "layer": layer,
                    "slot": "v_C" if tag == "vc" else "v_A",
                    "model_id": f"synthetic/{model}",
                    "template_sha": "selftest",
                    "drops": {},
                    "n_selected_texts": n,
                    "metadata": {"phase": "selftest"},
                },
                tmp,
            )


def phase_selftest(args) -> None:
    """Tiny-synthetic end-to-end fits -> report -> atlas (CPU, minutes). Also
    asserts the beta-payload path reproduces the reused core's predictions at the
    selected lambda (the ridge_beta_at_lambda equivalence contract)."""
    print("[phase=selftest]", flush=True)
    import tempfile

    work = Path(tempfile.mkdtemp(prefix="i2569-atlas-selftest-"))
    cap = work / "captures"
    _make_synthetic_captures(cap, n=240, d_q=8, d_l=12, seed=0)
    argv = [
        "--phase",
        "fits",
        "--capture-dir",
        str(cap),
        "--fits-dir",
        str(work / "fits"),
        "--leg7-dir",
        str(work / "leg7"),
        "--pairs",
        "19,22",
        "--device",
        "cpu",
        "--val-rows",
        "24",
        "--null-draws",
        "24",
        "--smoke",
        "--skip-upload",
        "--skip-passb",
        "--synthetic-qwen-map",
        "--map-root",
        str(work / "no-banked-maps-here"),  # banked rows DROP deterministically
    ]
    a2 = _parse_args(argv)
    # synthetic-qwen-map: substitute the matched comparator for the banked map
    phase_fits_synthetic(a2)
    a2.phase = "report"
    phase_report(a2)
    a2.phase = "atlas"
    phase_atlas(a2)
    tt = json.loads((work / "leg7" / "three_tier.json").read_text())
    at = json.loads((work / "leg7" / "atlas_distances.json").read_text())
    assert tt["tier2_routes"]["native"]["r2"] > 0.5, tt["tier2_routes"]["native"]["r2"]
    assert at["rows"], "selftest atlas resolved no rows"
    # ridge_beta_at_lambda equivalence vs the reused core at a fixed lambda
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 6))
    Y = rng.standard_normal((60, 4))
    tr, va, te = np.arange(40), np.arange(40, 50), np.arange(50, 60)
    fit_fn = _fit_core()[0]
    lam = 1.0
    pred_core, _m = fit_fn(X, Y, tr, va, te, [lam], torch.device("cpu"))
    payload = ridge_beta_at_lambda(X, Y, tr, lam)
    pred_beta = OP.predict(payload, X[te])
    assert np.allclose(pred_core, pred_beta, rtol=1e-8, atol=1e-8), (
        "beta path diverges from the reused core at fixed lambda"
    )
    print(f"[selftest] PASS (workdir {work})", flush=True)


def phase_fits_synthetic(args) -> None:
    """Selftest fits: identical to phase_fits but the composed route's center
    operator is the matched comparator (no banked map exists at synthetic d).
    Implemented by monkeypatching the loader seam INSIDE this process only."""
    global _load_qwen_operator
    real = _load_qwen_operator
    try:
        _load_qwen_operator = _synthetic_loader_wrapper
        phase_fits(args)
    finally:
        _load_qwen_operator = real


def _synthetic_loader_wrapper(args, q_layer: int, realized: int) -> OP.MapPayload:
    """Selftest center-operator: refit the matched comparator payload from the
    synthetic captures (same shape contract as the banked map)."""
    caps = load_captures(args, {"qwen": [q_layer]})
    join_q = caps["qwen"][("vc", q_layer)]
    X = join_q["x"]
    Y = caps["qwen"][("va", q_layer)]["x"]
    tr = np.arange(int(0.8 * len(X)))
    return ridge_beta_at_lambda(X, Y, tr, 1.0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PHASES = {
    "fits": phase_fits,
    "report": phase_report,
    "atlas": phase_atlas,
    "selftest": phase_selftest,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI (argparse — per-issue phase-dispatch driver convention, code-style.md)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=sorted(PHASES), default=None)
    ap.add_argument("--import-check", action="store_true", help="argcheck + exit 0")
    ap.add_argument(
        "--capture-dir",
        default=str(PROJECT_ROOT / "data" / "issue_2569" / "xmodel" / "final"),
        help="finalized capture bundles ({model}_{vc,va}_L{K}.pt)",
    )
    ap.add_argument(
        "--fits-dir", default=str(PROJECT_ROOT / "data" / "issue_2569" / "xmodel" / "fits")
    )
    ap.add_argument("--leg7-dir", default=str(LEG7_DIR))
    ap.add_argument("--pairs", default="", help="';'-joined 'q,l' matched layer pairs (override)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-rows", type=int, default=512)
    ap.add_argument("--null-draws", type=int, default=200)
    ap.add_argument("--hf-data-repo", default="superkaiba1/explore-persona-space-data")
    ap.add_argument("--stage-from-hf", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="short lambda grid (selftest)")
    ap.add_argument("--map-root", default="", help="banked n1m map root override (pods)")
    ap.add_argument("--skip-passb", action="store_true")
    ap.add_argument(
        "--synthetic-qwen-map",
        action="store_true",
        help="selftest only: matched comparator as the composed-route center",
    )
    ap.add_argument("--featmap-dir", default="", help="leg-4 P-B out_root/leg4 (encodes + metrics)")
    ap.add_argument("--sae-ctx-path", default="", help="fresh ctx SAE ae.pt (leg-4 P-B output)")
    ap.add_argument("--answer-sae-dir", default="", help="banked answer SAE local dir")
    ap.add_argument("--leg6-dir", default="", help="leg-6 P-A out_root/leg6 (write-map sidecars)")
    ap.add_argument("--i2378-maps", default="", help="#2378 operator payload dir (soft dep)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point: dispatch one phase; explicit flush + exit 0 (#1689 atexit rc)."""
    args = _parse_args(argv)
    if args.import_check:
        # Execute EVERY deferred (function-body) import on the real branch —
        # argcheck alone does not resolve them (#1689 false-pass class).
        import math as _math  # noqa: F401
        import tempfile as _tempfile  # noqa: F401

        import issue1345_operator_comparison as _oc  # noqa: F401
        import issue2476_turnavg_sae as _t24  # noqa: F401
        import issue2569_gateladder as _gl  # noqa: F401
        import issue2569_rowbattery as _rb  # noqa: F401
        import issue2569_xmodel_capture as _xc  # noqa: F401

        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    assert args.phase, "--phase is required (or --import-check)"
    PHASES[args.phase](args)
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
