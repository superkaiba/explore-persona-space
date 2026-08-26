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
  The pre-registered H7 demote branch (plan §7.5 leg 7) is COMPUTED here and the
  verdict written into the artifact (``h7_demote``): per evaluable pair,
  within-operator split-half distance (1 - floor cosine, max over the two members)
  vs between-operator distance (1 - aligned raw cosine, matched vec-cosine units);
  noise-dominated for > half of evaluable pairs -> demote (descriptive only).
  Pair statistics are BATCHED (concern atlas-pair-loop-unbatched-unresumable):
  one SVD per row + ONE shared Haar draw set serving every pair's rotation null
  (``shared_rotation_null_draws`` — exact in distribution via Haar invariance),
  checkpointed per row / per draw-chunk under ``<fits-dir>/atlas_ckpt`` with
  content-fingerprint resume keys (never status strings).
  Writes ``eval_results/issue_2569/leg7/atlas_distances.json``.
- ``--phase selftest``: tiny-synthetic end-to-end pass (fits -> report -> atlas at
  d=32/48, n~240) — the committed CPU smoke for the whole assembly path.

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


def _file_sha(path: Path) -> str:
    """sha256 of on-disk file bytes — the content fingerprint for resume keys
    (#1336-safe: bit-exact bytes READ FROM A FILE, never a recomputed float array)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def shared_rotation_null_draws(
    spectra_unit: np.ndarray,
    *,
    n_draws: int,
    seed: int,
    device: str = "cpu",
    chunk_draws: int = 25,
    ckpt_dir: Path | None = None,
    regime_key: str = "",
) -> np.ndarray:
    """Batched EXACT form of the ``issue1345_operator_comparison
    .raw_cosine_with_rotation_null`` two-sided rotation null — ONE shared draw set
    serving every square-operator pair (concern atlas-pair-loop-unbatched-unresumable).

    Identity (equality in distribution via Haar invariance; pinned by
    ``tests/test_issue2569_atlas.py::test_shared_null_algebraic_identity``): for
    square A = Ua Sa Va^T, B = Ub Sb Vb^T (full SVDs) and Haar Q1, Q2,

        cos(vec(A), vec(Q1^T B Q2))
          = tr(Sa [Ua^T Q1^T Ub] Sb [Vb^T Q2 Va]) / (||A||_F ||B||_F)
          =d sa_hat^T (G1 * G2^T) sb_hat,   G1, G2 iid Haar O(d),

    because Ua^T Q1^T Ub and Vb^T Q2 Va are themselves iid Haar (two-sided
    invariance) and ||Q1^T B Q2||_F == ||B||_F. Cost: 2 Haar QRs per draw TOTAL
    (shared across ALL pairs) instead of 2 QRs + 2 dense d^3 GEMMs per draw PER
    PAIR. Haar samples come from ``issue825_map_alignment._random_orthogonal``
    (gaussian QR + sign fix; CPU gaussian from the seeded generator, QR on
    ``device``) — the SAME construction the serial convention uses. Draws are
    therefore shared across pairs (pairs' null bands are correlated; each pair
    still receives ``n_draws`` draws from its exact null distribution).

    ``spectra_unit``: (n_rows, d) — row i = svdvals(A_i) / ||A_i||_F. Returns
    X (n_draws, n, n): X[t, i, j] = one null draw of the pair (i, j) cosine with
    row j in the rotated (beta_b) slot. Draws are generated in chunks of
    ``chunk_draws`` (per-chunk generator seed = seed + chunk index) and, when
    ``ckpt_dir`` is set, each chunk is checkpointed keyed on ``regime_key`` +
    chunk shape (#1482 checkpoint-cadence duty), so a mid-phase death resumes at
    the first missing chunk instead of restarting every pair.
    """
    import issue825_map_alignment as MA

    sn = torch.as_tensor(np.asarray(spectra_unit), dtype=torch.float64)
    n, d = sn.shape
    assert n_draws >= 1, n_draws
    dev = torch.device(device)
    sn_dev = sn.to(dev)
    out: list[np.ndarray] = []
    n_chunks = (n_draws + chunk_draws - 1) // chunk_draws
    t0 = time.time()
    for c in range(n_chunks):
        take = min(chunk_draws, n_draws - c * chunk_draws)
        cpath = (ckpt_dir / f"null_chunk_{c:03d}.pt") if ckpt_dir is not None else None
        if cpath is not None and cpath.exists():
            prior = torch.load(cpath, map_location="cpu", weights_only=False)
            x_prior = np.asarray(prior.get("x"), dtype=np.float64)
            if prior.get("regime_key") == regime_key and x_prior.shape == (take, n, n):
                out.append(x_prior)
                print(
                    f"[atlas] null chunk {c + 1}/{n_chunks} RESUME (checkpoint) "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
                continue
        gen = torch.Generator().manual_seed(seed + c)
        xs = torch.empty((take, n, n), dtype=torch.float64)
        for t in range(take):
            g1 = MA._random_orthogonal(d, gen, dev)
            g2 = MA._random_orthogonal(d, gen, dev)
            h = g1 * g2.T
            xs[t] = (sn_dev @ h @ sn_dev.T).cpu()
        if cpath is not None:
            _atomic_torch_save(
                {"x": xs, "regime_key": regime_key, "chunk": c, "take": take, "seed": seed + c},
                cpath,
            )
        out.append(xs.numpy())
        print(
            f"[atlas] null chunk {c + 1}/{n_chunks} draws={take} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    return np.concatenate(out, axis=0)


def _null_stats(draws: np.ndarray, d: int) -> dict:
    """Rotation-null band in the ``issue1345_operator_comparison`` output shape."""
    arr = np.asarray(draws, dtype=np.float64)
    return {
        "n_draws": int(len(arr)),
        "null_mean": float(arr.mean()),
        "null_std": float(arr.std()),
        "null_p975": float(np.quantile(arr, 0.975)),
        "analytic_sd_1_over_d": float(1.0 / d),
        "null_form": (
            "shared spectra-bilinear Haar draws (exact in distribution; draws "
            "shared across pairs — see shared_rotation_null_draws)"
        ),
    }


def tier2_aligned_operator_cosine(
    qwen_ops: dict[str, np.ndarray],
    A_llama: np.ndarray,
    R_in: np.ndarray,
    R_out: np.ndarray,
    *,
    n_draws: int,
    seed: int,
    device: str,
    n_rows_alignment: int,
) -> dict:
    """Tier-2 activation-Procrustes aligned operator cosine vs the random-rotation
    null (plan §4 leg 7 item 3; concern leg7-tier2-aligned-cosine-missing).

    Construction = ``issue825_map_alignment._procrustes_cosine_null``'s observed
    statistic, transposed into the atlas row convention: the Llama native operator
    is conjugated into the shared qwen basis by the FIXED activation-fitted
    semi-orthogonal alignments (``A_l_in_q = R_in @ A_llama @ R_out.T``; R_in from
    paired v_C TRAIN rows, R_out from paired v_A TRAIN rows — identical to the
    #825 ``R_in^T beta_b R_out`` modulo transpose naming) and compared to each
    qwen-side operator by raw vec-cosine. The null rotates the ALIGNED Llama
    operator two-sidedly (Haar) in the shared basis — the #1345 pair convention
    applied post-alignment; in the square full-orthogonal case this equals the
    #825 pre-alignment null exactly, and it is the well-defined extension when
    the alignment is semi-orthogonal (cross-model d_q != d_l). Read against the
    #825 base<->instruct anchor 0.6864. Direction-aware BY CONSTRUCTION — this
    statistic can never be produced by a spectrum cosine, which is
    rotation-invariant-only and cannot support "same operator up to rotation".
    """
    A_al = (
        np.asarray(R_in, np.float64)
        @ np.asarray(A_llama, np.float64)
        @ np.asarray(R_out, np.float64).T
    )
    d = int(A_al.shape[0])
    assert A_al.shape == (d, d), A_al.shape
    names = list(qwen_ops)
    mats = [np.asarray(qwen_ops[nm], np.float64) for nm in names]
    for m in mats:
        assert m.shape == (d, d), (m.shape, d)
    all_mats = mats + [A_al]
    spectra = []
    for m in all_mats:
        s = np.linalg.svd(m, compute_uv=False)
        spectra.append(s / (np.linalg.norm(s) + 1e-12))
    x = shared_rotation_null_draws(np.stack(spectra), n_draws=n_draws, seed=seed, device=device)
    j = len(all_mats) - 1
    v_al = A_al.reshape(-1)
    n_al = float(np.linalg.norm(v_al))
    per_op: dict[str, dict] = {}
    for i, nm in enumerate(names):
        v = mats[i].reshape(-1)
        obs = float(v @ v_al / (np.linalg.norm(v) * n_al + 1e-12))
        null = _null_stats(x[:, i, j], d)
        if np.asarray(A_llama).shape == mats[i].shape:
            v2 = np.asarray(A_llama, np.float64).reshape(-1)
            raw = {
                "applicable": True,
                "value": float(v @ v2 / (np.linalg.norm(v) * np.linalg.norm(v2) + 1e-12)),
            }
        else:
            raw = {
                "applicable": False,
                "reason": (
                    f"pre-alignment operator shapes differ ({mats[i].shape} vs "
                    f"{tuple(np.asarray(A_llama).shape)}) — cross-model spaces"
                ),
            }
        per_op[nm] = {
            "observed_aligned_cosine": obs,
            "rotation_null": null,
            "z_observed_vs_null": float((obs - null["null_mean"]) / (null["null_std"] + 1e-12)),
            "raw_vec_cosine": raw,
        }
    return {
        "per_operator": per_op,
        "anchor_825_aligned_cosine": ANCHOR_825_ALIGNED_COSINE,
        "n_rows_alignment": int(n_rows_alignment),
        "alignment": (
            "activation-fitted semi-orthogonal Procrustes (R_in from paired v_C "
            "TRAIN rows, R_out from paired v_A TRAIN rows; the "
            "issue825_map_alignment._procrustes_cosine_null construction)"
        ),
        "statistic_class": (
            "direction-aware under the FIXED activation-Procrustes alignment "
            "(vs a two-sided random-rotation null; NOT rotation-invariant — a "
            "spectrum cosine can never produce this read)"
        ),
    }


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

    # -- tier 2 aligned operator cosine (plan §4 leg 7 item 3; the #825
    # _procrustes_cosine_null construction — concern leg7-tier2-aligned-cosine-missing)
    R_in = orth_procrustes(Xq_c[folds["tr"]], Xl_c[folds["tr"]])
    R_out = orth_procrustes(Xq_a[folds["tr"]], Xl_a[folds["tr"]])
    A_banked, _b0 = OP.row_operator(a_qwen)
    A_matched, _b1 = OP.row_operator(named[f"qwen_matched_L{qL}"]["payload"])
    A_ll_native, _b2 = OP.row_operator(named[f"llama_native_L{lL}"]["payload"])
    t2 = tier2_aligned_operator_cosine(
        {"a_qwen": A_banked, "qwen_matched": A_matched},
        A_ll_native,
        R_in,
        R_out,
        n_draws=args.null_draws,
        seed=20250826,
        device=args.device,
        n_rows_alignment=int(len(folds["tr"])),
    )
    t2["a_qwen_source"] = a_qwen_source(a_qwen)
    summary["tier2_aligned_operator_cosine"] = t2
    print(
        "[fits] tier2 aligned operator cosine "
        f"a_qwen={t2['per_operator']['a_qwen']['observed_aligned_cosine']:.4f} "
        f"qwen_matched={t2['per_operator']['qwen_matched']['observed_aligned_cosine']:.4f} "
        f"(anchor 0.6864)",
        flush=True,
    )

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
            "aligned_operator_cosine": s["tier2_aligned_operator_cosine"],
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
# Phase: atlas (operator rows + pairwise distance table + MDS + H7 demote)
# ---------------------------------------------------------------------------

H7_PREDICATE = (
    "within-operator split-half distance >= between-operator distance for > half "
    "of pairs -> atlas reported noise-dominated, descriptive only"
)


def _floor_cos(floor) -> float | None:
    """Normalize a split-half floor record to its COSINE: fitted rows persist
    ``{'floor': cos, 'n_half': [...]}`` (``_split_half_floor``); the leg-6
    producer persists a BARE FLOAT (``issue2569_leg6.write_operator_factors``,
    key ``split_half_floor``); absent/None -> None."""
    if floor is None:
        return None
    if isinstance(floor, dict):
        v = floor.get("floor")
        return None if v is None else float(v)
    return float(floor)


def h7_demote_block(rows: list[dict], table: list[dict]) -> dict:
    """The pre-registered H7 demote branch (plan §7.5 leg 7; concerns
    h7-demote-branch-not-implemented + leg7-atlas-noise-demotion-missing) —
    decided FROM THE ARTIFACT ALONE, never at interpretation time.

    Reading (stated because the plan text leaves two details open):

    - **Units:** distances derive as 1 - cosine in MATCHED vec-cosine units —
      within-operator distance = 1 - split_half_floor cosine (the raw vec-cosine
      of the two half-refit operators) and between-operator distance = 1 - the
      pair's direction-aware ALIGNED raw cosine. The rotation-invariant spectrum
      fallback is NEVER used here (unmatched units); such pairs are excluded
      from the EVALUABLE set with a recorded reason. Cross-basis asymmetry
      (disclosure, concern h7-cross-basis-unit-asymmetry): for cross-basis pairs
      the within floor is a NATIVE-basis split-half cosine while the between
      distance is measured after the semi-orthogonal activation-Procrustes
      alignment, so "matched vec-cosine units" is exact on same-basis pairs and
      approximate on cross-model pairs (plausibly under-firing the demote
      there); both conventions are the plan's own — disclosed, not changed.
    - **Pair-level within distance:** the MAX of the two members' within
      distances — a pair's separation is resolvable only if it exceeds BOTH
      members' split-half self-distance, so the noisier member binds (the
      conservative reading: the demote fires more readily, never less).
    - **Denominator (registered — concern h7-denominator-evaluable-not-all-pairs):**
      ALL enumerated atlas pairs (``n_pairs_total``). The plan registers the
      criterion twice (v4 line 57 H7 statement + line 295 leg-7 Demote clause)
      as "> half of pairs" with NO evaluability qualifier, so the verdict is the
      strict majority over the full pair table: ``n_noise * 2 > n_pairs_total``.
      What an excluded pair means (a stated READING — the plan text did not
      anticipate exclusions): a pair excluded for incommensurate units (the
      spectrum fallback) or a missing floor is not evidence either way, and
      under the registered "> half of pairs" text it stays in the denominator
      while never entering the numerator — so exclusions bias the verdict toward
      NOT demoting, the OPPOSITE bias from the evaluable-only majority, which is
      retained as ``diagnostic_evaluable_majority`` (diagnostic only, never the
      verdict; disagreement between the readings is flagged). Zero evaluable
      pairs -> verdict UNDECIDABLE and the atlas ships descriptive-only (no
      above-noise separation claim is possible) — never a silent not-demoted
      default from zero evidence.

    Mutates each table entry with an ``h7`` record; returns the verdict block.
    """
    floors = {r["name"]: _floor_cos(r.get("floor")) for r in rows}
    n_eval = 0
    n_noise = 0
    excluded: dict[str, int] = {}
    for e in table:
        a, b = e["pair"]
        fa, fb = floors.get(a), floors.get(b)
        if e.get("cosine") is None:
            rec = {
                "evaluable": False,
                "reason": (
                    "no direction-aware aligned cosine (the spectrum fallback is "
                    "rotation-invariant — unmatched units for the floor comparison)"
                ),
            }
        elif fa is None or fb is None:
            missing = [nm for nm, f in ((a, fa), (b, fb)) if f is None]
            rec = {"evaluable": False, "reason": f"no split-half floor on: {', '.join(missing)}"}
        else:
            within = 1.0 - min(fa, fb)  # == max(1 - fa, 1 - fb)
            between = 1.0 - float(e["cosine"]["raw_cosine"])
            nd = bool(within >= between)
            rec = {
                "evaluable": True,
                "within_distance_max": float(within),
                "between_distance": float(between),
                "noise_dominated": nd,
            }
            n_eval += 1
            n_noise += int(nd)
        if not rec["evaluable"]:
            excluded[rec["reason"]] = excluded.get(rec["reason"], 0) + 1
        e["h7"] = rec
    n_total = len(table)
    if n_eval == 0:
        noise_dominated = None
        disposition = (
            "undecidable — descriptive only (zero pairs carry both split-half "
            "floors and a direction-aware aligned cosine)"
        )
    elif n_noise * 2 > n_total:
        noise_dominated = True
        disposition = (
            "noise-dominated — descriptive only (pre-registered H7 demote fired: "
            f"within >= between on {n_noise} of {n_total} total pairs, a strict "
            "majority of ALL pairs — the registered denominator)"
        )
    else:
        noise_dominated = False
        disposition = (
            "not demoted — within-operator split-half distance >= between-operator "
            f"distance for {n_noise} of {n_total} total pairs, not > half of ALL "
            "pairs (the registered denominator)"
        )
    eval_majority = (n_noise * 2 > n_eval) if n_eval else None
    return {
        "predicate": H7_PREDICATE,
        "reading": {
            "units": (
                "distances are 1 - cosine in matched vec-cosine units: within = "
                "1 - split_half_floor cosine (raw vec-cosine of the two half-refit "
                "operators); between = 1 - the pair's direction-aware aligned raw "
                "cosine; the rotation-invariant spectrum fallback is never used"
            ),
            "cross_basis_unit_asymmetry": (
                "disclosure: for cross-basis pairs the within floor is a "
                "NATIVE-basis split-half cosine while the between distance is "
                "measured after the semi-orthogonal activation-Procrustes "
                "alignment, so 'matched vec-cosine units' is exact on same-basis "
                "pairs and approximate on cross-model pairs (plausibly "
                "under-firing the demote there); both conventions are the plan's "
                "own — disclosed, not changed"
            ),
            "pair_within": (
                "max over the two members' within-operator distances (the noisier "
                "member binds: a separation is resolvable only above BOTH self-floors)"
            ),
            "denominator": (
                "ALL enumerated atlas pairs (n_pairs_total) — the plan registers "
                "'> half of pairs' twice (v4 line 57 + line 295) with no "
                "evaluability qualifier, so the verdict is the strict majority "
                "over the full pair table: n_noise_dominated * 2 > n_pairs_total"
            ),
            "excluded_pairs": (
                "a pair excluded for incommensurate units (spectrum fallback) or "
                "a missing floor is not evidence either way; under the registered "
                "'> half of pairs' text it stays in the denominator and never "
                "enters the numerator, so exclusions bias the verdict toward NOT "
                "demoting — a stated READING of plan text that did not anticipate "
                "exclusions (the opposite bias from the evaluable-only majority, "
                "kept as diagnostic_evaluable_majority); zero evaluable pairs -> "
                "verdict undecidable, atlas ships descriptive-only regardless"
            ),
        },
        "n_pairs_total": int(n_total),
        "n_evaluable": int(n_eval),
        "n_noise_dominated": int(n_noise),
        "verdict_denominator": "all_pairs",
        "fraction_noise_dominated_all_pairs": (float(n_noise) / n_total) if n_total else None,
        "fraction_noise_dominated_evaluable": (float(n_noise) / n_eval) if n_eval else None,
        "noise_dominated": noise_dominated,
        "disposition": disposition,
        "diagnostic_evaluable_majority": {
            "noise_dominated": eval_majority,
            "readings_disagree": (
                None
                if (noise_dominated is None or eval_majority is None)
                else bool(eval_majority != noise_dominated)
            ),
            "note": (
                "strict majority over EVALUABLE pairs only — the alternative "
                "reading (and the pre-fix implementation's); DIAGNOSTIC ONLY, "
                "never the verdict"
            ),
        },
        "excluded_pair_reasons": excluded,
        "conservatism_note": (
            "floors are split-half refits at n/2 train rows, which overstates the "
            "full-n operator's noise — per-pair noise_dominated flags fire "
            "conservatively (the registered all-pairs denominator, by contrast, "
            "biases the AGGREGATE verdict toward not-demoting when pairs are "
            "excluded — see reading.excluded_pairs)"
        ),
    }


def _resolve_atlas_rows(args) -> tuple[list[dict], list[dict]]:
    """Resolve every plan-named atlas row or DROP it with a named reason (plan §4
    leg 7 step 4: a miss drops the row, never a silent substitute).

    Returns (rows, dropped). Each row: {name, basis ('qwen'|'llama'), A (d x d or
    d_in x d_out np), floor (dict|float|None), floor_label, source, fp (content
    fingerprint of the row's on-disk inputs — the resume key component)}."""
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
                    "fp": _file_sha(p.path),
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
                        "fp": _file_sha(dest),
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
                    "fp": _file_sha(path),
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
                center_path = Path(center["payload"].path)
                center_fp = _file_sha(center_path) if center_path.is_file() else str(center_path)
                rows.append(
                    {
                        "name": f"composed_L{lL}",
                        "basis": "llama",
                        "A": Ac @ Am @ Aa,
                        "floor": None,
                        "floor_label": "no floor — composition of three fitted maps",
                        "source": f"{align_path} (center: {center['source']})",
                        "fp": hashlib.sha256(
                            (_file_sha(align_path) + center_fp).encode()
                        ).hexdigest(),
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
                    # producer schema: BARE FLOAT (issue2569_leg6.write_operator_factors)
                    "floor": d.get("split_half_floor"),
                    "floor_label": "leg6 sidecar (operator split-half self-cosine)",
                    "source": str(path),
                    "fp": _file_sha(path),
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
                    "fp": _file_sha(path),
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
    # Fingerprint covers the small derivation-pinning files + the recorded lambda;
    # the fp16 encode arrays are deliberately not hashed (multi-GB) — their
    # generation is pinned by enc_meta's row counts + the metrics-recorded lambda.
    fp = hashlib.sha256(
        (
            _file_sha(fdir / "enc_meta.json")
            + _file_sha(fdir / "feature_map_metrics.json")
            + _file_sha(fdir / "counts_ctx.npy")
            + f"lam={lam:g}"
        ).encode()
    ).hexdigest()
    return {
        "name": "featmap_L19",
        "basis": "qwen",
        "A": A_feat,
        "floor": floor,
        "floor_label": "split-half refit of M_feat at the recorded lambda (linearized)",
        "source": f"{fdir} (re-derived at recorded lambda={lam:g}; ReLU/TopK gates ignored)",
        "fp": fp,
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
    # content fingerprint of the alignment inputs (int64 ci read from the capture
    # files — machine-stable to hash; #1336) for the aligned-spectra resume keys
    fp = hashlib.sha256(
        json.dumps(
            {"ci": _sha_int64(join["ci"]), "n_tr": int(len(tr)), "val_rows": int(args.val_rows)},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "R_in": orth_procrustes(Xq_c, Xl_c),  # (d_q, d_l)
        "R_out": orth_procrustes(Yq_a, Yl_a),  # (d_q, d_l)
        "n_rows": int(len(tr)),
        "fp": fp,
    }


def _safe_name(name: str) -> str:
    """Filesystem-safe checkpoint stem for a row name."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _spectra_key(row: dict, proc_fp: str) -> str:
    """Resume key for one row's spectra checkpoint: content fingerprints of the
    row's inputs + (for llama-basis rows) the alignment inputs — generating
    parameters and file-byte shas only, never a recomputed float array (#1336)."""
    payload = {
        "name": row["name"],
        "fp": row.get("fp", ""),
        "basis": row["basis"],
        "shape": [int(x) for x in np.asarray(row["A"]).shape],
        "aligned": bool(row.get("aligned_A") is not None),
        "proc_fp": proc_fp if row["basis"] == "llama" else "",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _pair_statistics(
    rows: list[dict],
    *,
    n_draws: int,
    seed: int,
    device: str,
    ckpt_dir: Path,
    proc_fp: str,
    chunk_draws: int,
) -> list[dict]:
    """Batched pair statistics (concern atlas-pair-loop-unbatched-unresumable).

    Structure — the axes the old per-pair loop recomputed are hoisted:

    1. **One SVD per row** (checkpointed per row, keyed on the row's content
       fingerprint) instead of two dense SVDs per PAIR: spectrum cosines are dot
       products of the stored singular values (numerically identical to
       ``spectrum_cosine`` — pinned by
       ``tests/test_issue2569_atlas.py::test_pair_table_matches_direct_statistics``).
    2. **One shared Haar draw set for every pair's rotation null**
       (``shared_rotation_null_draws``; exact in distribution; 2 QRs per draw
       TOTAL instead of 2 QRs + 2 dense GEMMs per draw per pair), chunked +
       checkpointed. SVDs run on CPU deliberately (cuSOLVER svd non-convergence
       class, gotchas.md); the Haar QRs honor ``device`` (well-conditioned
       gaussian inputs).
    3. Per-pair raw aligned cosines are O(d^2) dots — the residual python loop
       is assembly-only and keeps the per-pair progress lines.
    """
    assert n_draws >= 1, "--null-draws must be >= 1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    t0 = time.time()
    for i, r in enumerate(rows):
        key = _spectra_key(r, proc_fp)
        f = ckpt_dir / f"spectra_{i:02d}_{_safe_name(r['name'])}.pt"
        blob = None
        if f.exists():
            prior = torch.load(f, map_location="cpu", weights_only=False)
            if prior.get("key") == key:
                blob = prior
                print(
                    f"[atlas] row {i + 1}/{n} {r['name']} spectra RESUME "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
        if blob is None:
            s_raw = torch.linalg.svdvals(torch.as_tensor(np.asarray(r["A"]), dtype=torch.float64))
            if r.get("aligned_A") is None:
                s_al = None
            elif r["aligned_A"] is r["A"]:
                s_al = s_raw  # qwen-basis rows alias aligned_A == A
            else:
                s_al = torch.linalg.svdvals(
                    torch.as_tensor(np.asarray(r["aligned_A"]), dtype=torch.float64)
                )
            blob = {"key": key, "s_raw": s_raw, "s_al": s_al}
            _atomic_torch_save(blob, f)
            print(
                f"[atlas] row {i + 1}/{n} {r['name']} spectra elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
        r["_s_raw"] = np.asarray(blob["s_raw"], dtype=np.float64)
        r["_s_al"] = None if blob["s_al"] is None else np.asarray(blob["s_al"], dtype=np.float64)

    al_idx = [i for i, r in enumerate(rows) if r.get("aligned_A") is not None]
    x_draws = None
    d_al = 0
    if len(al_idx) >= 2:
        d_al = int(np.asarray(rows[al_idx[0]]["aligned_A"]).shape[0])
        for i in al_idx:
            shape = tuple(np.asarray(rows[i]["aligned_A"]).shape)
            assert shape == (d_al, d_al), (rows[i]["name"], shape, d_al)
        spectra_unit = np.stack(
            [rows[i]["_s_al"] / (np.linalg.norm(rows[i]["_s_al"]) + 1e-12) for i in al_idx]
        )
        regime = hashlib.sha256(
            json.dumps(
                {
                    "rows": [_spectra_key(rows[i], proc_fp) for i in al_idx],
                    "seed": seed,
                    "chunk_draws": chunk_draws,
                    "d": d_al,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        x_draws = shared_rotation_null_draws(
            spectra_unit,
            n_draws=n_draws,
            seed=seed,
            device=device,
            chunk_draws=chunk_draws,
            ckpt_dir=ckpt_dir,
            regime_key=regime,
        )
    al_pos = {i: p for p, i in enumerate(al_idx)}

    table: list[dict] = []
    n_pairs = n * (n - 1) // 2
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            k += 1
            a, b = rows[i], rows[j]
            sa, sb = a["_s_raw"], b["_s_raw"]
            kk = int(min(len(sa), len(sb)))
            spec = float(
                (sa[:kk] * sb[:kk]).sum()
                / (np.linalg.norm(sa[:kk]) * np.linalg.norm(sb[:kk]) + 1e-12)
            )
            entry = {
                "pair": [a["name"], b["name"]],
                "bases": [a["basis"], b["basis"]],
                "spectrum": {
                    "spectrum_cosine": spec,
                    "truncated": bool(len(sa) != len(sb)),
                    "k": kk,
                    "statistic_class": "spectrum/rotation-invariant-only (descriptive ceiling)",
                },
            }
            if i in al_pos and j in al_pos and x_draws is not None:
                va = np.asarray(a["aligned_A"], dtype=np.float64).reshape(-1)
                vb = np.asarray(b["aligned_A"], dtype=np.float64).reshape(-1)
                raw = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
                same_basis = a["basis"] == b["basis"]
                entry["cosine"] = {
                    "raw_cosine": raw,
                    "rotation_null": _null_stats(x_draws[:, al_pos[i], al_pos[j]], d_al),
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
                f"[atlas] pair {k}/{n_pairs} {a['name']}~{b['name']} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    return table


def phase_atlas(args) -> None:
    """Operator atlas: resolve rows, align bases, batched pairwise statistics
    (every statistic class-labeled), split-half floors, the pre-registered H7
    demote verdict, MDS coords (presentation-only).
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

    table = _pair_statistics(
        rows,
        n_draws=args.null_draws,
        seed=1345,
        device=args.device,
        ckpt_dir=Path(args.fits_dir) / "atlas_ckpt",
        proc_fp=proc["fp"] if proc else "",
        chunk_draws=args.null_chunk_draws,
    )
    h7 = h7_demote_block(rows, table)
    print(f"[atlas] h7_demote: {h7['disposition']}", flush=True)

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
                # explicit derived forms so no downstream reader re-derives units
                # (the H7 predicate is phrased in DISTANCES; floors store COSINES)
                "floor_cos": _floor_cos(r.get("floor")),
                "within_distance": (
                    None
                    if _floor_cos(r.get("floor")) is None
                    else float(1.0 - _floor_cos(r.get("floor")))
                ),
                "floor_label": r["floor_label"],
                "source": r["source"],
                "fp": r.get("fp", ""),
                "procrustes_aligned": bool(
                    r.get("aligned_A") is not None and r["basis"] == "llama"
                ),
            }
            for r in rows
        ],
        "dropped_rows": dropped,
        "h7_demote": h7,
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
    pair 19,22) with a real linear cross-model structure + a REAL noise floor.

    The noise level (sigma=0.3) and d ~ n_train/5 are calibrated so val-selected
    lambda is INTERIOR on the smoke grid: a near-noiseless fixture (or d << n)
    makes the val curve flat toward lambda->0 and the C4 edge-refusal fires
    (probed 2026-08-25: sigma=0.3, d=32/48, n=240 — edge-free across 8 seeds,
    worst native-route test R2 = 0.626; the earlier sigma=0.05, d=8/12 fixture
    low-edged through all 6 widenings)."""
    import issue2569_xmodel_capture as XC

    sigma = 0.3
    rng = np.random.default_rng(seed)
    ci = np.arange(n, dtype=np.int64) * 3 + 1
    corpus = ["lmsys" if i % 2 == 0 else "wildchat" for i in range(n)]
    z = rng.standard_normal((n, d_q)).astype(np.float32)
    Mq = rng.standard_normal((d_q, d_q)).astype(np.float32) / np.sqrt(d_q)
    T = rng.standard_normal((d_q, d_l)).astype(np.float32) / np.sqrt(d_q)
    data = {
        ("qwen", "vc", 19): z,
        ("qwen", "va", 19): z @ Mq + sigma * rng.standard_normal((n, d_q)).astype(np.float32),
        ("llama", "vc", 22): z @ T + sigma * rng.standard_normal((n, d_l)).astype(np.float32),
    }
    data[("llama", "va", 22)] = data[("qwen", "va", 19)] @ T + sigma * rng.standard_normal(
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
    _make_synthetic_captures(cap, n=240, d_q=32, d_l=48, seed=0)
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
    routes = tt["tier2_operator_similarity"]["routes"]
    assert routes["native"]["r2"] > 0.5, routes["native"]["r2"]
    # the composed route must beat the alignment-only baseline on TRUE linear structure
    assert routes["composed_matched"]["r2"] > routes["alignment_only_baseline"]["r2"], routes
    t2 = tt["tier2_operator_similarity"]["aligned_operator_cosine"]
    assert set(t2["per_operator"]) == {"a_qwen", "qwen_matched"}, sorted(t2["per_operator"])
    for rec in t2["per_operator"].values():
        assert -1.0 <= rec["observed_aligned_cosine"] <= 1.0, rec
        assert rec["rotation_null"]["n_draws"] == 24, rec["rotation_null"]
    assert at["rows"], "selftest atlas resolved no rows"
    h7 = at["h7_demote"]
    assert h7["disposition"] and h7["n_pairs_total"] == len(at["distance_table"]), h7
    assert all("h7" in e for e in at["distance_table"]), "per-pair h7 records missing"
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
    ap.add_argument(
        "--null-chunk-draws",
        type=int,
        default=25,
        help="rotation-null draws per checkpoint chunk (atlas pair statistics)",
    )
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

        import issue825_map_alignment as _ma  # noqa: F401
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
