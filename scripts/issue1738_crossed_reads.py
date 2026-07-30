#!/usr/bin/env python3
"""Issue #1738 follow-up `crossed-multiturn-averaged` — S2 reads driver (plan v9 §4.3/§6).

Consumes the S1 crossed capture store (``issue1738_crossed/capture/*.pt`` —
px/cx/v_x at {14,19,26} + prefix_id/query_id + the SAE L19 fold-in fields) and
runs the six registered reads on the pinned prefix-grouped split
(``split_1738_crossed.json``, parent 4-way schema — ``MTF.load_split`` verbatim):

R1  Held-out R² per arm × layer (context / prefix / bare / averaged-independent),
    identity+learned-bias + kNN retrieval per fitted map, n-vs-d stated per fit.
R2  AVERAGED-GRAIN PRIMARY = the INDUCED map (per-row context fit applied to
    x̄_p = mean_q cx(p,q), scored vs ȳ_p — the #1092 operator-coincidence
    design); independently-fit averaged map SECONDARY (λ by inner CV WITHIN the
    train prefixes — never the 20-prefix val; n=4,430 vs d=3,584 stated).
R3  Crossed ANOVA at scale (prefix/query/interaction shares of v_x, complete-case
    grid, vectorized grid means — CCS.anova_shares) + per-direction top-48
    answer-PCA versions; parent K-resample sampling floor embedded as a
    COMPARATOR (never a subtraction).
R4  Disjoint stitch: ridge on [prefix-end; bare-query] (7,168-d) vs the
    full-context map; identity+bias DIMENSION-MISMATCHED → recorded
    "inapplicable" explicitly; kNN still runs.
R5  Operator geometry across arms: principal angles between top-k left/right
    singular subspaces of the fitted operators at each pair's matched λ, k=48 +
    k@90% energy, vs 200 Haar-subspace null draws (== the spectrum-matched
    random-map null for ANGLE reads: Haar U/V make every top-k subspace Haar;
    the spectrum only orders them). Inputs+targets mean-centred by the fit;
    gamma (mean-norm fraction) reported.
R6  SAE per-feature crossed reads, MECHANICAL ONLY (0 judge/API calls —
    judged-label freeze): per-feature prefix/query/interaction shares (K=20
    label-permutation nulls, seeds 1738000–1738019), encode-then-average
    feature maps (induced + independent), dense-latent stratification, and the
    decoder–r_B alignment (raw + scaffold-projected companion) on the top
    prefix-share tail; evidence artifacts persisted for #1773. Skips cleanly
    when the pilot verdict is sae_enabled=false.

Fitter: shared-Gram ridge ONLY (plan §4.3 stated refinement — the five-fitter
comparison is settled on the main corpus); parent λ grid logspace(−3,8,23),
val-selected on the pinned ~400-row val (except the averaged-independent inner
CV). Compute: 1× A100-80. Refusal-safety: chunk text fields are never
printed/logged (digest-only).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # #847: thread caps land BEFORE numpy/torch import on the shared VM

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as PF  # noqa: E402
import issue779_ffc_n50k_generate_capture as N50  # noqa: E402
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue1092_crossed_core_sae as CCS  # noqa: E402
import issue1092_partb_operator as OPS  # noqa: E402
import issue1482_shuffle_null as SN  # noqa: E402
import issue1738_multiturn_fits as MTF  # noqa: E402
import issue1738_multiturn_generate_capture as GG  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.orchestrate import hub  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1738_crossed_reads")

LAMBDAS = MTF.LAMBDAS  # parent grid logspace(-3, 8, 23), verbatim
ARMS = ("context", "prefix", "bare", "avg")  # the four-input-object decomposition
ARM_MM = {"context": "cx", "prefix": "px"}
PERM_SEEDS = tuple(range(1_738_000, 1_738_020))  # plan §10: K=20 permutation seeds
OPERATOR_NULL_SEED = 1738
DEFAULT_OUT_EVAL = PROJECT_ROOT / "eval_results" / "issue_1738" / "crossed"
DEFAULT_OUT_LOCAL = PROJECT_ROOT / "data" / "issue_1738" / "crossed_reads"
KRESAMPLE_SUMMARY_DEFAULT = (
    PROJECT_ROOT / "eval_results" / "issue_1738" / "kresample" / "floor_summary.json"
)
RC_FENCE = MTF.RC_FENCE  # 24 — G3 designed-halt rc (report written first)


def _eigh_robust(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """cuda eigh with the #1335 cuSOLVER non-convergence CPU fallback."""
    try:
        return torch.linalg.eigh(a)
    except torch.linalg.LinAlgError:
        logger.warning(
            "[eigh] cuda eigh failed to converge — CPU LAPACK fallback (n=%d)", a.shape[0]
        )
        w, v = torch.linalg.eigh(a.cpu())
        return w.to(a.device), v.to(a.device)


# ── assembly: stream crossed chunks → per-(array, layer) memmaps + sae sidecars ────


def _crossed_chunk_names(args, n_q: int) -> list[str]:
    """Chunk basenames for the REALIZED-n_q family ONLY (r2 blocker-1): a G1
    query-ladder descope leaves stale foreign-n_q chunks (and the pilot's
    ``pilotpartial`` files) under capture/ — scanning them would assemble a
    mixed-grid store (the ``qid.max() < n_q`` assert's crash) or double-count
    the pilot's partial rows. Names carry the family tag ``_qQQ_chunk``."""
    tag = f"_q{int(n_q):02d}_chunk"
    if args.local_capture_dir:
        pool = sorted(p.name for p in Path(args.local_capture_dir).glob("shard*.pt"))
    else:
        pool = sorted(
            n
            for n in N50._remote_index(f"{args.hf_prefix}/{GG.CAPTURE_SUBDIR}")
            if n.endswith(".pt")
        )
    names = [n for n in pool if tag in n]
    if not names:
        fams = sorted({p.split("_")[1] for p in pool if "_q" in p})
        raise SystemExit(
            f"no crossed capture chunks for realized n_q={n_q} (tag {tag!r}); "
            f"{len(pool)} .pt files present (n_q families: {fams or 'none'}) — "
            "run the S1 capture at this n_q first (the G1 ladder re-runs the "
            "pilot + fleet with --queries-per-prefix; stale families are inert)"
        )
    return names


def assemble_crossed(args, layers: list[int], n_q: int):
    """Stream crossed chunks (HF or local) into append-only fp32 binaries per
    (array, layer) + int64 ci/prefix-pos/query binaries + per-chunk SAE sidecars
    (the sae-arm chunk schema fields only — text never re-persisted), with a
    cursor checkpoint (resume truncates to the cursor — the external-stream
    checkpoint law; the MTF.assemble_streams pattern + h_dim inference).

    Returns (mm dict, ci, meta) with meta carrying h_dim + sae flags."""
    mm_dir = Path(args.mm_dir)
    mm_dir.mkdir(parents=True, exist_ok=True)
    side_dir = mm_dir / "sae_side"
    side_dir.mkdir(exist_ok=True)
    names = _crossed_chunk_names(args, n_q)
    fp = hashlib.sha256(
        ("\n".join(names) + f"|{args.hf_prefix}|{sorted(layers)}|crossed-v1").encode()
    ).hexdigest()
    paths: dict = {"cursor": mm_dir / "cursor.json", "ci": mm_dir / "ci.bin", "meta": {}}
    for arr in ("px", "cx", "vx"):
        for li in layers:
            paths[(arr, li)] = mm_dir / f"{arr}_L{li}.bin"
    cursor = {"fingerprint": fp, "n_chunks_done": 0, "n_rows": 0, "h_dim": 0, "sae_any": False}
    if paths["cursor"].exists():
        prev = json.loads(paths["cursor"].read_text())
        if prev.get("fingerprint") == fp:
            cursor = prev
            logger.info(
                "[assemble] resume: %d chunks / %d rows done",
                cursor["n_chunks_done"],
                cursor["n_rows"],
            )
        else:
            logger.info("[assemble] cursor fingerprint mismatch — fresh assembly")
            for k, p in paths.items():
                if k != "meta" and Path(p).exists():
                    Path(p).unlink()
            for p in side_dir.glob("*.pt"):
                p.unlink()
    n_rows = int(cursor["n_rows"])
    h_dim = int(cursor.get("h_dim", 0))
    sae_any = bool(cursor.get("sae_any", False))
    for arr in ("px", "cx", "vx"):
        for li in layers:
            p = paths[(arr, li)]
            want = n_rows * h_dim * 4
            if p.exists() and p.stat().st_size != want:
                with open(p, "r+b") as f:
                    f.truncate(want)
            elif not p.exists():
                p.touch()
    ci_p = paths["ci"]
    if ci_p.exists() and ci_p.stat().st_size != n_rows * 8:
        with open(ci_p, "r+b") as f:
            f.truncate(n_rows * 8)
    elif not ci_p.exists():
        ci_p.touch()

    key_of = {"px": "px_last", "cx": "cx_last", "vx": "v_x"}
    sae_keys = (
        "ci",
        "corpus",
        "dropped_ci",
        "feat_idx",
        "row_ptr",
        "ans_mean",
        "ans_max",
        "ans_frac",
        "nm_feat_idx",  # unmasked pooling twin (plan §6 mask-robustness; r2 blocker-2);
        "nm_row_ptr",  # absent from pre-r2 chunks — the save below is key-tolerant
        "nm_mean",
        "nm_max",
        "nm_frac",
        "px_feat_idx",
        "px_row_ptr",
        "px_feat_val",
        "cx_feat_idx",
        "cx_row_ptr",
        "cx_feat_val",
        "px_dense19",
        "cx_dense19",
        "n_ans_tokens",
        "n_inlier_tokens",
        "sae",
    )
    cache = mm_dir / "dl_cache"
    cache.mkdir(exist_ok=True)
    handles = {k: open(paths[k], "ab") for k in paths if isinstance(k, tuple)}
    ci_f = open(ci_p, "ab")
    try:
        for k, name in enumerate(names):
            if k < cursor["n_chunks_done"]:
                continue
            if args.local_capture_dir:
                local = Path(args.local_capture_dir) / name
            else:
                local = Path(
                    PF._download_chunk_with_retry(
                        C.HF_DATA_REPO, f"{args.hf_prefix}/{GG.CAPTURE_SUBDIR}/{name}", cache
                    )
                )
            bundle = torch.load(local, map_location="cpu", weights_only=False)
            blayers = list(bundle["layers"])
            li_pos = {li: blayers.index(li) for li in layers}
            n = len(bundle["ci"])
            if h_dim == 0:
                h_dim = int(bundle["v_x"].shape[2])
            for arr in ("px", "cx", "vx"):
                t = bundle[key_of[arr]]
                assert t.shape == (n, len(blayers), h_dim), (name, arr, t.shape)
                for li in layers:
                    handles[(arr, li)].write(
                        np.ascontiguousarray(
                            t[:, li_pos[li], :].numpy().astype(np.float32)
                        ).tobytes()
                    )
            ci_f.write(np.asarray(bundle["ci"], dtype=np.int64).tobytes())
            if bundle.get("sae_enabled"):
                sae_any = True
                torch.save({key: bundle[key] for key in sae_keys if key in bundle}, side_dir / name)
            n_rows += n
            cursor.update(
                {
                    "n_chunks_done": k + 1,
                    "n_rows": n_rows,
                    "h_dim": h_dim,
                    "sae_any": sae_any,
                }
            )
            if not args.local_capture_dir:
                local.unlink(missing_ok=True)  # purge — peak footprint ~one chunk
            if (k + 1) % 25 == 0 or (k + 1) == len(names):
                for h in handles.values():
                    h.flush()
                ci_f.flush()
                GG.N1M._atomic_write_json(paths["cursor"], cursor)
                logger.info("[assemble] chunk %d/%d (%d rows)", k + 1, len(names), n_rows)
    finally:
        for h in handles.values():
            h.close()
        ci_f.close()
    GG.N1M._atomic_write_json(paths["cursor"], cursor)
    ci = np.fromfile(ci_p, dtype=np.int64)
    assert len(ci) == n_rows, (len(ci), n_rows)
    assert len(set(ci.tolist())) == n_rows, "duplicate ci across chunks"
    mm = {
        (arr, li): np.memmap(paths[(arr, li)], dtype=np.float32, mode="r", shape=(n_rows, h_dim))
        for arr in ("px", "cx", "vx")
        for li in layers
    }
    meta = {"n_rows": n_rows, "n_chunks": len(names), "h_dim": h_dim, "sae_any": sae_any}
    return mm, ci, meta


def _grid_positions(ci: np.ndarray, n_q_grid: int, n_q: int):
    """Complete-case grid: (prefix positions cc, row_pos (P_cc, n_q) into the
    captured arrays, n_excluded). Row ci = prefix_pos * n_q_grid + q."""
    pid = ci // n_q_grid
    qid = ci % n_q_grid
    assert int(qid.max()) < n_q, (int(qid.max()), n_q)
    pos: dict[tuple[int, int], int] = {
        (int(p), int(q)): r for r, (p, q) in enumerate(zip(pid.tolist(), qid.tolist()))
    }
    prefixes = sorted(set(pid.tolist()))
    cc = [p for p in prefixes if all((p, q) in pos for q in range(n_q))]
    row_pos = np.asarray([[pos[(p, q)] for q in range(n_q)] for p in cc], dtype=np.int64)
    return np.asarray(cc, dtype=np.int64), row_pos, len(prefixes) - len(cc)


# ── ridge fits (parent shared-Gram helpers verbatim; fac retained for R2/R5) ──────


def _fit_dense_cell(X, Y, tr, val, ho, dev, block, lambdas=LAMBDAS):
    """One shared-Gram ridge cell; λ selected on the pinned val (parent
    fit_ridge selection order verbatim). Returns (fac, pred_ho, meta,
    val_stats-by-λ) — fac retained for the R2 induced read + R5 operators."""
    fac = PF._ridge_factorize(X, Y, tr, dev, block)
    y_val = np.asarray(Y[val], dtype=np.float64)
    val_stats: dict[float, dict] = {}
    best_lam, best_vr2 = float(lambdas[0]), -np.inf
    for lam in lambdas:
        pred = PF._ridge_predict_one(X, val, fac, lam, dev, block)
        r2 = PR._pooled_r2(pred, y_val)
        val_stats[float(lam)] = {
            "r2": float(r2),
            "mse": float(((pred - y_val) ** 2).mean()),
        }
        if np.isfinite(r2) and r2 > best_vr2:
            best_vr2, best_lam = float(r2), float(lam)
    pred_ho = PF._ridge_predict_one(X, ho, fac, best_lam, dev, block)
    edge = None
    if np.isclose(best_lam, float(lambdas[0])):
        edge = "low"
    elif np.isclose(best_lam, float(lambdas[-1])):
        edge = "high"
    meta = {
        "fitter": "ridge (shared-Gram primal, parent verbatim)",
        "selection": "val-lambda (pinned val rows)",
        "selected_lambda": best_lam,
        "val_r2_at_selected": float(best_vr2),
        "lambda_grid_edge": edge,
        "n_train": int(len(tr)),
        "d": int(X.shape[1]),
    }
    return fac, pred_ho, meta, val_stats


def _inner_cv_lambda(X, Y, tr, dev, block, n_folds, seed, lambdas=LAMBDAS):
    """Averaged-independent λ selection: k-fold CV WITHIN the train prefixes
    (critic Should-Fix — never the 20-prefix val; group == prefix == row at the
    averaged grain). Returns (λ*, cv table {λ: sse})."""
    n_folds = max(2, min(n_folds, len(tr)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tr))
    folds = np.array_split(order, n_folds)
    sse = {float(lam): 0.0 for lam in lambdas}
    for k, f in enumerate(folds):
        va = tr[f]
        trf = tr[np.concatenate([folds[j] for j in range(n_folds) if j != k])]
        fac = PF._ridge_factorize(X, Y, trf, dev, block)
        y_va = np.asarray(Y[va], dtype=np.float64)
        for lam in lambdas:
            pred = PF._ridge_predict_one(X, va, fac, lam, dev, block)
            sse[float(lam)] += float(((pred - y_va) ** 2).sum())
    finite = {lam: v for lam, v in sse.items() if np.isfinite(v)}
    assert finite, "inner CV: every λ produced non-finite SSE"
    lam_sel = min(finite, key=finite.get)
    return lam_sel, {"cv_sse_by_lambda": sse, "n_folds": int(n_folds), "seed": int(seed)}


def _w_raw(fac, lam: float) -> torch.Tensor:
    """(H_in, D) ridge operator in RAW-input coordinates at λ: fold the per-dim
    1/xsd back into the standardized-space W (the partb _operator_raw fold), so
    every arm's input subspace shares one raw basis. Maps centered raw x →
    centered y (inputs + targets mean-centred by the fit — R5 requirement)."""
    w_std = fac["U"] @ (fac["UtXtY"] / (fac["s_eig"] + float(lam))[:, None])
    return w_std / fac["xsd"][:, None]


def _ridge_payload(fac, lam: float) -> dict:
    """PF.apply_map-compatible ridge payload from a retained factorization."""
    w_std = fac["U"] @ (fac["UtXtY"] / (fac["s_eig"] + float(lam))[:, None])
    return {
        "kind": "ridge",
        "selected_lambda": float(lam),
        "xmu": fac["xmu"].detach().cpu().to(torch.float32),
        "xsd": fac["xsd"].detach().cpu().to(torch.float32),
        "ymu": fac["ymu"].detach().cpu().to(torch.float32),
        "W": w_std.detach().cpu().to(torch.float32),
    }


def _gamma(X, rows: np.ndarray, mu: torch.Tensor, block: int) -> float:
    """Mean-norm fraction ||μ||² / E[||x||²] over ``rows`` (R5 gamma report)."""
    rows = rows[: min(len(rows), 10_000)]
    tot = 0.0
    for s in range(0, len(rows), block):
        xb = np.asarray(X[rows[s : s + block]], dtype=np.float64)
        tot += float((xb * xb).sum())
    mean_sq = tot / max(1, len(rows))
    return float((mu.double() ** 2).sum().item() / max(mean_sq, 1e-30))


def _cell_summary(pred: np.ndarray, true: np.ndarray, n_boot: int) -> dict:
    r2, cos = F._recon_point(pred, true)
    ci = MTF._boot_recon_ci_batched(pred, true, n_boot, MTF.BOOT_SEED)
    return {"holdout_r2": float(r2), "holdout_mean_cosine": float(cos), "bootstrap": ci}


def _baselines_cell(Xtr, Ytr, Xho, Yho, preds: dict[str, np.ndarray], *, identity_note=""):
    """Standing mapping-baselines pair: identity+learned-bias (or an EXPLICIT
    'inapplicable' record on a dimension mismatch — R4) + kNN retrieval per
    prediction (euclidean + cosine, ks clamped to the pool)."""
    out: dict = {}
    y_ho = np.asarray(Yho, dtype=np.float64)
    preds = dict(preds)
    if Xtr is not None and Xtr.shape[1] == np.asarray(Ytr).shape[1]:
        pred_ib = identity_bias_predict(np.asarray(Xtr), np.asarray(Ytr), np.asarray(Xho))
        r2_ib, cos_ib = F._recon_point(pred_ib, y_ho)
        out["identity_bias"] = {
            "holdout_r2": float(r2_ib),
            "holdout_mean_cosine": float(cos_ib),
        }
        preds["identity_bias"] = pred_ib
    else:
        out["identity_bias"] = {
            "status": "inapplicable",
            "reason": identity_note
            or f"input dim {None if Xtr is None else Xtr.shape[1]} != target dim "
            f"{np.asarray(Ytr).shape[1]} (dimension-mismatched)",
        }
    ks = tuple(k for k in (1, 5, 10) if k <= len(y_ho)) or (1,)
    out["ks"] = list(ks)
    out["chance_note"] = f"chance = k / n_pool (n_pool = {len(y_ho)})"
    out["knn"] = {
        name: {m: knn_retrieval(pv, y_ho, ks=ks, metric=m) for m in ("euclidean", "cosine")}
        for name, pv in preds.items()
    }
    return out


def _answer_pca_dirs(Y, tr: np.ndarray, k: int, dev, block: int) -> np.ndarray:
    """Top-k answer-PCA directions (H, k), TRAIN-fit (streamed covariance +
    robust eigh; descending). The #1092 pca48 convention at this run's n."""
    h = Y.shape[1]
    s1 = torch.zeros(h, dtype=torch.float64, device=dev)
    s2 = torch.zeros((h, h), dtype=torch.float64, device=dev)
    for s in range(0, len(tr), block):
        yb = torch.as_tensor(np.asarray(Y[tr[s : s + block]]), dtype=torch.float64, device=dev)
        s1 += yb.sum(0)
        s2 += yb.T @ yb
    n = len(tr)
    mu = s1 / n
    cov = s2 / n - torch.outer(mu, mu)
    _, v = _eigh_robust(cov)
    k = int(min(k, h, max(1, n - 1)))
    return v[:, -k:].flip(-1).cpu().numpy()  # (H, k) descending


# ── run: the six reads ─────────────────────────────────────────────────────────────


def _regime_fp(args, split_doc, n_q: int) -> str:
    """Resume regime fingerprint: every output-affecting key (#722 r3 law).
    Recorded-only today; the realized n_q (G1 ladder) is output-affecting."""
    payload = json.dumps(
        {
            "split_shas": {k: v["sha256"] for k, v in split_doc["sets"].items()},
            "layers": args.layers,
            "lambdas": [float(x) for x in LAMBDAS],
            "n_boot": args.n_boot,
            "pca_dirs": args.pca_dirs,
            "inner_cv_folds": args.inner_cv_folds,
            "queries_per_prefix_realized": int(n_q),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _ladder_split_view(split: dict, n_q_grid: int, n_q: int) -> dict:
    """G1-ladder-aware coverage denominators (r2 blocker-1(b)): the split doc's
    intended counts were built on the FULL n_q_grid grid, so a clean descope
    capture (only q < n_q realized per prefix) would read n_q/n_q_grid coverage
    and trip the floor. Intersect each set's intended rows with q < realized
    n_q; identity when the ladder never fired. Row ci = prefix_pos*n_q_grid+q."""
    if int(n_q) == int(n_q_grid):
        return split
    sets = {
        name: {**s, "n": int(sum(1 for c in s["ci"] if int(c) % n_q_grid < n_q))}
        for name, s in split["sets"].items()
    }
    return {**split, "sets": sets}


def _load_manifest_bundle(args):
    """Crossed manifest bundle via the S1 loader (shim Namespace: the loader
    resolves {out_dir}/sampling_manifest; --local-manifest-dir must BE such a
    dir)."""
    if args.local_manifest_dir:
        local = Path(args.local_manifest_dir)
        assert local.name == GG.CROSSED_MANIFEST_LOCAL, (
            f"--local-manifest-dir must be a '{GG.CROSSED_MANIFEST_LOCAL}' dir, got {local}"
        )
        shim = SimpleNamespace(
            out_dir=local.parent, manifest_from_hf=False, crossed_hf_prefix=args.hf_prefix
        )
    else:
        shim = SimpleNamespace(
            out_dir=Path(args.out_local), manifest_from_hf=True, crossed_hf_prefix=args.hf_prefix
        )
    return GG._load_crossed_manifest(shim)


def _stage_pilot_meta(args) -> dict | None:
    """Stage crossed_pilot_meta.json into git out_eval (plan §6.5) + return it.
    Local path wins; else Hub; else None (recorded skip — smoke without one)."""
    if args.pilot_meta:
        doc = json.loads(Path(args.pilot_meta).read_text())
    elif not args.local_capture_dir:
        from huggingface_hub import hf_hub_download

        p = hub.retry_transient(
            lambda: hf_hub_download(
                C.HF_DATA_REPO,
                f"{args.hf_prefix}/{GG.CAPTURE_SUBDIR}/{GG.CROSSED_PILOT_META_NAME}",
                repo_type="dataset",
            ),
            what="crossed pilot meta fetch",
        )
        doc = json.loads(Path(p).read_text())
    else:
        logger.warning(
            "[reads] no pilot meta (local capture, none passed) — sae verdict from chunks"
        )
        return None
    C.write_json_atomic(Path(args.out_eval) / GG.CROSSED_PILOT_META_NAME, doc)
    return doc


def run_reads(args) -> int:  # noqa: C901 — the six-read pipeline is one linear pass
    t0 = time.time()
    layers = [int(x) for x in args.layers.split(",")]
    dev = torch.device(args.device)
    out_eval = Path(args.out_eval)
    out_eval.mkdir(parents=True, exist_ok=True)
    (out_eval / "cells").mkdir(exist_ok=True)
    pred_dir = Path(args.out_local) / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    perfeature_dir = Path(args.out_local) / "perfeature"
    perfeature_dir.mkdir(parents=True, exist_ok=True)

    prefix_rows, bank, split_doc, cmeta = _load_manifest_bundle(args)
    n_q_grid = int(cmeta["n_queries"])
    pilot_meta = _stage_pilot_meta(args)
    n_q = int(
        (pilot_meta or {}).get("queries_per_prefix_realized", args.queries_per_prefix or n_q_grid)
    )
    split = MTF.load_split(_split_path(args))
    for name, s in split["prefix_sets"].items():  # prefix-set shas (row sets via load_split)
        got = GG._sha_int_list([int(x) for x in s["pi"]])
        assert got == s["sha256"], f"prefix set {name!r} sha mismatch"

    C.phase("crossed-assemble")
    mm, ci, ameta = assemble_crossed(args, layers, n_q)
    h_dim = int(ameta["h_dim"])
    sets = MTF.split_positions(split, ci)
    bad = MTF._coverage_shortfalls(
        sets, _ladder_split_view(split, n_q_grid, n_q), args.coverage_floor
    )
    if bad:
        raise SystemExit(f"capture coverage below floor: {bad} (missing fleet shard?)")
    tr, val, ho = sets["train"], sets["val"], sets["holdout"]
    pid = ci // n_q_grid

    # averaged grain: complete-case grid + per-prefix means of cx / vx per layer
    cc, row_pos, n_excluded = _grid_positions(ci, n_q_grid, n_q)
    cc_set = {int(p) for p in cc}
    cc_index = {int(p): i for i, p in enumerate(cc)}
    avg_sets = {
        name: np.asarray(
            sorted(cc_index[int(p)] for p in split["prefix_sets"][name]["pi"] if int(p) in cc_set),
            dtype=np.int64,
        )
        for name in ("train", "val", "test", "holdout")
    }
    tr_p, ho_p = avg_sets["train"], avg_sets["holdout"]
    logger.info(
        "[reads] %d rows; grid %d complete prefixes x %d queries (%d excluded); "
        "avg train/holdout = %d/%d prefixes",
        len(ci),
        len(cc),
        n_q,
        n_excluded,
        len(tr_p),
        len(ho_p),
    )

    # bank bare states (the bare arm's 20 unique inputs)
    bq = _load_bank_bare(args, layers, n_q, h_dim)
    qid = (ci % n_q_grid).astype(np.int64)

    regime = _regime_fp(args, split, n_q)
    n_cells_fence = (len(ARMS) + 1) * len(layers)  # +1 = the stitch cell per layer
    first_wall: float | None = None
    fits_out: dict = {"cells": {}, "induced": {}, "stitch": {}, "regime_fp": regime}
    baselines_out: dict = {"cells": {}}
    operator_out: dict = {
        "per_layer": {},
        "null": {
            "kind": (
                "Haar-random subspace band == spectrum-matched random-map null for angle "
                "reads (Haar U/V make the top-k subspaces Haar; the spectrum only orders them)"
            ),
            "n_draws": int(args.n_operator_nulls),
            "seed": OPERATOR_NULL_SEED,
        },
    }

    C.phase("crossed-fits")
    fence_t0 = time.time()
    for li in layers:
        Y = mm[("vx", li)]
        y_ho = np.asarray(Y[ho], dtype=np.float64)
        # averaged targets/inputs for this layer (complete-case grid means)
        flat = row_pos.reshape(-1)
        Yg = np.asarray(Y[flat], dtype=np.float32).reshape(len(cc), n_q, h_dim)
        Ya = Yg.mean(axis=1, dtype=np.float64)
        del Yg
        Xg = np.asarray(mm[("cx", li)][flat], dtype=np.float32).reshape(len(cc), n_q, h_dim)
        Xa = Xg.mean(axis=1, dtype=np.float64)
        del Xg
        ya_ho = Ya[ho_p]

        arm_fits: dict[str, dict] = {}
        for arm in ARMS:
            cell = f"{arm}_L{li}"
            t_cell = time.time()
            if arm in ("context", "prefix"):
                X = mm[(ARM_MM[arm], li)]
                fac, pred_ho, meta, val_stats = _fit_dense_cell(
                    X, Y, tr, val, ho, dev, args.ridge_block
                )
                meta["n_train_unique_x"] = (
                    int(len(set(pid[tr].tolist()))) if arm == "prefix" else int(len(tr))
                )
                summ = _cell_summary(pred_ho, y_ho, args.n_boot)
                Xtr_b, Xho_b = np.asarray(X[tr]), np.asarray(X[ho])
                mse_tbl = {lam: v["mse"] for lam, v in val_stats.items()}
            elif arm == "bare":
                Xb = np.asarray(bq[:, layers.index(li), :], dtype=np.float32)[qid]
                fac, pred_ho, meta, val_stats = _fit_dense_cell(
                    Xb, Y, tr, val, ho, dev, args.ridge_block
                )
                meta["n_train_unique_x"] = int(n_q)
                meta["note"] = (
                    f"rank(X) <= {n_q} BY CONSTRUCTION — the query-main-effect read; "
                    "predictions take <= n_q distinct values (stated caveat)"
                )
                summ = _cell_summary(pred_ho, y_ho, args.n_boot)
                Xtr_b, Xho_b = Xb[tr], Xb[ho]
                mse_tbl = {lam: v["mse"] for lam, v in val_stats.items()}
            else:  # averaged-independent (R1's 4th arm + R2 SECONDARY)
                lam_sel, cv_meta = _inner_cv_lambda(
                    Xa, Ya, tr_p, dev, args.ridge_block, args.inner_cv_folds, GG.CROSSED_SEED
                )
                fac = PF._ridge_factorize(Xa, Ya, tr_p, dev, args.ridge_block)
                pred_ho = PF._ridge_predict_one(Xa, ho_p, fac, lam_sel, dev, args.ridge_block)
                meta = {
                    "fitter": "ridge (shared-Gram primal, parent verbatim)",
                    "selection": "inner CV within the train prefixes (never the val set)",
                    "selected_lambda": float(lam_sel),
                    "inner_cv": cv_meta,
                    "n_train": int(len(tr_p)),
                    "n_train_unique_x": int(len(tr_p)),
                    "d": int(h_dim),
                    "regime_note": (
                        f"n_train {len(tr_p)} vs d {h_dim} "
                        f"({len(tr_p) / max(1, h_dim):.2f}x — stated per plan §6 R1)"
                    ),
                }
                summ = _cell_summary(pred_ho, ya_ho, args.n_boot)
                Xtr_b, Xho_b = Xa[tr_p], Xa[ho_p]
                mse_tbl = {
                    lam: v / max(1, len(tr_p)) for lam, v in cv_meta["cv_sse_by_lambda"].items()
                }
            np.savez(pred_dir / f"{cell}.tmp.npz", pred16=pred_ho.astype(np.float16))
            (pred_dir / f"{cell}.tmp.npz").replace(pred_dir / f"{cell}_ridge.npz")
            true_b = Ya[ho_p] if arm == "avg" else y_ho
            baselines_out["cells"][cell] = _baselines_cell(
                Xtr_b,
                np.asarray((Ya[tr_p] if arm == "avg" else Y[tr])),
                Xho_b,
                true_b,
                {"ridge": pred_ho},
            )
            fits_out["cells"][cell] = {**meta, **summ}
            arm_fits[arm] = {"fac": fac, "meta": meta, "mse": mse_tbl}
            C.write_json_atomic(
                out_eval / "cells" / f"{cell}.json",
                {**meta, **summ, "regime_fp": regime, "wall_s": round(time.time() - t_cell, 1)},
            )
            logger.info(
                "[fits] unit %s: R2=%.4f lambda=%.3g elapsed=%.0fs",
                cell,
                summ["holdout_r2"],
                meta["selected_lambda"],
                time.time() - t_cell,
            )
            if first_wall is None:
                first_wall = time.time() - t_cell
            elif MTF._fence_should_halt(
                time.time() - fence_t0, first_wall, n_cells_fence, args.fence_mult
            ):
                rep = {
                    "gate": "G3",
                    "first_cell_wall_s": first_wall,
                    "elapsed_s": time.time() - fence_t0,
                    "n_cells": n_cells_fence,
                    "fence_mult": args.fence_mult,
                }
                C.write_json_atomic(out_eval / "fence_report.json", rep)
                logger.error("[G3] fits fence tripped: %s", rep)
                return RC_FENCE

        # R2: induced averaged map PRIMARY (context fit applied to averaged inputs)
        ctx = arm_fits["context"]
        payload = _ridge_payload(ctx["fac"], ctx["meta"]["selected_lambda"])
        pred_ind = PF.apply_map(payload, Xa[ho_p], dev)
        ind_summ = _cell_summary(pred_ind, ya_ho, args.n_boot)
        np.savez(pred_dir / f"induced_avg_L{li}.tmp.npz", pred16=pred_ind.astype(np.float16))
        (pred_dir / f"induced_avg_L{li}.tmp.npz").replace(pred_dir / f"induced_avg_L{li}_ridge.npz")
        baselines_out["cells"][f"induced_avg_L{li}"] = _baselines_cell(
            Xa[tr_p], Ya[tr_p], Xa[ho_p], ya_ho, {"induced": pred_ind}
        )
        fits_out["induced"][f"L{li}"] = {
            **ind_summ,
            "applied_lambda": ctx["meta"]["selected_lambda"],
            "definition": (
                "per-row context-arm ridge (train rows) applied to x̄_p = mean_q cx(p,q), "
                "scored vs ȳ_p on holdout prefixes (the #1092 operator-coincidence design)"
            ),
            "gap_induced_minus_independent_r2": float(
                ind_summ["holdout_r2"] - fits_out["cells"][f"avg_L{li}"]["holdout_r2"]
            ),
        }
        logger.info(
            "[R2] L%d induced R2=%.4f independent R2=%.4f gap=%.4f",
            li,
            ind_summ["holdout_r2"],
            fits_out["cells"][f"avg_L{li}"]["holdout_r2"],
            fits_out["induced"][f"L{li}"]["gap_induced_minus_independent_r2"],
        )

        # R4: disjoint stitch [prefix-end; bare-query] -> v_x
        t_cell = time.time()
        Xb_layer = np.asarray(bq[:, layers.index(li), :], dtype=np.float32)[qid]
        X_st = np.empty((len(ci), 2 * h_dim), dtype=np.float32)
        X_st[:, :h_dim] = mm[("px", li)][:]
        X_st[:, h_dim:] = Xb_layer
        fac_st, pred_st, meta_st, val_st = _fit_dense_cell(
            X_st, Y, tr, val, ho, dev, args.ridge_block
        )
        st_summ = _cell_summary(pred_st, y_ho, args.n_boot)
        np.savez(pred_dir / f"stitch_L{li}.tmp.npz", pred16=pred_st.astype(np.float16))
        (pred_dir / f"stitch_L{li}.tmp.npz").replace(pred_dir / f"stitch_L{li}_ridge.npz")
        baselines_out["cells"][f"stitch_L{li}"] = _baselines_cell(
            None,
            np.asarray(Y[tr]),
            None,
            y_ho,
            {"ridge": pred_st},
            identity_note=(
                f"inapplicable — stitched input dim {2 * h_dim} != target dim {h_dim} "
                "(dimension-mismatched; plan §6 R4 / critic Should-Fix)"
            ),
        )
        fits_out["stitch"][f"L{li}"] = {
            **meta_st,
            **st_summ,
            "reference_r2": {
                "context": fits_out["cells"][f"context_L{li}"]["holdout_r2"],
                "prefix_only": fits_out["cells"][f"prefix_L{li}"]["holdout_r2"],
                "bare_only": fits_out["cells"][f"bare_L{li}"]["holdout_r2"],
            },
        }
        del X_st
        logger.info(
            "[R4] L%d stitch R2=%.4f vs context %.4f (%.0fs)",
            li,
            st_summ["holdout_r2"],
            fits_out["cells"][f"context_L{li}"]["holdout_r2"],
            time.time() - t_cell,
        )

        # R5: operator geometry across arms at matched λ (facs still alive)
        arm_fits["stitch"] = {
            "fac": fac_st,
            "meta": meta_st,
            "mse": {lam: v["mse"] for lam, v in val_st.items()},
        }
        operator_out["per_layer"][str(li)] = _operator_geometry_layer(
            arm_fits, mm, li, tr, tr_p, Xa, Ya, args, dev
        )
        del arm_fits, fac_st
    C.write_json_atomic(out_eval / "crossed_fits.json", _with_meta(fits_out, args, t0))
    C.write_json_atomic(out_eval / "mapping_baselines.json", _with_meta(baselines_out, args, t0))
    C.write_json_atomic(out_eval / "operator_geometry.json", _with_meta(operator_out, args, t0))

    C.phase("crossed-anova")
    anova = _phase_anova(args, mm, layers, cc, row_pos, n_q, tr, dev, h_dim)
    C.write_json_atomic(out_eval / "anova.json", _with_meta(anova, args, t0))

    C.phase("crossed-transfer")
    transfer = _phase_transfer(
        args, mm, layers, ci, n_q_grid, n_q, prefix_rows, sets, cc, row_pos, avg_sets, dev, h_dim
    )
    fits_out["transfer"] = transfer
    C.write_json_atomic(out_eval / "crossed_fits.json", _with_meta(fits_out, args, t0))
    # stitch.json (plan §6.5 glob) = the per-layer stitch block, standalone
    C.write_json_atomic(
        out_eval / "stitch.json", _with_meta({"per_layer": fits_out["stitch"]}, args, t0)
    )

    C.phase("crossed-sae")
    sae_verdict = _sae_verdict(pilot_meta, ameta)
    _phase_sae(
        args,
        Path(args.mm_dir),
        ci,
        n_q_grid,
        n_q,
        sets,
        cc,
        row_pos,
        avg_sets,
        dev,
        sae_verdict,
        out_eval,
        perfeature_dir,
        mm,
    )

    C.phase("crossed-upload")
    if not args.no_upload:
        shim = SimpleNamespace(hf_prefix=args.hf_prefix, upload_prefix="")
        files = sorted(
            str(p.relative_to(out_eval))
            for p in [
                out_eval / "crossed_fits.json",
                out_eval / "mapping_baselines.json",
                out_eval / "operator_geometry.json",
                out_eval / "anova.json",
                out_eval / "stitch.json",
                out_eval / "sae_perfeature.json",
                out_eval / "perfeature_crossed_summary.csv",
                out_eval / GG.CROSSED_PILOT_META_NAME,
                *sorted((out_eval / "cells").glob("*.json")),
            ]
            if p.is_file()
        )
        MTF._upload_analysis_tensors(
            shim,
            [
                ("summaries", out_eval, files),
                ("pred", pred_dir, None),
                ("perfeature", perfeature_dir, None),
            ],
        )
    C.phase("done")
    logger.info("[reads] done in %.1f min", (time.time() - t0) / 60)
    return 0


def _split_path(args) -> Path:
    if args.local_manifest_dir:
        return Path(args.local_manifest_dir) / "split_1738_crossed.json"
    return Path(args.out_local) / GG.CROSSED_MANIFEST_LOCAL / "split_1738_crossed.json"


def _load_bank_bare(args, layers, n_q, h_dim) -> torch.Tensor:
    """bank_bare.pt (Q, L, H) — local dir first (smoke), else Hub."""
    if args.local_capture_dir:
        p = Path(args.local_capture_dir) / "bank_bare.pt"
    else:
        from huggingface_hub import hf_hub_download

        p = Path(
            hub.retry_transient(
                lambda: hf_hub_download(
                    C.HF_DATA_REPO,
                    f"{args.hf_prefix}/{GG.BARE_BANK_SUBDIR}/bank_bare.pt",
                    repo_type="dataset",
                ),
                what="bank bare capture fetch",
            )
        )
    d = torch.load(p, map_location="cpu", weights_only=False)
    bq = d["bq_last"].float()
    assert list(d["layers"]) == layers, (d["layers"], layers)
    assert bq.shape[0] >= n_q and bq.shape[2] == h_dim, (bq.shape, n_q, h_dim)
    return bq


def _sae_verdict(pilot_meta: dict | None, ameta: dict) -> dict:
    if pilot_meta is not None:
        s = pilot_meta.get("sae", {})
        return {"enabled": bool(s.get("enabled")), "source": "pilot meta", "gate": s}
    return {
        "enabled": bool(ameta.get("sae_any")),
        "source": "chunk sae_enabled flags (no pilot meta — smoke/local)",
    }


def _operator_geometry_layer(arm_fits, mm, li, tr, tr_p, Xa, Ya, args, dev) -> dict:
    """R5 for one layer: per-arm operator SVDs + pairwise subspace angles at the
    pair's matched λ (argmin of summed normalized val/cv MSE — the #1092
    matched-λ convention on this round's selection surfaces) vs Haar nulls."""
    out: dict = {"arms": {}, "pairs": {}}
    null_cache: dict = {}
    gen = torch.Generator().manual_seed(OPERATOR_NULL_SEED)
    svds: dict[str, dict] = {}
    arm_rows = {"context": ("cx", tr), "prefix": ("px", tr), "stitch": (None, tr)}
    for arm, f in arm_fits.items():
        lam = f["meta"]["selected_lambda"]
        w = _w_raw(f["fac"], lam)  # (H_in, D)
        m = w.T  # operator y = M x_c: (D, H_in)
        u, s, vh = torch.linalg.svd(m, full_matrices=False)
        rank = int((s > s.max() * 1e-10).sum()) if s.numel() else 0
        svds[arm] = {"u": u, "s": s, "vh": vh, "rank": rank}
        gx = float("nan")
        if arm in ("context", "prefix"):
            gx = _gamma(mm[(arm_rows[arm][0], li)], tr, f["fac"]["xmu"], args.ridge_block)
        elif arm == "avg":
            gx = _gamma(Xa, tr_p, f["fac"]["xmu"], args.ridge_block)
        gy = _gamma(
            Ya if arm == "avg" else mm[("vx", li)],
            tr_p if arm == "avg" else tr,
            f["fac"]["ymu"],
            args.ridge_block,
        )
        entry = {
            "selected_lambda": lam,
            "rank": rank,
            "k90_output": OPS._k90(s.cpu()) if s.numel() else 0,
            "gamma_x_mean_norm_fraction": gx,
            "gamma_y_mean_norm_fraction": gy,
        }
        if arm == "bare":
            entry["gamma_x_note"] = (
                "N/A: bare X = the <=n_q-row query bank tiled over rows (no per-prefix "
                "x-variation; a train-row mean-norm fraction is degenerate)"
            )
        elif arm == "stitch":
            entry["gamma_x_note"] = (
                "N/A: stitch X concatenates prefix-end + bare-query spaces (2H); a single "
                "mean-norm fraction is not comparable to the H-dim arms"
            )
        out["arms"][arm] = entry
    pairs = [
        ("context", "prefix"),
        ("context", "bare"),
        ("prefix", "bare"),
        ("context", "avg"),
        ("prefix", "avg"),
        ("bare", "avg"),
    ]
    for a, b in pairs:
        fa, fb = arm_fits[a], arm_fits[b]
        lams = sorted(set(fa["mse"]) & set(fb["mse"]))
        ma = min(v for v in fa["mse"].values() if np.isfinite(v))
        mb = min(v for v in fb["mse"].values() if np.isfinite(v))
        lam_m = min(lams, key=lambda x: fa["mse"][x] / ma + fb["mse"][x] / mb)
        sv_a = _svd_at(fa, lam_m)
        sv_b = _svd_at(fb, lam_m)
        pair_out: dict = {"matched_lambda": float(lam_m)}
        k48 = min(48, sv_a["rank"], sv_b["rank"])
        # OPS._k90 builds a CPU scalar for searchsorted; the fac tensors (and hence
        # the SVD outputs) are CUDA on the GPU lane, so device-match via .cpu().
        k90a, k90b = OPS._k90(sv_a["s"].cpu()), OPS._k90(sv_b["s"].cpu())
        for name, k1, k2, A, B in (
            ("output_k48", k48, k48, sv_a["u"], sv_b["u"]),
            ("input_k48", k48, k48, sv_a["vh"].T, sv_b["vh"].T),
            ("output_k90", min(k90a, sv_a["rank"]), min(k90b, sv_b["rank"]), sv_a["u"], sv_b["u"]),
            (
                "input_k90",
                min(k90a, sv_a["rank"]),
                min(k90b, sv_b["rank"]),
                sv_a["vh"].T,
                sv_b["vh"].T,
            ),
        ):
            if min(k1, k2) < 1:
                pair_out[name] = {"degenerate": f"k=({k1},{k2})"}
                continue
            if A.shape[1] < k1 or B.shape[1] < k2:
                k1, k2 = min(k1, A.shape[1]), min(k2, B.shape[1])
            angles = OPS._angles_between(A[:, :k1].double(), B[:, :k2].double())
            d = int(A.shape[0])
            key = (d, k1, k2)
            if key not in null_cache:
                null_cache[key] = OPS._angle_null_band(
                    d, k1, k2, args.n_operator_nulls, 16, gen, 384
                )
            null = dict(null_cache[key])
            null.pop("draws_mean_angle_rad", None)  # keep the JSON small
            pair_out[name] = {
                "k": [int(k1), int(k2)],
                "mean_angle_deg": float(np.degrees(np.mean(angles))),
                "null": null,
            }
        out["pairs"][f"{a}|{b}"] = pair_out
    return out


def _svd_at(f: dict, lam: float) -> dict:
    w = _w_raw(f["fac"], lam)
    u, s, vh = torch.linalg.svd(w.T, full_matrices=False)
    return {"u": u, "s": s, "vh": vh, "rank": int((s > s.max() * 1e-10).sum()) if s.numel() else 0}


def _phase_anova(args, mm, layers, cc, row_pos, n_q, tr, dev, h_dim) -> dict:
    """R3: crossed ANOVA at scale — vectorized grid means (CCS.anova_shares on
    the (P, Q, H) complete-case grid; overall = SS-weighted reaggregation of the
    per-dim shares) + per-direction top-48 answer-PCA versions; parent
    K-resample floor embedded as a comparator (never a subtraction)."""
    out: dict = {
        "n_complete_prefixes": int(len(cc)),
        "n_queries": int(n_q),
        "per_layer": {},
        "interaction_note": (
            "temp-1.0 sampling noise is CONFOUNDED into the interaction share (one draw "
            "per cell; #1092 generated greedy — stated); the parent K-resample per-context "
            "answer-sampling floor below is the REFERENCE comparator, not a subtraction"
        ),
    }
    flat = row_pos.reshape(-1)
    for li in layers:
        g = np.asarray(mm[("vx", li)][flat], dtype=np.float64).reshape(len(cc), n_q, h_dim)
        sh = CCS.anova_shares(g)
        w = np.nan_to_num(sh["ss_tot"], nan=0.0)
        tot = float(w.sum())
        overall = {
            k: float((np.nan_to_num(sh[k], nan=0.0) * w).sum() / max(tot, 1e-30))
            for k in ("share_prefix", "share_query", "share_inter")
        }
        dirs = _answer_pca_dirs(mm[("vx", li)], tr, args.pca_dirs, dev, args.ridge_block)
        gd = g.reshape(-1, h_dim) @ dirs
        shd = CCS.anova_shares(gd.reshape(len(cc), n_q, dirs.shape[1]))
        out["per_layer"][str(li)] = {
            "overall": overall,
            "n_pca_dirs": int(dirs.shape[1]),
            "per_direction": {
                k: [float(x) for x in shd[k]]
                for k in ("share_prefix", "share_query", "share_inter")
            },
        }
        logger.info("[R3] L%d shares %s", li, {k: round(v, 4) for k, v in overall.items()})
        del g, gd
    ks_path = Path(args.kresample_summary)
    if ks_path.is_file():
        ks = json.loads(ks_path.read_text())
        out["kresample_reference"] = {
            "path": str(ks_path),
            "per_layer": {
                k: {
                    "floor_share_median": v.get("floor_share_median"),
                    "floor_share_mean": v.get("floor_share_mean"),
                }
                for k, v in ks.get("per_layer", {}).items()
            },
        }
    else:
        out["kresample_reference"] = {"skipped": f"missing {ks_path}"}
        logger.warning("[R3] kresample reference missing at %s — comparator skipped", ks_path)
    return out


def _phase_transfer(
    args, mm, layers, ci, n_q_grid, n_q, prefix_rows, sets, cc, row_pos, avg_sets, dev, h_dim
) -> dict:
    """lmsys_transfer twin (minor; MTF._compute_transfer convention): ridge fit
    on LMSYS-prefix train rows, scored on WildChat-prefix holdout rows — context
    (per-row) + averaged arms, at L19 (or the last layer when 19 is absent —
    smoke layers; recorded)."""
    li = 19 if 19 in layers else layers[-1]
    corpus_by_pos = {int(r["i"]): r["corpus"] for r in prefix_rows}
    corp = np.asarray([corpus_by_pos.get(int(p), "?") for p in (ci // n_q_grid)])
    tr, val, ho = sets["train"], sets["val"], sets["holdout"]
    out: dict = {"control": "lmsys_transfer", "layer": int(li), "cells": {}}
    tr_lm = tr[corp[tr] == "lmsys"]
    val_lm = val[corp[val] == "lmsys"]
    ho_wc = ho[corp[ho] == "wildchat"]
    ho_lm = ho[corp[ho] == "lmsys"]
    if len(ho_wc) == 0 or len(tr_lm) == 0 or len(val_lm) == 0:
        out["skipped"] = (
            f"empty cells (tr_lm={len(tr_lm)}, val_lm={len(val_lm)}, ho_wc={len(ho_wc)})"
        )
        return out
    X = mm[("cx", li)]
    Y = mm[("vx", li)]
    eval_idx = np.concatenate([ho_wc, ho_lm])
    pred, meta = PF.fit_ridge(X, Y, tr_lm, val_lm, eval_idx, LAMBDAS, dev, args.ridge_block)
    r2_wc, _ = F._recon_point(pred[: len(ho_wc)], np.asarray(Y[ho_wc], dtype=np.float64))
    r2_lm, _ = F._recon_point(pred[len(ho_wc) :], np.asarray(Y[ho_lm], dtype=np.float64))
    out["cells"]["context"] = {
        "n_train_lmsys": int(len(tr_lm)),
        "n_holdout_wildchat": int(len(ho_wc)),
        "n_holdout_lmsys": int(len(ho_lm)),
        "transfer_r2_wildchat_holdout": float(r2_wc),
        "within_r2_lmsys_holdout": float(r2_lm),
        "selected_lambda": meta["selected_lambda"],
    }
    # averaged arm: fit on LMSYS averaged train prefixes (inner-CV λ), score WC holdout
    cc_corp = np.asarray([corpus_by_pos.get(int(p), "?") for p in cc])
    tr_p, ho_p = avg_sets["train"], avg_sets["holdout"]
    tr_p_lm = tr_p[cc_corp[tr_p] == "lmsys"]
    ho_p_wc = ho_p[cc_corp[ho_p] == "wildchat"]
    ho_p_lm = ho_p[cc_corp[ho_p] == "lmsys"]
    if len(tr_p_lm) < 2 or len(ho_p_wc) == 0:
        out["cells"]["avg"] = {
            "skipped": f"empty averaged cells (tr_p_lm={len(tr_p_lm)}, ho_p_wc={len(ho_p_wc)})"
        }
        return out
    flat = row_pos.reshape(-1)
    Xa = (
        np.asarray(mm[("cx", li)][flat], dtype=np.float32)
        .reshape(len(cc), n_q, h_dim)
        .mean(axis=1, dtype=np.float64)
    )
    Ya = (
        np.asarray(mm[("vx", li)][flat], dtype=np.float32)
        .reshape(len(cc), n_q, h_dim)
        .mean(axis=1, dtype=np.float64)
    )
    lam_avg, _cv = _inner_cv_lambda(
        Xa, Ya, tr_p_lm, dev, args.ridge_block, args.inner_cv_folds, GG.CROSSED_SEED
    )
    fac = PF._ridge_factorize(Xa, Ya, tr_p_lm, dev, args.ridge_block)
    pred_wc = PF._ridge_predict_one(Xa, ho_p_wc, fac, lam_avg, dev, args.ridge_block)
    r2_p_wc, _ = F._recon_point(pred_wc, Ya[ho_p_wc])
    r2_p_lm = float("nan")
    if len(ho_p_lm):
        pred_plm = PF._ridge_predict_one(Xa, ho_p_lm, fac, lam_avg, dev, args.ridge_block)
        r2_p_lm, _ = F._recon_point(pred_plm, Ya[ho_p_lm])
    out["cells"]["avg"] = {
        "n_train_lmsys_prefixes": int(len(tr_p_lm)),
        "n_holdout_wildchat_prefixes": int(len(ho_p_wc)),
        "n_holdout_lmsys_prefixes": int(len(ho_p_lm)),
        "transfer_r2_wildchat_holdout": float(r2_p_wc),
        "within_r2_lmsys_holdout": float(r2_p_lm),
        "selected_lambda": float(lam_avg),
        "selection": "inner CV within the LMSYS train prefixes",
    }
    return out


def _mask_robustness(side: list[Path]) -> dict:
    """Plan §6 with/without-mask robustness twin (r2 blocker-2): compare the SAE
    answer pooling WITH the #1482 token-inlier mask (the registered R6 inputs)
    against the UNMASKED pooling of the SAME per-token features. Capture stores
    the unmasked (``nm_*``) pooled trio ONLY where the mask bites (n_inl < n_ans
    — ``encode`` is row-independent, so masked == unmasked EXACTLY on
    outlier-free rows, reconstructed here as cos=1.0 rows). Per differing row:
    cosine(masked, unmasked) per pooling; plus the all-tokens-outlier count
    (masked family empty — the maximal mask effect, excluded from cosines)."""
    per_pool: dict[str, list[float]] = {"mean": [], "max": [], "frac": []}
    n_equal = n_diff = n_all_outlier = 0
    missing_schema = 0
    val_key = {"mean": "ans_mean", "max": "ans_max", "frac": "ans_frac"}
    for p in side:
        d = torch.load(p, map_location="cpu", weights_only=False)
        if "nm_row_ptr" not in d:
            missing_schema += 1
            continue
        n_ans = np.asarray(d["n_ans_tokens"], dtype=np.int64)
        n_inl = np.asarray(d["n_inlier_tokens"], dtype=np.int64)
        n_all_outlier += len(d.get("sae_skipped_ci", []))
        n_equal += int(((n_inl == n_ans) & (n_ans > 0)).sum())
        rp = np.asarray(d["row_ptr"], dtype=np.int64)
        nrp = np.asarray(d["nm_row_ptr"], dtype=np.int64)
        fi = np.asarray(d["feat_idx"], dtype=np.int64)
        nfi = np.asarray(d["nm_feat_idx"], dtype=np.int64)
        for j in np.nonzero((n_ans > 0) & (n_inl < n_ans))[0]:
            n_diff += 1
            mi = fi[rp[j] : rp[j + 1]]
            ui = nfi[nrp[j] : nrp[j + 1]]
            _common, ia, ib = np.intersect1d(mi, ui, return_indices=True)
            for pool in ("mean", "max", "frac"):
                mv = np.asarray(d[val_key[pool]][rp[j] : rp[j + 1]], dtype=np.float64)
                uv = np.asarray(d[f"nm_{pool}"][nrp[j] : nrp[j + 1]], dtype=np.float64)
                den = float(np.linalg.norm(mv) * np.linalg.norm(uv))
                per_pool[pool].append(
                    float((mv[ia] * uv[ib]).sum()) / den if den > 0 else float("nan")
                )
    if missing_schema and not (n_diff or n_equal):
        return {"skipped": "capture chunks predate the mask-twin (nm_*) schema"}
    out: dict = {
        "n_rows_mask_equal": int(n_equal),
        "n_rows_mask_differs": int(n_diff),
        "n_rows_all_tokens_outlier": int(n_all_outlier),
        "n_chunks_missing_nm_schema": int(missing_schema),
        "note": (
            "cosine(masked, unmasked pooled answer features) per pooling over the "
            "mask-differing rows; mask-equal rows are cos=1.0 by construction "
            "(row-independent encode); all-outlier rows have an empty masked family"
        ),
    }
    for pool, vals in per_pool.items():
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=np.float64)
        out[pool] = (
            {
                "n": int(len(v)),
                "cos_min": float(v.min()),
                "cos_p05": float(np.quantile(v, 0.05)),
                "cos_median": float(np.median(v)),
                "cos_mean": float(v.mean()),
            }
            if len(v)
            else {"n": 0}
        )
    return out


def _phase_sae(
    args,
    mm_dir: Path,
    ci,
    n_q_grid,
    n_q,
    sets,
    cc,
    row_pos,
    avg_sets,
    dev,
    verdict: dict,
    out_eval: Path,
    perfeature_dir: Path,
    mm,
) -> None:
    """R6 (mechanical only — 0 judge calls, judged-label freeze): per-feature
    prefix/query/interaction shares on the grid (K=20 label-permutation nulls),
    encode-then-average feature maps (induced + independent), dense-latent
    stratification, decoder–r_B alignment (raw + scaffold-projected) on the top
    prefix-share tail; evidence persisted for #1773. Skips cleanly when the
    pilot verdict is sae_enabled=false."""
    import issue1738_sae_arm as SAEARM

    side = sorted((mm_dir / "sae_side").glob("*.pt"))
    if not verdict.get("enabled") or not side:
        doc = {
            "skipped": True,
            "reason": (
                "sae_enabled=false (pilot fitness gate FAIL or --no-sae) — R6 skipped "
                "cleanly; R1-R5 unaffected"
                if not verdict.get("enabled")
                else "no sae sidecars in the capture store"
            ),
            "verdict": verdict,
        }
        C.write_json_atomic(out_eval / "sae_perfeature.json", doc)
        logger.warning("[R6] %s", doc["reason"])
        return
    mask_rob = _mask_robustness(side)  # plan §6 nuisance-control twin (r2 blocker-2)
    logger.info("[R6] mask_robustness: %s", {k: v for k, v in mask_rob.items() if k != "note"})
    first = torch.load(side[0], map_location="cpu", weights_only=False)
    dict_size = int(first["sae"]["dict_size"])
    train_ci = {int(ci[r]) for r in sets["train"]}
    scan = SAEARM._scan_sae(side, train_ci, dict_size)
    assert np.array_equal(scan["ci"], ci), "sae sidecar ci order != assembly order"
    f_out, floor = SN.restrict(
        scan["out_fit"], scan["n_fit"], min(SAEARM.MAX_FEATURES_OUT, dict_size)
    )
    f_in = {
        arm: SN.restrict(
            scan["in_fit"][arm], scan["n_fit"], min(SAEARM.MAX_FEATURES_IN, dict_size)
        )[0]
        for arm in ("px", "cx")
    }
    logger.info(
        "[R6] restriction: F_out=%d F_in_px=%d F_in_cx=%d floor=%d (1%% of %d fit rows)",
        len(f_out),
        len(f_in["px"]),
        len(f_in["cx"]),
        floor,
        scan["n_fit"],
    )
    X, Y, _dense_mm, _h_dense = SAEARM._build_sae_matrices(
        side, scan, f_out, f_in, mm_dir / "sae_mm"
    )
    Ym = Y["mean"].tocsc()
    n_rows, f_dim = Y["mean"].shape
    flat = row_pos.reshape(-1)

    # per-feature grid ANOVA + K=20 label-permutation nulls (feature blocks)
    shares = {
        k: np.full(f_dim, np.nan, np.float32)
        for k in ("share_prefix", "share_query", "share_inter")
    }
    null_p = np.zeros((len(PERM_SEEDS), f_dim), np.float16)
    null_q = np.zeros((len(PERM_SEEDS), f_dim), np.float16)
    fb = int(args.sae_feature_block)
    for s in range(0, f_dim, fb):
        cols = np.arange(s, min(s + fb, f_dim))
        blk = np.asarray(Ym[:, cols].toarray(), dtype=np.float64)
        g = blk[flat].reshape(len(cc), n_q, len(cols))
        sh = CCS.anova_shares(g)
        for k in shares:
            shares[k][cols] = sh[k].astype(np.float32)
        for j, seed in enumerate(PERM_SEEDS):
            null_p[j, cols] = CCS.permutation_null_shares(g, "prefix", 1, seed, args.device)[0]
            null_q[j, cols] = CCS.permutation_null_shares(g, "query", 1, seed, args.device)[0]
        logger.info("[R6] anova feature block %d-%d/%d", s, int(cols[-1]) + 1, f_dim)
    p_prefix = CCS.perm_pvalues(shares["share_prefix"], null_p)
    p_query = CCS.perm_pvalues(shares["share_query"], null_q)

    # dense-latent stratification (#1092 density convention)
    frac_active = np.diff(Ym.indptr) / max(1, n_rows)
    dense_latent = frac_active >= np.nanpercentile(frac_active, CCS.DENSE_LATENT_PCTL)
    dec = np.nanquantile(frac_active, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(dec[1:-1], frac_active, side="right"), 0, 9)
    density_bands = {
        str(b): {
            "n": int((bins == b).sum()),
            "share_prefix_mean": float(np.nanmean(shares["share_prefix"][bins == b]))
            if (bins == b).any()
            else float("nan"),
            "share_query_mean": float(np.nanmean(shares["share_query"][bins == b]))
            if (bins == b).any()
            else float("nan"),
        }
        for b in range(10)
    }

    # encode-then-average feature maps (induced + independent; #1092 non-collapse read)
    tr, val, ho = sets["train"], sets["val"], sets["holdout"]
    tr_p, ho_p = avg_sets["train"], avg_sets["holdout"]
    fac_feat = SAEARM._GramFactor(X["cx"], tr, dev, args.ridge_block)
    cell = SAEARM._fit_cell(fac_feat, Y["mean"], tr, val, ho, LAMBDAS)
    lam_sel = float(cell["selected_lambda"])
    import scipy.sparse as sp

    ind = sp.coo_matrix(
        (
            np.full(len(flat), 1.0 / n_q, dtype=np.float64),
            (np.repeat(np.arange(len(cc)), n_q), flat),
        ),
        shape=(len(cc), n_rows),
    ).tocsr()
    Xavg = np.asarray((ind @ X["cx"]).todense() if sp.issparse(X["cx"]) else ind @ X["cx"])
    Yavg = np.asarray((ind @ Y["mean"]).todense())
    y_ho_avg = Yavg[ho_p]
    ymu = cell["ymu"]
    B = fac_feat.U.T @ fac_feat.xty_centered(Y["mean"], tr, ymu)
    inv = 1.0 / (fac_feat.s_eig + lam_sel)
    e_avg = fac_feat._std_np(np.asarray(Xavg[ho_p], dtype=np.float32)) @ fac_feat.U
    pred_ind_avg = ((e_avg * inv) @ B + ymu).cpu().numpy()
    r2_ind = SAEARM._pooled_r2(pred_ind_avg, y_ho_avg)
    lam_avg, cv_meta = _inner_cv_lambda(
        Xavg, Yavg, tr_p, dev, args.ridge_block, args.inner_cv_folds, GG.CROSSED_SEED
    )
    fac_avg = PF._ridge_factorize(Xavg, Yavg, tr_p, dev, args.ridge_block)
    pred_avg = PF._ridge_predict_one(Xavg, ho_p, fac_avg, lam_avg, dev, args.ridge_block)
    r2_avg = SAEARM._pooled_r2(pred_avg, y_ho_avg)
    feature_maps = {
        "per_row_cx_feat_to_ans_mean": {
            "holdout_r2": float(
                SAEARM._pooled_r2(
                    cell["pred_ho"],
                    np.asarray(Y["mean"][ho].toarray(), dtype=np.float64),
                )
            ),
            "selected_lambda": lam_sel,
            "n_train": int(len(tr)),
            "d_in": int(X["cx"].shape[1]),
        },
        "induced_averaged": {
            "holdout_r2": float(r2_ind),
            "definition": "per-row cx_feat→ans_mean map applied to averaged features",
        },
        "independent_averaged": {
            "holdout_r2": float(r2_avg),
            "selected_lambda": float(lam_avg),
            "selection": "inner CV within the train prefixes",
            "regime_note": f"n_train {len(tr_p)} vs d {Xavg.shape[1]}",
        },
    }

    # decoder–r_B alignment (raw + scaffold-projected) on the top prefix-share tail
    rb_block: dict = {"skipped": True, "reason": "--skip-rb-align (smoke)"}
    tail_n = min(int(args.rb_tail_n), f_dim)
    tail_pos = np.argsort(np.nan_to_num(shares["share_prefix"], nan=-1.0))[::-1][:tail_n]
    if not args.skip_rb_align:
        dm_rows = sets["train"][: min(20_000, len(sets["train"]))]
        li_sae = int(first["sae"]["layer"])
        dm = np.asarray(mm[("vx", li_sae)][dm_rows], dtype=np.float32)
        rb = CCS.rb_cosine_join(
            f_out[tail_pos].astype(np.int64),
            Path(args.out_local),
            args.n_operator_nulls,
            OPERATOR_NULL_SEED,
            args.device,
            dm,
        )
        rb_block = {
            "n_tail": int(tail_n),
            "null_p95_raw": rb["null_p95"],
            "null_p95_proj": rb["null_p95_proj"],
            "scaffold_rank": rb["scaffold_rank"],
            "rb_scaffold_mass_frac": rb["rb_scaffold_mass_frac"],
        }
    else:
        rb = None
        import inspect

        inspect.signature(CCS.rb_cosine_join).bind(  # smoke signature-bind (fenced branch)
            np.zeros(1, np.int64), Path("."), 1, 1, "cpu", np.zeros((2, 4), np.float32)
        )

    # evidence artifacts for #1773 (digest-only: ids + stats + top-row cis, never text)
    top_rows: dict[str, list[int]] = {}
    for pos in tail_pos[: min(100, tail_n)]:
        s0, s1 = Ym.indptr[pos], Ym.indptr[pos + 1]
        rws = Ym.indices[s0:s1]
        vals = Ym.data[s0:s1]
        top = rws[np.argsort(vals)[::-1][:8]]
        top_rows[str(int(f_out[pos]))] = [int(ci[r]) for r in top]
    np.savez(
        perfeature_dir / "perfeature_crossed.npz",
        feature_ids=f_out.astype(np.int64),
        share_prefix=shares["share_prefix"],
        share_query=shares["share_query"],
        share_inter=shares["share_inter"],
        p_prefix=p_prefix,
        p_query=p_query,
        frac_active=frac_active.astype(np.float32),
        dense_latent=dense_latent,
        null_prefix_draws=null_p,
        null_query_draws=null_q,
        rb_cos_max=(rb["cos_max"] if rb else np.zeros(0, np.float32)),
        rb_cos_max_proj=(rb["cos_max_proj"] if rb else np.zeros(0, np.float32)),
        rb_tail_positions=tail_pos.astype(np.int64),
    )
    (perfeature_dir / "top_rows_ci.json").write_text(json.dumps(top_rows))
    csv_path = out_eval / "perfeature_crossed_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(
            [
                "feature_id",
                "frac_active",
                "dense_latent",
                "share_prefix",
                "share_query",
                "share_inter",
                "p_prefix",
                "p_query",
            ]
        )
        for i in range(f_dim):
            wr.writerow(
                [
                    int(f_out[i]),
                    f"{frac_active[i]:.6f}",
                    int(dense_latent[i]),
                    f"{shares['share_prefix'][i]:.6f}",
                    f"{shares['share_query'][i]:.6f}",
                    f"{shares['share_inter'][i]:.6f}",
                    f"{p_prefix[i]:.6f}",
                    f"{p_query[i]:.6f}",
                ]
            )
    doc = {
        "n_features_restricted": int(f_dim),
        "restriction_floor_rows": int(floor),
        "n_complete_prefixes": int(len(cc)),
        "n_queries": int(n_q),
        "perm_seeds": list(PERM_SEEDS),
        "density_bands": density_bands,
        "n_dense_latent": int(dense_latent.sum()),
        "feature_maps": feature_maps,
        "rb_alignment": rb_block,
        "mask_robustness": mask_rob,
        "judged_label_freeze": (
            "0 judge/API calls this round — evidence artifacts persisted for #1773"
        ),
        "verdict": verdict,
    }
    C.write_json_atomic(out_eval / "sae_perfeature.json", doc)
    logger.info(
        "[R6] done: %d features; induced avg R2=%.4f independent %.4f",
        f_dim,
        feature_maps["induced_averaged"]["holdout_r2"],
        feature_maps["independent_averaged"]["holdout_r2"],
    )


def _with_meta(doc: dict, args, t0: float) -> dict:
    return {
        **doc,
        "meta": {
            "script": "scripts/issue1738_crossed_reads.py",
            "git_commit": MTF._git_head(),
            "hf_prefix": args.hf_prefix,
            "layers": args.layers,
            "n_boot": int(args.n_boot),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "wall_s": round(time.time() - t0, 1),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #1738 crossed-multiturn-averaged S2 reads.")
    # UPLOAD_PREFIX_EXEMPT: the crossed round's OWN self-contained prefix (plan v9 §10)
    ap.add_argument("--hf-prefix", default=GG.CROSSED_HF_PREFIX)
    ap.add_argument("--local-capture-dir", default="", help="smoke: local chunk dir (no Hub)")
    ap.add_argument("--local-manifest-dir", default="", help="smoke: local sampling_manifest dir")
    ap.add_argument("--pilot-meta", default="", help="local crossed_pilot_meta.json (smoke)")
    ap.add_argument("--out-eval", type=Path, default=DEFAULT_OUT_EVAL)
    ap.add_argument("--out-local", type=Path, default=DEFAULT_OUT_LOCAL)
    ap.add_argument("--mm-dir", type=Path, default=None)
    ap.add_argument("--layers", default=",".join(str(x) for x in MTF.LAYERS_DEFAULT))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--ridge-block", type=int, default=8192)
    ap.add_argument("--fence-mult", type=float, default=2.0)
    ap.add_argument("--coverage-floor", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=MTF.N_BOOT)
    ap.add_argument("--pca-dirs", type=int, default=48)
    ap.add_argument("--inner-cv-folds", type=int, default=5)
    ap.add_argument("--n-operator-nulls", type=int, default=200)
    ap.add_argument("--sae-feature-block", type=int, default=1024)
    ap.add_argument("--rb-tail-n", type=int, default=200)
    ap.add_argument("--skip-rb-align", action="store_true")
    ap.add_argument("--queries-per-prefix", type=int, default=0)
    ap.add_argument("--kresample-summary", default=str(KRESAMPLE_SUMMARY_DEFAULT))
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    if args.mm_dir is None:
        args.mm_dir = Path(args.out_local) / "mm"
    rc = run_reads(args)
    # heavy C-extension entrypoint: explicit exit dodges the finalize-time
    # PyGILState_Release atexit race (#1689 gotcha).
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)


if __name__ == "__main__":
    main()
