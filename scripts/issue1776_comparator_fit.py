"""#1776 Phase 0.5: slot-matched fitted comparator M' = ridge cx_last(14) -> v(19).

Reuses the #779 Gram-shared refit machinery VERBATIM (``fit_ridge_with_weights``
in ``issue779_ffc_n1m_fits.py`` — streamed (H,H) X^TX + one eigh, all lambdas
off one factorization) on a NEW layer-14 input Gram:

  - X = cx_last at layer 14 (the J differentiation / DeltaHook edit slot),
    Y = v_x at layer 19 (the readout) — CROSS-layer by design (a same-layer
    ell_in == L' Jacobian is structurally zero; plan §4 slot-pinning block).
  - n_train = 50,000 rows seeded from the n1m TRAIN pool (n >> d = 3584,
    well-posed — plan §10 check (l)); ``--lmsys-only`` restricts the pool to
    LMSYS-provenance rows (the Phase-5a H3 comparator ``m_ridge_lmsys50k``).
  - EXTENDED lambda grid 1e-6..1e8, 28 pts (plan §11: the shipped M's
    lambda=1e-3 was LOW-edge-selected on the 23-pt grid); val-400 selection,
    pinned split shas re-asserted inside ``assemble_multilayer``.
  - Reports the guideline-11 companions on test: identity+learned-bias
    baseline (input/output share H=3584) + kNN retrieval (euclidean + cosine).

Weights payload (fp32 standardizer + W) persists locally and uploads to HF
``issue1776_jacobian/analysis_tensors/`` in the dispatcher's upload phase.

CPU smoke: ``--smoke-synthetic`` runs the FULL fit body (same functions, same
grid) on seeded synthetic data at small H — plus a serial-oracle equivalence
probe (fit_ridge vs fit_ridge_with_weights predictions identical).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import issue1776_common as C76
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # bind shared-VM thread caps BEFORE numpy/torch import (#847 gate)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_fits as N1M  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)


def _pooled_r2(pred: np.ndarray, y: np.ndarray) -> float:
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def select_train_rows(
    prov: np.ndarray,
    val: np.ndarray,
    te: np.ndarray,
    n_train: int,
    seed: int,
    lmsys_only: bool,
    extra_excluded: np.ndarray | None = None,
) -> np.ndarray:
    """Seeded n_train-row draw from the n1m TRAIN pool (all rows minus pinned
    val/test), optionally restricted to LMSYS-provenance rows (H3 refit).
    ``extra_excluded`` (row indices) implements the plan-§3 G-PARITY exclusion:
    rows failing the parity rig are dropped from the pool (review v1 concern
    parity-exclusion-list-unconsumed)."""
    n_rows = prov.shape[0]
    held = np.zeros(n_rows, dtype=bool)
    held[np.asarray(val)] = True
    held[np.asarray(te)] = True
    if extra_excluded is not None and len(extra_excluded):
        held[np.asarray(extra_excluded)] = True
    pool = np.arange(n_rows)[~held]
    if lmsys_only:
        pool = pool[np.asarray([prov[i] == "lmsys" for i in pool])]
    assert len(pool) >= n_train, (
        f"train pool {len(pool)} < n_train {n_train} (lmsys_only={lmsys_only})"
    )
    rng = np.random.default_rng(seed)
    tr = np.sort(rng.choice(pool, size=n_train, replace=False))
    assert not (set(tr.tolist()) & set(np.asarray(val).tolist())), "train/val overlap"
    assert not (set(tr.tolist()) & set(np.asarray(te).tolist())), "train/test overlap"
    return tr


def fit_comparator(X, Y, tr, val, te, lambdas, dev, block, *, tag: str, out_dir: Path) -> dict:
    """One Gram-shared ridge fit + guideline-11 companion reads; persists payload."""
    d = X.shape[1]
    assert len(tr) > d, f"n_train={len(tr)} <= d={d} — estimator-degenerate regime refused"
    t0 = time.time()
    pred_te, meta, payload = N1M.fit_ridge_with_weights(X, Y, tr, val, te, lambdas, dev, block)
    y_te = np.asarray(Y[np.asarray(te)], dtype=np.float64)
    test_r2 = _pooled_r2(np.asarray(pred_te, dtype=np.float64), y_te)

    # Guideline 11 (a): identity + learned-bias baseline (shared dim H -> H).
    ib_pred = identity_bias_predict(
        x_train=np.asarray(X[np.asarray(tr)], dtype=np.float64),
        y_train=np.asarray(Y[np.asarray(tr)], dtype=np.float64),
        x_eval=np.asarray(X[np.asarray(te)], dtype=np.float64),
    )
    ib_r2 = _pooled_r2(np.asarray(ib_pred, dtype=np.float64), y_te)
    # Guideline 11 (b): kNN retrieval among the held-out pool; chance = k/n_pool.
    knn = {
        metric: knn_retrieval(np.asarray(pred_te), y_te, ks=(1, 5, 10), metric=metric)
        for metric in ("euclidean", "cosine")
    }
    knn_ib = {
        metric: knn_retrieval(np.asarray(ib_pred), y_te, ks=(1, 5, 10), metric=metric)
        for metric in ("euclidean", "cosine")
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / f"{tag}.pt"
    payload["tag"] = tag
    payload["input_layer"] = C76.SOURCE_LAYER
    payload["output_layer"] = C76.READOUT_LAYER
    torch.save(payload, weights_path)
    report = {
        "tag": tag,
        "input_slot": f"cx_last(L{C76.SOURCE_LAYER})",
        "output_slot": f"v_x(L{C76.READOUT_LAYER})",
        "n_train": int(len(tr)),
        "d": int(d),
        "well_posed": bool(len(tr) > d),
        "lambda_grid": [float(lambdas[0]), float(lambdas[-1]), len(lambdas)],
        "meta": meta,
        "test_pooled_r2": test_r2,
        "identity_bias_test_pooled_r2": ib_r2,
        "knn_retrieval": knn,
        "knn_retrieval_identity_bias": knn_ib,
        "knn_chance_at_1": 1.0 / len(te),
        "weights_path": str(weights_path),
        "elapsed_s": time.time() - t0,
        "repro": C76.repro_meta(),
    }
    C76.atomic_write_json(out_dir / f"{tag}_report.json", report)
    print(
        f"[comparator] [phase=fit_done tag={tag}] lambda={meta['selected_lambda']:.3g} "
        f"(edge={meta['lambda_grid_edge']}) val_r2={meta['val_r2_at_selected']:.4f} "
        f"test_r2={test_r2:.4f} ib_r2={ib_r2:.4f} elapsed={report['elapsed_s']:.1f}s",
        flush=True,
    )
    return report


def smoke_synthetic(args) -> int:
    """CPU smoke: full fit body on seeded synthetic data (n=2,600, small H) +
    a fit_ridge vs fit_ridge_with_weights prediction-equivalence probe."""
    rng = np.random.default_rng(args.seed)
    n, h = 2600, args.smoke_h
    w_true = rng.standard_normal((h, h)) / np.sqrt(h)
    x = rng.standard_normal((n, h)).astype(np.float32)
    y = (x @ w_true + 0.1 * rng.standard_normal((n, h))).astype(np.float32)
    prov = np.array(["lmsys"] * (n // 2) + ["wildchat"] * (n - n // 2), dtype=object)
    val = np.arange(2000, 2300)
    te = np.arange(2300, 2600)
    tr = select_train_rows(prov, val, te, args.smoke_n_train, args.seed, args.lmsys_only)
    # Parity-exclusion branch probe (review v1): excluded rows never enter tr.
    # (n_train shrunk by the exclusion count — the probe exercises the branch,
    # not the pool-floor assert, which its own guard covers at production n.)
    tr_ex = select_train_rows(
        prov,
        val,
        te,
        args.smoke_n_train - 3,
        args.seed,
        args.lmsys_only,
        extra_excluded=np.asarray([0, 1, 2]),
    )
    assert not ({0, 1, 2} & set(tr_ex.tolist())), "parity-excluded rows leaked into train"
    lambdas = C76.EXTENDED_LAMBDA_GRID
    report = fit_comparator(
        x, y, tr, val, te, lambdas, "cpu", 512, tag="smoke_synth", out_dir=args.out_dir
    )
    assert report["test_pooled_r2"] > 0.5, f"synthetic linear map should fit: {report}"
    assert report["meta"]["lambda_grid_edge"] != "high", report["meta"]
    # Equivalence probe: fit_ridge (no-weights twin) predictions identical.
    pred_a, meta_a = N1M.fit_ridge(x, y, tr, val, te, lambdas, "cpu", 512)
    pred_b, meta_b, _ = N1M.fit_ridge_with_weights(x, y, tr, val, te, lambdas, "cpu", 512)
    assert meta_a["selected_lambda"] == meta_b["selected_lambda"], (meta_a, meta_b)
    np.testing.assert_allclose(pred_a, pred_b, rtol=0, atol=1e-10)
    print("[comparator] [phase=smoke_done] PASS (r2>0.5; twins prediction-identical)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="m_ridge_x50k")
    ap.add_argument("--n-train", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lmsys-only", action="store_true", help="H3 comparator m_ridge_lmsys50k")
    ap.add_argument("--out-dir", type=Path, default=C76.DATA_DIR / "comparator")
    ap.add_argument("--ridge-block", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # assemble_multilayer plumbing (pod path; defaults mirror issue779_ffc_n1m_fits).
    ap.add_argument("--pass-b", type=Path, default=N1M.N1G.PASS_B_LOCAL)
    ap.add_argument("--assemble-out-dir", type=Path, default=C76.DATA_DIR / "ffc_n1m")
    ap.add_argument("--manifest-from-hf", action="store_true", default=True)
    ap.add_argument(
        "--manifest-hf-prefix", default="issue779_monitoring/fitter-fair-comparison-n1m"
    )
    # N1M contract (issue779_ffc_n1m_fits.py L949-953 + its own CLI default):
    # --hf-prefix is the CAPTURE prefix <round-root>/final_token_capture — the chunk
    # stream reads <hf_prefix>/shardNN_chunkNNNN.pt directly — while
    # --manifest-hf-prefix is the ROUND ROOT (N1G._resolve_manifest_dir appends
    # sampling_manifest itself). Crash-fix r6 (att-20260729-082617): a round-root
    # default here 404'd the first chunk on the pod.
    ap.add_argument("--hf-prefix", default=f"{N1M.N1G.HF_PREFIX}/final_token_capture")
    ap.add_argument("--n1m-capture-dir", type=Path, default=None)
    ap.add_argument("--mm-dir", type=Path, default=C76.DATA_DIR / "n1m_mm")
    # N1M contract (issue779_ffc_n1m_fits.py L840 -> N50._pinned_original_shas):
    # orig_dir MUST be a real dir holding the ORIGINAL round's fair_comparison.json
    # (committed at eval_results/issue_779/fitter-fair-comparison/, repo-relative on
    # the pod clone). Crash-fix r7 (att-20260729-060640 attempt 3): default=None
    # crashed `None / "fair_comparison.json"` — reuse the module's own constant.
    ap.add_argument("--orig-dir", type=Path, default=N1M.DEFAULT_ORIG_DIR)
    ap.add_argument("--fresh-stream", action="store_true")
    ap.add_argument("--prefetch", type=int, default=2)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument(
        "--parity-exclusion",
        type=Path,
        default=None,
        help="G-PARITY exclusion_list.json — failed-parity ci rows leave the train pool (§3)",
    )
    ap.add_argument("--smoke-synthetic", action="store_true", help="CPU smoke, no staged data")
    ap.add_argument("--smoke-h", type=int, default=64)
    ap.add_argument("--smoke-n-train", type=int, default=2000)
    args = ap.parse_args(argv)

    if args.smoke_synthetic:
        return smoke_synthetic(args)

    ns = argparse.Namespace(
        pass_b=args.pass_b,
        out_dir=args.assemble_out_dir,
        manifest_from_hf=args.manifest_from_hf,
        manifest_hf_prefix=args.manifest_hf_prefix,
        hf_prefix=args.hf_prefix,
        n1m_capture_dir=args.n1m_capture_dir,
        mm_dir=args.mm_dir,
        orig_dir=args.orig_dir,
        fresh_stream=args.fresh_stream,
        prefetch=args.prefetch,
        max_chunks=args.max_chunks,
    )
    layers = [C76.SOURCE_LAYER, C76.READOUT_LAYER]
    per_layer, prov, _orig_train, val, te, split = N1M.assemble_multilayer(ns, layers)
    x14, _ = per_layer[C76.SOURCE_LAYER]
    _, y19 = per_layer[C76.READOUT_LAYER]
    assert x14.shape[1] == C.EXPECTED_HIDDEN and y19.shape == x14.shape, (x14.shape, y19.shape)
    tag = args.tag if not args.lmsys_only else "m_ridge_lmsys50k"
    # Plan §3 G-PARITY exclusion (review v1): rows whose ci failed the parity
    # rig are dropped from the train pool. The ci memmap is the row identity
    # the assemble pass persists (pass_b head rows carry ci=-1 and can never
    # match a parity ci — parity samples new-capture chunks only).
    excl_rows: np.ndarray | None = None
    n_parity_excluded = 0
    if args.parity_exclusion is not None:
        assert args.parity_exclusion.exists(), (
            f"--parity-exclusion missing: {args.parity_exclusion}"
        )
        excluded = json.loads(args.parity_exclusion.read_text())["excluded"]
        excl_ci = sorted({int(r["ci"]) for r in excluded})
        if excl_ci:
            ci = np.load(Path(args.mm_dir) / "ci.npy", mmap_mode="r")[: x14.shape[0]]
            excl_rows = np.nonzero(np.isin(np.asarray(ci), np.asarray(excl_ci)))[0]
            n_parity_excluded = int(len(excl_rows))
        print(
            f"[comparator] parity exclusion: {len(excl_ci)} ci -> "
            f"{n_parity_excluded} train-pool rows dropped",
            flush=True,
        )
    tr = select_train_rows(
        prov, val, te, args.n_train, args.seed, args.lmsys_only, extra_excluded=excl_rows
    )
    report = fit_comparator(
        x14,
        y19,
        tr,
        val,
        te,
        C76.EXTENDED_LAMBDA_GRID,
        args.device,
        args.ridge_block,
        tag=tag,
        out_dir=args.out_dir,
    )
    report["split"] = split
    report["n_parity_excluded"] = n_parity_excluded
    report["parity_exclusion_file"] = str(args.parity_exclusion) if args.parity_exclusion else None
    C76.atomic_write_json(args.out_dir / f"{tag}_report.json", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
