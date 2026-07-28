#!/usr/bin/env python3
"""Issue #1738 — multi-turn prefix/context-arm fits at ~100k (extends the #779
n1m fits driver by IMPORTING its fitters verbatim; plan §4 deltas only).

Deltas vs ``issue779_ffc_n1m_fits`` (whose five fitters + numerics are reused
UNCHANGED via import):

(a) ``--input-arm {prefix,context,both}`` selects X ∈ {``px_last``, ``cx_last``}
    (Y = ``v_x`` always) — the paired mapping arms (standing prefix+context rule).
(b) The split loader reads the Phase-0 pinned ``split_1738.json`` (sha-asserted)
    instead of the parent's ``fixed_split`` + pass_b recovery.
(c) Fit points reduce to ONE (``multiturn_100k``).
(d) After fitting, per-context HOLDOUT predictions are RETAINED per fitter
    (fp16, the #1482 pdshrink convention) and the standing mapping-baselines
    pair runs per arm × layer: identity+learned-bias
    (``analysis/mapping_baselines.identity_bias_predict``; dims match,
    3584→3584) and kNN retrieval (``knn_retrieval``; ks {1,5,10} over the
    holdout pool, euclidean + cosine, chance = k/n_pool stated).

Also: the ``lmsys_transfer`` group-level OOD control (fit ridge on LMSYS-only
train rows, score WildChat holdout rows; layer 19, both arms), and the G2
in-run first-cell timing fence (designed halt: fence_report.json + rc 24 —
never a bare rc=1, per the pilot-gate routing rule).

KRR Nystrom m is passed EXPLICITLY (default 16384 here — the parent script
default is 8192; plan §11 NOTE) with ``--krr-solver cholesky``.

Compute: 1× A100-80 (`capture-7b`). Streams capture chunks from HF into
per-(array, layer) append-only fp32 binaries (+ cursor checkpoint; resume-safe),
then memmaps them. Refusal-safety: chunk text fields are never printed/logged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
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
import issue779_fitter_fair_comparison as F  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
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
logger = logging.getLogger("issue1738_mt_fits")

FIT_POINT = "multiturn_100k"
ARMS = {"prefix": "px_last", "context": "cx_last"}
LAYERS_DEFAULT = (14, 19, 26)
LAMBDAS = PF.LAMBDAS_N1M  # logspace(-3, 8, 23), parent grid verbatim
PREDICTORS = PF.PREDICTORS
MLP_LRS = tuple(F.MLP_LRS)  # (1e-3, 3e-4) — round-1 grid (plan §11)
KRR_CENTERS_DEFAULT = 16_384  # plan §11 NOTE: NOT the parent script default (8192)
N_BOOT = 10_000
BOOT_SEED = 1738
RC_FENCE = 24  # G2 designed-halt rc (report written first; never a bare rc=1)
H_DIM = C.EXPECTED_HIDDEN  # 3584
DEFAULT_OUT_EVAL = PROJECT_ROOT / "eval_results" / "issue_1738"
DEFAULT_OUT_LOCAL = PROJECT_ROOT / "data" / "issue_1738" / "mt100k" / "fits"
ANALYSIS_TENSORS_SUBDIR = "analysis_tensors"


# ── split loader (delta b: pinned split_1738.json, sha-asserted) ──────────────────


def load_split(split_path: Path) -> dict:
    """Load + sha-assert the pinned split doc: every set's recorded sha256 must
    reproduce from its own ci list (a corrupted/hand-edited doc fails loud)."""
    doc = json.loads(split_path.read_text())
    for name, s in doc["sets"].items():
        got = GG._sha_int_list([int(c) for c in s["ci"]])
        assert got == s["sha256"], f"split set {name!r} sha mismatch: {got} != {s['sha256']}"
    return doc


def split_positions(doc: dict, ci: np.ndarray) -> dict[str, np.ndarray]:
    """Map the split doc's GLOBAL ci sets to CAPTURED row positions. Reports
    realized coverage (over-length/violation skips make captured ⊂ manifest)."""
    pos_of = {int(c): p for p, c in enumerate(ci.tolist())}
    out = {}
    for name, s in doc["sets"].items():
        rows = np.asarray(sorted(pos_of[c] for c in s["ci"] if int(c) in pos_of), dtype=np.int64)
        out[name] = rows
        logger.info("[split] %s: %d/%d intended rows captured", name, len(rows), s["n"])
    inter = set(out["train"].tolist()) & (
        set(out["val"].tolist()) | set(out["test"].tolist()) | set(out["holdout"].tolist())
    )
    assert not inter, f"train overlaps eval sets at {len(inter)} rows"
    return out


# ── streaming chunk assembly → per-(array, layer) fp32 binaries + memmaps ─────────


def _chunk_names(args) -> list[str]:
    if args.local_capture_dir:
        names = sorted(p.name for p in Path(args.local_capture_dir).glob("*.pt"))
    else:
        names = sorted(
            n
            for n in GG.N50._remote_index(f"{args.hf_prefix}/{GG.CAPTURE_SUBDIR}")
            if n.endswith(".pt")
        )
    if not names:
        raise SystemExit("no capture chunks found — run the capture phase first")
    return names


def _mm_paths(mm_dir: Path, layers: list[int]) -> dict:
    out = {"cursor": mm_dir / "cursor.json", "ci": mm_dir / "ci.bin", "meta": {}}
    for arr in ("px", "cx", "vx"):
        for li in layers:
            out[(arr, li)] = mm_dir / f"{arr}_L{li}.bin"
    return out


def assemble_streams(args, layers: list[int]):
    """Stream capture chunks (HF or local) into append-only fp32 binaries per
    (array, layer) + an int64 ci binary, with a cursor checkpoint every chunk
    batch (resume truncates to the cursor row count — the external-stream
    checkpoint law). Returns (mm dict of np.memmap, ci int64 array, meta)."""
    mm_dir = Path(args.mm_dir)
    mm_dir.mkdir(parents=True, exist_ok=True)
    names = _chunk_names(args)
    fp = hashlib.sha256(
        ("\n".join(names) + f"|{args.hf_prefix}|{sorted(layers)}").encode()
    ).hexdigest()
    paths = _mm_paths(mm_dir, layers)
    cursor = {"fingerprint": fp, "n_chunks_done": 0, "n_rows": 0}
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
    # truncate binaries to the cursor row count (crash mid-append leaves a tail)
    n_rows = int(cursor["n_rows"])
    for arr in ("px", "cx", "vx"):
        for li in layers:
            p = paths[(arr, li)]
            want = n_rows * H_DIM * 4
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
            for arr in ("px", "cx", "vx"):
                t = bundle[key_of[arr]]
                assert t.shape == (n, len(blayers), H_DIM), (name, arr, t.shape)
                for li in layers:
                    handles[(arr, li)].write(
                        np.ascontiguousarray(
                            t[:, li_pos[li], :].numpy().astype(np.float32)
                        ).tobytes()
                    )
            ci_f.write(np.asarray(bundle["ci"], dtype=np.int64).tobytes())
            n_rows += n
            cursor.update({"n_chunks_done": k + 1, "n_rows": n_rows})
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
        (arr, li): np.memmap(paths[(arr, li)], dtype=np.float32, mode="r", shape=(n_rows, H_DIM))
        for arr in ("px", "cx", "vx")
        for li in layers
    }
    return mm, ci, {"n_rows": n_rows, "n_chunks": len(names), "fingerprint": fp}


# ── fit dispatch (parent fitters verbatim; MLP lr grid val-selected) ──────────────


def _fitargs(args, mlp_lr: float) -> SimpleNamespace:
    return SimpleNamespace(
        ridge_block=args.ridge_block,
        mlp_lr=mlp_lr,
        mlp_max_epochs=args.mlp_max_epochs,
        mlp_batch=args.mlp_batch,
        seed=args.seed,
        krr_nystrom_centers=args.krr_nystrom_centers,
        krr_solver=args.krr_solver,
    )


def fit_predictor(name, X, Y, tr, val, te_all, args, dev, *, resid_lr: float):
    """One predictor on eval rows ``te_all`` (= test ∥ holdout). MLP fitters run
    the round-1 lr grid {1e-3, 3e-4} with val-R² selection (val rides in the
    eval concat); ridge/KRR select on val inside the parent fitters."""
    if name in ("mlp_w8192", "mlp_w32768"):
        width = PF.MLP_W_PROTOCOL if name == "mlp_w8192" else PF.MLP_W_CAPACITY
        eval_idx = np.concatenate([val, te_all])
        best = None
        for lr in args.mlp_lrs_list:
            pred, meta = PF.fit_mlp(
                X,
                Y,
                tr,
                eval_idx,
                width,
                lr,
                args.mlp_max_epochs,
                args.mlp_batch,
                args.seed,
                dev,
                capacity_arm=(name == "mlp_w32768"),
            )
            vr2 = PR._pooled_r2(pred[: len(val)], np.asarray(Y[val]))
            meta = {**meta, "lr": float(lr), "val_r2": float(vr2)}
            if best is None or (np.isfinite(vr2) and vr2 > best[0]):
                best = (vr2, pred[len(val) :], meta)
        pred_te, meta = best[1], best[2]
        meta["selected_lr"] = meta["lr"]
        meta["lr_grid"] = [float(x) for x in args.mlp_lrs_list]
        return pred_te, meta
    fa = _fitargs(args, resid_lr)
    return PF._fit_one_predictor(
        name, X, Y, tr, val, te_all, LAMBDAS, args.krr_gamma_mult, args.krr_lambdas_list, fa, dev
    )


def _percontext_nerr(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """nerr(x) = ||v̂−v||² / ||v−μ_eval||² per context (the #1482 convention)."""
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    mu = true.mean(axis=0)
    num = ((true - pred) ** 2).sum(axis=1)
    den = ((true - mu) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def _fence_should_halt(elapsed_s: float, first_wall_s: float, n_cells: int, mult: float) -> bool:
    """G2 pure predicate: halt once elapsed exceeds mult × first-cell wall ×
    n_cells (the ≥2× first-cell extrapolation fence, plan §7)."""
    return elapsed_s > mult * first_wall_s * n_cells


def _boot_recon_ci_batched(pred: np.ndarray, true: np.ndarray, n_boot: int, seed: int, chunk=500):
    """BATCHED context bootstrap of (R², mean cosine) — draw-identical to
    ``F._bootstrap_recon_ci`` (same rng stream, same per-draw math: multiplicity-
    weighted SS_res and SS_tot around the RESAMPLE mean) but expressed as chunked
    counts@pool GEMMs instead of a per-draw Python loop (the vectorize-first law;
    equivalence pinned in the fits ``--smoke``). Returns F's dict shape."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    n = pred.shape[0]
    res_i = ((true - pred) ** 2).sum(axis=1)
    cos_i = PR._per_context_cosine(pred, true)
    r2_point, cos_point = F._recon_point(pred, true)
    y_norm2 = (true**2).sum(axis=1)
    cos_fill = np.nan_to_num(cos_i, nan=0.0, posinf=0.0, neginf=0.0)
    cos_fin = np.isfinite(cos_i).astype(np.float64)
    rng = np.random.default_rng(seed)
    r2s = np.full(n_boot, np.nan)
    coss = np.full(n_boot, np.nan)
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        idx = rng.integers(0, n, size=(b, n))
        counts = np.zeros((b, n), dtype=np.float64)
        np.add.at(counts, (np.repeat(np.arange(b), n), idx.ravel()), 1.0)
        ss_res = counts @ res_i
        sum_y = counts @ true  # (b, H)
        ss_tot = counts @ y_norm2 - ((sum_y / n) ** 2).sum(axis=1) * n
        ok = ss_tot > 1e-12
        r2s[s : s + b][ok] = 1.0 - ss_res[ok] / ss_tot[ok]
        c_cnt = counts @ cos_fin
        with np.errstate(invalid="ignore", divide="ignore"):
            coss[s : s + b] = (counts @ cos_fill) / c_cnt

    def _ci(pt, boots):
        boots = boots[np.isfinite(boots)]
        if boots.size == 0:
            return {"point": pt, "lo": float("nan"), "hi": float("nan")}
        return {
            "point": pt,
            "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)),
        }

    return {"r2": _ci(r2_point, r2s), "mean_cosine": _ci(cos_point, coss), "n_test": int(n)}


def run_fits(args) -> int:
    layers = [int(x) for x in args.layers.split(",")]
    arms = list(ARMS) if args.input_arm == "both" else [args.input_arm]
    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    args.mlp_lrs_list = [float(x) for x in args.mlp_lrs.split(",")]
    args.krr_gamma_mult = tuple(float(x) for x in args.krr_gamma_mult_s.split(","))
    args.krr_lambdas_list = tuple(float(x) for x in args.krr_lambdas_s.split(","))

    C.phase("fits-assemble")
    mm, ci, ameta = assemble_streams(args, layers)
    split = load_split(Path(args.split_file))
    sets = split_positions(split, ci)
    tr, val, te, ho = sets["train"], sets["val"], sets["test"], sets["holdout"]
    n_tr, d = len(tr), H_DIM
    if n_tr < d and not args.allow_underdetermined:
        raise SystemExit(
            f"n_train={n_tr} < d={d}: estimator-degenerate regime — pass "
            "--allow-underdetermined only for a deliberate smoke shape"
        )
    te_all = np.concatenate([te, ho])
    corpus_by_ci = {}
    if args.manifest_dir:
        pool, _m = GG.N1M.read_manifest_pool(Path(args.manifest_dir))
        corpus_by_ci = {int(r["i"]): r["corpus"] for r in pool}

    # cell order: (context, L19) FIRST — the G2 in-run timing pilot cell.
    layer_order = ([19] if 19 in layers else []) + [x for x in layers if x != 19]
    arm_order = [a for a in ("context", "prefix") if a in arms]
    cells = [(a, li) for a in arm_order for li in layer_order]
    cells_dir = args.out_eval / "fits" / "cells"
    pc_dir = args.out_eval / "percontext"
    pred_dir = args.out_local / "pred16"
    yh_dir = args.out_local / "y_holdout"
    for p in (cells_dir, pc_dir, pred_dir, yh_dir, args.out_local / "weights"):
        p.mkdir(parents=True, exist_ok=True)

    # persist Y holdout (fp16) per layer — the off-pod H1 bootstrap input
    for li in layers:
        yhp = yh_dir / f"L{li}.npz"
        if not yhp.exists():
            np.savez(yhp, y16=np.asarray(mm[("vx", li)][ho], dtype=np.float16), ci=ci[ho])

    C.phase("fits")
    first_wall: float | None = None
    t_cells0 = time.time()
    n_cells_total = len(cells)
    summary: dict = {
        "fit_point": FIT_POINT,
        "layers": layers,
        "arms": arms,
        "n_rows_captured": ameta["n_rows"],
        "split_counts": {k: int(len(v)) for k, v in sets.items()},
        "split_shas": {k: split["sets"][k]["sha256"] for k in split["sets"]},
        "lambdas": [float(x) for x in LAMBDAS],
        "mlp_lr_grid": args.mlp_lrs_list,
        "krr": {"m_centers": int(args.krr_nystrom_centers), "solver": args.krr_solver},
        "n_boot": int(args.n_boot),
        "boot_seed": BOOT_SEED,
        "cells": {},
        "git_commit": os.environ.get("EPM_GIT_COMMIT", ""),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for cell_i, (arm, li) in enumerate(cells):
        # G2 fence: designed halt once elapsed exceeds fence_mult × first-cell
        # extrapolation (report JSON + distinct rc — never an anonymous crash).
        if first_wall is not None:
            budget = args.fence_mult * first_wall * n_cells_total
            elapsed = time.time() - t_cells0
            if _fence_should_halt(elapsed, first_wall, n_cells_total, args.fence_mult):
                rep = {
                    "gate": "G2",
                    "first_cell_wall_s": first_wall,
                    "elapsed_s": elapsed,
                    "budget_s": budget,
                    "fence_mult": args.fence_mult,
                    "cells_done": cell_i,
                    "cells_total": n_cells_total,
                }
                GG.N1M._atomic_write_json(args.out_eval / "fits" / "fence_report.json", rep)
                logger.error("[G2] fence tripped: %s", rep)
                sys.exit(RC_FENCE)
        X = mm[(("px" if arm == "prefix" else "cx"), li)]
        Y = mm[("vx", li)]
        t_cell = time.time()
        resid_lr = float(args.mlp_lrs_list[-1])
        for name in [p for p in PREDICTORS if p in args.predictors.split(",")]:
            cj = cells_dir / f"{arm}_L{li}_{name}.json"
            if cj.exists() and not args.no_resume:
                doc = json.loads(cj.read_text())
                summary["cells"][f"{arm}_L{li}_{name}"] = doc["metrics"]
                if name == "mlp_w8192" and "selected_lr" in doc.get("fit_meta", {}):
                    resid_lr = float(doc["fit_meta"]["selected_lr"])
                logger.info("[cell] %s_L%d_%s: resume-skip", arm, li, name)
                continue
            t0 = time.time()
            pred_all, meta = fit_predictor(
                name, X, Y, tr, val, te_all, args, dev, resid_lr=resid_lr
            )
            if name == "mlp_w8192" and "selected_lr" in meta:
                resid_lr = float(meta["selected_lr"])  # residual_skip inherits the selected lr
            pred_te, pred_ho = pred_all[: len(te)], pred_all[len(te) :]
            y_te, y_ho = np.asarray(Y[te], dtype=np.float64), np.asarray(Y[ho], dtype=np.float64)
            r2_te, cos_te = F._recon_point(pred_te, y_te)
            r2_ho, cos_ho = F._recon_point(pred_ho, y_ho)
            ci_boot = _boot_recon_ci_batched(pred_ho, y_ho, args.n_boot, BOOT_SEED)
            nerr = _percontext_nerr(pred_ho, y_ho).astype(np.float32)
            np.savez(pc_dir / f"{arm}_L{li}_{name}.npz", nerr=nerr, ci=ci[ho])
            np.savez(
                pred_dir / f"{arm}_L{li}_{name}.npz",
                pred16=pred_ho.astype(np.float16),
                ci=ci[ho],
            )
            metrics = {
                "test_r2": float(r2_te),
                "test_mean_cosine": float(cos_te),
                "holdout_r2": float(r2_ho),
                "holdout_mean_cosine": float(cos_ho),
                "holdout_bootstrap_ci": ci_boot,
                "n_test": int(len(te)),
                "n_holdout": int(len(ho)),
                "wall_s": time.time() - t0,
            }
            GG.N1M._atomic_write_json(
                cj, {"arm": arm, "layer": li, "fitter": name, "metrics": metrics, "fit_meta": meta}
            )
            summary["cells"][f"{arm}_L{li}_{name}"] = metrics
            logger.info(
                "[cell] %s_L%d_%s: test R2=%.4f holdout R2=%.4f (%.0fs)",
                arm,
                li,
                name,
                r2_te,
                r2_ho,
                metrics["wall_s"],
            )
        # ridge weights persisted per cell (small; apply_map-compatible)
        wj = args.out_local / "weights" / f"L{li}" / f"{arm}_ridge.pt"
        if not wj.exists() and "ridge" in args.predictors.split(","):
            _, _, payload = PF.fit_ridge_with_weights(
                X, Y, tr, val, te, LAMBDAS, dev, args.ridge_block
            )
            wj.parent.mkdir(parents=True, exist_ok=True)
            tmp = wj.parent / (wj.name + ".tmp")
            torch.save({**payload, "arm": arm, "layer": li}, tmp)
            os.replace(tmp, wj)
        if first_wall is None:
            first_wall = time.time() - t_cell
            logger.info(
                "[G2] first cell (%s, L%d) wall=%.0fs → projected total ≈ %.1f h (fence %.1f h)",
                arm,
                li,
                first_wall,
                first_wall * n_cells_total / 3600,
                args.fence_mult * first_wall * n_cells_total / 3600,
            )

    C.phase("fits-baselines")
    baselines: dict = {"ks": [1, 5, 10], "metrics": ["euclidean", "cosine"], "cells": {}}
    for arm, li in cells:
        X = mm[(("px" if arm == "prefix" else "cx"), li)]
        Y = mm[("vx", li)]
        y_ho = np.asarray(Y[ho], dtype=np.float64)
        pred_ib = identity_bias_predict(np.asarray(X[tr]), np.asarray(Y[tr]), np.asarray(X[ho]))
        r2_ib, cos_ib = F._recon_point(pred_ib, y_ho)
        cellb: dict = {
            "identity_bias": {"holdout_r2": float(r2_ib), "holdout_mean_cosine": float(cos_ib)},
            "knn": {},
        }
        ridge_npz = pred_dir / f"{arm}_L{li}_ridge.npz"
        preds = {"identity_bias": pred_ib}
        if ridge_npz.exists():
            preds["ridge"] = np.load(ridge_npz)["pred16"].astype(np.float64)
        for pname, pv in preds.items():
            cellb["knn"][pname] = {
                m: knn_retrieval(pv, y_ho, ks=(1, 5, 10), metric=m) for m in ("euclidean", "cosine")
            }
        baselines["cells"][f"{arm}_L{li}"] = cellb
        logger.info("[baseline] %s_L%d: identity+bias holdout R2=%.4f", arm, li, r2_ib)
    GG.N1M._atomic_write_json(args.out_eval / "mapping_baselines.json", baselines)

    # lmsys_transfer (group-level OOD): ridge, layer 19, both arms — fit on
    # LMSYS-only train rows, score WildChat holdout rows.
    transfer: dict = {"control": "lmsys_transfer", "layer": 19, "cells": {}}
    if split.get("transfer_descoped"):
        transfer["descoped"] = True
        logger.warning("[transfer] DESCOPED at manifest time (E_W < 5000)")
    elif corpus_by_ci:
        corp = np.asarray([corpus_by_ci.get(int(c), "?") for c in ci])
        tr_lm = tr[corp[tr] == "lmsys"]
        val_lm = val[corp[val] == "lmsys"]
        ho_wc = ho[corp[ho] == "wildchat"]
        ho_lm = ho[corp[ho] == "lmsys"]
        if len(ho_wc) == 0 or len(tr_lm) == 0:
            transfer["skipped"] = f"empty cells (tr_lm={len(tr_lm)}, ho_wc={len(ho_wc)})"
        else:
            for arm in arms:
                X = mm[(("px" if arm == "prefix" else "cx"), 19)]
                Y = mm[("vx", 19)]
                eval_idx = np.concatenate([ho_wc, ho_lm])
                pred, meta = PF.fit_ridge(
                    X, Y, tr_lm, val_lm, eval_idx, LAMBDAS, dev, args.ridge_block
                )
                r2_wc, _ = F._recon_point(
                    pred[: len(ho_wc)], np.asarray(Y[ho_wc], dtype=np.float64)
                )
                r2_lm, _ = F._recon_point(
                    pred[len(ho_wc) :], np.asarray(Y[ho_lm], dtype=np.float64)
                )
                transfer["cells"][arm] = {
                    "n_train_lmsys": int(len(tr_lm)),
                    "n_holdout_wildchat": int(len(ho_wc)),
                    "n_holdout_lmsys": int(len(ho_lm)),
                    "transfer_r2_wildchat_holdout": float(r2_wc),
                    "within_r2_lmsys_holdout": float(r2_lm),
                    "selected_lambda": meta["selected_lambda"],
                }
                logger.info("[transfer] %s: wc R2=%.4f lmsys R2=%.4f", arm, r2_wc, r2_lm)
    else:
        transfer["skipped"] = "no --manifest-dir (corpus provenance unavailable)"
    summary["lmsys_transfer"] = transfer
    GG.N1M._atomic_write_json(args.out_eval / "fits" / f"{FIT_POINT}_fits.json", summary)

    if not args.no_upload:
        C.phase("fits-upload")
        for sub in ("pred16", "y_holdout", "weights"):
            local = args.out_local / sub
            files = sorted(str(p.relative_to(local)) for p in local.rglob("*") if p.is_file())
            if not files:
                continue
            url = hub._upload_folder_filtered(
                local,
                repo_id=C.HF_DATA_REPO,
                repo_type="dataset",
                path_in_repo=f"{args.hf_prefix}/{ANALYSIS_TENSORS_SUBDIR}/{sub}",
                allow_patterns=files,
                expected_repo_paths=[
                    f"{args.hf_prefix}/{ANALYSIS_TENSORS_SUBDIR}/{sub}/{f}" for f in files
                ],
            )
            if not url:
                raise RuntimeError(f"analysis-tensors upload ({sub}) returned no URL")
    C.phase("done")
    return 0


# ── smoke: tiny synthetic capture store through the PRODUCTION entrypoint ─────────


def _write_smoke_store(root: Path, *, n_rows=140, layers=(14, 19, 26), seed=0) -> tuple[Path, Path]:
    """Synthetic capture chunks in the PRODUCTION chunk schema + a matching
    manifest/split doc. Y = linear(X_cx) + noise so ridge finds real signal."""
    rng = np.random.default_rng(seed)
    cap = root / "capture"
    man = root / "manifest"
    cap.mkdir(parents=True, exist_ok=True)
    man.mkdir(parents=True, exist_ok=True)
    W = rng.standard_normal((H_DIM, H_DIM)).astype(np.float32) * 0.01
    rows_per_chunk = (n_rows + 2) // 3
    pool_rows = []
    ci0 = 0
    for k in range(3):
        n = min(rows_per_chunk, n_rows - ci0)
        cx = rng.standard_normal((n, len(layers), H_DIM)).astype(np.float32)
        px = cx + 0.5 * rng.standard_normal((n, len(layers), H_DIM)).astype(np.float32)
        vx = np.einsum("nlh,hd->nld", cx, W).astype(np.float32)
        vx += 0.05 * rng.standard_normal(vx.shape).astype(np.float32)
        cis = list(range(ci0, ci0 + n))
        torch.save(
            {
                "px_last": torch.from_numpy(px),
                "cx_last": torch.from_numpy(cx),
                "v_x": torch.from_numpy(vx),
                "ci": cis,
                "prompts": ["[]"] * n,
                "response": ["stub"] * n,
                "depth": [2 + (c % 4) for c in cis],
                "corpus": ["lmsys" if c % 3 else "wildchat" for c in cis],
                "layers": list(layers),
                "shard_index": 0,
                "chunk": k,
            },
            cap / f"shard00_chunk{k:04d}.pt",
        )
        for c in cis:
            pool_rows.append(
                {
                    "i": c,
                    "messages": [{"role": "user", "content": f"q{c}"}],
                    "depth": 2 + (c % 4),
                    "corpus": "lmsys" if c % 3 else "wildchat",
                    "source_hash": f"s{c}",
                    "stream_pos": c,
                    "n_chars": 10,
                    "split": "train",
                }
            )
        ci0 += n
    order = rng.permutation(n_rows).tolist()
    sets = {
        "val": sorted(order[:20]),
        "test": sorted(order[20:40]),
        "holdout": sorted(order[40:80]),
        "train": sorted(order[80:]),
    }
    doc = {"seed": 1738, "n_manifest": n_rows, "sets": {}, "transfer_descoped": False}
    for name, ids in sets.items():
        doc["sets"][name] = {"ci": ids, "n": len(ids), "sha256": GG._sha_int_list(ids)}
        for c in ids:
            pool_rows[c]["split"] = name
    meta = {"n_new": n_rows, "capture_layers": list(layers)}
    GG.N1M._write_manifest_parts(man, pool_rows, meta)
    GG.N1M._atomic_write_json(man / "split_1738.json", doc)
    return cap, man


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #1738 multi-turn prefix/context fits.")
    ap.add_argument("--input-arm", choices=["prefix", "context", "both"], default="both")
    ap.add_argument("--layers", default=",".join(str(x) for x in LAYERS_DEFAULT))
    ap.add_argument("--hf-prefix", default=GG.HF_PREFIX)
    ap.add_argument("--local-capture-dir", default="", help="read chunks locally (smoke/pod)")
    ap.add_argument("--manifest-dir", default="", help="local manifest dir (corpus provenance)")
    ap.add_argument("--manifest-from-hf", action="store_true")
    ap.add_argument("--split-file", default="", help="split_1738.json (default: manifest dir)")
    ap.add_argument("--mm-dir", default=str(DEFAULT_OUT_LOCAL / "mm"))
    ap.add_argument("--out-eval", type=Path, default=DEFAULT_OUT_EVAL)
    ap.add_argument("--out-local", type=Path, default=DEFAULT_OUT_LOCAL)
    ap.add_argument("--predictors", default=",".join(PREDICTORS))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--mlp-lrs", default=",".join(str(x) for x in MLP_LRS))
    ap.add_argument("--mlp-max-epochs", type=int, default=F.MLP_MAX_EPOCHS)
    ap.add_argument("--mlp-batch", type=int, default=PF.MLP_BATCH)
    ap.add_argument("--ridge-block", type=int, default=PF.RIDGE_BLOCK)
    # plan §11 NOTE: m=16384 EXPLICIT here — the parent script default is 8192.
    ap.add_argument("--krr-nystrom-centers", type=int, default=KRR_CENTERS_DEFAULT)
    ap.add_argument("--krr-solver", choices=["eigh", "cholesky"], default="cholesky")
    ap.add_argument("--krr-gamma-mult", dest="krr_gamma_mult_s", default="1.0")
    ap.add_argument("--krr-lambdas", dest="krr_lambdas_s", default="0.1,10")
    ap.add_argument("--fence-mult", type=float, default=2.0, help="G2: halt past mult×first-cell")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--allow-underdetermined", action="store_true", help="smoke shape: n_train<d")
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic store, production path")
    args = ap.parse_args()

    if args.smoke:
        root = Path(args.mm_dir).parent / "_smoke_fits"
        if root.exists():
            import shutil

            shutil.rmtree(root)
        cap, man = _write_smoke_store(root)
        args = argparse.Namespace(
            **{
                **vars(args),
                "local_capture_dir": str(cap),
                "manifest_dir": str(man),
                "split_file": str(man / "split_1738.json"),
                "mm_dir": str(root / "mm"),
                "out_eval": root / "eval",
                "out_local": root / "local",
                "device": "cpu",
                "mlp_max_epochs": 2,
                "krr_nystrom_centers": 32,
                "n_boot": 50,
                "mlp_lrs": "0.0003",
                "layers": "19",
                "no_upload": True,
                "allow_underdetermined": True,  # deliberate smoke shape: n_train 60 < d 3584
            }
        )
        rc = run_fits(args)
        summ = json.loads((args.out_eval / "fits" / f"{FIT_POINT}_fits.json").read_text())
        assert len(summ["cells"]) == 2 * 1 * 5, sorted(summ["cells"])
        bl = json.loads((args.out_eval / "mapping_baselines.json").read_text())
        assert "context_L19" in bl["cells"] and "prefix_L19" in bl["cells"]
        for cell in bl["cells"].values():
            assert "identity_bias" in cell and "ridge" in cell["knn"]
        for f in (args.out_eval / "percontext").glob("*.npz"):
            z = np.load(f)
            assert z["nerr"].shape == z["ci"].shape
        assert "cells" in summ["lmsys_transfer"] and summ["lmsys_transfer"]["cells"], summ[
            "lmsys_transfer"
        ]
        # degenerate-input gate probes (data-dependent-gates duty): the G2 fence
        # predicate fires past the budget and stays quiet under it; the
        # n_train<d refusal fires without --allow-underdetermined.
        assert _fence_should_halt(101.0, 10.0, 5, 2.0) and not _fence_should_halt(
            99.0, 10.0, 5, 2.0
        )
        ud = argparse.Namespace(**{**vars(args), "allow_underdetermined": False})
        try:
            run_fits(ud)
            raise AssertionError("n_train<d gate did not refuse")
        except SystemExit as e:
            assert "n_train" in str(e.code) or (isinstance(e.code, str) and "d=" in e.code), e.code
        # batched-rewrite equivalence gate (vectorize rule item 6): the batched
        # bootstrap CI must reproduce F._bootstrap_recon_ci draw-for-draw.
        rng = np.random.default_rng(7)
        tp = rng.standard_normal((30, 8))
        tt = tp + 0.3 * rng.standard_normal((30, 8))
        a = _boot_recon_ci_batched(tp, tt, 200, 11, chunk=64)
        b = F._bootstrap_recon_ci(tp, tt, 200, 11)
        for k in ("r2", "mean_cosine"):
            for f_ in ("point", "lo", "hi"):
                assert abs(a[k][f_] - b[k][f_]) < 1e-9, (k, f_, a[k], b[k])
        logger.info("[smoke] batched bootstrap CI == serial reference (200 draws)")
        logger.info("[smoke] fits OK: %d cells + baselines + transfer", len(summ["cells"]))
    else:
        if args.manifest_from_hf and not args.manifest_dir:
            mdir = GG.N1M._download_manifest(
                args.hf_prefix, Path(args.mm_dir).parent / GG.MANIFEST_SUBDIR
            )
            args.manifest_dir = str(mdir)
        if not args.split_file:
            if not args.manifest_dir:
                raise SystemExit("--split-file or --manifest-dir/--manifest-from-hf required")
            args.split_file = str(Path(args.manifest_dir) / "split_1738.json")
        rc = run_fits(args)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)


if __name__ == "__main__":
    main()
