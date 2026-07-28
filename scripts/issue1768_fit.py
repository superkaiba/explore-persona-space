"""#1768 fit driver — p8 corpus map fits + floors + baselines + verdicts (+ p9 hand-off).

Estimator (plan §4.5): full-dim primal ridge with val-selected λ, reusing the
#779 streaming-primal machinery (`issue779_ffc_n1m_fits.fit_ridge_with_weights`
— streaming X^TX blocks, ONE eigh, multi-λ solve; measured 87.1 s @ n=963k).
Per (arm, layer): M0 (c0→v0), M⁺ (c⁺→v⁺), M⁺_tf (c⁺→v⁺_tf); every fitted map
reports held-out R² + mean cosine + the identity+learned-bias baseline + the
kNN retrieval read (standing rule; `analysis/mapping_baselines`). Map-change
verdicts ride Δ_med vs a B-refit bootstrap noise floor (batched GEMM-Gram +
Cholesky solves — no per-draw factorization loop). The panel `fit_cell` path
(instrument continuity with #722) runs in its home n≈120 regime.

`--smoke` runs the same code paths on the tiny pilot store (PASS_UNIFIED).
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # before torch/numpy: shared-VM thread caps

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.fit")

N_FLOOR_REFITS = 200  # B refits -> B/2 disjoint floor pairs (plan §4.5; min 100)
N_BOOT_TEST = 1000  # bootstrap draws over test rows (seed 1768)
N_CI_DRAWS = 500  # paired row-bootstrap draws for the D CI (chunk-bounded)


def _phase(name: str) -> None:
    print(f"[phase={name}]", flush=True)


def _meta() -> dict:
    import torch

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": commit,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "issue": X.ISSUE,
    }


def _atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def _device():
    import torch

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def lambda_grid(lo_exp: float = -3.0, hi_exp: float = 8.0, n: int = 23) -> list[float]:
    """The #779 23-point log grid 1e-3..1e8 (`n1m_multilayer_fits.json`)."""
    return [float(x) for x in np.logspace(lo_exp, hi_exp, n)]


# ── loader: pooled.pt pair -> fit matrices (the plan-named new code) ─────────


def _load_store(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _rows_from_store(store: dict, span: str, layer: int) -> tuple[np.ndarray, list[str]]:
    t = store["arms"][span][layer]
    return np.asarray(t.float().numpy(), dtype=np.float64), list(store["row_sha"])


def load_corpus_cell(arm_id: str, layer: int, out_root: Path) -> dict:
    """pooled.pt pair -> fit matrices for one (arm, layer) (plan §4.5 loader).

    Row alignment is by shared prompt shas (manifest row shas), never order:
    the kept-row sets can differ per unit (per-arm empty-response drops), so
    the join intersects on sha, in the BASE store's order. Split membership
    comes from the sample's global question index (train | val | test).
    """
    out_root = Path(out_root)
    base_unit = X.base_unit_for(arm_id)
    base = _load_store(out_root / "corpus_capture" / base_unit / "pooled.pt")
    plus = _load_store(out_root / "corpus_capture" / arm_id / "pooled.pt")
    tf = _load_store(out_root / "corpus_capture_tf" / arm_id / "pooled_tf.pt")

    C0, base_sha = _rows_from_store(base, "context", layer)
    V0, _ = _rows_from_store(base, "response", layer)
    Cp, plus_sha = _rows_from_store(plus, "context", layer)
    Vp, _ = _rows_from_store(plus, "response", layer)
    Vtf, tf_sha = _rows_from_store(tf, "response", layer)

    plus_ix = {s: i for i, s in enumerate(plus_sha)}
    tf_ix = {s: i for i, s in enumerate(tf_sha)}
    keep = [i for i, s in enumerate(base_sha) if s in plus_ix and s in tf_ix]
    assert len(keep) >= 0.9 * len(base_sha), (arm_id, layer, len(keep), len(base_sha))
    b = np.asarray(keep)
    p = np.asarray([plus_ix[base_sha[i]] for i in keep])
    t_ = np.asarray([tf_ix[base_sha[i]] for i in keep])

    sample = X.load_corpus_sample(out_root)
    qidx = np.asarray(base["row_question_idx"])[b]
    n_train, n_val = sample["n_train"], sample["n_val"]
    split = np.where(qidx < n_train, "train", np.where(qidx < n_train + n_val, "val", "test"))
    corpus = np.asarray([sample["rows"][q]["corpus"] for q in qidx])
    shas = [base_sha[i] for i in keep]
    for i, s in enumerate(shas):  # alignment is sha-keyed, assert it held
        assert sample["rows"][qidx[i]]["sha"] == s, (arm_id, i, s)
    return {
        "C0": C0[b],
        "V0": V0[b],
        "Cplus": Cp[p],
        "Vplus": Vp[p],
        "Vplus_tf": Vtf[t_],
        "sha": shas,
        "qidx": qidx,
        "split": split,
        "corpus": corpus,
    }


def _split_idx(split: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.where(split == "train")[0],
        np.where(split == "val")[0],
        np.where(split == "test")[0],
    )


# ── ridge fit + reads (reused #779 primal machinery) ─────────────────────────


def _pooled_r2(pred: np.ndarray, y: np.ndarray) -> float:
    resid = float(((pred - y) ** 2).sum())
    tot = float(((y - y.mean(axis=0)) ** 2).sum())
    return 1.0 - resid / tot if tot > 0 else float("nan")


def _mean_cos(pred: np.ndarray, y: np.ndarray) -> float:
    num = (pred * y).sum(axis=1)
    den = np.linalg.norm(pred, axis=1) * np.linalg.norm(y, axis=1) + 1e-12
    return float((num / den).mean())


def _boot_ci(stat_fn, pred: np.ndarray, y: np.ndarray, n_draws: int, seed: int) -> list[float]:
    """Bootstrap CI over test ROWS (vectorized index resampling)."""
    rng = np.random.default_rng(seed)
    n = pred.shape[0]
    vals = np.empty(n_draws)
    for d in range(n_draws):  # stat_fn is O(n*D) — the loop is cheap vs a fit
        idx = rng.integers(0, n, n)
        vals[d] = stat_fn(pred[idx], y[idx])
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def _fit_map(Xd: np.ndarray, Yd: np.ndarray, tr, val, te, dev) -> tuple[np.ndarray, dict, dict]:
    """Val-selected primal ridge with grid-edge extension (plan §4.5)."""
    import issue779_ffc_n1m_fits as n1m

    d = Xd.shape[1]
    assert len(tr) > d, (  # estimator validity: n_train > d (#1701 regime duty)
        f"n_train={len(tr)} <= d={d} — under-determined regime refused (plan §11)"
    )
    lo, hi, n = -3.0, 8.0, 23
    for _ext in range(4):
        grid = lambda_grid(lo, hi, n)
        pred_te, meta, payload = n1m.fit_ridge_with_weights(
            Xd, Yd, tr, val, te, grid, dev, n1m.RIDGE_BLOCK
        )
        edge = meta.get("lambda_grid_edge")
        if edge is None:
            break
        if edge == "low":
            lo -= 1.0
        else:
            hi += 1.0
        n += 2
        logger.info("[fit] lambda grid edge %s — extending to [1e%s, 1e%s]", edge, lo, hi)
    meta["lambda_grid"] = [lo, hi, n]
    return pred_te, meta, payload


def _apply_payload(payload: dict, X_eval: np.ndarray, dev) -> np.ndarray:
    import issue779_ffc_n1m_fits as n1m

    return n1m.apply_map(payload, X_eval, dev)


def _map_reads(pred_te: np.ndarray, V_te: np.ndarray, seed: int = X.FLOOR_SEED) -> dict:
    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    n_pool = V_te.shape[0]
    ks = tuple(k for k in (1, 10) if k <= n_pool) or (1,)
    return {
        "heldout_r2": _pooled_r2(pred_te, V_te),
        "heldout_r2_ci95": _boot_ci(_pooled_r2, pred_te, V_te, N_BOOT_TEST, seed),
        "mean_cos": _mean_cos(pred_te, V_te),
        "mean_cos_ci95": _boot_ci(_mean_cos, pred_te, V_te, N_BOOT_TEST, seed + 1),
        "knn_euclidean": knn_retrieval(pred_te, V_te, ks=ks, metric="euclidean"),
        "knn_cosine": knn_retrieval(pred_te, V_te, ks=ks, metric="cosine"),
    }


def _identity_bias_reads(C_tr, V_tr, C_te, V_te) -> dict:
    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    pred = identity_bias_predict(C_tr, V_tr, C_te)
    return {"applicable": True, **_map_reads(pred, V_te, seed=X.FLOOR_SEED + 7)}


# ── refit-noise floor (batched Gram + Cholesky; vectorize-first law) ─────────


def verdict_from(d_stat: float, ci: list[float]) -> str:
    """The §3 verdict lattice (DISJOINT + exhaustive): Changed ⇔ D>0 AND the
    95% CI excludes 0 on the positive side; Unchanged ⇔ CI wholly below 0."""
    if d_stat > 0 and ci[0] > 0:
        return "Changed"
    if ci[1] < 0:
        return "Unchanged"
    return "Unresolved"


def _floor_refit_preds(
    Xd: np.ndarray,
    Yd: np.ndarray,
    tr: np.ndarray,
    C_grid_te: np.ndarray,
    lam: float,
    n_refits: int,
    seed: int,
    dev,
) -> np.ndarray:
    """(n_refits, n_te, D) predictions of bootstrap-row-resampled refits.

    Fixed full-train standardizer + per-draw multinomial row weights; each
    draw is ONE weighted GEMM-Gram + ONE Cholesky solve on `dev` (no per-draw
    SVD/eigh loop — plan §4.5). fp32 Grams: the floor is a NOISE band, and the
    selected-λ solve is well-conditioned at these shapes.
    """
    import torch

    rng = np.random.default_rng(seed)
    n, d = len(tr), Xd.shape[1]
    xmu = Xd[tr].mean(axis=0)
    xsd = Xd[tr].std(axis=0) + 1e-8
    ymu = Yd[tr].mean(axis=0)
    Xs = torch.as_tensor((Xd[tr] - xmu) / xsd, dtype=torch.float32, device=dev)
    Yc = torch.as_tensor(Yd[tr] - ymu, dtype=torch.float32, device=dev)
    Cte = torch.as_tensor((C_grid_te - xmu) / xsd, dtype=torch.float32, device=dev)
    eye = torch.eye(d, dtype=torch.float32, device=dev)
    preds = np.empty((n_refits, C_grid_te.shape[0], Yd.shape[1]), dtype=np.float32)
    for b in range(n_refits):
        w = torch.as_tensor(rng.multinomial(n, np.full(n, 1.0 / n)).astype(np.float32), device=dev)
        Xw = Xs * w[:, None]
        A = Xw.T @ Xs + float(lam) * eye
        XtY = Xw.T @ Yc
        L = torch.linalg.cholesky(A)
        W = torch.cholesky_solve(XtY, L)
        preds[b] = ((Cte @ W) + torch.as_tensor(ymu, dtype=torch.float32, device=dev)).cpu().numpy()
    return preds


def _map_change_block(
    cell: dict, m0_payload: dict, mp_payload: dict, lam0: float, dev, smoke: bool
) -> dict:
    """Δ_med vs refit floor + verdict + CI (plan §4.5 / §3 lattice)."""
    tr, _val, te = _split_idx(cell["split"])
    C0_te = cell["C0"][te]
    pred0 = _apply_payload(m0_payload, C0_te, dev)
    predp = _apply_payload(mp_payload, C0_te, dev)  # common base-c grid (the #722 read)
    delta_rows = np.linalg.norm(predp - pred0, axis=1)
    delta_med = float(np.median(delta_rows))

    n_refits = 8 if smoke else N_FLOOR_REFITS
    floors = {}
    floor_rows_by_cond = {}
    for cond, (Xd, Yd) in {
        "M0": (cell["C0"], cell["V0"]),
        "Mplus": (cell["Cplus"], cell["Vplus"]),
    }.items():
        preds = _floor_refit_preds(
            Xd, Yd, tr, C0_te, lam0, n_refits, X.FLOOR_SEED + hash(cond) % 1000, dev
        )
        pair_rows = np.linalg.norm(preds[0::2] - preds[1::2], axis=2)  # (B/2, n_te)
        floor_rows_by_cond[cond] = pair_rows
        draws = np.median(pair_rows, axis=1)
        floors[cond] = {
            "floor_draw_medians": [float(x) for x in draws],
            "floor_p95": float(np.quantile(draws, 0.95)),
            "n_refits": n_refits,
        }
    floor_rows = floor_rows_by_cond["M0"]
    floor_p95 = floors["M0"]["floor_p95"]
    d_stat = delta_med - floor_p95

    # paired bootstrap over test rows × refit draws (plan §4.5)
    rng = np.random.default_rng(X.FLOOR_SEED + 99)
    n_te = len(te)
    d_draws = np.empty(N_CI_DRAWS)
    for k in range(N_CI_DRAWS):
        ridx = rng.integers(0, n_te, n_te)
        dm = float(np.median(delta_rows[ridx]))
        fm = np.median(floor_rows[:, ridx], axis=1)
        d_draws[k] = dm - float(np.quantile(fm, 0.95))
    ci = [float(np.quantile(d_draws, 0.025)), float(np.quantile(d_draws, 0.975))]
    verdict = verdict_from(d_stat, ci)
    return {
        "delta_med": delta_med,
        "floor_p95": floor_p95,
        "floors": floors,
        "D": d_stat,
        "D_ci95": ci,
        "verdict": verdict,
        "floor_lambda": lam0,
        "floor_standardizer": "full-train (fixed across draws)",
    }


def _decomposition_block(cell: dict, m0_payload: dict, mp_payload, dev, tf: bool = False) -> dict:
    """v⁺(x) − v⁰(x) = [M⁺(c⁺) − M0(c⁺)] + [M0(c⁺) − M0(c⁰)] + residual."""
    _tr, _val, te = _split_idx(cell["split"])
    Vp = (cell["Vplus_tf"] if tf else cell["Vplus"])[te]
    V0 = cell["V0"][te]
    m0_cplus = _apply_payload(m0_payload, cell["Cplus"][te], dev)
    m0_c0 = _apply_payload(m0_payload, cell["C0"][te], dev)
    mp_cplus = _apply_payload(mp_payload, cell["Cplus"][te], dev)
    total = Vp - V0
    map_change = mp_cplus - m0_cplus
    input_move = m0_cplus - m0_c0
    residual = total - map_change - input_move

    def norms(a):
        return float(np.linalg.norm(a, axis=1).mean())

    tot_sq = float((total**2).sum())
    return {
        "mean_norm_total": norms(total),
        "mean_norm_map_change": norms(map_change),
        "mean_norm_input_movement": norms(input_move),
        "mean_norm_residual": norms(residual),
        "sq_share_map_change": float((map_change**2).sum()) / tot_sq if tot_sq else float("nan"),
        "sq_share_input_movement": float((input_move**2).sum()) / tot_sq
        if tot_sq
        else float("nan"),
        "sq_share_residual": float((residual**2).sum()) / tot_sq if tot_sq else float("nan"),
        "matched_text": tf,
    }


def _transfer_fold(cell: dict, dev) -> dict:
    """Corpus-provenance transfer (group = source corpus; ood-folds rule)."""
    tr, val, _te = _split_idx(cell["split"])
    out = {}
    for src, dst in (("lmsys", "wildchat"), ("wildchat", "lmsys")):
        tr_src = tr[cell["corpus"][tr] == src]
        ev_dst = tr[cell["corpus"][tr] == dst]
        d = cell["C0"].shape[1]
        if len(tr_src) <= d or len(ev_dst) < 4:  # per-corpus split halves n_train
            out[f"{src}->{dst}"] = {
                "skipped": f"n_tr={len(tr_src)} <= d={d} or n_ev={len(ev_dst)} < 4"
            }
            continue
        pred_te, meta, payload = _fit_map(cell["C0"], cell["V0"], tr_src, val, ev_dst, dev)
        out[f"{src}->{dst}"] = {
            "heldout_r2": _pooled_r2(pred_te, cell["V0"][ev_dst]),
            "n_train": int(len(tr_src)),
            "n_eval": int(len(ev_dst)),
            "selected_lambda": meta["selected_lambda"],
        }
    return out


def fit_arm_layer(out_root: Path, results_dir: Path, arm_id: str, layer: int, smoke: bool) -> dict:
    """All p8 reads for one (arm, layer); persisted the moment it completes."""
    dest = results_dir / "fits" / f"{arm_id}_L{layer}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    dev = _device()
    cell = load_corpus_cell(arm_id, layer, out_root)
    tr, val, te = _split_idx(cell["split"])

    fits = {}
    payloads = {}
    for name, (Xd, Yd) in {
        "M0": (cell["C0"], cell["V0"]),
        "Mplus": (cell["Cplus"], cell["Vplus"]),
        "Mplus_tf": (cell["Cplus"], cell["Vplus_tf"]),
    }.items():
        pred_te, meta, payload = _fit_map(Xd, Yd, tr, val, te, dev)
        payloads[name] = payload
        fits[name] = {
            **meta,
            **_map_reads(pred_te, Yd[te]),
            "identity_bias": _identity_bias_reads(Xd[tr], Yd[tr], Xd[te], Yd[te]),
        }
    block = _map_change_block(
        cell, payloads["M0"], payloads["Mplus"], payloads["M0"]["selected_lambda"], dev, smoke
    )
    result = {
        "arm_id": arm_id,
        "layer": layer,
        "n_rows": int(len(cell["sha"])),
        "n_train": int(len(tr)),
        "n_val": int(len(val)),
        "n_test": int(len(te)),
        "fits": fits,
        "map_change": block,
        "decomposition": _decomposition_block(cell, payloads["M0"], payloads["Mplus"], dev),
        "decomposition_tf": _decomposition_block(
            cell, payloads["M0"], payloads["Mplus_tf"], dev, tf=True
        ),
        "transfer_fold": _transfer_fold(cell, dev),
        "smoke": smoke,
        **_meta(),
    }
    _atomic_json(dest, result)
    return result


def pilot_m0_fit(out_root: Path, base_unit: str, layer: int, smoke: bool) -> dict:
    """Gate-2 re-anchor: M0 fit on the pilot base store (capture p1 consumer)."""
    dev = _device()
    store = _load_store(Path(out_root) / "corpus_capture" / base_unit / "pooled.pt")
    C0, _ = _rows_from_store(store, "context", layer)
    V0, _ = _rows_from_store(store, "response", layer)
    sample = X.load_corpus_sample(Path(out_root))
    qidx = np.asarray(store["row_question_idx"])
    n_train, n_val = sample["n_train"], sample["n_val"]
    tr = np.where(qidx < n_train)[0]
    val = np.where((qidx >= n_train) & (qidx < n_train + n_val))[0]
    te = np.where(qidx >= n_train + n_val)[0]
    pred_te, meta, _payload = _fit_map(C0, V0, tr, val, te, dev)
    return {
        "layer": layer,
        "heldout_r2": _pooled_r2(pred_te, V0[te]),
        "mean_cos": _mean_cos(pred_te, V0[te]),
        "selected_lambda": meta["selected_lambda"],
        "n_train": int(len(tr)),
    }


# ── panel fit_cell continuity (home regime n≈120; plan §4.5) ─────────────────

FIT_CELL_BEHAVIOR_COL = "sycophancy"  # RB_COLUMN_KEY slot the rb stack rides in


def panel_cell_records(out_root: Path, arm_id: str, layer: int, span: str = "context") -> list:
    """CellRecord list from the panel pooled trees (family = read context)."""
    from issue722_load_activations import CellRecord

    base_store = _load_store(
        Path(out_root) / "panel_capture" / f"base_{arm_id.split('-')[0]}" / "pooled.pt"
    )
    arm_store = _load_store(Path(out_root) / "panel_capture" / arm_id / "pooled.pt")

    def rowmap(store):
        return {(m["context_id"], m["question_idx"]): i for i, m in enumerate(store["row_meta"])}

    b_ix, a_ix = rowmap(base_store), rowmap(arm_store)
    keys = sorted(set(b_ix) & set(a_ix))
    assert len(keys) >= 4, (arm_id, layer, len(keys), "fit_cell needs n>=4")
    recs = []
    for cid, q in keys:
        recs.append(
            CellRecord(
                behavior=arm_id,
                source_cid=cid,
                target_cid=cid,
                layer=layer,
                c0=np.asarray(
                    base_store["arms"][span][layer][b_ix[(cid, q)]].float().numpy(),
                    dtype=np.float64,
                ),
                cplus=np.asarray(
                    arm_store["arms"][span][layer][a_ix[(cid, q)]].float().numpy(),
                    dtype=np.float64,
                ),
                v0=np.asarray(
                    base_store["arms"]["response"][layer][b_ix[(cid, q)]].float().numpy(),
                    dtype=np.float64,
                ),
                vplus=np.asarray(
                    arm_store["arms"]["response"][layer][a_ix[(cid, q)]].float().numpy(),
                    dtype=np.float64,
                ),
                family=cid,
            )
        )
    return recs


def panel_fit_for_arm(
    out_root: Path, results_dir: Path, arm_id: str, layer: int, rb_stack: np.ndarray, span: str
) -> dict:
    """`fit_cell` verbatim in its home regime (ridge-only; plan call shape)."""
    import issue722_fit_M as fit_m

    dest = results_dir / "panel_fits" / f"{arm_id}_L{layer}_{span}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    cells = panel_cell_records(out_root, arm_id, layer, span=span)
    rb_main = {"r_b": {FIT_CELL_BEHAVIOR_COL: {"diffmeans": rb_stack}}}
    cell_json = fit_m.fit_cell(
        FIT_CELL_BEHAVIOR_COL,
        layer,
        cells,
        rb_main,
        None,
        include_mlp=False,
        floors="batched",
        loco="batched",
    )
    out = {
        "arm_id": arm_id,
        "layer": layer,
        "input_span": span,
        "prefix_arm_rank_limited": span == "prefix",  # §4.8 caveat carrier
        "fit_cell": cell_json,
        **_meta(),
    }
    _atomic_json(dest, out)
    return out


# ── phase drivers ────────────────────────────────────────────────────────────


def _arms_in_scope(smoke: bool, arms_filter: tuple[str, ...]) -> list[X.Arm]:
    arms = X.all_arms()
    if smoke and not arms_filter:
        return [a for a in arms if a.arm_id == X.PILOT_ARM]
    if arms_filter:
        return [a for a in arms if a.arm_id in set(arms_filter)]
    return arms


def phase_p8(out_root: Path, results_dir: Path, layers, smoke: bool, arms_filter) -> None:
    _phase("p8_fits")
    arms = _arms_in_scope(smoke, arms_filter)
    units = [a.arm_id for a in arms]
    total = len(units) * len(layers)
    k = 0
    verdicts = {}
    for arm_id in units:
        for layer in layers:
            t0 = time.time()
            res = fit_arm_layer(out_root, results_dir, arm_id, layer, smoke)
            verdicts[f"{arm_id}_L{layer}"] = {
                "verdict": res["map_change"]["verdict"],
                "D": res["map_change"]["D"],
                "D_ci95": res["map_change"]["D_ci95"],
                "m0_r2": res["fits"]["M0"]["heldout_r2"],
                "mplus_r2": res["fits"]["Mplus"]["heldout_r2"],
            }
            k += 1
            print(
                f"[p8] unit {k}/{total} {arm_id}_L{layer} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    counts: dict[str, int] = {}
    for v in verdicts.values():
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    _atomic_json(
        results_dir / "map_change_summary.json",
        {
            "verdicts": verdicts,
            "verdict_counts": counts,
            "n_cells": total,
            "unresolved_frac": counts.get("Unresolved", 0) / max(1, total),
            "smoke": smoke,
            **_meta(),
        },
    )
    # panel fit_cell continuity (home regime): production dims only — the
    # tiny-model smoke's 16-dim panel cannot satisfy fit_cell's (28, 3584)
    # r_B contract; the call-shape bind runs in tests/test_issue1768.py.
    if not smoke:
        import issue1768_directions as dirs

        rb = dirs.load_rb_tensors(out_root)
        for arm in arms:
            stack = rb[arm.beh_key]
            for layer in layers:
                for span in ("context", "prefix"):
                    try:
                        panel_fit_for_arm(out_root, results_dir, arm.arm_id, layer, stack, span)
                    except AssertionError as e:  # missing panel rows fail loud per arm
                        logger.warning("[p8-panel] %s L%d %s: %s", arm.arm_id, layer, span, e)
                        raise


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "eval_results" / "issue_1768")
    ap.add_argument("--phases", default="p8,p9")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default="")
    ap.add_argument("--layers", default=",".join(str(x) for x in X.LAYERS))
    ap.add_argument("--rb-dir", type=Path, default=None, help="fixture rb dir (smoke)")
    ap.add_argument("--wu-model", default=None, help="W_U source model path (smoke)")
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args(argv)
    if args.import_check:
        import issue1768_directions  # noqa: F401
        import issue722_fit_M  # noqa: F401
        import issue779_ffc_n1m_fits  # noqa: F401
        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
        )
        from issue722_load_activations import CellRecord  # noqa: F401

        print("[import-check] OK", flush=True)
        return 0
    layers = tuple(int(x) for x in args.layers.split(","))
    arms_filter = tuple(a for a in args.arms.split(",") if a)
    phases = tuple(p for p in args.phases.split(",") if p)
    for phase in phases:
        if phase == "p8":
            phase_p8(args.out_root, args.results_dir, layers, args.smoke, arms_filter)
        elif phase == "p9":
            import issue1768_directions as dirs

            dirs.run_p9(
                args.out_root,
                args.results_dir,
                layers,
                args.smoke,
                arms_filter,
                rb_dir=args.rb_dir,
                wu_model=args.wu_model,
            )
        else:
            raise ValueError(phase)
    _phase("done")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit: finalize-race guard (#1689)
