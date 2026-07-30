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


def _fit_map(
    Xd: np.ndarray,
    Yd: np.ndarray,
    tr,
    val,
    te,
    dev,
    *,
    allow_underdetermined: bool = False,
) -> tuple[np.ndarray, dict, dict]:
    """Val-selected primal ridge with grid-edge extension (plan §4.5).

    ``allow_underdetermined=True`` is the pfx round's DELIBERATE n_train < d
    opt-in (plan v8 §10 (l) / §12 assumption 5: the val-selected ridge is
    regularization-identified in this regime; D compares maps fitted under
    IDENTICAL n/λ-grid/rows within each condition, and the floor is recomputed
    at the same n). Round-1 callers keep the default refusal byte-identically.
    """
    import issue779_ffc_n1m_fits as n1m

    d = Xd.shape[1]
    if not allow_underdetermined:
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
        try:
            W = torch.cholesky_solve(XtY, torch.linalg.cholesky(A))
        except torch.linalg.LinAlgError:
            # fp32 Gram numerically non-PD (near-constant dims inflate the
            # standardizer at tiny n; the cuSOLVER-family gotcha) — exact
            # numerical-backend swap to a symmetrized CPU float64 solve of the
            # SAME system; never a jitter (it would change the floor).
            A64 = A.double().cpu()
            A64 = 0.5 * (A64 + A64.T)
            W = torch.linalg.solve(A64, XtY.double().cpu()).to(dtype=torch.float32, device=dev)
        preds[b] = ((Cte @ W) + torch.as_tensor(ymu, dtype=torch.float32, device=dev)).cpu().numpy()
    return preds


_FLOOR_COND_OFFSET = {"M0": 0, "Mplus": 1}


def floor_seed_for(cond: str) -> int:
    """Deterministic floor-refit seed under the plan §10 seed contract (1768).

    NEVER Python ``hash()`` here: string hashing is PYTHONHASHSEED-randomized
    per process, so the B floor draws would be non-reproducible across runs
    and a crash-resumed p8 would seed remaining cells differently from
    completed ones (round-1 Major 3; verdict-adjacent — floor_p95 feeds D).
    """
    return X.FLOOR_SEED + _FLOOR_COND_OFFSET[cond]


def _map_change_block(
    cell: dict,
    m0_payload: dict,
    mp_payload: dict,
    lam0: float,
    dev,
    smoke: bool,
    state_out: dict | None = None,
) -> dict:
    """Δ_med vs refit floor + verdict + CI (plan §4.5 / §3 lattice).

    ``state_out`` (pfx rounds): receives the per-row statistical inputs the
    pfx7 paired-contrast reduce re-consumes — ``delta_rows`` (n_te,), the M0
    ``floor_rows`` (B/2, n_te) pair matrix, and the test indices ``te``.
    """
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
        preds = _floor_refit_preds(Xd, Yd, tr, C0_te, lam0, n_refits, floor_seed_for(cond), dev)
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
    if state_out is not None:
        state_out["delta_rows"] = delta_rows.astype(np.float32)
        state_out["floor_rows"] = floor_rows.astype(np.float32)
        state_out["te"] = te

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


def _transfer_fold(cell: dict, dev, *, allow_underdetermined: bool = False) -> dict:
    """Corpus-provenance transfer (group = source corpus; ood-folds rule).

    ``allow_underdetermined`` (pfx): the per-corpus halves of an n=3,000 train
    split sit below d — the same §10 (l) regularization-identified opt-in as
    the main pfx fits; the floor then drops to a minimum-viable n of 8.
    """
    tr, val, _te = _split_idx(cell["split"])
    out = {}
    for src, dst in (("lmsys", "wildchat"), ("wildchat", "lmsys")):
        tr_src = tr[cell["corpus"][tr] == src]
        ev_dst = tr[cell["corpus"][tr] == dst]
        d = cell["C0"].shape[1]
        n_floor = 8 if allow_underdetermined else d
        if len(tr_src) <= n_floor or len(ev_dst) < 4:  # per-corpus split halves n_train
            out[f"{src}->{dst}"] = {
                "skipped": f"n_tr={len(tr_src)} <= {n_floor} or n_ev={len(ev_dst)} < 4"
            }
            continue
        pred_te, meta, payload = _fit_map(
            cell["C0"],
            cell["V0"],
            tr_src,
            val,
            ev_dst,
            dev,
            allow_underdetermined=allow_underdetermined,
        )
        out[f"{src}->{dst}"] = {
            "heldout_r2": _pooled_r2(pred_te, cell["V0"][ev_dst]),
            "n_train": int(len(tr_src)),
            "n_eval": int(len(ev_dst)),
            "selected_lambda": meta["selected_lambda"],
        }
    # STATED DEVIATION (round-1 Minor; carried for the analyzer): this fold
    # evaluates on the dst corpus's TRAIN rows — held out from the src-corpus
    # fit, but not the pinned "test rows" plan §4.5 names, which are
    # unsatisfiable as written (the pinned test set is entirely LMSYS-derived).
    out["note"] = (
        "eval rows = dst-corpus TRAIN rows (held out from the src fit); plan §4.5 "
        "'test rows' is unsatisfiable — the pinned test set is all LMSYS-derived"
    )
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
        "method": X.arm_method(arm_id),  # lora | ft — the amendment's grouping column
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


def _panel_baseline_reads(cells) -> dict:
    """Identity+bias baseline + kNN retrieval over the panel cell arrays.

    The standing mapping-baselines pair attached to the panel `fit_cell` maps
    (plan §6 "both reads attached to EVERY fitted map"; round-1 Major 6) —
    same c→v spaces as fit_cell's ridge (d_in = d_out), deterministic
    even/odd-index halves over the ~120 records. `_identity_bias_reads`
    carries the kNN reads (euclidean + cosine) via `_map_reads`.
    """
    C0 = np.stack([c.c0 for c in cells])
    V0 = np.stack([c.v0 for c in cells])
    Cp = np.stack([c.cplus for c in cells])
    Vp = np.stack([c.vplus for c in cells])
    tr = np.arange(0, len(cells), 2)
    te = np.arange(1, len(cells), 2)
    out: dict = {"split": "even-index train / odd-index test (deterministic)"}
    for name, (Xd, Yd) in {"M0": (C0, V0), "Mplus": (Cp, Vp)}.items():
        out[name] = _identity_bias_reads(Xd[tr], Yd[tr], Xd[te], Yd[te])
    return out


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
        "baselines": _panel_baseline_reads(cells),  # identity+bias + kNN (Major 6)
        **_meta(),
    }
    _atomic_json(dest, out)
    return out


# ── pfx: on-target per-condition fits + contrast (plan v8 round 3) ───────────
#
# pfx5  per-(arm, layer, condition) fits on the on_target stores (M0/M⁺/M⁺_tf
#       + baselines + floors B=200 recomputed at n=3,000) — 8-way arm-sharded
#       from the start (the round-1 p8 width lesson).
# pfx6  bare-at-n refits of the ROUND-1 stores on the SAME shas (n-matched
#       off-target comparator) + the exploratory M0_own-vs-M0_bare read.
# pfx7  CPU contrast reduce: paired ΔD = D_own − D_bare@n per arm × layer
#       (shared pinned test rows × refit draws) -> map_change_on_target.json.
# pfx8  prefix-mapping arm (both-arms rule): per-condition prefix Δ-reads +
#       pooled rank-limited prefix->response fits; then the results mirror
#       upload (plan §10 fellows-lane note).

PFX_PERCELL_SUFFIX = {"own": "own", "ctrl": "control"}  # plan §4.5 file names


def _pfx_out(out_root: Path) -> Path:
    return Path(out_root) / "on_target"


def _pfx_results(results_dir: Path) -> Path:
    return Path(results_dir) / "on_target"


def pfx_cell_paths(
    out_root: Path, results_dir: Path, arm_id: str, layer: int, cond_label: str
) -> tuple[Path, Path, Path]:
    """The ONE canonical (fits_json, fit_state_npz, percell_json) path triple
    per pfx cell. Producer (`_pfx_fit_core`) and consumer (`phase_pfx7`) BOTH
    compose paths through this helper so a writer/reader name drift cannot
    recur (r3-v2 Critical 1: the ctrl fits JSON was written `_ctrl` and read
    `_control`). ``cond_label`` ∈ {"own", "ctrl", "bare_n"}; file suffixes are
    the plan §4.5 names {own, control, bare_n}."""
    res = _pfx_results(results_dir)
    if cond_label == "bare_n":
        suffix = "bare_n"
        stem = f"{arm_id}_L{layer}"
        fits = res / "fits_bare_n" / f"{stem}.json"
    else:
        suffix = PFX_PERCELL_SUFFIX[cond_label]  # KeyError = unknown condition, loud
        stem = f"{arm_id}_L{layer}_{suffix}"
        fits = res / "fits" / f"{stem}.json"
    npz = _pfx_out(out_root) / "fit_state" / f"{stem}.npz"
    percell = res / "percell" / f"{arm_id}_L{layer}_{suffix}.json"
    return fits, npz, percell


def _pfx_fit_arms(smoke: bool, arms_filter: tuple[str, ...]) -> list[str]:
    if arms_filter:
        want = set(arms_filter)
        unknown = want - set(X.PFX_ARMS)
        assert not unknown, f"--arms outside the pfx arm set: {sorted(unknown)}"
        return [a for a in X.PFX_ARMS if a in want]
    if smoke:
        return [X.PILOT_ARM]
    return list(X.PFX_ARMS)


def _pfx_conds(smoke: bool, arm_id: str) -> tuple[str, ...]:
    return ("own",) if smoke else X.pfx_conditions_for(arm_id)


def _pfx_split_from_qidx(qidx: np.ndarray, sample: dict) -> np.ndarray:
    n_train, n_val = sample["n_train"], sample["n_val"]
    return np.where(qidx < n_train, "train", np.where(qidx < n_train + n_val, "val", "test"))


def _store_span_rows(store: dict, span: str, layer: int) -> np.ndarray:
    return np.asarray(store["arms"][span][layer].float().numpy(), dtype=np.float64)


def _join_pfx_cell(tag: str, layer: int, base: dict, plus: dict, tf: dict, sample: dict) -> dict:
    """Store triple -> fit matrices, rows joined by question_idx (UNIQUE within
    the pfx sample — the pinned valtest block carries duplicate SHAS, so qidx
    is the exact pairing key; shas ride along for reporting)."""
    C0 = _store_span_rows(base, "context", layer)
    V0 = _store_span_rows(base, "response", layer)
    Cp = _store_span_rows(plus, "context", layer)
    Vp = _store_span_rows(plus, "response", layer)
    Vtf = _store_span_rows(tf, "response", layer)
    base_q = list(base["row_question_idx"])
    plus_ix = {q: i for i, q in enumerate(plus["row_question_idx"])}
    tf_ix = {q: i for i, q in enumerate(tf["row_question_idx"])}
    keep = [i for i, q in enumerate(base_q) if q in plus_ix and q in tf_ix]
    assert len(keep) >= 0.9 * len(base_q), (tag, layer, len(keep), len(base_q))
    b = np.asarray(keep)
    p = np.asarray([plus_ix[base_q[i]] for i in keep])
    t_ = np.asarray([tf_ix[base_q[i]] for i in keep])
    qidx = np.asarray([base_q[i] for i in keep])
    rows = sample["rows"]
    for i, q in zip(keep, qidx, strict=True):  # sha-level alignment fail-loud
        assert base["row_sha"][i] == rows[int(q)]["sha"], (tag, int(q), base["row_sha"][i])
    return {
        "C0": C0[b],
        "V0": V0[b],
        "Cplus": Cp[p],
        "Vplus": Vp[p],
        "Vplus_tf": Vtf[t_],
        "sha": [rows[int(q)]["sha"] for q in qidx],
        "qidx": qidx,
        "src_qidx": np.asarray([rows[int(q)]["src_qidx"] for q in qidx]),
        "split": _pfx_split_from_qidx(qidx, sample),
        "corpus": np.asarray([rows[int(q)]["corpus"] for q in qidx]),
    }


def load_pfx_cell(arm_id: str, cond: str, layer: int, out_root: Path) -> dict:
    """on_target stores -> fit matrices for one (arm, condition, layer)."""
    root = _pfx_out(out_root)
    base = _load_store(root / "corpus_capture" / X.pfx_base_unit(arm_id, cond) / "pooled.pt")
    plus = _load_store(root / "corpus_capture" / X.pfx_trained_unit(arm_id, cond) / "pooled.pt")
    tf = _load_store(root / "corpus_capture_tf" / X.pfx_trained_unit(arm_id, cond) / "pooled_tf.pt")
    sample = X.load_pfx_sample(out_root)
    return _join_pfx_cell(f"{arm_id}@{cond}", layer, base, plus, tf, sample)


def _bare_store_path(out_root: Path, tree: str, unit: str, fname: str) -> Path:
    """Round-1 store path: canonical local (same-out-root resume / smoke) else
    the pfx6 staging dest (exact per-file target — no mirror-root arithmetic)."""
    local = Path(out_root) / tree / unit / fname
    if local.exists():
        return local
    return _pfx_out(out_root) / "bare_staging" / tree / unit / fname


def stage_bare_inputs(out_root: Path, arms: list[str]) -> None:
    """Parent pre-stage of every pfx6 input BEFORE any fan-out (the #1315
    fanout-shared-staging lesson: units never race a shared staging dest).
    Sources are the round-1 PRODUCTION prefix (`X.HF_PREFIX`)."""
    from explore_persona_space.orchestrate import hub

    files: list[tuple[str, str, str]] = [
        ("corpus_capture", u, "pooled.pt") for u in sorted({X.base_unit_for(a) for a in arms})
    ]
    for a in arms:
        files.append(("corpus_capture", a, "pooled.pt"))
        files.append(("corpus_capture_tf", a, "pooled_tf.pt"))
    for tree, unit, fname in files:
        target = _bare_store_path(out_root, tree, unit, fname)
        if not target.exists():
            logger.info("[pfx6] staging %s/%s/%s", tree, unit, fname)
            hub.stage_hub_file(
                X.HF_DATA_REPO, f"{X.HF_PREFIX}/{tree}/{unit}/{fname}", target, repo_type="dataset"
            )


def load_bare_n_cell(arm_id: str, layer: int, out_root: Path) -> dict:
    """ROUND-1 stores subset to the pfx shas, refit-ready at n=3,000.

    Rows are selected by `src_qidx` (the pfx sample's index into the round-1
    sample — unique by construction) and REMAPPED to pfx question indices so
    splits/percell keys share the own-cells' key space.
    """
    base = _load_store(
        _bare_store_path(out_root, "corpus_capture", X.base_unit_for(arm_id), "pooled.pt")
    )
    plus = _load_store(_bare_store_path(out_root, "corpus_capture", arm_id, "pooled.pt"))
    tf = _load_store(_bare_store_path(out_root, "corpus_capture_tf", arm_id, "pooled_tf.pt"))
    sample = X.load_pfx_sample(out_root)
    pfx_by_src = {int(r["src_qidx"]): j for j, r in enumerate(sample["rows"])}

    def _subset(store: dict) -> dict:
        keep = [i for i, q in enumerate(store["row_question_idx"]) if int(q) in pfx_by_src]
        out = {
            "row_question_idx": [pfx_by_src[int(store["row_question_idx"][i])] for i in keep],
            "row_sha": [store["row_sha"][i] for i in keep],
            "arms": {
                span: {li: t[keep] for li, t in per.items()} for span, per in store["arms"].items()
            },
        }
        return out

    cell = _join_pfx_cell(
        f"{arm_id}@bare_n", layer, _subset(base), _subset(plus), _subset(tf), sample
    )
    return cell


def _atomic_npz(path: Path, **arrays) -> None:
    """np.savez with tmp-rename atomicity; tmp keeps the .npz suffix so numpy
    does not silently append one (the #1092 `.npz.tmp.npz` gotcha)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _pfx_fit_core(
    out_root: Path,
    results_dir: Path,
    arm_id: str,
    layer: int,
    cond_label: str,
    cell: dict,
    smoke: bool,
    *,
    run_transfer_fold: bool,
) -> dict:
    """One pfx cell: 3 maps + baselines + floor + verdict, persisted the moment
    it completes (fits JSON + percell rows JSON + fit_state npz). Every output
    path composes through `pfx_cell_paths` — the same helper `phase_pfx7`
    reads through (r3-v2 Critical 1)."""
    dest, npz_path, percell_path = pfx_cell_paths(out_root, results_dir, arm_id, layer, cond_label)
    if dest.exists():
        return json.loads(dest.read_text())
    dev = _device()
    tr, val, te = _split_idx(cell["split"])
    fits = {}
    payloads = {}
    for name, (Xd, Yd) in {
        "M0": (cell["C0"], cell["V0"]),
        "Mplus": (cell["Cplus"], cell["Vplus"]),
        "Mplus_tf": (cell["Cplus"], cell["Vplus_tf"]),
    }.items():
        # allow_underdetermined: the pfx n_train=3,000 < d=3,584 opt-in
        # (plan v8 §10 (l) / §12 assumption 5)
        pred_te, meta, payload = _fit_map(Xd, Yd, tr, val, te, dev, allow_underdetermined=True)
        payloads[name] = payload
        fits[name] = {
            **meta,
            **_map_reads(pred_te, Yd[te]),
            "identity_bias": _identity_bias_reads(Xd[tr], Yd[tr], Xd[te], Yd[te]),
        }
    state: dict = {}
    block = _map_change_block(
        cell,
        payloads["M0"],
        payloads["Mplus"],
        payloads["M0"]["selected_lambda"],
        dev,
        smoke,
        state_out=state,
    )
    percell_suffix = "bare_n" if cond_label == "bare_n" else PFX_PERCELL_SUFFIX[cond_label]
    rows = [
        {
            "sha": cell["sha"][i],
            "qidx": int(cell["qidx"][i]),
            "src_qidx": int(cell["src_qidx"][i]),
            "delta": float(state["delta_rows"][k]),
        }
        for k, i in enumerate(state["te"])
    ]
    _atomic_json(
        percell_path,
        {
            "arm_id": arm_id,
            "layer": layer,
            "condition": percell_suffix,
            "n_rows": len(rows),
            "rows": rows,
            **_meta(),
        },
    )
    _atomic_npz(
        npz_path,
        delta_rows=state["delta_rows"],
        floor_rows=state["floor_rows"],
        test_qidx=cell["qidx"][state["te"]].astype(np.int64),
        test_src_qidx=cell["src_qidx"][state["te"]].astype(np.int64),
        delta_med=np.float64(block["delta_med"]),
        floor_p95=np.float64(block["floor_p95"]),
    )
    result = {
        "arm_id": arm_id,
        "method": X.arm_method(arm_id),
        "layer": layer,
        "condition": percell_suffix,
        "n_rows": int(len(cell["sha"])),
        "n_train": int(len(tr)),
        "n_val": int(len(val)),
        "n_test": int(len(te)),
        "underdetermined_n_lt_d": bool(len(tr) < cell["C0"].shape[1]),
        "fits": fits,
        "map_change": block,
        "decomposition": _decomposition_block(cell, payloads["M0"], payloads["Mplus"], dev),
        "decomposition_tf": _decomposition_block(
            cell, payloads["M0"], payloads["Mplus_tf"], dev, tf=True
        ),
        "smoke": smoke,
        **_meta(),
    }
    if run_transfer_fold:  # plan §6: the LMSYS<->WildChat fold on OWN cells
        result["transfer_fold"] = _transfer_fold(cell, dev, allow_underdetermined=True)
    _atomic_json(dest, result)
    return result


def fit_pfx_cell(
    out_root: Path, results_dir: Path, arm_id: str, cond: str, layer: int, smoke: bool
) -> dict:
    cell = load_pfx_cell(arm_id, cond, layer, out_root)
    return _pfx_fit_core(
        out_root,
        results_dir,
        arm_id,
        layer,
        cond,
        cell,
        smoke,
        run_transfer_fold=(cond == "own"),
    )


def fit_bare_n_cell(
    out_root: Path, results_dir: Path, arm_id: str, layer: int, smoke: bool
) -> dict:
    cell = load_bare_n_cell(arm_id, layer, out_root)
    return _pfx_fit_core(
        out_root, results_dir, arm_id, layer, "bare_n", cell, smoke, run_transfer_fold=False
    )


def _m0_prefix_effect(
    out_root: Path, results_dir: Path, layers, smoke: bool, arms: list[str]
) -> None:
    """Exploratory: does the PREFIX alone change the BASE map (plan §4.5 pfx7
    extras — M0_own vs M0_bare)? Per (base@prefix unit, layer): fit M0 on the
    prefixed base store and on the round-1 bare store (same shas, same n),
    then Δ_med between the two maps on both cells' shared test grids."""
    dest = _pfx_results(results_dir) / "m0_prefix_effect.json"
    if dest.exists():
        return
    dev = _device()
    sample = X.load_pfx_sample(out_root)
    base_units = sorted({X.pfx_base_unit(a, c) for a in arms for c in _pfx_conds(smoke, a)})
    out = {}
    for bu in base_units:
        bare_name = bu.split("@")[0]  # base_content | base_mk
        own_store = _load_store(_pfx_out(out_root) / "corpus_capture" / bu / "pooled.pt")
        bare_store = _load_store(
            _bare_store_path(out_root, "corpus_capture", bare_name, "pooled.pt")
        )
        pfx_by_src = {int(r["src_qidx"]): j for j, r in enumerate(sample["rows"])}
        bare_keep = [
            i for i, q in enumerate(bare_store["row_question_idx"]) if int(q) in pfx_by_src
        ]
        bare_q = np.asarray([pfx_by_src[int(bare_store["row_question_idx"][i])] for i in bare_keep])
        for layer in layers:
            C_own = _store_span_rows(own_store, "context", layer)
            V_own = _store_span_rows(own_store, "response", layer)
            own_q = np.asarray([int(q) for q in own_store["row_question_idx"]])
            C_bare = _store_span_rows(bare_store, "context", layer)[bare_keep]
            V_bare = _store_span_rows(bare_store, "response", layer)[bare_keep]
            rec = {}
            for name, (Xd, Yd, qidx) in {
                "own": (C_own, V_own, own_q),
                "bare": (C_bare, V_bare, bare_q),
            }.items():
                split = _pfx_split_from_qidx(qidx, sample)
                tr, val, te = _split_idx(split)
                pred_te, meta, payload = _fit_map(
                    Xd, Yd, tr, val, te, dev, allow_underdetermined=True
                )
                rec[name] = {
                    "payload": payload,
                    "grid": Xd[te],
                    "qidx_te": qidx[te],
                    "heldout_r2": _pooled_r2(pred_te, Yd[te]),
                    "selected_lambda": meta["selected_lambda"],
                }
            reads = {}
            for grid_name in ("own", "bare"):
                grid = rec[grid_name]["grid"]
                pred_own = _apply_payload(rec["own"]["payload"], grid, dev)
                pred_bare = _apply_payload(rec["bare"]["payload"], grid, dev)
                reads[f"delta_med_on_{grid_name}_grid"] = float(
                    np.median(np.linalg.norm(pred_own - pred_bare, axis=1))
                )
            out[f"{bu}_L{layer}"] = {
                "base_unit": bu,
                "layer": int(layer),
                "m0_own_r2": rec["own"]["heldout_r2"],
                "m0_bare_r2": rec["bare"]["heldout_r2"],
                **reads,
                "note": "exploratory — prefix-vs-bare BASE-map read (plan §4.5)",
            }
            print(f"[pfx6] m0_prefix_effect {bu}_L{layer}", flush=True)
    _atomic_json(dest, {"cells": out, "smoke": smoke, **_meta()})


# ── pfx fan-out (8-way arm shards from the start; round-1 p8 width lesson) ───


def _physical_gpus() -> list[int]:
    """Visible GPUs via CVD/nvidia-smi subprocess (never torch.cuda; gotchas).

    Fail-loud split (r3-v2 Minor): a MISSING nvidia-smi binary means a CPU box
    (VM smoke) — serial in-process is correct there — but nvidia-smi PRESENT
    and FAILING is a broken driver on a GPU box, where a silent `[]` would
    quietly run the fits serially at width 1 (the round-1 p8 ~12x lesson)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        return [int(x) for x in cvd.split(",") if x.strip() != ""]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except FileNotFoundError:
        return []  # no nvidia-smi binary: CPU box — serial is the correct width
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "[pfx] nvidia-smi present but failed — refusing the silent serial "
            f"width-1 fallback on a GPU box (set CUDA_VISIBLE_DEVICES to override): {e}"
        ) from e
    try:
        return [int(line) for line in out.split("\n") if line.strip()]
    except ValueError as e:
        raise RuntimeError(f"[pfx] unparseable nvidia-smi index output: {out!r}") from e


def _fanout_fit_arms(
    phase: str,
    arms: list[str],
    out_root: Path,
    results_dir: Path,
    layers,
    smoke: bool,
    upload: bool,
    hf_prefix: str | None,
) -> bool:
    """Arm-sharded subprocess fan-out across every visible GPU (CVD-pinned per
    shard — the #545 launcher-env rule). Returns False when the caller should
    run in-process (<=1 GPU or <=1 arm); per-cell dest.exists() resume makes
    shard re-runs idempotent."""
    gpus = _physical_gpus()
    if len(gpus) <= 1 or len(arms) <= 1:
        return False
    shards: dict[int, list[str]] = {g: [] for g in gpus}
    for i, arm in enumerate(arms):
        shards[gpus[i % len(gpus)]].append(arm)
    procs: dict[int, tuple[subprocess.Popen, list[str]]] = {}
    log_handles: list = []
    log_dir = _pfx_out(out_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for gpu, shard in shards.items():
        if not shard:
            continue
        cmd = [
            "uv",
            "run",
            "python",
            str(Path(__file__).resolve()),
            "--out-root",
            str(out_root),
            "--results-dir",
            str(results_dir),
            "--phases",
            phase,
            "--arms",
            ",".join(shard),
            "--layers",
            ",".join(str(x) for x in layers),
            "--worker",
        ]
        if smoke:
            cmd.append("--smoke")
        if not upload:
            cmd.append("--no-upload")
        if hf_prefix:
            cmd += ["--hf-prefix", hf_prefix]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        log_path = log_dir / f"{phase}_gpu{gpu}.log"
        logf = log_path.open("a")
        log_handles.append(logf)
        procs[gpu] = (
            subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=logf, stderr=logf),
            shard,
        )
        logger.info(
            "[%s] shard on gpu %d: %d arms (pid %d)", phase, gpu, len(shard), procs[gpu][0].pid
        )
    try:
        failed: list[tuple[int, int]] = []
        for gpu, (proc, _shard) in procs.items():
            rc = proc.wait()
            if rc != 0:
                failed.append((gpu, rc))
        if failed:
            for gpu, (proc, _shard) in procs.items():  # reap any stragglers
                if proc.poll() is None:
                    proc.terminate()
            gpu, rc = failed[0]
            tail_path = log_dir / f"{phase}_gpu{gpu}.log"
            tail = tail_path.read_text()[-4000:] if tail_path.exists() else "(no log)"
            raise RuntimeError(f"[{phase}] shard gpu{gpu} exited rc={rc}\n--- log tail ---\n{tail}")
    finally:
        for h in log_handles:  # r3-v2 Minor: close shard log handles
            h.close()
    return True


def phase_pfx5(
    out_root: Path,
    results_dir: Path,
    layers,
    smoke: bool,
    arms_filter,
    *,
    worker: bool = False,
    upload: bool = True,
    hf_prefix: str | None = None,
) -> None:
    _phase("pfx5_fits")
    arms = _pfx_fit_arms(smoke, arms_filter)
    if not worker and _fanout_fit_arms(
        "pfx5", arms, out_root, results_dir, layers, smoke, upload, hf_prefix
    ):
        return
    cells = [(a, c, layer) for a in arms for c in _pfx_conds(smoke, a) for layer in layers]
    for k, (a, c, layer) in enumerate(cells):
        t0 = time.time()
        fit_pfx_cell(out_root, results_dir, a, c, layer, smoke)
        print(
            f"[pfx5] unit {k + 1}/{len(cells)} {a}@{c}_L{layer} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )


def phase_pfx6(
    out_root: Path,
    results_dir: Path,
    layers,
    smoke: bool,
    arms_filter,
    *,
    worker: bool = False,
    upload: bool = True,
    hf_prefix: str | None = None,
) -> None:
    _phase("pfx6_bare_refit")
    arms = _pfx_fit_arms(smoke, arms_filter)
    if not worker:
        stage_bare_inputs(out_root, arms)  # parent pre-stage (#1315 race lesson)
    fanned = not worker and _fanout_fit_arms(
        "pfx6", arms, out_root, results_dir, layers, smoke, upload, hf_prefix
    )
    if not fanned:
        cells = [(a, layer) for a in arms for layer in layers]
        for k, (a, layer) in enumerate(cells):
            t0 = time.time()
            fit_bare_n_cell(out_root, results_dir, a, layer, smoke)
            print(
                f"[pfx6] unit {k + 1}/{len(cells)} {a}_L{layer} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    if not worker:
        _m0_prefix_effect(out_root, results_dir, layers, smoke, arms)


# ── pfx7: paired ΔD contrast reduce (CPU) ────────────────────────────────────

PFX_REFERENCE_BARE_ARMS = (
    # ctx=bare syc arms: trained context IS the bare corpus, so on-target ==
    # round-1 bare by construction — labeled ΔD ≡ 0 reference rows (plan §4.1).
    "syc-bare-con-lr1e5-s42",
    "syc-bare-po-lr1e5-s42",
    "syc-bare-con-lr1e5-s137",
    "syc-bare-po-lr1e5-s137",
)


def _pfx_primary_layer(arm_id: str) -> int:
    """Pre-registered primary layer per arm class (plan §6: v5 §3 — L19
    content / L25 marker)."""
    return 25 if arm_id.startswith("mk-") else 19


def contrast_verdict(
    ci: list[float],
    *,
    positive: str = "On-target-amplified",
    negative: str = "On-target-attenuated",
) -> str:
    """The §3 per-arm contrast lattice (DISJOINT + exhaustive) — the verdict
    derives from the CI alone (r3-v2 Minor: the point estimate was an unused
    parameter). ``positive``/``negative`` label the two significant sides so a
    control−own contrast never reuses the on/off-target vocabulary (r3-v2
    Minor: inverted semantics)."""
    if ci[0] > 0:
        return positive
    if ci[1] < 0:
        return negative
    return "Indistinguishable"


def _paired_d_contrast(
    a_state: dict,
    b_state: dict,
    seed: int,
    n_draws: int = N_CI_DRAWS,
    *,
    positive: str = "On-target-amplified",
    negative: str = "On-target-attenuated",
) -> dict:
    """Paired ΔD = D_a − D_b over SHARED test rows × refit draws.

    Rows are paired by src_qidx (the round-1 sample index both sides carry);
    each bootstrap draw resamples ONE shared row-index vector applied to both
    sides (paired), recomputing Δ_med − floor_p95 per side per draw.
    ``positive``/``negative`` thread the verdict vocabulary (control reads
    pass control-specific labels).
    """
    a_src = [int(q) for q in a_state["test_src_qidx"]]
    b_src = [int(q) for q in b_state["test_src_qidx"]]
    shared = sorted(set(a_src) & set(b_src))
    assert shared, "no shared test rows between contrast sides"
    a_ix = {q: i for i, q in enumerate(a_src)}
    b_ix = {q: i for i, q in enumerate(b_src)}
    a_cols = np.asarray([a_ix[q] for q in shared])
    b_cols = np.asarray([b_ix[q] for q in shared])
    a_delta = np.asarray(a_state["delta_rows"])[a_cols]
    b_delta = np.asarray(b_state["delta_rows"])[b_cols]
    a_floor = np.asarray(a_state["floor_rows"])[:, a_cols]
    b_floor = np.asarray(b_state["floor_rows"])[:, b_cols]

    def d_stat(delta: np.ndarray, floor: np.ndarray, ridx: np.ndarray | None) -> float:
        if ridx is None:
            dm = float(np.median(delta))
            fm = np.median(floor, axis=1)
        else:
            dm = float(np.median(delta[ridx]))
            fm = np.median(floor[:, ridx], axis=1)
        return dm - float(np.quantile(fm, 0.95))

    d_a = d_stat(a_delta, a_floor, None)
    d_b = d_stat(b_delta, b_floor, None)
    rng = np.random.default_rng(seed)
    n = len(shared)
    draws = np.empty(n_draws)
    for k in range(n_draws):
        ridx = rng.integers(0, n, n)
        draws[k] = d_stat(a_delta, a_floor, ridx) - d_stat(b_delta, b_floor, ridx)
    ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
    return {
        "n_shared_rows": n,
        "d_a_shared": d_a,
        "d_b_shared": d_b,
        "delta_d": d_a - d_b,
        "delta_d_ci95": ci,
        "verdict": contrast_verdict(ci, positive=positive, negative=negative),
    }


def _pfx_cell_inputs(
    out_root: Path, results_dir: Path, arm_id: str, layer: int, cond_label: str
) -> tuple[dict, dict]:
    """(fits JSON, fit_state arrays) for one cell, read through the SAME
    `pfx_cell_paths` helper the producer wrote through (r3-v2 Critical 1);
    a missing cell fails loud NAMING the phase to re-run, not with a raw
    FileNotFoundError (r3 verdict § Unaddressed Cases)."""
    fits_path, npz_path, _ = pfx_cell_paths(out_root, results_dir, arm_id, layer, cond_label)
    for p in (fits_path, npz_path):
        if not p.exists():
            rerun = "pfx6" if cond_label == "bare_n" else "pfx5"
            raise RuntimeError(
                f"[pfx7] missing {p} — cell {arm_id}_L{layer}@{cond_label} not fitted; "
                f"re-run {rerun} (a partially-fanned shard may have been aborted)"
            )
    with np.load(npz_path) as z:
        state = {k: z[k] for k in z.files}
    return json.loads(fits_path.read_text()), state


def phase_pfx7(out_root: Path, results_dir: Path, layers, smoke: bool, arms_filter) -> None:
    _phase("pfx7_contrast")
    arms = _pfx_fit_arms(smoke, arms_filter)
    res = _pfx_results(results_dir)

    table: dict[str, dict] = {}
    for arm in arms:
        # stable per-arm seed index (r3-v2 Minor: an --arms-filtered re-run
        # must draw the same CIs as the full run)
        ai = X.PFX_ARMS.index(arm)
        for layer in layers:
            own_fit, own_state = _pfx_cell_inputs(out_root, results_dir, arm, layer, "own")
            bare_fit, bare_state = _pfx_cell_inputs(out_root, results_dir, arm, layer, "bare_n")
            seed = X.FLOOR_SEED + 7000 + int(layer) * 100 + ai
            row = {
                "arm_id": arm,
                "method": own_fit["method"],
                "layer": int(layer),
                "primary_layer": _pfx_primary_layer(arm) == int(layer),
                "D_own": own_fit["map_change"]["D"],
                "D_own_ci95": own_fit["map_change"]["D_ci95"],
                "verdict_own": own_fit["map_change"]["verdict"],
                "D_bare_n": bare_fit["map_change"]["D"],
                "D_bare_n_ci95": bare_fit["map_change"]["D_ci95"],
                "verdict_bare_n": bare_fit["map_change"]["verdict"],
                "contrast": _paired_d_contrast(own_state, bare_state, seed),
            }
            if "ctrl" in _pfx_conds(smoke, arm):
                ctrl_fit, ctrl_state = _pfx_cell_inputs(out_root, results_dir, arm, layer, "ctrl")
                row["D_control"] = ctrl_fit["map_change"]["D"]
                row["verdict_control"] = ctrl_fit["map_change"]["verdict"]
                # ΔD = D_ctrl − D_own; "Control-below-own" (upper CI < 0) is
                # the prefix-SPECIFIC read (control-specific vocabulary —
                # r3-v2 Minor: never the on/off-target labels here)
                row["control_contrast"] = _paired_d_contrast(
                    ctrl_state,
                    own_state,
                    seed + 50,
                    positive="Control-above-own",
                    negative="Control-below-own",
                )
            table[f"{arm}_L{layer}"] = row
            print(f"[pfx7] contrast {arm}_L{layer}", flush=True)
    primary = {
        arm: table.get(f"{arm}_L{_pfx_primary_layer(arm)}")
        for arm in arms
        if f"{arm}_L{_pfx_primary_layer(arm)}" in table
    }
    n_amplified = sum(
        1 for r in primary.values() if r["contrast"]["verdict"] == "On-target-amplified"
    )
    ctrl_rows = [r for r in primary.values() if "control_contrast" in r]
    n_prefix_specific = sum(1 for r in ctrl_rows if r["D_control"] < r["D_own"])
    summary = {
        "contrast": table,
        "reference_bare_arms": [
            {"arm_id": a, "delta_d": 0.0, "note": "trained context IS the bare corpus (plan §4.1)"}
            for a in PFX_REFERENCE_BARE_ARMS
        ],
        "success_criteria": {
            "n_arms_on_target_amplified_at_primary_layer": n_amplified,
            "n_primary_rows": len(primary),
            "n_control_arms_below_own": n_prefix_specific,
            "n_control_arms": len(ctrl_rows),
        },
        "n_cells": len(table),
        "smoke": smoke,
        **_meta(),
    }
    _atomic_json(res / "map_change_on_target.json", summary)
    logger.info(
        "[pfx7] %d cells; amplified@primary %d/%d; control<own %d/%d",
        len(table),
        n_amplified,
        len(primary),
        n_prefix_specific,
        len(ctrl_rows),
    )


# ── pfx8: prefix-mapping arm (both-arms rule) + results mirror upload ────────


def phase_pfx8(
    out_root: Path,
    results_dir: Path,
    layers,
    smoke: bool,
    arms_filter,
    *,
    upload: bool = True,
    hf_prefix: str | None = None,
) -> None:
    _phase("pfx8_prefix_arm")
    arms = _pfx_fit_arms(smoke, arms_filter)
    res = _pfx_results(results_dir)
    dest = res / "prefix_arm.json"
    if not dest.exists():
        dev = _device()
        sample = X.load_pfx_sample(out_root)
        root = _pfx_out(out_root)
        # model groups: base_<decode> pooled over its prefix conditions; each
        # trained arm pooled over its own(+ctrl) conditions.
        groups: dict[str, list[str]] = {}
        for a in arms:
            for c in _pfx_conds(smoke, a):
                groups.setdefault(f"base:{X.base_unit_for(a)}", []).append(X.pfx_base_unit(a, c))
                groups.setdefault(f"arm:{a}", []).append(X.pfx_trained_unit(a, c))
        groups = {g: sorted(set(us)) for g, us in groups.items()}
        stores = {
            u: _load_store(root / "corpus_capture" / u / "pooled.pt")
            for us in groups.values()
            for u in us
        }
        # (a) per-condition prefix Δ-reads: trained − base prefix span-mean
        delta_reads = []
        for a in arms:
            for c in _pfx_conds(smoke, a):
                unit, base_unit = X.pfx_trained_unit(a, c), X.pfx_base_unit(a, c)
                for layer in layers:
                    import torch

                    p_tr = stores[unit]["arms"]["prefix"][layer].float().mean(dim=0)
                    p_b = stores[base_unit]["arms"]["prefix"][layer].float().mean(dim=0)
                    delta_reads.append(
                        {
                            "arm_id": a,
                            "condition": PFX_PERCELL_SUFFIX[c],
                            "layer": int(layer),
                            "prefix_delta_norm": float((p_tr - p_b).norm()),
                            "prefix_cos": float(
                                torch.nn.functional.cosine_similarity(p_tr, p_b, dim=0)
                            ),
                            "prefix_norm_base": float(p_b.norm()),
                            "prefix_norm_trained": float(p_tr.norm()),
                        }
                    )
        # (b) pooled prefix->response fits per model group (rank-limited)
        pooled_fits = {}
        for gname, units in groups.items():
            for layer in layers:
                Xs, Ys, qs = [], [], []
                for u in units:
                    Xs.append(_store_span_rows(stores[u], "prefix", layer))
                    Ys.append(_store_span_rows(stores[u], "response", layer))
                    qs.append(np.asarray([int(q) for q in stores[u]["row_question_idx"]]))
                Xd, Yd, qidx = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(qs)
                split = _pfx_split_from_qidx(qidx, sample)
                tr, val, te = _split_idx(split)
                pred_te, meta, _payload = _fit_map(
                    Xd, Yd, tr, val, te, dev, allow_underdetermined=True
                )
                pooled_fits[f"{gname}_L{layer}"] = {
                    "group": gname,
                    "layer": int(layer),
                    "n_units_pooled": len(units),
                    "n_distinct_prefix_conditions": len(units),
                    "rank_limited": True,
                    "selected_lambda": meta["selected_lambda"],
                    **_map_reads(pred_te, Yd[te]),
                    "identity_bias": _identity_bias_reads(Xd[tr], Yd[tr], Xd[te], Yd[te]),
                }
                print(f"[pfx8] pooled prefix fit {gname}_L{layer}", flush=True)
        _atomic_json(
            dest,
            {
                "prefix_delta_reads": delta_reads,
                "pooled_prefix_fits": pooled_fits,
                "note": (
                    "prefix-based mapping arm, POOLED across the round's distinct-prefix "
                    "conditions per model; input rank <= n distinct prefixes (prefix "
                    "span-mean ~constant within a condition) — rank-limited, exploratory "
                    "(plan §4.8; the v7 >=100-distinct re-open threshold is untouched)"
                ),
                "smoke": smoke,
                **_meta(),
            },
        )
    if upload:
        _pfx_results_upload(out_root, results_dir, hf_prefix)
    else:
        logger.info("[pfx8] results-mirror upload disabled (--no-upload)")


def _pfx_results_upload(out_root: Path, results_dir: Path, hf_prefix: str | None) -> None:
    """Results mirror (plan §10): on_target eval_results JSONs + fit_state npz
    to the data repo — the fellows lane rsync-excludes eval_results/, so the
    Hub mirror is the durable route for the reduce outputs.

    ``hf_prefix`` is REQUIRED (fail-loud raise, never an `or X.HF_PREFIX`
    fallback at the upload destination — the #1005 clobber shape the
    `--check-upload-prefix-clobber` lint bans: a child issue reusing this
    script would silently inherit this issue's prefix). Every caller passes
    the `resolve_fit_hf_prefix` result, which smoke-suffixes `_smoke`.
    """
    from explore_persona_space.orchestrate import hub

    if not hf_prefix:
        raise ValueError(
            "_pfx_results_upload requires an explicit hf_prefix (no hardcoded "
            "issue-prefix fallback at an upload destination — #1005 class)"
        )
    res = _pfx_results(results_dir)
    url = hub._upload(
        res,
        repo_id=X.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{hf_prefix}/on_target/eval_results",
    )
    if not url:
        raise RuntimeError(f"pfx8 results-mirror upload of {res} returned no path")
    state = _pfx_out(out_root) / "fit_state"
    if state.exists():
        url = hub._upload(
            state,
            repo_id=X.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{hf_prefix}/on_target/fit_state",
        )
        if not url:
            raise RuntimeError(f"pfx8 fit_state upload of {state} returned no path")


# ── phase drivers ────────────────────────────────────────────────────────────


def _arms_in_scope(smoke: bool, arms_filter: tuple[str, ...]) -> list[X.Arm]:
    arms = X.all_arms()
    if smoke and not arms_filter:
        return [a for a in arms if a.arm_id == X.PILOT_ARM]
    if arms_filter:
        return [a for a in arms if a.arm_id in set(arms_filter)]
    return arms


def _gate2_block(out_root: Path, smoke: bool) -> dict | None:
    """The pilot-recorded (re-anchored) gate-2 threshold, or None (smoke /
    standalone p8 rerun with no pilot report — logged, not fatal)."""
    if smoke:
        return None
    path = Path(out_root) / "pilot" / "pilot_report.json"
    if not path.exists():
        logger.warning("[p8] no pilot_report.json — gate-2 p8 halt not armed")
        return None
    return json.loads(path.read_text()).get("gate2")


def phase_p8(out_root: Path, results_dir: Path, layers, smoke: bool, arms_filter) -> None:
    _phase("p8_fits")
    arms = _arms_in_scope(smoke, arms_filter)
    units = [a.arm_id for a in arms]
    total = len(units) * len(layers)
    k = 0
    verdicts = {}
    gate2 = _gate2_block(out_root, smoke)
    anchor_layer = 19 if 19 in layers else layers[0]
    for arm_id in units:
        for layer in layers:
            t0 = time.time()
            res = fit_arm_layer(out_root, results_dir, arm_id, layer, smoke)
            if gate2 is not None and layer == anchor_layer:
                r2 = res["fits"]["M0"]["heldout_r2"]
                if r2 < gate2["threshold"]:
                    # plan §7 gate 2: "Fail at p8 -> halt fits" (round-1 Minor)
                    raise RuntimeError(
                        f"[p8] GATE2 halt: {arm_id} L{layer} M0 R2={r2:.3f} < "
                        f"re-anchored threshold {gate2['threshold']} "
                        f"(pilot R2={gate2.get('pilot_r2')}) — rig-level regression"
                    )
            verdicts[f"{arm_id}_L{layer}"] = {
                "verdict": res["map_change"]["verdict"],
                "method": res.get("method", X.arm_method(arm_id)),  # analyzer method split
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
    # r_B contract; the call shape below is signature-bound by
    # tests/test_issue1768.py::test_fit_cell_call_shape_binds (round-2 pin).
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


def resolve_fit_hf_prefix(smoke: bool, hf_prefix: str | None) -> str:
    """Upload prefix; --smoke ALWAYS lands under a `_smoke` suffix (the capture
    driver's round-2 rule: a smoke run must never write the production bucket)."""
    prefix = hf_prefix or X.HF_PREFIX
    if smoke and not prefix.endswith("_smoke"):
        prefix = f"{prefix}_smoke"
    return prefix


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
    ap.add_argument("--worker", action="store_true", help="internal: in-process arm shard")
    ap.add_argument("--no-upload", action="store_true", help="pfx8 results-mirror upload off")
    ap.add_argument("--hf-prefix", default=None, help="upload prefix (smoke: <prefix>_smoke)")
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
        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            _upload,
            stage_hub_file,
        )
        from issue722_load_activations import CellRecord  # noqa: F401

        print("[import-check] OK", flush=True)
        return 0
    layers = tuple(int(x) for x in args.layers.split(","))
    arms_filter = tuple(a for a in args.arms.split(",") if a)
    phases = tuple(p for p in args.phases.split(",") if p)
    upload = not args.no_upload
    hf_prefix = resolve_fit_hf_prefix(args.smoke, args.hf_prefix)
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
        elif phase == "pfx5":
            phase_pfx5(
                args.out_root,
                args.results_dir,
                layers,
                args.smoke,
                arms_filter,
                worker=args.worker,
                upload=upload,
                hf_prefix=hf_prefix,
            )
        elif phase == "pfx6":
            phase_pfx6(
                args.out_root,
                args.results_dir,
                layers,
                args.smoke,
                arms_filter,
                worker=args.worker,
                upload=upload,
                hf_prefix=hf_prefix,
            )
        elif phase == "pfx7":
            phase_pfx7(args.out_root, args.results_dir, layers, args.smoke, arms_filter)
        elif phase == "pfx8":
            phase_pfx8(
                args.out_root,
                args.results_dir,
                layers,
                args.smoke,
                arms_filter,
                upload=upload,
                hf_prefix=hf_prefix,
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
