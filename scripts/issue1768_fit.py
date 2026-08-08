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
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import issue1768_cells as X  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.fit")

# Arms fitted through this core by external reusers (#1947's battery) register
# their method here; #1768's own arms keep resolving through X.arm_method.
EXTERNAL_ARM_METHOD: dict[str, str] = {}


def _resolve_arm_method(arm_id: str) -> str:
    """Fit-record method label ("lora" | "ft"): an external-registered arm
    (e.g. a #1947 slug, absent from #1768's arm registry) resolves from
    EXTERNAL_ARM_METHOD; #1768's own arms via X.arm_method; a truly unknown
    arm still fails fast with the registry's KeyError."""
    return EXTERNAL_ARM_METHOD.get(arm_id) or X.arm_method(arm_id)


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

# plan §4.5 file names; round 4 EXTENDS the map with the three rung labels
# and round 5 with the three b_rel labels (the designed KeyError-loud
# extension point — plan v10/v13 §4.5): rung percell files are
# `<arm>_L<L>_r_<rung>.json` under on_target_r4/; b_rel percell files are
# `<arm>_L<L>_b_rel<j>.json` under on_target_r5/.
PFX_PERCELL_SUFFIX = {
    "own": "own",
    "ctrl": "control",
    "r_short": "r_short",
    "r_mid": "r_mid",
    "r_long": "r_long",
    "b_rel1": "b_rel1",
    "b_rel2": "b_rel2",
    "b_rel3": "b_rel3",
}


def _pfx_out(out_root: Path) -> Path:
    return Path(out_root) / "on_target"


def _pfx_results(results_dir: Path) -> Path:
    return Path(results_dir) / "on_target"


def _lad_out(out_root: Path) -> Path:
    return Path(out_root) / "on_target_r4"


def _lad_results(results_dir: Path) -> Path:
    return Path(results_dir) / "on_target_r4"


def _brl_out(out_root: Path) -> Path:
    return Path(out_root) / "on_target_r5"


def _brl_results(results_dir: Path) -> Path:
    return Path(results_dir) / "on_target_r5"


def pfx_cell_paths(
    out_root: Path, results_dir: Path, arm_id: str, layer: int, cond_label: str
) -> tuple[Path, Path, Path]:
    """The ONE canonical (fits_json, fit_state_npz, percell_json) path triple
    per pfx/lad cell. Producer (`_pfx_fit_core`) and consumer
    (`phase_pfx7`/`phase_lad7`/`phase_brl7`) BOTH compose paths through this
    helper so a writer/reader name drift cannot recur (r3-v2 Critical 1: the
    ctrl fits JSON was written `_ctrl` and read `_control`). ``cond_label`` ∈
    {"own", "ctrl", "bare_n"} (round-3 trees, on_target/), a round-4 rung
    label in `X.R4_CONDS` (on_target_r4/ trees), or a round-5 b_rel label in
    `X.R5_CONDS` (on_target_r5/ trees)."""
    if cond_label in X.R5_CONDS:
        res, state_root = _brl_results(results_dir), _brl_out(out_root)
    elif cond_label in X.R4_CONDS:
        res, state_root = _lad_results(results_dir), _lad_out(out_root)
    else:
        res, state_root = _pfx_results(results_dir), _pfx_out(out_root)
    if cond_label == "bare_n":
        suffix = "bare_n"
        stem = f"{arm_id}_L{layer}"
        fits = res / "fits_bare_n" / f"{stem}.json"
    else:
        suffix = PFX_PERCELL_SUFFIX[cond_label]  # KeyError = unknown condition, loud
        stem = f"{arm_id}_L{layer}_{suffix}"
        fits = res / "fits" / f"{stem}.json"
    npz = state_root / "fit_state" / f"{stem}.npz"
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


def _join_pfx_cell(
    tag: str, layer: int, base: dict, plus: dict, tf: dict | None, sample: dict
) -> dict:
    """Store triple -> fit matrices, rows joined by question_idx (UNIQUE within
    the pfx sample — the pinned valtest block carries duplicate SHAS, so qidx
    is the exact pairing key; shas ride along for reporting). ``tf=None`` is
    the round-4 rung shape (no matched-text TF trees — plan v10 Method delta
    (b)): the returned cell then carries no ``Vplus_tf`` key and
    `_pfx_fit_core` skips the Mplus_tf map."""
    C0 = _store_span_rows(base, "context", layer)
    V0 = _store_span_rows(base, "response", layer)
    Cp = _store_span_rows(plus, "context", layer)
    Vp = _store_span_rows(plus, "response", layer)
    Vtf = None if tf is None else _store_span_rows(tf, "response", layer)
    base_q = list(base["row_question_idx"])
    plus_ix = {q: i for i, q in enumerate(plus["row_question_idx"])}
    tf_ix = None if tf is None else {q: i for i, q in enumerate(tf["row_question_idx"])}
    keep = [i for i, q in enumerate(base_q) if q in plus_ix and (tf_ix is None or q in tf_ix)]
    assert len(keep) >= 0.9 * len(base_q), (tag, layer, len(keep), len(base_q))
    b = np.asarray(keep)
    p = np.asarray([plus_ix[base_q[i]] for i in keep])
    qidx = np.asarray([base_q[i] for i in keep])
    rows = sample["rows"]
    for i, q in zip(keep, qidx, strict=True):  # sha-level alignment fail-loud
        assert base["row_sha"][i] == rows[int(q)]["sha"], (tag, int(q), base["row_sha"][i])
    cell = {
        "C0": C0[b],
        "V0": V0[b],
        "Cplus": Cp[p],
        "Vplus": Vp[p],
        "sha": [rows[int(q)]["sha"] for q in qidx],
        "qidx": qidx,
        "src_qidx": np.asarray([rows[int(q)]["src_qidx"] for q in qidx]),
        "split": _pfx_split_from_qidx(qidx, sample),
        "corpus": np.asarray([rows[int(q)]["corpus"] for q in qidx]),
    }
    if tf is not None:
        t_ = np.asarray([tf_ix[base_q[i]] for i in keep])
        cell["Vplus_tf"] = Vtf[t_]
    return cell


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
    maps = {
        "M0": (cell["C0"], cell["V0"]),
        "Mplus": (cell["Cplus"], cell["Vplus"]),
    }
    if "Vplus_tf" in cell:  # round-4 rung cells carry no TF tree (Method delta (b))
        maps["Mplus_tf"] = (cell["Cplus"], cell["Vplus_tf"])
    for name, (Xd, Yd) in maps.items():
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
        "method": _resolve_arm_method(arm_id),
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
        "smoke": smoke,
        **_meta(),
    }
    if "Mplus_tf" in payloads:
        result["decomposition_tf"] = _decomposition_block(
            cell, payloads["M0"], payloads["Mplus_tf"], dev, tf=True
        )
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
    out_root: Path,
    results_dir: Path,
    layers,
    smoke: bool,
    arms: list[str],
    *,
    base_units: list[str] | None = None,
    store_root: Path | None = None,
    dest: Path | None = None,
    log_tag: str = "pfx6",
) -> None:
    """Exploratory: does the PREFIX alone change the BASE map (plan §4.5 pfx7
    extras — M0_own vs M0_bare)? Per (base@prefix unit, layer): fit M0 on the
    prefixed base store and on the round-1 bare store (same shas, same n),
    then Δ_med between the two maps on both cells' shared test grids.

    Round 4 threads ``base_units`` = the rung base units, ``store_root`` =
    on_target_r4, ``dest`` = m0_rung_effect.json (the lad8 dose-curve input);
    defaults preserve the round-3 behavior byte-identically."""
    dest = dest or (_pfx_results(results_dir) / "m0_prefix_effect.json")
    if dest.exists():
        return
    dev = _device()
    sample = X.load_pfx_sample(out_root)
    if base_units is None:
        base_units = sorted({X.pfx_base_unit(a, c) for a in arms for c in _pfx_conds(smoke, a)})
    store_root = store_root or _pfx_out(out_root)
    out = {}
    for bu in base_units:
        bare_name = bu.split("@")[0]  # base_content | base_mk
        own_store = _load_store(store_root / "corpus_capture" / bu / "pooled.pt")
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
            print(f"[{log_tag}] m0_prefix_effect {bu}_L{layer}", flush=True)
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
            if cond_label == "bare_n":
                rerun = "pfx6 (round 4/5: stage the r3 bare_n cell — lad7/brl7 staging)"
            elif cond_label in X.R4_CONDS:
                rerun = "lad5 (round 5: stage the r4 percell — brl7 staging)"
            elif cond_label in X.R5_CONDS:
                rerun = "brl5"
            else:
                rerun = "pfx5"
            raise RuntimeError(
                f"[pfx7/lad7/brl7] missing {p} — cell {arm_id}_L{layer}@{cond_label} not "
                f"fitted; re-run {rerun} (a partially-fanned shard may have been aborted)"
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


# ── lad: round-4 rung fits + ladder contrasts (plan v10) ────────────────────
#
# lad5  per-rung fits + floors (36 cells = 4 arms x 3 layers x 3 rungs; no TF
#       maps — Method delta (b)) + the base-bare M0 dose refit (pfx6 machinery)
#       -> on_target_r4/{fits,percell,fit_state} + m0_rung_effect.json.
# lad7  contrast reduce: ΔD = D_rung − D_bare@n per (arm, layer, rung) vs the
#       ROUND-3 bare_n cells + the registered m-contrasts (m_long−m_ctrl,
#       m_long−m_short, comparator m_long−m_own), (sha, qidx)-joined over the
#       pinned 1,000 test rows -> map_change_ladder.json.
# lad8  prefix-mapping arm (both-arms rule): per-rung prefix Δ-reads + pooled
#       prefix->response fits (base pools r3 pers/conv/icl + 3 rungs) + the
#       base-map dose curve -> prefix_ladder_reads.json + results mirror.

LAD_R3_MIRROR = "on_target_r4/inputs/r3_results"  # lad_build's Hub mirror prefix


def _lad_fit_arms(smoke: bool, arms_filter: tuple[str, ...]) -> list[str]:
    if arms_filter:
        want = set(arms_filter)
        unknown = want - set(X.R4_ARMS)
        assert not unknown, f"--arms outside the r4 arm set: {sorted(unknown)}"
        return [a for a in X.R4_ARMS if a in want]
    if smoke:
        return ["syc-pers-con-lr1e5-s42"]  # plan §4 smoke-parity arm
    return list(X.R4_ARMS)


def _lad_conds(smoke: bool) -> tuple[str, ...]:
    return ("r_long",) if smoke else X.R4_CONDS


def load_lad_cell(arm_id: str, cond: str, layer: int, out_root: Path) -> dict:
    """on_target_r4 stores -> fit matrices for one (arm, rung, layer); no TF
    store this round (plan v10 Method delta (b))."""
    root = _lad_out(out_root)
    base = _load_store(root / "corpus_capture" / X.r4_base_unit(cond) / "pooled.pt")
    plus = _load_store(root / "corpus_capture" / X.r4_trained_unit(arm_id, cond) / "pooled.pt")
    sample = X.load_pfx_sample(out_root)
    return _join_pfx_cell(f"{arm_id}@{cond}", layer, base, plus, None, sample)


def fit_lad_cell(
    out_root: Path, results_dir: Path, arm_id: str, cond: str, layer: int, smoke: bool
) -> dict:
    cell = load_lad_cell(arm_id, cond, layer, out_root)
    return _pfx_fit_core(
        out_root,
        results_dir,
        arm_id,
        layer,
        cond,
        cell,
        smoke,
        run_transfer_fold=True,  # plan §6: LMSYS<->WildChat fold inherited on rung fits
    )


def _stage_lad_bare_base(out_root: Path) -> None:
    """Parent pre-stage (the #1315 fanout-shared-staging lesson) of the
    round-1 bare base_content store the lad5 dose refit consumes."""
    from explore_persona_space.orchestrate import hub

    target = _bare_store_path(out_root, "corpus_capture", "base_content", "pooled.pt")
    if not target.exists():
        logger.info("[lad5] staging round-1 base_content store for the dose refit")
        hub.stage_hub_file(
            X.HF_DATA_REPO,
            f"{X.HF_PREFIX}/corpus_capture/base_content/pooled.pt",
            target,
            repo_type="dataset",
        )


def phase_lad5(
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
    _phase("lad5_rung_fits")
    arms = _lad_fit_arms(smoke, arms_filter)
    fanned = not worker and _fanout_fit_arms(
        "lad5", arms, out_root, results_dir, layers, smoke, upload, hf_prefix
    )
    if not fanned:
        cells = [(a, c, layer) for a in arms for c in _lad_conds(smoke) for layer in layers]
        for k, (a, c, layer) in enumerate(cells):
            t0 = time.time()
            fit_lad_cell(out_root, results_dir, a, c, layer, smoke)
            print(
                f"[lad5] unit {k + 1}/{len(cells)} {a}@{c}_L{layer} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    if not worker:
        # base-bare M0 dose refit (plan §9 lad5 row: +3 base-bare cells) — the
        # rung base units vs the ROUND-1 bare store, the pfx6 machinery.
        _stage_lad_bare_base(out_root)
        _m0_prefix_effect(
            out_root,
            results_dir,
            layers,
            smoke,
            arms,
            base_units=[X.r4_base_unit(c) for c in _lad_conds(smoke)],
            store_root=_lad_out(out_root),
            dest=_lad_results(results_dir) / "m0_rung_effect.json",
            log_tag="lad5",
        )


def _stage_r3_contrast_inputs(out_root: Path, results_dir: Path, arms, layers) -> None:
    """Ensure the ROUND-3 contrast inputs sit at the `pfx_cell_paths` read
    locations: percell {own, control, bare_n} + fits_bare_n JSONs (repo tree
    else the lad_build HF mirror) and the bare_n fit_state npz (round-3 HF
    fit_state prefix — never in git). Idempotent; fail-loud on both-miss."""
    from explore_persona_space.orchestrate import hub

    repo_r3 = REPO_ROOT / "eval_results" / "issue_1768" / "on_target"
    for arm in arms:
        for layer in layers:
            for cond in ("own", "ctrl", "bare_n"):
                fits_path, npz_path, percell_path = pfx_cell_paths(
                    out_root, results_dir, arm, layer, cond
                )
                needed = [percell_path] + ([fits_path] if cond == "bare_n" else [])
                for p in needed:
                    if p.exists():
                        continue
                    rel = p.relative_to(_pfx_results(results_dir))
                    src = repo_r3 / rel
                    if src.exists():
                        p.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, p)
                    else:
                        hub.stage_hub_file(
                            X.HF_DATA_REPO,
                            f"{X.HF_PREFIX}/{LAD_R3_MIRROR}/{rel.as_posix()}",
                            p,
                            repo_type="dataset",
                        )
                if cond == "bare_n" and not npz_path.exists():
                    hub.stage_hub_file(
                        X.HF_DATA_REPO,
                        f"{X.HF_PREFIX}/on_target/fit_state/{npz_path.name}",
                        npz_path,
                        repo_type="dataset",
                    )


def _load_percell_rows(out_root: Path, results_dir: Path, arm: str, layer: int, cond: str) -> list:
    _, _, percell = pfx_cell_paths(out_root, results_dir, arm, layer, cond)
    assert percell.exists(), (str(percell), "percell rows missing — staging/lad5 incomplete")
    return json.loads(percell.read_text())["rows"]


def _paired_m_contrast(
    a_rows: list, b_rows: list, seed: int, *, expect_pairs: int | None = None, smoke: bool = False
) -> dict:
    """Paired contrast of raw per-context medians m_a − m_b over rows joined
    by the EXPLICIT (sha, qidx) key (plan v10 §3 + the r4 consistency-checker
    advisory: the pinned test block carries duplicate SHAS — 942 unique of
    1,000 — so a sha-only dict join silently collapses rows; (sha, qidx) is a
    true 1,000-row pairing since qidx is unique within the shared pfx sample).
    Production asserts EXACTLY ``expect_pairs`` pairs (kill criterion (c)
    class); the assert demotes to a log line under smoke (#1345 rule).
    Bootstrap: ONE shared row-index resample applied to both sides per draw
    (batched median over the draw axis — no per-draw python loop)."""
    a_map = {(r["sha"], int(r["qidx"])): float(r["delta"]) for r in a_rows}
    b_map = {(r["sha"], int(r["qidx"])): float(r["delta"]) for r in b_rows}
    assert len(a_map) == len(a_rows), "duplicate (sha, qidx) keys on contrast side a"
    assert len(b_map) == len(b_rows), "duplicate (sha, qidx) keys on contrast side b"
    shared = sorted(set(a_map) & set(b_map))
    if expect_pairs is not None:
        if smoke:
            logger.info(
                "[lad7] smoke: m-contrast join %d pairs (production expects %d — demoted)",
                len(shared),
                expect_pairs,
            )
        else:
            assert len(shared) == expect_pairs, (
                len(shared),
                expect_pairs,
                "lad7 m-contrast (sha, qidx) join drift (kill criterion c class)",
            )
    assert shared, "no shared (sha, qidx) rows between m-contrast sides"
    a = np.asarray([a_map[k] for k in shared])
    b = np.asarray([b_map[k] for k in shared])
    rng = np.random.default_rng(seed)
    n = len(shared)
    ridx = rng.integers(0, n, (N_CI_DRAWS, n))
    draws = np.median(a[ridx], axis=1) - np.median(b[ridx], axis=1)
    ci = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
    return {
        "n_pairs": n,
        "m_a": float(np.median(a)),
        "m_b": float(np.median(b)),
        "diff": float(np.median(a) - np.median(b)),
        "diff_ci95": ci,
        "join": "(sha, qidx) exact",
    }


def lad_richness_verdict(ci_long_ctrl: list[float], ci_long_short: list[float]) -> str:
    """Plan v10 §3 richness-vs-identity lattice (DISJOINT + exhaustive)."""
    if ci_long_ctrl[1] < 0:
        return "Identity-consistent"
    if ci_long_ctrl[1] >= 0 and ci_long_short[0] > 0:
        return "Richness-consistent"
    return "Mixed"


def lad_own_suppression_verdict(ci_long_own: list[float]) -> str:
    """Plan v10 §3 comparator own-suppression lattice (DISJOINT + exhaustive)."""
    if ci_long_own[0] > 0:
        return "Own-suppressed"
    if ci_long_own[1] < 0:
        return "Not-suppressed"
    return "Indeterminate"


def phase_lad7(
    out_root: Path, results_dir: Path, layers, smoke: bool, arms_filter, *, force: bool = False
) -> None:
    _phase("lad7_ladder_contrast")
    res = _lad_results(results_dir)
    dest = res / "map_change_ladder.json"
    if dest.exists() and not force:
        # skip-if-output-exists guard, matching the sibling phases' convention
        # (concern `lad7-no-resume-guard`); `--force` recomputes deliberately.
        logger.info("[lad7] map_change_ladder.json present — resume skip (--force to recompute)")
        return
    arms = _lad_fit_arms(smoke, arms_filter)
    conds = _lad_conds(smoke)
    if not smoke:
        _stage_r3_contrast_inputs(out_root, results_dir, arms, layers)
    ladder = X.load_r4_ladder(out_root)
    realized = {c: ladder["rungs"][c]["realized_tokens"] for c in X.R4_CONDS}
    # plan §3: verdict lattices bind at the pre-registered primary layer L19
    # (all four arms are content arms); tiny-layer smokes/tests anchor on the
    # first available layer (the phase_p8 anchor-layer convention).
    primary = 19 if 19 in {int(x) for x in layers} else int(layers[0])

    cells: dict[str, dict] = {}
    m_table: dict[str, dict] = {}
    for arm in arms:
        ai = X.R4_ARMS.index(arm)
        for layer in layers:
            bare_state = None
            bare_fit = None
            try:
                bare_fit, bare_state = _pfx_cell_inputs(out_root, results_dir, arm, layer, "bare_n")
            except RuntimeError:
                if not smoke:
                    raise
                logger.info(
                    "[lad7] smoke: bare_n cell absent for %s L%d — ΔD leg skipped "
                    "(production-mode pytest covers it)",
                    arm,
                    layer,
                )
            for cond in conds:
                rung_fit, rung_state = _pfx_cell_inputs(out_root, results_dir, arm, layer, cond)
                seed = X.FLOOR_SEED + 9000 + int(layer) * 100 + ai * 10 + X.R4_CONDS.index(cond)
                row = {
                    "arm_id": arm,
                    "method": rung_fit["method"],
                    "layer": int(layer),
                    "rung": cond,
                    "realized_tokens": realized[cond],
                    "primary_layer": int(layer) == primary,
                    "D_rung": rung_fit["map_change"]["D"],
                    "D_rung_ci95": rung_fit["map_change"]["D_ci95"],
                    "verdict_rung": rung_fit["map_change"]["verdict"],
                }
                if bare_state is not None:
                    row["D_bare_n"] = bare_fit["map_change"]["D"]
                    row["D_bare_n_ci95"] = bare_fit["map_change"]["D_ci95"]
                    row["contrast"] = _paired_d_contrast(
                        rung_state,
                        bare_state,
                        seed,
                        positive="Rung-amplified",
                        negative="Rung-attenuated",
                    )
                cells[f"{arm}_L{layer}_{cond}"] = row
                print(f"[lad7] contrast {arm}_L{layer}_{cond}", flush=True)
            m_table[f"{arm}_L{layer}"] = _lad_m_row(
                out_root, results_dir, arm, layer, conds, smoke, ai, realized
            )
    richness = {}
    own_supp = None
    for arm in arms:
        mrow = m_table.get(f"{arm}_L{primary}")
        if not mrow or "contrasts" not in mrow:
            continue
        con = mrow["contrasts"]
        if arm == X.R4_COMPARATOR_ARM:
            own_supp = {
                "arm_id": arm,
                "verdict": lad_own_suppression_verdict(con["long_minus_own"]["diff_ci95"]),
                "contrast": con["long_minus_own"],
            }
        else:
            richness[arm] = lad_richness_verdict(
                con["long_minus_ctrl"]["diff_ci95"], con["long_minus_short"]["diff_ci95"]
            )
    n_rich = sum(1 for v in richness.values() if v == "Richness-consistent")
    n_ident = sum(1 for v in richness.values() if v == "Identity-consistent")
    summary = {
        "cells": cells,
        "m_table": m_table,
        "richness_verdicts": richness,
        "own_suppression": own_supp,
        "success_criteria": {
            "n_persona_arms_richness_consistent": n_rich,
            "n_persona_arms_identity_consistent": n_ident,
            "n_persona_arms": len(richness),
            "comparator_verdict": (own_supp or {}).get("verdict"),
        },
        "realized_tokens": realized,
        "join_convention": (
            "(sha, qidx) exact join over the pinned test rows; 1,000 pairs asserted in "
            "production (sha-only would collapse to the 942-unique-sha set)"
        ),
        "n_cells": len(cells),
        "smoke": smoke,
        **_meta(),
    }
    _atomic_json(res / "map_change_ladder.json", summary)
    logger.info(
        "[lad7] %d cells; richness-consistent %d / identity-consistent %d of %d persona arms; "
        "comparator=%s",
        len(cells),
        n_rich,
        n_ident,
        len(richness),
        (own_supp or {}).get("verdict"),
    )


def _lad_m_row(
    out_root: Path,
    results_dir: Path,
    arm: str,
    layer: int,
    conds,
    smoke: bool,
    ai: int,
    realized: dict,
) -> dict:
    """Raw per-context medians m per rung + the registered m-contrasts vs the
    round-3 own/ctrl percell rows (plan §3 contrasts (ii)/(iii) + the dose
    ordering); the r3-side legs are production-only (smoke covers rung medians
    — the production-mode pytest executes the full branch)."""
    rung_rows = {c: _load_percell_rows(out_root, results_dir, arm, layer, c) for c in conds}
    row: dict = {
        "arm_id": arm,
        "layer": int(layer),
        "m_rung": {c: float(np.median([r["delta"] for r in rr])) for c, rr in rung_rows.items()},
        "realized_tokens": {c: realized[c] for c in conds},
    }
    if smoke:
        row["note"] = "smoke: m-contrast legs fenced (need own/ctrl + all rungs)"
        return row
    own_rows = _load_percell_rows(out_root, results_dir, arm, layer, "own")
    ctrl_rows = _load_percell_rows(out_root, results_dir, arm, layer, "ctrl")
    seed = X.FLOOR_SEED + 11000 + int(layer) * 100 + ai * 10
    con = {
        "long_minus_ctrl": _paired_m_contrast(
            rung_rows["r_long"], ctrl_rows, seed + 1, expect_pairs=X.N_TEST, smoke=smoke
        ),
        "long_minus_short": _paired_m_contrast(
            rung_rows["r_long"], rung_rows["r_short"], seed + 2, expect_pairs=X.N_TEST, smoke=smoke
        ),
        "long_minus_own": _paired_m_contrast(
            rung_rows["r_long"], own_rows, seed + 3, expect_pairs=X.N_TEST, smoke=smoke
        ),
        "mid_minus_short": _paired_m_contrast(
            rung_rows["r_mid"], rung_rows["r_short"], seed + 4, expect_pairs=X.N_TEST, smoke=smoke
        ),
        "long_minus_mid": _paired_m_contrast(
            rung_rows["r_long"], rung_rows["r_mid"], seed + 5, expect_pairs=X.N_TEST, smoke=smoke
        ),
    }
    row["m_own"] = float(np.median([r["delta"] for r in own_rows]))
    row["m_ctrl"] = float(np.median([r["delta"] for r in ctrl_rows]))
    row["contrasts"] = con
    return row


def _lad_pooled_store_path(out_root: Path, unit: str) -> Path:
    tag = unit.partition("@")[2]
    root = _lad_out(out_root) if tag in X.R4_CONDS else _pfx_out(out_root)
    return root / "corpus_capture" / unit / "pooled.pt"


def _stage_r3_stores_for_pooled(out_root: Path, arms: list[str]) -> None:
    """Stage the round-3 prefixed stores the lad8 pooled fits consume (base
    pers/conv/icl + each arm's own/ctrl) — idempotent, canonical local paths."""
    from explore_persona_space.orchestrate import hub

    units = ["base_content@pers", "base_content@conv", "base_content@icl_syc"]
    for a in arms:
        units += [X.pfx_trained_unit(a, c) for c in X.pfx_conditions_for(a)]
    for u in sorted(set(units)):
        target = _pfx_out(out_root) / "corpus_capture" / u / "pooled.pt"
        if not target.exists():
            logger.info("[lad8] staging r3 store %s", u)
            hub.stage_hub_file(
                X.HF_DATA_REPO,
                f"{X.HF_PREFIX}/on_target/corpus_capture/{u}/pooled.pt",
                target,
                repo_type="dataset",
            )


def _lad_results_upload(out_root: Path, results_dir: Path, hf_prefix: str | None) -> None:
    """Results mirror (plan §10): on_target_r4 eval_results JSONs + fit_state
    npz to the data repo. ``hf_prefix`` REQUIRED (the #1005 clobber rule)."""
    from explore_persona_space.orchestrate import hub

    if not hf_prefix:
        raise ValueError(
            "_lad_results_upload requires an explicit hf_prefix (no hardcoded "
            "issue-prefix fallback at an upload destination — #1005 class)"
        )
    res = _lad_results(results_dir)
    url = hub._upload(
        res,
        repo_id=X.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{hf_prefix}/on_target_r4/eval_results",
    )
    if not url:
        raise RuntimeError(f"lad8 results-mirror upload of {res} returned no path")
    state = _lad_out(out_root) / "fit_state"
    if state.exists():
        url = hub._upload(
            state,
            repo_id=X.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{hf_prefix}/on_target_r4/fit_state",
        )
        if not url:
            raise RuntimeError(f"lad8 fit_state upload of {state} returned no path")


def phase_lad8(
    out_root: Path,
    results_dir: Path,
    layers,
    smoke: bool,
    arms_filter,
    *,
    upload: bool = True,
    hf_prefix: str | None = None,
) -> None:
    _phase("lad8_prefix_arm_dose")
    arms = _lad_fit_arms(smoke, arms_filter)
    conds = _lad_conds(smoke)
    res = _lad_results(results_dir)
    dest = res / "prefix_ladder_reads.json"
    if not dest.exists():
        import torch

        dev = _device()
        sample = X.load_pfx_sample(out_root)
        ladder = X.load_r4_ladder(out_root)
        realized = {c: ladder["rungs"][c]["realized_tokens"] for c in X.R4_CONDS}
        groups: dict[str, list[str]] = {"base:base_content": [X.r4_base_unit(c) for c in conds]}
        for a in arms:
            groups[f"arm:{a}"] = [X.r4_trained_unit(a, c) for c in conds]
        if not smoke:  # pool the ROUND-3 prefix conditions too (plan §4.5 lad8b)
            _stage_r3_stores_for_pooled(out_root, arms)
            groups["base:base_content"] += [
                "base_content@pers",
                "base_content@conv",
                "base_content@icl_syc",
            ]
            for a in arms:
                groups[f"arm:{a}"] += [X.pfx_trained_unit(a, c) for c in X.pfx_conditions_for(a)]
        groups = {g: sorted(set(us)) for g, us in groups.items()}
        stores = {
            u: _load_store(_lad_pooled_store_path(out_root, u))
            for us in groups.values()
            for u in us
        }
        # (a) per-rung prefix Δ-reads: trained − base prefix span-mean movement
        delta_reads = []
        for a in arms:
            for c in conds:
                unit, base_unit = X.r4_trained_unit(a, c), X.r4_base_unit(c)
                for layer in layers:
                    p_tr = stores[unit]["arms"]["prefix"][layer].float().mean(dim=0)
                    p_b = stores[base_unit]["arms"]["prefix"][layer].float().mean(dim=0)
                    delta_reads.append(
                        {
                            "arm_id": a,
                            "rung": c,
                            "realized_tokens": realized[c],
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
                print(f"[lad8] pooled prefix fit {gname}_L{layer}", flush=True)
        # (c) base-map dose curve: M0_rung vs M0_bare (lad5's m0_rung_effect)
        # + the round-3 anchors (m0_prefix_effect, committed/mirrored)
        rung_eff_path = res / "m0_rung_effect.json"
        assert rung_eff_path.exists(), (
            str(rung_eff_path),
            "m0_rung_effect.json missing — re-run lad5 (parent leg)",
        )
        rung_eff = json.loads(rung_eff_path.read_text())["cells"]
        dose: dict = {"rungs": {}, "r3_anchors": {}}
        for c in conds:
            bu = X.r4_base_unit(c)
            for layer in layers:
                key = f"{bu}_L{layer}"
                assert key in rung_eff, (key, "dose cell missing from m0_rung_effect.json")
                dose["rungs"][f"{c}_L{layer}"] = {
                    "rung": c,
                    "realized_tokens": realized[c],
                    **rung_eff[key],
                }
        if smoke:
            dose["note"] = "smoke: r3 anchors fenced (m0_prefix_effect staging is production)"
        else:
            anch_path = _pfx_results(results_dir) / "m0_prefix_effect.json"
            if not anch_path.exists():
                from explore_persona_space.orchestrate import hub

                hub.stage_hub_file(
                    X.HF_DATA_REPO,
                    f"{X.HF_PREFIX}/{LAD_R3_MIRROR}/m0_prefix_effect.json",
                    anch_path,
                    repo_type="dataset",
                )
            anch = json.loads(anch_path.read_text())["cells"]
            for bu in ("base_content@pers", "base_content@conv", "base_content@icl_syc"):
                for layer in layers:
                    key = f"{bu}_L{layer}"
                    if key in anch:
                        dose["r3_anchors"][key] = anch[key]
        _atomic_json(
            dest,
            {
                "ladder": {
                    c: {
                        "context_id": ladder["rungs"][c]["context_id"],
                        "conversation_hash": ladder["rungs"][c]["conversation_hash"],
                        "dataset_index": ladder["rungs"][c]["dataset_index"],
                        "realized_tokens": realized[c],
                        "target_tokens": ladder["rungs"][c]["target_tokens"],
                    }
                    for c in X.R4_CONDS
                },
                "prefix_delta_reads": delta_reads,
                "pooled_prefix_fits": pooled_fits,
                "dose_curve": dose,
                "note": (
                    "prefix-based mapping arm at the identifiable level (both-arms rule): "
                    "pooled fits rank <= n distinct prefix conditions (base pools the r3 "
                    "pers/conv/icl prefixes + the 3 rungs in production) — rank-limited, "
                    "exploratory (r3 pfx8 convention; the v7 >=100-distinct re-open "
                    "threshold untouched)"
                ),
                "smoke": smoke,
                **_meta(),
            },
        )
    if upload:
        _lad_results_upload(out_root, results_dir, hf_prefix)
    else:
        logger.info("[lad8] results-mirror upload disabled (--no-upload)")


# ── brl: round-5 behavior-relevant panel fits + contrasts (plan v13) ────────
#
# brl5  per-prefix fits + floors (36 cells = 4 arms x 3 layers x 3 prefixes;
#       no TF maps — Method delta carried) + the base-bare M0 refit (pfx6
#       machinery) -> on_target_r5/{fits,percell,fit_state} +
#       m0_brel_effect.json. Launched (arm x prefix)-sharded 8-way FROM THE
#       START (plan §4.5/§8: job 16134's `_fanout_fit_arms` sharded the 4
#       arms onto 4 of 8 GPUs — the fit tail ran at width 4 with 9 serial
#       cells per shard; pair shards run 12-way work-conserving).
# brl7  contrast reduce: ΔD = D_brel − D_bare@n per (arm, layer, prefix) vs
#       the ROUND-3 bare_n cells + the registered behavior-relevance-vs-
#       identity m-contrasts vs the r3 ctrl/own + r4 r_long committed percell
#       vectors, (sha, qidx)-joined over the pinned 1,000 test rows +
#       gap-closure + the dose-interpolated secondary read ->
#       map_change_brel.json.
# brl8  prefix-mapping arm (both-arms rule): per-prefix Δ-reads + pooled
#       prefix->response fits (3 b_rel conds, rank-limited) + the base-side
#       content read vs the r4 dose anchors -> prefix_brel_reads.json +
#       results mirror.

BRL_R_RESULTS = "on_target_r5/inputs/r_results"  # brl_build's Hub mirror prefix


def _brl_fit_arms(smoke: bool, arms_filter: tuple[str, ...]) -> list[str]:
    if arms_filter:
        want = set(arms_filter)
        unknown = want - set(X.R5_ARMS)
        assert not unknown, f"--arms outside the r5 arm set: {sorted(unknown)}"
        return [a for a in X.R5_ARMS if a in want]
    if smoke:
        return ["syc-pers-con-lr1e5-s42"]  # plan §4 smoke-parity arm
    return list(X.R5_ARMS)


def _brl_conds(smoke: bool, conds_filter: tuple[str, ...] = ()) -> tuple[str, ...]:
    if conds_filter:
        unknown = set(conds_filter) - set(X.R5_CONDS)
        assert not unknown, f"--conds outside the r5 condition set: {sorted(unknown)}"
        return tuple(c for c in X.R5_CONDS if c in set(conds_filter))
    return ("b_rel1",) if smoke else X.R5_CONDS


def _stage_brl_panel(out_root: Path) -> dict:
    """The brl_build-pinned panel at the fit side: local out-root -> repo
    commit copy -> Hub (fail-loud both-miss); returns the loaded panel."""
    from explore_persona_space.orchestrate import hub

    path = Path(out_root) / "on_target_r5" / "inputs" / "prefix_ladder_r5.json"
    if not path.exists():
        repo_copy = (
            REPO_ROOT
            / "eval_results"
            / "issue_1768"
            / "on_target_r5"
            / "inputs"
            / "prefix_ladder_r5.json"
        )
        if repo_copy.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo_copy, path)
        else:
            hub.stage_hub_file(
                X.HF_DATA_REPO,
                f"{X.HF_PREFIX}/on_target_r5/inputs/prefix_ladder_r5.json",
                path,
                repo_type="dataset",
            )
    return X.load_r5_brel_panel(out_root)


def _stage_r4_ladder(out_root: Path) -> dict:
    """The r4 rung ladder at the fit side (realized r_mid/r_long tokens for
    the dose-interpolated secondary read): local -> repo -> Hub."""
    from explore_persona_space.orchestrate import hub

    path = Path(out_root) / "on_target_r4" / "inputs" / "prefix_ladder.json"
    if not path.exists():
        repo_copy = (
            REPO_ROOT / "eval_results" / "issue_1768" / "on_target_r4" / "inputs"
        ) / "prefix_ladder.json"
        if repo_copy.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo_copy, path)
        else:
            hub.stage_hub_file(
                X.HF_DATA_REPO,
                f"{X.HF_PREFIX}/on_target_r4/inputs/prefix_ladder.json",
                path,
                repo_type="dataset",
            )
    return X.load_r4_ladder(out_root)


def load_brl_cell(arm_id: str, cond: str, layer: int, out_root: Path) -> dict:
    """on_target_r5 stores -> fit matrices for one (arm, prefix, layer); no TF
    store this round (plan v13 Method delta, carried from r4)."""
    root = _brl_out(out_root)
    base = _load_store(root / "corpus_capture" / X.r5_base_unit(cond) / "pooled.pt")
    plus = _load_store(root / "corpus_capture" / X.r5_trained_unit(arm_id, cond) / "pooled.pt")
    sample = X.load_pfx_sample(out_root)
    return _join_pfx_cell(f"{arm_id}@{cond}", layer, base, plus, None, sample)


def fit_brl_cell(
    out_root: Path, results_dir: Path, arm_id: str, cond: str, layer: int, smoke: bool
) -> dict:
    cell = load_brl_cell(arm_id, cond, layer, out_root)
    return _pfx_fit_core(
        out_root,
        results_dir,
        arm_id,
        layer,
        cond,
        cell,
        smoke,
        run_transfer_fold=True,  # plan §6: LMSYS<->WildChat fold inherited on b_rel fits
    )


def _fanout_fit_pairs(
    phase: str,
    pairs: list[tuple[str, str]],
    out_root: Path,
    results_dir: Path,
    layers,
    smoke: bool,
    upload: bool,
    hf_prefix: str | None,
) -> bool:
    """(arm x cond)-pair subprocess fan-out across every visible GPU — the
    brl5 width fix (plan §4.5/§8: job 16134's arm-sharded `_fanout_fit_arms`
    left 4 of 8 GPUs idle through the whole lad5 fit tail). Work-conserving
    queue: one subprocess per pair (`--arms <a> --conds <c> --worker`), at
    most one live per GPU, CVD-pinned per shard (#545). Returns False when
    the caller should run in-process (<=1 GPU or <=1 pair); per-cell
    dest.exists() resume makes shard re-runs idempotent."""
    gpus = _physical_gpus()
    if len(gpus) <= 1 or len(pairs) <= 1:
        return False
    log_dir = _brl_out(out_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = list(pairs)
    running: dict[int, tuple[subprocess.Popen, tuple[str, str], float]] = {}
    log_handles: list = []
    done_count = 0
    try:
        while queue or running:
            for gpu in [g for g in gpus if g not in running]:
                if not queue:
                    break
                arm, cond = queue.pop(0)
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
                    arm,
                    "--conds",
                    cond,
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
                logf = (log_dir / f"{phase}_{arm}_{cond}.log").open("a")
                log_handles.append(logf)
                proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=logf, stderr=logf)
                running[gpu] = (proc, (arm, cond), time.time())
                logger.info(
                    "[%s] pair shard %s@%s on gpu %d (pid %d)", phase, arm, cond, gpu, proc.pid
                )
            time.sleep(3)
            for gpu, (proc, pair, t0) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                del running[gpu]
                arm, cond = pair
                if rc != 0:
                    for sib_proc, sib_pair, _t in running.values():
                        logger.warning("[%s] terminating sibling %s on failure", phase, sib_pair)
                        sib_proc.terminate()
                    deadline = time.time() + 15
                    for sib_proc, _sib_pair, _t in running.values():
                        try:
                            sib_proc.wait(timeout=max(0.1, deadline - time.time()))
                        except subprocess.TimeoutExpired:
                            sib_proc.kill()
                    running.clear()
                    tail_path = log_dir / f"{phase}_{arm}_{cond}.log"
                    tail = tail_path.read_text()[-4000:] if tail_path.exists() else "(no log)"
                    raise RuntimeError(
                        f"[{phase}] pair shard {arm}@{cond} exited rc={rc}\n"
                        f"--- log tail ---\n{tail}"
                    )
                done_count += 1
                print(
                    f"[{phase}] unit {done_count}/{len(pairs)} {arm}@{cond} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    finally:
        for h in log_handles:
            h.close()
    return True


def phase_brl5(
    out_root: Path,
    results_dir: Path,
    layers,
    smoke: bool,
    arms_filter,
    conds_filter: tuple[str, ...] = (),
    *,
    worker: bool = False,
    upload: bool = True,
    hf_prefix: str | None = None,
) -> None:
    _phase("brl5_brel_fits")
    arms = _brl_fit_arms(smoke, arms_filter)
    conds = _brl_conds(smoke, conds_filter)
    pairs = [(a, c) for a in arms for c in conds]
    fanned = not worker and _fanout_fit_pairs(
        "brl5", pairs, out_root, results_dir, layers, smoke, upload, hf_prefix
    )
    if not fanned:
        cells = [(a, c, layer) for a, c in pairs for layer in layers]
        for k, (a, c, layer) in enumerate(cells):
            t0 = time.time()
            fit_brl_cell(out_root, results_dir, a, c, layer, smoke)
            print(
                f"[brl5] unit {k + 1}/{len(cells)} {a}@{c}_L{layer} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    if not worker:
        # base-bare M0 refit (plan §9 brl5 row: +3 base-bare cells) — the b_rel
        # base units vs the ROUND-1 bare store, the pfx6 machinery (as lad5 did).
        _stage_lad_bare_base(out_root)
        _m0_prefix_effect(
            out_root,
            results_dir,
            layers,
            smoke,
            arms,
            base_units=[X.r5_base_unit(c) for c in conds],
            store_root=_brl_out(out_root),
            dest=_brl_results(results_dir) / "m0_brel_effect.json",
            log_tag="brl5",
        )


def _stage_r_results_inputs(out_root: Path, results_dir: Path, arms, layers) -> None:
    """Ensure the ROUND-3 + ROUND-4 contrast inputs sit at the `pfx_cell_paths`
    read locations: r3 percell {own, control, bare_n} + fits_bare_n JSONs and
    r4 percell {r_mid, r_long} JSONs (repo tree else the brl_build `r_results`
    HF mirror, whose layout is results_dir-relative), plus the bare_n
    fit_state npz (round-3 HF fit_state prefix — never in git). r_short rides
    the mirror for figures only — brl7 never reads it, so it is not staged.
    Idempotent; fail-loud on both-miss."""
    from explore_persona_space.orchestrate import hub

    repo_root = REPO_ROOT / "eval_results" / "issue_1768"
    for arm in arms:
        for layer in layers:
            for cond in ("own", "ctrl", "bare_n", "r_mid", "r_long"):
                fits_path, npz_path, percell_path = pfx_cell_paths(
                    out_root, results_dir, arm, layer, cond
                )
                needed = [percell_path] + ([fits_path] if cond == "bare_n" else [])
                for p in needed:
                    if p.exists():
                        continue
                    rel = p.relative_to(Path(results_dir))
                    src = repo_root / rel
                    if src.exists():
                        p.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, p)
                    else:
                        hub.stage_hub_file(
                            X.HF_DATA_REPO,
                            f"{X.HF_PREFIX}/{BRL_R_RESULTS}/{rel.as_posix()}",
                            p,
                            repo_type="dataset",
                        )
                if cond == "bare_n" and not npz_path.exists():
                    hub.stage_hub_file(
                        X.HF_DATA_REPO,
                        f"{X.HF_PREFIX}/on_target/fit_state/{npz_path.name}",
                        npz_path,
                        repo_type="dataset",
                    )


def brl_behavior_relevance_verdict(ci_brel_ctrl: list[float], ci_brel_rlong: list[float]) -> str:
    """Plan v13 §3 behavior-relevance-vs-identity lattice, per (persona arm,
    prefix j) at L19 (DISJOINT + exhaustive): Behavior-relevance-consistent ⇔
    CI95(m_brel_j − m_ctrl).hi >= 0 AND CI95(m_brel_j − m_rlong).lo > 0;
    Identity-consistent ⇔ CI95(m_brel_j − m_ctrl).hi < 0; Mixed otherwise."""
    if ci_brel_ctrl[1] < 0:
        return "Identity-consistent"
    if ci_brel_ctrl[1] >= 0 and ci_brel_rlong[0] > 0:
        return "Behavior-relevance-consistent"
    return "Mixed"


def brl_comparator_verdict(ci_brel_own: list[float]) -> str:
    """Plan v13 §3 comparator content-proximity lattice (DISJOINT + exhaustive)."""
    if ci_brel_own[0] > 0:
        return "Above-own"
    if ci_brel_own[1] < 0:
        return "Below-own"
    return "Indistinguishable"


def brl_arm_majority(labels: list[str]) -> str:
    """Plan v13 §3 arm-level label: the majority label over the arm's 3 prefix
    draws (>= 2/3); no majority ⇔ Mixed. Deterministic iteration order."""
    assert len(labels) == 3, labels
    for lab in ("Behavior-relevance-consistent", "Identity-consistent", "Mixed"):
        if labels.count(lab) >= 2:
            return lab
    return "Mixed"


def _brl_dose_interp_rows(
    mid_rows: list, long_rows: list, t_real: float, t_mid: float, t_long: float
) -> tuple[list, float]:
    """Registered SECONDARY read input (plan §8 row 2): per-context neutral
    reference at the prefix's realized T — linear interpolation of the r4
    r_mid/r_long per-context deltas in log-token space, keyed (sha, qidx).
    Returns (interp rows in the percell row shape, the interpolation weight w
    on the r_long side)."""
    w = (np.log(t_real) - np.log(t_mid)) / (np.log(t_long) - np.log(t_mid))
    mid_map = {(r["sha"], int(r["qidx"])): float(r["delta"]) for r in mid_rows}
    long_map = {(r["sha"], int(r["qidx"])): float(r["delta"]) for r in long_rows}
    assert len(mid_map) == len(mid_rows) and len(long_map) == len(long_rows)
    shared = sorted(set(mid_map) & set(long_map))
    assert shared, "no shared (sha, qidx) rows between the r_mid and r_long percell sides"
    rows = [
        {
            "sha": sha,
            "qidx": qidx,
            "delta": (1.0 - w) * mid_map[(sha, qidx)] + w * long_map[(sha, qidx)],
        }
        for sha, qidx in shared
    ]
    return rows, float(w)


def _brl_m_row(
    out_root: Path,
    results_dir: Path,
    arm: str,
    layer: int,
    conds,
    smoke: bool,
    ai: int,
    realized: dict,
    r4_realized: dict,
) -> dict:
    """Raw per-context medians m per b_rel prefix + the registered m-contrasts
    vs the round-3 ctrl/own AND round-4 r_long committed percell rows (plan §3
    contrasts (ii)/(iii)), gap-closure fractions, and the dose-interpolated
    neutral reference; the r3/r4-side legs are production-only (smoke covers
    the b_rel medians — the production-mode pytest executes the full branch)."""
    brel_rows = {c: _load_percell_rows(out_root, results_dir, arm, layer, c) for c in conds}
    row: dict = {
        "arm_id": arm,
        "layer": int(layer),
        "m_brel": {c: float(np.median([r["delta"] for r in rr])) for c, rr in brel_rows.items()},
        "realized_tokens": {c: realized[c] for c in conds},
    }
    if smoke:
        row["note"] = "smoke: m-contrast legs fenced (need r3 own/ctrl + r4 rung percell)"
        return row
    own_rows = _load_percell_rows(out_root, results_dir, arm, layer, "own")
    ctrl_rows = _load_percell_rows(out_root, results_dir, arm, layer, "ctrl")
    rlong_rows = _load_percell_rows(out_root, results_dir, arm, layer, "r_long")
    rmid_rows = _load_percell_rows(out_root, results_dir, arm, layer, "r_mid")
    seed = X.FLOOR_SEED + 15000 + int(layer) * 100 + ai * 10
    m_own = float(np.median([r["delta"] for r in own_rows]))
    m_ctrl = float(np.median([r["delta"] for r in ctrl_rows]))
    m_rlong = float(np.median([r["delta"] for r in rlong_rows]))
    m_rmid = float(np.median([r["delta"] for r in rmid_rows]))
    con: dict = {}
    dose: dict = {}
    gap: dict = {}
    for j, c in enumerate(conds):
        con[f"{c}_minus_ctrl"] = _paired_m_contrast(
            brel_rows[c], ctrl_rows, seed + 1 + j * 10, expect_pairs=X.N_TEST, smoke=smoke
        )
        con[f"{c}_minus_rlong"] = _paired_m_contrast(
            brel_rows[c], rlong_rows, seed + 2 + j * 10, expect_pairs=X.N_TEST, smoke=smoke
        )
        con[f"{c}_minus_own"] = _paired_m_contrast(
            brel_rows[c], own_rows, seed + 3 + j * 10, expect_pairs=X.N_TEST, smoke=smoke
        )
        m_b = row["m_brel"][c]
        gap[c] = (m_b - m_rlong) / (m_ctrl - m_rlong) if (m_ctrl - m_rlong) != 0 else None
        interp_rows, w = _brl_dose_interp_rows(
            rmid_rows,
            rlong_rows,
            float(realized[c]),
            float(r4_realized["r_mid"]),
            float(r4_realized["r_long"]),
        )
        dose[c] = {
            "interp_weight_on_rlong": w,
            "realized_tokens": realized[c],
            "contrast_vs_interp": _paired_m_contrast(
                brel_rows[c], interp_rows, seed + 4 + j * 10, expect_pairs=X.N_TEST, smoke=smoke
            ),
            "note": (
                "dose-interpolated neutral reference (plan §8 row 2): per-context "
                "linear interp of the r4 r_mid/r_long deltas in log-token space at "
                "the prefix's realized T"
            ),
        }
    row["m_own"] = m_own
    row["m_ctrl"] = m_ctrl
    row["m_rlong"] = m_rlong
    row["m_rmid"] = m_rmid
    row["contrasts"] = con
    row["gap_closure"] = gap
    row["dose_interp"] = dose
    return row


def phase_brl7(
    out_root: Path, results_dir: Path, layers, smoke: bool, arms_filter, *, force: bool = False
) -> None:
    _phase("brl7_brel_contrast")
    res = _brl_results(results_dir)
    dest = res / "map_change_brel.json"
    if dest.exists() and not force:
        # skip-if-output-exists guard (the lad7 resume-guard convention);
        # `--force` recomputes deliberately.
        logger.info("[brl7] map_change_brel.json present — resume skip (--force to recompute)")
        return
    arms = _brl_fit_arms(smoke, arms_filter)
    conds = _brl_conds(smoke)
    if not smoke:
        _stage_r_results_inputs(out_root, results_dir, arms, layers)
    panel = _stage_brl_panel(out_root)
    realized = {c: panel["prefixes"][c]["realized_tokens"] for c in X.R5_CONDS}
    in_band = {c: panel["prefixes"][c]["in_band"] for c in X.R5_CONDS}
    shared_q = {c: panel["prefixes"][c]["question_shared_request_ids"] for c in X.R5_CONDS}
    r4_ladder = _stage_r4_ladder(out_root) if not smoke else None
    r4_realized = (
        {c: r4_ladder["rungs"][c]["realized_tokens"] for c in X.R4_CONDS}
        if r4_ladder is not None
        else {}
    )
    # plan §3: verdict lattices bind at the pre-registered primary layer L19
    # (all four arms are content arms); tiny-layer smokes/tests anchor on the
    # first available layer (the phase_lad7 convention).
    primary = 19 if 19 in {int(x) for x in layers} else int(layers[0])

    cells: dict[str, dict] = {}
    m_table: dict[str, dict] = {}
    for arm in arms:
        ai = X.R5_ARMS.index(arm)
        for layer in layers:
            bare_state = None
            bare_fit = None
            try:
                bare_fit, bare_state = _pfx_cell_inputs(out_root, results_dir, arm, layer, "bare_n")
            except RuntimeError:
                if not smoke:
                    raise
                logger.info(
                    "[brl7] smoke: bare_n cell absent for %s L%d — ΔD leg skipped "
                    "(production-mode pytest covers it)",
                    arm,
                    layer,
                )
            for cond in conds:
                brel_fit, brel_state = _pfx_cell_inputs(out_root, results_dir, arm, layer, cond)
                seed = X.FLOOR_SEED + 13000 + int(layer) * 100 + ai * 10 + X.R5_CONDS.index(cond)
                row = {
                    "arm_id": arm,
                    "method": brel_fit["method"],
                    "layer": int(layer),
                    "prefix": cond,
                    "realized_tokens": realized[cond],
                    "in_band": in_band[cond],
                    "primary_layer": int(layer) == primary,
                    "D_brel": brel_fit["map_change"]["D"],
                    "D_brel_ci95": brel_fit["map_change"]["D_ci95"],
                    "verdict_brel": brel_fit["map_change"]["verdict"],
                }
                if bare_state is not None:
                    row["D_bare_n"] = bare_fit["map_change"]["D"]
                    row["D_bare_n_ci95"] = bare_fit["map_change"]["D_ci95"]
                    row["contrast"] = _paired_d_contrast(
                        brel_state,
                        bare_state,
                        seed,
                        positive="Prefix-amplified",
                        negative="Prefix-attenuated",
                    )
                cells[f"{arm}_L{layer}_{cond}"] = row
                print(f"[brl7] contrast {arm}_L{layer}_{cond}", flush=True)
            m_table[f"{arm}_L{layer}"] = _brl_m_row(
                out_root, results_dir, arm, layer, conds, smoke, ai, realized, r4_realized
            )
    brel_verdicts: dict[str, dict] = {}
    comparator = None
    for arm in arms:
        mrow = m_table.get(f"{arm}_L{primary}")
        if not mrow or "contrasts" not in mrow:
            continue
        con = mrow["contrasts"]
        if arm == X.R5_COMPARATOR_ARM:
            comparator = {
                "arm_id": arm,
                "per_prefix": {
                    c: {
                        "verdict": brl_comparator_verdict(con[f"{c}_minus_own"]["diff_ci95"]),
                        "contrast_vs_own": con[f"{c}_minus_own"],
                        "secondary_vs_rlong": con[f"{c}_minus_rlong"],
                    }
                    for c in conds
                },
            }
        else:
            per_prefix = {
                c: brl_behavior_relevance_verdict(
                    con[f"{c}_minus_ctrl"]["diff_ci95"], con[f"{c}_minus_rlong"]["diff_ci95"]
                )
                for c in conds
            }
            brel_verdicts[arm] = {
                "per_prefix": per_prefix,
                "arm_label": (
                    brl_arm_majority([per_prefix[c] for c in X.R5_CONDS])
                    if len(conds) == 3
                    else "n/a — partial condition subset"
                ),
            }
    arm_labels = [v["arm_label"] for v in brel_verdicts.values()]
    summary = {
        "cells": cells,
        "m_table": m_table,
        "behavior_relevance_verdicts": brel_verdicts,
        "comparator_content_proximity": comparator,
        "success_criteria": {
            "n_persona_arms_identity_consistent": arm_labels.count("Identity-consistent"),
            "n_persona_arms_behavior_relevance_consistent": arm_labels.count(
                "Behavior-relevance-consistent"
            ),
            "n_persona_arms_mixed": arm_labels.count("Mixed"),
            "n_persona_arms": len(arm_labels),
        },
        "realized_tokens": realized,
        "in_band": in_band,
        "question_shared_request_ids": shared_q,
        "pairing_fallback_engaged": panel["pairing"]["fallback_engaged"],
        "join_convention": (
            "(sha, qidx) exact join over the pinned test rows; 1,000 pairs asserted in "
            "production (sha-only would collapse to the 942-unique-sha set)"
        ),
        "n_cells": len(cells),
        "smoke": smoke,
        **_meta(),
    }
    _atomic_json(dest, summary)
    logger.info(
        "[brl7] %d cells; arm labels %s; comparator=%s",
        len(cells),
        {a: v["arm_label"] for a, v in brel_verdicts.items()},
        None
        if comparator is None
        else {c: v["verdict"] for c, v in comparator["per_prefix"].items()},
    )


def _brl_results_upload(out_root: Path, results_dir: Path, hf_prefix: str | None) -> None:
    """Results mirror (plan §10): on_target_r5 eval_results JSONs + fit_state
    npz to the data repo. ``hf_prefix`` REQUIRED (the #1005 clobber rule)."""
    from explore_persona_space.orchestrate import hub

    if not hf_prefix:
        raise ValueError(
            "_brl_results_upload requires an explicit hf_prefix (no hardcoded "
            "issue-prefix fallback at an upload destination — #1005 class)"
        )
    res = _brl_results(results_dir)
    url = hub._upload(
        res,
        repo_id=X.HF_DATA_REPO,
        repo_type="dataset",
        path_in_repo=f"{hf_prefix}/on_target_r5/eval_results",
    )
    if not url:
        raise RuntimeError(f"brl8 results-mirror upload of {res} returned no path")
    state = _brl_out(out_root) / "fit_state"
    if state.exists():
        url = hub._upload(
            state,
            repo_id=X.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{hf_prefix}/on_target_r5/fit_state",
        )
        if not url:
            raise RuntimeError(f"brl8 fit_state upload of {state} returned no path")


def phase_brl8(
    out_root: Path,
    results_dir: Path,
    layers,
    smoke: bool,
    arms_filter,
    *,
    upload: bool = True,
    hf_prefix: str | None = None,
) -> None:
    _phase("brl8_prefix_arm_reads")
    arms = _brl_fit_arms(smoke, arms_filter)
    conds = _brl_conds(smoke)
    res = _brl_results(results_dir)
    dest = res / "prefix_brel_reads.json"
    if not dest.exists():
        import torch

        dev = _device()
        sample = X.load_pfx_sample(out_root)
        panel = _stage_brl_panel(out_root)
        realized = {c: panel["prefixes"][c]["realized_tokens"] for c in X.R5_CONDS}
        groups: dict[str, list[str]] = {"base:base_content": [X.r5_base_unit(c) for c in conds]}
        for a in arms:
            groups[f"arm:{a}"] = [X.r5_trained_unit(a, c) for c in conds]
        stores = {
            u: _load_store(_brl_out(out_root) / "corpus_capture" / u / "pooled.pt")
            for us in groups.values()
            for u in us
        }
        # (a) per-prefix Δ-reads: trained − base prefix span-mean movement (the
        # conv comparator's read on its own-corpus content is the new cell)
        delta_reads = []
        for a in arms:
            for c in conds:
                unit, base_unit = X.r5_trained_unit(a, c), X.r5_base_unit(c)
                for layer in layers:
                    p_tr = stores[unit]["arms"]["prefix"][layer].float().mean(dim=0)
                    p_b = stores[base_unit]["arms"]["prefix"][layer].float().mean(dim=0)
                    delta_reads.append(
                        {
                            "arm_id": a,
                            "prefix": c,
                            "realized_tokens": realized[c],
                            "layer": int(layer),
                            "prefix_delta_norm": float((p_tr - p_b).norm()),
                            "prefix_cos": float(
                                torch.nn.functional.cosine_similarity(p_tr, p_b, dim=0)
                            ),
                            "prefix_norm_base": float(p_b.norm()),
                            "prefix_norm_trained": float(p_tr.norm()),
                        }
                    )
        # (b) pooled prefix->response fits per model group (rank <= 3 distinct
        # b_rel prefixes — reported rank-limited + exploratory, the r3/r4
        # pfx8/lad8 convention; no r3-condition pooling this round, plan §4.5)
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
                print(f"[brl8] pooled prefix fit {gname}_L{layer}", flush=True)
        # (c) base-side content read: M0_brel vs M0_bare (brl5's
        # m0_brel_effect) + the r4 neutral/r3 anchors for the dose comparison
        brel_eff_path = res / "m0_brel_effect.json"
        assert brel_eff_path.exists(), (
            str(brel_eff_path),
            "m0_brel_effect.json missing — re-run brl5 (parent leg)",
        )
        brel_eff = json.loads(brel_eff_path.read_text())["cells"]
        base_read: dict = {"b_rel": {}, "r4_anchors": {}, "r3_anchors": {}}
        for c in conds:
            bu = X.r5_base_unit(c)
            for layer in layers:
                key = f"{bu}_L{layer}"
                assert key in brel_eff, (key, "base cell missing from m0_brel_effect.json")
                base_read["b_rel"][f"{c}_L{layer}"] = {
                    "prefix": c,
                    "realized_tokens": realized[c],
                    **brel_eff[key],
                }
        if smoke:
            base_read["note"] = "smoke: r3/r4 anchors fenced (staging is production)"
        else:
            from explore_persona_space.orchestrate import hub

            r4_eff_path = _lad_results(results_dir) / "m0_rung_effect.json"
            if not r4_eff_path.exists():
                src = REPO_ROOT / "eval_results" / "issue_1768" / "on_target_r4"
                if (src / "m0_rung_effect.json").exists():
                    r4_eff_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src / "m0_rung_effect.json", r4_eff_path)
                else:
                    hub.stage_hub_file(
                        X.HF_DATA_REPO,
                        f"{X.HF_PREFIX}/{BRL_R_RESULTS}/on_target_r4/m0_rung_effect.json",
                        r4_eff_path,
                        repo_type="dataset",
                    )
            r4_eff = json.loads(r4_eff_path.read_text())["cells"]
            r4_ladder = _stage_r4_ladder(out_root)
            for c in X.R4_CONDS:
                for layer in layers:
                    key = f"{X.r4_base_unit(c)}_L{layer}"
                    if key in r4_eff:
                        base_read["r4_anchors"][f"{c}_L{layer}"] = {
                            "rung": c,
                            "realized_tokens": r4_ladder["rungs"][c]["realized_tokens"],
                            **r4_eff[key],
                        }
            anch_path = _pfx_results(results_dir) / "m0_prefix_effect.json"
            if not anch_path.exists():
                src = REPO_ROOT / "eval_results" / "issue_1768" / "on_target"
                if (src / "m0_prefix_effect.json").exists():
                    anch_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src / "m0_prefix_effect.json", anch_path)
                else:
                    hub.stage_hub_file(
                        X.HF_DATA_REPO,
                        f"{X.HF_PREFIX}/{BRL_R_RESULTS}/on_target/m0_prefix_effect.json",
                        anch_path,
                        repo_type="dataset",
                    )
            anch = json.loads(anch_path.read_text())["cells"]
            for bu in ("base_content@pers", "base_content@conv", "base_content@icl_syc"):
                for layer in layers:
                    key = f"{bu}_L{layer}"
                    if key in anch:
                        base_read["r3_anchors"][key] = anch[key]
        _atomic_json(
            dest,
            {
                "panel": {
                    c: {
                        "context_id": panel["prefixes"][c]["context_id"],
                        "request_ids": panel["prefixes"][c]["request_ids"],
                        "realized_tokens": realized[c],
                        "in_band": panel["prefixes"][c]["in_band"],
                        "question_shared_request_ids": panel["prefixes"][c][
                            "question_shared_request_ids"
                        ],
                    }
                    for c in X.R5_CONDS
                },
                "prefix_delta_reads": delta_reads,
                "pooled_prefix_fits": pooled_fits,
                "base_content_read": base_read,
                "note": (
                    "prefix-based mapping arm at the identifiable level (both-arms rule): "
                    "pooled fits rank <= 3 distinct b_rel prefix conditions — rank-limited, "
                    "exploratory (r3/r4 pfx8/lad8 convention; the v7 >=100-distinct re-open "
                    "threshold untouched)"
                ),
                "smoke": smoke,
                **_meta(),
            },
        )
    if upload:
        _brl_results_upload(out_root, results_dir, hf_prefix)
    else:
        logger.info("[brl8] results-mirror upload disabled (--no-upload)")


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
    ap.add_argument("--conds", default="", help="b_rel condition filter (brl5 pair shards)")
    ap.add_argument("--layers", default=",".join(str(x) for x in X.LAYERS))
    ap.add_argument("--rb-dir", type=Path, default=None, help="fixture rb dir (smoke)")
    ap.add_argument("--wu-model", default=None, help="W_U source model path (smoke)")
    ap.add_argument("--worker", action="store_true", help="internal: in-process arm shard")
    ap.add_argument("--no-upload", action="store_true", help="pfx8 results-mirror upload off")
    ap.add_argument("--hf-prefix", default=None, help="upload prefix (smoke: <prefix>_smoke)")
    ap.add_argument(
        "--force",
        action="store_true",
        help="recompute lad7 even when map_change_ladder.json exists (resume-guard override)",
    )
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
    conds_filter = tuple(c for c in args.conds.split(",") if c)
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
        elif phase == "lad5":
            phase_lad5(
                args.out_root,
                args.results_dir,
                layers,
                args.smoke,
                arms_filter,
                worker=args.worker,
                upload=upload,
                hf_prefix=hf_prefix,
            )
        elif phase == "lad7":
            phase_lad7(
                args.out_root, args.results_dir, layers, args.smoke, arms_filter, force=args.force
            )
        elif phase == "lad8":
            phase_lad8(
                args.out_root,
                args.results_dir,
                layers,
                args.smoke,
                arms_filter,
                upload=upload,
                hf_prefix=hf_prefix,
            )
        elif phase == "brl5":
            phase_brl5(
                args.out_root,
                args.results_dir,
                layers,
                args.smoke,
                arms_filter,
                conds_filter,
                worker=args.worker,
                upload=upload,
                hf_prefix=hf_prefix,
            )
        elif phase == "brl7":
            phase_brl7(
                args.out_root, args.results_dir, layers, args.smoke, arms_filter, force=args.force
            )
        elif phase == "brl8":
            phase_brl8(
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
