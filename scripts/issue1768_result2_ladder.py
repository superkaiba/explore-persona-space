"""#1768 Result 2 — ablation ladder: did fine-tuning move the CONTEXT VECTORS or the MAP?

Four predictions of the fine-tuned answer vectors, scored by held-out R2 on one scale
(natural floor + ceiling):

  rung 1  M0(c0)      nothing updated (floor)
  rung 2  M0(c+)      context updated only
  rung 3  M+(c0)      map updated only
  rung 4  M+(c+)      both updated (ceiling)

Run per TREE (on-policy: target v+ = trained answers on the trained model's OWN text —
the conflated read; matched-text: target v+_tf = trained model teacher-forced on the BASE
model's responses — the weights-carried isolated read) x CONTEXT SET (generic real-user
corpus at full n=15,000; the same corpus n-MATCHED to 3,000; the round-3 trained-in
own-prefix condition at n=3,000). Within a tree both maps come from that tree; M0 serves
both ladders (the base model's on-policy text IS its matched text — the round-1 tf tree
stores no base unit, verified at runtime + via the HF listing in `--phase summary`).

Estimator identity with the committed round-1/round-3 fits: the same primal streaming
ridge core (`issue779_ffc_n1m_fits.fit_ridge_with_weights` — streaming X^TX, ONE eigh
shared across the lambda grid, val-selected lambda with grid-edge extension). The
shared-X K-target wrapper here evaluates the lambda scan in eigenspace (an associativity
reorder of `_ridge_predict_one`); `--phase pilot` pins it equivalent (identical selected
lambda, allclose predictions) against the verbatim `issue1768_fit._fit_map` on synthetic
data AND on the real pilot cell, and cross-checks rung 4 against the committed round-1
`heldout_r2`. Cross rungs apply the persisted-payload path (`n1m.apply_map`).

Staging is strict download -> slice -> reap per unit (the ~104 GB round-1 store tree may
never be bulk-staged; dispatch marker 2026-08-03). Checkpoint per (context-set, arm,
layer) cell; resume skips completed cells.
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
import gc  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import OrderedDict  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import issue1768_cells as X  # noqa: E402
import issue1768_fit as F  # noqa: E402
import issue779_ffc_n1m_fits as n1m  # noqa: E402
from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue1768.result2_ladder")

DEV = torch.device("cpu")  # dispatch marker 2026-08-03: CPU + network only
LAYERS = (14, 19, 25)
REV1 = "c07267285d2cdbf3e0401ddc3e3accae50e496a7"  # round-1 corpus_capture{,_tf} stores
REV3 = "a45494ba5f9cce8863859d558290a1b336b1dac8"  # round-3 on_target stores
OUT_DIR = REPO_ROOT / "eval_results/issue_1768/result2_ladder"
FIG_DIR = REPO_ROOT / "figures/issue_1768/result2_ladder"
COMMITTED_FITS = REPO_ROOT / "eval_results/issue_1768/fits"
BARE_STAGING = REPO_ROOT / "eval_results/issue_1768/on_target/bare_staging"
RUNGS = ("r1_M0_c0", "r2_M0_cplus", "r3_Mplus_c0", "r4_Mplus_cplus")
# tree -> (target key in the cell dict, M+ map name)
TREES = {"onpolicy": ("Vplus", "Mplus"), "matched": ("Vplus_tf", "Mplus_tf")}
CTX_SETS = ("generic", "bare_n", "on_target")
MIN_FREE_GB = 5.0  # dispatch marker: assert headroom before every unit download
PILOT_LAYER = 19


def _arms72() -> list[str]:
    arms = sorted({p.name.rsplit("_L", 1)[0] for p in COMMITTED_FITS.glob("*_L19.json")})
    assert len(arms) == 72, f"expected the 72 round-1 arms, found {len(arms)}"
    return arms


# ── staging: symlink-or-download with reap tracking ──────────────────────────


def default_stage_dir() -> Path:
    """Probe the data-disk per-user dir first (dispatch marker), else data/issue_1768/."""
    user = os.environ.get("USER", "user")
    for cand in (
        Path(f"/mnt/eps-data/{user}/issue1768_result2"),
        REPO_ROOT / "data/issue_1768/hf_dl/result2_stage",
    ):
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            return cand
        except OSError as e:
            logger.warning("[stage] %s not writable (%s) — trying fallback", cand, e)
    raise RuntimeError("no writable staging dir found")


def _assert_headroom(stage: Path) -> None:
    st = os.statvfs(stage)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    assert free_gb >= MIN_FREE_GB, (
        f"staging filesystem below headroom floor: {free_gb:.1f} GB free < {MIN_FREE_GB} GB"
    )


def _symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(target.resolve())


def _stage_file(stage: Path, path_in_repo: str, target: Path, revision: str) -> Path:
    from explore_persona_space.orchestrate import hub

    _assert_headroom(stage)
    hub.stage_hub_file(X.HF_DATA_REPO, path_in_repo, target, repo_type="dataset", revision=revision)
    return target


def setup_stage(stage: Path) -> None:
    """Make `stage` a valid out_root for the issue1768_fit loaders (samples + local units)."""
    _symlink(
        REPO_ROOT / "eval_results/issue_1768/inputs/corpus_sample.json",
        stage / "inputs/corpus_sample.json",
    )
    assert BARE_STAGING.is_dir(), f"round-3 bare_staging missing: {BARE_STAGING}"
    _symlink(BARE_STAGING, stage / "on_target/bare_staging")
    for unit in X.BASE_UNITS:  # both round-1 base stores are locally pre-staged
        src = BARE_STAGING / "corpus_capture" / unit / "pooled.pt"
        assert src.exists(), f"base unit store missing from bare_staging: {src}"
        _symlink(src, stage / "corpus_capture" / unit / "pooled.pt")
    pfx_sample = stage / "on_target/inputs/corpus_sample_pfx.json"
    if not pfx_sample.exists():
        _stage_file(
            stage, f"{X.HF_PREFIX}/on_target/inputs/corpus_sample_pfx.json", pfx_sample, REV3
        )


def ensure_generic_arm(stage: Path, arm: str) -> list[Path]:
    """Stage one arm's round-1 pooled.pt + pooled_tf.pt; return the REAPABLE paths."""
    reapable: list[Path] = []
    for tree, fname in (("corpus_capture", "pooled.pt"), ("corpus_capture_tf", "pooled_tf.pt")):
        dest = stage / tree / arm / fname
        local = BARE_STAGING / tree / arm / fname
        if dest.exists():
            if not dest.is_symlink():
                reapable.append(dest)
            continue
        if local.exists():
            _symlink(local, dest)
        else:
            _stage_file(stage, f"{X.HF_PREFIX}/{tree}/{arm}/{fname}", dest, REV1)
            reapable.append(dest)
    return reapable


def ensure_pfx_unit(stage: Path, unit: str, *, tf: bool) -> list[Path]:
    tree, fname = ("corpus_capture_tf", "pooled_tf.pt") if tf else ("corpus_capture", "pooled.pt")
    dest = stage / "on_target" / tree / unit / fname
    if dest.exists():
        return [dest] if not dest.is_symlink() else []
    _stage_file(stage, f"{X.HF_PREFIX}/on_target/{tree}/{unit}/{fname}", dest, REV3)
    return [dest]


def reap(paths: list[Path]) -> None:
    for p in paths:
        if p.exists() and not p.is_symlink():
            _drop_cached_store(p)
            p.unlink()
            logger.info("[stage] reaped %s", p)


# ── store LRU cache (keeps issue1768_fit loaders verbatim while avoiding reloads) ──

_STORE_CACHE: OrderedDict[str, dict] = OrderedDict()
_STORE_CACHE_MAX = 6
_ORIG_LOAD_STORE = F._load_store


def _cached_load_store(path: Path) -> dict:
    key = str(path)
    if key in _STORE_CACHE:
        _STORE_CACHE.move_to_end(key)
        return _STORE_CACHE[key]
    store = _ORIG_LOAD_STORE(Path(path))
    _STORE_CACHE[key] = store
    while len(_STORE_CACHE) > _STORE_CACHE_MAX:
        _STORE_CACHE.popitem(last=False)
    return store


def _drop_cached_store(path: Path) -> None:
    _STORE_CACHE.pop(str(path), None)


F._load_store = _cached_load_store


# ── shared-X multi-target val-selected primal ridge (equivalence-pinned) ─────


def fit_maps_shared_x(
    xd: np.ndarray,
    ymap: dict[str, np.ndarray],
    tr: np.ndarray,
    val: np.ndarray,
    te: np.ndarray,
    *,
    allow_underdetermined: bool = False,
) -> dict[str, tuple[np.ndarray, dict, dict]]:
    """K val-selected primal ridge fits sharing ONE factorization of X.

    Equivalent to K sequential `issue1768_fit._fit_map` calls (same
    `_ridge_factorize` standardizer/Gram/eigh, same val-lambda selection loop
    order, same `np.isclose` grid-edge rule + lo/hi/n extension). The lambda
    scan runs in eigenspace: `val_std @ U` once, then per lambda
    `(val_u) @ (UtXtY_block / (s+lam))` — an associativity reorder of
    `_ridge_predict_one`'s `U @ (UtXtY/(s+lam))` then `val_std @ W`.
    Pinned identical by `--phase pilot`. Returns {name: (pred_te, meta, payload)}.
    """
    d = xd.shape[1]
    tr, val, te = np.asarray(tr), np.asarray(val), np.asarray(te)
    if not allow_underdetermined:
        assert len(tr) > d, f"n_train={len(tr)} <= d={d} — under-determined regime refused (#1701)"
    names = list(ymap)
    ycat = np.concatenate([np.asarray(ymap[k]) for k in names], axis=1)
    fac = n1m._ridge_factorize(xd, ycat, tr, DEV, n1m.RIDGE_BLOCK)
    u, s_eig, utxty = fac["U"], fac["s_eig"], fac["UtXtY"]
    xmu, xsd, ymu = fac["xmu"], fac["xsd"], fac["ymu"]

    def _std_u(idx: np.ndarray) -> torch.Tensor:
        xn = (torch.as_tensor(xd[idx], dtype=torch.float64, device=DEV) - xmu) / xsd
        return xn @ u

    val_u, te_u = _std_u(val), _std_u(te)
    out: dict[str, tuple[np.ndarray, dict, dict]] = {}
    col = 0
    for name in names:
        dout = ymap[name].shape[1]
        sl = slice(col, col + dout)
        col += dout
        ut_sl, ymu_sl = utxty[:, sl], ymu[sl]
        y_val = np.asarray(ymap[name])[val]
        lo, hi, n = -3.0, 8.0, 23
        edge: str | None = None
        best_lam, best_vr2 = 1e-3, -np.inf
        for _ext in range(4):
            grid = F.lambda_grid(lo, hi, n)
            best_lam, best_vr2 = float(grid[0]), -np.inf
            for lam in grid:
                pred_val = (val_u @ (ut_sl / (s_eig + float(lam))[:, None]) + ymu_sl).cpu().numpy()
                vr2 = n1m.PR._pooled_r2(pred_val, y_val)
                if np.isfinite(vr2) and vr2 > best_vr2:
                    best_vr2, best_lam = vr2, float(lam)
            edge = None
            if np.isclose(best_lam, float(grid[0])):
                edge = "low"
            elif np.isclose(best_lam, float(grid[-1])):
                edge = "high"
            if edge is None:
                break
            if edge == "low":
                lo -= 1.0
            else:
                hi += 1.0
            n += 2
            logger.info(
                "[fit] %s lambda grid edge %s — extending to [1e%s, 1e%s]", name, edge, lo, hi
            )
        w = u @ (ut_sl / (s_eig + best_lam)[:, None])
        pred_te = (te_u @ (ut_sl / (s_eig + best_lam)[:, None]) + ymu_sl).cpu().numpy()
        payload = {
            "kind": "ridge",
            "selected_lambda": best_lam,
            "xmu": xmu.detach().cpu().to(torch.float32),
            "xsd": xsd.detach().cpu().to(torch.float32),
            "ymu": ymu_sl.detach().cpu().to(torch.float32),
            "W": w.detach().cpu().to(torch.float32),
        }
        meta = {
            "n_train": int(len(tr)),
            "selection": "val-lambda (primal, streaming)",
            "selected_lambda": best_lam,
            "val_r2_at_selected": float(best_vr2),
            "lambda_grid_edge": edge,
            "ridge_block": int(n1m.RIDGE_BLOCK),
            "lambda_grid": [lo, hi, n],
            "fit_impl": "sharedX-eigspace-v1 (pilot-pinned == fit_ridge_with_weights)",
        }
        out[name] = (pred_te, meta, payload)
    del fac, ycat  # closure-captured names (u, val_u, ...) release at return
    gc.collect()
    return out


# ── reads ─────────────────────────────────────────────────────────────────────


def _light_reads(pred: np.ndarray, y_te: np.ndarray) -> dict:
    """R2 + cosine + the standing kNN retrieval read (no per-rung bootstrap CIs —
    the cross-arm distribution is the uncertainty read; per-arm points are plotted)."""
    n_pool = y_te.shape[0]
    ks = tuple(k for k in (1, 10) if k <= n_pool) or (1,)
    return {
        "heldout_r2": F._pooled_r2(pred, y_te),
        "mean_cos": F._mean_cos(pred, y_te),
        "knn_euclidean": knn_retrieval(pred, y_te, ks=ks, metric="euclidean"),
        "knn_cosine": knn_retrieval(pred, y_te, ks=ks, metric="cosine"),
    }


def _identity_reads(c_tr, y_tr, c_te, y_te) -> dict:
    pred = identity_bias_predict(
        np.asarray(c_tr, dtype=np.float64),
        np.asarray(y_tr, dtype=np.float64),
        np.asarray(c_te, dtype=np.float64),
    )
    return {"applicable": True, **_light_reads(pred, np.asarray(y_te, dtype=np.float64))}


def _rung_block(
    m0_payload: dict, mp_payload: dict, cell: dict, te: np.ndarray, target_key: str
) -> dict:
    y_te = np.asarray(cell[target_key], dtype=np.float64)[te]
    preds = {
        "r1_M0_c0": n1m.apply_map(m0_payload, cell["C0"][te], DEV),
        "r2_M0_cplus": n1m.apply_map(m0_payload, cell["Cplus"][te], DEV),
        "r3_Mplus_c0": n1m.apply_map(mp_payload, cell["C0"][te], DEV),
        "r4_Mplus_cplus": n1m.apply_map(mp_payload, cell["Cplus"][te], DEV),
    }
    rungs = {k: _light_reads(p, y_te) for k, p in preds.items()}
    r = {k: rungs[k]["heldout_r2"] for k in RUNGS}
    span = r["r4_Mplus_cplus"] - r["r1_M0_c0"]
    safe = abs(span) > 1e-9
    return {
        "rungs": rungs,
        "floor_r2": r["r1_M0_c0"],
        "ceiling_r2": r["r4_Mplus_cplus"],
        "gap_close_context_only": (r["r2_M0_cplus"] - r["r1_M0_c0"]) / span if safe else None,
        "gap_close_map_only": (r["r3_Mplus_c0"] - r["r1_M0_c0"]) / span if safe else None,
        "interaction_r2": r["r4_Mplus_cplus"] + r["r1_M0_c0"] - r["r2_M0_cplus"] - r["r3_Mplus_c0"],
    }


# ── M0 fits (shared per base unit x layer x context set) ─────────────────────


def _m0_cell(stage: Path, ctxset: str, unit: str, layer: int) -> dict:
    if ctxset == "generic":
        store = F._load_store(stage / "corpus_capture" / unit / "pooled.pt")
        c, _ = F._rows_from_store(store, "context", layer)
        v, _ = F._rows_from_store(store, "response", layer)
        sample = X.load_corpus_sample(stage)
        qidx = np.asarray(store["row_question_idx"])
        n_tr, n_val = sample["n_train"], sample["n_val"]
        split = np.where(qidx < n_tr, "train", np.where(qidx < n_tr + n_val, "val", "test"))
    elif ctxset == "bare_n":
        store = F._load_store(F._bare_store_path(stage, "corpus_capture", unit, "pooled.pt"))
        sample = X.load_pfx_sample(stage)
        pfx_by_src = {int(r["src_qidx"]): j for j, r in enumerate(sample["rows"])}
        keep = [i for i, q in enumerate(store["row_question_idx"]) if int(q) in pfx_by_src]
        qidx = np.asarray([pfx_by_src[int(store["row_question_idx"][i])] for i in keep])
        c, _ = F._rows_from_store(store, "context", layer)
        v, _ = F._rows_from_store(store, "response", layer)
        c, v = c[np.asarray(keep)], v[np.asarray(keep)]
        split = F._pfx_split_from_qidx(qidx, sample)
    elif ctxset == "on_target":
        store = F._load_store(stage / "on_target" / "corpus_capture" / unit / "pooled.pt")
        sample = X.load_pfx_sample(stage)
        qidx = np.asarray(store["row_question_idx"])
        c, _ = F._rows_from_store(store, "context", layer)
        v, _ = F._rows_from_store(store, "response", layer)
        split = F._pfx_split_from_qidx(qidx, sample)
    else:
        raise ValueError(ctxset)
    return {"C": c, "V": v, "split": split}


def _m0_key(ctxset: str, unit: str, layer: int) -> str:
    return f"{ctxset}__{unit}__L{layer}"


def get_m0(stage: Path, out: Path, ctxset: str, unit: str, layer: int) -> dict:
    """Fit (or load) the shared M0 for (ctxset, base unit, layer); returns the payload."""
    key = _m0_key(ctxset, unit, layer)
    cache = stage / "m0_cache" / f"{key}.pt"
    rec_path = out / "m0_fits" / f"{key}.json"
    if cache.exists() and rec_path.exists():
        # self-produced sha-implicit bundle; metadata carries tensors -> weights_only=False
        return torch.load(cache, map_location="cpu", weights_only=False)["payload"]
    t0 = time.time()
    cell = _m0_cell(stage, ctxset, unit, layer)
    tr, val, te = F._split_idx(cell["split"])
    res = fit_maps_shared_x(
        cell["C"], {"M0": cell["V"]}, tr, val, te, allow_underdetermined=ctxset != "generic"
    )
    pred_te, meta, payload = res["M0"]
    y_te = np.asarray(cell["V"], dtype=np.float64)[te]
    rec = {
        "m0_key": key,
        "context_set": ctxset,
        "base_unit": unit,
        "layer": layer,
        "n_train": int(len(tr)),
        "n_val": int(len(val)),
        "n_test": int(len(te)),
        "d": int(cell["C"].shape[1]),
        "underdetermined_n_lt_d": bool(len(tr) < cell["C"].shape[1]),
        "meta": meta,
        "own_heldout": _light_reads(pred_te, y_te),
        "identity_bias": _identity_reads(
            cell["C"][tr], cell["V"][tr], cell["C"][te], cell["V"][te]
        ),
        "elapsed_s": round(time.time() - t0, 1),
        **F._meta(),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"payload": payload, "meta": meta}, cache)
    F._atomic_json(rec_path, rec)
    logger.info(
        "[m0] fit %s in %.1fs (lambda=%.3g, own R2=%.4f)",
        key,
        rec["elapsed_s"],
        meta["selected_lambda"],
        rec["own_heldout"]["heldout_r2"],
    )
    del cell
    gc.collect()
    return payload


# ── per-cell driver ───────────────────────────────────────────────────────────


def percell_path(out: Path, ctxset: str, arm: str, layer: int, mode: str = "ladder") -> Path:
    sub = "percell" if mode == "ladder" else "spectra"
    return out / sub / f"{ctxset}__{arm}__L{layer}.json"


# ── data-space spectrum of the map update (scope-extension addendum) ─────────

DEFAULT_CONVENTION = {
    "map_pair": "M0 (base contexts -> base answers) vs Mplus (trained contexts -> trained "
    "answers), each model's OWN on-policy greedy text",
    "context_set": "generic real-user corpus (LMSYS + WildChat bare prompts, 16,400 rows, "
    "n_train=15,000)",
    "tree": "onpolicy",
    "pooling": "span-mean (round-1 convention)",
    "headline_panel": "generic__onpolicy",
    "limitation": "Under the on-policy default, Mplus is fit on data where BOTH the contexts "
    "AND the answer text differ from base, so a 'the map changed' reading is a change in the "
    "model's DEPLOYED input->output relation, not a weights-isolated claim. The matched-text "
    "tree (trained model teacher-forced on the BASE model's responses) is the control that "
    "separates weights-carried change from text-mediated change.",
}


def _delta_spectrum(a: np.ndarray, q: int = 64) -> dict:
    """Singular spectrum of the realized predicted-answer shift Delta_pred (n_test x d).

    Exact-trace pattern reused from `issue1768_operator_kv._spectrum`:
    sum(s^2) == ||A||_F^2 and sum(s^4) == ||A^T A||_F^2 are exact traces, so the
    participation ratio PR = (sum s^2)^2 / sum s^4 and all mass FRACTIONS are exact;
    only the top-q values ride a thin `svd_lowrank`. Direction counts to 90/95/99%
    Frobenius mass are exact when reached within q, else reported as None + gt_q flag.
    """
    at = torch.as_tensor(np.asarray(a), dtype=torch.float64)
    fro2 = float((at * at).sum())
    g = at.T @ at
    sum_s4 = float((g * g).sum())
    del g
    _, s_top, _ = torch.svd_lowrank(at, q=min(q, min(at.shape) - 1), niter=8)
    s2 = s_top.numpy().astype(np.float64) ** 2
    pr = (fro2**2) / sum_s4 if sum_s4 > 0 else float("nan")
    cum = np.cumsum(s2) / fro2 if fro2 > 0 else np.zeros_like(s2)

    def _n_to(frac: float) -> int | None:
        hit = np.nonzero(cum >= frac)[0]
        return int(hit[0]) + 1 if hit.size else None

    return {
        "participation_ratio_exact": pr,
        "top1_share": float(s2[0] / fro2) if fro2 > 0 else float("nan"),
        "top5_share": float(s2[:5].sum() / fro2) if fro2 > 0 else float("nan"),
        "n_dirs_90pct": _n_to(0.90),
        "n_dirs_95pct": _n_to(0.95),
        "n_dirs_99pct": _n_to(0.99),
        "counts_gt_q": bool(cum[-1] < 0.99) if cum.size else True,
        "fro2_exact": fro2,
        "sum_s4_exact": sum_s4,
        "svals_top": [float(v) for v in np.sqrt(s2)],
        "svd_q": int(s2.size),
    }


def run_spectra_cell(
    out: Path,
    ctxset: str,
    arm: str,
    layer: int,
    cell: dict,
    m0_payload: dict,
    m0_key: str,
    *,
    allow_ud: bool,
) -> dict:
    """Data-space spectrum of Delta_pred = M+(c) - M0(c) on held-out test contexts.

    Both trees (Mplus / Mplus_tf vs the shared M0) x both input choices:
    `inputs_c0` (both maps on base contexts — the pure map-update read, the
    rung3-vs-rung1 contrast) and `inputs_cplus` (both maps on fine-tuned
    contexts — the rung4-vs-rung2 contrast).
    """
    dest = percell_path(out, ctxset, arm, layer, mode="spectra")
    if dest.exists():
        return json.loads(dest.read_text())
    t0 = time.time()
    tr, val, te = F._split_idx(cell["split"])
    ymap = {"Mplus": cell["Vplus"], "Mplus_tf": cell["Vplus_tf"]}
    fits = fit_maps_shared_x(cell["Cplus"], ymap, tr, val, te, allow_underdetermined=allow_ud)
    pred_m0 = {
        "c0": n1m.apply_map(m0_payload, cell["C0"][te], DEV),
        "cplus": n1m.apply_map(m0_payload, cell["Cplus"][te], DEV),
    }
    trees = {}
    for tree, (_target_key, map_name) in TREES.items():
        _, meta, payload = fits[map_name]
        block = {"map": map_name, "selected_lambda": meta["selected_lambda"]}
        for inp, x_key in (("inputs_c0", "C0"), ("inputs_cplus", "Cplus")):
            pred_mp = n1m.apply_map(payload, cell[x_key][te], DEV)
            block[inp] = _delta_spectrum(pred_mp - pred_m0[inp.removeprefix("inputs_")])
        trees[tree] = block
    rec = {
        "arm_id": arm,
        "layer": layer,
        "context_set": ctxset,
        "pooling": "span-mean (round-1 convention)",
        "n_test": int(len(te)),
        "n_train": int(len(tr)),
        "d": int(cell["Cplus"].shape[1]),
        "underdetermined_n_lt_d": bool(len(tr) < cell["Cplus"].shape[1]),
        "m0_key": m0_key,
        "trees": trees,
        "elapsed_s": round(time.time() - t0, 1),
        **F._meta(),
    }
    F._atomic_json(dest, rec)
    return rec


def run_cell(
    out: Path,
    ctxset: str,
    arm: str,
    layer: int,
    cell: dict,
    m0_payload: dict,
    m0_key: str,
    *,
    allow_ud: bool,
) -> dict:
    dest = percell_path(out, ctxset, arm, layer)
    if dest.exists():
        return json.loads(dest.read_text())
    t0 = time.time()
    tr, val, te = F._split_idx(cell["split"])
    ymap = {"Mplus": cell["Vplus"], "Mplus_tf": cell["Vplus_tf"]}
    fits = fit_maps_shared_x(cell["Cplus"], ymap, tr, val, te, allow_underdetermined=allow_ud)
    trees = {}
    for tree, (target_key, map_name) in TREES.items():
        _, meta, payload = fits[map_name]
        trees[tree] = {
            "target": target_key,
            "map": map_name,
            "map_meta": meta,
            **_rung_block(m0_payload, payload, cell, te, target_key),
            "identity_bias_mplus": _identity_reads(
                cell["Cplus"][tr], cell[target_key][tr], cell["Cplus"][te], cell[target_key][te]
            ),
        }
    rec = {
        "arm_id": arm,
        "layer": layer,
        "context_set": ctxset,
        "pooling": "span-mean (round-1 convention)",
        "n_rows": int(len(cell["sha"])),
        "n_train": int(len(tr)),
        "n_val": int(len(val)),
        "n_test": int(len(te)),
        "d": int(cell["Cplus"].shape[1]),
        "underdetermined_n_lt_d": bool(len(tr) < cell["Cplus"].shape[1]),
        "m0_key": m0_key,
        "trees": trees,
        "elapsed_s": round(time.time() - t0, 1),
        **F._meta(),
    }
    F._atomic_json(dest, rec)
    return rec


def _record_store_spans(stage: Path, out: Path, arm: str, ctxset: str) -> None:
    """Runtime verification: which spans each tree's store actually carries (c+ identity)."""
    path = out / "verification_runtime.json"
    rec = json.loads(path.read_text()) if path.exists() else {}
    key = f"{ctxset}__{arm}"
    if key in rec:
        return
    if ctxset == "generic":
        plus = F._load_store(stage / "corpus_capture" / arm / "pooled.pt")
        tf = F._load_store(stage / "corpus_capture_tf" / arm / "pooled_tf.pt")
    else:
        unit = X.pfx_trained_unit(arm, "own")
        plus = F._load_store(stage / "on_target" / "corpus_capture" / unit / "pooled.pt")
        tf = F._load_store(stage / "on_target" / "corpus_capture_tf" / unit / "pooled_tf.pt")
    rec[key] = {
        "onpolicy_store_spans": sorted(plus["arms"].keys()),
        "tf_store_spans": sorted(tf["arms"].keys()),
        "cplus_single_stored_copy": sorted(tf["arms"].keys()) == ["response"],
    }
    F._atomic_json(path, rec)


# ── phases ────────────────────────────────────────────────────────────────────


def _pfx_own_base_units() -> list[str]:
    return sorted({X.pfx_base_unit(a, "own") for a in X.PFX_ARMS})


def phase_battery(
    stage: Path,
    out: Path,
    layers: tuple[int, ...],
    arms_filter: list[str] | None,
    mode: str = "ladder",
) -> None:
    """mode="ladder": the four-rung reads; mode="spectra": the Delta_pred data-space
    spectra (scope-extension addendum) — same staging/iteration/resume machinery,
    separate per-cell checkpoint dir."""
    assert mode in ("ladder", "spectra"), mode
    cell_fn = run_cell if mode == "ladder" else run_spectra_cell
    setup_stage(stage)
    (out / ("percell" if mode == "ladder" else "spectra")).mkdir(parents=True, exist_ok=True)
    (out / "m0_fits").mkdir(parents=True, exist_ok=True)

    # ---- trained-in (on_target, own-prefix; n=3,000 < d) ----
    pfx_arms = [a for a in X.PFX_ARMS if not arms_filter or a in arms_filter]
    total_units = len(pfx_arms) + len([a for a in _arms72() if not arms_filter or a in arms_filter])
    unit_i = 0
    base_dl: list[Path] = []
    if pfx_arms:
        for unit in _pfx_own_base_units():
            base_dl += ensure_pfx_unit(stage, unit, tf=False)
    for arm in pfx_arms:
        unit_i += 1
        if all(percell_path(out, "on_target", arm, ly, mode=mode).exists() for ly in layers):
            logger.info(
                "[%s] unit %d/%d on_target %s already complete", mode, unit_i, total_units, arm
            )
            continue
        t0 = time.time()
        unit = X.pfx_trained_unit(arm, "own")
        dl = ensure_pfx_unit(stage, unit, tf=False) + ensure_pfx_unit(stage, unit, tf=True)
        _record_store_spans(stage, out, arm, "on_target")
        base_unit = X.pfx_base_unit(arm, "own")
        for layer in layers:
            if percell_path(out, "on_target", arm, layer, mode=mode).exists():
                continue
            m0p = get_m0(stage, out, "on_target", base_unit, layer)
            cell = F.load_pfx_cell(arm, "own", layer, stage)
            cell_fn(
                out,
                "on_target",
                arm,
                layer,
                cell,
                m0p,
                _m0_key("on_target", base_unit, layer),
                allow_ud=True,
            )
            del cell
            gc.collect()
        reap(dl)
        logger.info(
            "[%s] unit %d/%d on_target %s elapsed=%.0fs",
            mode,
            unit_i,
            total_units,
            arm,
            time.time() - t0,
        )
    if base_dl:
        reap(base_dl)

    # ---- generic (full n=15,000) + n-matched comparator (bare_n, n=3,000) ----
    arms = [a for a in _arms72() if not arms_filter or a in arms_filter]
    pool = ThreadPoolExecutor(max_workers=1)

    def _needs(arm: str) -> bool:
        need = [percell_path(out, "generic", arm, ly, mode=mode) for ly in layers]
        if arm in X.PFX_ARMS:
            need += [percell_path(out, "bare_n", arm, ly, mode=mode) for ly in layers]
        return not all(p.exists() for p in need)

    fut = None
    if arms and _needs(arms[0]):
        fut = pool.submit(ensure_generic_arm, stage, arms[0])
    for i, arm in enumerate(arms):
        unit_i += 1
        if not _needs(arm):
            logger.info(
                "[%s] unit %d/%d generic %s already complete", mode, unit_i, total_units, arm
            )
            if fut is not None and i + 1 < len(arms) and _needs(arms[i + 1]):
                pass  # keep the pending prefetch
            continue
        t0 = time.time()
        dl = fut.result() if fut is not None else ensure_generic_arm(stage, arm)
        fut = None
        nxt = next((a for a in arms[i + 1 :] if _needs(a)), None)
        if nxt is not None:
            fut = pool.submit(ensure_generic_arm, stage, nxt)
        _record_store_spans(stage, out, arm, "generic")
        base_unit = X.base_unit_for(arm)
        for layer in layers:
            if not percell_path(out, "generic", arm, layer, mode=mode).exists():
                m0p = get_m0(stage, out, "generic", base_unit, layer)
                cell = F.load_corpus_cell(arm, layer, stage)
                cell_fn(
                    out,
                    "generic",
                    arm,
                    layer,
                    cell,
                    m0p,
                    _m0_key("generic", base_unit, layer),
                    allow_ud=False,
                )
                del cell
                gc.collect()
            if (
                arm in X.PFX_ARMS
                and not percell_path(out, "bare_n", arm, layer, mode=mode).exists()
            ):
                m0p = get_m0(stage, out, "bare_n", base_unit, layer)
                cell = F.load_bare_n_cell(arm, layer, stage)
                cell_fn(
                    out,
                    "bare_n",
                    arm,
                    layer,
                    cell,
                    m0p,
                    _m0_key("bare_n", base_unit, layer),
                    allow_ud=True,
                )
                del cell
                gc.collect()
        reap(dl)
        logger.info(
            "[ladder] unit %d/%d generic %s elapsed=%.0fs",
            unit_i,
            total_units,
            arm,
            time.time() - t0,
        )
    pool.shutdown(wait=False)


def phase_pilot(stage: Path, out: Path) -> None:
    """MEASURED 1-cell pilot through the production entrypoint + equivalence pins."""
    setup_stage(stage)
    out.mkdir(parents=True, exist_ok=True)
    rec: dict = {"pilot_arm": X.PILOT_ARM, "layer": PILOT_LAYER, **F._meta()}

    # (a) synthetic equivalence: shared-X wrapper == two verbatim _fit_map calls
    rng = np.random.default_rng(1768)
    xs = rng.standard_normal((240, 16))
    w_true = rng.standard_normal((16, 8))
    ya = xs @ w_true + 0.1 * rng.standard_normal((240, 8))
    yb = xs @ (w_true * 0.5) + 0.1 * rng.standard_normal((240, 8))
    tr_s, val_s, te_s = np.arange(160), np.arange(160, 200), np.arange(200, 240)
    shared = fit_maps_shared_x(xs, {"a": ya, "b": yb}, tr_s, val_s, te_s)
    syn = {}
    for name, y in (("a", ya), ("b", yb)):
        pred_v, meta_v, _ = F._fit_map(xs, y, tr_s, val_s, te_s, DEV)
        pred_s, meta_s, _ = shared[name]
        syn[name] = {
            "lambda_match": bool(np.isclose(meta_v["selected_lambda"], meta_s["selected_lambda"])),
            "max_abs_pred_diff": float(np.max(np.abs(pred_v - pred_s))),
        }
        assert syn[name]["lambda_match"], (name, meta_v, meta_s)
        assert syn[name]["max_abs_pred_diff"] < 1e-8, syn[name]
    rec["synthetic_equivalence"] = syn

    # (b) real pilot cell at production shape, timed
    arm, layer = X.PILOT_ARM, PILOT_LAYER
    dl = ensure_generic_arm(stage, arm)
    t0 = time.time()
    cell = F.load_corpus_cell(arm, layer, stage)
    rec["t_load_s"] = round(time.time() - t0, 1)
    tr, val, te = F._split_idx(cell["split"])

    t1 = time.time()
    pred_v, meta_v, payload_v = F._fit_map(cell["Cplus"], cell["Vplus"], tr, val, te, DEV)
    rec["t_verbatim_single_fit_s"] = round(time.time() - t1, 1)

    t2 = time.time()
    fits = fit_maps_shared_x(
        cell["Cplus"], {"Mplus": cell["Vplus"], "Mplus_tf": cell["Vplus_tf"]}, tr, val, te
    )
    rec["t_shared_pair_fit_s"] = round(time.time() - t2, 1)

    pred_s, meta_s, payload_s = fits["Mplus"]
    r2_v, r2_s = F._pooled_r2(pred_v, cell["Vplus"][te]), F._pooled_r2(pred_s, cell["Vplus"][te])
    rec["real_equivalence"] = {
        "lambda_verbatim": meta_v["selected_lambda"],
        "lambda_shared": meta_s["selected_lambda"],
        "lambda_match": bool(np.isclose(meta_v["selected_lambda"], meta_s["selected_lambda"])),
        "r2_verbatim": r2_v,
        "r2_shared": r2_s,
        "abs_r2_diff": abs(r2_v - r2_s),
        "max_abs_pred_diff": float(np.max(np.abs(pred_v - pred_s))),
    }
    assert rec["real_equivalence"]["lambda_match"], rec["real_equivalence"]
    assert rec["real_equivalence"]["abs_r2_diff"] < 1e-6, rec["real_equivalence"]

    # apply_map (fp32-payload production path) vs exact fp64 test predictions
    pred_apply = n1m.apply_map(payload_s, cell["Cplus"][te], DEV)
    rec["apply_map_r2_diff"] = abs(F._pooled_r2(pred_apply, cell["Vplus"][te]) - r2_s)
    assert rec["apply_map_r2_diff"] < 1e-4, rec["apply_map_r2_diff"]

    # committed round-1 cross-check (rung 4 == the committed heldout_r2, both trees)
    committed = json.loads((COMMITTED_FITS / f"{arm}_L{layer}.json").read_text())
    r2_tf = F._pooled_r2(fits["Mplus_tf"][0], cell["Vplus_tf"][te])
    rec["committed_crosscheck"] = {
        "mplus_committed": committed["fits"]["Mplus"]["heldout_r2"],
        "mplus_refit": r2_s,
        "mplus_abs_diff": abs(committed["fits"]["Mplus"]["heldout_r2"] - r2_s),
        "mplus_tf_committed": committed["fits"]["Mplus_tf"]["heldout_r2"],
        "mplus_tf_refit": r2_tf,
        "mplus_tf_abs_diff": abs(committed["fits"]["Mplus_tf"]["heldout_r2"] - r2_tf),
    }

    # M0 fit timing (shared per base unit x layer)
    t3 = time.time()
    m0p = get_m0(stage, out, "generic", X.base_unit_for(arm), layer)
    rec["t_m0_fit_s"] = round(time.time() - t3, 1)

    t4 = time.time()
    rung = _rung_block(m0p, payload_s, cell, te, "Vplus")
    rec["t_rung_block_s"] = round(time.time() - t4, 1)
    rec["pilot_rungs_onpolicy_r2"] = {k: rung["rungs"][k]["heldout_r2"] for k in RUNGS}

    n_big = 72 * 3
    n_small = (12 + 12) * 3
    per_big = rec["t_shared_pair_fit_s"] + 2 * rec["t_rung_block_s"] + rec["t_load_s"] / 3
    rec["extrapolation"] = {
        "per_big_cell_s": round(per_big, 1),
        "n_big_cells": n_big,
        "n_small_cells": n_small,
        "assumed_small_cell_frac": 0.6,
        "projected_battery_h": round((n_big * per_big + n_small * per_big * 0.6) / 3600, 2),
    }
    F._atomic_json(out / "pilot.json", rec)
    logger.info("[pilot] %s", json.dumps(rec["extrapolation"]))
    del cell
    gc.collect()
    reap(dl)


def _agg(vals: list[float]) -> dict:
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
    }


def phase_summary(stage: Path, out: Path, layers: tuple[int, ...]) -> None:
    cells = [json.loads(p.read_text()) for p in sorted((out / "percell").glob("*.json"))]
    assert cells, "no per-cell records — run --phase battery first"
    summary: dict = {
        "issue": 1768,
        "round": "result2_ladder",
        "default_convention": DEFAULT_CONVENTION,
        "pooling": "span-mean (round-1 convention)",
        "lasttoken_second_pooling": (
            "skipped — not cheap: the lasttoken_ctx tree adds ~26 GB of downloads and ~300 "
            "further n=15,000 fits on a shared VM at load ~40; span-mean is the primary "
            "convention and the committed round-1/round-3 fits it cross-checks against"
        ),
        "rung_definitions": {
            "r1_M0_c0": "base map on base contexts (nothing updated — floor)",
            "r2_M0_cplus": "base map on fine-tuned contexts (context updated only)",
            "r3_Mplus_c0": "fine-tuned map on base contexts (map updated only)",
            "r4_Mplus_cplus": "fine-tuned map on fine-tuned contexts (ceiling)",
        },
        "trees": {
            "onpolicy": "target = trained model's answer vectors on its OWN text (conflated)",
            "matched": "target = trained model teacher-forced on the BASE model's responses "
            "(weights-carried only)",
        },
        **F._meta(),
    }
    grid: dict = {}
    for ctxset in CTX_SETS:
        for tree in TREES:
            for layer in layers:
                sub = [c for c in cells if c["context_set"] == ctxset and c["layer"] == layer]
                if not sub:
                    continue
                block: dict = {
                    "n_arms": len(sub),
                    "n_train": sub[0]["n_train"],
                    "d": sub[0]["d"],
                    "underdetermined_n_lt_d": sub[0]["underdetermined_n_lt_d"],
                    "regime": (
                        "regularization-limited (n_train < d)"
                        if sub[0]["underdetermined_n_lt_d"]
                        else "well-posed"
                    ),
                }
                for rung in RUNGS:
                    block[rung] = _agg([c["trees"][tree]["rungs"][rung]["heldout_r2"] for c in sub])
                for k in ("gap_close_context_only", "gap_close_map_only", "interaction_r2"):
                    block[k] = _agg([c["trees"][tree][k] for c in sub])
                block["selected_lambda_median"] = float(
                    np.median([c["trees"][tree]["map_meta"]["selected_lambda"] for c in sub])
                )
                block["per_arm"] = {
                    c["arm_id"]: {
                        **{r: c["trees"][tree]["rungs"][r]["heldout_r2"] for r in RUNGS},
                        "gap_close_context_only": c["trees"][tree]["gap_close_context_only"],
                        "gap_close_map_only": c["trees"][tree]["gap_close_map_only"],
                        "interaction_r2": c["trees"][tree]["interaction_r2"],
                        "selected_lambda": c["trees"][tree]["map_meta"]["selected_lambda"],
                    }
                    for c in sub
                }
                grid[f"{ctxset}__{tree}__L{layer}"] = block
    summary["grid"] = grid

    # data-space spectrum of the map update (scope-extension addendum): why — the raw
    # 3584x3584 operator-update SVD (map_augmentation/operator_kv) de-standardizes
    # A = W / xsd, AMPLIFYING input dims the data varies along least, so its Frobenius
    # mass tilts toward directions the context distribution rarely excites (raw operator
    # PR ~80 at L19). Delta_pred = M+(c) - M0(c) on held-out test contexts measures how
    # many directions the update actually moves predicted answers along, over the real
    # context distribution — no basis tilt.
    spec_cells = [json.loads(p.read_text()) for p in sorted((out / "spectra").glob("*.json"))]
    if spec_cells:
        ds: dict = {
            "what": "singular spectrum of Delta_pred = M+(c) - M0(c) on held-out test "
            "contexts (n_test x d), per tree x input-choice (inputs_c0 = both maps on "
            "base contexts, the pure map-update read; inputs_cplus = both maps on "
            "fine-tuned contexts)",
            "why": "the raw 3584x3584 operator-update SVD (map_augmentation/operator_kv) "
            "de-standardizes A = W / xsd, amplifying input dimensions the context "
            "distribution varies along LEAST, so its Frobenius mass — and the measured "
            "rank (raw operator PR ~80 at layer 19) — is tilted toward directions the "
            "data rarely excites. Delta_pred measures how many directions the update "
            "actually moves predicted answers along, over the real context distribution, "
            "with no basis tilt — the functional (data-space) rank of the map update.",
            "per_cell": {},
            "aggregates": {},
        }
        for c in spec_cells:
            key = f"{c['context_set']}__{c['arm_id']}__L{c['layer']}"
            ds["per_cell"][key] = {
                tree: {
                    inp: {
                        k: c["trees"][tree][inp][k]
                        for k in (
                            "participation_ratio_exact",
                            "top1_share",
                            "top5_share",
                            "n_dirs_90pct",
                            "n_dirs_95pct",
                            "n_dirs_99pct",
                            "counts_gt_q",
                        )
                    }
                    for inp in ("inputs_c0", "inputs_cplus")
                }
                for tree in TREES
            }
        for ctxset in CTX_SETS:
            for tree in TREES:
                for layer in layers:
                    sub = [
                        c for c in spec_cells if c["context_set"] == ctxset and c["layer"] == layer
                    ]
                    if not sub:
                        continue
                    blk = {}
                    for inp in ("inputs_c0", "inputs_cplus"):
                        blk[inp] = {
                            "participation_ratio": _agg(
                                [c["trees"][tree][inp]["participation_ratio_exact"] for c in sub]
                            ),
                            "top1_share": _agg([c["trees"][tree][inp]["top1_share"] for c in sub]),
                            "top5_share": _agg([c["trees"][tree][inp]["top5_share"] for c in sub]),
                            "n_dirs_90pct": _agg(
                                [c["trees"][tree][inp]["n_dirs_90pct"] for c in sub]
                            ),
                            "n_dirs_95pct": _agg(
                                [c["trees"][tree][inp]["n_dirs_95pct"] for c in sub]
                            ),
                        }
                    ds["aggregates"][f"{ctxset}__{tree}__L{layer}"] = blk
        summary["data_space_update_rank"] = ds

    # committed round-1 rung-4 cross-check (estimator identity, both trees)
    diffs = {"onpolicy": [], "matched": []}
    for c in cells:
        if c["context_set"] != "generic":
            continue
        committed_path = COMMITTED_FITS / f"{c['arm_id']}_L{c['layer']}.json"
        if not committed_path.exists():
            continue
        committed = json.loads(committed_path.read_text())
        diffs["onpolicy"].append(
            abs(
                committed["fits"]["Mplus"]["heldout_r2"]
                - c["trees"]["onpolicy"]["rungs"]["r4_Mplus_cplus"]["heldout_r2"]
            )
        )
        diffs["matched"].append(
            abs(
                committed["fits"]["Mplus_tf"]["heldout_r2"]
                - c["trees"]["matched"]["rungs"]["r4_Mplus_cplus"]["heldout_r2"]
            )
        )
    summary["committed_rung4_crosscheck_max_abs_diff"] = {
        k: (float(np.max(v)) if v else None) for k, v in diffs.items()
    }

    # verification block: runtime store spans + HF tree listing at the pinned revisions
    runtime = out / "verification_runtime.json"
    ver: dict = {"runtime_store_spans": json.loads(runtime.read_text()) if runtime.exists() else {}}
    try:
        from huggingface_hub import HfApi

        from explore_persona_space.orchestrate import hub

        api = HfApi()

        def _tree_units(prefix: str, revision: str) -> list[str]:
            # retried + materialized INSIDE the retry (lazy-generator gotcha, #920/#997)
            items = hub.retry_transient(
                lambda: list(
                    # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient right above
                    api.list_repo_tree(
                        X.HF_DATA_REPO,
                        path_in_repo=prefix,
                        repo_type="dataset",
                        revision=revision,
                        recursive=False,
                    )
                ),
                what=f"list_repo_tree {prefix}",
            )
            return [i.path.split("/")[-1] for i in items]

        r1_tf = _tree_units(f"{X.HF_PREFIX}/corpus_capture_tf", REV1)
        ot_tf = _tree_units(f"{X.HF_PREFIX}/on_target/corpus_capture_tf", REV3)
        ver["hf_round1_tf_units"] = len(r1_tf)
        ver["hf_round1_tf_has_base_units"] = any(u.startswith("base") for u in r1_tf)
        ver["hf_on_target_tf_own_units"] = sorted(u for u in ot_tf if u.endswith("@own"))
        ver["m0_serves_both_trees"] = (
            "confirmed by construction: the round-1 tf tree stores NO base unit — the base "
            "model's on-policy greedy text IS the matched text it would be teacher-forced "
            "onto, so a base tf capture would recompute the base on-policy store"
        )
        ver["cplus_identity_across_trees"] = (
            "c+ is a SINGLE STORED COPY: tf stores carry spans=['response'] only (runtime "
            "check above), and both ladders read Cplus from the on-policy store. Prompt-side "
            "activations are identical across trees by causal masking (same model, same "
            "prompt tokens), so the shared copy is exact by construction, not approximate."
        )
    except Exception as e:  # network probe is best-effort; runtime spans are the hard check
        ver["hf_listing_error"] = repr(e)
    summary["verification"] = ver
    F._atomic_json(out / "summary.json", summary)
    logger.info("[summary] wrote %s (%d cells)", out / "summary.json", len(cells))


# ── figures ───────────────────────────────────────────────────────────────────

CTX_LABEL = {
    "generic": "generic contexts (n=15,000 train)",
    "bare_n": "generic, n-matched (n=3,000 train < d=3,584 — regularization-limited)",
    "on_target": "fine-tuned-into contexts (n=3,000 train < d=3,584 — regularization-limited)",
}
TREE_LABEL = {
    "onpolicy": "on-policy text (conflated: text + representation shift)",
    "matched": "matched text (weights-carried shift only)",
}
RUNG_LABEL = {
    "r1_M0_c0": "base map,\nbase contexts\n(floor)",
    "r2_M0_cplus": "base map,\nfine-tuned contexts\n(context only)",
    "r3_Mplus_c0": "fine-tuned map,\nbase contexts\n(map only)",
    "r4_Mplus_cplus": "fine-tuned map,\nfine-tuned contexts\n(ceiling)",
}


def _fig_main(summary: dict, layer: int) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette(4)  # one color = one meaning: color is the RUNG, everywhere
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharey="row")
    rng = np.random.default_rng(0)
    for row, tree in enumerate(TREES):
        for col, ctxset in enumerate(CTX_SETS):
            ax = axes[row][col]
            block = summary["grid"].get(f"{ctxset}__{tree}__L{layer}")
            if block is None:
                ax.set_axis_off()
                continue
            means = [block[r]["mean"] for r in RUNGS]
            sems = [
                block[r]["std"] / max(1, np.sqrt(block[r]["n"])) if block[r]["n"] > 1 else 0.0
                for r in RUNGS
            ]
            xs = np.arange(4)
            ax.bar(
                xs,
                means,
                yerr=sems,
                capsize=3,
                color=colors,
                width=0.62,
                zorder=2,
                error_kw={"elinewidth": 1.0, "ecolor": "0.2"},
            )
            for j, r in enumerate(RUNGS):
                pts = [v[r] for v in block["per_arm"].values()]
                jit = rng.uniform(-0.14, 0.14, size=len(pts))
                ax.scatter(xs[j] + jit, pts, s=9, color="0.25", alpha=0.55, zorder=3, linewidths=0)
            ax.axhline(means[0], color=colors[0], lw=1.0, ls="--", alpha=0.8, zorder=1)
            ax.axhline(means[3], color=colors[3], lw=1.0, ls="--", alpha=0.8, zorder=1)
            ax.set_xticks(xs)
            ax.set_xticklabels([RUNG_LABEL[r] for r in RUNGS], fontsize=7.2)
            is_default = tree == "onpolicy" and ctxset == "generic"
            prefix = "DEFAULT map convention\n" if is_default else ""
            ax.set_title(f"{prefix}{TREE_LABEL[tree]}\n{CTX_LABEL[ctxset]}", fontsize=9.5)
            if col == 0:
                ax.set_ylabel("held-out $R^2$ toward the\nfine-tuned answer vectors")
    fig.suptitle(
        f"Ablation ladder: held-out $R^2$ of four predictions of the fine-tuned answer "
        f"vectors, layer {layer} (bars: mean over arms ± s.e.m.; points: per-arm; dashed "
        "lines: floor + ceiling means).\nDefault map convention (top-left): generic "
        "real-user contexts (LMSYS+WildChat, n=15,000 train), on-policy text in both "
        "models, span-mean pooling; other panels are variations against it",
        fontsize=11.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    savefig_paper(fig, f"result2_ladder_main_L{layer}", dir=FIG_DIR)
    plt.close(fig)


ARM_FIELD_LABELS = {
    "beh": {"cas": "casual style", "imp": "impoliteness", "syc": "sycophancy", "mk": "marker"},
    "ctx": {
        "pers": "persona ctx",
        "bare": "bare ctx",
        "conv": "conversation ctx",
        "icl": "in-context-learning ctx",
    },
    "regime": {"con": "contrastive", "po": "positive-only"},
}


def _arm_plain(arm: str) -> str:
    """Plain-English arm label for rendered tick text (no raw slugs, §3.5)."""
    parts = arm.split("-")
    beh = ARM_FIELD_LABELS["beh"].get(parts[0], parts[0])
    ctx = ARM_FIELD_LABELS["ctx"].get(parts[1], parts[1])
    if "ft" in parts:  # full fine-tune method arm (e.g. syc-pers-ft-con-s42)
        rest = [p for p in parts[2:] if p != "ft"]
        regime = ARM_FIELD_LABELS["regime"].get(rest[0], rest[0]) if rest else ""
        tail = ", ".join(rest[1:])
        return f"{beh}, {ctx}, full-FT {regime}" + (f", {tail}" if tail else "")
    regime = ARM_FIELD_LABELS["regime"].get(parts[2], parts[2])
    tail = ", ".join(parts[3:])
    return f"{beh}, {ctx}, {regime}" + (f", {tail}" if tail else "")


def _fig_dotplot_smalln(summary: dict, layer: int) -> None:
    """Labeled per-arm dot plot for the two n=3,000 context sets (12 arms each) —
    the points-labeled low-level companion behind the aggregate bars."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette(4)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), sharex="col")
    for row, tree in enumerate(TREES):
        for col, ctxset in enumerate(("on_target", "bare_n")):
            ax = axes[row][col]
            block = summary["grid"].get(f"{ctxset}__{tree}__L{layer}")
            if block is None:
                ax.set_axis_off()
                continue
            arms = sorted(block["per_arm"])
            ys = np.arange(len(arms))
            for j, r in enumerate(RUNGS):
                ax.scatter(
                    [block["per_arm"][a][r] for a in arms],
                    ys,
                    s=26,
                    color=colors[j],
                    zorder=3,
                    linewidths=0,
                    label=RUNG_LABEL[r].replace("\n", " ") if row + col == 0 else None,
                )
            for y in ys:
                vals = [block["per_arm"][arms[y]][r] for r in RUNGS]
                ax.plot([min(vals), max(vals)], [y, y], color="0.85", lw=1.0, zorder=1)
            ax.set_yticks(ys)
            ax.set_yticklabels([_arm_plain(a) for a in arms], fontsize=7.5)
            ax.set_title(f"{TREE_LABEL[tree]}\n{CTX_LABEL[ctxset]}", fontsize=9.5)
            ax.set_xlabel("held-out $R^2$ toward the fine-tuned answer vectors")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle(
        f"Per-arm ladder values, n=3,000-train context sets, layer {layer} "
        "(one row per fine-tuned arm; color = rung)",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    savefig_paper(fig, f"result2_ladder_perarm_dots_smalln_L{layer}", dir=FIG_DIR)
    plt.close(fig)


def _fig_perarm(summary: dict, layer: int) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_blog,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    beh_keys = ("cas", "imp", "syc", "mk")
    beh_label = {
        "cas": "casual writing style",
        "imp": "impoliteness",
        "syc": "sycophancy",
        "mk": "marker",
    }
    # behavior palette deliberately DISTINCT from the rung palette (one color = one meaning
    # across the writeup: Wong palette = rung identity, blog palette = behavior identity)
    beh_colors = dict(zip(beh_keys, paper_palette_blog(4), strict=True))
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.2), sharey="row")
    xs = np.arange(4)
    for row, tree in enumerate(TREES):
        for col, ctxset in enumerate(CTX_SETS):
            ax = axes[row][col]
            block = summary["grid"].get(f"{ctxset}__{tree}__L{layer}")
            if block is None:
                ax.set_axis_off()
                continue
            seen: set[str] = set()
            for arm, vals in block["per_arm"].items():
                beh = arm.split("-")[0]
                lbl = beh_label.get(beh, beh) if beh not in seen else None
                seen.add(beh)
                ax.plot(
                    xs,
                    [vals[r] for r in RUNGS],
                    color=beh_colors.get(beh, "0.4"),
                    alpha=0.5,
                    lw=1.0,
                    marker="o",
                    ms=2.4,
                    label=lbl,
                )
            ax.set_xticks(xs)
            ax.set_xticklabels([RUNG_LABEL[r] for r in RUNGS], fontsize=7.2)
            ax.set_title(f"{TREE_LABEL[tree]}\n{CTX_LABEL[ctxset]}", fontsize=9.5)
            if col == 0:
                ax.set_ylabel("held-out $R^2$ toward the\nfine-tuned answer vectors")
            if row == 0 and col == 0:
                ax.legend(title="behavior", fontsize=7.5, title_fontsize=8)
    fig.suptitle(
        f"Per-arm ladder trajectories (one line per fine-tuned arm), layer {layer}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    savefig_paper(fig, f"result2_ladder_perarm_L{layer}", dir=FIG_DIR)
    plt.close(fig)


def phase_figs(out: Path, layers: tuple[int, ...]) -> None:
    summary = json.loads((out / "summary.json").read_text())
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        _fig_main(summary, layer)
    _fig_perarm(summary, PILOT_LAYER)
    _fig_dotplot_smalln(summary, PILOT_LAYER)
    logger.info("[figs] wrote figures to %s", FIG_DIR)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        required=True,
        choices=["pilot", "battery", "spectra", "summary", "figs", "all"],
    )
    ap.add_argument("--stage-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--layers", type=str, default="14,19,25")
    ap.add_argument(
        "--arms", type=str, default="", help="comma-separated arm subset (debug/smoke only)"
    )
    args = ap.parse_args()
    layers = tuple(int(x) for x in args.layers.split(","))
    arms_filter = [a for a in args.arms.split(",") if a] or None
    stage = args.stage_dir or default_stage_dir()
    logger.info("[main] phase=%s stage=%s out=%s layers=%s", args.phase, stage, args.out, layers)
    if args.phase in ("pilot", "all"):
        phase_pilot(stage, args.out)
    if args.phase in ("battery", "all"):
        phase_battery(stage, args.out, layers, arms_filter)
    if args.phase in ("spectra", "all"):
        phase_battery(stage, args.out, layers, arms_filter, mode="spectra")
    if args.phase in ("summary", "all"):
        phase_summary(stage, args.out, layers)
    if args.phase in ("figs", "all"):
        phase_figs(args.out, layers)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit: heavy C-extension atexit teardown must not rewrite rc


if __name__ == "__main__":
    main()
