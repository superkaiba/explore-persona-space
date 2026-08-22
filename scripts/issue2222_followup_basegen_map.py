"""Free-analysis follow-up A for issue #2222 — base-generation-target map.

Step 9a-ter zero-GPU round (analysis-only; no model calls). Four legs over the
EXISTING P1/P2 capture store (HF ``issue2222_pvscreen/analysis_tensors/capture``):

- ``ctx2base`` / ``pfx2base``: LOFO (8 family folds) ridge maps ctxend/pfxend ->
  base_respavg at row grain (k=250 rows/dataset, n=6000, n_train=5250 > d=3584 —
  well-posed primal regime, stated per the #1701/#1887 duty), plus the
  mapped-base-gen ΔP predictor (raw_proj - pred_proj vs the #778 y).
- ``widegrid_ctx2raw``: the parent tuned_map fit (ctxend -> raw_respavg, same
  k=250 stored-order rows) with the lambda grid extended to 1e6; narrow
  (1e-2..1e4, 13pt) AND wide (1e-2..1e6, 17pt) grids are scanned from the SAME
  per-(layer,fold) factorization so the grid effect is isolated within-estimator.
- CJK-excluded exact-ΔP recount: mask base-generation rows containing CJK-script
  codepoints (row masks from ``raw_completions/exact_dp_base_gen``), recompute
  the exact_dp per-trait r on surviving rows.

Estimator note (record-integrity duty): ``_ridge_lofo_layer`` is a vectorized
twin of ``issue2222_analysis.dof_capped_ridge_multi_y`` (itself the documented
multi-target sibling of the #825 core): identical fp32-Gram -> fp64 eigh ->
per-target GCV under the #1887 dof cap (0.9); the per-lambda ``pred_tr`` GEMMs
are replaced by the algebraically-identical closed-form GCV in the eigenbasis
(rss_k(lam) = y_ss_k - (2q - e q^2)^T alpha_k^2, using z^T z = diag(e) and
z^T y = alpha) and the per-target w-build Python loop is vectorized into one
GEMM. ``--phase selfcheck`` asserts selected-lambda + held-out-prediction
equivalence against the reference. No permissiveness is broadened.

Estimator DEVIATION vs the parent's committed tuned_map.json: the parent leg
used the #825 core's inner-group-cv single-lambda-per-(layer,fold) selector;
this follow-up uses the dof-capped GCV per-TARGET selector (the dispatch note's
registered estimator). The output JSONs carry the note; the parent's saturation
stat (203/224 fits at the 1e4 grid max) is quoted for reference.

CONTENT HYGIENE: dataset rows / base generations include harmful-content
families — this module touches activations, row ids, and counts; completion
TEXT is scanned in-process for CJK codepoints only and is never printed,
logged, or persisted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# load_dotenv BEFORE any heavy import (#847 shared-VM thread caps bind in-process).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:  # sibling-script imports in script mode (#823)
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2222_analysis as ana  # noqa: E402
import issue2222_lib as lib  # noqa: E402
import issue2222_reduce as red  # noqa: E402

N_LAYERS, DIM = lib.RB_SHAPE
K_ROWS = 250  # parity with the parent tuned_map leg (tuned_map.json rows_per_dataset)
NARROW_GRID = np.logspace(-2, 4, 13)  # the #825 module LAMBDAS default (parent grid)
WIDE_GRID = np.logspace(-2, 6, 17)  # same 0.5-decade spacing, extended to 1e6
GCV_DOF_CAP = 0.9
KNN_LAYERS = (15, 19)
KNN_POOL_CAP = 3000
JOIN_FLOOR = 0.8  # mirror of the parent percell 80% join floor
MM_KINDS = ("ctx_j", "pfx_j", "base_j", "raw_j", "ctx_nj", "raw_nj")
# fit name -> (X memmap kind, Y memmap kind, identity-baseline applicability)
FITS: dict[str, tuple[str, str]] = {
    "ctx2base": ("ctx_j", "base_j"),
    "pfx2base": ("pfx_j", "base_j"),
    "widegrid_ctx2raw": ("ctx_nj", "raw_nj"),
}
# CJK-script codepoint blocks (Han + kana + hangul; punctuation-only blocks
# excluded). Built via chr() so no non-ASCII literal transits tool JSON (#1364).
CJK_BLOCKS: dict[str, tuple[int, int]] = {
    "hangul_jamo": (0x1100, 0x11FF),
    "hiragana": (0x3040, 0x309F),
    "katakana": (0x30A0, 0x30FF),
    "katakana_ext": (0x31F0, 0x31FF),
    "cjk_ext_a": (0x3400, 0x4DBF),
    "cjk_unified": (0x4E00, 0x9FFF),
    "hangul_syllables": (0xAC00, 0xD7AF),
    "cjk_compat": (0xF900, 0xFAFF),
    "cjk_ext_b": (0x20000, 0x2A6DF),
}
_CJK_RE = re.compile("[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in CJK_BLOCKS.values()) + "]")


def default_workdir() -> Path:
    """Durable per-issue work root on the data disk (worktree data/ is on /)."""
    return (
        Path("/mnt/eps-data")
        / os.environ.get("USER", "thomasjiralerspong")
        / ("issue2222_followup")
    )


def _mode_root(args) -> Path:
    root = Path(args.workdir)
    return root / "smoke" if args.smoke else root


def code_fingerprint() -> str:
    """Output-affecting code fingerprint (this file + the analysis helpers)."""
    return ana.files_fingerprint(
        [
            _SCRIPTS_DIR / "issue2222_followup_basegen_map.py",
            _SCRIPTS_DIR / "issue2222_analysis.py",
        ]
    )


# --- Staging (stream-and-delete; shared cache across smoke/production) ----------


def _stage_capture(args, ds: str, fname: str) -> tuple[Path, bool]:
    """(local path, staged_by_us) for one capture file; canonical-local first."""
    local = lib.capture_dir(Path(args.data_root), ds) / fname
    if local.exists():
        return local, False
    target = Path(args.workdir) / "staging" / ds / fname
    if target.exists():
        return target, True
    from explore_persona_space.orchestrate import hub

    lib.log_phase("fu_stage", "fetching capture file from HF", dataset=ds, file=fname)
    return (
        Path(
            hub.stage_hub_file(
                lib.HF_DATA_REPO,
                f"{lib.hf_capture_prefix(ds)}/{fname}",
                target,
                repo_type="dataset",
            )
        ),
        True,
    )


def _open_memmaps(root: Path, n_rows: int, mode: str) -> dict[str, np.ndarray]:
    """Open (creating on 'w') the six fp16 slice memmaps of shape (n_rows, L, D)."""
    mm_dir = root / "mm"
    mm_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, np.ndarray] = {}
    for kind in MM_KINDS:
        path = mm_dir / f"{kind}.npy"
        if path.exists():
            arr = np.load(path, mmap_mode="r+" if mode == "w" else "r")
            assert arr.shape == (n_rows, N_LAYERS, DIM), (kind, arr.shape)
        else:
            if mode != "w":
                raise FileNotFoundError(f"{path} missing — run --phase stage first")
            arr = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float16, shape=(n_rows, N_LAYERS, DIM)
            )
        out[kind] = arr
    return out


def phase_stage(args) -> None:
    """Stage capture npzs per dataset, slice into memmaps + projections, delete."""
    root = _mode_root(args)
    datasets = lib.dataset_ids(args.datasets)
    vhat, meta = red.load_vhat(Path(args.data_root))  # (T, L, D) float64 unit rows
    vhat32 = vhat.astype(np.float32)
    n_rows = K_ROWS * len(datasets)
    mm = _open_memmaps(root, n_rows, mode="w")
    (root / "proj").mkdir(parents=True, exist_ok=True)
    (root / "stage_done").mkdir(parents=True, exist_ok=True)
    for di, ds in enumerate(datasets):
        t0 = time.time()
        sentinel = root / "stage_done" / f"{ds}.json"
        if sentinel.exists():
            prior = json.loads(sentinel.read_text())
            if prior.get("code_fingerprint") == code_fingerprint() and prior.get("di") == di:
                lib.log_phase(
                    "fu_stage", "skip (fresh)", dataset=ds, unit=f"{di + 1}/{len(datasets)}"
                )
                continue
            lib.log_phase("fu_stage", "stale sentinel — restaging", dataset=ds)
        summ_path, s_staged = _stage_capture(args, ds, "summaries.npz")
        base_path, b_staged = _stage_capture(args, ds, "base_respavg.npz")
        with np.load(summ_path) as z:
            raw = z["raw_respavg"]
            ctx = z["ctxend"]
            pfx = z["pfxend"]
            row_ids = z["row_ids"]
        with np.load(base_path) as z:
            basev = z["base_respavg"]
            base_ids = z["row_ids"]
        assert raw.shape == (len(row_ids), N_LAYERS, DIM), raw.shape
        assert basev.shape == (len(base_ids), N_LAYERS, DIM), basev.shape
        # Join on row ids in SUMMARIES STORED ORDER (parent percell contract);
        # the nojoin slices reproduce the parent tuned_map's stored-order [:k].
        base_pos = {int(r): i for i, r in enumerate(base_ids)}
        pairs = [(i, base_pos[int(r)]) for i, r in enumerate(row_ids) if int(r) in base_pos]
        n_all = len(pairs)
        if n_all < JOIN_FLOOR * len(row_ids):
            raise ValueError(
                f"{ds}: join kept {n_all}/{len(row_ids)} rows — below the {JOIN_FLOOR} floor"
            )
        if n_all < K_ROWS or len(row_ids) < K_ROWS:
            raise ValueError(f"{ds}: {n_all} joined / {len(row_ids)} total rows < k={K_ROWS}")
        ia = np.array([i for i, _ in pairs])
        ib = np.array([j for _, j in pairs])
        sl = slice(di * K_ROWS, (di + 1) * K_ROWS)
        mm["ctx_j"][sl] = ctx[ia[:K_ROWS]].astype(np.float16)
        mm["pfx_j"][sl] = pfx[ia[:K_ROWS]].astype(np.float16)
        mm["base_j"][sl] = basev[ib[:K_ROWS]].astype(np.float16)
        mm["raw_j"][sl] = raw[ia[:K_ROWS]].astype(np.float16)
        mm["ctx_nj"][sl] = ctx[:K_ROWS].astype(np.float16)
        mm["raw_nj"][sl] = raw[:K_ROWS].astype(np.float16)
        for kind in MM_KINDS:
            assert np.isfinite(np.asarray(mm[kind][sl], dtype=np.float32)).all(), (ds, kind)
            mm[kind].flush()
        # Full-join per-row projections (exact_dp recount inputs; fp32, small).
        raw_proj = np.einsum("nld,tld->ntl", raw[ia].astype(np.float32), vhat32, optimize=True)
        base_proj = np.einsum("nld,tld->ntl", basev[ib].astype(np.float32), vhat32, optimize=True)
        proj_tmp = root / "proj" / f"{ds}.tmp.npz"
        np.savez(
            proj_tmp, row_ids=row_ids[ia].astype(np.int64), raw_proj=raw_proj, base_proj=base_proj
        )
        proj_tmp.replace(root / "proj" / f"{ds}.npz")
        lib.write_json_atomic(
            sentinel,
            {
                "dataset": ds,
                "di": di,
                "n_summary_rows": int(len(row_ids)),
                "n_base_rows": int(len(base_ids)),
                "n_joined": int(n_all),
                "k_rows": K_ROWS,
                "rb_source": meta["rb_source"],
                "code_fingerprint": code_fingerprint(),
                **lib.run_metadata(),
            },
        )
        if args.delete_staged:
            for p, staged in ((summ_path, s_staged), (base_path, b_staged)):
                if staged:
                    p.unlink()
        lib.log_phase(
            "fu_stage",
            "done",
            dataset=ds,
            unit=f"{di + 1}/{len(datasets)}",
            n_joined=int(n_all),
            elapsed_s=round(time.time() - t0, 1),
        )


# --- CJK row masks ---------------------------------------------------------------


def phase_cjk(args) -> None:
    """Per-dataset CJK row masks from the base-generation JSONLs (digest-only)."""
    root = _mode_root(args)
    (root / "cjk").mkdir(parents=True, exist_ok=True)
    datasets = lib.dataset_ids(args.datasets)
    for di, ds in enumerate(datasets):
        out = root / "cjk" / f"{ds}.json"
        if out.exists():
            # Fingerprint-keyed resume skip (mirrors the stage sentinel; #722-r3
            # class): a CJK_BLOCKS / regex edit must recompute stale masks.
            prior = json.loads(out.read_text())
            if prior.get("code_fingerprint") == code_fingerprint():
                lib.log_phase("fu_cjk", "skip (fresh)", dataset=ds)
                continue
            lib.log_phase("fu_cjk", "stale mask — recomputing", dataset=ds)
        local = lib.rawcomp_path(Path(args.data_root), ds)
        if not local.exists():
            local = Path(args.workdir) / "rawcomp" / f"{ds}.jsonl"
            if not local.exists():
                from explore_persona_space.orchestrate import hub

                local = Path(
                    hub.stage_hub_file(
                        lib.HF_DATA_REPO, lib.hf_rawcomp_path(ds), local, repo_type="dataset"
                    )
                )
        rows = lib.read_jsonl(local)
        cjk_ids = sorted(
            int(rec["row_id"]) for rec in rows if _CJK_RE.search(rec["completion"]) is not None
        )
        lib.write_json_atomic(
            out,
            {
                "dataset": ds,
                "n_rows": len(rows),
                "n_cjk": len(cjk_ids),
                "cjk_row_ids": cjk_ids,
                "blocks": sorted(CJK_BLOCKS),
                "code_fingerprint": code_fingerprint(),
                **lib.run_metadata(),
            },
        )
        lib.log_phase(
            "fu_cjk",
            "done",
            dataset=ds,
            unit=f"{di + 1}/{len(datasets)}",
            n_rows=len(rows),
            n_cjk=len(cjk_ids),
        )


# --- Ridge core (vectorized primal twin of ana.dof_capped_ridge_multi_y) --------


def _ridge_lofo_layer(
    x_l: np.ndarray,
    y_l: np.ndarray,
    fold_ids: np.ndarray,
    *,
    grids: dict[str, np.ndarray],
    dof_cap: float = GCV_DOF_CAP,
) -> dict:
    """LOFO ridge for ONE layer, per-target dof-capped GCV, multi-grid.

    x_l (n, d) fp32; y_l (n, T) fp64; fold_ids (n,). Returns per grid:
    {pred (n, T) held-out, lam (F, T), df (F, T)}. One eigh per fold shared
    across every lambda, every grid, and every target (vectorize-first).
    Arithmetic path matches ``ana.dof_capped_ridge_multi_y`` (fp32 Gram ->
    fp64 eigh; per-target argmin-GCV under the dof cap); see module docstring.
    """
    import torch

    x_l = np.ascontiguousarray(x_l, dtype=np.float32)
    y_l = np.asarray(y_l, dtype=np.float64)
    n, d = x_l.shape
    t = y_l.shape[1]
    uniq = np.unique(fold_ids)
    out = {
        g: {
            "pred": np.full((n, t), np.nan),
            "lam": np.full((len(uniq), t), np.nan),
            "df": np.full((len(uniq), t), np.nan),
        }
        for g in grids
    }
    for fi, f in enumerate(uniq):
        hold = fold_ids == f
        tr = ~hold
        n_tr = int(tr.sum())
        if n_tr <= d:
            raise ValueError(f"fold {f!r}: n_train={n_tr} <= d={d} — under-determined refused")
        xt = torch.from_numpy(x_l[tr])
        x_mu = xt.mean(dim=0, keepdim=True)
        xt = xt - x_mu
        yt = torch.from_numpy(y_l[tr])
        y_mu = yt.mean(dim=0, keepdim=True)
        yt = yt - y_mu
        g64 = (xt.T @ xt).double()
        evals, vecs = ana._eigh_with_cpu_fallback(g64)
        evals = torch.clamp(evals, min=0.0)
        xty = xt.T.double() @ yt  # (d, T)
        alpha = vecs.T @ xty  # (d, T)
        y_ss = (yt**2).sum(dim=0)  # (T,)
        a2 = alpha**2  # (d, T)
        xh = torch.from_numpy(x_l[hold]).double()
        for gname, grid in grids.items():
            lam_t = torch.as_tensor(np.asarray(grid, dtype=np.float64))
            q = 1.0 / (evals.unsqueeze(0) + lam_t.unsqueeze(1))  # (Lm, d)
            dfs = (evals.unsqueeze(0) * q).sum(dim=1)  # (Lm,)
            admissible = dfs <= dof_cap * n_tr
            # Closed-form train RSS in the eigenbasis (== the reference's
            # y_ss - 2(pred*y).sum + (pred^2).sum; see module docstring).
            w_scan = 2.0 * q - evals.unsqueeze(0) * q**2  # (Lm, d)
            rss = y_ss.unsqueeze(0) - w_scan @ a2  # (Lm, T)
            gcv = torch.where(
                admissible.unsqueeze(1),
                n_tr * rss / (n_tr - dfs).unsqueeze(1) ** 2,
                torch.tensor(float("inf"), dtype=torch.float64),
            )
            if not torch.isfinite(gcv).any():
                raise ValueError(f"fold {f!r} grid {gname!r}: no admissible lambda")
            best = torch.argmin(gcv, dim=0)  # (T,) first-minimum, matching the reference
            lam_sel = lam_t[best]  # (T,)
            q_sel = 1.0 / (evals.unsqueeze(1) + lam_sel.unsqueeze(0))  # (d, T)
            w = vecs @ (alpha * q_sel)  # (d, T)
            b0 = y_mu.double() - x_mu.double() @ w  # (1, T)
            out[gname]["pred"][hold] = (xh @ w + b0).numpy()
            out[gname]["lam"][fi] = lam_sel.numpy()
            out[gname]["df"][fi] = dfs[best].numpy()
    return out


def _pooled_r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Pooled held-out R^2 across all rows x targets (tuned_map convention)."""
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def _per_target_r2(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    ss_res = ((y - pred) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


def _lam_hist(lam: np.ndarray, grid: np.ndarray) -> dict[str, int]:
    """{lambda value: count} over the (F, T) selected-lambda matrix."""
    vals, counts = np.unique(lam[np.isfinite(lam)], return_counts=True)
    hist = dict.fromkeys((f"{g:.6g}" for g in grid), 0)
    for v, c in zip(vals, counts, strict=True):
        hist[f"{v:.6g}"] = int(c)
    return hist


def phase_fit(args) -> None:
    """Per-(fit, layer) checkpointed LOFO ridge fits over the staged memmaps."""
    root = _mode_root(args)
    datasets = lib.dataset_ids(args.datasets)
    # Require a FRESH stage sentinel per dataset (mirrors the stage skip check):
    # a manual `--phase fit` after an INCOMPLETE/stale stage would otherwise fit
    # zero-filled memmap slices silently (zeros are finite, so the isfinite
    # assert passes). Fail loud instead (#2222 pre-merge review Minor 2).
    for di, ds in enumerate(datasets):
        sentinel = root / "stage_done" / f"{ds}.json"
        if not sentinel.exists():
            raise FileNotFoundError(f"{sentinel} missing — run --phase stage first")
        prior = json.loads(sentinel.read_text())
        if prior.get("code_fingerprint") != code_fingerprint() or prior.get("di") != di:
            raise RuntimeError(
                f"{ds}: stale stage sentinel (code_fingerprint/di mismatch) — "
                "re-run --phase stage before fit"
            )
    fam_idx, families = red._family_index(datasets)
    fold_ids = np.repeat(fam_idx, K_ROWS)
    n_rows = K_ROWS * len(datasets)
    mm = _open_memmaps(root, n_rows, mode="r")
    vhat, _meta = red.load_vhat(Path(args.data_root))
    dims = args.dims or DIM
    grids = {"narrow": NARROW_GRID, "wide": WIDE_GRID}
    layers = args.layers if args.layers else list(range(N_LAYERS))
    fits = args.fits if args.fits else list(FITS)
    for fit_name in fits:
        xk, yk = FITS[fit_name]
        ck_dir = root / "fitck" / fit_name
        ck_dir.mkdir(parents=True, exist_ok=True)
        for li, layer in enumerate(layers):
            t0 = time.time()
            ck_npz = ck_dir / f"layer{layer:02d}.npz"
            ck_json = ck_dir / f"layer{layer:02d}.json"
            if ck_npz.exists() and ck_json.exists():
                if json.loads(ck_json.read_text()).get("code_fingerprint") == code_fingerprint():
                    lib.log_phase("fu_fit", "skip (fresh)", fit=fit_name, layer=layer)
                    continue
            x_l = np.asarray(mm[xk][:, layer, :dims], dtype=np.float32)
            y_l = np.asarray(mm[yk][:, layer, :dims], dtype=np.float64)
            vhat_l = vhat[:, layer, :dims]  # (T_traits, dims)
            res = _ridge_lofo_layer(x_l, y_l, fold_ids, grids=grids, dof_cap=GCV_DOF_CAP)
            assert np.isfinite(res["wide"]["pred"]).all(), (fit_name, layer)
            # Identity(+LOFO-bias) baselines on the SAME rows (mapping rule).
            x64 = x_l.astype(np.float64)
            ss_tot = float(((y_l - y_l.mean(axis=0)) ** 2).sum())
            r2_identity = 1.0 - float(((y_l - x64) ** 2).sum()) / ss_tot
            ss_idb = 0.0
            for f in np.unique(fold_ids):
                hold = fold_ids == f
                b_f = (y_l[~hold] - x64[~hold]).mean(axis=0)
                ss_idb += float(((y_l[hold] - x64[hold] - b_f) ** 2).sum())
            r2_id_bias = 1.0 - ss_idb / ss_tot
            # kNN retrieval on wide-grid held-out preds at the steering layers.
            knn_reads: list[dict] = []
            if layer in KNN_LAYERS:
                from explore_persona_space.analysis.mapping_baselines import knn_retrieval

                for f in np.unique(fold_ids):
                    hold = fold_ids == f
                    pred32 = res["wide"]["pred"][hold].astype(np.float32)
                    true32 = y_l[hold].astype(np.float32)
                    if len(true32) > KNN_POOL_CAP:
                        rng = np.random.default_rng(lib.SUBSAMPLE_SEED + int(f))
                        sel = rng.choice(len(true32), size=KNN_POOL_CAP, replace=False)
                        pred32, true32 = pred32[sel], true32[sel]
                    for metric in ("euclidean", "cosine"):
                        knn_reads.append(
                            {
                                "fold_family": families[int(f)],
                                **knn_retrieval(pred32, true32, metric=metric),
                            }
                        )
            # ΔP ingredients: raw/base/pred projections onto vhat (n, T_traits).
            raw_kind = "raw_nj" if xk == "ctx_nj" else "raw_j"
            raw_l32 = np.asarray(mm[raw_kind][:, layer, :dims], dtype=np.float32)
            vh32 = vhat_l.astype(np.float32)
            raw_proj = raw_l32 @ vh32.T
            pred_proj = res["wide"]["pred"].astype(np.float32) @ vh32.T
            base_proj = (
                np.asarray(mm["base_j"][:, layer, :dims], dtype=np.float32) @ vh32.T
                if yk == "base_j"
                else None
            )
            tmp = ck_npz.with_name(ck_npz.stem + ".tmp.npz")
            np.savez(
                tmp,
                raw_proj=raw_proj,
                pred_proj=pred_proj,
                **({"base_proj": base_proj} if base_proj is not None else {}),
                lam_wide=res["wide"]["lam"],
                lam_narrow=res["narrow"]["lam"],
            )
            tmp.replace(ck_npz)
            per_t_wide = _per_target_r2(y_l, res["wide"]["pred"])
            lib.write_json_atomic(
                ck_json,
                {
                    "fit": fit_name,
                    "layer": layer,
                    "dims": dims,
                    "n_rows": n_rows,
                    "n_train_min": int(min((fold_ids != f).sum() for f in np.unique(fold_ids))),
                    "r2_pooled": {g: _pooled_r2(y_l, res[g]["pred"]) for g in grids},
                    "r2_per_target_quartiles_wide": [
                        float(np.nanpercentile(per_t_wide, p)) for p in (25, 50, 75)
                    ],
                    "lam_hist": {g: _lam_hist(res[g]["lam"], grids[g]) for g in grids},
                    "r2_identity": r2_identity,
                    "r2_identity_plus_bias": r2_id_bias,
                    "knn": knn_reads,
                    "code_fingerprint": code_fingerprint(),
                    **lib.run_metadata(),
                },
            )
            lib.log_phase(
                "fu_fit",
                "done",
                fit=fit_name,
                layer=layer,
                unit=f"{li + 1}/{len(layers)}",
                r2_wide=round(_pooled_r2(y_l, res["wide"]["pred"]), 4),
                elapsed_s=round(time.time() - t0, 1),
            )


# --- Reduce ----------------------------------------------------------------------


def _dp_records(
    vals: np.ndarray,
    datasets: list[str],
    fam_idx: np.ndarray,
    y_axis: dict,
    *,
    smoke: bool = False,
) -> list:
    """mapped_tuned-style records: r at the steering layer + LOFO sweep per trait.

    Smoke-only degeneracy guard: with 1 dataset per non-held family the LOFO
    train fold can carry ZERO y variance (e.g. trait_score(evil)=0.0 on both
    off-trait datasets), so ``pearson_r_cols`` standardizes y to NaN and
    ``lofo_layer_sweep`` raises on the all-NaN argmax. That is a smoke-slice
    artifact (production trains each fold on 21 datasets with real y spread);
    under ``smoke`` the sweep is recorded as degenerate-skipped, production
    stays fail-loud (smoke-blind-spot: the smoke does NOT certify the LOFO
    sweep leg at production n).
    """
    records = []
    for ti, trait in enumerate(lib.TRAITS):
        yv = np.array([y_axis[trait][ds]["trait_score"] for ds in datasets])
        r_layers = ana.pearson_r_cols(vals[:, ti, :], yv)
        if smoke:
            try:
                sweep = ana.lofo_layer_sweep(vals[:, ti, :], yv, fam_idx)
            except ValueError as exc:  # all-NaN train-fold r at smoke n
                sweep = {"skipped_smoke_degenerate": f"{type(exc).__name__}: {exc}"}
        else:
            sweep = ana.lofo_layer_sweep(vals[:, ti, :], yv, fam_idx)
        steer = red.STEER_IDX[trait]
        records.append(
            {
                "trait": trait,
                "r_steer": float(r_layers[steer]),
                "steer_layer": steer,
                "sweep": sweep,
                "r_per_layer": [float(v) for v in r_layers],
            }
        )
    return records


def _parent_reference() -> dict:
    """Parent committed reads quoted for comparison (never re-derived)."""
    out_root = lib.REPO_ROOT / "eval_results" / "issue_2222"
    pred = json.loads((out_root / "predictor_correlations.json").read_text())
    tuned = json.loads((out_root / "tuned_map.json").read_text())
    rows = [
        {k: r.get(k) for k in ("trait", "arm", "layer_regime", "layer", "r", "published_r")}
        for r in pred["records"]
        if r.get("arm") in ("exact_dp", "mapped_ctx")
    ]
    lam = np.asarray(tuned["selected_lambda_per_layer_fold"], dtype=float)
    return {
        "predictor_records": rows,
        "tuned_map": {
            "heldout_r2_per_layer": tuned["heldout_r2_per_layer"],
            "records": [
                {k: r.get(k) for k in ("trait", "arm", "r_steer", "steer_layer")}
                for r in tuned["records"]
            ],
            "n_selected_at_grid_max_1e4": int((lam >= 1e4).sum()),
            "n_fits": int(lam.size),
            "selector": "inner-group-cv (single lambda per layer x fold)",
        },
    }


def _mean_by_dataset(per_row: np.ndarray, n_datasets: int) -> np.ndarray:
    """(n_rows, T) row values -> (n_datasets, T) dataset means (K_ROWS blocks)."""
    return per_row.reshape(n_datasets, K_ROWS, -1).mean(axis=1)


def phase_reduce(args) -> None:
    """Assemble the three output JSONs from fit checkpoints + projections.

    Smoke out-root divert: under ``--smoke`` the JSONs land in the smoke
    workdir (``<workdir>/smoke/out/``), never the canonical committed
    ``eval_results/`` path (smoke outputs must not overwrite/pollute the
    canonical out-root).

    Smoke blind-spot enumeration: under ``--smoke``, ``_parent_reference()``
    is SUBSTITUTED with a note dict, so its committed-parent JSON key reads
    (``heldout_r2_per_layer``, ``selected_lambda_per_layer_fold``,
    ``records``) execute only on the production branch — statically verified
    against the committed parent artifacts (#2222 review Minor 4); a reduce
    crash is cheap (fits are checkpointed per layer).
    """
    root = _mode_root(args)
    out_root = root / "out" if args.smoke else Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    datasets = lib.dataset_ids(args.datasets)
    fam_idx, _families = red._family_index(datasets)
    y_axis = red.load_y_axis(datasets)
    n_ds = len(datasets)
    parent = _parent_reference() if not args.smoke else {"note": "smoke — parent quotes skipped"}

    def _load_ck(fit_name: str) -> tuple[dict, dict]:
        per_layer_json: dict[int, dict] = {}
        per_layer_npz: dict[int, dict] = {}
        for layer in range(N_LAYERS):
            jp = root / "fitck" / fit_name / f"layer{layer:02d}.json"
            zp = root / "fitck" / fit_name / f"layer{layer:02d}.npz"
            if not jp.exists():
                if args.allow_partial:
                    continue
                raise FileNotFoundError(f"{jp} missing — run --phase fit first")
            per_layer_json[layer] = json.loads(jp.read_text())
            with np.load(zp) as z:
                per_layer_npz[layer] = {k: z[k] for k in z.files}
        return per_layer_json, per_layer_npz

    def _fit_block(fit_name: str) -> dict:
        pj, pz = _load_ck(fit_name)
        layers = sorted(pj)
        vals = np.full((n_ds, len(lib.TRAITS), N_LAYERS), np.nan)
        for layer in layers:
            diff = pz[layer]["raw_proj"] - pz[layer]["pred_proj"]  # (n_rows, T)
            vals[:, :, layer] = _mean_by_dataset(diff, n_ds)
        agg_hist = {g: {} for g in ("narrow", "wide")}
        for layer in layers:
            for g in agg_hist:
                for k, v in pj[layer]["lam_hist"][g].items():
                    agg_hist[g][k] = agg_hist[g].get(k, 0) + v
        wide_total = sum(agg_hist["wide"].values()) or 1
        return {
            "n_rows": pj[layers[0]]["n_rows"],
            "rows_per_dataset": K_ROWS,
            "n_train_min_per_fold": pj[layers[0]]["n_train_min"],
            "d": pj[layers[0]]["dims"],
            "well_posed_n_gt_d": bool(pj[layers[0]]["n_train_min"] > pj[layers[0]]["dims"]),
            "heldout_r2_per_layer": {
                g: [pj[layer]["r2_pooled"][g] if layer in pj else None for layer in range(N_LAYERS)]
                for g in ("narrow", "wide")
            },
            "r2_identity_per_layer": [
                pj[layer]["r2_identity"] if layer in pj else None for layer in range(N_LAYERS)
            ],
            "r2_identity_plus_bias_per_layer": [
                pj[layer]["r2_identity_plus_bias"] if layer in pj else None
                for layer in range(N_LAYERS)
            ],
            "knn_retrieval": {
                f"layer{layer}": pj[layer]["knn"] for layer in layers if pj[layer]["knn"]
            },
            "selected_lambda_hist": agg_hist,
            "wide_frac_selected_above_1e4": float(
                sum(v for k, v in agg_hist["wide"].items() if float(k) > 1.0e4) / wide_total
            ),
            "records": _dp_records(vals, datasets, fam_idx, y_axis, smoke=args.smoke),
        }

    est_note = (
        "estimator: vectorized primal twin of issue2222_analysis.dof_capped_ridge_multi_y "
        "(#825-core sibling): per-target GCV under dof cap 0.9, LOFO over the 8 families; "
        "equivalence-gated via --phase selfcheck. DEVIATION vs the parent tuned_map leg: "
        "the parent used the #825 core's inner-group-cv single-lambda selector."
    )
    # 1) basegen_map.json — both mapping arms + the mapped-base-gen ΔP read.
    ctx_block = _fit_block("ctx2base")
    pfx_block = _fit_block("pfx2base")
    # exact_dp on the SAME k=250 subset (apples-to-apples companion).
    _, pz_ctx = _load_ck("ctx2base")
    vals_dp = np.full((n_ds, len(lib.TRAITS), N_LAYERS), np.nan)
    for layer in sorted(pz_ctx):
        diff = pz_ctx[layer]["raw_proj"] - pz_ctx[layer]["base_proj"]
        vals_dp[:, :, layer] = _mean_by_dataset(diff, n_ds)
    lib.write_json_atomic(
        out_root / "basegen_map.json",
        {
            "note": "base-generation-target map (ctxend/pfxend -> base_respavg, LOFO ridge) "
            "+ mapped-base-gen ΔP predictor (raw - predicted-base projections vs the #778 y). "
            + est_note,
            "arms": {"mapped_base_ctx": ctx_block, "mapped_base_pfx": pfx_block},
            "exact_dp_k250_companion": {
                "note": "exact_dp (raw - actual base projections) on the SAME k=250 row "
                "subset the maps consumed — the apples-to-apples reference for the "
                "mapped-base arms (the parent's committed exact_dp used all joined rows)",
                "records": _dp_records(vals_dp, datasets, fam_idx, y_axis, smoke=args.smoke),
            },
            "parent_reference": parent,
            "code_fingerprint": code_fingerprint(),
            **lib.run_metadata(),
        },
    )
    # 2) tuned_map_widegrid.json — the wider-lambda refit.
    wg_block = _fit_block("widegrid_ctx2raw")
    wg_records = wg_block.pop("records")
    for r in wg_records:
        r["arm"] = "mapped_tuned_widegrid"
    lib.write_json_atomic(
        out_root / "tuned_map_widegrid.json",
        {
            "note": "EXPLORATORY tuned-map refit (ctxend -> raw_respavg, LOFO over 8 "
            "families) with the lambda grid extended to 1e6; narrow (13pt, max 1e4) and "
            "wide (17pt, max 1e6) grids scanned from the SAME factorization. " + est_note,
            **wg_block,
            "records": wg_records,
            "parent_reference": parent if args.smoke else parent["tuned_map"],
            "code_fingerprint": code_fingerprint(),
            **lib.run_metadata(),
        },
    )
    # 3) exact_dp_cjk_excluded.json — CJK-masked recount over ALL joined rows.
    masks: dict[str, set[int]] = {}
    n_masked: dict[str, int] = {}
    for ds in datasets:
        rec = json.loads((root / "cjk" / f"{ds}.json").read_text())
        masks[ds] = set(rec["cjk_row_ids"])
        n_masked[ds] = rec["n_cjk"]
    vals_all = np.full((n_ds, len(lib.TRAITS), N_LAYERS), np.nan)
    vals_excl = np.full((n_ds, len(lib.TRAITS), N_LAYERS), np.nan)
    n_used: dict[str, dict[str, int]] = {}
    for di, ds in enumerate(datasets):
        with np.load(root / "proj" / f"{ds}.npz") as z:
            row_ids = z["row_ids"]
            diff = z["raw_proj"].astype(np.float64) - z["base_proj"].astype(np.float64)
        keep = np.array([int(r) not in masks[ds] for r in row_ids])
        if keep.sum() == 0:
            raise RuntimeError(f"{ds}: CJK mask removed every joined row")
        vals_all[di] = diff.mean(axis=0)
        vals_excl[di] = diff[keep].mean(axis=0)
        n_used[ds] = {
            "n_joined": int(len(row_ids)),
            "n_cjk_masked_in_join": int((~keep).sum()),
            "n_kept": int(keep.sum()),
        }
    lib.write_json_atomic(
        out_root / "exact_dp_cjk_excluded.json",
        {
            "note": "exact_dp recount over ALL joined rows, with base-generation rows "
            "containing CJK-script codepoints excluded (mask from "
            "raw_completions/exact_dp_base_gen; blocks listed). all_rows = the "
            "unmasked recount on the identical join (reproduction reference).",
            "cjk_blocks": sorted(CJK_BLOCKS),
            "n_cjk_per_dataset_file": n_masked,
            "rows_used_per_dataset": n_used,
            "records_all_rows": _dp_records(vals_all, datasets, fam_idx, y_axis, smoke=args.smoke),
            "records_cjk_excluded": _dp_records(
                vals_excl, datasets, fam_idx, y_axis, smoke=args.smoke
            ),
            "parent_reference": parent,
            "code_fingerprint": code_fingerprint(),
            **lib.run_metadata(),
        },
    )
    lib.log_phase("fu_reduce", "done", out_root=str(out_root))


# --- Self-check + pilot ------------------------------------------------------------


def phase_selfcheck(args) -> None:
    """Equivalence gate: _ridge_lofo_layer vs ana.dof_capped_ridge_multi_y."""
    rng = np.random.default_rng(0)
    n, d, t = 320, 24, 6
    fold_ids = np.repeat(np.arange(5), n // 5)
    x = rng.standard_normal((n, d)).astype(np.float32)
    w_true = rng.standard_normal((d, t))
    y = x.astype(np.float64) @ w_true + 0.5 * rng.standard_normal((n, t))
    mine = _ridge_lofo_layer(x, y, fold_ids, grids={"narrow": NARROW_GRID})
    ref = ana.dof_capped_ridge_multi_y(x, y, fold_ids, lambdas=NARROW_GRID, dof_cap=GCV_DOF_CAP)
    lam_mine = mine["narrow"]["lam"]
    lam_ref = np.asarray(ref["gcv_lambda"])
    n_match = int((lam_mine == lam_ref).sum())
    assert n_match == lam_ref.size, f"selected-lambda mismatch: {n_match}/{lam_ref.size}"
    assert np.allclose(mine["narrow"]["pred"], ref["heldout_pred"], rtol=1e-8, atol=1e-9), float(
        np.nanmax(np.abs(mine["narrow"]["pred"] - ref["heldout_pred"]))
    )
    r2_mine = _per_target_r2(y, mine["narrow"]["pred"])
    assert np.allclose(r2_mine, ref["heldout_r2"], rtol=1e-9, atol=1e-10)
    lib.log_phase(
        "fu_selfcheck",
        "PASS — lambda + heldout-pred + R2 equivalence vs dof_capped_ridge_multi_y",
        n_fits=int(lam_ref.size),
    )


def phase_pilot(args) -> None:
    """Measured 1-layer pilot at PRODUCTION shape (synthetic data, real kernel)."""
    rng = np.random.default_rng(0)
    n, d, t = K_ROWS * 24, DIM, DIM
    fold_ids = np.repeat(np.arange(8), n // 8)
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.standard_normal((n, t))
    t0 = time.time()
    _ridge_lofo_layer(x, y, fold_ids, grids={"narrow": NARROW_GRID, "wide": WIDE_GRID})
    wall = time.time() - t0
    lib.log_phase(
        "fu_pilot",
        "1-layer production-shape wall",
        n=n,
        d=d,
        wall_s=round(wall, 1),
        projected_per_fit_min=round(wall * N_LAYERS / 60, 1),
        projected_three_fits_h=round(3 * wall * N_LAYERS / 3600, 2),
    )


PHASES = {
    "selfcheck": phase_selfcheck,
    "pilot": phase_pilot,
    "stage": phase_stage,
    "cjk": phase_cjk,
    "fit": phase_fit,
    "reduce": phase_reduce,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", default="all", choices=["all", *PHASES])
    ap.add_argument("--data-root", default=str(lib.default_data_root()))
    ap.add_argument("--workdir", default=str(default_workdir()))
    ap.add_argument(
        "--out-root",
        default=str(lib.REPO_ROOT / "eval_results" / "issue_2222" / "followup_free_analysis"),
    )
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--fits", nargs="*", default=None, choices=list(FITS))
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--dims", type=int, default=None, help="feature-dim slice (smoke only)")
    ap.add_argument("--smoke", action="store_true", help="separate smoke workdir/out shapes")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="smoke-only: tolerate missing fit-layer checkpoints in reduce",
    )
    ap.add_argument("--delete-staged", action="store_true", help="stream-and-delete staging")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        # Deferred imports executed (smoke-architecture Axis 1):
        from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: F401
        from explore_persona_space.orchestrate import hub  # noqa: F401
        from explore_persona_space.orchestrate.argcheck import (  # noqa: F401
            assert_args_attributes_defined as _chk,
        )

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if args.smoke and args.phase in ("all", "fit") and not args.dims:
        raise SystemExit("--smoke fit needs --dims (< n_train) — production omits both")
    if args.dims and not args.smoke:
        raise SystemExit("--dims is a smoke-only dial; production runs the full DIM")
    if args.allow_partial and not args.smoke:
        # Missing layers leave NaNs in r_per_layer/r_steer and json.dumps
        # (allow_nan=True) would emit non-strict `NaN` literals (#2222 Minor 5).
        raise SystemExit("--allow-partial is a smoke-only dial; production reduces all layers")
    phases = list(PHASES) if args.phase == "all" else [args.phase]
    if args.phase == "all":
        phases = ["selfcheck", "stage", "cjk", "fit", "reduce"]  # pilot is on-demand
    for name in phases:
        lib.log_phase("fu_phase", "enter", phase_name=name)
        PHASES[name](args)
    lib.log_phase("done", "followup basegen-map complete", phases=phases)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
