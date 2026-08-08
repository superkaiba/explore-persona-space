"""Input-side mirror of the #1895 subspace-overlap analysis (inline round, 0 GPU).

Two legs off one shared assembly (dispatch + addendum notes on #1895, 2026-08-03):

Leg B (PRIMARY) — input-bottleneck functional comparison: the dense-input map
(v_C -> v_A) vs the SAE-INPUT map (SAE(v_C) codes -> v_A) on the SAME dense
answer target; per-answer-direction R2 profiles of both maps in the answer-PCA
basis + the gap profile, pooled reads, per-context gap quantiles.

Leg A (secondary) — geometric mirror of the #1895 output-side read: top-k input
eigendirections ranked by the dense map's read-energy vs the SAE-reconstruction
(and residual-complement) PCA subspaces of the context states, against
within-shell variance-matched rotation nulls (primary profile only).

Reuses the parent code paths verbatim: EA._assemble_with_ci (assembly + G1 sha
asserts), N1M._ridge_factorize/_ridge_predict_one (fits), issue1895_subspaces
shell_partition/overlap_observed/overlap_null_draws/_pca_basis (angle battery),
issue1482_sae.BatchTopKSAE (encode/decode).

Usage:
    uv run python scripts/issue1895_input_side_overlap.py --phase all
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(8)

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue779_percontext_recon as PR  # noqa: E402
import issue1482_error_analysis as EA  # noqa: E402
import issue1482_sae as S  # noqa: E402
import issue1895_subspaces as SUB  # noqa: E402

OUT_ROOT = Path(
    os.environ.get("EPM_IS1895_OUT_ROOT", "/mnt/eps-data/thomasjiralerspong/issue1895_inputside")
)
DEVICE = os.environ.get("EPM_IS1895_DEVICE", "cpu")
SCRATCH = OUT_ROOT / "scratch"
SAE_DIR = OUT_ROOT / "sae_dl"
EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_1895" / "input_side"
FIG_DIR = PROJECT_ROOT / "figures" / "issue_1895"
PASS_B = PROJECT_ROOT / "data" / "issue_779" / "pass_b" / "train_context_vectors.pt"

BANKED_POOLED_R2 = 0.716621  # eval_results/issue_1895/fits_summary.json t_vA_ctx
GATE_TOL = 2e-3
K_GRID = (16, 32, 64, 128, 256)
SHELL_GRID = (16, 32, 64)
N_DRAWS = 1000
SEED = 18950
PANEL_F_IN = 8192  # mirrors the banked bridge's f_in
BLOCK = 4096


def log(msg: str) -> None:
    print(f"[inputside {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def done_path(phase: str) -> Path:
    return OUT_ROOT / f"done_{phase}.json"


def mark_done(phase: str, info: dict) -> None:
    done_path(phase).write_text(json.dumps(info, indent=2) + "\n")
    log(f"phase {phase} done: {json.dumps(info)[:200]}")


def is_done(phase: str) -> bool:
    return done_path(phase).exists()


# ── phase 1: assemble ────────────────────────────────────────────────────────────


def phase_assemble_light() -> None:
    """Memory-lean assembly: parent internals VERBATIM (same loaders, same stream,
    same fixed_split + pinned-sha asserts, same G1 carve) with the final
    np.concatenate replaced by an INDEXED GATHER of the ~142k design rows.

    The concatenate is the OOM peak (full X+Y 27.7 GB on top of the streamer's
    own 27.6 GB -> ~55 GB; earlyoom killed the first attempt at 11:02Z). Nothing
    downstream needs the full arrays: phase_cov and _design() both read
    design_export when present. Correctness is gated end-to-end by the G1 split
    shas + the pinned val/test shas, which depend on prov + row ORDER, not on the
    materialized activations."""
    if is_done("assemble_light") or is_done("export_design"):
        log("assemble_light: resume-skip")
        return
    SCRATCH.mkdir(parents=True, exist_ok=True)
    exp = OUT_ROOT / "design_export"
    exp.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(
        pass_b=PASS_B,
        out_dir=SCRATCH,
        manifest_from_hf=True,
        hf_prefix=EA.CAPTURE_PREFIX,
        manifest_hf_prefix=EA.N1G.HF_PREFIX,
        n1m_capture_dir=None,
        fresh_stream=False,
        orig_dir=N1M.DEFAULT_ORIG_DIR,
    )
    layer = SUB.LAYER
    pb = EA.N1G._load_pass_b_bundle(ns.pass_b)
    for fld in ("cx_last", "v_x"):
        assert fld in pb, f"pass_b missing {fld}"
    assert int(pb["cx_last"].shape[0]) == N1M.N_PASS_B, pb["cx_last"].shape[0]
    pb_X = EA.N50._slice_layer(pb, "cx_last", layer)
    pb_Y = EA.N50._slice_layer(pb, "v_x", layer)
    del pb
    gc.collect()
    manifest_args = argparse.Namespace(
        out_dir=ns.out_dir, manifest_from_hf=ns.manifest_from_hf, hf_prefix=ns.manifest_hf_prefix
    )
    manifest_dir = EA.N1G._resolve_manifest_dir(manifest_args)
    pool, man_meta = EA.N1G.read_manifest_pool(manifest_dir)
    ci_to_corpus = {int(r["i"]): r["corpus"] for r in pool}
    new_X, new_Y, new_ci = N1M._stream_n1m_layer(
        ns.hf_prefix,
        layer,
        None,
        ns.out_dir / ".n1m_stream_cache",
        ckpt_dir=ns.out_dir / ".n1m_stream_ckpt",
        ckpt_every=N1M.STREAM_CKPT_EVERY,
        fresh=False,
    )
    new_prov = np.array([ci_to_corpus[int(c)] for c in new_ci], dtype=object)
    n_total = N1M.N_PASS_B + int(new_X.shape[0])
    assert pb_X.shape[1] == EA.C.EXPECTED_HIDDEN and new_X.shape[1] == EA.C.EXPECTED_HIDDEN
    prov = np.array(["lmsys"] * N1M.N_PASS_B + list(new_prov), dtype=object)
    assert prov.shape[0] == n_total, (prov.shape, n_total)
    # pinned val/test shas (parent asserts, verbatim)
    pinned = EA.N50._pinned_original_shas(ns.orig_dir)
    r1_train, val, test = EA.F.fixed_split(
        N1M.N_PASS_B, N1M.N_PASS_B - N1M.N_VAL - N1M.N_TEST, N1M.N_VAL, N1M.N_TEST, N1M.SPLIT_SEED
    )
    val_sha, test_sha = EA.F._sha_ids(val), EA.F._sha_ids(test)
    assert val_sha == pinned["val_sha256"], (val_sha, pinned["val_sha256"])
    assert test_sha == pinned["test_sha256"], (test_sha, pinned["test_sha256"])
    assert (val < N1M.N_PASS_B).all() and (test < N1M.N_PASS_B).all()
    committed = json.loads(EA.COMMITTED_SPLIT_1482.read_text())
    assert n_total == int(committed["n_total"]) == SUB.N_TOTAL_PROD, (n_total, committed["n_total"])
    prov_u8 = (prov == "wildchat").astype(np.uint8)
    np.save(SCRATCH / "prov.npy", prov_u8)
    pools = N1M._pool_rows(prov, r1_train, n_total, val, test)
    train_full = pools["full"]
    lmsys_frac = len(pools["lmsys"]) / len(train_full)
    new_rows = np.arange(N1M.N_PASS_B, n_total)
    rng = np.random.default_rng(EA.SPLIT_SEED_1482)
    holdout, _ = EA._stratified_sample(rng, new_rows, prov_u8, 20_000, lmsys_frac)
    remaining = np.setdiff1d(new_rows, holdout, assume_unique=False)
    sae_fit, _ = EA._stratified_sample(rng, remaining, prov_u8, 120_000, lmsys_frac)
    remaining2 = np.setdiff1d(remaining, sae_fit, assume_unique=False)
    sae_val, _ = EA._stratified_sample(rng, remaining2, prov_u8, 2_000, lmsys_frac)
    got = {
        "train_full_sha256": EA._sha_ids(train_full),
        "holdout_sha256": EA._sha_ids(holdout),
        "sae_fit_sha256": EA._sha_ids(sae_fit),
        "sae_val_sha256": EA._sha_ids(sae_val),
    }
    assert got == SUB.committed_split_shas(), f"G1 HALT: split shas diverge: {got}"
    log("assemble_light: G1 PASS (4 split shas match committed split_1482.json)")
    np.savez(
        OUT_ROOT / "split_indices.npz",
        train_full=train_full,
        holdout=holdout,
        sae_fit=sae_fit,
        sae_val=sae_val,
    )
    # indexed gather of the design rows (replaces the OOM-peak concatenate)
    pos_sel = np.concatenate([sae_fit, sae_val, holdout])
    is_pb = pos_sel < N1M.N_PASS_B
    X_sub = np.empty((len(pos_sel), EA.C.EXPECTED_HIDDEN), dtype=np.float32)
    Y_sub = np.empty_like(X_sub)
    X_sub[is_pb] = pb_X[pos_sel[is_pb]]
    Y_sub[is_pb] = pb_Y[pos_sel[is_pb]]
    nb = pos_sel[~is_pb] - N1M.N_PASS_B
    X_sub[~is_pb] = new_X[nb]
    Y_sub[~is_pb] = new_Y[nb]
    del new_X, new_Y, pb_X, pb_Y
    gc.collect()
    np.save(exp / "X_sub.npy", X_sub)
    np.save(exp / "Y_sub.npy", Y_sub)
    np.save(exp / "pos_sel.npy", pos_sel)
    np.savez(exp / "design_meta.npz", n_tr=len(sae_fit), n_va=len(sae_val), n_te=len(holdout))
    import shutil

    shutil.copy2(OUT_ROOT / "split_indices.npz", exp / "split_indices.npz")
    mark_done("export_design", {"n": int(X_sub.shape[0]), "via": "assemble_light"})
    mark_done(
        "assemble_light",
        {"n_total": n_total, "g1": "PASS", "n_design": int(X_sub.shape[0])},
    )


def phase_assemble() -> None:
    if is_done("assemble"):
        log("assemble: resume-skip")
        return
    SCRATCH.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(
        pass_b=PASS_B,
        out_dir=SCRATCH,
        manifest_from_hf=True,
        hf_prefix=EA.CAPTURE_PREFIX,
        manifest_hf_prefix=EA.N1G.HF_PREFIX,
        n1m_capture_dir=None,
        fresh_stream=False,
        orig_dir=N1M.DEFAULT_ORIG_DIR,
    )
    X, Y, prov, r1_train, val, test, split, new_ci = EA._assemble_with_ci(ns, SUB.LAYER)
    n_total = int(X.shape[0])
    committed = json.loads(EA.COMMITTED_SPLIT_1482.read_text())
    assert n_total == int(committed["n_total"]) == SUB.N_TOTAL_PROD, (n_total, committed["n_total"])
    np.save(SCRATCH / "X.npy", X)
    np.save(SCRATCH / "Y.npy", Y)
    prov_u8 = (prov == "wildchat").astype(np.uint8)
    np.save(SCRATCH / "prov.npy", prov_u8)
    pools = N1M._pool_rows(prov, r1_train, n_total, val, test)
    train_full = pools["full"]
    lmsys_frac = len(pools["lmsys"]) / len(train_full)
    new_rows = np.arange(N1M.N_PASS_B, n_total)
    rng = np.random.default_rng(EA.SPLIT_SEED_1482)
    holdout, _ = EA._stratified_sample(rng, new_rows, prov_u8, 20_000, lmsys_frac)
    remaining = np.setdiff1d(new_rows, holdout, assume_unique=False)
    sae_fit, _ = EA._stratified_sample(rng, remaining, prov_u8, 120_000, lmsys_frac)
    remaining2 = np.setdiff1d(remaining, sae_fit, assume_unique=False)
    sae_val, _ = EA._stratified_sample(rng, remaining2, prov_u8, 2_000, lmsys_frac)
    got = {
        "train_full_sha256": EA._sha_ids(train_full),
        "holdout_sha256": EA._sha_ids(holdout),
        "sae_fit_sha256": EA._sha_ids(sae_fit),
        "sae_val_sha256": EA._sha_ids(sae_val),
    }
    exp = SUB.committed_split_shas()
    assert got == exp, f"G1 HALT: split shas diverge from committed: {got} vs {exp}"
    np.savez(
        OUT_ROOT / "split_indices.npz",
        train_full=train_full,
        holdout=holdout,
        sae_fit=sae_fit,
        sae_val=sae_val,
    )
    del X, Y
    gc.collect()
    mark_done("assemble", {"n_total": n_total, "g1": "PASS"})


# ── phase 2: covariance eigenbases (context Q_c, answer Q_a; sae_fit rows) ───────


def _stream_cov_eigh(mm: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H = mm.shape[1]
    A = torch.zeros((H, H), dtype=torch.float64, device=DEVICE)
    mu_acc = torch.zeros(H, dtype=torch.float64, device=DEVICE)
    n_acc = 0
    for i, s in enumerate(range(0, len(rows), BLOCK)):
        b = torch.as_tensor(np.asarray(mm[rows[s : s + BLOCK]], dtype=np.float64), device=DEVICE)
        A += b.T @ b
        mu_acc += b.sum(0)
        n_acc += b.shape[0]
        if i % 8 == 0:
            log(f"cov: block {i + 1}/{(len(rows) + BLOCK - 1) // BLOCK}")
    mu = mu_acc / n_acc
    A = A / n_acc - torch.outer(mu, mu)
    evals, evecs = SUB._eigh_desc(A)
    return evals.cpu().numpy(), evecs.cpu().numpy().astype(np.float32)


def phase_export_design() -> None:
    """VM-side: persist the 142k design rows so a GPU pod can run every later
    phase without the 27.7 GB scratch assembly (addendum: wall-clock pivot)."""
    if is_done("export_design"):
        log("export_design: resume-skip")
        return
    d = _design()
    exp = OUT_ROOT / "design_export"
    exp.mkdir(exist_ok=True)
    np.save(exp / "X_sub.npy", d["X"])
    np.save(exp / "Y_sub.npy", d["Y"])
    np.save(exp / "pos_sel.npy", d["pos"])
    np.savez(
        exp / "design_meta.npz",
        n_tr=len(d["tr"]),
        n_va=len(d["va"]),
        n_te=len(d["te"]),
    )
    import shutil

    shutil.copy2(OUT_ROOT / "split_indices.npz", exp / "split_indices.npz")
    mark_done("export_design", {"n": int(d["X"].shape[0])})


def phase_cov() -> None:
    if is_done("cov"):
        log("cov: resume-skip")
        return
    exp = OUT_ROOT / "design_export"
    if (exp / "X_sub.npy").exists():
        d = _design()
        Xmm, Ymm = d["X"], d["Y"]
        fit_rows = d["tr"]
    else:
        sp = np.load(OUT_ROOT / "split_indices.npz")
        fit_rows = np.sort(sp["sae_fit"])
        Xmm = np.load(SCRATCH / "X.npy", mmap_mode="r")
        Ymm = np.load(SCRATCH / "Y.npy", mmap_mode="r")
    log("cov: context covariance (Q_c)")
    evals_c, Qc = _stream_cov_eigh(Xmm, fit_rows)
    log("cov: answer covariance (Q_a)")
    evals_a, Qa = _stream_cov_eigh(Ymm, fit_rows)
    banked = np.asarray(json.loads(EA.COMMITTED_PERDIR_PCA.read_text())["eigvals_top"], np.float64)
    reldev = float(
        np.max(np.abs(evals_a[: len(banked)] - banked) / np.maximum(np.abs(banked), 1e-12))
    )
    log(
        f"cov: Q_a vs banked train_full eigvals reldev {reldev:.3e} (informational — "
        "Q_a here is sae_fit-based by stated deviation)"
    )
    np.savez(OUT_ROOT / "qc_basis.npz", Q=Qc, eigvals=evals_c)
    np.savez(OUT_ROOT / "qa_basis.npz", Q=Qa, eigvals=evals_a)
    mark_done("cov", {"qa_vs_banked_reldev_informational": reldev})


# ── shared design ────────────────────────────────────────────────────────────────


def _design() -> dict:
    exp = OUT_ROOT / "design_export"
    if (exp / "X_sub.npy").exists() and (exp / "design_meta.npz").exists():
        X_sub = np.load(exp / "X_sub.npy")
        Y_sub = np.load(exp / "Y_sub.npy")
        pos_sel = np.load(exp / "pos_sel.npy")
        with np.load(exp / "design_meta.npz") as z:
            n_tr, n_va, n_te = int(z["n_tr"]), int(z["n_va"]), int(z["n_te"])
    else:
        sp = np.load(OUT_ROOT / "split_indices.npz")
        tr_pos, va_pos, te_pos = sp["sae_fit"], sp["sae_val"], sp["holdout"]
        pos_sel = np.concatenate([tr_pos, va_pos, te_pos])
        Xmm = np.load(SCRATCH / "X.npy", mmap_mode="r")
        Ymm = np.load(SCRATCH / "Y.npy", mmap_mode="r")
        X_sub = np.asarray(Xmm[pos_sel], dtype=np.float32)
        Y_sub = np.asarray(Ymm[pos_sel], dtype=np.float32)
        n_tr, n_va = len(tr_pos), len(va_pos)
        n_te = len(pos_sel) - n_tr - n_va
    tr = np.arange(n_tr)
    va = np.arange(n_tr, n_tr + n_va)
    te = np.arange(n_tr + n_va, n_tr + n_va + n_te)
    assert len(tr) >= X_sub.shape[1], (len(tr), X_sub.shape[1])
    return {"X": X_sub, "Y": Y_sub, "tr": tr, "va": va, "te": te, "pos": pos_sel}


def _fit_ridge(Z: np.ndarray, Y: np.ndarray, tr, va, te) -> tuple[np.ndarray, dict, dict]:
    """Parent-helper ridge: val-selected lambda over the banked grid; returns
    (te predictions, meta, factorization)."""
    fac = N1M._ridge_factorize(Z, Y, tr, DEVICE, BLOCK)
    best = (float(SUB.LAMBDAS[0]), -np.inf)
    for lam in SUB.LAMBDAS:
        pv = N1M._ridge_predict_one(Z, va, fac, lam, DEVICE, BLOCK)
        r2 = PR._pooled_r2(pv, Y[va])
        if np.isfinite(r2) and r2 > best[1]:
            best = (float(lam), float(r2))
    lam = best[0]
    pt = N1M._ridge_predict_one(Z, te, fac, lam, DEVICE, BLOCK)
    pooled = float(PR._pooled_r2(pt, Y[te]))
    return pt, {"selected_lambda": lam, "val_r2": best[1], "pooled_r2_te": pooled}, fac


def phase_fit_dense() -> None:
    if is_done("fit_dense"):
        log("fit_dense: resume-skip")
        return
    d = _design()
    log(f"fit_dense: ridge on dense v_C (n_tr={len(d['tr'])}, d={d['X'].shape[1]})")
    pt, meta, fac = _fit_ridge(d["X"], d["Y"], d["tr"], d["va"], d["te"])
    log(
        f"fit_dense: selected_lambda={meta['selected_lambda']} pooled_te={meta['pooled_r2_te']:.6f}"
    )
    dev = abs(meta["pooled_r2_te"] - BANKED_POOLED_R2)
    assert dev < GATE_TOL, (
        f"GATE HALT: dense-map pooled R2 {meta['pooled_r2_te']:.6f} deviates {dev:.2e} "
        f"from banked {BANKED_POOLED_R2} (tol {GATE_TOL})"
    )
    with np.load(OUT_ROOT / "qa_basis.npz") as z:
        Qa = z["Q"].astype(np.float32)
    r2_dense = EA._per_feature_metrics(pt @ Qa, d["Y"][d["te"]] @ Qa)["r2"]
    # raw-space operator + read profiles in the Q_c basis
    U, s_eig, UtXtY = fac["U"], fac["s_eig"], fac["UtXtY"]
    W_std = U @ (UtXtY / (s_eig + meta["selected_lambda"])[:, None])
    W_raw = (W_std / fac["xsd"][:, None]).cpu().numpy()
    with np.load(OUT_ROOT / "qc_basis.npz") as z:
        Qc, evals_c = z["Q"].astype(np.float64), z["eigvals"]
    M = Qc.T @ W_raw
    gain = (M**2).sum(axis=1)
    read_energy = evals_c * gain
    # per-context normalized error of the dense map (for the leg-B per-context gap)
    err_dense = ((d["Y"][d["te"]] - pt) ** 2).sum(axis=1)
    np.savez(
        OUT_ROOT / "fit_dense.npz",
        r2_perdir_qa=r2_dense.astype(np.float32),
        read_energy=read_energy.astype(np.float64),
        gain=gain.astype(np.float64),
        err_dense=err_dense.astype(np.float32),
    )
    (OUT_ROOT / "fit_dense_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    mark_done("fit_dense", {**meta, "gate": "PASS", "gate_dev": dev})


# ── phase 4: SAE encode (all selected rows) + te decode ──────────────────────────


def phase_sae_encode() -> None:
    if is_done("sae_encode"):
        log("sae_encode: resume-skip")
        return
    SAE_DIR.mkdir(parents=True, exist_ok=True)
    sae = S.BatchTopKSAE.load(k=64, device=DEVICE, cache_dir=SAE_DIR)
    d = _design()
    X = d["X"]
    n = X.shape[0]
    parts_dir = OUT_ROOT / "code_parts"
    parts_dir.mkdir(exist_ok=True)
    fire_counts = np.zeros(sae.dict_size, dtype=np.int64)
    te_start = int(d["te"][0])
    recon_te = np.zeros((len(d["te"]), X.shape[1]), dtype=np.float32)
    chunk = 2048
    n_chunks = (n + chunk - 1) // chunk
    for i, s in enumerate(range(0, n, chunk)):
        part = parts_dir / f"part_{i:05d}.npz"
        e = min(s + chunk, n)
        if part.exists():
            with np.load(part) as z:
                if s < len(d["tr"]):  # re-accumulate fit-row fire counts on resume
                    lim = min(e, len(d["tr"])) - s
                    mask = z["rows"] < lim
                    np.add.at(fire_counts, z["idx"][mask], 1)
                if e > te_start:
                    f = torch.zeros((e - s, sae.dict_size), dtype=torch.float32, device=DEVICE)
                    f[
                        torch.as_tensor(z["rows"].astype(np.int64), device=DEVICE),
                        torch.as_tensor(z["idx"].astype(np.int64), device=DEVICE),
                    ] = torch.as_tensor(z["vals"], dtype=torch.float32, device=DEVICE)
                    r = sae.decode(f).cpu().numpy()
                    a = max(s, te_start)
                    recon_te[a - te_start : e - te_start] = r[a - s :]
            continue
        t0 = time.time()
        f = sae.encode(torch.as_tensor(X[s:e]))
        rows, idx = torch.nonzero(f, as_tuple=True)
        vals = f[rows, idx]
        if s < len(d["tr"]):
            lim = min(e, len(d["tr"])) - s
            m = rows < lim
            np.add.at(fire_counts, idx[m].cpu().numpy(), 1)
        if e > te_start:
            r = sae.decode(f).cpu().numpy()
            a = max(s, te_start)
            recon_te[a - te_start : e - te_start] = r[a - s :]
        tmp = part.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            rows=rows.cpu().numpy().astype(np.int32),
            idx=idx.cpu().numpy().astype(np.int32),
            vals=vals.cpu().numpy().astype(np.float32),
        )
        tmp.replace(part)
        log(f"sae_encode: chunk {i + 1}/{n_chunks} rows={e - s} elapsed={time.time() - t0:.1f}s")
        del f
    np.save(OUT_ROOT / "fire_counts_fit.npy", fire_counts)
    np.save(OUT_ROOT / "recon_te.npy", recon_te)
    mark_done("sae_encode", {"n_rows": int(n), "n_chunks": n_chunks})


# ── phase 5: SAE-input ridge (leg B) ─────────────────────────────────────────────


def phase_fit_saein() -> None:
    if is_done("fit_saein"):
        log("fit_saein: resume-skip")
        return
    d = _design()
    fire = np.load(OUT_ROOT / "fire_counts_fit.npy")
    panel = np.argsort(-fire)[:PANEL_F_IN]
    col_of = np.full(len(fire), -1, dtype=np.int64)
    col_of[panel] = np.arange(PANEL_F_IN)
    n = d["X"].shape[0]
    Zp = np.zeros((n, PANEL_F_IN), dtype=np.float32)
    parts = sorted((OUT_ROOT / "code_parts").glob("part_*.npz"))
    chunk = 2048
    for i, part in enumerate(parts):
        with np.load(part) as z:
            cols = col_of[z["idx"]]
            keep = cols >= 0
            Zp[z["rows"][keep] + i * chunk, cols[keep]] = z["vals"][keep]
    log(f"fit_saein: panel matrix built ({Zp.shape}); ridge (n_tr={len(d['tr'])}, d={PANEL_F_IN})")
    pt, meta, fac = _fit_ridge(Zp, d["Y"], d["tr"], d["va"], d["te"])
    log(
        f"fit_saein: selected_lambda={meta['selected_lambda']} pooled_te={meta['pooled_r2_te']:.6f}"
    )
    with np.load(OUT_ROOT / "qa_basis.npz") as z:
        Qa = z["Q"].astype(np.float32)
    r2_saein = EA._per_feature_metrics(pt @ Qa, d["Y"][d["te"]] @ Qa)["r2"]
    err_saein = ((d["Y"][d["te"]] - pt) ** 2).sum(axis=1)
    np.savez(
        OUT_ROOT / "fit_saein.npz",
        r2_perdir_qa=r2_saein.astype(np.float32),
        err_saein=err_saein.astype(np.float32),
        panel=panel.astype(np.int64),
    )
    (OUT_ROOT / "fit_saein_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    mark_done("fit_saein", meta)


# ── phase 6: angle battery (leg A) ───────────────────────────────────────────────


def phase_angles() -> None:
    if is_done("angles"):
        log("angles: resume-skip")
        return
    d = _design()
    with np.load(OUT_ROOT / "qc_basis.npz") as z:
        Qc, evals_c = z["Q"].astype(np.float32), z["eigvals"]
    X_te = d["X"][d["te"]]
    R_te = np.load(OUT_ROOT / "recon_te.npy")
    E_te = X_te - R_te
    ctx_fve = 1.0 - float(((E_te) ** 2).sum()) / float(((X_te - X_te.mean(0)) ** 2).sum())
    log(f"angles: context-state SAE reconstruction FVE (te pool) = {ctx_fve:.4f}")
    bases = {
        "recon_pca": SUB._pca_basis(R_te @ Qc, max(K_GRID), DEVICE),
        "resid_pca": SUB._pca_basis(E_te @ Qc, max(K_GRID), DEVICE),
    }
    with np.load(OUT_ROOT / "fit_dense.npz") as z:
        profiles = {"read_energy": z["read_energy"], "gain": z["gain"]}
    cells = []
    cell_i = 0
    for pname, prof in profiles.items():
        order = np.argsort(-prof)
        for k in K_GRID:
            s_pred = order[:k]
            for bname, B in bases.items():
                Bk = B[:, : min(k, B.shape[1])]
                cs, obs = SUB.overlap_observed(s_pred, Bk)
                cell = {
                    "profile": pname,
                    "pair": bname,
                    "k": int(k),
                    "observed_O": float(obs),
                    "nulls": {},
                }
                if pname == "read_energy":
                    for n_sh in SHELL_GRID:
                        shells = SUB.shell_partition(evals_c, n_sh)
                        t0 = time.time()
                        draws = SUB.overlap_null_draws(
                            s_pred, Bk, shells, N_DRAWS, SEED + cell_i, DEVICE
                        )
                        q = float(
                            (np.sum(draws < obs) + 0.5 * np.sum(draws == obs)) / len(draws) * 100
                        )
                        cell["nulls"][str(n_sh)] = {
                            "p2.5": float(np.percentile(draws, 2.5)),
                            "p50": float(np.percentile(draws, 50)),
                            "p97.5": float(np.percentile(draws, 97.5)),
                            "q_percentile_of_observed": q,
                            "n_draws": N_DRAWS,
                        }
                        log(
                            f"angles: cell {pname}/{bname}/k{k}/sh{n_sh} obs={obs:.4f} "
                            f"q={q:.1f}% elapsed={time.time() - t0:.1f}s"
                        )
                cell_i += 1
                cells.append(cell)
    (OUT_ROOT / "angles.json").write_text(
        json.dumps({"cells": cells, "ctx_recon_fve_te": ctx_fve}, indent=2) + "\n"
    )
    mark_done("angles", {"n_cells": len(cells), "ctx_recon_fve_te": ctx_fve})


# ── phase 7: summary + figures ───────────────────────────────────────────────────


def phase_summary() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import set_paper_style

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(OUT_ROOT / "fit_dense.npz") as z:
        r2_dense, err_dense = z["r2_perdir_qa"], z["err_dense"]
    with np.load(OUT_ROOT / "fit_saein.npz") as z:
        r2_saein, err_saein = z["r2_perdir_qa"], z["err_saein"]
    with np.load(OUT_ROOT / "qa_basis.npz") as z:
        evals_a = z["eigvals"]
    angles = json.loads((OUT_ROOT / "angles.json").read_text())
    meta_dense = json.loads((OUT_ROOT / "fit_dense_meta.json").read_text())
    meta_saein = json.loads((OUT_ROOT / "fit_saein_meta.json").read_text())
    fair_grid = None
    if (OUT_ROOT / "fit_rbar_summary.json").exists():
        fair_grid = json.loads((OUT_ROOT / "fit_rbar_summary.json").read_text())
    gap = r2_dense - r2_saein
    ratio = err_saein / np.maximum(err_dense, 1e-12)
    summary = {
        "leg_b": {
            "pooled_r2_dense_input": meta_dense["pooled_r2_te"],
            "pooled_r2_sae_input": meta_saein["pooled_r2_te"],
            "selected_lambda_dense": meta_dense["selected_lambda"],
            "selected_lambda_sae_input": meta_saein["selected_lambda"],
            "banked_bridge_refs": {"dense_to_dense_120k": 0.7166, "sae_ctx_to_dense": 0.6606},
            "gap_perdir_top16_mean": float(gap[:16].mean()),
            "gap_perdir_rank17_64_mean": float(gap[16:64].mean()),
            "gap_perdir_rank65_256_mean": float(gap[64:256].mean()),
            "gap_perdir_rank257_1024_mean": float(gap[256:1024].mean()),
            "per_context_err_ratio_quantiles": {
                q: float(np.quantile(ratio, float(q)))
                for q in ("0.1", "0.25", "0.5", "0.75", "0.9")
            },
            "panel_f_in": PANEL_F_IN,
        },
        "leg_b_fair_grid": fair_grid,
        "leg_a": angles,
        "gate": json.loads(done_path("fit_dense").read_text()),
        "design_notes": [
            "Q_a/Q_c eigenbases computed over the 120k sae_fit rows (stated deviation: "
            "the parent's Q used train_full).",
            "Context arm only: the prefix arm is a registered null by construction on "
            "this single-turn corpus (banked #1895 fits: prefix R2 <= 0.0002).",
            "SAE-input panel = top-8192 context-active features by fit-row firing rate "
            "(mirrors the banked bridge f_in=8192); codes fresh-encoded (BatchTopK k=64).",
        ],
    }
    (EVAL_DIR / "input_side_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    set_paper_style()
    # Fig B: per-direction R2, dense-input vs SAE-input
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ranks = np.arange(1, len(r2_dense) + 1)
    ax.plot(
        ranks,
        r2_dense,
        color="#0072B2",
        linewidth=1.0,
        alpha=0.85,
        label=f"dense-input map (pooled $R^2$={meta_dense['pooled_r2_te']:.3f})",
    )
    ax.plot(
        ranks,
        r2_saein,
        color="#D55E00",
        linewidth=1.0,
        alpha=0.85,
        label=f"SAE-input map (pooled $R^2$={meta_saein['pooled_r2_te']:.3f})",
    )
    ax.set_xscale("log")
    ax.set_xlabel("answer-PCA direction rank (by variance)")
    ax.set_ylabel("held-out per-direction $R^2$")
    ax.set_ylim(-0.5, 1.0)
    ax.set_title(
        "Input bottleneck: predicting the mean answer state from $v_C$\n"
        "vs from SAE($v_C$) codes (same target, same fit recipe)"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"input_side_gap_perdirection.{ext}", dpi=200, bbox_inches="tight")
    (FIG_DIR / "input_side_gap_perdirection.meta.json").write_text(
        json.dumps(
            {
                "source": "eval_results/issue_1895/input_side/input_side_summary.json",
                "what_is_plotted": (
                    "Held-out per-direction R2 over answer-PCA directions (variance rank, "
                    "log x) for the dense-input map (v_C -> v_A) vs the SAE-input map "
                    "(SAE(v_C) top-8192 codes -> v_A); same 120k-fit/20k-holdout split, "
                    "same val-selected ridge recipe."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    plt.close(fig)

    # Fig A: overlap k-sweep, read-energy profile, recon pair, 64-shell band
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    cells = [
        c
        for c in angles["cells"]
        if c["profile"] == "read_energy" and c["pair"] == "recon_pca" and c["nulls"]
    ]
    cells = sorted(cells, key=lambda c: c["k"])
    ks = [c["k"] for c in cells]
    obs = [c["observed_O"] for c in cells]
    fine = [c["nulls"]["64"] for c in cells]
    ax.fill_between(
        ks,
        [n["p2.5"] for n in fine],
        [n["p97.5"] for n in fine],
        color="0.75",
        alpha=0.55,
        linewidth=0,
        label="variance-matched null, central 95% of 1,000 draws (64 shells)",
    )
    ax.plot(
        ks,
        [n["p50"] for n in fine],
        color="0.35",
        linestyle="--",
        linewidth=1.6,
        label="null median (64 shells)",
    )
    ax.plot(
        ks,
        obs,
        color="#0072B2",
        marker="o",
        markersize=5.5,
        linewidth=2.0,
        label="observed overlap",
        zorder=5,
    )
    for k, o, n in zip(ks, obs, fine):
        ax.annotate(
            f"{n['q_percentile_of_observed']:.0f}%",
            (k, o),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color="#0072B2",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("subspace size $k$ (top-$k$ directions of each ranking)")
    ax.set_ylabel(r"subspace overlap  $O(k)=\overline{\cos^2\theta}$")
    ax.set_title(
        "INPUT side: map read-energy subspace vs SAE-representable context subspace\n"
        "(outside the band = inconsistent with variance alone at the 5% level)"
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(
            FIG_DIR / f"input_side_overlap_ksweep_64shell.{ext}", dpi=200, bbox_inches="tight"
        )
    (FIG_DIR / "input_side_overlap_ksweep_64shell.meta.json").write_text(
        json.dumps(
            {
                "source": "eval_results/issue_1895/input_side/input_side_summary.json",
                "k": ks,
                "observed_O": obs,
                "what_is_plotted": (
                    "Per k: observed mean cos^2 principal angle between the dense map's "
                    "top-k read-energy input eigendirections and the SAE reconstruction-PCA "
                    "top-k subspace of the context states, vs the finest (64-shell) "
                    "variance-matched rotation null (median + central-95% band)."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    plt.close(fig)
    # Fig C: reconstruction fair grid — per-direction R2 for the 2x2 (+ residual input)
    if fair_grid is not None and (OUT_ROOT / "fit_rbar.npz").exists():
        with np.load(OUT_ROOT / "fit_rbar.npz") as z:
            prof = {k: z[k] for k in z.files}
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        series = (
            ("vC__to__vA", "#0072B2", "-", "$v_C \\to v_A$"),
            ("rbarC__to__vA", "#D55E00", "-", "$\\bar r_C \\to v_A$"),
            ("vC__to__rbarA", "#0072B2", "--", "$v_C \\to \\bar r_A$"),
            ("rbarC__to__rbarA", "#D55E00", "--", "$\\bar r_C \\to \\bar r_A$"),
            ("eC__to__vA", "0.45", "-", "$e_C \\to v_A$ (SAE-missed input)"),
        )
        ranks = np.arange(1, len(next(iter(prof.values()))) + 1)
        for key, color, ls, label in series:
            if key in prof:
                pooled = fair_grid["cells"][key]["pooled_r2_te"]
                ax.plot(
                    ranks,
                    prof[key],
                    color=color,
                    linestyle=ls,
                    linewidth=1.0,
                    alpha=0.85,
                    label=f"{label}  (pooled {pooled:.3f})",
                )
        ax.set_xscale("log")
        ax.set_xlabel("answer-PCA direction rank (by variance)")
        ax.set_ylabel("held-out per-direction $R^2$")
        ax.set_ylim(-0.5, 1.0)
        ax.set_title(
            "Fair reconstruction grid: full states vs SAE-representable components\n"
            "(all cells dense-space, same kept rows, same val-selected ridge recipe)"
        )
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"input_side_fair_grid.{ext}", dpi=200, bbox_inches="tight")
        (FIG_DIR / "input_side_fair_grid.meta.json").write_text(
            json.dumps(
                {
                    "source": "eval_results/issue_1895/input_side/input_side_summary.json",
                    "what_is_plotted": (
                        "Held-out per-direction R2 over answer-PCA directions (variance "
                        "rank, log x) for the reconstruction fair grid: inputs {v_C, "
                        "rbar_C = decode(SAE(v_C)), e_C = v_C - rbar_C} x targets {v_A, "
                        "rbar_A = decode(pooled answer SAE codes)}; all cells dense "
                        "3584-dim space, identical kept rows (ans_all_out excluded), "
                        "identical val-selected ridge recipe."
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        plt.close(fig)
    log("summary: wrote input_side_summary.json + figures")
    mark_done("summary", {"eval": str(EVAL_DIR / "input_side_summary.json")})


# ── phases 5b/5c: reconstruction fair grid (addendum 2) ─────────────────────────

STORE_DIR = OUT_ROOT / "store_dl" / "issue1482_error_analysis" / "analysis_tensors" / "sae_pooled"
N_STORE_SHARDS = 1920
BANKED_RBAR_POOLED_R2 = 0.745688  # fits_summary.json t_rbar_ctx


def _sparse_decode(
    sae, row_ids: np.ndarray, idx: np.ndarray, vals: np.ndarray, n: int
) -> np.ndarray:
    """decode(sparse codes) = f @ w_dec.T + b_dec via a SPARSE matmul.

    torch.sparse.mm never materializes the (nnz, act_dim) per-entry intermediate
    an index_add formulation would (2M nnz x 3584 fp32 = 28.7 GB — the shape that
    OOM-killed the first attempt's sibling path); peak here is the COO tensor
    (~20 B/nnz) plus the (n, act_dim) output."""
    i = torch.stack(
        [
            torch.as_tensor(row_ids.astype(np.int64), device=DEVICE),
            torch.as_tensor(idx.astype(np.int64), device=DEVICE),
        ]
    )
    f = torch.sparse_coo_tensor(
        i,
        torch.as_tensor(vals.astype(np.float32), device=DEVICE),
        (n, sae.dict_size),
    ).coalesce()
    out = torch.sparse.mm(f, sae.w_dec.T)
    return (out + sae.b_dec).cpu().numpy()


def phase_rbar_a() -> None:
    """Store -> rbar_a + aao. VM-runnable right after assemble (needs only the
    staged pooled store + split indices + SAE weights) so the GPU pod never
    stages the 9.8 GB store."""
    if is_done("rbar_a"):
        log("rbar_a: resume-skip")
        return
    shards = sorted(STORE_DIR.glob("pooled_*.npz"))
    assert len(shards) == N_STORE_SHARDS, (
        f"pooled store staging incomplete: {len(shards)}/{N_STORE_SHARDS} shards under {STORE_DIR}"
    )
    sae = S.BatchTopKSAE.load(k=64, device=DEVICE, cache_dir=SAE_DIR)
    sp = np.load(OUT_ROOT / "split_indices.npz")
    pos_sel = np.concatenate([sp["sae_fit"], sp["sae_val"], sp["holdout"]])
    n_design = len(pos_sel)
    pos_of = {}
    for i, p in enumerate(pos_sel):
        pos_of[int(p)] = i
    rbar_a = np.zeros((n_design, sae.act_dim), dtype=np.float32)
    aao = np.full(n_design, -1, dtype=np.int8)
    got = np.zeros(n_design, dtype=bool)
    # vectorized store-row -> design-row lookup (a python dict-get per row cost
    # ~1M interpreter round-trips across the 1920 shards)
    order = np.argsort(pos_sel)
    pos_sorted = pos_sel[order]
    t_start = time.time()
    for si, shard in enumerate(shards):
        with np.load(shard) as z:
            row_idx = z["row_idx"].astype(np.int64)
            ins = np.searchsorted(pos_sorted, row_idx)
            ins_c = np.clip(ins, 0, len(pos_sorted) - 1)
            m = pos_sorted[ins_c] == row_idx
            if not m.any():
                continue
            hit = order[ins_c]
            # store schema: idx_off holds the per-row COUNT (builder appends
            # len(sp["idx"])); the parent recovers offsets by cumsum
            # (issue1482_error_analysis.py:1841).
            counts = z["idx_off"].astype(np.int64)
            assert int(counts.sum()) == len(z["ans_idx"]), (counts.sum(), len(z["ans_idx"]))
            # decode ONLY the design rows of this shard (~15% of them): filter the
            # COO entries to those rows and compact-remap their row ids, so the
            # sparse mm does no work for rows nothing downstream reads.
            offs = np.concatenate([[0], np.cumsum(counts)])
            sel_rows = np.flatnonzero(m)
            keep = np.zeros(len(z["ans_idx"]), dtype=bool)
            for r in sel_rows:
                keep[offs[r] : offs[r + 1]] = True
            compact = np.full(len(row_idx), -1, dtype=np.int64)
            compact[sel_rows] = np.arange(len(sel_rows))
            row_ids_full = np.repeat(np.arange(len(row_idx)), counts)
            dec = _sparse_decode(
                sae,
                compact[row_ids_full[keep]],
                z["ans_idx"][keep],
                z["ans_mean"][keep],
                len(sel_rows),
            )
            dec = dec.astype(np.float16).astype(np.float32)  # parent stores rbar fp16
            rbar_a[hit[m]] = dec
            aao[hit[m]] = z["ans_all_out"][m]
            got[hit[m]] = True
        if si % 200 == 0:
            log(
                f"rbar_a: shard {si + 1}/{len(shards)} kept={int(got.sum())} "
                f"elapsed={time.time() - t_start:.0f}s"
            )
    assert got.all(), f"rbar_a: {int((~got).sum())} design rows missing from pooled store"
    np.save(OUT_ROOT / "rbar_a.npy", rbar_a)
    np.save(OUT_ROOT / "aao.npy", aao)
    del rbar_a
    gc.collect()
    mark_done("rbar_a", {"n_design": int(n_design), "n_ans_all_out": int((aao == 1).sum())})


def phase_build_rbar() -> None:
    """r_bar_C from the persisted code parts (chunk i covers design rows i*2048..)."""
    if is_done("build_rbar"):
        log("build_rbar: resume-skip")
        return
    sae = S.BatchTopKSAE.load(k=64, device=DEVICE, cache_dir=SAE_DIR)
    d = _design()
    n = d["X"].shape[0]
    rbar_c = np.zeros((n, sae.act_dim), dtype=np.float32)
    parts = sorted((OUT_ROOT / "code_parts").glob("part_*.npz"))
    chunk = 2048
    for i, part in enumerate(parts):
        with np.load(part) as z:
            n_rows = min(chunk, n - i * chunk)
            dec = _sparse_decode(sae, z["rows"], z["idx"], z["vals"], n_rows)
            rbar_c[i * chunk : i * chunk + n_rows] = dec
    np.save(OUT_ROOT / "rbar_c.npy", rbar_c)
    mark_done("build_rbar", {"n_rows": int(n)})


def phase_fit_rbar() -> None:
    if is_done("fit_rbar"):
        log("fit_rbar: resume-skip")
        return
    d = _design()
    rbar_c = np.load(OUT_ROOT / "rbar_c.npy")
    rbar_a = np.load(OUT_ROOT / "rbar_a.npy")
    aao = np.load(OUT_ROOT / "aao.npy")
    kept = aao == 0
    tr_k = d["tr"][kept[d["tr"]]]
    va_k = d["va"][kept[d["va"]]]
    te_k = d["te"][kept[d["te"]]]
    log(
        f"fit_rbar: kept tr/va/te = {len(tr_k)}/{len(va_k)}/{len(te_k)} "
        f"(ans_all_out excluded: {int((~kept).sum())})"
    )
    with np.load(OUT_ROOT / "qa_basis.npz") as z:
        Qa = z["Q"].astype(np.float32)
    e_c = d["X"] - rbar_c
    cells: dict = {}
    profiles: dict = {}
    grid = (
        ("vC", d["X"], {"vA": d["Y"], "rbarA": rbar_a}),
        ("rbarC", rbar_c, {"vA": d["Y"], "rbarA": rbar_a}),
        ("eC", e_c, {"vA": d["Y"]}),
    )
    for iname, Z, tg in grid:
        res = EA._shared_gram_ridge_multi(Z, tg, tr_k, va_k, te_k, SUB.LAMBDAS, DEVICE, BLOCK)
        for tname, (pt, meta) in res.items():
            key = f"{iname}__to__{tname}"
            true = tg[tname][te_k]
            pooled = float(PR._pooled_r2(pt, true))
            cells[key] = {**meta, "pooled_r2_te": pooled}
            profiles[key] = EA._per_feature_metrics(pt @ Qa, true @ Qa)["r2"].astype(np.float32)
            log(f"fit_rbar: {key} pooled_te={pooled:.6f} lambda={meta['selected_lambda']}")
    gate_dev = abs(cells["vC__to__rbarA"]["pooled_r2_te"] - BANKED_RBAR_POOLED_R2)
    assert gate_dev < 3e-3, (
        f"GATE HALT: vC->rbarA pooled {cells['vC__to__rbarA']['pooled_r2_te']:.6f} deviates "
        f"{gate_dev:.2e} from banked {BANKED_RBAR_POOLED_R2}"
    )
    cells["_gate_vC_rbarA_dev"] = gate_dev
    np.savez(OUT_ROOT / "fit_rbar.npz", **profiles)
    (OUT_ROOT / "fit_rbar_summary.json").write_text(
        json.dumps(
            {
                "cells": cells,
                "kept": {"tr": int(len(tr_k)), "va": int(len(va_k)), "te": int(len(te_k))},
                "n_excluded_ans_all_out": int((~kept).sum()),
            },
            indent=2,
        )
        + "\n"
    )
    # design export for the pod-side nonlinear (MLP) twin cells (addendum 3)
    exp = OUT_ROOT / "design_export"
    exp.mkdir(exist_ok=True)
    np.save(exp / "X_sub.npy", d["X"])
    np.save(exp / "Y_sub.npy", d["Y"])
    np.savez(exp / "kept_splits.npz", tr=tr_k, va=va_k, te=te_k)
    mark_done("fit_rbar", {"cells": sorted(cells)})


PHASES = {
    "assemble_light": phase_assemble_light,
    "assemble": phase_assemble,
    "export_design": phase_export_design,
    "rbar_a": phase_rbar_a,
    "cov": phase_cov,
    "fit_dense": phase_fit_dense,
    "sae_encode": phase_sae_encode,
    "fit_saein": phase_fit_saein,
    "build_rbar": phase_build_rbar,
    "fit_rbar": phase_fit_rbar,
    "angles": phase_angles,
    "summary": phase_summary,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=[*PHASES, "all"])
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    todo = list(PHASES) if args.phase == "all" else [args.phase]
    for p in todo:
        log(f"=== phase {p} ===")
        PHASES[p]()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
