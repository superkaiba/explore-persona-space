#!/usr/bin/env python3
"""Issue #922 free-analysis EXTENSION: characterize what lives in the slow shell.

Builds on ``issue922_fixed_point_slow_modes.py`` (round 1): reuses the cached
top-64 eigendecomposition per read-out block
(``/tmp/eps922fpsm_cache/ckpt/topvecs_block_<b>.npz``: eigvals, eigvecs, h*) and
the real teacher-forced test-context store, and adds:

A. PER-MODE PROFILE (all modes |lambda|>0.90, per block; complex pairs = one
   2-D real plane): |lambda|, time constant, rotation period; overlap^2 with the
   #779 trait directions, top-5 between-context PCs, mean-answer direction, and
   the fixed-point direction h* (the affine intercept a is ~parallel to h*,
   round-1 cos 0.91-0.96 — maps blob freed for disk, h* is its proxy); and the
   fraction of prompt-end BETWEEN-context variance the mode's plane captures.
B. REAL-TRAJECTORY PERSISTENCE: per slow mode, per-context mean coordinate +
   within-answer std across the stored answer positions; persistence ratio
   (between-context std / within-answer std) vs a modulus-matched random-direction
   null. (B2 trait-score correlation is SCOPE-LIMITED — see the JSON note.)
C. LOGIT-LENS DECODE (if the Qwen lm_head shard fits on disk): top-20 promoted /
   suppressed tokens for each top-10 slow mode (both signs), h*, and mean-answer.
D. CROSS-BLOCK principal angles between adjacent blocks' |lambda|>0.95 slow
   subspaces (raw 3584-dim residual stream), vs a random-subspace null.
E. ARTIFACT CHECKS: overlap of each slow mode with the global mean-state
   direction (attention-sink proxy — sequence position 0 is outside the capture
   window) and the all-ones/anisotropy direction, to rule out #744 sink
   artifacts.

All states are teacher-forced captures of the model's OWN completions
(provenance inherited from #922's lmsys_test store).
"""

from __future__ import annotations

import argparse
import gc
import glob
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue922_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue922_slow_shell")

CACHE_DIR = Path("/tmp/eps922fpsm_cache")
CKPT_DIR = CACHE_DIR / "ckpt"
REV_779 = "037fcbb210bc52c459959b0746cc268fe08bae96"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
IMAG_TOL = 1e-7
MODE_CUTOFF = 0.90  # A/B profile all modes above this
SUBSPACE_CUTOFF = 0.95  # D principal angles
N_NULL = 200
DISK_FLOOR_GB_FOR_C = 5.0  # need headroom for the ~3 GB lm_head shard


def _free_gb(path: str = "/") -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def load_block_eig(b: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(eigvals (K,) complex, eigvecs (H,K) complex, h_star (H,) real) for block b."""
    z = np.load(CKPT_DIR / f"topvecs_block_{b}.npz")
    return z[f"block{b}_eigvals"], z[f"block{b}_eigvecs"], z[f"block{b}_h_star"]


def mode_planes(evals: np.ndarray, evecs: np.ndarray, cutoff: float) -> list[dict]:
    """One entry per mode with |lambda|>cutoff (complex pair -> one 2-D real plane).

    Returns dicts with the orthonormal real basis P (H, d), lambda, is_real,
    time constant, rotation period. Dedup conjugate pairs via Im(lambda) >= 0.
    """
    out = []
    mod = np.abs(evals)
    for i in range(len(evals)):
        if mod[i] <= cutoff:
            continue
        im = float(evals[i].imag)
        if im < -IMAG_TOL:
            continue
        m = float(mod[i])
        tau = float(-1.0 / np.log(m)) if 0.0 < m < 1.0 else None
        if abs(im) <= IMAG_TOL:
            cols = np.real(evecs[:, i])[:, None]
            is_real, rot = True, None
        else:
            cols = np.stack([np.real(evecs[:, i]), np.imag(evecs[:, i])], axis=1)
            arg = abs(float(np.angle(evals[i])))
            is_real = False
            rot = float(2 * np.pi / arg) if arg > 0 else None
        q, _ = np.linalg.qr(cols)  # (H, d) orthonormal real basis of the mode plane
        out.append(
            {
                "eig_index": int(i),
                "modulus": m,
                "real_part": float(evals[i].real),
                "imag_part": im,
                "is_real": is_real,
                "tau_steps": tau,
                "rotation_period_steps": rot,
                "plane_dim": int(q.shape[1]),
                "P": q.astype(np.float64),
            }
        )
    return out


def proj_energy(P: np.ndarray, u: np.ndarray) -> float:
    """Fraction of ||u||^2 in the orthonormal subspace P (H, d)."""
    un2 = float(u @ u)
    if un2 == 0.0:
        return 0.0
    p = P.T @ u
    return float((p @ p) / un2)


def vec_null_p97(dim: int, h: int, rng: np.random.Generator, n: int = N_NULL) -> dict:
    """Random-subspace projection-energy null (exact via chi-square ratios)."""
    g = rng.standard_normal((n, h)) ** 2
    frac = g[:, :dim].sum(1) / g.sum(1)
    return {"mean": float(frac.mean()), "p97.5": float(np.percentile(frac, 97.5))}


def load_store_states(store_p: Path, blocks: list[int], h: int) -> dict:
    """Per block: prompt-end states (n_ctx, H); per-context answer states (list);
    mean-answer / mean-prompt-end / global-mean directions. fp64.
    """
    blob = torch.load(store_p, weights_only=False, mmap=True)
    assert blob.get("corpus") == "lmsys_test", blob.get("corpus")
    labels = list(blob["blocks"])  # ['emb','0',...,'27']
    row_of = {b: labels.index(str(b)) for b in blocks}
    ctxs = blob["contexts"]
    cids = sorted(ctxs)
    pe = {b: [] for b in blocks}
    ans = {b: [] for b in blocks}  # list of (n_ans, H) per context
    gsum = {b: np.zeros(h) for b in blocks}
    gcnt = {b: 0 for b in blocks}
    asum = {b: np.zeros(h) for b in blocks}
    acnt = {b: 0 for b in blocks}
    for ci in cids:
        rec = ctxs[ci]
        hh = rec["h"]  # (n_pos, R, H) fp16
        pl, ws = int(rec["prompt_len"]), int(rec["window_start"])
        t_row = pl - 1 - ws
        npos = hh.shape[0]
        assert 0 <= t_row < npos, (ci, t_row, npos)
        seg = np.asarray(rec["segments"])
        ans_idx = np.nonzero(seg == C.SEG_ANSWER)[0]
        for b in blocks:
            r = row_of[b]
            hr = hh[:, r, :].to(torch.float64).numpy()  # (n_pos, H)
            pe[b].append(hr[t_row].copy())
            gsum[b] += hr.sum(0)
            gcnt[b] += npos
            if len(ans_idx):
                a = hr[ans_idx]  # (n_ans, H)
                ans[b].append(a)
                asum[b] += a.sum(0)
                acnt[b] += len(a)
            else:
                ans[b].append(np.zeros((0, h)))
    out = {"n_ctx": len(cids), "rows": {}}
    for b in blocks:
        out["rows"][b] = {
            "promptend": np.stack(pe[b]),  # (n_ctx, H)
            "answer": ans[b],  # list of (n_ans, H)
            "mean_answer": asum[b] / max(acnt[b], 1),
            "mean_promptend": np.stack(pe[b]).mean(0),
            "global_mean": gsum[b] / max(gcnt[b], 1),
        }
    del blob, ctxs
    gc.collect()
    return out


def principal_angles(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
    """cos of principal angles between two orthonormal bases (descending)."""
    if S1.shape[1] == 0 or S2.shape[1] == 0:
        return np.zeros(0)
    s = np.linalg.svd(S1.T @ S2, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def subspace_095(evals: np.ndarray, evecs: np.ndarray) -> np.ndarray:
    """Real orthonormal basis of the |lambda|>0.95 invariant subspace (H, d)."""
    planes = mode_planes(evals, evecs, SUBSPACE_CUTOFF)
    if not planes:
        return np.zeros((evecs.shape[0], 0))
    M = np.concatenate([p["P"] for p in planes], axis=1)
    q, _ = np.linalg.qr(M)
    return q


def logit_lens_tokens(shard_path: Path, directions: dict, topk: int = 20) -> dict:
    """RMSNorm(dir) @ lm_head.T -> top-k promoted/suppressed tokens per direction."""
    from safetensors import safe_open
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN_MODEL, cache_dir=str(CACHE_DIR))
    with safe_open(str(shard_path), framework="pt") as f:
        W = f.get_tensor("lm_head.weight").to(torch.float32)  # (V, H)
        norm_w = f.get_tensor("model.norm.weight").to(torch.float32)  # (H,)
    out = {}
    for name, d in directions.items():
        v = torch.from_numpy(np.asarray(d, dtype=np.float32))
        # RMSNorm (no mean-subtraction), then the learned scale, then unembed
        v = v / (v.pow(2).mean().sqrt() + 1e-6) * norm_w
        logits = W @ v  # (V,)
        top = torch.topk(logits, topk)
        bot = torch.topk(-logits, topk)
        out[name] = {
            "promoted": [tok.decode([int(i)]) for i in top.indices.tolist()],
            "suppressed": [tok.decode([int(i)]) for i in bot.indices.tolist()],
        }
    return out


def make_figures(res: dict, blocks: list[int], fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()

    # Figure 1: per-mode profile heatmap at the evil primary block (20)
    b = 20
    modes = res["per_block"][str(b)]["modes"]
    rows = ["evil", "sycophancy", "hallucination", "mean_answer_dir", "hstar_dir", "ctx_var_frac"]
    row_labels = [
        "evil dir overlap^2",
        "sycophancy dir overlap^2",
        "hallucination dir overlap^2",
        "mean-answer dir overlap^2",
        "fixed-point h* overlap^2",
        "prompt-end ctx-var fraction",
    ]
    mat = np.zeros((len(rows), len(modes)))
    for j, m in enumerate(modes):
        mat[0, j] = m["overlap_sq"]["evil"]
        mat[1, j] = m["overlap_sq"]["sycophancy"]
        mat[2, j] = m["overlap_sq"]["hallucination"]
        mat[3, j] = m["overlap_sq"]["mean_answer_dir"]
        mat[4, j] = m["overlap_sq"]["hstar_dir"]
        mat[5, j] = m["promptend_ctx_var_fraction"]
    fig, ax = plt.subplots(figsize=(min(1.2 + 0.32 * len(modes), 13), 3.6), layout="constrained")
    im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.0)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([f"{m['modulus']:.3f}" for m in modes], fontsize=6, rotation=90)
    ax.set_xlabel(f"eigen-mode |lambda| (block {b}, |lambda|>{MODE_CUTOFF})")
    ax.set_title(f"Per-mode alignment + context-variance profile (block {b})")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    savefig_paper(fig, "fp_mode_profile", fig_dir)
    plt.close(fig)

    # Figure 2: persistence ratio (slow modes vs null) per block
    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    cols = dict(zip(blocks, paper_palette(len(blocks)), strict=True))
    for bl in blocks:
        pr = res["per_block"][str(bl)]["persistence"]
        slow = [m["persistence_ratio"] for m in pr["slow_modes"]]
        ax.scatter([bl] * len(slow), slow, color=cols[bl], s=14, alpha=0.7)
    nullhi = np.mean(
        [res["per_block"][str(bl)]["persistence"]["null_ratio_p97.5"] for bl in blocks]
    )
    ax.axhline(nullhi, color="black", ls="--", lw=1.0, label="random-direction null (97.5%)")
    ax.set_xlabel("read-out block")
    ax.set_ylabel("persistence ratio (between-context std / within-answer std)")
    ax.set_title("Slow-mode persistence on real answer trajectories")
    ax.legend(fontsize=8)
    savefig_paper(fig, "fp_trajectory_persistence", fig_dir)
    plt.close(fig)

    # Figure 3: cross-block principal-angle cos (adjacent 0.95 subspaces) vs null
    fig, ax = plt.subplots(figsize=(8.0, 4.2), layout="constrained")
    pairs = res["cross_block"]["adjacent_pairs"]
    xs = range(len(pairs))
    for i, p in enumerate(pairs):
        cosv = p["principal_cos"]
        ax.scatter([i] * len(cosv), cosv, color=paper_palette(2)[0], s=16)
        ax.plot(
            [i - 0.2, i + 0.2], [p["null_cos_p97.5"], p["null_cos_p97.5"]], color="black", lw=1.4
        )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([p["pair"] for p in pairs], fontsize=8)
    ax.set_ylabel("cos(principal angle) between adjacent 0.95 slow subspaces")
    ax.set_xlabel("adjacent block pair")
    ax.set_title("Cross-block slow-subspace overlap (black tick: random-subspace null 97.5%)")
    savefig_paper(fig, "fp_crossblock_angles", fig_dir)
    plt.close(fig)


def main() -> int:  # noqa: C901 — the A-E analysis sequence IS the spec
    ap = argparse.ArgumentParser(description="Issue #922 slow-shell characterization.")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results/issue_922/slow_shell_characterization.json",
    )
    ap.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures/issue_922")
    ap.add_argument("--skip-logit-lens", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(8)
    rng = np.random.default_rng(0)
    H = C.EXPECTED_HIDDEN
    blocks = list(C.READOUT_BLOCKS)

    store_p = next(
        iter(glob.glob(str(CACHE_DIR / "**/store_test_contexts.pt"), recursive=True)), None
    )
    assert store_p, "store_test_contexts.pt not cached"
    store_p = Path(store_p)

    # trait directions (block-indexed (28,H))
    from huggingface_hub import hf_hub_download

    rb = {}
    for t in C.TRAITS:
        p = hf_hub_download(
            HF_DATA_REPO,
            f"issue779_monitoring/r_b/{t}.pt",
            repo_type="dataset",
            revision=REV_779,
            cache_dir=str(CACHE_DIR),
        )
        rb[t] = torch.load(p, weights_only=False)["r_b"].to(torch.float64).numpy()

    logger.info("[store] loading real test-context states ...")
    t0 = time.time()
    ss = load_store_states(store_p, blocks, H)
    logger.info("[store] %d contexts in %.1fs", ss["n_ctx"], time.time() - t0)

    results: dict = {"per_block": {}, "cross_block": {}, "notes": []}
    mean_ans_dirs: dict = {}  # stash for logit lens after store is freed
    hstars: dict = {}

    for b in blocks:
        evals, evecs, hstar = load_block_eig(b)
        hstars[b] = hstar
        planes = mode_planes(evals, evecs, MODE_CUTOFF)
        row = ss["rows"][b]
        pe = row["promptend"]  # (n_ctx, H)
        pe_c = pe - pe.mean(0)
        total_bcvar = float((pe_c * pe_c).sum())
        mean_ans = row["mean_answer"]
        mean_ans_dirs[b] = mean_ans
        drift_reg = hstar  # a ~parallel to h* (round-1 cos 0.91-0.96); h* = register axis
        # top-5 between-context PCs of prompt-end states
        _, _, Vt = np.linalg.svd(pe_c, full_matrices=False)
        pcs = Vt[:5]

        # ── A: per-mode profile ──────────────────────────────────────────────
        mode_rows = []
        for m in planes:
            P = m["P"]
            proj = pe_c @ P  # (n_ctx, d)
            bc_frac = float((proj * proj).sum() / total_bcvar) if total_bcvar > 0 else 0.0
            mode_rows.append(
                {
                    "eig_index": m["eig_index"],
                    "modulus": m["modulus"],
                    "is_real": m["is_real"],
                    "tau_steps": m["tau_steps"],
                    "rotation_period_steps": m["rotation_period_steps"],
                    "plane_dim": m["plane_dim"],
                    "overlap_sq": {
                        "evil": proj_energy(P, rb["evil"][b]),
                        "sycophancy": proj_energy(P, rb["sycophancy"][b]),
                        "hallucination": proj_energy(P, rb["hallucination"][b]),
                        "mean_answer_dir": proj_energy(P, mean_ans),
                        "hstar_dir": proj_energy(P, drift_reg),
                        "top5_context_pcs": [proj_energy(P, pcs[k]) for k in range(pcs.shape[0])],
                    },
                    "promptend_ctx_var_fraction": bc_frac,
                }
            )

        # ── B1: persistence ratio on real answer trajectories ────────────────
        # scalar mode coordinate = projection onto the plane's first basis vector
        ans_list = row["answer"]
        slow_modes_pr = []
        for m in planes:
            e1 = m["P"][:, 0]  # (H,)
            ctx_mean = np.empty(len(ans_list))
            ctx_within = np.empty(len(ans_list))
            for ci, a in enumerate(ans_list):
                if a.shape[0] == 0:
                    ctx_mean[ci] = np.nan
                    ctx_within[ci] = np.nan
                    continue
                coord = a @ e1  # (n_ans,)
                ctx_mean[ci] = coord.mean()
                ctx_within[ci] = coord.std()
            valid = ~np.isnan(ctx_mean)
            between = float(np.std(ctx_mean[valid]))
            within = float(np.mean(ctx_within[valid]))
            slow_modes_pr.append(
                {
                    "eig_index": m["eig_index"],
                    "modulus": m["modulus"],
                    "between_context_std": between,
                    "within_answer_std": within,
                    "persistence_ratio": float(between / within) if within > 0 else None,
                }
            )
        # modulus-matched random-direction null: random unit dirs, same ratio
        null_ratios = []
        rng_b = np.random.default_rng(b)
        for _ in range(50):
            rd = rng_b.standard_normal(H)
            rd /= np.linalg.norm(rd)
            cms, cws = [], []
            for a in ans_list:
                if a.shape[0] == 0:
                    continue
                coord = a @ rd
                cms.append(coord.mean())
                cws.append(coord.std())
            b_s = float(np.std(cms))
            w_s = float(np.mean(cws))
            if w_s > 0:
                null_ratios.append(b_s / w_s)
        null_p97 = float(np.percentile(null_ratios, 97.5)) if null_ratios else None
        null_med = float(np.percentile(null_ratios, 50)) if null_ratios else None

        # ── E: artifact checks ───────────────────────────────────────────────
        gmean = row["global_mean"]
        ones = np.ones(H) / np.sqrt(H)
        artifact = []
        for m in planes:
            artifact.append(
                {
                    "eig_index": m["eig_index"],
                    "modulus": m["modulus"],
                    "overlap_sq_global_mean": proj_energy(m["P"], gmean),
                    "overlap_sq_all_ones": proj_energy(m["P"], ones),
                }
            )

        results["per_block"][str(b)] = {
            "n_modes_ge_0p90": len(planes),
            "modes": mode_rows,
            "persistence": {
                "slow_modes": slow_modes_pr,
                "null_ratio_median": null_med,
                "null_ratio_p97.5": null_p97,
            },
            "artifact_checks": artifact,
        }
        logger.info("[block %d] %d modes profiled", b, len(planes))
        del evals, evecs
        gc.collect()

    # ── D: cross-block principal angles (0.95 subspaces) ─────────────────────
    subs = {}
    for b in blocks:
        evals, evecs, _ = load_block_eig(b)
        subs[b] = subspace_095(evals, evecs)
        del evals, evecs
    pairs = []
    for b1, b2 in itertools.pairwise(blocks):
        S1, S2 = subs[b1], subs[b2]
        cosv = principal_angles(S1, S2)
        d1, d2 = S1.shape[1], S2.shape[1]
        # random-subspace null: principal cos between random d1,d2 subspaces
        null_top = []
        for _ in range(N_NULL):
            Q1, _ = np.linalg.qr(rng.standard_normal((H, d1)))
            Q2, _ = np.linalg.qr(rng.standard_normal((H, d2)))
            null_top.append(float(principal_angles(Q1, Q2).max()) if d1 and d2 else 0.0)
        pairs.append(
            {
                "pair": f"{b1}-{b2}",
                "dims": [int(d1), int(d2)],
                "principal_cos": [float(x) for x in cosv],
                "max_principal_cos": float(cosv.max()) if len(cosv) else 0.0,
                "mean_principal_cos": float(cosv.mean()) if len(cosv) else 0.0,
                "n_cos_gt_0p5": int((cosv > 0.5).sum()),
                "null_cos_p97.5": float(np.percentile(null_top, 97.5)),
            }
        )
    results["cross_block"]["adjacent_pairs"] = pairs
    del subs
    gc.collect()

    # ── B2 scope note ────────────────────────────────────────────────────────
    results["notes"].append(
        "B2 (slow-coordinate vs graded-trait-score correlation) is SCOPE-LIMITED: the 500 "
        "lmsys_test contexts carry NO per-context trait scores (lmsys_g_rollouts has only "
        "prompt+responses), and the eval-condition store that DOES carry graded y_trait was not "
        "fetched (VM disk at 100%); its correlation is additionally confounded for "
        "sycophancy/hallucination by the #922 regenerated-eval-question parity issue "
        "(prompt<->response mismatch, issue922_common.eval_questions_provenance). The static "
        "behavioral-relevance evidence is A's per-mode trait-direction (r_B) overlap and round-1's "
        "trait-direction slow-subspace projection (r_B is #779's validated behavioral direction)."
    )
    results["notes"].append(
        "All states are teacher-forced captures of the model's OWN on-policy completions "
        "(#922 lmsys_test store). Eigendecompositions reused from round-1 cache (top-64 per block, "
        "covers all |lambda|>0.90). The affine intercept a was not re-loaded (maps blob freed for "
        "disk); h* is its near-parallel proxy (round-1 cos(h*,a)=0.91-0.96)."
    )
    results["notes"].append(
        "E artifact check uses the GLOBAL mean-state direction as the attention-sink proxy: "
        "sequence position 0 (the true sink) is outside the W_P=8 / W_A=40 capture window."
    )

    # ── C: logit lens (needs the ~3 GB Qwen lm_head shard; free store first) ──
    del ss
    gc.collect()
    lens_done = False
    if not args.skip_logit_lens:
        # free the cached store to make room for the shard
        try:
            real_store = os.path.realpath(store_p)
            free_before = _free_gb("/")
            if free_before < DISK_FLOOR_GB_FOR_C:
                os.remove(real_store)
                logger.info(
                    "[logit-lens] removed store blob to free disk (was %.1fGB free)", free_before
                )
        except OSError as e:
            logger.warning("[logit-lens] could not free store: %s", e)
        free_now = _free_gb("/")
        if free_now < DISK_FLOOR_GB_FOR_C:
            results["notes"].append(
                f"C (logit lens) SKIPPED: only {free_now:.1f}GB free on / (need "
                f"{DISK_FLOOR_GB_FOR_C}); the ~3GB Qwen lm_head shard would not fit."
            )
            logger.warning("[logit-lens] SKIP — %.1fGB free", free_now)
        else:
            try:
                from huggingface_hub import hf_hub_download as _dl

                idx = _dl(QWEN_MODEL, "model.safetensors.index.json", cache_dir=str(CACHE_DIR))
                with open(idx) as _f:
                    wm = json.load(_f)["weight_map"]
                shard_rel = wm["lm_head.weight"]
                shard = Path(_dl(QWEN_MODEL, shard_rel, cache_dir=str(CACHE_DIR)))
                directions: dict = {}
                for b in blocks:
                    evals, evecs, _ = load_block_eig(b)
                    planes = mode_planes(evals, evecs, MODE_CUTOFF)[:10]
                    for m in planes:
                        e1 = m["P"][:, 0]
                        directions[f"b{b}_mode{m['eig_index']}_pos(|l|={m['modulus']:.3f})"] = e1
                        directions[f"b{b}_mode{m['eig_index']}_neg"] = -e1
                    directions[f"b{b}_hstar"] = hstars[b]
                    directions[f"b{b}_mean_answer"] = mean_ans_dirs[b]
                    del evals, evecs
                lens = logit_lens_tokens(shard, directions)
                results["logit_lens"] = {
                    "caveat": "logit lens at mid blocks (14-20) approximate; most valid at 24/26",
                    "shard": shard_rel,
                    "tokens": lens,
                }
                lens_done = True
                logger.info("[logit-lens] decoded %d directions", len(directions))
            except Exception as e:
                results["notes"].append(
                    f"C (logit lens) SKIPPED after fetch attempt: {type(e).__name__}: {e}"
                )
                logger.warning("[logit-lens] SKIP after attempt: %s", e)
    results["logit_lens_done"] = lens_done

    results["metadata"] = C.reproducibility_metadata(
        {"script": "issue922_slow_shell", "kind": "free_analysis"}
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(args.out_json, results)
    logger.info("[out] wrote %s", args.out_json)

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    make_figures(results, blocks, args.fig_dir)
    logger.info("[out] figures in %s", args.fig_dir)
    logger.info("[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
