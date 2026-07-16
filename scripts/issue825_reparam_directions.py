#!/usr/bin/env python
"""#825 free-analysis: WHICH directions did the base->instruct reparam change most.

Result-2 established that post-training re-expresses the context->answer map via a
large anisotropic general-linear context reparameterization A_ctx (||A_ctx-I||=91,
cond ~9e5, stretch sv 0..15.3 at L19; see
eval_results/issue_825/reparam_characterization/results.json). This analysis
IDENTIFIES the top-changed directions and INTERPRETS them, structurally and
semantically.

PART A (structural, 0-GPU, on the HF activation stores):
  A_ctx = ridge operator base-context -> instruct-context (right-multiply: the
          change of the context representation, dXi ~ dXb @ A_ctx). SVD(A_ctx - I)
          -> the "most-changed directions": right singular vectors are the
          directions the representation MOVES toward (in instruct-context space);
          left singular vectors are the base-context input features that DRIVE the
          change. For each top direction:
    - variance-rank : project onto the instruct-context covariance PCs (SVD of
                      centered Xi) -> variance percentile + PC-band (is the change
                      a high-variance dominant direction or a low-variance tail?).
    - map-relevance : squared-cosine overlap with U_w = the top predictive input
                      directions of the answer map W_inst (Xi->Yi) -> do the
                      most-changed directions drive the answer?
  Mean shift dmu = mean(Xi) - mean(Xb): magnitude vs typical ||x||, and overlap
    with the map subspace / top context PC / top (A_ctx-I) singular directions
    (is the single biggest change a constant assistant/persona bias?).

PART B (semantic logit-lens, CPU): apply the final RMSNorm gain then unembed each
  top-changed direction (and dmu), reporting top/bottom tokens. APPROXIMATE lens:
  a layer-19 direction pushed through the FINAL unembedding, skipping the
  intervening layers AND the input-dependent RMS normalization scale.

Reuses scripts/issue825_map_alignment.py (ma) loaders + ma._raw_ridge_operator
(the same primal-ridge operator the reparam_characterization used) + the HF stage
recipe. Ridge/SVD closed-form only; VM CPU; 0 GPU-h. Headline layer 19.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import issue825_map_alignment as ma  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "eval_results" / "issue_825" / "reparam_directions"
FIG_DIR = REPO / "figures" / "issue_825"
DL_DIR = REPO / "data" / "issue_825" / "reparam_directions_dl"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHARD = "model-00004-of-00004.safetensors"  # holds lm_head.weight + model.norm.weight
LM_HEAD_KEY = "lm_head.weight"
NORM_KEY = "model.norm.weight"

TOP_SPECTRUM = 20  # (A_ctx - I) singular values reported
TOP_DIRS = 10  # per-direction structural metrics (right + left)
TOP_LL_RIGHT = 8  # right singular vectors logit-lensed
TOP_LL_LEFT = 4  # left singular vectors logit-lensed
N_TOKENS = 15  # top/bottom tokens per direction
ENERGY_CUT = 0.90  # r90 = rank capturing 90% of the answer-map energy
FIXED_RANK = 200  # r200 = fixed comparison rank


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as e:
        return f"unresolved: {e}"


def _op(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Primal-ridge operator W (D_in, D_out) with GCV lambda: dY ~ dX_raw @ W."""
    w, lam = ma._raw_ridge_operator(x, y)
    return torch.as_tensor(w, dtype=torch.float64, device=x.device), float(lam)


def _dir_metrics(
    v: torch.Tensor,
    vhpc: torch.Tensor,
    var_pc: torch.Tensor,
    idx_pc: torch.Tensor,
    uw: torch.Tensor,
    r90: int,
    r200: int,
) -> dict:
    """Structural placement of a unit direction v (D,) in the shared neuron basis.

    variance-rank vs the instruct-context covariance PCs (rows of vhpc, per-PC
    variance var_pc), and map-relevance vs the answer-map input directions uw
    (columns orthonormal). All squared cosines (v is unit; both bases orthonormal).
    """
    v = v / (v.norm() + 1e-30)
    # --- variance placement vs instruct-context PCs ---
    coeff = vhpc @ v  # (D,) projection onto PC directions
    sq = coeff**2  # squared cosine per PC; sums to 1
    proj_var = float((sq * var_pc).sum())  # variance of Xi along v
    var_pctile = float((var_pc <= proj_var).to(torch.float64).mean())  # 1=dominant, 0=tail
    centroid_pc = float((sq * idx_pc).sum())  # variance-weighted mean PC index
    sqcos_top10_pc = float(sq[:10].sum())
    sqcos_top50_pc = float(sq[:50].sum())
    # --- map relevance vs answer-map input directions ---
    mcoeff = uw.transpose(0, 1) @ v  # (D,) projection onto U_w columns
    msq = mcoeff**2
    return {
        "variance_percentile": var_pctile,
        "proj_variance": proj_var,
        "centroid_pc_index": centroid_pc,
        "sqcos_top10_context_pcs": sqcos_top10_pc,
        "sqcos_top50_context_pcs": sqcos_top50_pc,
        "map_sqcos_top1": float(msq[0]),
        "map_sqcos_r90": float(msq[:r90].sum()),
        "map_sqcos_r200": float(msq[:r200].sum()),
    }


def _fetch_model_head() -> tuple[torch.Tensor, torch.Tensor, object]:
    """Download ONLY the shard holding lm_head + final-norm; load both on CPU fp32."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoTokenizer

    DL_DIR.mkdir(parents=True, exist_ok=True)
    shard = hf_hub_download(
        MODEL_ID, MODEL_SHARD, local_dir=str(DL_DIR), token=os.environ.get("HF_TOKEN")
    )
    with safe_open(shard, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        if LM_HEAD_KEY not in keys:  # tied-embedding fallback (untied for Qwen2.5-7B)
            raise KeyError(
                f"{LM_HEAD_KEY} absent from {MODEL_SHARD}; keys sample={sorted(keys)[:5]}"
            )
        lm_head = f.get_tensor(LM_HEAD_KEY).to(torch.float32)  # (V, D)
        gamma = f.get_tensor(NORM_KEY).to(torch.float32)  # (D,)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    return lm_head, gamma, tok


def _logit_lens(v: torch.Tensor, lm_head: torch.Tensor, gamma: torch.Tensor, tok) -> dict:
    """Approximate logit lens of a unit direction: lm_head @ (gamma * v_hat).

    Reports the +v orientation as top_tokens and the -v orientation as
    bottom_tokens (logits(-v) = -logits(v), so bottom-15 of +v == top-15 of -v):
    both sign orientations of a change direction are meaningful.
    """
    vhat = v / (v.norm() + 1e-30)
    vg = gamma * vhat.to(torch.float32)  # (D,) final-norm gain applied
    logits = lm_head @ vg  # (V,)
    top = torch.topk(logits, N_TOKENS)
    bot = torch.topk(-logits, N_TOKENS)

    def _decode(indices) -> list[dict]:
        out = []
        for i in indices.tolist():
            out.append(
                {"token_id": int(i), "token": tok.decode([int(i)]), "logit": float(logits[i])}
            )
        return out

    return {"top_tokens": _decode(top.indices), "bottom_tokens": _decode(bot.indices)}


def run() -> dict:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lh = ma.HEADLINE_LAYER

    from huggingface_hub import HfApi

    try:
        resolved = HfApi().repo_info(ma.HF_DATA_REPO, repo_type="dataset", revision=ma.HF_REV).sha
    except Exception as e:
        resolved = f"unresolved: {e}"

    # ---- stage the two stores (incremental per-shard; ~2 GB peak each) ----
    npz_i = ma.cm.extract_stem(ma.STEM_INSTRUCT, DL_DIR)
    npz_b = ma.cm.extract_stem(ma.STEM_BASE, DL_DIR)
    data, _conv, layers_loaded, al = ma._load_pair(npz_i, npz_b, [lh])
    n = al["n_common"]
    print(f"[load] n={n} layers={layers_loaded} in {time.time() - t0:.1f}s", flush=True)

    xi, yi, xb = data["Xi"][lh], data["Yi"][lh], data["Xb"][lh]  # (n, D) fp64 cpu
    d = xi.shape[1]

    # ---- operators ----
    a_ctx, lam_ctx = _op(xb, xi)  # base-context -> instruct-context
    w_inst, lam_w = _op(xi, yi)  # instruct-context -> instruct-answer (the map)
    identity = torch.eye(d, dtype=torch.float64)
    dev = a_ctx - identity
    print(
        f"[op] A_ctx, W_inst fit; ||A_ctx-I||={float(dev.norm()):.3f} in {time.time() - t0:.1f}s",
        flush=True,
    )

    # ---- SVD of the deviation (the most-changed directions) ----
    ud, sd, vhd = torch.linalg.svd(dev, full_matrices=False)  # ud (D,D), sd (D,), vhd (D,D)
    sd_np = sd.cpu().numpy()

    # ---- answer-map input directions (U_w) + energy ranks ----
    uw, sw, _vhw = torch.linalg.svd(w_inst, full_matrices=False)
    sw2 = sw.cpu().numpy() ** 2
    cum = np.cumsum(sw2) / (sw2.sum() + 1e-30)
    r90 = int(min(max(int(np.searchsorted(cum, ENERGY_CUT) + 1), 1), d))
    r200 = int(min(FIXED_RANK, d))

    # ---- instruct-context covariance PCs (for variance-rank) ----
    xi_c = xi - xi.mean(0)
    _upc, spc, vhpc = torch.linalg.svd(xi_c, full_matrices=False)  # vhpc rows = PC dirs
    var_pc = (spc.to(torch.float64) ** 2) / max(n - 1, 1)  # per-PC variance, descending
    idx_pc = torch.arange(d, dtype=torch.float64)
    total_var = float(var_pc.sum())
    print(f"[svd] deviation + map + context-PC SVDs done in {time.time() - t0:.1f}s", flush=True)

    # ---- per-direction structural metrics (top-10, right + left) ----
    top_directions = []
    for k in range(TOP_DIRS):
        right = vhd[k, :]  # right singular vector = direction the representation moves toward
        left = ud[:, k]  # left singular vector = base-context input feature driving the change
        top_directions.append(
            {
                "rank": k + 1,
                "singular_value": float(sd_np[k]),
                "right_singular_vector": _dir_metrics(right, vhpc, var_pc, idx_pc, uw, r90, r200),
                "left_singular_vector": _dir_metrics(left, vhpc, var_pc, idx_pc, uw, r90, r200),
            }
        )

    # ---- mean-shift dmu ----
    mu_i, mu_b = xi.mean(0), xb.mean(0)
    dmu = mu_i - mu_b
    dmu_norm = float(dmu.norm())
    dmu_hat = dmu / (dmu_norm + 1e-30)
    dmu_metrics = _dir_metrics(dmu_hat, vhpc, var_pc, idx_pc, uw, r90, r200)
    # overlap with the top (A_ctx - I) right singular directions (the change dirs)
    dmu_dev_coeff = (vhd @ dmu_hat).cpu().numpy()  # (D,)
    dmu_dev_sq = dmu_dev_coeff**2
    dmu_block = {
        "norm": dmu_norm,
        "typical_norm_instruct": float(xi.norm(dim=1).mean()),
        "typical_norm_base": float(xb.norm(dim=1).mean()),
        "norm_rel_to_typical_instruct": dmu_norm / (float(xi.norm(dim=1).mean()) + 1e-30),
        "variance_percentile": dmu_metrics["variance_percentile"],
        "sqcos_top_context_pc": float((vhpc[0] @ dmu_hat) ** 2),
        "sqcos_top10_context_pcs": dmu_metrics["sqcos_top10_context_pcs"],
        "map_sqcos_top1": dmu_metrics["map_sqcos_top1"],
        "map_sqcos_r90": dmu_metrics["map_sqcos_r90"],
        "map_sqcos_r200": dmu_metrics["map_sqcos_r200"],
        "sqcos_top_deviation_right_sv": float(dmu_dev_sq[0]),
        "sqcos_top10_deviation_right_sv": float(dmu_dev_sq[:10].sum()),
    }
    print(
        f"[struct] top-{TOP_DIRS} directions + dmu metrics done in {time.time() - t0:.1f}s",
        flush=True,
    )

    metadata = {
        "git_commit": _git_head(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reused_from": "scripts/issue825_map_alignment.py",
        "hf_repo": ma.HF_DATA_REPO,
        "hf_prefix": ma.HF_PREFIX,
        "hf_revision_pinned": ma.HF_REV,
        "hf_revision_resolved": resolved,
        "stems": [ma.STEM_INSTRUCT, ma.STEM_BASE],
        "role": ma.ROLE,
        "headline_layer": lh,
        "n_common": int(n),
        "hidden_dim": int(d),
        "lambda_a_ctx": lam_ctx,
        "lambda_w_inst": lam_w,
        "map_rank_e90": r90,
        "fixed_rank": r200,
        "frob_a_ctx_minus_I": float(dev.norm()),
        "context_total_variance": total_var,
        "device": str(ma._fit_device()),
        "model_id": MODEL_ID,
        "model_shard_loaded": MODEL_SHARD,
        "tie_word_embeddings": False,
        "script": "scripts/issue825_reparam_directions.py",
        "caveats": [
            "Descriptive geometry on a SINGLE seed / one pinned store; no mechanism claims.",
            "A_ctx is a right-multiply ridge operator (dXi ~ dXb @ A_ctx) in the SHARED "
            "neuron basis (instruct fine-tuned from base), so A_ctx - I is a valid measure "
            "of how the context ENCODING differs in the same coordinates.",
            "Right singular vectors of (A_ctx - I) = directions the representation moves "
            "toward (instruct-context space); left singular vectors = base-context input "
            "features that drive the change.",
            "APPROXIMATE logit lens: a layer-19 direction pushed through the FINAL unembed "
            "with the final-norm gain applied but WITHOUT the input-dependent RMSNorm scale "
            "and WITHOUT layers 20-27; token lists are indicative, not exact next-token "
            "predictions. Sign is arbitrary per SVD, so BOTH orientations (top/bottom) are "
            "reported.",
            "Per-layer ||A_ctx-I|| profile lives in the sibling "
            "eval_results/issue_825/reparam_characterization/results.json (headline L19 only "
            "here).",
        ],
        "wall_seconds": None,
    }

    result = {
        "metadata": metadata,
        "deviation_spectrum": {
            "top_singular_values": [float(x) for x in sd_np[:TOP_SPECTRUM]],
            "participation_ratio": float((sd_np.sum() ** 2) / ((sd_np**2).sum() + 1e-30)),
            "n_dims": len(sd_np),
        },
        "top_directions": top_directions,
        "mean_shift_dmu": dmu_block,
        "logit_lens": None,  # filled in Part B
    }

    # ---- partial write (Part A) so a Part-B failure does not lose the structural read ----
    (OUT_DIR / "results.json").write_text(json.dumps(result, indent=2))
    print(f"[write-partial] Part A -> {OUT_DIR / 'results.json'}", flush=True)

    # ---- PART B: semantic logit-lens ----
    lm_head, gamma, tok = _fetch_model_head()
    assert lm_head.shape[1] == d, (lm_head.shape, d)
    print(
        f"[model] lm_head {tuple(lm_head.shape)} + gamma {tuple(gamma.shape)} loaded in "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )

    ll = {"right_singular_vectors": [], "left_singular_vectors": [], "mean_shift_dmu": None}
    for k in range(TOP_LL_RIGHT):
        rec = _logit_lens(vhd[k, :], lm_head, gamma, tok)
        rec["rank"] = k + 1
        rec["singular_value"] = float(sd_np[k])
        ll["right_singular_vectors"].append(rec)
    for k in range(TOP_LL_LEFT):
        rec = _logit_lens(ud[:, k], lm_head, gamma, tok)
        rec["rank"] = k + 1
        rec["singular_value"] = float(sd_np[k])
        ll["left_singular_vectors"].append(rec)
    ll["mean_shift_dmu"] = _logit_lens(dmu_hat, lm_head, gamma, tok)
    result["logit_lens"] = ll
    print(f"[logit-lens] done in {time.time() - t0:.1f}s", flush=True)

    result["metadata"]["wall_seconds"] = round(time.time() - t0, 1)
    (OUT_DIR / "results.json").write_text(json.dumps(result, indent=2))
    print(
        f"[write] {OUT_DIR / 'results.json'} (wall {result['metadata']['wall_seconds']}s)",
        flush=True,
    )
    return result


def _make_figure(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    cb = paper_palette(3)
    lh = result["metadata"]["headline_layer"]
    r90 = result["metadata"]["map_rank_e90"]

    fig, (axs, axd) = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained")

    # ---- Panel 1: singular-value spectrum of (A_ctx - I) ----
    sv = result["deviation_spectrum"]["top_singular_values"]
    axs.plot(range(1, len(sv) + 1), sv, "o-", color=cb[0], ms=4)
    axs.set_xlabel("Singular-value index")
    axs.set_ylabel("Singular value of (A_ctx - I)")
    axs.set_title(f"Layer {lh}: most-changed-direction spectrum")

    # ---- Panel 2: top changed directions -- is the change high-variance? answer-driving? ----
    tds = result["top_directions"]
    xv = [t["right_singular_vector"]["variance_percentile"] for t in tds]
    yv = [t["right_singular_vector"]["map_sqcos_r90"] for t in tds]
    svv = [t["singular_value"] for t in tds]
    sc = axd.scatter(xv, yv, c=svv, cmap="viridis", s=70, edgecolor="k", linewidth=0.4, zorder=3)
    for t, x, y in zip(tds, xv, yv, strict=True):
        axd.annotate(str(t["rank"]), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    dmu = result["mean_shift_dmu"]
    axd.scatter(
        [dmu["variance_percentile"]],
        [dmu["map_sqcos_r90"]],
        marker="*",
        s=240,
        color=cb[1],
        edgecolor="k",
        linewidth=0.5,
        zorder=4,
        label="mean shift",
    )
    axd.set_xlabel("Context-variance percentile (0 = tail, 1 = dominant)")
    axd.set_ylabel(f"Answer-map overlap\n(sq-cos, top-{r90} predictive dirs)")
    axd.set_title(f"Layer {lh}: are the most-changed directions answer-driving?")
    axd.legend(fontsize=8, loc="best")
    fig.colorbar(sc, ax=axd, label="singular value of (A_ctx - I)")

    fig.suptitle(
        "#825: which base->instruct reparameterization directions changed most",
        fontsize=11,
        y=1.04,
    )
    saved = savefig_paper(fig, "issue_825/reparam_directions", dir="figures")
    plt.close(fig)

    # ---- augment the auto sidecar with a factual caption ----
    meta_path = FIG_DIR / "reparam_directions.meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        meta = {}
    meta["caption"] = (
        f"Which directions the base->instruct context reparameterization A_ctx changed most, "
        f"layer {lh} (n={result['metadata']['n_common']}). Left: singular-value spectrum of "
        f"(A_ctx - I) -- a few directions carry most of the reshaping. Right: the top "
        f"{len(result['top_directions'])} most-changed directions (labeled by rank, colored by "
        f"singular value) placed by instruct-context variance percentile (x) and squared-cosine "
        f"overlap with the answer map's top-{r90} predictive input directions (y); the star is "
        f"the mean shift dmu. Right singular vectors of (A_ctx - I) are the directions the "
        f"representation moves toward. Descriptive geometry, single pinned store; logit-lens "
        f"tokens are in results.json (approximate lens)."
    )
    meta["source"] = "eval_results/issue_825/reparam_directions/results.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[fig] wrote {saved.get('png')} + caption in {meta_path}", flush=True)


def main() -> None:
    result = run()
    _make_figure(result)


if __name__ == "__main__":
    main()
