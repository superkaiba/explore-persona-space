#!/usr/bin/env python
# ruff: noqa: RUF001, RUF003
# Intentional Unicode (→, −, ², v_C/v_A notation) in scientific strings + plot labels.
"""Per-position v_C → answer-activation skill (issue #722 free-analysis followup).

Hypothesis: the final-input-token context vector v_C predicts the answer-side
activation WELL at EARLY answer positions (right after the context) and
progressively WORSE at later positions.

Method (0-GPU, disk-safe streaming):
  - For each of 50 context files in
    superkaiba1/explore-persona-space-data:issue658_theory_assumptions/store/answer_spans/<id>.pt
    (each ~3.3 GB, dict with `spans` = 48 per-probe tensors (28, S_probe, 3584) fp16,
     `probes`, `capture_layers` = [0..27]):
      STREAM one file at a time -> for layers {14,16,18}, for each answer position
      p = 0..P_MAX, compute the MEAN over probes (probes whose S_probe > p) of the
      activation at position p -> a per-(context, layer, p) vector (3584) fp32.
      Record coverage (how many probes contributed). DELETE the 5 GB file + prune
      hf cache immediately. Abort if disk free < 30 GB before any download.
  - Predictor v_C = issue594_context_geometry/analysis_tensors/context_vectors_mean.pt
    `tensor` (50,28,3584), aligned by `instance_ids` to the context ids.
  - For each layer in {14,16,18} and position p: LOCO ridge v_C[layer] -> target at p
    across contexts with data at p; skill-over-mean R^2. Reuse the canonical helpers
    (ridge_predict_loco_centered, robust_pca_basis, skill_over_mean_r2). PCA target
    dim = min(48, n_contexts_at_p - 2).
  - Reference line: skill of v_C -> the MEAN-over-all-answer-tokens activation (v_A),
    which should ~= the known ~0.80 at L18 (sanity check).

Outputs:
  - figures/issue_722/per_position_vC_skill.png
  - eval_results/issue_722/per_position_vC_skill.json
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy/torch freeze their pools.
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    ridge_predict_loco_centered,
    robust_pca_basis,
    skill_over_mean_r2,
)

REPO = "superkaiba1/explore-persona-space-data"
ANSWER_SPANS_PREFIX = "issue658_theory_assumptions/store/answer_spans"
CONTEXT_VECTORS_PATH = "issue594_context_geometry/analysis_tensors/context_vectors_mean.pt"
LAYERS = [14, 16, 18]  # axis-0 index == model layer (capture_layers == [0..27])
P_MAX = 64  # answer-token positions 0..P_MAX inclusive
MIN_DISK_FREE_GB = 30.0
COVERAGE_MIN_CONTEXTS = 30  # >=60% of 50 contexts must have data to report a p
SEED = 658

OUT_FIG = Path("figures/issue_722/per_position_vC_skill.png")
OUT_JSON = Path("eval_results/issue_722/per_position_vC_skill.json")
CACHE_ROOT = Path("/tmp/eps_pp_stream")


def disk_free_gb(path: str = "/") -> float:
    return shutil.disk_usage(path).free / 1e9


def extract_one_context(local_pt: str) -> dict:
    """Per-(layer, p) mean over probes + per-(layer) mean-over-all-tokens (v_A).

    Returns {
      "context_id": str,
      "pos_means": {layer: (P_MAX+1, 3584) fp32 with NaN rows where no probe},
      "pos_cov":   {layer: (P_MAX+1,) int  # n probes contributing at p},
      "vA":        {layer: (3584,) fp32}   # mean over ALL answer tokens, all probes
    }
    """
    d = torch.load(local_pt, map_location="cpu", weights_only=False)
    cid = d["context_id"]
    caps = d["capture_layers"]
    layer_axis = {L: caps.index(L) for L in LAYERS}
    spans = d["spans"]  # list of (28, S_probe, 3584) fp16

    pos_means = {L: np.full((P_MAX + 1, 3584), np.nan, dtype=np.float32) for L in LAYERS}
    pos_cov = {L: np.zeros(P_MAX + 1, dtype=np.int64) for L in LAYERS}
    # accumulate per-position sums (float64) + counts
    pos_sum = {L: np.zeros((P_MAX + 1, 3584), dtype=np.float64) for L in LAYERS}
    # v_A accumulate: sum over ALL answer tokens of ALL probes / total tokens
    vA_sum = {L: np.zeros(3584, dtype=np.float64) for L in LAYERS}
    vA_tok = {L: 0 for L in LAYERS}

    for sp in spans:  # sp: (28, S_probe, 3584) fp16
        S = sp.shape[1]
        for L in LAYERS:
            ax = layer_axis[L]
            arr = sp[ax].to(torch.float32).numpy()  # (S, 3584)
            # v_A: all answer tokens
            vA_sum[L] += arr.sum(axis=0, dtype=np.float64)
            vA_tok[L] += S
            # per-position p = 0..min(S-1, P_MAX)
            up = min(S, P_MAX + 1)
            pos_sum[L][:up] += arr[:up].astype(np.float64)
            pos_cov[L][:up] += 1
        del sp
    for L in LAYERS:
        cov = pos_cov[L]
        mask = cov > 0
        pos_means[L][mask] = (pos_sum[L][mask] / cov[mask, None]).astype(np.float32)
    vA = {L: (vA_sum[L] / max(vA_tok[L], 1)).astype(np.float32) for L in LAYERS}
    del d, spans, pos_sum
    gc.collect()
    return {"context_id": cid, "pos_means": pos_means, "pos_cov": pos_cov, "vA": vA}


def stream_extract_all() -> dict:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(REPO, repo_type="dataset")
    ans = sorted(f for f in files if f.startswith(ANSWER_SPANS_PREFIX + "/") and f.endswith(".pt"))
    print(f"[stream] {len(ans)} context files to process")

    # per-context storage (small): for each layer, (n_ctx, P_MAX+1, 3584)
    ctx_ids = []
    pos_by_layer = {L: [] for L in LAYERS}  # list of (P_MAX+1, 3584)
    cov_by_layer = {L: [] for L in LAYERS}  # list of (P_MAX+1,)
    vA_by_layer = {L: [] for L in LAYERS}  # list of (3584,)

    for i, fn in enumerate(ans):
        name = fn.split("/")[-1]
        free = disk_free_gb()
        if free < MIN_DISK_FREE_GB:
            raise SystemExit(
                f"ABORT: disk free {free:.1f} GB < {MIN_DISK_FREE_GB} GB before "
                f"downloading {name} (processed {i}/{len(ans)})"
            )
        cdir = CACHE_ROOT / f"c{i}"
        shutil.rmtree(cdir, ignore_errors=True)
        cdir.mkdir(parents=True, exist_ok=True)
        local = hf_hub_download(REPO, fn, repo_type="dataset", cache_dir=str(cdir))
        szgb = os.path.getsize(local) / 1e9
        out = extract_one_context(local)
        # delete the heavy file + its whole cache dir immediately
        shutil.rmtree(cdir, ignore_errors=True)
        gc.collect()
        ctx_ids.append(out["context_id"])
        for L in LAYERS:
            pos_by_layer[L].append(out["pos_means"][L])
            cov_by_layer[L].append(out["pos_cov"][L])
            vA_by_layer[L].append(out["vA"][L])
        free_after = disk_free_gb()
        print(
            f"[{i + 1}/{len(ans)}] {name} ({szgb:.2f}GB)  ->  {out['context_id']}  "
            f"| disk free {free:.1f}->{free_after:.1f} GB"
        )
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    return {
        "ctx_ids": ctx_ids,
        "pos": {L: np.stack(pos_by_layer[L]) for L in LAYERS},  # (n,P+1,3584)
        "cov": {L: np.stack(cov_by_layer[L]) for L in LAYERS},  # (n,P+1)
        "vA": {L: np.stack(vA_by_layer[L]) for L in LAYERS},  # (n,3584)
    }


def load_vC():
    from huggingface_hub import hf_hub_download

    cdir = CACHE_ROOT / "cv"
    shutil.rmtree(cdir, ignore_errors=True)
    cdir.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(REPO, CONTEXT_VECTORS_PATH, repo_type="dataset", cache_dir=str(cdir))
    cv = torch.load(p, map_location="cpu", weights_only=False)
    tensor = cv["tensor"].to(torch.float32).numpy()  # (50,28,3584)
    ids = list(cv["instance_ids"])
    shutil.rmtree(cdir, ignore_errors=True)
    return tensor, ids


def ridge_skill(vC_layer: np.ndarray, target: np.ndarray) -> dict:
    """LOCO ridge skill-over-mean: vC_layer (n, 3584) -> target (n, 3584).

    Reduce target to PCA dim min(48, n-2), ridge LOCO predict each PCA dim,
    skill_over_mean_r2 in PCA space.
    """
    n = vC_layer.shape[0]
    if n < 4:
        return {"skill": float("nan"), "n": n, "pca_dim": 0}
    k = min(48, n - 2)
    mu, comps, _ = robust_pca_basis(target, k)  # comps (k', 3584)
    Yv = (target - mu) @ comps.T  # (n, k')
    Xc = vC_layer - vC_layer.mean(axis=0, keepdims=True)
    preds = ridge_predict_loco_centered(Xc, Yv)  # (n, k')
    res = skill_over_mean_r2(preds, Yv)
    res["n"] = n
    res["pca_dim"] = int(comps.shape[0])
    return res


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(16, os.cpu_count() or 8))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    print(f"[disk] free at start: {disk_free_gb():.1f} GB")
    vC, vC_ids = load_vC()
    print(f"[vC] {vC.shape}  {len(vC_ids)} ids")

    ext = stream_extract_all()
    ctx_ids = ext["ctx_ids"]

    # align contexts: vC_ids -> ext ctx_ids
    id_to_extidx = {cid: i for i, cid in enumerate(ctx_ids)}
    common = [cid for cid in vC_ids if cid in id_to_extidx]
    print(f"[align] {len(common)} common contexts (vC {len(vC_ids)}, spans {len(ctx_ids)})")
    vC_idx = [vC_ids.index(c) for c in common]
    ext_idx = [id_to_extidx[c] for c in common]

    results = {
        "meta": {
            "p_max": P_MAX,
            "layers": LAYERS,
            "n_contexts_total": len(common),
            "coverage_min_contexts": COVERAGE_MIN_CONTEXTS,
            "seed": SEED,
            "predictor": "v_C (last-input-token context vector, mean over probes)",
            "target": "answer-side activation at position p, mean over probes",
        },
        "per_layer": {},
    }

    for L in LAYERS:
        vC_L = vC[vC_idx, L, :]  # (nc, 3584)
        pos = ext["pos"][L][ext_idx]  # (nc, P+1, 3584)
        cov = ext["cov"][L][ext_idx]  # (nc, P+1)
        vA = ext["vA"][L][ext_idx]  # (nc, 3584)

        # reference: v_C -> v_A (mean over all answer tokens)
        ref = ridge_skill(vC_L, vA)
        print(f"[L{L}] reference v_C->v_A skill = {ref['skill']:.4f} (n={ref['n']})")

        skills = []
        ncov = []
        ps = []
        for p in range(P_MAX + 1):
            # contexts that have data at position p (cov>0)
            has = cov[:, p] > 0
            n_at_p = int(has.sum())
            ncov.append(n_at_p)
            if n_at_p < COVERAGE_MIN_CONTEXTS:
                skills.append(float("nan"))
                ps.append(p)
                continue
            tgt = pos[has, p, :]  # (n_at_p, 3584)
            x = vC_L[has, :]
            r = ridge_skill(x, tgt)
            skills.append(r["skill"])
            ps.append(p)
        results["per_layer"][str(L)] = {
            "p": ps,
            "skill": skills,
            "n_contexts_at_p": ncov,
            "reference_vA_skill": ref["skill"],
            "reference_vA_n": ref["n"],
        }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"[write] {OUT_JSON}")

    make_figure(results)
    print_table(results)
    return results


def make_figure(results: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette(len(LAYERS))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ref_L18 = results["per_layer"]["18"]["reference_vA_skill"]
    for i, L in enumerate(LAYERS):
        d = results["per_layer"][str(L)]
        p = np.array(d["p"], dtype=float)
        sk = np.array(d["skill"], dtype=float)
        m = np.isfinite(sk)
        ax.plot(p[m], sk[m], "-o", ms=3, lw=1.6, color=colors[i], label=f"L{L}")

    ax.axhline(
        ref_L18,
        color="0.35",
        ls="--",
        lw=1.3,
        label=f"v_C → v_A (mean answer), L18 = {ref_L18:.2f}",
    )
    ax.set_xlabel("answer-token position p")
    ax.set_ylabel("skill over mean  R²  (v_C → answer activation)")
    ax.set_title("v_C predicts early answer positions best, decays with p")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_xlim(-1, P_MAX + 1)
    ax.grid(True, alpha=0.25)

    # coverage caption (annotate min n over the plotted range, per task: no on-plot arrows)
    n18 = results["per_layer"]["18"]["n_contexts_at_p"]
    cap = (
        f"v_C = last-input-token context vector (mean over 48 probes); v_A = mean over all "
        f"answer tokens. LOCO ridge skill-over-mean R², PCA target dim = min(48, n−2). "
        f"n_contexts at p=0 is {n18[0]}, at p={P_MAX} is {n18[P_MAX]} (L18). "
        f"Dashed line: v_C→v_A reference at L18."
    )
    fig.text(0.5, -0.06, cap, ha="center", va="top", fontsize=6.5, wrap=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    # savefig_paper(fig, stem, dir) writes <dir>/<stem>.<fmt>; split the path.
    savefig_paper(fig, OUT_FIG.stem, dir=str(OUT_FIG.parent))
    plt.close(fig)
    print(f"[write] {OUT_FIG}")


def print_table(results: dict):
    print("\n=== Skill (R²) at selected positions ===")
    cols = [0, 4, 16, P_MAX]
    header = "layer | " + " | ".join(f"p={c}" for c in cols) + " | v_A ref | n@0 | n@pmax"
    print(header)
    print("-" * len(header))
    for L in LAYERS:
        d = results["per_layer"][str(L)]
        sk = {pp: ss for pp, ss in zip(d["p"], d["skill"], strict=False)}
        ncov = {pp: nn for pp, nn in zip(d["p"], d["n_contexts_at_p"], strict=False)}
        row = f"L{L:>4} | " + " | ".join(f"{sk.get(c, float('nan')):.3f}" for c in cols)
        row += f" | {d['reference_vA_skill']:.3f} | {ncov.get(0, 0)} | {ncov.get(P_MAX, 0)}"
        print(row)


if __name__ == "__main__":
    main()
