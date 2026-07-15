"""Inline free-analysis (issue #658, A3.5a within-condition coherence): download +
reduce per-probe context vectors c_x and per-probe answer profiles to compact cached
arrays. ANALYSIS-ONLY: reads stored activations from HF, no training/generation/GPU.

Sources (pinned rev b33429f77b86, dataset superkaiba1/explore-persona-space-data):
  - #594 issue594_context_geometry/analysis_tensors/per_probe/<ctx>.pt
       ['tensor'] (48,28,3584) fp16 = per-probe last-input-token context vectors c_x
       ['mean_fp32'] (28,3584) fp32 = condition centroid c_C ; ['family'] str
  - #658 issue658_theory_assumptions/store/single_context/<ctx>.pt
       ['per_probe'][i]['samples'][j]['act'] (28,3584) fp16 = per-sample answer profile
       (8 samples/probe) -> per-probe answer profile a_x = mean over samples
  - #658 store/r_b.pt ['r_b'][B]['diffmeans'] (28,3584) fp32 = behavior directions
  - #658 store/v0_summaries.pt ['summaries']['mean'][ctx] (28,3584) = condition v0(C)

Content hygiene: the single_context samples carry base-model completion `text`
(some contexts are behavior-eliciting). This script NEVER reads/prints `text`; only
the `act` tensors are touched.
"""

import json
import os

from explore_persona_space.orchestrate.env import load_dotenv

# #847: shared-VM thread caps bind in-process only when load_dotenv() runs
# BEFORE the import that freezes the BLAS/intra-op pools.
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"
REV = "b33429f77b86"
OUT = "/tmp/issue658_a35a"
os.makedirs(OUT, exist_ok=True)

N_LAYERS = 28
HID = 3584
BEHAVIORS = ["broad_em", "harmful_compliance", "sycophancy", "refusal"]


def _dl(path):
    return hf_hub_download(REPO, path, repo_type="dataset", revision=REV)


def main():
    # --- context ids in the canonical store order ---
    man = json.load(open(_dl("issue658_theory_assumptions/store/store_manifest.json")))
    ctx_ids = list(man["context_ids"])
    store_hash = man["probe_pool_hash"]
    assert len(ctx_ids) == 50, len(ctx_ids)

    # --- per-probe context vectors c_x (#594) ---
    cx = np.zeros((50, 48, N_LAYERS, HID), dtype=np.float32)
    cc_last = np.zeros((50, N_LAYERS, HID), dtype=np.float32)  # last-input-token centroid
    families = []
    hashes = set()
    for ci, ctx in enumerate(ctx_ids):
        d = torch.load(
            _dl(f"issue594_context_geometry/analysis_tensors/per_probe/{ctx}.pt"),
            map_location="cpu",
        )
        t = d["tensor"]  # (48,28,3584) fp16
        assert tuple(t.shape) == (48, N_LAYERS, HID), (ctx, tuple(t.shape))
        cx[ci] = t.float().numpy()
        cc_last[ci] = d["mean_fp32"].float().numpy()
        families.append(str(d.get("family", "unknown")))
        hashes.add(d.get("probe_pool_hash"))
    assert hashes == {store_hash}, ("c_x probe_pool_hash mismatch", hashes, store_hash)

    # --- per-probe answer profiles a_x (#658 single_context; mean over 8 samples) ---
    ax = np.zeros((50, 48, N_LAYERS, HID), dtype=np.float32)
    nsamp = np.zeros((50, 48), dtype=np.int32)
    for ci, ctx in enumerate(ctx_ids):
        d = torch.load(
            _dl(f"issue658_theory_assumptions/store/single_context/{ctx}.pt"),
            map_location="cpu",
        )
        pp = d["per_probe"]
        assert len(pp) == 48, (ctx, len(pp))
        for pi, entry in enumerate(pp):
            samples = entry["samples"]
            acc = None
            k = 0
            for s in samples:
                a = s["act"]  # (28,3584) fp16 ; NEVER touch s["text"]
                acc = a.float() if acc is None else acc + a.float()
                k += 1
            ax[ci, pi] = (acc / max(k, 1)).numpy()
            nsamp[ci, pi] = k

    # --- behavior directions r_B (diffmeans) ---
    rb_raw = torch.load(_dl("issue658_theory_assumptions/store/r_b.pt"), map_location="cpu")["r_b"]
    rB = np.stack(
        [rb_raw[b]["diffmeans"].float().numpy() for b in BEHAVIORS], axis=0
    )  # (4,28,3584)

    # --- condition v0(C) mean recipe (for consistency check) ---
    v0s = torch.load(_dl("issue658_theory_assumptions/store/v0_summaries.pt"), map_location="cpu")
    v0_mean = np.stack(
        [v0s["summaries"]["mean"][ctx].float().numpy() for ctx in ctx_ids], axis=0
    )  # (50,28,3584)

    # --- consistency checks ---
    # (i) c_x condition-mean reproduces the stored last-token centroid mean_fp32
    cx_cond_mean = cx.mean(axis=1)  # (50,28,3584)
    err_cc = np.abs(cx_cond_mean - cc_last).max()
    # (ii) a_x condition-mean reproduces v0_summaries['mean']
    ax_cond_mean = ax.mean(axis=1)  # (50,28,3584)
    # cosine per (ctx,layer) between our a_x cond-mean and stored v0 mean
    num = (ax_cond_mean * v0_mean).sum(-1)
    den = np.linalg.norm(ax_cond_mean, axis=-1) * np.linalg.norm(v0_mean, axis=-1) + 1e-12
    cos_v0 = num / den  # (50,28)
    checks = {
        "cx_condmean_vs_mean_fp32_maxabs": float(err_cc),
        "ax_condmean_vs_v0mean_cos_mean": float(np.nanmean(cos_v0)),
        "ax_condmean_vs_v0mean_cos_min": float(np.nanmin(cos_v0)),
        "ax_condmean_vs_v0mean_cos_median": float(np.nanmedian(cos_v0)),
        "n_samples_per_probe_min": int(nsamp.min()),
        "n_samples_per_probe_max": int(nsamp.max()),
        "probe_pool_hash": store_hash,
    }
    print("CONSISTENCY CHECKS:", json.dumps(checks, indent=1))

    np.savez(
        os.path.join(OUT, "reduced.npz"),
        cx=cx.astype(np.float16),  # (50,48,28,3584)
        ax=ax.astype(np.float16),  # (50,48,28,3584)
        cc_last=cc_last.astype(np.float32),  # (50,28,3584)
        v0_mean=v0_mean.astype(np.float32),  # (50,28,3584)
        rB=rB.astype(np.float32),  # (4,28,3584)
    )
    json.dump(
        {"ctx_ids": ctx_ids, "families": families, "behaviors": BEHAVIORS, "checks": checks},
        open(os.path.join(OUT, "meta.json"), "w"),
        indent=1,
    )
    print("SAVED", os.path.join(OUT, "reduced.npz"))
    print("families:", {f: families.count(f) for f in sorted(set(families))})


if __name__ == "__main__":
    main()
