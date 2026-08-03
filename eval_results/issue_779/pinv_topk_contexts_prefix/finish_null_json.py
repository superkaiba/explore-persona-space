"""Finisher for random_direction_null.py: writes random_direction_null.json from
the LOCALLY-persisted prefix activations, skipping the model recapture and the slow
full-repo list verification that wedged the parent run (the upload itself succeeded;
a targeted get_paths_info probe confirmed the HF file is present).

Recomputes the same two null statistics as the parent script from
prefix_activations.pt — no model, no re-download. 0 GPU-h, seconds.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from pinv_prefix_twin import (
    KSTAR_PREREG,
    OUT_DIR,
    PARENT_JSON,
    PASS_B,
    RB_DIR,
    READ_OUT_LAYER,
    ridge_fit_matrix,
)
from scipy.stats import rankdata

from explore_persona_space.experiments.issue_779 import fit_h as F

N_RANDOM = 1000
SEED = 0
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue779_monitoring/pinv_prefix_twin"


def spearman_matrix(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    ry = rankdata(y)
    rP = np.apply_along_axis(rankdata, 1, P)
    ry_c = ry - ry.mean()
    rP_c = rP - rP.mean(axis=1, keepdims=True)
    num = rP_c @ ry_c
    den = np.sqrt((rP_c**2).sum(axis=1) * (ry_c**2).sum()) + 1e-12
    return num / den


def main() -> int:
    t0 = time.time()
    twin = json.loads((OUT_DIR / "pinv_prefix_twin.json").read_text())
    parent = json.loads(PARENT_JSON.read_text())
    blob = torch.load(OUT_DIR / "prefix_activations.pt", weights_only=False)
    acts = {
        k: {int(li): v.to(torch.float64).numpy() for li, v in d.items()}
        for k, d in blob["activations"].items()
    }
    alias_to_key = {a: r["key"] for r in twin["prefixes"] for a in r["aliases"]}
    assert len(acts) == 25, len(acts)

    tb = torch.load(PASS_B, map_location="cpu", mmap=True, weights_only=False)
    layers_b = list(tb["layers"])
    results = {"n_random": N_RANDOM, "seed": SEED, "traits": {}}
    rng = np.random.default_rng(SEED)

    for trait, L in READ_OUT_LAYER.items():
        li = layers_b.index(L)
        Xtr = tb["cx_last"][:, li, :].to(torch.float64).numpy()
        Ytr = tb["v_x"][:, li, :].to(torch.float64).numpy()
        r_b = torch.load(RB_DIR / f"{trait}.pt", weights_only=False)["r_b"]
        r_b = r_b.to(torch.float64).numpy()[li]
        fit = ridge_fit_matrix(Xtr, Ytr)
        W, xmu, xsd, s, lam = fit["W"], fit["xmu"], fit["xsd"], fit["s"], fit["lam"]
        recon = F.reconstruction_metrics(((Xtr - xmu) / xsd) @ W + fit["ymu"], Ytr)
        assert abs(recon["r2"] - parent["traits"][trait]["recon_r2_committed"]) < 1e-3
        Um, Sm, Vmt = np.linalg.svd(W.T, full_matrices=False)
        UtRb = Um.T @ r_b
        k_ridge = int(np.sum(s**2 >= lam))
        assert k_ridge == KSTAR_PREREG[trait], (trait, k_ridge)
        w_pinv = Vmt[:k_ridge].T @ (UtRb[:k_ridge] / Sm[:k_ridge])

        cond_ids = [f"sys{i}" for i in range(8)] + ["shot0"]
        judge = np.array(
            [parent["traits"][trait]["eval_grid"]["cond_mean_judge_score"][c] for c in cond_ids]
        )
        C9 = np.stack([(acts[alias_to_key[f"{trait}:{c}"]][L] - xmu) / xsd for c in cond_ids])
        all_keys = sorted(acts)
        C25 = np.stack([(acts[k][L] - xmu) / xsd for k in all_keys])

        got = C9 @ w_pinv
        max_dev = float(
            np.max(
                np.abs(
                    got
                    - np.array(
                        [
                            twin["traits"][trait]["directions"]["w_pinv_kstar"][
                                "prefix_proj_per_condition"
                            ][c]
                            for c in cond_ids
                        ]
                    )
                )
            )
        )
        obs_rho = float(spearman_matrix(got[None, :], judge)[0])
        committed_rho = twin["traits"][trait]["directions"]["w_pinv_kstar"][
            "spearman_prefix_vs_judge_n9"
        ]
        assert abs(obs_rho - committed_rho) < 0.12, (trait, obs_rho, committed_rho)

        R = rng.standard_normal((N_RANDOM, C9.shape[1]))
        null_rho = spearman_matrix(R @ C9.T, judge)
        p_one_sided = float((np.sum(null_rho >= obs_rho) + 1) / (N_RANDOM + 1))

        own_idx = np.array([all_keys.index(f"{trait}:sys{i}") for i in range(4)])
        proj25 = R @ C25.T
        top4 = np.argsort(-proj25, axis=1)[:, :4]
        null_own_in_top4 = np.isin(top4, own_idx).sum(axis=1)
        obs_top4_own = int(np.isin(np.argsort(-(C25 @ w_pinv))[:4], own_idx).sum())
        p_top4 = float((np.sum(null_own_in_top4 >= obs_top4_own) + 1) / (N_RANDOM + 1))

        results["traits"][trait] = {
            "observed_spearman_w_pinv_kstar": round(obs_rho, 4),
            "twin_committed_spearman": committed_rho,
            "recapture_max_abs_proj_dev": round(max_dev, 4),
            "null_rho_band_2p5_97p5": [
                round(float(np.percentile(null_rho, 2.5)), 4),
                round(float(np.percentile(null_rho, 97.5)), 4),
            ],
            "null_rho_max": round(float(null_rho.max()), 4),
            "p_one_sided_spearman": round(p_one_sided, 4),
            "observed_own_strong_in_top4_of_25": obs_top4_own,
            "null_own_in_top4_mean": round(float(null_own_in_top4.mean()), 3),
            "null_own_in_top4_max": int(null_own_in_top4.max()),
            "p_one_sided_top4_selectivity": round(p_top4, 4),
            "own_strong_ranks_committed": twin["traits"][trait]["directions"]["w_pinv_kstar"][
                "own_strong_sys012_ranks"
            ],
        }
        print(
            f"[finish] {trait}: obs={obs_rho:+.3f} band "
            f"{results['traits'][trait]['null_rho_band_2p5_97p5']} p={p_one_sided:.4f}; "
            f"top4-own={obs_top4_own} p={p_top4:.4f}",
            flush=True,
        )

    # targeted verify (fast; the parent run's full list_repo_files wedged)
    from huggingface_hub import HfApi

    api = HfApi()
    path = f"{HF_PREFIX}/prefix_activations.pt"
    info = api.get_paths_info(HF_DATA_REPO, [path], repo_type="dataset")
    assert info and info[0].path == path, "HF upload verification (targeted) failed"

    results["metadata"] = {
        "script": "finish_null_json (recomputed from local prefix_activations.pt; parent "
        "random_direction_null.py wedged in full-repo list verification after a "
        "successful upload)",
        "activations_hf": f"{HF_DATA_REPO}/{path}",
        "capture_dtype": blob["dtype"],
        "wall_seconds": round(time.time() - t0, 1),
    }
    out = OUT_DIR / "random_direction_null.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"[finish] wrote {out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
