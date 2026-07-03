#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, ℓ, →, ĥ, ρ) in scientific docstrings/log messages.
"""Issue #841 scaling — Stage-1 transported-trait-monitor CURVE across fit-sizes.

Re-runs the parent's transported-projection benchmark at EVERY fitted n vs the
4k anchor, reading the transport advantage as a CURVE (not a single endpoint) so
a non-monotone / inverted-U response is not misread as FLAT (plan v9 §3 H2).

The map-INDEPENDENT rows (raw-source, id-transport) are computed ONCE; only the
transported (ridge) read depends on the fitted maps, so it is recomputed per n
by loading the per-n ridge maps ``issue841_scaling_stage0`` persisted. Everything
else — r_B, pass_a eval trajectories, cached judge scores, ℓ*, the
within-condition Pearson protocol, and the #779 ``bootstrap_delta_ci`` — is
#779/#841-verbatim. NO new judging.

Per fit-size n (vs the 4k anchor):
  win_count(n)         : # cells where the n-transported read beats BOTH raw-source
                         AND id-transport (bootstrap_delta_ci CI excludes 0, delta>0)
  net / newly / dropped: symmetric win accounting vs the 4k baseline
  mean_paired_delta(n) : mean over cells of r(transported@n) − r(transported@4k),
                         per-cell joint condition-resample bootstrap CI
  BH_survivors(n)      : newly-winning cells surviving Benjamini-Hochberg FDR
  transport_fidelity(n): eval-context Δ-reconstruction R²/cosine per (source,ℓ*)

--synthetic fabricates a tiny in-memory trait + maps (no HF, no GPU) that runs the
FULL win-count/paired-delta/BH/npz path end-to-end on CPU.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue841_common as C  # noqa: E402
import issue841_scaling_common as S  # noqa: E402
import issue841_stage1_benchmark as B1  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_779.metrics import (  # noqa: E402
    bootstrap_delta_ci,
    within_condition_pearson,
)
from explore_persona_space.experiments.issue_841 import maps as MP  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue841_scaling_stage1")

MODES = B1.MODES  # ("system", "many_shot")


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("--device cuda requested but no CUDA device; falling back to cpu")
        return "cpu"
    return requested


# ── paired bootstrap vs the anchor (p-value-augmented mirror of #779's) ──────────


def paired_delta_vs_anchor(cx_n, cx_4k, cy, *, n_boot, seed) -> dict:
    """Within-condition r delta (transported@n − transported@4k) + CI + one-sided p.

    Structurally IDENTICAL to #779's ``bootstrap_delta_ci`` (condition-resample
    ONCE per replicate, paired r_n − r_4k on the shared resample) — it additionally
    returns ``p_one_sided`` (fraction of replicates with delta ≤ 0, the H2 "does
    more data help" test) for the Benjamini-Hochberg refinement. cx_n / cx_4k are
    per-condition x-arrays for the SAME conditions in order; cy the matched y.
    """
    rng = np.random.default_rng(seed)
    r_n = within_condition_pearson(cx_n, cy)["r"]
    r_4k = within_condition_pearson(cx_4k, cy)["r"]
    point = (r_n - r_4k) if (np.isfinite(r_n) and np.isfinite(r_4k)) else float("nan")
    n_cond = len(cy)
    idx = np.arange(n_cond)
    deltas = []
    for _ in range(n_boot):
        samp = rng.choice(idx, size=n_cond, replace=True)
        ra = within_condition_pearson([cx_n[i] for i in samp], [cy[i] for i in samp])["r"]
        rb = within_condition_pearson([cx_4k[i] for i in samp], [cy[i] for i in samp])["r"]
        if np.isfinite(ra) and np.isfinite(rb):
            deltas.append(ra - rb)
    if not deltas:
        return {
            "delta": point,
            "lo": float("nan"),
            "hi": float("nan"),
            "excludes_zero": False,
            "p_one_sided": float("nan"),
            "n_boot_valid": 0,
        }
    arr = np.asarray(deltas)
    lo, hi = float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))
    return {
        "delta": point,
        "lo": lo,
        "hi": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "p_one_sided": float(np.mean(arr <= 0.0)),  # H2: delta > 0 (more data helps)
        "n_boot_valid": len(arr),
    }


def benjamini_hochberg(pvals: list[float], q: float = 0.05) -> list[bool]:
    """BH-FDR: return a boolean survivor mask (NaN p-values never survive)."""
    finite = [(i, p) for i, p in enumerate(pvals) if np.isfinite(p)]
    survive = [False] * len(pvals)
    if not finite:
        return survive
    finite.sort(key=lambda x: x[1])
    m = len(finite)
    k_max = 0
    for rank, (_, p) in enumerate(finite, start=1):
        if p <= q * rank / m:
            k_max = rank
    for rank, (i, _) in enumerate(finite, start=1):
        if rank <= k_max:
            survive[i] = True
    return survive


# ── per-cell win + paired records ────────────────────────────────────────────────


def _grouped(x, mat, mode):
    """Per-condition (x, y) arrays for one mode with NaN-x pruned + std/N gates."""
    cx, cy = C.group_by_condition(x, mat["y"], mat["cond"], mat["mode"], mode)
    gx, gy = [], []
    for xi, yi in zip(cx, cy, strict=True):
        m = np.isfinite(xi)
        if m.sum() >= 3:
            gx.append(xi[m])
            gy.append(yi[m])
    return gx, gy


def process_trait(
    trait, mat, r_b, maps_by_n, schemes, ns_all, anchor_n, *, n_boot, seed, device, smoke
):
    """Per-(scheme, source, mode) win + paired-delta records across all n; + fidelity + proj."""
    traj_dev = torch.from_numpy(np.ascontiguousarray(mat["traj"])).to(
        device=device, dtype=torch.float32
    )
    rb_dev = torch.from_numpy(np.ascontiguousarray(r_b)).to(device=device, dtype=torch.float32)

    cells: list[dict] = []
    fidelity: dict = {}
    proj_store: dict = {}
    for scheme, tgt in schemes.items():
        r_b_tgt = rb_dev[tgt]
        fidelity[scheme] = {}
        ceiling = B1._proj(traj_dev[:, tgt, :], r_b_tgt)  # row 3 raw-target (map-independent)
        proj_store[f"{scheme}__ceiling"] = ceiling
        for src in B1.source_grid(tgt, smoke):
            row2 = B1._proj(traj_dev[:, src, :], rb_dev[src])  # raw_source (map-independent)
            row1b = B1._proj(traj_dev[:, src, :], r_b_tgt)  # id_transport (map-independent)
            proj_store[f"{scheme}__{src}__raw_source"] = row2
            proj_store[f"{scheme}__{src}__id_transport"] = row1b
            transported: dict[int, np.ndarray] = {}
            fidelity[scheme][str(src)] = {}
            for n in ns_all:
                x_t, h_hat = B1._transport_proj(maps_by_n[n], traj_dev, src, tgt, r_b_tgt)
                transported[n] = x_t
                proj_store[f"{scheme}__{src}__ridge_n{n}"] = x_t
                fidelity[scheme][str(src)][str(n)] = B1.transport_fidelity(
                    h_hat, traj_dev, src, tgt
                )
            for mode in MODES:
                gx2, gy = _grouped(row2, mat, mode)
                gx1b, _ = _grouped(row1b, mat, mode)
                g_ceil, _ = _grouped(ceiling, mat, mode)
                if len(gy) < 2:
                    continue  # too few conditions in this mode for a within-condition read
                gt = {n: _grouped(transported[n], mat, mode)[0] for n in ns_all}
                per_n = {}
                for n in ns_all:
                    d_raw = bootstrap_delta_ci(gt[n], gx2, gy, n_boot=n_boot, seed=seed)
                    d_id = bootstrap_delta_ci(gt[n], gx1b, gy, n_boot=n_boot, seed=seed)
                    win = bool(
                        d_raw["excludes_zero"]
                        and d_raw["delta"] > 0
                        and d_id["excludes_zero"]
                        and d_id["delta"] > 0
                    )
                    # DV3 retention(n): row-r ÷ ceiling-r, JOINT condition-resample
                    # bootstrap (#779/#841 bootstrap_retention_ci, unclipped).
                    ret = (
                        B1.bootstrap_retention_ci(gt[n], g_ceil, gy, n_boot=n_boot, seed=seed)
                        if len(g_ceil) == len(gy)
                        else {"point": float("nan")}
                    )
                    entry = {
                        "win": win,
                        "vs_raw_source": d_raw,
                        "vs_id_transport": d_id,
                        "retention": ret,
                    }
                    if n != anchor_n:
                        entry["paired_vs_anchor"] = paired_delta_vs_anchor(
                            gt[n], gt[anchor_n], gy, n_boot=n_boot, seed=seed
                        )
                    per_n[n] = entry
                cells.append(
                    {
                        "scheme": scheme,
                        "target_layer": tgt,
                        "source": src,
                        "mode": mode,
                        "per_n": per_n,
                    }
                )
            logger.info("[%s/%s] source=%d done (%d n-points)", trait, scheme, src, len(ns_all))
    proj_store["y"] = mat["y"]
    proj_store["cond"] = mat["cond"]
    proj_store["mode_is_manyshot"] = (mat["mode"] == "many_shot").astype(np.int64)
    return {"cells": cells, "fidelity": fidelity, "proj": proj_store}


# ── curve aggregation across cells ────────────────────────────────────────────────


def aggregate_curve(all_cells, ns_scaling, anchor_n, *, bh_q, seed) -> dict:
    """win_count(n) + net/newly/dropped + mean_paired_delta(n) + BH survivors."""
    total_cells = len(all_cells)
    win_at = {n: [bool(c["per_n"][n]["win"]) for c in all_cells] for n in [anchor_n, *ns_scaling]}
    anchor_rets = np.array(
        [c["per_n"][anchor_n]["retention"].get("point", float("nan")) for c in all_cells],
        dtype=float,
    )
    anchor_rets_finite = anchor_rets[np.isfinite(anchor_rets)]
    curve: dict = {
        "total_cells": total_cells,
        "anchor_n": anchor_n,
        "win_count_anchor": int(sum(win_at[anchor_n])),
        "mean_retention_anchor": (
            float(np.mean(anchor_rets_finite)) if anchor_rets_finite.size else float("nan")
        ),
        "by_n": {},
    }
    rng = np.random.default_rng(seed)
    for n in ns_scaling:
        newly = [i for i in range(total_cells) if win_at[n][i] and not win_at[anchor_n][i]]
        dropped = [i for i in range(total_cells) if win_at[anchor_n][i] and not win_at[n][i]]
        deltas = np.array(
            [c["per_n"][n]["paired_vs_anchor"]["delta"] for c in all_cells], dtype=float
        )
        finite = deltas[np.isfinite(deltas)]
        mean_delta = float(np.mean(finite)) if finite.size else float("nan")
        # cell-level bootstrap CI of the mean paired-delta (resample cells).
        if finite.size:
            boot = [
                float(np.mean(rng.choice(finite, size=finite.size, replace=True)))
                for _ in range(1000)
            ]
            mlo, mhi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        else:
            mlo = mhi = float("nan")
        pos_sig = sum(
            1
            for c in all_cells
            if c["per_n"][n]["paired_vs_anchor"]["excludes_zero"]
            and c["per_n"][n]["paired_vs_anchor"]["delta"] > 0
        )
        neg_sig = sum(
            1
            for c in all_cells
            if c["per_n"][n]["paired_vs_anchor"]["excludes_zero"]
            and c["per_n"][n]["paired_vs_anchor"]["delta"] < 0
        )
        # BH over ALL cells' one-sided p (transported@n > anchor); newly-winning survivors.
        pvals = [c["per_n"][n]["paired_vs_anchor"]["p_one_sided"] for c in all_cells]
        survive = benjamini_hochberg(pvals, q=bh_q)
        bh_newly = sum(1 for i in newly if survive[i])
        # DV3 retention(n): mean over cells of the joint-bootstrap retention point.
        rets = np.array(
            [c["per_n"][n]["retention"].get("point", float("nan")) for c in all_cells], dtype=float
        )
        rets_finite = rets[np.isfinite(rets)]
        mean_ret = float(np.mean(rets_finite)) if rets_finite.size else float("nan")
        curve["by_n"][str(n)] = {
            "win_count": int(sum(win_at[n])),
            "newly_winning": len(newly),
            "dropped_out": len(dropped),
            "net_win_vs_anchor": len(newly) - len(dropped),
            "mean_paired_delta": mean_delta,
            "mean_paired_delta_ci": [mlo, mhi],
            "cells_delta_pos_sig": pos_sig,
            "cells_delta_neg_sig": neg_sig,
            "bh_survivors_total": int(sum(survive)),
            "bh_newly_winning_survivors": bh_newly,
            "bh_q": bh_q,
            "chance_expectation": bh_q * total_cells,
            "mean_retention": mean_ret,
        }
    return curve


# ── inputs (real HF vs synthetic smoke) ──────────────────────────────────────────


def _synthetic_inputs(hidden, ns_all, anchor_n):
    """Fabricate a tiny 1-trait eval matrix + r_B + per-n maps (CPU, no HF)."""
    rng = np.random.default_rng(0)
    n_q, n_layers = 60, C.EXPECTED_LAYERS
    traj = rng.standard_normal((n_q, n_layers, hidden)).astype(np.float32)
    y = rng.standard_normal(n_q).astype(np.float64) * 20 + 50
    cond = rng.integers(0, 6, size=n_q)
    mode = np.array(["system" if i % 2 else "many_shot" for i in range(n_q)], dtype=object)
    mat = {
        "traj": traj,
        "y": y,
        "cond": cond,
        "mode": mode,
        "cond_ids": [f"c{i}" for i in range(6)],
        "layers": list(range(n_layers)),
    }
    r_b = rng.standard_normal((n_layers, hidden)).astype(np.float32)
    maps_by_n = {}
    for n in ns_all:
        maps = {}
        for t in range(n_layers - 1):
            maps[t] = MP.RidgeMap(
                mu=torch.zeros(hidden),
                sd=torch.ones(hidden),
                w=torch.from_numpy(
                    (rng.standard_normal((hidden, hidden)) * 0.01).astype(np.float32)
                ),
                bias=torch.zeros(hidden),
                best_lam=1.0,
                sigma=1.0,
            )
        maps_by_n[n] = maps
    schemes = {"primary": 20}  # a single target with a real source grid
    return {"evil": (mat, r_b)}, maps_by_n, schemes


def _real_inputs(traits, maps_dir, ns_all, device, smoke):
    """Load per-n ridge maps (local-or-HF) + each trait's eval matrix + r_B."""
    maps_by_n = {}
    for n in ns_all:
        path = maps_dir / f"ridge_maps_n{n}.pt"
        if not path.exists():
            from huggingface_hub import hf_hub_download

            rel = f"{S.hf_ridge_maps_bucket(n)}/ridge_maps_n{n}.pt"
            logger.info("[maps] local %s absent; fetching %s", path, rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            local = hf_hub_download(C.HF_DATA_REPO, filename=rel, repo_type="dataset")
            if not path.exists():
                path.symlink_to(Path(local).resolve())
        maps_by_n[n] = S.load_ridge_maps(path, device)
    trait_inputs = {}
    for trait in traits:
        cells = C.load_eval_cells(trait)
        mat = C.build_eval_traj_matrix(cells)
        trait_inputs[trait] = (mat, C.load_rb(trait))
    return trait_inputs, maps_by_n


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #841 Stage-1 transport-scaling curve.")
    ap.add_argument("--traits", nargs="+", default=list(C.TRAITS))
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--ns", default="", help="comma-list of fit sizes incl. anchor (default SCALING_NS)"
    )
    ap.add_argument("--anchor-n", type=int, default=S.N_ANCHOR_FIT)
    ap.add_argument("--maps-dir", type=Path, default=S.RIDGE_MAPS_DIR)
    ap.add_argument("--out-dir", type=Path, default=S.EVAL_SCALING_DIR)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--bh-q", type=float, default=0.05)
    ap.add_argument("--synthetic", action="store_true", help="smoke: fabricate tiny inputs (no HF)")
    ap.add_argument("--synthetic-hidden", type=int, default=C.EXPECTED_HIDDEN)
    ap.add_argument(
        "--smoke", action="store_true", help="1 trait, coarse source grid, small n-boot"
    )
    args = ap.parse_args()

    device = _resolve_device(args.device)
    ns_all = [int(x) for x in args.ns.split(",") if x] or list(S.SCALING_NS)
    anchor_n = args.anchor_n
    assert anchor_n in ns_all, f"anchor_n={anchor_n} must be in ns={ns_all}"
    ns_scaling = [n for n in ns_all if n != anchor_n]
    traits = ["evil"] if (args.smoke or args.synthetic) else args.traits
    logger.info(
        "device=%s ns=%s anchor=%d traits=%s n_boot=%d",
        device,
        ns_all,
        anchor_n,
        traits,
        args.n_boot,
    )

    if args.synthetic:
        trait_inputs, maps_by_n, fixed_schemes = _synthetic_inputs(
            args.synthetic_hidden, ns_all, anchor_n
        )
    else:
        trait_inputs, maps_by_n = _real_inputs(traits, args.maps_dir, ns_all, device, args.smoke)
        fixed_schemes = None

    all_cells: list[dict] = []
    fidelity_all: dict = {"traits": {}}
    proj_all: dict = {}
    result: dict = {
        "ns": ns_all,
        "anchor_n": anchor_n,
        "n_boot": args.n_boot,
        "target_layers": {"primary": C.PRIMARY_TARGET_LAYER, "companion": C.COMPANION_TARGET_LAYER},
        "traits": {},
        "metadata": C.reproducibility_metadata({"phase": "stage1_scaling", "smoke": args.smoke}),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for trait, (mat, r_b) in trait_inputs.items():
        schemes = fixed_schemes if fixed_schemes is not None else B1.schemes_for(trait, args.smoke)
        out = process_trait(
            trait,
            mat,
            r_b,
            maps_by_n,
            schemes,
            ns_all,
            anchor_n,
            n_boot=args.n_boot,
            seed=C.BOOTSTRAP_SEED,
            device=device,
            smoke=args.smoke,
        )
        result["traits"][trait] = {
            "n_cells": len(out["cells"]),
            "curve": aggregate_curve(
                out["cells"], ns_scaling, anchor_n, bh_q=args.bh_q, seed=C.BOOTSTRAP_SEED
            ),
        }
        all_cells.extend(out["cells"])
        fidelity_all["traits"][trait] = out["fidelity"]
        for k, v in out["proj"].items():
            proj_all[f"{trait}__{k}"] = v
        C.write_json_atomic(args.out_dir / "stage1_scaling.json", result)  # checkpoint per trait

    # Pooled curve across ALL traits' cells (the ~136-cell headline).
    result["pooled_curve"] = aggregate_curve(
        all_cells, ns_scaling, anchor_n, bh_q=args.bh_q, seed=C.BOOTSTRAP_SEED
    )
    C.write_json_atomic(args.out_dir / "stage1_scaling.json", result)
    C.write_json_atomic(args.out_dir / "transport_fidelity_scaling.json", fidelity_all)
    np.savez(args.out_dir / "scaling_projections.npz", **proj_all)
    logger.info(
        "[done] pooled win_count(anchor)=%d over %d cells → %s",
        result["pooled_curve"]["win_count_anchor"],
        result["pooled_curve"]["total_cells"],
        args.out_dir / "stage1_scaling.json",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
