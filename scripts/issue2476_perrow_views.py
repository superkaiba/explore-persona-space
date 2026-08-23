"""Per-unit views for the #2476 clean-result round-2 revision (two reconciler
blockers): the per-CANDIDATE firing-fraction ECDF behind Result 2's attrition
census (`attrition-per-unit-view-missing`) and the per-ROW retrieval-rank view
behind Result 4's acc@k cells (`retrieval-per-row-view-nonexistent`).

Inputs (staged from HF `superkaiba1/explore-persona-space-data` +
`chanind/qwen2.5-7B-it-layer-20-saes` into --stage-dir; idempotent skip when
present):

  issue2476_turnavg/analysis_tensors/eval/{alive_c,alive_b,ib_c,armb_maps,
    ftrue_b}.npz + ftrue_c_all.fp16.npy      (this run's P5 stores)
  issue2476_turnavg/analysis_tensors/sae_c/  (this run's trained SAE weights)
  issue1482_error_analysis/analysis_tensors/percontext/
    refit_holdout__ridge__seed0.npz          (fresh-arm dense map predictions —
    CONVENTION DELTA, recorded: this run's own refit npz was not uploaded; the
    parent round's copy is the same map refit through the same EA.phase_p1_fit
    recipe, whose reproduction this run's G1 gate verified to |dR2| <= 3.5e-7,
    holdout_rows asserted identical to ib_c.npz rows; the surrogate is further
    validated below by reproducing the committed per-feature R2 arrays)

  All nine staged inputs are fetched at pinned revision 89cfa76cdcd4 — a pin
  added post-run with content-identity verified: every staged file's last HF
  commit (refit npz 2026-07-18, this run's stores 2026-08-23 11:23-11:39 UTC)
  predates this script's 2026-08-23 ~16:32 UTC unpinned-main fetch, so the
  pinned revision serves the same content the recompute consumed.

Recompute convention: `_encode_restricted` + the `_knn_retrieval_chunked` rank
formula VERBATIM from scripts/issue2476_turnavg_sae.py (imported, not copied,
where importable; the rank kernel is re-derived here because the driver's
returns aggregates only — same distance formula, same 1e-9 relative tolerance
mid-rank ties), on CPU float32 — the PRODUCTION convention's dtype (the
committed cells ran cuda fp32). Measured dtype sensitivity (2026-08-23
diagnostic, all 24 cells): CPU fp32 reproduces every committed acc@k cell
bitwise except c/t0/map/euclidean (5.0e-5 = 1 row of 20,000, present at BOTH
dtypes — the parent-surrogate prediction delta, not a backend effect) and
c/t1/map/cosine (1.0e-4 = 2 rows); CPU fp64 (the exact-parity reference the
driver's tests pin) additionally diverges by up to 1.18e-2 in sparse COSINE
cells only — its tighter arithmetic pulls large equal-distance blocks inside
the 1e-9 tie tolerance and mid-ranks them, the open
`retrieval-device-pin-residual` near-tie class made visible. The script
reports every per-cell delta and gates the worst at 2e-3.

Outputs (committed under eval_results/issue_2476/turnavg/ + figures/):
  perrow_retrieval_ranks.npz        rank_{arm}_t{tier}_{pred}_{metric} arrays
  perrow_retrieval_ranks_meta.json  provenance + acc@k reproduction table
  figures/issue_2476/i2476_candidate_firing_ecdf.{png,pdf,meta.json}
  figures/issue_2476/i2476_retrieval_rank_ecdf.{png,pdf,meta.json}

Run from the worktree root (VM CPU, ~10 min):
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue2476_perrow_views.py
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "eval_results" / "issue_2476" / "turnavg"
STAGE_DEFAULT = "/mnt/eps-data/thomasjiralerspong/issue2476_r2crc"
DATA_REPO = "superkaiba1/explore-persona-space-data"
# Post-run pin (see module docstring): main head at pin time; every staged file's
# last touching commit predates the run's unpinned fetch, so content is identical.
DATA_REPO_REVISION = "89cfa76cdcd4207d95c1fec1c3131f36e21beec0"
STAGE_FILES = (
    "issue2476_turnavg/analysis_tensors/eval/alive_c.npz",
    "issue2476_turnavg/analysis_tensors/eval/alive_b.npz",
    "issue2476_turnavg/analysis_tensors/eval/ftrue_c_all.fp16.npy",
    "issue2476_turnavg/analysis_tensors/eval/ib_c.npz",
    "issue2476_turnavg/analysis_tensors/eval/armb_maps.npz",
    "issue2476_turnavg/analysis_tensors/eval/ftrue_b.npz",
    "issue2476_turnavg/analysis_tensors/sae_c/sae_weights.safetensors",
    "issue2476_turnavg/analysis_tensors/sae_c/cfg.json",
    "issue1482_error_analysis/analysis_tensors/percontext/refit_holdout__ridge__seed0.npz",
)
ACC_REPRO_TOL = 2e-3  # backend near-tie ceiling; any nonzero delta is REPORTED per cell


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "issue2476_turnavg_sae", ROOT / "scripts" / "issue2476_turnavg_sae.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["issue2476_turnavg_sae"] = mod
    spec.loader.exec_module(mod)
    return mod


def _stage(stage: Path) -> None:
    """Idempotently pull the HF inputs (skip-if-present; ~2.6 GB total)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    for f in STAGE_FILES:
        tgt = stage / f
        if tgt.exists() and tgt.stat().st_size > 0:
            continue
        hub.retry_transient(
            lambda f=f: hf_hub_download(
                DATA_REPO, f, repo_type="dataset", revision=DATA_REPO_REVISION, local_dir=str(stage)
            ),
            what="hf_hub_download",
        )
        print(f"[stage] {f}", flush=True)


def _ranks_cpu32(pred, true, metric: str, np, torch, block: int = 2048):
    """Per-row retrieval ranks — the `_knn_retrieval_chunked` formula verbatim
    (pool == true rows; mid-rank ties at 1e-9 relative tolerance), CPU fp32
    (the production convention's dtype; see the module docstring's measured
    dtype-sensitivity note for why fp64 is NOT used here)."""
    pt = torch.as_tensor(np.asarray(pred), dtype=torch.float32)
    qt = torch.as_tensor(np.asarray(true), dtype=torch.float32)
    n = pt.shape[0]
    assert pt.shape == qt.shape, (pt.shape, qt.shape)
    if metric == "euclidean":
        q2 = (qt * qt).sum(dim=1).unsqueeze(0)
        pool_t = qt
    else:
        assert metric == "cosine", metric
        q2 = None
        pool_t = qt / (torch.sqrt((qt * qt).sum(dim=1, keepdim=True)) + 1e-12)
    ranks_t = torch.empty(n, dtype=torch.float64)
    for s in range(0, n, block):
        pb = pt[s : s + block]
        if metric == "euclidean":
            d = (pb * pb).sum(dim=1).unsqueeze(1) + q2 - 2.0 * (pb @ pool_t.T)
        else:
            pn = pb / (torch.sqrt((pb * pb).sum(dim=1, keepdim=True)) + 1e-12)
            d = 1.0 - pn @ pool_t.T
        rows = torch.arange(len(pb))
        d_true = d[rows, torch.arange(s, s + len(pb))]
        tol = 1e-9 * torch.clamp(d_true.abs(), min=1e-12).unsqueeze(1)
        dt1 = d_true.unsqueeze(1)
        closer = (d < dt1 - tol).sum(dim=1)
        tied = ((d - dt1).abs() <= tol).sum(dim=1) - 1
        ranks_t[s : s + len(pb)] = 1.0 + closer.to(torch.float64) + 0.5 * tied.to(torch.float64)
    return ranks_t.numpy()


def _arm_ranks(tag, f_true, preds, tier, committed, np, torch, arrays, repro):
    """All (tier x pred x metric) rank vectors for one arm + the acc@k
    reproduction table against the committed retrieval_{tag}.json cells."""
    for t in (0, 1, 2):
        m = tier == t
        if int(m.sum()) == 0:
            continue
        ft = np.ascontiguousarray(np.asarray(f_true[:, m], np.float32))
        cell = committed["tiers"][str(t)]
        for pname, parr in preds.items():
            pa = np.ascontiguousarray(np.asarray(parr[:, m], np.float32))
            for metric in ("euclidean", "cosine"):
                t0 = time.time()
                ranks = _ranks_cpu32(pa, ft, metric, np, torch)
                arrays[f"rank_{tag}_t{t}_{pname}_{metric}"] = ranks.astype(np.float32)
                row = {}
                for k in (1, 5, 10):
                    got = float((ranks <= k).mean())
                    want = float(cell[pname][metric]["acc_at_k"][str(k)])
                    row[f"acc@{k}"] = {"recomputed": got, "committed": want, "delta": got - want}
                repro[f"{tag}/t{t}/{pname}/{metric}"] = row
                print(
                    f"[ranks] {tag}/t{t}/{pname}/{metric} n_feat={int(m.sum())} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )


def _fig_firing_ecdf(stage: Path, drv, fig_dir: Path) -> None:
    """Per-CANDIDATE firing-fraction ECDF per tier, both instruments, with the
    1%-of-fit-rows alive floor marked (Result 2's per-unit view)."""
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
    )

    tier_colors = paper_palette(3)
    bins = (0,) + tuple(drv.S.MATRYOSHKA_TIER_BOUNDS)
    fig, axes = plt.subplots(1, 2, figsize=figsize_iclr_panels(2))
    for ax, tag in zip(axes, ("c", "b"), strict=True):
        z = np.load(stage / f"issue2476_turnavg/analysis_tensors/eval/alive_{tag}.npz")
        counts = np.asarray(z["counts"], np.int64)
        n_fit = int(z["n_fit_rows"])
        frac = counts / float(n_fit)
        for t in (0, 1, 2):
            fr = np.sort(frac[bins[t] : bins[t + 1]])
            n_tot = len(fr)
            nz = fr[fr > 0]
            n_zero = n_tot - len(nz)
            # ECDF over ALL candidates; log-x shows the nonzero tail, the curve
            # entering at y = P(candidate never fires on a fit row).
            y = (n_zero + np.arange(1, len(nz) + 1)) / n_tot
            label = (
                f"{drv.TIER_LABELS[t].splitlines()[0]} "
                f"({n_zero / n_tot:.0%} of {n_tot:,} never fire)"
            )
            ax.plot(nz, y, color=tier_colors[t], lw=1.0, label=label)
        ax.axvline(0.01, ls="--", lw=0.8, color="gray")
        ax.text(0.011, 0.05, "1% alive floor", rotation=90, fontsize=5, color="gray")
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("firing fraction over fit rows")
        ax.set_title(drv._ARM_LABELS[tag], fontsize=6)
    axes[0].set_ylabel("fraction of candidates")
    axes[0].legend(fontsize=5, loc="lower right")
    savefig_paper(fig, "i2476_candidate_firing_ecdf", dir=fig_dir)
    plt.close(fig)
    print("[fig] i2476_candidate_firing_ecdf", flush=True)


def _fig_rank_ecdf(arrays: dict, n_pools: dict, drv, fig_dir: Path) -> None:
    """Per-ROW retrieval-rank ECDF (euclidean; map solid, identity+bias dashed),
    per tier, one panel per arm (Result 4's per-unit view)."""
    import matplotlib.pyplot as plt
    import numpy as np

    from explore_persona_space.analysis.paper_plots import (
        figsize_iclr_panels,
        paper_palette,
        savefig_paper,
    )

    tier_colors = paper_palette(3)
    fig, axes = plt.subplots(1, 2, figsize=figsize_iclr_panels(2))
    for ax, tag in zip(axes, ("c", "b"), strict=True):
        for t in (0, 1, 2):
            for pred, ls in (("map", "-"), ("ib", "--")):
                key = f"rank_{tag}_t{t}_{pred}_euclidean"
                if key not in arrays:
                    continue
                r = np.sort(np.asarray(arrays[key], np.float64))
                y = np.arange(1, len(r) + 1) / len(r)
                lbl = f"{drv.TIER_LABELS[t].splitlines()[0]}, {pred}" if pred == "map" else None
                ax.plot(r, y, ls=ls, color=tier_colors[t], lw=0.9, label=lbl)
        ax.set_xscale("log")
        ax.set_xlim(1, n_pools[tag])
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel("rank of the true conversation (euclidean)")
        ax.set_title(drv._ARM_LABELS[tag], fontsize=6)
    axes[0].set_ylabel("fraction of held-out rows")
    axes[0].plot([], [], ls="--", color="gray", lw=0.9, label="identity+bias (dashed)")
    axes[0].legend(fontsize=5, loc="center left")
    savefig_paper(fig, "i2476_retrieval_rank_ecdf", dir=fig_dir)
    plt.close(fig)
    print("[fig] i2476_retrieval_rank_ecdf", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-dir", default=STAGE_DEFAULT)
    args = ap.parse_args()
    stage = Path(args.stage_dir)
    stage.mkdir(parents=True, exist_ok=True)
    _stage(stage)
    drv = _load_driver()
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np
    import torch

    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style("iclr")
    fig_dir = ROOT / "figures" / "issue_2476"
    e = stage / "issue2476_turnavg/analysis_tensors/eval"
    arrays: dict = {}
    repro: dict = {}
    r2_checks: dict = {}

    # ── fresh arm (c) ────────────────────────────────────────────────────────
    az = np.load(e / "alive_c.npz")
    alive_c = np.asarray(az["alive_ids"], np.int64)
    n_fit_c = int(az["n_fit_rows"])
    tier_c = drv.S.tier_of(alive_c)
    f_true_c = np.asarray(np.load(e / "ftrue_c_all.fp16.npy", mmap_mode="r")[n_fit_c:])
    hz = np.load(stage / STAGE_FILES[-1])
    vhat = np.asarray(hz["holdout_pred16"], np.float16)
    hold_rows = np.asarray(hz["holdout_rows"], np.int64)
    ibz = np.load(e / "ib_c.npz")
    assert (np.asarray(ibz["rows"], np.int64) == hold_rows).all(), "surrogate row drift vs ib_c"
    ib16 = np.asarray(ibz["pred16"], np.float16)
    sae_c = drv.MatryoshkaBatchTopKSAE.load_local(
        stage / "issue2476_turnavg/analysis_tensors/sae_c", device="cpu"
    )
    t0 = time.time()
    f_pred_c = drv._encode_restricted(sae_c, vhat, np.arange(len(vhat)), alive_c)
    f_ib_c = drv._encode_restricted(sae_c, ib16, np.arange(len(ib16)), alive_c)
    print(f"[encode] arm c done elapsed={time.time() - t0:.0f}s", flush=True)
    del sae_c
    # encode + surrogate validation: reproduce the committed per-feature R2
    for name, pred in (("map", f_pred_c), ("ib", f_ib_c)):
        want = np.asarray(
            np.load(EV / f"perfeature_c_{'encodepred' if name == 'map' else 'ib'}.npz")["r2"]
        )
        got = drv.EA._per_feature_metrics(pred, f_true_c)["r2"]
        fin = np.isfinite(want) & np.isfinite(got)
        assert (np.isfinite(want) == np.isfinite(got)).all()
        dmax = float(np.abs(got[fin] - want[fin]).max())
        r2_checks[f"c/{name}"] = {"max_abs_delta_r2": dmax, "n_features": int(fin.sum())}
        print(f"[r2-check] c/{name} max|dR2|={dmax:.2e}", flush=True)
    rc = json.loads((EV / "retrieval_c.json").read_text())
    _arm_ranks("c", f_true_c, {"map": f_pred_c, "ib": f_ib_c}, tier_c, rc, np, torch, arrays, repro)
    arrays["rows_c"] = hold_rows
    arrays["alive_ids_c"] = alive_c
    arrays["tier_c"] = tier_c
    del f_pred_c, f_ib_c, f_true_c

    # ── bridge arm (b) ───────────────────────────────────────────────────────
    ab = np.load(e / "alive_b.npz")
    alive_b = np.asarray(ab["alive_ids"], np.int64)
    tier_b = drv.S.tier_of(alive_b)
    bz = np.load(e / "armb_maps.npz")
    fz = np.load(e / "ftrue_b.npz")
    row_idx_all = np.asarray(fz["row_idx"], np.int64)
    row_idx_score = np.asarray(bz["row_idx_score"], np.int64)
    te_pos = np.searchsorted(row_idx_all, row_idx_score)
    assert (row_idx_all[te_pos] == row_idx_score).all(), "armb score-row alignment drift"
    f_true_b = np.asarray(fz["f_true"], np.float16)[te_pos]
    sae_lm = drv.S.SAELensJumpReLU.load(
        drv.M.SAE_IDS["lmsys"], device="cpu", cache_dir=stage / "chanind"
    )
    t0 = time.time()
    f_pred_b = drv._encode_restricted(
        sae_lm, np.asarray(bz["pred16"], np.float16), np.arange(len(row_idx_score)), alive_b
    )
    f_ib_b = drv._encode_restricted(
        sae_lm, np.asarray(bz["ib_pred16"], np.float16), np.arange(len(row_idx_score)), alive_b
    )
    print(f"[encode] arm b done elapsed={time.time() - t0:.0f}s", flush=True)
    del sae_lm
    for name, pred in (("map", f_pred_b), ("ib", f_ib_b)):
        want = np.asarray(
            np.load(EV / f"perfeature_b_{'encodepred' if name == 'map' else 'ib'}.npz")["r2"]
        )
        got = drv.EA._per_feature_metrics(pred, f_true_b)["r2"]
        fin = np.isfinite(want) & np.isfinite(got)
        assert (np.isfinite(want) == np.isfinite(got)).all()
        dmax = float(np.abs(got[fin] - want[fin]).max())
        r2_checks[f"b/{name}"] = {"max_abs_delta_r2": dmax, "n_features": int(fin.sum())}
        print(f"[r2-check] b/{name} max|dR2|={dmax:.2e}", flush=True)
    rb = json.loads((EV / "retrieval_b.json").read_text())
    _arm_ranks("b", f_true_b, {"map": f_pred_b, "ib": f_ib_b}, tier_b, rb, np, torch, arrays, repro)
    arrays["rows_b"] = row_idx_score
    arrays["alive_ids_b"] = alive_b
    arrays["tier_b"] = tier_b

    # ── reproduction gate + artifacts ────────────────────────────────────────
    worst = max(abs(v["delta"]) for row in repro.values() for v in row.values())
    nonzero = {k: row for k, row in repro.items() if any(v["delta"] != 0.0 for v in row.values())}
    print(
        f"[repro] worst |delta acc@k| = {worst:.2e}; nonzero cells: {sorted(nonzero)}", flush=True
    )
    assert worst <= ACC_REPRO_TOL, (
        f"acc@k reproduction FAILED (worst delta {worst:.2e} > {ACC_REPRO_TOL}) — "
        "a bug in this recompute, never a new result"
    )
    tmp = EV / ".tmp_perrow_retrieval_ranks.npz"
    np.savez(tmp, **arrays)
    tmp.replace(EV / "perrow_retrieval_ranks.npz")
    meta = {
        "what": (
            "per-row retrieval ranks (pool = held-out true feature rows; mid-rank ties) "
            "for every tier x {map, identity+bias} x {euclidean, cosine} cell of "
            "retrieval_c.json / retrieval_b.json; rows_{c,b} give the conversation row ids "
            "in rank-array order"
        ),
        "backend": (
            "cpu float32 — the production convention's dtype (committed cells ran cuda "
            "fp32). Measured dtype sensitivity: cpu fp64 diverges from the committed "
            "aggregates by up to 1.18e-2 in sparse cosine cells only (tie-tolerance "
            "mid-ranking of equal-distance blocks; the retrieval-device-pin-residual "
            "near-tie class); fp32 reproduces every cell to the deltas below"
        ),
        "fresh_arm_map_pred_source": (
            "issue1482_error_analysis/analysis_tensors/percontext/refit_holdout__ridge__seed0.npz "
            "(the parent round's copy of the SAME EA.phase_p1_fit ridge refit; this run's own "
            "refit npz was not uploaded — G1-verified |dR2| <= 3.5e-7 vs the committed values, "
            "holdout_rows asserted identical to ib_c.npz rows, and the encode reproduces the "
            "committed per-feature R2 arrays to the max deltas below)"
        ),
        "per_feature_r2_reproduction": r2_checks,
        "acc_at_k_reproduction": repro,
        "worst_abs_acc_delta": worst,
    }
    (EV / "perrow_retrieval_ranks_meta.json").write_text(json.dumps(meta, indent=1))
    print("[write] perrow_retrieval_ranks.npz + meta", flush=True)

    n_pools = {"c": int(rc["n_pool"]), "b": int(rb["n_pool"])}
    _fig_rank_ecdf(arrays, n_pools, drv, fig_dir)
    _fig_firing_ecdf(stage, drv, fig_dir)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
