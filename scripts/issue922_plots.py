#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #922 Phase 4: figures (+ artifact uploads via --upload).

Over-produces per the plan §6 figure list — the analyzer picks the heroes:

- ``hero1_position_atlas`` — per-layer single-step r2_id (context-only ridge /
  MLP, token-informed ridge, embedding-only ridge) + the shuffled-context null
  band + the identity-0 line; answer-content segment.
- ``hero2_rollout_skill`` — rollout skill vs horizon k at the 6 pre-registered
  read-out layers: context-only roll vs frozen null (0 line), mean-drift null,
  token-informed ceiling, MLP companion.
- ``hero3_readout_bars`` — per (trait × mode): frozen read, horizon-mean
  rolled read, #779 direct-predictor reference, true-answer ceiling, 95% CIs.
- exploratory: layer × answer-position R² heatmap, autocorrelation curve,
  ‖Δ‖ per layer, GCV-λ per (layer, arm), H2 ratio-vs-depth, per-context
  rollout-skill scatter, DV3 per-unit projection-vs-score scatters, transfer
  deltas.

``--upload`` additionally pushes the plan §10 artifacts to the HF data repo
(``issue922_nexttoken/``): all step-1 boundary-map weights + the ℓ* answer
maps (fp16), the test-context store subset, the eval-condition store, the
eval JSONs + figures (non-LFS path — uploads unconditionally).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/numpy so the shared-VM thread caps bind (#847)

import issue922_common as C  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette,
    savefig_paper,
    set_paper_style,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue922_plots")

LAYER_ORDER = ["emb"] + [str(i) for i in range(28)]


def _load(p: Path) -> dict | None:
    if not p.exists():
        logger.warning("[plots] %s absent — figures depending on it are skipped", p)
        return None
    with open(p) as f:
        return json.load(f)


def _xs_ys(atlas, cls, arm, space, seg, metric="r2_id"):
    xs, ys = [], []
    for li, bk in enumerate(LAYER_ORDER):
        cell = atlas["cells"].get(f"{bk}|{seg}", {}).get(f"{cls}_{arm}", {}).get(space, {})
        v = cell.get(metric)
        if v is not None:
            xs.append(li)
            ys.append(v)
    return xs, ys


def hero1(atlas, fig_dir):
    pal = paper_palette(5)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    series = [
        ("ridge", "ctx", "context-only ridge", pal[0], "-"),
        ("mlp", "ctx", "context-only MLP", pal[0], "--"),
        ("ridge", "tok", "token-informed ridge (ceiling)", pal[1], "-"),
        ("mlp", "tok", "token-informed MLP", pal[1], "--"),
        ("ridge", "emb", "embedding-only ridge", pal[2], "-"),
    ]
    for cls, arm, label, color, ls in series:
        xs, ys = _xs_ys(atlas, cls, arm, "raw", "answer")
        if xs:
            ax.plot(xs, ys, ls, color=color, label=label, marker="o", ms=3)
    band = atlas.get("shuffle_null", {}).get("band_by_row", {})
    bx, blo, bhi = [], [], []
    for li, bk in enumerate(LAYER_ORDER):
        if bk in band:
            bx.append(li)
            blo.append(band[bk]["p2_5"])
            bhi.append(band[bk]["p97_5"])
    if bx:
        ax.fill_between(bx, blo, bhi, color="gray", alpha=0.3, label="shuffled-context null band")
    ax.axhline(0.0, color="black", lw=0.8, label="copy-previous (identity)")
    ax.set_xticks(range(0, 29, 4))
    ax.set_xticklabels([LAYER_ORDER[i] for i in range(0, 29, 4)])
    ax.set_xlabel("layer (emb, block 0..27)")
    ax.set_ylabel("held-out identity-relative R² on Δ")
    ax.set_title("Single-step next-position Δ predictability (answer segment, raw)")
    ax.legend(fontsize=7)
    savefig_paper(fig, "hero1_position_atlas", fig_dir)
    plt.close(fig)


def hero2(roll, fig_dir, readout_blocks):
    pal = paper_palette(5)
    blocks = [str(b) for b in readout_blocks]
    blocks = [
        b for b in blocks if b in roll["variants"]["ridge_ctx_boundary_first"]["pooled_r2_id"]
    ]
    if not blocks:
        blocks = list(roll["variants"]["ridge_ctx_boundary_first"]["pooled_r2_id"])[:6]
    n = len(blocks)
    fig, axes = plt.subplots(
        max(1, (n + 2) // 3), min(3, n), figsize=(9.5, 3.0 * max(1, (n + 2) // 3)), squeeze=False
    )
    ks = list(range(1, roll["k_max"] + 1))
    for i, bk in enumerate(blocks):
        ax = axes[i // 3][i % 3]
        for name, label, color in [
            ("ridge_ctx_boundary_first", "context-only roll (boundary-first)", pal[0]),
            ("ridge_ctx_naive", "naive roll (answer map at k=1)", pal[3]),
            ("mlp_ctx_boundary_first", "MLP companion roll", pal[4]),
            ("mean_drift", "mean-drift null", pal[2]),
            ("tok_ceiling", "token-informed ceiling", pal[1]),
        ]:
            v = roll["variants"].get(name, {}).get("pooled_r2_id", {}).get(bk)
            if v:
                ax.plot(ks, v, label=label, color=color, lw=1.2)
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_title(f"block {bk}")
        ax.set_xlabel("horizon k")
        ax.set_ylabel("pooled R² (frozen-relative)")
        ax.set_ylim(-1.0, 1.0)
    axes[0][0].legend(fontsize=6)
    fig.suptitle("Rollout skill vs horizon (test contexts)")
    savefig_paper(fig, "hero2_rollout_skill", fig_dir)
    plt.close(fig)


def hero3(dv3, fig_dir):
    traits = [t for t, v in dv3.get("traits", {}).items() if "primary" in v and "skipped" not in v]
    if not traits:
        return
    pal = paper_palette(8)  # r2 review minor: 3 distinct v6-mode colors (were all pal[4])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), squeeze=False)
    for mi, mode in enumerate(("system", "many_shot")):
        ax = axes[0][mi]
        labels, vals, los, his, colors = [], [], [], [], []
        for t in traits:
            pr = dv3["traits"][t]["primary"]
            series = [
                ("frozen", pr.get("frozen", {}), pal[0]),
                ("rolled (hm)", pr.get("horizon_mean", {}), pal[1]),
            ]
            # v6 modes (plan HERO-3): rolled-b1 / rolled-b2 (FiLM) / direct-c —
            # one distinct color per mode (they are distinct series in the bars)
            for (name, key), color in zip(
                [
                    ("rolled b1", "rolled_b1_ridge"),
                    ("rolled b2 (FiLM)", "rolled_film"),
                    ("direct-c", "direct_c"),
                ],
                (pal[4], pal[5], pal[6]),
                strict=True,
            ):
                if key in pr:
                    series.append((name, pr[key].get("horizon_mean", {}), color))
            for name, met, color in series:
                m = met.get(mode, {})
                labels.append(f"{t}\n{name}")
                vals.append(m.get("point", np.nan))
                los.append(m.get("point", np.nan) - m.get("lo", np.nan))
                his.append(m.get("hi", np.nan) - m.get("point", np.nan))
                colors.append(color)
            ref = pr.get("pv_direct_reference", {}).get(mode)
            ref_pt = ref.get("pv_raw_r", ref.get("point")) if isinstance(ref, dict) else ref
            if ref_pt is not None:
                labels.append(f"{t}\n#779 direct")
                vals.append(ref_pt)
                los.append(0)
                his.append(0)
                colors.append(pal[2])
            rp = dv3["traits"][t].get("restricted_panel", {})
            ceil = rp.get("true_answer_ceiling_horizon_mean", {}).get(mode, {})
            if ceil.get("point") is not None:
                labels.append(f"{t}\ntrue-answer ceiling*")
                vals.append(ceil["point"])
                los.append(ceil["point"] - ceil.get("lo", np.nan))
                his.append(ceil.get("hi", np.nan) - ceil["point"])
                colors.append(pal[3])
        x = np.arange(len(labels))
        ax.bar(
            x,
            vals,
            yerr=[np.maximum(0.0, np.nan_to_num(los)), np.maximum(0.0, np.nan_to_num(his))],
            color=colors,
            capsize=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax.set_ylabel("within-condition Pearson r")
        ax.set_title(f"{mode} (* = captured-subset panel)")
        ax.axhline(0.0, color="black", lw=0.8)
    fig.suptitle("Trait read-out: rolled vs frozen vs references (primary ℓ*)")
    fig.tight_layout()
    savefig_paper(fig, "hero3_readout_bars", fig_dir)
    plt.close(fig)


def hero4(cond, roll, fig_dir, readout_blocks):
    """The H6/H7 exhibit (plan §6 HERO-4): (left) paired per-context single-step
    r2 delta b2_film − b1_grad at the ℓ* rows + companion form points; (right)
    rollout skill vs k at the primary ℓ* rows — ctx-roll vs direct-c vs
    rolled-b1 vs rolled-b2(FiLM) vs mean-drift, with the H7 paired
    horizon-mean delta inset as text."""
    pal = paper_palette(6)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), squeeze=False)
    ax = axes[0][0]
    h6 = (cond or {}).get("h6", {})
    plotted_left = False
    for ci_i, (comp, color) in enumerate(
        [("film", pal[0]), ("lowrank", pal[2]), ("mixture", pal[3])]
    ):
        reads = h6.get(f"{comp}_minus_b1_grad", {}).get("per_lstar", {})
        if not reads:
            continue
        xs = [int(bk) + 0.15 * ci_i for bk in reads]
        means = [reads[bk]["mean"] for bk in reads]
        lo = [reads[bk]["mean"] - reads[bk]["lo"] for bk in reads]
        hi = [reads[bk]["hi"] - reads[bk]["mean"] for bk in reads]
        ax.errorbar(
            xs,
            means,
            yerr=[np.maximum(0.0, lo), np.maximum(0.0, hi)],
            fmt="o",
            ms=4,
            color=color,
            capsize=2,
            label=f"{comp} − b1_grad" + (" (PRIMARY)" if comp == "film" else ""),
        )
        plotted_left = True
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("read-out block ℓ*")
    ax.set_ylabel("paired per-context Δ r2_id (single-step)")
    ax.set_title("H6: operator reshaping vs additive drift")
    if plotted_left:
        ax.legend(fontsize=7)
    ax = axes[0][1]
    variants = (roll or {}).get("variants", {})
    ks = list(range(1, (roll or {}).get("k_max", 0) + 1))
    bks = [
        str(b)
        for b in readout_blocks
        if str(b) in variants.get("direct_c", {}).get("pooled_r2_id", {})
    ]
    bk0 = bks[0] if bks else None
    if bk0 is None:  # fall back to any block direct_c carries
        cand = list(variants.get("direct_c", {}).get("pooled_r2_id", {}))
        bk0 = cand[0] if cand else None
    if bk0 is not None:
        for name, label, color in [
            ("ridge_ctx_boundary_first", "ctx roll (arm a)", pal[0]),
            ("direct_c", "direct-c (arm c)", pal[1]),
            ("b1_ridge_roll", "rolled b1 (ridge)", pal[2]),
            ("b1_grad_roll", "rolled b1 (grad twin)", pal[4]),
            ("film_roll", "rolled b2 FiLM", pal[3]),
            ("mean_drift", "mean-drift null", pal[5]),
        ]:
            v = variants.get(name, {}).get("pooled_r2_id", {}).get(bk0)
            if v:
                ax.plot(ks, v, label=label, color=color, lw=1.2)
        h7 = (roll or {}).get("h7_paired", {}).get("ctx_roll_minus_direct_c", {}).get(bk0)
        if h7:
            ax.text(
                0.02,
                0.02,
                f"H7 Δhm (roll−direct) = {h7['mean']:.3f} [{h7['lo']:.3f}, {h7['hi']:.3f}]",
                transform=ax.transAxes,
                fontsize=7,
            )
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(f"H7: recursion vs direct (block {bk0})")
        ax.set_xlabel("horizon k")
        ax.set_ylabel("pooled R² (frozen-relative)")
        ax.legend(fontsize=6)
    fig.suptitle("H6/H7: conditioned + direct transition structures")
    savefig_paper(fig, "hero4_h6_h7", fig_dir)
    plt.close(fig)


def exploratory(atlas, roll, dv3, dv4, fig_dir, out_dir):  # noqa: C901 — one block per exploratory figure
    pal = paper_palette(4)
    # layer × answer-position R² heatmap
    r2p = atlas.get("diagnostics", {}).get("r2_by_ansrel", {})
    if r2p:
        rows = [bk for bk in LAYER_ORDER if bk in r2p]
        width = max(len(v) for v in r2p.values())
        mat = np.full((len(rows), width), np.nan)
        for i, bk in enumerate(rows):
            for j, v in enumerate(r2p[bk]):
                if v is not None:
                    mat[i, j] = v
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=-0.1, vmax=np.nanmax(mat))
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(rows, fontsize=5)
        ax.set_xlabel("answer-relative source position")
        ax.set_ylabel("layer")
        ax.set_title("context-only ridge r2_id by (layer × answer position)")
        fig.colorbar(im, ax=ax, label="r2_id")
        savefig_paper(fig, "exploratory_layer_position_heatmap", fig_dir)
        plt.close(fig)
    # autocorr + ‖Δ‖ curves
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    ac = atlas.get("diagnostics", {}).get("autocorr_by_row", {})
    dn = atlas.get("diagnostics", {}).get("delta_norm_by_row", {})
    xs = [i for i, bk in enumerate(LAYER_ORDER) if bk in ac]
    axes[0].plot(xs, [ac[LAYER_ORDER[i]] for i in xs], "-o", ms=3, color=pal[0])
    axes[0].set_title("cos(h_t, h_{t+1}) per layer (answer, test)")
    axes[0].set_xlabel("layer index")
    axes[1].plot(xs, [dn[LAYER_ORDER[i]] for i in xs], "-o", ms=3, color=pal[1])
    axes[1].set_title("mean ‖Δ‖ per layer")
    axes[1].set_xlabel("layer index")
    savefig_paper(fig, "exploratory_autocorr_deltanorm", fig_dir)
    plt.close(fig)
    # H2 ratio vs depth
    h2 = atlas.get("h2_ratio", {})
    if h2.get("blocks"):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(
            h2["blocks"],
            h2["ratio_ctx_over_tok"],
            "-o",
            ms=3,
            color=pal[0],
            label="ridge ratio ctx/tok",
        )
        ax.plot(
            h2["blocks"],
            h2.get("mlp_ratio_ctx_over_tok", []),
            "--o",
            ms=3,
            color=pal[1],
            label="MLP ratio ctx/tok",
        )
        ax.set_xlabel("block")
        ax.set_ylabel("carried-context share r2_ctx / r2_tok")
        rho = h2.get("spearman_ratio", {})
        ax.set_title(f"H2 depth profile (Spearman ρ={rho.get('rho', float('nan')):.2f})")
        ax.legend(fontsize=7)
        savefig_paper(fig, "exploratory_h2_ratio_depth", fig_dir)
        plt.close(fig)
    # GCV λ per (layer, arm)
    lam = atlas.get("cells", {})
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for arm, color in (("ctx", pal[0]), ("tok", pal[1]), ("emb", pal[2])):
        xs, ys = [], []
        for li, bk in enumerate(LAYER_ORDER):
            v = lam.get(f"{bk}|answer", {}).get(f"ridge_{arm}", {}).get("raw", {}).get("best_lam")
            if v is not None:
                xs.append(li)
                ys.append(v)
        if xs:
            ax.semilogy(xs, ys, "-o", ms=3, color=color, label=f"ridge {arm}")
    ax.set_xlabel("layer index")
    ax.set_ylabel("GCV-selected λ")
    ax.legend(fontsize=7)
    savefig_paper(fig, "exploratory_gcv_lambda", fig_dir)
    plt.close(fig)
    # per-context rollout-skill scatter (k=4 vs k=16) at first read-out block present
    npz_p = out_dir / "rollout_skill_percontext.npz"
    if npz_p.exists() and roll is not None:
        z = np.load(npz_p)
        key = next((k for k in z.files if k.startswith("skill__ridge_ctx_boundary_first__")), None)
        if key is not None:
            sk = z[key]
            if sk.shape[1] >= 16:
                fig, ax = plt.subplots(figsize=(4.5, 4.5))
                ax.scatter(sk[:, 3], sk[:, 15], s=6, alpha=0.4, color=pal[0])
                ax.set_xlabel("per-context skill at k=4")
                ax.set_ylabel("per-context skill at k=16")
                ax.set_title(f"rollout skill per context ({key.split('__')[-1]})")
                ax.axhline(0, color="black", lw=0.6)
                ax.axvline(0, color="black", lw=0.6)
                savefig_paper(fig, "exploratory_percontext_rollout_scatter", fig_dir)
                plt.close(fig)
    # DV3 per-unit scatters
    proj_p = out_dir / "readout_projections.npz"
    if proj_p.exists() and dv3 is not None:
        z = np.load(proj_p, allow_pickle=True)
        for trait in C.TRAITS:
            pk = next((k for k in z.files if k.startswith(f"proj__{trait}__")), None)
            if pk is None:
                continue
            proj = z[pk]
            y = z[f"y__{trait}"]
            hm = proj[:, : C.READOUT_K_MAX].mean(axis=1)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            ax.scatter(hm, y, s=8, alpha=0.5, color=pal[0])
            ax.set_xlabel("horizon-mean rolled projection")
            ax.set_ylabel("judged trait score")
            ax.set_title(f"{trait} — per-unit rolled read vs score")
            savefig_paper(fig, f"exploratory_perunit_{trait}", fig_dir)
            plt.close(fig)
    # transfer vs in-corpus deltas
    if dv4 is not None and atlas is not None:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        for arm, color in (("ctx", pal[0]), ("tok", pal[1])):
            xs, ds = [], []
            for li, bk in enumerate(LAYER_ORDER):
                a = atlas["cells"].get(f"{bk}|answer", {}).get(f"ridge_{arm}", {}).get("raw", {})
                t = dv4.get("single_step", {}).get(bk, {}).get(f"ridge_{arm}", {})
                if a.get("r2_id") is not None and t.get("r2_id") is not None:
                    xs.append(li)
                    ds.append(t["r2_id"] - a["r2_id"])
            if xs:
                ax.plot(xs, ds, "-o", ms=3, color=color, label=f"ridge {arm}")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("layer index")
        ax.set_ylabel("transfer − in-corpus r2_id")
        ax.set_title("LMSYS→eval-condition transfer delta (single-step)")
        ax.legend(fontsize=7)
        savefig_paper(fig, "exploratory_transfer_delta", fig_dir)
        plt.close(fig)


def upload_artifacts(args) -> dict:
    """Plan §10 uploads: maps subset (fp16) + store subsets + JSONs/figures."""
    events = {}
    stage = Path(args.upload_stage)
    stage.mkdir(parents=True, exist_ok=True)
    # maps: ALL boundary weights + answer maps at the read-out blocks, fp16.
    ridge_p = args.maps / "maps_ridge.pt"
    if ridge_p.exists():
        blob = torch.load(ridge_p, weights_only=False)
        keep_rows = {C.block_to_row(b) for b in C.READOUT_BLOCKS}
        sub = {
            "boundary": {
                arm: {
                    r: {k: (v.half() if torch.is_tensor(v) else v) for k, v in st.items()}
                    for r, st in d.items()
                }
                for arm, d in blob["boundary"].items()
            },
            "answer_lstar": {
                arm: {
                    r: {k: (v.half() if torch.is_tensor(v) else v) for k, v in st.items()}
                    for r, st in d.items()
                    if r in keep_rows
                }
                for arm, d in blob["answer"].items()
            },
            "sigma_by_row": blob["sigma_by_row"],
            "lambdas": blob["lambdas"],
            "metadata": blob["metadata"],
        }
        mdir = stage / "maps"
        mdir.mkdir(exist_ok=True)
        # v6: b1 closed-form [h,c] maps at the read-out rows ride the same file
        if blob.get("b1_answer"):
            sub["b1_answer_lstar"] = {
                r: {k: (v.half() if torch.is_tensor(v) else v) for k, v in st.items()}
                for r, st in blob["b1_answer"].items()
                if r in keep_rows
            }
        torch.save(sub, mdir / "maps_boundary_and_lstar_fp16.pt")
        events["maps"] = C.upload_dir_bulk(
            mdir, f"{C.HF_OUT_PREFIX}/maps", commit_message="issue922 maps (boundary + lstar fp16)"
        )
    # v6: conditioned gradient forms (ℓ* rows, fp16) + arm-c ℓ* all-k map files
    cond_p = args.maps / "maps_conditioned.pt"
    keep_rows = {C.block_to_row(b) for b in C.READOUT_BLOCKS}
    cstage_written = False
    cdir = stage / "maps_conditioned"
    if cond_p.exists():
        cblob = torch.load(cond_p, weights_only=False)
        cdir.mkdir(exist_ok=True)
        sub_c = {
            "forms": {
                form: {
                    r: {
                        **{k: v for k, v in pb.items() if k != "weights"},
                        "weights": {k: v.half() for k, v in pb["weights"].items()},
                    }
                    for r, pb in per_row.items()
                    if r in keep_rows
                }
                for form, per_row in cblob["forms"].items()
            },
            "rank": cblob.get("rank"),
            "n_mix": cblob.get("n_mix"),
            "recipe": cblob.get("recipe"),
            "capacity": cblob.get("capacity"),
            "metadata": cblob.get("metadata"),
        }
        torch.save(sub_c, cdir / "maps_conditioned_lstar_fp16.pt")
        cstage_written = True
    ddir = args.maps / "direct"
    if ddir.is_dir():
        cdir.mkdir(exist_ok=True)
        import shutil

        for r in sorted(keep_rows):
            p = ddir / f"direct_row_{r:02d}.pt"
            if p.exists():  # weights already fp16 on disk (plan §10 pricing)
                shutil.copy2(p, cdir / p.name)
                cstage_written = True
    if cstage_written:
        events["maps_conditioned"] = C.upload_dir_bulk(
            cdir,
            f"{C.HF_OUT_PREFIX}/maps_conditioned",
            commit_message="issue922 conditioned/direct maps (lstar fp16)",
        )
    # store subsets: test contexts + eval store.
    lm = args.store / "lmsys"
    if lm.is_dir():
        store = C.load_store(args.store, "lmsys")
        split = C.make_split(
            len(store["ctx_ids"]), n_fit=C.N_FIT, n_val=C.N_VAL, n_test=C.N_TEST, seed=C.SPLIT_SEED
        )
        tdir = stage / "store_test"
        tdir.mkdir(exist_ok=True)
        recs = {}
        for i in split["test"]:
            ci = store["ctx_ids"][int(i)]
            lo, npos = int(store["pos_lo"][int(i)]), int(store["n_pos"][int(i)])
            recs[ci] = {
                "h": store["h"][:, lo : lo + npos, :].permute(1, 0, 2).clone(),
                "token_ids": store["token_ids"][lo : lo + npos].clone(),
                "segments": store["segments"][lo : lo + npos - 1].astype(np.uint8),
                **store["meta"][ci],
            }
        torch.save(
            {"corpus": "lmsys_test", "blocks": store["blocks"], "contexts": recs},
            tdir / "store_test_contexts.pt",
        )
        events["store_test"] = C.upload_dir_bulk(
            tdir, f"{C.HF_OUT_PREFIX}/store_test", commit_message="issue922 test-context store"
        )
    ev = args.store / "eval_subset"
    if ev.is_dir():
        events["store_eval"] = C.upload_dir_bulk(
            ev,
            f"{C.HF_OUT_PREFIX}/store_eval",
            allow_patterns=["*.pt", "*.json"],
            commit_message="issue922 eval-condition store",
        )
    # eval JSONs + figures. NOTE the >10 MB npz files force-route to LFS, so a
    # quota-403 must reroute to the overflow repo instead of killing the run
    # pre-sentinel (r1 review Minor) — the overflow event record keeps it loud.
    events["eval_results"] = C.upload_dir_bulk(
        args.results,
        f"{C.HF_OUT_PREFIX}/eval_results",
        allow_patterns=["*.json", "*.npz"],
        commit_message="issue922 eval results",
        allow_overflow=True,
    )
    events["figures"] = C.upload_dir_bulk(
        args.figures,
        f"{C.HF_OUT_PREFIX}/figures",
        allow_patterns=["*.png", "*.pdf", "*.json"],
        commit_message="issue922 figures",
        allow_overflow=False,
    )
    return events


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #922 figures + uploads.")
    ap.add_argument("--results", type=Path, default=Path("eval_results/issue_922"))
    ap.add_argument("--out", type=Path, default=Path("figures/issue_922"))
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--store", type=Path, default=Path("/workspace/issue922_store"))
    ap.add_argument("--maps", type=Path, default=Path("/workspace/issue922_maps"))
    ap.add_argument("--upload-stage", type=Path, default=Path("/workspace/issue922_upload_stage"))
    args = ap.parse_args()
    ap_figs = args.out
    ap_figs.mkdir(parents=True, exist_ok=True)
    set_paper_style()

    atlas = _load(args.results / "stage0_position_atlas.json")
    roll = _load(args.results / "rollout_skill.json")
    dv3 = _load(args.results / "readout_benchmark.json")
    dv4 = _load(args.results / "transfer_eval.json")
    cond = _load(args.results / "conditioned_arms.json")
    if atlas:
        hero1(atlas, ap_figs)
    if roll:
        hero2(roll, ap_figs, C.READOUT_BLOCKS)
    if dv3:
        hero3(dv3, ap_figs)
    if cond or (roll and "h7_paired" in roll):
        hero4(cond, roll, ap_figs, C.READOUT_BLOCKS)
    if atlas or roll or dv3 or dv4:
        exploratory(atlas or {}, roll, dv3, dv4, ap_figs, args.results)
    if args.upload:
        args.figures = ap_figs
        events = upload_artifacts(args)
        C.write_json_atomic(
            args.results / "upload_events.json",
            {
                "events": events,
                "metadata": C.reproducibility_metadata({"script": "issue922_plots"}),
            },
        )
        logger.info("[upload] %s", json.dumps(events))
    logger.info("DONE figures under %s", ap_figs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
