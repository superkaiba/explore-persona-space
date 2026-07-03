#!/usr/bin/env python3
"""Issue #779 follow-up: arm-headline rerun EXTENSION — pass-2 + cross-layer variants.

Extends ``issue779_arm_headline_summaries.py`` (REUSED, not rewritten) with:

  A. **Frozen-layer rerun** of the 3-arm headline for the NEW answer-summary
     variants x arms {A LMSYS, B trait-corpus 10-rollout mean, C natural mix}:
       - the 8 pass-2 next-turn-template summaries (``v_im_end``,
         ``v_im_start``, ``v_user``, ``v_nl_after_user``, ``v_tmpl_mean``,
         ``v_tmpl_max``, ``v_full_mean``, ``v_full_max``) captured by
         ``issue779_capture_answer_summaries_pass2.py``;
       - 8 cross-LAYER combos of the EXISTING pass-1 summaries: per rollout,
         the mean and the element-wise max ACROSS the 28 layers of each pass-1
         summary (``xlmean_<s>`` / ``xlmax_<s>``) — a single 3584-dim target;
         read out (dot/cos) against r_B at the frozen layer.
  B. **Per-layer readout sweep** (all 28 layers) for the POOLING variants only
     — pass-1 ``v_max`` + pass-2 ``v_full_mean`` / ``v_full_max`` — Arm A only:
     one shared Gram factorization per layer serves all 3 targets; read out on
     each trait's eval matrix at that layer.

``g`` unchanged (labels don't depend on the summary) — skipped. Rows invalid
for any pass-1 summary (content-empty responses) are dropped via the SAME
joint mask as the phase-2 rerun; pass-2 summaries are always valid. Output:
``eval_results/issue_779/arm_headline_summaries2.json`` (checkpointed per
section per (trait, mode) / per layer) + ONE composite figure. Descope ladder
(projected wall > ~3h): sweep -> frozen layers only, then drop arms B/C for
the new variants (recorded in metadata).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Project dotenv wrapper: .env load + the shared-VM thread caps (#847) — called
# BEFORE numpy/torch freeze their pools.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue779_arm_headline as AH  # noqa: E402
import issue779_arm_headline_summaries as AS  # noqa: E402
import issue779_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_arm_headline_summaries2")

P2_SUMMARIES = (
    "v_im_end",
    "v_im_start",
    "v_user",
    "v_nl_after_user",
    "v_tmpl_mean",
    "v_tmpl_max",
    "v_full_mean",
    "v_full_max",
)
XL_VARIANTS = tuple(
    f"{red}_{s}" for s in AS.SUMMARIES for red in ("xlmean", "xlmax")
)  # 8: cross-layer mean/max of each pass-1 summary
SWEEP_VARIANTS = ("v_max", "v_full_mean", "v_full_max")
ARMS = AS.ARMS

DEFAULT_CAPTURE_DIR = AS.DEFAULT_CAPTURE_DIR  # pass-1 shards
DEFAULT_P2_DIR = AS.DEFAULT_CAPTURE_DIR / "pass2"


# ── generalized shard loaders ─────────────────────────────────────────────────


def load_layer_gen(
    capture_dir: Path,
    tag: str,
    li: int,
    n_ctx: int,
    n_rollouts: int,
    expected: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized ``AS.load_summary_layer`` for an arbitrary summaries tuple."""
    shards = sorted(capture_dir.glob(f"{tag}_summaries_shard*.pt"))
    if not shards:
        raise FileNotFoundError(f"no capture shards for {tag} under {capture_dir}")
    k = len(expected)
    S: np.ndarray | None = None
    valid = np.zeros((n_ctx, n_rollouts, k), dtype=bool)
    seen = np.zeros((n_ctx, n_rollouts), dtype=bool)
    for sp in shards:
        blob = torch.load(sp, mmap=True, weights_only=False, map_location="cpu")
        assert list(blob["summaries"]) == list(expected), (sp.name, blob["summaries"])
        col = blob["layers"].index(li)
        summ = blob["summ"][:, :, col, :].to(torch.float32).numpy()  # (n, k, H)
        if S is None:
            S = np.full((n_ctx, n_rollouts, k, summ.shape[-1]), np.nan, dtype=np.float32)
        v = blob["valid"].numpy()
        for row, (ci, ri) in enumerate(blob["index"]):
            assert not seen[ci, ri], (sp.name, ci, ri)
            seen[ci, ri] = True
            S[ci, ri] = summ[row]
            valid[ci, ri] = v[row]
    assert S is not None
    if not seen.all():
        raise RuntimeError(f"{tag}: {int((~seen).sum())} rows missing from {capture_dir}")
    return S, valid


def load_crosslayer(
    capture_dir: Path, tag: str, n_ctx: int, n_rollouts: int
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Cross-LAYER mean/max of the pass-1 summaries, per rollout.

    Returns ``(targets {xlmean_<s>/xlmax_<s>: (n_ctx, n_r, H)}, valid
    (n_ctx, n_r, 4))``. Reduces each shard in fp32 WITHOUT materializing the
    full (n, 4, 28, H) fp32 tensor (chunked rows).
    """
    shards = sorted(capture_dir.glob(f"{tag}_summaries_shard*.pt"))
    if not shards:
        raise FileNotFoundError(f"no pass-1 shards for {tag} under {capture_dir}")
    out: dict[str, np.ndarray] | None = None
    valid = np.zeros((n_ctx, n_rollouts, len(AS.SUMMARIES)), dtype=bool)
    seen = np.zeros((n_ctx, n_rollouts), dtype=bool)
    for sp in shards:
        blob = torch.load(sp, mmap=True, weights_only=False, map_location="cpu")
        assert list(blob["summaries"]) == list(AS.SUMMARIES), sp.name
        summ = blob["summ"]  # (n, 4, L, H) fp16
        hidden = summ.shape[-1]
        if out is None:
            out = {
                name: np.full((n_ctx, n_rollouts, hidden), np.nan, dtype=np.float32)
                for name in XL_VARIANTS
            }
        v = blob["valid"].numpy()
        n = summ.shape[0]
        for lo in range(0, n, 500):
            hi = min(lo + 500, n)
            chunk = summ[lo:hi].to(torch.float32)  # (c, 4, L, H)
            xm = chunk.mean(dim=2).numpy()  # (c, 4, H)
            xx = chunk.max(dim=2).values.numpy()  # (c, 4, H)
            for row in range(lo, hi):
                ci, ri = blob["index"][row]
                assert not seen[ci, ri], (sp.name, ci, ri)
                seen[ci, ri] = True
                valid[ci, ri] = v[row]
                for si, s in enumerate(AS.SUMMARIES):
                    out[f"xlmean_{s}"][ci, ri] = xm[row - lo, si]
                    out[f"xlmax_{s}"][ci, ri] = xx[row - lo, si]
    assert out is not None
    if not seen.all():
        raise RuntimeError(f"{tag}: {int((~seen).sum())} rows missing (cross-layer)")
    return out, valid


def _mean_over_valid(S: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-context mean over valid rollouts. S (n_ctx, n_r, H), v (n_ctx, n_r)."""
    with np.errstate(invalid="ignore"):
        m = np.nanmean(np.where(v[:, :, None], S, np.nan), axis=1)
    ok = v.sum(axis=1) > 0
    return m.astype(np.float32), ok


# ── target assembly per (trait, layer) ────────────────────────────────────────


def assemble_targets(
    cap1: Path, cap2: Path, trait: str, li: int
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray, dict]:
    """All 16 new-variant targets for Arm A (LMSYS rows) and Arm B (ctx means).

    Returns (Ya {variant: (n_lmsys, H)}, mask_a, Yb {variant: (n_ctx, H)},
    mask_b, diag). Joint masks intersect pass-1 validity (pass-2 always valid).
    """
    n_ctx, n_r, n_lm = AS.N_CORPUS_CTX, AS.N_CORPUS_ROLLOUTS, AS.N_LMSYS
    # pass-2 per-layer summaries.
    S2c, _v2c = load_layer_gen(cap2, trait, li, n_ctx, n_r, P2_SUMMARIES)
    S2l, _v2l = load_layer_gen(cap2, "lmsys", li, n_lm, 1, P2_SUMMARIES)
    # pass-1 cross-layer combos (layer-free — cached per trait by the caller).
    XLc, v1c = load_crosslayer(cap1, trait, n_ctx, n_r)
    XLl, v1l = load_crosslayer(cap1, "lmsys", n_lm, 1)

    mask_a = v1l[:, 0, :].all(axis=1)  # pass-1 joint validity (same as phase-2 rerun)
    mask_b = np.ones(n_ctx, dtype=bool)
    Ya: dict[str, np.ndarray] = {}
    Yb: dict[str, np.ndarray] = {}
    for si, s in enumerate(P2_SUMMARIES):
        Ya[s] = S2l[:, 0, si, :]
        m, ok = _mean_over_valid(S2c[:, :, si, :], np.ones((n_ctx, n_r), dtype=bool))
        Yb[s] = m
        mask_b &= ok
    for s in XL_VARIANTS:
        base = s.split("_", 1)[1]  # pass-1 summary name
        si = AS.SUMMARIES.index(base)
        Ya[s] = XLl[s][:, 0, :]
        m, ok = _mean_over_valid(XLc[s], v1c[:, :, si])
        Yb[s] = m
        mask_b &= ok
    diag = {
        "n_lmsys_joint_dropped": int((~mask_a).sum()),
        "n_ctx_joint_dropped": int((~mask_b).sum()),
    }
    for s in Ya:
        Ya[s] = Ya[s][mask_a]
        assert np.isfinite(Ya[s]).all(), s
    for s in Yb:
        Yb[s] = Yb[s][mask_b]
        assert np.isfinite(Yb[s]).all(), s
    return Ya, mask_a, Yb, mask_b, diag


# ── Section A: frozen-layer rerun for the new variants ────────────────────────


def run_section_a(res: dict, ctx: AH.Ctx, args: argparse.Namespace, drops: list[str]) -> None:
    sec = res.setdefault("summaries2_headline", {})
    all_variants = [*P2_SUMMARIES, *XL_VARIANTS]
    arms_here = ARMS if "arms_bc_for_new_variants" not in drops else ("A_lmsys",)
    cell_cache: dict[tuple[str, int], dict] = {}
    for trait in C.TRAITS:
        tr = sec.setdefault(trait, {})
        for mode in AH.MODES:
            if mode in tr:
                logger.info("[A %s %s] already checkpointed; skipping", trait, mode)
                continue
            li = AH.FROZEN_LAYERS[trait][mode]
            if (trait, li) not in cell_cache:
                t0 = time.time()
                mat = ctx.mat(trait, li)
                Xev = mat["c_last"]
                rb_l = ctx.rb(trait)[li]
                Xa_full, _ = ctx.lmsys_layer(li)
                Xb_full, _vb, _yb = ctx.corpus_layer(trait, li)
                Ya, mask_a, Yb, mask_b, diag = assemble_targets(
                    args.capture_dir, args.p2_dir, trait, li
                )
                Xa, Xb = Xa_full[mask_a], Xb_full[mask_b]
                arm_data: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {
                    "A_lmsys": (Xa, Ya),
                    "B_trait": (Xb, Yb),
                    "C_mix": (
                        np.concatenate([Xa, Xb]),
                        {s: np.concatenate([Ya[s], Yb[s]]) for s in Ya},
                    ),
                }
                per_arm: dict[str, dict] = {}
                recon: dict[str, dict] = {}
                for arm in arms_here:
                    Xh, Yh = arm_data[arm]
                    logger.info(
                        "[A %s L%d] arm %s: shared Gram fit (n=%d)", trait, li, arm, len(Xh)
                    )
                    gr = AH.GramRidge(Xh)
                    sm: dict[str, dict] = {}
                    for s in all_variants:
                        pred = gr.predict(Yh[s], Xev)
                        dot = F.dot_readout(pred, rb_l)
                        cos = F.cosine_readout(pred, rb_l)
                        sm[s] = {
                            "dot": AH._mode_metrics(dot, mat, n_boot=args.n_boot, seed=args.seed),
                            "cos": AH._mode_metrics(cos, mat, n_boot=args.n_boot, seed=args.seed),
                            "gcv_lambda": gr.last_lambda,
                        }
                    per_arm[arm] = {"n_train": len(Xh), "summaries": sm}
                    logger.info(
                        "[A %s L%d] arm %s: %d-fold recon (%d targets)",
                        trait,
                        li,
                        arm,
                        args.n_folds,
                        len(Yh),
                    )
                    recon[arm] = AH.heldout_recon_multi(
                        Xh, Yh, n_folds=args.n_folds, seed=args.seed
                    )
                cell_cache[(trait, li)] = {
                    "per_arm": per_arm,
                    "recon": recon,
                    "diag": diag,
                    "mat_rows": len(mat["y"]),
                    "wall_s": time.time() - t0,
                }
            cell = cell_cache[(trait, li)]
            tr[mode] = {
                "layer": li,
                "n_eval_rows": cell["mat_rows"],
                "arms_evaluated": list(arms_here),
                "per_arm": {
                    arm: {
                        "n_train": d["n_train"],
                        "summaries": {
                            s: {
                                "dot": d["summaries"][s]["dot"][mode],
                                "cos": d["summaries"][s]["cos"][mode],
                                "gcv_lambda": d["summaries"][s]["gcv_lambda"],
                            }
                            for s in d["summaries"]
                        },
                    }
                    for arm, d in cell["per_arm"].items()
                },
                "recon_heldout": cell["recon"],
                "valid_diag": cell["diag"],
                "cell_wall_s": cell["wall_s"],
            }
            C.write_json_atomic(args.out_json, res)
            best = max(
                all_variants,
                key=lambda s: (
                    tr[mode]["per_arm"][arms_here[0]]["summaries"][s]["cos"]["point"]
                    if np.isfinite(
                        tr[mode]["per_arm"][arms_here[0]]["summaries"][s]["cos"]["point"]
                    )
                    else -9
                ),
            )
            logger.info(
                "[A %s %s L%d] done (%.0fs); best armA cos variant: %s (%.3f)",
                trait,
                mode,
                li,
                cell["wall_s"],
                best,
                tr[mode]["per_arm"][arms_here[0]]["summaries"][best]["cos"]["point"],
            )


# ── Section B: per-layer readout sweep (pooling variants, Arm A) ──────────────


def run_section_b(res: dict, ctx: AH.Ctx, args: argparse.Namespace, drops: list[str]) -> None:
    sec = res.setdefault("per_layer_sweep", {})
    layers = (
        sorted({li for t in C.TRAITS for li in AH.FROZEN_LAYERS[t].values()})
        if "sweep_frozen_only" in drops
        else list(range(C.EXPECTED_LAYERS))
    )
    n_lm = AS.N_LMSYS
    # Pass-1 lmsys validity (joint over the 4 pass-1 summaries; matches phase 2).
    _S1_probe, v1l = AS.load_summary_layer(args.capture_dir, "lmsys", 0, n_lm, 1)
    mask_a = v1l[:, 0, :].all(axis=1)
    del _S1_probe
    v_max_idx = AS.SUMMARIES.index("v_max")
    fm_idx = P2_SUMMARIES.index("v_full_mean")
    fx_idx = P2_SUMMARIES.index("v_full_max")
    t_start = time.time()
    for li in layers:
        lkey = f"L{li}"
        if lkey in sec:
            logger.info("[B %s] already checkpointed; skipping", lkey)
            continue
        S1, _ = AS.load_summary_layer(args.capture_dir, "lmsys", li, n_lm, 1)
        S2, _ = load_layer_gen(args.p2_dir, "lmsys", li, n_lm, 1, P2_SUMMARIES)
        targets = {
            "v_max": S1[:, 0, v_max_idx, :][mask_a],
            "v_full_mean": S2[:, 0, fm_idx, :][mask_a],
            "v_full_max": S2[:, 0, fx_idx, :][mask_a],
        }
        for name, Y in targets.items():
            assert np.isfinite(Y).all(), (lkey, name)
        Xa_full, _ = ctx.lmsys_layer(li)
        gr = AH.GramRidge(Xa_full[mask_a])
        entry: dict = {"n_train": int(mask_a.sum()), "traits": {}}
        for trait in C.TRAITS:
            mat = ctx.mat(trait, li)
            rb_l = ctx.rb(trait)[li]
            td: dict = {}
            for name, Y in targets.items():
                pred = gr.predict(Y, mat["c_last"])
                cos = F.cosine_readout(pred, rb_l)
                mm = AH._mode_metrics(cos, mat, n_boot=args.sweep_n_boot, seed=args.seed)
                td[name] = {m: mm[m] for m in AH.MODES}
            td["pv_raw"] = {
                m: AH._mode_metrics(mat["pv_raw"], mat, n_boot=args.sweep_n_boot, seed=args.seed)[m]
                for m in AH.MODES
            }
            entry["traits"][trait] = td
        sec[lkey] = entry
        C.write_json_atomic(args.out_json, res)
        elapsed = time.time() - t_start
        done_n = len(sec)
        logger.info(
            "[B %s] done (%d/%d layers, %.0fs elapsed, ~%.0fs projected)",
            lkey,
            done_n,
            len(layers),
            elapsed,
            elapsed / done_n * len(layers),
        )


# ── figure ────────────────────────────────────────────────────────────────────


def make_figure(res: dict, args: argparse.Namespace) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    all_variants = [*P2_SUMMARIES, *XL_VARIANTS]
    fig, axes = plt.subplots(3, 3, figsize=(20, 13), layout="tight")
    colors = paper_palette(max(len(ARMS), len(SWEEP_VARIANTS)))
    # Rows 0-1: frozen-layer grid (modes x traits), grouped bars per variant/arm.
    # Guard: with --only b (or a partial resume) section A is absent — leave the
    # panel empty instead of KeyError-ing after the sweep already ran.
    head = res.get("summaries2_headline", {})
    for row, mode in enumerate(AH.MODES):
        for col, trait in enumerate(C.TRAITS):
            ax = axes[row][col]
            entry = head.get(trait, {}).get(mode)
            if entry is None:
                ax.set_axis_off()
                continue
            arms_here = entry["arms_evaluated"]
            width = 0.8 / len(arms_here)
            xpos = np.arange(len(all_variants))
            for ai, arm in enumerate(arms_here):
                pts, errs = [], []
                for s in all_variants:
                    mm = entry["per_arm"][arm]["summaries"][s]["cos"]
                    pts.append(mm["point"])
                    errs.append(
                        [
                            max(0.0, mm["point"] - mm["lo"]) if np.isfinite(mm["lo"]) else 0,
                            max(0.0, mm["hi"] - mm["point"]) if np.isfinite(mm["hi"]) else 0,
                        ]
                    )
                ax.bar(
                    xpos + (ai - (len(arms_here) - 1) / 2) * width,
                    pts,
                    width,
                    yerr=np.array(errs).T,
                    capsize=1.5,
                    color=colors[ai],
                    label=AS.ARM_LABELS[arm] if (row == 0 and col == 0) else None,
                )
            ax.axhline(0.0, color="gray", lw=0.5)
            ax.set_xticks(xpos)
            ax.set_xticklabels(all_variants, rotation=60, ha="right", fontsize=6)
            mode_lbl = "system" if mode == "system" else "many-shot"
            ax.set_title(f"{trait} — {mode_lbl} (L{entry['layer']})", fontsize=9)
            if col == 0:
                ax.set_ylabel("within-cond r (cos readout)")
    # Row 2: per-layer sweep, one panel per trait, lines per variant (system mode)
    sweep = res.get("per_layer_sweep", {})
    layers_sorted = sorted(int(k[1:]) for k in sweep)
    for col, trait in enumerate(C.TRAITS):
        ax = axes[2][col]
        for vi, name in enumerate(SWEEP_VARIANTS):
            for mode, ls in zip(AH.MODES, ("-", "--"), strict=True):
                ys = [sweep[f"L{li}"]["traits"][trait][name][mode]["point"] for li in layers_sorted]
                ax.plot(
                    layers_sorted,
                    ys,
                    ls,
                    color=colors[vi],
                    lw=1.2,
                    label=f"{name} ({mode})" if col == 0 else None,
                )
        pv = [sweep[f"L{li}"]["traits"][trait]["pv_raw"]["system"]["point"] for li in layers_sorted]
        ax.plot(
            layers_sorted,
            pv,
            ":",
            color="gray",
            lw=1.0,
            label="pv_raw (system)" if col == 0 else None,
        )
        for mode in AH.MODES:
            ax.axvline(AH.FROZEN_LAYERS[trait][mode], color="black", lw=0.5, alpha=0.4)
        ax.set_xlabel("layer")
        ax.set_title(f"{trait} — Arm A per-layer sweep", fontsize=9)
        if col == 0:
            ax.set_ylabel("within-cond r (cos readout)")
    handles, labels = [], []
    for ax in (axes[0][0], axes[2][0]):
        h, lb = ax.get_legend_handles_labels()
        handles += h
        labels += lb
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=7)
    figs = savefig_paper(fig, "arm_headline_summaries2", dir=args.fig_dir)
    plt.close(fig)
    return str(figs.get("png", ""))


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 pass-2 + cross-layer rerun.")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--sweep-n-boot", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--k-draws", type=int, default=5)  # Ctx compat
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--p2-dir", type=Path, default=DEFAULT_P2_DIR)
    parser.add_argument("--only", choices=["a", "b", "ab"], default="ab")
    parser.add_argument(
        "--drop",
        nargs="*",
        default=[],
        choices=["sweep_frozen_only", "arms_bc_for_new_variants"],
        help="descope ladder switches (recorded in metadata)",
    )
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / "arm_headline_summaries2.json",
    )
    parser.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures" / "issue_779")
    args = parser.parse_args()
    torch.set_num_threads(int(args.n_threads))

    res: dict = {}
    params = {"n_boot": args.n_boot, "seed": args.seed, "n_folds": args.n_folds}
    if args.out_json.exists() and not args.fresh:
        with open(args.out_json) as f:
            res = json.load(f)
        prior = {k: res.get("metadata", {}).get(k) for k in params}
        if prior != params:
            raise SystemExit(f"existing {args.out_json} params {prior} != {params}; use --fresh")
        logger.info("Resuming from %s", args.out_json)
    res["metadata"] = C.reproducibility_metadata(
        {
            "script": "issue779_arm_headline_summaries2",
            **params,
            "sweep_n_boot": args.sweep_n_boot,
            "frozen_layers": AH.FROZEN_LAYERS,
            "p2_summaries": list(P2_SUMMARIES),
            "xl_variants": list(XL_VARIANTS),
            "sweep_variants": list(SWEEP_VARIANTS),
            "drops": list(args.drop),
            "caveats": [
                "cross-layer (xl*) targets are layer-collapsed; readout uses r_B at the "
                "frozen readout layer",
                "pass-2 positions are teacher-forced next-turn TEMPLATE tokens "
                "(<|im_end|>\\n<|im_start|>user\\n extension); pass-1 v_last_turn is the "
                "addendum's (e) — the \\n after <|im_end|> — and lives in "
                "arm_headline_summaries.json",
                "per-layer sweep is Arm A only, pooling variants only, by design",
                "g omitted: judge labels unchanged by summary choice",
            ],
        }
    )

    ctx = AH.Ctx(args)
    res["metadata"]["equivalence_gate"] = AH.equivalence_gate(ctx.bundle, args.seed)
    C.write_json_atomic(args.out_json, res)

    if "a" in args.only:
        logger.info("=== Section A: frozen-layer rerun (16 new variants) ===")
        run_section_a(res, ctx, args, args.drop)
    if "b" in args.only:
        logger.info("=== Section B: per-layer sweep (pooling variants, Arm A) ===")
        run_section_b(res, ctx, args, args.drop)

    res.setdefault("figures", {})["arm_headline_summaries2"] = make_figure(res, args)
    C.write_json_atomic(args.out_json, res)
    logger.info("Done. Wrote %s", args.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
