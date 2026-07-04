#!/usr/bin/env python3
"""Issue #779 follow-up: 3-arm headline rerun with ALTERNATIVE answer summaries.

Re-runs the arm-headline within-condition read (``issue779_arm_headline``
section 1 machinery, REUSED — GramRidge / Ctx / heldout_recon_multi /
method_metrics) with the answer summary S swapped from the original
mean-response ``v(x)`` to each of the four teacher-forced summaries captured by
``issue779_capture_answer_summaries.py``:

  S in {v_last_turn, v_last_content, v_max, v_first}
  arms: A (LMSYS 5000, single-rollout S), B (trait corpus 2400,
        10-rollout-mean of S), C (natural A+B concat)

Per (trait, frozen-layer mode): fit ridge ``h: c_last -> S-profile`` (shared
Gram factorization per arm across all 4 summary targets), read the
within-condition Pearson r of the dot/cos readouts on the FIXED eval rig at the
FROZEN step0 layers, plus 5-fold held-out recon R2 per arm (test-own-mean).
``g`` is UNCHANGED by summary choice (labels don't change) — skipped.

Eval-reference caveats (documented in the output):
  - The stored eval rig references are ``r2_last`` / ``r2_max`` — pooled
    PROJECTIONS over the response span [prompt_len, full_len) INCLUDING the
    ``<|im_end|>`` + trailing newline template tokens. The span-FINAL position
    (r2_last) is therefore the ``v_last_turn`` position (token id 198), NOT the
    last content token; ``r2_max`` is the max of per-token projections (not an
    element-wise max), so it matches ``v_max`` only loosely. No stored
    reference exists for v_last_content / v_first.
  - Arm-A targets come from the r7 g-label LMSYS rollouts (pass_b's original
    rollout text was never persisted), index-aligned to the cached pass_b
    ``cx_last``.

Rows invalid for ANY summary (content-empty responses) are dropped via a JOINT
mask so ONE factorization serves all summaries; drop counts are reported. NaN
never coerced. Output: ``eval_results/issue_779/arm_headline_summaries.json``
(checkpointed per trait x mode) + one figure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_arm_headline as AH  # noqa: E402
import issue779_common as C  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_arm_headline_summaries")

SUMMARIES = ("v_last_turn", "v_last_content", "v_max", "v_first")
ARMS = ("A_lmsys", "B_trait", "C_mix")
ARM_LABELS = {"A_lmsys": "LMSYS (generic)", "B_trait": "trait corpus", "C_mix": "natural mix"}
DEFAULT_CAPTURE_DIR = Path("/mnt/eps-data/thomasjiralerspong/issue779-grid/final_token_capture")
BASELINE_JSON = PROJECT_ROOT / "eval_results" / "issue_779" / "arm_headline.json"

N_CORPUS_CTX = 2400
N_CORPUS_ROLLOUTS = 10
N_LMSYS = 5000


# ── capture-shard loading ─────────────────────────────────────────────────────


def load_summary_layer(
    capture_dir: Path, tag: str, li: int, n_ctx: int, n_rollouts: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble one tag's summaries at ONE layer from the capture shards.

    Returns ``(S, valid)`` with ``S`` (n_ctx, n_rollouts, 4, H) fp32 (NaN on
    invalid) and ``valid`` (n_ctx, n_rollouts, 4) bool. Fail-loud on missing
    shards / rows or a summary-order mismatch.
    """
    shards = sorted(capture_dir.glob(f"{tag}_summaries_shard*.pt"))
    if not shards:
        raise FileNotFoundError(f"no capture shards for {tag} under {capture_dir}")
    S: np.ndarray | None = None
    valid = np.zeros((n_ctx, n_rollouts, 4), dtype=bool)
    seen = np.zeros((n_ctx, n_rollouts), dtype=bool)
    for sp in shards:
        blob = torch.load(sp, mmap=True, weights_only=False, map_location="cpu")
        assert list(blob["summaries"]) == list(SUMMARIES), blob["summaries"]
        col = blob["layers"].index(li)
        summ = blob["summ"][:, :, col, :].to(torch.float32).numpy()  # (n, 4, H)
        if S is None:
            S = np.full((n_ctx, n_rollouts, 4, summ.shape[-1]), np.nan, dtype=np.float32)
        v = blob["valid"].numpy()  # (n, 4)
        for row, (ci, ri) in enumerate(blob["index"]):
            assert not seen[ci, ri], (sp.name, ci, ri)
            seen[ci, ri] = True
            S[ci, ri] = summ[row]
            valid[ci, ri] = v[row]
    assert S is not None
    if not seen.all():
        missing = int((~seen).sum())
        raise RuntimeError(f"{tag}: {missing} (ctx, rollout) rows missing from shards")
    return S, valid


def corpus_summary_targets(
    capture_dir: Path, trait: str, li: int
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Per-context 10-rollout-MEAN of each summary at one layer (Arm B targets).

    Mean over VALID rollouts only (drop-never-coerce). Returns
    ``(targets {summary: (n_ctx, H)}, joint_mask (n_ctx,), diag)`` — the joint
    mask keeps contexts with >= 1 valid rollout for EVERY summary.
    """
    S, valid = load_summary_layer(capture_dir, trait, li, N_CORPUS_CTX, N_CORPUS_ROLLOUTS)
    targets: dict[str, np.ndarray] = {}
    ok = np.ones(N_CORPUS_CTX, dtype=bool)
    diag: dict = {}
    for si, name in enumerate(SUMMARIES):
        v = valid[:, :, si]  # (n_ctx, n_r)
        n_valid = v.sum(axis=1)
        ok &= n_valid > 0
        with np.errstate(invalid="ignore"):
            m = np.nanmean(np.where(v[:, :, None], S[:, :, si, :], np.nan), axis=1)
        targets[name] = m.astype(np.float32)
        diag[name] = {
            "n_invalid_rollouts": int((~v).sum()),
            "n_ctx_zero_valid": int((n_valid == 0).sum()),
        }
    diag["n_ctx_joint_dropped"] = int((~ok).sum())
    return targets, ok, diag


def lmsys_summary_targets(
    capture_dir: Path, li: int
) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Single-rollout LMSYS summaries at one layer (Arm A targets) + joint mask."""
    S, valid = load_summary_layer(capture_dir, "lmsys", li, N_LMSYS, 1)
    ok = valid[:, 0, :].all(axis=1)  # row valid for ALL 4 summaries
    targets = {name: S[:, 0, si, :].astype(np.float32) for si, name in enumerate(SUMMARIES)}
    diag = {
        name: {"n_invalid_rows": int((~valid[:, 0, si]).sum())} for si, name in enumerate(SUMMARIES)
    }
    diag["n_rows_joint_dropped"] = int((~ok).sum())
    return targets, ok, diag


# ── per-(trait, layer) cell ───────────────────────────────────────────────────


def summaries_fit_cell(ctx: AH.Ctx, capture_dir: Path, trait: str, li: int) -> dict:
    """All summary x arm fits + monitors + recon for one (trait, layer)."""
    args = ctx.args
    mat = ctx.mat(trait, li)
    Xev = mat["c_last"]
    rb_l = ctx.rb(trait)[li]

    Xa_full, _ = ctx.lmsys_layer(li)
    Xb_full, _vb, _yb = ctx.corpus_layer(trait, li)
    Sa, mask_a, diag_a = lmsys_summary_targets(capture_dir, li)
    Sb, mask_b, diag_b = corpus_summary_targets(capture_dir, trait, li)
    assert Xa_full.shape[0] == N_LMSYS and Xb_full.shape[0] == N_CORPUS_CTX
    Xa, Xb = Xa_full[mask_a], Xb_full[mask_b]
    Ya = {s: Sa[s][mask_a] for s in SUMMARIES}
    Yb = {s: Sb[s][mask_b] for s in SUMMARIES}
    for s in SUMMARIES:
        assert np.isfinite(Ya[s]).all() and np.isfinite(Yb[s]).all(), s

    arms: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {
        "A_lmsys": (Xa, Ya),
        "B_trait": (Xb, Yb),
        "C_mix": (
            np.concatenate([Xa, Xb]),
            {s: np.concatenate([Ya[s], Yb[s]]) for s in SUMMARIES},
        ),
    }

    # References (fit-free): pv_raw / oracle / the stored r2_last + r2_max.
    references = {
        name: AH._mode_metrics(mat[name], mat, n_boot=args.n_boot, seed=args.seed)
        for name in ("pv_raw", "oracle", "r2_last", "r2_max")
    }

    per_arm: dict[str, dict] = {}
    recon: dict[str, dict] = {}
    for arm, (Xh, Yh) in arms.items():
        logger.info("[%s L%d] arm %s: shared Gram fit (n=%d)", trait, li, arm, len(Xh))
        gr = AH.GramRidge(Xh)
        sm: dict[str, dict] = {}
        for s in SUMMARIES:
            pred = gr.predict(Yh[s], Xev)
            monitors = {
                "dot": F.dot_readout(pred, rb_l),
                "cos": F.cosine_readout(pred, rb_l),
            }
            sm[s] = {
                kind: AH._mode_metrics(x, mat, n_boot=args.n_boot, seed=args.seed)
                for kind, x in monitors.items()
            }
            sm[s]["delta_cos_vs_pv_raw"] = {
                mode: AH._delta_vs(
                    monitors["cos"], mat["pv_raw"], mat, mode, n_boot=args.n_boot, seed=args.seed
                )
                for mode in AH.MODES
            }
            sm[s]["gcv_lambda"] = gr.last_lambda
        per_arm[arm] = {"n_train": len(Xh), "summaries": sm}
        logger.info(
            "[%s L%d] arm %s: %d-fold held-out recon (4 targets)", trait, li, arm, args.n_folds
        )
        recon[arm] = AH.heldout_recon_multi(Xh, Yh, n_folds=args.n_folds, seed=args.seed)

    return {
        "mat_rows": len(mat["y"]),
        "references": references,
        "per_arm": per_arm,
        "recon": recon,
        "valid_diag": {"lmsys": diag_a, "corpus": diag_b},
    }


def baseline_entry(trait: str, mode: str) -> dict:
    """Pull the mean-summary (v_x) baseline numbers from arm_headline*.json.

    The committed baseline is SPLIT across two files (same params, disjoint
    trait coverage): ``arm_headline.json`` carries evil (the VM free-analysis
    run) and ``arm_headline_pod.json`` carries sycophancy + hallucination (the
    sibling pod rerun). Try the primary file first, then the pod file.
    """
    e = None
    src = None
    for path in (BASELINE_JSON, BASELINE_JSON.with_name("arm_headline_pod.json")):
        if not path.exists():
            continue
        with open(path) as f:
            base = json.load(f)
        e = base.get("arm_headline", {}).get(trait, {}).get(mode)
        if e is not None:
            src = path.name
            break
    if e is None:
        return {"note": "baseline entry missing"}
    out: dict = {"monitors": {}, "recon_r2_mean": {}, "source": src}
    for arm in ARMS:
        for kind in ("dot", "cos"):
            mm = e["monitors"].get(f"h_{arm}_{kind}")
            if mm:
                out["monitors"][f"h_{arm}_{kind}"] = {
                    "point": mm["point"],
                    "lo": mm["lo"],
                    "hi": mm["hi"],
                }
        rc = e.get("recon_heldout", {}).get(arm)
        if rc:
            out["recon_r2_mean"][arm] = rc["r2_mean"]
    return out


# ── figure ────────────────────────────────────────────────────────────────────


def make_figure(res: dict, fig_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette(len(ARMS))
    groups = [*SUMMARIES, "v_x mean (r7 baseline)"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout="tight")
    for col, trait in enumerate(C.TRAITS):
        for row, mode in enumerate(AH.MODES):
            ax = axes[row][col]
            entry = res["summaries_headline"][trait][mode]
            width = 0.25
            xpos = np.arange(len(groups))
            for ai, arm in enumerate(ARMS):
                pts, los, his = [], [], []
                for g in groups:
                    if g == "v_x mean (r7 baseline)":
                        mm = entry["baseline_mean_summary"].get("monitors", {}).get(f"h_{arm}_cos")
                        if mm is None:
                            pts.append(np.nan)
                            los.append(0.0)
                            his.append(0.0)
                            continue
                        pt, lo, hi = mm["point"], mm["lo"], mm["hi"]
                    else:
                        # saved entries are already mode-flattened (main stores
                        # d["summaries"][s]["cos"][mode]), so no [mode] here
                        mm = entry["per_arm"][arm]["summaries"][g]["cos"]
                        pt, lo, hi = mm["point"], mm["lo"], mm["hi"]
                    pts.append(pt)
                    los.append(max(0.0, pt - lo) if np.isfinite(lo) else 0.0)
                    his.append(max(0.0, hi - pt) if np.isfinite(hi) else 0.0)
                ax.bar(
                    xpos + (ai - 1) * width,
                    pts,
                    width,
                    yerr=np.array([los, his]),
                    capsize=2,
                    color=colors[ai],
                    label=f"h — {ARM_LABELS[arm]}" if (row == 0 and col == 0) else None,
                )
            pv = entry["references"]["pv_raw"]["point"]
            orc = entry["references"]["oracle"]["point"]
            ax.axhline(pv, color="gray", ls="--", lw=1.0)
            ax.axhline(orc, color="black", ls=":", lw=1.0)
            ax.axhline(0.0, color="gray", lw=0.5)
            ax.set_xticks(xpos)
            ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=7)
            mode_lbl = "system prompting" if mode == "system" else "many-shot"
            ax.set_title(
                f"{trait} — {mode_lbl} (L{entry['layer']}; pv_raw {pv:.2f}, oracle {orc:.2f})"
            )
            if col == 0:
                ax.set_ylabel("within-condition Pearson r (cos readout)")
    fig.legend(loc="lower center", ncol=3, fontsize=8)
    figs = savefig_paper(fig, "arm_headline_summaries", dir=fig_dir)
    plt.close(fig)
    return str(figs.get("png", ""))


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 answer-summary arm-headline rerun.")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--k-draws", type=int, default=5)  # unused; Ctx compat
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / "arm_headline_summaries.json",
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
            "script": "issue779_arm_headline_summaries",
            **params,
            "frozen_layers": AH.FROZEN_LAYERS,
            "capture_dir": str(args.capture_dir),
            "summaries": list(SUMMARIES),
            "caveats": [
                "r2_last reference = span-FINAL position (incl <|im_end|>+\\n) = the "
                "v_last_turn position (token id 198), NOT the last content token",
                "r2_max reference = max of per-token r_B projections over the full span "
                "(not an element-wise max) — loose reference for v_max",
                "no stored eval reference for v_last_content / v_first",
                "Arm-A targets from the r7 g-label LMSYS rollouts (pass_b originals "
                "were never persisted), index-aligned to pass_b cx_last",
                "g omitted: judge labels are unchanged by the summary choice",
            ],
        }
    )

    ctx = AH.Ctx(args)
    res["metadata"]["equivalence_gate"] = AH.equivalence_gate(ctx.bundle, args.seed)
    C.write_json_atomic(args.out_json, res)

    sec = res.setdefault("summaries_headline", {})
    cell_cache: dict[tuple[str, int], dict] = {}
    for trait in C.TRAITS:
        tr = sec.setdefault(trait, {})
        for mode in AH.MODES:
            if mode in tr:
                logger.info("[%s %s] already checkpointed; skipping", trait, mode)
                continue
            li = AH.FROZEN_LAYERS[trait][mode]
            if (trait, li) not in cell_cache:
                cell_cache[(trait, li)] = summaries_fit_cell(ctx, args.capture_dir, trait, li)
            cell = cell_cache[(trait, li)]
            entry = {
                "layer": li,
                "n_eval_rows": cell["mat_rows"],
                "references": {
                    name: {**mm[mode], "overall_r_both_modes": mm["overall_r"]}
                    for name, mm in cell["references"].items()
                },
                "per_arm": {
                    arm: {
                        "n_train": d["n_train"],
                        "summaries": {
                            s: {
                                "dot": d["summaries"][s]["dot"][mode],
                                "cos": d["summaries"][s]["cos"][mode],
                                "delta_cos_vs_pv_raw": d["summaries"][s]["delta_cos_vs_pv_raw"][
                                    mode
                                ],
                                "gcv_lambda": d["summaries"][s]["gcv_lambda"],
                            }
                            for s in SUMMARIES
                        },
                    }
                    for arm, d in cell["per_arm"].items()
                },
                "recon_heldout": cell["recon"],
                "valid_diag": cell["valid_diag"],
                "baseline_mean_summary": baseline_entry(trait, mode),
            }
            tr[mode] = entry
            C.write_json_atomic(args.out_json, res)
            logger.info(
                "[%s %s L%d] pv_raw=%.3f r2_last=%.3f | cos A/B/C per summary: %s",
                trait,
                mode,
                li,
                entry["references"]["pv_raw"]["point"],
                entry["references"]["r2_last"]["point"],
                {
                    s: "/".join(
                        f"{entry['per_arm'][a]['summaries'][s]['cos']['point']:.3f}" for a in ARMS
                    )
                    for s in SUMMARIES
                },
            )

    # Refresh baseline_mean_summary on EXISTING (resumed) entries too — the
    # fits above are skipped on resume, but a baseline that was missing when
    # the entry was checkpointed (e.g. the pod-file split) heals here.
    for trait in C.TRAITS:
        for mode in AH.MODES:
            if mode in sec.get(trait, {}):
                sec[trait][mode]["baseline_mean_summary"] = baseline_entry(trait, mode)

    res.setdefault("figures", {})["arm_headline_summaries"] = make_figure(res, args.fig_dir)
    C.write_json_atomic(args.out_json, res)
    logger.info("Done. Wrote %s", args.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
