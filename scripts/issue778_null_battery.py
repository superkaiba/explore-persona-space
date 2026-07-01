#!/usr/bin/env python
"""Issue #778 — CPU null-battery driver (runs off-pod on the VM, plan v2 §9).

Wires ``explore_persona_space.analysis.null_battery`` to the cached artifacts from
Phases 1-3 and writes the primary deliverables:

  - ``eval_results/issue_778/{trait}_monitoring_nullbattery.json`` (overall + within)
  - ``eval_results/issue_778/{trait}_finetune_nullbattery.json``
  - ``eval_results/issue_778/{trait}_{setting}_{null_kind}_draws.npy`` (per-draw x
    per-layer |r| matrices — the analyzer's honest-band recompute inputs; MUST
    upload to the HF data repo analysis_tensors/ per Upload Policy).
  - ``eval_results/issue_778/hero_bands_{trait}_{setting}.json`` + figure-array JSONs
    (violin/box + heatmap + scatter + leave-one-family-out) for the analyzer.

Inputs (from the pod phases, staged/downloaded locally):
  - ``data/issue_778/rb/{trait}.pt`` + ``activations/{trait}_{pos,neg}.pt`` (Phase 1)
  - ``eval_results/issue_778/monitoring_{trait}.jsonl`` (Phase 2)
  - ``data/issue_778/finetune_activations/{model_tag}.pt`` +
    ``eval_results/issue_778/finetune_{trait}_{family}_{version}.json`` +
    ``finetune_base_{trait}.json`` (Phase 3)

CPU-only closed-form / sampling stats; no model calls, no GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib

from explore_persona_space.analysis import null_battery as nb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue778.nullbattery")


# ── Loaders ─────────────────────────────────────────────────────────────────────


def _load_rb(out_root: Path, trait: str) -> np.ndarray:
    import torch

    rb = torch.load(out_root / "rb" / f"{trait}.pt", weights_only=False)
    return rb.numpy().astype(np.float64)  # (28, 3584)


def _load_pools(out_root: Path, trait: str) -> tuple[np.ndarray, np.ndarray]:
    import torch

    pos = torch.load(out_root / "activations" / f"{trait}_pos.pt", weights_only=False)
    neg = torch.load(out_root / "activations" / f"{trait}_neg.pt", weights_only=False)
    return pos.numpy().astype(np.float64), neg.numpy().astype(np.float64)


def _load_monitoring(eval_root: Path, trait: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read monitoring JSONL -> (predictor_acts (n,28,3584-proxy), target, condition_ids).

    NOTE: the JSONL stores projection_per_layer (already projected onto r_B), not
    the raw activation. For the MATCHED direction the projection is what we need,
    but the nulls require re-projecting the RAW last-prompt activation onto each
    null direction. So this driver instead reconstructs the raw predictor
    activation from the per-model capture is NOT available for monitoring (Phase 2
    only stored projections). We therefore compute the matched r from the stored
    projection, and the nulls from the raw activation tensor Phase 2 ALSO caches.
    """
    rows = []
    with open(eval_root / f"monitoring_{trait}.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = [r for r in rows if r["mean_trait_score"] is not None]
    target = np.array([r["mean_trait_score"] for r in rows], dtype=np.float64)
    condition_ids = np.array([r["condition_id"] for r in rows])
    proj = np.array([r["projection_per_layer"] for r in rows], dtype=np.float64)  # (n, 28)
    return proj, target, condition_ids


def _load_monitoring_raw_acts(out_root: Path, trait: str) -> np.ndarray | None:
    """Load Phase-2 raw last-prompt activations for the null re-projection.

    Phase 2 caches the raw predictor tensor at
    ``data/issue_778/monitoring/{trait}_acts.pt`` (n_cells, 28, 3584) aligned with
    the JSONL row order (pre-drop). Returns None if absent (older run).
    """
    import torch

    p = out_root / "monitoring" / f"{trait}_acts.pt"
    if not p.exists():
        return None
    return torch.load(p, weights_only=False).numpy().astype(np.float64)


def _load_finetune(
    out_root: Path, eval_root: Path, trait: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Assemble the n=24 finetune regression: (shift_acts (24,28,3584), target (24,), tags).

    shift = mean-last-prompt(finetuned) - mean-last-prompt(base), per (cell, trait).
    """
    import torch

    base_acts = torch.load(out_root / "finetune_activations" / "base.pt", weights_only=False)
    base_vec = base_acts[trait].numpy().astype(np.float64)  # (28, 3584)

    shifts = []
    targets = []
    tags = []
    for fam in lib.FAMILIES:
        for ver in lib.VERSIONS:
            tag = f"{fam}_{ver}"
            act_path = out_root / "finetune_activations" / f"{tag}.pt"
            expr_path = eval_root / f"finetune_{trait}_{fam}_{ver}.json"
            if not act_path.exists() or not expr_path.exists():
                logger.warning("finetune cell %s missing artifacts; skipping (%s)", tag, trait)
                continue
            ft = torch.load(act_path, weights_only=False)[trait].numpy().astype(np.float64)
            with open(expr_path) as f:
                score = json.load(f).get("trait_score")
            if score is None:
                logger.warning("finetune cell %s trait_score None; skipping", tag)
                continue
            shifts.append(ft - base_vec)
            targets.append(score)
            tags.append(tag)
    if not shifts:
        raise RuntimeError(f"trait={trait}: no usable finetune cells for the n=24 regression")
    return np.stack(shifts, axis=0), np.array(targets, dtype=np.float64), tags


# ── Leave-one-family-out (finetune only) ────────────────────────────────────────


def _leave_one_family_out(
    shift_acts: np.ndarray, target: np.ndarray, tags: list[str], rb: np.ndarray, sel_layer: int
) -> dict:
    """Recompute matched-trait r dropping each family's versions in turn."""
    families = sorted({lib.split_cell_tag(t)[0] for t in tags})
    out = {}
    for fam in families:
        keep = [i for i, t in enumerate(tags) if lib.split_cell_tag(t)[0] != fam]
        if len(keep) < 3:
            out[fam] = None
            continue
        sub_acts = shift_acts[keep]
        sub_target = target[keep]
        proj = nb.project(sub_acts[:, sel_layer, :], rb[sel_layer])
        out[fam] = nb._pearson(proj, sub_target)
    return out


# ── Per-(trait, setting) run ────────────────────────────────────────────────────


def run_finetune(
    trait: str,
    out_root: Path,
    eval_root: Path,
    other_rbs: dict[str, np.ndarray],
    *,
    n_draws: int,
    lam: float,
    pca_k: int,
    n_boot: int,
) -> dict:
    rb = _load_rb(out_root, trait)
    pos, neg = _load_pools(out_root, trait)
    shift_acts, target, tags = _load_finetune(out_root, eval_root, trait)
    result, draws = nb.compute_setting(
        trait,
        "finetune",
        predictor_acts=shift_acts,
        rb_per_layer=rb,
        target=target,
        pos_acts=pos,
        neg_acts=neg,
        other_rbs=other_rbs,
        n_draws=n_draws,
        lam=lam,
        pca_k=pca_k,
        n_boot=n_boot,
    )
    loco = _leave_one_family_out(shift_acts, target, tags, rb, result.matched_selected_layer)
    result.reproducibility = lib.repro_metadata()
    payload = result.to_json()
    payload["tags"] = tags
    payload["per_run_points"] = [
        {
            "tag": tags[i],
            "shift_proj_selected_layer": float(
                nb.project(
                    shift_acts[i : i + 1, result.matched_selected_layer, :],
                    rb[result.matched_selected_layer],
                )[0]
            ),
            "trait_score": float(target[i]),
        }
        for i in range(len(tags))
    ]
    payload["leave_one_family_out_r"] = loco
    _write_draws(eval_root, trait, "finetune", draws)
    _write_figure_arrays(eval_root, trait, "finetune", result, draws, payload)
    return payload


def run_monitoring(
    trait: str,
    out_root: Path,
    eval_root: Path,
    other_rbs: dict[str, np.ndarray],
    *,
    n_draws: int,
    lam: float,
    pca_k: int,
    n_boot: int,
) -> dict:
    """Monitoring: BOTH overall_r and within_condition_r get the full null battery.

    Requires the Phase-2 raw last-prompt activation tensor to re-project nulls.
    """
    rb = _load_rb(out_root, trait)
    pos, neg = _load_pools(out_root, trait)
    _proj_stored, target, condition_ids = _load_monitoring(eval_root, trait)
    raw_acts = _load_monitoring_raw_acts(out_root, trait)
    if raw_acts is None:
        raise RuntimeError(
            f"trait={trait}: monitoring raw activation tensor "
            f"{out_root}/monitoring/{trait}_acts.pt missing — the null re-projection "
            f"needs raw last-prompt activations, not the stored projections."
        )
    # Align raw_acts to the kept (non-dropped) rows: Phase 2 wrote JSONL in cell
    # order and the raw tensor in the same order; kept rows are those with a
    # non-None score. Re-derive the kept mask from the full JSONL.
    kept_mask = _monitoring_kept_mask(eval_root, trait)
    raw_kept = raw_acts[kept_mask]
    if raw_kept.shape[0] != target.shape[0]:
        raise RuntimeError(
            f"trait={trait}: monitoring raw acts kept {raw_kept.shape[0]} != "
            f"target {target.shape[0]} — row alignment broken"
        )

    out = {}
    for setting in ("monitoring_overall", "monitoring_within"):
        result, draws = nb.compute_setting(
            trait,
            setting,
            predictor_acts=raw_kept,
            rb_per_layer=rb,
            target=target,
            pos_acts=pos,
            neg_acts=neg,
            other_rbs=other_rbs,
            condition_ids=condition_ids,
            n_draws=n_draws,
            lam=lam,
            pca_k=pca_k,
            n_boot=n_boot,
        )
        result.reproducibility = lib.repro_metadata()
        payload = result.to_json()
        _write_draws(eval_root, trait, setting, draws)
        _write_figure_arrays(eval_root, trait, setting, result, draws, payload)
        out[setting] = payload
    return out


def _monitoring_kept_mask(eval_root: Path, trait: str) -> np.ndarray:
    rows = []
    with open(eval_root / f"monitoring_{trait}.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return np.array([r["mean_trait_score"] is not None for r in rows])


# ── Persistence ─────────────────────────────────────────────────────────────────


def _write_draws(eval_root: Path, trait: str, setting: str, draws: dict[str, np.ndarray]) -> None:
    for kind, mat in draws.items():
        path = eval_root / f"{trait}_{setting}_{kind}_draws.npy"
        np.save(path, mat.astype(np.float32))


def _write_figure_arrays(
    eval_root: Path, trait: str, setting: str, result, draws: dict[str, np.ndarray], payload: dict
) -> None:
    """Emit the raw numeric arrays the analyzer's paper-plots skill consumes."""
    # Hero: observed matched r + CI overlaid on the 4 nulls' max|r| distributions.
    hero = {
        "trait": trait,
        "setting": setting,
        "observed_matched_max_abs": result.matched_max_abs,
        "observed_matched_r": result.matched_r,
        "matched_ci95": list(result.matched_r_bootstrap_ci_95),
        "nulls": {
            k: {
                "draws_max_abs": v.draws_max_abs,
                "p2_5": v.r_p2_5,
                "p97_5": v.r_p97_5,
                "empirical_p": v.empirical_p_one_sided,
            }
            for k, v in result.nulls.items()
        },
    }
    with open(eval_root / f"hero_bands_{trait}_{setting}.json", "w") as f:
        json.dump(hero, f, indent=2)

    # Per-layer heatmap: draws already carry per-layer |r|; store the mean per-layer
    # per null (exploratory). An all-NaN layer column (a degenerate draw at that
    # layer) is stored as NaN via a warning-safe reduce, not a raised RuntimeWarning.
    def _safe_col_mean(mat: np.ndarray) -> list:
        with np.errstate(invalid="ignore"):
            cols = [
                float(np.nanmean(mat[:, j])) if not np.all(np.isnan(mat[:, j])) else float("nan")
                for j in range(mat.shape[1])
            ]
        return cols

    heatmap = {
        "trait": trait,
        "setting": setting,
        "nulls_per_layer_mean_abs_r": {k: _safe_col_mean(v) for k, v in draws.items()},
    }
    with open(eval_root / f"per_layer_heatmap_{trait}_{setting}.json", "w") as f:
        json.dump(heatmap, f, indent=2)

    # Scatter (finetune only): the per-run regression points.
    if setting == "finetune" and "per_run_points" in payload:
        scatter = {"trait": trait, "setting": setting, "points": payload["per_run_points"]}
        with open(eval_root / f"scatter_{trait}_{setting}.json", "w") as f:
            json.dump(scatter, f, indent=2)
        if "leave_one_family_out_r" in payload:
            with open(eval_root / f"leave_one_family_out_{trait}.json", "w") as f:
                json.dump(
                    {
                        "trait": trait,
                        "leave_one_family_out_r": payload["leave_one_family_out_r"],
                        "full_r": result.matched_r,
                    },
                    f,
                    indent=2,
                )


# ── Main ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #778 CPU null battery.")
    parser.add_argument("--out-root", default="data/issue_778")
    parser.add_argument("--eval-results-root", default="eval_results/issue_778")
    parser.add_argument("--traits", nargs="+", default=list(lib.TRAITS))
    parser.add_argument("--settings", nargs="+", default=["monitoring", "finetune"])
    parser.add_argument("--n-draws", type=int, default=nb.DEFAULT_N_DRAWS)
    parser.add_argument("--lam", type=float, default=nb.PRIMARY_LAMBDA)
    parser.add_argument("--pca-k", type=int, default=nb.DEFAULT_PCA_K)
    parser.add_argument("--n-boot", type=int, default=nb.DEFAULT_BOOTSTRAP)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    eval_root = Path(args.eval_results_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    traits = args.traits

    # Load all r_B up front for the cross-trait null.
    rbs = {t: _load_rb(out_root, t) for t in traits}

    lib.log_phase("null_battery", f"start traits={traits} settings={args.settings}")
    all_pvals: list[float] = []
    pval_index: list[tuple[str, str, str]] = []  # (trait, setting, null_kind)
    summary: dict = {}

    for trait in traits:
        other_rbs = {ot: rbs[ot] for ot in traits if ot != trait}
        summary[trait] = {}
        if "finetune" in args.settings:
            ft = run_finetune(
                trait,
                out_root,
                eval_root,
                other_rbs,
                n_draws=args.n_draws,
                lam=args.lam,
                pca_k=args.pca_k,
                n_boot=args.n_boot,
            )
            summary[trait]["finetune"] = ft
            for kind, nr in ft["nulls"].items():
                all_pvals.append(nr["empirical_p_one_sided"])
                pval_index.append((trait, "finetune", kind))
            with open(eval_root / f"{trait}_finetune_nullbattery.json", "w") as f:
                json.dump(ft, f, indent=2)
        if "monitoring" in args.settings:
            mon = run_monitoring(
                trait,
                out_root,
                eval_root,
                other_rbs,
                n_draws=args.n_draws,
                lam=args.lam,
                pca_k=args.pca_k,
                n_boot=args.n_boot,
            )
            summary[trait]["monitoring"] = mon
            for setting in ("monitoring_overall", "monitoring_within"):
                for kind, nr in mon[setting]["nulls"].items():
                    all_pvals.append(nr["empirical_p_one_sided"])
                    pval_index.append((trait, setting, kind))
            with open(eval_root / f"{trait}_monitoring_nullbattery.json", "w") as f:
                json.dump(mon, f, indent=2)

    # BH-adjust across ALL tests, thread back into each file.
    bh = nb.benjamini_hochberg(all_pvals)
    bh_map = {idx: bh[i] for i, idx in enumerate(pval_index)}
    _thread_bh(eval_root, traits, args.settings, bh_map)

    lib.log_phase("null_battery", "done", n_tests=len(all_pvals))
    print(json.dumps({"phase": "null_battery", "n_tests": len(all_pvals)}, indent=2))


def _thread_bh(eval_root: Path, traits, settings, bh_map: dict) -> None:
    """Write the BH-adjusted p per (trait,setting,null) back into the deliverables."""
    for trait in traits:
        if "finetune" in settings:
            p = eval_root / f"{trait}_finetune_nullbattery.json"
            with open(p) as f:
                data = json.load(f)
            for kind in data["nulls"]:
                key = (trait, "finetune", kind)
                data["nulls"][kind]["bh_adjusted_empirical_p"] = bh_map.get(key)
            with open(p, "w") as f:
                json.dump(data, f, indent=2)
        if "monitoring" in settings:
            p = eval_root / f"{trait}_monitoring_nullbattery.json"
            with open(p) as f:
                data = json.load(f)
            for setting in ("monitoring_overall", "monitoring_within"):
                for kind in data[setting]["nulls"]:
                    key = (trait, setting, kind)
                    data[setting]["nulls"][kind]["bh_adjusted_empirical_p"] = bh_map.get(key)
            with open(p, "w") as f:
                json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
