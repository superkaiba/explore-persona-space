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

# Stable HF DATA-repo prefix (relative to ``issue{issue}_{slug}/``) the POD phase
# uploads the primary-deliverable monitoring JSONLs to
# (issue778_upload_corrected.EVAL_JSONL_SUBPREFIX). Kept in lockstep with the
# producer; this consumer downloads absent JSONLs from here after pod teardown.
EVAL_JSONL_SUBPREFIX = "followup_corrected/eval_jsonl"


# ── Off-pod input resolution (JSONL fetch from HF) ────────────────────────────────


def _ensure_monitoring_jsonls_local(
    eval_root: Path,
    traits: list[str],
    input_tags: list[str],
    *,
    issue: int,
    slug: str,
    fetch_from_hf: bool,
) -> None:
    """Ensure every ``{input_tag}_{trait}.jsonl`` the run needs is present locally.

    The pod-side dispatch writes these primary deliverables to the pod's
    ``eval_results/issue_778/`` and (reconciler round-1 BLOCKER fix) also uploads
    them to ``issue{issue}_{slug}/{EVAL_JSONL_SUBPREFIX}/`` on the HF DATA repo. This
    off-pod driver runs on the VM AFTER the pod is released, so any JSONL absent
    locally is downloaded from that HF prefix into ``eval_root`` before the null
    battery opens it. Fail-loud if a required JSONL is neither local nor on HF.
    """
    eval_root.mkdir(parents=True, exist_ok=True)
    required = [(tag, t) for tag in input_tags for t in traits]
    missing_local = [
        (tag, t) for tag, t in required if not (eval_root / f"{tag}_{t}.jsonl").exists()
    ]
    if not missing_local:
        return
    if not fetch_from_hf:
        names = [f"{tag}_{t}.jsonl" for tag, t in missing_local]
        raise RuntimeError(
            f"monitoring JSONL(s) absent locally and --no-hf-fetch set: {names} "
            f"(expected under {eval_root} or HF prefix {slug}/{EVAL_JSONL_SUBPREFIX})"
        )
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.env import load_dotenv
    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO

    load_dotenv()
    exp_name = f"issue{issue}_{slug}"
    prefix = f"{exp_name}/{EVAL_JSONL_SUBPREFIX}"
    for tag, trait in missing_local:
        name = f"{tag}_{trait}.jsonl"
        logger.info("fetching primary-deliverable JSONL %s from HF %s/%s", name, prefix, name)
        local = hf_hub_download(
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            filename=f"{prefix}/{name}",
            revision="main",
        )
        import shutil

        shutil.copyfile(local, eval_root / name)
    # Hard assert: every required JSONL now resolves locally.
    still_missing = [
        f"{tag}_{t}.jsonl" for tag, t in required if not (eval_root / f"{tag}_{t}.jsonl").exists()
    ]
    if still_missing:
        raise RuntimeError(
            f"monitoring JSONL(s) still absent after HF fetch: {still_missing} "
            f"(HF prefix {DEFAULT_DATASET_REPO}/{prefix})"
        )


def _ensure_monitoring_acts_local(
    out_root: Path,
    traits: list[str],
    input_tags: list[str],
    *,
    issue: int,
    slug: str,
    fetch_from_hf: bool,
) -> None:
    """Ensure every ``{input_tag}/{trait}_acts.pt`` the null re-projection needs is local.

    ``run_monitoring`` RAISES on an absent raw last-prompt activation tensor (the nulls
    re-project the RAW activation onto each null direction — the stored projections are
    not enough). The pod-side dispatch writes these to ``data/issue_778/{tag}/{trait}_acts.pt``
    AND uploads them to the HF DATA repo under
    ``issue{issue}_{slug}/analysis_tensors/{tag}/{trait}_acts.pt`` (issue778_upload_corrected
    .upload_pod_phase: ``at_prefix = {exp_name}/analysis_tensors`` + rel-to-out_root path).
    This off-pod driver runs on the VM AFTER teardown, so any acts tensor absent locally is
    downloaded from that HF prefix into ``out_root`` before the battery opens it. Fail-loud
    if a required tensor is neither local nor on HF (BLOCKER offpod-monitoring-acts-never-
    fetched, r2). Respects ``fetch_from_hf`` (``--no-hf-fetch``).
    """
    required = [(tag, t) for tag in input_tags for t in traits]
    missing_local = [
        (tag, t) for tag, t in required if not (out_root / tag / f"{t}_acts.pt").exists()
    ]
    if not missing_local:
        return
    exp_name = f"issue{issue}_{slug}"
    # Mirror the producer layout: HF path preserves ``{tag}/{trait}_acts.pt`` under
    # ``{exp_name}/analysis_tensors/`` (upload_pod_phase uploads pt.relative_to(out_root)).
    at_prefix = f"{exp_name}/analysis_tensors"
    if not fetch_from_hf:
        names = [f"{tag}/{t}_acts.pt" for tag, t in missing_local]
        raise RuntimeError(
            f"monitoring raw-acts tensor(s) absent locally and --no-hf-fetch set: {names} "
            f"(expected under {out_root} or HF prefix {at_prefix}/{{tag}}/{{trait}}_acts.pt)"
        )
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.env import load_dotenv
    from explore_persona_space.orchestrate.hub import DEFAULT_DATASET_REPO

    load_dotenv()
    for tag, trait in missing_local:
        rel = f"{tag}/{trait}_acts.pt"
        logger.info("fetching raw-acts tensor %s from HF %s/%s", rel, at_prefix, rel)
        local = hf_hub_download(
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            filename=f"{at_prefix}/{rel}",
            revision="main",
        )
        import shutil

        dest = out_root / tag / f"{trait}_acts.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, dest)
    # Hard assert: every required acts tensor now resolves locally.
    still_missing = [
        f"{tag}/{t}_acts.pt"
        for tag, t in required
        if not (out_root / tag / f"{t}_acts.pt").exists()
    ]
    if still_missing:
        raise RuntimeError(
            f"monitoring raw-acts tensor(s) still absent after HF fetch: {still_missing} "
            f"(HF prefix {DEFAULT_DATASET_REPO}/{at_prefix})"
        )


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


def _load_monitoring(
    eval_root: Path, trait: str, input_tag: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read monitoring JSONL -> (predictor_acts (n,28,3584-proxy), target, condition_ids).

    ``input_tag`` selects the input leg: ``monitoring`` (parent extraction-prompt
    run), ``monitoring_corrected`` (Leg A), ``monitoring_manyshot`` (Leg B). The
    JSONL path is ``{input_tag}_{trait}.jsonl``.

    NOTE: the JSONL stores projection_per_layer (already projected onto r_B), not
    the raw activation. For the MATCHED direction the projection is what we need,
    but the nulls require re-projecting the RAW last-prompt activation onto each
    null direction. So this driver instead reconstructs the raw predictor
    activation from the per-model capture is NOT available for monitoring (Phase 2
    only stored projections). We therefore compute the matched r from the stored
    projection, and the nulls from the raw activation tensor Phase 2 ALSO caches.
    """
    rows = []
    with open(eval_root / f"{input_tag}_{trait}.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = [r for r in rows if r["mean_trait_score"] is not None]
    target = np.array([r["mean_trait_score"] for r in rows], dtype=np.float64)
    condition_ids = np.array([r["condition_id"] for r in rows])
    proj = np.array([r["projection_per_layer"] for r in rows], dtype=np.float64)  # (n, 28)
    return proj, target, condition_ids


def _load_monitoring_raw_acts(out_root: Path, trait: str, input_tag: str) -> np.ndarray | None:
    """Load the raw last-prompt activations for the null re-projection.

    The monitoring/Leg-A/Leg-B drivers cache the raw predictor tensor at
    ``data/issue_778/{input_tag}/{trait}_acts.pt`` (n_cells, 28, 3584) aligned with
    the JSONL row order (pre-drop). Returns None if absent (older run).
    """
    import torch

    p = out_root / input_tag / f"{trait}_acts.pt"
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


# Null-class split (reconciled statistics fix, plan v4 §5/§6/§11):
#   STOCHASTIC nulls (>=200 draws) carry the BH-adjusted one-sided empirical
#     p < 0.025 inferential gate.
#   FIXED-DIRECTION nulls (2 crosstrait dirs, 5 pca_topk dirs) carry a fixed-
#     control EXCEEDANCE check (observed matched max|r| > max over the null's
#     per-direction max|r|); their +1 empirical p bottoms out at 1/3 and 1/6, so
#     a p<0.025 gate is unsatisfiable by construction — they are EXCLUDED from
#     the BH family (24 stochastic tests when both legs run, NOT 48) and carry
#     the exceedance verdict + descriptive p only.
STOCHASTIC_NULL_KINDS: tuple[str, ...] = ("perm", "randnorm")
FIXED_NULL_KINDS: tuple[str, ...] = ("crosstrait", "pca_topk")


def _annotate_exceedance(payload: dict) -> None:
    """Tag each FIXED-direction null in ``payload['nulls']`` with an exceedance bool.

    ``exceedance`` = observed matched max-over-layers |r| STRICTLY exceeds the max
    over that null's per-direction max-over-layers |r| (``draws_max_abs``). NaN
    directions are ignored; an all-NaN fixed null yields ``exceedance = None``
    (undecidable). The stochastic nulls carry no exceedance key (they use the
    BH-adjusted empirical-p gate instead).
    """
    observed = payload.get("matched_max_abs")
    for kind, nr in payload.get("nulls", {}).items():
        if kind not in FIXED_NULL_KINDS:
            continue
        dm = np.asarray(nr.get("draws_max_abs", []), dtype=np.float64)
        valid = dm[~np.isnan(dm)]
        if (
            observed is None
            or (isinstance(observed, float) and np.isnan(observed))
            or valid.size == 0
        ):
            nr["exceedance"] = None
        else:
            nr["exceedance"] = bool(float(observed) > float(valid.max()))


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
    _annotate_exceedance(payload)
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
    input_tag: str,
    n_draws: int,
    lam: float,
    pca_k: int,
    n_boot: int,
) -> dict:
    """Monitoring: BOTH overall_r and within_condition_r get the full null battery.

    Requires the raw last-prompt activation tensor to re-project nulls. ``input_tag``
    selects the leg: ``monitoring`` (parent), ``monitoring_corrected`` (Leg A),
    ``monitoring_manyshot`` (Leg B). The ``nb.compute_setting`` setting names stay
    ``monitoring_overall``/``monitoring_within`` (only the file names carry the tag)
    so the within/overall correlation semantics are identical across legs; for Leg B
    the condition_id is the shot-count, so ``monitoring_within`` = within-shot-count.
    """
    rb = _load_rb(out_root, trait)
    pos, neg = _load_pools(out_root, trait)
    _proj_stored, target, condition_ids = _load_monitoring(eval_root, trait, input_tag)
    raw_acts = _load_monitoring_raw_acts(out_root, trait, input_tag)
    if raw_acts is None:
        raise RuntimeError(
            f"trait={trait}: monitoring raw activation tensor "
            f"{out_root}/{input_tag}/{trait}_acts.pt missing — the null re-projection "
            f"needs raw last-prompt activations, not the stored projections."
        )
    # Align raw_acts to the kept (non-dropped) rows: the driver wrote JSONL in cell
    # order and the raw tensor in the same order; kept rows are those with a
    # non-None score. Re-derive the kept mask from the full JSONL.
    kept_mask = _monitoring_kept_mask(eval_root, trait, input_tag)
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
        _annotate_exceedance(payload)
        # File-name portion carries the input_tag + the overall/within suffix
        # (draws.npy: {trait}_{input_tag}_{overall,within}_{null}_draws.npy).
        corr = "overall" if setting == "monitoring_overall" else "within"
        file_setting = f"{input_tag}_{corr}"
        _write_draws(eval_root, trait, file_setting, draws)
        _write_figure_arrays(eval_root, trait, file_setting, result, draws, payload)
        out[setting] = payload
    return out


def _monitoring_kept_mask(eval_root: Path, trait: str, input_tag: str) -> np.ndarray:
    rows = []
    with open(eval_root / f"{input_tag}_{trait}.jsonl") as f:
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
    parser.add_argument(
        "--input-tag",
        nargs="+",
        default=["monitoring"],
        help="monitoring input leg(s) to run (default: monitoring). Pass BOTH legs of "
        "the amendment in ONE invocation — "
        "'--input-tag monitoring_corrected monitoring_manyshot' — so the BH family "
        "pools across both legs (24 STOCHASTIC tests: 2 legs x 3 traits x 2 corr x "
        "2 stochastic nulls perm/randnorm; the fixed nulls crosstrait/pca_topk are "
        "EXCLUDED and carry the exceedance verdict). Consumes {input_tag}_{trait}.jsonl "
        "+ data/issue_778/{input_tag}/{trait}_acts.pt; writes "
        "{trait}_{input_tag}_nullbattery.json.",
    )
    # Raised from parent's 200 -> 1000 (statistics-critic BH-floor concern: at 200
    # draws the one-sided empirical-p floor 1/201 ~ 0.005 can never clear the
    # 0.025/24-test BH threshold, so the FALSIFYING outcome would be illegible).
    parser.add_argument("--n-draws", type=int, default=1000)
    parser.add_argument("--lam", type=float, default=nb.PRIMARY_LAMBDA)
    parser.add_argument("--pca-k", type=int, default=nb.DEFAULT_PCA_K)
    parser.add_argument("--n-boot", type=int, default=nb.DEFAULT_BOOTSTRAP)
    parser.add_argument("--issue", type=int, default=778, help="for the HF JSONL fetch prefix")
    parser.add_argument("--slug", default="persona_vectors", help="for the HF JSONL fetch prefix")
    parser.add_argument(
        "--no-hf-fetch",
        action="store_true",
        help="do NOT download absent monitoring JSONLs from HF (fail-loud on a local miss)",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    eval_root = Path(args.eval_results_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    traits = args.traits

    # Off-pod input resolution: the pod uploaded the primary-deliverable monitoring
    # JSONLs to HF before teardown; download any absent locally BEFORE the battery
    # opens them (reconciler round-1 BLOCKER jsonl-deliverables-never-promoted).
    if "monitoring" in args.settings:
        _ensure_monitoring_jsonls_local(
            eval_root,
            traits,
            list(args.input_tag),
            issue=args.issue,
            slug=args.slug,
            fetch_from_hf=not args.no_hf_fetch,
        )
        # The nulls re-project the RAW last-prompt activation; run_monitoring RAISES on
        # an absent {input_tag}/{trait}_acts.pt. The pod uploaded these to HF
        # analysis_tensors/ before teardown; download any absent locally BEFORE the
        # battery opens them (BLOCKER offpod-monitoring-acts-never-fetched, r2).
        _ensure_monitoring_acts_local(
            out_root,
            traits,
            list(args.input_tag),
            issue=args.issue,
            slug=args.slug,
            fetch_from_hf=not args.no_hf_fetch,
        )

    # Load all r_B up front for the cross-trait null.
    rbs = {t: _load_rb(out_root, t) for t in traits}

    lib.log_phase(
        "null_battery",
        f"start traits={traits} settings={args.settings} input_tags={args.input_tag} "
        f"n_draws={args.n_draws}",
    )
    all_pvals: list[float] = []
    # pval index carries the input_tag so the two legs' files are disambiguated.
    # (input_tag, trait, setting, null_kind). finetune uses input_tag="".
    pval_index: list[tuple[str, str, str, str]] = []
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
                if kind not in STOCHASTIC_NULL_KINDS:
                    continue  # fixed nulls carry the exceedance verdict, not BH
                all_pvals.append(nr["empirical_p_one_sided"])
                pval_index.append(("", trait, "finetune", kind))
            with open(eval_root / f"{trait}_finetune_nullbattery.json", "w") as f:
                json.dump(ft, f, indent=2)
        if "monitoring" in args.settings:
            summary[trait]["monitoring"] = {}
            for input_tag in args.input_tag:
                mon = run_monitoring(
                    trait,
                    out_root,
                    eval_root,
                    other_rbs,
                    input_tag=input_tag,
                    n_draws=args.n_draws,
                    lam=args.lam,
                    pca_k=args.pca_k,
                    n_boot=args.n_boot,
                )
                summary[trait]["monitoring"][input_tag] = mon
                for setting in ("monitoring_overall", "monitoring_within"):
                    for kind, nr in mon[setting]["nulls"].items():
                        if kind not in STOCHASTIC_NULL_KINDS:
                            continue  # exclude fixed nulls from the BH family
                        all_pvals.append(nr["empirical_p_one_sided"])
                        pval_index.append((input_tag, trait, setting, kind))
                with open(eval_root / f"{trait}_{input_tag}_nullbattery.json", "w") as f:
                    json.dump(mon, f, indent=2)

    # BH-adjust across the STOCHASTIC-null tests pooled (the 24-test family when
    # both legs run — perm/randnorm only; fixed nulls excluded, plan v4 §5/§6/§11),
    # AND compute a per-null-family BH (within each null_kind) for legibility.
    bh = nb.benjamini_hochberg(all_pvals)
    bh_map = {idx: bh[i] for i, idx in enumerate(pval_index)}
    bh_within_family_map = _bh_within_null_family(all_pvals, pval_index)
    _thread_bh(eval_root, traits, args.settings, args.input_tag, bh_map, bh_within_family_map)

    lib.log_phase("null_battery", "done", n_tests=len(all_pvals))
    print(json.dumps({"phase": "null_battery", "n_tests": len(all_pvals)}, indent=2))


def _bh_within_null_family(
    all_pvals: list[float], pval_index: list[tuple[str, str, str, str]]
) -> dict[tuple[str, str, str, str], float]:
    """Per-null-family BH: BH computed WITHIN each null_kind across all its tests.

    The pooled `bh` (24-test STOCHASTIC family) mixes the 2 stochastic null kinds;
    this legibility field (statistics-critic) adjusts within each null_kind
    separately (12 tests each when both legs run: 2 legs x 3 traits x 2 corr). Only
    perm/randnorm reach here (fixed nulls are excluded before pooling). Returns
    {idx: bh_within}.
    """
    families: dict[str, list[int]] = {}
    for i, idx in enumerate(pval_index):
        null_kind = idx[3]
        families.setdefault(null_kind, []).append(i)
    out: dict[tuple[str, str, str, str], float] = {}
    for _kind, positions in families.items():
        sub = nb.benjamini_hochberg([all_pvals[i] for i in positions])
        for j, i in enumerate(positions):
            out[pval_index[i]] = sub[j]
    return out


def _thread_bh(
    eval_root: Path,
    traits,
    settings,
    input_tags,
    bh_map: dict,
    bh_within_family_map: dict,
) -> None:
    """Write BH-adjusted p (24-pooled STOCHASTIC) + per-null-family BH per deliverable.

    Only stochastic nulls (perm/randnorm) are in the maps; the fixed nulls
    (crosstrait/pca_topk) get ``bh_adjusted_empirical_p: null`` via ``.get`` misses
    and carry the ``exceedance`` verdict (annotated in run_finetune/run_monitoring).
    """
    for trait in traits:
        if "finetune" in settings:
            p = eval_root / f"{trait}_finetune_nullbattery.json"
            with open(p) as f:
                data = json.load(f)
            for kind in data["nulls"]:
                key = ("", trait, "finetune", kind)
                data["nulls"][kind]["bh_adjusted_empirical_p"] = bh_map.get(key)
                data["nulls"][kind]["bh_adjusted_within_null_family"] = bh_within_family_map.get(
                    key
                )
            with open(p, "w") as f:
                json.dump(data, f, indent=2)
        if "monitoring" in settings:
            for input_tag in input_tags:
                p = eval_root / f"{trait}_{input_tag}_nullbattery.json"
                with open(p) as f:
                    data = json.load(f)
                for setting in ("monitoring_overall", "monitoring_within"):
                    for kind in data[setting]["nulls"]:
                        key = (input_tag, trait, setting, kind)
                        data[setting]["nulls"][kind]["bh_adjusted_empirical_p"] = bh_map.get(key)
                        data[setting]["nulls"][kind]["bh_adjusted_within_null_family"] = (
                            bh_within_family_map.get(key)
                        )
                with open(p, "w") as f:
                    json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
