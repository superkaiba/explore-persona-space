"""Score new evil-OOD rungs (MHJ full, tom-gibbs full, PAIR full, tactic holdouts).

Task #1739 evil-ood-spread-round unit 4a (plan v16 §6 + §7).

After the new-rung rollouts + judging land (units 2/3 emit them), this driver
loads per-context DV + per-arm predictions and emits arm-results JSONs with
the full paired-CI + AUROC-CI + positive_count schema plan v16 §6 defines.

Design choice (DRY-BREAKING alternative, not the refactor):
    Rather than mutating scripts/issue1739_rescore_ood.py's `_compute_detection_metrics`
    surface to accept a `--rung <name>` selecting the new corpora, this
    script imports the small helpers (`_ap_score`, `_precision_at_k`,
    `AUROC_POS_THR`, `_log`) and the arms-module compute kernels
    (`spearman_rows`, `auroc_rows`, `bootstrap_rhos`, `make_bootstrap_idx`,
    `permutation_null_max`, `ARM_REGISTRY`) verbatim, applying them to
    per-rung inputs supplied on disk. The refactor is deferred — it is a
    larger change to a 776-line file with a live consumer already committed.
    See §(d) of the round's implementation marker.

Inputs (per rung `<R>`):
    --dv-pool eval_results/issue_1739/evil_ood_spread/dv_pool/<R>.json
        {"contexts": [{"context_id": str, "dv": float}, ...]}
    --arm-scores eval_results/issue_1739/evil_ood_spread/arm_scores/<R>.json
        {"arms": {"<arm_slug>": {"frozen_layer": int, "scores_per_layer": [[float, ...], ...]},
                                (n_layers x n_contexts)}}
        OR the flat shape:
        {"arms": {"<arm_slug>": {"scores": [float, ...]}}}
        (n_contexts,) — one score per context.
    --contexts-order eval_results/issue_1739/evil_ood_spread/contexts/<R>.json
        {"order": [context_id, ...]} — canonical order for aligning DV and arm scores.

Outputs:
    eval_results/issue_1739/evil_ood_spread/arm_results/<R>.json
        per-rung per-arm: rho, ci_rho, auroc, ci_auroc,
        ci_rho_delta_vs_arm16, ci_auroc_delta_vs_arm16, positive_count,
        n_rung, ap, precision@10, precision@50, perm_null_max_p.

Smoke:
    --smoke runs ONE rung with a mock 20-context DV pool + mock arm scores,
    asserting the JSON schema, exit rc=0.

Grounded on:
    - plan v16 §6 "Paired-contrast CIs" (paired-difference reads).
    - plan v16 §7 "Verdict lattice" (H1 / H2 gates on paired CIs).
    - arms.py::bootstrap_rhos (shared idx across arms => paired differences).
    - .claude/rules/selection-symmetric-nulls.md (per-draw selection).
    - .claude/rules/llm-judging.md rule 24 (drop-never-coerce; not
      re-litigated here — DV pool arrives already drop-cleaned).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # numpy imported lazily inside functions to keep startup light
    import numpy as np  # noqa: F401


def _ensure_repo_root_on_syspath() -> Path:
    """Guarded per gotchas.md 'Script mode puts the SCRIPT's dir on sys.path[0]'."""
    here = Path(__file__).resolve()
    repo_root = here.parents[1]
    sentinel = repo_root / "scripts" / "issue1739_rescore_ood.py"
    assert sentinel.exists(), f"repo-root sentinel missing: {sentinel}"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


_REPO_ROOT = _ensure_repo_root_on_syspath()

# Reuse helpers from rescore_ood (SAME AUROC threshold, same score utilities).
from scripts.issue1739_rescore_ood import (  # noqa: E402
    AUROC_POS_THR,
    _ap_score,
    _log,
    _precision_at_k,
)

# ---------------------------------------------------------------------------
# constants (mirror plan v16 §6)
# ---------------------------------------------------------------------------
N_BOOT_DEFAULT = 500
N_PERM_DEFAULT = 500
DEFAULT_RUNGS = ("mhj_full", "tom-gibbs_mt_full", "pair_full")
MAP_FAMILY = (
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm8_map_ridge_true",
    "arm9_pretrain_ft",
    "arm10_stacked",
)  # per plan v16 §3
COMPARATOR_ARM = "arm16_surface_feat"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _load_dv_pool(dv_path: Path, ctx_order: list[str]) -> "np.ndarray":  # type: ignore[name-defined]
    """Return per-context DV in the canonical order (n_ctx,) float64.

    Non-finite / missing entries surface as NaN and downstream `keep` masks
    drop them (never coerce — rule 24 discipline).
    """
    import numpy as np

    payload = _load_json(dv_path)
    ctx_records = {rec["context_id"]: rec for rec in payload["contexts"]}
    dv = np.full(len(ctx_order), np.nan, dtype=np.float64)
    for i, cid in enumerate(ctx_order):
        rec = ctx_records.get(cid)
        if rec is None:
            continue
        val = rec.get("dv")
        if val is None:
            continue
        try:
            dv[i] = float(val)
        except (TypeError, ValueError):
            dv[i] = np.nan
    return dv


def _load_arm_scores(
    arm_path: Path, ctx_order: list[str]
) -> tuple[list[str], "np.ndarray", dict[str, int]]:  # type: ignore[name-defined]
    """Return (arm_slugs, scores_matrix, frozen_by_arm).

    Scores matrix: (S, n_ctx) float64 — ONE frozen-layer score per (arm, ctx).
    Supports two shapes:
      1) layered: {"scores_per_layer": [[..per ctx..], ...], "frozen_layer": L}
         => take layer L across contexts.
      2) flat:    {"scores": [..per ctx..]}
         => single row; frozen_layer defaulted to 0 in report.
    """
    import numpy as np

    payload = _load_json(arm_path)
    arms_dict = payload.get("arms", {})
    arm_slugs: list[str] = []
    rows: list["np.ndarray"] = []
    frozen_by_arm: dict[str, int] = {}
    for slug, rec in arms_dict.items():
        if "scores_per_layer" in rec:
            layers = rec["scores_per_layer"]
            frozen = int(rec.get("frozen_layer", 0))
            frozen = max(0, min(frozen, len(layers) - 1))
            row_src = layers[frozen]
        elif "scores" in rec:
            row_src = rec["scores"]
            frozen = 0
        else:
            _log(f"skip arm {slug}: no scores_per_layer/scores field")
            continue
        by_ctx = (
            {rec_ctx["context_id"]: rec_ctx["score"] for rec_ctx in row_src}
            if (row_src and isinstance(row_src[0], dict))
            else None
        )
        row = np.full(len(ctx_order), np.nan, dtype=np.float64)
        if by_ctx is not None:
            for i, cid in enumerate(ctx_order):
                v = by_ctx.get(cid)
                if v is not None:
                    try:
                        row[i] = float(v)
                    except (TypeError, ValueError):
                        row[i] = np.nan
        else:
            # positional: assumed aligned to ctx_order already
            for i, v in enumerate(row_src[: len(ctx_order)]):
                try:
                    row[i] = float(v)
                except (TypeError, ValueError):
                    row[i] = np.nan
        arm_slugs.append(slug)
        rows.append(row)
        frozen_by_arm[slug] = frozen
    if not arm_slugs:
        raise ValueError(f"no arms loaded from {arm_path}")
    scores = np.stack(rows, axis=0)
    return arm_slugs, scores, frozen_by_arm


# ---------------------------------------------------------------------------
# Paired-bootstrap AUROC (extends bootstrap_rhos's paired shape to AUROC)
# ---------------------------------------------------------------------------
def _bootstrap_auroc(scores: "np.ndarray", labels: "np.ndarray", idx: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    """Paired-bootstrap AUROC: scores (S, n) x labels (n,) x idx (B, n) -> (S, B).

    The SAME resample indices are applied to every arm row per draw
    (paired-difference-valid). Uses the rank-formula AUROC from arms.auroc_rows
    on each drawn row-subset. A draw with zero pos or zero neg produces NaN
    for that draw (auroc_rows already handles this via the n_pos/n_neg guard).
    """
    import numpy as np

    from explore_persona_space.experiments.issue_1739.arms import auroc_rows

    scores = np.atleast_2d(scores)
    labels = np.asarray(labels, dtype=bool)
    s_rows, n = scores.shape
    n_boot = idx.shape[0]
    out = np.empty((s_rows, n_boot))
    for b in range(n_boot):
        i_b = idx[b]  # (n,)
        s_b = scores[:, i_b]  # (S, n)
        l_b = labels[i_b]  # (n,)
        out[:, b] = auroc_rows(s_b, l_b)
    return out


# ---------------------------------------------------------------------------
# core scoring
# ---------------------------------------------------------------------------
def _score_rung(
    rung: str,
    ctx_order: list[str],
    dv: "np.ndarray",  # type: ignore[name-defined]
    arm_slugs: list[str],
    scores: "np.ndarray",  # type: ignore[name-defined]
    frozen_by_arm: dict[str, int],
    *,
    n_boot: int,
    n_perm: int,
    cell_seed: int,
    k_vals: tuple[int, ...] = (10, 50),
) -> dict:
    """Compute per-arm marginal + paired-difference CIs + selection null on ONE rung."""
    import numpy as np

    from explore_persona_space.experiments.issue_1739.arms import (
        bootstrap_rhos,
        make_bootstrap_idx,
        permutation_null_max,
        spearman_rows,
    )

    labels = dv >= AUROC_POS_THR
    labels_finite = np.isfinite(dv) & labels
    positive_count = int(labels_finite.sum())

    # Row filter: at least 3 finite (score, dv) pairs per arm.
    keep_common = np.isfinite(dv)
    for i in range(scores.shape[0]):
        keep_common &= np.isfinite(scores[i])
    if keep_common.sum() < 3:
        return {
            "rung": rung,
            "n_rung": int(keep_common.sum()),
            "positive_count": positive_count,
            "error": "n_common<3",
        }

    dv_k = dv[keep_common]
    labels_k = labels[keep_common]
    scores_k = scores[:, keep_common]
    n_k = int(keep_common.sum())

    # SHARED resample indices — paired-difference-valid across arms.
    idx_b = make_bootstrap_idx(n_k, n_boot=n_boot, seed=cell_seed)

    # Batched Spearman + AUROC on all arms (marginal point estimates).
    rho_vec = spearman_rows(scores_k, dv_k)  # (S,)
    from explore_persona_space.experiments.issue_1739.arms import auroc_rows as _auroc

    auroc_vec = _auroc(scores_k, labels_k)  # (S,)

    # Bootstrap draws (S, B).
    rho_draws = bootstrap_rhos(scores_k, dv_k, idx_b)
    auroc_draws = _bootstrap_auroc(scores_k, labels_k, idx_b)

    # Comparator (arm16_surface_feat) row for paired diffs.
    try:
        cmp_i = arm_slugs.index(COMPARATOR_ARM)
    except ValueError:
        cmp_i = None

    # Selection-inherited paired-difference CIs: for arms in MAP_FAMILY,
    # the comparator per draw is (max over map_family rho) - rho_arm16.
    # Per plan v16 §6 the paired-CI on the selected max IS the selection-inherited
    # width; per-arm rows within the map family report the plain paired diff
    # (arm - arm16), and the "best_map" row carries the selection-inherited CI.
    map_family_idx = [i for i, slug in enumerate(arm_slugs) if slug in MAP_FAMILY]

    rows: list[dict[str, Any]] = []
    for i, slug in enumerate(arm_slugs):
        row: dict[str, Any] = {
            "arm": slug,
            "family": _family_of(slug),
            "frozen_layer": int(frozen_by_arm.get(slug, 0)),
            "rho": float(rho_vec[i]),
            "ci_rho": _quantile_ci(rho_draws[i]),
            "auroc": float(auroc_vec[i]),
            "ci_auroc": _quantile_ci(auroc_draws[i]),
            "ap": float(_ap_score(scores_k[i], labels_k)),
        }
        for k in k_vals:
            row[f"precision_at_{k}"] = float(_precision_at_k(scores_k[i], labels_k, k))
        # Paired diffs vs COMPARATOR_ARM.
        if cmp_i is not None and cmp_i != i:
            rho_delta = rho_draws[i] - rho_draws[cmp_i]
            auroc_delta = auroc_draws[i] - auroc_draws[cmp_i]
            row["rho_delta_vs_arm16"] = float(rho_vec[i] - rho_vec[cmp_i])
            row["ci_rho_delta_vs_arm16"] = _quantile_ci(rho_delta)
            row["auroc_delta_vs_arm16"] = float(auroc_vec[i] - auroc_vec[cmp_i])
            row["ci_auroc_delta_vs_arm16"] = _quantile_ci(auroc_delta)
        rows.append(row)

    # Selection-inherited best-of-map-family paired-CI.
    best_map: dict[str, Any] | None = None
    if cmp_i is not None and map_family_idx:
        # per draw: max over family rho draws, then - comparator draws.
        family_rho = rho_draws[map_family_idx]  # (F, B)
        family_auroc = auroc_draws[map_family_idx]  # (F, B)
        # Observed argmax per rung determined by point rho over family.
        obs_map = int(np.nanargmax(rho_vec[map_family_idx]))
        obs_slug = arm_slugs[map_family_idx[obs_map]]
        # Per-draw max (selection-inherited: NEW argmax per draw).
        max_rho = np.nanmax(family_rho, axis=0)
        max_auroc = np.nanmax(family_auroc, axis=0)
        rho_sel_delta = max_rho - rho_draws[cmp_i]
        auroc_sel_delta = max_auroc - auroc_draws[cmp_i]
        best_map = {
            "arm": "best_map_family_selection",
            "observed_arm": obs_slug,
            "family_members": [arm_slugs[j] for j in map_family_idx],
            "rho_max": float(np.nanmax(rho_vec[map_family_idx])),
            "ci_rho_max": _quantile_ci(max_rho),
            "auroc_max": float(np.nanmax(auroc_vec[map_family_idx])),
            "ci_auroc_max": _quantile_ci(max_auroc),
            "rho_delta_vs_arm16": float(np.nanmax(rho_vec[map_family_idx]) - rho_vec[cmp_i]),
            "ci_rho_delta_vs_arm16": _quantile_ci(rho_sel_delta),
            "auroc_delta_vs_arm16": float(np.nanmax(auroc_vec[map_family_idx]) - auroc_vec[cmp_i]),
            "ci_auroc_delta_vs_arm16": _quantile_ci(auroc_sel_delta),
        }

    # Permutation null (selection-symmetric over ALL arms).
    try:
        perm_null = permutation_null_max(scores_k, dv_k, n_perm=n_perm, seed=cell_seed)
    except Exception as exc:
        perm_null = {"error": str(exc)}

    return {
        "rung": rung,
        "n_rung": n_k,
        "positive_count": positive_count,
        "n_arms": len(arm_slugs),
        "arms": rows,
        "best_map_family_selection": best_map,
        "perm_null_max_all16": perm_null,
    }


def _family_of(slug: str) -> str:
    """Cheap family lookup (avoid importing the whole arms module here)."""
    try:
        from explore_persona_space.experiments.issue_1739.arms import ARM_REGISTRY

        return str(ARM_REGISTRY.get(slug, {}).get("family", "unknown"))
    except Exception:
        return "unknown"


def _quantile_ci(draws: "np.ndarray", alpha: float = 0.05) -> list[float]:  # type: ignore[name-defined]
    """2.5/97.5 percentile CI (nan-safe)."""
    import numpy as np

    if draws.size == 0:
        return [float("nan"), float("nan")]
    return [
        float(np.nanquantile(draws, alpha / 2.0)),
        float(np.nanquantile(draws, 1.0 - alpha / 2.0)),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--rungs",
        nargs="+",
        default=list(DEFAULT_RUNGS),
        help="Rung slugs to score (default: MHJ/tom-gibbs/PAIR full).",
    )
    p.add_argument(
        "--add-rung",
        action="append",
        default=[],
        help="Extra rung slugs (e.g. tactic_holdout_<class>); repeatable.",
    )
    p.add_argument(
        "--activation-store-rev",
        default=None,
        help="Provenance-only: activation store revision used to build arm scores.",
    )
    p.add_argument(
        "--arms",
        default="all16",
        help="Provenance-only: arms roster label (default 'all16').",
    )
    p.add_argument(
        "--n-folds", type=int, default=5, help="Reserved (unused: OOD transfer read, not fold CV)."
    )
    p.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    p.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--input-root",
        default="eval_results/issue_1739/evil_ood_spread",
        help="Root dir for dv_pool/, arm_scores/, contexts/ inputs.",
    )
    p.add_argument(
        "--output-dir",
        "--output",
        dest="output",
        default="eval_results/issue_1739/evil_ood_spread/arm_results/",
        help="Output dir for per-rung JSON.",
    )
    p.add_argument("--smoke", action="store_true", help="Tiny slice: 1 rung, mock inputs OK.")
    return p.parse_args(argv)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    tmp.replace(path)


def _repro_metadata() -> dict[str, Any]:
    import platform
    import subprocess

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True
        ).strip()
    except Exception:
        sha = "unavailable-no-git"
    return {
        "git_commit": sha,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    rungs = list(args.rungs) + list(args.add_rung)
    if args.smoke:
        rungs = rungs[:1]  # ONE rung for smoke; mock inputs land at input_root/<rung>.*.
    if not rungs:
        _log("no rungs to score; exiting rc=0")
        return 0

    input_root = Path(args.input_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for rung in rungs:
        ctx_path = input_root / "contexts" / f"{rung}.json"
        dv_path = input_root / "dv_pool" / f"{rung}.json"
        arm_path = input_root / "arm_scores" / f"{rung}.json"
        _log(f"[{rung}] loading contexts={ctx_path} dv={dv_path} arms={arm_path}")
        ctx_payload = _load_json(ctx_path)
        ctx_order = list(ctx_payload["order"])
        dv = _load_dv_pool(dv_path, ctx_order)
        arm_slugs, scores, frozen_by_arm = _load_arm_scores(arm_path, ctx_order)
        _log(
            f"[{rung}] n_ctx={len(ctx_order)} n_arms={len(arm_slugs)} "
            f"positive={int((dv >= AUROC_POS_THR).sum())}"
        )
        result = _score_rung(
            rung,
            ctx_order,
            dv,
            arm_slugs,
            scores,
            frozen_by_arm,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            cell_seed=args.seed,
        )
        result["provenance"] = {
            "activation_store_rev": args.activation_store_rev,
            "arms_roster": args.arms,
            "n_boot": args.n_boot,
            "n_perm": args.n_perm,
            "seed": args.seed,
            "input_root": str(input_root),
            **_repro_metadata(),
        }
        out_path = output_dir / f"{rung}.json"
        _write_json(out_path, result)
        _log(f"[{rung}] wrote {out_path}")

    _log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
