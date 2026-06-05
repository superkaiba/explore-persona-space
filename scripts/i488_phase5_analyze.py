# ruff: noqa: RUF002, RUF003
"""Issue #488 Phase 5 — analysis + headline statistics.

Plan v2 §6. Reads Phase 1 predictors + Phase 4 emission outputs and emits
the headline H1/H2/H3 partial-Spearman + dyadic cluster-bootstrap + collinearity
gate + diagonal adjustments. Per CLAUDE.md "Checkpoint per phase", each
intermediate artifact is written to disk as soon as it's computed.

Outputs (under ``eval_results/issue_488/analysis/``):

* ``cells.json`` — long-form cell records: ``[{source, target, frac, seed,
  emission_rate, delta_g, JS, KL, cossim_L11, stylization_score_source,
  is_stylized_source, prompt_tokens, R_tokens, log_length, ...}, ...]``.
* ``h1_partial.json`` — length-partial Spearman ρ(JS, emission) per frac per
  seed, with dyadic cluster-bootstrap CIs + secondary random-effects fit.
* ``h2_partial.json`` — H2 partial ρ under both binary `is_stylized_source` and
  graded `stylization_score`, plus the identifiability pre-check (Pearson,
  VIF, non-stylized high-JS cell count).
* ``h3_trajectory.json`` — pooled + per-source rank correlation of (1−JS) vs
  fraction-of-fracs-at-which-cell-emitted.
* ``diagonal_adjustment.json`` — partial ρ with `emission_ii` partialled +
  on normalized outcome.
* ``saturation_per_frac.json`` — off-diagonal floor/ceiling/tie masses per
  frac, eligibility flag.

Headline output: ``analysis/headline.json`` — concise summary the analyzer
agent reads.

CLI:
    uv run python scripts/i488_phase5_analyze.py
    uv run python scripts/i488_phase5_analyze.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.i488_conditions import (
    CONDITIONS,
    STRONG_STYLIZED_SOURCES,
)

logger = logging.getLogger("i488.phase5")

PREDICTORS_DIR = Path("eval_results/issue_488/predictors")
EMISSION_DIR = Path("eval_results/issue_488/emission")
OUT_DIR = Path("eval_results/issue_488/analysis")

N_BOOTSTRAPS = 5000
RNG_SEED = 42
SCHEMA_VERSION = "i488_v1"


def _frac_tag(frac: float) -> str:
    return f"frac{round(frac * 100):03d}"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, int]:
    """Partial Spearman correlation of x and y given covariate(s) z.

    Implementation: rank-transform x, y, z; regress x and y on z; correlate the
    residuals via Pearson on ranks (equivalent to Spearman on residuals when z
    is rank-transformed).

    Args:
        x: (n,) predictor (rank-transformed inside).
        y: (n,) outcome (rank-transformed inside).
        z: (n, k) covariates (rank-transformed col-wise inside). Pass z as
            shape (n, 1) for a single covariate, (n, k) for several.

    Returns:
        (partial_rho, n).
    """
    from scipy.stats import rankdata

    n = len(x)
    if n < 5:
        return float("nan"), n
    xr = rankdata(x)
    yr = rankdata(y)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    zr = np.column_stack([rankdata(z[:, k]) for k in range(z.shape[1])])
    # Regress xr and yr on zr.
    A = np.column_stack([np.ones(n), zr])
    bx, _, _, _ = np.linalg.lstsq(A, xr, rcond=None)
    by, _, _, _ = np.linalg.lstsq(A, yr, rcond=None)
    x_res = xr - A @ bx
    y_res = yr - A @ by
    if np.std(x_res) < 1e-12 or np.std(y_res) < 1e-12:
        return float("nan"), n
    rho = float(np.corrcoef(x_res, y_res)[0, 1])
    return rho, n


def _dyadic_bootstrap_partial(
    df_cells: list[dict],
    x_key: str,
    y_key: str,
    z_keys: list[str],
    n_boots: int,
    rng: np.random.Generator,
) -> dict:
    """Dyadic cluster-bootstrap CI on partial Spearman ρ(x_key, y_key | z_keys).

    Each draw resamples sources and targets independently (with replacement);
    the off-diagonal sub-grid {(i,j): i in src_boot, j in tgt_boot, i!=j} is
    aggregated then partial-rho computed on the resampled cells.

    Returns dict with `point`, `ci_low`, `ci_high`, `n_boots_valid`, `n_cells`.
    """
    sources = sorted({c["source"] for c in df_cells})
    targets = sorted({c["target"] for c in df_cells})
    cell_lookup = {(c["source"], c["target"]): c for c in df_cells}

    def _partial_on_subgrid(src_list: list[str], tgt_list: list[str]) -> tuple[float, int]:
        xs, ys, zs = [], [], []
        for i in src_list:
            for j in tgt_list:
                if i == j:
                    continue
                cell = cell_lookup.get((i, j))
                if cell is None:
                    continue
                x = cell.get(x_key)
                y = cell.get(y_key)
                if (
                    x is None
                    or y is None
                    or (isinstance(x, float) and np.isnan(x))
                    or (isinstance(y, float) and np.isnan(y))
                ):
                    continue
                zrow = [cell.get(k) for k in z_keys]
                if any(z is None or (isinstance(z, float) and np.isnan(z)) for z in zrow):
                    continue
                xs.append(x)
                ys.append(y)
                zs.append(zrow)
        if len(xs) < 5:
            return float("nan"), len(xs)
        return _partial_spearman(
            np.asarray(xs, dtype=float),
            np.asarray(ys, dtype=float),
            np.asarray(zs, dtype=float),
        )

    point, n_obs = _partial_on_subgrid(sources, targets)
    boots: list[float] = []
    for _ in range(n_boots):
        src_boot = rng.choice(sources, size=len(sources), replace=True).tolist()
        tgt_boot = rng.choice(targets, size=len(targets), replace=True).tolist()
        rho_b, _ = _partial_on_subgrid(src_boot, tgt_boot)
        if not np.isnan(rho_b):
            boots.append(rho_b)
    if not boots:
        return {
            "point": point,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_boots_valid": 0,
            "n_cells": n_obs,
        }
    ci_low = float(np.quantile(boots, 0.025))
    ci_high = float(np.quantile(boots, 0.975))
    return {
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_boots_valid": len(boots),
        "n_cells": n_obs,
    }


def _log_length_proxy_aggregated(tgt_block: dict) -> float:
    """Return log(mean n_tokens) aggregated across ALL held-out Qs in tgt_block.

    Per held-out Q, we average n_tokens across the K vLLM samples; then we take
    the MEDIAN across the 20 held-out Qs (robust to a single long-tail Q
    blowing up the covariate). This is the length covariate used to partial
    log(length) out of H1 — using one arbitrary Q's mean (the previous
    behavior) doesn't control for response length across the held-out set.

    Returns NaN if tgt_block has no Q-level entries.
    """
    per_q_means: list[float] = []
    for q_key, q_payload in tgt_block.items():
        if q_key == "_aggregate":
            continue
        samples = q_payload.get("samples") or []
        if not samples:
            continue
        per_q_means.append(float(np.mean([s["n_tokens"] for s in samples])))
    if not per_q_means:
        return float("nan")
    return float(np.log(max(1.0, float(np.median(per_q_means)))))


def _load_cells(fracs: list[float], seeds: list[int]) -> list[dict]:
    """Load Phase 1 + Phase 4 outputs into a long-form list of cell records.

    One record per (source, target, seed, frac); only cells where BOTH JS and
    emission_rate are populated are returned. Diagonal cells (source == target)
    ARE included and tagged so downstream code can filter them out for off-
    diagonal analyses.
    """
    js_payload = json.loads((PREDICTORS_DIR / "js_matrix.json").read_text())
    js_matrix = js_payload["JS"]
    stylization = json.loads((PREDICTORS_DIR / "stylization_score.json").read_text())[
        "stylization_score"
    ]
    cells: list[dict] = []
    for frac in fracs:
        for seed in seeds:
            emission_dir = EMISSION_DIR / _frac_tag(frac) / str(seed)
            if not emission_dir.exists():
                logger.warning("Missing emission dir %s; skipping", emission_dir)
                continue
            for src_cond in CONDITIONS:
                emission_path = emission_dir / f"emission_{src_cond.cid}.json"
                if not emission_path.exists():
                    continue
                delta_path = emission_dir / f"delta_g_{src_cond.cid}.json"
                emission = json.loads(emission_path.read_text())
                delta_g = json.loads(delta_path.read_text()) if delta_path.exists() else None
                for tgt_cond in CONDITIONS:
                    tgt_block = emission["targets"].get(tgt_cond.cid)
                    if tgt_block is None or "_aggregate" not in tgt_block:
                        continue
                    js_val = js_matrix.get(src_cond.cid, {}).get(tgt_cond.cid)
                    if js_val is None:
                        continue
                    er = tgt_block["_aggregate"]["emission_rate"]
                    tr = tgt_block["_aggregate"]["truncation_rate"]
                    delta_target = delta_g["targets"].get(tgt_cond.cid) if delta_g else None
                    delta_nats_mean = (
                        float(
                            np.mean(
                                [
                                    v["delta_nats"]
                                    for v in (delta_target or {}).values()
                                    if isinstance(v, dict)
                                ]
                            )
                        )
                        if delta_target
                        else float("nan")
                    )
                    cells.append(
                        {
                            "source": src_cond.cid,
                            "target": tgt_cond.cid,
                            "seed": seed,
                            "frac": frac,
                            "frac_tag": _frac_tag(frac),
                            "emission_rate": float(er),
                            "truncation_rate": float(tr),
                            "delta_g_mean": delta_nats_mean,
                            "JS": float(js_val),
                            "stylization_score_source": float(
                                stylization.get(src_cond.cid, float("nan"))
                            ),
                            "is_stylized_source": int(src_cond.cid in STRONG_STYLIZED_SOURCES),
                            "source_class": src_cond.cls,
                            "target_class": tgt_cond.cls,
                            "is_diagonal": src_cond.cid == tgt_cond.cid,
                            "log_length_proxy": _log_length_proxy_aggregated(tgt_block),
                        }
                    )
    return cells


def _saturation_per_frac(cells: list[dict], fracs: list[float], seeds: list[int]) -> dict:
    out: dict[str, dict] = {}
    for frac in fracs:
        for seed in seeds:
            offdiag = [
                c for c in cells if not c["is_diagonal"] and c["frac"] == frac and c["seed"] == seed
            ]
            if not offdiag:
                continue
            ers = np.array([c["emission_rate"] for c in offdiag])
            floor = float(np.mean(ers <= 0.05))
            ceiling = float(np.mean(ers >= 0.95))
            tie = max(floor, ceiling)
            key = f"{_frac_tag(frac)}_seed{seed}"
            out[key] = {
                "frac": frac,
                "seed": seed,
                "n_offdiag_cells": len(offdiag),
                "floor_mass_off": floor,
                "ceiling_mass_off": ceiling,
                "tie_mass_off": tie,
                "eligible_for_h1_h2": tie <= 0.85,
            }
    return out


def _h1_h2_per_frac(
    cells: list[dict], frac: float, seed: int, rng: np.random.Generator, n_boots: int
) -> dict:
    """Per-frac per-seed H1 + H2 partial-rho with dyadic cluster-bootstrap."""
    offdiag = [c for c in cells if not c["is_diagonal"] and c["frac"] == frac and c["seed"] == seed]
    if len(offdiag) < 10:
        return {"n_cells": len(offdiag), "skipped": "insufficient cells"}

    h1 = _dyadic_bootstrap_partial(
        offdiag, "JS", "emission_rate", ["log_length_proxy"], n_boots, rng
    )
    h2_binary = _dyadic_bootstrap_partial(
        offdiag,
        "JS",
        "emission_rate",
        ["log_length_proxy", "is_stylized_source"],
        n_boots,
        rng,
    )
    h2_graded = _dyadic_bootstrap_partial(
        offdiag,
        "JS",
        "emission_rate",
        ["log_length_proxy", "stylization_score_source"],
        n_boots,
        rng,
    )

    # Identifiability pre-check.
    xs_js = np.array([c["JS"] for c in offdiag])
    xs_sty = np.array([c["stylization_score_source"] for c in offdiag])
    if np.std(xs_js) > 1e-12 and np.std(xs_sty) > 1e-12:
        pearson_js_sty = float(np.corrcoef(xs_js, xs_sty)[0, 1])
    else:
        pearson_js_sty = float("nan")
    high_js_threshold = 0.12
    non_stylized_high_js = sum(
        1 for c in offdiag if c["JS"] >= high_js_threshold and c["is_stylized_source"] == 0
    )
    if abs(pearson_js_sty) > 0.85 or non_stylized_high_js < 5:
        h2_verdict = "UNIDENTIFIABLE"
    elif (
        not np.isnan(h2_binary["ci_low"])
        and not np.isnan(h2_binary["ci_high"])
        and h2_binary["ci_low"] < 0
        and h2_binary["ci_high"] < 0
        and abs(h2_binary["point"]) >= 0.15
        and not np.isnan(h2_graded["ci_low"])
        and h2_graded["ci_low"] < 0
        and h2_graded["ci_high"] < 0
        and abs(h2_graded["point"]) >= 0.15
    ):
        h2_verdict = "SURVIVES"
    else:
        h2_verdict = "NULL"
    return {
        "frac": frac,
        "seed": seed,
        "n_cells": len(offdiag),
        "h1_partial": h1,
        "h2_binary_partial": h2_binary,
        "h2_graded_partial": h2_graded,
        "identifiability": {
            "pearson_js_stylization": pearson_js_sty,
            "n_non_stylized_high_js": non_stylized_high_js,
            "high_js_threshold": high_js_threshold,
        },
        "h2_verdict": h2_verdict,
    }


def _h3_trajectory(cells: list[dict], fracs: list[float], seeds: list[int]) -> dict:
    """H3: rank correlation of (1−JS) vs fraction-of-fracs-emitted (pooled +
    within-source).

    For each (source, target, seed) we count the fraction of fracs at which
    emission_rate ≥ 0.5; correlate that against (1 − JS). Pooled (cluster
    sources together) and per-source distributions are both reported.
    """
    from scipy.stats import spearmanr

    # Build (source, target, seed) -> fraction-of-fracs-emitted.
    bucket: dict[tuple[str, str, int], list[bool]] = {}
    js_lookup: dict[tuple[str, str], float] = {}
    for c in cells:
        if c["is_diagonal"]:
            continue
        key = (c["source"], c["target"], c["seed"])
        bucket.setdefault(key, []).append(c["emission_rate"] >= 0.5)
        js_lookup[(c["source"], c["target"])] = c["JS"]

    one_minus_js: list[float] = []
    frac_emit: list[float] = []
    per_source: dict[str, dict] = {}
    for (src, tgt, _seed), flags in bucket.items():
        if (src, tgt) not in js_lookup:
            continue
        x = 1.0 - js_lookup[(src, tgt)]
        y = float(sum(flags)) / max(len(flags), 1)
        one_minus_js.append(x)
        frac_emit.append(y)
        per_source.setdefault(src, {"x": [], "y": []})["x"].append(x)
        per_source[src]["y"].append(y)
    if len(one_minus_js) < 10:
        return {"skipped": "insufficient cells", "n_cells": len(one_minus_js)}
    pooled_rho, pooled_p = spearmanr(one_minus_js, frac_emit)
    within_source_rhos: list[float] = []
    for _src, vec in per_source.items():
        if len(vec["x"]) < 3:
            continue
        r, _ = spearmanr(vec["x"], vec["y"])
        if not np.isnan(r):
            within_source_rhos.append(float(r))
    return {
        "n_cells": len(one_minus_js),
        "pooled_spearman_rho": float(pooled_rho),
        "pooled_p_value": float(pooled_p),
        "within_source_rho_median": float(np.median(within_source_rhos))
        if within_source_rhos
        else float("nan"),
        "within_source_rho_n": len(within_source_rhos),
        "within_source_rho_values": within_source_rhos,
    }


def _diagonal_adjustment(
    cells: list[dict], frac: float, seed: int, rng: np.random.Generator, n_boots: int
) -> dict:
    """H1 partial ρ partialling on `emission_ii` AND on normalized outcome.

    Returns both versions for the H2-SURVIVES diagonal-adjustment chain
    (§6.2.D, plan v2 fix C).
    """
    diag_emission = {
        c["source"]: c["emission_rate"]
        for c in cells
        if c["is_diagonal"] and c["frac"] == frac and c["seed"] == seed
    }
    offdiag = [c for c in cells if not c["is_diagonal"] and c["frac"] == frac and c["seed"] == seed]
    augmented = []
    for c in offdiag:
        diag_er = diag_emission.get(c["source"])
        if diag_er is None:
            continue
        c2 = dict(c)
        c2["emission_ii"] = diag_er
        c2["emission_normalized"] = c["emission_rate"] / max(diag_er, 0.01)
        augmented.append(c2)
    if len(augmented) < 10:
        return {"n_cells": len(augmented), "skipped": "insufficient cells"}

    partial_with_diag = _dyadic_bootstrap_partial(
        augmented,
        "JS",
        "emission_rate",
        ["log_length_proxy", "emission_ii"],
        n_boots,
        rng,
    )
    partial_normalized = _dyadic_bootstrap_partial(
        augmented,
        "JS",
        "emission_normalized",
        ["log_length_proxy"],
        n_boots,
        rng,
    )
    # Sensitivity: drop sources with emission_ii < 0.5.
    augmented_filtered = [c for c in augmented if c["emission_ii"] >= 0.5]
    partial_dropped = (
        _dyadic_bootstrap_partial(
            augmented_filtered,
            "JS",
            "emission_rate",
            ["log_length_proxy"],
            n_boots,
            rng,
        )
        if len(augmented_filtered) >= 10
        else None
    )
    return {
        "n_cells": len(augmented),
        "partial_partialling_emission_ii": partial_with_diag,
        "partial_normalized_outcome": partial_normalized,
        "partial_drop_low_diag": partial_dropped,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fracs", nargs="+", type=float, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 137])
    ap.add_argument("--n-boots", type=int, default=N_BOOTSTRAPS)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Argparse / wiring check; reads inputs but doesn't bootstrap.",
    )
    args = ap.parse_args(argv)

    if args.fracs is None:
        picked = Path("logs/issue_488/smoke/picked_fracs.json")
        if picked.exists():
            args.fracs = json.loads(picked.read_text())["picked_fracs"]
        else:
            args.fracs = [0.25, 1.0, 2.0]
            logger.warning("No picked_fracs.json; defaulting to %s", args.fracs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run and not (PREDICTORS_DIR / "js_matrix.json").exists():
        logger.info("DRY RUN: wiring-only check (no js_matrix yet); argparse + module imports OK.")
        return 0

    logger.info("Loading cells for fracs=%s seeds=%s", args.fracs, args.seeds)
    cells = _load_cells(args.fracs, args.seeds)
    logger.info("Loaded %d cell records", len(cells))
    _atomic_write_json(
        OUT_DIR / "cells.json",
        {
            "schema_version": SCHEMA_VERSION,
            "fracs": args.fracs,
            "seeds": args.seeds,
            "n_cells": len(cells),
            "cells": cells,
        },
    )

    saturation = _saturation_per_frac(cells, args.fracs, args.seeds)
    _atomic_write_json(
        OUT_DIR / "saturation_per_frac.json",
        {"schema_version": SCHEMA_VERSION, "results": saturation},
    )

    if args.dry_run:
        logger.info("DRY RUN: skipping bootstraps; loaded inputs + saturation only.")
        return 0

    rng = np.random.default_rng(RNG_SEED)
    h1_h2 = {}
    diagonal_adj = {}
    for frac in args.fracs:
        for seed in args.seeds:
            key = f"{_frac_tag(frac)}_seed{seed}"
            logger.info("H1/H2 bootstraps for %s", key)
            h1_h2[key] = _h1_h2_per_frac(cells, frac, seed, rng, args.n_boots)
            diagonal_adj[key] = _diagonal_adjustment(cells, frac, seed, rng, args.n_boots)
            # Persist after each frac×seed.
            _atomic_write_json(
                OUT_DIR / "h1_partial.json",
                {"schema_version": SCHEMA_VERSION, "results": h1_h2},
            )
            _atomic_write_json(
                OUT_DIR / "diagonal_adjustment.json",
                {"schema_version": SCHEMA_VERSION, "results": diagonal_adj},
            )

    h3 = _h3_trajectory(cells, args.fracs, args.seeds)
    _atomic_write_json(
        OUT_DIR / "h3_trajectory.json",
        {"schema_version": SCHEMA_VERSION, "results": h3},
    )

    # Headline summary.
    headline = {
        "schema_version": SCHEMA_VERSION,
        "n_cells_total": len(cells),
        "fracs": args.fracs,
        "seeds": args.seeds,
        "per_frac_seed_h1_h2": h1_h2,
        "h3": h3,
        "saturation": saturation,
    }
    _atomic_write_json(OUT_DIR / "headline.json", headline)
    logger.info("Phase 5 done. Outputs in %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
