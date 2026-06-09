# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Greek Δ/β intentional
"""Task #505 follow-up `expanded-predictor-reanalysis` — zero-GPU covariate expansion.

Re-runs the #505 leave-one-out analysis over the EXISTING trajectory JSONs with
an expanded predictor / covariate set (followup spec, epm:followup-scope v1):

1. Close the pre-registered source-proximity control: add ``cos(b, source)``
   to (a) the per-arm partial OLS and (b) a pooled OLS analogue of the planned
   (singular) mixed model — arm + seed fixed effects with bystander
   cluster-robust standard errors.
2. Add the per-bystander base-model marker prior ``base_prior_b`` (mean
   ``b_logp`` at the headline read slice) as a covariate.
3. Add the drop-one-design geometry predictors ``shadow_angle(b; j)`` (angle
   between centroid_j − centroid_source and centroid_b − centroid_source) and
   ``d_nearest_remaining_neg(b; cell)`` (cosine distance from b to the nearest
   negative still PRESENT in the drop-j cell, qwen_default included), entered
   jointly with ``cos(b, j)``.
4. Secondary DV: the full-set-differenced ABSOLUTE trained log-prob
   ``delta_trained_abs(b; j, seed) = mean_q g_logp(drop-j) − mean_q
   g_logp(full-set)`` alongside the Δ-leakage shift (which additionally
   subtracts the base-side difference). Propensity can hide in the
   trained−base subtraction, so the two cuts are reported in parallel.

Reuses ``analyze.compute_delta_leakage_table`` VERBATIM for the Δ-leakage
frame (same headline frac 0.50 read slice), then merges per-row ``g_logp`` /
``b_logp`` means over the SAME question intersection and asserts the merged
reconstruction reproduces ``delta_leakage`` exactly. Geometry comes from the
already-published centroid bundles on the HF data repo
(``issue505_loo_contrastive/geometry/centroids_pv_L{7,14,21,27}.pt`` +
``issue472_neg_geometry/geometry/centroids_L10.pt``) — no new computation,
no model forward passes, CPU only.

Writes (all under the followup artifact dir
``eval_results/issue_505/expanded-predictor-reanalysis/``):
  - ``expanded_frame.json``        — 936-row pooled frame, both DVs + covariates.
  - ``geometry_predictors.json``   — per-layer shadow_angle + d_nearest tables.
  - ``per_arm_expanded_ols.json``  — per-arm OLS, original vs expanded covariates,
                                     both DVs, layers {21, 7, 14, 27, 10}.
  - ``pooled_expanded_ols.json``   — pooled OLS (cluster-robust by bystander),
                                     baseline vs expanded, raw + standardized.
  - ``headline_comparison.json``   — compact old-vs-new cos(b, j) table at L21.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from explore_persona_space.experiments.leave_one_out_505 import (
    ALL_SIMILARITY_LAYERS,
    ALWAYS_INCLUDE_NEGATIVE,
    HEADLINE_CHECKPOINT_FRAC,
    HEADLINE_LAYER,
    HF_DATA_PREFIX,
    HF_DATA_PREFIX_INHERIT,
    HF_DATA_REPO,
    INHERITED_L10_LAYER,
    SEEDS,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.leave_one_out_505.analyze import (
    _parse_trajectory_payload,
    compute_delta_leakage_table,
)

log = logging.getLogger("issue_505.analyze_expanded")

FULL_SET_SLUG = "c505_full_set"

# Continuous predictor column order for the pooled models (FE dummies appended).
EXPANDED_PREDICTORS = (
    "cos_b_j",
    "shadow_angle",
    "d_nearest_remaining",
    "cos_b_source",
    "base_prior_b",
    "delta_source_dg",
)
BASELINE_PREDICTORS = ("cos_b_j", "delta_source_dg")
PER_ARM_ORIGINAL = ("cos_b_j", "delta_source_dg")
PER_ARM_EXPANDED = ("cos_b_j", "delta_source_dg", "cos_b_source", "base_prior_b")
DVS = ("delta_leakage", "delta_trained_abs")


def _reproducibility_block(inputs: dict[str, str]) -> dict[str, Any]:
    """Git commit + env versions + timestamp + input paths for result JSONs."""
    import pandas
    import statsmodels

    from explore_persona_space.analysis.paper_plots import _git_commit_hash

    return {
        "git_commit": _git_commit_hash(),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions": {
            "numpy": np.__version__,
            "pandas": pandas.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "inputs": inputs,
    }


# ── Geometry predictors from the published centroid bundles ─────────────────


@dataclass(frozen=True)
class GeometryPredictors:
    """Per-layer geometry predictor tables for the drop-one design.

    cos_b_j / cos_b_source come from the bundle's stored ``cos_matrix`` (the
    exact values behind ``panel_similarity_matrix.json``); ``shadow_angle`` is
    computed from the RAW centroid vectors (pairwise cosines do not determine
    angles between difference vectors); ``d_nearest_remaining`` uses the
    stored cosine matrix over the negatives still present in each drop cell.
    """

    layer: int
    cos_b_j: dict[str, dict[str, float]]  # [b][j]
    cos_b_source: dict[str, float]  # [b]
    shadow_angle: dict[str, dict[str, float]]  # [b][j], radians
    d_nearest_remaining: dict[str, dict[str, float]]  # [b][j_dropped]
    d_nearest_full_set: dict[str, float]  # [b], all negatives present
    remaining_negative_sets: dict[str, list[str]]  # j_dropped -> negatives used


def download_centroid_bundles(
    layers: tuple[int, ...] = ALL_SIMILARITY_LAYERS,
) -> dict[int, Path]:
    """Fetch the per-layer centroid bundles from the HF data repo (cached).

    Layer 10 is the inherited #472 bundle (``centroids_L10.pt``); the rest are
    the #505 persona-vectors bundles (``centroids_pv_L<layer>.pt``). Public
    repo — works without HF_TOKEN; uses the standard hub cache, so no new
    binary lands in the worktree.
    """
    from huggingface_hub import hf_hub_download

    paths: dict[int, Path] = {}
    for layer in layers:
        if layer == INHERITED_L10_LAYER:
            repo_path = f"{HF_DATA_PREFIX_INHERIT}/geometry/centroids_L10.pt"
        else:
            repo_path = f"{HF_DATA_PREFIX}/geometry/centroids_pv_L{layer}.pt"
        local = hf_hub_download(HF_DATA_REPO, repo_path, repo_type="dataset")
        paths[layer] = Path(local)
        log.info("[expanded] centroid bundle L%d: %s", layer, local)
    return paths


def compute_geometry_predictors(
    bundle_path: Path,
    *,
    layer: int,
    panel: list[str],
    negatives: list[str],
    source: str = SOURCE_PERSONA,
    default_negative: str = ALWAYS_INCLUDE_NEGATIVE,
) -> GeometryPredictors:
    """Build the per-layer geometry predictor tables from one centroid bundle.

    Fails loud on any missing persona (panel, negatives, source, or the
    always-included default negative) and on a zero-norm difference vector.
    """
    import torch

    bundle = torch.load(bundle_path, weights_only=False)
    names: list[str] = list(bundle["persona_names"])
    idx = {n: i for i, n in enumerate(names)}
    required = [source, default_negative, *negatives, *panel]
    missing = [n for n in required if n not in idx]
    if missing:
        raise KeyError(
            f"layer {layer} centroid bundle {bundle_path} missing personas {missing}; "
            f"bundle has {len(names)} personas."
        )
    overlap = set(panel) & ({source, default_negative} | set(negatives))
    if overlap:
        raise ValueError(
            f"panel overlaps the negative/source set ({sorted(overlap)}) — "
            "d_nearest_remaining would be 0 by self-distance; refusing to proceed."
        )

    cos_t = bundle["cos_matrix"]
    cents = bundle["centroids"].to(torch.float64).numpy()  # (n, d)

    def stored_cos(a: str, b: str) -> float:
        return float(cos_t[idx[a], idx[b]].item())

    # shadow_angle from RAW centroids: angle between (c_j − c_s) and (c_b − c_s).
    c_source = cents[idx[source]]
    diff_j = {j: cents[idx[j]] - c_source for j in negatives}
    for j, v in diff_j.items():
        if float(np.linalg.norm(v)) == 0.0:
            raise ValueError(f"layer {layer}: centroid_{j} == centroid_{source}; degenerate.")

    cos_b_j: dict[str, dict[str, float]] = {}
    cos_b_source: dict[str, float] = {}
    shadow_angle: dict[str, dict[str, float]] = {}
    for b in panel:
        cos_b_source[b] = stored_cos(b, source)
        cos_b_j[b] = {j: stored_cos(b, j) for j in negatives}
        v_b = cents[idx[b]] - c_source
        norm_b = float(np.linalg.norm(v_b))
        if norm_b == 0.0:
            raise ValueError(f"layer {layer}: centroid_{b} == centroid_{source}; degenerate.")
        shadow_angle[b] = {}
        for j in negatives:
            u = diff_j[j]
            c = float(np.dot(u, v_b) / (np.linalg.norm(u) * norm_b))
            shadow_angle[b][j] = float(math.acos(max(-1.0, min(1.0, c))))

    # d_nearest over negatives PRESENT per cell (1 − cos). qwen_default is a
    # negative in EVERY cell and its centroid exists in the bundle, so it is
    # always part of the remaining set.
    remaining_sets = {j: [n for n in negatives if n != j] + [default_negative] for j in negatives}
    d_nearest_remaining: dict[str, dict[str, float]] = {}
    d_nearest_full: dict[str, float] = {}
    all_present = [*negatives, default_negative]
    for b in panel:
        d_nearest_full[b] = min(1.0 - stored_cos(b, n) for n in all_present)
        d_nearest_remaining[b] = {
            j: min(1.0 - stored_cos(b, n) for n in remaining)
            for j, remaining in remaining_sets.items()
        }

    return GeometryPredictors(
        layer=layer,
        cos_b_j=cos_b_j,
        cos_b_source=cos_b_source,
        shadow_angle=shadow_angle,
        d_nearest_remaining=d_nearest_remaining,
        d_nearest_full_set=d_nearest_full,
        remaining_negative_sets=remaining_sets,
    )


def assert_geometry_matches_original(
    geom: GeometryPredictors, similarity_matrix_path: Path, atol: float = 1e-6
) -> None:
    """Cross-check bundle-derived cosines against the original analysis artifact.

    Guards against reading a different bundle than the one
    ``panel_similarity_matrix.json`` was built from.
    """
    sim = json.loads(similarity_matrix_path.read_text())
    key = f"L{geom.layer}"
    if key not in sim:
        raise KeyError(f"{similarity_matrix_path} has no {key} block; layers: {sim.get('layers')}")
    for b, per_j in geom.cos_b_j.items():
        for j, v in per_j.items():
            ref = sim[key]["cos_b_j"][b][j]
            if abs(v - ref) > atol:
                raise AssertionError(
                    f"L{geom.layer} cos({b},{j}) mismatch vs panel_similarity_matrix.json: "
                    f"bundle {v} vs original {ref}"
                )
    for b, v in geom.cos_b_source.items():
        ref = sim[key]["cos_b_source"][b]
        if abs(v - ref) > atol:
            raise AssertionError(
                f"L{geom.layer} cos({b},source) mismatch: bundle {v} vs original {ref}"
            )


# ── Expanded frame: both DVs + base prior ───────────────────────────────────


def _gb_per_q(
    payload: dict, *, frac: float, persona: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-question ``(g_logp, b_logp)`` dicts for one persona at one frac."""
    target = frac

    def _frac_match(ckpt: dict) -> bool:
        raw = ckpt.get("frac")
        return isinstance(raw, (int, float)) and abs(float(raw) - target) < 1e-4

    ckpt = next((c for c in payload["checkpoints"] if _frac_match(c)), None)
    if ckpt is None:
        raise KeyError(
            f"trajectory has no checkpoint at frac={frac!r}; "
            f"checkpoints: {[c.get('frac') for c in payload['checkpoints']]}"
        )
    held = ckpt.get("held_out", {})
    if persona not in held:
        raise KeyError(f"frac={frac} missing held-out persona {persona!r}")
    g = {q: float(leaf["g_logp"]) for q, leaf in held[persona].items()}
    b = {q: float(leaf["b_logp"]) for q, leaf in held[persona].items()}
    return g, b


def build_expanded_frame(
    *,
    sweep_dir: Path,
    panel: list[str],
    non_default_negatives: list[str],
    seeds: tuple[int, ...] = SEEDS,
    frac: float = HEADLINE_CHECKPOINT_FRAC,
    source: str = SOURCE_PERSONA,
) -> dict:
    """The 936-row pooled frame with BOTH DVs + the per-bystander base prior.

    DV #1 ``delta_leakage`` is taken VERBATIM from
    ``analyze.compute_delta_leakage_table`` (same read slice as the original
    headline). DV #2 ``delta_trained_abs = mean_q g_logp(drop-j) − mean_q
    g_logp(full-set)`` over the SAME question intersection; the merge is
    validated by asserting ``(g_drop − b_drop) − (g_full − b_full)`` equals
    ``delta_leakage`` to 1e-6 on every row.

    ``base_prior_b`` is the per-bystander scalar mean of ``b_logp`` over the 7
    in-design cells (full-set + 6 drop arms) × seeds × questions at the read
    slice. The base model is fixed; b_logp varies only through the trained
    model's own on-policy response R, so pooling across cells/seeds gives the
    stable bystander-level prior the followup spec asks for.
    """
    table = compute_delta_leakage_table(
        sweep_dir=sweep_dir,
        panel=panel,
        non_default_negatives=non_default_negatives,
        seeds=seeds,
        frac=frac,
        source=source,
    )
    if table["missing_cells"]:
        raise RuntimeError(f"missing sweep cells: {table['missing_cells']} — frame incomplete.")
    expected = len(non_default_negatives) * len(seeds) * len(panel)
    if len(table["rows"]) != expected:
        raise RuntimeError(f"frame has {len(table['rows'])} rows; expected {expected}.")

    # Per-(cell, seed) trajectory payloads (full set + drop arms).
    slugs = [FULL_SET_SLUG] + [f"c505_drop_j{i}" for i in range(len(non_default_negatives))]
    payloads = {
        (slug, seed): _parse_trajectory_payload(
            sweep_dir / slug / f"seed_{seed}" / "trajectory.json"
        )
        for slug in slugs
        for seed in seeds
    }

    # Per-bystander base prior over all in-design cells × seeds × questions.
    base_prior_b: dict[str, float] = {}
    for b in panel:
        vals: list[float] = []
        for payload in payloads.values():
            _, b_map = _gb_per_q(payload, frac=frac, persona=b)
            vals.extend(b_map.values())
        base_prior_b[b] = float(np.mean(vals))

    rows: list[dict[str, Any]] = []
    for r in table["rows"]:
        slug = f"c505_drop_j{r['j_idx']}"
        g_full, b_full = _gb_per_q(payloads[(FULL_SET_SLUG, r["seed"])], frac=frac, persona=r["b"])
        g_drop, b_drop = _gb_per_q(payloads[(slug, r["seed"])], frac=frac, persona=r["b"])
        common_q = sorted(set(g_full) & set(g_drop))
        if len(common_q) != r["n_q"]:
            raise AssertionError(
                f"question-set mismatch for (b={r['b']}, j={r['j_i']}, seed={r['seed']}): "
                f"{len(common_q)} vs original n_q={r['n_q']}"
            )
        g_full_mean = float(np.mean([g_full[q] for q in common_q]))
        g_drop_mean = float(np.mean([g_drop[q] for q in common_q]))
        b_full_mean = float(np.mean([b_full[q] for q in common_q]))
        b_drop_mean = float(np.mean([b_drop[q] for q in common_q]))
        recon = (g_drop_mean - b_drop_mean) - (g_full_mean - b_full_mean)
        if abs(recon - r["delta_leakage"]) > 1e-6:
            raise AssertionError(
                f"Δ-leakage reconstruction mismatch for (b={r['b']}, j={r['j_i']}, "
                f"seed={r['seed']}): {recon} vs {r['delta_leakage']}"
            )
        rows.append(
            {
                **r,
                "g_full_mean": g_full_mean,
                "g_drop_mean": g_drop_mean,
                "b_full_mean": b_full_mean,
                "b_drop_mean": b_drop_mean,
                "delta_trained_abs": g_drop_mean - g_full_mean,
                "base_prior_b": base_prior_b[r["b"]],
            }
        )

    return {
        "rows": rows,
        "base_prior_b": base_prior_b,
        "frac": frac,
        "n_rows": len(rows),
        "dv_construction": {
            "delta_leakage": (
                "mean_q ΔG(drop-j) − mean_q ΔG(full-set); ΔG = g_logp − b_logp "
                "(trained − base log P(marker) at the post-response slot); verbatim "
                "from analyze.compute_delta_leakage_table at frac "
                f"{frac}"
            ),
            "delta_trained_abs": (
                "mean_q g_logp(drop-j) − mean_q g_logp(full-set) over the same question "
                "intersection — the full-set-differenced ABSOLUTE trained log-prob. "
                "Differs from delta_leakage by NOT subtracting the base-side difference "
                "(b_drop − b_full), so trained-propensity shifts that the trained−base "
                "subtraction hides remain visible."
            ),
            "base_prior_b": (
                "per-bystander mean of b_logp over the 7 in-design cells (full-set + 6 "
                "drop arms) × 3 seeds × eval questions at the read slice"
            ),
        },
    }


# ── Regressions ─────────────────────────────────────────────────────────────


def _attach_layer_predictors(rows: list[dict], geom: GeometryPredictors) -> pd.DataFrame:  # noqa: F821
    """Long-format frame with the per-layer geometry columns attached."""
    import pandas as pd

    df = pd.DataFrame(rows)
    df["cos_b_j"] = [geom.cos_b_j[r["b"]][r["j_i"]] for r in rows]
    df["cos_b_source"] = [geom.cos_b_source[r["b"]] for r in rows]
    df["shadow_angle"] = [geom.shadow_angle[r["b"]][r["j_i"]] for r in rows]
    df["d_nearest_remaining"] = [geom.d_nearest_remaining[r["b"]][r["j_i"]] for r in rows]
    return df


def _coef_table(res, names: list[str]) -> dict[str, dict[str, float]]:
    """Per-coefficient estimate / SE / p / 95% CI from a fitted statsmodels result."""
    ci = res.conf_int(alpha=0.05)
    out: dict[str, dict[str, float]] = {}
    for name in names:
        out[name] = {
            "beta": float(res.params[name]),
            "se": float(res.bse[name]),
            "p": float(res.pvalues[name]),
            "ci95_low": float(ci.loc[name][0]),
            "ci95_high": float(ci.loc[name][1]),
        }
    return out


def fit_per_arm_models(df, *, dv: str, predictors: tuple[str, ...]) -> dict:
    """Per-arm OLS ``dv ~ const + predictors`` with HC2 SEs + cos sign-agreement.

    Within one arm every geometry predictor is constant across seeds (it varies
    only over the 52 bystanders), so each arm pools 3 seeds × 52 b = 156 rows.
    """
    import statsmodels.api as sm

    per_arm: dict[str, Any] = {}
    for j_i, sub in df.groupby("j_i", sort=False):
        x = sm.add_constant(sub[list(predictors)])
        res = sm.OLS(sub[dv].to_numpy(), x).fit(cov_type="HC2")
        per_arm[str(j_i)] = {
            "n_rows": len(sub),
            "coefficients": _coef_table(res, list(predictors)),
            "intercept": float(res.params["const"]),
            "r_squared": float(res.rsquared),
        }
    betas = [stats["coefficients"]["cos_b_j"]["beta"] for stats in per_arm.values()]
    return {
        "per_arm": per_arm,
        "predictors": list(predictors),
        "dv": dv,
        "sign_agreement_cos_b_j": {
            "n_positive": int(sum(1 for b in betas if b > 0)),
            "n_total": len(betas),
            "pre_registered_threshold": ">= 5/6 positive",
        },
    }


def fit_pooled_model(
    df,
    *,
    dv: str,
    predictors: tuple[str, ...],
    standardize: bool = False,
) -> dict:
    """Pooled OLS ``dv ~ predictors + C(j_i) + C(seed)`` with bystander-cluster SEs.

    The pooled-OLS-with-cluster-robust-SEs design is the executable analogue of
    the plan's mixed model (which fit singular at every layer): arm + seed
    enter as fixed-effect dummies (drop-first) and the bystander random effect
    is replaced by clustering the SEs on bystander id (52 clusters,
    t-distribution small-sample correction, statsmodels default ``use_t``).

    ``standardize=True`` z-scores every continuous predictor (NOT the DV, NOT
    the FE dummies) so coefficients are comparable across predictors with
    different native scales (cosines vs radians vs nats).
    """
    import pandas as pd
    import statsmodels.api as sm

    work = df.copy()
    scale_info: dict[str, dict[str, float]] = {}
    if standardize:
        for col in predictors:
            mu = float(work[col].mean())
            sd = float(work[col].std(ddof=0))
            if sd == 0.0:
                raise ValueError(f"predictor {col} has zero variance; cannot standardize.")
            work[col] = (work[col] - mu) / sd
            scale_info[col] = {"mean": mu, "sd": sd}

    j_dummies = pd.get_dummies(work["j_i"], prefix="j", drop_first=True, dtype=float)
    seed_dummies = pd.get_dummies(
        work["seed"].astype(str), prefix="seed", drop_first=True, dtype=float
    )
    exog = sm.add_constant(pd.concat([work[list(predictors)], j_dummies, seed_dummies], axis=1))
    groups = pd.factorize(work["b"])[0]
    res = sm.OLS(work[dv].to_numpy(), exog).fit(cov_type="cluster", cov_kwds={"groups": groups})

    pred_corr = work[list(predictors)].corr()
    return {
        "dv": dv,
        "predictors": list(predictors),
        "standardized": standardize,
        "n_rows": len(work),
        "n_clusters_bystander": len(set(groups)),
        "coefficients": _coef_table(res, list(predictors)),
        "fixed_effects": "C(j_i) drop-first + C(seed) drop-first",
        "cov_type": "cluster (bystander id)",
        "r_squared": float(res.rsquared),
        "predictor_correlations": {
            a: {b: float(pred_corr.loc[a, b]) for b in predictors} for a in predictors
        },
        **({"standardization": scale_info} if standardize else {}),
    }


# ── End-to-end ──────────────────────────────────────────────────────────────


def run_expanded_analysis(
    *,
    panel_gate_path: Path,
    sweep_dir: Path,
    original_analysis_dir: Path,
    out_dir: Path,
    layers: tuple[int, ...] = ALL_SIMILARITY_LAYERS,
    frac: float = HEADLINE_CHECKPOINT_FRAC,
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    """Frame → geometry → per-arm + pooled regressions → comparison table.

    Checkpoint-per-phase: each artifact is written the moment its phase
    completes. The original analysis outputs under ``original_analysis_dir``
    are read-only (cross-checks + old-vs-new comparison); everything new lands
    under ``out_dir``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "panel_gate": str(panel_gate_path),
        "sweep_dir": str(sweep_dir),
        "original_analysis_dir": str(original_analysis_dir),
        "hf_data_repo": HF_DATA_REPO,
    }
    repro = _reproducibility_block(inputs)

    panel_payload = json.loads(panel_gate_path.read_text())
    if not panel_payload.get("gate_passed"):
        raise RuntimeError("panel coverage gate did not pass; refusing to analyze.")
    panel = list(panel_payload["panel"])
    non_default = list(panel_payload["non_default_negatives"])

    # Phase A: expanded frame (both DVs + base prior), validated against the
    # original Δ-leakage construction row by row.
    frame = build_expanded_frame(
        sweep_dir=sweep_dir,
        panel=panel,
        non_default_negatives=non_default,
        seeds=seeds,
        frac=frac,
    )
    (out_dir / "expanded_frame.json").write_text(
        json.dumps({**frame, "reproducibility": repro}, indent=2)
    )
    log.info("[expanded] frame: %d rows", frame["n_rows"])

    # Phase B: per-layer geometry predictors from the published bundles,
    # cross-checked against the original similarity matrix artifact.
    bundle_paths = download_centroid_bundles(layers)
    sim_path = original_analysis_dir / "panel_similarity_matrix.json"
    geoms: dict[int, GeometryPredictors] = {}
    for layer in layers:
        geom = compute_geometry_predictors(
            bundle_paths[layer],
            layer=layer,
            panel=panel,
            negatives=non_default,
        )
        assert_geometry_matches_original(geom, sim_path)
        geoms[layer] = geom
    (out_dir / "geometry_predictors.json").write_text(
        json.dumps(
            {
                "layers": list(layers),
                "per_layer": {str(layer): asdict(g) for layer, g in geoms.items()},
                "notes": (
                    "shadow_angle in radians from raw centroid difference vectors; "
                    "d_nearest_remaining = min over negatives present in the drop cell "
                    "(remaining 5 named + qwen_default, whose centroid exists in every "
                    "bundle) of 1 − cos(b, negative)."
                ),
                "reproducibility": repro,
            },
            indent=2,
        )
    )

    # Phase C: per-arm OLS (original vs expanded covariates, both DVs, all layers).
    per_arm_out: dict[str, Any] = {"layers": list(layers), "per_layer": {}}
    for layer in layers:
        df = _attach_layer_predictors(frame["rows"], geoms[layer])
        per_arm_out["per_layer"][str(layer)] = {
            dv: {
                "original_covariates": fit_per_arm_models(df, dv=dv, predictors=PER_ARM_ORIGINAL),
                "expanded_covariates": fit_per_arm_models(df, dv=dv, predictors=PER_ARM_EXPANDED),
            }
            for dv in DVS
        }
    per_arm_out["reproducibility"] = repro

    # Cross-check: the original-covariate per-arm betas at the headline layer
    # must reproduce the executed per_arm_partial_ols.json point estimates.
    orig_partial_path = original_analysis_dir / "per_arm_partial_ols.json"
    orig_partial = json.loads(orig_partial_path.read_text())
    mine = per_arm_out["per_layer"][str(HEADLINE_LAYER)]["delta_leakage"]["original_covariates"]
    for j_i, stats in orig_partial["per_arm_partial"].items():
        my_beta = mine["per_arm"][j_i]["coefficients"]["cos_b_j"]["beta"]
        if abs(my_beta - stats["beta_cos"]) > 1e-6:
            raise AssertionError(
                f"per-arm reproduction mismatch for {j_i}: {my_beta} vs "
                f"original {stats['beta_cos']} ({orig_partial_path})"
            )
    log.info("[expanded] per-arm original-covariate betas reproduce per_arm_partial_ols.json")
    (out_dir / "per_arm_expanded_ols.json").write_text(json.dumps(per_arm_out, indent=2))

    # Phase D: pooled OLS (baseline vs expanded; raw + standardized; both DVs).
    pooled_out: dict[str, Any] = {"layers": list(layers), "per_layer": {}}
    for layer in layers:
        df = _attach_layer_predictors(frame["rows"], geoms[layer])
        pooled_out["per_layer"][str(layer)] = {
            dv: {
                "baseline": fit_pooled_model(df, dv=dv, predictors=BASELINE_PREDICTORS),
                "expanded": fit_pooled_model(df, dv=dv, predictors=EXPANDED_PREDICTORS),
                "expanded_standardized": fit_pooled_model(
                    df, dv=dv, predictors=EXPANDED_PREDICTORS, standardize=True
                ),
            }
            for dv in DVS
        }
    pooled_out["reproducibility"] = repro
    (out_dir / "pooled_expanded_ols.json").write_text(json.dumps(pooled_out, indent=2))

    # Phase E: compact old-vs-new headline comparison at L21.
    hl = str(HEADLINE_LAYER)
    mixed_path = original_analysis_dir / "mixed_model_pooled.json"
    mixed = json.loads(mixed_path.read_text())
    comparison = {
        "headline_layer": HEADLINE_LAYER,
        "per_arm_cos_b_j": {
            j_i: {
                "original_delta_source_dg_only": orig_partial["per_arm_partial"][j_i]["beta_cos"],
                "expanded_delta_leakage": per_arm_out["per_layer"][hl]["delta_leakage"][
                    "expanded_covariates"
                ]["per_arm"][j_i]["coefficients"]["cos_b_j"],
                "expanded_delta_trained_abs": per_arm_out["per_layer"][hl]["delta_trained_abs"][
                    "expanded_covariates"
                ]["per_arm"][j_i]["coefficients"]["cos_b_j"],
            }
            for j_i in non_default
        },
        "sign_agreement": {
            "original_delta_source_dg_only": sum(
                1 for s in orig_partial["per_arm_partial"].values() if s["beta_cos"] > 0
            ),
            "expanded_delta_leakage": per_arm_out["per_layer"][hl]["delta_leakage"][
                "expanded_covariates"
            ]["sign_agreement_cos_b_j"]["n_positive"],
            "expanded_delta_trained_abs": per_arm_out["per_layer"][hl]["delta_trained_abs"][
                "expanded_covariates"
            ]["sign_agreement_cos_b_j"]["n_positive"],
            "n_arms": len(non_default),
        },
        "pooled_cos_b_j": {
            "original_mixed_model": mixed["per_layer"][hl],
            "baseline_pooled_ols": {
                dv: pooled_out["per_layer"][hl][dv]["baseline"]["coefficients"]["cos_b_j"]
                for dv in DVS
            },
            "expanded_pooled_ols": {
                dv: pooled_out["per_layer"][hl][dv]["expanded"]["coefficients"]["cos_b_j"]
                for dv in DVS
            },
        },
        "pooled_geometry_predictors_L21": {
            dv: {
                name: pooled_out["per_layer"][hl][dv]["expanded"]["coefficients"][name]
                for name in ("cos_b_j", "shadow_angle", "d_nearest_remaining")
            }
            for dv in DVS
        },
        "reproducibility": repro,
    }
    (out_dir / "headline_comparison.json").write_text(json.dumps(comparison, indent=2))

    return {
        "out_dir": str(out_dir),
        "n_rows": frame["n_rows"],
        "comparison": comparison,
        "frame": frame,
        "geoms": geoms,
        "per_arm": per_arm_out,
        "pooled": pooled_out,
    }


# ── Figures ─────────────────────────────────────────────────────────────────

_ARM_LABELS = {
    "hero": "Drop hero",
    "wizard": "Drop wizard",
    "quilter": "Drop quilter",
    "veterinarian": "Drop veterinarian",
    "child": "Drop child",
    "ai_assistant": "Drop AI assistant",
}

_PREDICTOR_LABELS = {
    "cos_b_j": "cos(bystander, dropped negative)",
    "shadow_angle": "shadow angle (bystander; dropped negative)",
    "d_nearest_remaining": "distance to nearest remaining negative",
    "cos_b_source": "cos(bystander, source)",
    "base_prior_b": "base-model marker prior",
    "delta_source_dg": "source implant shift (drop − full)",
}

_DV_LABELS = {
    "delta_leakage": "Δ-leakage (trained − base shift)",
    "delta_trained_abs": "absolute trained log-prob shift",
}


def figure_forest_cos_bj(results: dict, fig_dir: Path, *, layer: int = HEADLINE_LAYER) -> None:
    """Forest plot of the headline cos(b, j) slope: per-arm + pooled, original vs expanded."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    colors = paper_palette(2)
    hl = str(layer)
    per_arm = results["per_arm"]["per_layer"][hl]
    pooled = results["pooled"]["per_layer"][hl]
    arms = list(_ARM_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharey=True)
    for ax, dv in zip(axes, DVS, strict=True):
        labels: list[str] = []
        y = 0
        for j_i in arms:
            for variant, color, off in (
                ("original_covariates", colors[0], 0.16),
                ("expanded_covariates", colors[1], -0.16),
            ):
                c = per_arm[dv][variant]["per_arm"][j_i]["coefficients"]["cos_b_j"]
                ax.errorbar(
                    c["beta"],
                    y + off,
                    xerr=[[c["beta"] - c["ci95_low"]], [c["ci95_high"] - c["beta"]]],
                    fmt="o" if variant == "original_covariates" else "D",
                    color=color,
                    markersize=5,
                    capsize=2.5,
                    lw=1.4,
                )
            labels.append(_ARM_LABELS[j_i])
            y += 1
        for variant, color, off in (("baseline", colors[0], 0.16), ("expanded", colors[1], -0.16)):
            c = pooled[dv][variant]["coefficients"]["cos_b_j"]
            ax.errorbar(
                c["beta"],
                y + off,
                xerr=[[c["beta"] - c["ci95_low"]], [c["ci95_high"] - c["beta"]]],
                fmt="o" if variant == "baseline" else "D",
                color=color,
                markersize=6.5,
                capsize=2.5,
                lw=1.8,
            )
        labels.append("Pooled (cluster-robust)")
        ax.axvline(0.0, color="0.55", lw=0.9, ls="--", zorder=0)
        ax.axhline(y - 0.5, color="0.8", lw=0.8, zorder=0)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("cos(b, j) slope (nats per cosine unit)")
        ax.set_title(_DV_LABELS[dv])
        ax.invert_yaxis()
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=colors[0], label="Source-shift covariate only"),
        plt.Line2D([], [], marker="D", ls="", color=colors[1], label="Expanded covariates"),
    ]
    axes[0].legend(handles=handles, loc="lower left", fontsize=9)
    fig.suptitle(
        f"Per-arm cos(b, j) slopes stay mixed-sign; the pooled slope is positive at layer {layer}",
        x=0.02,
        ha="left",
        fontweight="semibold",
    )
    fig.text(
        0.02,
        0.91,
        f"Layer {layer} cosine; per-arm OLS (HC2) + pooled OLS with arm/seed fixed effects "
        "and bystander-clustered 95% CIs; 936 rows (6 drop arms × 3 seeds × 52 bystanders)",
        ha="left",
        fontsize=9,
        color="#5A5A5A",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    savefig_paper(fig, "forest_cos_bj_old_vs_expanded", dir=fig_dir)
    plt.close(fig)


def figure_pooled_geometry(results: dict, fig_dir: Path, *, layer: int = HEADLINE_LAYER) -> None:
    """Standardized pooled coefficients for every expanded predictor, both DVs."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style("blog")
    colors = paper_palette(2)
    pooled = results["pooled"]["per_layer"][str(layer)]
    preds = list(EXPANDED_PREDICTORS)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for k, (dv, color, off) in enumerate(zip(DVS, colors, (0.16, -0.16), strict=True)):
        coefs = pooled[dv]["expanded_standardized"]["coefficients"]
        for y, name in enumerate(preds):
            c = coefs[name]
            ax.errorbar(
                c["beta"],
                y + off,
                xerr=[[c["beta"] - c["ci95_low"]], [c["ci95_high"] - c["beta"]]],
                fmt=["o", "D"][k],
                color=color,
                markersize=5.5,
                capsize=2.5,
                lw=1.5,
                label=_DV_LABELS[dv] if y == 0 else None,
            )
    ax.axvline(0.0, color="0.55", lw=0.9, ls="--", zorder=0)
    ax.set_yticks(range(len(preds)))
    ax.set_yticklabels([_PREDICTOR_LABELS[p] for p in preds])
    ax.invert_yaxis()
    ax.set_xlabel("standardized slope (nats per predictor SD), bystander-clustered 95% CI")
    ax.legend(loc="center right", fontsize=9)
    set_title_subtitle(
        ax,
        "cos(b, j) and shadow angle split the joint positive weight at layer "
        f"{layer}; nearest-remaining distance is flat",
        f"Layer {layer}; pooled OLS, all predictors entered jointly with arm/seed fixed "
        "effects; cos(b, j) and shadow angle are strongly collinear (r = -0.92)",
    )
    fig.tight_layout()
    savefig_paper(fig, "pooled_expanded_coefficients", dir=fig_dir)
    plt.close(fig)
