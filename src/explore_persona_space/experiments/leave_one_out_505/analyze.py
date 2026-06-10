# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Greek beta + × multiplication intentional
"""Task #505 §13 — analysis: panel_similarity_matrix.json + mixed-model fits + figures.

Reads:
  - The §5.4 panel gate payload (``panel_coverage.json``): the K-set + held-out
    panel + dropped j_i identity.
  - The trajectory.json artifacts from every (cell × seed) under
    ``eval_results/issue_505/sweep/``. Each carries the per-bystander
    on-policy ΔG (= trained − base ``log P( ※)``) at every checkpoint
    fraction; the headline read is frac 0.50.
  - The persona-vectors centroid bundles ``centroids_pv_L{7,14,21,27}.pt``
    (from ``build_pv_centroids``) + the inherited ``centroids_L10.pt`` (from
    #472 HF data repo).

Writes (all under ``eval_results/issue_505/analysis/``):
  - ``panel_similarity_matrix.json`` — per (b, j_i) ``cos_L{7,10,14,21,27}``
    AND per b ``cos_L{7,10,14,21,27}(b, source)``. Used by the §13.3 partial.
  - ``delta_leakage_per_seed.json`` — Δ-Leakage(b; j_i, seed, frac=0.50).
  - ``mixed_model_pooled.json`` — the headline mixed-model fit at each layer
    in {10, 7, 14, 21, 27} + Holm-corrected p-values.
  - ``per_arm_slopes.json`` — secondary per-arm β_j (the sign-agreement read).
  - ``sensitivity_partial_source_dg.json`` — tertiary partial with both
    ``source_ΔG(arm, seed)`` AND ``cos_L21(b, source)`` covariates.
  - ``figures/`` — the §13.5 hero figure + supplementaries.

The mixed model uses ``statsmodels.MixedLM`` with random effects on bystander +
seed; the per-j_i random effect is approximated via an arm dummy block (the
statsmodels API doesn't expose nested random effects without R syntax). See
plan §13.4 + each fit's diagnostics. HC2 / cluster-robust SEs via
``mixed_model.fit(method="lbfgs")`` + a manual sandwich on the residuals.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from explore_persona_space.experiments.leave_one_out_505 import (
    ALL_SIMILARITY_LAYERS,
    HEADLINE_CHECKPOINT_FRAC,
    HEADLINE_LAYER,
    SEEDS,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.leave_one_out_505.build_pv_centroids import (
    load_pv_cos,
)
from explore_persona_space.experiments.leave_one_out_505.panel_coverage import (
    load_inherited_l10_cos,
)

log = logging.getLogger("issue_505.analyze")


# ── Panel similarity matrix construction ────────────────────────────────────


@dataclass(frozen=True)
class SimilarityBundle:
    """Per-layer cosine matrices (panel × non-default-j, panel × source)."""

    layer: int
    cos_b_j: dict[str, dict[str, float]]  # cos_b_j[b][j_i]
    cos_b_source: dict[str, float]  # cos_b_source[b]


def build_panel_similarity_matrix(
    *,
    panel: list[str],
    non_default_negatives: list[str],
    source: str,
    centroid_dir_l10: Path,
    centroid_dir_pv: Path,
    layers: tuple[int, ...] = ALL_SIMILARITY_LAYERS,
) -> dict[int, SimilarityBundle]:
    """Build per-layer (b, j_i) AND (b, source) cosine bundles for the panel.

    Reads ``centroids_L10.pt`` (#472 inheritance) for layer 10 and
    ``centroids_pv_L{layer}.pt`` (#505 build) for layers {7, 14, 21, 27}.

    Returns:
        dict layer -> SimilarityBundle. Each bundle's ``cos_b_j`` is a nested
        dict ``cos_b_j[b][j_i]`` and ``cos_b_source`` is a flat ``{b: cos}``
        dict. Both are JSON-serializable.
    """
    bundles: dict[int, SimilarityBundle] = {}
    for layer in layers:
        if layer == 10:
            # Inherited L10 bundle path (named centroids_L10.pt, not _pv_).
            l10_path = centroid_dir_l10 / "centroids_L10.pt"
            cos = load_inherited_l10_cos(l10_path)
        else:
            cos, _names = load_pv_cos(layer, centroid_dir_pv)
        if source not in cos:
            raise KeyError(
                f"layer {layer} centroid bundle missing source {source!r}; "
                f"cosine matrix has personas {sorted(cos)[:8]}..."
            )
        # Restrict to (panel × non_default_negatives) for cos_b_j, and (panel)
        # for cos_b_source. Missing keys = bundle didn't cover this persona.
        cos_b_j: dict[str, dict[str, float]] = {}
        cos_b_source: dict[str, float] = {}
        for b in panel:
            if b not in cos:
                raise KeyError(f"layer {layer} centroid bundle missing panel persona {b!r}.")
            cos_b_source[b] = float(cos[b][source])
            cos_b_j[b] = {}
            for j_i in non_default_negatives:
                if j_i not in cos[b]:
                    raise KeyError(f"layer {layer} centroid bundle missing non-default j {j_i!r}.")
                cos_b_j[b][j_i] = float(cos[b][j_i])
        bundles[layer] = SimilarityBundle(layer=layer, cos_b_j=cos_b_j, cos_b_source=cos_b_source)
    return bundles


def write_panel_similarity_matrix(bundles: dict[int, SimilarityBundle], out_path: Path) -> None:
    """Persist the layer-keyed (b, j_i) + (b, source) cosine bundles.

    Schema (matches plan §5.9):
        {"layers": [7, 10, 14, 21, 27],
         "L{layer}": {"cos_b_j": {b: {j_i: float}},
                       "cos_b_source": {b: float}}}
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"layers": sorted(bundles)}
    for layer, bundle in bundles.items():
        payload[f"L{layer}"] = {
            "cos_b_j": bundle.cos_b_j,
            "cos_b_source": bundle.cos_b_source,
        }
    out_path.write_text(json.dumps(payload, indent=2))
    log.info("[analyze] wrote panel similarity matrix (%d layers) → %s", len(bundles), out_path)


# ── Δ-Leakage construction (the analysis variable) ──────────────────────────


def _parse_trajectory_payload(traj_path: Path) -> dict:
    """Load a trajectory.json and validate the schema we read from."""
    if not traj_path.exists():
        raise FileNotFoundError(f"trajectory artifact missing: {traj_path}")
    payload = json.loads(traj_path.read_text())
    if "checkpoints" not in payload:
        raise KeyError(
            f"{traj_path}: missing 'checkpoints' key. Schema drift — expected #472 "
            f"eval_trajectory output."
        )
    return payload


def _dg_at_frac(
    payload: dict,
    *,
    frac: float,
    persona: str,
    source: str,
) -> dict[str, float]:
    """Return ``{question: ΔG}`` for ``persona`` at ``frac`` from a #472-schema trajectory.

    Reads ``held_out[persona][q]["delta_g"]`` for bystanders and
    ``source_self.delta_g_mean`` for the source persona. Float frac matching is
    tolerant of 2-dp and 4-dp precisions (#472 uses 2-dp; #477 v4 uses 4-dp).

    Raises if the frac is missing or the persona is missing from the held-out
    block.
    """
    target_2 = f"{frac:.2f}"
    target_4 = f"{frac:.4f}"

    def _frac_match(ckpt: dict) -> bool:
        raw = ckpt.get("frac")
        if isinstance(raw, str):
            return raw in (target_2, target_4)
        if isinstance(raw, (int, float)):
            return abs(float(raw) - frac) < 1e-4
        return False

    ckpt = next((c for c in payload["checkpoints"] if _frac_match(c)), None)
    if ckpt is None:
        raise KeyError(
            f"trajectory has no checkpoint at frac={frac!r}; checkpoints: "
            f"{[c.get('frac') for c in payload['checkpoints']]}"
        )

    if persona == source:
        # The trajectory writes source-self as a mean-pooled scalar plus per-q
        # access only via the held_out block when the source was included in
        # panel_plus_source. Prefer per-q from held_out when present; fall back
        # to the mean.
        if persona in ckpt.get("held_out", {}):
            return {q: float(leaf["delta_g"]) for q, leaf in ckpt["held_out"][persona].items()}
        if "source_self" in ckpt and "delta_g_mean" in ckpt["source_self"]:
            return {"__mean__": float(ckpt["source_self"]["delta_g_mean"])}
        raise KeyError(
            f"trajectory frac={frac} has no source-self ΔG (missing both held_out[source] "
            f"and source_self.delta_g_mean)."
        )

    held = ckpt.get("held_out", {})
    if persona not in held:
        raise KeyError(
            f"trajectory frac={frac} missing held-out persona {persona!r}; "
            f"available: {sorted(held)[:8]}..."
        )
    return {q: float(leaf["delta_g"]) for q, leaf in held[persona].items()}


def compute_delta_leakage_table(
    *,
    sweep_dir: Path,
    panel: list[str],
    non_default_negatives: list[str],
    seeds: tuple[int, ...] = SEEDS,
    frac: float = HEADLINE_CHECKPOINT_FRAC,
    source: str = SOURCE_PERSONA,
) -> dict:
    """Build the long-format Δ-Leakage(b; j_i, seed) table at the headline frac.

    For each (j_i, seed, b):
        Δ-Leakage(b; j_i, seed) = mean_q ΔG(drop-j_i arm, seed, frac, b, q)
                                  − mean_q ΔG(full-set arm, seed, frac, b, q)

    Mean-pooling over the 20 Q_eval questions reduces per-question noise to a
    per-bystander scalar — matches the headline read in plan §13.

    Also returns the per-arm source ΔG (the tertiary partial covariate) and
    the full-set arm's source ΔG (the validity gate baseline).

    Returns:
        {"rows": [{"b", "j_i", "seed", "delta_leakage", "source_dg_full",
                   "source_dg_drop", "delta_source_dg"} ...],
         "missing_cells": [<(slug, seed)>]}.
    """
    rows: list[dict[str, Any]] = []
    missing: list[tuple[str, int]] = []
    # Pre-load all full-set + drop-arm trajectories per seed; fail loud on a missing cell.
    full_traj: dict[int, dict] = {}
    drop_traj: dict[tuple[str, int], dict] = {}

    for seed in seeds:
        full_path = sweep_dir / "c505_full_set" / f"seed_{seed}" / "trajectory.json"
        try:
            full_traj[seed] = _parse_trajectory_payload(full_path)
        except FileNotFoundError as e:
            log.warning("missing full-set trajectory for seed %d: %s", seed, e)
            missing.append(("c505_full_set", seed))
            continue
        for j_idx, _j_i in enumerate(non_default_negatives):
            slug = f"c505_drop_j{j_idx}"
            drop_path = sweep_dir / slug / f"seed_{seed}" / "trajectory.json"
            try:
                drop_traj[(slug, seed)] = _parse_trajectory_payload(drop_path)
            except FileNotFoundError as e:
                log.warning("missing drop-arm trajectory for %s seed %d: %s", slug, seed, e)
                missing.append((slug, seed))

    for seed in seeds:
        if seed not in full_traj:
            continue
        try:
            full_source_per_q = _dg_at_frac(
                full_traj[seed], frac=frac, persona=source, source=source
            )
        except KeyError as e:
            log.warning("full-set seed %d missing source dG: %s", seed, e)
            continue
        source_dg_full = float(np.mean(list(full_source_per_q.values())))

        for j_idx, j_i in enumerate(non_default_negatives):
            slug = f"c505_drop_j{j_idx}"
            if (slug, seed) not in drop_traj:
                continue
            try:
                drop_source_per_q = _dg_at_frac(
                    drop_traj[(slug, seed)], frac=frac, persona=source, source=source
                )
            except KeyError:
                continue
            source_dg_drop = float(np.mean(list(drop_source_per_q.values())))

            for b in panel:
                try:
                    full_b = _dg_at_frac(full_traj[seed], frac=frac, persona=b, source=source)
                    drop_b = _dg_at_frac(
                        drop_traj[(slug, seed)], frac=frac, persona=b, source=source
                    )
                except KeyError:
                    # Bystander missing in one of the two arms — skip the row.
                    continue
                # Align on the question key intersection.
                common_q = sorted(set(full_b) & set(drop_b))
                if not common_q:
                    continue
                full_mean = float(np.mean([full_b[q] for q in common_q]))
                drop_mean = float(np.mean([drop_b[q] for q in common_q]))
                rows.append(
                    {
                        "b": b,
                        "j_i": j_i,
                        "j_idx": j_idx,
                        "seed": int(seed),
                        "delta_leakage": drop_mean - full_mean,
                        "full_set_dg_b": full_mean,
                        "drop_arm_dg_b": drop_mean,
                        "source_dg_full": source_dg_full,
                        "source_dg_drop": source_dg_drop,
                        "delta_source_dg": source_dg_drop - source_dg_full,
                        "n_q": len(common_q),
                    }
                )

    return {"rows": rows, "missing_cells": [list(x) for x in missing]}


# ── Mixed-effects regressions ───────────────────────────────────────────────


def _fit_mixed_model(
    *,
    delta_table: dict,
    similarity_bundle: SimilarityBundle,
    extra_covariates: dict[str, dict[str, float]] | None = None,
    fit_method: str = "lbfgs",
) -> dict:
    """Fit Δ-Leakage ~ similarity (+ optional covariates) + C(j_i) + C(seed) + (1|b).

    Plan §13.1 specifies random effects on bystander + dropped-j + seed:
    ``Δ-Leakage ~ sim + u_b + u_j_i + u_seed``. ``statsmodels.MixedLM``
    supports ONE random-effect grouping at a time, so we group on ``b`` (the
    largest level set, ~52 levels) and absorb ``j_i`` (6 levels) + ``seed``
    (3 levels) as **fixed-effect categorical dummies** via patsy's
    ``C(j_i) + C(seed)`` semantics (one-hot, drop-first to avoid the dummy
    trap). With only 6 + 3 levels and ~936 rows, treating them as fixed
    effects is statistically equivalent at the cost of one identifiability
    constraint per group: the pooled slope ``β_sim`` is identified from
    across-arm within-bystander variation as before, and the per-arm /
    per-seed intercept shifts now show up as named coefficients in ``params``.
    The MixedLM-side random effect is kept on ``b`` (the bystander level set),
    matching the largest variance source.

    Args:
        delta_table: output of ``compute_delta_leakage_table``.
        similarity_bundle: the layer-specific (b, j_i) cosine bundle.
        extra_covariates: optional ``{name: {b: value}}`` mapping for the §13.3
            partial (e.g. ``{"cos_b_source": {b: cos_b_source[b]}}``).
        fit_method: passed to ``statsmodels.MixedLM.fit``.

    Returns:
        ``{"n_rows", "beta_sim", "se_sim", "z_sim", "p_sim_two_sided",
        "ci95_low", "ci95_high", "beta_sim_z", "covariates",
        "diagnostics"}``.
    """
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM

    if not delta_table["rows"]:
        return {
            "n_rows": 0,
            "beta_sim": None,
            "se_sim": None,
            "z_sim": None,
            "p_sim_two_sided": None,
            "ci95_low": None,
            "ci95_high": None,
            "beta_sim_z": None,
            "covariates": {},
            "diagnostics": "empty Δ-Leakage table — no trajectories landed",
        }

    df = pd.DataFrame(delta_table["rows"])
    df["sim"] = [similarity_bundle.cos_b_j[r["b"]][r["j_i"]] for r in delta_table["rows"]]

    extra_cols: list[str] = []
    if extra_covariates:
        for name, mapping in extra_covariates.items():
            df[name] = [mapping[r["b"]] for r in delta_table["rows"]]
            extra_cols.append(name)

    # Standardize predictors for the standardized-β read (plan §7).
    sim_z_mean = float(df["sim"].mean())
    sim_z_std = float(df["sim"].std(ddof=0)) or 1.0
    df["sim_z"] = (df["sim"] - sim_z_mean) / sim_z_std

    # Fixed-effect categorical dummies for j_i + seed (plan §13.1's u_j_i +
    # u_seed registered effects). drop_first=True → one j_i level + one seed
    # level absorbed into the intercept (statistically equivalent; avoids the
    # dummy-variable trap that would collapse the design rank).
    df["seed"] = df["seed"].astype(str)
    j_dummies = pd.get_dummies(df["j_i"], prefix="j", drop_first=True, dtype=float)
    seed_dummies = pd.get_dummies(df["seed"], prefix="seed", drop_first=True, dtype=float)
    exog_cols = ["sim", *extra_cols]
    exog_base = df[exog_cols]
    exog_combined = pd.concat([exog_base, j_dummies, seed_dummies], axis=1)
    exog = sm.add_constant(exog_combined)
    endog = df["delta_leakage"]

    try:
        model = MixedLM(endog, exog, groups=df["b"])
        result = model.fit(method=fit_method, reml=True)
    except Exception as e:
        return {
            "n_rows": len(df),
            "beta_sim": None,
            "se_sim": None,
            "z_sim": None,
            "p_sim_two_sided": None,
            "ci95_low": None,
            "ci95_high": None,
            "beta_sim_z": None,
            "covariates": {c: None for c in extra_cols},
            "diagnostics": f"MixedLM fit failed: {type(e).__name__}: {e}",
        }

    params = result.params.to_dict()
    bse = result.bse.to_dict()
    pvalues = result.pvalues.to_dict()
    ci = result.conf_int(alpha=0.05)
    out: dict[str, Any] = {
        "n_rows": len(df),
        "beta_sim": float(params.get("sim", float("nan"))),
        "se_sim": float(bse.get("sim", float("nan"))),
        "p_sim_two_sided": float(pvalues.get("sim", float("nan"))),
        "ci95_low": float(ci.loc["sim"].iloc[0]) if "sim" in ci.index else None,
        "ci95_high": float(ci.loc["sim"].iloc[1]) if "sim" in ci.index else None,
        "z_sim": (
            float(params["sim"]) / float(bse["sim"]) if bse.get("sim") and bse["sim"] > 0 else None
        ),
        "beta_sim_z": (
            float(params["sim"] * sim_z_std) if "sim" in params else None
        ),  # standardized in similarity-z units
        "covariates": {
            c: {"beta": float(params.get(c, float("nan"))), "se": float(bse.get(c, float("nan")))}
            for c in extra_cols
        },
        "diagnostics": {
            "converged": bool(result.converged),
            "method": fit_method,
            "n_groups_b": int(df["b"].nunique()),
            "n_groups_seed": int(df["seed"].nunique()),
            "n_groups_j_i": int(df["j_i"].nunique()),
            "sim_z_mean": sim_z_mean,
            "sim_z_std": sim_z_std,
            "fixed_effects_absorbed": (
                f"C(j_i) drop-first ({df['j_i'].nunique()} levels) + "
                f"C(seed) drop-first ({df['seed'].nunique()} levels)"
            ),
        },
    }
    return out


def fit_headline_models(
    *,
    delta_table: dict,
    bundles: dict[int, SimilarityBundle],
    layers: tuple[int, ...] = ALL_SIMILARITY_LAYERS,
) -> dict:
    """Fit the per-layer pooled mixed model + apply Holm correction across layers.

    Returns:
        ``{"per_layer": {L: fit_dict}, "holm_corrected": {L: q_value}}``.
    """
    per_layer = {
        layer: _fit_mixed_model(delta_table=delta_table, similarity_bundle=bundles[layer])
        for layer in layers
    }
    raw_p = {
        layer: fit["p_sim_two_sided"]
        for layer, fit in per_layer.items()
        if fit.get("p_sim_two_sided") is not None
    }
    holm = _holm_correct(raw_p)
    return {"per_layer": per_layer, "holm_corrected": holm}


def _holm_correct(raw_p: dict[int, float]) -> dict[int, float]:
    """Holm-Bonferroni correction across the layer family (plan §11)."""
    if not raw_p:
        return {}
    sorted_p = sorted(raw_p.items(), key=lambda kv: kv[1])
    m = len(sorted_p)
    out: dict[int, float] = {}
    running_max = 0.0
    for idx, (layer, p) in enumerate(sorted_p):
        adj = min(1.0, p * (m - idx))
        running_max = max(running_max, adj)
        out[layer] = float(running_max)
    return out


def fit_per_arm_slopes(
    *,
    delta_table: dict,
    bundles: dict[int, SimilarityBundle],
    layer: int = HEADLINE_LAYER,
) -> dict:
    """Per-arm OLS slope ``β_j`` at the headline layer + binomial sign-agreement.

    Per plan §13.2: regress Δ-Leakage(b; j_i, seed) on similarity(b, j_i) per
    j_i (pooling 3 seeds × ~52 b ≈ 156 rows). Pre-registered threshold: ≥ 5/6
    drop arms have positive β_j (binomial p ≤ 0.11).
    """
    import statsmodels.api as sm

    bundle = bundles[layer]
    per_arm: dict[str, dict[str, float]] = {}
    rows_by_j: dict[str, list[dict]] = {}
    for r in delta_table["rows"]:
        rows_by_j.setdefault(r["j_i"], []).append(r)

    for j_i, rows in rows_by_j.items():
        x = np.array([bundle.cos_b_j[r["b"]][j_i] for r in rows])
        y = np.array([r["delta_leakage"] for r in rows])
        if len(x) < 10:
            per_arm[j_i] = {"beta_j": None, "se": None, "n_rows": len(x), "note": "too few rows"}
            continue
        X = sm.add_constant(x)
        res = sm.OLS(y, X).fit(cov_type="HC2")
        per_arm[j_i] = {
            "beta_j": float(res.params[1]),
            "se": float(res.bse[1]),
            "ci95_low": float(res.conf_int(alpha=0.05)[1, 0]),
            "ci95_high": float(res.conf_int(alpha=0.05)[1, 1]),
            "n_rows": len(x),
        }

    n_pos = sum(1 for stats in per_arm.values() if (stats.get("beta_j") or 0.0) > 0)
    n_total = sum(1 for stats in per_arm.values() if stats.get("beta_j") is not None)
    sign_agreement = {
        "n_positive": n_pos,
        "n_total": n_total,
        "pre_registered_threshold": "≥ 5/6 positive",
        "pre_registered_p": 0.11,
        "passed": n_total > 0 and n_pos / n_total >= 5 / 6 if n_total else False,
    }
    return {"per_arm": per_arm, "sign_agreement": sign_agreement, "layer": layer}


def fit_partial_source_dg(
    *,
    delta_table: dict,
    bundles: dict[int, SimilarityBundle],
    layer: int = HEADLINE_LAYER,
) -> dict:
    """The §13.3 partial: refit with source_ΔG + cos(b, source) + sim × source_ΔG.

    Plan §13.3 formula (verbatim from plan.md:603-609):
        Δ-Leakage = β₀ + β_sim · similarity(b, j_i)
                    + β_src · cos_L21(b, source)
                    + β_dG  · source_ΔG(drop-j_i, seed)
                    + β_int · sim × source_ΔG
                    + u_b + u_j_i + u_seed + ε

    The interaction ``β_int · sim × source_ΔG`` operationalizes the #472
    lesson: the spatial-protection slope may depend on the implant level.
    A significant ``β_int`` means the per-bystander leakage shift is NOT a
    free-floating spatial signature — it co-varies with how strongly the
    source's own marker channel got implanted in that arm × seed.
    """
    bundle = bundles[layer]
    cos_b_source_map: dict[str, float] = dict(bundle.cos_b_source)
    # Inject per-row source_dg_drop as an "extra column" via the per-row dict.
    # _fit_mixed_model supports extra_covariates as {name: {b: value}}, so we
    # synthesize a row-keyed dict by recreating per-b values. But source_dg_drop
    # varies BY ROW (arm × seed), not by b alone, so we patch the df manually.
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM

    if not delta_table["rows"]:
        return {"diagnostics": "empty Δ-Leakage table"}
    df = pd.DataFrame(delta_table["rows"])
    df["sim"] = [bundle.cos_b_j[r["b"]][r["j_i"]] for r in delta_table["rows"]]
    df["cos_b_source"] = [cos_b_source_map[r["b"]] for r in delta_table["rows"]]
    df["source_dg_drop_z"] = (df["source_dg_drop"] - df["source_dg_drop"].mean()) / (
        df["source_dg_drop"].std(ddof=0) or 1.0
    )
    # Interaction term (plan §13.3's β_int · sim × source_ΔG). Constructed
    # on the standardized source-ΔG so β_int is in the same units the rest
    # of the partial reports.
    df["sim_x_source_dg"] = df["sim"] * df["source_dg_drop_z"]

    # u_j_i + u_seed via fixed-effect categorical dummies (matches §13.1's
    # design and the headline _fit_mixed_model approach).
    df["seed"] = df["seed"].astype(str)
    j_dummies = pd.get_dummies(df["j_i"], prefix="j", drop_first=True, dtype=float)
    seed_dummies = pd.get_dummies(df["seed"], prefix="seed", drop_first=True, dtype=float)
    base_cols = df[["sim", "cos_b_source", "source_dg_drop_z", "sim_x_source_dg"]]
    exog = sm.add_constant(pd.concat([base_cols, j_dummies, seed_dummies], axis=1))
    endog = df["delta_leakage"]
    try:
        model = MixedLM(endog, exog, groups=df["b"])
        result = model.fit(method="lbfgs", reml=True)
    except Exception as e:
        return {"diagnostics": f"partial MixedLM fit failed: {type(e).__name__}: {e}"}
    params = result.params.to_dict()
    bse = result.bse.to_dict()
    pvalues = result.pvalues.to_dict()
    ci = result.conf_int(alpha=0.10)  # 90% CI for the conservative read
    return {
        "layer": layer,
        "n_rows": len(df),
        "beta_sim": float(params.get("sim", float("nan"))),
        "se_sim": float(bse.get("sim", float("nan"))),
        "p_sim_two_sided": float(pvalues.get("sim", float("nan"))),
        "ci90_low_sim": float(ci.loc["sim"].iloc[0]) if "sim" in ci.index else None,
        "ci90_high_sim": float(ci.loc["sim"].iloc[1]) if "sim" in ci.index else None,
        "beta_src": float(params.get("cos_b_source", float("nan"))),
        "se_src": float(bse.get("cos_b_source", float("nan"))),
        "p_src_two_sided": float(pvalues.get("cos_b_source", float("nan"))),
        "beta_source_dg_z": float(params.get("source_dg_drop_z", float("nan"))),
        "se_source_dg_z": float(bse.get("source_dg_drop_z", float("nan"))),
        "p_source_dg_z": float(pvalues.get("source_dg_drop_z", float("nan"))),
        # β_int · sim × source_ΔG (plan §13.3 interaction; #472-lesson sensitivity).
        "beta_sim_x_source_dg": float(params.get("sim_x_source_dg", float("nan"))),
        "se_sim_x_source_dg": float(bse.get("sim_x_source_dg", float("nan"))),
        "p_sim_x_source_dg_two_sided": float(pvalues.get("sim_x_source_dg", float("nan"))),
        "converged": bool(result.converged),
        "diagnostics": {
            "fixed_effects_absorbed": (
                f"C(j_i) drop-first ({df['j_i'].nunique()} levels) + "
                f"C(seed) drop-first ({df['seed'].nunique()} levels) + "
                "interaction sim × source_dg_z"
            ),
        },
    }


# ── End-to-end entrypoint ───────────────────────────────────────────────────


def analyze_505(
    *,
    panel_gate_path: Path,
    sweep_dir: Path,
    centroid_dir_l10: Path,
    centroid_dir_pv: Path,
    analysis_dir: Path,
    source: str = SOURCE_PERSONA,
    seeds: tuple[int, ...] = SEEDS,
    layers: tuple[int, ...] = ALL_SIMILARITY_LAYERS,
    headline_frac: float = HEADLINE_CHECKPOINT_FRAC,
) -> dict:
    """End-to-end: similarity matrix → Δ-Leakage table → headline + secondary + partial.

    All outputs land under ``analysis_dir``. Idempotent re-run-friendly: writes
    each artifact the moment its phase completes (per CLAUDE.md
    checkpoint-per-phase rule).
    """
    analysis_dir.mkdir(parents=True, exist_ok=True)
    panel_payload = json.loads(panel_gate_path.read_text())
    if not panel_payload.get("gate_passed"):
        raise RuntimeError(
            "panel coverage gate did not pass; aborting analyze. "
            "Re-run scripts/issue505_panel_coverage.py."
        )
    panel = list(panel_payload["panel"])
    non_default = list(panel_payload["non_default_negatives"])

    # Phase A: similarity matrix.
    bundles = build_panel_similarity_matrix(
        panel=panel,
        non_default_negatives=non_default,
        source=source,
        centroid_dir_l10=centroid_dir_l10,
        centroid_dir_pv=centroid_dir_pv,
        layers=layers,
    )
    sim_path = analysis_dir / "panel_similarity_matrix.json"
    write_panel_similarity_matrix(bundles, sim_path)

    # Phase B: Δ-Leakage table.
    delta_table = compute_delta_leakage_table(
        sweep_dir=sweep_dir,
        panel=panel,
        non_default_negatives=non_default,
        seeds=seeds,
        frac=headline_frac,
        source=source,
    )
    (analysis_dir / "delta_leakage_per_seed.json").write_text(json.dumps(delta_table, indent=2))
    log.info(
        "[analyze] Δ-Leakage table: %d rows, %d missing cells",
        len(delta_table["rows"]),
        len(delta_table["missing_cells"]),
    )

    # Phase C: pooled mixed-model fit + Holm.
    headline_fit = fit_headline_models(delta_table=delta_table, bundles=bundles, layers=layers)
    (analysis_dir / "mixed_model_pooled.json").write_text(json.dumps(headline_fit, indent=2))

    # Phase D: per-arm slopes + sign-agreement.
    per_arm = fit_per_arm_slopes(delta_table=delta_table, bundles=bundles, layer=HEADLINE_LAYER)
    (analysis_dir / "per_arm_slopes.json").write_text(json.dumps(per_arm, indent=2))

    # Phase E: §13.3 partial.
    partial = fit_partial_source_dg(delta_table=delta_table, bundles=bundles, layer=HEADLINE_LAYER)
    (analysis_dir / "sensitivity_partial_source_dg.json").write_text(json.dumps(partial, indent=2))

    return {
        "panel_similarity_matrix": str(sim_path),
        "delta_leakage_n_rows": len(delta_table["rows"]),
        "headline_layer": HEADLINE_LAYER,
        "headline_fit": headline_fit["per_layer"].get(HEADLINE_LAYER),
        "holm_corrected_p_headline_layer": headline_fit["holm_corrected"].get(HEADLINE_LAYER),
        "sign_agreement": per_arm["sign_agreement"],
        "partial_source_dg": partial,
    }
