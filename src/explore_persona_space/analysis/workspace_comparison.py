"""Paired comparisons of verified J/R fits, with explicit cohorts and control gaps."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from explore_persona_space.analysis.workspace_components import _bootstrap_summary

ROTATIONS = (None, 20260913, 20260914, 20260915)
TARGETS = ("full", "J", "restJ", "R", "restR")
PREDICTORS = ("ridge", "mlp", "mlp_seed42", "mlp_seed137", "mlp_seed271")


def cell_key(role, kind, k, rotation):
    """Give observational and affine-null cells distinct, stable identities."""
    if role not in ("primary", "comparison") or kind not in ("observed", "affine_null"):
        raise ValueError("Unknown role or experiment kind")
    if k not in (5, 10, 25) or rotation not in ROTATIONS:
        raise ValueError("Unregistered sparsity or rotation")
    return f"{role}/{kind}/k{k}/rotation{rotation}"


def paired_cohort(cells: Mapping[str, list[str]], *, eligible_ids=None) -> tuple[list[str], dict]:
    """Intersect explicitly and deterministically; retain every excluded ID."""
    if not cells:
        raise ValueError("No cells to pair")
    for name, ids in cells.items():
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid or duplicate context IDs: {name}")
    common = set.intersection(*(set(ids) for ids in cells.values()))
    if eligible_ids is not None:
        if len(eligible_ids) != len(set(eligible_ids)):
            raise ValueError("Duplicate eligible completion IDs")
        common &= set(eligible_ids)
    common = sorted(common)
    if len(common) < 2:
        raise ValueError("Fewer than two shared contexts; paired comparison is undefined")
    return common, {
        "policy": "explicit_sorted_common_context_intersection",
        "common_context_ids": common,
        "common_contexts": len(common),
        "completion_eligible_context_ids": eligible_ids,
        "cells": {
            name: {
                "original_contexts": len(ids),
                "excluded_from_pairing": sorted(set(ids) - set(common)),
            }
            for name, ids in cells.items()
        },
    }


def align_arrays(ids, common, targets, predictions):
    """Align all targets and predictors together; never infer pairing from row count."""
    if len(ids) != len(set(ids)) or len(common) != len(set(common)):
        raise ValueError("Duplicate contexts prevent paired alignment")
    lookup = {context: index for index, context in enumerate(ids)}
    if not set(common) <= set(lookup):
        raise ValueError("A requested paired context is missing")
    indices = [lookup[context] for context in common]
    aligned_targets = {}
    for name, value in targets.items():
        if value.ndim != 2 or len(value) != len(ids) or not np.isfinite(value).all():
            raise ValueError("Invalid target array for paired alignment")
        aligned_targets[name] = value[indices]
    aligned_predictions = {}
    for predictor, by_target in predictions.items():
        if set(by_target) != set(targets):
            raise ValueError("Every predictor must cover all targets")
        aligned_predictions[predictor] = {}
        for name, value in by_target.items():
            if value.shape != targets[name].shape or not np.isfinite(value).all():
                raise ValueError("Invalid prediction array for paired alignment")
            aligned_predictions[predictor][name] = value[indices]
    return aligned_targets, aligned_predictions


def gap_terms(cell, predictor, arm):
    """Use each component's own R² denominator before forming the gap."""
    return {f"{cell}/{predictor}/rest{arm}": 1.0, f"{cell}/{predictor}/{arm}": -1.0}


def linear_terms(*weighted_terms):
    """Combine contrasts algebraically, removing exactly cancelling terms."""
    result = {}
    for weight, terms in weighted_terms:
        for key, value in terms.items():
            result[key] = result.get(key, 0.0) + weight * value
    return {key: value for key, value in result.items() if value != 0}


def registered_contrasts(cells):
    """Specify all estimands without consulting their measured values."""
    available = set(cells)
    contrasts, missing = {}, []

    def add(name, coefficients, required):
        absent = sorted(set(required) - available)
        if absent:
            missing.append({"contrast": name, "missing_cells": absent})
        else:
            contrasts[name] = coefficients

    _within_model_contrasts(add)
    _cross_model_contrasts(add)
    return contrasts, missing


def _within_model_contrasts(add):
    """Native, rotated and affine-null within-model comparisons."""
    for role in ("primary", "comparison"):
        for kind in ("observed", "affine_null"):
            for k in (5, 10, 25):
                native = cell_key(role, kind, k, None)
                rotations = [cell_key(role, kind, k, seed) for seed in ROTATIONS[1:]]
                for cell in (native, *rotations):
                    for predictor in PREDICTORS:
                        for arm in ("J", "R"):
                            add(
                                f"{cell}/{predictor}/G_{arm}",
                                gap_terms(cell, predictor, arm),
                                [cell],
                            )
                        add(
                            f"{cell}/{predictor}/G_R_minus_G_J",
                            linear_terms(
                                (1, gap_terms(cell, predictor, "R")),
                                (-1, gap_terms(cell, predictor, "J")),
                            ),
                            [cell],
                        )
                    for target in TARGETS:
                        add(
                            f"{cell}/MLP_gain/{target}",
                            {f"{cell}/mlp/{target}": 1, f"{cell}/ridge/{target}": -1},
                            [cell],
                        )
                for predictor in PREDICTORS:
                    for arm in ("J", "R"):
                        add(
                            f"{native}/{predictor}/G_{arm}_minus_mean_rotated",
                            linear_terms(
                                (1, gap_terms(native, predictor, arm)),
                                *((-1 / 3, gap_terms(cell, predictor, arm)) for cell in rotations),
                            ),
                            [native, *rotations],
                        )
        for k in (5, 10, 25):
            for rotation in ROTATIONS:
                observed = cell_key(role, "observed", k, rotation)
                null = cell_key(role, "affine_null", k, rotation)
                for predictor in PREDICTORS:
                    for arm in ("J", "R"):
                        add(
                            f"{observed}/{predictor}/G_{arm}_minus_affine_null",
                            linear_terms(
                                (1, gap_terms(observed, predictor, arm)),
                                (-1, gap_terms(null, predictor, arm)),
                            ),
                            [observed, null],
                        )


def _cross_model_contrasts(add):
    """Capability-paired descriptive differences for shared test contexts."""
    for kind in ("observed", "affine_null"):
        for k in (5, 10, 25):
            for rotation in ROTATIONS:
                primary = cell_key("primary", kind, k, rotation)
                comparison = cell_key("comparison", kind, k, rotation)
                for predictor in PREDICTORS:
                    base = f"cross_model/{kind}/k{k}/rotation{rotation}/{predictor}"
                    differences = {}
                    for arm in ("J", "R"):
                        differences[arm] = linear_terms(
                            (1, gap_terms(primary, predictor, arm)),
                            (-1, gap_terms(comparison, predictor, arm)),
                        )
                        add(
                            f"{base}/G_{arm}_primary_minus_comparison",
                            differences[arm],
                            [primary, comparison],
                        )
                    add(
                        f"{base}/lens_disagreement_primary_minus_comparison",
                        linear_terms((1, differences["R"]), (-1, differences["J"])),
                        [primary, comparison],
                    )
            for predictor in PREDICTORS:
                for arm in ("J", "R"):
                    required, terms = [], []
                    for role, sign in (("primary", 1), ("comparison", -1)):
                        for rotation in ROTATIONS:
                            cell = cell_key(role, kind, k, rotation)
                            required.append(cell)
                            weight = sign if rotation is None else -sign / 3
                            terms.append((weight, gap_terms(cell, predictor, arm)))
                    add(
                        f"cross_model/{kind}/k{k}/{predictor}/G_{arm}_control_adjusted_primary_minus_comparison",
                        linear_terms(*terms),
                        required,
                    )


def quality_matched_contrasts(role, report, available):
    """Use only calibration-selected approximate matches; retain all mismatches."""
    contrasts, ledger = {}, {}
    for arm in ("J", "R"):
        for k in (5, 10, 25):
            native = cell_key(role, "observed", k, None)
            rows = report["matching"][arm][str(k)]
            if set(rows) != {str(seed) for seed in ROTATIONS[1:]}:
                raise ValueError("Quality matching lacks exactly the three registered rotations")
            matched = all(row["status"] == "approximate_within_range" for row in rows.values())
            controls = (
                [
                    cell_key(role, "observed", rows[str(seed)]["selected_k"], seed)
                    for seed in ROTATIONS[1:]
                ]
                if matched
                else []
            )
            missing = sorted(set([native, *controls]) - set(available))
            ledger[f"{role}/k{k}/{arm}"] = {
                "status": "included_approximate" if matched and not missing else "excluded",
                "calibration_matches": rows,
                "missing_cells": missing,
                "scope": "supplementary calibration-matched comparison; same-k remains primary",
            }
            if matched and not missing:
                for predictor in PREDICTORS:
                    contrasts[f"{native}/{predictor}/G_{arm}_minus_calibration_matched_rotated"] = (
                        linear_terms(
                            (1, gap_terms(native, predictor, arm)),
                            *((-1 / 3, gap_terms(cell, predictor, arm)) for cell in controls),
                        )
                    )
    return contrasts, ledger


def combine_paired(bootstraps, contrasts, *, confidence=0.95):
    """Subtract paired replicates, never marginal intervals or unpaired draws."""
    if not bootstraps:
        raise ValueError("No bootstraps supplied")
    pairing = ("counts_sha256", "n_bootstrap", "seed")
    reference = next(iter(bootstraps.values()))
    samples, points = {}, {}
    for cell, result in bootstraps.items():
        if tuple(result["context_ids"]) != tuple(reference["context_ids"]) or any(
            result[key] != reference[key] for key in pairing
        ):
            raise ValueError("Bootstrap cells do not share exact context order and draws")
        for metric, draws in result["samples"].items():
            key = f"{cell}/{metric}"
            values = np.asarray(draws)
            if values.shape != (reference["n_bootstrap"],) or np.isinf(values).any():
                raise ValueError("Invalid bootstrap samples")
            samples[key] = values
            points[key] = result["summary"][metric]["estimate"]
    summaries, combined = {}, {}
    for name, coefficients in contrasts.items():
        if not coefficients or not set(coefficients) <= samples.keys():
            raise ValueError(f"Contrast has missing statistics: {name}")
        combined[name] = sum(weight * samples[key] for key, weight in coefficients.items())
        point = (
            sum(weight * points[key] for key, weight in coefficients.items())
            if all(points[key] is not None for key in coefficients)
            else None
        )
        summaries[name] = _bootstrap_summary(combined[name], point, confidence)
        summaries[name]["coefficients"] = coefficients
    return summaries, combined


def rotation_variation(summaries):
    """Describe three fixed rotations separately from the context bootstrap CI."""
    result = {}
    for role in ("primary", "comparison"):
        for kind in ("observed", "affine_null"):
            for k in (5, 10, 25):
                for predictor in PREDICTORS:
                    for arm in ("J", "R"):
                        keys = [
                            f"{cell_key(role, kind, k, seed)}/{predictor}/G_{arm}"
                            for seed in ROTATIONS[1:]
                        ]
                        values = [
                            summaries[key]["estimate"] if key in summaries else None for key in keys
                        ]
                        name = f"{role}/{kind}/k{k}/{predictor}/G_{arm}"
                        valid = len(values) == 3 and all(value is not None for value in values)
                        result[name] = {
                            "rotations": list(ROTATIONS[1:]),
                            "estimates": values,
                            "status": "ok" if valid else "missing_or_undefined",
                            "mean": float(np.mean(values)) if valid else None,
                            "minimum": float(np.min(values)) if valid else None,
                            "maximum": float(np.max(values)) if valid else None,
                            "sample_standard_deviation": float(np.std(values, ddof=1))
                            if valid
                            else None,
                            "interpretation": (
                                "descriptive variation across three fixed rotations; "
                                "not a population dictionary confidence interval"
                            ),
                        }
    return result
