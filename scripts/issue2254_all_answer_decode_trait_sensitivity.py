#!/usr/bin/env python3
"""Trait administration-deviation sensitivity for #2254 plans v17/v20."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.issue2254_all_answer_decode_analysis as base
import scripts.issue2254_all_answer_decode_policy_recovery as policy
import scripts.issue2254_all_answer_decode_structured_recovery as structured
import scripts.issue2254_all_answer_decode_sweep as gen
import scripts.issue2254_all_answer_decode_trait_order_recovery as order_recovery


SENSITIVITY_VERSION = "issue2254-trait-administration-sensitivity-v1"
EXPECTED_DECISIONS = 13_200
MIN_RETAINED_REPEATS = 3
ORDER_ONLY_EXPECTED = {
    "excluded_decisions": 36,
    "retained_decisions": 13_164,
    "affected_items": 36,
    "remaining_repeat_count_distribution": {"4": 36, "5": 2_604},
    "estimable": True,
    "offending_items": [],
}


Decision = tuple[str, int]


def _decision_set(opaque_ids: list[str], pass_index: int) -> set[Decision]:
    if type(pass_index) is not int or not 0 <= pass_index < base.N_PASSES:
        raise base.AnalysisError(f"invalid recovered pass index {pass_index!r}")
    if any(not isinstance(opaque_id, str) for opaque_id in opaque_ids):
        raise base.AnalysisError("trait recovery receipt has a non-string opaque id")
    decisions = {(opaque_id, pass_index) for opaque_id in opaque_ids}
    if len(decisions) != len(opaque_ids):
        raise base.AnalysisError("trait recovery receipt has duplicate opaque ids")
    return decisions


def _administration_sets(args, items, instrument, rubrics) -> tuple[dict, dict, set]:
    """Derive all trait exclusions from validated receipts and current rosters."""
    order_recovery._ACTIVE_OUT_ROOT = args.out_root
    policy._ACTIVE_OUT_ROOT = args.out_root
    structured._ACTIVE_OUT_ROOT = args.out_root
    composed_jobs = []
    for rubric_id in ("trait_evil", "trait_sycophancy"):
        composed_jobs.extend(
            order_recovery._fully_recovered_jobs(items, instrument, rubrics, rubric_id)
        )

    prior_evil = policy._prior_recovered_jobs(items, instrument, rubrics, "trait_evil")
    target_matches = [job for job in prior_evil if job.job_id == order_recovery.TARGET_JOB_ID]
    if len(target_matches) != 1:
        raise base.AnalysisError("trait order parent no longer resolves exactly once")
    receipt = order_recovery._load_receipt(args.out_root, target_matches[0])
    order_only = _decision_set(receipt["ordered_opaque_item_ids"], receipt["parent"]["pass_index"])
    if len(order_only) != order_recovery.TARGET_N_ITEMS:
        raise base.AnalysisError("trait order exclusion count changed")

    policy_singleton: set[Decision] = set()
    policy_all: set[Decision] = set()
    registry = policy._load_registry(args.out_root)
    trait_receipts = []
    for record in registry.values():
        if record.get("scope") != "trait_production" or record.get("rubric_id") not in {
            "trait_evil",
            "trait_sycophancy",
        }:
            continue
        parent_decisions = _decision_set(record["opaque_item_ids"], record["pass_index"])
        singleton_decisions = _decision_set(
            [child["opaque_item_id"] for child in record["children"]],
            record["pass_index"],
        )
        if parent_decisions != singleton_decisions:
            raise base.AnalysisError("trait policy singleton decisions changed")
        policy_all |= parent_decisions
        policy_singleton |= singleton_decisions
        trait_receipts.append(
            {
                "job_id": record["job_id"],
                "rubric_id": record["rubric_id"],
                "entry_sha256": record["entry_sha256"],
            }
        )
    if not policy_singleton <= policy_all:
        raise base.AnalysisError("trait singleton decisions escape packetization set")

    scenarios = {
        "no_exclusion": set(),
        "trait_order_replacement_only": order_only,
        "trait_policy_singleton_only": policy_singleton,
        "all_trait_policy_packetization": policy_all,
        "all_trait_administration_deviations": order_only | policy_all,
    }
    valid_opaque = {item.opaque_id for item in items}
    if any(
        opaque_id not in valid_opaque
        for decisions in scenarios.values()
        for opaque_id, _pass in decisions
    ):
        raise base.AnalysisError("trait administration receipt references an unknown item")

    root = base.analysis_root(args.out_root)
    recovery_files = sorted(path for path in (root / "recovery").rglob("*.json") if path.is_file())
    required = {
        "runner_manifest_structured.json",
        "runner_manifest_trait_order.json",
        "trait_order_replacement_launch.json",
    }
    if not required <= {path.name for path in recovery_files}:
        raise base.AnalysisError("trait recovery runner/lease lineage is incomplete")
    replacement = order_recovery._replacement_job(target_matches[0])
    replacement_schema = order_recovery._schema_path(args.out_root, replacement)
    replacement_canonical = order_recovery.runner._job_record_path(root, replacement)
    instrument_path = root / "instrument_manifest.json"
    inputs_path = root / "inputs_manifest.json"
    evidence_files = [
        replacement_schema,
        replacement_canonical,
        instrument_path,
        inputs_path,
    ]
    if any(not path.is_file() for path in evidence_files):
        raise base.AnalysisError("trait sensitivity evidence is incomplete")
    provenance = {
        "composed_trait_jobs": len(composed_jobs),
        "composed_trait_job_registry_sha256": base._canonical_sha256(
            [policy._parent_metadata(job) for job in composed_jobs]
        ),
        "staged_instrument_manifest_sha256": base._sha256_file(instrument_path),
        "staged_inputs_manifest_sha256": base._sha256_file(inputs_path),
        "recovery_lineage": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": base._sha256_file(path),
            }
            for path in recovery_files
        ],
        "trait_policy_receipts": sorted(trait_receipts, key=lambda row: row["job_id"]),
        "trait_order_receipt_sha256": order_recovery._sha256_file(
            order_recovery._receipt_path(args.out_root)
        ),
        "trait_order_receipt_entry_sha256": receipt["entry_sha256"],
        "replacement_schema_sha256": base._sha256_file(replacement_schema),
        "replacement_canonical_sha256": base._sha256_file(replacement_canonical),
        "rejected_original_rows_used": 0,
    }
    return scenarios, provenance, order_only


def _exclusion_preflight(items, excluded: set[Decision]) -> dict:
    """Determine repeat-floor estimability before any numerical analysis."""
    by_opaque = {item.opaque_id: item for item in items}
    if len(by_opaque) != len(items):
        raise base.AnalysisError("trait sensitivity has duplicate opaque ids")
    excluded_by_item: dict[str, set[int]] = {}
    for opaque_id, pass_index in excluded:
        if opaque_id not in by_opaque:
            raise base.AnalysisError("trait exclusion references an unknown item")
        if type(pass_index) is not int or not 0 <= pass_index < base.N_PASSES:
            raise base.AnalysisError("trait exclusion references an invalid pass")
        excluded_by_item.setdefault(opaque_id, set()).add(pass_index)
    distribution = Counter()
    offending = []
    all_passes = set(range(base.N_PASSES))
    for item in items:
        excluded_passes = excluded_by_item.get(item.opaque_id, set())
        retained_passes = all_passes - excluded_passes
        distribution[len(retained_passes)] += 1
        if len(retained_passes) < MIN_RETAINED_REPEATS:
            offending.append(
                {
                    "source_item_id": item.source_item_id,
                    "opaque_item_id": item.opaque_id,
                    "cell_id": item.cell_id,
                    "behavior": item.behavior,
                    "dose": item.dose,
                    "question_index": item.qi,
                    "effective_seed": item.effective_seed,
                    "excluded_pass_indices": sorted(excluded_passes),
                    "retained_pass_indices": sorted(retained_passes),
                    "retained_repeats": len(retained_passes),
                }
            )
    return {
        "minimum_retained_repeats": MIN_RETAINED_REPEATS,
        "excluded_decisions": len(excluded),
        "retained_decisions": EXPECTED_DECISIONS - len(excluded),
        "affected_items": len(excluded_by_item),
        "remaining_repeat_count_distribution": {
            str(count): n_items for count, n_items in sorted(distribution.items())
        },
        "estimable": not offending,
        "offending_items": offending,
    }


def _validate_v20_preflights(
    preflights: dict[str, dict], scenarios: dict[str, set[Decision]], order_only: set[Decision]
) -> None:
    if set(preflights) != {
        "no_exclusion",
        "trait_order_replacement_only",
        "trait_policy_singleton_only",
        "all_trait_policy_packetization",
        "all_trait_administration_deviations",
    }:
        raise base.AnalysisError("v20 trait sensitivity scenario set changed")
    no_exclusion_expected = {
        "minimum_retained_repeats": MIN_RETAINED_REPEATS,
        "excluded_decisions": 0,
        "retained_decisions": EXPECTED_DECISIONS,
        "affected_items": 0,
        "remaining_repeat_count_distribution": {"5": 2_640},
        "estimable": True,
        "offending_items": [],
    }
    if preflights["no_exclusion"] != no_exclusion_expected:
        raise base.AnalysisError("v20 no-exclusion trait accounting changed")
    order_expected = {
        "minimum_retained_repeats": MIN_RETAINED_REPEATS,
        **ORDER_ONLY_EXPECTED,
    }
    if scenarios["trait_order_replacement_only"] != order_only:
        raise base.AnalysisError("v20 trait order exclusion set changed")
    if preflights["trait_order_replacement_only"] != order_expected:
        raise base.AnalysisError("v20 trait order accounting changed")
    if (
        not scenarios["trait_order_replacement_only"]
        <= scenarios["all_trait_administration_deviations"]
    ):
        raise base.AnalysisError("v20 union omitted trait order decisions")
    if (
        not scenarios["all_trait_policy_packetization"]
        <= scenarios["all_trait_administration_deviations"]
    ):
        raise base.AnalysisError("v20 union omitted trait policy decisions")


def _trait_arrays_with_exclusions(items, outcomes, excluded: set[Decision]):
    arrays = {}
    remaining_counts = Counter()
    affected_items = set()
    for cell_id in sorted({item.cell_id for item in items}):
        itt = np.full((20, 6), np.nan)
        numeric = np.full((20, 6), np.nan)
        refusal = np.full((20, 6), np.nan)
        numeric_draw_complete = np.full((20, 6), np.nan)
        for item in [row for row in items if row.cell_id == cell_id]:
            if set(outcomes[item.opaque_id]) != set(range(base.N_PASSES)):
                raise base.AnalysisError(f"{item.source_item_id}: incomplete trait repeats")
            kept = [
                outcomes[item.opaque_id][index]
                for index in range(base.N_PASSES)
                if (item.opaque_id, index) not in excluded
            ]
            if len(kept) < MIN_RETAINED_REPEATS:
                raise base.AnalysisError(
                    f"{item.source_item_id}: fewer than {MIN_RETAINED_REPEATS} repeats remain"
                )
            if any(
                not (value == "REFUSAL" or (type(value) is int and 0 <= value <= 100))
                for value in kept
            ):
                raise base.AnalysisError(f"{item.source_item_id}: invalid trait outcome")
            numeric_values = [value for value in kept if type(value) is int]
            slot = item.effective_seed - gen.SEED_BASE
            itt[item.qi, slot] = float(
                np.mean([0 if value == "REFUSAL" else value for value in kept])
            )
            if numeric_values:
                numeric[item.qi, slot] = float(np.mean(numeric_values))
            refusal[item.qi, slot] = (len(kept) - len(numeric_values)) / len(kept)
            numeric_draw_complete[item.qi, slot] = len(numeric_values) / len(kept)
            remaining_counts[len(kept)] += 1
            if len(kept) != base.N_PASSES:
                affected_items.add(item.opaque_id)
        if not np.isfinite(itt).all() or not np.isfinite(refusal).all():
            raise base.AnalysisError(f"{cell_id}: incomplete trait sensitivity matrix")
        arrays[cell_id] = {
            "itt": itt,
            "numeric": numeric,
            "refusal": refusal,
            "numeric_draw_complete": numeric_draw_complete,
        }
    return arrays, {
        "affected_items": len(affected_items),
        "remaining_repeat_count_distribution": {
            str(count): n_items for count, n_items in sorted(remaining_counts.items())
        },
    }


def _numeric_sensitivity(cell: dict) -> dict:
    numeric = cell["numeric"]
    item_completeness = float(np.isfinite(numeric).mean())
    draw_completeness = float(cell["numeric_draw_complete"].mean())
    return {
        "item_completeness": item_completeness,
        "draw_completeness": draw_completeness,
        "valid": bool(
            item_completeness >= base.NUMERIC_SENSITIVITY_FLOOR
            and draw_completeness >= base.NUMERIC_SENSITIVITY_FLOOR
        ),
        "mean": None if not np.isfinite(numeric).any() else float(np.nanmean(numeric)),
    }


def _behavior_metrics(behavior: str, arrays: dict, primary_result: dict) -> dict:
    full_idx = base._bootstrap_indices(f"full-{behavior}", 20)
    confirm_idx = base._bootstrap_indices(f"confirm-{behavior}", len(base.CONFIRMATION_QUESTIONS))
    context_cell = gen.CellSpec(behavior, "context", gen.CONTEXT_DOSE[behavior])
    zero_cell = gen.CellSpec(behavior, "decodeonly", 0.0)
    context = arrays[context_cell.cell_id]
    zero = arrays[zero_cell.cell_id]
    context_q = context["itt"].mean(axis=1)
    zero_q = zero["itt"].mean(axis=1)
    cells = []
    for dose in gen.DOSES:
        cell_spec = gen.CellSpec(behavior, "decodeonly", dose)
        cell = arrays[cell_spec.cell_id]
        trait_q = cell["itt"].mean(axis=1)
        row = {
            "cell_id": cell_spec.cell_id,
            "dose": dose,
            "trait_itt_full20": base._summary(trait_q, full_idx),
            "trait_refusal_fraction": float(cell["refusal"].mean()),
            "numeric_sensitivity": _numeric_sensitivity(cell),
            "trait_vs_zero_full20": base._difference(
                trait_q, zero_q, full_idx, base.TRAIT_CI_LEVEL
            ),
        }
        if dose != 0:
            row["trait_vs_context_confirmation"] = base._difference(
                trait_q[list(base.CONFIRMATION_QUESTIONS)],
                context_q[list(base.CONFIRMATION_QUESTIONS)],
                confirm_idx,
                base.TRAIT_CI_LEVEL,
            )
            row["trait_vs_context_full20_exploratory"] = base._difference(
                trait_q, context_q, full_idx, base.TRAIT_CI_LEVEL
            )
        cells.append(row)
    context_row = {
        "cell_id": context_cell.cell_id,
        "dose": context_cell.dose,
        "trait_itt_full20": base._summary(context_q, full_idx),
        "trait_refusal_fraction": float(context["refusal"].mean()),
        "numeric_sensitivity": _numeric_sensitivity(context),
        "trait_vs_zero_full20": base._difference(context_q, zero_q, full_idx, base.TRAIT_CI_LEVEL),
    }

    primary_quality = primary_result["behaviors"][behavior]["primary_confirmation"]
    selected_id = primary_quality.get("selected_cell_id")
    chosen = next((row for row in cells if row["cell_id"] == selected_id), None)
    if chosen is None:
        primary_confirmation = {
            "status": "not_applicable_no_selected_point_match",
            "selected_cell_id": None,
            "quality_equivalence_confirmed": False,
            "confirmatory_trait_tested": False,
            "trait_verdict": "not_tested",
        }
    else:
        quality_ok = bool(primary_quality["quality_equivalence_confirmed"])
        trait_verdict, descriptive = base._confirmatory_trait_verdict(
            chosen["trait_vs_context_confirmation"]["simultaneous_ci"], quality_ok
        )
        primary_confirmation = {
            "status": primary_quality["status"],
            "selected_cell_id": selected_id,
            "selected_dose": primary_quality["selected_dose"],
            "quality_equivalence_confirmed": quality_ok,
            "confirmatory_trait_tested": quality_ok,
            "trait_contrast": chosen["trait_vs_context_confirmation"],
            "trait_verdict": trait_verdict,
            "descriptive_trait_direction": descriptive,
        }
    return {
        "context": context_row,
        "answer_dose_frontier": cells,
        "primary_confirmation": primary_confirmation,
    }


def _validate_quality_lineage(
    selection: dict,
    quality_projection: dict,
    primary_result: dict,
    selection_sha256: str,
) -> dict:
    """Bind trait analysis to the frozen, integrity-only quality decision."""
    if selection.get("trait_scores_read") is not False:
        raise base.AnalysisError("trait sensitivity selection is not quality-only")
    if quality_projection.get("trait_scores_read") is not False:
        raise base.AnalysisError("quality primary projection is not integrity-only")
    if quality_projection.get("selection_sha256") != selection_sha256:
        raise base.AnalysisError("quality primary projection selection hash changed")
    if primary_result.get("selection") != selection:
        raise base.AnalysisError("primary reduction selection differs from frozen selection")
    if set(selection.get("behaviors", {})) != set(gen.BEHAVIORS):
        raise base.AnalysisError("frozen selection behavior set changed")
    if set(quality_projection.get("behaviors", {})) != set(gen.BEHAVIORS):
        raise base.AnalysisError("quality primary projection behavior set changed")
    if set(primary_result.get("behaviors", {})) != set(gen.BEHAVIORS):
        raise base.AnalysisError("primary reduction behavior set changed")

    frozen_quality = {"behaviors": {}}
    for behavior in gen.BEHAVIORS:
        frozen_selection = selection["behaviors"][behavior]
        projection_behavior = quality_projection["behaviors"][behavior]
        if projection_behavior.get("selection") != frozen_selection:
            raise base.AnalysisError(
                f"{behavior}: quality projection selection differs from frozen selection"
            )
        projected_confirmation = projection_behavior.get("fixed_primary_dose_confirmation")
        if not isinstance(projected_confirmation, dict):
            raise base.AnalysisError(f"{behavior}: quality projection confirmation is missing")
        if type(projected_confirmation.get("quality_equivalence_confirmed")) is not bool:
            raise base.AnalysisError(
                f"{behavior}: projected quality-equivalence flag is not boolean"
            )
        selected_cell_id = frozen_selection.get("selected_cell_id")
        selected_dose = frozen_selection.get("selected_dose")
        if (
            projected_confirmation.get("selected_cell_id") != selected_cell_id
            or projected_confirmation.get("selected_dose") != selected_dose
        ):
            raise base.AnalysisError(
                f"{behavior}: projected confirmation differs from frozen selected dose"
            )
        projected_status = projected_confirmation.get("status")
        projected_quality_ok = projected_confirmation["quality_equivalence_confirmed"]
        if selected_cell_id is None:
            internally_valid = (
                projected_status == "no_selected_point_match" and not projected_quality_ok
            )
        else:
            internally_valid = projected_status in {
                "quality_matched",
                "quality_not_confirmed",
            } and projected_quality_ok == (projected_status == "quality_matched")
        if not internally_valid:
            raise base.AnalysisError(
                f"{behavior}: quality projection confirmation is internally inconsistent"
            )

        primary_confirmation = primary_result["behaviors"][behavior].get(
            "primary_confirmation"
        )
        if not isinstance(primary_confirmation, dict):
            raise base.AnalysisError(f"{behavior}: primary confirmation is missing")
        comparison_keys = (
            "selected_cell_id",
            "selected_dose",
            "status",
            "quality_equivalence_confirmed",
        )
        projection_gate = {
            key: projected_confirmation.get(key) for key in comparison_keys
        }
        primary_gate = {key: primary_confirmation.get(key) for key in comparison_keys}
        if primary_gate != projection_gate:
            raise base.AnalysisError(
                f"{behavior}: primary quality gate differs from integrity-only projection"
            )
        frozen_quality["behaviors"][behavior] = {
            "primary_confirmation": projected_confirmation
        }
    return frozen_quality


def _affected_cells(items, excluded: set[Decision]) -> list[str]:
    excluded_opaque = {opaque_id for opaque_id, _pass in excluded}
    return sorted({item.cell_id for item in items if item.opaque_id in excluded_opaque})


def _comparison_to_primary(primary: dict, scenario: dict) -> dict:
    comparison = {"behaviors": {}}
    for behavior in gen.BEHAVIORS:
        p_behavior = primary["behaviors"][behavior]
        s_behavior = scenario["behaviors"][behavior]
        p_cells = {
            row["cell_id"]: row
            for row in [p_behavior["context"], *p_behavior["answer_dose_frontier"]]
        }
        s_cells = {
            row["cell_id"]: row
            for row in [s_behavior["context"], *s_behavior["answer_dose_frontier"]]
        }
        per_cell = {}
        for cell_id in sorted(p_cells):
            p_row = p_cells[cell_id]
            s_row = s_cells[cell_id]
            numeric_delta = None
            if (
                p_row["numeric_sensitivity"]["mean"] is not None
                and s_row["numeric_sensitivity"]["mean"] is not None
            ):
                numeric_delta = (
                    s_row["numeric_sensitivity"]["mean"] - p_row["numeric_sensitivity"]["mean"]
                )
            per_cell[cell_id] = {
                "trait_itt_estimate_delta": (
                    s_row["trait_itt_full20"]["estimate"] - p_row["trait_itt_full20"]["estimate"]
                ),
                "trait_refusal_fraction_delta": (
                    s_row["trait_refusal_fraction"] - p_row["trait_refusal_fraction"]
                ),
                "numeric_item_completeness_delta": (
                    s_row["numeric_sensitivity"]["item_completeness"]
                    - p_row["numeric_sensitivity"]["item_completeness"]
                ),
                "numeric_draw_completeness_delta": (
                    s_row["numeric_sensitivity"]["draw_completeness"]
                    - p_row["numeric_sensitivity"]["draw_completeness"]
                ),
                "numeric_mean_delta": numeric_delta,
                "trait_vs_zero_estimate_delta": (
                    s_row["trait_vs_zero_full20"]["estimate"]
                    - p_row["trait_vs_zero_full20"]["estimate"]
                ),
            }
            if "trait_vs_context_full20_exploratory" in p_row:
                per_cell[cell_id]["trait_vs_context_full20_estimate_delta"] = (
                    s_row["trait_vs_context_full20_exploratory"]["estimate"]
                    - p_row["trait_vs_context_full20_exploratory"]["estimate"]
                )
        comparison["behaviors"][behavior] = {
            "per_cell": per_cell,
            "maximum_absolute_trait_itt_estimate_delta": max(
                abs(row["trait_itt_estimate_delta"]) for row in per_cell.values()
            ),
            "primary_confirmation_before": p_behavior["primary_confirmation"],
            "primary_confirmation_after": s_behavior["primary_confirmation"],
        }
    return comparison


def _non_estimable_scenario(common: dict) -> dict:
    if common.get("estimable") is not False or not common.get("offending_items"):
        raise base.AnalysisError("non-estimable trait scenario requires an offending item")
    return {
        "status": "non_estimable_min_repeats",
        "analysis_performed": False,
        **common,
        "affected_cells": None,
        "behaviors": None,
        "comparison_to_primary": None,
    }


def _analyze_scenarios(
    items,
    outcomes,
    scenarios: dict[str, set[Decision]],
    order_only: set[Decision],
    primary_result: dict,
    provenance_sha256: str,
) -> dict:
    preflights = {
        name: _exclusion_preflight(items, excluded) for name, excluded in scenarios.items()
    }
    _validate_v20_preflights(preflights, scenarios, order_only)
    results = {}
    provenance_reference = {"location": "$.provenance", "sha256": provenance_sha256}
    for name, excluded in scenarios.items():
        preflight = preflights[name]
        common = {
            "excluded_decision_set_sha256": base._canonical_sha256(
                sorted([list(decision) for decision in excluded])
            ),
            "recovery_provenance_reference": provenance_reference,
            **preflight,
        }
        if not preflight["estimable"]:
            results[name] = _non_estimable_scenario(common)
            continue
        arrays, accounting = _trait_arrays_with_exclusions(items, outcomes, excluded)
        expected_accounting = {
            "affected_items": preflight["affected_items"],
            "remaining_repeat_count_distribution": preflight["remaining_repeat_count_distribution"],
        }
        if accounting != expected_accounting:
            raise base.AnalysisError(f"{name}: trait preflight accounting changed")
        results[name] = {
            "status": "estimable",
            "analysis_performed": True,
            **common,
            "affected_cells": _affected_cells(items, excluded),
            "behaviors": {
                behavior: _behavior_metrics(behavior, arrays, primary_result)
                for behavior in gen.BEHAVIORS
            },
        }
    primary = results["no_exclusion"]
    for scenario in results.values():
        if scenario["analysis_performed"]:
            scenario["comparison_to_primary"] = _comparison_to_primary(primary, scenario)
    return results


def _project_primary(primary_result: dict) -> dict:
    projected = {"behaviors": {}}
    for behavior in gen.BEHAVIORS:
        block = primary_result["behaviors"][behavior]
        context = block["context"]
        cells = []
        for row in block["answer_dose_frontier"]:
            projected_row = {
                key: row[key]
                for key in (
                    "cell_id",
                    "dose",
                    "trait_itt_full20",
                    "trait_refusal_fraction",
                    "numeric_sensitivity",
                    "trait_vs_zero_full20",
                )
            }
            for key in (
                "trait_vs_context_confirmation",
                "trait_vs_context_full20_exploratory",
            ):
                if key in row:
                    projected_row[key] = row[key]
            cells.append(projected_row)
        primary_confirmation = dict(block["primary_confirmation"])
        primary_confirmation.pop("interpretation", None)
        if primary_confirmation.get("status") == "no_selected_point_match":
            primary_confirmation = {
                "status": "not_applicable_no_selected_point_match",
                "selected_cell_id": None,
                "quality_equivalence_confirmed": False,
                "confirmatory_trait_tested": False,
                "trait_verdict": "not_tested",
            }
        projected["behaviors"][behavior] = {
            "context": {
                key: context[key]
                for key in (
                    "cell_id",
                    "dose",
                    "trait_itt_full20",
                    "trait_refusal_fraction",
                    "numeric_sensitivity",
                    "trait_vs_zero_full20",
                )
            },
            "answer_dose_frontier": cells,
            "primary_confirmation": primary_confirmation,
        }
    return projected


def run(args) -> Path:
    root = base.analysis_root(args.out_root)
    primary_path = root / "reduce" / "matched_results.json"
    selection_path = root / "selection" / "quality_only_selection.json"
    quality_projection_path = root / "reduce" / "quality_primary_projection.json"
    if not all(
        path.is_file() for path in (primary_path, selection_path, quality_projection_path)
    ):
        raise base.AnalysisError(
            "trait sensitivity requires primary reduction, frozen selection, and "
            "integrity-only quality projection"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    quality_projection = json.loads(quality_projection_path.read_text(encoding="utf-8"))
    primary_result = json.loads(primary_path.read_text(encoding="utf-8"))
    selection_sha256 = base._sha256_file(selection_path)
    frozen_quality = _validate_quality_lineage(
        selection,
        quality_projection,
        primary_result,
        selection_sha256,
    )
    items, instrument, rubrics = base._load_staged(args, require_cli=False)
    if len(items) * base.N_PASSES != EXPECTED_DECISIONS:
        raise base.AnalysisError("trait sensitivity total decision count changed")
    scenarios, provenance, order_only = _administration_sets(args, items, instrument, rubrics)
    jobs = []
    for rubric_id in ("trait_evil", "trait_sycophancy"):
        jobs.extend(order_recovery._fully_recovered_jobs(items, instrument, rubrics, rubric_id))
    outcomes = base._collect_outcomes(root, jobs)
    if len(outcomes) != len(items):
        raise base.AnalysisError(f"trait sensitivity coverage {len(outcomes)}/{len(items)}")
    provenance_sha256 = base._canonical_sha256(provenance)
    scenario_results = _analyze_scenarios(
        items,
        outcomes,
        scenarios,
        order_only,
        frozen_quality,
        provenance_sha256,
    )
    no_exclusion_projection = {"behaviors": scenario_results["no_exclusion"]["behaviors"]}
    expected_projection = _project_primary(primary_result)
    if no_exclusion_projection != expected_projection:
        raise base.AnalysisError("trait sensitivity does not reproduce primary reduction")
    estimable = [name for name, row in scenario_results.items() if row["estimable"]]
    non_estimable = [name for name, row in scenario_results.items() if not row["estimable"]]
    status = (
        "partially_estimable"
        if estimable and non_estimable
        else "fully_estimable"
        if estimable
        else "non_estimable"
    )
    artifact = {
        "version": SENSITIVITY_VERSION,
        "script_path": str(Path(__file__).resolve().relative_to(base._REPO_ROOT)),
        "script_sha256": base._sha256_file(Path(__file__).resolve()),
        "git_commit": base._git_commit(),
        "status": status,
        "analysis": "leave-trait-administration-deviation-decision-out influence audit",
        "interpretation_limit": "not a causal packet-size or judge-draw estimate",
        "rejected_original_rows_used": 0,
        "quality_selection_frozen": True,
        "quality_lineage_validation": "PASS",
        "selection_sha256": selection_sha256,
        "quality_primary_projection_sha256": base._sha256_file(quality_projection_path),
        "primary_results_sha256": base._sha256_file(primary_path),
        "expected_total_decisions": EXPECTED_DECISIONS,
        "minimum_retained_repeats": MIN_RETAINED_REPEATS,
        "n_bootstrap": base.N_BOOTSTRAP,
        "trait_simultaneous_level": base.TRAIT_CI_LEVEL,
        "provenance": provenance,
        "provenance_sha256": provenance_sha256,
        "no_exclusion_primary_reproduction": "PASS",
        "scenarios": scenario_results,
    }
    output = root / "reduce" / "trait_administration_sensitivity.json"
    base._immutable_json(output, artifact)
    print(f"[trait-sensitivity] COMPLETE status={status} {output}", flush=True)
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(base._REPO_ROOT))
    return parser


def main() -> None:
    try:
        run(build_argparser().parse_args())
    finally:
        order_recovery._ACTIVE_OUT_ROOT = None
        policy._ACTIVE_OUT_ROOT = None
        structured._ACTIVE_OUT_ROOT = None


if __name__ == "__main__":
    main()
