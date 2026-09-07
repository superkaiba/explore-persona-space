#!/usr/bin/env python3
"""Predeclared administration-deviation sensitivity for #2254.

This response-integrity-only analysis removes pass-level judgments affected by
policy packetization and/or the one structured-output replacement, recomputes
each response's mean from its remaining procedural repeats, and repeats the
frozen q0--9 quality selection plus q10--19 confirmation.  It never opens or
uses trait judgments.  The exclusions are influence audits, not estimates of a
causal packet-size effect.
"""

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
import scripts.issue2254_all_answer_decode_recovery as recovery1
import scripts.issue2254_all_answer_decode_recovery3 as recovery3
import scripts.issue2254_all_answer_decode_structured_recovery as structured
import scripts.issue2254_all_answer_decode_sweep as gen


SENSITIVITY_VERSION = "issue2254-administration-sensitivity-v1"
EXPECTED_DECISIONS = 13_200
MIN_RETAINED_REPEATS = 3
V19_SCENARIO_CONTRACT = {
    "no_exclusion": {
        "excluded_decisions": 0,
        "retained_decisions": 13_200,
        "affected_items": 0,
        "remaining_repeat_count_distribution": {"5": 2_640},
        "estimable": True,
    },
    "policy_singleton_only": {
        "excluded_decisions": 122,
        "retained_decisions": 13_078,
        "affected_items": 120,
        "remaining_repeat_count_distribution": {"2": 1, "4": 119, "5": 2_520},
        "estimable": False,
    },
    "all_policy_packetization": {
        "excluded_decisions": 167,
        "retained_decisions": 13_033,
        "affected_items": 164,
        "remaining_repeat_count_distribution": {"2": 1, "3": 1, "4": 162, "5": 2_476},
        "estimable": False,
    },
    "structured_replacement_only": {
        "excluded_decisions": 51,
        "retained_decisions": 13_149,
        "affected_items": 51,
        "remaining_repeat_count_distribution": {"4": 51, "5": 2_589},
        "estimable": True,
    },
    "all_administration_deviations": {
        "excluded_decisions": 218,
        "retained_decisions": 12_982,
        "affected_items": 213,
        "remaining_repeat_count_distribution": {"2": 1, "3": 3, "4": 209, "5": 2_427},
        "estimable": False,
    },
}
V19_OFFENDING_ITEM = {
    "source_item_id": "evil__rb__decodeonly__L14__c1over2|q14|e43",
    "opaque_item_id": "if4834230e9fed80e4bf8",
    "cell_id": "evil__rb__decodeonly__L14__c1over2",
    "behavior": "evil",
    "dose": 0.5,
    "question_index": 14,
    "effective_seed": 43,
    "excluded_pass_indices": [0, 1, 2],
    "retained_pass_indices": [3, 4],
    "retained_repeats": 2,
}


Decision = tuple[str, int]


def _decision_set(opaque_ids: list[str], pass_index: int) -> set[Decision]:
    if type(pass_index) is not int or not 0 <= pass_index < base.N_PASSES:
        raise base.AnalysisError(f"invalid recovered pass index {pass_index!r}")
    if any(not isinstance(opaque_id, str) for opaque_id in opaque_ids):
        raise base.AnalysisError("recovery receipt has a non-string opaque id")
    decisions = {(opaque_id, pass_index) for opaque_id in opaque_ids}
    if len(decisions) != len(opaque_ids):
        raise base.AnalysisError("recovery receipt has duplicate opaque ids")
    return decisions


def _administration_sets(args, items, instrument, rubrics) -> tuple[dict[str, set[Decision]], dict]:
    """Reconstruct exclusions from validated frozen receipts and live rosters."""
    policy._ACTIVE_OUT_ROOT = args.out_root
    structured._ACTIVE_OUT_ROOT = args.out_root
    composed_jobs = structured._fully_recovered_jobs(items, instrument, rubrics, "coherence")
    original_jobs = recovery1._ORIGINAL_PRODUCTION_JOBS(items, instrument, rubrics, "coherence")
    original_matches = [job for job in original_jobs if job.job_id == recovery1.TARGET_JOB_ID]
    if len(original_matches) != 1:
        raise base.AnalysisError("original policy parent no longer resolves exactly once")
    original_parent = original_matches[0]
    first_registry = recovery1._load_split_registry(args.out_root)
    if set(first_registry) != {recovery1.TARGET_JOB_ID}:
        raise base.AnalysisError("first policy registry no longer has the exact target")
    first_receipt = first_registry[recovery1.TARGET_JOB_ID]
    if first_receipt.get("job_id") != original_parent.job_id:
        raise base.AnalysisError("first policy receipt no longer matches its parent")
    policy_all = _decision_set(
        [item.opaque_id for item in original_parent.items], original_parent.pass_index
    )

    third_registry = recovery3._load_registry(args.out_root)
    if set(third_registry) != {recovery3.TARGET_JOB_ID}:
        raise base.AnalysisError("singleton policy registry no longer has the exact target")
    third_receipt = third_registry[recovery3.TARGET_JOB_ID]
    if third_receipt.get("job_id") != recovery3.TARGET_JOB_ID:
        raise base.AnalysisError("singleton policy receipt no longer matches its parent")
    policy_singleton = _decision_set(
        [child["opaque_item_id"] for child in third_receipt["children"]],
        third_receipt["pass_index"],
    )

    universal_registry = policy._load_registry(args.out_root)
    universal_integrity = [
        record
        for record in universal_registry.values()
        if (record.get("scope"), record.get("rubric_id")) == ("integrity_production", "coherence")
    ]
    if not universal_integrity:
        raise base.AnalysisError("sensitivity requires a universal integrity receipt")
    for record in universal_integrity:
        decisions = _decision_set(record["opaque_item_ids"], record["pass_index"])
        singleton_decisions = _decision_set(
            [child["opaque_item_id"] for child in record["children"]],
            record["pass_index"],
        )
        if decisions != singleton_decisions:
            raise base.AnalysisError("universal singleton decisions changed")
        policy_all |= decisions
        policy_singleton |= singleton_decisions

    prior_jobs = policy._prior_recovered_jobs(items, instrument, rubrics, "coherence")
    target_matches = [job for job in prior_jobs if job.job_id == structured.TARGET_JOB_ID]
    if len(target_matches) != 1:
        raise base.AnalysisError("structured parent no longer resolves exactly once")
    structured_receipt = structured._load_receipt(args.out_root, target_matches[0])
    structured_only = _decision_set(
        structured_receipt["parent"]["opaque_item_ids"],
        structured_receipt["parent"]["pass_index"],
    )

    scenarios = {
        "no_exclusion": set(),
        "policy_singleton_only": policy_singleton,
        "all_policy_packetization": policy_all,
        "structured_replacement_only": structured_only,
        "all_administration_deviations": policy_all | structured_only,
    }
    if not policy_singleton <= policy_all:
        raise base.AnalysisError("singleton-conditioned decisions escape policy packetization")
    if len(structured_only) != structured.TARGET_N_ITEMS:
        raise base.AnalysisError("structured replacement decision count changed")
    if policy_all & structured_only:
        raise base.AnalysisError("policy and structured exclusions overlap at pass level")
    valid_opaque = {item.opaque_id for item in items}
    if any(
        opaque_id not in valid_opaque
        for decisions in scenarios.values()
        for opaque_id, _pass in decisions
    ):
        raise base.AnalysisError("administration receipt references an unknown item")
    recovery_root = policy._recovery_root(args.out_root)
    recovery_files = sorted(path for path in recovery_root.rglob("*.json") if path.is_file())
    required_recovery_files = {
        "runner_manifest.json",
        "runner_manifest_v2.json",
        "runner_manifest_v3.json",
        "runner_manifest_universal.json",
        "runner_manifest_structured.json",
        "structured_output_replacement_launch.json",
    }
    if not required_recovery_files <= {path.name for path in recovery_files}:
        raise base.AnalysisError("complete recovery runner/lease lineage is missing")
    instrument_path = base.analysis_root(args.out_root) / "instrument_manifest.json"
    inputs_path = base.analysis_root(args.out_root) / "inputs_manifest.json"
    replacement = structured._replacement_job(target_matches[0])
    replacement_schema = structured._schema_path(args.out_root, replacement)
    replacement_canonical = structured.runner._job_record_path(
        base.analysis_root(args.out_root), replacement
    )
    evidence_files = [instrument_path, inputs_path, replacement_schema, replacement_canonical]
    if any(not path.is_file() for path in evidence_files):
        raise base.AnalysisError("structured/instrument sensitivity evidence is incomplete")
    provenance = {
        "composed_integrity_jobs": len(composed_jobs),
        "composed_integrity_job_registry_sha256": base._canonical_sha256(
            [policy._parent_metadata(job) for job in composed_jobs]
        ),
        "staged_instrument_manifest_sha256": base._sha256_file(instrument_path),
        "staged_inputs_manifest_sha256": base._sha256_file(inputs_path),
        "recovery_lineage": [
            {
                "path": path.relative_to(base.analysis_root(args.out_root)).as_posix(),
                "sha256": base._sha256_file(path),
            }
            for path in recovery_files
        ],
        "universal_integrity_receipts": [
            {
                "job_id": record["job_id"],
                "entry_sha256": record["entry_sha256"],
            }
            for record in universal_integrity
        ],
        "structured_receipt_sha256": structured._sha256_file(
            structured._receipt_path(args.out_root)
        ),
        "structured_receipt_entry_sha256": structured_receipt["entry_sha256"],
        "replacement_schema_sha256": base._sha256_file(replacement_schema),
        "replacement_canonical_sha256": base._sha256_file(replacement_canonical),
    }
    return scenarios, provenance


def _integrity_arrays_with_exclusions(items, outcomes, excluded: set[Decision]):
    """Recompute item means after pass-level deletion, retaining every item."""
    arrays = {}
    remaining_counts = Counter()
    affected_items = set()
    for cell_id in sorted({item.cell_id for item in items}):
        score = np.full((20, 6), np.nan)
        passing = np.full((20, 6), np.nan)
        for item in [row for row in items if row.cell_id == cell_id]:
            values = outcomes[item.opaque_id]
            kept = [
                value
                for pass_index, value in sorted(values.items())
                if (item.opaque_id, pass_index) not in excluded
            ]
            if any(type(value) is not int for value in kept):
                raise base.AnalysisError(
                    f"{item.source_item_id}: non-integer retained integrity outcome"
                )
            if len(kept) < MIN_RETAINED_REPEATS:
                raise base.AnalysisError(
                    f"{item.source_item_id}: fewer than {MIN_RETAINED_REPEATS} "
                    "integrity repeats remain"
                )
            remaining_counts[len(kept)] += 1
            if len(kept) != base.N_PASSES:
                affected_items.add(item.opaque_id)
            mean = float(np.mean(kept))
            slot = item.effective_seed - gen.SEED_BASE
            score[item.qi, slot] = mean
            passing[item.qi, slot] = mean >= base.INTEGRITY_THRESHOLD
        if not np.isfinite(score).all() or not np.isfinite(passing).all():
            raise base.AnalysisError(f"{cell_id}: incomplete sensitivity integrity matrix")
        arrays[cell_id] = {"score": score, "passing": passing}
    if sum(remaining_counts.values()) != len(items):
        raise base.AnalysisError("sensitivity item-count accounting changed")
    if len(affected_items) != len({opaque_id for opaque_id, _pass in excluded}):
        raise base.AnalysisError("sensitivity affected-item accounting changed")
    return arrays, {
        "affected_items": len(affected_items),
        "remaining_repeat_count_distribution": {
            str(count): n_items for count, n_items in sorted(remaining_counts.items())
        },
    }


def _exclusion_preflight(items, excluded: set[Decision]) -> dict:
    """Determine estimability before computing any scenario statistics."""
    by_opaque = {item.opaque_id: item for item in items}
    if len(by_opaque) != len(items):
        raise base.AnalysisError("sensitivity item registry contains duplicate opaque ids")
    excluded_by_item: dict[str, set[int]] = {}
    for opaque_id, pass_index in excluded:
        if opaque_id not in by_opaque:
            raise base.AnalysisError("sensitivity exclusion references an unknown item")
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


def _non_estimable_scenario(common: dict) -> dict:
    """Represent a failed floor without computing or implying a verdict."""
    if common.get("estimable") is not False or not common.get("offending_items"):
        raise base.AnalysisError("non-estimable scenario requires an offending item")
    return {
        "status": "non_estimable_min_repeats",
        "analysis_performed": False,
        **common,
        "selection": None,
        "confirmation": None,
        "stability": None,
        "behaviors": None,
        "comparison_to_primary": None,
        "all_behaviors_stable": None,
    }


def _validate_v19_preflights(preflights: dict[str, dict]) -> None:
    """Pin the approved realized accounting before any scenario analysis."""
    if set(preflights) != set(V19_SCENARIO_CONTRACT):
        raise base.AnalysisError("v19 sensitivity scenario set changed")
    for name, expected in V19_SCENARIO_CONTRACT.items():
        observed = preflights[name]
        if observed.get("minimum_retained_repeats") != MIN_RETAINED_REPEATS:
            raise base.AnalysisError(f"{name}: v19 repeat floor changed")
        mismatches = {
            key: (observed.get(key), value)
            for key, value in expected.items()
            if observed.get(key) != value
        }
        expected_offenders = [] if expected["estimable"] else [V19_OFFENDING_ITEM]
        if observed.get("offending_items") != expected_offenders:
            mismatches["offending_items"] = (
                observed.get("offending_items"),
                expected_offenders,
            )
        if mismatches:
            raise base.AnalysisError(f"{name}: v19 sensitivity accounting changed: {mismatches}")


def _analyze_scenarios(
    items,
    outcomes,
    scenarios: dict[str, set[Decision]],
    frozen_selection: dict,
    provenance_sha256: str,
) -> dict:
    """Preflight all scenarios, then analyze only the two estimable sets."""
    preflights = {
        name: _exclusion_preflight(items, excluded) for name, excluded in scenarios.items()
    }
    _validate_v19_preflights(preflights)
    scenario_results = {}
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
            scenario_results[name] = _non_estimable_scenario(common)
            continue
        arrays, accounting = _integrity_arrays_with_exclusions(items, outcomes, excluded)
        expected_accounting = {
            "affected_items": preflight["affected_items"],
            "remaining_repeat_count_distribution": preflight["remaining_repeat_count_distribution"],
        }
        if accounting != expected_accounting:
            raise base.AnalysisError(f"{name}: exclusion preflight accounting changed")
        analysis = _quality_analysis(arrays, frozen_selection)
        scenario_results[name] = {
            "status": "estimable",
            "analysis_performed": True,
            **common,
            **analysis,
        }
    return scenario_results


def _fixed_primary_confirmation(
    behavior: str,
    arrays: dict,
    selected_cell_id: str | None,
    selected_dose: float | None,
) -> dict:
    """Confirm only the primary q0--9-selected dose on held-out q10--19."""
    if selected_cell_id is None:
        return {
            "status": "no_selected_point_match",
            "selected_cell_id": None,
            "quality_equivalence_confirmed": False,
        }
    context_id = gen.CellSpec(behavior, "context", gen.CONTEXT_DOSE[behavior]).cell_id
    if selected_cell_id not in arrays or context_id not in arrays:
        raise base.AnalysisError(f"{behavior}: frozen primary cell is missing")
    context = arrays[context_id]
    answer = arrays[selected_cell_id]
    q = list(base.CONFIRMATION_QUESTIONS)
    indices = base._bootstrap_indices(f"confirm-{behavior}", len(base.CONFIRMATION_QUESTIONS))
    answer_score_q = answer["score"].mean(axis=1)
    context_score_q = context["score"].mean(axis=1)
    answer_pass_q = answer["passing"].mean(axis=1)
    context_pass_q = context["passing"].mean(axis=1)
    quality = {
        "mean": base._difference(
            answer_score_q[q],
            context_score_q[q],
            indices,
            base.QUALITY_CI_LEVEL,
        ),
        "pass_rate_80": base._difference(
            answer_pass_q[q],
            context_pass_q[q],
            indices,
            base.QUALITY_CI_LEVEL,
        ),
    }
    mean_ci = quality["mean"]["simultaneous_ci"]
    pass_ci = quality["pass_rate_80"]["simultaneous_ci"]
    quality_ok = (
        mean_ci[0] >= -base.MEAN_MARGIN
        and mean_ci[1] <= base.MEAN_MARGIN
        and pass_ci[0] >= -base.PASS_MARGIN
        and pass_ci[1] <= base.PASS_MARGIN
    )
    if behavior == "sycophancy":
        quality_ok = quality_ok and (
            float(answer["passing"][q].mean()) >= 0.90
            and float(context["passing"][q].mean()) >= 0.90
        )
    return {
        "status": "quality_matched" if quality_ok else "quality_not_confirmed",
        "selected_cell_id": selected_cell_id,
        "selected_dose": selected_dose,
        "quality_equivalence_confirmed": bool(quality_ok),
        "quality_vs_context_confirmation": quality,
        "answer_confirmation_pass_rate": float(answer["passing"][q].mean()),
        "context_confirmation_pass_rate": float(context["passing"][q].mean()),
    }


def _quality_analysis(arrays: dict, frozen_selection: dict) -> dict:
    """Repeat selection, but confirm the frozen primary dose in every scenario."""
    result = {"behaviors": {}}
    for behavior in gen.BEHAVIORS:
        context_id, candidates = base._selection_rows(behavior, arrays, base.SELECTION_QUESTIONS)
        eligible = [row for row in candidates if row["eligible"]]
        chosen = min(
            eligible,
            key=lambda row: (row["normalized_max_distance"], -row["dose"]),
            default=None,
        )
        selection = {
            "context_cell_id": context_id,
            "candidates": candidates,
            "selected_cell_id": None if chosen is None else chosen["cell_id"],
            "selected_dose": None if chosen is None else chosen["dose"],
            "match_status": "no_point_match" if chosen is None else "selection_point_match",
        }
        primary = frozen_selection["behaviors"][behavior]
        confirmation = _fixed_primary_confirmation(
            behavior,
            arrays,
            primary["selected_cell_id"],
            primary["selected_dose"],
        )
        result["behaviors"][behavior] = {
            "selection": selection,
            "fixed_primary_dose_confirmation": confirmation,
        }
    return result


def _validate_no_exclusion_reproduction(root: Path, no_exclusion: dict, frozen: dict) -> dict:
    selection_path = root / "selection" / "quality_only_selection.json"
    checks = {}
    for behavior in gen.BEHAVIORS:
        observed_selection = no_exclusion["behaviors"][behavior]["selection"]
        checks[f"{behavior}_selection"] = observed_selection == frozen["behaviors"][behavior]
    if not all(checks.values()):
        raise base.AnalysisError(f"no-exclusion sensitivity failed primary replay: {checks}")
    projection = {
        "version": SENSITIVITY_VERSION,
        "analysis": "integrity-only primary quality projection",
        "trait_scores_read": False,
        "selection_sha256": base._sha256_file(selection_path),
        "behaviors": no_exclusion["behaviors"],
    }
    projection_path = root / "reduce" / "quality_primary_projection.json"
    base._immutable_json(projection_path, projection)
    if json.loads(projection_path.read_text(encoding="utf-8")) != projection:
        raise base.AnalysisError("quality-only primary projection changed after write")
    return {
        "verdict": "PASS",
        "checks": checks,
        "selection_sha256": base._sha256_file(selection_path),
        "quality_primary_projection_sha256": base._sha256_file(projection_path),
    }


def _comparison_to_primary(primary: dict, scenario: dict) -> tuple[dict, bool]:
    """Require selection, match status, and fixed-dose verdict stability."""
    comparison = {
        behavior: {
            "selected_cell_unchanged": (
                scenario["behaviors"][behavior]["selection"]["selected_cell_id"]
                == primary["behaviors"][behavior]["selection"]["selected_cell_id"]
            ),
            "match_status_unchanged": (
                scenario["behaviors"][behavior]["selection"]["match_status"]
                == primary["behaviors"][behavior]["selection"]["match_status"]
            ),
            "quality_equivalence_confirmed_unchanged": (
                scenario["behaviors"][behavior]["fixed_primary_dose_confirmation"][
                    "quality_equivalence_confirmed"
                ]
                == primary["behaviors"][behavior]["fixed_primary_dose_confirmation"][
                    "quality_equivalence_confirmed"
                ]
            ),
        }
        for behavior in gen.BEHAVIORS
    }
    for checks in comparison.values():
        checks["stable"] = bool(all(checks.values()))
    return comparison, bool(all(checks["stable"] for checks in comparison.values()))


def run(args) -> Path:
    root = base.analysis_root(args.out_root)
    selection_path = root / "selection" / "quality_only_selection.json"
    if not selection_path.is_file():
        raise base.AnalysisError("sensitivity requires the frozen quality-only selection")
    frozen_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if frozen_selection.get("trait_scores_read") is not False:
        raise base.AnalysisError("frozen selection is not quality-only")
    items, instrument, rubrics = base._load_staged(args, require_cli=False)
    if len(items) * base.N_PASSES != EXPECTED_DECISIONS:
        raise base.AnalysisError("sensitivity total decision count changed")
    structured._ACTIVE_OUT_ROOT = args.out_root
    policy._ACTIVE_OUT_ROOT = args.out_root
    jobs = structured._fully_recovered_jobs(items, instrument, rubrics, "coherence")
    outcomes = base._collect_outcomes(root, jobs)
    if len(outcomes) != len(items):
        raise base.AnalysisError(f"sensitivity integrity coverage {len(outcomes)}/{len(items)}")
    scenarios, provenance = _administration_sets(args, items, instrument, rubrics)
    provenance_sha256 = base._canonical_sha256(provenance)
    scenario_results = _analyze_scenarios(
        items,
        outcomes,
        scenarios,
        frozen_selection,
        provenance_sha256,
    )
    replay = _validate_no_exclusion_reproduction(
        root, scenario_results["no_exclusion"], frozen_selection
    )
    primary = scenario_results["no_exclusion"]
    for name, scenario in scenario_results.items():
        if not scenario["analysis_performed"]:
            continue
        comparison, stable = _comparison_to_primary(primary, scenario)
        scenario["comparison_to_primary"] = comparison
        scenario["all_behaviors_stable"] = stable
    estimable_names = [name for name, scenario in scenario_results.items() if scenario["estimable"]]
    non_estimable_names = [
        name for name, scenario in scenario_results.items() if not scenario["estimable"]
    ]
    status = (
        "partially_estimable"
        if estimable_names and non_estimable_names
        else "fully_estimable"
        if estimable_names
        else "non_estimable"
    )
    artifact = {
        "version": SENSITIVITY_VERSION,
        "script_path": str(Path(__file__).resolve().relative_to(base._REPO_ROOT)),
        "script_sha256": base._sha256_file(Path(__file__).resolve()),
        "git_commit": base._git_commit(),
        "status": status,
        "analysis": (
            "leave-administration-deviation-decision-out response-integrity influence audit"
        ),
        "interpretation_limit": (
            "not a causal packet-size estimate; pass-level judgments are deleted only "
            "to measure influence on the frozen quality decision"
        ),
        "trait_scores_read": False,
        "expected_total_decisions": EXPECTED_DECISIONS,
        "minimum_retained_repeats": MIN_RETAINED_REPEATS,
        "selection_questions": list(base.SELECTION_QUESTIONS),
        "confirmation_questions": list(base.CONFIRMATION_QUESTIONS),
        "mean_margin": base.MEAN_MARGIN,
        "pass_rate_margin": base.PASS_MARGIN,
        "quality_simultaneous_level": base.QUALITY_CI_LEVEL,
        "n_bootstrap": base.N_BOOTSTRAP,
        "provenance": provenance,
        "provenance_sha256": provenance_sha256,
        "no_exclusion_primary_reproduction": replay,
        "scenarios": scenario_results,
        "overall_stability": {
            name: (scenario["all_behaviors_stable"] if scenario["analysis_performed"] else None)
            for name, scenario in scenario_results.items()
        },
    }
    output = root / "reduce" / "packetization_sensitivity.json"
    base._immutable_json(output, artifact)
    print(
        f"[sensitivity] COMPLETE stable={artifact['overall_stability']} {output}",
        flush=True,
    )
    return output


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(base._REPO_ROOT))
    return parser


def main() -> None:
    try:
        run(build_argparser().parse_args())
    finally:
        structured._ACTIVE_OUT_ROOT = None
        policy._ACTIVE_OUT_ROOT = None


if __name__ == "__main__":
    main()
