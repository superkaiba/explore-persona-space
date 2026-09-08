"""Separate literal benchmark rewards from applicability of the hacking-risk assay."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts import context_risk_highrate_design as design
from scripts.context_risk_impossiblebench_harness import submission_program

POLICY = design.DESIGN / "input_validity_policy.json"


def source_audit(root: Path) -> dict:
    """Apply the pilot-triggered static rule to every immutable source context."""
    policy = json.loads(POLICY.read_text())
    source = root / "manifests/source.jsonl"
    rows = design.source_rows(source)
    if design.sha256(source) != policy["source_sha256"]:
        raise ValueError("Validity policy and frozen benchmark source differ")
    invalid = {}
    for row in rows:
        key = f"{row['task_id']}:{row['condition']}"
        try:
            ast.parse(row["test"])
            ast.parse(submission_program(row, ""))
        except SyntaxError:
            invalid[key] = hashlib.sha256(row["test"].encode()).hexdigest()
    if invalid != policy["structurally_invalid_test_sha256"]:
        raise ValueError("Static invalid-input roster differs from the reviewed amendment")
    return {
        "source_sha256": design.sha256(source),
        "policy_sha256": design.sha256(POLICY),
        "contexts_checked": len(rows),
        "structurally_invalid_test_sha256": invalid,
    }


def annotate(contexts: list[dict], audit: dict) -> dict:
    """Keep native outcomes intact and label structurally non-assessable observations."""
    invalid = audit["structurally_invalid_test_sha256"]
    affected = []
    rank_rows = copy.deepcopy(contexts)
    for original, ranked in zip(contexts, rank_rows, strict=True):
        key = f"{original['task_id']}:{original['condition']}"
        if key not in invalid:
            continue
        if original["success"]:
            raise ValueError("A malformed input unexpectedly received reward; investigate")
        affected.append(copy.deepcopy(original))
        # This second view is only for semantic uncertainty bounds. Native scores
        # remain unchanged, and structural invalidity remains a separate reason.
        ranked["censored"] += ranked["failure"]
        ranked["failure"] = 0
    return {
        "static_audit": audit,
        "structurally_invalid_contexts": affected,
        "not_assessable_planned_trajectories": sum(r["planned"] for r in affected),
        "not_assessable_realized_trajectories": sum(r["realized"] for r in affected),
        "invalid_native_failure": sum(r["failure"] for r in affected),
        "invalid_native_censored": sum(r["censored"] for r in affected),
        "valid_input_success": sum(r["success"] for r in contexts),
        "valid_input_failure": sum(r["failure"] for r in contexts)
        - sum(r["failure"] for r in affected),
        "valid_input_censored": sum(r["censored"] for r in contexts)
        - sum(r["censored"] for r in affected),
        "valid_input_missing": sum(r["missing"] for r in contexts)
        - sum(r["missing"] for r in affected),
        "rank_rows_with_semantic_unknowns": rank_rows,
    }


def selection_annotation(contexts: list[dict], audit: dict) -> dict:
    """Prove that structural-validity accounting leaves every rank and split unchanged."""
    result = annotate(contexts, audit)
    native_rank, native_roles = design.rank_tasks(contexts)
    semantic_rank, semantic_roles = design.rank_tasks(
        result.pop("rank_rows_with_semantic_unknowns")
    )
    if [r["task_id"] for r in native_rank] != [
        r["task_id"] for r in semantic_rank
    ] or native_roles != semantic_roles:
        raise ValueError("Validity annotation changed the frozen task ranking or split")
    return {
        **result,
        "ranking_and_split_unchanged": True,
        "native_ranking": native_rank,
        "validity_aware_ranking": semantic_rank,
        "validity_aware_cutoff_lower": semantic_rank[29]["rate_lower"],
        "validity_aware_excluded_upper_overlap": [
            row["task_id"]
            for row in semantic_rank[30:]
            if row["rate_upper"] >= semantic_rank[29]["rate_lower"]
        ],
        "task_roles": native_roles,
        "unknown_bound_scope": "Native censoring and structural non-assessability contribute to these conservative bounds; their reasons and native scores remain separate.",
    }


def assessable_inputs(rows, raw, audit: dict):
    """Remove invalid contexts before FeatureBank parses any test, retaining full raw evidence."""
    invalid = audit["structurally_invalid_test_sha256"]
    keep = [
        i for i, row in enumerate(rows) if f"{row['task_id']}:{row['condition']}" not in invalid
    ]
    if not keep:
        raise ValueError("No structurally assessable contexts remain")
    return [rows[i] for i in keep], raw[keep]
