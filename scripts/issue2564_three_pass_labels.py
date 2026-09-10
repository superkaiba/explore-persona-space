"""Derive the approved three-pass product without changing original Codex evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import issue2564_codex_judgments as original
from issue2564_answer_behavior import PROPERTIES, digest, dump

PROVIDER = "codex_collaboration_subagents_three_pass"
SELECTED_REPETITIONS = (0, 1, 2)
DRAW_COUNT = 3


def output_dir(root: Path) -> Path:
    return root / "annotation_codex_three_pass/main"


def recipe() -> dict:
    return {
        **original.RECIPE,
        "provider": PROVIDER,
        "draws": DRAW_COUNT,
        "selected_repetitions": list(SELECTED_REPETITIONS),
        "source_provider": original.PROVIDER,
        "source_recipe_hash": digest(original.RECIPE),
        "scope_authorization": "2026-09-09 user approved first three complete passes; no new judging",
    }


def select_units(rows: list[dict], units: list[dict]) -> list[dict]:
    """Select chronological passes uniformly, without looking at probe results."""
    selected = [u for u in units if u["draw"] in SELECTED_REPETITIONS]
    expected = {(r["id"], p, d) for r in rows for p in PROPERTIES for d in SELECTED_REPETITIONS}
    keys = [(u["row_id"], u["property"], u["draw"]) for u in selected]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Selected three-pass keyset is incomplete or duplicated")
    agents = defaultdict(set)
    for unit in selected:
        if unit["drop"] is not None or unit["parsed"] is None:
            raise ValueError("Invalid selected rating; no imputation or silent drop")
        if not unit["agent_id"]:
            raise ValueError("Selected rating lacks its actual judge identity")
        agents[(unit["row_id"], unit["property"])].add(unit["agent_id"])
    if any(len(ids) != DRAW_COUNT for ids in agents.values()):
        raise ValueError("Each answer/property requires three distinct judge contexts")
    return selected


def _load_source(root: Path) -> tuple[list[dict], list[dict], dict, dict]:
    # Reuse all accepted-pilot, exact-packet, recovery and fresh-agent checks.
    acceptance = original.validate_codex_pilot(root, original.RECIPE)
    rows, units, source = original.collect(root, "main")
    selected = select_units(rows, units)
    packet_ids = sorted({u["packet_id"] for u in selected})
    selected_files = [x for x in source["packet_files"] if any(p in x[0] for p in packet_ids)]
    provenance = {
        "source_raw_records_hash": source["raw_records_hash"],
        "source_packet_files": source["packet_files"],
        "selected_raw_key_hash": digest(sorted(u["key"] for u in selected)),
        "selected_packet_ids": packet_ids,
        "selected_packet_files_hash": digest(selected_files),
        "selected_ratings": len(selected),
        "all_source_ratings": len(units),
        "retained_excluded_ratings": len(units) - len(selected),
        "original_expected_ratings": source["expected_draws"],
        "original_five_pass_complete": source["exact_keyset_complete"],
        "source_config_sha256": original.file_hash(root / "annotation_codex/main/config.json"),
        "pilot_acceptance_hash": digest(acceptance),
    }
    return rows, selected, provenance, acceptance


def expected_products(
    rows: list[dict], units: list[dict], provenance: dict
) -> tuple[list, dict, dict]:
    labels, properties = original.summarize_units(rows, units, SELECTED_REPETITIONS)
    quality = {
        "provider": PROVIDER,
        "ratings_per_answer_property": DRAW_COUNT,
        "properties": properties,
        "provenance": provenance,
        "human_agreement": "unmeasured",
        "sampling": original.RECIPE["sampling"],
        "new_model_judgments": 0,
        "scope_limit": "Three ratings instead of five; less averaging of shared-model variability.",
    }
    complete = {
        "provider": PROVIDER,
        "expected_annotations": len(rows) * len(PROPERTIES) * DRAW_COUNT,
        "persisted_expected": len(units),
        "selected_repetitions": list(SELECTED_REPETITIONS),
        "expected_keys_hash": provenance["selected_raw_key_hash"],
        "raw_records_hash": provenance["source_raw_records_hash"],
        "config_hash": digest(recipe()),
        "completion_basis": "Exact selected three-pass roster; original five-pass wave remains separate",
    }
    return labels, quality, complete


def expected_manifest(root: Path, rows: list[dict], provenance: dict) -> dict:
    out = output_dir(root)
    return {
        "provider": PROVIDER,
        "part": "main",
        "row_count": len(rows),
        "expected_draws": len(rows) * len(PROPERTIES) * DRAW_COUNT,
        "persisted_draws": provenance["selected_ratings"],
        "exact_keyset_complete": True,
        "expected_keys_hash": provenance["selected_raw_key_hash"],
        "config_hash": digest(recipe()),
        "rubric_schema_hash": original.instrument_hash(),
        "rows_sha256": original.file_hash(root / "prepared/rows.jsonl"),
        "labels_sha256": original.file_hash(out / "labels.json"),
        "quality_sha256": original.file_hash(out / "quality.json"),
        "complete_sha256": original.file_hash(out / "complete.json"),
        "provenance": provenance,
    }


def export_three_pass_main(root: Path) -> dict:
    out = output_dir(root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Derived product already exists; validate instead of overwriting")
    rows, units, provenance, acceptance = _load_source(root)
    labels, quality, complete = expected_products(rows, units, provenance)
    for name, value in (
        ("config", recipe()),
        ("labels", labels),
        ("quality", quality),
        ("complete", complete),
    ):
        dump(out / f"{name}.json", value)
    manifest = expected_manifest(root, rows, provenance)
    dump(out / "labels_manifest.json", manifest)
    return {
        "labels_path": str(out / "labels.json"),
        "labels_manifest": manifest,
        "config": recipe(),
        "complete": complete,
        "acceptance": acceptance,
    }


def validate_three_pass_main(root: Path) -> dict:
    out = output_dir(root)
    if not (out / "labels_manifest.json").exists():
        raise RuntimeError("Three-pass derived product has not completed")
    rows, units, provenance, acceptance = _load_source(root)
    labels, quality, complete = expected_products(rows, units, provenance)
    for name, expected in (
        ("config", recipe()),
        ("labels", labels),
        ("quality", quality),
        ("complete", complete),
    ):
        if json.loads((out / f"{name}.json").read_text()) != expected:
            raise ValueError(f"Three-pass {name} differs from validated original evidence")
    manifest = json.loads((out / "labels_manifest.json").read_text())
    if manifest != expected_manifest(root, rows, provenance):
        raise ValueError("Three-pass manifest is stale or inconsistent")
    return {
        "labels_path": str(out / "labels.json"),
        "labels_manifest": manifest,
        "config": recipe(),
        "complete": complete,
        "acceptance": acceptance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("export", "validate"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = (export_three_pass_main if args.mode == "export" else validate_three_pass_main)(
        args.root
    )
    print(json.dumps({"labels_path": result["labels_path"], "complete": result["complete"]}))


if __name__ == "__main__":
    main()
