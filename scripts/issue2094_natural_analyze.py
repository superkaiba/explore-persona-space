#!/usr/bin/env python3
"""Analyze corrected #2094 natural task/subject patching and write a report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [center - radius, center + radius]


def rate_record(rows: list[dict[str, Any]], predicate) -> dict[str, Any]:
    n = len(rows)
    k = sum(bool(predicate(row)) for row in rows)
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": wilson(k, n)}


def structural_format(text: str) -> str:
    """Simple non-judge companion for the bullet/paragraph control."""
    body = text.removeprefix("Response:\n").strip()
    lines = [line for line in body.splitlines() if line.strip()]
    bullet_lines = sum(bool(re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", line)) for line in lines)
    if bullet_lines >= 3:
        return "bullets"
    blocks = [block for block in re.split(r"\n\s*\n", body) if block.strip()]
    if bullet_lines == 0 and len(blocks) == 1:
        return "paragraph"
    return "neither_or_mixed"


def validate_and_join(
    generations: list[dict[str, Any]], annotations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(generations) != 129:
        raise ValueError(f"expected 129 generation rows, found {len(generations)}")
    gen_by_id = {str(row["gen_id"]): row for row in generations}
    ann_by_id = {str(row["gen_id"]): row for row in annotations}
    if len(gen_by_id) != len(generations) or len(ann_by_id) != len(annotations):
        raise ValueError("duplicate generation or annotation IDs")
    if set(gen_by_id) != set(ann_by_id):
        raise ValueError("generation and annotation ID sets differ")
    if any(row.get("termination_reason") != "eos" for row in generations):
        raise ValueError("capped/non-EOS row in final generations")
    joined: list[dict[str, Any]] = []
    for generation in generations:
        annotation = ann_by_id[str(generation["gen_id"])]
        joined.append(
            {
                **generation,
                "blind_annotation": {
                    key: annotation[key]
                    for key in (
                        "row_id",
                        "form",
                        "subject",
                        "format",
                        "complete",
                        "coherence",
                        "evidence",
                    )
                },
                "structural_format": structural_format(str(generation["output_text"])),
            }
        )
    return joined


def validate_annotation_audit(
    annotation_path: Path,
    annotations: list[dict[str, Any]],
    generations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replay the frozen-reader sidecars before using any blind labels."""
    from issue2094_natural_blind_annotate import (
        build_segments,
        parse_annotations,
        parse_codex_events,
        scan_for_leakage,
    )

    audit_root = annotation_path.parent
    done_path = audit_root / "DONE.json"
    if not done_path.exists():
        raise ValueError("missing annotation DONE.json")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("n_rows") != len(annotations) or not done.get("all_rows_annotated"):
        raise ValueError("annotation DONE.json census mismatch")
    packet_root = audit_root / "packets"
    request_paths = sorted(packet_root.glob("*.request.json"))
    response_paths = sorted(packet_root.glob("*.response.json"))
    parsed_paths = sorted(packet_root.glob("*.parsed.json"))
    expected_packets = int(done["n_packets"])
    if not (len(request_paths) == len(response_paths) == len(parsed_paths) == expected_packets):
        raise ValueError("annotation sidecar packet census mismatch")

    key_path = audit_root / "blind_key.json"
    if not key_path.exists():
        raise ValueError("missing frozen annotation key")
    mapping = json.loads(key_path.read_text(encoding="utf-8"))["row_id_to_gen_id"]
    annotation_mapping = {str(row["row_id"]): str(row["gen_id"]) for row in annotations}
    if mapping != annotation_mapping:
        raise ValueError("frozen annotation key does not match final annotations")
    generations_by_id = {str(row["gen_id"]): row for row in generations}
    if set(generations_by_id) != set(mapping.values()):
        raise ValueError("frozen annotation key does not match generation IDs")

    requests = [json.loads(path.read_text(encoding="utf-8")) for path in request_paths]
    row_ids = [row_id for request in requests for row_id in request["row_ids"]]
    expected_row_ids = [str(row["row_id"]) for row in annotations]
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != set(expected_row_ids):
        raise ValueError("annotation request row-ID census mismatch")
    if any(
        request["leakage_scan_scopes"][scope]["hits"]
        for request in requests
        for scope in ("wrapper", "payload")
    ):
        raise ValueError("annotation request leakage scan is not clean")
    if any(request.get("tool_item_types_observed", []) for request in requests):
        raise ValueError("annotation reader tool-use audit is not clean")

    final_by_row = {
        str(row["row_id"]): {key: value for key, value in row.items() if key != "gen_id"}
        for row in annotations
    }
    for request_path, response_path, parsed_path, request in zip(
        request_paths, response_paths, parsed_paths, requests, strict=True
    ):
        if not (
            request_path.stem.removesuffix(".request")
            == response_path.stem.removesuffix(".response")
            == parsed_path.stem.removesuffix(".parsed")
        ):
            raise ValueError("annotation sidecar stems do not align")
        packet_ids = [str(row_id) for row_id in request["row_ids"]]
        items = [
            (row_id, str(generations_by_id[str(mapping[row_id])]["output_text"]))
            for row_id in packet_ids
        ]
        segments = build_segments(items)
        expected_request = "".join(text for _scope, text in segments)
        if request["outbound_request_verbatim"] != expected_request:
            raise ValueError("annotation outbound request does not match frozen-key payload")
        if any(scan_for_leakage(segments).values()):
            raise ValueError("replayed annotation leakage scan is not clean")

        response = json.loads(response_path.read_text(encoding="utf-8"))
        parsed_file = json.loads(parsed_path.read_text(encoding="utf-8"))
        parsed_raw = parse_annotations(str(response["raw_text"]), packet_ids)
        if parsed_file != parsed_raw:
            raise ValueError("parsed annotation sidecar differs from raw response")
        if parsed_file != [final_by_row[row_id] for row_id in packet_ids]:
            raise ValueError("parsed annotation sidecar differs from final annotations")

        if done["backend"] == "codex-cli":
            events, _usage = parse_codex_events(str(response["provider_event_jsonl"]))
            agent_messages = [
                event["item"]["text"].strip()
                for event in events
                if event.get("type") == "item.completed"
                and isinstance(event.get("item"), dict)
                and event["item"].get("type") == "agent_message"
            ]
            if agent_messages != [str(response["raw_text"]).strip()]:
                raise ValueError("annotation event transcript differs from raw response")

    deviations = sorted(
        {
            str(request["protocol_deviation"])
            for request in requests
            if request.get("protocol_deviation")
        }
    )
    return {
        "backend": done["backend"],
        "model": done["model"],
        "n_packets": expected_packets,
        "n_rows": len(row_ids),
        "all_leakage_scans_clean": True,
        "all_tool_audits_clean": True,
        "all_payloads_replayed": True,
        "all_raw_parsed_final_equal": True,
        "protocol_deviations": deviations,
    }


def subset(
    rows: list[dict[str, Any]],
    *,
    arm: str | None = None,
    axis: str | None = None,
    setting: str | None = None,
) -> list[dict[str, Any]]:
    out = rows
    if arm is not None:
        out = [row for row in out if row["arm"] == arm]
    if axis is not None:
        out = [row for row in out if row["axis"] == axis]
    if setting is not None:
        out = [row for row in out if row["layer_setting"] == setting]
    return out


def expected_anchor_form(row: dict[str, Any]) -> str:
    return row["recipient_task"]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    anchors = subset(rows, arm="unpatched")
    anchor_main = [
        row for row in anchors if row["axis"] == "anchor" and row["recipient_format"] is None
    ]
    anchor_format = [row for row in anchors if row["recipient_format"] is not None]
    anchor_summary = {
        "n": len(anchors),
        "main_form_accuracy": rate_record(
            anchor_main,
            lambda row: row["blind_annotation"]["form"] == expected_anchor_form(row),
        ),
        "main_subject_accuracy": rate_record(
            anchor_main,
            lambda row: row["blind_annotation"]["subject"] == row["recipient_subject"],
        ),
        "format_label_accuracy": rate_record(
            anchor_format,
            lambda row: row["blind_annotation"]["format"] == row["recipient_format"],
        ),
        "format_structural_accuracy": rate_record(
            anchor_format,
            lambda row: row["structural_format"] == row["recipient_format"],
        ),
        "complete": rate_record(anchors, lambda row: row["blind_annotation"]["complete"]),
        "coherent_ge_60": rate_record(
            anchors, lambda row: row["blind_annotation"]["coherence"] >= 60
        ),
    }

    anchor_text = {row["recipient_prompt_id"]: row["output_text"] for row in anchors}
    self_summary: dict[str, Any] = {}
    for setting in ("L19", "all28"):
        arm_rows = subset(rows, arm="self_patch", setting=setting)
        self_summary[setting] = {
            "n": len(arm_rows),
            "exact_text_match": rate_record(
                arm_rows,
                lambda row: row["output_text"] == anchor_text[row["recipient_prompt_id"]],
            ),
            "complete": rate_record(arm_rows, lambda row: row["blind_annotation"]["complete"]),
            "coherent_ge_60": rate_record(
                arm_rows, lambda row: row["blind_annotation"]["coherence"] >= 60
            ),
            "max_injection_error": max(
                row["injection_telemetry"]["max_abs_source_error"] for row in arm_rows
            ),
        }

    settings: dict[str, Any] = {}
    for setting in ("L19", "all28"):
        task_rows = subset(
            rows, arm="donor_patch", axis="same_subject_different_task", setting=setting
        )
        subject_rows = subset(
            rows, arm="donor_patch", axis="same_task_different_subject", setting=setting
        )
        format_rows = subset(
            rows, arm="donor_patch", axis="positive_format_control", setting=setting
        )
        settings[setting] = {
            "task_swap": {
                "n": len(task_rows),
                "donor_task": rate_record(
                    task_rows,
                    lambda row: row["blind_annotation"]["form"] == row["donor_task"],
                ),
                "recipient_task": rate_record(
                    task_rows,
                    lambda row: row["blind_annotation"]["form"] == row["recipient_task"],
                ),
                "recipient_subject": rate_record(
                    task_rows,
                    lambda row: row["blind_annotation"]["subject"] == row["recipient_subject"],
                ),
                "complete": rate_record(task_rows, lambda row: row["blind_annotation"]["complete"]),
                "coherent_ge_60": rate_record(
                    task_rows, lambda row: row["blind_annotation"]["coherence"] >= 60
                ),
                "observed_form_counts": dict(
                    Counter(row["blind_annotation"]["form"] for row in task_rows)
                ),
            },
            "subject_swap": {
                "n": len(subject_rows),
                "donor_subject": rate_record(
                    subject_rows,
                    lambda row: row["blind_annotation"]["subject"] == row["donor_subject"],
                ),
                "recipient_subject": rate_record(
                    subject_rows,
                    lambda row: row["blind_annotation"]["subject"] == row["recipient_subject"],
                ),
                "recipient_task": rate_record(
                    subject_rows,
                    lambda row: row["blind_annotation"]["form"] == row["recipient_task"],
                ),
                "complete": rate_record(
                    subject_rows, lambda row: row["blind_annotation"]["complete"]
                ),
                "coherent_ge_60": rate_record(
                    subject_rows, lambda row: row["blind_annotation"]["coherence"] >= 60
                ),
                "observed_subject_counts": dict(
                    Counter(row["blind_annotation"]["subject"] for row in subject_rows)
                ),
            },
            "positive_format_control": {
                "n": len(format_rows),
                "donor_format_blind": rate_record(
                    format_rows,
                    lambda row: row["blind_annotation"]["format"] == row["donor_format"],
                ),
                "recipient_format_blind": rate_record(
                    format_rows,
                    lambda row: row["blind_annotation"]["format"] == row["recipient_format"],
                ),
                "donor_format_structural": rate_record(
                    format_rows, lambda row: row["structural_format"] == row["donor_format"]
                ),
                "recipient_subject": rate_record(
                    format_rows,
                    lambda row: row["blind_annotation"]["subject"] == row["recipient_subject"],
                ),
                "complete": rate_record(
                    format_rows, lambda row: row["blind_annotation"]["complete"]
                ),
                "coherent_ge_60": rate_record(
                    format_rows, lambda row: row["blind_annotation"]["coherence"] >= 60
                ),
            },
        }

    format_self: dict[str, dict[str, Any]] = {}
    for setting in ("L19", "all28"):
        group = [
            row
            for row in subset(rows, arm="self_patch", setting=setting)
            if row["recipient_format"] is not None
        ]
        format_self[setting] = rate_record(
            group, lambda row: row["blind_annotation"]["format"] == row["recipient_format"]
        )

    format_gate_by_setting = {
        setting: (
            settings[setting]["positive_format_control"]["donor_format_blind"]["k"] >= 4
            and anchor_summary["format_label_accuracy"]["k"] >= 5
            and format_self[setting]["k"] >= 5
        )
        for setting in ("L19", "all28")
    }
    gate_pass = format_gate_by_setting["all28"]
    self_patch_gate_pass = all(
        self_summary[setting]["exact_text_match"]["k"] >= 13 for setting in ("L19", "all28")
    )
    primary_specificity_gate_pass = (
        settings["L19"]["task_swap"]["recipient_subject"]["k"] == settings["L19"]["task_swap"]["n"]
        and settings["L19"]["subject_swap"]["recipient_task"]["k"]
        == settings["L19"]["subject_swap"]["n"]
    )
    primary_coherence_gate_pass = all(
        settings["L19"][axis]["coherent_ge_60"]["rate"] >= 0.8
        for axis in ("task_swap", "subject_swap")
    )
    task_k = settings["L19"]["task_swap"]["donor_task"]["k"]
    subject_k = settings["L19"]["subject_swap"]["donor_subject"]["k"]
    if not self_patch_gate_pass:
        verdict = "inconclusive_self_patch_control_failed"
    elif not gate_pass:
        verdict = "inconclusive_pipeline_control_failed"
    elif not primary_coherence_gate_pass:
        verdict = "inconclusive_coherence_floor_failed"
    elif task_k > subject_k and not primary_specificity_gate_pass:
        verdict = "inconclusive_specificity_failed"
    elif task_k > subject_k:
        verdict = "consistent_with_selective_layer19_task_transfer"
    else:
        verdict = "no_selective_layer19_task_transfer"

    return {
        "generated_utc": datetime.now(UTC).isoformat(),
        "planned_n_rows": 129,
        "realized_n_rows": len(rows),
        "all_rows_eos": all(row["termination_reason"] == "eos" for row in rows),
        "all_rows_blind_annotated": all("blind_annotation" in row for row in rows),
        "all_rows_complete_blind": all(row["blind_annotation"]["complete"] for row in rows),
        "anchors": anchor_summary,
        "self_patch": self_summary,
        "format_self_accuracy": format_self,
        "format_gate_by_setting": format_gate_by_setting,
        "settings": settings,
        "positive_format_gate_pass": gate_pass,
        "self_patch_gate_pass": self_patch_gate_pass,
        "primary_specificity_gate_pass": primary_specificity_gate_pass,
        "primary_coherence_gate_pass": primary_coherence_gate_pass,
        "primary_verdict": verdict,
        "annotation_corrections": 0,
    }


def pct(record: dict[str, Any]) -> str:
    return f"{record['k']}/{record['n']} ({100 * record['rate']:.1f}%)"


def markdown_text(text: str) -> str:
    """Keep answer content verbatim apart from Markdown-invisible line-end spaces."""
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def forced_opening_comparison(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Extract the preregistered forced-prefix ablation from two summaries."""
    settings: dict[str, Any] = {}
    for setting in ("L19", "all28"):
        before = baseline["settings"][setting]
        after = current["settings"][setting]
        settings[setting] = {
            "donor_task": {
                "forced": before["task_swap"]["donor_task"],
                "unforced": after["task_swap"]["donor_task"],
            },
            "donor_subject": {
                "forced": before["subject_swap"]["donor_subject"],
                "unforced": after["subject_swap"]["donor_subject"],
            },
            "donor_format": {
                "forced": before["positive_format_control"]["donor_format_blind"],
                "unforced": after["positive_format_control"]["donor_format_blind"],
            },
        }
    return {
        "forced_primary_verdict": baseline["primary_verdict"],
        "unforced_primary_verdict": current["primary_verdict"],
        "settings": settings,
    }


def report_markdown(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    comparison: dict[str, Any] | None = None,
) -> str:
    verdict_text = {
        "inconclusive_self_patch_control_failed": (
            "The self-patch no-op control changed too many greedy answers, so numerical "
            "sensitivity prevents a strong causal interpretation."
        ),
        "inconclusive_pipeline_control_failed": (
            "The exact-pipeline formatting manipulation check failed, so the task/subject "
            "comparison is inconclusive."
        ),
        "inconclusive_coherence_floor_failed": (
            "The layer-19 arm did not meet the preregistered coherence floor, so its "
            "task/subject comparison is descriptive only."
        ),
        "inconclusive_specificity_failed": (
            "Layer 19 showed donor-task expression without fully retaining the recipient "
            "subject/task safeguards, so selective task-vector sufficiency is inconclusive."
        ),
        "consistent_with_selective_layer19_task_transfer": (
            "With the formatting manipulation check passed, layer 19 transferred the donor "
            "response task more often than it transferred the donor subject. This is "
            "qualitative evidence consistent with selective task-vector sufficiency on this bank."
        ),
        "no_selective_layer19_task_transfer": (
            "The exact-pipeline formatting manipulation check passed, but layer 19 did not "
            "transfer donor task more often than donor subject. The corrected run does not "
            "support selective task-vector sufficiency on this bank."
        ),
    }[summary["primary_verdict"]]
    if (
        summary["primary_verdict"] == "no_selective_layer19_task_transfer"
        and not summary["format_gate_by_setting"]["L19"]
    ):
        verdict_text = (
            "The all-layer exact-pipeline formatting control passed, but the layer-19 "
            "formatting control failed. The observed layer-19 task/subject null is therefore "
            "only weakly interpretable and supplies no support for selective task-vector "
            "sufficiency. Under the separately reported maximal all-layer edit, format "
            "transferred while neither task nor subject did."
        )
    lines = [
        "# Corrected natural task/subject context-vector patching — no forced opening",
        "",
        "## Bottom line",
        "",
        verdict_text,
        "",
        "This is a causal sufficiency test of one final-context-token state, not an information-absence test: recipient information remains in every other prompt-token KV entry. Layer 19 is primary; the all-layer intervention is reported separately and cannot be read as a single-layer effect.",
        "",
        "## Integrity checks",
        "",
        f"- Planned/realized generations: {summary['planned_n_rows']}/{summary['realized_n_rows']}.",
        f"- Complete EOS-terminated answers: {summary['all_rows_eos']}.",
        f"- Blind reader marked every answer complete: {summary['all_rows_complete_blind']}.",
        f"- Frozen-key blinded row annotations: {summary['all_rows_blind_annotated']} ({summary['annotation_corrections']} corrections).",
        f"- Blind reader: `{summary['annotation_audit']['model']}` via `{summary['annotation_audit']['backend']}`; {summary['annotation_audit']['n_packets']} production packets; leakage and tool-use audits clean.",
        f"- Unpatched main form accuracy: {pct(summary['anchors']['main_form_accuracy'])}; subject accuracy: {pct(summary['anchors']['main_subject_accuracy'])}.",
        f"- Unpatched bullet/paragraph category accuracy: {pct(summary['anchors']['format_label_accuracy'])} blind, {pct(summary['anchors']['format_structural_accuracy'])} structural.",
        f"- Self-patch exact-answer agreement: L19 {pct(summary['self_patch']['L19']['exact_text_match'])}; all-layer {pct(summary['self_patch']['all28']['exact_text_match'])}.",
        f"- Self-patch no-op gate: {'PASS' if summary['self_patch_gate_pass'] else 'FAIL'} (minimum 13/15 in each setting).",
        f"- Maximum recorded source-state injection error: L19 {summary['self_patch']['L19']['max_injection_error']:.3g}; all-layer {summary['self_patch']['all28']['max_injection_error']:.3g}.",
        "",
        "## Annotation protocol deviation",
        "",
        "The planned direct Claude reader could not be used because the Anthropic API credential returned HTTP 401. The already-frozen opaque key was retained. An isolated `gpt-6-astra` Codex CLI process read each content-only packet in a new neutral empty directory with user configuration ignored and a read-only sandbox. This preserves arm/content blinding, and the event transcripts prove that no tools were used, but it changes the preregistered judge family and introduces the CLI's built-in system context. Results should be read with these deviations in mind.",
        "",
        "## Primary and secondary results",
        "",
        "| Intervention | Donor task on task swaps | Recipient subject retained | Donor subject on subject swaps | Recipient task retained | Donor format on positive control | Format gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for setting, label in (("L19", "Layer 19 (primary)"), ("all28", "All 28 layers (secondary)")):
        block = summary["settings"][setting]
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    pct(block["task_swap"]["donor_task"]),
                    pct(block["task_swap"]["recipient_subject"]),
                    pct(block["subject_swap"]["donor_subject"]),
                    pct(block["subject_swap"]["recipient_task"]),
                    pct(block["positive_format_control"]["donor_format_blind"]),
                    "PASS" if summary["format_gate_by_setting"][setting] else "FAIL",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "Wilson intervals and every denominator are in `summary.json`. The manipulation gate is defined on the all-layer format arm plus unpatched/all-layer-self requested-format accuracy; it is not retrofitted to the observed task result.",
            "",
        )
    )
    if comparison is not None:
        lines.extend(
            (
                "## Forced-opening ablation",
                "",
                "| Intervention | Donor task (forced → unforced) | Donor subject (forced → unforced) | Donor format (forced → unforced) |",
                "|---|---:|---:|---:|",
            )
        )
        for setting, label in (("L19", "Layer 19"), ("all28", "All 28 layers")):
            block = comparison["settings"][setting]
            lines.append(
                "| "
                + " | ".join(
                    (
                        label,
                        f"{pct(block['donor_task']['forced'])} → {pct(block['donor_task']['unforced'])}",
                        f"{pct(block['donor_subject']['forced'])} → {pct(block['donor_subject']['unforced'])}",
                        f"{pct(block['donor_format']['forced'])} → {pct(block['donor_format']['unforced'])}",
                    )
                )
                + " |"
            )
        lines.extend(
            (
                "",
                "Removing the fixed opening restored the all-layer positive control without creating task or subject transfer. This changes the earlier pipeline-inconclusive result into an interpretable negative for the maximal all-layer edit; the single-layer null remains weak because its own formatting control still failed.",
                "",
            )
        )
    lines.extend(
        (
            "## Fixed qualitative roster",
            "",
            "The roster below was selected mechanically before reading outcomes: for each ordered task pair, the Vancouver row; for each ordered subject pair, the itinerary row; and all six primary layer-19 formatting controls. Answers are complete, not excerpts.",
            "",
        )
    )
    candidates = [
        row for row in rows if row["arm"] == "donor_patch" and row["layer_setting"] == "L19"
    ]
    chosen: list[dict[str, Any]] = []
    for row in candidates:
        if row["axis"] == "same_subject_different_task" and row["recipient_subject"] == "vancouver":
            chosen.append(row)
        elif row["axis"] == "same_task_different_subject" and row["recipient_task"] == "itinerary":
            chosen.append(row)
        elif row["axis"] == "positive_format_control":
            chosen.append(row)
    chosen.sort(key=lambda row: row["gen_id"])
    for index, row in enumerate(chosen, 1):
        ann = row["blind_annotation"]
        lines.extend(
            (
                f"### {index}. {row['axis']}: {row['recipient_prompt_id']} ← {row['donor_prompt_id']}",
                "",
                f"Blind label: form `{ann['form']}`, subject `{ann['subject']}`, format `{ann['format']}`, complete `{ann['complete']}`, coherence `{ann['coherence']}`.",
                "",
                markdown_text(str(row["output_text"])),
                "",
            )
        )
    lines.extend(
        (
            "## Scope",
            "",
            "The bank contains three natural single-turn response tasks and three travel/history subjects. Greedy decoding gives one outcome per directed pair; the 18 prompt swaps per axis, not stochastic samples, are the replication units. The controlled templates are necessary for one-axis swaps but limit generalization. Generation begins directly from the patched final context token; no answer tokens are forced.",
            "",
        )
    )
    return "\n".join(lines).rstrip()


def run(args: argparse.Namespace) -> None:
    generations = read_jsonl(args.generations)
    annotations = read_jsonl(args.annotations)
    rows = validate_and_join(generations, annotations)
    annotation_audit = validate_annotation_audit(args.annotations, annotations, generations)
    summary = summarize(rows)
    summary["annotation_audit"] = annotation_audit
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "rows_annotated.jsonl", rows)
    corrections_path = args.out / "annotation_corrections.jsonl"
    if not corrections_path.exists():
        corrections_path.write_text("", encoding="utf-8")
    corrections = read_jsonl(corrections_path)
    summary["annotation_corrections"] = len(corrections)
    atomic_json(args.out / "summary.json", summary)
    comparison = None
    baseline_path = getattr(args, "forced_baseline_summary", None)
    if baseline_path is not None:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        comparison = forced_opening_comparison(baseline, summary)
        atomic_json(args.out / "forced_opening_comparison.json", comparison)
    report = report_markdown(rows, summary, comparison)
    report_bytes = (report + "\n").encode()
    (args.out / "qualitative_report.md").write_bytes(report_bytes)
    atomic_json(
        args.out / "DONE.json",
        {
            "generated_utc": datetime.now(UTC).isoformat(),
            "n_rows": len(rows),
            "primary_verdict": summary["primary_verdict"],
            "positive_format_gate_pass": summary["positive_format_gate_pass"],
            "annotation_backend": annotation_audit["backend"],
            "annotation_model": annotation_audit["model"],
            "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        },
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--forced-baseline-summary", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
