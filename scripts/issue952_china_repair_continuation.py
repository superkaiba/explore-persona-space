"""Continue issue 952 judging under an explicitly new Luna model-switch phase.

This module never rewrites original packets, receipts, or authored decisions.  It
builds fresh packets only for incomplete original assignments and emits a strict
index that joins old and new triples by the original assignment identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts import issue952_china_repair_judges as judges

PHASE = "production-continuation-gpt-5.6-luna-medium-v1"
MODEL = "gpt-5.6-luna"
REASONING = "medium"
AGENTS = judges.AGENTS


def _rebase(root: Path, value: str, marker: str) -> Path:
    """Resolve a preparation path after moving the continuation tree."""
    path = Path(value)
    if path.exists():
        return path
    if marker in path.parts:
        return root.joinpath(*path.parts[path.parts.index(marker) :])
    return root / path


def _identity(agent_id: str) -> dict[str, Any]:
    """Record the requested runtime pins; do not impersonate an old reviewer."""
    if not agent_id.strip():
        raise ValueError("continuation reviewer identity must be non-empty")
    return {
        "agent_id": agent_id,
        "model": MODEL,
        "reasoning_effort": REASONING,
        "fork_turns": "none",
        "runtime_note": "New model-switch continuation; not the original reviewer identity.",
        "unavailable": ["temperature", "top_p", "service_tier", "stop_reason"],
    }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _load_manifest(path: Path) -> dict:
    manifest = judges.read_json(path)
    if manifest.get("phase") != "production":
        raise ValueError("continuation requires the original production manifest")
    judges._validate_manifest(manifest)
    return manifest


def _old_triples(old_dir: Path, manifest: dict) -> tuple[set[tuple[str, str]], dict]:
    """Validate complete original triples and return completed keys plus census."""
    lookup = judges.read_json(Path(manifest["lookup_path"]))
    by_key = {
        (lane, info["opaque_id"]): entry
        for entry in lookup
        for lane, info in entry["lanes"].items()
    }
    completed: set[tuple[str, str]] = set()
    census = []
    for record in manifest["packets"]:
        packet_path = Path(record["packet_path"])
        receipt_path = Path(record["receipt_path"])
        output_path = Path(record["output_path"])
        if judges.sha_file(packet_path) != record["packet_sha256"]:
            raise ValueError("original packet bytes changed")
        packet = judges.read_json(packet_path)
        judges._validate_packet(packet, record, manifest, by_key)
        present = [path.exists() for path in (receipt_path, output_path)]
        if any(present) and not all(present):
            raise ValueError("original packet has a receipt/output half-triple")
        if not all(present):
            continue
        receipt = judges.read_json(receipt_path)
        judges._validate_receipt(receipt, packet, record)
        receipt_sha = judges.sha_file(receipt_path)
        rows = judges.read_jsonl(output_path)
        if [row.get("opaque_id") for row in rows] != record["opaque_ids"]:
            raise ValueError("original output does not preserve packet order")
        for row, item in zip(rows, packet["items"], strict=True):
            judges.validate_decision(row, item, packet, record, receipt_sha)
            key = (record["lane"], item["opaque_id"])
            if key in completed:
                raise ValueError("duplicate original completed assignment")
            completed.add(key)
        authored_path = packet_path.with_suffix(".authored.json")
        if authored_path.exists():
            authored = judges.read_json(authored_path)
            authored_fields = {
                "opaque_id", "rationale", "category", *judges.SCORES,
                *judges.BEHAVIOR_FLAGS, "unassessable", "unassessable_reason",
            }
            if (not isinstance(authored, list) or len(authored) != len(rows)
                    or any({key: row[key] for key in authored_fields} != supplied
                           for row, supplied in zip(rows, authored, strict=True))):
                raise ValueError("original authored JSON differs from submitted output")
        census.append({
            "lane": record["lane"],
            "packet_path": str(packet_path),
            "packet_sha256": record["packet_sha256"],
            "receipt_path": str(receipt_path),
            "receipt_sha256": receipt_sha,
            "output_path": str(output_path),
            "output_sha256": judges.sha_file(output_path),
            "n_items": len(rows),
        })
    return completed, {"lookup": lookup, "census": census}


def prepare(old_dir: Path, out_dir: Path, agent_ids: tuple[str, str]) -> dict:
    """Prepare only pending original assignments in two independent Luna lanes."""
    if len(set(agent_ids)) != 2:
        raise ValueError("continuation requires two distinct lane identities")
    old_dir, out_dir = Path(old_dir).resolve(), Path(out_dir).resolve()
    if out_dir == old_dir or out_dir.is_relative_to(old_dir):
        raise ValueError("continuation output must be outside the original judge tree")
    old_manifest_path = old_dir / "manifest.json"
    old_manifest = _load_manifest(old_manifest_path)
    completed, old = _old_triples(old_dir, old_manifest)
    runtime = {lane: _identity(agent) for lane, agent in zip(AGENTS, agent_ids, strict=True)}
    # Keep the original manifest byte-for-byte; this is a provenance reference,
    # not a reserialized substitute for the original control-plane artifact.
    judges.write_immutable(
        out_dir / "private" / "original_manifest.json", old_manifest_path.read_bytes()
    )
    original_files = []
    copy_sources = [
        old_manifest_path,
        Path(old_manifest["lookup_path"]),
        Path(old_manifest["source_manifest_path"]),
        *[Path(record[key]) for record in old_manifest["packets"] for key in
          ("packet_path", "receipt_path", "output_path") if Path(record[key]).exists()],
        *[Path(record["packet_path"]).with_suffix(".authored.json")
          for record in old_manifest["packets"]
          if Path(record["packet_path"]).with_suffix(".authored.json").exists()],
    ]
    for source_path in copy_sources:
        relative = Path("private/original/files") / source_path.name
        # Names are disambiguated below for packet trees; preserve exact bytes.
        if source_path.parent.name in AGENTS:
            relative = Path("private/original/files") / source_path.parent.name / source_path.name
        elif source_path.name == "lookup.json":
            relative = Path("private/original/files") / "lookup.json"
        elif source_path.name == "manifest.json":
            relative = Path("private/original/files") / "manifest.json"
        target = out_dir / relative
        judges.write_immutable(target, source_path.read_bytes())
        original_files.append({"path": relative.as_posix(), "sha256": judges.sha_file(source_path),
                               "bytes": source_path.stat().st_size})
    # A relocatable structural view points at the exact-byte copies above.  The
    # original manifest remains separately preserved byte-for-byte.
    runtime_manifest = json.loads(old_manifest_path.read_bytes())
    runtime_manifest["lookup_path"] = str((out_dir / "private/original/files/lookup.json").resolve())
    runtime_manifest["source_manifest_path"] = str(
        (out_dir / "private/original/files" / Path(old_manifest["source_manifest_path"]).name).resolve()
    )
    for record in runtime_manifest["packets"]:
        lane = record["lane"]
        for key in ("packet_path", "receipt_path", "output_path"):
            record[key] = str(
                (out_dir / "private/original/files" / lane / Path(record[key]).name).resolve()
            )
    runtime_manifest_path = out_dir / "private/original/runtime_manifest.json"
    judges.write_immutable(runtime_manifest_path, _json_bytes(runtime_manifest))
    old_lookup = old["lookup"]
    pending: dict[str, list[dict]] = {lane: [] for lane in AGENTS}
    continuation_lookup = []
    for entry in old_lookup:
        for lane, old_info in entry["lanes"].items():
            old_key = (lane, old_info["opaque_id"])
            if old_key in completed:
                continue
            old_record = next(
                record for record in old_manifest["packets"]
                if record["lane"] == lane and old_info["opaque_id"] in record["opaque_ids"]
            )
            old_packet = judges.read_json(Path(old_record["packet_path"]))
            source_item = next(item for item in old_packet["items"] if item["opaque_id"] == old_info["opaque_id"])
            identity = runtime[lane]
            request = judges._request_hash(source_item, entry, PHASE, identity)
            item = {
                "opaque_id": "o-" + request[:24],
                "request_sha256": request,
                "question": source_item["question"],
                "response": source_item["response"],
                "question_sha256": source_item["question_sha256"],
                "response_sha256": source_item["response_sha256"],
                "question_utf8_bytes": source_item["question_utf8_bytes"],
                "response_utf8_bytes": source_item["response_utf8_bytes"],
            }
            pending[lane].append((entry, old_record, old_info, item))
    packets = []
    for lane in AGENTS:
        rows = sorted(pending[lane], key=lambda value: judges.sha_object({"lane": lane, "id": value[3]["opaque_id"]}))
        for start in range(0, len(rows), judges.PACKET_SIZE):
            batch = rows[start : start + judges.PACKET_SIZE]
            base = out_dir / "packets" / lane / f"batch_{start // judges.PACKET_SIZE:04d}"
            paths = {
                "packet_path": str(base.with_suffix(".json")),
                "output_path": str(base.with_suffix(".output.jsonl")),
                "receipt_path": str(base.with_suffix(".read_receipt.json")),
            }
            items = [value[3] for value in batch]
            payload = {
                "contract": judges.CONTRACT,
                "phase": PHASE,
                "rubric": judges.RUBRIC,
                "rubric_sha256": judges.RUBRIC_SHA256,
                "runtime_identity": runtime[lane],
                "runtime_identity_sha256": judges.sha_object(runtime[lane]),
                "output_fields": list(judges.ROW_FIELDS),
                "categories": list(judges.CATEGORIES),
                "paths": paths,
                "items": items,
                "continuation_note": "Fresh Luna phase; original reviewer decisions and identities are untouched.",
            }
            judges.write_immutable(Path(paths["packet_path"]), _json_bytes(payload))
            packets.append({
                "lane": lane,
                **paths,
                "packet_sha256": judges.sha_file(Path(paths["packet_path"])),
                "n_items": len(items),
                "opaque_ids": [item["opaque_id"] for item in items],
                "original_assignment_refs": [
                    {"lane": lane, "old_opaque_id": value[2]["opaque_id"],
                     "old_packet_sha256": value[1]["packet_sha256"],
                     "item_id": value[0]["item_id"]}
                    for value in batch
                ],
            })
    for entry in old_lookup:
        for lane, info in entry["lanes"].items():
            if (lane, info["opaque_id"]) in completed:
                continue
            packet = next(p for p in packets if info["opaque_id"] in {r["old_opaque_id"] for r in p["original_assignment_refs"]})
            ref = next(r for r in packet["original_assignment_refs"] if r["old_opaque_id"] == info["opaque_id"])
            new_info = next(i for i in judges.read_json(Path(packet["packet_path"]))["items"] if i["opaque_id"] == packet["opaque_ids"][packet["original_assignment_refs"].index(ref)])
            entry_copy = {key: value for key, value in entry.items() if key not in {"lanes"}}
            entry_copy.update({"lane": lane, "old_opaque_id": info["opaque_id"], "old_packet_sha256": ref["old_packet_sha256"],
                               "new_opaque_id": new_info["opaque_id"], "new_packet_sha256": packet["packet_sha256"],
                               "new_packet_path": packet["packet_path"]})
            continuation_lookup.append(entry_copy)
    lookup_path = out_dir / "private" / "continuation_lookup.json"
    judges.write_immutable(lookup_path, _json_bytes(continuation_lookup))
    manifest = {
        "contract": judges.CONTRACT,
        "phase": PHASE,
        "rubric_sha256": judges.RUBRIC_SHA256,
        "runtime": runtime,
        "original_manifest_sha256": judges.sha_file(old_manifest_path),
        "original_manifest_path": str((out_dir / "private" / "original_manifest.json").resolve()),
        "original_runtime_manifest_path": str(runtime_manifest_path.resolve()),
        "original_n_assignments": old_manifest["n_assignments"],
        "original_n_overlap": old_manifest["n_overlap"],
        "original_completed_assignments": len(completed),
        "n_assignments": sum(len(v) for v in pending.values()),
        "n_packets": len(packets),
        "lookup_path": str(lookup_path.resolve()),
        "lookup_sha256": judges.sha_file(lookup_path),
        "packets": packets,
        "completed_census": old["census"],
        "model_switch": {"model": MODEL, "reasoning_effort": REASONING, "fork_turns": "none"},
        "original_files": original_files,
    }
    judges.write_immutable(out_dir / "manifest.json", _json_bytes(manifest))
    return manifest


def validate_continuation(out_dir: Path) -> dict:
    """Validate pending new triples and return structural coverage metadata."""
    out_dir = Path(out_dir).resolve()
    manifest = judges.read_json(out_dir / "manifest.json")
    lookup_path = _rebase(out_dir, manifest["lookup_path"], "private")
    if manifest.get("phase") != PHASE or manifest["lookup_sha256"] != judges.sha_file(lookup_path):
        raise ValueError("continuation manifest/index hash mismatch")
    lookup = judges.read_json(lookup_path)
    by_new = {(row["lane"], row["new_opaque_id"]): row for row in lookup}
    if len(by_new) != manifest["n_assignments"]:
        raise ValueError("continuation lookup assignment census mismatch")
    decisions = {}
    census = []
    for record in manifest["packets"]:
        packet_path = _rebase(out_dir, record["packet_path"], "packets")
        if judges.sha_file(packet_path) != record["packet_sha256"]:
            raise ValueError("continuation packet bytes changed")
        packet = judges.read_json(packet_path)
        lane = record["lane"]
        identity = manifest["runtime"][lane]
        if packet["phase"] != PHASE or packet["runtime_identity"] != identity:
            raise ValueError("continuation packet runtime/phase mismatch")
        if len(packet["items"]) != record["n_items"] or [i["opaque_id"] for i in packet["items"]] != record["opaque_ids"]:
            raise ValueError("continuation packet item census mismatch")
        receipt_path = _rebase(out_dir, record["receipt_path"], "packets")
        output_path = _rebase(out_dir, record["output_path"], "packets")
        if not receipt_path.exists() or not output_path.exists():
            raise FileNotFoundError(f"incomplete continuation triple: {packet_path}")
        judges._validate_receipt(judges.read_json(receipt_path), packet, record)
        receipt_sha = judges.sha_file(receipt_path)
        rows = judges.read_jsonl(output_path)
        if [row.get("opaque_id") for row in rows] != record["opaque_ids"]:
            raise ValueError("continuation output order mismatch")
        for row, item in zip(rows, packet["items"], strict=True):
            judges.validate_decision(row, item, packet, record, receipt_sha)
            key = (lane, item["opaque_id"])
            if key not in by_new or key in decisions:
                raise ValueError("continuation decision lacks unique original reference")
            decisions[key] = row
        census.append({"lane": lane, "packet_sha256": record["packet_sha256"], "receipt_sha256": receipt_sha,
                       "output_sha256": judges.sha_file(output_path), "n_rows": len(rows)})
    if set(decisions) != set(by_new):
        raise ValueError("continuation output coverage incomplete")
    return {"technical_complete": True, "phase": PHASE, "model_switch": manifest["model_switch"],
            "n_assignments": len(decisions), "completed_census": manifest["completed_census"], "continuation_census": census,
            "original_manifest_sha256": manifest["original_manifest_sha256"], "continuation_manifest_sha256": judges.sha_file(out_dir / "manifest.json")}


def reconstruct_mixed_scores(out_dir: Path) -> dict:
    """Join original and Luna decisions without changing either authored row."""
    out_dir = Path(out_dir).resolve()
    manifest = judges.read_json(out_dir / "manifest.json")
    validate_continuation(out_dir)
    original = judges.read_json(
        _rebase(out_dir, manifest["original_runtime_manifest_path"], "private")
    )
    original["lookup_path"] = str(_rebase(out_dir, original["lookup_path"], "private"))
    original["source_manifest_path"] = str(
        _rebase(out_dir, original["source_manifest_path"], "private")
    )
    for record in original["packets"]:
        for key in ("packet_path", "receipt_path", "output_path"):
            record[key] = str(_rebase(out_dir, record[key], "private"))
    completed, old = _old_triples(out_dir, original)
    old_lookup = old["lookup"]
    old_decisions = {}
    for record in original["packets"]:
        if (record["lane"], record["opaque_ids"][0]) not in completed:
            continue
        rows = judges.read_jsonl(Path(record["output_path"]))
        for row in rows:
            old_decisions[(record["lane"], row["opaque_id"])] = row
    new_lookup = judges.read_json(_rebase(out_dir, manifest["lookup_path"], "private"))
    new_decisions = {}
    for record in manifest["packets"]:
        rows = judges.read_jsonl(_rebase(out_dir, record["output_path"], "packets"))
        for row in rows:
            new_decisions[(record["lane"], row["opaque_id"])] = row
    by_old = {(row["lane"], row["old_opaque_id"]): row for row in new_lookup}
    scores, overlap = [], []
    for entry in old_lookup:
        metadata = {key: value for key, value in entry.items()
                    if key not in {"lanes", "primary_agent", "assigned_agents"}}
        lane_rows = {}
        for lane, info in entry["lanes"].items():
            key = (lane, info["opaque_id"])
            if key in old_decisions:
                decision, phase = old_decisions[key], original["phase"]
            else:
                ref = by_old.get(key)
                if ref is None:
                    raise ValueError("mixed reconstruction omitted an original assignment")
                decision, phase = new_decisions[(lane, ref["new_opaque_id"])], PHASE
            lane_rows[lane] = {**decision, "judging_phase": phase}
        primary = entry["primary_agent"]
        scores.append({**metadata, **lane_rows[primary], "judge_id": primary})
        if len(lane_rows) == 2:
            overlap.append({**metadata, **lane_rows})
    score_path = out_dir / "mixed_scores.jsonl"
    overlap_path = out_dir / "mixed_overlap.jsonl"
    judges.write_immutable(score_path, judges._jsonl_bytes(scores))
    judges.write_immutable(overlap_path, judges._jsonl_bytes(overlap))
    result = {"contract": judges.CONTRACT, "phase": "mixed-original-plus-luna",
              "original_manifest_sha256": manifest["original_manifest_sha256"],
              "continuation_manifest_sha256": judges.sha_file(out_dir / "manifest.json"),
              "n_items": len(scores), "n_assignments": sum(len(e["lanes"]) for e in old_lookup),
              "n_overlap": len(overlap), "scores_sha256": judges.sha_file(score_path),
              "overlap_sha256": judges.sha_file(overlap_path),
              "model_switch": manifest["model_switch"]}
    judges.write_immutable(out_dir / "mixed_manifest.json", _json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--old-dir", type=Path, required=True)
    prep.add_argument("--out-dir", type=Path, required=True)
    prep.add_argument("--agent-id", nargs=2, required=True)
    check = sub.add_parser("validate")
    check.add_argument("--out-dir", type=Path, required=True)
    mixed = sub.add_parser("reconstruct")
    mixed.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.old_dir, args.out_dir, tuple(args.agent_id))
    elif args.command == "validate":
        result = validate_continuation(args.out_dir)
    else:
        result = reconstruct_mixed_scores(args.out_dir)
    print(json.dumps({key: value for key, value in result.items() if key not in {"completed_census", "continuation_census"}}, sort_keys=True))


if __name__ == "__main__":
    main()
