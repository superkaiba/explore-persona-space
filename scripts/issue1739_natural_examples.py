"""Preserve selected natural prompts, five cached answers and archived judgments.

Local-only export: no generation, judging, downloading, or uploading occurs.
Example:
  python scripts/issue1739_natural_examples.py --metadata-root RUN/inputs \
    --selection-root RUN/results --output-root RUN/selected_examples
Selection files are BEHAVIOR/selection.json or BEHAVIOR/L*/selection.json.
Use --behaviors evil for the existing pilot. All successful selections with
explicit high_ids/low_ids are exported by default, including sensitivity runs.
The output manifest explicitly lists controls without recorded membership IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

BEHAVIORS = ("evil", "sycophancy", "hallucination")
MAX_SHARD_BYTES = 8_000_000
HALLU_JUDGE_ROOT = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1739_natpv/inputs/issue1739_ctxmap/judge/hallucination"
)
HALLU_JUDGE_REVISION = "7a47ff5ce42f16308bebaba29c1286a4e9bc8008"
# Independently matched against the pinned HF Git-blob identities, 2026-09-18.
HALLU_JUDGE_PARTS = {
    "labeling_scores.json.part000": "87907bf85c8acbb91100b5d0b1d66dd8bdb179e6b5466f020adef7aa7cc4dcd6",
    "labeling_scores.json.part001": "89c2a53c8bf8dc96a86a510418804f1b7ff28511d43ff28b715d56b3675d7249",
}


def digest(data):
    """Hash the exact bytes used by a source or output artifact."""
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    """Serialize strict, deterministic JSON without altering source strings."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def read_verified(path, expected=None):
    """Read one immutable snapshot and reject a different source digest."""
    data = Path(path).read_bytes()
    actual = digest(data)
    if expected is not None and actual != expected:
        raise ValueError(f"Source hash mismatch: {path}")
    return json.loads(data), actual


def selections(root, behavior, primary_only):
    """Require identical selection membership across every available layer."""
    base = Path(root) / behavior
    paths = sorted(base.glob("L*/selection.json"))
    if (base / "selection.json").exists():
        paths.append(base / "selection.json")
    if not paths:
        raise ValueError(f"No selection files for {behavior}: {base}")
    canonical, sources = None, {}
    for path in paths:
        data, sha = read_verified(path)
        if canonical is not None and data != canonical:
            raise ValueError(f"Selection records differ across layers: {path}")
        canonical = data
        sources[str(path.resolve())] = sha
    roles, included, omitted = defaultdict(list), {}, {}
    for key, record in sorted(canonical.items()):
        if key.startswith("_"):
            continue
        dataset, variant = key.split("/", 1)
        if record.get("status") != "ok":
            omitted[key] = {"reason": "selection_unavailable", "record": record}
            continue
        if primary_only and variant != "q01_s0":
            omitted[key] = {"reason": "explicit_primary_only_scope"}
            continue
        if "high_ids" not in record or "low_ids" not in record:
            if variant.startswith("q"):
                raise ValueError(f"Successful quantile selection lacks membership IDs: {key}")
            omitted[key] = {"reason": "no_explicit_membership_ids", "record": record}
            continue
        if set(record["high_ids"]) & set(record["low_ids"]):
            raise ValueError(f"Overlapping high/low tails: {key}")
        for tail in ("high", "low"):
            ids = record[f"{tail}_ids"]
            if not ids or len(ids) != len(set(ids)) or len(ids) != record[tail]["n_prompts"]:
                raise ValueError(f"Invalid selected context set: {key}/{tail}")
            for cid in ids:
                roles[cid].append({"dataset": dataset, "variant": variant, "tail": tail})
        included[key] = record
    if not roles or not any(key.endswith("/q01_s0") for key in included):
        raise ValueError(f"No successful primary selection for {behavior}")
    return dict(roles), included, omitted, sources


def load_inputs(root, behavior, selected):
    """Verify the bundle, context parts, and original label/rollout-score joins."""
    root = Path(root)
    bundle, bundle_sha = read_verified(root / "bundle_manifest.json")
    files = bundle["files"][behavior]
    base = root / behavior
    metadata, metadata_sha = read_verified(base / "metadata.json", files["metadata.json"]["sha256"])
    if metadata["behavior"] != behavior:
        raise ValueError("Metadata behavior mismatch")
    sources = {
        str((root / "bundle_manifest.json").resolve()): bundle_sha,
        str((base / "metadata.json").resolve()): metadata_sha,
    }
    contexts, seen = {}, set()
    for part in metadata["context_parts"]:
        name = part["path"]
        if Path(name).name != name or part["sha256"] != files[name]["sha256"]:
            raise ValueError(f"Invalid context part identity: {name}")
        rows, sha = read_verified(base / name, part["sha256"])
        if len(rows) != part["n"] or seen.intersection(rows):
            raise ValueError(f"Context part count/identity mismatch: {name}")
        seen.update(rows)
        contexts.update((cid, row) for cid, row in rows.items() if cid in selected)
        sources[str((base / name).resolve())] = sha
    if len(seen) != metadata["n_contexts"] or set(contexts) != set(selected):
        raise ValueError(f"Incomplete context metadata for {behavior}")
    labels, label_sha = read_verified(
        base / "train_labels.json", files["train_labels.json"]["sha256"]
    )
    label_origins = [p for p, sha in metadata["source_files"].items() if sha == label_sha]
    if not label_origins:
        raise ValueError("Copied labels lack a verified original source")
    sources[str((base / "train_labels.json").resolve())] = label_sha
    for path in label_origins:
        _, actual = read_verified(path, label_sha)
        sources[path] = actual
    context_labels = {}
    for index, row in enumerate(labels["rows"]):
        cid = row["context_id"]
        if cid in selected:
            if cid in context_labels:
                raise ValueError(f"Duplicate judgment context: {cid}")
            context_labels[cid] = (row, index)
    if set(context_labels) != set(selected):
        raise ValueError("Selected context absent from original judgments")
    score_data, score_path, score_sha = labels, label_origins[0], label_sha
    if behavior == "hallucination":
        candidates = [
            p for p in metadata["source_files"] if Path(p).name == "hallucination_per_rollout.json"
        ]
        if len(candidates) != 1:
            raise ValueError("Ambiguous hallucination per-response judgment source")
        score_path = candidates[0]
        score_data, score_sha = read_verified(score_path, metadata["source_files"][score_path])
        sources[score_path] = score_sha
    score_rows = {}
    for index, row in enumerate(score_data["rows"]):
        cid = row["context_id"]
        if cid in selected:
            if cid in score_rows:
                raise ValueError(f"Duplicate per-response judgment context: {cid}")
            score_rows[cid] = (row, index)
    means = {}
    for cid, info in contexts.items():
        if (
            info["natural_eligible"] is not True
            or info["behavior"] != behavior
            or info["exclusion_reason"] is not None
        ):
            raise ValueError(f"Selected prompt is not eligible: {cid}")
        if cid not in score_rows:
            raise ValueError(f"Missing per-response judgment source: {cid}")
        score_row = score_rows[cid][0]
        scores = score_row["per_rollout_scores"]
        if scores != metadata["scores"][cid]:
            raise ValueError(f"Frozen per-response scores changed: {cid}")
        if set(scores) - {f"k{k:02d}" for k in range(5)}:
            raise ValueError(f"Unexpected judgment draw key: {cid}")
        valid = [v for v in scores.values() if v is not None]
        if not valid or any(
            isinstance(v, bool)
            or not isinstance(v, (float, int))
            or not math.isfinite(v)
            or not 0 <= v <= 100
            for v in valid
        ):
            raise ValueError(f"Invalid archived judgment values: {cid}")
        means[cid] = sum(valid) / len(valid) / (100 if behavior == "hallucination" else 1)
        label = context_labels[cid][0]
        if not math.isclose(means[cid], label["dv"], abs_tol=1e-12):
            raise ValueError(f"Native prompt mean differs from archived DV: {cid}")
        if any(label[key] != info[key] for key in ("behavior", "split", "rung")):
            raise ValueError(f"Context judgment metadata mismatch: {cid}")
        if sorted(info["rollout_ks"]) != list(range(5)):
            raise ValueError(f"Metadata does not identify five captured responses: {cid}")
    return {
        "metadata": metadata,
        "contexts": contexts,
        "labels": context_labels,
        "score_rows": score_rows,
        "means": means,
        "source_files": sources,
        "label_provenance": {"paths": label_origins, "sha256": label_sha},
        "score_provenance": {"path": score_path, "sha256": score_sha},
        "judge_meta": labels.get("judge_meta"),
        "score_provenance_note": score_data.get("per_rollout_scores_provenance"),
    }


def load_categories(root, inputs):
    """Recover actual three-way labels, never treating a zero score as correct."""
    parts = []
    for name, expected in HALLU_JUDGE_PARTS.items():
        path = Path(root) / name
        data = path.read_bytes()
        if digest(data) != expected:
            raise ValueError(f"Pinned three-way source differs: {path}")
        inputs["source_files"][str(path.resolve())] = expected
        parts.append(data)
    raw = json.loads(b"".join(parts))
    categories, judge_rows = {}, {}
    mapping = {"fabricated": 100.0, "correct": 0.0, "abstained": 0.0, "unjudged": None}
    for cid, (row, _) in inputs["score_rows"].items():
        counts = Counter()
        for draw in range(5):
            item = f"{cid}_k{draw:02d}"
            category = raw["three_way"][item]
            if (
                category not in mapping
                or row["per_rollout_scores"].get(f"k{draw:02d}") != mapping[category]
            ):
                raise ValueError(f"Three-way category and archived score disagree: {item}")
            categories[item] = category
            counts[category] += 1
            judge_rows[item] = {
                key: value[item]
                for key, value in raw["abstain_judge"].items()
                if isinstance(value, dict) and item in value
            }
        if any(counts[key] != row["counts"][key] for key in ("correct", "abstained", "fabricated")):
            raise ValueError(f"Three-way category counts differ from frozen metadata: {cid}")
        if counts["unjudged"] != row["n_unjudged"]:
            raise ValueError(f"Three-way unjudged count mismatch: {cid}")
    inputs["categories"], inputs["category_judge_rows"] = categories, judge_rows
    inputs["category_provenance"] = {
        "repo": "superkaiba1/explore-persona-space-data",
        "revision": HALLU_JUDGE_REVISION,
        "parts": {
            "issue1739_ctxmap/judge/hallucination/" + k: v for k, v in HALLU_JUDGE_PARTS.items()
        },
        "join_key": "context_id + '_k' + two-digit rollout_k",
    }


class Shards:
    """Write bounded JSONL shards, publishing filenames only after validation."""

    def __init__(self, root, max_bytes):
        self.root, self.max_bytes = root, max_bytes
        self.parts, self.current, self.size, self.rows = [], [], 0, 0

    def add(self, row):
        data = encoded(row) + b"\n"
        if len(data) > self.max_bytes:
            raise ValueError("One example exceeds the declared shard byte limit")
        if self.size + len(data) > self.max_bytes:
            self.flush()
        self.current.append(data)
        self.size += len(data)
        self.rows += 1

    def flush(self):
        if not self.current:
            return
        name = f"examples.shard{len(self.parts):04d}.jsonl"
        data = b"".join(self.current)
        (self.root / (name + ".partial")).write_bytes(data)
        self.parts.append(
            {"path": name, "bytes": len(data), "sha256": digest(data), "n_rows": len(self.current)}
        )
        self.current, self.size = [], 0

    def publish(self):
        self.flush()
        for part in self.parts:
            (self.root / (part["path"] + ".partial")).replace(self.root / part["path"])


def export_behavior(inputs, roles, included, writer):
    """Stream the frozen raw shards; preserve dropped judgments and exact draw coverage."""
    metadata, contexts = inputs["metadata"], inputs["contexts"]
    behavior = metadata["behavior"]
    frozen = metadata["source_files"]
    paths = sorted(
        p
        for p in frozen
        if Path(p).name.startswith(f"labeling_{behavior}.shard") and Path(p).suffix == ".jsonl"
    )
    if not paths or any(info["source_path"] not in paths for info in contexts.values()):
        raise ValueError("Selected raw source is outside the frozen original corpus")
    for key, record in included.items():
        for tail in ("high", "low"):
            values = [inputs["means"][cid] for cid in record[f"{tail}_ids"]]
            if not math.isclose(
                sum(values) / len(values), record[tail]["mean_score"], abs_tol=1e-10
            ):
                raise ValueError(
                    f"Tail mean differs from selected original judgments: {key}/{tail}"
                )
    seen, missing_judgments, first_sources = set(), 0, set()
    for path in paths:
        hasher = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                hasher.update(raw)
                if not raw.strip():
                    continue
                packed = json.loads(raw)
                doc = packed["doc"]
                cid = doc.get("context_id")
                if cid not in contexts:
                    continue
                info = contexts[cid]
                draw = doc["rollout_k"]
                if isinstance(draw, bool) or not isinstance(draw, int) or draw not in range(5):
                    raise ValueError(f"Invalid raw rollout index: {cid}")
                pair = (cid, draw)
                if pair in seen:
                    raise ValueError(f"Duplicate captured response: {cid}/{draw}")
                if any(
                    doc[key] != info[key]
                    for key in ("query", "prompt_text", "behavior", "rung", "split")
                ):
                    raise ValueError(f"Exact cached prompt/metadata mismatch: {cid}/{draw}")
                if not isinstance(doc["completion"], str) or not isinstance(doc["meta"], dict):
                    raise ValueError(f"Invalid raw completion schema: {cid}/{draw}")
                if path == info["source_path"] and packed["src"] == info["source_record"]:
                    if doc["meta"] != info["generation_meta"]:
                        raise ValueError(f"Generation provenance mismatch: {cid}")
                    first_sources.add(cid)
                label, label_index = inputs["labels"][cid]
                if doc["group_key"] != label["group_key"]:
                    raise ValueError(f"Raw context group mismatch: {cid}")
                score_row, score_index = inputs["score_rows"][cid]
                key = f"k{draw:02d}"
                score = score_row["per_rollout_scores"].get(key)
                is_missing = score is None
                missing_judgments += is_missing
                seen.add(pair)
                writer.add(
                    {
                        "behavior": behavior,
                        "context_id": cid,
                        "rollout_k": draw,
                        "selection_roles": roles[cid],
                        "generation": doc,
                        "judgment": {
                            "archived_response_score": score,
                            "archived_score_units": "0..100",
                            "score_key_present": key in score_row["per_rollout_scores"],
                            "judge_missing": is_missing,
                            "native_response_score": None
                            if is_missing
                            else score / (100 if behavior == "hallucination" else 1),
                            "native_score_units": "0..1"
                            if behavior == "hallucination"
                            else "0..100",
                            "native_prompt_mean": inputs["means"][cid],
                            "original_context_judgment": label,
                            "original_score_row": score_row,
                            "three_way_category": inputs.get("categories", {}).get(f"{cid}_{key}"),
                            "original_abstain_judge": inputs.get("category_judge_rows", {}).get(
                                f"{cid}_{key}"
                            ),
                        },
                        "provenance": {
                            "raw_path": path,
                            "raw_sha256": frozen[path],
                            "physical_line": line_number,
                            "line_sha256": digest(raw),
                            "packed_source_record": packed["src"],
                            "query_sha256": digest(doc["query"].encode()),
                            "prompt_sha256": digest(doc["prompt_text"].encode()),
                            "completion_sha256": digest(doc["completion"].encode()),
                            "context_judgment": {
                                **inputs["label_provenance"],
                                "row_index": label_index,
                            },
                            "response_judgment": {
                                **inputs["score_provenance"],
                                "row_index": score_index,
                                "key": key,
                            },
                        },
                    }
                )
        if hasher.hexdigest() != frozen[path]:
            raise ValueError(f"Frozen raw shard hash mismatch: {path}")
        inputs["source_files"][path] = frozen[path]
    expected = {(cid, draw) for cid in roles for draw in range(5)}
    if seen != expected or first_sources != set(roles):
        raise ValueError(
            f"Incomplete selected response/provenance coverage: missing={len(expected - seen)}"
        )
    return {
        "contexts": len(roles),
        "expected_response_rows": len(expected),
        "realized_unique_response_rows": len(seen),
        "missing_judgments": missing_judgments,
        "draws_per_context": 5,
        "sources": dict(Counter(c["rung"] for c in contexts.values())),
    }


def export(
    metadata_root,
    selection_root,
    output_root,
    behaviors=BEHAVIORS,
    primary_only=False,
    max_shard_bytes=MAX_SHARD_BYTES,
    max_contexts=20000,
    hallucination_judge_root=HALLU_JUDGE_ROOT,
):
    """Build a complete self-contained export, or leave only unverified partial files."""
    if not 1024 <= max_shard_bytes <= MAX_SHARD_BYTES:
        raise ValueError("Shard bound must be between 1024 and 8000000 bytes")
    if not behaviors or len(set(behaviors)) != len(behaviors) or set(behaviors) - set(BEHAVIORS):
        raise ValueError("Choose distinct supported behaviors")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Output directory must be empty; never overwrite an existing export")
    writer = Shards(root, max_shard_bytes)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "exporter_sha256": digest(Path(__file__).read_bytes()),
        "metadata_root": str(Path(metadata_root).resolve()),
        "selection_root": str(Path(selection_root).resolve()),
        "primary_only": primary_only,
        "coverage": {},
        "selection_files": {},
        "selection_coverage": {},
        "unexported_selections": {},
        "verified_source_files": {},
        "judge_metadata": {},
        "notes": [
            "All five captured responses are preserved; unavailable judgments remain null, never zero.",
            "Archived per-response scores are the original aggregates used by selection; individual judge draws are not reconstructed.",
            "Only explicit selection IDs are exported; coefficient-only E2/E2p controls and unrecorded split-half membership are not inferred.",
            "Selection-file layer coverage is the available input set; a pilot export does not establish final-run completion.",
        ],
    }
    n_contexts = 0
    for behavior in behaviors:
        roles, included, omitted, selection_files = selections(
            selection_root, behavior, primary_only
        )
        n_contexts += len(roles)
        if n_contexts > max_contexts:
            raise ValueError(
                "Selected union exceeds --max-contexts; choose an explicit larger bound"
            )
        inputs = load_inputs(metadata_root, behavior, roles)
        if behavior == "hallucination":
            load_categories(hallucination_judge_root, inputs)
        manifest["coverage"][behavior] = export_behavior(inputs, roles, included, writer)
        manifest["verified_source_files"].update(inputs["source_files"])
        manifest["selection_files"].update(selection_files)
        manifest["selection_coverage"][behavior] = {
            key: {"high": len(r["high_ids"]), "low": len(r["low_ids"])}
            for key, r in included.items()
        }
        manifest["unexported_selections"][behavior] = omitted
        manifest["judge_metadata"][behavior] = {
            "judge_meta": inputs["judge_meta"],
            "score_provenance_note": inputs["score_provenance_note"],
            "category_provenance": inputs.get("category_provenance"),
        }
        if behavior == "hallucination":
            for key, record in included.items():
                manifest["selection_coverage"][behavior][key]["three_way_counts"] = {
                    tail: dict(
                        Counter(
                            inputs["categories"][f"{cid}_k{draw:02d}"]
                            for cid in record[f"{tail}_ids"]
                            for draw in range(5)
                        )
                    )
                    for tail in ("high", "low")
                }
    writer.publish()
    manifest["parts"] = writer.parts
    manifest["n_contexts"] = n_contexts
    manifest["n_rows"] = writer.rows
    data = encoded(manifest) + b"\n"
    if len(data) > MAX_SHARD_BYTES:
        raise ValueError("Export manifest exceeds upload byte limit")
    temporary = root / "examples.manifest.json.partial"
    temporary.write_bytes(data)
    temporary.replace(root / "examples.manifest.json")
    return {
        "manifest": str((root / "examples.manifest.json").resolve()),
        "sha256": digest(data),
        "n_contexts": n_contexts,
        "n_rows": writer.rows,
        "n_shards": len(writer.parts),
        "bytes": sum(p["bytes"] for p in writer.parts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--behaviors", nargs="+", choices=BEHAVIORS, default=list(BEHAVIORS))
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--max-shard-bytes", type=int, default=MAX_SHARD_BYTES)
    parser.add_argument("--max-contexts", type=int, default=20000)
    parser.add_argument("--hallucination-judge-root", type=Path, default=HALLU_JUDGE_ROOT)
    args = parser.parse_args()
    result = export(
        args.metadata_root,
        args.selection_root,
        args.output_root,
        args.behaviors,
        args.primary_only,
        args.max_shard_bytes,
        args.max_contexts,
        args.hallucination_judge_root,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
